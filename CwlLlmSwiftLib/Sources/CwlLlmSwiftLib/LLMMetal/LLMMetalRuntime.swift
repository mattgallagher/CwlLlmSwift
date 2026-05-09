import Foundation

private struct LLMMetalTokenDataLoader {
    private static let headerCount = 256
    private static let magic: UInt32 = 20240520
    private static let version: UInt32 = 1

    let batchSize: Int
    let sequenceLength: Int
    let tokenCount: Int
    let sampleCount: Int

    private let tokens: [UInt16]
    private(set) var currentSampleIndex = 0
    private(set) var inputs: [UInt32]
    private(set) var targets: [UInt32]

    init(data: Data, batchSize: Int, sequenceLength: Int) throws {
        let headerBytes = Self.headerCount * MemoryLayout<UInt32>.size
        guard data.count >= headerBytes else {
            throw LLMMetalRuntimeError.invalidDataset("Dataset shard is too small to contain a valid header.")
        }

        let header = data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: UInt32.self).prefix(Self.headerCount))
        }
        guard header.count >= 3 else {
            throw LLMMetalRuntimeError.invalidDataset("Dataset shard header is incomplete.")
        }
        guard header[0] == Self.magic else {
            throw LLMMetalRuntimeError.invalidDataset("Dataset shard has invalid magic value \(header[0]).")
        }
        guard header[1] == Self.version else {
            throw LLMMetalRuntimeError.invalidDataset("Dataset shard uses unsupported version \(header[1]).")
        }

        let tokenCount = Int(header[2])
        let payload = data.dropFirst(headerBytes)
        let tokens = payload.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: UInt16.self))
        }
        guard tokens.count == tokenCount else {
            throw LLMMetalRuntimeError.invalidDataset("Dataset shard token count mismatch: header says \(tokenCount), file contains \(tokens.count).")
        }
        guard tokenCount >= batchSize * sequenceLength + 1 else {
            throw LLMMetalRuntimeError.invalidDataset("Dataset shard does not contain enough tokens for a batch of size \(batchSize) and sequence length \(sequenceLength).")
        }

        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.tokenCount = tokenCount
        self.sampleCount = max(1, (tokenCount - 1) / (batchSize * sequenceLength))
        self.tokens = tokens
        self.inputs = Array(repeating: 0, count: batchSize * sequenceLength)
        self.targets = Array(repeating: 0, count: batchSize * sequenceLength)
    }

    mutating func reset() {
        currentSampleIndex = 0
    }

    mutating func nextBatch() {
        if currentSampleIndex >= sampleCount {
            currentSampleIndex = 0
        }
        let start = currentSampleIndex * batchSize * sequenceLength
        for index in 0..<(batchSize * sequenceLength) {
            inputs[index] = UInt32(tokens[start + index])
            targets[index] = UInt32(tokens[start + index + 1])
        }
        currentSampleIndex += 1
    }
}

actor LLMMetalRuntime {
    private var model: GPT2Metal?
    private var currentCheckpointData: Data?
    private var trainData: Data?
    private var validationData: Data?
    private var tokenizerData: Data?
    private var batchSize = 4
    private var sequenceLength = 64
    private var completedStepCount = 0

    func loadCheckpoint(data: Data) async throws {
        currentCheckpointData = data
        completedStepCount = 0
        model = try buildModel(from: data)
    }

    func exportCheckpoint() async throws -> Data {
        try ensureModel()
        guard let model else { throw LLMMetalRuntimeError.missingCheckpoint }
        let checkpointData = try model.exportCheckpoint()
        currentCheckpointData = checkpointData
        return checkpointData
    }

    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        trainData = request.trainData
        validationData = request.validationData
        tokenizerData = request.tokenizerData
        batchSize = request.batchSize
        sequenceLength = request.sequenceLength
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        completedStepCount = 0
        try recreateModel()

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.trainingLoop(request: request, continuation: continuation)
                    self.releaseTrainingState()
                    continuation.finish()
                } catch {
                    self.releaseTrainingState()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        if let tokenizerData = request.tokenizerData {
            self.tokenizerData = tokenizerData
        }
        guard let tokenizerData else { throw LLMMetalRuntimeError.tokenizerRequired }

        let tokenizer = try LLMTokenizer(data: tokenizerData)
        try ensureModel()
        guard let model else { throw LLMMetalRuntimeError.failedToCreateModel }

        batchSize = 1
        sequenceLength = min(model.config.max_seq_len, max(1, request.maximumTokenCount))
        let promptTokens = tokenizer.encodePrompt(
            request.prompt,
            maximumTokenCount: model.config.max_seq_len
        )

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.inferenceLoop(
                        tokenizer: tokenizer,
                        promptTokens: promptTokens,
                        request: request,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    private func trainingLoop(
        request: LLMTrainingRequest,
        continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation
    ) async throws {
        try ensureModel()
        guard let model else { throw LLMMetalRuntimeError.failedToCreateModel }
        guard let trainData, let validationData else { throw LLMMetalRuntimeError.failedToCreateModel }

        let effectiveSequenceLength = min(sequenceLength, model.config.max_seq_len)
        var trainLoader = try LLMMetalTokenDataLoader(
            data: trainData,
            batchSize: batchSize,
            sequenceLength: effectiveSequenceLength
        )
        var validationLoader = try LLMMetalTokenDataLoader(
            data: validationData,
            batchSize: batchSize,
            sequenceLength: effectiveSequenceLength
        )

        for _ in 0..<request.stepCount {
            try Task.checkCancellation()
            let stepStart = ContinuousClock.now
            trainLoader.nextBatch()
            let forwardStart = ContinuousClock.now
            model.forward(
                inputs: trainLoader.inputs,
                targets: trainLoader.targets,
                B: batchSize,
                T: effectiveSequenceLength
            )
            let forwardPassMilliseconds = forwardStart.duration(to: .now).timeInterval * 1_000
            let backwardStart = ContinuousClock.now
            model.backward()
            let backwardPassMilliseconds = backwardStart.duration(to: .now).timeInterval * 1_000
            completedStepCount += 1
            model.update(
                learningRate: Float(request.learningRate),
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weightDecay: 0,
                t: completedStepCount
            )
            let elapsed = stepStart.duration(to: .now)
            let trainingLoss = Double(model.mean_loss)
            let validationLoss = computeValidationLoss(
                using: model,
                loader: &validationLoader,
                batchCount: request.validationBatchCount,
                sequenceLength: effectiveSequenceLength
            )

            continuation.yield(
                LLMTrainingProgress(
                    step: completedStepCount,
                    iterationsPerSecond: elapsed.components.seconds > 0 || elapsed.components.attoseconds > 0 ? 1 / elapsed.timeInterval : nil,
                    forwardPassMilliseconds: forwardPassMilliseconds,
                    backwardPassMilliseconds: backwardPassMilliseconds,
                    trainingLoss: trainingLoss,
                    validationLoss: validationLoss
                )
            )
        }

        currentCheckpointData = try model.exportCheckpoint()
    }

    private func inferenceLoop(
        tokenizer: LLMTokenizer,
        promptTokens: [Int],
        request: LLMInferenceRequest,
        continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation
    ) async throws {
        try ensureModel()
        guard let model else { throw LLMMetalRuntimeError.failedToCreateModel }

        let effectiveSequenceLength = min(
            model.config.max_seq_len,
            max(promptTokens.count + request.maximumTokenCount, 1)
        )
        let totalCount = min(effectiveSequenceLength, promptTokens.count + request.maximumTokenCount)
        var generationTokens = Array(repeating: UInt32(tokenizer.eotToken), count: effectiveSequenceLength)
        for (index, token) in promptTokens.prefix(totalCount).enumerated() {
            generationTokens[index] = UInt32(token)
        }

        var generatedTokenCount = 0
        var state = UInt64.random(in: UInt64.min...UInt64.max)
        let startIndex = min(promptTokens.count, totalCount)
        guard startIndex < totalCount else { return }

        for tokenIndex in startIndex..<totalCount {
            try Task.checkCancellation()
            let logits = model.performInference(inputs: generationTokens, B: 1, T: effectiveSequenceLength)
            let rowStart = max(0, tokenIndex - 1) * model.config.vocab_size
            let rowEnd = rowStart + model.config.vocab_size
            let nextToken = LLMSwift.sample(logits: ArraySlice(logits[rowStart..<rowEnd]), temperature: request.temperature, state: &state)
            generationTokens[tokenIndex] = UInt32(nextToken)
            generatedTokenCount += 1
            continuation.yield(LLMInferenceChunk(text: tokenizer.decode(token: nextToken), generatedTokenCount: generatedTokenCount))
        }
    }

    private func ensureModel() throws {
        if model == nil { try recreateModel() }
    }

    private func recreateModel() throws {
        guard let currentCheckpointData else { throw LLMMetalRuntimeError.missingCheckpoint }
        model = try buildModel(from: currentCheckpointData)
    }

    // Drops the in-memory model and dataset buffers after a training run so
    // running multiple engines back-to-back (e.g. on the Comparison screen)
    // doesn't accumulate ~3-4GB of state per engine. The model can be rebuilt
    // on demand from `currentCheckpointData` if inference or export follows.
    private func releaseTrainingState() {
        model = nil
        trainData = nil
        validationData = nil
        tokenizerData = nil
    }

    // Public counterpart to `releaseTrainingState` that also drops the cached
    // checkpoint, so a fully idle engine retains effectively nothing. Callers
    // (e.g. the Comparison screen) invoke this when they're done with an
    // engine for now and don't intend to immediately call exportCheckpoint.
    func releaseResources() {
        releaseTrainingState()
        currentCheckpointData = nil
        completedStepCount = 0
    }

    private func buildModel(from data: Data) throws -> GPT2Metal {
        let (header, parameters) = try LLMCheckpointCodec.decode(data)
        let config = LLMGPT2Config(header: header)
        let parameterData = parameters.withUnsafeBufferPointer { Data(buffer: $0) }
        return try GPT2Metal(config: config, parameterData: parameterData)
    }

    private func computeValidationLoss(using model: GPT2Metal, loader: inout LLMMetalTokenDataLoader, batchCount: Int, sequenceLength: Int) -> Double? {
        guard batchCount > 0 else { return nil }
        loader.reset()
        var total: Double = 0
        for _ in 0..<batchCount {
            loader.nextBatch()
            model.forward(inputs: loader.inputs, targets: loader.targets, B: batchSize, T: sequenceLength)
            total += Double(model.mean_loss)
        }
        return total / Double(batchCount)
    }
}

private extension Duration {
    var timeInterval: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
