import Foundation

enum LLMSwiftRuntimeError: LocalizedError {
    case missingCheckpoint
    case failedToCreateModel
    case invalidDataset(String)
    case tokenizerRequired

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint:
            return "A checkpoint is required before running the fast Swift backend."
        case .failedToCreateModel:
            return "Failed to create the fast Swift model runtime."
        case .invalidDataset(let message):
            return message
        case .tokenizerRequired:
            return "Inference requires a tokenizer asset."
        }
    }
}

private struct LLMSwiftTokenDataLoader {
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
            throw LLMSwiftRuntimeError.invalidDataset("Dataset shard is too small to contain a valid header.")
        }

        let header = data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: UInt32.self).prefix(Self.headerCount))
        }
        guard header.count >= 3 else {
            throw LLMSwiftRuntimeError.invalidDataset("Dataset shard header is incomplete.")
        }
        guard header[0] == Self.magic else {
            throw LLMSwiftRuntimeError.invalidDataset("Dataset shard has invalid magic value \(header[0]).")
        }
        guard header[1] == Self.version else {
            throw LLMSwiftRuntimeError.invalidDataset("Dataset shard uses unsupported version \(header[1]).")
        }

        let tokenCount = Int(header[2])
        let payload = data.dropFirst(headerBytes)
        let tokens = payload.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: UInt16.self))
        }
        guard tokens.count == tokenCount else {
            throw LLMSwiftRuntimeError.invalidDataset("Dataset shard token count mismatch: header says \(tokenCount), file contains \(tokens.count).")
        }
        guard tokenCount >= batchSize * sequenceLength + 1 else {
            throw LLMSwiftRuntimeError.invalidDataset("Dataset shard does not contain enough tokens for a batch of size \(batchSize) and sequence length \(sequenceLength).")
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

actor LLMSwiftRuntime {
    private var model: LLMSwift.GPT2?
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
        model = try LLMSwift.buildModel(from: data)
    }

    func exportCheckpoint() async throws -> Data {
        try ensureModel()
        guard let model else {
            throw LLMSwiftRuntimeError.missingCheckpoint
        }
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
        guard let tokenizerData else {
            throw LLMSwiftRuntimeError.tokenizerRequired
        }

        let tokenizer = try LLMTokenizer(data: tokenizerData)
        try ensureModel()
        guard let model else {
            throw LLMSwiftRuntimeError.failedToCreateModel
        }

        let promptTokens = tokenizer.encodePrompt(request.prompt, maximumTokenCount: min(request.maximumTokenCount, model.config.max_seq_len))
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.inferenceLoop(tokenizer: tokenizer, promptTokens: promptTokens, request: request, continuation: continuation)
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

    private func trainingLoop(request: LLMTrainingRequest, continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation) async throws {
        try ensureModel()
        guard var model else {
            throw LLMSwiftRuntimeError.failedToCreateModel
        }
        guard let trainData, let validationData else {
            throw LLMSwiftRuntimeError.failedToCreateModel
        }

        var trainLoader = try LLMSwiftTokenDataLoader(data: trainData, batchSize: batchSize, sequenceLength: min(sequenceLength, model.config.max_seq_len))
        var validationLoader = try LLMSwiftTokenDataLoader(data: validationData, batchSize: batchSize, sequenceLength: min(sequenceLength, model.config.max_seq_len))
        let effectiveSequenceLength = min(sequenceLength, model.config.max_seq_len)

        for _ in 0..<request.stepCount {
            try Task.checkCancellation()
            let stepStart = ContinuousClock.now
            trainLoader.nextBatch()
            let forwardStart = ContinuousClock.now
            LLMSwift.gpt2_forward(model: &model, inputs: trainLoader.inputs, targets: trainLoader.targets, B: batchSize, T: effectiveSequenceLength)
            let forwardPassMilliseconds = forwardStart.duration(to: .now).timeInterval * 1_000
            try Task.checkCancellation()
            LLMSwift.gpt2_zero_grad(model: &model)
            let backwardStart = ContinuousClock.now
            LLMSwift.gpt2_backward(model: &model)
            let backwardPassMilliseconds = backwardStart.duration(to: .now).timeInterval * 1_000
            try Task.checkCancellation()
            LLMSwift.gpt2_update(
                model: &model,
                update_params: LLMSwift.UpdateParams(
                    learning_rate: Float(request.learningRate),
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0,
                    t: completedStepCount + 1
                )
            )
            let elapsed = stepStart.duration(to: .now)
            completedStepCount += 1
            let trainingLoss = Double(model.mean_loss)
            let validationLoss = try computeValidationLoss(using: &model, loader: &validationLoader, batchCount: request.validationBatchCount, sequenceLength: effectiveSequenceLength)

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

    private func inferenceLoop(tokenizer: LLMTokenizer, promptTokens: [Int], request: LLMInferenceRequest, continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation) async throws {
        try ensureModel()
        guard var model else {
            throw LLMSwiftRuntimeError.failedToCreateModel
        }

        let effectiveSequenceLength = min(sequenceLength, model.config.max_seq_len)
        let totalCount = min(effectiveSequenceLength, promptTokens.count + request.maximumTokenCount)
        var generationTokens = Array(repeating: UInt32(tokenizer.eotToken), count: batchSize * effectiveSequenceLength)
        for (index, token) in promptTokens.prefix(totalCount).enumerated() {
            generationTokens[index] = UInt32(token)
        }

        var generatedTokenCount = 0
        var state = UInt64.random(in: UInt64.min...UInt64.max)
        let startIndex = min(promptTokens.count, totalCount)
        guard startIndex < totalCount else {
            return
        }

        for tokenIndex in startIndex..<totalCount {
            try Task.checkCancellation()
            LLMSwift.gpt2_forward(model: &model, inputs: generationTokens, targets: [], B: batchSize, T: effectiveSequenceLength)
            let rowStart = max(0, tokenIndex - 1) * model.config.padded_vocab_size
            let rowEnd = rowStart + model.config.vocab_size
            let nextToken = LLMSwift.sample(logits: model.acts.logits[rowStart..<rowEnd], temperature: request.temperature, state: &state)
            generationTokens[tokenIndex] = UInt32(nextToken)
            generatedTokenCount += 1
            continuation.yield(LLMInferenceChunk(text: tokenizer.decode(token: nextToken), generatedTokenCount: generatedTokenCount))
        }

        self.model = model
    }

    private func ensureModel() throws {
        if model == nil {
            try recreateModel()
        }
    }

    private func recreateModel() throws {
        guard let currentCheckpointData else {
            throw LLMSwiftRuntimeError.missingCheckpoint
        }
        model = try LLMSwift.buildModel(from: currentCheckpointData)
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

    private func computeValidationLoss(using model: inout LLMSwift.GPT2, loader: inout LLMSwiftTokenDataLoader, batchCount: Int, sequenceLength: Int) throws -> Double? {
        guard batchCount > 0 else {
            return nil
        }

        loader.reset()
        var total: Double = 0
        for _ in 0..<batchCount {
            try Task.checkCancellation()
            loader.nextBatch()
            LLMSwift.gpt2_forward(model: &model, inputs: loader.inputs, targets: loader.targets, B: batchSize, T: sequenceLength)
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
