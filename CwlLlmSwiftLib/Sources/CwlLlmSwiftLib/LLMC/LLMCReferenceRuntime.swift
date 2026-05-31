import CLLMCReference
import Foundation

enum LLMCReferenceRuntimeError: LocalizedError {
    case missingCheckpoint
    case failedToCreateSession
    case failedToWriteCheckpoint
    case tokenizerRequired

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint:
            return "A checkpoint is required before running the llm.c reference backend."
        case .failedToCreateSession:
            return "Failed to create the llm.c reference session."
        case .failedToWriteCheckpoint:
            return "Failed to export the llm.c checkpoint."
        case .tokenizerRequired:
            return "Inference requires a tokenizer asset."
        }
    }
}

extension LLMCReferenceRuntime: LLMTrainingStreamRuntime {
    typealias Model = LLMCReferenceSession

    nonisolated var descriptor: LLMEngineDescriptor { LLMEngineDescriptor(
        id: .cReference,
        displayName: "llm.c",
        summary: "Vendored CPU reference path from train_gpt2.c for training, inference, checkpointing, and numerical validation.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "Reference backend supports training, generation, and checkpoint export/load."
    ) }
}

final class LLMCReferenceSession {
    private let model = UnsafeMutablePointer<GPT2>.allocate(capacity: 1)
    private let trainLoader = UnsafeMutablePointer<DataLoader>.allocate(capacity: 1)
    private let validationLoader = UnsafeMutablePointer<DataLoader>.allocate(capacity: 1)
    private let baseURL: URL
    private(set) var hasTrainLoader = false
    private(set) var hasValidationLoader = false
    let batchSize: Int
    let sequenceLength: Int
    let vocabSize: Int
    let paddedVocabSize: Int

    init(checkpointData: Data, trainData: Data?, validationData: Data?, batchSize: Int, sequenceLength: Int) throws {
        self.baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        self.batchSize = batchSize

        model.initialize(to: GPT2())
        trainLoader.initialize(to: DataLoader())
        validationLoader.initialize(to: DataLoader())

        try FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
        let checkpointURL = baseURL.appendingPathComponent("checkpoint.bin")
        try checkpointData.write(to: checkpointURL)

        gpt2_build_from_checkpoint(model, checkpointURL.path)
        self.sequenceLength = min(sequenceLength, Int(model.pointee.config.max_seq_len))
        self.vocabSize = Int(model.pointee.config.vocab_size)
        self.paddedVocabSize = Int(model.pointee.config.padded_vocab_size)

        if let trainData {
            let trainURL = baseURL.appendingPathComponent("train.bin")
            try trainData.write(to: trainURL)
            dataloader_init(trainLoader, trainURL.path, batchSize, self.sequenceLength, 0, 1, 0)
            hasTrainLoader = true
        }

        if let validationData {
            let validationURL = baseURL.appendingPathComponent("val.bin")
            try validationData.write(to: validationURL)
            dataloader_init(validationLoader, validationURL.path, batchSize, self.sequenceLength, 0, 1, 0)
            hasValidationLoader = true
        }
    }

    deinit {
        if hasTrainLoader {
            dataloader_free(trainLoader)
        }
        if hasValidationLoader {
            dataloader_free(validationLoader)
        }
        if model.pointee.params_memory != nil {
            gpt2_free(model)
        }
        model.deallocate()
        trainLoader.deallocate()
        validationLoader.deallocate()
        try? FileManager.default.removeItem(at: baseURL)
    }

    func trainStep(step: Int, learningRate: Double, validationBatchCount: Int) throws -> LLMTrainingProgress {
        var stepStart = timespec()
        var forwardStart = timespec()
        var forwardEnd = timespec()
        var backwardStart = timespec()
        var backwardEnd = timespec()
        var stepEnd = timespec()
        clock_gettime(CLOCK_MONOTONIC, &stepStart)

        dataloader_next_batch(trainLoader)
        clock_gettime(CLOCK_MONOTONIC, &forwardStart)
        gpt2_forward(model, trainLoader.pointee.inputs, trainLoader.pointee.targets, batchSize, sequenceLength)
        clock_gettime(CLOCK_MONOTONIC, &forwardEnd)
        try Task.checkCancellation()
        gpt2_zero_grad(model)
        clock_gettime(CLOCK_MONOTONIC, &backwardStart)
        gpt2_backward(model)
        clock_gettime(CLOCK_MONOTONIC, &backwardEnd)
        try Task.checkCancellation()
        gpt2_update(model, Float(learningRate), 0.9, 0.999, 1e-8, 0, Int32(step))

        clock_gettime(CLOCK_MONOTONIC, &stepEnd)
        let totalElapsedSeconds = elapsedSeconds(from: stepStart, to: stepEnd)
        let trainingLoss = Double(model.pointee.mean_loss)
        let validationLoss = try validationBatchCount > 0 ? computeValidationLoss(batchCount: validationBatchCount) : nil
        return LLMTrainingProgress(
            step: step,
            iterationsPerSecond: totalElapsedSeconds > 0 ? 1 / totalElapsedSeconds : nil,
            forwardPassMilliseconds: elapsedSeconds(from: forwardStart, to: forwardEnd) * 1_000,
            backwardPassMilliseconds: elapsedSeconds(from: backwardStart, to: backwardEnd) * 1_000,
            trainingLoss: trainingLoss,
            validationLoss: validationLoss
        )
    }

    func generate(promptTokens: [Int], maximumTokenCount: Int, temperature: Double, eotToken: Int) -> [Int] {
        let totalCount = min(sequenceLength, promptTokens.count + maximumTokenCount)
        var generationTokens = Array(repeating: Int32(eotToken), count: batchSize * sequenceLength)
        for (index, token) in promptTokens.prefix(totalCount).enumerated() {
            generationTokens[index] = Int32(token)
        }

        var generated = Array(promptTokens.prefix(totalCount))
        var rng = UInt64.random(in: UInt64.min...UInt64.max)
        let startIndex = generated.count
        guard startIndex < totalCount else {
            return generated
        }

        for tokenIndex in startIndex..<totalCount {
            generationTokens.withUnsafeMutableBufferPointer { buffer in
                gpt2_forward(model, buffer.baseAddress, nil, batchSize, sequenceLength)
            }
            let logits = UnsafeBufferPointer(
                start: model.pointee.acts.logits + (tokenIndex - 1) * paddedVocabSize,
                count: vocabSize
            )
            let nextToken = sample(from: logits, temperature: temperature, rng: &rng)
            generationTokens[tokenIndex] = Int32(nextToken)
            generated.append(nextToken)
        }

        return generated
    }

    func prepareGenerationTokens(promptTokens: [Int], maximumTokenCount: Int, eotToken: Int) -> ([Int32], Int, Int) {
        let totalCount = min(sequenceLength, promptTokens.count + maximumTokenCount)
        var generationTokens = Array(repeating: Int32(eotToken), count: batchSize * sequenceLength)
        for (index, token) in promptTokens.prefix(totalCount).enumerated() {
            generationTokens[index] = Int32(token)
        }
        return (generationTokens, min(promptTokens.count, totalCount), totalCount)
    }

    func generateNextToken(generationTokens: inout [Int32], tokenIndex: Int, temperature: Double, rng: inout UInt64) -> Int {
        generationTokens.withUnsafeMutableBufferPointer { buffer in
            gpt2_forward(model, buffer.baseAddress, nil, batchSize, sequenceLength)
        }
        let logits = UnsafeBufferPointer(
            start: model.pointee.acts.logits + (tokenIndex - 1) * paddedVocabSize,
            count: vocabSize
        )
        let nextToken = sample(from: logits, temperature: temperature, rng: &rng)
        generationTokens[tokenIndex] = Int32(nextToken)
        return nextToken
    }

    func exportCheckpoint() throws -> Data {
        guard let paramsMemory = model.pointee.params_memory else {
            throw LLMCReferenceRuntimeError.failedToWriteCheckpoint
        }

        let header = LLMCheckpointHeader(
            maxSequenceLength: Int(model.pointee.config.max_seq_len),
            vocabularySize: Int(model.pointee.config.vocab_size),
            layerCount: Int(model.pointee.config.num_layers),
            headCount: Int(model.pointee.config.num_heads),
            channelCount: Int(model.pointee.config.channels),
            paddedVocabularySize: Int(model.pointee.config.padded_vocab_size)
        )
        let parameterCount = Int(model.pointee.num_parameters)
        let parameters = Array(UnsafeBufferPointer(start: paramsMemory, count: parameterCount))
        return try LLMCheckpointCodec.encode(header: header, parameters: parameters, expectedParameterCount: parameterCount)
    }

    private func computeValidationLoss(batchCount: Int) throws -> Double? {
        guard hasValidationLoader else {
            return nil
        }

        dataloader_reset(validationLoader)
        var total: Double = 0
        for _ in 0..<batchCount {
            try Task.checkCancellation()
            dataloader_next_batch(validationLoader)
            gpt2_forward(model, validationLoader.pointee.inputs, validationLoader.pointee.targets, batchSize, sequenceLength)
            total += Double(model.pointee.mean_loss)
        }
        return total / Double(batchCount)
    }

    private func sample(from logits: UnsafeBufferPointer<Float>, temperature: Double, rng: inout UInt64) -> Int {
        if temperature <= 0 {
            return logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }

        let maxLogit = logits.max() ?? 0
        var probabilities = logits.map { exp(Double(($0 - maxLogit) / Float(temperature))) }
        let sum = probabilities.reduce(0, +)
        guard sum > 0 else {
            return 0
        }
        for index in probabilities.indices {
            probabilities[index] /= sum
        }

        rng ^= rng >> 12
        rng ^= rng << 25
        rng ^= rng >> 27
        let sample = Double((rng & 0xFFFFFFFF)) / Double(UInt32.max)

        var cumulative = 0.0
        for (index, probability) in probabilities.enumerated() {
            cumulative += probability
            if sample <= cumulative {
                return index
            }
        }
        return max(0, probabilities.count - 1)
    }
}

private struct LLMCReferenceInferenceContext: @unchecked Sendable {
    let session: LLMCReferenceSession
    let tokenizer: LLMTokenizer
    let promptTokens: [Int]
}

actor LLMCReferenceRuntime {
    private var currentCheckpointData: Data?
    private var tokenizerData: Data?
    private var batchSize = 4
    private var sequenceLength = 64

    func loadCheckpoint(data: Data) async throws {
        currentCheckpointData = data
    }

    func exportCheckpoint() async throws -> Data {
        guard let currentCheckpointData else {
            throw LLMCReferenceRuntimeError.missingCheckpoint
        }
        return currentCheckpointData
    }

    func prepareTraining(request: LLMTrainingRequest) async throws -> LLMCReferenceSession {
        tokenizerData = request.tokenizerData
        batchSize = request.batchSize
        sequenceLength = request.sequenceLength
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        guard let currentCheckpointData else {
            throw LLMCReferenceRuntimeError.missingCheckpoint
        }
        return try LLMCReferenceSession(
            checkpointData: currentCheckpointData,
            trainData: request.trainData,
            validationData: request.validationData,
            batchSize: batchSize,
            sequenceLength: sequenceLength
        )
    }

    fileprivate func prepareInference(request: LLMInferenceRequest) async throws -> LLMCReferenceInferenceContext {
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        if let tokenizerData = request.tokenizerData {
            self.tokenizerData = tokenizerData
        }
        guard let tokenizerData else {
            throw LLMCReferenceRuntimeError.tokenizerRequired
        }
        let tokenizer = try LLMTokenizer(data: tokenizerData)
        let session = try recreateSession()

        let promptTokens = tokenizer.encodePrompt(request.prompt, maximumTokenCount: session.sequenceLength)
        return LLMCReferenceInferenceContext(session: session, tokenizer: tokenizer, promptTokens: promptTokens)
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        let context = try await prepareInference(request: request)

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.inferenceLoop(context: context, request: request, continuation: continuation)
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

    func trainingLoop(
        model session: inout LLMCReferenceSession,
        request: LLMTrainingRequest,
        preparationStart: ContinuousClock.Instant,
        continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation
    ) async throws {
        guard request.stepCount > 0 else {
            currentCheckpointData = try session.exportCheckpoint()
            return
        }

        for step in 1...request.stepCount {
            try Task.checkCancellation()
            let stepStart = ContinuousClock.now
            let progress = try session.trainStep(step: step, learningRate: request.learningRate, validationBatchCount: request.validationBatchCount)
            let elapsed = step == 1
                ? preparationStart.duration(to: .now)
                : stepStart.duration(to: .now)
            continuation.yield(
                LLMTrainingProgress(
                    step: step,
                    iterationsPerSecond: elapsed.iterationsPerSecond,
                    forwardPassMilliseconds: progress.forwardPassMilliseconds,
                    backwardPassMilliseconds: progress.backwardPassMilliseconds,
                    trainingLoss: progress.trainingLoss,
                    validationLoss: progress.validationLoss
                )
            )
        }

        currentCheckpointData = try session.exportCheckpoint()
    }

    fileprivate func inferenceLoop(
        context: LLMCReferenceInferenceContext,
        request: LLMInferenceRequest,
        continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation
    ) async throws {
        let session = context.session
        let tokenizer = context.tokenizer
        let promptTokens = context.promptTokens
        var (generationTokens, startIndex, totalCount) = session.prepareGenerationTokens(
            promptTokens: promptTokens,
            maximumTokenCount: request.maximumTokenCount,
            eotToken: tokenizer.eotToken
        )
        guard startIndex < totalCount else {
            return
        }

        var generatedTokenCount = 0
        var rng = UInt64.random(in: UInt64.min...UInt64.max)
        for tokenIndex in startIndex..<totalCount {
            try Task.checkCancellation()
            let nextToken = session.generateNextToken(
                generationTokens: &generationTokens,
                tokenIndex: tokenIndex,
                temperature: request.temperature,
                rng: &rng
            )
            generatedTokenCount += 1
            continuation.yield(
                LLMInferenceChunk(
                    text: tokenizer.decode(token: nextToken),
                    generatedTokenCount: generatedTokenCount
                )
            )
        }
    }

    private func recreateSession() throws -> LLMCReferenceSession {
        guard let checkpointData = currentCheckpointData else {
            throw LLMCReferenceRuntimeError.missingCheckpoint
        }

        return try LLMCReferenceSession(
            checkpointData: checkpointData,
            trainData: nil,
            validationData: nil,
            batchSize: batchSize,
            sequenceLength: sequenceLength
        )
    }

    // Drops request-scoped training assets after a run while retaining the
    // latest checkpoint for export or follow-up inference.
    func releaseTrainingState() async {
        tokenizerData = nil
    }

    // Public counterpart to `releaseTrainingState` that also drops the cached
    // checkpoint, so a fully idle engine retains effectively nothing. Callers
    // (e.g. the Comparison screen) invoke this when they're done with an
    // engine for now and don't intend to immediately call exportCheckpoint.
    func releaseResources() async {
        await releaseTrainingState()
        currentCheckpointData = nil
    }
}

private func elapsedSeconds(from start: timespec, to end: timespec) -> Double {
    Double(end.tv_sec - start.tv_sec) + Double(end.tv_nsec - start.tv_nsec) / 1e9
}
