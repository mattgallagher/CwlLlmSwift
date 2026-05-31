import Foundation

enum LLMMPSRuntimeError: LocalizedError {
    case missingCheckpoint
    case tokenizerRequired
    case noMetalDevice
    case failedToCreateCommandQueue

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint:
            return "A checkpoint is required before running the MPS backend."
        case .tokenizerRequired:
            return "Inference requires a tokenizer asset."
        case .noMetalDevice:
            return "No Metal device available."
        case .failedToCreateCommandQueue:
            return "Failed to create a Metal command queue."
        }
    }
}

extension GPT2MPS: LLMTrainingSequenceLengthProviding {
    var maximumTrainingSequenceLength: Int { config.max_seq_len }
}

extension LLMMPSRuntime: LLMTrainingStreamRuntime, LLMInferenceStreamRuntime {
    typealias Model = GPT2MPS

    nonisolated var descriptor: LLMEngineDescriptor { LLMEngineDescriptor(
        id: .mpsGraph,
        displayName: "MPSGraph",
        summary: "Metal graph backend that compiles the GPT-2 forward and backward passes into MPSGraph executables for Apple GPU execution.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "MPSGraph backend supports training, generation, and checkpoint export/load via Metal Performance Shaders."
    ) }
}

actor LLMMPSRuntime {
    private var currentCheckpointData: Data?
    private var tokenizerData: Data?

    func loadCheckpoint(data: Data) async throws {
        currentCheckpointData = data
    }

    func exportCheckpoint() async throws -> Data {
        guard let currentCheckpointData else {
            throw LLMMPSRuntimeError.missingCheckpoint
        }
        return currentCheckpointData
    }

    func prepareTraining(request: LLMTrainingRequest) async throws -> GPT2MPS {
        tokenizerData = request.tokenizerData
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        return try recreateModel()
    }

    func prepareInference(request: LLMInferenceRequest) async throws -> LLMInferenceContext<GPT2MPS> {
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        if let tokenizerData = request.tokenizerData {
            self.tokenizerData = tokenizerData
        }
        guard let tokenizerData else {
            throw LLMMPSRuntimeError.tokenizerRequired
        }

        let tokenizer = try LLMTokenizer(data: tokenizerData)
        let model = try recreateModel()

        let promptTokens = tokenizer.encodePrompt(request.prompt, maximumTokenCount: model.config.max_seq_len)
        return LLMInferenceContext(model: model, tokenizer: tokenizer, promptTokens: promptTokens)
    }

    func trainingLoop(model: inout GPT2MPS, request: LLMTrainingRequest, preparationStart: ContinuousClock.Instant, continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation) async throws {
        try await LLMTrainingLoop.run(
            isolatedTo: self,
            model: &model,
            request: request,
            preparationStart: preparationStart,
            continuation: continuation,
            trainStep: { _, model, trainLoader, _ in
                trainLoader.nextBatch()
                let forwardStart = ContinuousClock.now
                let loss = model.performTrainingStep(
                    inputs: trainLoader.inputs,
                    targets: trainLoader.targets,
                    B: trainLoader.batchSize,
                    T: trainLoader.sequenceLength,
                    learningRate: Float(request.learningRate)
                )
                let forwardPassMilliseconds = forwardStart.duration(to: .now).timeInterval * 1_000
                return LLMTrainingStepResult(
                    forwardPassMilliseconds: forwardPassMilliseconds,
                    trainingLoss: Double(loss)
                )
            },
            validationLoss: { engine, model, validationLoader, _, _ in
                engine.computeValidationLoss(
                    using: model,
                    loader: &validationLoader,
                    batchCount: request.validationBatchCount
                )
            },
            exportCheckpoint: { engine, model in
                engine.currentCheckpointData = try model.exportCheckpoint()
            }
        )
    }

    func inferenceLoop(context: LLMInferenceContext<GPT2MPS>, request: LLMInferenceRequest, continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation) async throws {
        let model = context.model
        let tokenizer = context.tokenizer
        let promptTokens = context.promptTokens

        let effectiveSequenceLength = min(model.config.max_seq_len, max(promptTokens.count + request.maximumTokenCount, 1))
        let totalCount = min(effectiveSequenceLength, promptTokens.count + request.maximumTokenCount)
        var generationTokens = Array(repeating: UInt32(tokenizer.eotToken), count: effectiveSequenceLength)
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
            let logits = model.performLatestTokenInference(
                inputs: generationTokens,
                rowIndex: max(0, tokenIndex - 1),
                B: 1,
                T: effectiveSequenceLength
            )
            let nextToken = LLMSwift.sample(logits: logits.prefix(model.config.vocab_size), temperature: request.temperature, state: &state)
            generationTokens[tokenIndex] = UInt32(nextToken)
            generatedTokenCount += 1
            continuation.yield(LLMInferenceChunk(text: tokenizer.decode(token: nextToken), generatedTokenCount: generatedTokenCount))
        }
    }

    private func recreateModel() throws -> GPT2MPS {
        guard let currentCheckpointData else {
            throw LLMMPSRuntimeError.missingCheckpoint
        }
        return try buildModel(from: currentCheckpointData)
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

    private func buildModel(from data: Data) throws -> GPT2MPS {
        let (header, parameters) = try LLMCheckpointCodec.decode(data)
        let config = LLMGPT2Config(header: header)
        let parameterData = parameters.withUnsafeBufferPointer { Data(buffer: $0) }
        return try GPT2MPS(config: config, parameterData: parameterData)
    }

    private func computeValidationLoss(using model: GPT2MPS, loader: inout LLMTokenDataLoader, batchCount: Int) -> Double? {
        guard batchCount > 0 else {
            return nil
        }

        loader.reset()
        var total: Double = 0
        for _ in 0..<batchCount {
            loader.nextBatch()
            let loss = model.performLossEstimation(inputs: loader.inputs, targets: loader.targets, B: loader.batchSize, T: loader.sequenceLength)
            total += Double(loss)
        }
        return total / Double(batchCount)
    }
}
