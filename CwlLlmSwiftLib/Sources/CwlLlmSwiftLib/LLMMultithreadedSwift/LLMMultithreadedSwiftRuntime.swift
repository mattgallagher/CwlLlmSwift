import Foundation

enum LLMMultithreadedSwiftRuntimeError: LocalizedError {
    case missingCheckpoint
    case failedToCreateModel
    case tokenizerRequired

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint:
            return "A checkpoint is required before running the multithreaded Swift backend."
        case .failedToCreateModel:
            return "Failed to create the multithreaded Swift model runtime."
        case .tokenizerRequired:
            return "Inference requires a tokenizer asset."
        }
    }
}

extension LLMMultithreadedSwiftRuntime: LLMTrainingStreamRuntime, LLMInferenceStreamRuntime {
    typealias Model = LLMMultithreadedSwift.GPT2

    nonisolated var descriptor: LLMEngineDescriptor { LLMEngineDescriptor(
        id: .multithreadedSwift,
        displayName: "Multithreaded Swift",
        summary: "Optimized pure Swift backend derived from the fast Swift implementation, adding CPU-parallel matmul and attention kernels with Dispatch concurrent work sharing.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "Multithreaded Swift backend supports training, generation, and checkpoint export/load."
    ) }
}

actor LLMMultithreadedSwiftRuntime {
    private var currentCheckpointData: Data?
    private var tokenizerData: Data?
    private var batchSize = 4
    private var sequenceLength = 64

    func loadCheckpoint(data: Data) async throws {
        currentCheckpointData = data
    }

    func exportCheckpoint() async throws -> Data {
        guard let currentCheckpointData else {
            throw LLMMultithreadedSwiftRuntimeError.missingCheckpoint
        }
        return currentCheckpointData
    }

    func prepareTraining(request: LLMTrainingRequest) async throws -> LLMMultithreadedSwift.GPT2 {
        tokenizerData = request.tokenizerData
        batchSize = request.batchSize
        sequenceLength = request.sequenceLength
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        return try recreateModel()
    }

    func prepareInference(request: LLMInferenceRequest) async throws -> LLMInferenceContext<LLMMultithreadedSwift.GPT2> {
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        if let tokenizerData = request.tokenizerData {
            self.tokenizerData = tokenizerData
        }
        guard let tokenizerData else {
            throw LLMMultithreadedSwiftRuntimeError.tokenizerRequired
        }

        let tokenizer = try LLMTokenizer(data: tokenizerData)
        let model = try recreateModel()

        let promptTokens = tokenizer.encodePrompt(request.prompt, maximumTokenCount: min(request.maximumTokenCount, model.config.max_seq_len))
        return LLMInferenceContext(model: model, tokenizer: tokenizer, promptTokens: promptTokens)
    }

    func trainingLoop(model: inout LLMMultithreadedSwift.GPT2, request: LLMTrainingRequest, preparationStart: ContinuousClock.Instant, continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation) async throws {
        try await LLMTrainingLoop.run(
            isolatedTo: self,
            model: &model,
            request: request,
            preparationStart: preparationStart,
            continuation: continuation,
            trainStep: { _, model, trainLoader, optimizerStep in
                trainLoader.nextBatch()
                let forwardStart = ContinuousClock.now
                LLMMultithreadedSwift.gpt2_forward(model: &model, inputs: trainLoader.inputs, targets: trainLoader.targets, B: trainLoader.batchSize, T: trainLoader.sequenceLength)
                let forwardPassMilliseconds = forwardStart.duration(to: .now).timeInterval * 1_000
                try Task.checkCancellation()
                LLMMultithreadedSwift.gpt2_zero_grad(model: &model)
                let backwardStart = ContinuousClock.now
                LLMMultithreadedSwift.gpt2_backward(model: &model)
                let backwardPassMilliseconds = backwardStart.duration(to: .now).timeInterval * 1_000
                try Task.checkCancellation()
                LLMMultithreadedSwift.gpt2_update(
                    model: &model,
                    update_params: LLMMultithreadedSwift.UpdateParams(
                        learning_rate: Float(request.learningRate),
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        weight_decay: 0,
                        t: optimizerStep
                    )
                )
                return LLMTrainingStepResult(
                    forwardPassMilliseconds: forwardPassMilliseconds,
                    backwardPassMilliseconds: backwardPassMilliseconds,
                    trainingLoss: Double(model.mean_loss)
                )
            },
            validationLoss: { engine, model, validationLoader, _, _ in
                try engine.computeValidationLoss(
                    using: &model,
                    loader: &validationLoader,
                    batchCount: request.validationBatchCount,
                    sequenceLength: validationLoader.sequenceLength
                )
            },
            exportCheckpoint: { engine, model in
                engine.currentCheckpointData = try model.exportCheckpoint()
            }
        )
    }

    func inferenceLoop(context: LLMInferenceContext<LLMMultithreadedSwift.GPT2>, request: LLMInferenceRequest, continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation) async throws {
        var model = context.model
        let tokenizer = context.tokenizer
        let promptTokens = context.promptTokens

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
            LLMMultithreadedSwift.gpt2_forward(model: &model, inputs: generationTokens, targets: [], B: batchSize, T: effectiveSequenceLength)
            let rowStart = max(0, tokenIndex - 1) * model.config.padded_vocab_size
            let rowEnd = rowStart + model.config.vocab_size
            let nextToken = LLMMultithreadedSwift.sample(logits: model.acts.logits[rowStart..<rowEnd], temperature: request.temperature, state: &state)
            generationTokens[tokenIndex] = UInt32(nextToken)
            generatedTokenCount += 1
            continuation.yield(LLMInferenceChunk(text: tokenizer.decode(token: nextToken), generatedTokenCount: generatedTokenCount))
        }
    }

    private func recreateModel() throws -> LLMMultithreadedSwift.GPT2 {
        guard let currentCheckpointData else {
            throw LLMMultithreadedSwiftRuntimeError.missingCheckpoint
        }
        return try LLMMultithreadedSwift.buildModel(from: currentCheckpointData)
    }

    func releaseTrainingState() async {
        tokenizerData = nil
    }

    func releaseResources() async {
        await releaseTrainingState()
        currentCheckpointData = nil
    }

    private func computeValidationLoss(using model: inout LLMMultithreadedSwift.GPT2, loader: inout LLMTokenDataLoader, batchCount: Int, sequenceLength: Int) throws -> Double? {
        guard batchCount > 0 else {
            return nil
        }

        loader.reset()
        var total: Double = 0
        for _ in 0..<batchCount {
            try Task.checkCancellation()
            loader.nextBatch()
            LLMMultithreadedSwift.gpt2_forward(model: &model, inputs: loader.inputs, targets: loader.targets, B: batchSize, T: sequenceLength)
            total += Double(model.mean_loss)
        }
        return total / Double(batchCount)
    }
}
