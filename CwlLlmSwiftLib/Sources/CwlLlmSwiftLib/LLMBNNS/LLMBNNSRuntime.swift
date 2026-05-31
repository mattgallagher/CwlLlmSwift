import Foundation

enum LLMBNNSRuntimeError: LocalizedError {
    case missingCheckpoint
    case failedToCreateModel
    case tokenizerRequired

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint:
            return "A checkpoint is required before running the BNNS backend."
        case .failedToCreateModel:
            return "Failed to create the BNNS model runtime."
        case .tokenizerRequired:
            return "Inference requires a tokenizer asset."
        }
    }
}

extension LLMBNNS.GraphModel: LLMTrainingSequenceLengthProviding {
    var maximumTrainingSequenceLength: Int { config.max_seq_len }
}

extension LLMBNNSRuntime: LLMTrainingStreamRuntime, LLMInferenceStreamRuntime {
    typealias Model = LLMBNNS.GraphModel

    nonisolated var descriptor: LLMEngineDescriptor { LLMEngineDescriptor(
        id: .bnns,
        displayName: "Accelerate BNNS",
        summary: "Accelerate BNNS graph backend that compiles the GPT-2 forward pass into a fixed-shape BNNSGraph context and trains with shared reverse-pass tensors.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "BNNS backend currently uses a compiled BNNS forward graph with shared Swift tensor storage for backward and optimizer state."
    ) }
}

actor LLMBNNSRuntime {
    private var currentCheckpointData: Data?
    private var tokenizerData: Data?
    private var batchSize = 4
    private var sequenceLength = 64

    func loadCheckpoint(data: Data) async throws {
        currentCheckpointData = data
    }

    func exportCheckpoint() async throws -> Data {
        guard let currentCheckpointData else {
            throw LLMBNNSRuntimeError.missingCheckpoint
        }
        return currentCheckpointData
    }

    func prepareTraining(request: LLMTrainingRequest) async throws -> LLMBNNS.GraphModel {
        tokenizerData = request.tokenizerData
        batchSize = request.batchSize
        sequenceLength = request.sequenceLength
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        return try recreateModel()
    }

    func prepareInference(request: LLMInferenceRequest) async throws -> LLMInferenceContext<LLMBNNS.GraphModel> {
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        if let tokenizerData = request.tokenizerData {
            self.tokenizerData = tokenizerData
        }
        guard let tokenizerData else {
            throw LLMBNNSRuntimeError.tokenizerRequired
        }

        let tokenizer = try LLMTokenizer(data: tokenizerData)
        let model = try recreateModel()

        batchSize = 1
        sequenceLength = min(model.config.max_seq_len, max(1, request.maximumTokenCount))
        let promptTokens = tokenizer.encodePrompt(request.prompt, maximumTokenCount: model.config.max_seq_len)
        return LLMInferenceContext(model: model, tokenizer: tokenizer, promptTokens: promptTokens)
    }

    func inferenceLoop(context: LLMInferenceContext<LLMBNNS.GraphModel>, request: LLMInferenceRequest, continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation) async throws {
        var model = context.model
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
            let logits = try LLMBNNS.gpt2_latest_token_logits(
                model: &model,
                inputs: generationTokens,
                rowIndex: max(0, tokenIndex - 1),
                B: 1,
                T: effectiveSequenceLength
            )
            let nextToken = LLMBNNS.sample(logits: logits, temperature: request.temperature, state: &state)
            generationTokens[tokenIndex] = UInt32(nextToken)
            generatedTokenCount += 1
            continuation.yield(LLMInferenceChunk(text: tokenizer.decode(token: nextToken), generatedTokenCount: generatedTokenCount))
        }
    }

    func trainingLoop(model: inout LLMBNNS.GraphModel, request: LLMTrainingRequest, preparationStart: ContinuousClock.Instant, continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation) async throws {
        try await LLMTrainingLoop.run(
            isolatedTo: self,
            model: &model,
            request: request,
            preparationStart: preparationStart,
            continuation: continuation,
            trainStep: { _, model, trainLoader, optimizerStep in
                trainLoader.nextBatch()
                let forwardStart = ContinuousClock.now
                try LLMBNNS.gpt2_forward(model: &model, inputs: trainLoader.inputs, targets: trainLoader.targets, B: trainLoader.batchSize, T: trainLoader.sequenceLength)
                let forwardPassMilliseconds = forwardStart.duration(to: .now).timeInterval * 1_000
                LLMBNNS.gpt2_zero_grad(model: &model)
                let backwardStart = ContinuousClock.now
                LLMBNNS.gpt2_backward(model: &model)
                let backwardPassMilliseconds = backwardStart.duration(to: .now).timeInterval * 1_000
                LLMBNNS.gpt2_update(
                    model: &model,
                    update_params: LLMBNNS.UpdateParams(
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

    private func recreateModel() throws -> LLMBNNS.GraphModel {
        guard let currentCheckpointData else {
            throw LLMBNNSRuntimeError.missingCheckpoint
        }
        return try LLMBNNS.buildModel(from: currentCheckpointData)
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

    private func computeValidationLoss(using model: inout LLMBNNS.GraphModel, loader: inout LLMTokenDataLoader, batchCount: Int, sequenceLength: Int) throws -> Double? {
        guard batchCount > 0 else {
            return nil
        }

        loader.reset()
        var total: Double = 0
        for _ in 0..<batchCount {
            loader.nextBatch()
            try LLMBNNS.gpt2_forward(model: &model, inputs: loader.inputs, targets: loader.targets, B: batchSize, T: sequenceLength)
            total += Double(model.mean_loss)
        }
        return total / Double(batchCount)
    }
}
