import CoreML
import Foundation

enum LLMMLTensorRuntimeError: LocalizedError {
    case missingCheckpoint
    case failedToCreateModel
    case tokenizerRequired

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint:
            return "A checkpoint is required before running the MLTensor backend."
        case .failedToCreateModel:
            return "Failed to create the MLTensor model runtime."
        case .tokenizerRequired:
            return "Inference requires a tokenizer asset."
        }
    }
}

extension GPT2MLTensor: LLMTrainingSequenceLengthProviding {
    var maximumTrainingSequenceLength: Int { config.max_seq_len }
}

extension LLMMLTensorRuntime: LLMTrainingStreamRuntime, LLMInferenceStreamRuntime {
    typealias Model = GPT2MLTensor

    nonisolated var descriptor: LLMEngineDescriptor { LLMEngineDescriptor(
        id: .mlTensor,
        displayName: "MLTensor",
        summary: "MLTensor backend that runs the GPT-2 forward and backward passes using Apple's MLTensor compute graph.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "MLTensor backend supports training, generation, and checkpoint export/load via Apple's MLTensor compute graph."
    ) }
}

actor LLMMLTensorRuntime {
    private static let computePolicy = MLComputePolicy.cpuAndGPU
    private var currentCheckpointData: Data?
    private var tokenizerData: Data?
    private var batchSize = 4
    private var sequenceLength = 64

    // Validation runs a full extra forward pass per call. MLTensor has high
    // per-call dispatch overhead and no graph fusion, so doing this every
    // step roughly doubles training time. Cadence > 1 means validation is
    // only computed on multiples of this value (and the final step). Other
    // engines validate every step; the MLTensor number is therefore not
    // directly comparable when this is > 1.
    private static let validationStepInterval = 5

    func loadCheckpoint(data: Data) async throws {
        currentCheckpointData = data
    }

    func exportCheckpoint() async throws -> Data {
        guard let currentCheckpointData else {
            throw LLMMLTensorRuntimeError.missingCheckpoint
        }
        return currentCheckpointData
    }

    func prepareTraining(request: LLMTrainingRequest) async throws -> GPT2MLTensor {
        tokenizerData = request.tokenizerData
        batchSize = request.batchSize
        sequenceLength = request.sequenceLength
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        return try recreateModel()
    }

    func prepareInference(request: LLMInferenceRequest) async throws -> LLMInferenceContext<GPT2MLTensor> {
        if let checkpointData = request.checkpointData {
            currentCheckpointData = checkpointData
        }
        if let tokenizerData = request.tokenizerData {
            self.tokenizerData = tokenizerData
        }
        guard let tokenizerData else {
            throw LLMMLTensorRuntimeError.tokenizerRequired
        }

        let tokenizer = try LLMTokenizer(data: tokenizerData)
        let model = try recreateModel()

        batchSize = 1
        sequenceLength = min(model.config.max_seq_len, max(1, request.maximumTokenCount))
        let promptTokens = tokenizer.encodePrompt(request.prompt, maximumTokenCount: model.config.max_seq_len)
        return LLMInferenceContext(model: model, tokenizer: tokenizer, promptTokens: promptTokens)
    }

    func trainingLoop(model: inout GPT2MLTensor, request: LLMTrainingRequest, preparationStart: ContinuousClock.Instant, continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation) async throws {
        try await LLMTrainingLoop.run(
            isolatedTo: self,
            model: &model,
            request: request,
            preparationStart: preparationStart,
            continuation: continuation,
            trainStep: { _, model, trainLoader, optimizerStep in
                let stepStart = ContinuousClock.now
                trainLoader.nextBatch()

                // Build the full forward + backward + AdamW graph without forcing
                // GPU evaluation. None of these calls actually materialize values;
                // they all just compose MLTensor graph nodes.
                let lossTensor = withMLTensorComputePolicy(Self.computePolicy) {
                    let (acts, _) = model.forward(inputs: trainLoader.inputs, targets: trainLoader.targets, B: trainLoader.batchSize, T: trainLoader.sequenceLength)
                    let grads = model.backwardSync(acts: acts, inputs: trainLoader.inputs, targets: trainLoader.targets, B: trainLoader.batchSize, T: trainLoader.sequenceLength)
                    LLMMLTensor.adamw_update(
                        params: &model.params, grads: grads, m: &model.m_memory, v: &model.v_memory,
                        learningRate: Float(request.learningRate), beta1: 0.9, beta2: 0.999, eps: 1e-8,
                        weightDecay: 0, t: optimizerStep
                    )
                    return acts.losses
                }

                // Single sync point per step: forces evaluation of forward + backward
                // + optimizer apply as one fused graph submission.
                let meanLoss = await lossTensor.mean().shapedArray(of: Float.self).scalar ?? 0
                let trainComputeMilliseconds = stepStart.duration(to: .now).timeInterval * 1_000

                return LLMTrainingStepResult(
                    forwardPassMilliseconds: trainComputeMilliseconds,
                    backwardPassMilliseconds: nil,
                    trainingLoss: Double(meanLoss)
                )
            },
            validationLoss: { engine, model, validationLoader, completedStep, isLastStep in
                // Validation gating — see Self.validationStepInterval.
                let shouldValidate = request.validationBatchCount > 0 && (isLastStep || completedStep % Self.validationStepInterval == 0)
                return shouldValidate
                    ? await engine.computeValidationLoss(using: model, loader: &validationLoader, batchCount: request.validationBatchCount, sequenceLength: validationLoader.sequenceLength)
                    : nil
            },
            exportCheckpoint: { engine, model in
                engine.currentCheckpointData = try await model.exportCheckpoint()
            }
        )
    }

    func inferenceLoop(context: LLMInferenceContext<GPT2MLTensor>, request: LLMInferenceRequest, continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation) async throws {
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
            // Only forward over the populated prefix (positions 0..<currentT)
            // rather than the full padded `effectiveSequenceLength`. This drops
            // the per-token cost from O(maxT^2) to O(tokenIndex^2) in attention
            // and shrinks every other op proportionally.
            let currentT = max(tokenIndex, 1)
            let inputSlice = Array(generationTokens.prefix(currentT))
            let logitsTensor = withMLTensorComputePolicy(Self.computePolicy) {
                model.performInference(inputs: inputSlice, B: 1, T: currentT)
            }
            // Slice the single relevant row on-GPU before materializing scalars,
            // so we transfer V floats instead of the full 1*currentT*V tensor.
            let lastRowIndex = currentT - 1
            let lastRowTensor = logitsTensor[0, lastRowIndex, 0...]
            let logits = await lastRowTensor.shapedArray(of: Float.self).scalars
            let nextToken = LLMSwift.sample(logits: ArraySlice(logits), temperature: request.temperature, state: &state)
            generationTokens[tokenIndex] = UInt32(nextToken)
            generatedTokenCount += 1
            continuation.yield(LLMInferenceChunk(text: tokenizer.decode(token: nextToken), generatedTokenCount: generatedTokenCount))
        }
    }

    private func recreateModel() throws -> GPT2MLTensor {
        guard let currentCheckpointData else {
            throw LLMMLTensorRuntimeError.missingCheckpoint
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

    private func buildModel(from data: Data) throws -> GPT2MLTensor {
        let (header, parameters) = try LLMCheckpointCodec.decode(data)
        let config = LLMGPT2Config(header: header)
        return GPT2MLTensor(config: config, parameters: parameters)
    }

    private func computeValidationLoss(using model: GPT2MLTensor, loader: inout LLMTokenDataLoader, batchCount: Int, sequenceLength: Int) async -> Double? {
        guard batchCount > 0 else {
            return nil
        }

        loader.reset()
        var total: Double = 0
        for _ in 0..<batchCount {
            loader.nextBatch()
            let lossTensor = withMLTensorComputePolicy(Self.computePolicy) {
                model.lossTensor(inputs: loader.inputs, targets: loader.targets, B: batchSize, T: sequenceLength)
            }
            let loss = await lossTensor.mean().shapedArray(of: Float.self).scalar ?? 0
            total += Double(loss)
        }
        return total / Double(batchCount)
    }
}
