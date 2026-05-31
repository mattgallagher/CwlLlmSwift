import Foundation

enum LLMTokenDataLoaderError: LocalizedError, Equatable {
    case invalidDataset(String)

    var errorDescription: String? {
        switch self {
        case .invalidDataset(let message):
            return message
        }
    }
}

struct LLMTokenDataLoader {
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
            throw LLMTokenDataLoaderError.invalidDataset("Dataset shard is too small to contain a valid header.")
        }

        let header = data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: UInt32.self).prefix(Self.headerCount))
        }
        guard header.count >= 3 else {
            throw LLMTokenDataLoaderError.invalidDataset("Dataset shard header is incomplete.")
        }
        guard header[0] == Self.magic else {
            throw LLMTokenDataLoaderError.invalidDataset("Dataset shard has invalid magic value \(header[0]).")
        }
        guard header[1] == Self.version else {
            throw LLMTokenDataLoaderError.invalidDataset("Dataset shard uses unsupported version \(header[1]).")
        }

        let tokenCount = Int(header[2])
        let payload = data.dropFirst(headerBytes)
        let tokens = payload.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: UInt16.self))
        }
        guard tokens.count == tokenCount else {
            throw LLMTokenDataLoaderError.invalidDataset("Dataset shard token count mismatch: header says \(tokenCount), file contains \(tokens.count).")
        }
        guard tokenCount >= batchSize * sequenceLength + 1 else {
            throw LLMTokenDataLoaderError.invalidDataset("Dataset shard does not contain enough tokens for a batch of size \(batchSize) and sequence length \(sequenceLength).")
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

extension Duration {
    var timeInterval: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }

    var iterationsPerSecond: Double? {
        timeInterval > 0 ? 1 / timeInterval : nil
    }
}

struct LLMTrainingStepResult {
    let forwardPassMilliseconds: Double?
    let backwardPassMilliseconds: Double?
    let trainingLoss: Double?

    init(
        forwardPassMilliseconds: Double? = nil,
        backwardPassMilliseconds: Double? = nil,
        trainingLoss: Double?
    ) {
        self.forwardPassMilliseconds = forwardPassMilliseconds
        self.backwardPassMilliseconds = backwardPassMilliseconds
        self.trainingLoss = trainingLoss
    }
}

protocol LLMTrainingSequenceLengthProviding {
    var maximumTrainingSequenceLength: Int { get }
}

extension LLMTrainingSequenceLengthProviding {
    func effectiveTrainingSequenceLength(for request: LLMTrainingRequest) -> Int {
        min(request.sequenceLength, maximumTrainingSequenceLength)
    }
}

enum LLMTrainingLoop {
    static func run<Engine: Actor, Model: LLMTrainingSequenceLengthProviding>(
        isolatedTo engine: isolated Engine,
        model: inout Model,
        request: LLMTrainingRequest,
        preparationStart: ContinuousClock.Instant,
        continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation,
        trainStep: (_ engine: isolated Engine, _ model: inout Model, _ trainLoader: inout LLMTokenDataLoader, _ optimizerStep: Int) async throws -> LLMTrainingStepResult,
        validationLoss: (_ engine: isolated Engine, _ model: inout Model, _ validationLoader: inout LLMTokenDataLoader, _ completedStep: Int, _ isLastStep: Bool) async throws -> Double?,
        exportCheckpoint: (_ engine: isolated Engine, _ model: inout Model) async throws -> Void
    ) async throws {
        let effectiveSequenceLength = model.effectiveTrainingSequenceLength(for: request)
        var trainLoader = try LLMTokenDataLoader(
            data: request.trainData,
            batchSize: request.batchSize,
            sequenceLength: effectiveSequenceLength
        )
        var validationLoader = try LLMTokenDataLoader(
            data: request.validationData,
            batchSize: request.batchSize,
            sequenceLength: effectiveSequenceLength
        )

        guard request.stepCount > 0 else {
            try await exportCheckpoint(engine, &model)
            return
        }

        for completedStepCount in 1...request.stepCount {
            try Task.checkCancellation()
            let stepStart = ContinuousClock.now
            let stepResult = try await trainStep(engine, &model, &trainLoader, completedStepCount)
            let elapsed = completedStepCount == 1
                ? preparationStart.duration(to: .now)
                : stepStart.duration(to: .now)
            let validationLoss = try await validationLoss(engine, &model, &validationLoader, completedStepCount, completedStepCount == request.stepCount)

            continuation.yield(
                LLMTrainingProgress(
                    step: completedStepCount,
                    iterationsPerSecond: elapsed.iterationsPerSecond,
                    forwardPassMilliseconds: stepResult.forwardPassMilliseconds,
                    backwardPassMilliseconds: stepResult.backwardPassMilliseconds,
                    trainingLoss: stepResult.trainingLoss,
                    validationLoss: validationLoss
                )
            )
        }

        try await exportCheckpoint(engine, &model)
    }
}
