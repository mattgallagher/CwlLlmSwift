import Foundation

public enum LLMEngineIdentifier: String, CaseIterable, Codable, Sendable {
    case basicSwift = "basic-swift"
    case fastSwift = "fast-swift"
    case multithreadedSwift = "multithreaded-swift"
    case blas = "blas"
    case amx = "amx"
    case bnns = "bnns"
    case mpsGraph = "mps-graph"
    case mlTensor = "ml-tensor"
    case metal = "metal"
    case cReference = "c-reference"
}

public struct LLMEngineCapabilities: OptionSet, Codable, Sendable, Hashable {
    public let rawValue: Int

    public static let training = LLMEngineCapabilities(rawValue: 1 << 0)
    public static let inference = LLMEngineCapabilities(rawValue: 1 << 1)
    public static let checkpointing = LLMEngineCapabilities(rawValue: 1 << 2)

    public static let none: LLMEngineCapabilities = []

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public var statusSummary: String {
        isEmpty ? "Stub" : "Partial"
    }

    public var featureSummaries: [String] {
        [
            contains(.training) ? "Training" : nil,
            contains(.inference) ? "Inference" : nil,
            contains(.checkpointing) ? "Checkpointing" : nil,
        ].compactMap { $0 }
    }
}

public struct LLMEngineDescriptor: Identifiable, Codable, Sendable, Hashable {
    public let id: LLMEngineIdentifier
    public let displayName: String
    public let summary: String
    public let capabilities: LLMEngineCapabilities
    public let isAvailable: Bool
    public let availabilityNote: String?

    public var capabilitySummary: String {
        let features = capabilities.featureSummaries
        return features.isEmpty ? "No runtime features yet" : features.joined(separator: ", ")
    }
}

public struct LLMTrainingRequest: Sendable {
    public let trainData: Data
    public let validationData: Data
    public let tokenizerData: Data?
    public let checkpointData: Data?
    public let stepCount: Int
    public let batchSize: Int
    public let sequenceLength: Int
    public let learningRate: Double
    public let validationBatchCount: Int

    public init(
        trainData: Data,
        validationData: Data,
        tokenizerData: Data? = nil,
        checkpointData: Data? = nil,
        stepCount: Int = 100,
        batchSize: Int = 1,
        sequenceLength: Int = 64,
        learningRate: Double = 3e-4,
        validationBatchCount: Int = 1
    ) {
        self.trainData = trainData
        self.validationData = validationData
        self.tokenizerData = tokenizerData
        self.checkpointData = checkpointData
        self.stepCount = stepCount
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.learningRate = learningRate
        self.validationBatchCount = validationBatchCount
    }
}

public struct LLMInferenceRequest: Sendable, Equatable {
    public let prompt: String
    public let maximumTokenCount: Int
    public let temperature: Double
    public let tokenizerData: Data?
    public let checkpointData: Data?

    public init(
        prompt: String,
        maximumTokenCount: Int = 128,
        temperature: Double = 1.0,
        tokenizerData: Data? = nil,
        checkpointData: Data? = nil
    ) {
        self.prompt = prompt
        self.maximumTokenCount = maximumTokenCount
        self.temperature = temperature
        self.tokenizerData = tokenizerData
        self.checkpointData = checkpointData
    }
}

public struct LLMTrainingProgress: Codable, Sendable, Equatable {
    public let step: Int
    public let iterationsPerSecond: Double?
    public let forwardPassMilliseconds: Double?
    public let backwardPassMilliseconds: Double?
    public let trainingLoss: Double?
    public let validationLoss: Double?

    public init(
        step: Int,
        iterationsPerSecond: Double?,
        forwardPassMilliseconds: Double? = nil,
        backwardPassMilliseconds: Double? = nil,
        trainingLoss: Double?,
        validationLoss: Double?
    ) {
        self.step = step
        self.iterationsPerSecond = iterationsPerSecond
        self.forwardPassMilliseconds = forwardPassMilliseconds
        self.backwardPassMilliseconds = backwardPassMilliseconds
        self.trainingLoss = trainingLoss
        self.validationLoss = validationLoss
    }

    public static let zero = LLMTrainingProgress(
        step: 0,
        iterationsPerSecond: nil,
        forwardPassMilliseconds: nil,
        backwardPassMilliseconds: nil,
        trainingLoss: nil,
        validationLoss: nil
    )
}

public enum LLMCheckpointKind: String, Codable, CaseIterable, Sendable {
    case initial
    case latest
    case manual
}

public struct LLMCheckpointDescriptor: Identifiable, Codable, Sendable, Equatable, Hashable {
    public let kind: LLMCheckpointKind
    public let name: String
    public let fileName: String

    public init(kind: LLMCheckpointKind, name: String, fileName: String) {
        self.kind = kind
        self.name = name
        self.fileName = fileName
    }

    public var id: String {
        "\(kind.rawValue):\(fileName)"
    }
}

public struct LLMInferenceResponse: Sendable, Equatable {
    public let generatedText: String
    public let generatedTokenCount: Int

    public init(generatedText: String, generatedTokenCount: Int) {
        self.generatedText = generatedText
        self.generatedTokenCount = generatedTokenCount
    }
}

public struct LLMInferenceChunk: Sendable, Equatable {
    public let text: String
    public let generatedTokenCount: Int

    public init(text: String, generatedTokenCount: Int) {
        self.text = text
        self.generatedTokenCount = generatedTokenCount
    }
}

public enum LLMEngineError: LocalizedError, Equatable {
    case unavailableEngine(String)
    case unsupportedOperation(String)

    public var errorDescription: String? {
        switch self {
        case .unavailableEngine(let message), .unsupportedOperation(let message):
            return message
        }
    }
}

public protocol LLMEngine: Sendable {
    associatedtype Model

    var descriptor: LLMEngineDescriptor { get }

    func loadCheckpoint(data: Data) async throws
    func exportCheckpoint() async throws -> Data
    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error>
    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error>
    func runInference(request: LLMInferenceRequest) async throws -> LLMInferenceResponse
    /// Drops cached training assets and the post-training checkpoint copy. Use
    /// this when a caller knows it's done with an engine for now (e.g. the
    /// Comparison screen between engines) to avoid retaining ~500 MB of GPT-2
    /// checkpoint per idle engine.
    func releaseResources() async
}

public extension LLMEngine {
    func runInference(request: LLMInferenceRequest) async throws -> LLMInferenceResponse {
        let stream = try await startInference(request: request)
        var generatedText = ""
        var generatedTokenCount = 0
        for try await chunk in stream {
            generatedText += chunk.text
            generatedTokenCount = chunk.generatedTokenCount
        }
        return LLMInferenceResponse(generatedText: generatedText, generatedTokenCount: generatedTokenCount)
    }

    func releaseResources() async {}
}

protocol LLMTrainingStreamRuntime: LLMEngine, Actor {
    func prepareTraining(request: LLMTrainingRequest) async throws -> Model
    func trainingLoop(
        model: inout Model,
        request: LLMTrainingRequest,
        preparationStart: ContinuousClock.Instant,
        continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation
    ) async throws
    func releaseTrainingState() async
}

extension LLMTrainingStreamRuntime {
    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runTrainingStream(request: request, continuation: continuation)
                    continuation.finish()
                } catch {
                    await self.releaseTrainingState()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    private func runTrainingStream(
        request: LLMTrainingRequest,
        continuation: AsyncThrowingStream<LLMTrainingProgress, Error>.Continuation
    ) async throws {
        let preparationStart = ContinuousClock.now
        var model = try await prepareTraining(request: request)
        try await trainingLoop(model: &model, request: request, preparationStart: preparationStart, continuation: continuation)
        await releaseTrainingState()
    }
}

struct LLMInferenceContext<Model>: @unchecked Sendable {
    let model: Model
    let tokenizer: LLMTokenizer
    let promptTokens: [Int]
}

protocol LLMInferenceStreamRuntime: LLMEngine {
    associatedtype InferenceContext: Sendable

    func prepareInference(request: LLMInferenceRequest) async throws -> InferenceContext
    func inferenceLoop(
        context: InferenceContext,
        request: LLMInferenceRequest,
        continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation
    ) async throws
}

extension LLMInferenceStreamRuntime {
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
}

public struct LLMEngineRegistry: Sendable {
    private let engines: [any LLMEngine]

    public init() {
        self.engines = [
            LLMCReferenceRuntime(),
            LLMBasicSwiftRuntime(),
            LLMSwiftRuntime(),
            LLMMultithreadedSwiftRuntime(),
            LLMAMXRuntime(),
            LLMMetalRuntime(),
            LLMBLASRuntime(),
            LLMBNNSRuntime(),
            LLMMPSRuntime(),
            LLMMLTensorRuntime()
        ]
    }

    public var descriptors: [LLMEngineDescriptor] {
        engines.map(\.descriptor)
    }

    public func engine(for identifier: LLMEngineIdentifier) -> (any LLMEngine)? {
        engines.first { $0.descriptor.id == identifier }
    }
}

private struct StubEngineBase: LLMEngine {
    typealias Model = Void

    let descriptor: LLMEngineDescriptor

    func loadCheckpoint(data: Data) async throws {
        throw LLMEngineError.unsupportedOperation("\(descriptor.displayName) checkpoint loading is not implemented yet.")
    }

    func exportCheckpoint() async throws -> Data {
        throw LLMEngineError.unsupportedOperation("\(descriptor.displayName) checkpoint export is not implemented yet.")
    }

    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        throw LLMEngineError.unsupportedOperation("\(descriptor.displayName) training is not implemented yet.")
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        throw LLMEngineError.unsupportedOperation("\(descriptor.displayName) inference is not implemented yet.")
    }
}
