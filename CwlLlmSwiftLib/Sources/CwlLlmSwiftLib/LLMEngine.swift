import Foundation

public enum LLMEngineIdentifier: String, CaseIterable, Codable, Sendable {
    case basicSwift = "basic-swift"
    case fastSwift = "fast-swift"
    case multithreadedSwift = "multithreaded-swift"
    case amx = "amx"
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
    var descriptor: LLMEngineDescriptor { get }

    func loadCheckpoint(data: Data) async throws
    func exportCheckpoint() async throws -> Data
    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error>
    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error>
    func runInference(request: LLMInferenceRequest) async throws -> LLMInferenceResponse
    /// Drops every cached buffer on the engine — model, datasets, and the
    /// post-training checkpoint copy. Use this when a caller knows it's done
    /// with an engine for now (e.g. the Comparison screen between engines)
    /// to avoid retaining ~500 MB of GPT-2 checkpoint per idle engine.
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

public struct LLMEngineRegistry: Sendable {
    private let engines: [any LLMEngine]

    public init() {
        self.engines = [
            CReferenceValidationEngine(),
            BasicSwiftEngine(),
            FastSwiftEngine(),
            MultithreadedSwiftEngine(),
            AMXEngine(),
            MetalEngine()
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

struct BasicSwiftEngine: LLMEngine {
    let descriptor = LLMEngineDescriptor(
        id: .basicSwift,
        displayName: "Basic Swift",
        summary: "Naive pure Swift backend that keeps the train_gpt2.swift structure close to the straightforward reference implementation.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "Basic Swift backend supports training, generation, and checkpoint export/load."
    )
    let runtime = LLMBasicSwiftRuntime()

    func loadCheckpoint(data: Data) async throws {
        try await runtime.loadCheckpoint(data: data)
    }

    func exportCheckpoint() async throws -> Data {
        try await runtime.exportCheckpoint()
    }

    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        try await runtime.startTraining(request: request)
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        try await runtime.startInference(request: request)
    }

    func releaseResources() async {
        await runtime.releaseResources()
    }
}

struct FastSwiftEngine: LLMEngine {
    let descriptor = LLMEngineDescriptor(
        id: .fastSwift,
        displayName: "Fast Swift",
        summary: "Optimized pure Swift backend derived from the train_gpt2.swift translation, keeping the shared Swift runtime while applying targeted low-level speedups.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "Fast Swift backend supports training, generation, and checkpoint export/load."
    )
    let runtime = LLMSwiftRuntime()

    func loadCheckpoint(data: Data) async throws {
        try await runtime.loadCheckpoint(data: data)
    }

    func exportCheckpoint() async throws -> Data {
        try await runtime.exportCheckpoint()
    }

    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        try await runtime.startTraining(request: request)
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        try await runtime.startInference(request: request)
    }

    func releaseResources() async {
        await runtime.releaseResources()
    }
}

struct AMXEngine: LLMEngine {
    let descriptor = LLMEngineDescriptor(
        id: .amx,
        displayName: "Direct AMX",
        summary: "Direct Apple AMX backend that replaces the BLAS engine's GEMM calls with explicit AMX microkernels while keeping the surrounding train_gpt2-style Swift structure comparable.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: LLMAMXBridge.isAvailable,
        availabilityNote: LLMAMXBridge.isAvailable
            ? "AMX backend supports training, generation, and checkpoint export/load via direct AMX matmul kernels."
            : "AMX backend requires Apple Silicon with the private AMX instruction set."
    )
    let runtime = LLMAMXRuntime()

    func loadCheckpoint(data: Data) async throws {
        try await runtime.loadCheckpoint(data: data)
    }

    func exportCheckpoint() async throws -> Data {
        try await runtime.exportCheckpoint()
    }

    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        try await runtime.startTraining(request: request)
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        try await runtime.startInference(request: request)
    }

    func releaseResources() async {
        await runtime.releaseResources()
    }
}

struct MultithreadedSwiftEngine: LLMEngine {
    let descriptor = LLMEngineDescriptor(
        id: .multithreadedSwift,
        displayName: "Multithreaded Swift",
        summary: "Optimized pure Swift backend derived from the fast Swift implementation, adding CPU-parallel matmul and attention kernels with Dispatch concurrent work sharing.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "Multithreaded Swift backend supports training, generation, and checkpoint export/load."
    )
    let runtime = LLMMultithreadedSwiftRuntime()

    func loadCheckpoint(data: Data) async throws {
        try await runtime.loadCheckpoint(data: data)
    }

    func exportCheckpoint() async throws -> Data {
        try await runtime.exportCheckpoint()
    }

    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        try await runtime.startTraining(request: request)
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        try await runtime.startInference(request: request)
    }

    func releaseResources() async {
        await runtime.releaseResources()
    }
}

struct MetalEngine: LLMEngine {
    let descriptor = LLMEngineDescriptor(
        id: .metal,
        displayName: "Metal",
        summary: "Custom Metal compute backend using explicit GPU kernels for forward/backward passes and MPS for matrix multiply.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "Metal backend supports training, generation, and checkpoint export/load via GPU compute shaders."
    )
    let runtime = LLMMetalRuntime()

    func loadCheckpoint(data: Data) async throws {
        try await runtime.loadCheckpoint(data: data)
    }

    func exportCheckpoint() async throws -> Data {
        try await runtime.exportCheckpoint()
    }

    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        try await runtime.startTraining(request: request)
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        try await runtime.startInference(request: request)
    }

    func releaseResources() async {
        await runtime.releaseResources()
    }
}

struct CReferenceValidationEngine: LLMEngine {
    let descriptor = LLMEngineDescriptor(
        id: .cReference,
        displayName: "llm.c",
        summary: "Vendored CPU reference path from train_gpt2.c for training, inference, checkpointing, and numerical validation.",
        capabilities: [.training, .inference, .checkpointing],
        isAvailable: true,
        availabilityNote: "Reference backend supports training, generation, and checkpoint export/load."
    )
    let runtime = LLMCReferenceRuntime()

    func loadCheckpoint(data: Data) async throws {
        try await runtime.loadCheckpoint(data: data)
    }

    func exportCheckpoint() async throws -> Data {
        try await runtime.exportCheckpoint()
    }

    func startTraining(request: LLMTrainingRequest) async throws -> AsyncThrowingStream<LLMTrainingProgress, Error> {
        try await runtime.startTraining(request: request)
    }

    func startInference(request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        try await runtime.startInference(request: request)
    }

    func releaseResources() async {
        await runtime.releaseResources()
    }
}
