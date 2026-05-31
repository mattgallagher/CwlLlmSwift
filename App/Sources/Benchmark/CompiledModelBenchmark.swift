import CoreML
import CwlLlmSwiftLib
import Foundation

enum CompiledModelInferenceError: LocalizedError {
    case tokenizerRequired
    case invalidModel(String)
    case missingOutput(String)

    var errorDescription: String? {
        switch self {
        case .tokenizerRequired:
            return "Inference requires a tokenizer asset."
        case .invalidModel(let message), .missingOutput(let message):
            return message
        }
    }
}

private struct CompiledModelTokenizer: Sendable {
    private let tokenDataByID: [Data]
    private let tokenIDByData: [Data: Int]
    private let maxTokenLength: Int
    let eotToken: Int

    init(data: Data) throws {
        var cursor = 0

        func readUInt32() throws -> UInt32 {
            guard cursor + 4 <= data.count else { throw CocoaError(.fileReadCorruptFile) }
            defer { cursor += 4 }
            let bytes = Array(data[cursor..<(cursor + 4)])
            return UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) | (UInt32(bytes[2]) << 16) | (UInt32(bytes[3]) << 24)
        }

        guard data.count >= 256 * 4 else { throw CocoaError(.fileReadCorruptFile) }
        let magic = try readUInt32()
        guard magic == 20240328 else { throw CocoaError(.fileReadCorruptFile) }

        let version = try readUInt32()
        let vocabSize = Int(try readUInt32())
        let eotToken: Int
        switch version {
        case 1:
            eotToken = 50256
        case 2:
            eotToken = Int(try readUInt32())
        default:
            throw CocoaError(.fileReadCorruptFile)
        }

        cursor = 256 * 4
        var tokenDataByID: [Data] = []
        tokenDataByID.reserveCapacity(vocabSize)
        var tokenIDByData: [Data: Int] = [:]
        tokenIDByData.reserveCapacity(vocabSize)
        var maxTokenLength = 0

        for tokenID in 0..<vocabSize {
            guard cursor < data.count else { throw CocoaError(.fileReadCorruptFile) }
            let length = Int(data[cursor])
            cursor += 1
            guard length > 0, cursor + length <= data.count else { throw CocoaError(.fileReadCorruptFile) }
            let tokenData = Data(data[cursor..<(cursor + length)])
            cursor += length
            tokenDataByID.append(tokenData)
            tokenIDByData[tokenData] = tokenID
            maxTokenLength = max(maxTokenLength, length)
        }

        self.tokenDataByID = tokenDataByID
        self.tokenIDByData = tokenIDByData
        self.maxTokenLength = maxTokenLength
        self.eotToken = eotToken
    }

    func encodePrompt(_ prompt: String, maximumTokenCount: Int) -> [Int] {
        let utf8 = Array(prompt.utf8)
        guard !utf8.isEmpty else { return [eotToken] }

        var tokens: [Int] = []
        var index = 0
        while index < utf8.count, tokens.count < maximumTokenCount {
            let remainingCount = utf8.count - index
            let candidateLength = min(maxTokenLength, remainingCount)
            var matchedToken: Int?
            var matchedLength = 0

            for length in stride(from: candidateLength, through: 1, by: -1) {
                let slice = Data(utf8[index..<(index + length)])
                if let token = tokenIDByData[slice] {
                    matchedToken = token
                    matchedLength = length
                    break
                }
            }

            if let matchedToken {
                tokens.append(matchedToken)
                index += matchedLength
            } else {
                tokens.append(eotToken)
                index += 1
            }
        }

        return tokens.isEmpty ? [eotToken] : tokens
    }

    func decode(token: Int) -> String {
        guard token >= 0, token < tokenDataByID.count else { return "" }
        return String(decoding: tokenDataByID[token], as: UTF8.self)
    }
}

private struct CompiledModelSession: @unchecked Sendable {
    let model: MLModel
    let inputName: String
    let outputName: String
    let batchSize: Int
    let sequenceLength: Int
}

enum CompiledModelInferenceRunner {
    static func startInference(modelURL: URL, request: LLMInferenceRequest) async throws -> AsyncThrowingStream<LLMInferenceChunk, Error> {
        guard let tokenizerData = request.tokenizerData else {
            throw CompiledModelInferenceError.tokenizerRequired
        }
        let tokenizer = try CompiledModelTokenizer(data: tokenizerData)
        let session = try loadSession(modelURL: modelURL)
        guard session.batchSize == 1 else {
            throw CompiledModelInferenceError.invalidModel(
                "Compiled model inference requires batch size 1, but the selected mlpackage expects B=\(session.batchSize)."
            )
        }

        let promptTokens = tokenizer.encodePrompt(request.prompt, maximumTokenCount: session.sequenceLength)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try runLoop(session: session, tokenizer: tokenizer, promptTokens: promptTokens, request: request, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private static func loadSession(modelURL: URL) throws -> CompiledModelSession {
        let isSecurityScoped = modelURL.startAccessingSecurityScopedResource()
        defer {
            if isSecurityScoped {
                modelURL.stopAccessingSecurityScopedResource()
            }
        }

        let compiledURL = try MLModel.compileModel(at: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)

        let inputName = model.modelDescription.inputDescriptionsByName["tokens"] != nil
            ? "tokens"
            : model.modelDescription.inputDescriptionsByName.keys.sorted().first
        guard let inputName,
              let inputDescription = model.modelDescription.inputDescriptionsByName[inputName],
              let constraint = inputDescription.multiArrayConstraint else {
            throw CompiledModelInferenceError.invalidModel("Compiled model must expose a multi-array input.")
        }

        let shape = constraint.shape.map(\.intValue)
        guard shape.count == 2 else {
            throw CompiledModelInferenceError.invalidModel("Compiled model input must have shape [B, T].")
        }

        guard let outputName = model.modelDescription.outputDescriptionsByName.keys.sorted().first else {
            throw CompiledModelInferenceError.invalidModel("Compiled model has no outputs.")
        }

        return CompiledModelSession(
            model: model,
            inputName: inputName,
            outputName: outputName,
            batchSize: shape[0],
            sequenceLength: shape[1]
        )
    }

    private static func runLoop(
        session: CompiledModelSession,
        tokenizer: CompiledModelTokenizer,
        promptTokens: [Int],
        request: LLMInferenceRequest,
        continuation: AsyncThrowingStream<LLMInferenceChunk, Error>.Continuation
    ) throws {
        let totalCount = min(session.sequenceLength, promptTokens.count + request.maximumTokenCount)
        var generationTokens = Array(repeating: Int32(tokenizer.eotToken), count: session.sequenceLength)
        for (index, token) in promptTokens.prefix(totalCount).enumerated() {
            generationTokens[index] = Int32(token)
        }

        let startIndex = min(promptTokens.count, totalCount)
        guard startIndex < totalCount else { return }

        var generatedTokenCount = 0
        var state = UInt64.random(in: UInt64.min...UInt64.max)
        for tokenIndex in startIndex..<totalCount {
            try Task.checkCancellation()
            let inputArray = try makeInputArray(tokens: generationTokens, sequenceLength: session.sequenceLength)
            let outputArray = try predict(model: session.model, inputName: session.inputName, outputName: session.outputName, inputArray: inputArray)
            let rowIndex = max(0, tokenIndex - 1)
            let logits = try logitsRow(from: outputArray, rowIndex: rowIndex)
            let nextToken = sample(logits: ArraySlice(logits), temperature: request.temperature, state: &state)
            generationTokens[tokenIndex] = Int32(nextToken)
            generatedTokenCount += 1
            continuation.yield(LLMInferenceChunk(text: tokenizer.decode(token: nextToken), generatedTokenCount: generatedTokenCount))
        }
    }

    private static func makeInputArray(tokens: [Int32], sequenceLength: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, sequenceLength as NSNumber], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: tokens.count)
        for (index, token) in tokens.enumerated() {
            pointer[index] = token
        }
        return array
    }

    private static func predict(model: MLModel, inputName: String, outputName: String, inputArray: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: inputArray)])
        let output = try model.prediction(from: provider)
        guard let value = output.featureValue(for: outputName)?.multiArrayValue else {
            throw CompiledModelInferenceError.missingOutput("Compiled model output '\(outputName)' is missing or not a multi-array.")
        }
        return value
    }

    private static func logitsRow(from output: MLMultiArray, rowIndex: Int) throws -> [Float] {
        let shape = output.shape.map(\.intValue)
        let strides = output.strides.map(\.intValue)

        switch output.dataType {
        case .float32:
            let pointer = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
            if shape.count == 3 {
                let vocabSize = shape[2]
                guard rowIndex < shape[1] else {
                    throw CompiledModelInferenceError.invalidModel("Requested logits row \(rowIndex) exceeds output sequence length \(shape[1]).")
                }
                return (0..<vocabSize).map { pointer[rowIndex * strides[1] + $0 * strides[2]] }
            }
            if shape.count == 2 {
                let vocabSize = shape[1]
                return (0..<vocabSize).map { pointer[$0 * strides[1]] }
            }
        case .double:
            let pointer = output.dataPointer.bindMemory(to: Double.self, capacity: output.count)
            if shape.count == 3 {
                let vocabSize = shape[2]
                guard rowIndex < shape[1] else {
                    throw CompiledModelInferenceError.invalidModel("Requested logits row \(rowIndex) exceeds output sequence length \(shape[1]).")
                }
                return (0..<vocabSize).map { Float(pointer[rowIndex * strides[1] + $0 * strides[2]]) }
            }
            if shape.count == 2 {
                let vocabSize = shape[1]
                return (0..<vocabSize).map { Float(pointer[$0 * strides[1]]) }
            }
        default:
            break
        }

        throw CompiledModelInferenceError.invalidModel("Compiled model output must be float32/double with shape [1, V] or [1, T, V].")
    }

    private static func randomU32(state: inout UInt64) -> UInt32 {
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return UInt32((state &* 0x2545F4914F6CDD1D) >> 32)
    }

    private static func randomF32(state: inout UInt64) -> Float {
        Float(randomU32(state: &state) >> 8) / 16_777_216.0
    }

    private static func sample(logits: ArraySlice<Float>, temperature: Double, state: inout UInt64) -> Int {
        if temperature <= 0 {
            return logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }

        let maxLogit = logits.max() ?? 0
        var probabilities = logits.map { exp(Double(($0 - maxLogit) / Float(temperature))) }
        let sum = probabilities.reduce(0, +)
        guard sum > 0 else { return 0 }
        for index in probabilities.indices {
            probabilities[index] /= sum
        }

        let sample = Double(randomF32(state: &state))
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
