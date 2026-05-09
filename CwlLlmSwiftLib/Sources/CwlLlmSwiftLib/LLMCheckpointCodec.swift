import Foundation

struct LLMCheckpointHeader: Sendable, Equatable {
    let maxSequenceLength: Int
    let vocabularySize: Int
    let layerCount: Int
    let headCount: Int
    let channelCount: Int
    let paddedVocabularySize: Int
}

enum LLMCheckpointCodecError: LocalizedError, Equatable {
    case fileTooSmall
    case invalidMagic(UInt32)
    case unsupportedVersion(UInt32)
    case invalidParameterCount(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .fileTooSmall:
            return "Checkpoint data is too small to contain a valid header."
        case .invalidMagic(let magic):
            return "Checkpoint data has invalid magic value \(magic)."
        case .unsupportedVersion(let version):
            return "Checkpoint data uses unsupported version \(version)."
        case .invalidParameterCount(let expected, let actual):
            return "Checkpoint data contained \(actual) parameters, expected \(expected)."
        }
    }
}

enum LLMCheckpointCodec {
    static let magic: UInt32 = 20240326
    static let version: UInt32 = 3
    private static let headerCount = 256

    static func decode(_ data: Data) throws -> (header: LLMCheckpointHeader, parameters: [Float]) {
        guard data.count >= headerCount * MemoryLayout<UInt32>.size else {
            throw LLMCheckpointCodecError.fileTooSmall
        }

        let headerValues = data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: UInt32.self).prefix(headerCount))
        }
        guard headerValues.count >= 8 else {
            throw LLMCheckpointCodecError.fileTooSmall
        }
        guard headerValues[0] == magic else {
            throw LLMCheckpointCodecError.invalidMagic(headerValues[0])
        }
        guard headerValues[1] == version else {
            throw LLMCheckpointCodecError.unsupportedVersion(headerValues[1])
        }

        let header = LLMCheckpointHeader(
            maxSequenceLength: Int(headerValues[2]),
            vocabularySize: Int(headerValues[3]),
            layerCount: Int(headerValues[4]),
            headCount: Int(headerValues[5]),
            channelCount: Int(headerValues[6]),
            paddedVocabularySize: Int(headerValues[7])
        )

        let parameterData = data.dropFirst(headerCount * MemoryLayout<UInt32>.size)
        let parameters = parameterData.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
        return (header, parameters)
    }

    static func encode(header: LLMCheckpointHeader, parameters: [Float], expectedParameterCount: Int) throws -> Data {
        guard parameters.count == expectedParameterCount else {
            throw LLMCheckpointCodecError.invalidParameterCount(expected: expectedParameterCount, actual: parameters.count)
        }

        var rawHeader = Array(repeating: UInt32.zero, count: headerCount)
        rawHeader[0] = magic
        rawHeader[1] = version
        rawHeader[2] = UInt32(header.maxSequenceLength)
        rawHeader[3] = UInt32(header.vocabularySize)
        rawHeader[4] = UInt32(header.layerCount)
        rawHeader[5] = UInt32(header.headCount)
        rawHeader[6] = UInt32(header.channelCount)
        rawHeader[7] = UInt32(header.paddedVocabularySize)

        let headerData = rawHeader.withUnsafeBufferPointer { Data(buffer: $0) }
        let parameterData = parameters.withUnsafeBufferPointer { Data(buffer: $0) }
        var data = headerData
        data.append(parameterData)
        return data
    }
}
