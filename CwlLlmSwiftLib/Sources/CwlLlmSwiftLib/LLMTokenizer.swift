import Foundation

struct LLMTokenizer {
    private let tokenDataByID: [Data]
    private let tokenIDByData: [Data: Int]
    private let maxTokenLength: Int
    let eotToken: Int

    init(data: Data) throws {
        var cursor = 0

        func readUInt32() throws -> UInt32 {
            guard cursor + 4 <= data.count else {
                throw CocoaError(.fileReadCorruptFile)
            }
            defer { cursor += 4 }
            let bytes = Array(data[cursor..<(cursor + 4)])
            return UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) | (UInt32(bytes[2]) << 16) | (UInt32(bytes[3]) << 24)
        }

        guard data.count >= 256 * 4 else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let magic = try readUInt32()
        guard magic == 20240328 else {
            throw CocoaError(.fileReadCorruptFile)
        }

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
            guard cursor < data.count else {
                throw CocoaError(.fileReadCorruptFile)
            }
            let length = Int(data[cursor])
            cursor += 1
            guard length > 0, cursor + length <= data.count else {
                throw CocoaError(.fileReadCorruptFile)
            }
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
        guard !utf8.isEmpty else {
            return [eotToken]
        }

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

    func decode(_ tokens: ArraySlice<Int>) -> String {
        var output = Data()
        for token in tokens {
            guard token >= 0, token < tokenDataByID.count else {
                continue
            }
            output.append(tokenDataByID[token])
        }
        return String(decoding: output, as: UTF8.self)
    }

    func decode(token: Int) -> String {
        guard token >= 0, token < tokenDataByID.count else {
            return ""
        }
        return String(decoding: tokenDataByID[token], as: UTF8.self)
    }
}
