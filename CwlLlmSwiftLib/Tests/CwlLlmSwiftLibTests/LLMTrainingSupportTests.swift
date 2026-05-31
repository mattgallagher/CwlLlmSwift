import Foundation
import XCTest
@testable import CwlLlmSwiftLib

final class LLMTrainingSupportTests: XCTestCase {
    func testTokenDataLoaderDecodesShardAndWrapsBatches() throws {
        let data = makeTokenShard(tokens: [10, 11, 12, 13, 14, 15, 16, 17, 18])
        var loader = try LLMTokenDataLoader(data: data, batchSize: 2, sequenceLength: 2)

        XCTAssertEqual(loader.tokenCount, 9)
        XCTAssertEqual(loader.sampleCount, 2)

        loader.nextBatch()
        XCTAssertEqual(loader.inputs, [10, 11, 12, 13])
        XCTAssertEqual(loader.targets, [11, 12, 13, 14])

        loader.nextBatch()
        XCTAssertEqual(loader.inputs, [14, 15, 16, 17])
        XCTAssertEqual(loader.targets, [15, 16, 17, 18])

        loader.nextBatch()
        XCTAssertEqual(loader.inputs, [10, 11, 12, 13])
        XCTAssertEqual(loader.targets, [11, 12, 13, 14])
    }

    func testTokenDataLoaderRejectsMismatchedTokenCount() {
        let header = makeTokenShardHeader(tokenCount: 4)
        let payload = [UInt16(1), UInt16(2), UInt16(3)]
        var data = header.withUnsafeBufferPointer { Data(buffer: $0) }
        data.append(payload.withUnsafeBufferPointer { Data(buffer: $0) })

        XCTAssertThrowsError(try LLMTokenDataLoader(data: data, batchSize: 1, sequenceLength: 2)) { error in
            XCTAssertEqual(
                error as? LLMTokenDataLoaderError,
                .invalidDataset("Dataset shard token count mismatch: header says 4, file contains 3.")
            )
        }
    }

    private func makeTokenShard(tokens: [UInt16]) -> Data {
        let header = makeTokenShardHeader(tokenCount: tokens.count)
        var data = header.withUnsafeBufferPointer { Data(buffer: $0) }
        data.append(tokens.withUnsafeBufferPointer { Data(buffer: $0) })
        return data
    }

    private func makeTokenShardHeader(tokenCount: Int) -> [UInt32] {
        var header = Array(repeating: UInt32.zero, count: 256)
        header[0] = 20240520
        header[1] = 1
        header[2] = UInt32(tokenCount)
        return header
    }
}
