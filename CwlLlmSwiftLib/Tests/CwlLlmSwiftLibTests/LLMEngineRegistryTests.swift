import XCTest
@testable import CwlLlmSwiftLib

final class LLMEngineRegistryTests: XCTestCase {
    func testDefaultRegistryIncludesOnlyPhaseOneEngines() {
        let descriptors = LLMEngineRegistry().descriptors

        XCTAssertEqual(
            descriptors.map(\.id),
            [.cReference, .basicSwift, .fastSwift, .multithreadedSwift, .amx, .metal]
        )
    }

    func testAllEnginesAdvertiseExpectedCapabilities() throws {
        let basicSwiftDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .basicSwift)?.descriptor)
        XCTAssertEqual(basicSwiftDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(basicSwiftDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(basicSwiftDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let fastSwiftDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .fastSwift)?.descriptor)
        XCTAssertEqual(fastSwiftDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(fastSwiftDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(fastSwiftDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let cReferenceDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .cReference)?.descriptor)
        XCTAssertEqual(cReferenceDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(cReferenceDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(cReferenceDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let amxDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .amx)?.descriptor)
        XCTAssertEqual(amxDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(amxDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(amxDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let metalDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .metal)?.descriptor)
        XCTAssertEqual(metalDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(metalDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(metalDescriptor.capabilitySummary, "Training, Inference, Checkpointing")
    }
}
