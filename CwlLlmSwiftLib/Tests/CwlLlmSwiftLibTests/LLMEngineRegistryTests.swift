import XCTest
@testable import CwlLlmSwiftLib

final class LLMEngineRegistryTests: XCTestCase {
    func testDefaultRegistryIncludesAllPhaseFiveEngines() {
        let descriptors = LLMEngineRegistry().descriptors

        XCTAssertEqual(
            descriptors.map(\.id),
            [.cReference, .basicSwift, .fastSwift, .multithreadedSwift, .amx, .metal, .blas, .bnns, .mpsGraph, .mlTensor]
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

        let blasDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .blas)?.descriptor)
        XCTAssertEqual(blasDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(blasDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(blasDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let amxDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .amx)?.descriptor)
        XCTAssertEqual(amxDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(amxDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(amxDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let bnnsDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .bnns)?.descriptor)
        XCTAssertEqual(bnnsDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(bnnsDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(bnnsDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let mpsGraphDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .mpsGraph)?.descriptor)
        XCTAssertEqual(mpsGraphDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(mpsGraphDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(mpsGraphDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let mlTensorDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .mlTensor)?.descriptor)
        XCTAssertEqual(mlTensorDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(mlTensorDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(mlTensorDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

        let metalDescriptor = try XCTUnwrap(LLMEngineRegistry().engine(for: .metal)?.descriptor)
        XCTAssertEqual(metalDescriptor.capabilities, [.training, .inference, .checkpointing])
        XCTAssertEqual(metalDescriptor.capabilities.statusSummary, "Partial")
        XCTAssertEqual(metalDescriptor.capabilitySummary, "Training, Inference, Checkpointing")

    }
}
