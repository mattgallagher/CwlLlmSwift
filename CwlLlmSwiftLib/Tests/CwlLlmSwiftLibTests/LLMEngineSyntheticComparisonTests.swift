import CLLMCReference
import CoreML
import Foundation
import MetalPerformanceShadersGraph
import XCTest
@testable import CwlLlmSwiftLib

private struct ComparisonTolerance {
    let loss: Float
    let activations: Float
    let gradients: Float
    let parameters: Float
}

private extension Array where Element == Double {
    var average: Double? {
        guard isEmpty == false else {
            return nil
        }
        return reduce(0, +) / Double(count)
    }
}

private struct SyntheticFixture {
    let checkpointData: Data
    let batchSize: Int
    let sequenceLength: Int
    let inputs: [UInt32]
    let targets: [UInt32]
    let updateParams: LLMSwift.UpdateParams

    static func make() throws -> SyntheticFixture {
        let header = LLMCheckpointHeader(
            maxSequenceLength: 8,
            vocabularySize: 16,
            layerCount: 2,
            headCount: 2,
            channelCount: 4,
            paddedVocabularySize: 16
        )
        let config = LLMGPT2Config(header: header)
        var state = UInt64(0x5EED_F00D_DECA_FBAD)
        let parameters = (0..<config.num_parameters).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let value = Float((state >> 40) & 0xFFFF) / Float(0xFFFF)
            return (value - 0.5) * 0.24
        }

        let checkpointData = try LLMCheckpointCodec.encode(
            header: header,
            parameters: parameters,
            expectedParameterCount: config.num_parameters
        )

        return SyntheticFixture(
            checkpointData: checkpointData,
            batchSize: 2,
            sequenceLength: 4,
            inputs: [1, 5, 7, 3, 4, 2, 6, 8],
            targets: [5, 7, 3, 4, 2, 6, 8, 9],
            updateParams: LLMSwift.UpdateParams(
                learning_rate: 1e-3,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0,
                t: 1
            )
        )
    }
}

private struct ModelSnapshot {
    let meanLoss: Float
    let parameters: [Float]
    let gradients: [Float]
    let activations: [Float]
}

private struct TrainingRunSnapshot {
    let progress: [LLMTrainingProgress]
    let checkpointData: Data
}

final class LLMEngineSyntheticComparisonTests: XCTestCase {
    private enum EngineCase: CaseIterable {
        case basicSwift
        case fastSwift
        case blas
        case amx
        case bnns

        var name: String {
            switch self {
            case .basicSwift: "Basic Swift"
            case .fastSwift: "Fast Swift"
            case .blas: "BLAS"
            case .amx: "Direct AMX"
            case .bnns: "BNNS"
            }
        }

        var tolerance: ComparisonTolerance {
            switch self {
            case .basicSwift, .fastSwift:
                ComparisonTolerance(loss: 1e-5, activations: 5e-5, gradients: 1e-5, parameters: 1e-5)
            case .blas, .amx:
                ComparisonTolerance(loss: 5e-4, activations: 5e-3, gradients: 5e-4, parameters: 5e-4)
            case .bnns:
                ComparisonTolerance(loss: 5e-3, activations: 7e-3, gradients: 7e-3, parameters: 7e-3)
            }
        }
    }

    private static let mlTensorTolerance = ComparisonTolerance(loss: 5e-3, activations: 7e-3, gradients: 7e-3, parameters: 7e-3)

    func testSyntheticForwardMatchesCReferenceAcrossEngines() throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runForward(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)

        for engine in EngineCase.allCases {
            let snapshot = try snapshot(for: engine, fixture: fixture, phase: .forward)
            assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: engine.tolerance, engineName: engine.name, phaseName: "forward")
        }
    }

    func testSyntheticBackwardMatchesCReferenceAcrossEngines() throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runBackward(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)

        for engine in EngineCase.allCases {
            let snapshot = try snapshot(for: engine, fixture: fixture, phase: .backward)
            assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: engine.tolerance, engineName: engine.name, phaseName: "backward")
        }
    }

    func testSyntheticTrainingStepMatchesCReferenceAcrossEngines() throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runTrainingStep(
            inputs: fixture.inputs,
            targets: fixture.targets,
            B: fixture.batchSize,
            T: fixture.sequenceLength,
            updateParams: fixture.updateParams
        )

        for engine in EngineCase.allCases {
            let snapshot = try snapshot(for: engine, fixture: fixture, phase: .trainingStep)
            assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: engine.tolerance, engineName: engine.name, phaseName: "training step")
        }
    }

    func testSyntheticCheckpointRoundTripsAcrossEngines() throws {
        let fixture = try SyntheticFixture.make()
        let expected = try LLMCheckpointCodec.decode(fixture.checkpointData)

        let basicSwift = try LLMBasicSwift.buildModel(from: fixture.checkpointData)
        let fastSwift = try LLMSwift.buildModel(from: fixture.checkpointData)
        let blas = try LLMBLAS.buildModel(from: fixture.checkpointData)
        let amx = try LLMAMX.buildModel(from: fixture.checkpointData)
        let bnns = try LLMBNNS.buildModel(from: fixture.checkpointData)

        try assertCheckpointRoundTrip(basicSwift.exportCheckpoint(), expected: expected, engineName: EngineCase.basicSwift.name)
        try assertCheckpointRoundTrip(fastSwift.exportCheckpoint(), expected: expected, engineName: EngineCase.fastSwift.name)
        try assertCheckpointRoundTrip(blas.exportCheckpoint(), expected: expected, engineName: EngineCase.blas.name)
        try assertCheckpointRoundTrip(amx.exportCheckpoint(), expected: expected, engineName: EngineCase.amx.name)
        try assertCheckpointRoundTrip(bnns.exportCheckpoint(), expected: expected, engineName: EngineCase.bnns.name)
    }

    func testBNNSLatestTokenInferenceMatchesFullLogitsRow() throws {
        let fixture = try SyntheticFixture.make()
        var model = try LLMBNNS.buildModel(from: fixture.checkpointData)
        let inputs = Array(fixture.inputs.prefix(fixture.sequenceLength))
        let rowIndex = 2

        try LLMBNNS.gpt2_forward(model: &model, inputs: inputs, targets: [], B: 1, T: fixture.sequenceLength)
        let rowLogits = try LLMBNNS.gpt2_latest_token_logits(model: &model, inputs: inputs, rowIndex: rowIndex, B: 1, T: fixture.sequenceLength)
        let rowStart = rowIndex * model.config.padded_vocab_size

        XCTAssertEqual(rowLogits.count, model.config.vocab_size)
        for index in rowLogits.indices {
            XCTAssertEqual(
                rowLogits[index],
                model.acts.logits[rowStart + index],
                accuracy: EngineCase.bnns.tolerance.loss,
                "BNNS latest-token logits mismatch at index \(index)"
            )
        }
    }

    // MARK: - Metal compute engine tests

    private static let metalTolerance = ComparisonTolerance(loss: 5e-3, activations: 7e-3, gradients: 7e-3, parameters: 7e-3)

    func testMetalSyntheticForwardMatchesCReference() throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runForward(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)

        let snapshot = try metalSnapshot(fixture: fixture, phase: .forward)
        assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: Self.metalTolerance, engineName: "Metal", phaseName: "forward")
    }

    func testMetalSyntheticBackwardMatchesCReference() throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runBackward(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)

        let snapshot = try metalSnapshot(fixture: fixture, phase: .backward)
        assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: Self.metalTolerance, engineName: "Metal", phaseName: "backward")
    }

    func testMetalSyntheticTrainingStepMatchesCReference() throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runTrainingStep(
            inputs: fixture.inputs,
            targets: fixture.targets,
            B: fixture.batchSize,
            T: fixture.sequenceLength,
            updateParams: fixture.updateParams
        )

        let snapshot = try metalSnapshot(fixture: fixture, phase: .trainingStep)
        assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: Self.metalTolerance, engineName: "Metal", phaseName: "training step")
    }

    func testMetalSyntheticCheckpointRoundTrip() throws {
        let fixture = try SyntheticFixture.make()
        let expected = try LLMCheckpointCodec.decode(fixture.checkpointData)

        let model = try buildMetalModel(from: fixture.checkpointData)
        let exported = try model.exportCheckpoint()
        try assertCheckpointRoundTrip(exported, expected: expected, engineName: "Metal")
    }

    private func buildMetalModel(from checkpointData: Data) throws -> GPT2Metal {
        let (header, parameters) = try LLMCheckpointCodec.decode(checkpointData)
        let config = LLMGPT2Config(header: header)
        let parameterData = parameters.withUnsafeBufferPointer { Data(buffer: $0) }
        return try GPT2Metal(config: config, parameterData: parameterData)
    }

    private func metalSnapshot(fixture: SyntheticFixture, phase: ExecutionPhase) throws -> ModelSnapshot {
        let model = try buildMetalModel(from: fixture.checkpointData)
        let B = fixture.batchSize
        let T = fixture.sequenceLength

        model.forward(inputs: fixture.inputs, targets: fixture.targets, B: B, T: T)

        if phase != .forward {
            model.backward()
        }
        if phase == .trainingStep {
            model.update(
                learningRate: fixture.updateParams.learning_rate,
                beta1: fixture.updateParams.beta1,
                beta2: fixture.updateParams.beta2,
                eps: fixture.updateParams.eps,
                weightDecay: fixture.updateParams.weight_decay,
                t: fixture.updateParams.t
            )
            model.forward(inputs: fixture.inputs, targets: fixture.targets, B: B, T: T)
            // After update + re-forward, we need fresh backward for gradient comparison
            if phase == .trainingStep {
                model.backward()
            }
        }

        return ModelSnapshot(
            meanLoss: model.mean_loss,
            parameters: model.params.flattened(),
            gradients: model.grads.flattened(),
            activations: selectedMetalActivations(model: model, B: B, T: T)
        )
    }

    private func selectedMetalActivations(model: GPT2Metal, B: Int, T: Int) -> [Float] {
        guard let acts = model.acts else { return [] }
        let config = model.config
        let L = config.num_layers
        let V = config.vocab_size
        let Vp = config.padded_vocab_size

        var values = [Float]()

        func read(_ buffer: MTLBuffer) -> [Float] {
            let count = buffer.length / MemoryLayout<Float>.stride
            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }

        values.append(contentsOf: read(acts.encoded))
        for l in 0..<L { values.append(contentsOf: read(acts.ln1[l])) }
        for l in 0..<L { values.append(contentsOf: read(acts.qkv[l])) }
        for l in 0..<L { values.append(contentsOf: read(acts.atty[l])) }
        for l in 0..<L { values.append(contentsOf: read(acts.att[l])) }
        for l in 0..<L { values.append(contentsOf: read(acts.attproj[l])) }
        for l in 0..<L { values.append(contentsOf: read(acts.fch_gelu[l])) }
        for l in 0..<L { values.append(contentsOf: read(acts.residual3[l])) }
        values.append(contentsOf: read(acts.lnf))

        // Metal logits/probs use V columns; reference uses Vp. Pad to match.
        let logits = read(acts.logits)
        let probs = read(acts.probs)
        for bt in 0..<(B * T) {
            values.append(contentsOf: logits[(bt * V)..<(bt * V + V)])
            values.append(contentsOf: [Float](repeating: 0, count: Vp - V))
        }
        for bt in 0..<(B * T) {
            values.append(contentsOf: probs[(bt * V)..<(bt * V + V)])
            values.append(contentsOf: [Float](repeating: 0, count: Vp - V))
        }
        values.append(contentsOf: read(acts.losses))
        return values
    }

    // MARK: - MPSGraph engine tests
    // MPSGraph compiles forward+backward+update into a fused executable, so only
    // loss-level and checkpoint comparisons are possible (no activation/gradient access).

    private static let mpsGraphTolerance = ComparisonTolerance(loss: 5e-3, activations: 0, gradients: 0, parameters: 5e-3)

    func testMPSGraphSyntheticForwardLossMatchesCReference() throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runForward(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)

        let model = try buildMPSModel(from: fixture.checkpointData)
        let loss = model.performLossEstimation(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)
        XCTAssertEqual(loss, referenceSnapshot.meanLoss, accuracy: Self.mpsGraphTolerance.loss, "MPSGraph forward loss mismatch")
    }

    func testMPSGraphLatestTokenInferenceMatchesFullLogitsRow() throws {
        let fixture = try SyntheticFixture.make()
        let model = try buildMPSModel(from: fixture.checkpointData)
        let inputs = Array(fixture.inputs.prefix(fixture.sequenceLength))
        let rowIndex = 2
        let fullLogits = model.performInference(inputs: inputs, B: 1, T: fixture.sequenceLength)
        let rowLogits = model.performLatestTokenInference(inputs: inputs, rowIndex: rowIndex, B: 1, T: fixture.sequenceLength)
        let rowStart = rowIndex * rowLogits.count

        XCTAssertEqual(rowLogits.count, 16)
        for index in rowLogits.indices {
            XCTAssertEqual(
                rowLogits[index],
                fullLogits[rowStart + index],
                accuracy: Self.mpsGraphTolerance.loss,
                "MPSGraph inference-only row logits mismatch at index \(index)"
            )
        }
    }

    func testMPSGraphSyntheticTrainingStepLossMatchesCReference() throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runTrainingStep(
            inputs: fixture.inputs,
            targets: fixture.targets,
            B: fixture.batchSize,
            T: fixture.sequenceLength,
            updateParams: fixture.updateParams
        )

        let model = try buildMPSModel(from: fixture.checkpointData)
        _ = model.performTrainingStep(
            inputs: fixture.inputs,
            targets: fixture.targets,
            B: fixture.batchSize,
            T: fixture.sequenceLength,
            learningRate: fixture.updateParams.learning_rate
        )
        let postUpdateLoss = model.performLossEstimation(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)
        XCTAssertEqual(postUpdateLoss, referenceSnapshot.meanLoss, accuracy: Self.mpsGraphTolerance.loss, "MPSGraph training step loss mismatch")
    }

    func testMPSGraphSyntheticCheckpointRoundTrip() throws {
        let fixture = try SyntheticFixture.make()
        let expected = try LLMCheckpointCodec.decode(fixture.checkpointData)

        let model = try buildMPSModel(from: fixture.checkpointData)
        let exported = try model.exportCheckpoint()
        try assertCheckpointRoundTrip(exported, expected: expected, engineName: "MPSGraph")
    }

    private func buildMPSModel(from checkpointData: Data) throws -> GPT2MPS {
        let (header, parameters) = try LLMCheckpointCodec.decode(checkpointData)
        let config = LLMGPT2Config(header: header)
        let parameterData = parameters.withUnsafeBufferPointer { Data(buffer: $0) }
        return try GPT2MPS(config: config, parameterData: parameterData)
    }

    // MARK: - MLTensor engine tests

    func testMLTensorSyntheticForwardMatchesCReference() async throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runForward(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)

        let snapshot = try await mlTensorSnapshot(fixture: fixture, phase: .forward)
        assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: Self.mlTensorTolerance, engineName: "MLTensor", phaseName: "forward")
    }

    func testMLTensorSyntheticBackwardMatchesCReference() async throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runBackward(inputs: fixture.inputs, targets: fixture.targets, B: fixture.batchSize, T: fixture.sequenceLength)

        let snapshot = try await mlTensorSnapshot(fixture: fixture, phase: .backward)
        assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: Self.mlTensorTolerance, engineName: "MLTensor", phaseName: "backward")
    }

    func testMLTensorSyntheticTrainingStepMatchesCReference() async throws {
        let fixture = try SyntheticFixture.make()
        let reference = try CReferenceHarness(checkpointData: fixture.checkpointData)
        let referenceSnapshot = try reference.runTrainingStep(
            inputs: fixture.inputs,
            targets: fixture.targets,
            B: fixture.batchSize,
            T: fixture.sequenceLength,
            updateParams: fixture.updateParams
        )

        let snapshot = try await mlTensorSnapshot(fixture: fixture, phase: .trainingStep)
        assertSnapshotMatchesReference(snapshot, reference: referenceSnapshot, tolerance: Self.mlTensorTolerance, engineName: "MLTensor", phaseName: "training step")
    }

    func testMLTensorSyntheticCheckpointRoundTrip() async throws {
        let fixture = try SyntheticFixture.make()
        let expected = try LLMCheckpointCodec.decode(fixture.checkpointData)

        let (header, parameters) = try LLMCheckpointCodec.decode(fixture.checkpointData)
        let config = LLMGPT2Config(header: header)
        let model = GPT2MLTensor(config: config, parameters: parameters)
        let exported = try await model.exportCheckpoint()
        try assertCheckpointRoundTrip(exported, expected: expected, engineName: "MLTensor")
    }

    private func mlTensorSnapshot(fixture: SyntheticFixture, phase: ExecutionPhase) async throws -> ModelSnapshot {
        let (header, parameters) = try LLMCheckpointCodec.decode(fixture.checkpointData)
        let config = LLMGPT2Config(header: header)
        let model = GPT2MLTensor(config: config, parameters: parameters)

        let B = fixture.batchSize
        let T = fixture.sequenceLength

        let (acts, _) = model.forward(inputs: fixture.inputs, targets: fixture.targets, B: B, T: T)
        let meanLoss = await acts.losses.mean().shapedArray(of: Float.self).scalar ?? 0

        var gradients = [Float]()
        if phase != .forward {
            let grads = await model.backward(acts: acts, inputs: fixture.inputs, targets: fixture.targets, B: B, T: T)
            if phase == .trainingStep {
                LLMMLTensor.adamw_update(
                    params: &model.params, grads: grads, m: &model.m_memory, v: &model.v_memory,
                    learningRate: fixture.updateParams.learning_rate, beta1: fixture.updateParams.beta1,
                    beta2: fixture.updateParams.beta2, eps: fixture.updateParams.eps,
                    weightDecay: fixture.updateParams.weight_decay, t: fixture.updateParams.t
                )
                // Re-run forward after update to get post-update loss
                let (acts2, _) = model.forward(inputs: fixture.inputs, targets: fixture.targets, B: B, T: T)
                let postUpdateLoss = await acts2.losses.mean().shapedArray(of: Float.self).scalar ?? 0
                let activations = await selectedMLTensorActivations(acts: acts2, config: config, B: B, T: T)
                let params = await mlTensorParameters(model: model, config: config)
                // Re-run backward to get post-update gradients
                let postGrads = await model.backward(acts: acts2, inputs: fixture.inputs, targets: fixture.targets, B: B, T: T)
                gradients = await mlTensorGradients(grads: postGrads, config: config)
                return ModelSnapshot(meanLoss: postUpdateLoss, parameters: params, gradients: gradients, activations: activations)
            }
            gradients = await mlTensorGradients(grads: grads, config: config)
        }

        let activations = await selectedMLTensorActivations(acts: acts, config: config, B: B, T: T)
        let params = await mlTensorParameters(model: model, config: config)
        return ModelSnapshot(meanLoss: meanLoss, parameters: params, gradients: gradients, activations: activations)
    }

    private func mlTensorParameters(model: GPT2MLTensor, config: LLMGPT2Config) async -> [Float] {
        // Must match the flattened order of LLMGPT2ParameterTensors
        var result = [Float]()
        result.reserveCapacity(config.num_parameters)

        let V = config.vocab_size
        let Vp = config.padded_vocab_size
        let C = config.channels

        // wte: pad from (V, C) to (Vp, C)
        let wteScalars = await model.params.wte.shapedArray(of: Float.self).scalars
        var paddedWte = [Float](repeating: 0, count: Vp * C)
        for row in 0..<V {
            paddedWte.replaceSubrange((row * C)..<(row * C + C), with: wteScalars[(row * C)..<(row * C + C)])
        }
        result.append(contentsOf: paddedWte)

        result.append(contentsOf: await model.params.wpe.shapedArray(of: Float.self).scalars)
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.ln1w[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.ln1b[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.qkvw[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.qkvb[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.attprojw[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.attprojb[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.ln2w[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.ln2b[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.fcw[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.fcb[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.fcprojw[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await model.params.fcprojb[l].shapedArray(of: Float.self).scalars) }
        result.append(contentsOf: await model.params.lnfw.shapedArray(of: Float.self).scalars)
        result.append(contentsOf: await model.params.lnfb.shapedArray(of: Float.self).scalars)
        return result
    }

    private func mlTensorGradients(grads: LLMMLTensor.MLParameterTensors, config: LLMGPT2Config) async -> [Float] {
        var result = [Float]()
        result.reserveCapacity(config.num_parameters)

        let V = config.vocab_size
        let Vp = config.padded_vocab_size
        let C = config.channels

        // dwte: pad from (V, C) to (Vp, C)
        let dwteScalars = await grads.wte.shapedArray(of: Float.self).scalars
        var paddedDwte = [Float](repeating: 0, count: Vp * C)
        for row in 0..<V {
            paddedDwte.replaceSubrange((row * C)..<(row * C + C), with: dwteScalars[(row * C)..<(row * C + C)])
        }
        result.append(contentsOf: paddedDwte)

        // dwpe: may be (T, C) rather than (maxT, C), pad to full size
        let dwpeScalars = await grads.wpe.shapedArray(of: Float.self).scalars
        var paddedDwpe = [Float](repeating: 0, count: config.max_seq_len * C)
        paddedDwpe.replaceSubrange(0..<dwpeScalars.count, with: dwpeScalars)
        result.append(contentsOf: paddedDwpe)
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.ln1w[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.ln1b[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.qkvw[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.qkvb[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.attprojw[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.attprojb[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.ln2w[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.ln2b[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.fcw[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.fcb[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.fcprojw[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { result.append(contentsOf: await grads.fcprojb[l].shapedArray(of: Float.self).scalars) }
        result.append(contentsOf: await grads.lnfw.shapedArray(of: Float.self).scalars)
        result.append(contentsOf: await grads.lnfb.shapedArray(of: Float.self).scalars)
        return result
    }

    private func selectedMLTensorActivations(acts: LLMMLTensor.MLActivationTensors, config: LLMGPT2Config, B: Int, T: Int) async -> [Float] {
        let Vp = config.padded_vocab_size
        let V = config.vocab_size
        var values = [Float]()

        values.append(contentsOf: await acts.encoded.shapedArray(of: Float.self).scalars)
        for l in 0..<config.num_layers { values.append(contentsOf: await acts.ln1[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { values.append(contentsOf: await acts.qkv[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { values.append(contentsOf: await acts.atty[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { values.append(contentsOf: await acts.att[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { values.append(contentsOf: await acts.attproj[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { values.append(contentsOf: await acts.fch_gelu[l].shapedArray(of: Float.self).scalars) }
        for l in 0..<config.num_layers { values.append(contentsOf: await acts.residual3[l].shapedArray(of: Float.self).scalars) }
        values.append(contentsOf: await acts.lnf.shapedArray(of: Float.self).scalars)

        // logits and probs: MLTensor uses V (unpadded), reference uses Vp (padded)
        let logitsScalars = await acts.logits.shapedArray(of: Float.self).scalars
        let probsScalars = await acts.probs.shapedArray(of: Float.self).scalars
        // Pad logits from (B*T, V) to (B*T, Vp)
        for bt in 0..<(B * T) {
            values.append(contentsOf: logitsScalars[(bt * V)..<(bt * V + V)])
            values.append(contentsOf: [Float](repeating: 0, count: Vp - V))
        }
        // Pad probs from (B*T, V) to (B*T, Vp)
        for bt in 0..<(B * T) {
            values.append(contentsOf: probsScalars[(bt * V)..<(bt * V + V)])
            values.append(contentsOf: [Float](repeating: 0, count: Vp - V))
        }
        values.append(contentsOf: await acts.losses.shapedArray(of: Float.self).scalars)
        return values
    }

    private func collectTrainingProgress(
        from stream: AsyncThrowingStream<LLMTrainingProgress, Error>
    ) async throws -> [LLMTrainingProgress] {
        var progress: [LLMTrainingProgress] = []
        for try await sample in stream {
            progress.append(sample)
        }
        return progress
    }

    private func unwrapTrainingLosses(
        from progress: [LLMTrainingProgress],
        engineName: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> [Double] {
        try progress.enumerated().map { index, sample in
            guard let trainingLoss = sample.trainingLoss else {
                XCTFail("\(engineName) step \(index + 1) did not report a training loss", file: file, line: line)
                struct MissingTrainingLoss: Error {}
                throw MissingTrainingLoss()
            }
            return trainingLoss
        }
    }

    private func attachTrainingSummary(
        for progress: [LLMTrainingProgress],
        engineName: String,
        wallClockSeconds: Double
    ) {
        guard progress.isEmpty == false else {
            return
        }

        let averageIterationsPerSecond = progress.compactMap(\.iterationsPerSecond).average
        let averageForwardMilliseconds = progress.compactMap(\.forwardPassMilliseconds).average
        let averageBackwardMilliseconds = progress.compactMap(\.backwardPassMilliseconds).average
        let lossSummary = progress.enumerated().compactMap { index, sample -> String? in
            guard let trainingLoss = sample.trainingLoss else {
                return nil
            }
            return "step \(index + 1): \(String(format: "%.5f", trainingLoss))"
        }.joined(separator: ", ")

        let summary = """
        Engine: \(engineName)
        Steps: \(progress.count)
        Wall clock seconds: \(String(format: "%.3f", wallClockSeconds))
        Average iterations/s: \(averageIterationsPerSecond.map { String(format: "%.4f", $0) } ?? "n/a")
        Average forward ms: \(averageForwardMilliseconds.map { String(format: "%.3f", $0) } ?? "n/a")
        Average backward ms: \(averageBackwardMilliseconds.map { String(format: "%.3f", $0) } ?? "n/a")
        Training losses: \(lossSummary)
        """

        print(summary)
    }

    private enum ExecutionPhase {
        case forward
        case backward
        case trainingStep
    }

    private func snapshot(for engine: EngineCase, fixture: SyntheticFixture, phase: ExecutionPhase) throws -> ModelSnapshot {
        switch engine {
        case .basicSwift:
            var model = try LLMBasicSwift.buildModel(from: fixture.checkpointData)
            return try execute(
                model: &model,
                fixture: fixture,
                phase: phase,
                forward: { LLMBasicSwift.gpt2_forward(model: &$0, inputs: $1, targets: $2, B: $3, T: $4) },
                zeroGrad: { LLMBasicSwift.gpt2_zero_grad(model: &$0) },
                backward: { LLMBasicSwift.gpt2_backward(model: &$0) },
                update: { LLMBasicSwift.gpt2_update(model: &$0, update_params: $1) },
                parameters: { $0.params.flattened() },
                gradients: { $0.grads.flattened() },
                activations: { selectedForwardActivations(from: $0.acts) },
                meanLoss: { $0.mean_loss }
            )
        case .fastSwift:
            var model = try LLMSwift.buildModel(from: fixture.checkpointData)
            return try execute(
                model: &model,
                fixture: fixture,
                phase: phase,
                forward: { LLMSwift.gpt2_forward(model: &$0, inputs: $1, targets: $2, B: $3, T: $4) },
                zeroGrad: { LLMSwift.gpt2_zero_grad(model: &$0) },
                backward: { LLMSwift.gpt2_backward(model: &$0) },
                update: { LLMSwift.gpt2_update(model: &$0, update_params: $1) },
                parameters: { $0.params.flattened() },
                gradients: { $0.grads.flattened() },
                activations: { selectedForwardActivations(from: $0.acts) },
                meanLoss: { $0.mean_loss }
            )
        case .blas:
            var model = try LLMBLAS.buildModel(from: fixture.checkpointData)
            return try execute(
                model: &model,
                fixture: fixture,
                phase: phase,
                forward: { LLMBLAS.gpt2_forward(model: &$0, inputs: $1, targets: $2, B: $3, T: $4) },
                zeroGrad: { LLMBLAS.gpt2_zero_grad(model: &$0) },
                backward: { LLMBLAS.gpt2_backward(model: &$0) },
                update: { LLMBLAS.gpt2_update(model: &$0, update_params: $1) },
                parameters: { $0.params.flattened() },
                gradients: { $0.grads.flattened() },
                activations: { selectedForwardActivations(from: $0.acts) },
                meanLoss: { $0.mean_loss }
            )
        case .amx:
            var model = try LLMAMX.buildModel(from: fixture.checkpointData)
            return try execute(
                model: &model,
                fixture: fixture,
                phase: phase,
                forward: { LLMAMX.gpt2_forward(model: &$0, inputs: $1, targets: $2, B: $3, T: $4) },
                zeroGrad: { LLMAMX.gpt2_zero_grad(model: &$0) },
                backward: { LLMAMX.gpt2_backward(model: &$0) },
                update: { LLMAMX.gpt2_update(model: &$0, update_params: $1) },
                parameters: { $0.params.flattened() },
                gradients: { $0.grads.flattened() },
                activations: { selectedForwardActivations(from: $0.acts) },
                meanLoss: { $0.mean_loss }
            )
        case .bnns:
            var model = try LLMBNNS.buildModel(from: fixture.checkpointData)
            return try execute(
                model: &model,
                fixture: fixture,
                phase: phase,
                forward: { try LLMBNNS.gpt2_forward(model: &$0, inputs: $1, targets: $2, B: $3, T: $4) },
                zeroGrad: { LLMBNNS.gpt2_zero_grad(model: &$0) },
                backward: { LLMBNNS.gpt2_backward(model: &$0) },
                update: { LLMBNNS.gpt2_update(model: &$0, update_params: $1) },
                parameters: { $0.params.flattened() },
                gradients: { $0.grads.flattened() },
                activations: { selectedForwardActivations(from: $0.acts) },
                meanLoss: { $0.mean_loss }
            )
        }
    }

    private func execute<Model>(
        model: inout Model,
        fixture: SyntheticFixture,
        phase: ExecutionPhase,
        forward: (inout Model, [UInt32], [UInt32], Int, Int) throws -> Void,
        zeroGrad: (inout Model) -> Void,
        backward: (inout Model) -> Void,
        update: (inout Model, LLMSwift.UpdateParams) -> Void,
        parameters: (Model) -> [Float],
        gradients: (Model) -> [Float],
        activations: (Model) -> [Float],
        meanLoss: (Model) -> Float
    ) throws -> ModelSnapshot {
        try forward(&model, fixture.inputs, fixture.targets, fixture.batchSize, fixture.sequenceLength)
        if phase != .forward {
            zeroGrad(&model)
            backward(&model)
        }
        if phase == .trainingStep {
            update(&model, fixture.updateParams)
            try forward(&model, fixture.inputs, fixture.targets, fixture.batchSize, fixture.sequenceLength)
        }

        return ModelSnapshot(
            meanLoss: meanLoss(model),
            parameters: parameters(model),
            gradients: gradients(model),
            activations: activations(model)
        )
    }

    private func selectedForwardActivations(from activations: LLMGPT2ActivationTensors) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(
            activations.encoded.count +
            activations.lnf.count +
            activations.logits.count +
            activations.probs.count +
            activations.losses.count +
            activations.ln1.reduce(0) { $0 + $1.count } +
            activations.qkv.reduce(0) { $0 + $1.count } +
            activations.atty.reduce(0) { $0 + $1.count } +
            activations.att.reduce(0) { $0 + $1.count } +
            activations.attproj.reduce(0) { $0 + $1.count } +
            activations.fch_gelu.reduce(0) { $0 + $1.count } +
            activations.residual3.reduce(0) { $0 + $1.count }
        )
        values.append(contentsOf: activations.encoded)
        values.append(contentsOf: activations.ln1.flatMap { $0 })
        values.append(contentsOf: activations.qkv.flatMap { $0 })
        values.append(contentsOf: activations.atty.flatMap { $0 })
        values.append(contentsOf: activations.att.flatMap { $0 })
        values.append(contentsOf: activations.attproj.flatMap { $0 })
        values.append(contentsOf: activations.fch_gelu.flatMap { $0 })
        values.append(contentsOf: activations.residual3.flatMap { $0 })
        values.append(contentsOf: activations.lnf)
        values.append(contentsOf: activations.logits)
        values.append(contentsOf: activations.probs)
        values.append(contentsOf: activations.losses)
        return values
    }

    private func assertSnapshotMatchesReference(
        _ snapshot: ModelSnapshot,
        reference: ModelSnapshot,
        tolerance: ComparisonTolerance,
        engineName: String,
        phaseName: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let compareGradients = phaseName != "forward"
        let compareParameters = phaseName == "training step"

        XCTAssertEqual(
            snapshot.meanLoss,
            reference.meanLoss,
            accuracy: tolerance.loss,
            "\(engineName) \(phaseName) loss mismatch",
            file: file,
            line: line
        )
        assertMaxDifference(
            actual: snapshot.activations,
            expected: reference.activations,
            accuracy: tolerance.activations,
            label: "\(engineName) \(phaseName) activations",
            file: file,
            line: line
        )
        if compareGradients {
            assertMaxDifference(
                actual: snapshot.gradients,
                expected: reference.gradients,
                accuracy: tolerance.gradients,
                label: "\(engineName) \(phaseName) gradients",
                file: file,
                line: line
            )
        }
        if compareParameters {
            assertMaxDifference(
                actual: snapshot.parameters,
                expected: reference.parameters,
                accuracy: tolerance.parameters,
                label: "\(engineName) \(phaseName) parameters",
                file: file,
                line: line
            )
        }
    }

    private func assertCheckpointRoundTrip(
        _ data: Data,
        expected: (header: LLMCheckpointHeader, parameters: [Float]),
        engineName: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let decoded = try LLMCheckpointCodec.decode(data)
        XCTAssertEqual(decoded.header, expected.header, "\(engineName) round-trip header mismatch", file: file, line: line)
        assertMaxDifference(
            actual: decoded.parameters,
            expected: expected.parameters,
            accuracy: 0,
            label: "\(engineName) round-trip parameters",
            file: file,
            line: line
        )
    }

    private func assertMaxDifference(
        actual: [Float],
        expected: [Float],
        accuracy: Float,
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "\(label) count mismatch", file: file, line: line)
        guard actual.count == expected.count else { return }

        var maxDifference: Float = 0
        var maxIndex = 0
        for index in actual.indices {
            let difference = abs(actual[index] - expected[index])
            if difference > maxDifference {
                maxDifference = difference
                maxIndex = index
            }
        }

        XCTAssertLessThanOrEqual(
            maxDifference,
            accuracy,
            "\(label) max diff \(maxDifference) at index \(maxIndex) exceeds \(accuracy)",
            file: file,
            line: line
        )
    }
}

private final class CReferenceHarness {
    private let model = UnsafeMutablePointer<GPT2>.allocate(capacity: 1)
    private let baseURL: URL

    init(checkpointData: Data) throws {
        baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
        let checkpointURL = baseURL.appendingPathComponent("checkpoint.bin")
        try checkpointData.write(to: checkpointURL)
        model.initialize(to: GPT2())
        gpt2_build_from_checkpoint(model, checkpointURL.path)
    }

    deinit {
        if model.pointee.params_memory != nil {
            gpt2_free(model)
        }
        model.deallocate()
        try? FileManager.default.removeItem(at: baseURL)
    }

    func runForward(inputs: [UInt32], targets: [UInt32], B: Int, T: Int) throws -> ModelSnapshot {
        try run(inputs: inputs, targets: targets, B: B, T: T, includeBackward: false, updateParams: nil)
    }

    func runBackward(inputs: [UInt32], targets: [UInt32], B: Int, T: Int) throws -> ModelSnapshot {
        try run(inputs: inputs, targets: targets, B: B, T: T, includeBackward: true, updateParams: nil)
    }

    func runTrainingStep(
        inputs: [UInt32],
        targets: [UInt32],
        B: Int,
        T: Int,
        updateParams: LLMSwift.UpdateParams
    ) throws -> ModelSnapshot {
        try run(inputs: inputs, targets: targets, B: B, T: T, includeBackward: true, updateParams: updateParams)
    }

    private func run(
        inputs: [UInt32],
        targets: [UInt32],
        B: Int,
        T: Int,
        includeBackward: Bool,
        updateParams: LLMSwift.UpdateParams?
    ) throws -> ModelSnapshot {
        var inputs = try convertToCInts(inputs)
        var targets = try convertToCInts(targets)

        inputs.withUnsafeMutableBufferPointer { inputBuffer in
            targets.withUnsafeMutableBufferPointer { targetBuffer in
                gpt2_forward(model, inputBuffer.baseAddress, targetBuffer.baseAddress, B, T)
            }
        }

        if includeBackward {
            gpt2_zero_grad(model)
            gpt2_backward(model)
        }

        if let updateParams {
            gpt2_update(
                model,
                updateParams.learning_rate,
                updateParams.beta1,
                updateParams.beta2,
                updateParams.eps,
                updateParams.weight_decay,
                Int32(updateParams.t)
            )
            inputs.withUnsafeMutableBufferPointer { inputBuffer in
                targets.withUnsafeMutableBufferPointer { targetBuffer in
                    gpt2_forward(model, inputBuffer.baseAddress, targetBuffer.baseAddress, B, T)
                }
            }
        }

        return ModelSnapshot(
            meanLoss: model.pointee.mean_loss,
            parameters: copy(pointer: model.pointee.params_memory, count: Int(model.pointee.num_parameters)),
            gradients: copy(pointer: model.pointee.grads_memory, count: Int(model.pointee.num_parameters)),
            activations: selectedForwardActivations(B: B, T: T)
        )
    }

    private func convertToCInts(_ values: [UInt32]) throws -> [Int32] {
        try values.map { value in
            guard let converted = Int32(exactly: value) else {
                struct ConversionError: Error {}
                throw ConversionError()
            }
            return converted
        }
    }

    private func copy(pointer: UnsafeMutablePointer<Float>?, count: Int) -> [Float] {
        guard let pointer, count > 0 else {
            return []
        }
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func selectedForwardActivations(B: Int, T: Int) -> [Float] {
        let config = model.pointee.config
        let L = Int(config.num_layers)
        let NH = Int(config.num_heads)
        let C = Int(config.channels)
        let Vp = Int(config.padded_vocab_size)

        var values: [Float] = []
        values.reserveCapacity(
            B * T * C +
            L * B * T * C +
            L * B * T * 3 * C +
            L * B * T * C +
            L * B * NH * T * T +
            L * B * T * C +
            L * B * T * 4 * C +
            L * B * T * C +
            B * T * C +
            B * T * Vp +
            B * T * Vp +
            B * T
        )
        values.append(contentsOf: copy(pointer: model.pointee.acts.encoded, count: B * T * C))
        values.append(contentsOf: copy(pointer: model.pointee.acts.ln1, count: L * B * T * C))
        values.append(contentsOf: copy(pointer: model.pointee.acts.qkv, count: L * B * T * 3 * C))
        values.append(contentsOf: copy(pointer: model.pointee.acts.atty, count: L * B * T * C))
        values.append(contentsOf: copy(pointer: model.pointee.acts.att, count: L * B * NH * T * T))
        values.append(contentsOf: copy(pointer: model.pointee.acts.attproj, count: L * B * T * C))
        values.append(contentsOf: copy(pointer: model.pointee.acts.fch_gelu, count: L * B * T * 4 * C))
        values.append(contentsOf: copy(pointer: model.pointee.acts.residual3, count: L * B * T * C))
        values.append(contentsOf: copy(pointer: model.pointee.acts.lnf, count: B * T * C))
        values.append(contentsOf: copy(pointer: model.pointee.acts.logits, count: B * T * Vp))
        values.append(contentsOf: copy(pointer: model.pointee.acts.probs, count: B * T * Vp))
        values.append(contentsOf: copy(pointer: model.pointee.acts.losses, count: B * T))
        return values
    }
}
