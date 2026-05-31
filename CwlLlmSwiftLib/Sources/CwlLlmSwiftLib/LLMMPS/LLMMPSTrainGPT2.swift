import Foundation
import Metal
import MetalPerformanceShadersGraph

enum LLMMPS {
    typealias GPT2Config = LLMGPT2Config

    struct MPSParameterTensors {
        let wte: MPSGraphTensor
        let wpe: MPSGraphTensor
        let ln1w: [MPSGraphTensor]
        let ln1b: [MPSGraphTensor]
        let qkvw: [MPSGraphTensor]
        let qkvb: [MPSGraphTensor]
        let attprojw: [MPSGraphTensor]
        let attprojb: [MPSGraphTensor]
        let ln2w: [MPSGraphTensor]
        let ln2b: [MPSGraphTensor]
        let fcw: [MPSGraphTensor]
        let fcb: [MPSGraphTensor]
        let fcprojw: [MPSGraphTensor]
        let fcprojb: [MPSGraphTensor]
        let lnfw: MPSGraphTensor
        let lnfb: MPSGraphTensor
        let trainingStep: MPSGraphTensor

        var allVariables: [MPSGraphTensor] {
            [wte, wpe]
                + ln1w + ln1b
                + qkvw + qkvb
                + attprojw + attprojb
                + ln2w + ln2b
                + fcw + fcb
                + fcprojw + fcprojb
                + [lnfw, lnfb]
        }

        func variablesMomentumsVelocities(graph: MPSGraph) -> [(variable: MPSGraphTensor, momentum: MPSGraphTensor, velocity: MPSGraphTensor)] {
            allVariables.map {
                (
                    variable: $0,
                    momentum: graph.variableFromTensor(graph.broadcast(graph.constant(0, dataType: .float32), shape: $0.shape!, name: "momentumValues"), name: "momentum"),
                    velocity: graph.variableFromTensor(graph.broadcast(graph.constant(0, dataType: .float32), shape: $0.shape!, name: "velocityValues"), name: "velocity")
                )
            }
        }
    }

    struct CachedTrainingExecutables {
        let B: Int
        let T: Int
        let learningRate: Float
        let training: MPSGraphExecutable
        let lossEstimation: MPSGraphExecutable
    }

    struct CachedInferenceExecutable {
        let B: Int
        let T: Int
        let executable: MPSGraphExecutable
    }

    final class CachedLatestTokenIO {
        let B: Int
        let T: Int
        let inputBuffer: MTLBuffer
        let rowIndexBuffer: MTLBuffer
        let logitsBuffer: MTLBuffer
        let inputData: MPSGraphTensorData
        let rowIndexData: MPSGraphTensorData
        let logitsData: MPSGraphTensorData

        init?(device: MTLDevice, B: Int, T: Int, vocabularySize: Int) {
            let inputByteCount = B * T * MemoryLayout<UInt32>.stride
            let logitsByteCount = vocabularySize * MemoryLayout<Float>.stride
            guard let inputBuffer = device.makeBuffer(length: inputByteCount, options: .storageModeShared),
                  let rowIndexBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared),
                  let logitsBuffer = device.makeBuffer(length: logitsByteCount, options: .storageModeShared) else {
                return nil
            }

            self.B = B
            self.T = T
            self.inputBuffer = inputBuffer
            self.rowIndexBuffer = rowIndexBuffer
            self.logitsBuffer = logitsBuffer
            self.inputData = MPSGraphTensorData(inputBuffer, shape: [B, T] as [NSNumber], dataType: .uInt32)
            self.rowIndexData = MPSGraphTensorData(rowIndexBuffer, shape: [1] as [NSNumber], dataType: .int32)
            self.logitsData = MPSGraphTensorData(logitsBuffer, shape: [B, 1, vocabularySize] as [NSNumber], dataType: .float32)
        }
    }

    // MARK: - Graph layer operations

    static func encoder_forward(
        graph: MPSGraph,
        inp: MPSGraphTensor,
        wte: MPSGraphTensor,
        wpe: MPSGraphTensor,
        B: Int,
        T: Int,
        C: Int,
        requiresPositionGradient: Bool
    ) -> MPSGraphTensor {
        // Token embedding lookup: [V, C] gathered by indices [B, T] -> [B, T, C].
        // The gather op requires int32/int64 indices.
        let indices = graph.cast(inp, to: .int32, name: "inputIndices")
        let tokenEmbedding = graph.gather(withUpdatesTensor: wte, indicesTensor: indices, axis: 0, batchDimensions: 0, name: "tokenEmbedding")
        let wpeSlice = graph.sliceTensor(wpe, dimension: 0, start: 0, length: T, name: "wpeSlice")
        let expandedWpe = graph.reshape(wpeSlice, shape: [1, T, C] as [NSNumber], name: "expandedWpe")
        let positionEmbedding: MPSGraphTensor
        if requiresPositionGradient {
            // The explicit broadcast lets MPSGraph autodiff resolve the gradient
            // path to wpe. Inference can use normal broadcast semantics.
            let btcZeros = graph.constant(0, shape: [B, T, C] as [NSNumber], dataType: .float32)
            positionEmbedding = graph.addition(expandedWpe, btcZeros, name: "positionEmbedding")
        } else {
            positionEmbedding = expandedWpe
        }
        return graph.addition(tokenEmbedding, positionEmbedding, name: "encoder_forward")
    }

    static func layernorm_forward(graph: MPSGraph, inp: MPSGraphTensor, weight: MPSGraphTensor, bias: MPSGraphTensor) -> MPSGraphTensor {
        let mean = graph.mean(of: inp, axes: [2], name: "mean")
        let variance = graph.variance(of: inp, axes: [2], name: "variance")
        return graph.normalize(inp, mean: mean, variance: variance, gamma: weight, beta: bias, epsilon: 1e-5, name: "out")
    }

    static func matmul_forward(graph: MPSGraph, inp: MPSGraphTensor, weight: MPSGraphTensor, bias: MPSGraphTensor?) -> MPSGraphTensor {
        let transposedWeight = graph.transposeTensor(weight, dimension: 0, withDimension: 1, name: "transposedWeight")
        var output = graph.matrixMultiplication(primary: inp, secondary: transposedWeight, name: "output")
        if let bias {
            output = graph.addition(output, bias, name: "biased_output")
        }
        return output
    }

    static func attention_forward(graph: MPSGraph, inp: MPSGraphTensor, mask: MPSGraphTensor, B: Int, T: Int, C: Int, NH: Int) -> MPSGraphTensor {
        let HS = C / NH
        let reshaped = graph.reshape(inp, shape: [B, T, 3, NH, HS] as [NSNumber], name: "reshaped")
        let q = attentionProjection(graph: graph, reshaped: reshaped, index: 0, name: "q", B: B, T: T, NH: NH, HS: HS)
        let k = attentionProjection(graph: graph, reshaped: reshaped, index: 1, name: "k", B: B, T: T, NH: NH, HS: HS)
        let v = attentionProjection(graph: graph, reshaped: reshaped, index: 2, name: "v", B: B, T: T, NH: NH, HS: HS)
        let preatt = graph.matrixMultiplication(primary: q, secondary: graph.transposeTensor(k, dimension: 2, withDimension: 3, name: "ktranspose"), name: "preatt")
        let preattScaled = graph.multiplication(preatt, graph.constant(1 / Double(HS).squareRoot(), dataType: .float32), name: "preattScaled")

        let masked = graph.addition(preattScaled, mask, name: "masked")
        let att = graph.softMax(with: masked, axis: 3, name: "att")
        let out = graph.matrixMultiplication(primary: att, secondary: v, name: "atty")

        return graph.reshape(graph.transposeTensor(out, dimension: 1, withDimension: 2, name: "transposedOut"), shape: [B, T, C] as [NSNumber], name: "reshaped_atty")
    }

    private static func attentionProjection(graph: MPSGraph, reshaped: MPSGraphTensor, index: Int, name: String, B: Int, T: Int, NH: Int, HS: Int) -> MPSGraphTensor {
        let slice = graph.sliceTensor(reshaped, dimension: 2, start: index, length: 1, name: "\(name)Slice")
        let squeezed = graph.reshape(slice, shape: [B, T, NH, HS] as [NSNumber], name: "\(name)Squeezed")
        return graph.transposeTensor(squeezed, dimension: 1, withDimension: 2, name: name)
    }

    /// Builds a `[B, NH, T, T]` additive causal mask (0 below diagonal, -inf above) once per executable.
    static func causal_mask(graph: MPSGraph, B: Int, NH: Int, T: Int) -> MPSGraphTensor {
        let t1Coord = graph.coordinate(alongAxis: 2, withShape: [B, NH, T, T] as [NSNumber], name: "t1Coord")
        let t2Coord = graph.coordinate(alongAxis: 3, withShape: [B, NH, T, T] as [NSNumber], name: "t2Coord")
        let upper = graph.greaterThan(t2Coord, t1Coord, name: "upperTriangle")
        return graph.select(
            predicate: upper,
            trueTensor: graph.constant(-Double.infinity, dataType: .float32),
            falseTensor: graph.constant(0, dataType: .float32),
            name: "causalMask"
        )
    }

    static func gelu_forward(graph: MPSGraph, inp: MPSGraphTensor) -> MPSGraphTensor {
        let inpCubed = graph.power(inp, graph.constant(3, dataType: .float32), name: "inpCubed")
        let cubeTerm = graph.multiplication(inpCubed, graph.constant(0.044715, dataType: .float32), name: "cubeTerm")
        let inpPlusCubeTerm = graph.addition(inp, cubeTerm, name: "inpPlusCubeTerm")
        let geluScalingFactor = graph.constant((2 / Double.pi).squareRoot(), dataType: .float32)
        let tanhArg = graph.multiplication(geluScalingFactor, inpPlusCubeTerm, name: "tanhArg")
        let tanhOut = graph.tanh(with: tanhArg, name: "tanhOut")
        let onePlusTanh = graph.addition(graph.constant(1, dataType: .float32), tanhOut, name: "onePlusTanh")
        let halfInp = graph.multiplication(inp, graph.constant(0.5, dataType: .float32), name: "halfInp")
        return graph.multiplication(halfInp, onePlusTanh, name: "gelu")
    }

    static func residual_forward(graph: MPSGraph, inp1: MPSGraphTensor, inp2: MPSGraphTensor) -> MPSGraphTensor {
        graph.addition(inp1, inp2, name: "residual")
    }
}

// MARK: - GPT2MPS

final class GPT2MPS {
    let commandQueue: MTLCommandQueue
    let config: LLMGPT2Config
    let device: MPSGraphDevice
    let graph = MPSGraph()
    let parameters: LLMMPS.MPSParameterTensors
    var cachedTrainingExecutables: LLMMPS.CachedTrainingExecutables?
    var cachedFullInferenceExecutable: LLMMPS.CachedInferenceExecutable?
    var cachedLatestTokenInferenceExecutable: LLMMPS.CachedInferenceExecutable?
    var cachedLatestTokenIO: LLMMPS.CachedLatestTokenIO?

    static func variable(_ graph: MPSGraph, _ shape: [Int], _ data: Data, _ offset: inout Int, _ name: String) -> MPSGraphTensor {
        let size = MemoryLayout<Float>.stride * shape.reduce(1, *)
        let variable = graph.variable(with: data[offset..<(offset + size)], shape: shape as [NSNumber], dataType: .float32, name: name)
        offset += size
        return variable
    }

    static func variable(_ graph: MPSGraph, sourceShape: (Int, Int), targetShape: (Int, Int), _ data: Data, _ offset: inout Int, _ name: String) -> MPSGraphTensor {
        let sourceSize = MemoryLayout<Float>.stride * sourceShape.0 * sourceShape.1
        var reshapedData = Data(capacity: MemoryLayout<Float>.stride * targetShape.0 * targetShape.1)
        for outer in 0..<targetShape.0 {
            reshapedData.append(
                data.subdata(in: (offset + outer * sourceShape.1 * MemoryLayout<Float>.stride)..<(offset + (outer * sourceShape.1 + targetShape.1) * MemoryLayout<Float>.stride))
            )
        }
        let variable = graph.variable(with: reshapedData, shape: [targetShape.0, targetShape.1] as [NSNumber], dataType: .float32, name: name)
        offset += sourceSize
        return variable
    }

    init(config: LLMGPT2Config, parameterData: Data) throws {
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            throw LLMMPSRuntimeError.noMetalDevice
        }
        guard let queue = metalDevice.makeCommandQueue() else {
            throw LLMMPSRuntimeError.failedToCreateCommandQueue
        }
        self.config = config
        self.device = MPSGraphDevice(mtlDevice: metalDevice)
        self.commandQueue = queue

        let (maxT, V, Vp, L, _, C) = (config.max_seq_len, config.vocab_size, config.padded_vocab_size, config.num_layers, config.num_heads, config.channels)

        var offset = 0
        self.parameters = LLMMPS.MPSParameterTensors(
            wte: Self.variable(graph, sourceShape: (Vp, C), targetShape: (V, C), parameterData, &offset, "wte"),
            wpe: Self.variable(graph, [maxT, C], parameterData, &offset, "wpe"),
            ln1w: (0..<L).map { [graph] l in Self.variable(graph, [C], parameterData, &offset, "ln1w-\(l)") },
            ln1b: (0..<L).map { [graph] l in Self.variable(graph, [C], parameterData, &offset, "ln1b-\(l)") },
            qkvw: (0..<L).map { [graph] l in Self.variable(graph, [(3 * C), C], parameterData, &offset, "qkvw-\(l)") },
            qkvb: (0..<L).map { [graph] l in Self.variable(graph, [(3 * C)], parameterData, &offset, "qkvb-\(l)") },
            attprojw: (0..<L).map { [graph] l in Self.variable(graph, [C, C], parameterData, &offset, "attprojw-\(l)") },
            attprojb: (0..<L).map { [graph] l in Self.variable(graph, [C], parameterData, &offset, "attprojb-\(l)") },
            ln2w: (0..<L).map { [graph] l in Self.variable(graph, [C], parameterData, &offset, "ln2w-\(l)") },
            ln2b: (0..<L).map { [graph] l in Self.variable(graph, [C], parameterData, &offset, "ln2b-\(l)") },
            fcw: (0..<L).map { [graph] l in Self.variable(graph, [(4 * C), C], parameterData, &offset, "fcw-\(l)") },
            fcb: (0..<L).map { [graph] l in Self.variable(graph, [(4 * C)], parameterData, &offset, "fcb-\(l)") },
            fcprojw: (0..<L).map { [graph] l in Self.variable(graph, [C, (4 * C)], parameterData, &offset, "fcprojw-\(l)") },
            fcprojb: (0..<L).map { [graph] l in Self.variable(graph, [C], parameterData, &offset, "fcprojb-\(l)") },
            lnfw: Self.variable(graph, [C], parameterData, &offset, "lnfw"),
            lnfb: Self.variable(graph, [C], parameterData, &offset, "lnfb"),
            trainingStep: graph.variableFromTensor(graph.constant(1, dataType: .float32), name: "step")
        )
    }

    func buildTrainingExecutables(B: Int, T: Int, learningRate: Float) -> LLMMPS.CachedTrainingExecutables {
        let inputPlaceholder = graph.placeholder(shape: [B, T] as [NSNumber], dataType: .uInt32, name: "input")
        let targetPlaceholder = graph.placeholder(shape: [B, T] as [NSNumber], dataType: .uInt32, name: "target")
        let encoded = buildTransformer(inputPlaceholder: inputPlaceholder, B: B, T: T, requiresPositionGradient: true)
        let finalHidden = LLMMPS.layernorm_forward(graph: graph, inp: encoded, weight: parameters.lnfw, bias: parameters.lnfb)
        let logits = LLMMPS.matmul_forward(graph: graph, inp: finalHidden, weight: parameters.wte, bias: nil)

        let targetOneHot = graph.oneHot(withIndicesTensor: targetPlaceholder, depth: config.vocab_size, name: "targetOneHot")
        let meanLoss = graph.softMaxCrossEntropy(logits, labels: targetOneHot, axis: 2, reuctionType: .mean, name: "meanLoss")

        let variablesMomentumsVelocities = parameters.variablesMomentumsVelocities(graph: graph)
        let gradTensors = graph.gradients(of: meanLoss, with: variablesMomentumsVelocities.map(\.variable), name: nil)
        let beta1 = graph.constant(0.9, dataType: .float32)
        let beta2 = graph.constant(0.999, dataType: .float32)
        var updateOperations = [MPSGraphOperation]()
        for variableMomentumVelocity in variablesMomentumsVelocities {
            let gradient = gradTensors[variableMomentumVelocity.variable]!
            let updateMomentumVelocity = graph.adam(
                learningRate: graph.constant(Double(learningRate), dataType: .float32),
                beta1: beta1,
                beta2: beta2,
                epsilon: graph.constant(1e-8, dataType: .float32),
                beta1Power: graph.power(beta1, parameters.trainingStep, name: "beta1pow"),
                beta2Power: graph.power(beta2, parameters.trainingStep, name: "beta2pow"),
                values: variableMomentumVelocity.variable,
                momentum: variableMomentumVelocity.momentum,
                velocity: variableMomentumVelocity.velocity,
                maximumVelocity: nil,
                gradient: gradient,
                name: "adamW"
            )
            updateOperations.append(graph.assign(variableMomentumVelocity.variable, tensor: updateMomentumVelocity[0], name: nil))
            updateOperations.append(graph.assign(variableMomentumVelocity.momentum, tensor: updateMomentumVelocity[1], name: nil))
            updateOperations.append(graph.assign(variableMomentumVelocity.velocity, tensor: updateMomentumVelocity[2], name: nil))
        }
        let incrementStep = graph.addition(graph.constant(1, dataType: .float32), parameters.trainingStep, name: nil)
        updateOperations.append(graph.assign(parameters.trainingStep, tensor: incrementStep, name: nil))

        let inputTargetFeeds = [
            inputPlaceholder: MPSGraphShapedType(shape: [B, T] as [NSNumber], dataType: .uInt32),
            targetPlaceholder: MPSGraphShapedType(shape: [B, T] as [NSNumber], dataType: .uInt32)
        ]
        let training = graph.compile(with: device, feeds: inputTargetFeeds, targetTensors: [meanLoss], targetOperations: updateOperations, compilationDescriptor: nil)
        let lossEstimation = graph.compile(with: device, feeds: inputTargetFeeds, targetTensors: [meanLoss], targetOperations: nil, compilationDescriptor: nil)

        let executables = LLMMPS.CachedTrainingExecutables(
            B: B,
            T: T,
            learningRate: learningRate,
            training: training,
            lossEstimation: lossEstimation
        )
        self.cachedTrainingExecutables = executables
        return executables
    }

    private func buildTransformer(
        inputPlaceholder: MPSGraphTensor,
        B: Int,
        T: Int,
        requiresPositionGradient: Bool
    ) -> MPSGraphTensor {
        var encoded = LLMMPS.encoder_forward(
            graph: graph,
            inp: inputPlaceholder,
            wte: parameters.wte,
            wpe: parameters.wpe,
            B: B,
            T: T,
            C: config.channels,
            requiresPositionGradient: requiresPositionGradient
        )

        // Build the causal mask once and reuse across every layer.
        let causalMask = LLMMPS.causal_mask(graph: graph, B: B, NH: config.num_heads, T: T)

        for l in 0..<config.num_layers {
            let ln1 = LLMMPS.layernorm_forward(graph: graph, inp: encoded, weight: parameters.ln1w[l], bias: parameters.ln1b[l])
            let qkv = LLMMPS.matmul_forward(graph: graph, inp: ln1, weight: parameters.qkvw[l], bias: parameters.qkvb[l])
            let atty = LLMMPS.attention_forward(graph: graph, inp: qkv, mask: causalMask, B: B, T: T, C: config.channels, NH: config.num_heads)
            let attproj = LLMMPS.matmul_forward(graph: graph, inp: atty, weight: parameters.attprojw[l], bias: parameters.attprojb[l])
            let residual2 = LLMMPS.residual_forward(graph: graph, inp1: encoded, inp2: attproj)
            let ln2 = LLMMPS.layernorm_forward(graph: graph, inp: residual2, weight: parameters.ln2w[l], bias: parameters.ln2b[l])
            let fch = LLMMPS.matmul_forward(graph: graph, inp: ln2, weight: parameters.fcw[l], bias: parameters.fcb[l])
            let fch_gelu = LLMMPS.gelu_forward(graph: graph, inp: fch)
            let fcproj = LLMMPS.matmul_forward(graph: graph, inp: fch_gelu, weight: parameters.fcprojw[l], bias: parameters.fcprojb[l])
            encoded = LLMMPS.residual_forward(graph: graph, inp1: residual2, inp2: fcproj)
        }

        return encoded
    }

    private func ensureTrainingExecutables(B: Int, T: Int, learningRate: Float) -> LLMMPS.CachedTrainingExecutables {
        if let cached = cachedTrainingExecutables,
           cached.B == B,
           cached.T == T,
           cached.learningRate == learningRate {
            return cached
        }
        return buildTrainingExecutables(B: B, T: T, learningRate: learningRate)
    }

    private func ensureFullInferenceExecutable(B: Int, T: Int) -> MPSGraphExecutable {
        if let cached = cachedFullInferenceExecutable, cached.B == B, cached.T == T {
            return cached.executable
        }

        let input = graph.placeholder(shape: [B, T] as [NSNumber], dataType: .uInt32, name: "fullInferenceInput")
        let encoded = buildTransformer(inputPlaceholder: input, B: B, T: T, requiresPositionGradient: false)
        let finalHidden = LLMMPS.layernorm_forward(graph: graph, inp: encoded, weight: parameters.lnfw, bias: parameters.lnfb)
        let logits = LLMMPS.matmul_forward(graph: graph, inp: finalHidden, weight: parameters.wte, bias: nil)
        let feeds = [input: MPSGraphShapedType(shape: [B, T] as [NSNumber], dataType: .uInt32)]
        let executable = graph.compile(with: device, feeds: feeds, targetTensors: [logits], targetOperations: nil, compilationDescriptor: nil)
        cachedFullInferenceExecutable = LLMMPS.CachedInferenceExecutable(B: B, T: T, executable: executable)
        return executable
    }

    private func ensureLatestTokenInferenceExecutable(B: Int, T: Int) -> MPSGraphExecutable {
        if let cached = cachedLatestTokenInferenceExecutable, cached.B == B, cached.T == T {
            return cached.executable
        }

        let input = graph.placeholder(shape: [B, T] as [NSNumber], dataType: .uInt32, name: "latestTokenInput")
        let rowIndex = graph.placeholder(shape: [1] as [NSNumber], dataType: .int32, name: "latestTokenRowIndex")
        let encoded = buildTransformer(inputPlaceholder: input, B: B, T: T, requiresPositionGradient: false)
        let selectedRow = graph.gather(
            withUpdatesTensor: encoded,
            indicesTensor: rowIndex,
            axis: 1,
            batchDimensions: 0,
            name: "latestTokenRow"
        )
        let latestHidden = LLMMPS.layernorm_forward(
            graph: graph,
            inp: selectedRow,
            weight: parameters.lnfw,
            bias: parameters.lnfb
        )
        let latestLogits = LLMMPS.matmul_forward(
            graph: graph,
            inp: latestHidden,
            weight: parameters.wte,
            bias: nil
        )
        let feeds = [
            input: MPSGraphShapedType(shape: [B, T] as [NSNumber], dataType: .uInt32),
            rowIndex: MPSGraphShapedType(shape: [1] as [NSNumber], dataType: .int32)
        ]
        let executable = graph.compile(with: device, feeds: feeds, targetTensors: [latestLogits], targetOperations: nil, compilationDescriptor: nil)
        cachedLatestTokenInferenceExecutable = LLMMPS.CachedInferenceExecutable(B: B, T: T, executable: executable)
        return executable
    }

    private func ensureLatestTokenIO(B: Int, T: Int) -> LLMMPS.CachedLatestTokenIO {
        if let cached = cachedLatestTokenIO, cached.B == B, cached.T == T {
            return cached
        }
        guard let io = LLMMPS.CachedLatestTokenIO(
            device: commandQueue.device,
            B: B,
            T: T,
            vocabularySize: config.vocab_size
        ) else {
            preconditionFailure("Failed to allocate MPSGraph inference buffers.")
        }
        cachedLatestTokenIO = io
        return io
    }

    private func readFloats(from tensorData: MPSGraphTensorData) -> [Float] {
        var array = [Float](repeating: 0, count: tensorData.shape.map { $0.intValue }.reduce(1, *))
        array.withUnsafeMutableBufferPointer { bp in
            tensorData.mpsndarray().readBytes(UnsafeMutableRawPointer(bp.baseAddress!), strideBytes: nil)
        }
        return array
    }

    private func tokenTensorData(_ tokens: [UInt32], B: Int, T: Int) -> MPSGraphTensorData {
        tokens.withUnsafeBufferPointer { bp in
            MPSGraphTensorData(device: device, data: Data(buffer: bp), shape: [B, T] as [NSNumber], dataType: .uInt32)
        }
    }

    private func scalarTensorData(_ value: Float) -> MPSGraphTensorData {
        [value].withUnsafeBufferPointer { bp in
            MPSGraphTensorData(device: device, data: Data(buffer: bp), shape: [1] as [NSNumber], dataType: .float32)
        }
    }

    func performInference(inputs: [UInt32], B: Int, T: Int) -> [Float] {
        let executable = ensureFullInferenceExecutable(B: B, T: T)
        let inputData = tokenTensorData(inputs, B: B, T: T)
        let outputTensors = executable.run(with: commandQueue, inputs: [inputData], results: nil, executionDescriptor: nil)
        return readFloats(from: outputTensors[0])
    }

    func performLatestTokenInference(inputs: [UInt32], rowIndex: Int, B: Int, T: Int) -> [Float] {
        precondition(B == 1, "Latest-token MPS inference requires batch size 1.")
        precondition(inputs.count == B * T, "Latest-token MPS inference input shape mismatch.")
        let executable = ensureLatestTokenInferenceExecutable(B: B, T: T)
        let io = ensureLatestTokenIO(B: B, T: T)
        inputs.withUnsafeBytes { bytes in
            io.inputBuffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        io.rowIndexBuffer.contents().storeBytes(of: Int32(rowIndex), as: Int32.self)
        _ = executable.run(
            with: commandQueue,
            inputs: [io.inputData, io.rowIndexData],
            results: [io.logitsData],
            executionDescriptor: nil
        )
        let logits = io.logitsBuffer.contents().bindMemory(to: Float.self, capacity: config.vocab_size)
        return Array(UnsafeBufferPointer(start: logits, count: config.vocab_size))
    }

    func performLossEstimation(inputs: [UInt32], targets: [UInt32], B: Int, T: Int) -> Float {
        let executables = ensureTrainingExecutables(B: B, T: T, learningRate: 1e-4)
        let inputData = tokenTensorData(inputs, B: B, T: T)
        let targetsData = tokenTensorData(targets, B: B, T: T)
        let outputTensors = executables.lossEstimation.run(with: commandQueue, inputs: [inputData, targetsData], results: nil, executionDescriptor: nil)
        let lossValues = readFloats(from: outputTensors[0])
        return lossValues[0]
    }

    func performTrainingStep(inputs: [UInt32], targets: [UInt32], B: Int, T: Int, learningRate: Float) -> Float {
        let executables = ensureTrainingExecutables(B: B, T: T, learningRate: learningRate)
        let inputData = tokenTensorData(inputs, B: B, T: T)
        let targetsData = tokenTensorData(targets, B: B, T: T)
        let outputTensors = executables.training.run(with: commandQueue, inputs: [inputData, targetsData], results: nil, executionDescriptor: nil)
        let lossValues = readFloats(from: outputTensors[0])
        return lossValues[0]
    }

    func exportCheckpoint() throws -> Data {
        let variables = parameters.allVariables

        let dummyInput = graph.placeholder(shape: [1] as [NSNumber], dataType: .float32, name: "dummyExport")
        let feeds = [dummyInput: MPSGraphShapedType(shape: [1] as [NSNumber], dataType: .float32)]
        let exe = graph.compile(with: device, feeds: feeds, targetTensors: variables, targetOperations: nil, compilationDescriptor: nil)
        let dummyData = scalarTensorData(0)
        let results = exe.run(with: commandQueue, inputs: [dummyData], results: nil, executionDescriptor: nil)

        var allParams = [Float]()
        allParams.reserveCapacity(config.num_parameters)

        for (index, variable) in variables.enumerated() {
            let floats = readFloats(from: results[index])

            if variable === parameters.wte {
                allParams.append(contentsOf: paddedTokenEmbedding(floats))
            } else {
                allParams.append(contentsOf: floats)
            }
        }

        return try LLMCheckpointCodec.encode(
            header: config.checkpointHeader,
            parameters: allParams,
            expectedParameterCount: config.num_parameters
        )
    }

    private func paddedTokenEmbedding(_ floats: [Float]) -> [Float] {
        let V = config.vocab_size
        let Vp = config.padded_vocab_size
        let C = config.channels
        var padded = [Float](repeating: 0, count: Vp * C)
        for row in 0..<V {
            let rowRange = (row * C)..<(row * C + C)
            padded.replaceSubrange(rowRange, with: floats[rowRange])
        }
        return padded
    }
}
