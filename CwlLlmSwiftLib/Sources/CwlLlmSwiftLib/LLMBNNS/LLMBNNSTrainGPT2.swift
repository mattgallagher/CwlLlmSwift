import Accelerate
import Foundation

enum LLMBNNS {
    typealias GPT2Config = LLMGPT2Config
    typealias ParameterTensors = LLMGPT2ParameterTensors
    typealias ActivationTensors = LLMGPT2ActivationTensors
    typealias UpdateParams = LLMSwift.UpdateParams

    struct GraphKey: Hashable {
        let batchSize: Int
        let sequenceLength: Int
    }

    final class FinalBackwardGraph {
        let batchSize: Int
        let sequenceLength: Int
        let context: BNNSGraph.Context

        init(config: GPT2Config, batchSize: Int, sequenceLength: Int) throws {
            self.batchSize = batchSize
            self.sequenceLength = sequenceLength
            let C = config.channels
            let Vp = config.padded_vocab_size
            let BT = batchSize * sequenceLength

            context = try BNNSGraph.makeContext { builder in
                let dout = builder.argument(name: "dout", dataType: Float.self, shape: [BT, Vp])
                let projectionInput = builder.argument(name: "projection_input", dataType: Float.self, shape: [BT, C])
                let layerNormInput = builder.argument(name: "layernorm_input", dataType: Float.self, shape: [BT, C])
                let weight = builder.argument(name: "weight", dataType: Float.self, shape: [Vp, C])
                let lnWeight = builder.argument(name: "ln_weight", dataType: Float.self, shape: [C])
                let mean = builder.argument(name: "mean", dataType: Float.self, shape: [BT])
                let rstd = builder.argument(name: "rstd", dataType: Float.self, shape: [BT])

                let dlnf = dout.matmul(other: weight)
                let dWte = dout.matmul(transpose: true, other: projectionInput)

                let meanExpanded = mean.reshape(to: [BT, 1])
                let rstdExpanded = rstd.reshape(to: [BT, 1])
                let normalized = (layerNormInput - meanExpanded) * rstdExpanded
                let dnorm = dlnf * lnWeight
                let dnormMean = dnorm.mean(axes: [1], keepDimensions: true)
                let dnormNormMean = (dnorm * normalized).mean(axes: [1], keepDimensions: true)
                let dResidual = (dnorm - dnormMean - normalized * dnormNormMean) * rstdExpanded
                let dLnWeight = (normalized * dlnf).sum(axes: [0], keepDimensions: false)
                let dLnBias = dlnf.sum(axes: [0], keepDimensions: false)

                return [dResidual, dWte, dLnWeight, dLnBias]
            }
        }

        func execute(model: inout GraphModel) throws {
            let BT = batchSize * sequenceLength
            let C = model.config.channels
            let Vp = model.config.padded_vocab_size
            let L = model.config.num_layers
            let strideBTC = bnnsRowMajorStride(for: [BT, C])
            let strideBTVp = bnnsRowMajorStride(for: [BT, Vp])
            let strideVpC = bnnsRowMajorStride(for: [Vp, C])

            var arguments: [BNNSTensor] = [
                // Outputs (graph builder return order: dResidual, dWte, dLnWeight, dLnBias)
                BNNSTensor(data: model.grads_acts.residual3[L - 1].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.grads.wte.bnnsPinnedBuffer(), shape: [Vp, C], stride: strideVpC),
                BNNSTensor(data: model.grads.lnfw.bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.grads.lnfb.bnnsPinnedBuffer(), shape: [C], stride: [1]),
                // Inputs (graph builder argument order: dout, projectionInput, layerNormInput, weight, lnWeight, mean, rstd)
                BNNSTensor(data: model.grads_acts.logits.bnnsPinnedBuffer(), shape: [BT, Vp], stride: strideBTVp),
                BNNSTensor(data: model.acts.lnf.bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.acts.residual3[L - 1].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.params.wte.bnnsPinnedBuffer(), shape: [Vp, C], stride: strideVpC),
                BNNSTensor(data: model.params.lnfw.bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.acts.lnf_mean.bnnsPinnedBuffer(), shape: [BT], stride: [1]),
                BNNSTensor(data: model.acts.lnf_rstd.bnnsPinnedBuffer(), shape: [BT], stride: [1]),
            ]

            try context.executeFunction(arguments: &arguments)
            // Outputs were written directly into model.grads.* and model.grads_acts.*; no copy-back needed.
        }
    }

    final class LayerBackwardGraph {
        let batchSize: Int
        let sequenceLength: Int
        let context: BNNSGraph.Context

        init(config: GPT2Config, batchSize: Int, sequenceLength: Int) throws {
            self.batchSize = batchSize
            self.sequenceLength = sequenceLength
            let C = config.channels
            let NH = config.num_heads
            let HS = config.headSize
            let fourC = 4 * C
            let C3 = 3 * C
            let BT = batchSize * sequenceLength
            let geluScale = LLMSwift.gelu_scaling_factor
            let attentionScale = Float(1) / Float(HS).squareRoot()

            context = try BNNSGraph.makeContext { builder in
                let one = builder.constant(name: "layer_backward_one", value: 1 as Float)
                let doutResidual3 = builder.argument(name: "dout_residual3", dataType: Float.self, shape: [BT, C])
                let fchGelu = builder.argument(name: "fch_gelu", dataType: Float.self, shape: [BT, fourC])
                let fch = builder.argument(name: "fch", dataType: Float.self, shape: [BT, fourC])
                let ln2 = builder.argument(name: "ln2", dataType: Float.self, shape: [BT, C])
                let residual2 = builder.argument(name: "residual2", dataType: Float.self, shape: [BT, C])
                let fcprojw = builder.argument(name: "fcprojw", dataType: Float.self, shape: [C, fourC])
                let fcw = builder.argument(name: "fcw", dataType: Float.self, shape: [fourC, C])
                let ln2w = builder.argument(name: "ln2w", dataType: Float.self, shape: [C])
                let ln2Mean = builder.argument(name: "ln2_mean", dataType: Float.self, shape: [BT])
                let ln2RSTD = builder.argument(name: "ln2_rstd", dataType: Float.self, shape: [BT])
                let atty = builder.argument(name: "atty", dataType: Float.self, shape: [BT, C])
                let attprojw = builder.argument(name: "attprojw", dataType: Float.self, shape: [C, C])
                let qkv = builder.argument(name: "qkv", dataType: Float.self, shape: [batchSize, sequenceLength, C3])
                let att = builder.argument(name: "att", dataType: Float.self, shape: [batchSize, NH, sequenceLength, sequenceLength])
                let ln1 = builder.argument(name: "ln1", dataType: Float.self, shape: [BT, C])
                let qkvw = builder.argument(name: "qkvw", dataType: Float.self, shape: [C3, C])
                let ln1Input = builder.argument(name: "ln1_input", dataType: Float.self, shape: [BT, C])
                let ln1w = builder.argument(name: "ln1w", dataType: Float.self, shape: [C])
                let ln1Mean = builder.argument(name: "ln1_mean", dataType: Float.self, shape: [BT])
                let ln1RSTD = builder.argument(name: "ln1_rstd", dataType: Float.self, shape: [BT])

                let dFcproj = doutResidual3
                let residualIdentity = doutResidual3
                let dFchGelu = dFcproj.matmul(other: fcprojw)
                let dFcprojw = dFcproj.matmul(transpose: true, other: fchGelu)
                let dFcprojb = dFcproj.sum(axes: [0], keepDimensions: false)

                let x2 = fch.pow(y: 2 as Float)
                let cube = fch * x2 * Float(0.044715)
                let tanhArgument = (fch + cube) * geluScale
                let tanhOutput = tanhArgument.tanh()
                let sechSquared = tanhOutput.pow(y: 2 as Float) * Float(-1) + one
                let localGradient = Float(0.5) * (one + tanhOutput) + fch * Float(0.5) * sechSquared * geluScale * (one + Float(3 * 0.044715) * x2)
                let dFch = dFchGelu * localGradient

                let dLn2 = dFch.matmul(other: fcw)
                let dFcw = dFch.matmul(transpose: true, other: ln2)
                let dFcb = dFch.sum(axes: [0], keepDimensions: false)

                let ln2MeanExpanded = ln2Mean.reshape(to: [BT, 1])
                let ln2RSTDExpanded = ln2RSTD.reshape(to: [BT, 1])
                let ln2Normalized = (residual2 - ln2MeanExpanded) * ln2RSTDExpanded
                let dNorm2 = dLn2 * ln2w
                let dNorm2Mean = dNorm2.mean(axes: [1], keepDimensions: true)
                let dNorm2NormMean = (dNorm2 * ln2Normalized).mean(axes: [1], keepDimensions: true)
                let dResidualNorm = (dNorm2 - dNorm2Mean - ln2Normalized * dNorm2NormMean) * ln2RSTDExpanded
                let dResidual2 = residualIdentity + dResidualNorm
                let dLn2w = (ln2Normalized * dLn2).sum(axes: [0], keepDimensions: false)
                let dLn2b = dLn2.sum(axes: [0], keepDimensions: false)

                let dAttproj = dResidual2
                let dResidualSkip = dAttproj
                let dAtty = dAttproj.matmul(other: attprojw)
                let dAttprojw = dAttproj.matmul(transpose: true, other: atty)
                let dAttprojb = dAttproj.sum(axes: [0], keepDimensions: false)

                let qkvTensor = qkv.reshape(to: [batchSize, sequenceLength, 3, NH, HS])
                let q = qkvTensor[0..<batchSize, 0..<sequenceLength, 0..<1, 0..<NH, 0..<HS].squeeze(axis: 2).transpose(axes: [0, 2, 1, 3])
                let k = qkvTensor[0..<batchSize, 0..<sequenceLength, 1..<2, 0..<NH, 0..<HS].squeeze(axis: 2).transpose(axes: [0, 2, 1, 3])
                let v = qkvTensor[0..<batchSize, 0..<sequenceLength, 2..<3, 0..<NH, 0..<HS].squeeze(axis: 2).transpose(axes: [0, 2, 1, 3])
                let dOutHeads = dAtty.reshape(to: [batchSize, sequenceLength, NH, HS]).transpose(axes: [0, 2, 1, 3])

                let dAtt = dOutHeads.matmul(other: v.transpose(axes: [0, 1, 3, 2]))
                let dV = att.transpose(axes: [0, 1, 3, 2]).matmul(other: dOutHeads)
                let attDot = (dAtt * att).sum(axes: [3], keepDimensions: true)
                let dPreatt = att * (dAtt - attDot)
                let dQ = dPreatt.matmul(other: k) * attentionScale
                let dK = dPreatt.transpose(axes: [0, 1, 3, 2]).matmul(other: q) * attentionScale

                let dQBT = dQ.transpose(axes: [0, 2, 1, 3]).reshape(to: [batchSize, sequenceLength, 1, NH, HS])
                let dKBT = dK.transpose(axes: [0, 2, 1, 3]).reshape(to: [batchSize, sequenceLength, 1, NH, HS])
                let dVBT = dV.transpose(axes: [0, 2, 1, 3]).reshape(to: [batchSize, sequenceLength, 1, NH, HS])
                let dQKV = builder.concatenate([dQBT, dKBT, dVBT], axis: 2).reshape(to: [batchSize, sequenceLength, C3])
                let dQKVFlat = dQKV.reshape(to: [BT, C3])

                let dLn1 = dQKVFlat.matmul(other: qkvw)
                let dQKVw = dQKVFlat.matmul(transpose: true, other: ln1)
                let dQKVb = dQKVFlat.sum(axes: [0], keepDimensions: false)

                let ln1MeanExpanded = ln1Mean.reshape(to: [BT, 1])
                let ln1RSTDExpanded = ln1RSTD.reshape(to: [BT, 1])
                let ln1Normalized = (ln1Input - ln1MeanExpanded) * ln1RSTDExpanded
                let dNorm1 = dLn1 * ln1w
                let dNorm1Mean = dNorm1.mean(axes: [1], keepDimensions: true)
                let dNorm1NormMean = (dNorm1 * ln1Normalized).mean(axes: [1], keepDimensions: true)
                let dLn1Input = (dNorm1 - dNorm1Mean - ln1Normalized * dNorm1NormMean) * ln1RSTDExpanded
                let dPreviousResidual = dResidualSkip + dLn1Input
                let dLn1w = (ln1Normalized * dLn1).sum(axes: [0], keepDimensions: false)
                let dLn1b = dLn1.sum(axes: [0], keepDimensions: false)

                return [
                    dPreviousResidual,
                    dResidual2,
                    dAtty,
                    dQKV,
                    dPreatt,
                    dAtt,
                    dLn1,
                    dFcprojw,
                    dFcprojb,
                    dFcw,
                    dFcb,
                    dLn2w,
                    dLn2b,
                    dAttprojw,
                    dAttprojb,
                    dQKVw,
                    dQKVb,
                    dLn1w,
                    dLn1b,
                ]
            }
        }

        func execute(layer: Int, model: inout GraphModel) throws {
            let C = model.config.channels
            let NH = model.config.num_heads
            let fourC = 4 * C
            let C3 = 3 * C
            let BT = batchSize * sequenceLength
            let strideBTC = bnnsRowMajorStride(for: [BT, C])
            let strideBT4C = bnnsRowMajorStride(for: [BT, fourC])
            let strideC4C = bnnsRowMajorStride(for: [C, fourC])
            let stride4CC = bnnsRowMajorStride(for: [fourC, C])
            let strideCC = bnnsRowMajorStride(for: [C, C])
            let strideBSC3 = bnnsRowMajorStride(for: [batchSize, sequenceLength, C3])
            let strideBNHTT = bnnsRowMajorStride(for: [batchSize, NH, sequenceLength, sequenceLength])
            let strideC3C = bnnsRowMajorStride(for: [C3, C])

            // dPreviousResidual writes to either acts.encoded (layer 0) or grads_acts.residual3[layer - 1] (otherwise).
            let dPreviousResidualBuffer: UnsafeMutableBufferPointer<Float>
            if layer == 0 {
                dPreviousResidualBuffer = model.grads_acts.encoded.bnnsPinnedBuffer()
            } else {
                dPreviousResidualBuffer = model.grads_acts.residual3[layer - 1].bnnsPinnedBuffer()
            }
            // ln1Input reads the same field on the activations side.
            let ln1InputBuffer: UnsafeMutableBufferPointer<Float>
            if layer == 0 {
                ln1InputBuffer = model.acts.encoded.bnnsPinnedBuffer()
            } else {
                ln1InputBuffer = model.acts.residual3[layer - 1].bnnsPinnedBuffer()
            }

            var arguments: [BNNSTensor] = [
                // Outputs (graph builder return order)
                BNNSTensor(data: dPreviousResidualBuffer, shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.grads_acts.residual2[layer].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.grads_acts.atty[layer].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.grads_acts.qkv[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C3], stride: strideBSC3),
                BNNSTensor(data: model.grads_acts.preatt[layer].bnnsPinnedBuffer(), shape: [batchSize, NH, sequenceLength, sequenceLength], stride: strideBNHTT),
                BNNSTensor(data: model.grads_acts.att[layer].bnnsPinnedBuffer(), shape: [batchSize, NH, sequenceLength, sequenceLength], stride: strideBNHTT),
                BNNSTensor(data: model.grads_acts.ln1[layer].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.grads.fcprojw[layer].bnnsPinnedBuffer(), shape: [C, fourC], stride: strideC4C),
                BNNSTensor(data: model.grads.fcprojb[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.grads.fcw[layer].bnnsPinnedBuffer(), shape: [fourC, C], stride: stride4CC),
                BNNSTensor(data: model.grads.fcb[layer].bnnsPinnedBuffer(), shape: [fourC], stride: [1]),
                BNNSTensor(data: model.grads.ln2w[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.grads.ln2b[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.grads.attprojw[layer].bnnsPinnedBuffer(), shape: [C, C], stride: strideCC),
                BNNSTensor(data: model.grads.attprojb[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.grads.qkvw[layer].bnnsPinnedBuffer(), shape: [C3, C], stride: strideC3C),
                BNNSTensor(data: model.grads.qkvb[layer].bnnsPinnedBuffer(), shape: [C3], stride: [1]),
                BNNSTensor(data: model.grads.ln1w[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.grads.ln1b[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]),
                // Inputs (graph builder argument order)
                BNNSTensor(data: model.grads_acts.residual3[layer].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.acts.fch_gelu[layer].bnnsPinnedBuffer(), shape: [BT, fourC], stride: strideBT4C),
                BNNSTensor(data: model.acts.fch[layer].bnnsPinnedBuffer(), shape: [BT, fourC], stride: strideBT4C),
                BNNSTensor(data: model.acts.ln2[layer].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.acts.residual2[layer].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.params.fcprojw[layer].bnnsPinnedBuffer(), shape: [C, fourC], stride: strideC4C),
                BNNSTensor(data: model.params.fcw[layer].bnnsPinnedBuffer(), shape: [fourC, C], stride: stride4CC),
                BNNSTensor(data: model.params.ln2w[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.acts.ln2_mean[layer].bnnsPinnedBuffer(), shape: [BT], stride: [1]),
                BNNSTensor(data: model.acts.ln2_rstd[layer].bnnsPinnedBuffer(), shape: [BT], stride: [1]),
                BNNSTensor(data: model.acts.atty[layer].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.params.attprojw[layer].bnnsPinnedBuffer(), shape: [C, C], stride: strideCC),
                BNNSTensor(data: model.acts.qkv[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C3], stride: strideBSC3),
                BNNSTensor(data: model.acts.att[layer].bnnsPinnedBuffer(), shape: [batchSize, NH, sequenceLength, sequenceLength], stride: strideBNHTT),
                BNNSTensor(data: model.acts.ln1[layer].bnnsPinnedBuffer(), shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.params.qkvw[layer].bnnsPinnedBuffer(), shape: [C3, C], stride: strideC3C),
                BNNSTensor(data: ln1InputBuffer, shape: [BT, C], stride: strideBTC),
                BNNSTensor(data: model.params.ln1w[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]),
                BNNSTensor(data: model.acts.ln1_mean[layer].bnnsPinnedBuffer(), shape: [BT], stride: [1]),
                BNNSTensor(data: model.acts.ln1_rstd[layer].bnnsPinnedBuffer(), shape: [BT], stride: [1]),
            ]

            try context.executeFunction(arguments: &arguments)
            // Outputs were written directly into model.grads.* and model.grads_acts.*; no copy-back needed.
        }
    }

    final class EncoderBackwardGraph {
        let batchSize: Int
        let sequenceLength: Int
        let context: BNNSGraph.Context
        private let wteShape: [Int]
        private let wpeShape: [Int]
        private let wteStride: [Int]
        private let wpeStride: [Int]
        // The graph reads `zero_wte` and `zero_wpe` as the starting accumulator, then
        // scatter-adds the per-batch contribution into the output. We feed always-zero
        // scratch buffers as the input and a scratch buffer as the output so the graph
        // doesn't have to allocate or copy. The result is then accumulated into the
        // model's gradient buffers via vDSP.
        private var zeroWteScratch: [Float]
        private var zeroWpeScratch: [Float]
        private var dWteScratch: [Float]
        private var dWpeScratch: [Float]

        init(config: GPT2Config, batchSize: Int, sequenceLength: Int) throws {
            self.batchSize = batchSize
            self.sequenceLength = sequenceLength
            let C = config.channels
            let Vp = config.padded_vocab_size
            let flatPositions = Array(repeating: Array(0..<sequenceLength).map(Int32.init), count: batchSize).flatMap { $0 }

            context = try BNNSGraph.makeContext { builder in
                let dout = builder.argument(name: "dout", dataType: Float.self, shape: [batchSize * sequenceLength, C])
                let tokenIndices = builder.argument(name: "token_indices", dataType: Int32.self, shape: [batchSize * sequenceLength])
                let zeroWte = builder.argument(name: "zero_wte", dataType: Float.self, shape: [Vp, C])
                let zeroWpe = builder.argument(name: "zero_wpe", dataType: Float.self, shape: [config.max_seq_len, C])
                let positionIndices = builder.constant(name: "encoder_positions", values: flatPositions, shape: [batchSize * sequenceLength])

                let dWte = zeroWte.scatter(updates: dout, indices: tokenIndices, mode: .add, axis: 0)
                let dWpe = zeroWpe.scatter(updates: dout, indices: positionIndices, mode: .add, axis: 0)
                return [dWte, dWpe]
            }

            self.wteShape = [Vp, C]
            self.wpeShape = [config.max_seq_len, C]
            self.wteStride = bnnsRowMajorStride(for: [Vp, C])
            self.wpeStride = bnnsRowMajorStride(for: [config.max_seq_len, C])
            self.zeroWteScratch = Array(repeating: 0, count: Vp * C)
            self.zeroWpeScratch = Array(repeating: 0, count: config.max_seq_len * C)
            self.dWteScratch = Array(repeating: 0, count: Vp * C)
            self.dWpeScratch = Array(repeating: 0, count: config.max_seq_len * C)
        }

        func execute(model: inout GraphModel) throws {
            let BT = batchSize * sequenceLength
            let C = model.config.channels
            let strideEncoded = bnnsRowMajorStride(for: [BT, C])

            var arguments: [BNNSTensor] = [
                // Outputs: dWte, dWpe (per-batch contributions)
                BNNSTensor(data: dWteScratch.bnnsPinnedBuffer(), shape: wteShape, stride: wteStride),
                BNNSTensor(data: dWpeScratch.bnnsPinnedBuffer(), shape: wpeShape, stride: wpeStride),
                // Inputs: dout, tokenIndices, zero_wte (always zero), zero_wpe (always zero)
                BNNSTensor(data: model.grads_acts.encoded.bnnsPinnedBuffer(), shape: [BT, C], stride: strideEncoded),
                BNNSTensor(data: model.inputsInt32.bnnsPinnedBuffer(), shape: [BT], stride: [1]),
                BNNSTensor(data: zeroWteScratch.bnnsPinnedBuffer(), shape: wteShape, stride: wteStride),
                BNNSTensor(data: zeroWpeScratch.bnnsPinnedBuffer(), shape: wpeShape, stride: wpeStride),
            ]
            try context.executeFunction(arguments: &arguments)

            // Accumulate the contributions into the model gradient buffers in place.
            vDSP.add(dWteScratch, model.grads.wte, result: &model.grads.wte)
            vDSP.add(dWpeScratch, model.grads.wpe, result: &model.grads.wpe)
        }
    }

    final class ForwardGraph {
        typealias Tensor = BNNSGraph.Builder.Tensor<Float>
        typealias BoolTensor = BNNSGraph.Builder.Tensor<Bool>

        let batchSize: Int
        let sequenceLength: Int
        let context: BNNSGraph.Context

        init(config: GPT2Config, batchSize: Int, sequenceLength: Int) throws {
            self.batchSize = batchSize
            self.sequenceLength = sequenceLength
            let C = config.channels
            let NH = config.num_heads
            let HS = config.headSize
            let Vp = config.padded_vocab_size
            let L = config.num_layers
            let positions = Self.positionIndices(batchSize: batchSize, sequenceLength: sequenceLength)
            let causalMask = Self.causalMask(sequenceLength: sequenceLength)
            let scale = Float(1) / Float(HS).squareRoot()

            context = try BNNSGraph.makeContext { builder in
                let inputIndices = builder.argument(name: "inputs", dataType: Int32.self, shape: [batchSize, sequenceLength])
                let wte = builder.argument(name: "wte", dataType: Float.self, shape: [Vp, C])
                let wpe = builder.argument(name: "wpe", dataType: Float.self, shape: [config.max_seq_len, C])
                let positionIndices = builder.constant(name: "positions", values: positions, shape: [batchSize, sequenceLength])
                let mask = builder.constant(name: "mask", values: causalMask, shape: [1, 1, sequenceLength, sequenceLength])
                let zero = builder.constant(name: "zero", value: 0 as Float)
                let validAttention = mask .== zero

                var residual = Self.encoder_forward(
                    inp: inputIndices,
                    wte: wte,
                    wpe: wpe,
                    positionIndices: positionIndices
                )
                var outputs: [any BNNSGraph.TensorDescriptor] = []
                outputs.append(residual)

                for layer in 0..<L {
                    let ln1w = builder.argument(name: "ln1w_\(layer)", dataType: Float.self, shape: [C])
                    let ln1b = builder.argument(name: "ln1b_\(layer)", dataType: Float.self, shape: [C])
                    let qkvw = builder.argument(name: "qkvw_\(layer)", dataType: Float.self, shape: [3 * C, C])
                    let qkvb = builder.argument(name: "qkvb_\(layer)", dataType: Float.self, shape: [3 * C])
                    let attprojw = builder.argument(name: "attprojw_\(layer)", dataType: Float.self, shape: [C, C])
                    let attprojb = builder.argument(name: "attprojb_\(layer)", dataType: Float.self, shape: [C])
                    let ln2w = builder.argument(name: "ln2w_\(layer)", dataType: Float.self, shape: [C])
                    let ln2b = builder.argument(name: "ln2b_\(layer)", dataType: Float.self, shape: [C])
                    let fcw = builder.argument(name: "fcw_\(layer)", dataType: Float.self, shape: [4 * C, C])
                    let fcb = builder.argument(name: "fcb_\(layer)", dataType: Float.self, shape: [4 * C])
                    let fcprojw = builder.argument(name: "fcprojw_\(layer)", dataType: Float.self, shape: [C, 4 * C])
                    let fcprojb = builder.argument(name: "fcprojb_\(layer)", dataType: Float.self, shape: [C])

                    let ln1 = Self.layernorm_forward(inp: residual, weight: ln1w, bias: ln1b, B: batchSize, T: sequenceLength)
                    let qkv = Self.matmul_forward(inp: ln1.out, weight: qkvw, bias: qkvb, B: batchSize, T: sequenceLength, C: C, OC: 3 * C)
                        .reshape(to: [batchSize, sequenceLength, 3, NH, HS])
                    let attention = Self.attention_forward(inp: qkv, mask: mask, validAttention: validAttention, B: batchSize, T: sequenceLength, C: C, NH: NH, HS: HS, scale: scale)
                    let attproj = Self.matmul_forward(inp: attention.out, weight: attprojw, bias: attprojb, B: batchSize, T: sequenceLength, C: C, OC: C)
                    let residual2 = Self.residual_forward(inp1: residual, inp2: attproj)
                    let ln2 = Self.layernorm_forward(inp: residual2, weight: ln2w, bias: ln2b, B: batchSize, T: sequenceLength)
                    let fch = Self.matmul_forward(inp: ln2.out, weight: fcw, bias: fcb, B: batchSize, T: sequenceLength, C: C, OC: 4 * C)
                    let fchGelu = Self.gelu_forward(inp: fch)
                    let fcproj = Self.matmul_forward(inp: fchGelu, weight: fcprojw, bias: fcprojb, B: batchSize, T: sequenceLength, C: 4 * C, OC: C)
                    residual = Self.residual_forward(inp1: residual2, inp2: fcproj)

                    outputs.append(ln1.out)
                    outputs.append(ln1.mean)
                    outputs.append(ln1.rstd)
                    outputs.append(qkv.reshape(to: [batchSize, sequenceLength, 3 * C]))
                    outputs.append(attention.out.reshape(to: [batchSize, sequenceLength, C]))
                    outputs.append(attention.preatt)
                    outputs.append(attention.att)
                    outputs.append(attproj)
                    outputs.append(residual2)
                    outputs.append(ln2.out)
                    outputs.append(ln2.mean)
                    outputs.append(ln2.rstd)
                    outputs.append(fch)
                    outputs.append(fchGelu)
                    outputs.append(fcproj)
                    outputs.append(residual)
                }

                let lnfw = builder.argument(name: "lnfw", dataType: Float.self, shape: [C])
                let lnfb = builder.argument(name: "lnfb", dataType: Float.self, shape: [C])
                let lnf = Self.layernorm_forward(inp: residual, weight: lnfw, bias: lnfb, B: batchSize, T: sequenceLength)
                let logits = Self.matmul_forward(inp: lnf.out, weight: wte, B: batchSize, T: sequenceLength, C: C, OC: Vp)
                let logitsOutput = logits + zero
                let vocabMask = builder.constant(name: "vocab_mask", values: Self.vocabularyMask(vocabSize: config.vocab_size, paddedVocabSize: Vp), shape: [1, 1, Vp])
                let probs = (logits + vocabMask).softmax(axis: 2)
                outputs.append(lnf.out)
                outputs.append(lnf.mean)
                outputs.append(lnf.rstd)
                outputs.append(logitsOutput)
                outputs.append(probs)
                return outputs
            }
        }

        func execute(model: inout GraphModel) throws {
            let C = model.config.channels
            let NH = model.config.num_heads
            let Vp = model.config.padded_vocab_size
            let outputCount = 1 + 16 * model.config.num_layers + 5
            let strideBSC = bnnsRowMajorStride(for: [batchSize, sequenceLength, C])
            let strideBS = bnnsRowMajorStride(for: [batchSize, sequenceLength])
            let strideBS3C = bnnsRowMajorStride(for: [batchSize, sequenceLength, 3 * C])
            let strideBS4C = bnnsRowMajorStride(for: [batchSize, sequenceLength, 4 * C])
            let strideBSVp = bnnsRowMajorStride(for: [batchSize, sequenceLength, Vp])
            let strideBNHTT = bnnsRowMajorStride(for: [batchSize, NH, sequenceLength, sequenceLength])
            let strideInputs = bnnsRowMajorStride(for: [batchSize, sequenceLength])
            let strideVpC = bnnsRowMajorStride(for: [Vp, C])
            let strideMaxSeqC = bnnsRowMajorStride(for: [model.config.max_seq_len, C])
            let stride3CC = bnnsRowMajorStride(for: [3 * C, C])
            let strideCC = bnnsRowMajorStride(for: [C, C])
            let stride4CC = bnnsRowMajorStride(for: [4 * C, C])
            let strideC4C = bnnsRowMajorStride(for: [C, 4 * C])

            var arguments: [BNNSTensor] = []
            arguments.reserveCapacity(outputCount + 5 + 12 * model.config.num_layers)

            // Outputs (write directly into model.acts buffers)
            arguments.append(BNNSTensor(data: model.acts.encoded.bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))

            for layer in 0..<model.config.num_layers {
                arguments.append(BNNSTensor(data: model.acts.ln1[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))
                arguments.append(BNNSTensor(data: model.acts.ln1_mean[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength], stride: strideBS))
                arguments.append(BNNSTensor(data: model.acts.ln1_rstd[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength], stride: strideBS))
                arguments.append(BNNSTensor(data: model.acts.qkv[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, 3 * C], stride: strideBS3C))
                arguments.append(BNNSTensor(data: model.acts.atty[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))
                arguments.append(BNNSTensor(data: model.acts.preatt[layer].bnnsPinnedBuffer(), shape: [batchSize, NH, sequenceLength, sequenceLength], stride: strideBNHTT))
                arguments.append(BNNSTensor(data: model.acts.att[layer].bnnsPinnedBuffer(), shape: [batchSize, NH, sequenceLength, sequenceLength], stride: strideBNHTT))
                arguments.append(BNNSTensor(data: model.acts.attproj[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))
                arguments.append(BNNSTensor(data: model.acts.residual2[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))
                arguments.append(BNNSTensor(data: model.acts.ln2[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))
                arguments.append(BNNSTensor(data: model.acts.ln2_mean[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength], stride: strideBS))
                arguments.append(BNNSTensor(data: model.acts.ln2_rstd[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength], stride: strideBS))
                arguments.append(BNNSTensor(data: model.acts.fch[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, 4 * C], stride: strideBS4C))
                arguments.append(BNNSTensor(data: model.acts.fch_gelu[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, 4 * C], stride: strideBS4C))
                arguments.append(BNNSTensor(data: model.acts.fcproj[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))
                arguments.append(BNNSTensor(data: model.acts.residual3[layer].bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))
            }

            arguments.append(BNNSTensor(data: model.acts.lnf.bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, C], stride: strideBSC))
            arguments.append(BNNSTensor(data: model.acts.lnf_mean.bnnsPinnedBuffer(), shape: [batchSize, sequenceLength], stride: strideBS))
            arguments.append(BNNSTensor(data: model.acts.lnf_rstd.bnnsPinnedBuffer(), shape: [batchSize, sequenceLength], stride: strideBS))
            arguments.append(BNNSTensor(data: model.acts.logits.bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, Vp], stride: strideBSVp))
            arguments.append(BNNSTensor(data: model.acts.probs.bnnsPinnedBuffer(), shape: [batchSize, sequenceLength, Vp], stride: strideBSVp))

            // Inputs (graph builder argument order)
            arguments.append(BNNSTensor(data: model.inputsInt32.bnnsPinnedBuffer(), shape: [batchSize, sequenceLength], stride: strideInputs))
            arguments.append(BNNSTensor(data: model.params.wte.bnnsPinnedBuffer(), shape: [Vp, C], stride: strideVpC))
            arguments.append(BNNSTensor(data: model.params.wpe.bnnsPinnedBuffer(), shape: [model.config.max_seq_len, C], stride: strideMaxSeqC))

            for layer in 0..<model.config.num_layers {
                arguments.append(BNNSTensor(data: model.params.ln1w[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.ln1b[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.qkvw[layer].bnnsPinnedBuffer(), shape: [3 * C, C], stride: stride3CC))
                arguments.append(BNNSTensor(data: model.params.qkvb[layer].bnnsPinnedBuffer(), shape: [3 * C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.attprojw[layer].bnnsPinnedBuffer(), shape: [C, C], stride: strideCC))
                arguments.append(BNNSTensor(data: model.params.attprojb[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.ln2w[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.ln2b[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.fcw[layer].bnnsPinnedBuffer(), shape: [4 * C, C], stride: stride4CC))
                arguments.append(BNNSTensor(data: model.params.fcb[layer].bnnsPinnedBuffer(), shape: [4 * C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.fcprojw[layer].bnnsPinnedBuffer(), shape: [C, 4 * C], stride: strideC4C))
                arguments.append(BNNSTensor(data: model.params.fcprojb[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
            }

            arguments.append(BNNSTensor(data: model.params.lnfw.bnnsPinnedBuffer(), shape: [C], stride: [1]))
            arguments.append(BNNSTensor(data: model.params.lnfb.bnnsPinnedBuffer(), shape: [C], stride: [1]))

            try context.executeFunction(arguments: &arguments)
            // Outputs were written directly into model.acts; no copy-back needed.
        }

        struct LayerNormForwardResult {
            let out: Tensor
            let mean: Tensor
            let rstd: Tensor
        }

        struct AttentionForwardResult {
            let out: Tensor
            let preatt: Tensor
            let att: Tensor
        }

        static func encoder_forward(inp: BNNSGraph.Builder.Tensor<Int32>, wte: Tensor, wpe: Tensor, positionIndices: BNNSGraph.Builder.Tensor<Int32>) -> Tensor {
            let tokenEmbedding = wte.gather(indices: inp, axis: 0, batchDimensionCount: 0)
            let positionEmbedding = wpe.gather(indices: positionIndices, axis: 0, batchDimensionCount: 0)
            return tokenEmbedding + positionEmbedding
        }

        static func layernorm_forward(inp: Tensor, weight: Tensor, bias: Tensor, B: Int, T: Int) -> LayerNormForwardResult {
            let mean = inp.mean(axes: [2], keepDimensions: false)
            let centered = inp - mean.reshape(to: [B, T, 1])
            let variance = centered.pow(y: 2 as Float).mean(axes: [2], keepDimensions: false)
            let rstd = (variance + Float(1e-5)).rsqrt()
            let out = inp.layerNorm(weight: weight, bias: bias, axes: [2], epsilon: 1e-5)
            return LayerNormForwardResult(out: out, mean: mean, rstd: rstd)
        }

        static func matmul_forward(inp: Tensor, weight: Tensor, bias: Tensor, B: Int, T: Int, C: Int, OC: Int) -> Tensor {
            inp.reshape(to: [B * T, C])
                .linear(weight: weight, bias: bias)
                .reshape(to: [B, T, OC])
        }

        static func matmul_forward(inp: Tensor, weight: Tensor, B: Int, T: Int, C: Int, OC: Int) -> Tensor {
            inp.reshape(to: [B * T, C])
                .linear(weight: weight)
                .reshape(to: [B, T, OC])
        }

        static func attention_forward(inp: Tensor, mask: Tensor, validAttention: BoolTensor, B: Int, T: Int, C: Int, NH: Int, HS: Int, scale: Float) -> AttentionForwardResult {
            let q = inp[0..<B, 0..<T, 0..<1, 0..<NH, 0..<HS].squeeze(axis: 2).transpose(axes: [0, 2, 1, 3])
            let k = inp[0..<B, 0..<T, 1..<2, 0..<NH, 0..<HS].squeeze(axis: 2).transpose(axes: [0, 2, 3, 1])
            let v = inp[0..<B, 0..<T, 2..<3, 0..<NH, 0..<HS].squeeze(axis: 2).transpose(axes: [0, 2, 1, 3])
            let preatt = q.matmul(other: k) * scale
            let savedPreatt = validAttention.select(preatt, 0 as Float)
            let att = (preatt + mask).softmax(axis: 3)
            let out = att.matmul(other: v)
                .transpose(axes: [0, 2, 1, 3])
                .reshape(to: [B * T, C])
            return AttentionForwardResult(out: out, preatt: savedPreatt, att: att)
        }

        static func gelu_forward(inp: Tensor) -> Tensor {
            inp.geluTanhApproximation()
        }

        static func residual_forward(inp1: Tensor, inp2: Tensor) -> Tensor {
            inp1 + inp2
        }

        static func positionIndices(batchSize: Int, sequenceLength: Int) -> [Int32] {
            Array(repeating: Array(0..<sequenceLength).map(Int32.init), count: batchSize).flatMap { $0 }
        }

        static func causalMask(sequenceLength: Int) -> [Float] {
            var mask = Array(repeating: Float.zero, count: sequenceLength * sequenceLength)
            for queryIndex in 0..<sequenceLength {
                for keyIndex in (queryIndex + 1)..<sequenceLength {
                    mask[queryIndex * sequenceLength + keyIndex] = -1e10
                }
            }
            return mask
        }

        private static func vocabularyMask(vocabSize V: Int, paddedVocabSize Vp: Int) -> [Float] {
            var mask = Array(repeating: Float.zero, count: Vp)
            for i in V..<Vp {
                mask[i] = -1e10
            }
            return mask
        }
    }

    final class LatestTokenInferenceGraph {
        typealias Tensor = BNNSGraph.Builder.Tensor<Float>
        typealias BoolTensor = BNNSGraph.Builder.Tensor<Bool>

        let batchSize: Int
        let sequenceLength: Int
        let context: BNNSGraph.Context

        init(config: GPT2Config, batchSize: Int, sequenceLength: Int) throws {
            self.batchSize = batchSize
            self.sequenceLength = sequenceLength
            let C = config.channels
            let NH = config.num_heads
            let HS = config.headSize
            let Vp = config.padded_vocab_size
            let L = config.num_layers
            let positions = ForwardGraph.positionIndices(batchSize: batchSize, sequenceLength: sequenceLength)
            let causalMask = ForwardGraph.causalMask(sequenceLength: sequenceLength)
            let scale = Float(1) / Float(HS).squareRoot()

            context = try BNNSGraph.makeContext { builder in
                let inputIndices = builder.argument(name: "inputs", dataType: Int32.self, shape: [batchSize, sequenceLength])
                let rowIndex = builder.argument(name: "rowIndex", dataType: Int32.self, shape: [1])
                let wte = builder.argument(name: "wte", dataType: Float.self, shape: [Vp, C])
                let wpe = builder.argument(name: "wpe", dataType: Float.self, shape: [config.max_seq_len, C])
                let positionIndices = builder.constant(name: "positions", values: positions, shape: [batchSize, sequenceLength])
                let mask = builder.constant(name: "mask", values: causalMask, shape: [1, 1, sequenceLength, sequenceLength])
                let zero = builder.constant(name: "zero", value: 0 as Float)
                let validAttention = mask .== zero

                var residual = ForwardGraph.encoder_forward(
                    inp: inputIndices,
                    wte: wte,
                    wpe: wpe,
                    positionIndices: positionIndices
                )

                for layer in 0..<L {
                    let ln1w = builder.argument(name: "ln1w_\(layer)", dataType: Float.self, shape: [C])
                    let ln1b = builder.argument(name: "ln1b_\(layer)", dataType: Float.self, shape: [C])
                    let qkvw = builder.argument(name: "qkvw_\(layer)", dataType: Float.self, shape: [3 * C, C])
                    let qkvb = builder.argument(name: "qkvb_\(layer)", dataType: Float.self, shape: [3 * C])
                    let attprojw = builder.argument(name: "attprojw_\(layer)", dataType: Float.self, shape: [C, C])
                    let attprojb = builder.argument(name: "attprojb_\(layer)", dataType: Float.self, shape: [C])
                    let ln2w = builder.argument(name: "ln2w_\(layer)", dataType: Float.self, shape: [C])
                    let ln2b = builder.argument(name: "ln2b_\(layer)", dataType: Float.self, shape: [C])
                    let fcw = builder.argument(name: "fcw_\(layer)", dataType: Float.self, shape: [4 * C, C])
                    let fcb = builder.argument(name: "fcb_\(layer)", dataType: Float.self, shape: [4 * C])
                    let fcprojw = builder.argument(name: "fcprojw_\(layer)", dataType: Float.self, shape: [C, 4 * C])
                    let fcprojb = builder.argument(name: "fcprojb_\(layer)", dataType: Float.self, shape: [C])

                    let ln1 = ForwardGraph.layernorm_forward(inp: residual, weight: ln1w, bias: ln1b, B: batchSize, T: sequenceLength)
                    let qkv = ForwardGraph.matmul_forward(inp: ln1.out, weight: qkvw, bias: qkvb, B: batchSize, T: sequenceLength, C: C, OC: 3 * C)
                        .reshape(to: [batchSize, sequenceLength, 3, NH, HS])
                    let attention = ForwardGraph.attention_forward(inp: qkv, mask: mask, validAttention: validAttention, B: batchSize, T: sequenceLength, C: C, NH: NH, HS: HS, scale: scale)
                    let attproj = ForwardGraph.matmul_forward(inp: attention.out, weight: attprojw, bias: attprojb, B: batchSize, T: sequenceLength, C: C, OC: C)
                    let residual2 = ForwardGraph.residual_forward(inp1: residual, inp2: attproj)
                    let ln2 = ForwardGraph.layernorm_forward(inp: residual2, weight: ln2w, bias: ln2b, B: batchSize, T: sequenceLength)
                    let fch = ForwardGraph.matmul_forward(inp: ln2.out, weight: fcw, bias: fcb, B: batchSize, T: sequenceLength, C: C, OC: 4 * C)
                    let fchGelu = ForwardGraph.gelu_forward(inp: fch)
                    let fcproj = ForwardGraph.matmul_forward(inp: fchGelu, weight: fcprojw, bias: fcprojb, B: batchSize, T: sequenceLength, C: 4 * C, OC: C)
                    residual = ForwardGraph.residual_forward(inp1: residual2, inp2: fcproj)
                }

                let lnfw = builder.argument(name: "lnfw", dataType: Float.self, shape: [C])
                let lnfb = builder.argument(name: "lnfb", dataType: Float.self, shape: [C])
                let lnf = ForwardGraph.layernorm_forward(inp: residual, weight: lnfw, bias: lnfb, B: batchSize, T: sequenceLength)
                let latestHidden = lnf.out.gather(indices: rowIndex, axis: 1, batchDimensionCount: 0)
                let latestLogits = ForwardGraph.matmul_forward(inp: latestHidden, weight: wte, B: batchSize, T: 1, C: C, OC: Vp)
                return [latestLogits + zero]
            }
        }

        func execute(model: inout GraphModel, rowIndex: Int) throws {
            let C = model.config.channels
            let NH = model.config.num_heads
            let Vp = model.config.padded_vocab_size
            let strideInputs = bnnsRowMajorStride(for: [batchSize, sequenceLength])
            let strideVpC = bnnsRowMajorStride(for: [Vp, C])
            let strideMaxSeqC = bnnsRowMajorStride(for: [model.config.max_seq_len, C])
            let stride3CC = bnnsRowMajorStride(for: [3 * C, C])
            let strideCC = bnnsRowMajorStride(for: [C, C])
            let stride4CC = bnnsRowMajorStride(for: [4 * C, C])
            let strideC4C = bnnsRowMajorStride(for: [C, 4 * C])
            let strideLogits = bnnsRowMajorStride(for: [batchSize, 1, Vp])
            model.latestInferenceRowIndex[0] = Int32(rowIndex)

            var arguments: [BNNSTensor] = []
            arguments.reserveCapacity(3 + 12 * model.config.num_layers)
            arguments.append(BNNSTensor(data: model.latestTokenLogits.bnnsPinnedBuffer(), shape: [batchSize, 1, Vp], stride: strideLogits))
            arguments.append(BNNSTensor(data: model.inputsInt32.bnnsPinnedBuffer(), shape: [batchSize, sequenceLength], stride: strideInputs))
            arguments.append(BNNSTensor(data: model.latestInferenceRowIndex.bnnsPinnedBuffer(), shape: [1], stride: [1]))
            arguments.append(BNNSTensor(data: model.params.wte.bnnsPinnedBuffer(), shape: [Vp, C], stride: strideVpC))
            arguments.append(BNNSTensor(data: model.params.wpe.bnnsPinnedBuffer(), shape: [model.config.max_seq_len, C], stride: strideMaxSeqC))

            for layer in 0..<model.config.num_layers {
                arguments.append(BNNSTensor(data: model.params.ln1w[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.ln1b[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.qkvw[layer].bnnsPinnedBuffer(), shape: [3 * C, C], stride: stride3CC))
                arguments.append(BNNSTensor(data: model.params.qkvb[layer].bnnsPinnedBuffer(), shape: [3 * C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.attprojw[layer].bnnsPinnedBuffer(), shape: [C, C], stride: strideCC))
                arguments.append(BNNSTensor(data: model.params.attprojb[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.ln2w[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.ln2b[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.fcw[layer].bnnsPinnedBuffer(), shape: [4 * C, C], stride: stride4CC))
                arguments.append(BNNSTensor(data: model.params.fcb[layer].bnnsPinnedBuffer(), shape: [4 * C], stride: [1]))
                arguments.append(BNNSTensor(data: model.params.fcprojw[layer].bnnsPinnedBuffer(), shape: [C, 4 * C], stride: strideC4C))
                arguments.append(BNNSTensor(data: model.params.fcprojb[layer].bnnsPinnedBuffer(), shape: [C], stride: [1]))
            }

            arguments.append(BNNSTensor(data: model.params.lnfw.bnnsPinnedBuffer(), shape: [C], stride: [1]))
            arguments.append(BNNSTensor(data: model.params.lnfb.bnnsPinnedBuffer(), shape: [C], stride: [1]))

            try context.executeFunction(arguments: &arguments)
        }
    }

    struct GraphModel {
        let config: GPT2Config
        var params: ParameterTensors
        var grads: ParameterTensors
        var m_memory: ParameterTensors
        var v_memory: ParameterTensors
        var acts = ActivationTensors()
        var grads_acts = ActivationTensors()
        var forwardGraphs: [GraphKey: ForwardGraph] = [:]
        var latestTokenInferenceGraphs: [GraphKey: LatestTokenInferenceGraph] = [:]
        var finalBackwardGraphs: [GraphKey: FinalBackwardGraph] = [:]
        var layerBackwardGraphs: [GraphKey: LayerBackwardGraph] = [:]
        var encoderBackwardGraphs: [GraphKey: EncoderBackwardGraph] = [:]
        var batch_size = 0
        var seq_len = 0
        var inputs: [UInt32] = []
        var targets: [UInt32] = []
        var inputsInt32: [Int32] = []
        var latestInferenceRowIndex = [Int32](repeating: 0, count: 1)
        var latestTokenLogits: [Float] = []
        var mean_loss: Float = -1

        mutating func ensureInputsInt32(count: Int) {
            if inputsInt32.count != count {
                inputsInt32 = Array(repeating: 0, count: count)
            }
            for i in 0..<count {
                inputsInt32[i] = Int32(bitPattern: inputs[i])
            }
        }

        mutating func ensureLatestTokenLogits(count: Int) {
            if latestTokenLogits.count != count {
                latestTokenLogits = Array(repeating: 0, count: count)
            }
        }

        init(config: GPT2Config, float_params: [Float]) {
            self.config = config
            self.params = ParameterTensors(config: config, float_params: float_params)
            self.grads = ParameterTensors(config: config, float_params: Array(repeating: 0, count: config.num_parameters))
            self.m_memory = ParameterTensors(config: config, float_params: Array(repeating: 0, count: config.num_parameters))
            self.v_memory = ParameterTensors(config: config, float_params: Array(repeating: 0, count: config.num_parameters))
        }

        func exportCheckpoint() throws -> Data {
            try LLMCheckpointCodec.encode(
                header: config.checkpointHeader,
                parameters: params.flattened(),
                expectedParameterCount: config.num_parameters
            )
        }
    }

    static func buildModel(from checkpointData: Data) throws -> GraphModel {
        let (header, parameters) = try LLMCheckpointCodec.decode(checkpointData)
        let config = GPT2Config(header: header)
        guard parameters.count == config.num_parameters else {
            throw LLMCheckpointCodecError.invalidParameterCount(expected: config.num_parameters, actual: parameters.count)
        }
        return GraphModel(config: config, float_params: parameters)
    }

    static func gpt2_forward(model: inout GraphModel, inputs: [UInt32], targets: [UInt32], B: Int, T: Int) throws {
        let V = model.config.vocab_size
        let Vp = model.config.padded_vocab_size
        let key = GraphKey(batchSize: B, sequenceLength: T)

        model.batch_size = B
        model.seq_len = T
        _ = model.acts.resizeIfNeeded(config: model.config, B: B, T: T)
        precondition(inputs.count == B * T)
        precondition(inputs.allSatisfy { Int($0) >= 0 && Int($0) < V })
        precondition(targets.isEmpty || (targets.count == B * T && targets.allSatisfy { Int($0) >= 0 && Int($0) < V }))
        model.inputs = inputs
        model.targets = targets
        model.ensureInputsInt32(count: B * T)

        let graph = try cachedForwardGraph(for: key, model: &model)
        try graph.execute(model: &model)

        if !targets.isEmpty {
            crossentropy_forward(losses: &model.acts.losses, probs: model.acts.probs, targets: targets, B: B, T: T, Vp: Vp)
            var meanLoss: Float = 0
            for loss in model.acts.losses {
                meanLoss += loss
            }
            model.mean_loss = meanLoss / Float(B * T)
        } else {
            model.mean_loss = -1
        }
    }

    static func gpt2_latest_token_logits(model: inout GraphModel, inputs: [UInt32], rowIndex: Int, B: Int, T: Int) throws -> ArraySlice<Float> {
        let V = model.config.vocab_size
        let Vp = model.config.padded_vocab_size
        let key = GraphKey(batchSize: B, sequenceLength: T)

        model.batch_size = B
        model.seq_len = T
        precondition(inputs.count == B * T)
        precondition(inputs.allSatisfy { Int($0) >= 0 && Int($0) < V })
        precondition(rowIndex >= 0 && rowIndex < T)
        model.inputs = inputs
        model.targets = []
        model.ensureInputsInt32(count: B * T)
        model.ensureLatestTokenLogits(count: B * Vp)

        let graph = try cachedLatestTokenInferenceGraph(for: key, model: &model)
        try graph.execute(model: &model, rowIndex: rowIndex)
        model.mean_loss = -1
        return model.latestTokenLogits.prefix(V)
    }

    static func sample(logits: ArraySlice<Float>, temperature: Double, state: inout UInt64) -> Int {
        LLMSwift.sample(logits: logits, temperature: temperature, state: &state)
    }

    static func gpt2_zero_grad(model: inout GraphModel) {
        model.grads.zero()
        model.grads_acts.zero()
    }

    static func gpt2_backward(model: inout GraphModel) {
        let B = model.batch_size
        let T = model.seq_len
        let V = model.config.vocab_size
        let Vp = model.config.padded_vocab_size
        let L = model.config.num_layers
        _ = model.grads_acts.resizeIfNeeded(config: model.config, B: B, T: T)

        let lossCount = max(1, B * T)
        let dlossMean: Float = 1 / Float(lossCount)
        for index in 0..<lossCount {
            model.grads_acts.losses[index] = dlossMean
        }

        LLMBLAS.crossentropy_softmax_backward(dlogits: &model.grads_acts.logits, dlosses: model.grads_acts.losses, probs: model.acts.probs, targets: model.targets, B: B, T: T, V: V, Vp: Vp)

        let key = GraphKey(batchSize: B, sequenceLength: T)
        let finalBackwardGraph = try! cachedFinalBackwardGraph(for: key, model: &model)
        try! finalBackwardGraph.execute(model: &model)

        for layer in stride(from: L - 1, through: 0, by: -1) {
            try! executeUnifiedLayerBackward(layer: layer, key: key, model: &model)
        }

        // ensureInputsInt32 was already called in gpt2_forward; encoder backward reuses it.
        let encoderBackwardGraph = try! cachedEncoderBackwardGraph(for: key, model: &model)
        try! encoderBackwardGraph.execute(model: &model)
    }

    static func gpt2_update(model: inout GraphModel, update_params: UpdateParams) {
        LLMSwift.gpt2_update_field(params: &model.params.wte, grads: model.grads.wte, m_memory: &model.m_memory.wte, v_memory: &model.v_memory.wte, update_params: update_params)
        LLMSwift.gpt2_update_field(params: &model.params.wpe, grads: model.grads.wpe, m_memory: &model.m_memory.wpe, v_memory: &model.v_memory.wpe, update_params: update_params)
        for layer in 0..<model.config.num_layers {
            LLMSwift.gpt2_update_field(params: &model.params.ln1w[layer], grads: model.grads.ln1w[layer], m_memory: &model.m_memory.ln1w[layer], v_memory: &model.v_memory.ln1w[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.ln1b[layer], grads: model.grads.ln1b[layer], m_memory: &model.m_memory.ln1b[layer], v_memory: &model.v_memory.ln1b[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.qkvw[layer], grads: model.grads.qkvw[layer], m_memory: &model.m_memory.qkvw[layer], v_memory: &model.v_memory.qkvw[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.qkvb[layer], grads: model.grads.qkvb[layer], m_memory: &model.m_memory.qkvb[layer], v_memory: &model.v_memory.qkvb[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.attprojw[layer], grads: model.grads.attprojw[layer], m_memory: &model.m_memory.attprojw[layer], v_memory: &model.v_memory.attprojw[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.attprojb[layer], grads: model.grads.attprojb[layer], m_memory: &model.m_memory.attprojb[layer], v_memory: &model.v_memory.attprojb[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.ln2w[layer], grads: model.grads.ln2w[layer], m_memory: &model.m_memory.ln2w[layer], v_memory: &model.v_memory.ln2w[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.ln2b[layer], grads: model.grads.ln2b[layer], m_memory: &model.m_memory.ln2b[layer], v_memory: &model.v_memory.ln2b[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.fcw[layer], grads: model.grads.fcw[layer], m_memory: &model.m_memory.fcw[layer], v_memory: &model.v_memory.fcw[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.fcb[layer], grads: model.grads.fcb[layer], m_memory: &model.m_memory.fcb[layer], v_memory: &model.v_memory.fcb[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.fcprojw[layer], grads: model.grads.fcprojw[layer], m_memory: &model.m_memory.fcprojw[layer], v_memory: &model.v_memory.fcprojw[layer], update_params: update_params)
            LLMSwift.gpt2_update_field(params: &model.params.fcprojb[layer], grads: model.grads.fcprojb[layer], m_memory: &model.m_memory.fcprojb[layer], v_memory: &model.v_memory.fcprojb[layer], update_params: update_params)
        }
        LLMSwift.gpt2_update_field(params: &model.params.lnfw, grads: model.grads.lnfw, m_memory: &model.m_memory.lnfw, v_memory: &model.v_memory.lnfw, update_params: update_params)
        LLMSwift.gpt2_update_field(params: &model.params.lnfb, grads: model.grads.lnfb, m_memory: &model.m_memory.lnfb, v_memory: &model.v_memory.lnfb, update_params: update_params)
    }

    private static func cachedForwardGraph(for key: GraphKey, model: inout GraphModel) throws -> ForwardGraph {
        if let graph = model.forwardGraphs[key] {
            return graph
        }
        let graph = try ForwardGraph(config: model.config, batchSize: key.batchSize, sequenceLength: key.sequenceLength)
        model.forwardGraphs[key] = graph
        return graph
    }

    private static func cachedLatestTokenInferenceGraph(for key: GraphKey, model: inout GraphModel) throws -> LatestTokenInferenceGraph {
        if let graph = model.latestTokenInferenceGraphs[key] {
            return graph
        }
        let graph = try LatestTokenInferenceGraph(config: model.config, batchSize: key.batchSize, sequenceLength: key.sequenceLength)
        model.latestTokenInferenceGraphs[key] = graph
        return graph
    }

    private static func cachedFinalBackwardGraph(for key: GraphKey, model: inout GraphModel) throws -> FinalBackwardGraph {
        if let graph = model.finalBackwardGraphs[key] {
            return graph
        }
        let graph = try FinalBackwardGraph(config: model.config, batchSize: key.batchSize, sequenceLength: key.sequenceLength)
        model.finalBackwardGraphs[key] = graph
        return graph
    }

    private static func cachedLayerBackwardGraph(for key: GraphKey, model: inout GraphModel) throws -> LayerBackwardGraph {
        if let graph = model.layerBackwardGraphs[key] {
            return graph
        }
        let graph = try LayerBackwardGraph(config: model.config, batchSize: key.batchSize, sequenceLength: key.sequenceLength)
        model.layerBackwardGraphs[key] = graph
        return graph
    }

    private static func cachedEncoderBackwardGraph(for key: GraphKey, model: inout GraphModel) throws -> EncoderBackwardGraph {
        if let graph = model.encoderBackwardGraphs[key] {
            return graph
        }
        let graph = try EncoderBackwardGraph(config: model.config, batchSize: key.batchSize, sequenceLength: key.sequenceLength)
        model.encoderBackwardGraphs[key] = graph
        return graph
    }

    private static func executeUnifiedLayerBackward(layer: Int, key: GraphKey, model: inout GraphModel) throws {
        let graph = try cachedLayerBackwardGraph(for: key, model: &model)
        try graph.execute(layer: layer, model: &model)

        // Preserved from the prior implementation: mirror the residual2 gradient into
        // the attproj gradient slot so downstream readers see the post-skip value.
        model.grads_acts.attproj[layer] = model.grads_acts.residual2[layer]
    }


    private static func crossentropy_forward(losses: inout [Float], probs: [Float], targets: [UInt32], B: Int, T: Int, Vp: Int) {
        for batchIndex in 0..<B {
            for timeIndex in 0..<T {
                let index = batchIndex * T + timeIndex
                let target = Int(targets[index])
                losses[index] = -log(max(probs[index * Vp + target], 1e-30))
            }
        }
    }
}

// Returns the contiguous row-major stride for a shape.
private func bnnsRowMajorStride(for shape: [Int]) -> [Int] {
    guard !shape.isEmpty else { return [] }
    var stride = Array(repeating: 1, count: shape.count)
    if shape.count > 1 {
        for index in stride.indices.dropLast().reversed() {
            stride[index] = stride[index + 1] * shape[index + 1]
        }
    }
    return stride
}

// SAFETY: returns a buffer pointer that escapes the `withUnsafeMutableBufferPointer`
// closure. The pointer remains valid as long as the underlying Array's storage is not
// reallocated (i.e. the array is not grown, and is not subjected to copy-on-write while
// the pointer is in use). All callers in this file invoke the returned pointer
// exclusively inside a single `BNNSGraph.Context.executeFunction` call where no other
// Swift code mutates these arrays, which satisfies that requirement in practice.
private extension Array where Element == Float {
    @inline(__always)
    mutating func bnnsPinnedBuffer() -> UnsafeMutableBufferPointer<Float> {
        return withUnsafeMutableBufferPointer { $0 }
    }
}

private extension Array where Element == Int32 {
    @inline(__always)
    mutating func bnnsPinnedBuffer() -> UnsafeMutableBufferPointer<Int32> {
        return withUnsafeMutableBufferPointer { $0 }
    }
}
