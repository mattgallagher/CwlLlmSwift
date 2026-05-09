import Dispatch
import Foundation
import Numerics

private struct SendableUnsafeMutableBuffer<Element>: @unchecked Sendable {
    let baseAddress: UnsafeMutablePointer<Element>

    subscript(index: Int) -> Element {
        get { baseAddress[index] }
        nonmutating set { baseAddress[index] = newValue }
    }
}

enum LLMMultithreadedSwift {
    typealias GPT2Config = LLMSwift.GPT2Config
    typealias ParameterTensors = LLMSwift.ParameterTensors
    typealias ActivationTensors = LLMSwift.ActivationTensors
    typealias GPT2 = LLMSwift.GPT2
    typealias UpdateParams = LLMSwift.UpdateParams

    static let gelu_scaling_factor = LLMSwift.gelu_scaling_factor

    static func encoder_forward(out: inout [Float], inp: [UInt32], wte: [Float], wpe: [Float], B: Int, T: Int, C: Int) {
        LLMSwift.encoder_forward(out: &out, inp: inp, wte: wte, wpe: wpe, B: B, T: T, C: C)
    }

    static func encoder_backward(dwte: inout [Float], dwpe: inout [Float], dout: [Float], inp: [UInt32], B: Int, T: Int, C: Int) {
        LLMSwift.encoder_backward(dwte: &dwte, dwpe: &dwpe, dout: dout, inp: inp, B: B, T: T, C: C)
    }

    static func layernorm_forward(out: inout [Float], mean: inout [Float], rstd: inout [Float], inp: [Float], weight: [Float], bias: [Float], B: Int, T: Int, C: Int) {
        LLMSwift.layernorm_forward(out: &out, mean: &mean, rstd: &rstd, inp: inp, weight: weight, bias: bias, B: B, T: T, C: C)
    }

    static func layernorm_backward(dinp: inout [Float], dweight: inout [Float], dbias: inout [Float], dout: [Float], inp: [Float], weight: [Float], mean: [Float], rstd: [Float], B: Int, T: Int, C: Int) {
        LLMSwift.layernorm_backward(dinp: &dinp, dweight: &dweight, dbias: &dbias, dout: dout, inp: inp, weight: weight, mean: mean, rstd: rstd, B: B, T: T, C: C)
    }

    static func matmul_forward_naive(out: inout [Float], inp: [Float], weight: [Float], bias: [Float]?, B: Int, T: Int, C: Int, OC: Int) {
        LLMSwift.matmul_forward_naive(out: &out, inp: inp, weight: weight, bias: bias, B: B, T: T, C: C, OC: OC)
    }

    static func matmul_forward(out: inout [Float], inp: [Float], weight: [Float], bias: [Float]?, B: Int, T: Int, C: Int, OC: Int) {
        let LOOP_UNROLL = 8
        let BT = B * T
        if BT % LOOP_UNROLL != 0 {
            matmul_forward_naive(out: &out, inp: inp, weight: weight, bias: bias, B: B, T: T, C: C, OC: OC)
            return
        }

        let tileCount = BT / LOOP_UNROLL
        let workerCount = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let chunkSize = max(1, (tileCount + workerCount - 1) / workerCount)
        let chunkCount = (tileCount + chunkSize - 1) / chunkSize
        let bias = bias?.span
        let inp = inp.span
        let weight = weight.span

        out.withUnsafeMutableBufferPointer { outBuffer in
            let outStorage = SendableUnsafeMutableBuffer(baseAddress: outBuffer.baseAddress!)

            DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
                let startTile = chunk * chunkSize
                let endTile = min(tileCount, startTile + chunkSize)

                for tile in startTile..<endTile {
                    let obt = tile * LOOP_UNROLL

                    for o in 0..<OC {
                        var result = InlineArray<8, Float>(repeating: bias?[o] ?? 0)
                        let bt = inp.extracting(droppingFirst: obt * C)
                        let w = weight.extracting(droppingFirst: o * C)

                        for i in 0..<C {
                            for r in result.indices {
                                result[r] = Relaxed.multiplyAdd(bt[r * C + i], w[i], result[r])
                            }
                        }

                        for r in result.indices {
                            outStorage[(obt + r) * OC + o] = result[r]
                        }
                    }
                }
            }
        }
    }

    static func matmul_backward(dinp: inout [Float], dweight: inout [Float], dbias: inout [Float], dout: [Float], inp: [Float], weight: [Float], B: Int, T: Int, C: Int, OC: Int) {
        let bt = B * T
        let workerCount = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let btChunkSize = max(1, (bt + workerCount - 1) / workerCount)
        let btChunkCount = (bt + btChunkSize - 1) / btChunkSize
        let ocChunkSize = max(1, (OC + workerCount - 1) / workerCount)
        let ocChunkCount = (OC + ocChunkSize - 1) / ocChunkSize
        let hasBiasGradient = !dbias.isEmpty
        let dout = dout.span
        let inp = inp.span
        let weight = weight.span

        dinp.withUnsafeMutableBufferPointer { dinpBuffer in
            dweight.withUnsafeMutableBufferPointer { dweightBuffer in
                dbias.withUnsafeMutableBufferPointer { dbiasBuffer in
                    let dinpStorage = SendableUnsafeMutableBuffer(baseAddress: dinpBuffer.baseAddress!)
                    let dweightStorage = SendableUnsafeMutableBuffer(baseAddress: dweightBuffer.baseAddress!)
                    let dbiasStorage = SendableUnsafeMutableBuffer(baseAddress: dbiasBuffer.baseAddress!)

                    DispatchQueue.concurrentPerform(iterations: btChunkCount) { chunk in
                        let startBT = chunk * btChunkSize
                        let endBT = min(bt, startBT + btChunkSize)

                        for row in startBT..<endBT {
                            let doutBT = row * OC
                            let dinpBT = row * C
                            for o in stride(from: 0, to: OC - 1, by: 2) {
                                let wrow0 = o * C
                                let wrow1 = (o + 1) * C
                                let d0 = dout[doutBT + o]
                                let d1 = dout[doutBT + o + 1]
                                for i in 0..<C {
                                    dinpStorage[dinpBT + i] = Relaxed.multiplyAdd(weight[wrow0 + i], d0, dinpStorage[dinpBT + i])
                                    dinpStorage[dinpBT + i] = Relaxed.multiplyAdd(weight[wrow1 + i], d1, dinpStorage[dinpBT + i])
                                }
                            }
                            if OC & 1 == 1 {
                                let wrow = (OC - 1) * C
                                let d = dout[doutBT + (OC - 1)]
                                for i in 0..<C {
                                    dinpStorage[dinpBT + i] = Relaxed.multiplyAdd(weight[wrow + i], d, dinpStorage[dinpBT + i])
                                }
                            }
                        }
                    }

                    DispatchQueue.concurrentPerform(iterations: ocChunkCount) { chunk in
                        let startOC = chunk * ocChunkSize
                        let endOC = min(OC, startOC + ocChunkSize)

                        for o in startOC..<endOC {
                            let dwrow = o * C
                            for row in 0..<bt {
                                let doutBT = row * OC
                                let inpBT = row * C
                                let d = dout[doutBT + o]
                                if hasBiasGradient {
                                    dbiasStorage[o] = Relaxed.sum(dbiasStorage[o], d)
                                }
                                for i in 0..<C {
                                    dweightStorage[dwrow + i] = Relaxed.multiplyAdd(inp[inpBT + i], d, dweightStorage[dwrow + i])
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static func attention_forward(out: inout [Float], preatt: inout [Float], att: inout [Float], inp: [Float], B: Int, T: Int, C: Int, NH: Int) {
        let c3 = C * 3
        let hs = C / NH
        let scale = 1 / Float(hs).squareRoot()
        let bthCount = B * T * NH
        let workerCount = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let chunkSize = max(1, (bthCount + workerCount - 1) / workerCount)
        let chunkCount = (bthCount + chunkSize - 1) / chunkSize
        let inp = inp.span

        out.withUnsafeMutableBufferPointer { outBuffer in
            preatt.withUnsafeMutableBufferPointer { preattBuffer in
                att.withUnsafeMutableBufferPointer { attBuffer in
                    let outStorage = SendableUnsafeMutableBuffer(baseAddress: outBuffer.baseAddress!)
                    let preattStorage = SendableUnsafeMutableBuffer(baseAddress: preattBuffer.baseAddress!)
                    let attStorage = SendableUnsafeMutableBuffer(baseAddress: attBuffer.baseAddress!)

                    DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
                        let start = chunk * chunkSize
                        let end = min(bthCount, start + chunkSize)

                        for bth in start..<end {
                            let b = bth / (T * NH)
                            let th = bth % (T * NH)
                            let t = th / NH
                            let h = th % NH

                            let query_t = b * T * c3 + t * c3 + h * hs
                            let preatt_bth = b * NH * T * T + h * T * T + t * T
                            let att_bth = b * NH * T * T + h * T * T + t * T

                            var maxval: Float = -10000
                            for t2 in 0...t {
                                let key_t2 = b * T * c3 + t2 * c3 + h * hs + C
                                var val: Float = 0
                                for i in 0..<hs {
                                    val = Relaxed.multiplyAdd(inp[query_t + i], inp[key_t2 + i], val)
                                }
                                val *= scale
                                maxval = max(maxval, val)
                                preattStorage[preatt_bth + t2] = val
                            }

                            var expsum: Float = 0
                            for t2 in 0...t {
                                let expv = exp(preattStorage[preatt_bth + t2] - maxval)
                                expsum = Relaxed.sum(expsum, expv)
                                attStorage[att_bth + t2] = expv
                            }
                            let expsumInv = expsum == 0 ? 0 : 1 / expsum

                            for t2 in 0..<T {
                                if t2 <= t {
                                    attStorage[att_bth + t2] *= expsumInv
                                } else {
                                    attStorage[att_bth + t2] = 0
                                }
                            }

                            let out_bth = b * T * C + t * C + h * hs
                            for i in 0..<hs {
                                outStorage[out_bth + i] = 0
                            }
                            for t2 in 0...t {
                                let value_t2 = b * T * c3 + t2 * c3 + h * hs + 2 * C
                                let att_btht2 = attStorage[att_bth + t2]
                                for i in 0..<hs {
                                    outStorage[out_bth + i] = Relaxed.multiplyAdd(att_btht2, inp[value_t2 + i], outStorage[out_bth + i])
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static func attention_backward(dinp: inout [Float], dpreatt: inout [Float], datt: inout [Float], dout: [Float], inp: [Float], att: [Float], B: Int, T: Int, C: Int, NH: Int) {
        let c3 = C * 3
        let hs = C / NH
        let scale = 1 / Float(hs).squareRoot()
        let bhCount = B * NH
        let workerCount = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let chunkSize = max(1, (bhCount + workerCount - 1) / workerCount)
        let chunkCount = (bhCount + chunkSize - 1) / chunkSize
        let dout = dout.span
        let inp = inp.span
        let att = att.span

        dinp.withUnsafeMutableBufferPointer { dinpBuffer in
            dpreatt.withUnsafeMutableBufferPointer { dpreattBuffer in
                datt.withUnsafeMutableBufferPointer { dattBuffer in
                    let dinpStorage = SendableUnsafeMutableBuffer(baseAddress: dinpBuffer.baseAddress!)
                    let dpreattStorage = SendableUnsafeMutableBuffer(baseAddress: dpreattBuffer.baseAddress!)
                    let dattStorage = SendableUnsafeMutableBuffer(baseAddress: dattBuffer.baseAddress!)

                    DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
                        let start = chunk * chunkSize
                        let end = min(bhCount, start + chunkSize)

                        for bh in start..<end {
                            let b = bh / NH
                            let h = bh % NH

                            for t in 0..<T {
                                let att_bth = b * NH * T * T + h * T * T + t * T
                                let datt_bth = b * NH * T * T + h * T * T + t * T
                                let dpreatt_bth = b * NH * T * T + h * T * T + t * T
                                let dquery_t = b * T * c3 + t * c3 + h * hs
                                let query_t = b * T * c3 + t * c3 + h * hs
                                let dout_bth = b * T * C + t * C + h * hs

                                for t2 in 0...t {
                                    let value_t2 = b * T * c3 + t2 * c3 + h * hs + 2 * C
                                    let dvalue_t2 = b * T * c3 + t2 * c3 + h * hs + 2 * C
                                    for i in 0..<hs {
                                        dattStorage[datt_bth + t2] = Relaxed.multiplyAdd(inp[value_t2 + i], dout[dout_bth + i], dattStorage[datt_bth + t2])
                                        dinpStorage[dvalue_t2 + i] = Relaxed.multiplyAdd(att[att_bth + t2], dout[dout_bth + i], dinpStorage[dvalue_t2 + i])
                                    }
                                }

                                for t2 in 0...t {
                                    for t3 in 0...t {
                                        let indicator: Float = t2 == t3 ? 1 : 0
                                        let localDerivative = att[att_bth + t2] * (indicator - att[att_bth + t3])
                                        dpreattStorage[dpreatt_bth + t3] = Relaxed.multiplyAdd(localDerivative, dattStorage[datt_bth + t2], dpreattStorage[dpreatt_bth + t3])
                                    }
                                }

                                for t2 in 0...t {
                                    let key_t2 = b * T * c3 + t2 * c3 + h * hs + C
                                    let dkey_t2 = b * T * c3 + t2 * c3 + h * hs + C
                                    let scaledGradient = dpreattStorage[dpreatt_bth + t2] * scale
                                    for i in 0..<hs {
                                        dinpStorage[dquery_t + i] = Relaxed.multiplyAdd(inp[key_t2 + i], scaledGradient, dinpStorage[dquery_t + i])
                                        dinpStorage[dkey_t2 + i] = Relaxed.multiplyAdd(inp[query_t + i], scaledGradient, dinpStorage[dkey_t2 + i])
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static func gelu_forward(out: inout [Float], inp: [Float], N: Int) {
        LLMSwift.gelu_forward(out: &out, inp: inp, N: N)
    }

    static func gelu_backward(dinp: inout [Float], inp: [Float], dout: [Float], N: Int) {
        LLMSwift.gelu_backward(dinp: &dinp, inp: inp, dout: dout, N: N)
    }

    static func residual_forward(out: inout [Float], inp1: [Float], inp2: [Float], N: Int) {
        LLMSwift.residual_forward(out: &out, inp1: inp1, inp2: inp2, N: N)
    }

    static func residual_backward(dinp1: inout [Float], dinp2: inout [Float], dout: [Float], N: Int) {
        LLMSwift.residual_backward(dinp1: &dinp1, dinp2: &dinp2, dout: dout, N: N)
    }

    static func softmax_forward(probs: inout [Float], logits: [Float], B: Int, T: Int, V: Int, Vp: Int) {
        LLMSwift.softmax_forward(probs: &probs, logits: logits, B: B, T: T, V: V, Vp: Vp)
    }

    static func crossentropy_forward(losses: inout [Float], probs: [Float], targets: [UInt32], B: Int, T: Int, Vp: Int) {
        LLMSwift.crossentropy_forward(losses: &losses, probs: probs, targets: targets, B: B, T: T, Vp: Vp)
    }

    static func crossentropy_softmax_backward(dlogits: inout [Float], dlosses: [Float], probs: [Float], targets: [UInt32], B: Int, T: Int, V: Int, Vp: Int) {
        LLMSwift.crossentropy_softmax_backward(dlogits: &dlogits, dlosses: dlosses, probs: probs, targets: targets, B: B, T: T, V: V, Vp: Vp)
    }

    static func buildModel(from checkpointData: Data) throws -> GPT2 {
        try LLMSwift.buildModel(from: checkpointData)
    }

    static func gpt2_forward(model: inout GPT2, inputs: [UInt32], targets: [UInt32], B: Int, T: Int) {
        let V = model.config.vocab_size
        let Vp = model.config.padded_vocab_size
        let L = model.config.num_layers
        let NH = model.config.num_heads
        let C = model.config.channels

        model.batch_size = B
        model.seq_len = T
        _ = model.acts.resizeIfNeeded(config: model.config, B: B, T: T)
        precondition(inputs.allSatisfy { Int($0) >= 0 && Int($0) < V })
        precondition(targets.isEmpty || targets.allSatisfy { Int($0) >= 0 && Int($0) < V })
        model.inputs = inputs
        model.targets = targets

        encoder_forward(out: &model.acts.encoded, inp: inputs, wte: model.params.wte, wpe: model.params.wpe, B: B, T: T, C: C)
        for l in 0..<L {
            let residual = l == 0 ? model.acts.encoded : model.acts.residual3[l - 1]
            layernorm_forward(out: &model.acts.ln1[l], mean: &model.acts.ln1_mean[l], rstd: &model.acts.ln1_rstd[l], inp: residual, weight: model.params.ln1w[l], bias: model.params.ln1b[l], B: B, T: T, C: C)
            matmul_forward(out: &model.acts.qkv[l], inp: model.acts.ln1[l], weight: model.params.qkvw[l], bias: model.params.qkvb[l], B: B, T: T, C: C, OC: 3 * C)
            attention_forward(out: &model.acts.atty[l], preatt: &model.acts.preatt[l], att: &model.acts.att[l], inp: model.acts.qkv[l], B: B, T: T, C: C, NH: NH)
            matmul_forward(out: &model.acts.attproj[l], inp: model.acts.atty[l], weight: model.params.attprojw[l], bias: model.params.attprojb[l], B: B, T: T, C: C, OC: C)
            residual_forward(out: &model.acts.residual2[l], inp1: residual, inp2: model.acts.attproj[l], N: B * T * C)
            layernorm_forward(out: &model.acts.ln2[l], mean: &model.acts.ln2_mean[l], rstd: &model.acts.ln2_rstd[l], inp: model.acts.residual2[l], weight: model.params.ln2w[l], bias: model.params.ln2b[l], B: B, T: T, C: C)
            matmul_forward(out: &model.acts.fch[l], inp: model.acts.ln2[l], weight: model.params.fcw[l], bias: model.params.fcb[l], B: B, T: T, C: C, OC: 4 * C)
            gelu_forward(out: &model.acts.fch_gelu[l], inp: model.acts.fch[l], N: B * T * 4 * C)
            matmul_forward(out: &model.acts.fcproj[l], inp: model.acts.fch_gelu[l], weight: model.params.fcprojw[l], bias: model.params.fcprojb[l], B: B, T: T, C: 4 * C, OC: C)
            residual_forward(out: &model.acts.residual3[l], inp1: model.acts.residual2[l], inp2: model.acts.fcproj[l], N: B * T * C)
        }
        layernorm_forward(out: &model.acts.lnf, mean: &model.acts.lnf_mean, rstd: &model.acts.lnf_rstd, inp: model.acts.residual3[L - 1], weight: model.params.lnfw, bias: model.params.lnfb, B: B, T: T, C: C)
        matmul_forward(out: &model.acts.logits, inp: model.acts.lnf, weight: model.params.wte, bias: nil, B: B, T: T, C: C, OC: Vp)
        softmax_forward(probs: &model.acts.probs, logits: model.acts.logits, B: B, T: T, V: V, Vp: Vp)

        if !targets.isEmpty {
            crossentropy_forward(losses: &model.acts.losses, probs: model.acts.probs, targets: targets, B: B, T: T, Vp: Vp)
            var mean_loss: Float = 0
            for i in 0..<(B * T) {
                mean_loss = Relaxed.sum(mean_loss, model.acts.losses[i])
            }
            mean_loss /= Float(B * T)
            model.mean_loss = mean_loss
        } else {
            model.mean_loss = -1
        }
    }

    static func gpt2_zero_grad(model: inout GPT2) {
        LLMSwift.gpt2_zero_grad(model: &model)
    }

    static func gpt2_backward(model: inout GPT2) {
        precondition(model.mean_loss != -1, "Must forward with targets before backward")

        let B = model.batch_size
        let T = model.seq_len
        let V = model.config.vocab_size
        let Vp = model.config.padded_vocab_size
        let L = model.config.num_layers
        let NH = model.config.num_heads
        let C = model.config.channels

        _ = model.grads_acts.resizeIfNeeded(config: model.config, B: B, T: T)
        gpt2_zero_grad(model: &model)

        let dloss_mean = 1 / Float(B * T)
        for i in 0..<(B * T) {
            model.grads_acts.losses[i] = dloss_mean
        }

        crossentropy_softmax_backward(dlogits: &model.grads_acts.logits, dlosses: model.grads_acts.losses, probs: model.acts.probs, targets: model.targets, B: B, T: T, V: V, Vp: Vp)
        var emptyBias: [Float] = []
        matmul_backward(dinp: &model.grads_acts.lnf, dweight: &model.grads.wte, dbias: &emptyBias, dout: model.grads_acts.logits, inp: model.acts.lnf, weight: model.params.wte, B: B, T: T, C: C, OC: Vp)
        layernorm_backward(dinp: &model.grads_acts.residual3[L - 1], dweight: &model.grads.lnfw, dbias: &model.grads.lnfb, dout: model.grads_acts.lnf, inp: model.acts.residual3[L - 1], weight: model.params.lnfw, mean: model.acts.lnf_mean, rstd: model.acts.lnf_rstd, B: B, T: T, C: C)

        for l in stride(from: L - 1, through: 0, by: -1) {
            residual_backward(dinp1: &model.grads_acts.residual2[l], dinp2: &model.grads_acts.fcproj[l], dout: model.grads_acts.residual3[l], N: B * T * C)
            matmul_backward(dinp: &model.grads_acts.fch_gelu[l], dweight: &model.grads.fcprojw[l], dbias: &model.grads.fcprojb[l], dout: model.grads_acts.fcproj[l], inp: model.acts.fch_gelu[l], weight: model.params.fcprojw[l], B: B, T: T, C: 4 * C, OC: C)
            gelu_backward(dinp: &model.grads_acts.fch[l], inp: model.acts.fch[l], dout: model.grads_acts.fch_gelu[l], N: B * T * 4 * C)
            matmul_backward(dinp: &model.grads_acts.ln2[l], dweight: &model.grads.fcw[l], dbias: &model.grads.fcb[l], dout: model.grads_acts.fch[l], inp: model.acts.ln2[l], weight: model.params.fcw[l], B: B, T: T, C: C, OC: 4 * C)
            layernorm_backward(dinp: &model.grads_acts.residual2[l], dweight: &model.grads.ln2w[l], dbias: &model.grads.ln2b[l], dout: model.grads_acts.ln2[l], inp: model.acts.residual2[l], weight: model.params.ln2w[l], mean: model.acts.ln2_mean[l], rstd: model.acts.ln2_rstd[l], B: B, T: T, C: C)
            if l == 0 {
                residual_backward(dinp1: &model.grads_acts.encoded, dinp2: &model.grads_acts.attproj[l], dout: model.grads_acts.residual2[l], N: B * T * C)
            } else {
                residual_backward(dinp1: &model.grads_acts.residual3[l - 1], dinp2: &model.grads_acts.attproj[l], dout: model.grads_acts.residual2[l], N: B * T * C)
            }
            matmul_backward(dinp: &model.grads_acts.atty[l], dweight: &model.grads.attprojw[l], dbias: &model.grads.attprojb[l], dout: model.grads_acts.attproj[l], inp: model.acts.atty[l], weight: model.params.attprojw[l], B: B, T: T, C: C, OC: C)
            attention_backward(dinp: &model.grads_acts.qkv[l], dpreatt: &model.grads_acts.preatt[l], datt: &model.grads_acts.att[l], dout: model.grads_acts.atty[l], inp: model.acts.qkv[l], att: model.acts.att[l], B: B, T: T, C: C, NH: NH)
            matmul_backward(dinp: &model.grads_acts.ln1[l], dweight: &model.grads.qkvw[l], dbias: &model.grads.qkvb[l], dout: model.grads_acts.qkv[l], inp: model.acts.ln1[l], weight: model.params.qkvw[l], B: B, T: T, C: C, OC: 3 * C)
            if l == 0 {
                layernorm_backward(dinp: &model.grads_acts.encoded, dweight: &model.grads.ln1w[l], dbias: &model.grads.ln1b[l], dout: model.grads_acts.ln1[l], inp: model.acts.encoded, weight: model.params.ln1w[l], mean: model.acts.ln1_mean[l], rstd: model.acts.ln1_rstd[l], B: B, T: T, C: C)
            } else {
                layernorm_backward(dinp: &model.grads_acts.residual3[l - 1], dweight: &model.grads.ln1w[l], dbias: &model.grads.ln1b[l], dout: model.grads_acts.ln1[l], inp: model.acts.residual3[l - 1], weight: model.params.ln1w[l], mean: model.acts.ln1_mean[l], rstd: model.acts.ln1_rstd[l], B: B, T: T, C: C)
            }
        }

        encoder_backward(dwte: &model.grads.wte, dwpe: &model.grads.wpe, dout: model.grads_acts.encoded, inp: model.inputs, B: B, T: T, C: C)
    }

    static func gpt2_update(model: inout GPT2, update_params: UpdateParams) {
        LLMSwift.gpt2_update(model: &model, update_params: update_params)
    }

    static func sample(logits: ArraySlice<Float>, temperature: Double, state: inout UInt64) -> Int {
        LLMSwift.sample(logits: logits, temperature: temperature, state: &state)
    }
}
