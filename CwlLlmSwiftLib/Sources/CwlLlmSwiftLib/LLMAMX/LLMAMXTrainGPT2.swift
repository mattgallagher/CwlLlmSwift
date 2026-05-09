import Accelerate
import Foundation
import Numerics

enum LLMAMX {
    typealias GPT2 = LLMSwift.GPT2
    typealias UpdateParams = LLMSwift.UpdateParams

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

    static func matmul_forward(out: inout [Float], inp: [Float], weight: [Float], bias: [Float]?, B: Int, T: Int, C: Int, OC: Int) {
        LLMAMXBridge.gemm(
            out: &out,
            outRowStride: OC,
            lhs: inp,
            lhsRowStride: C,
            lhsColStride: 1,
            rhs: weight,
            rhsRowStride: C,
            rhsColStride: 1,
            rowCount: B * T,
            columnCount: OC,
            innerCount: C,
            accumulate: false
        )

        guard let bias else {
            return
        }
        bias.withUnsafeBufferPointer { biasBuffer in
            out.withUnsafeMutableBufferPointer { outBuffer in
                guard let biasBase = biasBuffer.baseAddress, let outBase = outBuffer.baseAddress else {
                    return
                }
                for bt in 0..<(B * T) {
                    cblas_saxpy(Int32(OC), 1, biasBase, 1, outBase.advanced(by: bt * OC), 1)
                }
            }
        }
    }

    static func matmul_backward(dinp: inout [Float], dweight: inout [Float], dbias: inout [Float], dout: [Float], inp: [Float], weight: [Float], B: Int, T: Int, C: Int, OC: Int) {
        LLMAMXBridge.gemm(
            out: &dinp,
            outRowStride: C,
            lhs: dout,
            lhsRowStride: OC,
            lhsColStride: 1,
            rhs: weight,
            rhsRowStride: 1,
            rhsColStride: C,
            rowCount: B * T,
            columnCount: C,
            innerCount: OC,
            accumulate: true
        )

        LLMAMXBridge.gemm(
            out: &dweight,
            outRowStride: C,
            lhs: dout,
            lhsRowStride: 1,
            lhsColStride: OC,
            rhs: inp,
            rhsRowStride: 1,
            rhsColStride: C,
            rowCount: OC,
            columnCount: C,
            innerCount: B * T,
            accumulate: true
        )

        guard !dbias.isEmpty else {
            return
        }
        dout.withUnsafeBufferPointer { doutBuffer in
            dbias.withUnsafeMutableBufferPointer { dbiasBuffer in
                guard let doutBase = doutBuffer.baseAddress, let dbiasBase = dbiasBuffer.baseAddress else {
                    return
                }
                for bt in 0..<(B * T) {
                    cblas_saxpy(Int32(OC), 1, doutBase.advanced(by: bt * OC), 1, dbiasBase, 1)
                }
            }
        }
    }

    static func attention_forward(out: inout [Float], preatt: inout [Float], att: inout [Float], inp: [Float], B: Int, T: Int, C: Int, NH: Int) {
        let headSize = C / NH
        let scale = 1 / Float(headSize).squareRoot()
        let iterationCount = B * NH
        var q = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var k = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var v = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var vAccum = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)

        for b in 0..<B {
            for h in 0..<NH {
                let iterationIndex = b * NH + h
                for t in 0..<T {
                    for hs in 0..<headSize {
                        let index = t * headSize + hs
                        let inputIndex = b * T * 3 * NH * headSize + t * 3 * NH * headSize + h * headSize + hs
                        q[iterationIndex][index] = inp[inputIndex]
                        k[iterationIndex][index] = inp[inputIndex + NH * headSize]
                        v[iterationIndex][index] = inp[inputIndex + 2 * NH * headSize]
                    }
                }

                let attBase = b * NH * T * T + h * T * T
                var preattBlock = Array(repeating: Float.zero, count: T * T)
                LLMAMXBridge.gemm(
                    out: &preattBlock,
                    outRowStride: T,
                    lhs: q[iterationIndex],
                    lhsRowStride: headSize,
                    lhsColStride: 1,
                    rhs: k[iterationIndex],
                    rhsRowStride: headSize,
                    rhsColStride: 1,
                    rowCount: T,
                    columnCount: T,
                    innerCount: headSize,
                    accumulate: false
                )
                preatt.replaceSubrange(attBase..<(attBase + T * T), with: preattBlock)

                for t in 0..<T {
                    var maxValue = -Float.infinity
                    var sumValue: Float = 0
                    let rowBase = attBase + t * T
                    for t2 in 0...t {
                        let valueIndex = rowBase + t2
                        let previousMax = maxValue
                        maxValue = max(maxValue, preatt[valueIndex])
                        sumValue *= exp(scale * (previousMax - maxValue))
                        sumValue += exp(scale * (preatt[valueIndex] - maxValue))
                    }
                    let normalization = 1 / sumValue
                    for t2 in 0..<T {
                        let valueIndex = rowBase + t2
                        att[valueIndex] = t2 <= t ? exp(scale * (preatt[valueIndex] - maxValue)) * normalization : 0
                    }
                }

                LLMAMXBridge.gemm(
                    out: &vAccum[iterationIndex],
                    outRowStride: headSize,
                    lhs: att[attBase..<(attBase + T * T)].map { $0 },
                    lhsRowStride: T,
                    lhsColStride: 1,
                    rhs: v[iterationIndex],
                    rhsRowStride: 1,
                    rhsColStride: headSize,
                    rowCount: T,
                    columnCount: headSize,
                    innerCount: T,
                    accumulate: false
                )

                for t in 0..<T {
                    for hs in 0..<headSize {
                        let valueIndex = t * headSize + hs
                        let outIndex = b * T * C + t * C + h * headSize + hs
                        out[outIndex] = vAccum[iterationIndex][valueIndex]
                    }
                }
            }
        }
    }

    static func attention_backward(dinp: inout [Float], dpreatt: inout [Float], datt: inout [Float], dout: [Float], inp: [Float], att: [Float], B: Int, T: Int, C: Int, NH: Int) {
        let headSize = C / NH
        let scale = 1 / Float(headSize).squareRoot()
        let iterationCount = B * NH
        var q = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var k = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var v = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var dq = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var dk = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var dv = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)
        var dvAccum = Array(repeating: Array(repeating: 0 as Float, count: T * headSize), count: iterationCount)

        for b in 0..<B {
            for h in 0..<NH {
                let iterationIndex = b * NH + h
                for t in 0..<T {
                    for hs in 0..<headSize {
                        let index = t * headSize + hs
                        let inputIndex = b * T * 3 * NH * headSize + t * 3 * NH * headSize + h * headSize + hs
                        q[iterationIndex][index] = inp[inputIndex]
                        k[iterationIndex][index] = inp[inputIndex + NH * headSize]
                        v[iterationIndex][index] = inp[inputIndex + 2 * NH * headSize]
                        dvAccum[iterationIndex][index] = dout[b * T * C + t * C + h * headSize + hs]
                    }
                }

                let attBase = b * NH * T * T + h * T * T
                let attBlock = att[attBase..<(attBase + T * T)].map { $0 }
                var dattBlock = Array(repeating: Float.zero, count: T * T)
                LLMAMXBridge.gemm(
                    out: &dattBlock,
                    outRowStride: T,
                    lhs: dvAccum[iterationIndex],
                    lhsRowStride: headSize,
                    lhsColStride: 1,
                    rhs: v[iterationIndex],
                    rhsRowStride: headSize,
                    rhsColStride: 1,
                    rowCount: T,
                    columnCount: T,
                    innerCount: headSize,
                    accumulate: false
                )
                datt.replaceSubrange(attBase..<(attBase + T * T), with: dattBlock)

                LLMAMXBridge.gemm(
                    out: &dv[iterationIndex],
                    outRowStride: headSize,
                    lhs: attBlock,
                    lhsRowStride: 1,
                    lhsColStride: T,
                    rhs: dvAccum[iterationIndex],
                    rhsRowStride: 1,
                    rhsColStride: headSize,
                    rowCount: T,
                    columnCount: headSize,
                    innerCount: T,
                    accumulate: false
                )

                for t in 0..<T {
                    let attRowBase = attBase + t * T
                    for t2 in 0...t {
                        for t3 in 0...t {
                            let indicator: Float = t2 == t3 ? 1 : 0
                            let derivative = att[attRowBase + t2] * (indicator - att[attRowBase + t3]) * scale
                            dpreatt[attRowBase + t3] = Relaxed.multiplyAdd(derivative, datt[attRowBase + t2], dpreatt[attRowBase + t3])
                        }
                    }
                }

                let dpreattBlock = dpreatt[attBase..<(attBase + T * T)].map { $0 }
                LLMAMXBridge.gemm(
                    out: &dq[iterationIndex],
                    outRowStride: headSize,
                    lhs: dpreattBlock,
                    lhsRowStride: T,
                    lhsColStride: 1,
                    rhs: k[iterationIndex],
                    rhsRowStride: 1,
                    rhsColStride: headSize,
                    rowCount: T,
                    columnCount: headSize,
                    innerCount: T,
                    accumulate: false
                )
                LLMAMXBridge.gemm(
                    out: &dk[iterationIndex],
                    outRowStride: headSize,
                    lhs: dpreattBlock,
                    lhsRowStride: 1,
                    lhsColStride: T,
                    rhs: q[iterationIndex],
                    rhsRowStride: 1,
                    rhsColStride: headSize,
                    rowCount: T,
                    columnCount: headSize,
                    innerCount: T,
                    accumulate: false
                )

                for t in 0..<T {
                    for hs in 0..<headSize {
                        let index = t * headSize + hs
                        let inputIndex = b * T * 3 * NH * headSize + t * 3 * NH * headSize + h * headSize + hs
                        dinp[inputIndex] = dq[iterationIndex][index]
                        dinp[inputIndex + NH * headSize] = dk[iterationIndex][index]
                        dinp[inputIndex + 2 * NH * headSize] = dv[iterationIndex][index]
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
        cblas_scopy(Int32(N), inp1, 1, &out, 1)
        cblas_saxpy(Int32(N), 1, inp2, 1, &out, 1)
    }

    static func residual_backward(dinp1: inout [Float], dinp2: inout [Float], dout: [Float], N: Int) {
        cblas_saxpy(Int32(N), 1, dout, 1, &dinp1, 1)
        cblas_saxpy(Int32(N), 1, dout, 1, &dinp2, 1)
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
            var meanLoss: Float = 0
            for i in 0..<(B * T) {
                meanLoss += model.acts.losses[i]
            }
            model.mean_loss = meanLoss / Float(B * T)
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

        let dlossMean = 1 / Float(B * T)
        for i in 0..<(B * T) {
            model.grads_acts.losses[i] = dlossMean
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
