import CoreML
import Foundation

enum LLMMLTensor {
    typealias GPT2Config = LLMGPT2Config

    struct ShapeKey: Hashable {
        let batchSize: Int
        let sequenceLength: Int
    }

    struct ShapeContext {
        let batchSize: Int
        let sequenceLength: Int
        let vocabSize: Int
        let attentionMask: MLTensor
        let crossEntropyBatchIndices: MLTensor
        let crossEntropyTimeIndices: MLTensor
        let targetColumnIndices: MLTensor
        let meanLossScale: MLTensor

        init(batchSize: Int, sequenceLength: Int, vocabSize: Int) {
            self.batchSize = batchSize
            self.sequenceLength = sequenceLength
            self.vocabSize = vocabSize

            var attentionMaskScalars = Array(repeating: false, count: sequenceLength * sequenceLength)
            for row in 0..<sequenceLength {
                for column in (row + 1)..<sequenceLength {
                    attentionMaskScalars[row * sequenceLength + column] = true
                }
            }
            attentionMask = MLTensor(shape: [1, 1, sequenceLength, sequenceLength], scalars: attentionMaskScalars, scalarType: Bool.self)

            var batchIndexScalars = [Int32]()
            batchIndexScalars.reserveCapacity(batchSize * sequenceLength)
            for batchIndex in 0..<batchSize {
                batchIndexScalars.append(contentsOf: Array(repeating: Int32(batchIndex), count: sequenceLength))
            }
            crossEntropyBatchIndices = MLTensor(shape: [batchSize, sequenceLength, 1], scalars: batchIndexScalars)

            var timeIndexScalars = [Int32]()
            timeIndexScalars.reserveCapacity(batchSize * sequenceLength)
            for _ in 0..<batchSize {
                timeIndexScalars.append(contentsOf: Array(0..<sequenceLength).map(Int32.init))
            }
            crossEntropyTimeIndices = MLTensor(shape: [batchSize, sequenceLength, 1], scalars: timeIndexScalars)

            targetColumnIndices = MLTensor(Int32(0)..<Int32(vocabSize), scalarType: Int32.self).expandingShape(at: 0, 1)
            meanLossScale = MLTensor(repeating: 1 / Float(batchSize * sequenceLength), shape: [batchSize, sequenceLength])
        }
    }

    struct MLParameterTensors {
        var wte: MLTensor
        var wpe: MLTensor
        var ln1w: [MLTensor]
        var ln1b: [MLTensor]
        var qkvw: [MLTensor]
        var qkvb: [MLTensor]
        var attprojw: [MLTensor]
        var attprojb: [MLTensor]
        var ln2w: [MLTensor]
        var ln2b: [MLTensor]
        var fcw: [MLTensor]
        var fcb: [MLTensor]
        var fcprojw: [MLTensor]
        var fcprojb: [MLTensor]
        var lnfw: MLTensor
        var lnfb: MLTensor

        var allTensors: [MLTensor] {
            var result = [wte, wpe]
            result.append(contentsOf: ln1w)
            result.append(contentsOf: ln1b)
            result.append(contentsOf: qkvw)
            result.append(contentsOf: qkvb)
            result.append(contentsOf: attprojw)
            result.append(contentsOf: attprojb)
            result.append(contentsOf: ln2w)
            result.append(contentsOf: ln2b)
            result.append(contentsOf: fcw)
            result.append(contentsOf: fcb)
            result.append(contentsOf: fcprojw)
            result.append(contentsOf: fcprojb)
            result.append(contentsOf: [lnfw, lnfb])
            return result
        }

        static func zeros(config: GPT2Config) -> MLParameterTensors {
            let L = config.num_layers
            let zero = MLTensor(zeros: [1], scalarType: Float.self)
            return MLParameterTensors(
                wte: zero, wpe: zero,
                ln1w: [MLTensor](repeating: zero, count: L),
                ln1b: [MLTensor](repeating: zero, count: L),
                qkvw: [MLTensor](repeating: zero, count: L),
                qkvb: [MLTensor](repeating: zero, count: L),
                attprojw: [MLTensor](repeating: zero, count: L),
                attprojb: [MLTensor](repeating: zero, count: L),
                ln2w: [MLTensor](repeating: zero, count: L),
                ln2b: [MLTensor](repeating: zero, count: L),
                fcw: [MLTensor](repeating: zero, count: L),
                fcb: [MLTensor](repeating: zero, count: L),
                fcprojw: [MLTensor](repeating: zero, count: L),
                fcprojb: [MLTensor](repeating: zero, count: L),
                lnfw: zero, lnfb: zero
            )
        }
    }

    struct MLActivationTensors {
        var encoded = MLTensor(zeros: [1], scalarType: Float.self)
        var ln1: [MLTensor]
        var ln1_mean: [MLTensor]
        var ln1_rstd: [MLTensor]
        var qkv: [MLTensor]
        var atty: [MLTensor]
        var att: [MLTensor]
        var attproj: [MLTensor]
        var residual2: [MLTensor]
        var ln2: [MLTensor]
        var ln2_mean: [MLTensor]
        var ln2_rstd: [MLTensor]
        var fch: [MLTensor]
        var fch_gelu: [MLTensor]
        var fcproj: [MLTensor]
        var residual3: [MLTensor]
        var lnf = MLTensor(zeros: [1], scalarType: Float.self)
        var lnf_mean = MLTensor(zeros: [1], scalarType: Float.self)
        var lnf_rstd = MLTensor(zeros: [1], scalarType: Float.self)
        var logits = MLTensor(zeros: [1], scalarType: Float.self)
        var probs = MLTensor(zeros: [1], scalarType: Float.self)
        var losses = MLTensor(zeros: [1], scalarType: Float.self)

        init(config: GPT2Config) {
            let zero = MLTensor(zeros: [1], scalarType: Float.self)
            ln1 = [MLTensor](repeating: zero, count: config.num_layers)
            ln1_mean = [MLTensor](repeating: zero, count: config.num_layers)
            ln1_rstd = [MLTensor](repeating: zero, count: config.num_layers)
            qkv = [MLTensor](repeating: zero, count: config.num_layers)
            atty = [MLTensor](repeating: zero, count: config.num_layers)
            att = [MLTensor](repeating: zero, count: config.num_layers)
            attproj = [MLTensor](repeating: zero, count: config.num_layers)
            residual2 = [MLTensor](repeating: zero, count: config.num_layers)
            ln2 = [MLTensor](repeating: zero, count: config.num_layers)
            ln2_mean = [MLTensor](repeating: zero, count: config.num_layers)
            ln2_rstd = [MLTensor](repeating: zero, count: config.num_layers)
            fch = [MLTensor](repeating: zero, count: config.num_layers)
            fch_gelu = [MLTensor](repeating: zero, count: config.num_layers)
            fcproj = [MLTensor](repeating: zero, count: config.num_layers)
            residual3 = [MLTensor](repeating: zero, count: config.num_layers)
        }
    }

    struct AttentionForwardResult {
        let out: MLTensor
        let preatt: MLTensor
        let att: MLTensor
    }

    // MARK: - Layer operations (forward)

    static func encoder_forward(inp: MLTensor, wte: MLTensor, wpe: MLTensor, B: Int, T: Int, C: Int, V: Int) -> MLTensor {
        let inputs = inp.reshaped(to: [B, T])
        let tokenEmbedding = wte.gathering(atIndices: inputs, alongAxis: 0)
        let positionEmbedding = wpe[0..<T, 0..<C].expandingShape(at: 0)
        return tokenEmbedding + positionEmbedding
    }

    static func layernorm_forward(inp: MLTensor, weight: MLTensor, bias: MLTensor) -> (out: MLTensor, mean: MLTensor, rstd: MLTensor) {
        let mean = inp.mean(alongAxes: 2, keepRank: true)
        let variance = (inp - mean).squared().mean(alongAxes: 2, keepRank: true)
        let eps: Float = 1e-5
        let rstd = (variance + eps).rsqrt()
        let out = rstd * (inp - mean) * weight + bias
        return (out, mean, rstd)
    }

    static func matmul_forward(inp: MLTensor, weight: MLTensor, bias: MLTensor?) -> MLTensor {
        let product = inp.matmul(weight.transposed(permutation: 1, 0))
        if let bias {
            return product + bias
        }
        return product
    }

    static func attention_forward(inp: MLTensor, mask: MLTensor, B: Int, T: Int, C: Int, NH: Int, HS: Int, scale: Float) -> AttentionForwardResult {
        let q = inp[0..., 0..., 0, 0..., 0...].transposed(permutation: 0, 2, 1, 3)
        let k = inp[0..., 0..., 1, 0..., 0...].transposed(permutation: 0, 2, 3, 1)
        let v = inp[0..., 0..., 2, 0..., 0...].transposed(permutation: 0, 2, 1, 3)
        let preatt = q.matmul(k) * scale
        let att = preatt.replacing(with: -Float.infinity, where: mask).softmax()
        let out = att.matmul(v)
            .transposed(permutation: 0, 2, 1, 3)
            .reshaped(to: [B, T, C])
        return AttentionForwardResult(out: out, preatt: preatt, att: att)
    }

    private static let geluScalingFactor = (2 / Float.pi).squareRoot()

    static func gelu_forward(inp: MLTensor) -> MLTensor {
        let inpPlusCubeTerm = inp + inp.pow(3) * 0.044715
        return inp * 0.5 * ((geluScalingFactor * inpPlusCubeTerm).tanh() + 1)
    }

    static func residual_forward(inp1: MLTensor, inp2: MLTensor) -> MLTensor {
        inp1 + inp2
    }

    static func softmax_forward(logits: MLTensor) -> MLTensor {
        logits.softmax()
    }

    static func crossentropy_forward(probs: MLTensor, targets: MLTensor, context: ShapeContext, B: Int, T: Int) -> MLTensor {
        let indices = context.crossEntropyBatchIndices
            .concatenated(with: context.crossEntropyTimeIndices, alongAxis: 2)
            .concatenated(with: targets.reshaped(to: [B, T]).expandingShape(at: 2), alongAxis: 2)
        return -probs.gathering(atIndices: indices).log()
    }

    // MARK: - Layer operations (backward)

    static func crossentropy_softmax_backward(dlosses: MLTensor, probs: MLTensor, targets: MLTensor, context: ShapeContext, B: Int, T: Int, V: Int) -> MLTensor {
        // Equivalent to `(probs - oneHot) * dlosses` but skips materialising
        // a dense [B, T, V] one-hot float tensor. We compute `probs * dlosses`
        // everywhere, then splice in `(probs * dlosses) - dlosses` (i.e.
        // `(probs - 1) * dlosses`) at the target column of each (b, t) row,
        // selected via a broadcast index-equality mask.
        let dlossesExpanded = dlosses.expandingShape(at: 2) // [B, T, 1]
        let dlogitsBase = probs * dlossesExpanded // [B, T, V]
        let dlogitsAtTarget = dlogitsBase - dlossesExpanded // [B, T, V] (only target columns read)
        let targetIndices = targets.reshaped(to: [B, T]).expandingShape(at: 2).cast(to: Int32.self) // [B, T, 1]
        let mask = context.targetColumnIndices .== targetIndices // broadcast → [B, T, V] bool
        return dlogitsBase.replacing(with: dlogitsAtTarget, where: mask)
    }

    static func matmul_backward(dout: MLTensor, inp: MLTensor, weight: MLTensor, B: Int, T: Int, C: Int, OC: Int) -> (dinp: MLTensor, dweight: MLTensor, dbias: MLTensor) {
        let dout_reshaped = dout.reshaped(to: [B * T, OC])
        let dinp = dout_reshaped.matmul(weight).reshaped(to: [B, T, C])
        let dweight = dout_reshaped.transposed().matmul(inp.reshaped(to: [B * T, C]))
        return (dinp, dweight, dout_reshaped.sum(alongAxes: [0]).squeezingShape())
    }

    static func layernorm_backward(dout: MLTensor, inp: MLTensor, weight: MLTensor, mean: MLTensor, rstd: MLTensor) -> (dinp: MLTensor, dweight: MLTensor, dbias: MLTensor) {
        let dnorm = weight * dout
        let dnorm_mean = dnorm.mean(alongAxes: 2)
        let norm_bti = (inp - mean) * rstd
        let dnorm_norm_mean = (dnorm * norm_bti).mean(alongAxes: 2)

        let dbias = dout.sum(alongAxes: [0, 1])
        let dweight = (norm_bti * dout).sum(alongAxes: [0, 1])
        let dinp = (dnorm - dnorm_mean.expandingShape(at: 2) - (norm_bti * dnorm_norm_mean.expandingShape(at: 2))) * rstd
        return (dinp, dweight, dbias)
    }

    static func attention_backward(dout: MLTensor, att: MLTensor, inp: MLTensor, B: Int, T: Int, C: Int, NH: Int) -> MLTensor {
        let HS = C / NH
        let scale = 1 / Float(HS).squareRoot()

        // Recover q, k, v from inp (same decomposition as forward)
        let reshaped = inp.reshaped(to: [B, T, 3, NH, HS])
        let q = reshaped[0..., 0..., 0, 0..., 0...].transposed(permutation: 0, 2, 1, 3) // (B, NH, T, HS)
        let k = reshaped[0..., 0..., 1, 0..., 0...].transposed(permutation: 0, 2, 1, 3) // (B, NH, T, HS)
        let v = reshaped[0..., 0..., 2, 0..., 0...].transposed(permutation: 0, 2, 1, 3) // (B, NH, T, HS)

        // Backward pass 4: through value accumulation
        // Forward was: out_nhw = att @ v, then transposed to (B,T,NH,HS) and reshaped to (B,T,C)
        // Undo reshape+transpose to get dout in (B,NH,T,HS)
        let dout_nhtw = dout.reshaped(to: [B, T, NH, HS]).transposed(permutation: 0, 2, 1, 3)
        let datt = dout_nhtw.matmul(v.transposed(permutation: 0, 1, 3, 2)) // (B,NH,T,T)
        let dv = att.transposed(permutation: 0, 1, 3, 2).matmul(dout_nhtw) // (B,NH,T,HS)

        // Backward pass 2 & 3: softmax backward
        // dpreatt = scale * att * (datt - sum(att * datt, axis=-1))
        let dpreatt = scale * att * (datt - (att * datt).sum(alongAxes: 3, keepRank: true))

        // Backward pass 1: query @ key matmul backward
        let dq = dpreatt.matmul(k) // (B,NH,T,HS)
        let dk = dpreatt.transposed(permutation: 0, 1, 3, 2).matmul(q) // (B,NH,T,HS)

        // Transpose back to (B,T,NH,HS) and concatenate as (B,T,3*C)
        let dq_out = dq.transposed(permutation: 0, 2, 1, 3) // (B,T,NH,HS)
        let dk_out = dk.transposed(permutation: 0, 2, 1, 3) // (B,T,NH,HS)
        let dv_out = dv.transposed(permutation: 0, 2, 1, 3) // (B,T,NH,HS)
        return dq_out.concatenated(with: dk_out, alongAxis: 2).concatenated(with: dv_out, alongAxis: 2).reshaped(to: [B, T, 3 * C])
    }

    static func gelu_backward(inp: MLTensor, dout: MLTensor) -> MLTensor {
        let tanh_arg = geluScalingFactor * (inp + 0.044715 * inp * inp * inp)
        let tanh_out = tanh_arg.tanh()
        let cosh_out = tanh_arg.cosh()
        let sech_out = 1 / (cosh_out * cosh_out)
        let local_grad = 0.5 * (1 + tanh_out) + inp * 0.5 * sech_out * geluScalingFactor * (1 + 3 * 0.044715 * inp * inp)
        return local_grad * dout
    }

    static func residual_backward(dout: MLTensor) -> (dinp1: MLTensor, dinp2: MLTensor) {
        (dout, dout)
    }

    static func encoder_backward(dout: MLTensor, inp: MLTensor, dwte: MLTensor, B: Int, T: Int, maxT: Int, V: Int, C: Int) -> (dwte: MLTensor, dwpe: MLTensor) {
        // dwte: scatter-add dout rows into token embedding gradient
        // oneHot is (B*T, V), dout_flat is (B*T, C) → oneHot^T @ dout_flat = (V, C)
        let indices = inp.reshaped(to: [B * T]).cast(to: Int32.self)
        let oneHot = MLTensor(zeros: [B * T, V], scalarType: Float.self)
            .replacing(atIndices: indices.expandingShape(at: 1), with: Float(1), alongAxis: 1)
            .squeezingShape()
        let dout_flat = dout.reshaped(to: [B * T, C])
        let dwte_update = oneHot.transposed().matmul(dout_flat) // (V, C)
        let dwte_result = dwte + dwte_update

        // dwpe is (T, C) from the sum but must be (maxT, C) to match the parameter shape
        let dwpe_t = dout.sum(alongAxes: 0) // (T, C)
        let padding = maxT - T
        let dwpe: MLTensor
        if padding > 0 {
            dwpe = dwpe_t.concatenated(with: MLTensor(zeros: [padding, C], scalarType: Float.self), alongAxis: 0)
        } else {
            dwpe = dwpe_t
        }
        return (dwte_result, dwpe)
    }

    // MARK: - AdamW weight update

    // Synchronous: every operation here just composes MLTensor graph nodes; nothing
    // forces materialization. Marking it `async` previously caused 150+ tiny actor
    // hops per step, fragmenting what should be a single fused graph build.
    static func adamw_update(params: inout MLParameterTensors, grads: MLParameterTensors, m: inout MLParameterTensors, v: inout MLParameterTensors, learningRate: Float, beta1: Float, beta2: Float, eps: Float, weightDecay: Float, t: Int) {
        let beta1CorrectionFactor = 1 / (1 - pow(beta1, Float(t)))
        let beta2CorrectionFactor = 1 / (1 - pow(beta2, Float(t)))
        let oneMinusBeta1 = 1 - beta1
        let oneMinusBeta2 = 1 - beta2
        let mScale = learningRate * beta1CorrectionFactor
        let hasWeightDecay = weightDecay != 0
        let weightDecayScale = learningRate * weightDecay

        func updateField(param: inout MLTensor, grad: MLTensor, mField: inout MLTensor, vField: inout MLTensor) {
            mField = beta1 * mField + oneMinusBeta1 * grad
            vField = beta2 * vField + oneMinusBeta2 * grad.squared()
            let adamStep = (mField * mScale) * (vField * beta2CorrectionFactor + eps).rsqrt()
            if hasWeightDecay {
                param = param - adamStep - weightDecayScale * param
            } else {
                param = param - adamStep
            }
        }

        updateField(param: &params.wte, grad: grads.wte, mField: &m.wte, vField: &v.wte)
        updateField(param: &params.wpe, grad: grads.wpe, mField: &m.wpe, vField: &v.wpe)
        for l in 0..<params.ln1w.count {
            updateField(param: &params.ln1w[l], grad: grads.ln1w[l], mField: &m.ln1w[l], vField: &v.ln1w[l])
            updateField(param: &params.ln1b[l], grad: grads.ln1b[l], mField: &m.ln1b[l], vField: &v.ln1b[l])
            updateField(param: &params.qkvw[l], grad: grads.qkvw[l], mField: &m.qkvw[l], vField: &v.qkvw[l])
            updateField(param: &params.qkvb[l], grad: grads.qkvb[l], mField: &m.qkvb[l], vField: &v.qkvb[l])
            updateField(param: &params.attprojw[l], grad: grads.attprojw[l], mField: &m.attprojw[l], vField: &v.attprojw[l])
            updateField(param: &params.attprojb[l], grad: grads.attprojb[l], mField: &m.attprojb[l], vField: &v.attprojb[l])
            updateField(param: &params.ln2w[l], grad: grads.ln2w[l], mField: &m.ln2w[l], vField: &v.ln2w[l])
            updateField(param: &params.ln2b[l], grad: grads.ln2b[l], mField: &m.ln2b[l], vField: &v.ln2b[l])
            updateField(param: &params.fcw[l], grad: grads.fcw[l], mField: &m.fcw[l], vField: &v.fcw[l])
            updateField(param: &params.fcb[l], grad: grads.fcb[l], mField: &m.fcb[l], vField: &v.fcb[l])
            updateField(param: &params.fcprojw[l], grad: grads.fcprojw[l], mField: &m.fcprojw[l], vField: &v.fcprojw[l])
            updateField(param: &params.fcprojb[l], grad: grads.fcprojb[l], mField: &m.fcprojb[l], vField: &v.fcprojb[l])
        }
        updateField(param: &params.lnfw, grad: grads.lnfw, mField: &m.lnfw, vField: &v.lnfw)
        updateField(param: &params.lnfb, grad: grads.lnfb, mField: &m.lnfb, vField: &v.lnfb)
    }
}

// MARK: - GPT2MLTensor model

final class GPT2MLTensor: @unchecked Sendable {
    let config: LLMGPT2Config
    var params: LLMMLTensor.MLParameterTensors
    var m_memory: LLMMLTensor.MLParameterTensors
    var v_memory: LLMMLTensor.MLParameterTensors
    private var shapeContexts: [LLMMLTensor.ShapeKey: LLMMLTensor.ShapeContext] = [:]

    private static func mlTensor(_ values: [Float], offset: inout Int, shape: [Int]) -> MLTensor {
        let scalarCount = shape.reduce(1, *)
        let tensor = MLTensor(shape: shape, scalars: values[offset..<(offset + scalarCount)])
        offset += scalarCount
        return tensor
    }

    init(config: LLMGPT2Config, parameters: [Float]) {
        self.config = config
        let (maxT, V, Vp, L, _, C) = (config.max_seq_len, config.vocab_size, config.padded_vocab_size, config.num_layers, config.num_heads, config.channels)
        var offset = 0
        self.params = LLMMLTensor.MLParameterTensors(
            wte: Self.mlTensor(parameters, offset: &offset, shape: [Vp, C])[0..<V, 0..<C],
            wpe: Self.mlTensor(parameters, offset: &offset, shape: [maxT, C]),
            ln1w: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [C]) },
            ln1b: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [C]) },
            qkvw: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [3 * C, C]) },
            qkvb: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [3 * C]) },
            attprojw: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [C, C]) },
            attprojb: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [C]) },
            ln2w: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [C]) },
            ln2b: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [C]) },
            fcw: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [4 * C, C]) },
            fcb: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [4 * C]) },
            fcprojw: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [C, 4 * C]) },
            fcprojb: (0..<L).map { _ in Self.mlTensor(parameters, offset: &offset, shape: [C]) },
            lnfw: Self.mlTensor(parameters, offset: &offset, shape: [C]),
            lnfb: Self.mlTensor(parameters, offset: &offset, shape: [C])
        )
        self.m_memory = LLMMLTensor.MLParameterTensors.zeros(config: config)
        self.v_memory = LLMMLTensor.MLParameterTensors.zeros(config: config)
    }

    private func shapeContext(batchSize: Int, sequenceLength: Int) -> LLMMLTensor.ShapeContext {
        let key = LLMMLTensor.ShapeKey(batchSize: batchSize, sequenceLength: sequenceLength)
        if let cached = shapeContexts[key] {
            return cached
        }
        let context = LLMMLTensor.ShapeContext(batchSize: batchSize, sequenceLength: sequenceLength, vocabSize: config.vocab_size)
        shapeContexts[key] = context
        return context
    }

    func forward(inputs: [UInt32], targets: [UInt32], B: Int, T: Int) -> (acts: LLMMLTensor.MLActivationTensors, meanLoss: Float?) {
        let V = config.vocab_size
        let C = config.channels
        let NH = config.num_heads
        let HS = C / NH
        let scale = Float(1) / Float(HS).squareRoot()
        let context = shapeContext(batchSize: B, sequenceLength: T)

        var acts = LLMMLTensor.MLActivationTensors(config: config)
        let inp = MLTensor(inputs)

        acts.encoded = LLMMLTensor.encoder_forward(inp: inp, wte: params.wte, wpe: params.wpe, B: B, T: T, C: C, V: V)

        for l in 0..<config.num_layers {
            let residual = l == 0 ? acts.encoded : acts.residual3[l - 1]
            (acts.ln1[l], acts.ln1_mean[l], acts.ln1_rstd[l]) = LLMMLTensor.layernorm_forward(inp: residual, weight: params.ln1w[l], bias: params.ln1b[l])
            acts.qkv[l] = LLMMLTensor.matmul_forward(inp: acts.ln1[l], weight: params.qkvw[l], bias: params.qkvb[l])
            let qkv = acts.qkv[l].reshaped(to: [B, T, 3, NH, HS])
            let attention = LLMMLTensor.attention_forward(inp: qkv, mask: context.attentionMask, B: B, T: T, C: C, NH: NH, HS: HS, scale: scale)
            acts.atty[l] = attention.out
            acts.att[l] = attention.att
            acts.attproj[l] = LLMMLTensor.matmul_forward(inp: acts.atty[l], weight: params.attprojw[l], bias: params.attprojb[l])
            acts.residual2[l] = LLMMLTensor.residual_forward(inp1: residual, inp2: acts.attproj[l])
            (acts.ln2[l], acts.ln2_mean[l], acts.ln2_rstd[l]) = LLMMLTensor.layernorm_forward(inp: acts.residual2[l], weight: params.ln2w[l], bias: params.ln2b[l])
            acts.fch[l] = LLMMLTensor.matmul_forward(inp: acts.ln2[l], weight: params.fcw[l], bias: params.fcb[l])
            acts.fch_gelu[l] = LLMMLTensor.gelu_forward(inp: acts.fch[l])
            acts.fcproj[l] = LLMMLTensor.matmul_forward(inp: acts.fch_gelu[l], weight: params.fcprojw[l], bias: params.fcprojb[l])
            acts.residual3[l] = LLMMLTensor.residual_forward(inp1: acts.residual2[l], inp2: acts.fcproj[l])
        }

        let L = config.num_layers
        (acts.lnf, acts.lnf_mean, acts.lnf_rstd) = LLMMLTensor.layernorm_forward(inp: acts.residual3[L - 1], weight: params.lnfw, bias: params.lnfb)
        acts.logits = LLMMLTensor.matmul_forward(inp: acts.lnf, weight: params.wte, bias: nil)
        acts.probs = LLMMLTensor.softmax_forward(logits: acts.logits)

        if targets.isEmpty {
            return (acts, nil)
        }

        let tgt = MLTensor(targets)
        acts.losses = LLMMLTensor.crossentropy_forward(probs: acts.probs, targets: tgt, context: context, B: B, T: T)
        return (acts, nil) // meanLoss computed asynchronously by caller
    }

    func backwardSync(acts: LLMMLTensor.MLActivationTensors, inputs: [UInt32], targets: [UInt32], B: Int, T: Int) -> LLMMLTensor.MLParameterTensors {
        let V = config.vocab_size
        let L = config.num_layers
        let NH = config.num_heads
        let C = config.channels
        let context = shapeContext(batchSize: B, sequenceLength: T)

        var grads = LLMMLTensor.MLParameterTensors.zeros(config: config)

        let tgt = MLTensor(targets)
        let dlogits = LLMMLTensor.crossentropy_softmax_backward(dlosses: context.meanLossScale, probs: acts.probs, targets: tgt, context: context, B: B, T: T, V: V)

        var dlnf: MLTensor
        (dlnf, grads.wte, _) = LLMMLTensor.matmul_backward(dout: dlogits, inp: acts.lnf, weight: params.wte, B: B, T: T, C: C, OC: V)

        var dresidual: MLTensor
        (dresidual, grads.lnfw, grads.lnfb) = LLMMLTensor.layernorm_backward(dout: dlnf, inp: acts.residual3[L - 1], weight: params.lnfw, mean: acts.lnf_mean, rstd: acts.lnf_rstd)

        for l in stride(from: L - 1, through: 0, by: -1) {
            let (dresidual2, dfcproj) = LLMMLTensor.residual_backward(dout: dresidual)
            var dfch_gelu: MLTensor
            (dfch_gelu, grads.fcprojw[l], grads.fcprojb[l]) = LLMMLTensor.matmul_backward(dout: dfcproj, inp: acts.fch_gelu[l], weight: params.fcprojw[l], B: B, T: T, C: 4 * C, OC: C)
            let dfch = LLMMLTensor.gelu_backward(inp: acts.fch[l], dout: dfch_gelu)
            var dln2: MLTensor
            (dln2, grads.fcw[l], grads.fcb[l]) = LLMMLTensor.matmul_backward(dout: dfch, inp: acts.ln2[l], weight: params.fcw[l], B: B, T: T, C: C, OC: 4 * C)
            var dresidual2_accum: MLTensor
            (dresidual2_accum, grads.ln2w[l], grads.ln2b[l]) = LLMMLTensor.layernorm_backward(dout: dln2, inp: acts.residual2[l], weight: params.ln2w[l], mean: acts.ln2_mean[l], rstd: acts.ln2_rstd[l])
            dresidual2_accum = dresidual2_accum + dresidual2

            let (dresidual_prev, dattproj) = LLMMLTensor.residual_backward(dout: dresidual2_accum)
            var datty: MLTensor
            (datty, grads.attprojw[l], grads.attprojb[l]) = LLMMLTensor.matmul_backward(dout: dattproj, inp: acts.atty[l], weight: params.attprojw[l], B: B, T: T, C: C, OC: C)
            let dqkv = LLMMLTensor.attention_backward(dout: datty, att: acts.att[l], inp: acts.qkv[l], B: B, T: T, C: C, NH: NH)
            var dln1: MLTensor
            let residual_inp = l == 0 ? acts.encoded : acts.residual3[l - 1]
            (dln1, grads.qkvw[l], grads.qkvb[l]) = LLMMLTensor.matmul_backward(dout: dqkv, inp: acts.ln1[l], weight: params.qkvw[l], B: B, T: T, C: C, OC: 3 * C)
            var dresidual_ln1: MLTensor
            (dresidual_ln1, grads.ln1w[l], grads.ln1b[l]) = LLMMLTensor.layernorm_backward(dout: dln1, inp: residual_inp, weight: params.ln1w[l], mean: acts.ln1_mean[l], rstd: acts.ln1_rstd[l])
            dresidual = dresidual_ln1 + dresidual_prev
        }

        let inp = MLTensor(inputs)
        (grads.wte, grads.wpe) = LLMMLTensor.encoder_backward(dout: dresidual, inp: inp, dwte: grads.wte, B: B, T: T, maxT: config.max_seq_len, V: V, C: C)

        return grads
    }

    func backward(acts: LLMMLTensor.MLActivationTensors, inputs: [UInt32], targets: [UInt32], B: Int, T: Int) async -> LLMMLTensor.MLParameterTensors {
        backwardSync(acts: acts, inputs: inputs, targets: targets, B: B, T: T)
    }

    func performTrainingStep(inputs: [UInt32], targets: [UInt32], B: Int, T: Int, learningRate: Float, step: Int) async -> Float {
        let (acts, _) = forward(inputs: inputs, targets: targets, B: B, T: T)
        let meanLoss = await acts.losses.mean().shapedArray(of: Float.self).scalar ?? 0
        let grads = backwardSync(acts: acts, inputs: inputs, targets: targets, B: B, T: T)
        LLMMLTensor.adamw_update(
            params: &params, grads: grads, m: &m_memory, v: &v_memory,
            learningRate: learningRate, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0, t: step
        )
        return meanLoss
    }

    func performInference(inputs: [UInt32], B: Int, T: Int) -> MLTensor {
        let (acts, _) = forward(inputs: inputs, targets: [], B: B, T: T)
        return acts.logits
    }

    func lossTensor(inputs: [UInt32], targets: [UInt32], B: Int, T: Int) -> MLTensor {
        let (acts, _) = forward(inputs: inputs, targets: targets, B: B, T: T)
        return acts.losses
    }

    func performLossEstimation(inputs: [UInt32], targets: [UInt32], B: Int, T: Int) async -> Float {
        return await lossTensor(inputs: inputs, targets: targets, B: B, T: T).mean().shapedArray(of: Float.self).scalar ?? 0
    }

    func exportCheckpoint() async throws -> Data {
        let V = config.vocab_size
        let Vp = config.padded_vocab_size
        let C = config.channels

        var allParams = [Float]()
        allParams.reserveCapacity(config.num_parameters)

        // wte needs padding from V to Vp
        let wteScalars = await params.wte.shapedArray(of: Float.self).scalars
        var paddedWte = [Float](repeating: 0, count: Vp * C)
        for row in 0..<V {
            paddedWte.replaceSubrange((row * C)..<(row * C + C), with: wteScalars[(row * C)..<(row * C + C)])
        }
        allParams.append(contentsOf: paddedWte)

        // All other parameters in checkpoint order (grouped by type, not by layer)
        var tensors: [MLTensor] = [params.wpe]
        tensors.append(contentsOf: params.ln1w)
        tensors.append(contentsOf: params.ln1b)
        tensors.append(contentsOf: params.qkvw)
        tensors.append(contentsOf: params.qkvb)
        tensors.append(contentsOf: params.attprojw)
        tensors.append(contentsOf: params.attprojb)
        tensors.append(contentsOf: params.ln2w)
        tensors.append(contentsOf: params.ln2b)
        tensors.append(contentsOf: params.fcw)
        tensors.append(contentsOf: params.fcb)
        tensors.append(contentsOf: params.fcprojw)
        tensors.append(contentsOf: params.fcprojb)
        tensors.append(contentsOf: [params.lnfw, params.lnfb])

        for tensor in tensors {
            let scalars = await tensor.shapedArray(of: Float.self).scalars
            allParams.append(contentsOf: scalars)
        }

        return try LLMCheckpointCodec.encode(
            header: config.checkpointHeader,
            parameters: allParams,
            expectedParameterCount: config.num_parameters
        )
    }
}
