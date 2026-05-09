import Foundation
import Numerics

enum LLMSwift {
    typealias GPT2Config = LLMGPT2Config
    typealias ParameterTensors = LLMGPT2ParameterTensors
    typealias ActivationTensors = LLMGPT2ActivationTensors
    
    static let gelu_scaling_factor = (2 / Float.pi).squareRoot()

    static func encoder_forward(out: inout [Float], inp: [UInt32], wte: [Float], wpe: [Float], B: Int, T: Int, C: Int) {
        var out = out.mutableSpan

        // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        // inp is (B,T) of integers, holding the token ids at each (b,t) position
        // wte is (V,C) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
        for b in 0..<B {
            for t in 0..<T {
                // seek to the output position in out[b,t,:]
                let out_bt = b * T * C + t * C
                // get the index of the token at inp[b, t]
                let ix = Int(inp[b * T + t])
                // seek to the position in wte corresponding to the token
                let wte_ix = ix * C
                // seek to the position in wpe corresponding to the position
                let wpe_t = t * C
                // add the two vectors and store the result in out[b,t,:]
                for i in 0..<C {
                    out[out_bt + i] = Relaxed.sum(wte[wte_ix + i], wpe[wpe_t + i])
                }
            }
        }
    }

    static func encoder_backward(dwte: inout [Float], dwpe: inout [Float], dout: [Float], inp: [UInt32], B: Int, T: Int, C: Int) {
        var dwte = dwte.mutableSpan
        var dwpe = dwpe.mutableSpan

        for b in 0..<B {
            for t in 0..<T {
                let dout_bt = b * T * C + t * C
                let ix = Int(inp[b * T + t])
                let dwte_ix = ix * C
                let dwpe_t = t * C
                for i in 0..<C {
                    let d = dout[dout_bt + i]
                    dwte[dwte_ix + i] += d
                    dwpe[dwpe_t + i] += d
                }
            }
        }
    }

    static func layernorm_forward(out: inout [Float], mean: inout [Float], rstd: inout [Float], inp: [Float], weight: [Float], bias: [Float], B: Int, T: Int, C: Int) {
        var out = out.mutableSpan
        var mean = mean.mutableSpan
        var rstd = rstd.mutableSpan

        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both inp and out are (B,T,C) of the activations
        // mean and rstd are (B,T) buffers, to be used later in backward pass
        // at each position (b,t) of the input, the C-dimensional vector
        // of activations gets normalized, then scaled and shifted
        let eps: Float = 1e-5
        for b in 0..<B {
            for t in 0..<T {
                // seek to the input position inp[b,t,:]
                let x = b * T * C + t * C
                // calculate the mean
                var m: Float = 0
                for i in 0..<C {
                    m = Relaxed.sum(m, inp[x + i])
                }
                m /= Float(C)

                // calculate the variance (without any bias correction)
                var v: Float = 0
                for i in 0..<C {
                    let xshift = inp[x + i] - m
                    v = Relaxed.multiplyAdd(xshift, xshift, v)
                }
                v /= Float(C)

                // calculate the rstd (reciprocal standard deviation)
                let s = 1 / (v + eps).squareRoot()
                // seek to the output position in out[b,t,:]
                for i in 0..<C {
                    let n = s * (inp[x + i] - m) // normalize
                    let o = Relaxed.multiplyAdd(n, weight[i], bias[i]) // scale and shift
                    out[x + i] = o // write
                }
                // cache the mean and rstd for the backward pass later
                mean[b * T + t] = m
                rstd[b * T + t] = s
            }
        }
    }

    static func layernorm_backward(dinp: inout [Float], dweight: inout [Float], dbias: inout [Float], dout: [Float], inp: [Float], weight: [Float], mean: [Float], rstd: [Float], B: Int, T: Int, C: Int) {
        var dinp = dinp.mutableSpan
        var dweight = dweight.mutableSpan
        var dbias = dbias.mutableSpan

        for b in 0..<B {
            for t in 0..<T {
                let dout_bt = b * T * C + t * C
                let inp_bt = b * T * C + t * C
                let dinp_bt = b * T * C + t * C
                let mean_bt = mean[b * T + t]
                let rstd_bt = rstd[b * T + t]

                // first: two reduce operations
                var dnorm_mean: Float = 0
                var dnorm_norm_mean: Float = 0
                for i in 0..<C {
                    let norm_bti = (inp[inp_bt + i] - mean_bt) * rstd_bt
                    let dnorm_i = weight[i] * dout[dout_bt + i]
                    dnorm_mean = Relaxed.sum(dnorm_mean, dnorm_i)
                    dnorm_norm_mean = Relaxed.multiplyAdd(dnorm_i, norm_bti, dnorm_norm_mean)
                }
                dnorm_mean /= Float(C)
                dnorm_norm_mean /= Float(C)

                // now iterate again and accumulate all the gradients
                for i in 0..<C {
                    let norm_bti = (inp[inp_bt + i] - mean_bt) * rstd_bt
                    let dnorm_i = weight[i] * dout[dout_bt + i]
                    // gradient contribution to bias
                    dbias[i] = Relaxed.sum(dbias[i], dout[dout_bt + i])
                    // gradient contribution to weight
                    dweight[i] = Relaxed.multiplyAdd(norm_bti, dout[dout_bt + i], dweight[i])
                    // gradient contribution to input
                    var dval: Float = 0
                    dval = Relaxed.sum(dval, dnorm_i) // term 1
                    dval -= dnorm_mean // term 2
                    dval -= norm_bti * dnorm_norm_mean // term 3
                    dval *= rstd_bt // final scale
                    dinp[dinp_bt + i] = Relaxed.sum(dinp[dinp_bt + i], dval)
                }
            }
        }
    }

    static func matmul_forward_naive(out: inout [Float], inp: [Float], weight: [Float], bias: [Float]?, B: Int, T: Int, C: Int, OC: Int) {
        var out = out.mutableSpan

        // OC is short for "output channels"
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        // out will be (B,T,OC)
        for b in 0..<B {
            for t in 0..<T {
                let bt = b * T + t
                for o in 0..<OC {
                    var val = bias?[o] ?? 0
                    for i in 0..<C {
                        val = Relaxed.multiplyAdd(inp[bt * C + i], weight[o * C + i], val)
                    }
                    out[bt * OC + o] = val
                }
            }
        }
    }

    static func matmul_forward(out: inout [Float], inp: [Float], weight: [Float], bias: [Float]?, B: Int, T: Int, C: Int, OC: Int) {
        // Tile over batch rows (B*T) so each weight element is loaded once and reused across
        // LOOP_UNROLL independent output rows. This mirrors the hot path in train_gpt2.c:
        // one pass over inp/weight per output channel, with four partial sums kept in registers.
        let LOOP_UNROLL = 8
        let BT = B * T
        if BT % LOOP_UNROLL != 0 {
            matmul_forward_naive(out: &out, inp: inp, weight: weight, bias: bias, B: B, T: T, C: C, OC: OC)
            return
        }

        var out = out.mutableSpan

        // OC is short for "output channels"
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        // out will be (B,T,OC)
        for obt in stride(from: 0, to: BT, by: LOOP_UNROLL) {
            for o in 0..<OC {
                var result = InlineArray<8, Float>(repeating: bias?[o] ?? 0)
                let bt = inp.span.extracting(droppingFirst: obt * C)
                let w = weight.span.extracting(droppingFirst: o * C)

                for i in 0..<C {
                    for r in result.indices {
                        result[r] = Relaxed.multiplyAdd(bt[r * C + i], w[i], result[r])
                    }
                }

                for r in result.indices {
                    out[(obt + r) * OC + o] = result[r]
                }
            }
        }
    }

    static func matmul_backward(dinp: inout [Float], dweight: inout [Float], dbias: inout [Float], dout: [Float], inp: [Float], weight: [Float], B: Int, T: Int, C: Int, OC: Int) {
        var dinp = dinp.mutableSpan
        var dweight = dweight.mutableSpan
        var dbias = dbias.mutableSpan

        // backward into inp first
        for b in 0..<B {
            for t in 0..<T {
                let dout_bt = b * T * OC + t * OC
                let dinp_bt = b * T * C + t * C
                for o in stride(from: 0, to: OC - 1, by: 2) {
                    let wrow0 = o * C
                    let wrow1 = (o + 1) * C
                    let d0 = dout[dout_bt + o]
                    let d1 = dout[dout_bt + o + 1]
                    for i in 0..<C {
                        dinp[dinp_bt + i] = Relaxed.multiplyAdd(weight[wrow0 + i], d0, dinp[dinp_bt + i])
                        dinp[dinp_bt + i] = Relaxed.multiplyAdd(weight[wrow1 + i], d1, dinp[dinp_bt + i])
                    }
                }
                if OC & 1 == 1 {
                    let wrow = (OC - 1) * C
                    let d = dout[dout_bt + (OC - 1)]
                    for i in 0..<C {
                        dinp[dinp_bt + i] = Relaxed.multiplyAdd(weight[wrow + i], d, dinp[dinp_bt + i])
                    }
                }
            }
        }

        // backward into weight/bias
        for o in 0..<OC {
            for b in 0..<B {
                for t in 0..<T {
                    let dout_bt = b * T * OC + t * OC
                    let inp_bt = b * T * C + t * C
                    let dwrow = o * C
                    let d = dout[dout_bt + o]
                    if !dbias.isEmpty {
                        dbias[o] = Relaxed.sum(dbias[o], d)
                    }
                    for i in 0..<C {
                        dweight[dwrow + i] = Relaxed.multiplyAdd(inp[inp_bt + i], d, dweight[dwrow + i])
                    }
                }
            }
        }
    }

    static func attention_forward(out: inout [Float], preatt: inout [Float], att: inout [Float], inp: [Float], B: Int, T: Int, C: Int, NH: Int) {
        var out = out.mutableSpan
        var preatt = preatt.mutableSpan
        var att = att.mutableSpan

        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        let C3 = C * 3
        let hs = C / NH // head size
        let scale = 1 / Float(hs).squareRoot()

        for b in 0..<B {
            for t in 0..<T {
                for h in 0..<NH {
                    let query_t = b * T * C3 + t * C3 + h * hs
                    let preatt_bth = b * NH * T * T + h * T * T + t * T
                    let att_bth = b * NH * T * T + h * T * T + t * T

                    // pass 1: calculate query dot key and maxval
                    var maxval: Float = -10000
                    for t2 in 0...t {
                        let key_t2 = b * T * C3 + t2 * C3 + h * hs + C // +C because it's key
                        // (query_t) dot (key_t2)
                        var val: Float = 0
                        for i in 0..<hs {
                            val = Relaxed.multiplyAdd(inp[query_t + i], inp[key_t2 + i], val)
                        }
                        val *= scale
                        maxval = max(maxval, val)
                        preatt[preatt_bth + t2] = val
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    var expsum: Float = 0
                    for t2 in 0...t {
                        let expv = exp(preatt[preatt_bth + t2] - maxval)
                        expsum = Relaxed.sum(expsum, expv)
                        att[att_bth + t2] = expv
                    }
                    let expsum_inv = expsum == 0 ? 0 : 1 / expsum

                    // pass 3: normalize to get the softmax
                    for t2 in 0..<T {
                        if t2 <= t {
                            att[att_bth + t2] *= expsum_inv
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att[att_bth + t2] = 0
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    let out_bth = b * T * C + t * C + h * hs
                    for i in 0..<hs {
                        out[out_bth + i] = 0
                    }
                    for t2 in 0...t {
                        let value_t2 = b * T * C3 + t2 * C3 + h * hs + 2 * C // +C*2 because it's value
                        let att_btht2 = att[att_bth + t2]
                        for i in 0..<hs {
                            out[out_bth + i] = Relaxed.multiplyAdd(att_btht2, inp[value_t2 + i], out[out_bth + i])
                        }
                    }
                }
            }
        }
    }

    static func attention_backward(dinp: inout [Float], dpreatt: inout [Float], datt: inout [Float], dout: [Float], inp: [Float], att: [Float], B: Int, T: Int, C: Int, NH: Int) {
        var dinp = dinp.mutableSpan
        var dpreatt = dpreatt.mutableSpan
        var datt = datt.mutableSpan

        // inp/dinp are (B, T, 3C) Q,K,V
        // att/datt/dpreatt are (B, NH, T, T)
        // dout is (B, T, C)
        let C3 = C * 3
        let hs = C / NH // head size
        let scale = 1 / Float(hs).squareRoot()

        for b in 0..<B {
            for t in 0..<T {
                for h in 0..<NH {
                    let att_bth = b * NH * T * T + h * T * T + t * T
                    let datt_bth = b * NH * T * T + h * T * T + t * T
                    let dpreatt_bth = b * NH * T * T + h * T * T + t * T
                    let dquery_t = b * T * C3 + t * C3 + h * hs
                    let query_t = b * T * C3 + t * C3 + h * hs
                    let dout_bth = b * T * C + t * C + h * hs

                    // backward pass 4, through the value accumulation
                    for t2 in 0...t {
                        let value_t2 = b * T * C3 + t2 * C3 + h * hs + 2 * C // +C*2 because it's value
                        let dvalue_t2 = b * T * C3 + t2 * C3 + h * hs + 2 * C
                        for i in 0..<hs {
                            // in the forward pass this was:
                            // out_bth[i] += att_bth[t2] * value_t2[i];
                            // so now we have:
                            datt[datt_bth + t2] = Relaxed.multiplyAdd(inp[value_t2 + i], dout[dout_bth + i], datt[datt_bth + t2])
                            dinp[dvalue_t2 + i] = Relaxed.multiplyAdd(att[att_bth + t2], dout[dout_bth + i], dinp[dvalue_t2 + i])
                        }
                    }

                    // backward pass 2 & 3, the softmax
                    // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                    for t2 in 0...t {
                        for t3 in 0...t {
                            let indicator: Float = t2 == t3 ? 1 : 0
                            let local_derivative = att[att_bth + t2] * (indicator - att[att_bth + t3])
                            dpreatt[dpreatt_bth + t3] = Relaxed.multiplyAdd(local_derivative, datt[datt_bth + t2], dpreatt[dpreatt_bth + t3])
                        }
                    }

                    // backward pass 1, the query @ key matmul
                    for t2 in 0...t {
                        let key_t2 = b * T * C3 + t2 * C3 + h * hs + C // +C because it's key
                        let dkey_t2 = b * T * C3 + t2 * C3 + h * hs + C // +C because it's key
                        for i in 0..<hs {
                            // in the forward pass this was:
                            // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                            // so now we have:
                            dinp[dquery_t + i] = Relaxed.multiplyAdd(inp[key_t2 + i], dpreatt[dpreatt_bth + t2] * scale, dinp[dquery_t + i])
                            dinp[dkey_t2 + i] = Relaxed.multiplyAdd(inp[query_t + i], dpreatt[dpreatt_bth + t2] * scale, dinp[dkey_t2 + i])
                        }
                    }
                }
            }
        }
    }

    static func gelu_forward(out: inout [Float], inp: [Float], N: Int) {
        var out = out.mutableSpan

        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        for i in 0..<N {
            let x = inp[i]
            let cube = 0.044715 * x * x * x
            out[i] = 0.5 * x * (1 + tanh(gelu_scaling_factor * (x + cube)))
        }
    }

    static func gelu_backward(dinp: inout [Float], inp: [Float], dout: [Float], N: Int) {
        var dinp = dinp.mutableSpan

        for i in 0..<N {
            let x = inp[i]
            let cube = 0.044715 * x * x * x
            let tanh_arg = gelu_scaling_factor * (x + cube)
            let tanh_out = tanh(tanh_arg)
            let coshf_out = cosh(tanh_arg)
            let sech_out = 1 / (coshf_out * coshf_out)
            let local_grad = 0.5 * (1 + tanh_out) + x * 0.5 * sech_out * gelu_scaling_factor * (1 + 3 * 0.044715 * x * x)
            dinp[i] += local_grad * dout[i]
        }
    }

    static func residual_forward(out: inout [Float], inp1: [Float], inp2: [Float], N: Int) {
        var out = out.mutableSpan
        let inp1 = inp1
        let inp2 = inp2
        for i in 0..<N {
            out[i] = Relaxed.sum(inp1[i], inp2[i])
        }
    }

    static func residual_backward(dinp1: inout [Float], dinp2: inout [Float], dout: [Float], N: Int) {
        var dinp1 = dinp1.mutableSpan
        var dinp2 = dinp2.mutableSpan

        for i in 0..<N {
            dinp1[i] = Relaxed.sum(dinp1[i], dout[i])
            dinp2[i] = Relaxed.sum(dinp2[i], dout[i])
        }
    }

    static func softmax_forward(probs: inout [Float], logits: [Float], B: Int, T: Int, V: Int, Vp: Int) {
        var probs = probs.mutableSpan

        // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (B,T,Vp) of the unnormalized log probabilities
        // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
        // example: Vp is 50304 and V is 50257
        for b in 0..<B {
            for t in 0..<T {
                let logits_bt = b * T * Vp + t * Vp
                let probs_bt = b * T * Vp + t * Vp

                // probs <- softmax(logits)

                // maxval is only calculated and subtracted for numerical stability
                var maxval: Float = -10000
                for i in 0..<V {
                    maxval = max(maxval, logits[logits_bt + i])
                }
                var sum: Float = 0
                for i in 0..<V {
                    probs[probs_bt + i] = exp(logits[logits_bt + i] - maxval)
                    sum = Relaxed.sum(sum, probs[probs_bt + i])
                }
                // note we only loop to V, leaving the padded dimensions
                for i in 0..<V {
                    probs[probs_bt + i] /= sum
                }
                // for extra super safety we may wish to include this too,
                // forcing the probabilities here to be zero, but it shouldn't matter
                for i in V..<Vp {
                    probs[probs_bt + i] = 0
                }
            }
        }
    }

    static func crossentropy_forward(losses: inout [Float], probs: [Float], targets: [UInt32], B: Int, T: Int, Vp: Int) {
        var losses = losses.mutableSpan

        // output: losses is (B,T) of the individual losses at each position
        // input: probs are (B,T,Vp) of the probabilities
        // input: targets is (B,T) of integers giving the correct index in logits
        for b in 0..<B {
            for t in 0..<T {
                // loss = -log(probs[target])
                let probs_bt = b * T * Vp + t * Vp
                let ix = Int(targets[b * T + t])
                losses[b * T + t] = -log(probs[probs_bt + ix])
            }
        }
    }

    static func crossentropy_softmax_backward(dlogits: inout [Float], dlosses: [Float], probs: [Float], targets: [UInt32], B: Int, T: Int, V: Int, Vp: Int) {
        var dlogits = dlogits.mutableSpan

        // backwards through both softmax and crossentropy
        for b in 0..<B {
            for t in 0..<T {
                let dlogits_bt = b * T * Vp + t * Vp
                let probs_bt = b * T * Vp + t * Vp
                let dloss = dlosses[b * T + t]
                let ix = Int(targets[b * T + t])
                // note we only loop to V, leaving the padded dimensions
                // of dlogits untouched, so gradient there stays at zero
                for i in 0..<V {
                    let p = probs[probs_bt + i]
                    let indicator: Float = i == ix ? 1 : 0
                    dlogits[dlogits_bt + i] = Relaxed.multiplyAdd(p - indicator, dloss, dlogits[dlogits_bt + i])
                }
            }
        }
    }

    // ----------------------------------------------------------------------------
    // GPT-2 model definition

    struct GPT2: Sendable {
        let config: GPT2Config
        var params: ParameterTensors
        var grads: ParameterTensors
        var m_memory: ParameterTensors
        var v_memory: ParameterTensors
        var acts = ActivationTensors()
        var grads_acts = ActivationTensors()
        var batch_size = 0
        var seq_len = 0
        var inputs: [UInt32] = []
        var targets: [UInt32] = []
        var mean_loss: Float = -1

        init(config: GPT2Config, float_params: [Float]) {
            self.config = config
            self.params = ParameterTensors(config: config, float_params: float_params)
            self.grads = ParameterTensors(config: config, float_params: Array(repeating: 0, count: config.num_parameters))
            self.m_memory = ParameterTensors(config: config, float_params: Array(repeating: 0, count: config.num_parameters))
            self.v_memory = ParameterTensors(config: config, float_params: Array(repeating: 0, count: config.num_parameters))
        }

        func exportCheckpoint() throws -> Data {
            let flattened = params.flattened()
            return try LLMCheckpointCodec.encode(header: config.checkpointHeader, parameters: flattened, expectedParameterCount: config.num_parameters)
        }
    }

    struct UpdateParams: Sendable {
        let learning_rate: Float
        let beta1: Float
        let beta2: Float
        let eps: Float
        let weight_decay: Float
        let t: Int
    }

    static func buildModel(from checkpointData: Data) throws -> GPT2 {
        let (header, parameters) = try LLMCheckpointCodec.decode(checkpointData)
        let config = GPT2Config(header: header)
        guard parameters.count == config.num_parameters else {
            throw LLMCheckpointCodecError.invalidParameterCount(expected: config.num_parameters, actual: parameters.count)
        }
        return GPT2(config: config, float_params: parameters)
    }

    static func gpt2_forward(model: inout GPT2, inputs: [UInt32], targets: [UInt32], B: Int, T: Int) {
        // targets are optional and could be empty
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

        // forward pass
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

        // also forward the cross-entropy loss function if we have the targets
        if !targets.isEmpty {
            crossentropy_forward(losses: &model.acts.losses, probs: model.acts.probs, targets: targets, B: B, T: T, Vp: Vp)
            // for convenience also evaluate the mean loss
            var mean_loss: Float = 0
            for i in 0..<(B * T) {
                mean_loss = Relaxed.sum(mean_loss, model.acts.losses[i])
            }
            mean_loss /= Float(B * T)
            model.mean_loss = mean_loss
        } else {
            // if we don't have targets, we don't have a loss
            model.mean_loss = -1
        }
    }

    static func gpt2_zero_grad(model: inout GPT2) {
        model.grads.zero()
        model.grads_acts.zero()
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

        // we kick off the chain rule by filling in dlosses with 1.0f / (B * T)
        let dloss_mean = 1 / Float(B * T)
        for i in 0..<(B * T) {
            model.grads_acts.losses[i] = dloss_mean
        }

        crossentropy_softmax_backward(dlogits: &model.grads_acts.logits, dlosses: model.grads_acts.losses, probs: model.acts.probs, targets: model.targets, B: B, T: T, V: V, Vp: Vp)
        var empty_bias: [Float] = []
        matmul_backward(dinp: &model.grads_acts.lnf, dweight: &model.grads.wte, dbias: &empty_bias, dout: model.grads_acts.logits, inp: model.acts.lnf, weight: model.params.wte, B: B, T: T, C: C, OC: Vp)
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

    static func gpt2_update_field(params: inout [Float], grads: [Float], m_memory: inout [Float], v_memory: inout [Float], update_params: UpdateParams) {
        var params = params.mutableSpan
        var m_memory = m_memory.mutableSpan
        var v_memory = v_memory.mutableSpan

        for i in 0..<params.count {
            let param = params[i]
            let grad = grads[i]
            // update the first moment (momentum)
            let m = Relaxed.multiplyAdd(update_params.beta1, m_memory[i], (1 - update_params.beta1) * grad)
            // update the second moment (RMSprop)
            let v = Relaxed.multiplyAdd(update_params.beta2, v_memory[i], (1 - update_params.beta2) * grad * grad)
            // bias-correct both moments
            let m_hat = m / (1 - pow(update_params.beta1, Float(update_params.t)))
            let v_hat = v / (1 - pow(update_params.beta2, Float(update_params.t)))
            m_memory[i] = m
            v_memory[i] = v
            // update
            params[i] -= update_params.learning_rate * (m_hat / (v_hat.squareRoot() + update_params.eps) + update_params.weight_decay * param)
        }
    }

    static func gpt2_update(model: inout GPT2, update_params: UpdateParams) {
        gpt2_update_field(params: &model.params.wte, grads: model.grads.wte, m_memory: &model.m_memory.wte, v_memory: &model.v_memory.wte, update_params: update_params)
        gpt2_update_field(params: &model.params.wpe, grads: model.grads.wpe, m_memory: &model.m_memory.wpe, v_memory: &model.v_memory.wpe, update_params: update_params)
        for l in 0..<model.config.num_layers {
            gpt2_update_field(params: &model.params.ln1w[l], grads: model.grads.ln1w[l], m_memory: &model.m_memory.ln1w[l], v_memory: &model.v_memory.ln1w[l], update_params: update_params)
            gpt2_update_field(params: &model.params.ln1b[l], grads: model.grads.ln1b[l], m_memory: &model.m_memory.ln1b[l], v_memory: &model.v_memory.ln1b[l], update_params: update_params)
            gpt2_update_field(params: &model.params.qkvw[l], grads: model.grads.qkvw[l], m_memory: &model.m_memory.qkvw[l], v_memory: &model.v_memory.qkvw[l], update_params: update_params)
            gpt2_update_field(params: &model.params.qkvb[l], grads: model.grads.qkvb[l], m_memory: &model.m_memory.qkvb[l], v_memory: &model.v_memory.qkvb[l], update_params: update_params)
            gpt2_update_field(params: &model.params.attprojw[l], grads: model.grads.attprojw[l], m_memory: &model.m_memory.attprojw[l], v_memory: &model.v_memory.attprojw[l], update_params: update_params)
            gpt2_update_field(params: &model.params.attprojb[l], grads: model.grads.attprojb[l], m_memory: &model.m_memory.attprojb[l], v_memory: &model.v_memory.attprojb[l], update_params: update_params)
            gpt2_update_field(params: &model.params.ln2w[l], grads: model.grads.ln2w[l], m_memory: &model.m_memory.ln2w[l], v_memory: &model.v_memory.ln2w[l], update_params: update_params)
            gpt2_update_field(params: &model.params.ln2b[l], grads: model.grads.ln2b[l], m_memory: &model.m_memory.ln2b[l], v_memory: &model.v_memory.ln2b[l], update_params: update_params)
            gpt2_update_field(params: &model.params.fcw[l], grads: model.grads.fcw[l], m_memory: &model.m_memory.fcw[l], v_memory: &model.v_memory.fcw[l], update_params: update_params)
            gpt2_update_field(params: &model.params.fcb[l], grads: model.grads.fcb[l], m_memory: &model.m_memory.fcb[l], v_memory: &model.v_memory.fcb[l], update_params: update_params)
            gpt2_update_field(params: &model.params.fcprojw[l], grads: model.grads.fcprojw[l], m_memory: &model.m_memory.fcprojw[l], v_memory: &model.v_memory.fcprojw[l], update_params: update_params)
            gpt2_update_field(params: &model.params.fcprojb[l], grads: model.grads.fcprojb[l], m_memory: &model.m_memory.fcprojb[l], v_memory: &model.v_memory.fcprojb[l], update_params: update_params)
        }
        gpt2_update_field(params: &model.params.lnfw, grads: model.grads.lnfw, m_memory: &model.m_memory.lnfw, v_memory: &model.v_memory.lnfw, update_params: update_params)
        gpt2_update_field(params: &model.params.lnfb, grads: model.grads.lnfb, m_memory: &model.m_memory.lnfb, v_memory: &model.v_memory.lnfb, update_params: update_params)
    }

    static func random_u32(state: inout UInt64) -> UInt32 {
        state ^= state &>> 12
        state ^= state &<< 25
        state ^= state &>> 27
        return UInt32((state &* 0x2545F4914F6CDD1D) &>> 32)
    }

    static func random_f32(state: inout UInt64) -> Float {
        Float(random_u32(state: &state) &>> 8) / 16777216.0
    }

    static func sample_mult(probabilities: ArraySlice<Float>, coin: Float) -> UInt32 {
        var cdf: Float = 0
        for (index, probability) in probabilities.enumerated() {
            cdf = Relaxed.sum(cdf, probability)
            if coin < cdf {
                return UInt32(index)
            }
        }
        return UInt32(probabilities.count - 1)
    }

    static func sample(logits: ArraySlice<Float>, temperature: Double, state: inout UInt64) -> Int {
        // C reference has only sample_mult on normalized probabilities.
        // This helper keeps the same RNG path but also handles temperature directly.
        if temperature <= 0 {
            return logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }

        let max_logit = logits.max() ?? 0
        var probabilities = logits.map { exp(Double(($0 - max_logit) / Float(temperature))) }
        let sum = probabilities.reduce(0, +)
        guard sum > 0 else {
            return 0
        }
        for index in probabilities.indices {
            probabilities[index] /= sum
        }
        let sample = Double(random_f32(state: &state))

        var cumulative = 0.0
        for (index, probability) in probabilities.enumerated() {
            cumulative += probability
            if sample <= cumulative {
                return index
            }
        }
        return max(0, probabilities.count - 1)
    }
}
