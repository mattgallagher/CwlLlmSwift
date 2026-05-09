import Foundation
import Metal

private extension GPT2Metal {
    func encoder_forward( out: MTLBuffer, inp: MTLBuffer, wte: MTLBuffer, wpe: MTLBuffer, B: Int, T: Int, C: Int, ctx: MetalCommandContext) {
        ctx.compute.compute(
            pipelines.encoderForward,
            threads: B * T * C,
            blockSize: 256,
            buffers: [out, inp, wte, wpe],
            uints: [UInt32(B), UInt32(T), UInt32(C)]
        )
        ctx.endCompute()
    }
    
    func encoder_backward(dwte: MTLBuffer, dwpe: MTLBuffer, dout: MTLBuffer, inp: MTLBuffer, B: Int, T: Int, C: Int, ctx: MetalCommandContext) {
        ctx.compute.compute(
            pipelines.encoderBackward,
            threads: B * T * C,
            blockSize: 256,
            buffers: [dwte, dwpe, dout, inp],
            uints: [UInt32(B), UInt32(T), UInt32(C)]
        )
        ctx.endCompute()
    }
    
    func layernorm_forward(out: MTLBuffer, mean: MTLBuffer, rstd: MTLBuffer, inp: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, B: Int, T: Int, C: Int, ctx: MetalCommandContext) {
        let blockSize = 512
        let N = B * T
        
        let enc = ctx.compute
        enc.setComputePipelineState(pipelines.mean)
        enc.setBuffer(mean, offset: 0, index: 0)
        enc.setBuffer(inp, offset: 0, index: 1)
        var meanN = Int32(N)
        enc.setBytes(&meanN, length: MemoryLayout<Int32>.stride, index: 2)
        var meanC = Int32(C)
        enc.setBytes(&meanC, length: MemoryLayout<Int32>.stride, index: 3)
        enc.setThreadgroupMemoryLength(blockSize * floatStride, index: 0)
        let tpg = min(blockSize, pipelines.mean.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(
            MTLSize(width: N, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        
        ctx.compute.threadgroups(
            pipelines.rstd,
            threadgroups: N,
            blockSize: blockSize,
            buffers: [rstd, inp, mean],
            uints: [UInt32(N), UInt32(C)],
            threadgroupMemoryLength: blockSize * floatStride
        )
        ctx.compute.compute(
            pipelines.normalization,
            threads: B * T * C,
            blockSize: 128,
            buffers: [out, inp, mean, rstd, weight, bias],
            uints: [UInt32(B), UInt32(T), UInt32(C)]
        )
        ctx.endCompute()
    }
    
    func layernorm_backward(dinp: MTLBuffer, dweight: MTLBuffer, dbias: MTLBuffer, dout: MTLBuffer, inp: MTLBuffer, weight: MTLBuffer, mean: MTLBuffer, rstd: MTLBuffer, B: Int, T: Int, C: Int, ctx: MetalCommandContext) {
        let blockSize = 256
        ctx.compute.threadgroups(
            pipelines.layernormBackward,
            threadgroups: B * T,
            blockSize: blockSize,
            buffers: [dinp, dweight, dbias, dout, inp, weight, mean, rstd],
            uints: [UInt32(B), UInt32(T), UInt32(C)],
            threadgroupMemoryLength: 2 * blockSize * floatStride
        )
        ctx.endCompute()
    }
    
    func matmul_forward(out: MTLBuffer, inp: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, B: Int, T: Int, C: Int, OC: Int, ctx: MetalCommandContext) {
        ctx.compute.threadgroups2D(
            pipelines.matmulForward,
            width: OC,
            height: B * T,
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1),
            buffers: [out, inp, weight, bias],
            uints: [UInt32(B * T), UInt32(C), UInt32(OC)]
        )
        ctx.endCompute()
    }
    
    func matmul_backward(dinp: MTLBuffer, dweight: MTLBuffer, dbias: MTLBuffer?, dout: MTLBuffer, inp: MTLBuffer, weight: MTLBuffer, B: Int, T: Int, C: Int, OC: Int, ctx: MetalCommandContext) {
        ctx.compute.threadgroups2D(
            pipelines.matmulBackwardDinp,
            width: C,
            height: B * T,
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1),
            buffers: [dinp, dout, weight],
            uints: [UInt32(B * T), UInt32(C), UInt32(OC)]
        )
        ctx.endCompute()
        ctx.compute.threadgroups2D(
            pipelines.matmulBackwardDweight,
            width: C,
            height: OC,
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1),
            buffers: [dweight, dout, inp],
            uints: [UInt32(B * T), UInt32(C), UInt32(OC)]
        )
        ctx.endCompute()
        if let dbias {
            ctx.compute.compute(
                pipelines.addBiasBackward,
                threads: B * T * OC,
                blockSize: 128,
                buffers: [dbias, dout],
                uints: [UInt32(OC)]
            )
            ctx.endCompute()
        }
    }
    
    func attention_forward(out: MTLBuffer, preatt: MTLBuffer, att: MTLBuffer, qkvr: MTLBuffer, v_accum: MTLBuffer, inp: MTLBuffer, B: Int, T: Int, C: Int, NH: Int, HS: Int, ctx: MetalCommandContext) {
        let totalQKV = B * NH * T * HS
        let qOffset = 0
        let kOffset = B * T * C * floatStride
        let vOffset = 2 * B * T * C * floatStride
        
        let enc = ctx.compute
        enc.setComputePipelineState(pipelines.permute)
        enc.setBuffer(qkvr, offset: qOffset, index: 0)
        enc.setBuffer(qkvr, offset: kOffset, index: 1)
        enc.setBuffer(qkvr, offset: vOffset, index: 2)
        enc.setBuffer(inp, offset: 0, index: 3)
        var b = UInt32(B), t = UInt32(T), nh = UInt32(NH), hs = UInt32(HS)
        enc.setBytes(&b, length: 4, index: 4)
        enc.setBytes(&t, length: 4, index: 5)
        enc.setBytes(&nh, length: 4, index: 6)
        enc.setBytes(&hs, length: 4, index: 7)
        let tpg = min(128, pipelines.permute.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: totalQKV, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        ctx.endCompute()
        
        attention_query_key(out: preatt, qkvr: qkvr, qOffset: qOffset, kOffset: kOffset, batchCount: B * NH, T: T, HS: HS, ctx: ctx)
        scale_forward(buffer: preatt, scale: 1.0 / Float(HS).squareRoot(), B: B, NH: NH, T: T, ctx: ctx)
        softmax_forward(out: att, inp: preatt, N: B * NH * T, C: T, ctx: ctx)
        attention_accumulate_values(out: v_accum, att: att, qkvr: qkvr, vOffset: vOffset, batchCount: B * NH, T: T, HS: HS, ctx: ctx)
        
        ctx.compute.compute(
            pipelines.unpermute,
            threads: B * T * C,
            blockSize: 128,
            buffers: [v_accum, out],
            uints: [UInt32(B), UInt32(T), UInt32(NH), UInt32(HS)]
        )
        ctx.endCompute()
    }
    
    func attention_backward(dinp: MTLBuffer, dpreatt: MTLBuffer, datt: MTLBuffer, dqkvr: MTLBuffer, dv_accum: MTLBuffer, dout: MTLBuffer, qkvr: MTLBuffer, att: MTLBuffer, B: Int, T: Int, C: Int, NH: Int, HS: Int, ctx: MetalCommandContext) {
        let qOff = 0
        let kOff = B * T * C * floatStride
        let vOff = 2 * B * T * C * floatStride
        
        ctx.compute.compute(
            pipelines.unpermuteBackward,
            threads: B * T * C,
            blockSize: 128,
            buffers: [dout, dv_accum],
            uints: [UInt32(B), UInt32(T), UInt32(NH), UInt32(HS)]
        )
        ctx.endCompute()
        
        attention_backward_datt(datt: datt, dvAccum: dv_accum, qkvr: qkvr, vOffset: vOff, batchCount: B * NH, T: T, HS: HS, ctx: ctx)
        attention_backward_dv(dv: dqkvr, dvOffset: vOff, att: att, dvAccum: dv_accum, batchCount: B * NH, T: T, HS: HS, ctx: ctx)
        
        let dscaledBuf = device.makeBuffer(length: B * NH * T * T * floatStride, options: .storageModeShared)!
        softmax_backward(dpreatt: dscaledBuf, datt: datt, att: att, N: B * NH * T, C: T, ctx: ctx)
        scale_backward(dinp: dpreatt, dout: dscaledBuf, scale: 1.0 / Float(HS).squareRoot(), B: B, NH: NH, T: T, ctx: ctx)
        
        attention_backward_dq(dq: dqkvr, dqOffset: qOff, dpreatt: dpreatt, qkvr: qkvr, kOffset: kOff, batchCount: B * NH, T: T, HS: HS, ctx: ctx)
        attention_backward_dk(dk: dqkvr, dkOffset: kOff, dpreatt: dpreatt, qkvr: qkvr, qOffset: qOff, batchCount: B * NH, T: T, HS: HS, ctx: ctx)
        
        let enc = ctx.compute
        enc.setComputePipelineState(pipelines.permuteBackward)
        enc.setBuffer(dqkvr, offset: qOff, index: 0)
        enc.setBuffer(dqkvr, offset: kOff, index: 1)
        enc.setBuffer(dqkvr, offset: vOff, index: 2)
        enc.setBuffer(dinp, offset: 0, index: 3)
        var b = UInt32(B), t = UInt32(T), nh = UInt32(NH), hs = UInt32(HS)
        enc.setBytes(&b, length: 4, index: 4)
        enc.setBytes(&t, length: 4, index: 5)
        enc.setBytes(&nh, length: 4, index: 6)
        enc.setBytes(&hs, length: 4, index: 7)
        let tpg = min(128, pipelines.permuteBackward.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: B * NH * T * HS, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        ctx.endCompute()
    }
    
    func gelu_forward(out: MTLBuffer, inp: MTLBuffer, N: Int, ctx: MetalCommandContext) {
        ctx.compute.compute(pipelines.gelu, threads: N, blockSize: 256, buffers: [out, inp])
        ctx.endCompute()
    }
    
    func gelu_backward(dinp: MTLBuffer, inp: MTLBuffer, dout: MTLBuffer, N: Int, ctx: MetalCommandContext) {
        ctx.compute.compute(pipelines.geluBackward, threads: N, blockSize: 256, buffers: [dinp, inp, dout])
        ctx.endCompute()
    }
    
    func residual_forward(out: MTLBuffer, inp1: MTLBuffer, inp2: MTLBuffer, N: Int, ctx: MetalCommandContext) {
        ctx.compute.compute(pipelines.residualForward, threads: N, blockSize: 256, buffers: [out, inp1, inp2])
        ctx.endCompute()
    }
    
    func residual_backward(dinp1: MTLBuffer, dinp2: MTLBuffer, dout: MTLBuffer, N: Int, ctx: MetalCommandContext) {
        ctx.compute.compute(pipelines.residualBackward, threads: N, blockSize: 256, buffers: [dinp1, dinp2, dout])
        ctx.endCompute()
    }
    
    func softmax_forward(out: MTLBuffer, inp: MTLBuffer, N: Int, C: Int, ctx: MetalCommandContext) {
        let enc = ctx.compute
        enc.setComputePipelineState(pipelines.softmaxForward)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBuffer(inp, offset: 0, index: 1)
        var n = Int32(N)
        enc.setBytes(&n, length: MemoryLayout<Int32>.stride, index: 2)
        var c = Int32(C)
        enc.setBytes(&c, length: MemoryLayout<Int32>.stride, index: 3)
        let tpg = min(128, pipelines.softmaxForward.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        ctx.endCompute()
    }
    
    func crossentropy_forward(losses: MTLBuffer, probs: MTLBuffer, targets: MTLBuffer, T: Int, V: Int, ctx: MetalCommandContext) {
        ctx.compute.compute(
            pipelines.crossentropyForward,
            threads: losses.length / floatStride,
            blockSize: 256,
            buffers: [losses, probs, targets],
            uints: [UInt32(T), UInt32(V)]
        )
        ctx.endCompute()
    }
    
    func crossentropy_softmax_backward(dlogits: MTLBuffer, dlosses: MTLBuffer, probs: MTLBuffer, targets: MTLBuffer, B: Int, T: Int, V: Int, ctx: MetalCommandContext) {
        ctx.compute.compute(
            pipelines.crossentropySoftmaxBackward,
            threads: B * T * V,
            blockSize: 256,
            buffers: [dlogits, dlosses, probs, targets],
            uints: [UInt32(B), UInt32(T), UInt32(V), UInt32(V)]
        )
        ctx.endCompute()
    }
    
    func scale_forward(buffer: MTLBuffer, scale: Float, B: Int, NH: Int, T: Int, ctx: MetalCommandContext) {
        let enc = ctx.compute
        enc.setComputePipelineState(pipelines.scale)
        enc.setBuffer(buffer, offset: 0, index: 0)
        var s = scale
        enc.setBytes(&s, length: floatStride, index: 1)
        var b = UInt32(B)
        enc.setBytes(&b, length: 4, index: 2)
        var nh = UInt32(NH)
        enc.setBytes(&nh, length: 4, index: 3)
        var t = UInt32(T)
        enc.setBytes(&t, length: 4, index: 4)
        let tpg = min(128, pipelines.scale.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: B * NH * T * T, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        ctx.endCompute()
    }
    
    func fill(buffer: MTLBuffer, value: Float, count: Int, ctx: MetalCommandContext) {
        let enc = ctx.compute
        enc.setComputePipelineState(pipelines.fill)
        enc.setBuffer(buffer, offset: 0, index: 0)
        var v = value
        enc.setBytes(&v, length: floatStride, index: 1)
        let tpg = min(256, pipelines.fill.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        ctx.endCompute()
    }
    
    func softmax_backward(dpreatt: MTLBuffer, datt: MTLBuffer, att: MTLBuffer, N: Int, C: Int, ctx: MetalCommandContext) {
        let enc = ctx.compute
        enc.setComputePipelineState(pipelines.softmaxBackward)
        enc.setBuffer(dpreatt, offset: 0, index: 0)
        enc.setBuffer(datt, offset: 0, index: 1)
        enc.setBuffer(att, offset: 0, index: 2)
        var n = Int32(N)
        enc.setBytes(&n, length: MemoryLayout<Int32>.stride, index: 3)
        var c = Int32(C)
        enc.setBytes(&c, length: MemoryLayout<Int32>.stride, index: 4)
        let tpg = min(128, pipelines.softmaxBackward.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        ctx.endCompute()
    }
    
    func scale_backward(dinp: MTLBuffer, dout: MTLBuffer, scale: Float, B: Int, NH: Int, T: Int, ctx: MetalCommandContext) {
        let enc = ctx.compute
        enc.setComputePipelineState(pipelines.scaleBackward)
        enc.setBuffer(dinp, offset: 0, index: 0)
        enc.setBuffer(dout, offset: 0, index: 1)
        var s = scale
        enc.setBytes(&s, length: floatStride, index: 2)
        var b = UInt32(B)
        enc.setBytes(&b, length: 4, index: 3)
        var nh = UInt32(NH)
        enc.setBytes(&nh, length: 4, index: 4)
        var t = UInt32(T)
        enc.setBytes(&t, length: 4, index: 5)
        let tpg = min(128, pipelines.scaleBackward.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: B * NH * T * T, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        ctx.endCompute()
    }
    
    func attention_query_key(out: MTLBuffer, qkvr: MTLBuffer, qOffset: Int, kOffset: Int, batchCount: Int, T: Int, HS: Int, ctx: MetalCommandContext) {
        ctx.compute.threadgroups3D(
            pipelines.attentionQK,
            width: T,
            height: T,
            depth: batchCount,
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1),
            buffers: [out, qkvr, qkvr],
            offsets: [0, qOffset, kOffset],
            uints: [UInt32(T), UInt32(HS)]
        )
        ctx.endCompute()
    }
    
    func attention_accumulate_values(out: MTLBuffer, att: MTLBuffer, qkvr: MTLBuffer, vOffset: Int, batchCount: Int, T: Int, HS: Int, ctx: MetalCommandContext) {
        ctx.compute.threadgroups3D(
            pipelines.attentionAV,
            width: T,
            height: HS,
            depth: batchCount,
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1),
            buffers: [out, att, qkvr],
            offsets: [0, 0, vOffset],
            uints: [UInt32(T), UInt32(HS)]
        )
        ctx.endCompute()
    }
    
    func attention_backward_datt(datt: MTLBuffer, dvAccum: MTLBuffer, qkvr: MTLBuffer, vOffset: Int, batchCount: Int, T: Int, HS: Int, ctx: MetalCommandContext) {
        ctx.compute.threadgroups3D(
            pipelines.attentionBackwardDatt,
            width: T,
            height: T,
            depth: batchCount,
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1),
            buffers: [datt, dvAccum, qkvr],
            offsets: [0, 0, vOffset],
            uints: [UInt32(T), UInt32(HS)]
        )
        ctx.endCompute()
    }
    
    func attention_backward_dv(dv: MTLBuffer, dvOffset: Int, att: MTLBuffer, dvAccum: MTLBuffer, batchCount: Int, T: Int, HS: Int, ctx: MetalCommandContext) {
        ctx.compute.threadgroups3D(
            pipelines.attentionBackwardDV,
            width: HS,
            height: T,
            depth: batchCount,
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1),
            buffers: [dv, att, dvAccum],
            offsets: [dvOffset, 0, 0],
            uints: [UInt32(T), UInt32(HS)]
        )
        ctx.endCompute()
    }
    
    func attention_backward_dq(dq: MTLBuffer, dqOffset: Int, dpreatt: MTLBuffer, qkvr: MTLBuffer, kOffset: Int, batchCount: Int, T: Int, HS: Int, ctx: MetalCommandContext) {
        ctx.compute.threadgroups3D(
            pipelines.attentionBackwardDQ,
            width: HS,
            height: T,
            depth: batchCount,
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1),
            buffers: [dq, dpreatt, qkvr],
            offsets: [dqOffset, 0, kOffset],
            uints: [UInt32(T), UInt32(HS)]
        )
        ctx.endCompute()
    }
    
    func attention_backward_dk(dk: MTLBuffer, dkOffset: Int, dpreatt: MTLBuffer, qkvr: MTLBuffer, qOffset: Int, batchCount: Int, T: Int, HS: Int, ctx: MetalCommandContext) {
        ctx.compute.threadgroups3D(
            pipelines.attentionBackwardDK,
            width: HS,
            height: T,
            depth: batchCount,
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1),
            buffers: [dk, dpreatt, qkvr],
            offsets: [dkOffset, 0, qOffset],
            uints: [UInt32(T), UInt32(HS)]
        )
        ctx.endCompute()
    }
    
    func makeIntBuffer(from array: [UInt32]) -> MTLBuffer {
        array.withUnsafeBufferPointer { bp in
            device.makeBuffer(
                bytes: bp.baseAddress!,
                length: bp.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )!
        }
    }
}

// MARK: - GPT2Metal

final class GPT2Metal {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    let config: LLMGPT2Config
    let pipelines: MetalPipelines

    var params: MetalParameterBuffers
    var grads: MetalParameterBuffers
    var m_memory: MetalParameterBuffers
    var v_memory: MetalParameterBuffers

    var acts: MetalActivationBuffers?
    var grads_acts: MetalActivationBuffers?

    var inputs: MTLBuffer?
    var targets: MTLBuffer?
    let zeroLogitsBias: MTLBuffer

    var batch_size = 0
    var seq_len = 0
    var mean_loss: Float = -1
    var completed_steps = 0

    init(config: LLMGPT2Config, parameterData: Data) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LLMMetalRuntimeError.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw LLMMetalRuntimeError.failedToCreateCommandQueue
        }

        let library: MTLLibrary
        #if SWIFT_MODULE_RESOURCE_BUNDLE_AVAILABLE
            guard let libraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib") else {
                throw LLMMetalRuntimeError.failedToLoadShaders("default.metallib was not found in the module bundle.")
            }
            do {
                library = try device.makeLibrary(URL: libraryURL)
            } catch {
                throw LLMMetalRuntimeError.failedToLoadShaders(error.localizedDescription)
            }
        #else
            guard let defaultLibrary = device.makeDefaultLibrary() else {
                throw LLMMetalRuntimeError.failedToLoadShaders("Metal default library is unavailable.")
            }
            library = defaultLibrary
        #endif

        self.device = device
        self.commandQueue = queue
        self.library = library
        self.config = config
        self.pipelines = try MetalPipelines.create(library: library)

        let floats = parameterData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        self.params = MetalParameterBuffers.fromCheckpoint(
            device: device,
            config: config,
            floats: floats
        )
        let zeroLogitsBias = device.makeBuffer(length: config.vocab_size * floatStride, options: .storageModeShared)!
        memset(zeroLogitsBias.contents(), 0, zeroLogitsBias.length)
        self.zeroLogitsBias = zeroLogitsBias
        self.grads = MetalParameterBuffers.zeros(device: device, config: config)
        self.m_memory = MetalParameterBuffers.zeros(device: device, config: config)
        self.v_memory = MetalParameterBuffers.zeros(device: device, config: config)
    }

    // MARK: - Iterations
    
    func forward(inputs: [UInt32], targets: [UInt32], B: Int, T: Int) {
        let V = config.vocab_size
        let L = config.num_layers
        let NH = config.num_heads
        let C = config.channels
        let HS = C / NH

        batch_size = B
        seq_len = T
        if acts == nil || batch_size != B || seq_len != T {
            acts = MetalActivationBuffers.allocate(
                device: device,
                config: config,
                B: B,
                T: T
            )
        }

        self.inputs = makeIntBuffer(from: inputs)
        if !targets.isEmpty {
            self.targets = makeIntBuffer(from: targets)
        }

        let acts = acts!
        let ctx = MetalCommandContext(commandQueue.makeCommandBuffer()!)

        encoder_forward(out: acts.encoded, inp: self.inputs!, wte: params.wte, wpe: params.wpe, B: B, T: T, C: C, ctx: ctx)

        for l in 0..<L {
            let residual = l == 0 ? acts.encoded : acts.residual3[l - 1]

            layernorm_forward(out: acts.ln1[l], mean: acts.ln1_mean[l], rstd: acts.ln1_rstd[l], inp: residual, weight: params.ln1w[l], bias: params.ln1b[l], B: B, T: T, C: C, ctx: ctx)
            matmul_forward(out: acts.qkv[l], inp: acts.ln1[l], weight: params.qkvw[l], bias: params.qkvb[l], B: B, T: T, C: C, OC: 3 * C, ctx: ctx)
            attention_forward(out: acts.atty[l], preatt: acts.preatt[l], att: acts.att[l], qkvr: acts.qkvr[l], v_accum: acts.v_accum[l], inp: acts.qkv[l], B: B, T: T, C: C, NH: NH, HS: HS, ctx: ctx)
            matmul_forward(out: acts.attproj[l], inp: acts.atty[l], weight: params.attprojw[l], bias: params.attprojb[l], B: B, T: T, C: C, OC: C, ctx: ctx)
            residual_forward(out: acts.residual2[l], inp1: residual, inp2: acts.attproj[l], N: B * T * C, ctx: ctx)

            layernorm_forward(out: acts.ln2[l], mean: acts.ln2_mean[l], rstd: acts.ln2_rstd[l], inp: acts.residual2[l], weight: params.ln2w[l], bias: params.ln2b[l], B: B, T: T, C: C, ctx: ctx)
            matmul_forward(out: acts.fch[l], inp: acts.ln2[l], weight: params.fcw[l], bias: params.fcb[l], B: B, T: T, C: C, OC: 4 * C, ctx: ctx)
            gelu_forward(out: acts.fch_gelu[l], inp: acts.fch[l], N: B * T * 4 * C, ctx: ctx)
            matmul_forward(out: acts.fcproj[l], inp: acts.fch_gelu[l], weight: params.fcprojw[l], bias: params.fcprojb[l], B: B, T: T, C: 4 * C, OC: C, ctx: ctx)
            residual_forward(out: acts.residual3[l], inp1: acts.residual2[l], inp2: acts.fcproj[l], N: B * T * C, ctx: ctx)
        }

        layernorm_forward(out: acts.lnf, mean: acts.lnf_mean, rstd: acts.lnf_rstd, inp: acts.residual3[L - 1], weight: params.lnfw, bias: params.lnfb, B: B, T: T, C: C, ctx: ctx)
        matmul_forward(out: acts.logits, inp: acts.lnf, weight: params.wte, bias: zeroLogitsBias, B: B, T: T, C: C, OC: V, ctx: ctx)
        softmax_forward(out: acts.probs, inp: acts.logits, N: B * T, C: V, ctx: ctx)

        if !targets.isEmpty {
            crossentropy_forward(losses: acts.losses, probs: acts.probs, targets: self.targets!, T: T, V: V, ctx: ctx)
        }

        ctx.commitAndWait()

        if !targets.isEmpty {
            let lossPtr = acts.losses.contents().bindMemory(to: Float.self, capacity: B * T)
            var total: Float = 0
            for i in 0..<(B * T) { total += lossPtr[i] }
            mean_loss = total / Float(B * T)
        } else {
            mean_loss = -1
        }
    }

    func backward() {
        precondition(mean_loss != -1, "Must forward with targets before backward")
        let B = batch_size
        let T = seq_len
        let V = config.vocab_size
        let L = config.num_layers
        let NH = config.num_heads
        let C = config.channels
        let HS = C / NH

        if grads_acts == nil {
            grads_acts = MetalActivationBuffers.allocate(
                device: device,
                config: config,
                B: B,
                T: T
            )
        }
        let acts = acts!
        let ga = grads_acts!

        // Zero all gradient buffers
        let blitBuffer = commandQueue.makeCommandBuffer()!
        let blit = blitBuffer.makeBlitCommandEncoder()!
        for buf in grads.allBuffers + ga.allBuffers {
            blit.fill(buffer: buf, range: 0..<buf.length, value: 0)
        }
        blit.endEncoding()
        blitBuffer.commit()
        blitBuffer.waitUntilCompleted()

        let ctx = MetalCommandContext(commandQueue.makeCommandBuffer()!)

        fill(buffer: ga.losses, value: 1.0 / Float(B * T), count: B * T, ctx: ctx)
        crossentropy_softmax_backward(dlogits: ga.logits, dlosses: ga.losses, probs: acts.probs, targets: self.targets!, B: B, T: T, V: V, ctx: ctx)

        let empty_bias: MTLBuffer? = nil
        matmul_backward(dinp: ga.lnf, dweight: grads.wte, dbias: empty_bias, dout: ga.logits, inp: acts.lnf, weight: params.wte, B: B, T: T, C: C, OC: V, ctx: ctx)

        layernorm_backward(dinp: ga.residual3[L - 1], dweight: grads.lnfw, dbias: grads.lnfb, dout: ga.lnf, inp: acts.residual3[L - 1], weight: params.lnfw, mean: acts.lnf_mean, rstd: acts.lnf_rstd, B: B, T: T, C: C, ctx: ctx)

        for l in stride(from: L - 1, through: 0, by: -1) {
            let dresidual = l == 0 ? ga.encoded : ga.residual3[l - 1]

            residual_backward(dinp1: ga.residual2[l], dinp2: ga.fcproj[l], dout: ga.residual3[l], N: B * T * C, ctx: ctx)
            matmul_backward(dinp: ga.fch_gelu[l], dweight: grads.fcprojw[l], dbias: grads.fcprojb[l], dout: ga.fcproj[l], inp: acts.fch_gelu[l], weight: params.fcprojw[l], B: B, T: T, C: 4 * C, OC: C, ctx: ctx)
            gelu_backward(dinp: ga.fch[l], inp: acts.fch[l], dout: ga.fch_gelu[l], N: B * T * 4 * C, ctx: ctx)
            matmul_backward(dinp: ga.ln2[l], dweight: grads.fcw[l], dbias: grads.fcb[l], dout: ga.fch[l], inp: acts.ln2[l], weight: params.fcw[l], B: B, T: T, C: C, OC: 4 * C, ctx: ctx)
            layernorm_backward(dinp: ga.residual2[l], dweight: grads.ln2w[l], dbias: grads.ln2b[l], dout: ga.ln2[l], inp: acts.residual2[l], weight: params.ln2w[l], mean: acts.ln2_mean[l], rstd: acts.ln2_rstd[l], B: B, T: T, C: C, ctx: ctx)

            residual_backward(dinp1: dresidual, dinp2: ga.attproj[l], dout: ga.residual2[l], N: B * T * C, ctx: ctx)
            matmul_backward(dinp: ga.atty[l], dweight: grads.attprojw[l], dbias: grads.attprojb[l], dout: ga.attproj[l], inp: acts.atty[l], weight: params.attprojw[l], B: B, T: T, C: C, OC: C, ctx: ctx)
            attention_backward(dinp: ga.qkv[l], dpreatt: ga.preatt[l], datt: ga.att[l], dqkvr: ga.qkvr[l], dv_accum: ga.v_accum[l], dout: ga.atty[l], qkvr: acts.qkvr[l], att: acts.att[l], B: B, T: T, C: C, NH: NH, HS: HS, ctx: ctx)
            matmul_backward(dinp: ga.ln1[l], dweight: grads.qkvw[l], dbias: grads.qkvb[l], dout: ga.qkv[l], inp: acts.ln1[l], weight: params.qkvw[l], B: B, T: T, C: C, OC: 3 * C, ctx: ctx)

            let lnResidual = l == 0 ? acts.encoded : acts.residual3[l - 1]
            layernorm_backward(dinp: dresidual, dweight: grads.ln1w[l], dbias: grads.ln1b[l], dout: ga.ln1[l], inp: lnResidual, weight: params.ln1w[l], mean: acts.ln1_mean[l], rstd: acts.ln1_rstd[l], B: B, T: T, C: C, ctx: ctx)
        }

        encoder_backward(dwte: grads.wte, dwpe: grads.wpe, dout: ga.encoded, inp: self.inputs!, B: B, T: T, C: C, ctx: ctx)

        ctx.commitAndWait()
    }

    func update(
        learningRate: Float,
        beta1: Float,
        beta2: Float,
        eps: Float,
        weightDecay: Float,
        t: Int
    ) {
        let beta1_correction = 1 - pow(beta1, Float(t))
        let beta2_correction = 1 - pow(beta2, Float(t))

        let paramsAll = params.allBuffers
        let gradsAll = grads.allBuffers
        let mAll = m_memory.allBuffers
        let vAll = v_memory.allBuffers

        let ctx = MetalCommandContext(commandQueue.makeCommandBuffer()!)
        for i in 0..<paramsAll.count {
            let count = paramsAll[i].length / floatStride
            var lr = learningRate, b1 = beta1, b2 = beta2, e = eps, wd = weightDecay
            var b1c = beta1_correction, b2c = beta2_correction
            let enc = ctx.compute
            enc.setComputePipelineState(pipelines.adamw)
            enc.setBuffer(paramsAll[i], offset: 0, index: 0)
            enc.setBuffer(gradsAll[i], offset: 0, index: 1)
            enc.setBuffer(mAll[i], offset: 0, index: 2)
            enc.setBuffer(vAll[i], offset: 0, index: 3)
            enc.setBytes(&lr, length: floatStride, index: 4)
            enc.setBytes(&b1, length: floatStride, index: 5)
            enc.setBytes(&b2, length: floatStride, index: 6)
            enc.setBytes(&e, length: floatStride, index: 7)
            enc.setBytes(&wd, length: floatStride, index: 8)
            enc.setBytes(&b1c, length: floatStride, index: 9)
            enc.setBytes(&b2c, length: floatStride, index: 10)
            enc.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(256, pipelines.adamw.maxTotalThreadsPerThreadgroup),
                    height: 1,
                    depth: 1
                )
            )
        }
        ctx.commitAndWait()
        completed_steps = t
    }

    // MARK: - Checkpoint

    func exportCheckpoint() throws -> Data {
        try LLMCheckpointCodec.encode(
            header: config.checkpointHeader,
            parameters: params.flattened(),
            expectedParameterCount: config.num_parameters
        )
    }

    // MARK: - Inference helper

    func performInference(inputs: [UInt32], B: Int, T: Int) -> [Float] {
        forward(inputs: inputs, targets: [], B: B, T: T)
        let count = B * T * config.vocab_size
        let ptr = acts!.logits.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

}


// ----------------------------------------------------------------------------
// GPT-2 model definition

struct MetalParameterBuffers {
    let wte: MTLBuffer
    let wpe: MTLBuffer
    let ln1w: [MTLBuffer]
    let ln1b: [MTLBuffer]
    let qkvw: [MTLBuffer]
    let qkvb: [MTLBuffer]
    let attprojw: [MTLBuffer]
    let attprojb: [MTLBuffer]
    let ln2w: [MTLBuffer]
    let ln2b: [MTLBuffer]
    let fcw: [MTLBuffer]
    let fcb: [MTLBuffer]
    let fcprojw: [MTLBuffer]
    let fcprojb: [MTLBuffer]
    let lnfw: MTLBuffer
    let lnfb: MTLBuffer

    static func fromCheckpoint(device: MTLDevice, config: LLMGPT2Config, floats: [Float]) -> MetalParameterBuffers {
        var offset = 0
        func take(_ count: Int) -> MTLBuffer {
            let buf = floats.withUnsafeBufferPointer { bp in
                device.makeBuffer(bytes: bp.baseAddress! + offset, length: count * floatStride, options: .storageModeShared)!
            }
            offset += count
            return buf
        }
        func takePerLayer(_ size: Int) -> [MTLBuffer] {
            (0..<config.num_layers).map { _ in take(size) }
        }
        return MetalParameterBuffers(
            wte: take(config.wte_size),
            wpe: take(config.wpe_size),
            ln1w: takePerLayer(config.ln1w_size),
            ln1b: takePerLayer(config.ln1b_size),
            qkvw: takePerLayer(config.qkvw_size),
            qkvb: takePerLayer(config.qkvb_size),
            attprojw: takePerLayer(config.attprojw_size),
            attprojb: takePerLayer(config.attprojb_size),
            ln2w: takePerLayer(config.ln2w_size),
            ln2b: takePerLayer(config.ln2b_size),
            fcw: takePerLayer(config.fcw_size),
            fcb: takePerLayer(config.fcb_size),
            fcprojw: takePerLayer(config.fcprojw_size),
            fcprojb: takePerLayer(config.fcprojb_size),
            lnfw: take(config.lnfw_size),
            lnfb: take(config.lnfb_size)
        )
    }

    static func zeros(device: MTLDevice, config: LLMGPT2Config) -> MetalParameterBuffers {
        func zero(_ count: Int) -> MTLBuffer {
            device.makeBuffer(length: count * floatStride, options: .storageModeShared)!
        }
        func zeroPerLayer(_ size: Int) -> [MTLBuffer] {
            (0..<config.num_layers).map { _ in zero(size) }
        }
        return MetalParameterBuffers(
            wte: zero(config.wte_size),
            wpe: zero(config.wpe_size),
            ln1w: zeroPerLayer(config.ln1w_size),
            ln1b: zeroPerLayer(config.ln1b_size),
            qkvw: zeroPerLayer(config.qkvw_size),
            qkvb: zeroPerLayer(config.qkvb_size),
            attprojw: zeroPerLayer(config.attprojw_size),
            attprojb: zeroPerLayer(config.attprojb_size),
            ln2w: zeroPerLayer(config.ln2w_size),
            ln2b: zeroPerLayer(config.ln2b_size),
            fcw: zeroPerLayer(config.fcw_size),
            fcb: zeroPerLayer(config.fcb_size),
            fcprojw: zeroPerLayer(config.fcprojw_size),
            fcprojb: zeroPerLayer(config.fcprojb_size),
            lnfw: zero(config.lnfw_size),
            lnfb: zero(config.lnfb_size)
        )
    }

    var allBuffers: [MTLBuffer] {
        var result = [wte, wpe]
        for arrays in [ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb] {
            result.append(contentsOf: arrays)
        }
        result.append(contentsOf: [lnfw, lnfb])
        return result
    }

    func flattened() -> [Float] {
        var result = [Float]()
        for buffer in allBuffers {
            let count = buffer.length / floatStride
            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
            result.append(contentsOf: UnsafeBufferPointer(start: ptr, count: count))
        }
        return result
    }
}

struct MetalActivationBuffers {
    let encoded: MTLBuffer
    let ln1: [MTLBuffer]
    let ln1_mean: [MTLBuffer]
    let ln1_rstd: [MTLBuffer]
    let qkv: [MTLBuffer]
    let atty: [MTLBuffer]
    let preatt: [MTLBuffer]
    let att: [MTLBuffer]
    let attproj: [MTLBuffer]
    let residual2: [MTLBuffer]
    let ln2: [MTLBuffer]
    let ln2_mean: [MTLBuffer]
    let ln2_rstd: [MTLBuffer]
    let fch: [MTLBuffer]
    let fch_gelu: [MTLBuffer]
    let fcproj: [MTLBuffer]
    let residual3: [MTLBuffer]
    let lnf: MTLBuffer
    let lnf_mean: MTLBuffer
    let lnf_rstd: MTLBuffer
    let logits: MTLBuffer
    let probs: MTLBuffer
    let losses: MTLBuffer
    let qkvr: [MTLBuffer]
    let v_accum: [MTLBuffer]

    static func allocate(device: MTLDevice, config: LLMGPT2Config, B: Int, T: Int) -> MetalActivationBuffers {
        let L = config.num_layers
        let C = config.channels
        let NH = config.num_heads
        let V = config.vocab_size
        func buf(_ count: Int) -> MTLBuffer {
            device.makeBuffer(length: count * floatStride, options: .storageModeShared)!
        }
        func perLayer(_ count: Int) -> [MTLBuffer] {
            (0..<L).map { _ in buf(count) }
        }
        return MetalActivationBuffers(
            encoded: buf(B * T * C),
            ln1: perLayer(B * T * C),
            ln1_mean: perLayer(B * T),
            ln1_rstd: perLayer(B * T),
            qkv: perLayer(B * T * 3 * C),
            atty: perLayer(B * T * C),
            preatt: perLayer(B * NH * T * T),
            att: perLayer(B * NH * T * T),
            attproj: perLayer(B * T * C),
            residual2: perLayer(B * T * C),
            ln2: perLayer(B * T * C),
            ln2_mean: perLayer(B * T),
            ln2_rstd: perLayer(B * T),
            fch: perLayer(B * T * 4 * C),
            fch_gelu: perLayer(B * T * 4 * C),
            fcproj: perLayer(B * T * C),
            residual3: perLayer(B * T * C),
            lnf: buf(B * T * C),
            lnf_mean: buf(B * T),
            lnf_rstd: buf(B * T),
            logits: buf(B * T * V),
            probs: buf(B * T * V),
            losses: buf(B * T),
            qkvr: perLayer(B * T * 3 * C),
            v_accum: perLayer(B * T * C)
        )
    }

    var allBuffers: [MTLBuffer] {
        var result = [encoded]
        for arrays in [ln1, ln1_mean, ln1_rstd, qkv, atty, preatt, att, attproj, residual2, ln2, ln2_mean, ln2_rstd, fch, fch_gelu, fcproj, residual3, qkvr, v_accum] {
            result.append(contentsOf: arrays)
        }
        result.append(contentsOf: [lnf, lnf_mean, lnf_rstd, logits, probs, losses])
        return result
    }
}

enum LLMMetalRuntimeError: LocalizedError {
    case missingCheckpoint
    case failedToCreateModel
    case invalidDataset(String)
    case tokenizerRequired
    case noMetalDevice
    case failedToCreateCommandQueue
    case failedToLoadShaders(String)
    case failedToCreatePipeline(String)

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint: "A checkpoint is required before running the Metal backend."
        case .failedToCreateModel: "Failed to create the Metal model runtime."
        case .invalidDataset(let msg): msg
        case .tokenizerRequired: "Inference requires a tokenizer asset."
        case .noMetalDevice: "No Metal device available."
        case .failedToCreateCommandQueue: "Failed to create a Metal command queue."
        case .failedToLoadShaders(let msg): "Failed to load Metal shaders: \(msg)"
        case .failedToCreatePipeline(let msg): "Failed to create compute pipeline: \(msg)"
        }
    }
}

private let floatStride = MemoryLayout<Float>.stride

// MARK: - Command encoding helper

private class MetalCommandContext {
    let commandBuffer: MTLCommandBuffer
    private var encoder: MTLComputeCommandEncoder?

    init(_ commandBuffer: MTLCommandBuffer) {
        self.commandBuffer = commandBuffer
    }

    var compute: MTLComputeCommandEncoder {
        if let e = encoder { return e }
        let e = commandBuffer.makeComputeCommandEncoder()!
        encoder = e
        return e
    }

    func endCompute() {
        encoder?.endEncoding()
        encoder = nil
    }

    func commitAndWait() {
        endCompute()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

// MARK: - Compute pipelines

struct MetalPipelines {
    let encoderForward: MTLComputePipelineState
    let mean: MTLComputePipelineState
    let rstd: MTLComputePipelineState
    let normalization: MTLComputePipelineState
    let permute: MTLComputePipelineState
    let unpermute: MTLComputePipelineState
    let addBias: MTLComputePipelineState
    let scale: MTLComputePipelineState
    let softmaxForward: MTLComputePipelineState
    let residualForward: MTLComputePipelineState
    let gelu: MTLComputePipelineState
    let crossentropyForward: MTLComputePipelineState
    let encoderBackward: MTLComputePipelineState
    let layernormBackward: MTLComputePipelineState
    let geluBackward: MTLComputePipelineState
    let residualBackward: MTLComputePipelineState
    let crossentropySoftmaxBackward: MTLComputePipelineState
    let softmaxBackward: MTLComputePipelineState
    let scaleBackward: MTLComputePipelineState
    let permuteBackward: MTLComputePipelineState
    let unpermuteBackward: MTLComputePipelineState
    let addBiasBackward: MTLComputePipelineState
    let matmulForward: MTLComputePipelineState
    let matmulBackwardDinp: MTLComputePipelineState
    let matmulBackwardDweight: MTLComputePipelineState
    let attentionQK: MTLComputePipelineState
    let attentionAV: MTLComputePipelineState
    let attentionBackwardDatt: MTLComputePipelineState
    let attentionBackwardDV: MTLComputePipelineState
    let attentionBackwardDQ: MTLComputePipelineState
    let attentionBackwardDK: MTLComputePipelineState
    let adamw: MTLComputePipelineState
    let fill: MTLComputePipelineState

    static func create(library: MTLLibrary) throws -> MetalPipelines {
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw LLMMetalRuntimeError.failedToCreatePipeline("Function '\(name)' not found")
            }
            return try library.device.makeComputePipelineState(function: function)
        }
        return try MetalPipelines(
            encoderForward: pipeline("encoder_forward_kernel2"),
            mean: pipeline("mean_kernel"),
            rstd: pipeline("rstd_kernel"),
            normalization: pipeline("normalization_kernel"),
            permute: pipeline("permute_kernel"),
            unpermute: pipeline("unpermute_kernel"),
            addBias: pipeline("add_bias_kernel"),
            scale: pipeline("scale_kernel"),
            softmaxForward: pipeline("softmax_forward_kernel1"),
            residualForward: pipeline("residual_forward_kernel"),
            gelu: pipeline("gelu_kernel"),
            crossentropyForward: pipeline("crossentropy_forward_kernel1"),
            encoderBackward: pipeline("encoder_backward_kernel"),
            layernormBackward: pipeline("layernorm_backward_kernel"),
            geluBackward: pipeline("gelu_backward_kernel"),
            residualBackward: pipeline("residual_backward_kernel"),
            crossentropySoftmaxBackward: pipeline("crossentropy_softmax_backward_kernel"),
            softmaxBackward: pipeline("softmax_backward_kernel"),
            scaleBackward: pipeline("scale_backward_kernel"),
            permuteBackward: pipeline("permute_backward_kernel"),
            unpermuteBackward: pipeline("unpermute_backward_kernel"),
            addBiasBackward: pipeline("add_bias_backward_kernel"),
            matmulForward: pipeline("matmul_forward_kernel"),
            matmulBackwardDinp: pipeline("matmul_backward_dinp_kernel"),
            matmulBackwardDweight: pipeline("matmul_backward_dweight_kernel"),
            attentionQK: pipeline("attention_qk_kernel"),
            attentionAV: pipeline("attention_av_kernel"),
            attentionBackwardDatt: pipeline("attention_backward_datt_kernel"),
            attentionBackwardDV: pipeline("attention_backward_dv_kernel"),
            attentionBackwardDQ: pipeline("attention_backward_dq_kernel"),
            attentionBackwardDK: pipeline("attention_backward_dk_kernel"),
            adamw: pipeline("adamw_kernel"),
            fill: pipeline("fill_kernel")
        )
    }
}

private extension MTLComputeCommandEncoder {
    func compute(
        _ pipeline: MTLComputePipelineState,
        threads: Int,
        blockSize: Int,
        buffers: [MTLBuffer],
        uints: [UInt32] = []
    ) {
        setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            setBuffer(buf, offset: 0, index: i)
        }
        var uintsCopy = uints
        for i in 0..<uintsCopy.count {
            setBytes(&uintsCopy[i], length: MemoryLayout<UInt32>.stride, index: buffers.count + i)
        }
        let tpg = min(blockSize, pipeline.maxTotalThreadsPerThreadgroup)
        dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
    }

    func threadgroups(
        _ pipeline: MTLComputePipelineState,
        threadgroups: Int,
        blockSize: Int,
        buffers: [MTLBuffer],
        uints: [UInt32] = [],
        threadgroupMemoryLength: Int = 0
    ) {
        setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            setBuffer(buf, offset: 0, index: i)
        }
        var uintsCopy = uints
        for i in 0..<uintsCopy.count {
            setBytes(&uintsCopy[i], length: MemoryLayout<UInt32>.stride, index: buffers.count + i)
        }
        if threadgroupMemoryLength > 0 {
            setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)
        }
        let tpg = min(blockSize, pipeline.maxTotalThreadsPerThreadgroup)
        dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
    }

    func compute2D(
        _ pipeline: MTLComputePipelineState,
        width: Int,
        height: Int,
        threadsPerThreadgroup: MTLSize,
        buffers: [MTLBuffer],
        uints: [UInt32] = []
    ) {
        setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            setBuffer(buf, offset: 0, index: i)
        }
        var uintsCopy = uints
        for i in 0..<uintsCopy.count {
            setBytes(&uintsCopy[i], length: MemoryLayout<UInt32>.stride, index: buffers.count + i)
        }
        dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    func threadgroups2D(
        _ pipeline: MTLComputePipelineState,
        width: Int,
        height: Int,
        threadsPerThreadgroup: MTLSize,
        buffers: [MTLBuffer],
        uints: [UInt32] = []
    ) {
        setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            setBuffer(buf, offset: 0, index: i)
        }
        var uintsCopy = uints
        for i in 0..<uintsCopy.count {
            setBytes(&uintsCopy[i], length: MemoryLayout<UInt32>.stride, index: buffers.count + i)
        }
        let groupsX = (width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width
        let groupsY = (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height
        dispatchThreadgroups(
            MTLSize(width: groupsX, height: groupsY, depth: 1),
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    func compute3D(
        _ pipeline: MTLComputePipelineState,
        width: Int,
        height: Int,
        depth: Int,
        threadsPerThreadgroup: MTLSize,
        buffers: [MTLBuffer],
        uints: [UInt32] = []
    ) {
        setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            setBuffer(buf, offset: 0, index: i)
        }
        var uintsCopy = uints
        for i in 0..<uintsCopy.count {
            setBytes(&uintsCopy[i], length: MemoryLayout<UInt32>.stride, index: buffers.count + i)
        }
        dispatchThreads(
            MTLSize(width: width, height: height, depth: depth),
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    func threadgroups3D(
        _ pipeline: MTLComputePipelineState,
        width: Int,
        height: Int,
        depth: Int,
        threadsPerThreadgroup: MTLSize,
        buffers: [MTLBuffer],
        offsets: [Int] = [],
        uints: [UInt32] = []
    ) {
        setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            let offset = i < offsets.count ? offsets[i] : 0
            setBuffer(buf, offset: offset, index: i)
        }
        var uintsCopy = uints
        for i in 0..<uintsCopy.count {
            setBytes(&uintsCopy[i], length: MemoryLayout<UInt32>.stride, index: buffers.count + i)
        }
        let groupsX = (width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width
        let groupsY = (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height
        let groupsZ = (depth + threadsPerThreadgroup.depth - 1) / threadsPerThreadgroup.depth
        dispatchThreadgroups(
            MTLSize(width: groupsX, height: groupsY, depth: groupsZ),
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }
}
