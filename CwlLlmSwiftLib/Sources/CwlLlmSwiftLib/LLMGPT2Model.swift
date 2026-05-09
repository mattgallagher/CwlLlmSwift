import Foundation

struct LLMGPT2Config: Sendable, Equatable {
    let max_seq_len: Int
    let vocab_size: Int
    let padded_vocab_size: Int
    let num_layers: Int
    let num_heads: Int
    let channels: Int

    init(max_seq_len: Int, vocab_size: Int, padded_vocab_size: Int, num_layers: Int, num_heads: Int, channels: Int) {
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.channels = channels
    }

    init(header: LLMCheckpointHeader) {
        self.init(
            max_seq_len: header.maxSequenceLength,
            vocab_size: header.vocabularySize,
            padded_vocab_size: header.paddedVocabularySize,
            num_layers: header.layerCount,
            num_heads: header.headCount,
            channels: header.channelCount
        )
    }

    var checkpointHeader: LLMCheckpointHeader {
        LLMCheckpointHeader(
            maxSequenceLength: max_seq_len,
            vocabularySize: vocab_size,
            layerCount: num_layers,
            headCount: num_heads,
            channelCount: channels,
            paddedVocabularySize: padded_vocab_size
        )
    }

    var headSize: Int { channels / num_heads }
    var wte_size: Int { padded_vocab_size * channels }
    var wpe_size: Int { max_seq_len * channels }
    var ln1w_size: Int { channels }
    var ln1b_size: Int { channels }
    var qkvw_size: Int { 3 * channels * channels }
    var qkvb_size: Int { 3 * channels }
    var attprojw_size: Int { channels * channels }
    var attprojb_size: Int { channels }
    var ln2w_size: Int { channels }
    var ln2b_size: Int { channels }
    var fcw_size: Int { 4 * channels * channels }
    var fcb_size: Int { 4 * channels }
    var fcprojw_size: Int { channels * (4 * channels) }
    var fcprojb_size: Int { channels }
    var lnfw_size: Int { channels }
    var lnfb_size: Int { channels }
    var num_parameters: Int {
        wte_size + wpe_size + num_layers * (ln1w_size + ln1b_size + qkvw_size + qkvb_size + attprojw_size + attprojb_size + ln2w_size + ln2b_size + fcw_size + fcb_size + fcprojw_size + fcprojb_size) + lnfw_size + lnfb_size
    }
}

struct LLMGPT2ParameterTensors: Sendable {
    var wte: [Float]
    var wpe: [Float]
    var ln1w: [[Float]]
    var ln1b: [[Float]]
    var qkvw: [[Float]]
    var qkvb: [[Float]]
    var attprojw: [[Float]]
    var attprojb: [[Float]]
    var ln2w: [[Float]]
    var ln2b: [[Float]]
    var fcw: [[Float]]
    var fcb: [[Float]]
    var fcprojw: [[Float]]
    var fcprojb: [[Float]]
    var lnfw: [Float]
    var lnfb: [Float]

    init(config: LLMGPT2Config, float_params: [Float]) {
        var offset = 0
        wte = Array(float_params[offset..<(offset + config.wte_size)]); offset += config.wte_size
        wpe = Array(float_params[offset..<(offset + config.wpe_size)]); offset += config.wpe_size
        ln1w = Self.layerArray(count: config.num_layers, size: config.ln1w_size, values: float_params, offset: &offset)
        ln1b = Self.layerArray(count: config.num_layers, size: config.ln1b_size, values: float_params, offset: &offset)
        qkvw = Self.layerArray(count: config.num_layers, size: config.qkvw_size, values: float_params, offset: &offset)
        qkvb = Self.layerArray(count: config.num_layers, size: config.qkvb_size, values: float_params, offset: &offset)
        attprojw = Self.layerArray(count: config.num_layers, size: config.attprojw_size, values: float_params, offset: &offset)
        attprojb = Self.layerArray(count: config.num_layers, size: config.attprojb_size, values: float_params, offset: &offset)
        ln2w = Self.layerArray(count: config.num_layers, size: config.ln2w_size, values: float_params, offset: &offset)
        ln2b = Self.layerArray(count: config.num_layers, size: config.ln2b_size, values: float_params, offset: &offset)
        fcw = Self.layerArray(count: config.num_layers, size: config.fcw_size, values: float_params, offset: &offset)
        fcb = Self.layerArray(count: config.num_layers, size: config.fcb_size, values: float_params, offset: &offset)
        fcprojw = Self.layerArray(count: config.num_layers, size: config.fcprojw_size, values: float_params, offset: &offset)
        fcprojb = Self.layerArray(count: config.num_layers, size: config.fcprojb_size, values: float_params, offset: &offset)
        lnfw = Array(float_params[offset..<(offset + config.lnfw_size)]); offset += config.lnfw_size
        lnfb = Array(float_params[offset..<(offset + config.lnfb_size)]); offset += config.lnfb_size
    }

    static func layerArray(count: Int, size: Int, values: [Float], offset: inout Int) -> [[Float]] {
        let result = (0..<count).map { index in
            Array(values[(offset + index * size)..<(offset + (index + 1) * size)])
        }
        offset += count * size
        return result
    }

    mutating func zero() {
        wte.zeroFill()
        wpe.zeroFill()
        ln1w.zeroFillLayers()
        ln1b.zeroFillLayers()
        qkvw.zeroFillLayers()
        qkvb.zeroFillLayers()
        attprojw.zeroFillLayers()
        attprojb.zeroFillLayers()
        ln2w.zeroFillLayers()
        ln2b.zeroFillLayers()
        fcw.zeroFillLayers()
        fcb.zeroFillLayers()
        fcprojw.zeroFillLayers()
        fcprojb.zeroFillLayers()
        lnfw.zeroFill()
        lnfb.zeroFill()
    }

    func flattened() -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(wte.count + wpe.count + lnfw.count + lnfb.count + ln1w.reduce(0) { $0 + $1.count } * 12)
        values.append(contentsOf: wte)
        values.append(contentsOf: wpe)
        values.append(contentsOf: ln1w.flatMap { $0 })
        values.append(contentsOf: ln1b.flatMap { $0 })
        values.append(contentsOf: qkvw.flatMap { $0 })
        values.append(contentsOf: qkvb.flatMap { $0 })
        values.append(contentsOf: attprojw.flatMap { $0 })
        values.append(contentsOf: attprojb.flatMap { $0 })
        values.append(contentsOf: ln2w.flatMap { $0 })
        values.append(contentsOf: ln2b.flatMap { $0 })
        values.append(contentsOf: fcw.flatMap { $0 })
        values.append(contentsOf: fcb.flatMap { $0 })
        values.append(contentsOf: fcprojw.flatMap { $0 })
        values.append(contentsOf: fcprojb.flatMap { $0 })
        values.append(contentsOf: lnfw)
        values.append(contentsOf: lnfb)
        return values
    }
}

struct LLMGPT2ActivationTensors: Sendable {
    var encoded: [Float] = []
    var ln1: [[Float]] = []
    var ln1_mean: [[Float]] = []
    var ln1_rstd: [[Float]] = []
    var qkv: [[Float]] = []
    var atty: [[Float]] = []
    var preatt: [[Float]] = []
    var att: [[Float]] = []
    var attproj: [[Float]] = []
    var residual2: [[Float]] = []
    var ln2: [[Float]] = []
    var ln2_mean: [[Float]] = []
    var ln2_rstd: [[Float]] = []
    var fch: [[Float]] = []
    var fch_gelu: [[Float]] = []
    var fcproj: [[Float]] = []
    var residual3: [[Float]] = []
    var lnf: [Float] = []
    var lnf_mean: [Float] = []
    var lnf_rstd: [Float] = []
    var logits: [Float] = []
    var probs: [Float] = []
    var losses: [Float] = []
    var allocatedBatchSize = 0
    var allocatedSequenceLength = 0

    mutating func resizeIfNeeded(config: LLMGPT2Config, B: Int, T: Int) -> Int? {
        guard allocatedBatchSize != B || allocatedSequenceLength != T || encoded.isEmpty else {
            return nil
        }

        let Vp = config.padded_vocab_size
        let L = config.num_layers
        let NH = config.num_heads
        let C = config.channels

        allocatedBatchSize = B
        allocatedSequenceLength = T
        encoded = Array(repeating: 0, count: B * T * C)
        ln1 = Self.makeLayers(count: L, elementCount: B * T * C)
        ln1_mean = Self.makeLayers(count: L, elementCount: B * T)
        ln1_rstd = Self.makeLayers(count: L, elementCount: B * T)
        qkv = Self.makeLayers(count: L, elementCount: B * T * 3 * C)
        atty = Self.makeLayers(count: L, elementCount: B * T * C)
        preatt = Self.makeLayers(count: L, elementCount: B * NH * T * T)
        att = Self.makeLayers(count: L, elementCount: B * NH * T * T)
        attproj = Self.makeLayers(count: L, elementCount: B * T * C)
        residual2 = Self.makeLayers(count: L, elementCount: B * T * C)
        ln2 = Self.makeLayers(count: L, elementCount: B * T * C)
        ln2_mean = Self.makeLayers(count: L, elementCount: B * T)
        ln2_rstd = Self.makeLayers(count: L, elementCount: B * T)
        fch = Self.makeLayers(count: L, elementCount: B * T * 4 * C)
        fch_gelu = Self.makeLayers(count: L, elementCount: B * T * 4 * C)
        fcproj = Self.makeLayers(count: L, elementCount: B * T * C)
        residual3 = Self.makeLayers(count: L, elementCount: B * T * C)
        lnf = Array(repeating: 0, count: B * T * C)
        lnf_mean = Array(repeating: 0, count: B * T)
        lnf_rstd = Array(repeating: 0, count: B * T)
        logits = Array(repeating: 0, count: B * T * Vp)
        probs = Array(repeating: 0, count: B * T * Vp)
        losses = Array(repeating: 0, count: B * T)

        return encoded.count + ln1.reduce(0) { $0 + $1.count } + ln1_mean.reduce(0) { $0 + $1.count } + ln1_rstd.reduce(0) { $0 + $1.count } + qkv.reduce(0) { $0 + $1.count } + atty.reduce(0) { $0 + $1.count } + preatt.reduce(0) { $0 + $1.count } + att.reduce(0) { $0 + $1.count } + attproj.reduce(0) { $0 + $1.count } + residual2.reduce(0) { $0 + $1.count } + ln2.reduce(0) { $0 + $1.count } + ln2_mean.reduce(0) { $0 + $1.count } + ln2_rstd.reduce(0) { $0 + $1.count } + fch.reduce(0) { $0 + $1.count } + fch_gelu.reduce(0) { $0 + $1.count } + fcproj.reduce(0) { $0 + $1.count } + residual3.reduce(0) { $0 + $1.count } + lnf.count + lnf_mean.count + lnf_rstd.count + logits.count + probs.count + losses.count
    }

    mutating func zero() {
        encoded.zeroFill()
        ln1.zeroFillLayers()
        ln1_mean.zeroFillLayers()
        ln1_rstd.zeroFillLayers()
        qkv.zeroFillLayers()
        atty.zeroFillLayers()
        preatt.zeroFillLayers()
        att.zeroFillLayers()
        attproj.zeroFillLayers()
        residual2.zeroFillLayers()
        ln2.zeroFillLayers()
        ln2_mean.zeroFillLayers()
        ln2_rstd.zeroFillLayers()
        fch.zeroFillLayers()
        fch_gelu.zeroFillLayers()
        fcproj.zeroFillLayers()
        residual3.zeroFillLayers()
        lnf.zeroFill()
        lnf_mean.zeroFill()
        lnf_rstd.zeroFill()
        logits.zeroFill()
        probs.zeroFill()
        losses.zeroFill()
    }

    private static func makeLayers(count: Int, elementCount: Int) -> [[Float]] {
        (0..<count).map { _ in Array(repeating: 0, count: elementCount) }
    }
}

extension Array where Element == Float {
    mutating func zeroFill() {
        self = Array(repeating: 0, count: count)
    }
}

extension Array where Element == [Float] {
    mutating func zeroFillLayers() {
        for index in indices {
            self[index].zeroFill()
        }
    }
}
