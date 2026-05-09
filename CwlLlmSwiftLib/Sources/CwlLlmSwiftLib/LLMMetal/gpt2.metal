#include <metal_stdlib>
using namespace metal;

#define MAX_RANK 5
#define MATMUL_TILE 16
#define ATTN_TILE 8
#define M_PI 3.14159265358979323846264338327950288

// -----------------------------------------------------------------------
// Helper functions (from llm.metal)
// -----------------------------------------------------------------------

inline void flat_to_nd(int index, int inShape[MAX_RANK], int inStrides[MAX_RANK], int outCoords[MAX_RANK]) {
    int flatIdx = index;
    for (int i = 0; i < MAX_RANK; i++) {
        int coord = flatIdx / inStrides[i];
        outCoords[i] = min(coord, inShape[i] - 1);
        flatIdx -= coord * inStrides[i];
    }
}

inline int nd_to_flat(int inCoords[MAX_RANK], int inStrides[MAX_RANK]) {
    int flatIdx = 0;
    for (int i = 0; i < MAX_RANK; i++) {
        flatIdx += inCoords[i] * inStrides[i];
    }
    return flatIdx;
}

inline void calc_strides(const int srcShape[MAX_RANK], int outStrides[MAX_RANK]) {
    for (int i = 0; i < MAX_RANK; i++) {
        int prod = 1;
        for (int j = i + 1; j < MAX_RANK; j++) {
            prod *= srcShape[j];
        }
        outStrides[i] = prod;
    }
}

inline int calc_perm_idx(const int srcShape[MAX_RANK], const int permOrder[MAX_RANK], const uint index) {
    int strides[MAX_RANK];
    calc_strides(srcShape, strides);
    int permuted_shape[MAX_RANK];
    int permuted_strides[MAX_RANK];
    for (int i = 0; i < MAX_RANK; i++) {
        int axis = permOrder[i];
        permuted_strides[i] = strides[axis];
        permuted_shape[i] = srcShape[axis];
    }
    int strides_T[MAX_RANK];
    calc_strides(permuted_shape, strides_T);
    int coords[MAX_RANK];
    flat_to_nd(index, permuted_shape, strides_T, coords);
    int permIdx = nd_to_flat(coords, permuted_strides);
    return permIdx;
}

// -----------------------------------------------------------------------
// Forward kernels (from llm.metal gpt2.metal)
// -----------------------------------------------------------------------

kernel void encoder_forward_kernel2(
    device float* out [[buffer(0)]],
    device int* inp [[buffer(1)]],
    device float* wte [[buffer(2)]],
    device float* wpe [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint N = B * T * C;
    if (tid < N) {
        int bt = tid / C;
        int b = bt / T;
        int t = bt % T;
        int c = tid % C;
        int ix = inp[b * T + t];
        out[tid] = wte[ix * C + c] + wpe[t * C + c];
    }
}

kernel void mean_kernel(
    device float* mean [[buffer(0)]],
    device float* inp [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& C [[buffer(3)]],
    uint block_size [[threads_per_threadgroup]],
    uint idx [[threadgroup_position_in_grid]],
    uint tgid [[thread_position_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    device float* x = inp + idx * C;
    float sum = 0.0f;
    for (int i = tgid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgid < (uint)stride) {
            shared[tgid] += shared[tgid + stride];
        }
    }
    if (tgid == 0) {
        mean[idx] = shared[0] / C;
    }
}

kernel void rstd_kernel(
    device float* rstd [[buffer(0)]],
    device float* inp [[buffer(1)]],
    device float* mean [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    uint idx [[threadgroup_position_in_grid]],
    uint tgid [[thread_position_in_threadgroup]],
    uint bsize [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    device float* x = inp + idx * C;
    float m = mean[idx];
    float sum = 0.0f;
    for (uint i = tgid; i < C; i += bsize) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = bsize / 2; stride >= 1; stride /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgid < stride) {
            shared[tgid] += shared[tgid + stride];
        }
    }
    if (tgid == 0) {
        rstd[idx] = 1.0f / precise::sqrt(shared[0] / C + 1e-5f);
    }
}

kernel void normalization_kernel(
    device float* out [[buffer(0)]],
    device float* inp [[buffer(1)]],
    device float* mean [[buffer(2)]],
    device float* rstd [[buffer(3)]],
    device float* weight [[buffer(4)]],
    device float* bias [[buffer(5)]],
    constant uint& B [[buffer(6)]],
    constant uint& T [[buffer(7)]],
    constant uint& C [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint bt = tid / C;
    uint c = tid % C;
    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[tid];
    float n = s * (xi - m);
    out[tid] = bias[c] + n * weight[c];
}

kernel void permute_kernel(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device float* v [[buffer(2)]],
    const device float* inp [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& NH [[buffer(6)]],
    constant uint& d [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < B * NH * N * d) {
        uint b = tid / (NH * N * d);
        uint rest = tid % (NH * N * d);
        uint nh_ = rest / (N * d);
        rest = rest % (N * d);
        uint n = rest / d;
        uint d_ = rest % d;
        uint inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[tid] = inp[inp_idx];
        k[tid] = inp[inp_idx + NH * d];
        v[tid] = inp[inp_idx + 2 * NH * d];
    }
}

kernel void unpermute_kernel(
    const device float* inp [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& NH [[buffer(4)]],
    constant uint& HS [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const int src_shape[5] = {(int)B, (int)NH, (int)HS, (int)T, 1};
    const int perm_order[5] = {0, 3, 1, 2, 4};
    int permIdx = calc_perm_idx(src_shape, perm_order, tid);
    out[tid] = inp[permIdx];
}

kernel void add_bias_kernel(
    device float* out [[buffer(0)]],
    device float* bias [[buffer(1)]],
    constant uint& OC [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = out[tid] + bias[tid % OC];
}

kernel void scale_kernel(
    device float* inout [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& NH [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int rest = tid % (NH * T * T);
    rest = rest % (T * T);
    int t2 = rest / T;
    int t = rest % T;
    if (t > t2) {
        inout[tid] = -INFINITY;
    } else {
        inout[tid] *= scale;
    }
}

kernel void softmax_forward_kernel1(
    device float* out [[buffer(0)]],
    device float* inp [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& C [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    device float* inp_row = inp + tid * C;
    device float* out_row = out + tid * C;
    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
        if (inp_row[j] > maxval) {
            maxval = inp_row[j];
        }
    }
    float sum = 0.0f;
    for (int j = 0; j < C; j++) {
        out_row[j] = exp(inp_row[j] - maxval);
        sum += out_row[j];
    }
    for (int j = 0; j < C; j++) {
        out_row[j] /= sum;
    }
}

kernel void residual_forward_kernel(
    device float* out [[buffer(0)]],
    device float* inp1 [[buffer(1)]],
    device float* inp2 [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = inp1[tid] + inp2[tid];
}

kernel void gelu_kernel(
    device float* out [[buffer(0)]],
    device float* inp [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float xi = inp[tid];
    float s = sqrt(2.0f / M_PI);
    float cube = 0.044715f * xi * xi * xi;
    out[tid] = 0.5f * xi * (1.0f + precise::tanh(s * (xi + cube)));
}

kernel void crossentropy_forward_kernel1(
    device float* losses [[buffer(0)]],
    device float* probs [[buffer(1)]],
    device int* targets [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& V [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint b = tid / T;
    uint t = tid % T;
    device float* probs_bt = probs + b * T * V + t * V;
    int ix = targets[b * T + t];
    losses[b * T + t] = -log(probs_bt[ix]);
}

// -----------------------------------------------------------------------
// Backward kernels
// -----------------------------------------------------------------------

kernel void encoder_backward_kernel(
    device float* dwte [[buffer(0)]],
    device float* dwpe [[buffer(1)]],
    const device float* dout [[buffer(2)]],
    const device int* inp [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint N = B * T * C;
    if (tid < N) {
        uint bt = tid / C;
        uint b = bt / T;
        uint t = bt % T;
        uint c = tid % C;
        float gradient = dout[tid];
        int ix = inp[b * T + t];
        device atomic<float>* dwte_ptr = (device atomic<float>*)(dwte + ix * C + c);
        atomic_fetch_add_explicit(dwte_ptr, gradient, memory_order_relaxed);
        device atomic<float>* dwpe_ptr = (device atomic<float>*)(dwpe + t * C + c);
        atomic_fetch_add_explicit(dwpe_ptr, gradient, memory_order_relaxed);
    }
}

kernel void layernorm_backward_kernel(
    device float* dinp [[buffer(0)]],
    device float* dweight [[buffer(1)]],
    device float* dbias [[buffer(2)]],
    const device float* dout [[buffer(3)]],
    const device float* inp [[buffer(4)]],
    const device float* weight [[buffer(5)]],
    const device float* mean [[buffer(6)]],
    const device float* rstd [[buffer(7)]],
    constant uint& B [[buffer(8)]],
    constant uint& T [[buffer(9)]],
    constant uint& C [[buffer(10)]],
    uint idx [[threadgroup_position_in_grid]],
    uint tgid [[thread_position_in_threadgroup]],
    uint bsize [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Each threadgroup handles one row (one bt position)
    const device float* x = inp + idx * C;
    const device float* dl = dout + idx * C;
    device float* dx = dinp + idx * C;
    float m = mean[idx];
    float s = rstd[idx];

    // Phase 1: Compute dnorm_mean and dnorm_norm_mean via reduction
    float dnorm_sum = 0.0f;
    float dnorm_norm_sum = 0.0f;
    for (uint i = tgid; i < C; i += bsize) {
        float norm_i = (x[i] - m) * s;
        float dnorm_i = weight[i] * dl[i];
        dnorm_sum += dnorm_i;
        dnorm_norm_sum += dnorm_i * norm_i;
    }
    shared[tgid] = dnorm_sum;
    shared[tgid + bsize] = dnorm_norm_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = bsize / 2; stride >= 1; stride /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgid < stride) {
            shared[tgid] += shared[tgid + stride];
            shared[tgid + bsize] += shared[tgid + bsize + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dnorm_mean = shared[0] / C;
    float dnorm_norm_mean = shared[bsize] / C;

    // Phase 2: Compute dinp and accumulate dweight/dbias
    for (uint i = tgid; i < C; i += bsize) {
        float norm_i = (x[i] - m) * s;
        float dnorm_i = weight[i] * dl[i];
        float grad = (dnorm_i - dnorm_mean - norm_i * dnorm_norm_mean) * s;
        device atomic<float>* dx_ptr = (device atomic<float>*)(dx + i);
        atomic_fetch_add_explicit(dx_ptr, grad, memory_order_relaxed);
        device atomic<float>* dw_ptr = (device atomic<float>*)(dweight + i);
        atomic_fetch_add_explicit(dw_ptr, norm_i * dl[i], memory_order_relaxed);
        device atomic<float>* db_ptr = (device atomic<float>*)(dbias + i);
        atomic_fetch_add_explicit(db_ptr, dl[i], memory_order_relaxed);
    }
}

kernel void gelu_backward_kernel(
    device float* dinp [[buffer(0)]],
    const device float* inp [[buffer(1)]],
    const device float* dout [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float x = inp[tid];
    float s = sqrt(2.0f / M_PI);
    float cube = 0.044715f * x * x * x;
    float tanh_arg = s * (x + cube);
    float tanh_out = precise::tanh(tanh_arg);
    float cosh_out = precise::cosh(tanh_arg);
    float sech_out = 1.0f / (cosh_out * cosh_out);
    float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x);
    dinp[tid] += local_grad * dout[tid];
}

kernel void residual_backward_kernel(
    device float* dinp1 [[buffer(0)]],
    device float* dinp2 [[buffer(1)]],
    const device float* dout [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    dinp1[tid] += dout[tid];
    dinp2[tid] += dout[tid];
}

kernel void crossentropy_softmax_backward_kernel(
    device float* dlogits [[buffer(0)]],
    const device float* dlosses [[buffer(1)]],
    const device float* probs [[buffer(2)]],
    const device int* targets [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& V [[buffer(6)]],
    constant uint& Vp [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint bt = tid / V;
    uint v = tid % V;
    if (bt < B * T) {
        uint base = bt * Vp;
        float dloss = dlosses[bt];
        int target = targets[bt];
        float indicator = (v == (uint)target) ? 1.0f : 0.0f;
        dlogits[base + v] += (probs[base + v] - indicator) * dloss;
    }
}

kernel void softmax_backward_kernel(
    device float* dpreatt [[buffer(0)]],
    const device float* datt [[buffer(1)]],
    const device float* att [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& C [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    const device float* att_row = att + tid * C;
    const device float* datt_row = datt + tid * C;
    device float* dpreatt_row = dpreatt + tid * C;
    float dot = 0.0f;
    for (int j = 0; j < C; j++) {
        dot += att_row[j] * datt_row[j];
    }
    for (int j = 0; j < C; j++) {
        dpreatt_row[j] = att_row[j] * (datt_row[j] - dot);
    }
}

kernel void scale_backward_kernel(
    device float* dinp [[buffer(0)]],
    const device float* dout [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& B [[buffer(3)]],
    constant uint& NH [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    int rest = tid % (NH * T * T);
    rest = rest % (T * T);
    int t2 = rest / T;
    int t = rest % T;
    if (t > t2) {
        dinp[tid] = 0.0f;
    } else {
        dinp[tid] = dout[tid] * scale;
    }
}

kernel void permute_backward_kernel(
    const device float* dq [[buffer(0)]],
    const device float* dk [[buffer(1)]],
    const device float* dv [[buffer(2)]],
    device float* dinp [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& NH [[buffer(6)]],
    constant uint& d [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < B * NH * N * d) {
        uint b = tid / (NH * N * d);
        uint rest = tid % (NH * N * d);
        uint nh_ = rest / (N * d);
        rest = rest % (N * d);
        uint n = rest / d;
        uint d_ = rest % d;
        uint inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[tid];
        dinp[inp_idx + NH * d] = dk[tid];
        dinp[inp_idx + 2 * NH * d] = dv[tid];
    }
}

kernel void unpermute_backward_kernel(
    const device float* dout [[buffer(0)]],
    device float* dinp [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& NH [[buffer(4)]],
    constant uint& HS [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const int src_shape[5] = {(int)B, (int)NH, (int)HS, (int)T, 1};
    const int perm_order[5] = {0, 3, 1, 2, 4};
    int permIdx = calc_perm_idx(src_shape, perm_order, tid);
    dinp[permIdx] = dout[tid];
}

kernel void add_bias_backward_kernel(
    device float* dbias [[buffer(0)]],
    const device float* dout [[buffer(1)]],
    constant uint& OC [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint oc = tid % OC;
    device atomic<float>* db_ptr = (device atomic<float>*)(dbias + oc);
    atomic_fetch_add_explicit(db_ptr, dout[tid], memory_order_relaxed);
}

// -----------------------------------------------------------------------
// Specialized matmul kernels
// -----------------------------------------------------------------------

kernel void matmul_forward_kernel(
    device float* out [[buffer(0)]],
    const device float* inp [[buffer(1)]],
    const device float* weight [[buffer(2)]],
    const device float* bias [[buffer(3)]],
    constant uint& BT [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& OC [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float inpTile[MATMUL_TILE][MATMUL_TILE];
    threadgroup float weightTile[MATMUL_TILE][MATMUL_TILE];

    uint oc = gid.x;
    uint bt = gid.y;
    bool inBounds = bt < BT && oc < OC;
    float sum = inBounds ? bias[oc] : 0.0f;

    for (uint kBase = 0; kBase < C; kBase += MATMUL_TILE) {
        uint inpK = kBase + lid.x;
        uint weightK = kBase + lid.y;
        inpTile[lid.y][lid.x] = (bt < BT && inpK < C) ? inp[bt * C + inpK] : 0.0f;
        weightTile[lid.y][lid.x] = (oc < OC && weightK < C) ? weight[oc * C + weightK] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < MATMUL_TILE; k++) {
            sum += inpTile[lid.y][k] * weightTile[k][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        out[bt * OC + oc] = sum;
    }
}

kernel void matmul_backward_dinp_kernel(
    device float* dinp [[buffer(0)]],
    const device float* dout [[buffer(1)]],
    const device float* weight [[buffer(2)]],
    constant uint& BT [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& OC [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float doutTile[MATMUL_TILE][MATMUL_TILE];
    threadgroup float weightTile[MATMUL_TILE][MATMUL_TILE];

    uint c = gid.x;
    uint bt = gid.y;
    bool inBounds = bt < BT && c < C;
    float sum = 0.0f;

    for (uint kBase = 0; kBase < OC; kBase += MATMUL_TILE) {
        uint doutK = kBase + lid.x;
        uint weightK = kBase + lid.y;
        doutTile[lid.y][lid.x] = (bt < BT && doutK < OC) ? dout[bt * OC + doutK] : 0.0f;
        weightTile[lid.y][lid.x] = (c < C && weightK < OC) ? weight[weightK * C + c] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < MATMUL_TILE; k++) {
            sum += doutTile[lid.y][k] * weightTile[k][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        dinp[bt * C + c] += sum;
    }
}

kernel void matmul_backward_dweight_kernel(
    device float* dweight [[buffer(0)]],
    const device float* dout [[buffer(1)]],
    const device float* inp [[buffer(2)]],
    constant uint& BT [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& OC [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float doutTile[MATMUL_TILE][MATMUL_TILE];
    threadgroup float inpTile[MATMUL_TILE][MATMUL_TILE];

    uint c = gid.x;
    uint oc = gid.y;
    bool inBounds = oc < OC && c < C;
    float sum = 0.0f;

    for (uint kBase = 0; kBase < BT; kBase += MATMUL_TILE) {
        uint doutK = kBase + lid.x;
        uint inpK = kBase + lid.y;
        doutTile[lid.y][lid.x] = (oc < OC && doutK < BT) ? dout[doutK * OC + oc] : 0.0f;
        inpTile[lid.y][lid.x] = (c < C && inpK < BT) ? inp[inpK * C + c] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < MATMUL_TILE; k++) {
            sum += doutTile[lid.y][k] * inpTile[k][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        dweight[oc * C + c] += sum;
    }
}

kernel void attention_qk_kernel(
    device float* out [[buffer(0)]],
    const device float* q [[buffer(1)]],
    const device float* k [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& HS [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float qTile[ATTN_TILE][ATTN_TILE];
    threadgroup float kTile[ATTN_TILE][ATTN_TILE];

    uint batch = gid.z;
    uint t2 = gid.x;
    uint t = gid.y;
    bool inBounds = t2 < T && t < T;
    uint matrixStride = T * HS;
    const device float* qBase = q + batch * matrixStride;
    const device float* kBase = k + batch * matrixStride;
    float sum = 0.0f;

    for (uint kBaseIndex = 0; kBaseIndex < HS; kBaseIndex += ATTN_TILE) {
        uint qIndex = kBaseIndex + lid.x;
        uint kIndex = kBaseIndex + lid.y;
        qTile[lid.y][lid.x] = (t < T && qIndex < HS) ? qBase[t * HS + qIndex] : 0.0f;
        kTile[lid.y][lid.x] = (t2 < T && kIndex < HS) ? kBase[t2 * HS + kIndex] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kTileIndex = 0; kTileIndex < ATTN_TILE; kTileIndex++) {
            sum += qTile[lid.y][kTileIndex] * kTile[kTileIndex][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        out[batch * T * T + t * T + t2] = sum;
    }
}

kernel void attention_av_kernel(
    device float* out [[buffer(0)]],
    const device float* att [[buffer(1)]],
    const device float* v [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& HS [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float vTile[ATTN_TILE][ATTN_TILE];
    threadgroup float attTile[ATTN_TILE][ATTN_TILE];

    uint batch = gid.z;
    uint t = gid.x;
    uint hs = gid.y;
    bool inBounds = t < T && hs < HS;
    uint matrixStride = T * HS;
    const device float* attBase = att + batch * T * T;
    const device float* vBase = v + batch * matrixStride;
    float sum = 0.0f;

    for (uint kBaseIndex = 0; kBaseIndex < T; kBaseIndex += ATTN_TILE) {
        uint vIndex = kBaseIndex + lid.x;
        uint attIndex = kBaseIndex + lid.y;
        vTile[lid.y][lid.x] = (hs < HS && vIndex < T) ? vBase[vIndex * HS + hs] : 0.0f;
        attTile[lid.y][lid.x] = (t < T && attIndex < T) ? attBase[t * T + attIndex] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kTileIndex = 0; kTileIndex < ATTN_TILE; kTileIndex++) {
            sum += vTile[lid.y][kTileIndex] * attTile[kTileIndex][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        out[batch * matrixStride + hs * T + t] = sum;
    }
}

kernel void attention_backward_datt_kernel(
    device float* datt [[buffer(0)]],
    const device float* dv_accum [[buffer(1)]],
    const device float* v [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& HS [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float dvTile[ATTN_TILE][ATTN_TILE];
    threadgroup float vTile[ATTN_TILE][ATTN_TILE];

    uint batch = gid.z;
    uint t2 = gid.x;
    uint t = gid.y;
    bool inBounds = t2 < T && t < T;
    uint matrixStride = T * HS;
    const device float* dvBase = dv_accum + batch * matrixStride;
    const device float* vBase = v + batch * matrixStride;
    float sum = 0.0f;

    for (uint kBaseIndex = 0; kBaseIndex < HS; kBaseIndex += ATTN_TILE) {
        uint dvIndex = kBaseIndex + lid.x;
        uint vIndex = kBaseIndex + lid.y;
        dvTile[lid.y][lid.x] = (t < T && dvIndex < HS) ? dvBase[dvIndex * T + t] : 0.0f;
        vTile[lid.y][lid.x] = (t2 < T && vIndex < HS) ? vBase[t2 * HS + vIndex] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kTileIndex = 0; kTileIndex < ATTN_TILE; kTileIndex++) {
            sum += dvTile[lid.y][kTileIndex] * vTile[kTileIndex][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        datt[batch * T * T + t * T + t2] = sum;
    }
}

kernel void attention_backward_dv_kernel(
    device float* dv [[buffer(0)]],
    const device float* att [[buffer(1)]],
    const device float* dv_accum [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& HS [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float attTile[ATTN_TILE][ATTN_TILE];
    threadgroup float dvAccumTile[ATTN_TILE][ATTN_TILE];

    uint batch = gid.z;
    uint hs = gid.x;
    uint t2 = gid.y;
    bool inBounds = hs < HS && t2 < T;
    uint matrixStride = T * HS;
    const device float* attBase = att + batch * T * T;
    const device float* dvAccumBase = dv_accum + batch * matrixStride;
    float sum = 0.0f;

    for (uint kBaseIndex = 0; kBaseIndex < T; kBaseIndex += ATTN_TILE) {
        uint attIndex = kBaseIndex + lid.x;
        uint dvAccumIndex = kBaseIndex + lid.y;
        attTile[lid.y][lid.x] = (t2 < T && attIndex < T) ? attBase[attIndex * T + t2] : 0.0f;
        dvAccumTile[lid.y][lid.x] = (hs < HS && dvAccumIndex < T) ? dvAccumBase[hs * T + dvAccumIndex] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kTileIndex = 0; kTileIndex < ATTN_TILE; kTileIndex++) {
            sum += attTile[lid.y][kTileIndex] * dvAccumTile[kTileIndex][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        dv[batch * matrixStride + t2 * HS + hs] = sum;
    }
}

kernel void attention_backward_dq_kernel(
    device float* dq [[buffer(0)]],
    const device float* dpreatt [[buffer(1)]],
    const device float* k [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& HS [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float dpreattTile[ATTN_TILE][ATTN_TILE];
    threadgroup float kTile[ATTN_TILE][ATTN_TILE];

    uint batch = gid.z;
    uint hs = gid.x;
    uint t = gid.y;
    bool inBounds = hs < HS && t < T;
    uint matrixStride = T * HS;
    const device float* dpreattBase = dpreatt + batch * T * T;
    const device float* kBase = k + batch * matrixStride;
    float sum = 0.0f;

    for (uint kBaseIndex = 0; kBaseIndex < T; kBaseIndex += ATTN_TILE) {
        uint dpreattIndex = kBaseIndex + lid.x;
        uint kIndex = kBaseIndex + lid.y;
        dpreattTile[lid.y][lid.x] = (t < T && dpreattIndex < T) ? dpreattBase[t * T + dpreattIndex] : 0.0f;
        kTile[lid.y][lid.x] = (hs < HS && kIndex < T) ? kBase[kIndex * HS + hs] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kTileIndex = 0; kTileIndex < ATTN_TILE; kTileIndex++) {
            sum += dpreattTile[lid.y][kTileIndex] * kTile[kTileIndex][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        dq[batch * matrixStride + t * HS + hs] = sum;
    }
}

kernel void attention_backward_dk_kernel(
    device float* dk [[buffer(0)]],
    const device float* dpreatt [[buffer(1)]],
    const device float* q [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& HS [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float dpreattTile[ATTN_TILE][ATTN_TILE];
    threadgroup float qTile[ATTN_TILE][ATTN_TILE];

    uint batch = gid.z;
    uint hs = gid.x;
    uint t2 = gid.y;
    bool inBounds = hs < HS && t2 < T;
    uint matrixStride = T * HS;
    const device float* dpreattBase = dpreatt + batch * T * T;
    const device float* qBase = q + batch * matrixStride;
    float sum = 0.0f;

    for (uint kBaseIndex = 0; kBaseIndex < T; kBaseIndex += ATTN_TILE) {
        uint dpreattIndex = kBaseIndex + lid.x;
        uint qIndex = kBaseIndex + lid.y;
        dpreattTile[lid.y][lid.x] = (t2 < T && dpreattIndex < T) ? dpreattBase[dpreattIndex * T + t2] : 0.0f;
        qTile[lid.y][lid.x] = (hs < HS && qIndex < T) ? qBase[qIndex * HS + hs] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kTileIndex = 0; kTileIndex < ATTN_TILE; kTileIndex++) {
            sum += dpreattTile[lid.y][kTileIndex] * qTile[kTileIndex][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        dk[batch * matrixStride + t2 * HS + hs] = sum;
    }
}

// -----------------------------------------------------------------------
// Optimizer and utility kernels
// -----------------------------------------------------------------------

kernel void adamw_kernel(
    device float* params [[buffer(0)]],
    const device float* grads [[buffer(1)]],
    device float* m_memory [[buffer(2)]],
    device float* v_memory [[buffer(3)]],
    constant float& learning_rate [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& beta1_correction [[buffer(9)]],
    constant float& beta2_correction [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    float p = params[tid];
    float g = grads[tid];
    float m = beta1 * m_memory[tid] + (1.0f - beta1) * g;
    float v = beta2 * v_memory[tid] + (1.0f - beta2) * g * g;
    float m_hat = m / beta1_correction;
    float v_hat = v / beta2_correction;
    m_memory[tid] = m;
    v_memory[tid] = v;
    params[tid] = p - learning_rate * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p);
}

kernel void fill_kernel(
    device float* buffer [[buffer(0)]],
    constant float& value [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    buffer[tid] = value;
}
