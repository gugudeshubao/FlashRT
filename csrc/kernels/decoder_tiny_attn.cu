// ================================================================
// FlashRT — Split-K GQA Cross-Attention for Orin Decoder
//
// Problem: FA2 for (q=10, kv=532, NH=8, NKV=1, HD=256) launches
// only 8 blocks → 4 warps/SM → poor latency hiding → 103 µs/call.
//
// Key insight: SM87 (Orin) stalls on memory loads due to few warps.
// With 4 warps/SM, the GPU can't hide the ~100-cycle L2 latency.
// This kernel uses Grid(q=10, NH=8, KV_SPLITS=4) = 320 blocks,
// giving 20 warps/SM — 5× better latency hiding.
//
// Design
// ------
// Phase 1  gqa_split_kv_p1:
//   Grid: (q_seq=10, NH=8, KV_SPLITS=4) = 320 blocks (1 warp each)
//   Block: 32 threads = 1 warp
//   Each warp handles ONE (q_idx, h_idx) row for kv_split portion.
//   Register pressure: ~34 regs/thread (very low, excellent ILP).
//   No shared memory, no intra-block synchronisation.
//   Output: O_partial (q, NH, KV_SPLITS, HD) float32  ~320KB
//           m_partial, l_partial (q, NH, KV_SPLITS)   ~1KB each
//
// Phase 2  gqa_split_kv_p2:
//   Grid: (q_seq=10, NH=8) = 80 blocks (1 warp each)
//   Block: 32 threads = 1 warp
//   Merges KV_SPLITS=4 partial (m, l, O) per (q, h) using the
//   Flash Attention split-K merge formula.
//   Output: O (q_seq, NH, HD) BF16
//
// SM occupancy vs FA2:
//   FA2 : 8 blocks × 4 warps = 32 warps / 16 SMs = 4 warps/SM
//   Ours: 320 blocks × 1 warp = 320 warps / 16 SMs = 20 warps/SM
//
// Unique DRAM traffic: NH × 544KB ≈ 4.35 MB (K+V for 8 heads),
// same as FA2. Speedup comes entirely from 5× better latency hiding.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

static constexpr int KV_SPLITS = 4;
static constexpr int Q_PAD     = 16;   // round q_seq up for merge

// ── Phase 1 ────────────────────────────────────────────────────────
// Grid: (q_seq, NH, KV_SPLITS) = 10 × 8 × 4 = 320 blocks
// Block: 32 threads = 1 warp
// Each block handles ONE q_row × ONE head × ONE kv_split range.

template<int STEPS>   // = HD / 32
__global__ void gqa_split_kv_p1(
        const __nv_bfloat16* __restrict__ Q,   // (q_seq, NH, HD)
        const __nv_bfloat16* __restrict__ K,   // (kv_seq, NKV, HD) NKV=1
        const __nv_bfloat16* __restrict__ V,   // (kv_seq, NKV, HD)
        float* __restrict__ O_partial,          // (q_seq, NH, KV_SPLITS, HD)
        float* __restrict__ m_partial,          // (q_seq, NH, KV_SPLITS)
        float* __restrict__ l_partial,          // (q_seq, NH, KV_SPLITS)
        int q_seq, int kv_seq, int NH, int NKV, int HD, float scale)
{
    int q_idx    = blockIdx.x;   // 0 .. q_seq-1
    int h_idx    = blockIdx.y;   // 0 .. NH-1
    int kv_split = blockIdx.z;   // 0 .. KV_SPLITS-1
    int lane     = threadIdx.x;  // 0 .. 31
    int kv_head  = h_idx * NKV / NH;  // 0 for GQA NKV=1

    // ── KV range for this block ───────────────────────────────────
    int kv_per_split = (kv_seq + KV_SPLITS - 1) / KV_SPLITS;
    int kv_start     = kv_split * kv_per_split;
    int kv_end       = min(kv_start + kv_per_split, kv_seq);

    // ── Load Q[q_idx, h_idx, :] into registers ───────────────────
    float q_reg[STEPS];
    #pragma unroll
    for (int s = 0; s < STEPS; ++s) {
        int d = lane + s * 32;
        q_reg[s] = (d < HD) ?
            __bfloat162float(Q[q_idx * NH * HD + h_idx * HD + d]) : 0.f;
    }

    // ── Online softmax state ──────────────────────────────────────
    float m = -1e20f, l = 0.f;
    float acc[STEPS];
    #pragma unroll
    for (int s = 0; s < STEPS; ++s) acc[s] = 0.f;

    // ── Iterate over kv positions in this split ───────────────────
    for (int kv_i = kv_start; kv_i < kv_end; ++kv_i) {
        int kv_base = kv_i * NKV * HD + kv_head * HD;

        // Partial QK dot product
        float dot = 0.f;
        #pragma unroll
        for (int s = 0; s < STEPS; ++s) {
            int d = lane + s * 32;
            float k = (d < HD) ? __bfloat162float(K[kv_base + d]) : 0.f;
            dot += q_reg[s] * k;
        }
        // Warp reduce → scalar score
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, off);
        float score = __shfl_sync(0xffffffff, dot, 0) * scale;

        // Online softmax update
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float exp_s = expf(score - m_new);

        // Accumulate V
        #pragma unroll
        for (int s = 0; s < STEPS; ++s) {
            int d = lane + s * 32;
            float v = (d < HD) ? __bfloat162float(V[kv_base + d]) : 0.f;
            acc[s] = acc[s] * alpha + exp_s * v;
        }
        m = m_new;
        l = l * alpha + exp_s;
    }

    // ── Write partial results to global memory ────────────────────
    int base_o  = ((q_idx * NH + h_idx) * KV_SPLITS + kv_split) * HD;
    int base_ml =  (q_idx * NH + h_idx) * KV_SPLITS + kv_split;

    #pragma unroll
    for (int s = 0; s < STEPS; ++s) {
        int d = lane + s * 32;
        if (d < HD) O_partial[base_o + d] = acc[s];
    }
    if (lane == 0) {
        m_partial[base_ml] = m;
        l_partial[base_ml] = l;
    }
}


// ── Phase 2: merge KV_SPLITS partial outputs ───────────────────────
// Grid: (q_seq=10, NH=8) = 80 blocks (1 warp each)
// Block: 32 threads = 1 warp

__global__ void gqa_split_kv_p2(
        const float* __restrict__ O_partial,   // (q_seq, NH, KV_SPLITS, HD)
        const float* __restrict__ m_partial,   // (q_seq, NH, KV_SPLITS)
        const float* __restrict__ l_partial,   // (q_seq, NH, KV_SPLITS)
        __nv_bfloat16* __restrict__ O,         // (q_seq, NH, HD) BF16
        int q_seq, int NH, int HD)
{
    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int lane  = threadIdx.x;  // 0..31

    // Find global max m and total l across KV_SPLITS
    float global_m = -1e20f;
    int   base_ml  = (q_idx * NH + h_idx) * KV_SPLITS;
    #pragma unroll
    for (int kv_s = 0; kv_s < KV_SPLITS; ++kv_s)
        global_m = fmaxf(global_m, m_partial[base_ml + kv_s]);

    float total_l = 0.f;
    #pragma unroll
    for (int kv_s = 0; kv_s < KV_SPLITS; ++kv_s)
        total_l += l_partial[base_ml + kv_s]
                 * expf(m_partial[base_ml + kv_s] - global_m);
    float inv_l = 1.f / total_l;

    // Merge O_partial across KV_SPLITS for this thread's HD dims
    int base_o  = (q_idx * NH + h_idx) * KV_SPLITS * HD;
    int steps   = (HD + 31) / 32;
    for (int s = 0; s < steps; ++s) {
        int d = lane + s * 32;
        if (d >= HD) continue;
        float merged = 0.f;
        #pragma unroll
        for (int kv_s = 0; kv_s < KV_SPLITS; ++kv_s) {
            float corr = expf(m_partial[base_ml + kv_s] - global_m);
            merged += O_partial[base_o + kv_s * HD + d] * corr;
        }
        O[q_idx * NH * HD + h_idx * HD + d] =
            __float2bfloat16(merged * inv_l);
    }
}


// ── Host launcher ──────────────────────────────────────────────────
void gqa_split_kv_bf16(
        const __nv_bfloat16* Q,
        const __nv_bfloat16* K,
        const __nv_bfloat16* V,
        __nv_bfloat16* O,
        float* O_partial,   // (q_seq, NH, KV_SPLITS, HD) float32
        float* m_partial,   // (q_seq, NH, KV_SPLITS)     float32
        float* l_partial,
        int q_seq, int kv_seq, int NH, int NKV, int HD,
        cudaStream_t stream)
{
    float scale = 1.f / sqrtf(static_cast<float>(HD));
    dim3 grid1(q_seq, NH, KV_SPLITS);
    dim3 block1(32, 1);

    if (HD == 256) {
        gqa_split_kv_p1<8><<<grid1, block1, 0, stream>>>(
            Q, K, V, O_partial, m_partial, l_partial,
            q_seq, kv_seq, NH, NKV, HD, scale);
    } else if (HD == 128) {
        gqa_split_kv_p1<4><<<grid1, block1, 0, stream>>>(
            Q, K, V, O_partial, m_partial, l_partial,
            q_seq, kv_seq, NH, NKV, HD, scale);
    } else {
        gqa_split_kv_p1<16><<<grid1, block1, 0, stream>>>(
            Q, K, V, O_partial, m_partial, l_partial,
            q_seq, kv_seq, NH, NKV, HD, scale);
    }

    // Phase 2: merge
    dim3 grid2(q_seq, NH);
    dim3 block2(32, 1);
    gqa_split_kv_p2<<<grid2, block2, 0, stream>>>(
        O_partial, m_partial, l_partial, O, q_seq, NH, HD);
}
