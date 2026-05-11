// ================================================================
// FlashRT — Full-WMMA Two-Pass Split-K Attention  (Orin decoder)
//
// Uses tensor cores (WMMA m16n16k16 BF16) for BOTH QK^T and AV.
//
// Why WMMA for AV matters (the key insight):
//   Scalar AV: for each kv position, iterate q_seq=10 rows sequentially
//              → 10× serial inner loops per kv position = bottleneck
//   WMMA AV:   scores(16,16) × V(16,16) → output(16,16) in ONE call
//              → all 16 q_rows computed simultaneously = 16× faster
//
// Two-pass design (avoids online-softmax WMMA fragment issue):
//   Within each warp's kv sub-range (~34 positions, 3 WMMA tiles):
//   Pass A: QK^T → store all score tiles to shared memory (3×1KB)
//   Softmax: find per-row max over stored tiles, apply exp in-place
//   Pass B: AV with WMMA using the BF16 softmax scores from shared
//   → Only need per-warp m_max[16] and l_sum[16] (no per-tile alpha)
//
// Occupancy:
//   Grid: (NH=8, KV_SPLITS=4) = 32 blocks → 2 blocks/SM
//   Block: 4 warps × 32 threads = 128 threads → 8 warps/SM
//   Shared per block: Q(8KB) + scores(12KB) + BF16-temp(2KB) = 22KB
//   22KB × 2 blocks = 44KB < 48KB per SM limit ✓
//   vs FA2: ~4 warps/SM  → 2× better latency hiding
//
// Expected: ~30-60 µs per call (vs FA2 103 µs) → saves ~8-13 ms
// ================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

static constexpr int KW_SPLITS   = 4;
static constexpr int KW_NWARPS   = 4;
static constexpr int KW_QPAD     = 16;    // q_seq padded to WMMA_M=16
static constexpr int KW_HD_TILES = 16;    // HD=256 / WMMA_N=16
static constexpr int KW_KV_TILES = 3;     // ceil(ceil(kv_per_split/4)/16) ≤ 3

// ── Phase 1 ────────────────────────────────────────────────────────
// Grid: (NH, KV_SPLITS)  Block: (32, NWARPS)
__global__ void wmma_two_pass_p1(
        const __nv_bfloat16* __restrict__ Q,   // (q_seq, NH, HD)
        const __nv_bfloat16* __restrict__ K,   // (kv_seq, 1, HD)
        const __nv_bfloat16* __restrict__ V,   // (kv_seq, 1, HD)
        float* __restrict__ O_partial,          // (NH,SP,NWARPS,QPAD,HD) float32
        float* __restrict__ m_partial,          // (NH,SP,NWARPS,QPAD)
        float* __restrict__ l_partial,
        int q_seq, int kv_seq, int NH, int HD, float scale)
{
    constexpr int WM = 16, WN = 16, WK = 16;

    int h_idx    = blockIdx.x;
    int kv_split = blockIdx.y;
    int wid      = threadIdx.y;
    int lane     = threadIdx.x;

    // ── Shared memory layout ──────────────────────────────────────
    // Q_smem    : [QPAD × HD] BF16         = 8192 B
    // score_smem: [NWARPS][KV_TILES][QPAD×WN] float32  = 4 × 3 × 1024 = 12288 B
    // bf16_tmp  : [NWARPS][QPAD×WN] BF16   = 4 × 512   =  2048 B
    extern __shared__ char smem_raw[];
    auto* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* score_smem = reinterpret_cast<float*>(
        smem_raw + KW_QPAD * HD * sizeof(__nv_bfloat16));
    auto* bf16_tmp = reinterpret_cast<__nv_bfloat16*>(
        score_smem + KW_NWARPS * KW_KV_TILES * KW_QPAD * WN);

    // ── Load Q cooperatively ──────────────────────────────────────
    for (int idx = wid * 32 + lane; idx < KW_QPAD * HD; idx += KW_NWARPS * 32) {
        int qi = idx / HD, di = idx % HD;
        Q_smem[qi * HD + di] = (qi < q_seq) ?
            Q[qi * NH * HD + h_idx * HD + di] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // ── Warp's kv range ───────────────────────────────────────────
    int kv_per_split = (kv_seq + KW_SPLITS - 1) / KW_SPLITS;
    int blk_start    = kv_split * kv_per_split;
    int blk_end      = min(blk_start + kv_per_split, kv_seq);
    int blk_len      = blk_end - blk_start;
    // Warp wid strides over the block's kv range with stride KW_NWARPS×WN
    // kv tiles for warp wid: t={0,1,...}: kv_tile = blk_start + wid*WN + t*KW_NWARPS*WN
    // Max tiles per warp: ceil(blk_len / (KW_NWARPS * WN))
    // For blk_len≈133 and KW_NWARPS×WN=64: ceil(133/64)=3 → KW_KV_TILES=3 max
    int n_my_tiles = 0;
    float* my_score = score_smem + wid * KW_KV_TILES * KW_QPAD * WN;
    __nv_bfloat16* my_bf16  = bf16_tmp + wid * KW_QPAD * WN;

    // ── PASS A: QK^T → store score tiles to shared ────────────────
    for (int t = 0; t < KW_KV_TILES; ++t) {
        int kv_tile = blk_start + wid * WN + t * KW_NWARPS * WN;
        if (kv_tile >= blk_end) break;
        ++n_my_tiles;
        int kv_valid = min(kv_tile + WN, blk_end) - kv_tile;

        // WMMA QK^T: Q(16,256) × K^T(256,16) → qk_acc(16,16)
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> qk_acc;
        wmma::fill_fragment(qk_acc, 0.f);

        for (int k = 0; k < HD / WK; ++k) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> q_f;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> k_f;
            wmma::load_matrix_sync(q_f, Q_smem + k * WK, HD);
            if (kv_valid == WN) {
                wmma::load_matrix_sync(k_f, K + kv_tile * HD + k * WK, HD);
            } else {
                wmma::fill_fragment(k_f, 0.f);  // partial tile: zero fill
            }
            wmma::mma_sync(qk_acc, q_f, k_f, qk_acc);
        }

        // Store QK result to shared float32 buffer (tile t of this warp)
        float* tile_smem = my_score + t * KW_QPAD * WN;
        wmma::store_matrix_sync(tile_smem, qk_acc, WN, wmma::mem_row_major);
        __syncwarp();

        // Mask out-of-range kv columns with -inf (for softmax correctness)
        if (kv_valid < WN) {
            for (int e = lane; e < KW_QPAD * WN; e += 32) {
                int n = e % WN;
                if (n >= kv_valid) tile_smem[e] = -1e20f;
            }
        }
        // Mask padding q rows (qi >= q_seq)
        if (q_seq < KW_QPAD) {
            for (int e = lane; e < KW_QPAD * WN; e += 32) {
                int qi = e / WN;
                if (qi >= q_seq) my_score[t * KW_QPAD * WN + e] = -1e20f;
            }
        }
    }
    __syncwarp();

    // ── SOFTMAX: per-row max and exp (scalar, in shared) ──────────
    // Thread qi (0..15) handles row qi for all tiles
    float m_row[KW_QPAD], l_row[KW_QPAD];
    for (int qi = 0; qi < KW_QPAD; ++qi) { m_row[qi] = -1e20f; l_row[qi] = 0.f; }

    // Scale + find per-row max across all my_tiles
    for (int t = 0; t < n_my_tiles; ++t) {
        float* tile = my_score + t * KW_QPAD * WN;
        for (int e = lane; e < KW_QPAD * WN; e += 32) {
            int qi = e / WN;
            float s = tile[e] * scale;
            tile[e] = s;
            // Contribute to per-row max (each thread covers a subset of rows)
        }
    }
    __syncwarp();

    // Per-row max: thread qi covers row qi (for qi=0..15, lane qi handles it directly)
    // Use all 32 threads: thread i handles rows {i, i+16} (if i<16) or just {i%16}
    for (int qi = lane < KW_QPAD ? lane : 0; qi < KW_QPAD; qi += 1) {
        if (lane != qi % 32) continue;  // only thread `qi` handles row `qi`
        float rmax = -1e20f;
        for (int t = 0; t < n_my_tiles; ++t) {
            float* tile = my_score + t * KW_QPAD * WN;
            for (int n = 0; n < WN; ++n) rmax = fmaxf(rmax, tile[qi * WN + n]);
        }
        m_row[qi] = rmax;
    }
    // Threads 0..15 own m_row[0..15]; broadcast within warp
    for (int qi = 0; qi < KW_QPAD; ++qi)
        m_row[qi] = __shfl_sync(0xffffffff, m_row[qi], qi);  // broadcast from thread qi

    // Apply exp and accumulate l_row (thread qi handles row qi)
    for (int t = 0; t < n_my_tiles; ++t) {
        float* tile = my_score + t * KW_QPAD * WN;
        for (int e = lane; e < KW_QPAD * WN; e += 32) {
            int qi = e / WN;
            float es = expf(tile[e] - m_row[qi]);
            tile[e] = es;
        }
    }
    // Sum per row (thread qi sums its row)
    for (int qi = 0; qi < KW_QPAD; ++qi) {
        float rsum = 0.f;
        if (lane == qi % 32) {  // only responsible thread
            for (int t = 0; t < n_my_tiles; ++t) {
                float* tile = my_score + t * KW_QPAD * WN;
                for (int n = 0; n < WN; ++n) rsum += tile[qi * WN + n];
            }
            l_row[qi] = rsum;
        }
        l_row[qi] = __shfl_sync(0xffffffff, l_row[qi], qi % 32);
    }
    __syncwarp();

    // ── PASS B: WMMA AV using softmax scores ──────────────────────
    // out_acc[s]: accumulator for HD tile s (s×16 .. (s+1)×16 output dims)
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> out_acc[KW_HD_TILES];
    for (int s = 0; s < KW_HD_TILES; ++s) wmma::fill_fragment(out_acc[s], 0.f);

    for (int t = 0; t < n_my_tiles; ++t) {
        int kv_tile = blk_start + wid * WN + t * KW_NWARPS * WN;

        // Convert float32 scores to BF16 in my_bf16 temp (warp's own temp area)
        float* tile_f = my_score + t * KW_QPAD * WN;
        for (int e = lane; e < KW_QPAD * WN; e += 32)
            my_bf16[e] = __float2bfloat16(tile_f[e]);
        __syncwarp();

        // Load score A fragment (row_major, 16×16 BF16)
        wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> s_frag;
        wmma::load_matrix_sync(s_frag, my_bf16, WN);

        // WMMA AV: scores(16,16) × V(16,16) → out_acc[s] contribution per HD tile
        for (int s = 0; s < KW_HD_TILES; ++s) {
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> v_frag;
            // V[kv_tile:kv_tile+16, s*16:(s+1)*16] — row_major
            if (kv_tile + WN <= kv_seq) {
                wmma::load_matrix_sync(v_frag, V + kv_tile * HD + s * WN, HD);
            } else {
                wmma::fill_fragment(v_frag, 0.f);
            }
            wmma::mma_sync(out_acc[s], s_frag, v_frag, out_acc[s]);
        }
    }

    // ── Write partial results to global ───────────────────────────
    // Use shared as staging area for storing WMMA fragments
    int o_base  = ((h_idx * KW_SPLITS + kv_split) * KW_NWARPS + wid) * KW_QPAD * HD;
    int ml_base = ((h_idx * KW_SPLITS + kv_split) * KW_NWARPS + wid) * KW_QPAD;

    // Store each HD tile's WMMA accumulator fragment to smem, then copy to global
    // Reuse score_smem as staging (warp wid's portion = my_score, first tile, 1KB)
    for (int s = 0; s < KW_HD_TILES; ++s) {
        float* tmp = my_score;  // 1KB staging area (reuse tile 0 space)
        wmma::store_matrix_sync(tmp, out_acc[s], WN, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < KW_QPAD * WN; e += 32) {
            int qi = e / WN, n = e % WN;
            int d  = s * WN + n;
            O_partial[o_base + qi * HD + d] = tmp[e];
        }
        __syncwarp();
    }

    // Write m_row and l_row (threads 0..15)
    if (lane < KW_QPAD) {
        m_partial[ml_base + lane] = m_row[lane];
        l_partial[ml_base + lane] = l_row[lane];
    }
}


// ── Phase 2: merge all partials ────────────────────────────────────
// Grid: (q_seq=10, NH=8)  Block: (32, 1)
__global__ void wmma_two_pass_p2(
        const float* __restrict__ O_partial,   // (NH, SP, NW, QPAD, HD)
        const float* __restrict__ m_partial,   // (NH, SP, NW, QPAD)
        const float* __restrict__ l_partial,
        __nv_bfloat16* __restrict__ O,         // (q_seq, NH, HD)
        int q_seq, int NH, int HD)
{
    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int lane  = threadIdx.x;

    // Global max m across all KW_SPLITS × KW_NWARPS partials
    float global_m = -1e20f;
    for (int sp = 0; sp < KW_SPLITS; ++sp)
        for (int w = 0; w < KW_NWARPS; ++w) {
            int idx = ((h_idx * KW_SPLITS + sp) * KW_NWARPS + w) * KW_QPAD + q_idx;
            global_m = fmaxf(global_m, m_partial[idx]);
        }

    // Total l (weighted by exp(m_w - m_global))
    float total_l = 0.f;
    for (int sp = 0; sp < KW_SPLITS; ++sp)
        for (int w = 0; w < KW_NWARPS; ++w) {
            int idx = ((h_idx * KW_SPLITS + sp) * KW_NWARPS + w) * KW_QPAD + q_idx;
            total_l += l_partial[idx] * expf(m_partial[idx] - global_m);
        }
    float inv_l = 1.f / fmaxf(total_l, 1e-10f);

    // Merge O_partial: O = sum_w( exp(m_w - m_global) × O_w ) / total_l
    int steps = (HD + 31) / 32;
    for (int s = 0; s < steps; ++s) {
        int d = lane + s * 32;
        if (d >= HD) continue;
        float merged = 0.f;
        for (int sp = 0; sp < KW_SPLITS; ++sp)
            for (int w = 0; w < KW_NWARPS; ++w) {
                int ml_idx = ((h_idx * KW_SPLITS + sp) * KW_NWARPS + w) * KW_QPAD + q_idx;
                float corr = expf(m_partial[ml_idx] - global_m);
                int   o_b  = ((h_idx * KW_SPLITS + sp) * KW_NWARPS + w) * KW_QPAD * HD;
                merged += O_partial[o_b + q_idx * HD + d] * corr;
            }
        O[q_idx * NH * HD + h_idx * HD + d] = __float2bfloat16(merged * inv_l);
    }
}


// ── Host launcher ──────────────────────────────────────────────────
void gqa_wmma_split_kv_bf16(
        const __nv_bfloat16* Q,
        const __nv_bfloat16* K,
        const __nv_bfloat16* V,
        __nv_bfloat16* O,
        float* O_partial,
        float* m_partial,
        float* l_partial,
        int q_seq, int kv_seq, int NH, int NKV, int HD,
        cudaStream_t stream)
{
    float scale = 1.f / sqrtf(static_cast<float>(HD));

    // smem: Q(8KB) + scores(12KB) + bf16_tmp(2KB) = 22KB
    int smem_bytes = KW_QPAD * HD * sizeof(__nv_bfloat16)            // Q_smem
                   + KW_NWARPS * KW_KV_TILES * KW_QPAD * 16 * 4      // score_smem
                   + KW_NWARPS * KW_QPAD * 16 * sizeof(__nv_bfloat16);// bf16_tmp

    dim3 grid1(NH, KW_SPLITS);
    dim3 block1(32, KW_NWARPS);
    wmma_two_pass_p1<<<grid1, block1, smem_bytes, stream>>>(
        Q, K, V, O_partial, m_partial, l_partial,
        q_seq, kv_seq, NH, HD, scale);

    dim3 grid2(q_seq, NH);
    dim3 block2(32, 1);
    wmma_two_pass_p2<<<grid2, block2, 0, stream>>>(
        O_partial, m_partial, l_partial, O, q_seq, NH, HD);
}
