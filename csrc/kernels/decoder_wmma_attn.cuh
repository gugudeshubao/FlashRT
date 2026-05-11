#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// WMMA QK^T + Scalar AV split-k attention for decoder (q≤16, kv>>q).
// Grid: (NH=8, KV_SPLITS=4) = 32 blocks → 2 blocks/SM → 8 warps/SM
// vs FA2's ~4 warps/SM.  WMMA for QK^T gives 4-8× compute speedup.
//
// Scratch buffers (caller pre-allocates):
//   O_partial: (NH × KV_SPLITS × NWARPS × QPAD × HD) float32  ≈ 2 MB
//   m_partial, l_partial: (NH × KV_SPLITS × NWARPS × QPAD) float32  ≈ 8 KB each
void gqa_wmma_split_kv_bf16(
        const __nv_bfloat16* Q,
        const __nv_bfloat16* K,
        const __nv_bfloat16* V,
        __nv_bfloat16* O,
        float* O_partial,
        float* m_partial,
        float* l_partial,
        int q_seq, int kv_seq, int NH, int NKV, int HD,
        cudaStream_t stream = 0);
