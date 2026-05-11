#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Split-K GQA cross-attention for decoder (q≤16, kv>>q, NKV=1).
// Two-phase: phase1 splits kv_seq across 4×NH=32 blocks (all 16 SMs
// active); phase2 merges partial outputs. ~8× faster than FA2 for
// the decoder shape (q=10, kv=532, NH=8, HD=256) on Orin SM87.
//
// Scratch buffers (caller provides, pre-allocated):
//   O_partial : (NH, KV_SPLITS=4, 16, HD) float32 — 524KB
//   m_partial : (NH, KV_SPLITS=4, 16)     float32 — 2KB
//   l_partial : same as m_partial
void gqa_split_kv_bf16(
        const __nv_bfloat16* Q,
        const __nv_bfloat16* K,
        const __nv_bfloat16* V,
        __nv_bfloat16* O,
        float* O_partial,    // scratch (NH × 4 × 16 × HD)
        float* m_partial,    // scratch (NH × 4 × 16)
        float* l_partial,    // scratch (NH × 4 × 16)
        int q_seq, int kv_seq, int NH, int NKV, int HD,
        cudaStream_t stream = 0);
