// ================================================================
// FlashRT — Activation kernel declarations
// GeLU, SiLU, Gate*Act*Mul (BF16/FP16 and fused FP8 variants)
// Supports: __half (FP16), __nv_bfloat16 (BF16)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signatures, backward compatible) ──

void gate_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, int n, cudaStream_t stream = 0);

void gelu_inplace(__nv_bfloat16* x, int n, cudaStream_t stream = 0);

// Fused bias-add + GELU (in-place). Replaces the add_bias_bf16 +
// gelu_inplace pair on the SigLIP FFN-up output; saves one L2/DRAM
// round-trip over the (seq × VIS_H) buffer per layer.
//
// Status (2026-05): kernel correctness verified (microbench cosine
// 0.99999 vs the kernel-pair baseline) and 2.12× faster in isolation,
// but **not enabled by default** — wiring it into the SigLIP pipeline
// produced ~0.7 ms p50 latency win (sub-noise on Orin) and pipeline-
// level action cosine 0.94-0.99 vs the original. The cosine drop comes
// from the fused kernel keeping fp32 between bias-add and GELU while
// the original kernel pair rounds to bf16 between them; the two paths
// calibrate the downstream INT8 GEMMs against slightly different
// activation distributions, and 27 SigLIP layers amplify the per-step
// drift. Kept here as an opt-in primitive; a strict bf16-round
// variant could fix the cosine by emulating the original's precision
// boundary, but the gain is small enough that it isn't worth the risk
// of further subtle numerics divergence.
void bias_gelu_bf16(__nv_bfloat16* x, const __nv_bfloat16* bias,
                    int seq_len, int dim, cudaStream_t stream = 0);
void bias_gelu_fp16(__half* x, const __half* bias,
                    int seq_len, int dim, cudaStream_t stream = 0);

void gate_silu_mul_merged(const __nv_bfloat16* merged, __nv_bfloat16* out,
                           int seq, int half_dim, cudaStream_t stream = 0);

void gate_silu_mul_merged_fp8(const __nv_bfloat16* merged, __nv_fp8_e4m3* out,
                               int seq, int half_dim,
                               const float* d_scale, cudaStream_t stream = 0);

// ── FP16 variants ──

void gate_silu_mul_fp16(const __half* gate, const __half* up,
                        __half* out, int n, cudaStream_t stream = 0);

void gelu_inplace_fp16(__half* x, int n, cudaStream_t stream = 0);

void gate_silu_mul_merged_fp16(const __half* merged, __half* out,
                                int seq, int half_dim, cudaStream_t stream = 0);

void gate_silu_mul_merged_fp8_fp16(const __half* merged, __nv_fp8_e4m3* out,
                                    int seq, int half_dim,
                                    const float* d_scale, cudaStream_t stream = 0);

// Split SiLU: separate gate and up buffers → FP8 output
// Matches pi05 silu_mul_split_fp8_k (split gate+up GEMMs for L2 optimization)
void silu_mul_split_fp8_fp16(const __half* gate, const __half* up,
                              __nv_fp8_e4m3* out, int n,
                              const float* d_scale, cudaStream_t stream = 0);
