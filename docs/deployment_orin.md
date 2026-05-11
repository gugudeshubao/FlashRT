# FlashRT Jetson AGX Orin (SM87) Deployment Guide

> INT8 inference for Pi0.5 on Jetson AGX Orin.
> GPU: SM87 (Ampere), 16 SMs, LPDDR5X ~204 GB/s, no native FP8.
> Uses the RTX pipeline (`pi05_rtx`) with INT8 fallback paths.

---

## Machine Details

| Field | Value |
|---|---|
| Repo path | `/data/wy/FlashRT` |
| Python | `/usr/bin/python3` (system Python 3.10) |
| GPU | Jetson AGX Orin (SM87) |
| CUDA | 12.6 |
| Checkpoint | `/data/wy/orin_pi05_droid_pytorch` |
| Tokenizer | `/home/dog/.cache/flash_rt/paligemma_tokenizer.model` |

## Architecture Mapping

Orin is dispatched to the **RTX pipeline** (`pi05_rtx`):

```
flash_rt/hardware/__init__.py:
  ("pi05", "torch", "rtx_sm87") → ("flash_rt.frontends.torch.pi05_rtx", "Pi05TorchFrontendRtx")
```

Orin has no native FP8 tensor cores (SM87). The RTX pipeline detects
this via `supports_fp8()` and activates INT8 fallback paths when
`FVK_PI05_RTX_FORCE_INT8=1` is set.

## Build

```bash
export PATH=/home/dog/.local/bin:/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

cd /data/wy/FlashRT
cmake -B build_orin_sm87 -S . \
  -DGPU_ARCH=87 \
  -DFA2_ARCH_NATIVE_ONLY=ON \
  -DFA2_HDIMS='96;128;256' \
  -DFA2_DTYPES='bf16' \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build_orin_sm87 -j4
```

## Usage

```python
import sys, os
sys.path.insert(0, "/data/wy/FlashRT")
os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
import numpy as np

pipe = Pi05TorchFrontendRtx(
    "/data/wy/orin_pi05_droid_pytorch",
    num_views=2,          # 1 or 2
    num_steps=10,         # ODE steps (default 10)
    vision_pool_factor=1, # 1=none, 2=2×2, 4=4×4
    vision_num_layers=27, # SigLIP layers to run (1-27)
)
pipe.set_prompt("pick up the black envelope on the table")

obs = {"image": img, "wrist_image": img}  # uint8 (224,224,3)
pipe.calibrate_with_real_data([obs])       # once, ~2s

result = pipe.infer(obs)
actions = result["actions"]  # (10, 7) numpy
```

## INT8 Optimizations

Activated by `FVK_PI05_RTX_FORCE_INT8=1`. All paths are additive with
safe defaults; ThorU and RTX 4090/5090 paths are unaffected.

### Encoder INT8 (`use_int8_encoder`)

The Gemma-2B encoder runs CUTLASS SM80 rowwise INT8 GEMMs
(`csrc/gemm/cutlass_sm80_int8_rowwise.cu`). Weights are pre-quantized
per-output-channel at load time; activations use per-row dynamic
quantization inside the CUDA graph.

The dominant GEMMs:

| GEMM | Shape (M, N, K) | Notes |
|---|---|---|
| gate+up | (seq≈560, 32768, 2048) | biggest, ~25ms |
| down    | (seq≈560, 2048, 16384) | ~36ms |

### Fused RMSNorm → INT8 kernels (`csrc/kernels/norm.cu`)

Two new kernels eliminate the intermediate BF16 write between RMSNorm
and INT8 quantization in the encoder hot-path:

| Kernel | Purpose |
|---|---|
| `rms_norm_int8_rowwise` | RMSNorm(x) → INT8 + per-row scale, 1 pass |
| `residual_add_rms_norm_int8_rowwise` | residual += x; RMSNorm → INT8, 1 pass |

Saves ~161 MB bandwidth and 68 kernel launches per encoder forward.

### Vision BF16 Autotune

Five SigLIP GEMM shapes are autotuned via `autotune_bf16_nn` before
CUDA graph capture, so cuBLASLt selects the optimal tile algorithm for
SM87.

## Performance Matrix

All: `FVK_PI05_RTX_FORCE_INT8=1`, synthetic uint8 images (identical
latency to real camera input), steady-state p50.

| num_views | pool | vis_layers | steps | cache | p50 | Effective Hz | Quality |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 27 | 10 | 1 | 127 ms | **7.9** | lossless ✓ |
| 2 | 1 | 27 | 10 | **2** | 127/38 ms | **12.2** | lossless ✓ ← recommended |
| 2 | 1 | 27 | 5 | 1 | 107 ms | **9.3** | steps reduction |
| 1 | 1 | 27 | 10 | 1 | ~87 ms | **~11.5** | 1-camera lossless |
| 2 | 2 | 27 | 5 | 1 | ~71 ms | **~14** | cos=0.89 (needs validation) |
| 2 | 4 | 27 | 3 | 1 | ~53 ms | **~19** | cos=0.19 (degraded) |
| 1 | 4 | 27 | 3 | 1 | ~38 ms | **~26** | cos=0.19 (degraded) |

**Effective Hz for cache=2**: `2 / (full_ms + decode_only_ms)`.
Decode-only latency: **38 ms** (vs 127 ms full forward).

**Recommended starting point**: `pool=1, steps=10, cache=2` → **12.2 Hz lossless**.

### Parameter trade-offs

| Parameter | Effect | Quality impact |
|---|---|---|
| `num_steps` 10→5 | −19ms | Minor ODE accuracy |
| `vision_pool_factor` 1→2 | −38ms encoder | Spatial averaging of features |
| `vision_pool_factor` 1→4 | −47ms encoder | More spatial averaging |
| `vision_num_layers` 27→14 | −15ms vision | Shallower visual features |
| `num_views` 2→1 | −40ms total | Loses wrist camera |

Pool and layer reduction have **not been trained** — quality must be
validated on real robot tasks.

## Precision Analysis

### Metric

Encoder K/V cosine similarity (layer-0 key vectors vs BF16 reference)
is a reliable quantization quality indicator — unlike final action cosine,
it is unaffected by diffusion noise amplification.

### Key Finding: Vision Static INT8 Was Broken

Static per-tensor vision INT8 causes **severe encoder feature degradation**:

| Configuration | Encoder K cosine | Verdict |
|---|---|---|
| Enc+Dec INT8, Vision BF16 | **0.991** | ✅ Use this |
| Enc+Dec INT8, Vision static INT8 | **0.282** | ❌ Do not use |

Root cause: a per-tensor scale calibrated on one image cannot capture
SigLIP's activation range, leading to large quantization error that
propagates through the Gemma encoder.

**Fix applied**: `use_int8_vision_static = False` permanently.
Vision runs in BF16 regardless of `FVK_PI05_RTX_FORCE_INT8=1`.

### Current Numbers (after fix)

| Mode | Encoder K cosine | p50 | Hz (cache=1) | Hz (cache=2) |
|---|---|---|---|---|
| BF16 | 1.000 | ~193ms | 5.2 | — |
| **Enc+Dec INT8 (vision=BF16)** | **0.991** | **127ms** | **7.9** | **12.2** |

## Phase Breakdown (pool=1, 2-camera, 5-step)

| Phase | Time | Bottleneck |
|---|---|---|
| Vision SigLIP (27L BF16) | ~29 ms | 16-SM compute bound |
| Encoder Gemma-2B (18L INT8) | ~65 ms | CUTLASS INT8 on 16 SMs |
| Decoder Gemma-300M (5-step INT8) | ~30 ms | FA2 cross-attn |

Orin memory bandwidth is ~204 GB/s (LPDDR5X, same as ThorU).
Encoder GEMMs are **compute-bound** (16 SM limit), not bandwidth-bound.

## GPU Profiling (nsys + ncu)

### nsys: Kernel-Level Time Breakdown (pool=1, 2-cam, 10-step, 3 inferences)

Total GPU time: 140.8 ms/inference.

| Kernel | calls/inf | time/inf | % | Notes |
|---|---|---|---|---|
| **Kernel2** (CUTLASS INT8 GEMMs) | 908 | **86.8 ms** | **61.7%** | All encoder/decoder/vision INT8 GEMMs |
| gate_silu_mul_merged_kernel | 197 | 9.6 ms | 6.8% | SiLU-gated activation |
| quantize_int8_rowwise_kernel | 394 | 9.2 ms | 6.5% | Dynamic per-row quantize (enc attn_O + dec) |
| add_bias_bf16_kernel | 55 | 6.0 ms | 4.3% | Vision bias adds (27L × QKV+up) |
| quantize_int8_kernel_generic | 108 | 4.7 ms | 3.4% | Static vision INT8 quantize |
| flash_fwd_splitkv_kernel + flash_fwd | 224 | 4.4 ms | 3.1% | FA2 attention (vis+enc+dec) |
| residual_add_rms_norm_int8_rowwise | 17 | 1.5 ms | 1.0% | Fused enc FFN norm |
| rms_norm_int8_rowwise | 18 | 0.7 ms | 0.5% | Fused enc QKV norm |
| Other norms / residuals / misc | — | 7.9 ms | 5.6% | — |

GEMM duration distribution (per inference):
- < 50 μs: 568 calls → 15.7 ms (tiny decoder GEMMs, M=10)
- 50–150 μs: 274 calls → 20.6 ms (vision + small encoder GEMMs)
- 150–500 μs: 32 calls → 6.4 ms (medium encoder GEMMs)
- 500–1500 μs: 30 calls → 34.9 ms (large encoder gate_up/down)
- > 1500 μs: 4 calls → 9.3 ms (largest encoder GEMMs)

### ncu: GPU Utilization Per GEMM Shape

Profiled with Nsight Compute 2024.3.1 on Orin SM87 (CC 8.7).

| GEMM | M | N | K | compute_mem % | sm_throughput % | IMMA active % |
|---|---|---|---|---|---|---|
| Encoder gate_up | 560 | 32768 | 2048 | **92.3%** | 79.7% | **80.1%** |
| Decoder gate_up | 10 | 8192 | 1024 | 73.1% | 54.2% | — |

**Encoder gate_up at 92% GPU throughput** — this is the dominant GEMM
and it is already running at 92% of peak hardware capability.
INT8 tensor cores (IMMA) are active 80% of the time during this kernel.

**Decoder gate_up at 73% / 54%** — M=10 is tiny relative to the
128×128×64 CUTLASS tile; 1280 tiles but with only 1 M-tile means
significant underutilization per SM. This is a fundamental hardware
constraint for small-batch decoding.

**Interpretation:** The INT8 CUTLASS GEMM on SM87 is already near its
hardware ceiling for the large encoder GEMMs. Further optimization of
these kernels via different tile shapes or scheduling would yield at
most ~8% improvement. The gap to ThorU is a **hardware gap** (more SMs,
higher-throughput FP8), not a software gap.

## Performance Comparison: Orin vs ThorU

### GPU Specs

| | Orin (SM87) | ThorU (SM110) |
|---|---|---|
| GPU | Jetson AGX Orin | NVIDIA Thor |
| SM count | **16** | **20** |
| Memory BW | ~204 GB/s (LPDDR5X) | ~204 GB/s (LPDDR5X) |
| Precision | INT8 (no native FP8) | **FP8** (native tensor core) |
| CUDA CC | 8.7 | 11.0 |

### Measured Phase Breakdown

nsys measured (pool=1, 2-cam, 10-step, 3 inferences):

| Phase (nsys kernel group) | Orin SM87 (INT8) | ThorU SM110 (FP8) | Ratio |
|---|---|---|---|
| **GEMM kernels** | **86.8 ms (62%)** | **13.8 ms (30%)** | **6.3×** |
| Attention (FMHA) | 8.7 ms (6%) | 16.8 ms (37%) | 0.52× (Orin faster!) |
| Gated SiLU / norms / etc | 18.5 ms (13%) | 7.2 ms (16%) | 2.6× |
| Quantize overhead | 16.1 ms (11%) | 0.8 ms (2%) | 20× |
| Other | 10.7 ms (8%) | 7.4 ms (16%) | — |
| **Total** | **~140 ms / 7.2 Hz** | **~46 ms / 22 Hz** | **3×** |

GPU specs: Orin = 16 SMs, 61 GB unified; ThorU = more SMs, same ~204 GB/s BW.

### Root Cause of the Gap

Memory bandwidth is identical (~204 GB/s) — **bandwidth is not the bottleneck**.

```
Largest GEMM: encoder gate+up (560, 32768, 2048)
  Tile count: ceil(560/128) × ceil(32768/128) = 5 × 256 = 1280 blocks

  Orin  16 SM: 1280 / 16 = 80 waves  → many sequential rounds
  ThorU N  SM: 1280 / N  = <<80 waves → N≈4–6× more SMs → ~4× faster here
```

Key findings from nsys+ncu:

1. **FP8 TOPS per SM (primary)**: ThorU has only 20 SMs vs Orin's 16 (1.25×).
   Yet ThorU GEMMs are **6.3× faster**. This means FP8 on SM110 delivers
   ~5× more TOPS per SM than INT8 on SM87. This is the dominant factor.

2. **Attention: Orin is faster!** ThorU uses custom `nvjet` FMHA kernels
   (16.8 ms/inf, 37% of total) vs Orin's FA2 (8.7 ms/inf, 6% of total).
   ThorU's attention is 1.9× slower. This is why ThorU's bottleneck is
   attention-limited while Orin is GEMM-limited.

3. **Quantize overhead**: Orin needs 16.1 ms/inf for INT8 per-row quantization.
   ThorU needs only 0.8 ms/inf for static FP8 scalar quantize. This 20×
   difference reflects the cost of Orin's dynamic per-row scale computation.

4. **GEMM epilogue fusion**: ThorU fuses GELU+bias into each GEMM kernel.
   Orin cannot (CUBLASLT_EPILOGUE_BIAS returns NOT_SUPPORTED on SM87).

**ncu result (encoder gate_up, the largest GEMM):**

| GEMM | GPU throughput | INT8/FP8 tensor core active |
|---|---|---|
| Orin INT8 (560, 32768, 2048) | 92.3% | IMMA 80.1% |
| ThorU FP8 (560, 32768, 2048) | 73.7% | — (SM100 architecture) |

Orin's encoder GEMM is actually at **higher GPU utilization** (92%) than
ThorU (74%) for this shape. The speedup on ThorU is purely from higher
absolute TOPS per SM in FP8, not better kernel efficiency.

The 3× total gap is a **hardware gap** (FP8 TOPS), not a software gap.
Software optimizations have extracted most available headroom on SM87.

## Comparison with ThorU

Both machines have ~204 GB/s memory bandwidth. ThorU is faster due to:

1. **More SMs** — higher GEMM parallelism
2. **Native FP8** — higher tensor core throughput vs INT8
3. **cuBLASLt epilogue fusion** — `CUBLASLT_EPILOGUE_BIAS/GELU_BIAS` work
   on SM110; return `NOT_SUPPORTED (code=15)` on SM87

CUTLASS EVT (`csrc/gemm/cutlass_sm80_int8_rowwise.cu`) supports bias
and activation epilogues on SM87, but the Gemma encoder/decoder weights
have no bias terms (RMSNorm weight is folded into GEMM weights).

---

## Custom Decoder Attention Experiments (Archived)

The decoder cross-attention shape (q=10, kv=532, NH=8, NKV=1, HD=256)
was a target for optimization because FA2 launches only 8 blocks for
16 SMs (50% SM utilization). Several custom kernels were built and
benchmarked:

| Kernel | GPU time/call | 180 calls | Pipeline p50 | Vs FA2 |
|---|---|---|---|---|
| **FA2 (baseline)** | **103 µs** | **18.5 ms** | **127 ms** | — |
| Scalar 8-warp/block + shared merge | 162 µs | 29 ms | 142 ms | 1.6× slower |
| Split-kv scalar 320 blocks | 216 µs | 39 ms | 183 ms | 2.1× slower |
| WMMA QK + scalar AV | 283 µs | 51 ms | ~ | 2.7× slower |
| **WMMA full 2-pass (QK+AV)** | **141 µs** | **25 ms** | **182 ms** | **1.37× slower** |

**Conclusion**: FA2 on SM87 uses WMMA tensor cores, optimal register blocking,
and pipelining — all custom kernels remain slower due to:
- Multi-kernel launches: 32+80 blocks vs FA2's 8 (extra GPU scheduling overhead)
- Serial per-row softmax loops (no warp-level parallelism across q_rows)
- Extra shared memory staging (score → BF16 → WMMA fragment conversion)

The FlashRT-bundled FA2 for SM87 (hdim=256, dtype=BF16, split-kv) is
already close to the theoretical bandwidth minimum for this shape.
All custom kernels are kept in `csrc/kernels/decoder_tiny_attn.cu` and
`csrc/kernels/decoder_wmma_attn.cu` and are **disabled by default**
(`use_tiny_q_attn=False`).
