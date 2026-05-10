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

| num_views | pool | vis_layers | steps | p50 | Hz |
|---|---|---|---|---|---|
| 2 | 1 | 27 | 10 | 132 ms | 7.6 |
| 2 | 1 | 27 | 5 | 113 ms | 8.8 |
| 1 | 1 | 27 | 5 | 72 ms | 13.9 |
| 2 | 2 | 27 | 5 | 74 ms | 13.5 |
| 2 | 2 | 27 | 3 | 67 ms | 15.0 |
| 2 | 4 | 27 | 3 | 56 ms | 17.9 |
| 2 | 4 | 14 | 3 | 42 ms | 23.9 |
| 2 | 4 | 10 | 3 | 38 ms | 26.5 |
| 1 | 4 | 27 | 2 | 38 ms | 26.6 |
| 1 | 4 | 14 | 2 | 30 ms | 33.2 |
| 1 | 4 | 10 | 2 | 28 ms | 35.7 |

**Recommended starting point**: `pool=1, steps=10` (7.6 Hz, lossless).
To reach 10 Hz without pooling: `num_views=1, pool=1, steps=10` → 11.2 Hz.

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

## Phase Breakdown (pool=1, 2-camera, 5-step)

| Phase | Time | Bottleneck |
|---|---|---|
| Vision SigLIP (27L BF16) | ~29 ms | 16-SM compute bound |
| Encoder Gemma-2B (18L INT8) | ~65 ms | CUTLASS INT8 on 16 SMs |
| Decoder Gemma-300M (5-step INT8) | ~30 ms | FA2 cross-attn |

Orin memory bandwidth is ~204 GB/s (LPDDR5X, same as ThorU).
Encoder GEMMs are **compute-bound** (16 SM limit), not bandwidth-bound.

## Comparison with ThorU

Both machines have ~204 GB/s memory bandwidth. ThorU is faster due to:

1. **More SMs** — higher GEMM parallelism
2. **Native FP8** — higher tensor core throughput vs INT8
3. **cuBLASLt epilogue fusion** — `CUBLASLT_EPILOGUE_BIAS/GELU_BIAS` work
   on SM110; return `NOT_SUPPORTED (code=15)` on SM87

CUTLASS EVT (`csrc/gemm/cutlass_sm80_int8_rowwise.cu`) supports bias
and activation epilogues on SM87, but the Gemma encoder/decoder weights
have no bias terms (RMSNorm weight is folded into GEMM weights).
