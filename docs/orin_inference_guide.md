# FlashRT Orin Inference Guide

> How to run Pi0.5 inference on Jetson AGX Orin (SM87) with FlashRT.
> Covers build, configuration, benchmark presets, and API usage.

---

## Quick Start

```bash
# On Orin
cd /data/wy/FlashRT

# Lossless quality (recommended first test)
FVK_PI05_RTX_FORCE_INT8=1 python3 examples/orin/bench_pi05.py \
    --checkpoint /data/wy/orin_pi05_droid_pytorch \
    --preset lossless
```

---

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

Build time: ~20 seconds.

---

## Benchmark Script

`examples/orin/bench_pi05.py` provides preset-based benchmarking.

### Presets

All numbers measured on Jetson AGX Orin (SM87, 16 SMs, LPDDR5X ~204 GB/s)
with `FVK_PI05_RTX_FORCE_INT8=1`, stable conditions, p50.

| Preset | num_views | pool | layers | steps | p50 | Hz | Notes |
|---|---|---|---|---|---|---|---|
| `lossless` | 2 | 1 | 27 | 10 | 128.6 ms | **7.8 Hz** | No quality trade-offs |
| `balanced` | 2 | 1 | 27 | 5 | 109.6 ms | **9.1 Hz** | Fewer ODE steps |
| `fast` | 2 | 2 | 27 | 5 | 70.9 ms | **14.1 Hz** | 2×2 vision pooling |
| `faster` | 2 | 4 | 27 | 3 | 52.6 ms | **19.0 Hz** | 4×4 pooling + 3 steps |
| `fastest` | 1 | 4 | 27 | 3 | 38.1 ms | **26.3 Hz** | Single camera |
| *(custom)* | 1 | 1 | 27 | 10 | 86.7 ms | **11.5 Hz** | 1-camera lossless |

### Usage

```bash
# Named preset
FVK_PI05_RTX_FORCE_INT8=1 python3 examples/orin/bench_pi05.py \
    --checkpoint /path/to/pi05_droid_pytorch \
    --preset lossless

# Manual parameters
FVK_PI05_RTX_FORCE_INT8=1 python3 examples/orin/bench_pi05.py \
    --checkpoint /path/to/pi05_droid_pytorch \
    --num-views 2 \
    --pool 1 \
    --layers 27 \
    --steps 10 \
    --warmup 8 \
    --reps 15

# Help
python3 examples/orin/bench_pi05.py --help
```

---

## Python API

```python
import os
os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"

import sys
sys.path.insert(0, "/data/wy/FlashRT")

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
import numpy as np

pipe = Pi05TorchFrontendRtx(
    "/data/wy/orin_pi05_droid_pytorch",
    num_views=2,              # 1 or 2 cameras
    num_steps=10,             # ODE steps (10=best quality, 5=faster)
    vision_pool_factor=1,     # 1=none, 2=2×2 pool, 4=4×4 pool
    vision_num_layers=27,     # SigLIP layers (1-27, default=27)
)

pipe.set_prompt("pick up the black envelope on the table")

# Observations: uint8 numpy arrays (224, 224, 3)
img = np.zeros((224, 224, 3), dtype=np.uint8)
obs = {"image": img, "wrist_image": img}  # always pass both keys

# One-time calibration (~2s)
pipe.calibrate_with_real_data([obs])

# Inference
result = pipe.infer(obs)
actions = result["actions"]    # numpy (10, 7) — 10-step chunk, 7 DoF
```

---

## Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `num_views` | 2 | Number of cameras. `1`=base only, `2`=base+wrist |
| `num_steps` | 10 | Diffusion ODE steps. Higher = better quality, slower |
| `vision_pool_factor` | 1 | SigLIP output spatial pooling. `1`=none (lossless), `2`=2×2→64 tok/view, `4`=4×4→16 tok/view |
| `vision_num_layers` | 27 | SigLIP layers to run. `27`=full (lossless). Reducing is experimental |

### `vision_pool_factor` — what it does

SigLIP produces 256 tokens per view (16×16 patch grid). Pooling averages
adjacent patches before passing tokens to the Gemma encoder:

```
pool=1: 16×16 = 256 tok/view  (full spatial resolution)
pool=2:  8×8  =  64 tok/view  (2×2 average, moderate loss)
pool=4:  4×4  =  16 tok/view  (4×4 average, more loss)
```

SigLIP still runs all 27 layers on full 512 tokens — pooling happens
**after** SigLIP, reducing only what the Gemma encoder sees. Quality
impact must be validated on real robot tasks.

### Quality trade-off ordering (least → most impact)

```
fewer steps (10→5) → pool=2 → pool=4 → fewer layers → single camera
```

---

## Environment Variable

| Variable | Value | Effect |
|---|---|---|
| `FVK_PI05_RTX_FORCE_INT8` | `1` | Enable encoder + decoder + static-INT8 vision |
| *(unset)* | — | BF16 everywhere (~193ms baseline, ~5.2 Hz) |

Always set `FVK_PI05_RTX_FORCE_INT8=1` on Orin for best performance.

---

## Performance Notes

- **Orin has 16 SMs and no native FP8** — INT8 is the best available precision
- Encoder `gate_up` GEMM (560×32768×2048) runs at **92% GPU utilization** (ncu verified)
  — there is very little software headroom remaining
- ThorU (SM110) is 3× faster primarily due to **Blackwell FP8 per-SM TOPS**, not SM count
  (ThorU: 20 SMs, Orin: 16 SMs — only 25% difference)
- Vision attention (FA2) is faster on Orin than ThorU's custom nvjet kernels for these shapes
- See `docs/deployment_orin.md` for full nsys+ncu profiling data

---

## Troubleshooting

**CUTLASS INT8 workspace error between sequential model loads:**
Free GPU memory between model instances (`del pipe; gc.collect(); time.sleep(5)`).

**Slow first inference:**
Always call `calibrate_with_real_data([obs])` before `infer()`.
The calibration also captures the CUDA graph — first call after calibration
may be slower; subsequent calls reach steady-state.

**`KeyError: 'wrist_image'` with num_views=1:**
Always pass both `image` and `wrist_image` keys in the obs dict.
The pipeline slices to `[:num_views]` internally.
