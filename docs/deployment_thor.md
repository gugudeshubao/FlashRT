# FlashRT ThorU (SM110) Deployment Guide

> Build and run Pi0.5 inference on Jetson AGX Thor.
> GPU: NVIDIA Thor, SM110 (cc 11.0), CUDA 13.0, Ubuntu 24.04.3, aarch64.

---

## Machine Details

| Field | Value |
|---|---|
| Repo path | `/home/user/wy/FlashRT` |
| Python env | `/home/user/wy/openpi-full-venv` |
| GPU | NVIDIA Thor (SM110) |
| CUDA | 13.0 |
| Checkpoint | `/home/user/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch` |

## Build

```bash
export PATH=/home/user/wy/openpi-full-venv/bin:/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=/home/user/wy/nvpl/lib:$LD_LIBRARY_PATH

cd /home/user/wy/FlashRT
cmake -B build -S . -DGPU_ARCH=110 \
  -DPython3_EXECUTABLE=/home/user/wy/openpi-full-venv/bin/python
cmake --build build -j4
```

Key notes:
- `nvcc` must be on `PATH` (not always the case by default)
- `pybind11` must be installed into the same venv as CMake Python
- Point CMake at the venv Python explicitly via `-DPython3_EXECUTABLE`
- CUTLASS v4.4.2 vendored under `third_party/cutlass`

## Checkpoint

The `pi05_droid_pytorch` checkpoint is a PyTorch safetensors conversion
of `pi05_droid`. It stores norm-stats at `assets/droid/norm_stats.json`
(not the LIBERO-style path). FlashRT's norm-stats loader handles this
automatically since the DROID norm-stats fix (see `flash_rt/core/utils/norm_stats.py`).

## Quickstart

```bash
python examples/quickstart.py \
  --checkpoint /home/user/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch \
  --config pi05 \
  --framework torch \
  --hardware thor
```

## Performance

All numbers: steady-state, `p50`, single-sample inference.

| Precision | p50 |
|---|---|
| FP8 | ~45.8–46.2 ms |
| FP4 | ~43.9–44.4 ms |

Versus `openpi` eager on the same machine:

| openpi mode | p50 |
|---|---|
| 10 steps | 267.9 ms |
| 1 step | 129.1 ms |

FlashRT is **~5.7–5.9×** lower latency than openpi eager.

Real-photo smoke test (single tabletop photo duplicated into 2 views):

| Backend | latency |
|---|---|
| openpi eager | 274.9 ms |
| FlashRT FP8 | 48.1 ms |
| FlashRT FP4 | 46.5 ms |

Output shape: `(10, 7)` — 10-step action chunk, 7 DoF.
