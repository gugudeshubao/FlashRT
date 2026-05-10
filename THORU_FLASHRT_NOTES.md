# ThorU FlashRT Notes

## Summary

`FlashRT` has been built and validated on `ThorU` with the
`pi05_droid_pytorch` checkpoint from `openpi`.

Test machine:

- OS: `Ubuntu 24.04.3`
- arch: `aarch64`
- GPU: `NVIDIA Thor`
- CUDA: `13.0`

## Build Notes

Repo on ThorU:

- `/home/user/wy/FlashRT`

Working Python environment:

- `/home/user/wy/openpi-full-venv`

Important environment variables during build / run:

- `PATH=/home/user/wy/openpi-full-venv/bin:/usr/local/cuda/bin:$PATH`
- `CUDACXX=/usr/local/cuda/bin/nvcc`
- `LD_LIBRARY_PATH=/home/user/wy/nvpl/lib:$LD_LIBRARY_PATH`

Important build details:

- `nvcc` existed on the machine but was not initially on `PATH`
- `pybind11` had to be installed into the same venv used by CMake
- CMake had to be pointed at the venv Python explicitly:
  - `-DPython3_EXECUTABLE=/home/user/wy/openpi-full-venv/bin/python`
- `CUTLASS v4.4.2` was vendored under:
  - `third_party/cutlass`

Successful configure/build pattern:

```bash
cmake -B build -S . -DGPU_ARCH=110 \
  -DPython3_EXECUTABLE=/home/user/wy/openpi-full-venv/bin/python
cmake --build build -j4
```

## Checkpoint Used

Checkpoint:

- `/home/user/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch`

This checkpoint is the converted PyTorch version of `pi05_droid`.

## DROID Norm-Stats Fix

Initial problem:

- `FlashRT` `pi05` frontends assumed LIBERO-style norm-stats paths
- the DROID checkpoint instead stores stats at:
  - `assets/droid/norm_stats.json`

Also, the loader could crash when probing inaccessible paths like:

- `/root/.cache/openpi/...`

### Files updated

- `flash_rt/core/utils/norm_stats.py`
- `flash_rt/frontends/torch/pi05_thor.py`
- `flash_rt/frontends/torch/pi05_rtx.py`
- `flash_rt/frontends/jax/pi05_thor.py`

### Behavior after fix

- native support for:
  - `pi05_droid`
  - `pi05_droid_pytorch`
- inaccessible norm-stats candidates are skipped instead of raising
- no manual copy of `norm_stats.json` to checkpoint root is needed anymore

## Quickstart Results on ThorU

Using:

```bash
python examples/quickstart.py \
  --checkpoint /home/user/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch \
  --config pi05 \
  --framework torch \
  --hardware thor
```

Results:

- FP8:
  - `P50 ~45.8-46.2 ms`
- FP4:
  - `P50 ~43.9-44.4 ms`

Output interface:

- shape: `(10, 7)`

Sanity checks passed:

- non-NaN output
- prompt reuse path OK

## Comparison Against openpi

Current `openpi` ThorU eager numbers for the same model family:

- `10 steps` steady `p50`: `267.9 ms`
- `1 step` steady `p50`: `129.1 ms`

So on ThorU, the current FlashRT `pi05` path is roughly:

- `~5.7x - 5.9x` lower latency than `openpi` eager in the tested setup

## Real-Photo Smoke Comparison

A rough comparison was run using one real tabletop photo duplicated into
2 views:

- image:
  - `/home/user/wy/wall-x/test_images/real_tabletop_1.jpg`
- prompt:
  - `pick up the black envelope on the table`

Important caveat:

- this is not a true robot base/wrist frame pair
- current interfaces are not shape-aligned:
  - `openpi`: `(15, 8)`
  - `FlashRT`: `(10, 7)`

Latency on that input:

- `openpi eager`: `274.9 ms`
- `FlashRT FP8`: `48.1 ms`
- `FlashRT FP4`: `46.5 ms`

## Current Conclusion

For local low-latency `pi0.5` inference on `ThorU`, `FlashRT` is now:

- buildable
- runnable
- compatible with the DROID PyTorch checkpoint
- clearly faster than `openpi` eager

The next meaningful validation step would be running with a real
base-camera + wrist-camera pair and a fixed state vector.

---

## Jetson Orin (SM87) INT8 Path

### Hardware

- GPU: Jetson AGX Orin, SM87 (Ampere), 16 SMs
- Memory: LPDDR5X, ~204 GB/s (unified CPU+GPU)
- No native FP8 tensor core support (INT8 tensor cores available)
- Checkpoint: `pi05_droid_pytorch`

### Optimization Journey

All measurements: `FVK_PI05_RTX_FORCE_INT8=1`, cool conditions, p50.

| Step | Change | 2-cam p50 | Hz |
|---|---|---|---|
| Baseline | BF16 全量 | ~193 ms | 5.2 Hz |
| 1 | INT8 decoder only | ~170 ms | 5.9 Hz |
| 2 | **INT8 encoder** (CUTLASS SM8x rowwise) | ~134 ms | 7.4 Hz |
| 3 | **fused `rms_norm_int8_rowwise` kernel** | ~132 ms | 7.6 Hz |
| 4 | **Vision BF16 autotune** (5 shapes) | ~130 ms | 7.7 Hz |
| 5 | **`num_steps=5`** + `add_bias_bf16` | ~113 ms | 8.8 Hz |
| — | **1-camera mode** (num_views=1, 5步) | **72 ms** | **13.9 Hz** |
| — | **1-camera mode** (num_views=1, 10步) | **89 ms** | **11.2 Hz** |

### Optimization 1: Encoder INT8

`FVK_PI05_RTX_FORCE_INT8=1` 原先只开 decoder INT8，encoder 跑 BF16。
现在自动同时开 encoder INT8。

最大瓶颈 GEMM 形状：

| Layer | (M, N, K) | 改善 |
|-------|-----------|------|
| Encoder gate+up | (560, 32768, 2048) | ~2× |
| Encoder down    | (560, 2048, 16384) | ~2× |

权重静态 per-output-channel 量化，激活 per-row 动态量化（CUDA graph 内）。

### Optimization 2: Fused RMSNorm → INT8 Kernel

两个新内核（`csrc/kernels/norm.cu`）消除 encoder 每层的额外 BF16 中间写：

- `rms_norm_int8_rowwise(x, weight, out_i8, scales, seq, dim, eps)`
  - RMSNorm(x, weight) → INT8 + per-row scale，一次 global memory pass
- `residual_add_rms_norm_int8_rowwise(residual, x, weight, out_i8, scales, ...)`
  - residual += x；RMSNorm → INT8，一次 pass

效果：减少 ~161 MB BW + 68 次 kernel launch（18 层 encoder）。

### Optimization 3: Vision BF16 Autotune

在 `autotune_gemms()` 中为 5 个 vision BF16 GEMM shape 调用 `autotune_bf16_nn`：

- QKV: (512, 3456, 1152)
- O:   (512, 1152, 1152)
- up:  (512, 4304, 1152)
- down: (512, 1152, 4304)
- projector: (512, 2048, 1152)

让 cuBLASLt 在 SM87 上选择最优 tile 策略，视觉阶段节省 ~1-2 ms。

### Optimization 4: 可配置 num_steps

`Pi05TorchFrontendRtx` 新增 `num_steps` 参数（默认 10）。
时间嵌入按正确的 `dt = -1/num_steps` 调度重新生成，而不截取 10步表。

```python
pipe = Pi05TorchFrontendRtx(ckpt, num_views=2, num_steps=5)
```

5步 vs 10步权衡：
- 5步：~113 ms (8.8 Hz)，action 轨迹略粗
- 10步：~132 ms (7.6 Hz)，质量最好

### Optimization 5: `add_bias_bf16`（借鉴 ThorU）

ThorU 用 `add_bias_fp16` 做正确的 in-place bias 加法。Orin 原先用
`bias_residual(x, zero_buf, bias)` —— 会读一个全零缓冲区（浪费带宽）。
替换为 `fvk.add_bias_bf16(x, bias, S, D, stream)` 节省 ~33% bandwidth
在 54 次 vision bias add 上（QKV × 27 + FFN-up × 27）。

### 最终性能（合成随机图像，等价于真实数据延迟）

| 配置 | p50 | min | Hz |
|---|---|---|---|
| 2-camera, 5步 | 113.1 ms | 110.9 ms | **8.84 Hz** |
| **1-camera, 5步** | **71.9 ms** | **71.5 ms** | **13.91 Hz** |
| 2-camera, 10步 | 131.8 ms | 130.5 ms | 7.59 Hz |
| 1-camera, 10步 | 89.3 ms | 88.8 ms | 11.20 Hz |

测量说明：随机 `uint8` 图像形状与真实数据相同，计算量完全一致，
延迟结论等价于真实相机输入。

1-camera 比 2-camera 快 57%：`encoder_seq` 从 522 → 266，
encoder gate_up GEMM 计算量和 vision pass 同时减半。

### Phase Breakdown (2-camera, 5 步)

| 阶段 | 耗时 | 主要限制 |
|---|---|---|
| Vision SigLIP (27L BF16) | ~29 ms | 16 SMs compute-bound；INT8 overhead > 收益 |
| Encoder Gemma-2B (18L INT8) | ~62 ms | CUTLASS INT8 限于 16 SM 吞吐量上限 |
| Decoder Gemma-300M (5步 INT8) | ~30 ms | FA2 cross-attn 9ms + tiny GEMM 21ms |

注：Orin 内存带宽 ~204 GB/s（LPDDR5X），与 ThorU 基本相当。
encoder GEMM 是 compute-bound（16 SM 限制），不是带宽限制。

### ThorU vs Orin 优化对比

ThorU (SM110) 在相似带宽下更快的原因：

1. **更多 SM**：更高并行度，GEMM 完工更快
2. **原生 FP8**：SM110 FP8 tensor core 吞吐量 >> SM87 INT8
3. **cuBLASLt epilogue 融合**：SM110 支持 `CUBLASLT_EPILOGUE_BIAS/GELU_BIAS/BIAS_RES`；
   SM87 全部返回 `NOT_SUPPORTED (code=15)`（已验证）

ThorU 可借鉴、已移植到 Orin 的部分：

| ThorU | Orin 等价 | 状态 |
|---|---|---|
| `rms_norm_fp8_noweight` fused | `rms_norm_int8_rowwise` fused | ✅ 已实现 |
| 静态 FP8 weight scale → GEMM alpha | INT8 per-output-channel 静态量化 | ✅ 已实现 |
| `add_bias_fp16` | `add_bias_bf16` | ✅ 已实现 |
| Vision shape autotune | `autotune_bf16_nn` 5 shapes | ✅ 已实现 |
| `fmha_strided_full` | 未加载（需 libfmha_fp16_strided.so） | ❌ 未实现 |
| FP8 GEMM epilogue | cuBLASLt NOT_SUPPORTED on SM87 | ❌ 不可用 |

### CUTLASS INT8 vs FP8 Epilogue Fusion 说明

CUTLASS 的 **EVT（Epilogue Visitor Tree）** 框架对 INT8 和 FP8 都支持
bias、residual、GELU 等 epilogue 融合（这是 CUTLASS raw API 层面的能力）。

区别在于：
- cuBLASLt 高层 API：FP8 epilogue 在 SM89+ 有效；BF16 bias epilogue 在 SM87 无效
- 裸 CUTLASS EVT：INT8 + bias 在 SM87 **技术上可行**，但我们 encoder/decoder
  的 Gemma 权重均无 bias（RMSNorm 已 fold），fusion 无处应用
- **可行的未来扩展**：将 `gate_geglu_merged`（SiLU-gated activation）融合进
  gate_up INT8 GEMM 的 EVT，省去 18 次单独 kernel launch

### Quickstart

```python
import sys
sys.path.insert(0, "/data/wy/FlashRT")
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
import numpy as np, os

os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"

# ── 选择配置 ──────────────────────────────────────────────
# 最快：单摄 5步 → ~72ms / 13.9 Hz
pipe = Pi05TorchFrontendRtx(
    "/data/wy/orin_pi05_droid_pytorch",
    num_views=1,   # 只用 base camera
    num_steps=5,
)

# 最优质量：双摄 10步 → ~132ms / 7.6 Hz
# pipe = Pi05TorchFrontendRtx(..., num_views=2, num_steps=10)

# 双摄平衡：双摄 5步 → ~113ms / 8.8 Hz
# pipe = Pi05TorchFrontendRtx(..., num_views=2, num_steps=5)
# ─────────────────────────────────────────────────────────

pipe.set_prompt("pick up the black envelope on the table")

img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
obs = {"image": img}                          # num_views=1 只传 image
# obs = {"image": img, "wrist_image": img}   # num_views=2

pipe.calibrate_with_real_data([obs])          # 一次性初始化，~2s

result = pipe.infer(obs)
actions = result["actions"]  # (10, 7) numpy — 10 chunk, 7 DoF
```

### 构建命令（Orin SM87）

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
