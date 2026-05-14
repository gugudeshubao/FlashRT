# Pipelined dual-stream inference (Orin SM87) — WIP

> **Status**: infrastructure landed; SERIAL mode is bit-identical to the
> cache_frames=1 baseline; PARALLEL mode currently has ~0.96-0.99 cosine
> drift (root cause + fix paths documented below). 10 Hz lossless target
> not yet hit on Orin alone — fundamentally limited by 16-SM concurrency.

## Goal

Push 2-camera Pi0.5 INT8 lossless inference on Jetson AGX Orin from
**7.86 Hz → ≥10 Hz** with zero precision loss vs the existing
`cache_frames=1` baseline.

The target is the same `pool=1 / 27 SigLIP layers / 10 ODE steps` config
that produces `encoder K cosine = 0.991` against the BF16 reference; we
must keep **every numerical knob unchanged** — no pooling, no fewer
layers, no dropped ODE steps, no cache_frames>1 staleness.

## Approach

**Cross-call pipelining** — overlap frame N's encoder with frame N-1's
decoder on two CUDA streams. The action returned at call N is for image
N-1 (1-frame action delay), but each action is computed from the same
fresh K/V and noise the baseline would have used, so the output is
**bit-identical** to baseline.

```
                    │── stream A ───────────────────────────────────
   call N (enc)     │  vision + Gemma encoder → snap_K_pair[N % 2]
                    │
                    │── stream B ───────────────────────────────────
   call N (dec)     │  decoder graph (reads snap_K_pair[(N-1) % 2])
                    │  → action_{N-1}
```

Theoretical wall = `max(t_enc, t_dec) ≈ max(86, 37) ms = 86 ms = 11.6 Hz`,
beating the 10 Hz target. **In practice on Orin, 16 SMs are saturated
by either side alone, so concurrent streams interleave on the SM array
rather than truly running in parallel.**

## Implementation (`flash_rt/frontends/torch/pi05_rtx.py`)

* `Pi05TorchFrontendRtx._init_pipelined_state()` — clones two snap K/V
  buffers from the original `attn.enc_K`/`enc_V`, allocates two CUDA
  streams (`_enc_stream`, `_dec_stream`), double-buffers the noise
  draws so each frame's noise outlives one call.
* `Pi05TorchFrontendRtx.infer_pipelined(obs)` — per call:
  1. Draw fresh noise into `_noise_pair[curr]` (consumed by the *next*
     call's decoder).
  2. **Decoder side** (stream B): D2D-restore `snap[prev] → orig_K` (5 MB
     each, ~30 µs), copy `_noise_pair[prev] → input_noise_buf`, replay
     the existing `pipe._decoder_only_graph`, D2D-copy out the actions.
  3. **Encoder side** (stream A): re-aim `attn.enc_K = snap[curr]`,
     run vision + encoder Python; writes new K/V into `snap[curr]`.
  4. Sync both streams.
* `bench_pi05.py --pipelined` flag for measurement.

The decoder reuses the **existing** `_decoder_only_graph` captured during
`record_infer_graph` (always built; used by `cache_frames>1`). Reusing
it avoids a fresh graph capture, which empirically produces wrong
outputs when the capture spans changes to `attn.enc_K` and other
shared-pool tensors.

## Numerical equivalence

`tests/` style harness (run baseline first, save fixed `(image, noise)`
sequence, run pipelined, compare):

| N | cosine        | maxabs diff | verdict |
|---|--------------:|------------:|---------|
| 1 | 1.0000000000  | 0           | OK      |
| 2 | 0.9999998810  | 0           | OK      |
| 3 | 1.0000000000  | 0           | OK      |
| 4 | 1.0000000747  | 0           | OK      |
| 5 | 1.0000000000  | 0           | OK      |

`maxabs diff = 0` across all frames — **bit-identical** to the baseline
inference at the same `(image_{N-1}, noise_{N-1})` pair. (Tiny cosine
deviations >1.0 are floating-point precision in the dot-product
calculation, not actual data difference.)

Runs in **SERIAL** mode (decoder finishes before encoder starts) by
default. Setting `pipe._pipelined_force_serial = False` opts into the
parallel mode.

## Measured latency (Jetson AGX Orin, INT8 lossless preset)

| Mode | p50 | Hz | Cosine vs baseline |
|---|---:|---:|---|
| Baseline (cache=1, no pipelining) | 127.2 ms | 7.86 | 1.000 |
| **Pipelined SERIAL** (default) | 130.2 ms | 7.68 | **1.000 ✓** |
| Pipelined PARALLEL (opt-in)    | 119.4 ms | 8.37 | ~0.97-0.99 ✗ |
| `cache_frames=2` (existing) | 83.3 ms | 12.01 | 0.991 (1-frame K/V stale) |

**Conclusion**: pipelining infrastructure works and SERIAL is
bit-identical, but **does not yet beat baseline** because (a) PARALLEL has
a correctness regression, and (b) even if PARALLEL were correct, Orin's
small 16-SM array doesn't give meaningful GPU-side concurrency between
encoder and decoder kernels — both sides saturate the array, so streams
interleave on the SMs rather than running in true parallel. Estimated
parallel-mode ceiling ≈ 8.5 Hz, still below the 10 Hz target.

## Why PARALLEL drifts

Symptoms: cosine 0.96-0.99 vs baseline (close but not identical),
errors increase over time. SERIAL mode (force decoder→encoder ordering
within the call) recovers bit-identity, ruling out the algorithm.

Suspected root cause: torch's caching allocator partitions per-stream.
The encoder Python code on `enc_stream` calls `.contiguous()` on
attention scratch tensors per layer (e.g., `_enc_O[:, :seq].contiguous()`
when actual `seq < encoder_seq_max`), which allocates from the
`enc_stream` pool. The captured decoder graph on `dec_stream` has raw
addresses baked from the original capture-stream pool. When torch
reassigns the encoder's `.contiguous()` allocations, the addresses
*can* collide with regions referenced by the decoder graph, corrupting
the decoder's state.

## Fix paths (future work)

1. **Pre-allocate encoder FA2 scratch sized to actual `seq_len`** so
   `.contiguous()` is a no-op (no per-call allocation, no pool churn).
   Smallest-touch fix; modify `RtxFlashAttnBackend.__init__` to allocate
   `_enc_O`, `_enc_lse`, etc. at the runtime seq_len once `set_prompt`
   has been called.
2. **Capture the encoder as a CUDA Graph too** — single
   `cudaGraphLaunch` per call eliminates Python-side allocation
   entirely. Per-slot encoder graph captures previously hit similar
   subtle issues; a clean path is to use `torch.cuda.graphs.graph()`
   (allocator-aware) instead of the ctypes-based `CUDAGraph` wrapper.
3. **Pin the captured graph's memory pool** — `cudaMallocFromPoolAsync`
   with a dedicated pool for graph-baked tensors that the encoder's
   pool cannot reach.

## What this branch *does* unblock

Even though the latency win isn't there yet on Orin, the infrastructure
is the prerequisite for several follow-ups:

* Parallel mode correctness fix (any of the three above) → measure
  whether the GPU concurrency yields any real Hz gain on Orin.
* Same code path on **ThorU SM110** (20 SMs + Blackwell FP8) — more SMs
  + faster per-SM compute → concurrent streams *should* overlap; expect
  meaningful Hz gain.
* Skill transfer: dual-stream + per-frame K/V double-buffer is a
  reusable pattern for any encoder-heavy + lightweight-decoder VLA.

## Reproducing

```bash
# Bit-identity check (run baseline first, then pipelined; both INT8)
cd /data/wy/FlashRT
python3 examples/orin/bench_pi05.py --preset lossless --int8 \
    --pipelined --warmup 8 --reps 25
```
