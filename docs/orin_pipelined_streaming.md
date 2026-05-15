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
| **Pipelined SERIAL** (default) | 131.1 ms | 7.63 | **1.000 ✓ (maxabs=0)** |
| Pipelined PARALLEL (opt-in) — after `_enc_O` fix | 120.2 ms | 8.32 | ~0.96-0.99 ✗ |
| `cache_frames=2` (existing) | 83.3 ms | 12.01 | 0.991 (1-frame K/V stale) |

**Conclusion**: pipelining infrastructure works and SERIAL is
bit-identical (cosine=1.000, maxabs=0), but PARALLEL **caps at 8.32 Hz
in measurement** — Orin's 16 SMs are saturated by either side alone, so
streams interleave on the SM array rather than running in true parallel.
Even *if* the cosine drift in PARALLEL were eliminated, the wall-time
ceiling sits at ~8.3-8.5 Hz, **still below the 10 Hz target**. Cross-call
pipelining on Orin alone is therefore not a viable lossless path to 10 Hz.

## Why PARALLEL drifts

Symptoms: cosine 0.96-0.99 vs baseline (close but not identical),
errors compound over a few frames. SERIAL mode (force decoder→encoder
ordering within the call) recovers bit-identity, ruling out the
algorithm and per-frame K/V plumbing.

**Initially suspected**: torch's per-stream caching allocator
reassigning `_enc_O[:, :seq].contiguous()` addresses that collide with
the captured decoder graph's baked addresses (since `_enc_O` was sized
to `encoder_seq_max=560` but actual `seq=522`, `.contiguous()` did
allocate fresh memory each layer).

**Tried and verified harmless**: `_init_pipelined_state` now resizes
`attn._enc_O` to exactly `encoder_seq_len` so `[:, :seq]` is the full
tensor (already contiguous, no per-call allocation). The fix is
landed — but PARALLEL cosine drift is *unchanged*. So allocator churn
was not the root cause.

**Updated hypothesis**: non-deterministic floating-point reductions in
FA2 split-KV when encoder and decoder kernels interleave on the same
SM array. FA2 uses online softmax with split-KV accumulator buffers
(`_enc_lse_accum`, `_enc_o_accum`); the cross-split reduction step uses
GPU atomics whose ordering depends on SM scheduling. When the encoder
runs in isolation (SERIAL), the scheduling is reproducible. When the
encoder and decoder kernels interleave on Orin's 16 SMs (PARALLEL),
the encoder's reductions land in different orders → tiny K/V
perturbations → amplified across the decoder's 10 ODE steps to
cosine 0.96-0.99 visible drift.

This is harder to fix than the allocator theory — it's not a memory
race, it's a fundamental property of concurrent kernel execution with
non-deterministic atomics.

## Fix paths (future work)

1. ~~**Pre-allocate encoder FA2 scratch**~~ (landed — no measurable
   improvement; root cause was not allocator churn).
2. **Disable split-KV in encoder FA2 when in pipelined mode**. With
   `lse_accum=None, o_accum=None` the FA2 wrapper falls back to a
   single-block kernel without atomic cross-split reductions →
   deterministic regardless of co-tenant kernel activity. Cost: may
   reduce SigLIP throughput slightly when split-KV would otherwise
   help; needs re-measurement.
3. **Capture the encoder as a CUDA Graph too** — single
   `cudaGraphLaunch` per call. The captured kernels have fixed launch
   ordering and the GPU's command buffer schedules them deterministically
   relative to other graphs. (Per-slot encoder graphs previously hit
   subtle capture-time issues; a `torch.cuda.graphs.graph()`
   allocator-aware path is cleaner than the ctypes-based `CUDAGraph`
   wrapper.)
4. **Pin the captured graph's memory pool** — `cudaMallocFromPoolAsync`
   with a dedicated pool for graph-baked tensors that the encoder's
   pool cannot reach. (Defensive; not needed if root cause is FA2
   atomics, but cheap insurance.)

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
