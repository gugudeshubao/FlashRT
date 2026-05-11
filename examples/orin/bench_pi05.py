#!/usr/bin/env python3
"""FlashRT — Pi0.5 Orin benchmark script.

Measures steady-state inference latency across configurable
num_views / vision_pool_factor / vision_num_layers / num_steps.

Usage:
    FVK_PI05_RTX_FORCE_INT8=1 python3 examples/orin/bench_pi05.py \
        --checkpoint /data/wy/orin_pi05_droid_pytorch \
        --num-views 2 \
        --pool 1 \
        --steps 10

Quick presets (pool=1 is lossless, pool>1 trades spatial detail for speed):
    --preset lossless    → 2cam pool=1 steps=10  (~132ms / 7.6 Hz)
    --preset balanced    → 2cam pool=1 steps=5   (~110ms / 9.1 Hz)
    --preset fast        → 2cam pool=2 steps=5   (~74ms  / 13.5 Hz)
    --preset fastest     → 1cam pool=4 steps=3   (~41ms  / 24.3 Hz)
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------

PRESETS = {
    "lossless":  dict(num_views=2, pool=1, layers=27, steps=10),
    "balanced":  dict(num_views=2, pool=1, layers=27, steps=5),
    "fast":      dict(num_views=2, pool=2, layers=27, steps=5),
    "faster":    dict(num_views=2, pool=4, layers=27, steps=3),
    "fastest":   dict(num_views=1, pool=4, layers=27, steps=3),
}


def parse_args():
    p = argparse.ArgumentParser(description="FlashRT Pi0.5 Orin benchmark")
    p.add_argument("--checkpoint", "-c",
                   default="/data/wy/orin_pi05_droid_pytorch",
                   help="Path to pi05_droid_pytorch checkpoint")
    p.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="Quick config preset (overrides individual flags)")
    p.add_argument("--num-views", type=int, default=2, choices=[1, 2])
    p.add_argument("--pool", type=int, default=1, choices=[1, 2, 4],
                   help="vision_pool_factor (1=no pool, 2=2x2, 4=4x4)")
    p.add_argument("--layers", type=int, default=27,
                   help="SigLIP layers to run (1-27, default=27 for lossless)")
    p.add_argument("--steps", type=int, default=10,
                   help="Diffusion ODE steps (default=10 for best quality)")
    p.add_argument("--prompt", default="pick up the black envelope on the table")
    p.add_argument("--warmup", type=int, default=8,
                   help="Warmup iterations before measurement")
    p.add_argument("--reps", type=int, default=15,
                   help="Measurement iterations")
    p.add_argument("--int8", action="store_true", default=False,
                   help="Enable INT8 encoder/decoder/vision (sets FVK_PI05_RTX_FORCE_INT8=1)")
    p.add_argument("--no-int8", dest="int8", action="store_false",
                   help="Disable INT8, run BF16 (default)")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply preset if given
    if args.preset:
        cfg = PRESETS[args.preset]
        args.num_views = cfg["num_views"]
        args.pool      = cfg["pool"]
        args.layers    = cfg["layers"]
        args.steps     = cfg["steps"]
        print(f"Preset '{args.preset}': "
              f"num_views={args.num_views} pool={args.pool} "
              f"layers={args.layers} steps={args.steps}")

    # Enforce INT8 env var
    if args.int8 and not os.environ.get("FVK_PI05_RTX_FORCE_INT8"):
        os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"
        print("Auto-set FVK_PI05_RTX_FORCE_INT8=1")

    import logging
    logging.basicConfig(level=logging.WARNING)

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)

    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    print(f"\nLoading checkpoint: {args.checkpoint}")
    pipe = Pi05TorchFrontendRtx(
        args.checkpoint,
        num_views=args.num_views,
        num_steps=args.steps,
        vision_pool_factor=args.pool,
        vision_num_layers=args.layers,
    )

    pipe.set_prompt(args.prompt)

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Always include wrist_image — _stack_images slices to [:num_views] internally,
    # so the extra key is harmless when num_views=1.
    obs = {"image": img, "wrist_image": img}

    print("Calibrating…")
    pipe.calibrate_with_real_data([obs])

    print(f"Warmup ({args.warmup} iters)…")
    for _ in range(args.warmup):
        pipe.infer(obs)

    print(f"Measuring ({args.reps} iters)…")
    lat = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = pipe.infer(obs)
        lat.append((time.perf_counter() - t0) * 1000)

    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[int(len(lat) * 0.95)]

    print()
    print("=" * 55)
    print(f"  Config : num_views={args.num_views}  pool={args.pool}"
          f"  layers={args.layers}  steps={args.steps}")
    print(f"  enc_seq: {pipe.pipeline.encoder_seq_len}")
    print(f"  Actions: {out['actions'].shape}")
    print(f"  p50    : {p50:.1f} ms   → {1000/p50:.2f} Hz")
    print(f"  p95    : {p95:.1f} ms")
    print(f"  min    : {lat[0]:.1f} ms   → {1000/lat[0]:.2f} Hz")
    print("=" * 55)

    stats = pipe.get_latency_stats()
    if stats:
        print(f"  Pipeline p50: {stats['p50_ms']:.1f} ms")


if __name__ == "__main__":
    main()
