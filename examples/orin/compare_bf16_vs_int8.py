#!/usr/bin/env python3
"""FlashRT Orin — BF16 vs INT8 action output comparison.

Quantifies the numerical difference between BF16 and INT8 inference
on the same input, without requiring MuJoCo/LIBERO/robosuite.

Metrics:
  - cosine_similarity: 1.0 = identical direction, 0.0 = orthogonal
  - l2_norm_ratio: ||int8 - bf16|| / ||bf16||, lower is better
  - max_abs_diff: per-DoF max absolute action difference

Usage:
    python3 examples/orin/compare_bf16_vs_int8.py \\
        --checkpoint /data/wy/orin_pi05_droid_pytorch \\
        --n-images 10 \\
        --prompt "pick up the black bowl"
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def run_inference(checkpoint, int8: bool, images, prompts, steps=10, seed=42):
    """Run inference with given precision and fixed random seed."""
    import logging
    logging.basicConfig(level=logging.WARNING)

    if int8:
        os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"
    else:
        os.environ.pop("FVK_PI05_RTX_FORCE_INT8", None)

    import torch
    import flash_rt

    pipe = flash_rt.load_model(
        checkpoint=checkpoint,
        framework="torch",
        num_views=2,
        num_steps=steps,
    )

    results = []
    latencies = []
    for i, (img_pair, prompt) in enumerate(zip(images, prompts)):
        # Fix the random noise seed so BF16 and INT8 start from identical noise.
        # The diffusion denoiser starts from this noise — without a fixed seed,
        # results are incomparable even for the same inputs.
        torch.manual_seed(seed + i)

        t0 = time.perf_counter()
        actions = pipe.predict(images=img_pair, prompt=prompt)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append(actions)
        if i == 0:
            print(f"  Sample 0 actions[0]: {actions[0][:4]}")

    print(f"  Median latency (excl. first): {sorted(latencies[1:])[len(latencies)//2]:.1f} ms")
    return np.array(results), np.median(latencies[1:])  # exclude first (calibration overhead)


def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    return float(np.dot(a_flat, b_flat) /
                 (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/data/wy/orin_pi05_droid_pytorch")
    p.add_argument("--n-images", type=int, default=10,
                   help="Number of random test images")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--prompt", default="pick up the black bowl between the plate and the ramekin and place it on the plate")
    p.add_argument("--pool", type=int, default=1, help="vision_pool_factor for INT8 run")
    args = p.parse_args()

    np.random.seed(42)
    images = [
        [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
         np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
        for _ in range(args.n_images)
    ]
    prompts = [args.prompt] * args.n_images

    print("=" * 60)
    print(f"FlashRT Orin — BF16 vs INT8 Action Comparison")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Steps:      {args.steps}")
    print(f"  N images:   {args.n_images}")
    print(f"  Prompt:     {args.prompt[:50]}...")
    print("=" * 60)

    print("\n[1/2] Running BF16 inference (fixed seed per sample)...")
    actions_bf16, bf16_ms = run_inference(args.checkpoint, int8=False,
                                           images=images, prompts=prompts,
                                           steps=args.steps, seed=42)

    print("\n[2/2] Running INT8 inference (same fixed seed per sample)...")
    actions_int8, int8_ms = run_inference(args.checkpoint, int8=True,
                                           images=images, prompts=prompts,
                                           steps=args.steps, seed=42)

    # Comparison metrics
    print("\n" + "=" * 60)
    print("Comparison Results:")
    print("-" * 60)

    cos_sims = []
    l2_ratios = []
    max_diffs = []

    for i in range(args.n_images):
        b = actions_bf16[i].astype(np.float64)
        q = actions_int8[i].astype(np.float64)
        cos = cosine_sim(b, q)
        l2r = float(np.linalg.norm(b - q) / (np.linalg.norm(b) + 1e-12))
        md = float(np.abs(b - q).max())
        cos_sims.append(cos)
        l2_ratios.append(l2r)
        max_diffs.append(md)

    print(f"  cosine_similarity:  {np.mean(cos_sims):.6f} ± {np.std(cos_sims):.6f}  (1.0=identical)")
    print(f"  l2_norm_ratio:      {np.mean(l2_ratios):.6f} ± {np.std(l2_ratios):.6f}  (0.0=identical)")
    print(f"  max_abs_diff:       {np.mean(max_diffs):.6f} ± {np.std(max_diffs):.6f}  (per-sample max)")
    print(f"  per-DoF mean diff:  {np.abs(actions_bf16 - actions_int8).mean(axis=(0,1))}")
    print()
    print(f"  BF16 median:   {bf16_ms:.1f} ms")
    print(f"  INT8 median:   {int8_ms:.1f} ms  ({bf16_ms/int8_ms:.2f}x speedup)")
    print("=" * 60)

    # Interpretation
    mean_cos = np.mean(cos_sims)
    if mean_cos > 0.999:
        interp = "EXCELLENT — INT8 output virtually identical to BF16"
    elif mean_cos > 0.99:
        interp = "GOOD — Very small quantization error"
    elif mean_cos > 0.95:
        interp = "ACCEPTABLE — Small but noticeable error"
    else:
        interp = "SIGNIFICANT — Quantization error may affect policy"
    print(f"\nVerdict: {interp}")
    print(f"  (cosine={mean_cos:.6f})")


if __name__ == "__main__":
    main()
