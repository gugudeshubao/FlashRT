#!/usr/bin/env python3
"""FlashRT Orin — LIBERO simulation benchmark.

Evaluates Pi0.5 on LIBERO tasks in MuJoCo simulation.
Supports BF16 baseline vs INT8 quality comparison (same config).

Install dependencies on Orin first:
    pip install mujoco
    pip install robosuite
    pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git

Usage:
    # Quick smoke test (3 tasks × 3 trials), BF16
    python3 examples/orin/eval_libero.py \\
        --checkpoint /data/wy/orin_pi05_droid_pytorch \\
        --task_suite libero_spatial --quick

    # INT8 lossless (7.8 Hz on Orin)
    python3 examples/orin/eval_libero.py \\
        --checkpoint /data/wy/orin_pi05_droid_pytorch \\
        --task_suite libero_spatial --quick --int8

    # INT8 with 2x2 pooling for speed (14 Hz)
    python3 examples/orin/eval_libero.py \\
        --checkpoint /data/wy/orin_pi05_droid_pytorch \\
        --task_suite libero_spatial --quick --int8 --pool 2

    # Full evaluation (10 tasks × 50 trials)
    python3 examples/orin/eval_libero.py \\
        --checkpoint /data/wy/orin_pi05_droid_pytorch \\
        --task_suite libero_spatial --int8

Notes:
    - Uses pi05_droid_pytorch by default (DROID-trained, not LIBERO-trained).
      Expect lower success rates than a LIBERO-finetuned checkpoint, but
      BF16 vs INT8 comparison is still valid.
    - For absolute success rate, use pi05_libero_pytorch checkpoint.
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import sys
import tempfile
import time

import numpy as np

# MuJoCo EGL (off-screen rendering on Orin)
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')


def _patch_egl_cleanup():
    """Patch EGL cleanup to prevent CUDA memory corruption on Jetson unified memory."""
    try:
        import robosuite.renderers.context.egl_context as _egl
        _egl.EGLGLContext.free = lambda self: None
        _egl.EGLGLContext.__del__ = lambda self: None
    except Exception:
        pass
    try:
        import robosuite.utils.binding_utils as _bu
        _bu.MjRenderContext.__del__ = lambda self: None
    except Exception:
        pass


_patch_egl_cleanup()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIBERO_ENV_RESOLUTION = 256
DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS_DICT = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}


def resize_with_pad(img, target_h, target_w):
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    result = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return result


def get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def run_episode(env, model, task_description, max_steps,
                replan_steps=5, num_steps_wait=10, obs=None):
    action_plan = collections.deque()
    if obs is None:
        obs = env.reset()
    t = 0
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, info = env.step(DUMMY_ACTION)
            t += 1
            continue

        if len(action_plan) == 0:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = resize_with_pad(img, 224, 224)
            wrist_img = resize_with_pad(wrist_img, 224, 224)

            actions = model.predict(
                images=[img, wrist_img],
                prompt=task_description,
            )
            action_chunk = actions[:replan_steps]
            action_plan.extend(action_chunk)

        action = action_plan.popleft()
        if hasattr(action, 'tolist'):
            action = action.tolist()
        obs, reward, done, info = env.step(action)
        if done:
            return True
        t += 1
    return False


def load_orin_model(args):
    """Load the Orin-optimized Pi05 model via unified API."""
    import flash_rt

    # INT8 must be set before import of the frontend
    if args.int8:
        os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"
        logger.info("INT8 mode enabled (FVK_PI05_RTX_FORCE_INT8=1)")

    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework="torch",
        num_views=2,
        # Orin-specific performance params (only passed when supported)
        num_steps=args.steps,
        vision_pool_factor=args.pool,
        vision_num_layers=args.layers,
    )
    return model


def eval_single_task(args, task_id):
    from libero.libero import benchmark
    import tqdm

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    max_steps = MAX_STEPS_DICT[args.task_suite]

    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    logger.info(f"Task {task_id}: {task_description}")

    model = load_orin_model(args)

    successes = 0
    latencies = []
    for trial in tqdm.tqdm(range(args.num_trials), desc=f"Task {task_id}"):
        env.reset()
        obs = env.set_init_state(initial_states[trial % len(initial_states)])

        t0 = time.perf_counter()
        success = run_episode(env, model, task_description, max_steps,
                              replan_steps=args.replan_steps, obs=obs)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        if success:
            successes += 1
        logger.info(f"  Trial {trial}: {'SUCCESS' if success else 'FAIL'} ({elapsed:.1f}s)")

    rate = successes / args.num_trials
    env.close()
    logger.info(f"Task {task_id}: {successes}/{args.num_trials} = {rate:.1%}")

    return {
        "task_id": task_id,
        "task_description": task_description,
        "successes": successes,
        "num_trials": args.num_trials,
        "success_rate": rate,
        "mean_episode_time": float(np.mean(latencies)),
    }


def main():
    parser = argparse.ArgumentParser(description="FlashRT Orin LIBERO benchmark")
    parser.add_argument("--checkpoint", default="/data/wy/orin_pi05_droid_pytorch",
                        help="Checkpoint dir (pi05_droid_pytorch or pi05_libero_pytorch)")
    parser.add_argument("--task_suite", default="libero_spatial",
                        choices=list(MAX_STEPS_DICT.keys()))
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 3 tasks × 3 trials")
    parser.add_argument("--output", type=str, default=None)
    # Orin precision / speed options
    parser.add_argument("--int8", action="store_true",
                        help="Enable INT8 (sets FVK_PI05_RTX_FORCE_INT8=1)")
    parser.add_argument("--pool", type=int, default=1, choices=[1, 2, 4],
                        help="vision_pool_factor (1=lossless, 2=2x2, 4=4x4)")
    parser.add_argument("--steps", type=int, default=10,
                        help="ODE steps (10=best quality, 5=faster)")
    parser.add_argument("--layers", type=int, default=27,
                        help="SigLIP layers (27=full/lossless)")
    args = parser.parse_args()

    if args.quick:
        task_end = 3
        args.num_trials = 3
    else:
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite]()
        task_end = task_suite.n_tasks

    if args.output is None:
        mode = "int8" if args.int8 else "bf16"
        args.output = (f"orin_libero_{args.task_suite}_{mode}"
                       f"_pool{args.pool}_steps{args.steps}.json")

    print("=" * 60)
    print("FlashRT Orin LIBERO Benchmark")
    print(f"  Suite:      {args.task_suite}")
    print(f"  Precision:  {'INT8' if args.int8 else 'BF16'}")
    print(f"  pool:       {args.pool}  steps: {args.steps}  layers: {args.layers}")
    print(f"  Tasks:      0..{task_end - 1}")
    print(f"  Trials:     {args.num_trials}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    all_results = []
    total_s, total_e = 0, 0

    for tid in range(task_end):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            out_path = f.name
        try:
            result = eval_single_task(args, tid)
            all_results.append(result)
            total_s += result["successes"]
            total_e += result["num_trials"]
        except Exception as e:
            logger.error(f"Task {tid} failed: {e}")
            all_results.append({"task_id": tid, "error": str(e)})
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    overall_rate = total_s / total_e if total_e > 0 else 0.0
    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "task_suite": args.task_suite,
            "int8": args.int8,
            "pool": args.pool,
            "steps": args.steps,
            "layers": args.layers,
        },
        "overall_success_rate": overall_rate,
        "total_successes": total_s,
        "total_episodes": total_e,
        "tasks": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Overall: {total_s}/{total_e} = {overall_rate:.1%}")
    print(f"Results: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
