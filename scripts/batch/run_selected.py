#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for PartCrafter inference with parallel execution support.
Edits you may want to make are all in the CONSTANTS section below.
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
import time

# =========================
# ====== CONSTANTS ========
# =========================

# GPU to use
CUDA_VISIBLE_DEVICES = "0,1,2,3"

# Python executable (leave as "python" unless you need a specific env)
PYTHON_BIN = "python"

# Inference script
INFERENCE_SCRIPT = "scripts/inference/part_images.py"

# Base folder that contains all selected/<ID> directories
BASE_SELECTED_DIR = Path("/nas/yumenghe/comparison/quantitative")

# Inference common args
TAG = "FT5_1"
SEED = 1
NUM_INFERENCE_STEPS = 1000
GUIDANCE_SCALE = 1.0
RENDER = True  # add --render if True

# Behavior controls
CREATE_OUTPUT_DIR = True    # create output dir if it doesn't exist
STOP_ON_ERROR = False       # if True, stop when any job fails

# Parallel execution settings
ENABLE_PARALLEL = True      # if True, run jobs in parallel
MAX_PARALLEL_JOBS = 8       # max number of parallel jobs (adjust based on GPU memory)

IMAGE_SUBDIR = None 
# IMAGE_SUBDIR = "gt"

# IDs to run (replace/modify as needed)
SELECT = [
    "100162","100202",
    "100448","100473",
    "101362","102834",
    "103048","103105",
    "12092","12540",
    "8867","8897",
    "100021","100051",
    "10090","10101",
    "7221","7236",
    "101773","101931",
    "103863","103872",
    "10627","10900",
    "100828","101011",
    "101593","102301",
    "35059","44817",
    "100550","100825",
    "20411","25913",
    "103477","103482",
    "101323","102632",
    "101377","102155",
    "103518","103521",
]

# =========================
# ===== END CONSTANTS =====
# =========================


def build_cmd(image_dir: Path, output_dir: Path) -> list:
    cmd = [
        PYTHON_BIN,
        INFERENCE_SCRIPT,
        "--image_folder", str(image_dir),
        "--output_dir", str(output_dir),
        "--tag", str(TAG),
        "--seed", str(SEED),
        "--num_inference_steps", str(NUM_INFERENCE_STEPS),
        "--guidance_scale", str(GUIDANCE_SCALE),
    ]
    if RENDER:
        cmd.append("--render")
    return cmd


def run_once(case_id: str, gpu_id: str = None, verbose: bool = True) -> Tuple[str, int]:
    """
    Run inference for a single case on a specific GPU.

    Args:
        case_id: Case ID to process
        gpu_id: GPU ID to use (if None, uses CUDA_VISIBLE_DEVICES as-is)
        verbose: Whether to print detailed output

    Returns:
        Tuple[str, int]: (case_id, return_code)
    """
    base_dir = BASE_SELECTED_DIR / case_id
    image_dir = base_dir / IMAGE_SUBDIR if IMAGE_SUBDIR else base_dir
    output_dir = base_dir / "inference"

    if CREATE_OUTPUT_DIR:
        output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(image_dir, output_dir)

    # Prepare env with desired CUDA device
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    if verbose:
        print("=" * 80)
        print(f"[RUN] ID={case_id}")
        print(f"      CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        print("      CMD:", " ".join(cmd))
        print(f"      image_folder={image_dir}")
        print(f"      output_dir  ={output_dir}")

    try:
        # Suppress output in parallel mode to avoid clutter
        stdout = None if verbose else subprocess.DEVNULL
        stderr = None if verbose else subprocess.DEVNULL
        subprocess.run(cmd, check=True, env=env, stdout=stdout, stderr=stderr)
        if verbose:
            print(f"[OK ] ID={case_id} finished.")
        return (case_id, 0)
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[ERR] ID={case_id} failed with return code {e.returncode}.")
        return (case_id, e.returncode)
    except FileNotFoundError as e:
        if verbose:
            print(f"[ERR] Missing file or executable: {e}")
        return (case_id, 127)


def main():
    # Parse GPU list
    gpu_list = [gpu.strip() for gpu in CUDA_VISIBLE_DEVICES.split(',') if gpu.strip()]
    num_gpus = len(gpu_list)

    print("=" * 80)
    print("PartCrafter Batch Inference")
    print("=" * 80)
    print(f"Total jobs: {len(SELECT)}")
    print(f"Parallel mode: {'ENABLED' if ENABLE_PARALLEL else 'DISABLED'}")
    if ENABLE_PARALLEL:
        print(f"Max parallel jobs: {MAX_PARALLEL_JOBS}")
        print(f"Available GPUs: {gpu_list} ({num_gpus} GPUs)")
        if MAX_PARALLEL_JOBS > num_gpus:
            print(f"⚠️  Warning: MAX_PARALLEL_JOBS ({MAX_PARALLEL_JOBS}) > num_gpus ({num_gpus})")
            print(f"   Multiple jobs will share GPUs")
    else:
        print(f"GPU: {CUDA_VISIBLE_DEVICES}")
    print("=" * 80)
    print()

    start_time = time.time()
    failed_ids = []

    if ENABLE_PARALLEL:
        # Parallel execution with round-robin GPU assignment
        print(f"Starting parallel execution with {MAX_PARALLEL_JOBS} workers...\n")

        completed = 0
        with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
            # Submit all jobs with GPU assignment (round-robin)
            future_to_id = {}
            for idx, cid in enumerate(SELECT):
                # Assign GPU using round-robin
                gpu_id = gpu_list[idx % num_gpus]
                future = executor.submit(run_once, cid, gpu_id, False)
                future_to_id[future] = (cid, gpu_id)

            # Process completed jobs as they finish
            for future in as_completed(future_to_id):
                case_id, gpu_id = future_to_id[future]
                result_case_id, return_code = future.result()
                completed += 1

                if return_code == 0:
                    print(f"[{completed}/{len(SELECT)}] ✓ {case_id} completed (GPU {gpu_id})")
                else:
                    print(f"[{completed}/{len(SELECT)}] ✗ {case_id} failed (GPU {gpu_id}, code={return_code})")
                    failed_ids.append(case_id)

                    if STOP_ON_ERROR:
                        print("\n[STOP_ON_ERROR] Cancelling remaining jobs...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        # Sequential execution (original behavior)
        print("Starting sequential execution...\n")

        for i, cid in enumerate(SELECT, 1):
            print(f"\n[{i}/{len(SELECT)}] Processing {cid}...")
            case_id, return_code = run_once(cid, gpu_id=None, verbose=True)

            if return_code != 0:
                failed_ids.append(case_id)
                if STOP_ON_ERROR:
                    print("\n[STOP_ON_ERROR] Halting due to failure.")
                    break

    elapsed_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total jobs: {len(SELECT)}")
    print(f"Successful: {len(SELECT) - len(failed_ids)}")
    print(f"Failed: {len(failed_ids)}")
    print(f"Elapsed time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
    if len(SELECT) - len(failed_ids) > 0:
        print(f"Average time per job: {elapsed_time / (len(SELECT) - len(failed_ids)):.1f}s")

    if failed_ids:
        print("\nFailed IDs:")
        for fid in failed_ids:
            print(f"  - {fid}")
        print("\nYou can copy this list:")
        print(f"  {failed_ids}")
    else:
        print("\n🎉 All jobs completed successfully!")


if __name__ == "__main__":
    main()
