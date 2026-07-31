#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for part-image inference with selected seeds.
Edits you may want to make are all in the CONSTANTS section below.
"""

import os
import subprocess
import sys

# =========================
# ====== CONSTANTS ========
# =========================

PYTHON_BIN = sys.executable
INFERENCE_SCRIPT = "scripts/inference/part_images.py"

IMAGE_FOLDER = "/nas/yumenghe/comparison/realimage/10"
OUTPUT_DIR = "/nas/yumenghe/comparison/realimage/10/inference"
TAG_PREFIX = "FT5_G1"

NUM_INFERENCE_STEPS = 1000
GUIDANCE_SCALE = 1.0
RENDER = True
RMBG = False
USE_FLASH_DECODER = False
NUM_TOKENS = None
MAX_NUM_EXPANDED_COORDS = None

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 3407]

# =========================
# ===== END CONSTANTS =====
# =========================


def build_cmd(seed: int) -> list:
    tag = f"{TAG_PREFIX}_{seed}"
    cmd = [
        PYTHON_BIN,
        INFERENCE_SCRIPT,
        "--image_folder", IMAGE_FOLDER,
        "--output_dir", OUTPUT_DIR,
        "--tag", tag,
        "--seed", str(seed),
        "--num_inference_steps", str(NUM_INFERENCE_STEPS),
        "--guidance_scale", str(GUIDANCE_SCALE),
    ]
    if RENDER:
        cmd.append("--render")
    if RMBG:
        cmd.append("--rmbg")
    if USE_FLASH_DECODER:
        cmd.append("--use_flash_decoder")
    if NUM_TOKENS is not None:
        cmd.extend(["--num_tokens", str(NUM_TOKENS)])
    if MAX_NUM_EXPANDED_COORDS is not None:
        cmd.extend(["--max_num_expanded_coords", str(MAX_NUM_EXPANDED_COORDS)])
    return cmd


def run_once(seed: int, verbose: bool = True) -> tuple[int, int]:
    tag = f"{TAG_PREFIX}_{seed}"
    cmd = build_cmd(seed)

    if verbose:
        print("=" * 80)
        print(f"[RUN] seed={seed}, tag={tag}")
        print("      CMD:", " ".join(cmd))
        print(f"      image_folder={IMAGE_FOLDER}")
        print(f"      output_dir  ={OUTPUT_DIR}")

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    try:
        subprocess.run(cmd, env=env, check=True, text=True)
        if verbose:
            print(f"[OK ] seed={seed} finished.")
        return seed, 0
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[ERR] seed={seed} failed with return code {e.returncode}.")
        return seed, e.returncode
    except FileNotFoundError as e:
        if verbose:
            print(f"[ERR] Missing file or executable: {e}")
        return seed, 127


def main():
    print("=" * 80)
    print("Running inference with selected seeds")
    print("=" * 80)
    print(f"Total seeds to run: {len(SEEDS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Image folder: {IMAGE_FOLDER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Error: Image folder not found: {IMAGE_FOLDER}")
        sys.exit(1)

    successful = []
    failed = []

    for index, seed in enumerate(SEEDS, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing seed {index}/{len(SEEDS)}: {seed}")
        print(f"{'=' * 80}\n")

        _, return_code = run_once(seed, verbose=True)
        if return_code == 0:
            successful.append(seed)
        else:
            failed.append(seed)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total seeds: {len(SEEDS)}")
    print(f"Successful: {len(successful)} - {successful}")
    print(f"Failed: {len(failed)} - {failed}")
    print("=" * 80)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
