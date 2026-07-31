#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for PartCrafter inference.
Edits you may want to make are all in the CONSTANTS section below.
"""

import os
import subprocess
from pathlib import Path

# =========================
# ====== CONSTANTS ========
# =========================

# GPU to use
CUDA_VISIBLE_DEVICES = "7"

# Python executable (leave as "python" unless you need a specific env)
PYTHON_BIN = "python"

# Inference script
INFERENCE_SCRIPT = "scripts/inference/part_images.py"

# Base folder that contains all selected/<ID> directories
BASE_SELECTED_DIR = Path("/nas/yumenghe/comparison/GAPartNet_PartNetMobility/selected")

# Inference common args
TAG = "FT13_10086"
SEED = 10086
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 1.0
RENDER = True  # add --render if True

# Behavior controls
CREATE_OUTPUT_DIR = True    # create output dir if it doesn't exist
STOP_ON_ERROR = False       # if True, stop when any job fails

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


def run_once(case_id: str) -> int:
    image_dir = BASE_SELECTED_DIR / case_id
    output_dir = image_dir / "inference"

    if CREATE_OUTPUT_DIR:
        output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(image_dir, output_dir)

    # Prepare env with desired CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    print("=" * 80)
    print(f"[RUN] ID={case_id}")
    print(f"      CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print("      CMD:", " ".join(cmd))
    print(f"      image_folder={image_dir}")
    print(f"      output_dir  ={output_dir}")

    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"[OK ] ID={case_id} finished.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[ERR] ID={case_id} failed with return code {e.returncode}.")
        return e.returncode
    except FileNotFoundError as e:
        print(f"[ERR] Missing file or executable: {e}")
        return 127


def main():
    failed_ids = []
    for cid in SELECT:
        rc = run_once(cid)
        if rc != 0:
            failed_ids.append(cid)
            if STOP_ON_ERROR:
                print("[STOP_ON_ERROR] Halting due to failure.")
                break

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total jobs: {len(SELECT)}")
    print(f"Successful: {len(SELECT) - len(failed_ids)}")
    print(f"Failed: {len(failed_ids)}")

    if failed_ids:
        print("\nFailed IDs:")
        for fid in failed_ids:
            print(f"  - {fid}")
        print("\nYou can copy this list:")
        print(f"  {failed_ids}")
    else:
        print("\nAll jobs completed successfully!")


if __name__ == "__main__":
    main()
