#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import csv
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image

# PSNR / SSIM
from skimage.metrics import structural_similarity as ssim

# LPIPS
import torch
import lpips


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB float32 in [0,1], shape (H,W,3)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def load_mask(path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Load mask as float32 in {0,1}, shape (H,W).
    Accepts grayscale/rgba. Resizes to target size with NEAREST.
    """
    m = Image.open(path)
    if m.mode in ["RGBA", "LA"]:
        # Use alpha channel if present
        alpha = m.split()[-1]
        m = alpha
    else:
        m = m.convert("L")

    m = m.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
    m = (np.asarray(m).astype(np.float32) / 255.0)
    m = (m > 0.5).astype(np.float32)
    return m


def resize_to_match(img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize B to match A if shapes differ (using bilinear)."""
    if img_a.shape[:2] == img_b.shape[:2]:
        return img_a, img_b

    ha, wa = img_a.shape[:2]
    b = Image.fromarray((img_b * 255.0).clip(0, 255).astype(np.uint8))
    b = b.resize((wa, ha), resample=Image.BILINEAR)
    img_b2 = np.asarray(b).astype(np.float32) / 255.0
    return img_a, img_b2


def compute_psnr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    PSNR in dB. a,b: float32 [0,1].
    If mask is provided (H,W in {0,1}), compute MSE only on masked pixels.
    """
    diff2 = (a - b) ** 2  # (H,W,3)
    if mask is not None:
        m = mask[..., None]  # (H,W,1)
        denom = float(np.sum(m) * 3.0)
        if denom <= 0:
            return float("nan")
        mse = float(np.sum(diff2 * m) / denom)
    else:
        mse = float(np.mean(diff2))

    if mse <= 1e-12:
        return float("inf")
    maxv = 1.0
    psnr_val = 20.0 * np.log10(maxv / np.sqrt(mse))
    return float(psnr_val)


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (y0,y1,x0,x1) bbox of mask==1. If empty, return None."""
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1


def compute_ssim(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    SSIM in [0,1] (typically). If mask is provided, we compute SSIM on the
    mask bounding box to reduce background influence.
    """
    if mask is not None:
        bb = bbox_from_mask(mask)
        if bb is None:
            return float("nan")
        y0, y1, x0, x1 = bb
        a = a[y0:y1, x0:x1, :]
        b = b[y0:y1, x0:x1, :]

    # skimage SSIM expects data_range for float images
    val = ssim(a, b, channel_axis=2, data_range=1.0)
    return float(val)


@torch.no_grad()
def compute_lpips(a: np.ndarray, b: np.ndarray, lpips_model, device: torch.device,
                  mask: Optional[np.ndarray] = None) -> float:
    """
    LPIPS (lower is better). a,b: float32 [0,1], (H,W,3).
    If mask provided, we apply bbox crop for fairness.
    """
    if mask is not None:
        bb = bbox_from_mask(mask)
        if bb is None:
            return float("nan")
        y0, y1, x0, x1 = bb
        a = a[y0:y1, x0:x1, :]
        b = b[y0:y1, x0:x1, :]

    # Convert to torch tensor in [-1,1], shape (1,3,H,W)
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(device)
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).to(device)
    ta = ta * 2.0 - 1.0
    tb = tb * 2.0 - 1.0
    d = lpips_model(ta, tb)
    return float(d.item())


def list_pairs(render_path: str, input_path: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (name, render_file, input_file).
    If both are files: single pair.
    If both are dirs: match by filename intersection.
    """
    if os.path.isfile(render_path) and os.path.isfile(input_path):
        name = os.path.splitext(os.path.basename(render_path))[0]
        return [(name, render_path, input_path)]

    if not (os.path.isdir(render_path) and os.path.isdir(input_path)):
        raise ValueError("render_path and input_path must both be files or both be directories.")

    def is_img(fn: str) -> bool:
        ext = os.path.splitext(fn.lower())[1]
        return ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

    r_files = {fn: os.path.join(render_path, fn) for fn in os.listdir(render_path) if is_img(fn)}
    i_files = {fn: os.path.join(input_path, fn) for fn in os.listdir(input_path) if is_img(fn)}

    common = sorted(set(r_files.keys()) & set(i_files.keys()))
    pairs = []
    for fn in common:
        name = os.path.splitext(fn)[0]
        pairs.append((name, r_files[fn], i_files[fn]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True, help="Rendered image path OR directory")
    ap.add_argument("--input", required=True, help="Input/GT image path OR directory")
    ap.add_argument("--mask", default=None,
                    help="Optional mask path OR directory (same filenames). White=foreground.")
    ap.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"],
                    help="LPIPS backbone. vgg is common.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Device for LPIPS.")
    ap.add_argument("--out_csv", default="metrics.csv", help="Output CSV filename")
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_model.eval()

    pairs = list_pairs(args.render, args.input)
    if len(pairs) == 0:
        raise RuntimeError("No matched image pairs found.")

    rows: List[Dict[str, float]] = []
    for name, r_path, i_path in pairs:
        r = load_image_rgb(r_path)
        gt = load_image_rgb(i_path)
        r, gt = resize_to_match(r, gt)

        m = None
        if args.mask is not None:
            if os.path.isfile(args.mask):
                mpath = args.mask
            else:
                # directory mask: match exact filename
                fn = os.path.basename(r_path)
                mpath = os.path.join(args.mask, fn)
                if not os.path.exists(mpath):
                    mpath = None
            if mpath is not None and os.path.exists(mpath):
                m = load_mask(mpath, target_hw=r.shape[:2])

        psnr_val = compute_psnr(r, gt, mask=m)
        ssim_val = compute_ssim(r, gt, mask=m)
        lpips_val = compute_lpips(r, gt, lpips_model, device=device, mask=m)

        rows.append({
            "name": name,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
        })
        print(f"[{name}] PSNR={psnr_val:.4f} dB | SSIM={ssim_val:.4f} | LPIPS={lpips_val:.4f}")

    # Summary
    psnrs = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
    ssims = [r["ssim"] for r in rows if np.isfinite(r["ssim"])]
    lps   = [r["lpips"] for r in rows if np.isfinite(r["lpips"])]

    def mean(x): return float(np.mean(x)) if len(x) else float("nan")

    print("\n=== SUMMARY ===")
    print(f"Mean PSNR : {mean(psnrs):.4f} dB")
    print(f"Mean SSIM : {mean(ssims):.4f}")
    print(f"Mean LPIPS: {mean(lps):.4f}")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim", "lpips"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
