"""Command line interface for articulation-only SPARK refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .optimizer import OptimizationConfig, optimize_urdf_against_open_silhouette
from .renderer import RendererConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SPARK-faithful articulation-only URDF refinement. Uses existing segmented "
            "meshes referenced by URDF; no mesh, texture, camera, axis, or joint-type optimization."
        )
    )
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing mobility.urdf and open.png")
    parser.add_argument("--urdf", type=str, default=None, help="Path to coarse URDF")
    parser.add_argument("--open-image", type=str, default=None, help="VLM-generated open-state image")
    parser.add_argument("--target-mask", type=str, default=None, help="Optional binary open-state mask")
    parser.add_argument("--out-urdf", type=str, default=None, help="Output refined URDF path")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for debug silhouettes and summary JSON")
    parser.add_argument("--joint-name", action="append", default=None, help="Revolute joint to optimize; repeatable")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr-origin", type=float, default=5e-3)
    parser.add_argument("--lr-angle", type=float, default=1e-2)
    parser.add_argument("--origin-reg", type=float, default=1e-3)
    parser.add_argument("--angle-reg", type=float, default=1e-4)
    parser.add_argument("--region-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--unit-scale", type=float, default=1.0)
    parser.add_argument("--default-open-angle-deg", type=float, default=120.0)
    parser.add_argument("--init-angle-deg", type=float, default=None)
    parser.add_argument("--save-debug-every", type=int, default=0)
    parser.add_argument("--no-auto-normalize", action="store_true")
    parser.add_argument("--no-preserve-zero-pose", action="store_true")

    parser.add_argument("--camera-radius", type=float, default=4.0)
    parser.add_argument("--camera-fov", type=float, default=40.0)
    parser.add_argument("--camera-elevation", type=float, default=0.0)
    parser.add_argument("--camera-azimuth", type=float, default=0.0)
    parser.add_argument("--camera-y", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
        args.urdf = args.urdf or str(input_dir / "mobility.urdf")
        args.open_image = args.open_image or str(input_dir / "open.png")
        args.out_dir = args.out_dir or str(input_dir / "URDFoptimize_spark")
        args.out_urdf = args.out_urdf or str(input_dir / "mobility_refined.urdf")

    if args.urdf is None:
        raise SystemExit("--urdf is required unless --input-dir is provided")
    if args.open_image is None and args.target_mask is None:
        raise SystemExit("--open-image or --target-mask is required")

    config = OptimizationConfig(
        image_size=args.image_size,
        threshold=args.threshold,
        iterations=args.iters,
        lr_origin=args.lr_origin,
        lr_angle=args.lr_angle,
        region_weight=args.region_weight,
        edge_weight=args.edge_weight,
        origin_reg_weight=args.origin_reg,
        angle_reg_weight=args.angle_reg,
        unit_scale=args.unit_scale,
        default_open_angle_deg=args.default_open_angle_deg,
        save_debug_every=args.save_debug_every,
        auto_normalize=not args.no_auto_normalize,
        preserve_zero_pose=not args.no_preserve_zero_pose,
    )
    renderer_config = RendererConfig(
        image_size=args.image_size,
        radius=args.camera_radius,
        fov=args.camera_fov,
        elevation=args.camera_elevation,
        azimuth=args.camera_azimuth,
        camera_y=args.camera_y,
        target_y=args.target_y,
    )

    result = optimize_urdf_against_open_silhouette(
        urdf_path=args.urdf,
        open_image_path=args.open_image,
        target_mask_path=args.target_mask,
        output_urdf_path=args.out_urdf,
        output_dir=args.out_dir,
        joint_names=args.joint_name,
        initial_angle_deg=args.init_angle_deg,
        config=config,
        renderer_config=renderer_config,
        device=args.device,
    )

    summary = {
        "final_loss": result.final_loss,
        "origin_deltas": result.origin_deltas,
        "open_state_joint_values_rad": result.joint_values,
        "output_urdf": str(result.output_urdf) if result.output_urdf is not None else None,
        "baseline_constraints": {
            "optimized": ["revolute joint origin offsets", "open-state revolute joint angles"],
            "frozen": ["joint type", "joint axis", "camera pose", "camera intrinsics", "mesh vertices"],
            "losses": ["silhouette region overlap", "edge alignment", "origin regularization", "angle regularization"],
            "excluded": ["RGB photometric loss", "multi-state supervision", "mesh generation", "texture generation"],
        },
    }
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "optimization_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
