#!/usr/bin/env python3
"""SPARK articulation-only pipeline for existing segmented meshes.

This intentionally excludes the SPARK mesh generation stages: DiT, VAE latent
generation, DINOv2 conditioning, part-image-guided mesh synthesis, and texture
generation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from URDFoptimizer.render.localize_urdf_meshes import localize_urdf_meshes


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SPARK-faithful articulation-only refinement.")
    parser.add_argument("--image", required=True, help="Input RGB image")
    parser.add_argument("--output-dir", required=True, help="Output working directory")
    parser.add_argument("--metadata", default=None, help="Optional existing coarse metadata.json")
    parser.add_argument("--urdf", default=None, help="Optional existing coarse URDF")
    parser.add_argument("--open-image", default=None, help="Optional existing VLM-generated open-state image")
    parser.add_argument("--mesh-dir", default=None, help="Directory containing existing segmented mesh files")
    parser.add_argument("--mesh-map", default=None, help="JSON mapping link labels or semantic part names to mesh paths")
    parser.add_argument("--mesh-pattern", default="part_{index:02d}.glb", help="Segment mesh filename pattern")
    parser.add_argument("--absolute-mesh-paths", action="store_true", help="Write absolute mesh paths into generated URDF")
    parser.add_argument(
        "--localize-meshes",
        action="store_true",
        help="Copy URDF mesh references into output-dir/meshes and rewrite mobility.urdf before optimization",
    )
    parser.add_argument("--localized-mesh-dir", default="meshes", help="Mesh subdirectory used with --localize-meshes")
    parser.add_argument(
        "--localized-mesh-format",
        default="keep",
        choices=["keep", "obj", "stl"],
        help="Mesh format used with --localize-meshes. Use obj or stl for Isaac Sim URDF import",
    )
    parser.add_argument(
        "--add-visual-collisions",
        action="store_true",
        help="Add collision meshes copied from visual meshes when localizing URDF meshes",
    )
    parser.add_argument("--csv", default="./VLMguidance/partnet-mobility-data-analysis.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-vlm-structure", action="store_true", help="Require existing --metadata or --urdf")
    parser.add_argument("--skip-discrete-refinement", action="store_true", help="Do not re-predict frozen joint type/axis")
    parser.add_argument("--skip-open-generation", action="store_true", help="Require existing --open-image")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--camera-radius", type=float, default=4.0)
    parser.add_argument("--camera-fov", type=float, default=40.0)
    parser.add_argument("--camera-azimuth", type=float, default=0.0)
    parser.add_argument("--camera-elevation", type=float, default=0.0)
    parser.add_argument("--camera-y", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--joint-name", action="append", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_image = out_dir / "input.png"
    shutil.copy2(args.image, input_image)

    metadata_path = out_dir / "metadata.json"
    if args.metadata:
        shutil.copy2(args.metadata, metadata_path)
    elif not args.skip_vlm_structure:
        _run(
            [
                sys.executable,
                "VLMguidance/generate_json.py",
                "--input",
                str(input_image),
                "--output",
                str(metadata_path),
                "--csv",
                args.csv,
                "--seed",
                str(args.seed),
            ]
        )

    if args.urdf:
        urdf_path = out_dir / "mobility.urdf"
        if args.localize_meshes:
            summary = localize_urdf_meshes(
                args.urdf,
                output_urdf_path=urdf_path,
                mesh_dir=out_dir / args.localized_mesh_dir,
                absolute_paths=args.absolute_mesh_paths,
                mesh_format=args.localized_mesh_format,
                add_collisions=args.add_visual_collisions,
            )
            print(
                f"Localized {summary['localized_meshes']} mesh file(s) into "
                f"{summary['mesh_dir']} ({summary['copied_meshes']} copied)"
            )
        else:
            shutil.copy2(args.urdf, urdf_path)
    else:
        if not metadata_path.exists():
            raise SystemExit("metadata.json is required to generate a URDF")

        if not args.skip_discrete_refinement:
            _run(
                [
                    sys.executable,
                    "VLMguidance/repredict_axis.py",
                    "--input",
                    str(out_dir),
                    "--image",
                    "input.png",
                    "--single",
                    "--seed",
                    str(args.seed),
                ]
            )

        gen_cmd = [
            sys.executable,
            "VLMguidance/generate_urdf.py",
            "--input-dir",
            str(out_dir),
            "--single",
            "--mesh-pattern",
            args.mesh_pattern,
        ]
        if args.mesh_dir:
            gen_cmd.extend(["--mesh-dir", args.mesh_dir])
        if args.mesh_map:
            gen_cmd.extend(["--mesh-map", args.mesh_map])
        if args.absolute_mesh_paths:
            gen_cmd.append("--absolute-mesh-paths")
        _run(gen_cmd)
        urdf_path = out_dir / "mobility.urdf"

        if args.localize_meshes:
            summary = localize_urdf_meshes(
                urdf_path,
                output_urdf_path=urdf_path,
                mesh_dir=out_dir / args.localized_mesh_dir,
                absolute_paths=args.absolute_mesh_paths,
                mesh_format=args.localized_mesh_format,
                add_collisions=args.add_visual_collisions,
            )
            print(
                f"Localized {summary['localized_meshes']} mesh file(s) into "
                f"{summary['mesh_dir']} ({summary['copied_meshes']} copied)"
            )

    if args.open_image:
        open_image = out_dir / "open.png"
        shutil.copy2(args.open_image, open_image)
    else:
        if args.skip_open_generation:
            raise SystemExit("--open-image is required when --skip-open-generation is set")
        _run([sys.executable, "VLMguidance/generate_prompt_open.py", "--input", str(out_dir), "--seed", str(args.seed)])
        _run([sys.executable, "VLMguidance/generate_image_open.py", "--input", str(out_dir), "--seed", str(args.seed)])
        open_image = out_dir / "open.png"

    opt_cmd = [
        sys.executable,
        "URDFoptimizer/optimize_spark_articulation.py",
        "--urdf",
        str(urdf_path),
        "--open-image",
        str(open_image),
        "--out-urdf",
        str(out_dir / "mobility_refined.urdf"),
        "--out-dir",
        str(out_dir / "URDFoptimize_spark"),
        "--iters",
        str(args.iters),
        "--image-size",
        str(args.image_size),
        "--camera-radius",
        str(args.camera_radius),
        "--camera-fov",
        str(args.camera_fov),
        "--camera-azimuth",
        str(args.camera_azimuth),
        "--camera-elevation",
        str(args.camera_elevation),
        "--target-y",
        str(args.target_y),
    ]
    if args.camera_y is not None:
        opt_cmd.extend(["--camera-y", str(args.camera_y)])
    if args.device:
        opt_cmd.extend(["--device", args.device])
    if args.joint_name:
        for joint_name in args.joint_name:
            opt_cmd.extend(["--joint-name", joint_name])
    _run(opt_cmd)


if __name__ == "__main__":
    main()
