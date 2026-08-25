#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a GLB file into separate GLB files for each part/geometry.

This script:
1. Loads a GLB file containing multiple geometries (parts)
2. Extracts each geometry as a separate mesh
3. Saves each part as an individual GLB file

Usage:
    python split_glb.py --input input/textured.glb --output output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def _safe_name(name: str) -> str:
    return name.replace('/', '_').replace('\\', '_').replace(' ', '_')


def _ensure_rgba_vertex_colors(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    trimesh's GLB exporter expects vertex colors as RGBA. Some GLBs load as
    RGB vertex colors, which later fails with "cannot reshape ... into shape
    (4)". Keep geometry unchanged and only normalize color channel count.
    """
    import numpy as np
    import trimesh

    visual = getattr(mesh, "visual", None)
    colors = getattr(visual, "vertex_colors", None)
    if colors is None:
        return mesh

    colors = np.asarray(colors)
    if colors.size == 0:
        return mesh

    num_vertices = len(mesh.vertices)
    if colors.ndim == 1:
        if colors.size == num_vertices * 3:
            colors = colors.reshape((num_vertices, 3))
        elif colors.size == num_vertices * 4:
            colors = colors.reshape((num_vertices, 4))
        else:
            return mesh

    if colors.ndim != 2 or colors.shape[0] != num_vertices:
        return mesh

    if colors.shape[1] == 3:
        alpha_value = 1.0 if np.issubdtype(colors.dtype, np.floating) and colors.max(initial=0.0) <= 1.0 else 255
        alpha = np.full((num_vertices, 1), alpha_value, dtype=colors.dtype)
        colors = np.concatenate([colors, alpha], axis=1)
    elif colors.shape[1] != 4:
        return mesh

    if colors.dtype != np.uint8:
        colors = colors.astype(np.float32)
        if colors.max(initial=0.0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh


def _scene_parts(scene: trimesh.Scene) -> List[Tuple[str, str, trimesh.Trimesh]]:
    """Return scene node meshes with node transforms baked into vertices."""
    parts: List[Tuple[str, str, trimesh.Trimesh]] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geom = scene.geometry[geometry_name].copy()
        geom.apply_transform(transform)
        parts.append((node_name, geometry_name, geom))
    return parts


def _write_manifest(output_path: Path, manifest: list[dict]) -> Path:
    manifest_path = output_path / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved split manifest to: {manifest_path}")
    return manifest_path


def split_glb(glb_path: str, output_dir: str) -> Path:
    """
    Split GLB file into separate files for each geometry/part.

    Args:
        glb_path: Path to input GLB file
        output_dir: Directory to save split GLB files
    """
    import trimesh

    print(f"Loading GLB from: {glb_path}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(glb_path)

    if isinstance(scene, trimesh.Trimesh):
        # Single mesh - save as is
        print("[WARN] GLB contains a single mesh (no separate parts)")
        output_file = output_path / "part_0.glb"
        scene = _ensure_rgba_vertex_colors(scene.copy())
        scene.export(str(output_file))
        print(f"Saved single mesh to: {output_file}")
        _write_manifest(
            output_path,
            [
                {
                    "index": 0,
                    "node_name": "",
                    "geometry_name": Path(glb_path).stem,
                    "file": output_file.name,
                    "vertices": int(len(scene.vertices)),
                    "faces": int(len(scene.faces)),
                }
            ],
        )
        return output_path

    if not isinstance(scene, trimesh.Scene):
        raise ValueError(f"Unexpected GLB type: {type(scene)}")

    # Extract scene nodes and bake node transforms into each mesh.
    parts = _scene_parts(scene)
    print(f"\nFound {len(parts)} parts in GLB:")

    for i, (node_name, geometry_name, geom) in enumerate(parts):
        print(f"  [{i}] {geometry_name} (node={node_name}): {len(geom.vertices)} verts, {len(geom.faces)} faces")

    if len(parts) == 0:
        raise ValueError("No geometries found in GLB")

    # Save each part
    print(f"\nSaving parts to: {output_dir}")
    manifest = []
    for i, (node_name, geometry_name, geom) in enumerate(parts):
        # Clean up name for filename (remove special characters)
        safe_name = _safe_name(geometry_name)
        output_file = output_path / f"part_{i}_{safe_name}.glb"

        geom = _ensure_rgba_vertex_colors(geom)
        geom.export(str(output_file))
        manifest.append({
            "index": i,
            "node_name": node_name,
            "geometry_name": geometry_name,
            "file": output_file.name,
            "vertices": int(len(geom.vertices)),
            "faces": int(len(geom.faces)),
        })
        print(f"  Saved part {i} ({geometry_name}) to: {output_file.name}")

    _write_manifest(output_path, manifest)

    print(f"\n[DONE] Split {len(parts)} parts")
    return output_path


def _default_mesh_map_output(metadata_path: Path) -> Path:
    return metadata_path.resolve().parent / "mesh_map.json"


def _launch_mesh_map_gui(
    metadata_path: Path,
    split_dir: Path,
    mesh_map_output: Path,
    absolute_paths: bool,
    host: str,
    port: int,
    share: bool,
) -> None:
    try:
        from URDFoptimizer.render.mesh_map_gui import build_app
    except ImportError:
        from mesh_map_gui import build_app

    app = build_app(
        metadata_path=metadata_path,
        split_dir=split_dir,
        output_path=mesh_map_output,
        absolute_paths=absolute_paths,
    )
    print(f"Launching mesh map GUI: http://{host}:{port}")
    print(f"Mesh map output: {mesh_map_output}")
    app.launch(server_name=host, server_port=port, share=share)


def _launch_mesh_map_web(
    metadata_path: Path,
    split_dir: Path,
    mesh_map_output: Path,
    absolute_paths: bool,
    host: str,
    port: int,
    viewer_script_url: str,
) -> None:
    try:
        from URDFoptimizer.render.mesh_map_web import serve_app
    except ImportError:
        from mesh_map_web import serve_app

    serve_app(
        metadata_path=metadata_path,
        split_dir=split_dir,
        output_path=mesh_map_output,
        absolute_paths=absolute_paths,
        host=host,
        port=port,
        viewer_script_url=viewer_script_url,
    )


def main():
    parser = argparse.ArgumentParser(description="Split GLB file into separate parts")
    parser.add_argument("--input", type=str, default="outputs/test4/voxel.glb", help="Path to input GLB file")
    parser.add_argument("--output", type=str, default="outputs/test4/glb",
                       help="Output directory for split GLB files")
    parser.add_argument("--metadata", type=str, default=None, help="metadata.json containing the LLM-generated link graph")
    parser.add_argument("--mesh-map-output", type=str, default=None, help="Path to write mesh_map.json")
    parser.add_argument("--launch-web", action="store_true", help="Open the headless HTTP web mapper after splitting")
    parser.add_argument("--launch-gui", action="store_true", help="Open the Gradio mapper after splitting")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute mesh paths from the mapper")
    parser.add_argument("--host", default="0.0.0.0", help="Mapper host when --launch-web or --launch-gui is set")
    parser.add_argument("--port", type=int, default=7860, help="Mapper port when --launch-web or --launch-gui is set")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link when --launch-gui is set")
    parser.add_argument(
        "--viewer-script-url",
        default="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js",
        help="Browser GLB viewer script URL when --launch-web is set",
    )
    args = parser.parse_args()

    split_dir = split_glb(args.input, args.output)

    if args.launch_web and args.launch_gui:
        raise SystemExit("Use either --launch-web or --launch-gui, not both")

    if args.launch_web or args.launch_gui:
        if not args.metadata:
            raise SystemExit("--metadata is required when launching the mapper")
        metadata_path = Path(args.metadata)
        mesh_map_output = Path(args.mesh_map_output) if args.mesh_map_output else _default_mesh_map_output(metadata_path)

    if args.launch_web:
        _launch_mesh_map_web(
            metadata_path=metadata_path,
            split_dir=split_dir,
            mesh_map_output=mesh_map_output,
            absolute_paths=args.absolute_paths,
            host=args.host,
            port=args.port,
            viewer_script_url=args.viewer_script_url,
        )
    elif args.launch_gui:
        _launch_mesh_map_gui(
            metadata_path=metadata_path,
            split_dir=split_dir,
            mesh_map_output=mesh_map_output,
            absolute_paths=args.absolute_paths,
            host=args.host,
            port=args.port,
            share=args.share,
        )


if __name__ == "__main__":
    main()
