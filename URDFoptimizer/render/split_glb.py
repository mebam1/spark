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

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh


def _safe_name(name: str) -> str:
    return name.replace('/', '_').replace('\\', '_').replace(' ', '_')


def _ensure_rgba_vertex_colors(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    trimesh's GLB exporter expects vertex colors as RGBA. Some GLBs load as
    RGB vertex colors, which later fails with "cannot reshape ... into shape
    (4)". Keep geometry unchanged and only normalize color channel count.
    """
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


def split_glb(glb_path: str, output_dir: str):
    """
    Split GLB file into separate files for each geometry/part.

    Args:
        glb_path: Path to input GLB file
        output_dir: Directory to save split GLB files
    """
    print(f"Loading GLB from: {glb_path}")
    scene = trimesh.load(glb_path)

    if isinstance(scene, trimesh.Trimesh):
        # Single mesh - save as is
        print("[WARN] GLB contains a single mesh (no separate parts)")
        output_path = Path(output_dir) / "part_0.glb"
        scene = _ensure_rgba_vertex_colors(scene.copy())
        scene.export(str(output_path))
        print(f"Saved single mesh to: {output_path}")
        return

    if not isinstance(scene, trimesh.Scene):
        raise ValueError(f"Unexpected GLB type: {type(scene)}")

    # Extract scene nodes and bake node transforms into each mesh.
    parts = _scene_parts(scene)
    print(f"\nFound {len(parts)} parts in GLB:")

    for i, (node_name, geometry_name, geom) in enumerate(parts):
        print(f"  [{i}] {geometry_name} (node={node_name}): {len(geom.vertices)} verts, {len(geom.faces)} faces")

    if len(parts) == 0:
        raise ValueError("No geometries found in GLB")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    manifest_path = output_path / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved split manifest to: {manifest_path}")

    print(f"\n[DONE] Split {len(parts)} parts")


def main():
    parser = argparse.ArgumentParser(description="Split GLB file into separate parts")
    parser.add_argument("--input", type=str, default="outputs/test4/voxel.glb", help="Path to input GLB file")
    parser.add_argument("--output", type=str, default="outputs/test4/glb",
                       help="Output directory for split GLB files")
    args = parser.parse_args()

    split_glb(args.input, args.output)


if __name__ == "__main__":
    main()
