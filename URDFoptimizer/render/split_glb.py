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
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


Vec3 = Tuple[float, float, float]
Matrix4 = List[List[float]]


def _identity_matrix() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _parse_vec3(value: Any, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if value is None:
        return default
    if isinstance(value, str):
        values = [float(item) for item in value.strip().split()]
    elif isinstance(value, Sequence) and len(value) == 3:
        values = [float(item) for item in value]
    else:
        return default
    if len(values) != 3:
        raise ValueError(f"Expected a 3-vector, got: {value}")
    return values[0], values[1], values[2]


def _matmul(a: Matrix4, b: Matrix4) -> Matrix4:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def _origin_matrix(xyz: Vec3, rpy: Vec3) -> Matrix4:
    """Build the same URDF origin transform used by the articulation FK."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    matrix = _identity_matrix()
    matrix[0][0] = cy * cp
    matrix[0][1] = cy * sp * sr - sy * cr
    matrix[0][2] = cy * sp * cr + sy * sr
    matrix[1][0] = sy * cp
    matrix[1][1] = sy * sp * sr + cy * cr
    matrix[1][2] = sy * sp * cr - cy * sr
    matrix[2][0] = -sp
    matrix[2][1] = cp * sr
    matrix[2][2] = cp * cr
    matrix[0][3] = xyz[0]
    matrix[1][3] = xyz[1]
    matrix[2][3] = xyz[2]
    return matrix


def _invert_rigid_matrix(matrix: Matrix4) -> Matrix4:
    inverse = _identity_matrix()
    for row in range(3):
        for col in range(3):
            inverse[row][col] = matrix[col][row]
    for row in range(3):
        inverse[row][3] = -sum(inverse[row][col] * matrix[col][3] for col in range(3))
    return inverse


def _metadata_key_to_link_label(metadata: Mapping[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    name_counts: Dict[str, int] = {}
    for part in metadata.get("parts", []):
        name = part.get("name")
        if name:
            name_counts[str(name)] = name_counts.get(str(name), 0) + 1

    for index, part in enumerate(metadata.get("parts", [])):
        label = str(part.get("label", f"link{index}"))
        lookup[label] = label
        name = part.get("name")
        if name and name_counts.get(str(name), 0) == 1:
            lookup[str(name)] = label
    return lookup


def _link_zero_pose_transforms(metadata: Mapping[str, Any]) -> Dict[str, Matrix4]:
    transforms: Dict[str, Matrix4] = {"base": _identity_matrix()}
    pending = {
        str(part.get("label", f"link{index}")): part
        for index, part in enumerate(metadata.get("parts", []))
    }

    while pending:
        progressed = False
        for label, part in list(pending.items()):
            parent = str(part.get("parent", "base"))
            if parent not in transforms:
                continue
            xyz = _parse_vec3(part.get("origin_xyz") or part.get("joint_origin_xyz"))
            rpy = _parse_vec3(part.get("origin_rpy") or part.get("joint_origin_rpy"))
            transforms[label] = _matmul(transforms[parent], _origin_matrix(xyz, rpy))
            del pending[label]
            progressed = True

        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise ValueError(f"Cannot resolve metadata link zero-pose transforms for: {unresolved}")

    return transforms


def _iter_mesh_map_values(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _add_unique_assignment(assignments: Dict[str, str], key: str, link_label: str) -> None:
    existing = assignments.get(key)
    if existing is not None and existing != link_label:
        raise ValueError(f"Mesh map assigns {key} to both {existing} and {link_label}")
    assignments[key] = link_label


def _mesh_map_link_assignments(
    mesh_map: Mapping[str, Any],
    metadata: Mapping[str, Any],
    mesh_map_path: Path,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    key_to_label = _metadata_key_to_link_label(metadata)
    by_resolved_path: Dict[str, str] = {}
    by_filename: Dict[str, str] = {}

    for key, value in mesh_map.items():
        link_label = key_to_label.get(str(key))
        if link_label is None:
            raise ValueError(f"Mesh map key does not match a metadata link label or unique part name: {key}")

        for mesh_ref in _iter_mesh_map_values(value):
            mesh_path = Path(mesh_ref)
            resolved = mesh_path if mesh_path.is_absolute() else mesh_map_path.parent / mesh_path
            _add_unique_assignment(by_resolved_path, str(resolved.resolve()), link_label)
            _add_unique_assignment(by_filename, mesh_path.name, link_label)

    return by_resolved_path, by_filename


def _load_link_local_context(
    metadata_path: str | Path,
    mesh_map_path: str | Path,
) -> Tuple[Dict[str, Matrix4], Dict[str, str], Dict[str, str]]:
    metadata_path = Path(metadata_path)
    mesh_map_path = Path(mesh_map_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mesh_map = json.loads(mesh_map_path.read_text(encoding="utf-8"))
    transforms = _link_zero_pose_transforms(metadata)
    by_resolved_path, by_filename = _mesh_map_link_assignments(mesh_map, metadata, mesh_map_path)
    return transforms, by_resolved_path, by_filename


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


def _link_label_for_output_mesh(
    output_file: Path,
    by_resolved_path: Mapping[str, str],
    by_filename: Mapping[str, str],
) -> str:
    link_label = by_resolved_path.get(str(output_file.resolve()))
    if link_label is not None:
        return link_label

    link_label = by_filename.get(output_file.name)
    if link_label is not None:
        return link_label

    raise ValueError(
        f"--link-local could not find {output_file.name} in --mesh-map. "
        "Assign every split mesh to a metadata link before exporting link-local parts."
    )


def _apply_link_local_transform(
    geom: trimesh.Trimesh,
    output_file: Path,
    transforms_by_link: Mapping[str, Matrix4],
    by_resolved_path: Mapping[str, str],
    by_filename: Mapping[str, str],
) -> Tuple[trimesh.Trimesh, str, Matrix4]:
    link_label = _link_label_for_output_mesh(output_file, by_resolved_path, by_filename)
    if link_label not in transforms_by_link:
        raise ValueError(f"No zero-pose transform found for metadata link: {link_label}")

    link_transform = transforms_by_link[link_label]
    geom.apply_transform(_invert_rigid_matrix(link_transform))
    return geom, link_label, link_transform


def split_glb(
    glb_path: str,
    output_dir: str,
    *,
    link_local: bool = False,
    metadata_path: str | Path | None = None,
    mesh_map_path: str | Path | None = None,
) -> Path:
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

    transforms_by_link: Dict[str, Matrix4] = {}
    by_resolved_path: Dict[str, str] = {}
    by_filename: Dict[str, str] = {}
    if link_local:
        if metadata_path is None or mesh_map_path is None:
            raise ValueError("--link-local requires both --metadata and --mesh-map")
        transforms_by_link, by_resolved_path, by_filename = _load_link_local_context(metadata_path, mesh_map_path)

    if isinstance(scene, trimesh.Trimesh):
        # Single mesh - save as is
        print("[WARN] GLB contains a single mesh (no separate parts)")
        output_file = output_path / "part_0.glb"
        scene = _ensure_rgba_vertex_colors(scene.copy())
        link_label = None
        link_transform = None
        if link_local:
            scene, link_label, link_transform = _apply_link_local_transform(
                scene,
                output_file,
                transforms_by_link,
                by_resolved_path,
                by_filename,
            )
        scene.export(str(output_file))
        print(f"Saved single mesh to: {output_file}")
        manifest_item = {
            "index": 0,
            "node_name": "",
            "geometry_name": Path(glb_path).stem,
            "file": output_file.name,
            "vertices": int(len(scene.vertices)),
            "faces": int(len(scene.faces)),
        }
        if link_local:
            manifest_item.update(
                {
                    "frame": "link_local",
                    "link_label": link_label,
                    "zero_pose_link_transform": link_transform,
                }
            )
        _write_manifest(output_path, [manifest_item])
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
    frame_desc = "link-local" if link_local else "object-space"
    print(f"\nSaving {frame_desc} parts to: {output_dir}")
    manifest = []
    for i, (node_name, geometry_name, geom) in enumerate(parts):
        # Clean up name for filename (remove special characters)
        safe_name = _safe_name(geometry_name)
        output_file = output_path / f"part_{i}_{safe_name}.glb"

        link_label = None
        link_transform = None
        if link_local:
            geom, link_label, link_transform = _apply_link_local_transform(
                geom,
                output_file,
                transforms_by_link,
                by_resolved_path,
                by_filename,
            )

        geom = _ensure_rgba_vertex_colors(geom)
        geom.export(str(output_file))
        manifest_item = {
            "index": i,
            "node_name": node_name,
            "geometry_name": geometry_name,
            "file": output_file.name,
            "vertices": int(len(geom.vertices)),
            "faces": int(len(geom.faces)),
        }
        if link_local:
            manifest_item.update(
                {
                    "frame": "link_local",
                    "link_label": link_label,
                    "zero_pose_link_transform": link_transform,
                }
            )
        manifest.append(manifest_item)
        link_note = f" [{link_label} local]" if link_label else ""
        print(f"  Saved part {i} ({geometry_name}) to: {output_file.name}{link_note}")

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
    parser.add_argument("--mesh-map", type=str, default=None, help="Existing mesh_map.json used by --link-local")
    parser.add_argument("--mesh-map-output", type=str, default=None, help="Path to write mesh_map.json")
    parser.add_argument(
        "--link-local",
        action="store_true",
        help=(
            "Export split meshes in their assigned URDF link-local zero-pose frame. "
            "Requires --metadata and --mesh-map."
        ),
    )
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

    if args.launch_web and args.launch_gui:
        raise SystemExit("Use either --launch-web or --launch-gui, not both")

    if args.link_local and (not args.metadata or not args.mesh_map):
        raise SystemExit("--link-local requires both --metadata and --mesh-map")

    split_dir = split_glb(
        args.input,
        args.output,
        link_local=args.link_local,
        metadata_path=args.metadata,
        mesh_map_path=args.mesh_map,
    )

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
