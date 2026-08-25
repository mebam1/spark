#!/usr/bin/env python3
"""Copy/convert URDF mesh references into a local mesh directory and rewrite paths."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote, urlparse


def _resolve_mesh_path(filename: str, urdf_dir: Path) -> Path:
    parsed = urlparse(filename)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()

    normalized = filename
    if normalized.startswith("package://"):
        normalized = normalized.replace("package://", "", 1)

    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (urdf_dir / path).resolve()


def _unique_target_path(source: Path, mesh_dir: Path, used_names: set[str], suffix: str | None = None) -> Path:
    candidate_name = source.with_suffix(suffix).name if suffix else source.name
    if candidate_name not in used_names:
        used_names.add(candidate_name)
        return mesh_dir / candidate_name

    stem = source.stem
    target_suffix = suffix if suffix is not None else source.suffix
    index = 1
    while True:
        candidate_name = f"{stem}_{index}{target_suffix}"
        if candidate_name not in used_names:
            used_names.add(candidate_name)
            return mesh_dir / candidate_name
        index += 1


def _mesh_filename_for_urdf(target: Path, output_urdf: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return target.resolve().as_posix()
    return Path(os.path.relpath(target.resolve(), output_urdf.resolve().parent)).as_posix()


def _normalize_mesh_format(mesh_format: str | None) -> str | None:
    if mesh_format is None:
        return None
    normalized = mesh_format.lower().lstrip(".")
    if normalized in ("", "keep", "copy", "none"):
        return None
    if normalized not in ("obj", "stl"):
        raise ValueError("--mesh-format must be one of: keep, obj, stl")
    return normalized


def _write_mesh(source: Path, target: Path, mesh_format: str | None) -> None:
    if mesh_format is None:
        shutil.copy2(source, target)
        return

    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("trimesh is required when --mesh-format is used") from exc

    mesh = trimesh.load(str(source), force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {source}")
    mesh.export(str(target), file_type=mesh_format)


def _add_collision_meshes_from_visuals(root: ET.Element) -> int:
    added = 0
    for link in root.findall("link"):
        if link.findall("collision"):
            continue
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None or geometry.find("mesh") is None:
                continue

            collision = ET.Element("collision")
            origin = visual.find("origin")
            if origin is not None:
                collision.append(copy.deepcopy(origin))
            collision.append(copy.deepcopy(geometry))
            link.append(collision)
            added += 1
    return added


def localize_urdf_meshes(
    urdf_path: str | os.PathLike[str],
    output_urdf_path: str | os.PathLike[str] | None = None,
    mesh_dir: str | os.PathLike[str] = "meshes",
    *,
    absolute_paths: bool = False,
    mesh_format: str | None = None,
    add_collisions: bool = False,
) -> Dict[str, Any]:
    """Copy or convert meshes referenced by a URDF and rewrite mesh filenames.

    Relative source mesh paths are resolved from the input URDF directory. If
    ``output_urdf_path`` is omitted, the input URDF is rewritten in place.
    Relative ``mesh_dir`` values are resolved from the output URDF directory.
    """
    mesh_format = _normalize_mesh_format(mesh_format)
    input_urdf = Path(urdf_path).resolve()
    output_urdf = Path(output_urdf_path).resolve() if output_urdf_path else input_urdf
    input_urdf_dir = input_urdf.parent

    mesh_dir_path = Path(mesh_dir)
    if not mesh_dir_path.is_absolute():
        mesh_dir_path = output_urdf.parent / mesh_dir_path
    mesh_dir_path = mesh_dir_path.resolve()
    mesh_dir_path.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_urdf)
    root = tree.getroot()

    copied_by_source: Dict[tuple[Path, str | None], Path] = {}
    used_names: set[str] = set()
    missing: list[str] = []
    rewritten = 0
    written = 0
    target_suffix = f".{mesh_format}" if mesh_format is not None else None

    for mesh_el in root.findall(".//mesh"):
        filename = mesh_el.attrib.get("filename")
        if not filename:
            continue

        source = _resolve_mesh_path(filename, input_urdf_dir)
        if not source.exists():
            missing.append(f"{filename} -> {source}")
            continue

        source_key = (source, mesh_format)
        target = copied_by_source.get(source_key)
        if target is None:
            if mesh_format is None and source.parent.resolve() == mesh_dir_path:
                target = source
                used_names.add(target.name)
            else:
                target = _unique_target_path(source, mesh_dir_path, used_names, suffix=target_suffix)
                _write_mesh(source, target, mesh_format)
                written += 1
            copied_by_source[source_key] = target

        mesh_el.attrib["filename"] = _mesh_filename_for_urdf(target, output_urdf, absolute_paths)
        rewritten += 1

    added_collision_meshes = _add_collision_meshes_from_visuals(root) if add_collisions else 0

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Some URDF mesh references were not found:\n{joined}")

    ET.indent(tree, space="  ", level=0)
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)

    return {
        "input_urdf": str(input_urdf),
        "output_urdf": str(output_urdf),
        "mesh_dir": str(mesh_dir_path),
        "localized_meshes": len(copied_by_source),
        "copied_meshes": written if mesh_format is None else 0,
        "converted_meshes": written if mesh_format is not None else 0,
        "rewritten_mesh_references": rewritten,
        "added_collision_meshes": added_collision_meshes,
        "mesh_format": mesh_format or "keep",
        "mesh_files": [target.name for target in copied_by_source.values()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy or convert URDF mesh references into a local meshes directory.")
    parser.add_argument("--urdf", required=True, help="Input URDF path")
    parser.add_argument("--output-urdf", default=None, help="Output URDF path. Default: rewrite --urdf in place")
    parser.add_argument("--mesh-dir", default="meshes", help="Directory for copied meshes. Relative to output URDF dir")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute mesh paths into the output URDF")
    parser.add_argument(
        "--mesh-format",
        default="keep",
        choices=["keep", "obj", "stl"],
        help="Keep original mesh format or convert mesh files. OBJ/STL are safer for Isaac Sim URDF import",
    )
    parser.add_argument("--add-collisions", action="store_true", help="Add collision meshes copied from visuals when absent")
    args = parser.parse_args()

    summary = localize_urdf_meshes(
        urdf_path=args.urdf,
        output_urdf_path=args.output_urdf,
        mesh_dir=args.mesh_dir,
        absolute_paths=args.absolute_paths,
        mesh_format=args.mesh_format,
        add_collisions=args.add_collisions,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
