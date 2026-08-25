#!/usr/bin/env python3
"""Copy URDF mesh references into a local meshes directory and rewrite paths."""

from __future__ import annotations

import argparse
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


def _unique_target_path(source: Path, mesh_dir: Path, used_names: set[str]) -> Path:
    candidate_name = source.name
    if candidate_name not in used_names:
        used_names.add(candidate_name)
        return mesh_dir / candidate_name

    stem = source.stem
    suffix = source.suffix
    index = 1
    while True:
        candidate_name = f"{stem}_{index}{suffix}"
        if candidate_name not in used_names:
            used_names.add(candidate_name)
            return mesh_dir / candidate_name
        index += 1


def _mesh_filename_for_urdf(target: Path, output_urdf: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return target.resolve().as_posix()
    return Path(os.path.relpath(target.resolve(), output_urdf.resolve().parent)).as_posix()


def localize_urdf_meshes(
    urdf_path: str | os.PathLike[str],
    output_urdf_path: str | os.PathLike[str] | None = None,
    mesh_dir: str | os.PathLike[str] = "meshes",
    *,
    absolute_paths: bool = False,
) -> Dict[str, Any]:
    """Copy meshes referenced by a URDF and rewrite mesh filenames.

    Relative source mesh paths are resolved from the input URDF directory. If
    ``output_urdf_path`` is omitted, the input URDF is rewritten in place.
    Relative ``mesh_dir`` values are resolved from the output URDF directory.
    """
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

    copied_by_source: Dict[Path, Path] = {}
    used_names = {path.name for path in mesh_dir_path.iterdir() if path.is_file()}
    missing: list[str] = []
    rewritten = 0
    copied = 0

    for mesh_el in root.findall(".//mesh"):
        filename = mesh_el.attrib.get("filename")
        if not filename:
            continue

        source = _resolve_mesh_path(filename, input_urdf_dir)
        if not source.exists():
            missing.append(f"{filename} -> {source}")
            continue

        target = copied_by_source.get(source)
        if target is None:
            if source.parent.resolve() == mesh_dir_path:
                target = source
                used_names.add(target.name)
            else:
                target = _unique_target_path(source, mesh_dir_path, used_names)
                shutil.copy2(source, target)
                copied += 1
            copied_by_source[source] = target

        mesh_el.attrib["filename"] = _mesh_filename_for_urdf(target, output_urdf, absolute_paths)
        rewritten += 1

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
        "copied_meshes": copied,
        "rewritten_mesh_references": rewritten,
        "mesh_files": [target.name for target in copied_by_source.values()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy URDF mesh references into a local meshes directory.")
    parser.add_argument("--urdf", required=True, help="Input URDF path")
    parser.add_argument("--output-urdf", default=None, help="Output URDF path. Default: rewrite --urdf in place")
    parser.add_argument("--mesh-dir", default="meshes", help="Directory for copied meshes. Relative to output URDF dir")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute mesh paths into the output URDF")
    args = parser.parse_args()

    summary = localize_urdf_meshes(
        urdf_path=args.urdf,
        output_urdf_path=args.output_urdf,
        mesh_dir=args.mesh_dir,
        absolute_paths=args.absolute_paths,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
