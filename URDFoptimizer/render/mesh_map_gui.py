#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI for mapping split GLB meshes to VLM-predicted URDF links.

Each split GLB can be assigned to one metadata link. Assigning multiple GLBs to
the same link makes them one rigid URDF link with multiple visual meshes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


UNASSIGNED = "(unassigned)"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _metadata_links(metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    links = []
    for part in metadata.get("parts", []):
        links.append(
            {
                "label": str(part.get("label", "")),
                "name": str(part.get("name", "")),
                "parent": str(part.get("parent", "")),
                "joint_type": str(part.get("joint_type", "")),
                "axis": str(part.get("axis", "")),
                "origin_xyz": str(part.get("origin_xyz") or part.get("joint_origin_xyz") or "0 0 0"),
            }
        )
    return links


def _link_markdown(metadata: Dict[str, Any]) -> str:
    rows = [
        "| Link | Name | Parent | Joint | Axis | Origin |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for link in _metadata_links(metadata):
        rows.append(
            f"| `{link['label']}` | {link['name']} | `{link['parent']}` | "
            f"`{link['joint_type']}` | `{link['axis']}` | `{link['origin_xyz']}` |"
        )
    return "\n".join(rows)


def _load_split_parts(split_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = split_dir / "split_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        parts = []
        for item in manifest:
            file_path = split_dir / item["file"]
            if file_path.exists():
                parts.append({**item, "path": file_path})
        return parts

    parts = []
    for index, path in enumerate(sorted(split_dir.glob("*.glb"))):
        parts.append(
            {
                "index": index,
                "node_name": "",
                "geometry_name": path.stem,
                "file": path.name,
                "vertices": "",
                "faces": "",
                "path": path,
            }
        )
    return parts


def _resolve_map_value(value: str, output_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (output_dir / path).resolve()


def _load_existing_assignments(output_path: Path, split_parts: List[Dict[str, Any]]) -> Dict[str, str]:
    if not output_path.exists():
        return {}

    data = _load_json(output_path)
    assignment_by_file: Dict[str, str] = {}
    output_dir = output_path.parent.resolve()

    resolved_to_file = {str(part["path"].resolve()): part["file"] for part in split_parts}
    basename_to_file = {part["path"].name: part["file"] for part in split_parts}

    for link, values in data.items():
        if isinstance(values, str):
            values = [values]
        for value in values:
            resolved = str(_resolve_map_value(str(value), output_dir))
            file_name = resolved_to_file.get(resolved) or basename_to_file.get(Path(str(value)).name)
            if file_name is not None:
                assignment_by_file[file_name] = link
    return assignment_by_file


def _mesh_path_for_json(mesh_path: Path, output_path: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return mesh_path.resolve().as_posix()
    return Path(os.path.relpath(mesh_path.resolve(), output_path.parent.resolve())).as_posix()


def _build_mesh_map(
    split_parts: List[Dict[str, Any]],
    assignments: List[str],
    output_path: Path,
    absolute_paths: bool,
) -> Dict[str, Any]:
    grouped: Dict[str, List[str]] = {}
    for part, link in zip(split_parts, assignments):
        if link == UNASSIGNED:
            continue
        grouped.setdefault(link, []).append(_mesh_path_for_json(part["path"], output_path, absolute_paths))

    # Keep the compact string form for single-mesh links; use lists only when
    # one rigid link is composed of multiple split meshes.
    return {link: values[0] if len(values) == 1 else values for link, values in grouped.items()}


def build_app(
    metadata_path: Path,
    split_dir: Path,
    output_path: Path,
    absolute_paths: bool = False,
) -> gr.Blocks:
    import gradio as gr

    metadata = _load_json(metadata_path)
    links = _metadata_links(metadata)
    link_choices = [UNASSIGNED] + [link["label"] for link in links]
    split_parts = _load_split_parts(split_dir)
    existing = _load_existing_assignments(output_path, split_parts)

    if not links:
        raise ValueError(f"No parts found in metadata: {metadata_path}")
    if not split_parts:
        raise ValueError(f"No split GLB files found in: {split_dir}")

    def save_mapping(*assignments: str):
        mesh_map = _build_mesh_map(split_parts, list(assignments), output_path, absolute_paths)
        _save_json(output_path, mesh_map)

        missing_links = [link["label"] for link in links if link["label"] not in mesh_map]
        summary = {
            "output": str(output_path),
            "assigned_links": len(mesh_map),
            "assigned_meshes": sum(1 for item in assignments if item != UNASSIGNED),
            "unassigned_meshes": sum(1 for item in assignments if item == UNASSIGNED),
            "links_without_mesh": missing_links,
            "mesh_map": mesh_map,
        }
        status = f"Saved {output_path}"
        if missing_links:
            status += f"\nLinks without mesh: {', '.join(missing_links)}"
        return status, json.dumps(summary, indent=2)

    with gr.Blocks(title="SPARK Mesh Map GUI") as app:
        gr.Markdown("# SPARK Mesh Map GUI")
        gr.Markdown(
            "Assign each split GLB mesh to one VLM-predicted link. "
            "Assigning multiple GLBs to the same link treats them as one rigid moving part."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## LLM Link Graph")
                gr.Markdown(_link_markdown(metadata))
                gr.Markdown(f"Metadata: `{metadata_path}`\n\nSplit dir: `{split_dir}`\n\nOutput: `{output_path}`")
            with gr.Column(scale=1):
                status = gr.Textbox(label="Status", interactive=False)
                mesh_map_json = gr.Code(label="mesh_map.json preview", language="json")
                save_button = gr.Button("Save mesh_map.json", variant="primary")

        dropdowns = []
        gr.Markdown("## Split Mesh Assignments")
        for part in split_parts:
            default_link = existing.get(part["file"], UNASSIGNED)
            if default_link not in link_choices:
                default_link = UNASSIGNED

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Model3D(value=str(part["path"]), label=f"{part['index']}: {part['file']}", height=280)
                with gr.Column(scale=1):
                    gr.Markdown(
                        f"**Geometry:** `{part.get('geometry_name', '')}`\n\n"
                        f"**Node:** `{part.get('node_name', '')}`\n\n"
                        f"**Vertices/Faces:** `{part.get('vertices', '')}` / `{part.get('faces', '')}`"
                    )
                    dropdown = gr.Dropdown(
                        choices=link_choices,
                        value=default_link,
                        label="Assign to link",
                    )
                    dropdowns.append(dropdown)

        save_button.click(save_mapping, inputs=dropdowns, outputs=[status, mesh_map_json])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign split GLB meshes to VLM-generated links.")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--split-dir", required=True, help="Directory containing split GLB files")
    parser.add_argument("--output", required=True, help="Path to write mesh_map.json")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute mesh paths")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(
        metadata_path=Path(args.metadata),
        split_dir=Path(args.split_dir),
        output_path=Path(args.output),
        absolute_paths=args.absolute_paths,
    )
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
