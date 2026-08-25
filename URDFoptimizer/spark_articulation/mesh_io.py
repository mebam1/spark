"""Mesh loading for existing part-segmented meshes."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from .model import MeshPartTensors
from .urdf_model import ArticulatedURDF, resolve_mesh_path


def load_mesh_parts_from_urdf(
    urdf: ArticulatedURDF,
    *,
    unit_scale: float = 1.0,
    device: torch.device | None = None,
) -> List[MeshPartTensors]:
    """Load visual meshes referenced by a URDF without modifying geometry."""
    if urdf.source_path is None:
        raise ValueError("URDF source_path is required to resolve relative mesh filenames")

    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        raise RuntimeError("trimesh is required to load existing segmented meshes") from exc

    device = device or torch.device("cpu")
    parts: List[MeshPartTensors] = []

    for link in urdf.links.values():
        for visual_index, visual in enumerate(link.visuals):
            path = resolve_mesh_path(visual.mesh_filename, urdf.source_path)
            if path is None:
                continue
            if not Path(path).exists():
                raise FileNotFoundError(f"Mesh referenced by link {link.name} not found: {path}")

            mesh = trimesh.load(str(path), force="mesh", process=False)
            if mesh.is_empty:
                raise ValueError(f"Mesh referenced by link {link.name} is empty: {path}")

            vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=device)

            if visual.scale is not None:
                vertices = vertices * torch.tensor(visual.scale, dtype=torch.float32, device=device)
            if unit_scale != 1.0:
                vertices = vertices * float(unit_scale)

            parts.append(
                MeshPartTensors(
                    link_name=link.name,
                    visual=visual,
                    vertices=vertices,
                    faces=faces,
                    visual_index=visual_index,
                )
            )

    if not parts:
        raise ValueError("No mesh visuals were loaded from the URDF")
    return parts

