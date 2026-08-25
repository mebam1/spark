"""Differentiable articulated mesh module for silhouette optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import nn

from .fk import compute_link_transforms, transform_visual_vertices
from .urdf_model import ArticulatedURDF, VisualSpec


@dataclass(frozen=True)
class MeshPartTensors:
    link_name: str
    visual: VisualSpec
    vertices: torch.Tensor
    faces: torch.Tensor
    visual_index: int = 0


def _initial_revolute_angle(joint, default_open_angle_rad: float) -> float:
    if joint.limit_upper is not None and joint.limit_upper > 0:
        return float(joint.limit_upper)
    return default_open_angle_rad


class DifferentiableArticulationModel(nn.Module):
    """URDF FK + existing segmented meshes + learnable continuous parameters."""

    def __init__(
        self,
        urdf: ArticulatedURDF,
        mesh_parts: Sequence[MeshPartTensors],
        *,
        optimize_joint_names: Optional[Sequence[str]] = None,
        initial_joint_values: Optional[Mapping[str, float]] = None,
        default_open_angle_rad: float = math.radians(120.0),
        learn_origin: bool = True,
        learn_angle: bool = True,
        auto_normalize: bool = True,
        preserve_zero_pose: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.urdf = urdf
        self.preserve_zero_pose = preserve_zero_pose
        self.part_specs: List[tuple[str, VisualSpec, int]] = []

        device = device or torch.device("cpu")
        joints_by_name = urdf.joints_by_name
        if optimize_joint_names is None:
            optimize_joint_names = [joint.name for joint in urdf.joints if joint.is_revolute]
        self.optimize_joint_names = list(optimize_joint_names)

        for name in self.optimize_joint_names:
            joint = joints_by_name[name]
            if not joint.is_revolute:
                raise ValueError(f"Baseline continuous optimization supports revolute joints only: {name}")

        self._origin_param_indices: Dict[str, int] = {}
        self._angle_param_indices: Dict[str, int] = {}
        self.origin_deltas = nn.ParameterList()
        self.angle_deltas = nn.ParameterList()

        if learn_origin:
            for name in self.optimize_joint_names:
                self._origin_param_indices[name] = len(self.origin_deltas)
                self.origin_deltas.append(nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device)))

        initial_joint_values = dict(initial_joint_values or {})
        self.initial_joint_values: Dict[str, float] = {}
        for joint in urdf.joints:
            if joint.is_revolute:
                self.initial_joint_values[joint.name] = initial_joint_values.get(
                    joint.name,
                    _initial_revolute_angle(joint, default_open_angle_rad),
                )
            else:
                self.initial_joint_values[joint.name] = initial_joint_values.get(joint.name, 0.0)

        if learn_angle:
            for name in self.optimize_joint_names:
                self._angle_param_indices[name] = len(self.angle_deltas)
                self.angle_deltas.append(nn.Parameter(torch.zeros((), dtype=torch.float32, device=device)))

        for idx, part in enumerate(mesh_parts):
            verts = part.vertices.detach().clone().to(device=device, dtype=torch.float32)
            faces = part.faces.detach().clone().to(device=device, dtype=torch.int64)
            self.register_buffer(f"verts_{idx}", verts)
            self.register_buffer(f"faces_{idx}", faces)
            self.part_specs.append((part.link_name, part.visual, part.visual_index))

        if auto_normalize:
            with torch.no_grad():
                vertices, _ = self.transformed_vertices_and_faces(use_learned=False, closed_state=True, normalize=False)
                if vertices.numel() == 0:
                    translation = torch.zeros(3, dtype=torch.float32, device=device)
                    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
                else:
                    bbox_min = vertices.min(dim=0).values
                    bbox_max = vertices.max(dim=0).values
                    center = (bbox_min + bbox_max) * 0.5
                    extent = (bbox_max - bbox_min).max().clamp_min(1e-6)
                    translation = -center
                    scale = 2.0 / extent
        else:
            translation = torch.zeros(3, dtype=torch.float32, device=device)
            scale = torch.tensor(1.0, dtype=torch.float32, device=device)

        self.register_buffer("norm_translation", translation)
        self.register_buffer("norm_scale", scale)

    def origin_parameters(self) -> Iterable[nn.Parameter]:
        return self.origin_deltas

    def angle_parameters(self) -> Iterable[nn.Parameter]:
        return self.angle_deltas

    def current_origin_offsets(self) -> Dict[str, torch.Tensor]:
        offsets: Dict[str, torch.Tensor] = {}
        for name, idx in self._origin_param_indices.items():
            offsets[name] = self.origin_deltas[idx]
        return offsets

    def current_joint_values(self, *, closed_state: bool = False) -> Dict[str, torch.Tensor]:
        values: Dict[str, torch.Tensor] = {}
        device = self.norm_translation.device if hasattr(self, "norm_translation") else torch.device("cpu")
        for name, initial_value in self.initial_joint_values.items():
            if closed_state:
                values[name] = torch.tensor(0.0, dtype=torch.float32, device=device)
            elif name in self._angle_param_indices:
                values[name] = torch.tensor(initial_value, dtype=torch.float32, device=device) + self.angle_deltas[
                    self._angle_param_indices[name]
                ]
            else:
                values[name] = torch.tensor(initial_value, dtype=torch.float32, device=device)
        return values

    def clamp_angles_to_limits(self) -> None:
        joints = self.urdf.joints_by_name
        with torch.no_grad():
            for name, idx in self._angle_param_indices.items():
                joint = joints[name]
                lower = joint.limit_lower
                upper = joint.limit_upper
                if lower is None or upper is None or upper <= lower:
                    continue
                initial = self.initial_joint_values[name]
                total = torch.clamp(self.angle_deltas[idx] + initial, min=lower, max=upper)
                self.angle_deltas[idx].copy_(total - initial)

    def transformed_vertices_and_faces(
        self,
        *,
        use_learned: bool = True,
        closed_state: bool = False,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.part_specs) == 0:
            device = self.norm_translation.device if hasattr(self, "norm_translation") else torch.device("cpu")
            return (
                torch.empty((0, 3), dtype=torch.float32, device=device),
                torch.empty((0, 3), dtype=torch.int64, device=device),
            )

        sample_vertices = getattr(self, "verts_0")
        device = sample_vertices.device
        dtype = sample_vertices.dtype
        if use_learned:
            joint_values = self.current_joint_values(closed_state=closed_state)
            origin_offsets = self.current_origin_offsets()
        else:
            joint_values = {joint.name: torch.tensor(0.0, dtype=dtype, device=device) for joint in self.urdf.joints}
            origin_offsets = {}

        link_transforms = compute_link_transforms(
            self.urdf,
            joint_values=joint_values,
            origin_offsets=origin_offsets,
            preserve_zero_pose=self.preserve_zero_pose,
            device=device,
            dtype=dtype,
        )

        transformed_vertices: List[torch.Tensor] = []
        transformed_faces: List[torch.Tensor] = []
        vertex_offset = 0
        for idx, (link_name, visual, _visual_index) in enumerate(self.part_specs):
            vertices = getattr(self, f"verts_{idx}")
            faces = getattr(self, f"faces_{idx}")
            if link_name not in link_transforms:
                continue

            verts_world = transform_visual_vertices(vertices, link_transforms[link_name], visual)
            transformed_vertices.append(verts_world)
            transformed_faces.append(faces + vertex_offset)
            vertex_offset += vertices.shape[0]

        if not transformed_vertices:
            return (
                torch.empty((0, 3), dtype=dtype, device=device),
                torch.empty((0, 3), dtype=torch.int64, device=device),
            )

        vertices = torch.cat(transformed_vertices, dim=0)
        faces = torch.cat(transformed_faces, dim=0)
        if normalize:
            vertices = (vertices + self.norm_translation.unsqueeze(0)) * self.norm_scale
        return vertices, faces

    def regularization_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.norm_translation.device
        origin_reg = torch.zeros((), dtype=torch.float32, device=device)
        for param in self.origin_deltas:
            origin_reg = origin_reg + torch.sum(param * param)

        angle_reg = torch.zeros((), dtype=torch.float32, device=device)
        for param in self.angle_deltas:
            angle_reg = angle_reg + param * param
        return origin_reg, angle_reg

    def learned_origin_deltas(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for name, idx in self._origin_param_indices.items():
            out[name] = [float(x) for x in self.origin_deltas[idx].detach().cpu()]
        return out

    def learned_joint_values(self) -> Dict[str, float]:
        values = self.current_joint_values(closed_state=False)
        return {name: float(value.detach().cpu()) for name, value in values.items() if name in self.optimize_joint_names}

    def forward(self):
        vertices, faces = self.transformed_vertices_and_faces(use_learned=True, closed_state=False, normalize=True)
        try:
            from pytorch3d.renderer import TexturesVertex
            from pytorch3d.structures import Meshes
        except Exception as exc:  # pragma: no cover - exercised only when dependency is absent.
            raise RuntimeError("PyTorch3D is required for differentiable rendering") from exc

        vertex_colors = torch.ones_like(vertices)
        return Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=[vertex_colors]))

