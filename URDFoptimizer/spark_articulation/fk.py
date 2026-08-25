"""Differentiable hierarchical URDF forward kinematics."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from .math_utils import (
    as_vec3,
    axis_angle_to_matrix,
    invert_transform,
    make_transform,
    origin_transform,
    transform_points,
    translation_transform,
)
from .urdf_model import ArticulatedURDF, JointSpec, VisualSpec


def _zero_transform(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype)


def _joint_motion_transform(
    joint: JointSpec,
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if joint.is_revolute:
        axis = as_vec3(joint.axis, device=device, dtype=dtype)
        R = axis_angle_to_matrix(axis, value)
        return make_transform(R, torch.zeros(3, device=device, dtype=dtype))
    if joint.is_prismatic:
        axis = as_vec3(joint.axis, device=device, dtype=dtype)
        axis = axis / (torch.linalg.norm(axis) + 1e-9)
        return translation_transform(axis * value)
    return _zero_transform(device, dtype)


def compute_link_transforms(
    urdf: ArticulatedURDF,
    joint_values: Optional[Mapping[str, torch.Tensor | float]] = None,
    origin_offsets: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    preserve_zero_pose: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Compute world-from-link transforms for the full URDF tree.

    ``origin_offsets`` are continuous pivot refinements in the parent link
    frame. Discrete parameters such as joint type and axis are read from URDF
    and are never optimized here.
    """
    device = device or torch.device("cpu")
    joint_values = joint_values or {}
    origin_offsets = origin_offsets or {}

    transforms: Dict[str, torch.Tensor] = {}
    for root_link in urdf.root_links:
        transforms[root_link] = _zero_transform(device, dtype)

    for joint in urdf.topological_joints():
        if joint.parent not in transforms:
            raise ValueError(f"Parent link transform not ready for joint {joint.name}: {joint.parent}")

        parent_T = transforms[joint.parent]
        base_xyz = as_vec3(joint.xyz, device=device, dtype=dtype)
        offset = origin_offsets.get(joint.name)
        if offset is None:
            offset = torch.zeros(3, device=device, dtype=dtype)
        else:
            offset = offset.to(device=device, dtype=dtype)

        T_joint_initial = origin_transform(joint.xyz, joint.rpy, device=device, dtype=dtype)
        T_joint_current = origin_transform(base_xyz + offset, joint.rpy, device=device, dtype=dtype)

        raw_value = joint_values.get(joint.name, 0.0)
        value = torch.as_tensor(raw_value, device=device, dtype=dtype)
        T_motion = _joint_motion_transform(joint, value, device=device, dtype=dtype)

        if preserve_zero_pose and joint.name in origin_offsets:
            T_compensation = invert_transform(T_joint_current) @ T_joint_initial
        else:
            T_compensation = _zero_transform(device, dtype)

        transforms[joint.child] = parent_T @ T_joint_current @ T_motion @ T_compensation

    return transforms


def compute_visual_transform(
    link_transform: torch.Tensor,
    visual: VisualSpec,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = device or link_transform.device
    T_visual = origin_transform(visual.xyz, visual.rpy, device=device, dtype=dtype)
    return link_transform @ T_visual


def transform_visual_vertices(
    vertices: torch.Tensor,
    link_transform: torch.Tensor,
    visual: VisualSpec,
) -> torch.Tensor:
    T = compute_visual_transform(link_transform, visual, device=vertices.device, dtype=vertices.dtype)
    return transform_points(T, vertices)

