"""Torch SE(3) helpers used by the differentiable FK path."""

from __future__ import annotations

from typing import Sequence

import torch


def as_vec3(
    values: Sequence[float] | torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {tuple(tensor.shape)}")
    return tensor


def rpy_to_matrix(
    rpy: Sequence[float] | torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert URDF roll-pitch-yaw to a rotation matrix.

    URDF rpy is represented as fixed-axis roll, pitch, yaw, equivalent to
    Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    rpy_t = as_vec3(rpy, device=device, dtype=dtype)
    r, p, y = rpy_t.unbind()
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)

    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ]
    )


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation matrix for a fixed joint axis and scalar angle."""
    axis = axis / (torch.linalg.norm(axis) + 1e-9)
    x, y, z = axis.unbind()
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c

    return torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s]),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s]),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]),
        ]
    )


def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build a homogeneous 4x4 transform from rotation and translation."""
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def translation_transform(t: torch.Tensor) -> torch.Tensor:
    R = torch.eye(3, dtype=t.dtype, device=t.device)
    return make_transform(R, t)


def origin_transform(
    xyz: Sequence[float] | torch.Tensor,
    rpy: Sequence[float] | torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    R = rpy_to_matrix(rpy, device=device, dtype=dtype)
    t = as_vec3(xyz, device=device, dtype=dtype)
    return make_transform(R, t)


def invert_transform(T: torch.Tensor) -> torch.Tensor:
    """Invert a rigid homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R.transpose(0, 1)
    T_inv[:3, 3] = -(R.transpose(0, 1) @ t)
    return T_inv


def transform_points(T: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous transform to Nx3 vertices."""
    ones = torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    vertices_h = torch.cat([vertices, ones], dim=1)
    return (T @ vertices_h.T).T[:, :3]

