"""Continuous SPARK-style articulation optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from PIL import Image

from .losses import image_to_silhouette, mask_image_to_silhouette, silhouette_pixel_loss
from .mesh_io import load_mesh_parts_from_urdf
from .model import DifferentiableArticulationModel
from .renderer import RendererConfig, build_silhouette_renderer, render_silhouette
from .urdf_model import parse_urdf, write_refined_urdf


@dataclass
class OptimizationConfig:
    image_size: int = 256
    threshold: float = 0.9
    iterations: int = 200
    lr_origin: float = 5e-3
    lr_angle: float = 1e-2
    region_weight: float = 1.0
    edge_weight: float = 0.2
    origin_reg_weight: float = 1e-3
    angle_reg_weight: float = 1e-4
    unit_scale: float = 1.0
    default_open_angle_deg: float = 120.0
    save_debug_every: int = 0
    auto_normalize: bool = True
    preserve_zero_pose: bool = True


@dataclass
class OptimizationResult:
    final_loss: float
    origin_deltas: Dict[str, Sequence[float]]
    joint_values: Dict[str, float]
    output_urdf: Optional[Path]


def _save_silhouette(path: Path, silhouette: torch.Tensor) -> None:
    arr = (silhouette.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype("uint8")
    Image.fromarray(arr, mode="L").save(path)


def optimize_urdf_against_open_silhouette(
    *,
    urdf_path: str | Path,
    open_image_path: Optional[str | Path] = None,
    target_mask_path: Optional[str | Path] = None,
    output_urdf_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    joint_names: Optional[Sequence[str]] = None,
    initial_angle_deg: Optional[float] = None,
    config: OptimizationConfig = OptimizationConfig(),
    renderer_config: RendererConfig = RendererConfig(),
    device: torch.device | str = "cpu",
) -> OptimizationResult:
    if open_image_path is None and target_mask_path is None:
        raise ValueError("Provide either open_image_path or target_mask_path")

    device = torch.device(device)
    urdf = parse_urdf(urdf_path)
    mesh_parts = load_mesh_parts_from_urdf(urdf, unit_scale=config.unit_scale, device=device)

    initial_joint_values = None
    if initial_angle_deg is not None:
        angle_rad = torch.deg2rad(torch.tensor(float(initial_angle_deg))).item()
        selected = joint_names or [joint.name for joint in urdf.joints if joint.is_revolute]
        initial_joint_values = {name: angle_rad for name in selected}

    model = DifferentiableArticulationModel(
        urdf,
        mesh_parts,
        optimize_joint_names=joint_names,
        initial_joint_values=initial_joint_values,
        default_open_angle_rad=torch.deg2rad(torch.tensor(config.default_open_angle_deg)).item(),
        learn_origin=True,
        learn_angle=True,
        auto_normalize=config.auto_normalize,
        preserve_zero_pose=config.preserve_zero_pose,
        device=device,
    ).to(device)

    renderer_config = RendererConfig(
        image_size=config.image_size,
        radius=renderer_config.radius,
        fov=renderer_config.fov,
        elevation=renderer_config.elevation,
        azimuth=renderer_config.azimuth,
        camera_y=renderer_config.camera_y,
        target_y=renderer_config.target_y,
        faces_per_pixel=renderer_config.faces_per_pixel,
        sigma=renderer_config.sigma,
        gamma=renderer_config.gamma,
    )
    renderer = build_silhouette_renderer(renderer_config, device)

    if target_mask_path is not None:
        target = mask_image_to_silhouette(target_mask_path, config.image_size, foreground_is_white=True)
    else:
        target = image_to_silhouette(open_image_path, config.image_size, threshold=config.threshold)
    target = target.to(device)

    params = []
    origin_params = list(model.origin_parameters())
    angle_params = list(model.angle_parameters())
    if origin_params:
        params.append({"params": origin_params, "lr": config.lr_origin})
    if angle_params:
        params.append({"params": angle_params, "lr": config.lr_angle})
    if not params:
        raise ValueError("No continuous revolute parameters selected for optimization")

    optimizer = torch.optim.Adam(params)
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    final_loss = None
    for iteration in range(1, config.iterations + 1):
        optimizer.zero_grad()
        silhouette = render_silhouette(renderer, model())
        pixel_loss, terms = silhouette_pixel_loss(
            silhouette,
            target,
            region_weight=config.region_weight,
            edge_weight=config.edge_weight,
        )
        origin_reg, angle_reg = model.regularization_terms()
        loss = pixel_loss + config.origin_reg_weight * origin_reg + config.angle_reg_weight * angle_reg
        loss.backward()
        optimizer.step()
        model.clamp_angles_to_limits()
        final_loss = float(loss.detach().cpu())

        if out_dir is not None and config.save_debug_every > 0 and iteration % config.save_debug_every == 0:
            _save_silhouette(out_dir / f"iter_{iteration:04d}_silhouette.png", silhouette)
            with (out_dir / f"iter_{iteration:04d}_loss.txt").open("w", encoding="utf-8") as f:
                f.write(f"total={final_loss:.8f}\n")
                f.write(f"region={float(terms['region'].detach().cpu()):.8f}\n")
                f.write(f"edge={float(terms['edge'].detach().cpu()):.8f}\n")
                f.write(f"origin_reg={float(origin_reg.detach().cpu()):.8f}\n")
                f.write(f"angle_reg={float(angle_reg.detach().cpu()):.8f}\n")

    origin_deltas = model.learned_origin_deltas()
    joint_values = model.learned_joint_values()

    output_urdf = Path(output_urdf_path) if output_urdf_path is not None else None
    if output_urdf is not None:
        write_refined_urdf(
            urdf_path,
            output_urdf,
            origin_deltas,
            preserve_zero_pose=config.preserve_zero_pose,
        )

    return OptimizationResult(
        final_loss=float(final_loss if final_loss is not None else 0.0),
        origin_deltas=origin_deltas,
        joint_values=joint_values,
        output_urdf=output_urdf,
    )
