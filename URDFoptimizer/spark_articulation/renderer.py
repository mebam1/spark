"""PyTorch3D soft silhouette rendering with fixed camera parameters."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RendererConfig:
    image_size: int = 256
    radius: float = 4.0
    fov: float = 40.0
    elevation: float = 0.0
    azimuth: float = 0.0
    faces_per_pixel: int = 50
    sigma: float = 1e-5
    gamma: float = 1e-6


def build_silhouette_renderer(config: RendererConfig, device: torch.device):
    try:
        from pytorch3d.renderer import (
            BlendParams,
            FoVPerspectiveCameras,
            MeshRasterizer,
            MeshRenderer,
            RasterizationSettings,
            SoftSilhouetteShader,
            look_at_view_transform,
        )
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        raise RuntimeError("PyTorch3D is required for differentiable silhouette rendering") from exc

    R, T = look_at_view_transform(
        dist=config.radius,
        elev=config.elevation,
        azim=config.azimuth,
        device=device,
    )
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=config.fov)

    blur_radius = torch.log(torch.tensor(1.0 / 1e-4 - 1.0, device=device)) * config.sigma
    raster_settings = RasterizationSettings(
        image_size=config.image_size,
        blur_radius=float(blur_radius.detach().cpu()),
        faces_per_pixel=config.faces_per_pixel,
        cull_backfaces=True,
        perspective_correct=True,
        bin_size=None,
    )
    blend_params = BlendParams(
        sigma=config.sigma,
        gamma=config.gamma,
        background_color=(0.0, 0.0, 0.0),
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )


def render_silhouette(renderer, meshes) -> torch.Tensor:
    images = renderer(meshes)
    return images[..., 3].squeeze(0).clamp(0.0, 1.0)

