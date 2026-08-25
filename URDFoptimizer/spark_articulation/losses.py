"""SPARK-style silhouette losses for articulation refinement."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def image_to_silhouette(path: str | Path, image_size: int, threshold: float = 0.9) -> torch.Tensor:
    """Convert an open-state image to a foreground mask using a white-background heuristic."""
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = torch.as_tensor(list(img.getdata()), dtype=torch.float32).view(image_size, image_size, 3) / 255.0
    brightness = arr.mean(dim=2)
    return (brightness < threshold).float()


def mask_image_to_silhouette(path: str | Path, image_size: int, *, foreground_is_white: bool = True) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    mask = torch.as_tensor(list(img.getdata()), dtype=torch.float32).view(image_size, image_size) / 255.0
    return (mask > 0.5).float() if foreground_is_white else (mask < 0.5).float()


def silhouette_overlap_loss(rendered: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rendered = rendered.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    intersection = torch.sum(rendered * target)
    union = torch.sum(rendered + target - rendered * target)
    return 1.0 - (intersection + eps) / (union + eps)


def _edge_map(mask: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3) / 8.0
    x = mask.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8).squeeze(0).squeeze(0)


def edge_alignment_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(_edge_map(rendered), _edge_map(target))


def silhouette_pixel_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    region_weight: float = 1.0,
    edge_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    region = silhouette_overlap_loss(rendered, target)
    edge = edge_alignment_loss(rendered, target)
    loss = region_weight * region + edge_weight * edge
    return loss, {"region": region, "edge": edge}

