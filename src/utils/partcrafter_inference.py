from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image

from src.models.briarmbg import BriaRMBG
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.download_weights import ensure_weights
from src.utils.image_utils import prepare_image
from src.utils.render_utils import (
    export_renderings,
    make_grid_for_images_or_videos,
    render_normal_views_around_mesh,
    render_single_view,
    render_views_around_mesh,
)

MAX_NUM_PARTS = 16


def validate_num_parts(num_parts: int) -> None:
    if not 1 <= num_parts <= MAX_NUM_PARTS:
        raise ValueError(f"num_parts must be in [1, {MAX_NUM_PARTS}]")


def load_models(device: str = "cuda", dtype: torch.dtype = torch.float16) -> tuple[PartCrafterPipeline, BriaRMBG]:
    partcrafter_weights_dir = ensure_weights("PartCrafter")
    rmbg_weights_dir = ensure_weights("RMBG-1.4")

    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval()

    pipe = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)
    return pipe, rmbg_net


def prepare_input_image(
    image_input: str | Path | Image.Image,
    rmbg_net: Any,
    *,
    rmbg: bool = False,
) -> Image.Image:
    if rmbg:
        return prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    if isinstance(image_input, Image.Image):
        return image_input
    return Image.open(image_input)


def sanitize_mesh_outputs(outputs: Sequence[trimesh.Trimesh | None], *, verbose: bool = False) -> list[trimesh.Trimesh]:
    sanitized_outputs: list[trimesh.Trimesh] = []
    for index, mesh in enumerate(outputs):
        if mesh is None:
            if verbose:
                print(f"WARNING: Part {index} generated None mesh, using dummy mesh")
            mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        sanitized_outputs.append(mesh)
    return sanitized_outputs


@torch.no_grad()
def run_object_inference(
    pipe: Any,
    image_input: str | Path | Image.Image,
    num_parts: int,
    rmbg_net: Any,
    seed: int,
    *,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = int(1e9),
    use_flash_decoder: bool = False,
    rmbg: bool = False,
) -> tuple[list[trimesh.Trimesh], Image.Image]:
    validate_num_parts(num_parts)
    processed_image = prepare_input_image(image_input, rmbg_net, rmbg=rmbg)

    start_time = time.time()
    outputs = pipe(
        image=[processed_image] * num_parts,
        attention_kwargs={
            "num_parts": num_parts,
            "part_positions": list(range(num_parts)),
        },
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
    ).meshes
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

    return sanitize_mesh_outputs(outputs), processed_image


def load_num_parts_from_metadata(image_folder: str | Path) -> tuple[int, Path, dict[str, Any]]:
    image_folder = Path(image_folder)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    for filename in ("metadata.json", "num_parts.json"):
        metadata_path = image_folder / filename
        if metadata_path.exists():
            with metadata_path.open("r") as handle:
                metadata = json.load(handle)
            num_parts = metadata.get("num_parts")
            if num_parts is None:
                raise ValueError(f"'num_parts' field not found in {metadata_path.name}")
            validate_num_parts(num_parts)
            return num_parts, metadata_path, metadata

    raise FileNotFoundError(f"Neither metadata.json nor num_parts.json found in {image_folder}")


def get_part_image_paths(image_folder: str | Path, num_parts: int) -> list[Path]:
    validate_num_parts(num_parts)
    image_folder = Path(image_folder)
    paths = [image_folder / f"link{i}.png" for i in range(num_parts)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing part images in {image_folder}: {missing}")
    return paths


def load_part_images(
    image_folder: str | Path,
    num_parts: int,
    rmbg_net: Any = None,
    *,
    rmbg: bool = False,
) -> list[Image.Image]:
    return [
        prepare_input_image(path, rmbg_net, rmbg=rmbg) if rmbg and rmbg_net is not None else Image.open(path)
        for path in get_part_image_paths(image_folder, num_parts)
    ]


def load_global_images(
    image_folder: str | Path,
    num_parts: int,
    rmbg_net: Any = None,
    *,
    rmbg: bool = False,
) -> list[Image.Image] | None:
    global_image_path = Path(image_folder) / "input.png"
    if not global_image_path.exists():
        return None

    global_image = prepare_input_image(global_image_path, rmbg_net, rmbg=rmbg) if rmbg and rmbg_net is not None else Image.open(global_image_path)
    return [global_image.copy() for _ in range(num_parts)]


@torch.no_grad()
def run_part_image_inference(
    pipe: Any,
    image_folder: str | Path,
    num_parts: int,
    rmbg_net: Any,
    seed: int,
    *,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = int(1e9),
    use_flash_decoder: bool = False,
    rmbg: bool = False,
    verbose: bool = False,
) -> tuple[list[trimesh.Trimesh], list[Image.Image], list[Image.Image] | None]:
    validate_num_parts(num_parts)
    part_images = load_part_images(image_folder, num_parts, rmbg_net, rmbg=rmbg)
    global_images = load_global_images(image_folder, num_parts, rmbg_net, rmbg=rmbg)
    part_positions = list(range(num_parts))

    pipe_kwargs = {
        "attention_kwargs": {
            "num_parts": num_parts,
            "part_positions": part_positions,
        },
        "num_tokens": num_tokens,
        "generator": torch.Generator(device=pipe.device).manual_seed(seed),
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "max_num_expanded_coords": max_num_expanded_coords,
        "use_flash_decoder": use_flash_decoder,
    }

    start_time = time.time()
    if global_images is not None:
        outputs = pipe(
            image=part_images,
            global_image=global_images,
            local_images=part_images,
            **pipe_kwargs,
        ).meshes
    else:
        outputs = pipe(image=part_images, **pipe_kwargs).meshes

    if verbose:
        print(f"DEBUG: Pipeline completed in {time.time() - start_time:.2f} seconds")
        print(f"DEBUG: Generated {len(outputs)} meshes")

    return sanitize_mesh_outputs(outputs, verbose=verbose), part_images, global_images


def create_export_dir(output_dir: str | Path, tag: str | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / (tag or time.strftime("%Y%m%d_%H_%M_%S"))
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def export_mesh_outputs(
    outputs: Sequence[trimesh.Trimesh],
    export_dir: str | Path,
    *,
    merged_mesh_kwargs: dict[str, Any] | None = None,
) -> trimesh.Trimesh:
    export_dir = Path(export_dir)
    for index, mesh in enumerate(outputs):
        mesh.export(export_dir / f"part_{index:02}.glb")

    merged_mesh = get_colored_mesh_composition(outputs, **(merged_mesh_kwargs or {}))
    merged_mesh.export(export_dir / "object.glb")
    return merged_mesh


def save_reference_images(images: Iterable[Image.Image], export_dir: str | Path, prefix: str) -> None:
    export_dir = Path(export_dir)
    for index, image in enumerate(images):
        image.save(export_dir / f"{prefix}_{index:02}.png")


def save_global_reference_image(global_images: list[Image.Image] | None, export_dir: str | Path) -> None:
    if global_images is not None:
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        global_images[0].save(Path(export_dir) / "input_global.png")


def render_mesh_outputs(
    merged_mesh: trimesh.Trimesh,
    export_dir: str | Path,
    *,
    processed_image: Image.Image | None = None,
    num_views: int = 36,
    radius: float = 4,
    fps: int = 18,
    use_rgba: bool = True,
    save_angle_view: bool = False,
    angle_azimuth: float = 30.0,
    angle_elevation: float = 30.0,
) -> None:
    export_dir = Path(export_dir)
    render_kwargs = {"flags": pyrender.constants.RenderFlags.RGBA} if use_rgba else {}

    rendered_images = render_views_around_mesh(
        merged_mesh,
        num_views=num_views,
        radius=radius,
        **render_kwargs,
    )
    rendered_normals = render_normal_views_around_mesh(
        merged_mesh,
        num_views=num_views,
        radius=radius,
    )

    grid_inputs: list[list[Image.Image]] = []
    if processed_image is not None:
        grid_inputs.append([processed_image] * num_views)
    grid_inputs.extend([rendered_images, rendered_normals])
    rendered_grids = make_grid_for_images_or_videos(grid_inputs, nrow=3)

    export_renderings(rendered_images, export_dir / "rendering.gif", fps=fps)
    export_renderings(rendered_normals, export_dir / "rendering_normal.gif", fps=fps)
    export_renderings(rendered_grids, export_dir / "rendering_grid.gif", fps=fps)

    rendered_images[0].save(export_dir / "rendering.png")
    rendered_normals[0].save(export_dir / "rendering_normal.png")
    rendered_grids[0].save(export_dir / "rendering_grid.png")

    if save_angle_view:
        rendered_angle = render_single_view(
            merged_mesh,
            azimuth=angle_azimuth,
            elevation=angle_elevation,
            radius=radius,
            **render_kwargs,
        )
        rendered_angle.save(export_dir / "rendering_angle.png")
