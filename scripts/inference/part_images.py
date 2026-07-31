"""
PartCrafter Per-Part Images Inference Script

This script performs inference using PartCrafter with per-part images and optional global image.
It uses individual images for each part (local attention) and optionally a global image for global attention.
If no global image (rendering.png) is provided, it falls back to using per-part images for both attentions.

Usage:
    python scripts/inference/part_images.py \
        --image_folder /path/to/part/images \
        --output_dir ./results \
        --render

Expected folder structure:
    /path/to/part/images/
    ├── metadata.json (or num_parts.json)  # Contains num_parts and other metadata
    ├── rendering.png                      # Global image for global attention
    ├── link0.png                          # Image for part 0
    ├── link1.png                          # Image for part 1
    ├── link2.png                          # Image for part 2
    └── ...

The number of part image files (link0.png to link{num_parts-1}.png) is determined from
metadata.json or num_parts.json (both formats are supported).
The global image (rendering.png) is optional but recommended for better results.

Output structure:
    ./results/{timestamp}/
    ├── part_00.glb              # Individual part meshes
    ├── part_01.glb
    ├── part_02.glb
    ├── object.glb               # Merged colored mesh
    ├── input_part_00.png        # Copy of input part images for reference
    ├── input_part_01.png
    ├── input_part_02.png
    ├── input_global.png         # Copy of global image for reference (if provided)
    ├── rendering.gif            # Rendered animation (if --render)
    ├── rendering_normal.gif     # Normal rendering animation
    ├── rendering_grid.gif       # Grid rendering animation
    ├── rendering.png            # Single frame renders
    ├── rendering_normal.png
    ├── rendering_grid.png
    └── rendering_angle.png      # Angled view (45° right, 45° up)

Requirements:
    - PartCrafter model weights in pretrained_weights/PartCrafter/
    - Per-part images named as link0.png, link1.png, etc.
    - CUDA-compatible GPU
"""

import argparse
import os
import torch
from accelerate.utils import set_seed

from src.utils.partcrafter_inference import (
    create_export_dir,
    export_mesh_outputs,
    load_models,
    load_num_parts_from_metadata,
    render_mesh_outputs,
    run_part_image_inference,
    save_global_reference_image,
    save_reference_images,
)

# =========================
# ====== CLI SETUP ========
# =========================

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser(description="PartCrafter inference with per-part images")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to folder containing part images (link0.png, link1.png, ...) and metadata.json")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--max_num_expanded_coords", type=int, default=1e9)
    parser.add_argument("--use_flash_decoder", action="store_true")
    parser.add_argument("--rmbg", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("DEBUG: PartCrafter Per-Part Images Inference")
    print("=" * 80)
    print(f"DEBUG: Image folder: {args.image_folder}")
    print(f"DEBUG: Output directory: {args.output_dir}")
    print(f"DEBUG: Seed: {args.seed}")
    print(f"DEBUG: Guidance scale: {args.guidance_scale}")
    print(f"DEBUG: Use RMBG: {args.rmbg}")
    print("=" * 80)

    # =========================
    # ====== PREPARE =========
    # =========================

    if not os.path.exists(args.image_folder):
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")

    num_parts, metadata_path, _ = load_num_parts_from_metadata(args.image_folder)
    print(f"DEBUG: Loading metadata from: {metadata_path}")
    print(f"DEBUG: Number of parts (from {metadata_path.name}): {num_parts}")

    print("DEBUG: Loading PartCrafter weights and RMBG model...")
    pipe, rmbg_net = load_models(device=device, dtype=dtype)

    set_seed(args.seed)
    print(f"DEBUG: Set random seed to: {args.seed}")

    # =========================
    # ===== INFERENCE ========
    # =========================

    print(f"DEBUG: Starting inference...")
    outputs, part_images, global_images = run_part_image_inference(
        pipe=pipe,
        image_folder=args.image_folder,
        num_parts=num_parts,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_num_expanded_coords=args.max_num_expanded_coords,
        use_flash_decoder=args.use_flash_decoder,
        rmbg=args.rmbg,
        verbose=True,
    )

    # =========================
    # ======== SAVE ==========
    # =========================

    print(f"DEBUG: Inference completed, saving results...")

    export_dir = create_export_dir(args.output_dir, args.tag)
    print(f"DEBUG: Saving results to: {export_dir}")

    print(f"DEBUG: Saving {len(outputs)} individual part meshes...")
    print(f"DEBUG: Creating and saving merged mesh...")
    merged_mesh = export_mesh_outputs(outputs, export_dir, merged_mesh_kwargs={"is_random": False, "alpha": 255})

    print(f"DEBUG: Saving input part images for reference...")
    save_reference_images(part_images, export_dir, "input_part")

    if global_images is not None:
        save_global_reference_image(global_images, export_dir)
        print(f"DEBUG: Saved global image to: {export_dir / 'input_global.png'}")

    print(f"✅ Generated {len(outputs)} parts and saved to {export_dir}")

    # =========================
    # ======= RENDER =========
    # =========================

    if args.render:
        print("DEBUG: Starting rendering...")
        render_mesh_outputs(
            merged_mesh,
            export_dir,
            num_views=36,
            radius=4,
            fps=18,
            use_rgba=True,
            save_angle_view=True,
            angle_azimuth=30.0,
            angle_elevation=30.0,
        )

        print("✅ Rendering completed!")

    print("=" * 80)
    print(f"🎉 ALL RESULTS SAVED TO: {export_dir}")
    print("   📁 Individual part meshes: part_00.glb, part_01.glb, ...")
    print("   🔗 Merged object mesh: object.glb")
    print("   🖼️  Input part images: input_part_00.png, input_part_01.png, ...")
    if global_images is not None:
        print("   🌐 Input global image: input_global.png")
    if args.render:
        print("   🎬 Rendered animations: rendering.gif, rendering_normal.gif, rendering_grid.gif")
        print("   📸 Rendered frames: rendering.png, rendering_normal.png, rendering_grid.png")
        print("   📐 Angled view: rendering_angle.png (45° right, 45° up)")
    print("=" * 80)