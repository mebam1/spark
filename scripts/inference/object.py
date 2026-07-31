import argparse
import torch
from accelerate.utils import set_seed

from src.utils.partcrafter_inference import (
    create_export_dir,
    export_mesh_outputs,
    load_models,
    render_mesh_outputs,
    run_object_inference,
    validate_num_parts,
)

# =========================
# ====== CLI SETUP ========
# =========================

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--num_parts", type=int, required=True, help="number of parts to generate")
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

    # =========================
    # ====== PREPARE =========
    # =========================

    validate_num_parts(args.num_parts)

    pipe, rmbg_net = load_models(device=device, dtype=dtype)

    set_seed(args.seed)

    # =========================
    # ===== INFERENCE ========
    # =========================

    outputs, processed_image = run_triposg(
        pipe=pipe,
        image_input=args.image_path,
        num_parts=args.num_parts,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_num_expanded_coords=args.max_num_expanded_coords,
        use_flash_decoder=args.use_flash_decoder,
        rmbg=args.rmbg,
    )

    # =========================
    # ======== SAVE ==========
    # =========================

    export_dir = create_export_dir(args.output_dir, args.tag)
    merged_mesh = export_mesh_outputs(outputs, export_dir)
    print(f"Generated {len(outputs)} parts and saved to {export_dir}")

    # =========================
    # ======= RENDER =========
    # =========================

    if args.render:
        print("Start rendering...")
        render_mesh_outputs(
            merged_mesh,
            export_dir,
            processed_image=processed_image,
        )
        print("Rendering done.")

