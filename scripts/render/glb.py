"""
GLB/OBJ Rendering Script with GIF Animation

Renders a GLB file or directory of OBJ+MTL files from two viewpoints and creates a 360° rotation GIF:
1. Front view (azimuth=0°, elevation=0°)
2. Angled view (azimuth=30°, elevation=30°) - optional, skipped with --single
3. 360° rotation GIF animation

Usage:
    # Render GLB with default texture
    python scripts/render/glb.py --input /path/to/model.glb --output_dir ./output

    # Render only front view
    python scripts/render/glb.py --input /path/to/model.glb --output_dir ./output --single

    # Render with custom light intensity
    python scripts/render/glb.py --input /path/to/model.glb --output_dir ./output --light 10.0

    # Render OBJ+MTL directory with colored parts and GIF
    python scripts/render/glb.py --input /path/to/textured_objs --output_dir ./output --color --gif

    # Render with custom GIF settings
    python scripts/render/glb.py --input /path/to/model.glb --output_dir ./output --gif --num_views 72 --fps 24
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyrender
import numpy as np
import trimesh

from src.utils.render_utils import render_single_view, render_views_around_mesh, export_renderings
from src.utils.data_utils import get_colored_mesh_composition


def load_mesh_from_path(input_path: str) -> trimesh.Scene:
    """
    Load mesh from GLB file or directory of OBJ+MTL files.

    Args:
        input_path: Path to GLB file or directory containing OBJ+MTL files

    Returns:
        trimesh.Scene object
    """
    path = Path(input_path)

    if path.is_file():
        # Load single file (GLB, OBJ, etc.)
        print(f"Loading single mesh file: {input_path}")
        mesh = trimesh.load(input_path)
        if isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.Scene(mesh)
        return mesh

    elif path.is_dir():
        # Load all OBJ files from directory
        obj_files = sorted(glob(os.path.join(input_path, "*.obj")))
        if not obj_files:
            raise FileNotFoundError(f"No OBJ files found in directory: {input_path}")

        print(f"Loading {len(obj_files)} OBJ files from directory: {input_path}")

        # Load all meshes and combine into a scene
        scene = trimesh.Scene()
        for i, obj_file in enumerate(obj_files):
            print(f"  Loading {os.path.basename(obj_file)}...")
            mesh = trimesh.load(obj_file)
            if isinstance(mesh, trimesh.Scene):
                # If it's already a scene, add all geometries
                for name, geom in mesh.geometry.items():
                    scene.add_geometry(geom, node_name=f"part_{i}_{name}")
            else:
                # Single mesh
                scene.add_geometry(mesh, node_name=f"part_{i}")

        print(f"Successfully loaded {len(scene.geometry)} mesh parts")
        return scene

    else:
        raise ValueError(f"Invalid input path: {input_path}")


def render_mesh(
    input_path: str,
    output_dir: str,
    add_color: bool = False,
    create_gif: bool = False,
    num_views: int = 36,
    fps: int = 18,
    radius: float = 4.0,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    single_view: bool = False,
    light_intensity: float = 5.0,
):
    """
    Render a mesh from GLB file or OBJ+MTL directory.

    Generates:
    1. Front view image (azimuth=0°, elevation=0°)
    2. Angled view image (azimuth=30°, elevation=30°) - unless single_view is True
    3. Optional: 360° rotation GIF animation

    Args:
        input_path: Path to GLB file or directory containing OBJ+MTL files
        output_dir: Directory to save rendered images
        add_color: If True, add colors to mesh parts; otherwise use default texture
        create_gif: If True, generate 360° rotation GIF
        num_views: Number of views for GIF rotation (default: 36)
        fps: Frames per second for GIF (default: 18)
        radius: Camera distance from object (default: 4.0)
        single_view: If True, only render front view (default: False)
        light_intensity: Light intensity for rendering (default: 5.0)
    """
    print("=" * 80)
    print(f"Rendering mesh from: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Add color: {add_color}")
    print(f"Single view mode: {single_view}")
    print(f"Front view camera: azimuth={azimuth}, elevation={elevation}")
    print(f"Light intensity: {light_intensity}")
    print(f"Create GIF: {create_gif}")
    if create_gif:
        print(f"GIF settings: {num_views} views @ {fps} fps")
    print("=" * 80)

    # Check if input exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Load mesh
    mesh = load_mesh_from_path(input_path)
    print(f"Loaded mesh type: {type(mesh)}")

    # Add color if requested
    if add_color:
        print("Adding colors to mesh parts...")
        if isinstance(mesh, trimesh.Scene):
            # Multi-part mesh: use color composition
            mesh = get_colored_mesh_composition(mesh, is_random=False, alpha=255)
            print(f"Applied colors to {len(mesh.geometry)} parts")
        else:
            # Single mesh: add a single color
            RGB = [(248, 210, 161)]  # Default peach color
            color = np.array(RGB[0])
            color_with_alpha = np.append(color, 255)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=color_with_alpha,
            )
            print("Applied single color to mesh")
    else:
        print("Using default texture (no color added)")

    # Render front view with the requested camera orientation.
    print(f"\nRendering front view (azimuth={azimuth}°, elevation={elevation}°)...")
    front_view = render_single_view(
        mesh,
        azimuth=azimuth,
        elevation=elevation,
        radius=radius,
        light_intensity=light_intensity,
        flags=pyrender.constants.RenderFlags.RGBA,
    )
    print("Front view rendered successfully")

    # Render angled view (azimuth=30°, elevation=30°) - same as inference_partcrafter_part.py
    if not single_view:
        print("Rendering angled view (azimuth=30°, elevation=30°)...")
        angle_view = render_single_view(
            mesh,
            azimuth=30.0,
            elevation=30.0,
            radius=radius,
            light_intensity=light_intensity,
            flags=pyrender.constants.RenderFlags.RGBA,
        )
        print("Angled view rendered successfully")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")

    # Save static images
    front_path = os.path.join(output_dir, "front_view.png")
    front_view.save(front_path)

    print("\n" + "=" * 80)
    print("✅ Static images rendered!")
    print(f"   📸 Front view saved to: {front_path}")

    if not single_view:
        angle_path = os.path.join(output_dir, "angle_view.png")
        angle_view.save(angle_path)
        print(f"   📸 Angled view saved to: {angle_path}")

    # Generate 360° rotation GIF if requested
    if create_gif:
        print("\n" + "=" * 80)
        print(f"Generating 360° rotation GIF with {num_views} views...")

        rendered_images = render_views_around_mesh(
            mesh,
            num_views=num_views,
            radius=radius,
            light_intensity=light_intensity,
            flags=pyrender.constants.RenderFlags.RGBA,
        )
        print(f"Rendered {len(rendered_images)} views")

        gif_path = os.path.join(output_dir, "rotation.gif")
        export_renderings(rendered_images, gif_path, fps=fps)

        print(f"✅ GIF animation saved to: {gif_path}")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render GLB file or OBJ+MTL directory with optional 360° GIF animation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to GLB file or directory containing OBJ+MTL files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./render_output",
        help="Output directory for rendered images (default: ./render_output)"
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Add colors to mesh parts (default: use original texture)"
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Generate 360° rotation GIF animation"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=36,
        help="Number of views for GIF rotation (default: 36)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=18,
        help="Frames per second for GIF (default: 18)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=4.0,
        help="Camera distance from object (default: 4.0)"
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="Front-view camera azimuth in degrees (default: 0.0)"
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        help="Front-view camera elevation in degrees (default: 0.0)"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Only render front view (skip angled view)"
    )
    parser.add_argument(
        "--light",
        type=float,
        default=5.0,
        help="Light intensity for rendering (default: 5.0)"
    )

    args = parser.parse_args()

    render_mesh(
        input_path=args.input,
        output_dir=args.output_dir,
        add_color=args.color,
        create_gif=args.gif,
        num_views=args.num_views,
        fps=args.fps,
        radius=args.radius,
        azimuth=args.azimuth,
        elevation=args.elevation,
        single_view=args.single,
        light_intensity=args.light,
    )
