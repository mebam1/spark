#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align and scale one GLB file to match another GLB file.

This script aligns a target GLB mesh to a reference GLB mesh by:
1. Computing the scale factor based on bounding box dimensions
2. Scaling the target mesh (including textures)
3. Aligning the centers to match

Usage:
    # Basic alignment
    python align_glb.py --reference model1.glb --target model2.glb --output aligned.glb

    # With ICP refinement (requires similar geometry)
    python align_glb.py --reference model1.glb --target model2.glb --output aligned.glb --icp
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import trimesh


def get_mesh_info(mesh: trimesh.Trimesh) -> dict:
    """Get mesh bounding box information."""
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    dimensions = bounds[1] - bounds[0]
    max_dim = np.max(dimensions)

    # Debug: check if dimensions are suspiciously small
    if max_dim < 1e-6:
        print(f"  WARNING: Mesh has very small or zero dimensions!")
        print(f"  Bounds: min={bounds[0]}, max={bounds[1]}")
        print(f"  First 5 vertices:\n{mesh.vertices[:5]}")

    return {
        'center': center,
        'dimensions': dimensions,
        'max_dimension': max_dim,
        'min_bound': bounds[0],
        'max_bound': bounds[1]
    }


def compute_scale_factor(
    ref_info: dict,
    target_info: dict,
    method: str = 'max_dim'
) -> float:
    """
    Compute scale factor to align target to reference.

    Args:
        ref_info: Reference mesh info
        target_info: Target mesh info
        method: 'max_dim' (scale by max dimension) or 'volume' (scale by volume)

    Returns:
        Scale factor to apply to target mesh
    """
    if method == 'max_dim':
        # Check for invalid dimensions
        if target_info['max_dimension'] < 1e-8:
            raise ValueError(f"Target mesh has invalid dimensions (max_dim={target_info['max_dimension']})")
        if ref_info['max_dimension'] < 1e-8:
            raise ValueError(f"Reference mesh has invalid dimensions (max_dim={ref_info['max_dimension']})")

        scale = ref_info['max_dimension'] / target_info['max_dimension']
    elif method == 'volume':
        ref_vol = np.prod(ref_info['dimensions'])
        target_vol = np.prod(target_info['dimensions'])

        if target_vol < 1e-12:
            raise ValueError(f"Target mesh has invalid volume ({target_vol})")
        if ref_vol < 1e-12:
            raise ValueError(f"Reference mesh has invalid volume ({ref_vol})")

        scale = (ref_vol / target_vol) ** (1/3)  # Cube root for 3D
    else:
        raise ValueError(f"Unknown scale method: {method}")

    return scale


def align_mesh(
    target_mesh: trimesh.Trimesh,
    reference_mesh: trimesh.Trimesh,
    scale_method: str = 'max_dim',
    use_icp: bool = False,
    verbose: bool = True
) -> Tuple[trimesh.Trimesh, dict]:
    """
    Align target mesh to reference mesh.

    Args:
        target_mesh: Mesh to be aligned
        reference_mesh: Reference mesh to align to
        scale_method: Method for computing scale factor
        use_icp: Whether to use ICP for fine alignment
        verbose: Print alignment information

    Returns:
        Tuple of (aligned_mesh, transform_info)
    """
    # Get mesh information
    ref_info = get_mesh_info(reference_mesh)
    target_info = get_mesh_info(target_mesh)

    if verbose:
        print("\nReference mesh:")
        print(f"  Center: {ref_info['center']}")
        print(f"  Dimensions: {ref_info['dimensions']}")
        print(f"  Max dimension: {ref_info['max_dimension']:.4f}")

        print("\nTarget mesh (before alignment):")
        print(f"  Center: {target_info['center']}")
        print(f"  Dimensions: {target_info['dimensions']}")
        print(f"  Max dimension: {target_info['max_dimension']:.4f}")

    # Create a copy to avoid modifying original
    aligned = target_mesh.copy()

    # Step 1: Compute scale factor
    scale_factor = compute_scale_factor(ref_info, target_info, method=scale_method)

    if verbose:
        print(f"\nComputed scale factor: {scale_factor:.6f}")

    # Step 2: Center the target mesh at origin
    aligned.vertices -= target_info['center']

    # Step 3: Apply scaling
    # Scale vertices
    aligned.vertices *= scale_factor

    # Note: UV coordinates should NOT be scaled as they are in texture space [0,1]
    # Trimesh automatically handles this correctly

    # Step 4: Translate to reference center
    aligned.vertices += ref_info['center']

    # Get new info after alignment
    aligned_info = get_mesh_info(aligned)

    if verbose:
        print("\nTarget mesh (after alignment):")
        print(f"  Center: {aligned_info['center']}")
        print(f"  Dimensions: {aligned_info['dimensions']}")
        print(f"  Max dimension: {aligned_info['max_dimension']:.4f}")

    # Step 5: Optional ICP refinement
    icp_matrix = None
    if use_icp:
        if verbose:
            print("\nApplying ICP refinement...")

        # Check if meshes are valid for ICP
        aligned_dim = np.max(aligned_info['dimensions'])
        ref_dim = np.max(ref_info['dimensions'])

        if aligned_dim < 1e-6 or ref_dim < 1e-6:
            if verbose:
                print(f"  WARNING: Mesh dimensions too small for ICP")
                print(f"  Aligned mesh max_dim: {aligned_dim}")
                print(f"  Reference mesh max_dim: {ref_dim}")
                print(f"  Skipping ICP refinement")
        else:
            try:
                # Use trimesh's ICP implementation
                # This will fine-tune the alignment if geometries are similar
                icp_matrix, cost = trimesh.registration.icp(
                    aligned.vertices,
                    reference_mesh.vertices,
                    max_iterations=50,
                    threshold=1e-5
                )

                # Apply ICP transformation
                aligned.apply_transform(icp_matrix)

                if verbose:
                    print(f"  ICP cost: {cost:.6f}")
                    print(f"  ICP transformation applied")

            except Exception as e:
                if verbose:
                    print(f"  ICP failed: {e}")
                    print(f"  Continuing without ICP refinement")

    # Prepare transform info
    transform_info = {
        'scale_factor': scale_factor,
        'translation': ref_info['center'] - target_info['center'] * scale_factor,
        'ref_center': ref_info['center'],
        'target_center_original': target_info['center'],
        'target_center_aligned': aligned_info['center'],
        'icp_matrix': icp_matrix
    }

    return aligned, transform_info


def load_mesh(path: Path, verbose: bool = True) -> trimesh.Trimesh:
    """Load a mesh from GLB file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if verbose:
        print(f"Loading: {path.name}")

    mesh = trimesh.load(str(path), force='mesh')

    # Handle scene (multiple meshes)
    if isinstance(mesh, trimesh.Scene):
        if verbose:
            print(f"  Scene contains {len(mesh.geometry)} geometries, merging...")
        mesh = trimesh.util.concatenate(
            [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh from {path}")

    if verbose:
        print(f"  Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        if mesh.visual.defined:
            print(f"  Has textures: Yes")
        else:
            print(f"  Has textures: No")

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Align and scale target GLB to match reference GLB"
    )

    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Reference GLB file (target will be aligned to this)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target GLB file to be aligned"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output GLB file path for aligned mesh"
    )
    parser.add_argument(
        "--scale-method",
        type=str,
        default="max_dim",
        choices=["max_dim", "volume"],
        help="Method for computing scale factor (default: max_dim)"
    )
    parser.add_argument(
        "--icp",
        action="store_true",
        help="Use ICP for fine alignment (requires similar geometry)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Convert paths
    ref_path = Path(args.reference).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    # Load meshes
    if verbose:
        print("=" * 80)
        print("Loading meshes...")
        print("=" * 80)

    try:
        reference_mesh = load_mesh(ref_path, verbose=verbose)
        target_mesh = load_mesh(target_path, verbose=verbose)
    except Exception as e:
        print(f"\nError loading meshes: {e}")
        sys.exit(1)

    # Align meshes
    if verbose:
        print("\n" + "=" * 80)
        print("Aligning meshes...")
        print("=" * 80)

    try:
        aligned_mesh, transform_info = align_mesh(
            target_mesh,
            reference_mesh,
            scale_method=args.scale_method,
            use_icp=args.icp,
            verbose=verbose
        )
    except Exception as e:
        print(f"\nError during alignment: {e}")
        sys.exit(1)

    # Export aligned mesh
    if verbose:
        print("\n" + "=" * 80)
        print("Exporting aligned mesh...")
        print("=" * 80)

    try:
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export as GLB (preserves textures)
        aligned_mesh.export(str(output_path))

        if verbose:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Saved to: {output_path}")
            print(f"  File size: {file_size_mb:.2f} MB")

            print("\nAlignment summary:")
            print(f"  Scale factor: {transform_info['scale_factor']:.6f}")
            print(f"  Translation: {transform_info['translation']}")

            if transform_info['icp_matrix'] is not None:
                print(f"  ICP refinement: Applied")

            print("\n" + "=" * 80)
            print("Alignment complete!")
            print("=" * 80)

    except Exception as e:
        print(f"\nError exporting mesh: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
