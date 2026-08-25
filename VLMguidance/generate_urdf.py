#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate URDF files for GAPartNet objects from metadata.json

This script generates mobility.urdf files for each object in GAPartNet_PartNetMobility/selected,
using the structure defined in metadata.json and placeholder GLB mesh files (part_00.glb, part_01.glb, etc.)

Usage:
    python VLMguidance/generate_urdf.py
    python VLMguidance/generate_urdf.py --input-dir GAPartNet_PartNetMobility/selected
"""

import json
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _value_to_xyz_string(value: Any, default: str = "0 0 0") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return f"{float(value[0])} {float(value[1])} {float(value[2])}"
    return default


def _safe_part_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


MeshMapValue = Union[str, List[str]]


def _as_mesh_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _resolve_one_mesh_filename(
    candidate: str,
    object_dir: Path,
    mesh_dir: Optional[Path] = None,
    absolute_mesh_paths: bool = False,
) -> str:
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        resolved_path = candidate_path.resolve()
    elif mesh_dir is not None:
        resolved_path = (mesh_dir / candidate_path).resolve()
    else:
        resolved_path = (object_dir / candidate_path).resolve()

    if absolute_mesh_paths:
        return resolved_path.as_posix()

    return Path(os.path.relpath(resolved_path, object_dir.resolve())).as_posix()


def _resolve_mesh_filenames(
    part: Dict[str, Any],
    link_num: int,
    object_dir: Path,
    mesh_map: Optional[Dict[str, MeshMapValue]] = None,
    mesh_dir: Optional[Path] = None,
    mesh_pattern: str = "part_{index:02d}.glb",
    absolute_mesh_paths: bool = False,
) -> List[str]:
    label = part.get("label", f"link{link_num}")
    part_name = part.get("name", f"part_{link_num}")

    candidate = (
        part.get("mesh_filenames")
        or part.get("mesh_paths")
        or part.get("mesh_filename")
        or part.get("mesh_path")
    )
    if candidate is None and mesh_map:
        candidate = mesh_map.get(label) or mesh_map.get(part_name)
    if candidate is None:
        candidate = mesh_pattern.format(index=link_num, label=label, name=_safe_part_name(part_name))

    candidates = _as_mesh_list(candidate) or []
    return [
        _resolve_one_mesh_filename(
            item,
            object_dir=object_dir,
            mesh_dir=mesh_dir,
            absolute_mesh_paths=absolute_mesh_paths,
        )
        for item in candidates
    ]


def generate_urdf_content(
    metadata: Dict[str, Any],
    object_dir: Path,
    mesh_map: Optional[Dict[str, MeshMapValue]] = None,
    mesh_dir: Optional[Path] = None,
    mesh_pattern: str = "part_{index:02d}.glb",
    absolute_mesh_paths: bool = False,
) -> str:
    """
    Generate URDF XML content from metadata.

    Args:
        metadata: Dictionary containing object metadata
        object_dir: Path to the object directory (for relative mesh paths)

    Returns:
        str: URDF XML content
    """
    object_name = metadata.get('object_name', 'object')
    num_parts = metadata.get('num_parts', 0)
    parts = metadata.get('parts', [])

    # Start URDF
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{object_name}">')

    # Add base link (empty, just for structure)
    lines.append('  <link name="base"/>')
    lines.append('')

    # Process each part
    for i, part in enumerate(parts):
        label = part.get('label', f'link{i}')
        part_name = part.get('name', f'part_{i}')
        parent = part.get('parent', 'base')
        joint_type = part.get('joint_type', 'fixed')
        axis_str = part.get('axis', '0 0 0')
        origin_xyz = _value_to_xyz_string(part.get('origin_xyz') or part.get('joint_origin_xyz'))
        origin_rpy = _value_to_xyz_string(part.get('origin_rpy') or part.get('joint_origin_rpy'))
        limit_lower = part.get('limit_lower', '0')
        limit_upper = part.get('limit_upper', '0')

        # Extract link number from label (e.g., "link0" -> 0)
        if 'link' in label:
            link_num_str = label.replace('link', '')
            link_num = int(link_num_str) if link_num_str.isdigit() else i
        else:
            link_num = i
        mesh_files = _resolve_mesh_filenames(
            part=part,
            link_num=link_num,
            object_dir=object_dir,
            mesh_map=mesh_map,
            mesh_dir=mesh_dir,
            mesh_pattern=mesh_pattern,
            absolute_mesh_paths=absolute_mesh_paths,
        )

        # Add link with visual mesh
        lines.append(f'  <link name="{label}">')
        for mesh_file in mesh_files:
            lines.append('    <visual>')
            lines.append('      <geometry>')
            lines.append(f'        <mesh filename="{mesh_file}"/>')
            lines.append('      </geometry>')
            lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            lines.append('    </visual>')
        lines.append('  </link>')
        lines.append('')

        # Add joint connecting to parent
        joint_name = f'{parent}_to_{label}'
        lines.append(f'  <joint name="{joint_name}" type="{joint_type}">')
        lines.append(f'    <parent link="{parent}"/>')
        lines.append(f'    <child link="{label}"/>')
        lines.append(f'    <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>')

        # Add axis for revolute/prismatic joints
        if joint_type in ('revolute', 'prismatic', 'continuous'):
            lines.append(f'    <axis xyz="{axis_str}"/>')

            # Add limits for revolute/prismatic joints
            if joint_type in ('revolute', 'prismatic'):
                lines.append(f'    <limit lower="{limit_lower}" upper="{limit_upper}" effort="100" velocity="1.0"/>')

        lines.append('  </joint>')
        lines.append('')

    # Close robot tag
    lines.append('</robot>')

    return '\n'.join(lines)


def process_object_directory(
    object_dir: Path,
    verbose: bool = True,
    mesh_map: Optional[Dict[str, MeshMapValue]] = None,
    mesh_dir: Optional[Path] = None,
    mesh_pattern: str = "part_{index:02d}.glb",
    absolute_mesh_paths: bool = False,
) -> bool:
    """
    Process a single object directory and generate mobility.urdf.

    Args:
        object_dir: Path to object directory
        verbose: Whether to print verbose output

    Returns:
        bool: True if successful, False otherwise
    """
    object_id = object_dir.name
    metadata_path = object_dir / 'metadata.json'
    output_path = object_dir / 'mobility.urdf'

    # Check if metadata.json exists
    if not metadata_path.exists():
        if verbose:
            print(f"[{object_id}] SKIP: metadata.json not found")
        return False

    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Generate URDF content
        urdf_content = generate_urdf_content(
            metadata,
            object_dir,
            mesh_map=mesh_map,
            mesh_dir=mesh_dir,
            mesh_pattern=mesh_pattern,
            absolute_mesh_paths=absolute_mesh_paths,
        )

        # Write URDF file
        with open(output_path, 'w') as f:
            f.write(urdf_content)

        if verbose:
            num_parts = metadata.get('num_parts', 0)
            print(f"[{object_id}] ✓ Generated mobility.urdf ({num_parts} parts)")

        return True

    except Exception as e:
        if verbose:
            print(f"[{object_id}] ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate URDF files from metadata.json for GAPartNet objects"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="GAPartNet_PartNetMobility/selected",
        help="Input directory containing object subdirectories (default: GAPartNet_PartNetMobility/selected)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Process input-dir as single object directory (not batch mode)"
    )
    parser.add_argument(
        "--mesh-map",
        type=str,
        default=None,
        help="Optional JSON mapping from link labels or semantic part names to existing mesh paths"
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help="Optional directory containing existing segmented mesh files"
    )
    parser.add_argument(
        "--mesh-pattern",
        type=str,
        default="part_{index:02d}.glb",
        help="Pattern for mesh filenames when metadata has no mesh_filename (fields: index, label, name)"
    )
    parser.add_argument(
        "--absolute-mesh-paths",
        action="store_true",
        help="Write absolute mesh paths into the URDF instead of paths relative to the object directory"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    verbose = not args.quiet
    mesh_map = None
    if args.mesh_map:
        with open(args.mesh_map, 'r', encoding='utf-8') as f:
            mesh_map = json.load(f)
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else None

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    # Single directory mode - process input_dir directly
    if args.single:
        print(f"Processing single directory: {input_dir}")
        if process_object_directory(
            input_dir,
            verbose=verbose,
            mesh_map=mesh_map,
            mesh_dir=mesh_dir,
            mesh_pattern=args.mesh_pattern,
            absolute_mesh_paths=args.absolute_mesh_paths,
        ):
            print("✓ Success")
        else:
            print("✗ Failed")
        return

    # Get all subdirectories
    subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not subdirs:
        print(f"ERROR: No subdirectories found in {input_dir}")
        return

    print(f"Processing {len(subdirs)} object directories...")
    print(f"Input: {input_dir}")
    print(f"{'='*80}\n")

    # Process each directory
    success_count = 0
    fail_count = 0

    for subdir in subdirs:
        if process_object_directory(
            subdir,
            verbose=verbose,
            mesh_map=mesh_map,
            mesh_dir=mesh_dir,
            mesh_pattern=args.mesh_pattern,
            absolute_mesh_paths=args.absolute_mesh_paths,
        ):
            success_count += 1
        else:
            fail_count += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total:      {len(subdirs)}")
    print(f"  Success:    {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"{'='*80}")

    # Show example output
    if success_count > 0:
        first_success = None
        for subdir in subdirs:
            if (subdir / 'mobility.urdf').exists():
                first_success = subdir
                break

        if first_success:
            print(f"\nExample output: {first_success / 'mobility.urdf'}")
            print(f"Mesh placeholders: part_00.glb, part_01.glb, ...")


if __name__ == "__main__":
    main()
