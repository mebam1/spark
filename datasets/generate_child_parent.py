#!/usr/bin/env python3
"""
Generate parent_indices.json for each object in preprocessed_data.

This script:
1. Iterates through all subfolders in preprocessed_data
2. Maps folder names like 148_mid, 148_max -> 148 base ID
3. Finds corresponding mesh/partnet-mobility/{base_id}/mobility.urdf
4. Parses URDF to extract parent-child relationships from joints
5. Saves parent_indices.json to preprocessed_data/{subfolder}/

Parent-child relationship:
- link_0 = first part (index 0)
- link_1 = second part (index 1)
- ...
- parent="base" means root node (parent_indices = -1)
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def extract_base_id(folder_name: str) -> str:
    """
    Extract base ID from folder name.

    Examples:
        148_mid -> 148
        148_max -> 148
        100013 -> 100013

    Args:
        folder_name: Folder name (e.g., "148_mid", "100013")

    Returns:
        base_id: Base ID without suffix (e.g., "148", "100013")
    """
    # Remove _mid or _max suffix if present
    if folder_name.endswith('_mid') or folder_name.endswith('_max'):
        return folder_name.rsplit('_', 1)[0]
    return folder_name


def parse_urdf_hierarchy(urdf_path: str) -> Tuple[List[int], int]:
    """
    Parse URDF file to extract parent-child relationships.

    URDF structure:
        <link name="link_0">...</link>
        <link name="link_1">...</link>
        <joint name="joint_0">
            <child link="link_0"/>
            <parent link="link_4"/>  <!-- link_0's parent is link_4 -->
        </joint>
        <joint name="joint_4">
            <child link="link_4"/>
            <parent link="base"/>    <!-- link_4 is root -->
        </joint>

    Args:
        urdf_path: Path to mobility.urdf

    Returns:
        parent_indices: List[int], parent index for each part (-1 for root)
        num_parts: int, total number of parts (links)
    """
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Step 1: Find all links (exclude "base" which is special)
        links = []
        for link in root.findall('link'):
            link_name = link.get('name')
            if link_name != 'base' and link_name.startswith('link_'):
                links.append(link_name)

        # Sort links by their index (link_0, link_1, ..., link_N)
        links.sort(key=lambda x: int(x.split('_')[1]))
        num_parts = len(links)

        if num_parts == 0:
            print(f"Warning: No links found in {urdf_path}")
            return [], 0

        # Step 2: Build link_name -> index mapping
        link_to_index = {link_name: idx for idx, link_name in enumerate(links)}

        # Step 3: Initialize parent_indices (default to -1 for root)
        parent_indices = [-1] * num_parts

        # Step 4: Parse joints to find parent-child relationships
        for joint in root.findall('joint'):
            child_elem = joint.find('child')
            parent_elem = joint.find('parent')

            if child_elem is None or parent_elem is None:
                continue

            child_link = child_elem.get('link')
            parent_link = parent_elem.get('link')

            # Only process if child is in our link list
            if child_link not in link_to_index:
                continue

            child_idx = link_to_index[child_link]

            # Check if parent is "base" (root) or another link
            if parent_link == 'base':
                parent_indices[child_idx] = -1  # Root node
            elif parent_link in link_to_index:
                parent_idx = link_to_index[parent_link]
                parent_indices[child_idx] = parent_idx
            else:
                # Parent not in our link list, treat as root
                parent_indices[child_idx] = -1

        return parent_indices, num_parts

    except Exception as e:
        print(f"Error parsing {urdf_path}: {e}")
        return [], 0


def validate_parent_indices(parent_indices: List[int], num_parts: int) -> bool:
    """
    Validate parent_indices for correctness.

    Checks:
    1. Length matches num_parts
    2. No self-referencing (parent_indices[i] != i)
    3. Parent indices are in valid range
    4. No cycles in the hierarchy

    Args:
        parent_indices: List of parent indices
        num_parts: Number of parts

    Returns:
        valid: True if valid, False otherwise
    """
    if len(parent_indices) != num_parts:
        print(f"Error: Length mismatch {len(parent_indices)} != {num_parts}")
        return False

    for i, parent_idx in enumerate(parent_indices):
        # Check range
        if parent_idx >= num_parts:
            print(f"Error: Invalid parent index {parent_idx} >= {num_parts} for part {i}")
            return False

        # Check self-reference
        if parent_idx == i:
            print(f"Error: Part {i} cannot be its own parent")
            return False

    # Check for cycles (optional but recommended)
    def has_cycle(start_idx: int) -> bool:
        visited = set()
        current = start_idx
        while current >= 0:
            if current in visited:
                return True  # Cycle detected
            visited.add(current)
            current = parent_indices[current]
        return False

    for i in range(num_parts):
        if has_cycle(i):
            print(f"Error: Cycle detected starting from part {i}")
            return False

    return True


def process_one_object(
    object_folder: str,
    preprocessed_data_dir: str,
    partnet_mobility_dir: str,
    force: bool = False
) -> Tuple[str, bool, Optional[str]]:
    """
    Process one object to generate parent_indices.json.

    Args:
        object_folder: Folder name (e.g., "148_mid", "100013")
        preprocessed_data_dir: Path to preprocessed_data
        partnet_mobility_dir: Path to mesh/partnet-mobility
        force: If True, overwrite existing parent_indices.json

    Returns:
        (object_folder, success, error_message)
    """
    # Extract base ID
    base_id = extract_base_id(object_folder)

    # Paths
    object_dir = os.path.join(preprocessed_data_dir, object_folder)
    output_path = os.path.join(object_dir, 'parent_indices.json')
    urdf_path = os.path.join(partnet_mobility_dir, base_id, 'mobility.urdf')

    # Check if output already exists
    if os.path.exists(output_path) and not force:
        return (object_folder, True, "Already exists (skipped)")

    # Check if URDF exists
    if not os.path.exists(urdf_path):
        return (object_folder, False, f"URDF not found: {urdf_path}")

    # Parse URDF
    parent_indices, num_parts = parse_urdf_hierarchy(urdf_path)

    if num_parts == 0:
        return (object_folder, False, "No parts found in URDF")

    # Validate
    if not validate_parent_indices(parent_indices, num_parts):
        return (object_folder, False, "Validation failed")

    # Save to JSON
    try:
        data = {"parent_indices": parent_indices}
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return (object_folder, True, None)

    except Exception as e:
        return (object_folder, False, f"Failed to save: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate parent_indices.json from URDF files"
    )
    parser.add_argument(
        '--preprocessed-data',
        type=str,
        default='preprocessed_data',
        help='Path to preprocessed_data directory'
    )
    parser.add_argument(
        '--partnet-mobility',
        type=str,
        default='mesh/partnet-mobility',
        help='Path to mesh/partnet-mobility directory'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing parent_indices.json'
    )
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Test on a single object folder (e.g., "148_mid")'
    )

    args = parser.parse_args()

    preprocessed_data_dir = args.preprocessed_data
    partnet_mobility_dir = args.partnet_mobility

    # Validate directories
    if not os.path.exists(preprocessed_data_dir):
        print(f"Error: {preprocessed_data_dir} does not exist")
        return

    if not os.path.exists(partnet_mobility_dir):
        print(f"Error: {partnet_mobility_dir} does not exist")
        return

    # Test mode: process single object
    if args.test:
        print(f"Testing on object: {args.test}")
        object_folder, success, error_msg = process_one_object(
            args.test,
            preprocessed_data_dir,
            partnet_mobility_dir,
            force=args.force
        )

        if success:
            print(f"✓ Success: {object_folder}")
            if error_msg:
                print(f"  Note: {error_msg}")

            # Print result
            output_path = os.path.join(preprocessed_data_dir, object_folder, 'parent_indices.json')
            with open(output_path, 'r') as f:
                data = json.load(f)
                parent_indices = data['parent_indices']
                base_id = extract_base_id(object_folder)
                print(f"\nResult:")
                print(f"  Base ID: {base_id}")
                print(f"  Num parts: {len(parent_indices)}")
                print(f"  Parent indices: {parent_indices}")
        else:
            print(f"✗ Failed: {object_folder}")
            print(f"  Error: {error_msg}")

        return

    # Get all subfolders
    all_folders = [
        item for item in os.listdir(preprocessed_data_dir)
        if os.path.isdir(os.path.join(preprocessed_data_dir, item))
        and item != 'object_part_configs.json'  # Skip config file
    ]

    print(f"Found {len(all_folders)} objects in {preprocessed_data_dir}")

    # Process all objects
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_objects = []

    for object_folder in tqdm(all_folders, desc="Processing objects"):
        object_folder_name, success, error_msg = process_one_object(
            object_folder,
            preprocessed_data_dir,
            partnet_mobility_dir,
            force=args.force
        )

        if success:
            if error_msg and "skipped" in error_msg.lower():
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
            failed_objects.append((object_folder_name, error_msg))

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Total objects: {len(all_folders)}")
    print(f"  ✓ Newly processed: {success_count}")
    print(f"  ⊙ Skipped (already exists): {skip_count}")
    print(f"  ✗ Failed: {fail_count}")

    if failed_objects:
        print("\nFailed objects:")
        for obj, err in failed_objects[:20]:  # Show first 20
            print(f"  {obj}: {err}")
        if len(failed_objects) > 20:
            print(f"  ... and {len(failed_objects) - 20} more")

    print("="*60)

    # Generate summary statistics
    print("\nGenerating statistics...")
    stats = {
        "total_objects": len(all_folders),
        "processed": success_count,
        "skipped": skip_count,
        "failed": fail_count,
        "hierarchy_depth_distribution": {},
        "num_parts_distribution": {},
    }

    # Analyze hierarchy depth and num_parts
    for object_folder in all_folders:
        parent_indices_path = os.path.join(
            preprocessed_data_dir, object_folder, 'parent_indices.json'
        )

        if not os.path.exists(parent_indices_path):
            continue

        with open(parent_indices_path, 'r') as f:
            data = json.load(f)
            parent_indices = data['parent_indices']
            num_parts = len(parent_indices)

            # Count num_parts
            stats['num_parts_distribution'][num_parts] = \
                stats['num_parts_distribution'].get(num_parts, 0) + 1

            # Calculate max hierarchy depth
            def get_depth(idx):
                depth = 0
                current = parent_indices[idx]
                while current >= 0:
                    depth += 1
                    current = parent_indices[current]
                return depth

            max_depth = max(get_depth(i) for i in range(num_parts))
            stats['hierarchy_depth_distribution'][max_depth] = \
                stats['hierarchy_depth_distribution'].get(max_depth, 0) + 1

    print("\nHierarchy Depth Distribution:")
    for depth in sorted(stats['hierarchy_depth_distribution'].keys()):
        count = stats['hierarchy_depth_distribution'][depth]
        print(f"  Depth {depth}: {count} objects")

    print("\nNum Parts Distribution (top 10):")
    sorted_parts = sorted(
        stats['num_parts_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for num_parts, count in sorted_parts:
        print(f"  {num_parts} parts: {count} objects")

    # Save statistics
    stats_path = os.path.join(preprocessed_data_dir, 'hierarchy_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")


if __name__ == '__main__':
    main()
