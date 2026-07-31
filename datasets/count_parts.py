#!/usr/bin/env python3
"""
Script to analyze PartNet-Mobility dataset structure.
Loops through subfolders, reads meta.json and semantics.txt,
and generates statistics per category.

Usage:
    python count_parts.py --input mesh/partnet-mobility --output part_statistics.txt
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


def parse_semantics(semantics_path: Path) -> List[str]:
    """
    Parse semantics.txt to extract part names.

    Args:
        semantics_path: Path to semantics.txt file

    Returns:
        List of part names (one per link)
    """
    part_names = []

    with open(semantics_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: link_X <joint_type> <part_name>
            parts = line.split()
            if len(parts) >= 3:
                part_name = parts[2]  # Third element is the part name
                part_names.append(part_name)

    return part_names


def analyze_dataset(input_dir: Path) -> Dict[str, Dict]:
    """
    Analyze all subfolders in the input directory.

    Args:
        input_dir: Root directory containing subfolders with meta.json and semantics.txt

    Returns:
        Dictionary mapping category to statistics
    """
    # Dictionary to store statistics per category
    category_stats = defaultdict(lambda: {
        'num_parts_list': [],      # List of part counts for this category
        'part_names': set(),       # Set of unique part names
        'object_ids': []           # List of object IDs in this category
    })

    # Find all subfolders
    subfolders = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"Found {len(subfolders)} subfolders in {input_dir}")

    processed = 0
    skipped = 0

    for subfolder in sorted(subfolders):
        meta_path = subfolder / "meta.json"
        semantics_path = subfolder / "semantics.txt"

        # Check if both files exist
        if not meta_path.exists() or not semantics_path.exists():
            skipped += 1
            continue

        try:
            # Read meta.json to get category
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            category = meta.get('model_cat', 'Unknown')

            # Parse semantics.txt to get part names
            part_names = parse_semantics(semantics_path)
            num_parts = len(part_names)

            # Update category statistics
            category_stats[category]['num_parts_list'].append(num_parts)
            category_stats[category]['part_names'].update(part_names)
            category_stats[category]['object_ids'].append(subfolder.name)

            processed += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {subfolder.name}: {e}")
            skipped += 1
            continue

    print(f"\nProcessed: {processed} objects")
    print(f"Skipped: {skipped} objects")

    return dict(category_stats)


def write_statistics(category_stats: Dict[str, Dict], output_path: Path):
    """
    Write statistics to a text file.

    Args:
        category_stats: Dictionary of statistics per category
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PartNet-Mobility Dataset Statistics\n")
        f.write("="*80 + "\n\n")

        # Overall statistics
        total_objects = sum(len(stats['object_ids']) for stats in category_stats.values())
        total_categories = len(category_stats)

        f.write(f"Total Categories: {total_categories}\n")
        f.write(f"Total Objects: {total_objects}\n\n")

        # Sort categories by name
        sorted_categories = sorted(category_stats.items(), key=lambda x: x[0])

        f.write("="*80 + "\n")
        f.write("Per-Category Statistics\n")
        f.write("="*80 + "\n\n")

        for category, stats in sorted_categories:
            num_parts_list = stats['num_parts_list']
            part_names = sorted(stats['part_names'])
            num_objects = len(stats['object_ids'])

            f.write(f"Category: {category}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Number of Objects: {num_objects}\n")

            # Count objects within part thresholds
            if num_parts_list:
                num_objects_le_8 = sum(1 for n in num_parts_list if n <= 8)
                num_objects_le_16 = sum(1 for n in num_parts_list if n <= 16)
                f.write(f"  Number of Objects within 8 parts: {num_objects_le_8}\n")
                f.write(f"  Number of Objects within 16 parts: {num_objects_le_16}\n")

            if num_parts_list:
                min_parts = min(num_parts_list)
                max_parts = max(num_parts_list)
                avg_parts = sum(num_parts_list) / len(num_parts_list)

                f.write(f"  Min Parts: {min_parts}\n")
                f.write(f"  Average Parts: {avg_parts:.2f}\n")
                f.write(f"  Max Parts: {max_parts}\n")

            f.write(f"  Unique Part Names ({len(part_names)}):\n")
            f.write(f"    {part_names}\n")
            f.write("\n")

        # Summary table
        f.write("="*100 + "\n")
        f.write("Summary Table\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Category':<20} {'Objects':>8} {'<=8':>6} {'<=16':>6} {'Min':>6} {'Avg':>8} {'Max':>6} {'Unique Parts':>15}\n")
        f.write("-" * 100 + "\n")

        for category, stats in sorted_categories:
            num_parts_list = stats['num_parts_list']
            num_objects = len(stats['object_ids'])
            num_unique_parts = len(stats['part_names'])

            if num_parts_list:
                num_objects_le_8 = sum(1 for n in num_parts_list if n <= 8)
                num_objects_le_16 = sum(1 for n in num_parts_list if n <= 16)
                min_parts = min(num_parts_list)
                max_parts = max(num_parts_list)
                avg_parts = sum(num_parts_list) / len(num_parts_list)

                f.write(f"{category:<20} {num_objects:>8} {num_objects_le_8:>6} {num_objects_le_16:>6} {min_parts:>6} {avg_parts:>8.2f} {max_parts:>6} {num_unique_parts:>15}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"\nStatistics written to: {output_path}")


def print_statistics(category_stats: Dict[str, Dict]):
    """
    Print statistics to console.

    Args:
        category_stats: Dictionary of statistics per category
    """
    print("\n" + "="*80)
    print("PartNet-Mobility Dataset Statistics")
    print("="*80 + "\n")

    # Overall statistics
    total_objects = sum(len(stats['object_ids']) for stats in category_stats.values())
    total_categories = len(category_stats)

    print(f"Total Categories: {total_categories}")
    print(f"Total Objects: {total_objects}\n")

    # Sort categories by name
    sorted_categories = sorted(category_stats.items(), key=lambda x: x[0])

    print("="*100)
    print("Summary Table")
    print("="*100)
    print(f"{'Category':<20} {'Objects':>8} {'<=8':>6} {'<=16':>6} {'Min':>6} {'Avg':>8} {'Max':>6} {'Unique Parts':>15}")
    print("-" * 100)

    for category, stats in sorted_categories:
        num_parts_list = stats['num_parts_list']
        num_objects = len(stats['object_ids'])
        num_unique_parts = len(stats['part_names'])

        if num_parts_list:
            num_objects_le_8 = sum(1 for n in num_parts_list if n <= 8)
            num_objects_le_16 = sum(1 for n in num_parts_list if n <= 16)
            min_parts = min(num_parts_list)
            max_parts = max(num_parts_list)
            avg_parts = sum(num_parts_list) / len(num_parts_list)

            print(f"{category:<20} {num_objects:>8} {num_objects_le_8:>6} {num_objects_le_16:>6} {min_parts:>6} {avg_parts:>8.2f} {max_parts:>6} {num_unique_parts:>15}")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze PartNet-Mobility dataset structure and generate statistics per category'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input folder containing subfolders with meta.json and semantics.txt (e.g., mesh/partnet-mobility)')
    parser.add_argument('--output', type=str, default='part_statistics.txt',
                       help='Output text file for statistics (default: part_statistics.txt)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return

    if not input_dir.is_dir():
        print(f"[ERROR] Input path is not a directory: {input_dir}")
        return

    print(f"Analyzing dataset in: {input_dir}")
    print(f"Output will be written to: {output_path}\n")

    # Analyze dataset
    category_stats = analyze_dataset(input_dir)

    # Print to console
    print_statistics(category_stats)

    # Write to file
    write_statistics(category_stats, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
