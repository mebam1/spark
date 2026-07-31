#!/usr/bin/env python3
"""
Script to create a test set by extracting 5% of preprocessed data.
Samples objects evenly (e.g., 1st, 20th, 40th, 60th, etc. for 5% sampling)
and moves them to preprocessed_data_test directory.

When using --manual flag, uses SELECT list to manually choose specific folders.
"""

import json
import shutil
import argparse
from pathlib import Path

# Manual selection list - add folder names here to manually select test data
SELECT=[
"100162","100202",
"100448","100473",
"101362","102834",
"103048","103105",
"12092","12540",
"8867","8897",
"100021","100051",
"10090","10101",
"7221","7236",
"101773","101931",
"103863","103872",
"10627","10900",
"100828","101011",
"101593","102301",
"35059","44817",
"100550","100825",
"20411","25913",
"103477","103482",
"101323","102632",
"101377","102155",
"103518","103521"
]

def create_test_set(preprocessed_dir, test_dir, config_file, test_ratio=0.05, manual_mode=False):
    """
    Create test set by sampling data at regular intervals or manual selection.

    Args:
        preprocessed_dir (str): Path to preprocessed_data directory
        test_dir (str): Path to preprocessed_data_test directory
        config_file (str): Path to object_part_configs.json
        test_ratio (float): Ratio of data to use for test set (default: 0.05 for 5%)
        manual_mode (bool): If True, use SELECT list for manual selection (default: False)
    """
    preprocessed_dir = Path(preprocessed_dir)
    test_dir = Path(test_dir)
    config_file = Path(config_file)

    # Create test directory if it doesn't exist
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created test directory: {test_dir}")

    # Load the config file
    print(f"\nLoading config file: {config_file}")
    with open(config_file, 'r') as f:
        configs = json.load(f)

    total_objects = len(configs)
    print(f"Total objects in config: {total_objects}")

    if manual_mode:
        # Manual selection mode - use SELECT list
        print(f"\n*** MANUAL MODE: Using SELECT list ***")
        print(f"SELECT list contains {len(SELECT)} folders:")
        for folder in SELECT:
            print(f"  - {folder}")

        # Find indices of objects in SELECT list
        test_indices = []
        for idx, config in enumerate(configs):
            if config['file'] in SELECT:
                test_indices.append(idx)

        num_test_objects = len(test_indices)
        print(f"\nFound {num_test_objects} objects from SELECT list in config")

        # Check for missing folders
        selected_folders_in_config = [configs[idx]['file'] for idx in test_indices]
        missing_folders = set(SELECT) - set(selected_folders_in_config)
        if missing_folders:
            print(f"WARNING: {len(missing_folders)} folders from SELECT not found in config:")
            for folder in missing_folders:
                print(f"  - {folder}")
    else:
        # Automatic sampling mode
        # Calculate sampling interval
        # For 5%, we sample every 20th object (1/0.05 = 20)
        sampling_interval = int(1 / test_ratio)
        print(f"Sampling interval: every {sampling_interval}th object")

        # Select indices to move to test set
        # Start from index 0, then sampling_interval-1, 2*sampling_interval-1, etc.
        test_indices = list(range(0, total_objects, sampling_interval))
        num_test_objects = len(test_indices)

        print(f"Number of objects to move to test set: {num_test_objects}")
        print(f"Actual test ratio: {num_test_objects/total_objects*100:.2f}%")

    # Separate configs into training and test
    train_configs = []
    test_configs = []
    moved_objects = []
    failed_objects = []

    for idx, config in enumerate(configs):
        if idx in test_indices:
            test_configs.append(config)
            object_id = config['file']

            # Move the object folder
            source_path = preprocessed_dir / object_id
            dest_path = test_dir / object_id

            if source_path.exists():
                try:
                    shutil.move(str(source_path), str(dest_path))
                    moved_objects.append(object_id)
                    print(f"Moved: {object_id} (index {idx})")
                except Exception as e:
                    print(f"Error moving {object_id}: {e}")
                    failed_objects.append(object_id)
                    # If move failed, keep in training set
                    train_configs.append(config)
                    test_configs.pop()
            else:
                print(f"Warning: {object_id} folder not found, skipping")
                failed_objects.append(object_id)
                # If folder doesn't exist, keep in training set
                train_configs.append(config)
                test_configs.pop()
        else:
            train_configs.append(config)

    # Save updated training config
    print(f"\nSaving updated training config...")
    with open(config_file, 'w') as f:
        json.dump(train_configs, f, indent=4)
    print(f"Training config saved with {len(train_configs)} objects")

    # Save test config
    test_config_file = test_dir / 'object_part_configs.json'
    print(f"\nSaving test config...")
    with open(test_config_file, 'w') as f:
        json.dump(test_configs, f, indent=4)
    print(f"Test config saved to {test_config_file} with {len(test_configs)} objects")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total objects processed: {total_objects}")
    print(f"Objects moved to test set: {len(moved_objects)}")
    print(f"Objects remaining in training set: {len(train_configs)}")
    print(f"Failed/missing objects: {len(failed_objects)}")

    if failed_objects:
        print(f"\nFailed/missing objects:")
        for obj in failed_objects:
            print(f"  - {obj}")

    print(f"\nTest set location: {test_dir}")
    print(f"Test config location: {test_config_file}")
    print(f"Training config updated at: {config_file}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create test set by sampling 5% of preprocessed data at regular intervals'
    )
    parser.add_argument('--preprocessed-dir', type=str,
                        default='preprocessed_data',
                        help='Path to preprocessed_data directory (default: preprocessed_data)')
    parser.add_argument('--test-dir', type=str,
                        default='preprocessed_data_test',
                        help='Path to test data directory (default: preprocessed_data_test)')
    parser.add_argument('--config', type=str,
                        default='preprocessed_data/object_part_configs.json',
                        help='Path to object_part_configs.json (default: preprocessed_data/object_part_configs.json)')
    parser.add_argument('--test-ratio', type=float,
                        default=0.05,
                        help='Ratio of data for test set (default: 0.05 for 5%%)')
    parser.add_argument('--manual', action='store_true',
                        help='Use manual selection mode with SELECT list instead of automatic sampling')

    args = parser.parse_args()

    print("Creating test set...")
    print(f"Preprocessed directory: {args.preprocessed_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Config file: {args.config}")
    if args.manual:
        print(f"Mode: Manual selection (using SELECT list)")
    else:
        print(f"Mode: Automatic sampling")
        print(f"Test ratio: {args.test_ratio*100}%")

    # Confirm before proceeding
    response = input("\nThis will move object folders from preprocessed_data to preprocessed_data_test. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Operation cancelled.")
        exit(0)

    create_test_set(
        args.preprocessed_dir,
        args.test_dir,
        args.config,
        args.test_ratio,
        manual_mode=args.manual
    )

    print("\nDone!")
