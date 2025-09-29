import json
from pathlib import Path

import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='Base dataset directory path', required=True)
parser.add_argument('--subset', choices=['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], help='Specific subset to process (if not specified, processes all subsets)', default=None)
parser.add_argument('--start_k', type=int, default=1, help='Starting K value (default: 1)')
parser.add_argument('--end_k', type=int, default=11, help='Ending K value (default: 11)')

args = parser.parse_args()


def process_single_subset(dataset_base_path: Path, subset: str):
    """Process a single subset directory"""
    print(f"\n=== Processing subset {subset} ===")

    DATASET_PATH = dataset_base_path / subset
    print(f"Processing: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        print(f"Warning: Dataset path {DATASET_PATH} does not exist, skipping...")
        return False

    # "train" or "test"
    for stage in ["train"]:
        stage_path = DATASET_PATH / stage

        if not stage_path.exists():
            print(f"Warning: Stage path {stage_path} does not exist, skipping...")
            continue

        index = {}
        torch_files = [f for f in stage_path.iterdir() if f.suffix == ".torch"]

        if not torch_files:
            print(f"Warning: No .torch files found in {stage_path}")
            continue

        for chunk_path in tqdm(
            sorted(torch_files), desc=f"Indexing {stage} for {subset}"
        ):
            chunk = torch.load(chunk_path)
            for example in chunk:
                index[example["key"]] = str(chunk_path.relative_to(stage_path))

        output_file = stage_path / "index.json"
        with output_file.open("w") as f:
            json.dump(index, f)

        print(f"Created index for {subset}/{stage}: {len(index)} entries -> {output_file}")

    return True

if __name__ == "__main__":
    dataset_base_path = Path(args.dataset_path)

    if not dataset_base_path.exists():
        print(f"Error: Base dataset path {dataset_base_path} does not exist!")
        exit(1)

    if args.subset:
        # Process single specified subset
        print(f"Processing single subset: {args.subset}")
        success = process_single_subset(dataset_base_path, args.subset)
        if success:
            print(f"\nCompleted processing subset {args.subset}")
        else:
            print(f"\nFailed to process subset {args.subset}")
    else:
        # Process all subsets from start_k to end_k
        print(f"Processing all subsets from {args.start_k}K to {args.end_k}K")

        total_subsets = args.end_k - args.start_k + 1
        processed_subsets = 0
        successful_subsets = 0

        for k in range(args.start_k, args.end_k + 1):
            subset = f"{k}K"

            print(f"\n{'='*50}")
            print(f"Processing subset {subset} ({processed_subsets + 1}/{total_subsets})")
            print(f"{'='*50}")

            success = process_single_subset(dataset_base_path, subset)
            processed_subsets += 1

            if success:
                successful_subsets += 1
                print(f"✓ Completed subset {subset} ({processed_subsets}/{total_subsets})")
            else:
                print(f"✗ Failed to process subset {subset} ({processed_subsets}/{total_subsets})")

        print(f"\n{'='*50}")
        print(f"Index generation complete!")
        print(f"Successfully processed: {successful_subsets}/{total_subsets} subsets")
        print(f"{'='*50}")
