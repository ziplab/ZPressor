import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict
from PIL import Image

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import argparse
import json
import os

from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--input_base_dir", type=str, help="base directory containing 1K, 2K, ..., 11K subdirectories")
parser.add_argument("--output_base_dir", type=str, help="base output directory for processed datasets")
parser.add_argument(
    "--img_subdir",
    type=str,
    default="images_8",
    help="image directory name",
    choices=[
        "images_4",
        "images_8",
    ],
)
parser.add_argument("--n_test", type=int, default=10, help="test skip")
parser.add_argument("--which_stage", type=str, default=None, help="dataset directory")
parser.add_argument("--detect_overlap", action="store_true")
parser.add_argument("--start_k", type=int, default=1, help="starting K value (default: 1)")
parser.add_argument("--end_k", type=int, default=11, help="ending K value (default: 11)")

args = parser.parse_args()


# Target 200 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(2e8)


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {
        int(path.stem.split("_")[-1]): load_raw(path)
        for path in example_path.iterdir()
        if path.suffix.lower() not in [".npz"]
    }


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(example_path: Path) -> Metadata:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    url = str(example_path).split("/")[-3]
    with open(example_path, "r") as f:
        meta_data = json.load(f)

    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    saved_fx = float(fx) / float(store_w)
    saved_fy = float(fy) / float(store_h)
    saved_cx = float(cx) / float(store_w)
    saved_cy = float(cy) / float(store_h)

    timestamps = []
    cameras = []
    opencv_c2ws = []  # will be used to calculate camera distance

    for frame in meta_data["frames"]:
        timestamps.append(
            int(os.path.basename(frame["file_path"]).split(".")[0].split("_")[-1])
        )
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        # transform_matrix is in blender c2w, while we need to store opencv w2c matrix here
        opencv_c2w = np.array(frame["transform_matrix"]) @ blender2opencv
        opencv_c2ws.append(opencv_c2w)
        camera.extend(np.linalg.inv(opencv_c2w)[:3].flatten().tolist())
        cameras.append(np.array(camera))

    # timestamp should be the one that match the above images keys, use for indexing
    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {"url": url, "timestamps": timestamps, "cameras": cameras}


def partition_train_test_splits(root_dir, n_test=10):
    sub_folders = sorted(glob(os.path.join(root_dir, "*/")))
    test_list = sub_folders[::n_test]
    train_list = [x for x in sub_folders if x not in test_list]
    out_dict = {"train": train_list, "test": test_list}
    return out_dict


def is_image_shape_matched(image_dir, target_shape):
    image_path = sorted(glob(str(image_dir / "*")))
    if len(image_path) == 0:
        return False

    image_path = image_path[0]
    try:
        im = Image.open(image_path)
    except:
        return False
    w, h = im.size
    if (h, w) == target_shape:
        return True
    else:
        return False


def legal_check_for_all_scenes(root_dir, target_shape):
    valid_folders = []
    sub_folders = sorted(glob(os.path.join(root_dir, "*/*")))
    for sub_folder in tqdm(sub_folders, desc="checking scenes..."):
        img_dir = os.path.join(sub_folder, 'images_8')
        # img_dir = os.path.join(sub_folder, "images_4")
        if not is_image_shape_matched(Path(img_dir), target_shape):
            print(f"image shape does not match for {sub_folder}")
            continue
        pose_file = os.path.join(sub_folder, "transforms.json")
        if not os.path.isfile(pose_file):
            print(f"cannot find pose file for {sub_folder}")
            continue

        valid_folders.append(sub_folder)

    return valid_folders


def process_single_directory(input_dir: Path, output_dir: Path):
    """Process a single K directory"""
    print(f"\n=== Processing {input_dir.name} ===")

    INPUT_DIR = input_dir
    OUTPUT_DIR = output_dir

    if "images_8" in args.img_subdir:
        target_shape = (270, 480)  # (h, w)
    elif "images_4" in args.img_subdir:
        target_shape = (540, 960)
    else:
        raise ValueError

    print("checking all scenes...")
    valid_scenes = legal_check_for_all_scenes(INPUT_DIR, target_shape)
    print("valid scenes:", len(valid_scenes))

    # test scenes
    test_scenes = "/home/lhmd/Project/depthsplat/datasets/DL3DV-10K/dl3dv_benchmark/test/index.json"
    with open(test_scenes, "r") as f:
        overlap_scenes = json.load(f)

    assert len(overlap_scenes) == 140, "test scenes should contain 140 scenes"

    for stage in ["train"]:

        error_logs = []
        image_dirs = valid_scenes

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            nonlocal chunk_size, chunk_index, chunk

            chunk_key = f"{chunk_index:0>6}"
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for image_dir in tqdm(image_dirs, desc=f"Processing {stage}"):
            key = os.path.basename(image_dir.strip("/"))
            # skip test scenes
            if key in overlap_scenes:
                print(f"scene {key} in benchmark, skip.")
                continue

            image_dir = Path(image_dir) / "images_8"  # 270x480
            # image_dir = Path(image_dir) / 'images_4'  # 540x960

            num_bytes = get_size(image_dir)

            # Read images and metadata.
            try:
                images = load_images(image_dir)
            except:
                print("image loading error")
                continue
            meta_path = image_dir.parent / "transforms.json"
            if not meta_path.is_file():
                error_msg = f"---------> [ERROR] no meta file in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue
            example = load_metadata(meta_path)

            # Merge the images into the example.
            try:
                example["images"] = [
                    images[timestamp.item()] for timestamp in example["timestamps"]
                ]
            except:
                error_msg = f"---------> [ERROR] Some images missing in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue

            # Add the key to the example.
            example["key"] = "dl3dv_" + key

            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()


if __name__ == "__main__":
    base_input_dir = Path(args.input_base_dir)
    base_output_dir = Path(args.output_base_dir)

    # Process all directories from start_k to end_k
    total_dirs = args.end_k - args.start_k + 1
    processed_dirs = 0

    for k in range(args.start_k, args.end_k + 1):
        k_dir = f"{k}K"
        input_dir = base_input_dir / k_dir
        output_dir = base_output_dir / k_dir

        if not input_dir.exists():
            print(f"Warning: Input directory {input_dir} does not exist, skipping...")
            continue

        print(f"\n{'='*50}")
        print(f"Processing directory {k_dir} ({processed_dirs + 1}/{total_dirs})")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"{'='*50}")

        # Process this directory
        process_single_directory(input_dir, output_dir)

        processed_dirs += 1
        print(f"\nCompleted {k_dir} ({processed_dirs}/{total_dirs})")

    print(f"\n{'='*50}")
    print(f"All processing complete! Processed {processed_dirs}/{total_dirs} directories.")
    print(f"{'='*50}")
