<p align="center">
  <img src="https://lhmd.top/zpressor/assets/favicon.svg" alt="ZPressor Logo" style="width: 50px; height: 50px; margin-right: 20px;">
  <h1 align="center">ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS</h1>
  <p align="center">
    <a href="https://lhmd.top">Weijie Wang</a>
    ·
    <a href="https://donydchen.github.io">Donny Y. Chen</a>
    ·
    <a href="https://steve-zeyu-zhang.github.io">Zeyu Zhang</a>
    ·
    <a href="https://openreview.net/profile?id=~Duochao_Shi1">Duochao Shi</a>
    ·
    <a href="https://github.com/AkideLiu">Akide Liu</a>
    ·
    <a href="https://bohanzhuang.github.io">Bohan Zhuang</a>
  </p>
  <h3 align="center">NeurIPS 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2505.23734">Paper</a> | <a href="https://lhmd.top/zpressor">Project Page</a> | <a href="https://github.com/ziplab/ZPressor">Code</a> | <a href="https://huggingface.co/lhmd/ZPressor">Models</a> </h3>
  <div align="center"></div>
</p>



<p align="center">
  <a href="">
    <img src="https://lhmd.top/zpressor/assets/teaser.jpg" alt="Logo" width="100%">
  </a>
</p>

<p align="center">
<strong>ZPressor is an architecture-agnostic module that compresses multi-view inputs for scalable feed-forward 3DGS.</strong>
</p>


## News
- **29/09/25 Update:** Check out our <a href="https://github.com/ziplab/VolSplat">VolSplat</a>, a fancy framework for improving multi-view consistency and geometric accuracy for feed-forward 3DGS with voxel-aligned prediction.
- **09/06/25 Update:** Check out our <a href="https://github.com/aim-uofa/PM-Loss">PM-Loss</a>, a novel regularization loss for improving feed-forward 3DGS quality based on a learned point map.

## Installation

Since the `pixelSplat`/`MVSplat`/`DepthSplat `environments are largely consistent, we will provide an environment capable of running all three codebases simultaneously:

```bash
conda create -n zpressor python=3.10
conda activate zpressor
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Then install `ZPressor` as a package:

```bash
cd zpressor
pip install -e . # install the zpressor package
cd ..
```

## Model Zoo

Our pre-trained models are hosted on [Hugging Face](https://huggingface.co/lhmd/ZPressor). Please download the required models to the `./[Baseline Folder]/pretrained/ `directory.

| Model                                  | Codebase            | Training Dataset    | Download                                                     |
| -------------------------------------- | ------------------- | ------------------- | ------------------------------------------------------------ |
| depthsplat-dl3dv-baseline-n50-256x448  | DepthSplat          | RealEstate10K+DL3DV | [download](https://huggingface.co/lhmd/ZPressor/resolve/main/depthsplat-dl3dv-baseline-n50-256x448.ckpt) |
| depthsplat-dl3dv-zpressor-n50-256x448  | DepthSplat+ZPressor | RealEstate10K+DL3DV | [download](https://huggingface.co/lhmd/ZPressor/resolve/main/depthsplat-dl3dv-zpressor-n50-256x448.ckpt) |
| mvsplat-re10k-baseline-n200-256x256    | MVSplat             | RealEstate10K       | [download](https://huggingface.co/lhmd/ZPressor/resolve/main/mvsplat-re10k-baseline-n200-256x256.ckpt) |
| mvsplat-re10k-zpressor-n200-256x256    | MVSplat+ZPressor    | RealEstate10K       | [download](https://huggingface.co/lhmd/ZPressor/resolve/main/mvsplat-re10k-zpressor-n200-256x256.ckpt) |
| pixelsplat-re10k-baseline-n200-256x256 | pixelSplat          | RealEstate10K       | [download](https://huggingface.co/lhmd/ZPressor/resolve/main/pixelsplat-re10k-baseline-n200-256x256.ckpt) |
| pixelsplat-re10k-zpressor-n200-256x256 | pixelSplat+ZPressor | RealEstate10K       | [download](https://huggingface.co/lhmd/ZPressor/resolve/main/pixelsplat-re10k-zpressor-n200-256x256.ckpt) |

## Datasets

### DL3DV-10K

First, download the `DL3DV-10K` dataset according to [the official script](https://github.com/DL3DV-10K/Dataset/blob/main/scripts/download.py), you can use [this script](https://github.com/DL3DV-10K/Dataset/blob/main/scripts/count.sh) to verify data integrity.

Then, we enter the `depthsplat `folder to process the dataset. We made modifications to the `DepthSplat`’s script for processing `DL3DV-10K`.

```bash
cd depthsplat
python src/scripts/convert_dl3dv_test.py --input_dir [ori_benchmark_path] --output_dir [benchmark_path]
python generate_dl3dv_index.py --dataset_path [benchmark_path] --bench --stage test
# You may modify the path of test stage index.json in convert_dl3dv_train.py
python src/scripts/convert_dl3dv_train.py \
    --input_base_dir [ori_dataset_path] \ # such as datasets/DL3DV-10K-480
    --output_base_dir [dataset_path] \ # such as datasets/DL3DV-10K-480P
    --start_k 1 \
    --end_k 11 \
    --img_subdir images_8 # for 480P
python src/scripts/generate_dl3dv_index.py \
    --dataset_path [dataset_path] \
    --start_k 1 \
    --end_k 11
```

### RealEstate10K / ACID

Please refer to [here](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) for acquiring preprocessed versions of the datasets following [pixelSplat](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets). If the link is broken or inaccessible, feel free to contact wangweijie@zju.edu.cn. 

### Some Notes

Expected folder structure of datasets:

```
├── datasets
│   ├── re10k
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── dl3dv
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
```

You can use a symbolic link to point the datasets folder to the correct location when running specific codebases, for example:

```bash
ln -s ./datasets ./depthsplat/
ln -s ./datasets ./mvsplat/
ln -s ./datasets ./pixelsplat/
```

## Running the Code

Each codebase operates differently; detailed instructions are provided in the README files within each code folder ([DepthSplat](./depthsplat/README.md) / [MVSplat](./mvsplat/README.md) / [pixelSplat](./pixelsplat/README.md)).

## Citation
If you find our work useful for your research, please consider citing us:

```bibtex
@article{wang2025zpressor,
  title={ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS},
  author={Wang, Weijie and Chen, Donny Y. and Zhang, Zeyu and Shi, Duochao and Liu, Akide and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2505.23734},
  year={2025}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at wangweijie@zju.edu.cn.

## Acknowledgements

This project is developed with several fantastic repos: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat) and [DepthSplat](https://github.com/cvg/depthsplat). We thank the original authors for their excellent work.
