# DepthSplat: Connecting Gaussian Splatting and Depth

## Test

You can use [test.py](./test.py) to test models provided by us or models you have trained yourself. We also provide a pre-trained Baseline model for comparison, please use [test_base.py](./test_base.py) to evaluate it. You can freely modify the parameters in the script to accommodate any test conditions you desire.

```python
python test.py
python test_base.py
```

To generate additional test files in a standardized manner, we provide a unified test metadata generation script. Please use it as follows:

```python
python -m src.scripts.generate_evaluation_index --data_dir=datasets/dl3dv/test --num_context_views=36 --num_target_views=8 --view_selection_num=50 --type dl3dv
python -m src.scripts.generate_evaluation_index --data_dir=datasets/re10k/test --num_context_views=36 --num_target_views=8 --view_selection_num=200 --type re10k
```

## Train

Before training, you need to download the pre-trained [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [UniMatch](https://github.com/autonomousvision/unimatch) weights and set up your [wandb account](https://github.com/lhmd/depthSplat/blob/rebuttal/config/main.yaml) for logging.

```bash
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth -P pretrained
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained
```

Once your dataset and pretrained model are ready, you can use the commands in [train.sh](./train.sh) to train `DepthSplat+ZPressor`.

