# pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction

## Test

You can use [test_re10k.py](./test_re10k.py) and [test_acid.py](./test_acid.py) to test models provided by us or models you have trained yourself. We also provide a pre-trained Baseline model for comparison, please use [test_re10k_base.py](./test_re10k_base.py) and [test_acid_base.py](./test_acid_base.py) to evaluate it. You can freely modify the parameters in the script to accommodate any test conditions you desire.

```python
python test_re10k.py
python test_acid.py
python test_re10k_base.py
python test_acid_base.py
```

You can use the scripts in the `DepthSplat` folder to generate more test files:

```python
cd ../depthsplat
python -m src.scripts.generate_evaluation_index --data_dir=datasets/re10k/test --num_context_views=36 --num_target_views=8 --view_selection_num=200 --type re10k
python -m src.scripts.generate_evaluation_index --data_dir=datasets/acid/test --num_context_views=36 --num_target_views=8 --view_selection_num=50 --type acid
```

## Train

```bash
python -m src.main +experiment=re10k_12_4_view
```
