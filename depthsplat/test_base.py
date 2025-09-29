import subprocess

# List of context_num values
context_nums = [36]  # Add your desired context_num values here
cuda_idx = [0]
cluster_num = [6]
gap_num = [50]

# Base command with placeholders for context_num
base_command = """
CUDA_VISIBLE_DEVICES=[cuda_idx] HF_ENDPOINT=https://hf-mirror.com python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=1 \
mode=test \
'dataset.roots'='["datasets/dl3dv"]' \
'dataset.image_shape'='[256,448]' \
dataset.near=1. \
dataset.far=200. \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=[context_num] \
dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_ctx_[context_num]v_tgt_8v_n50.json \
model.encoder.upsample_factor=8 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
model.encoder.cluster_num=[cluster_num] \
model.encoder.use_cluster=false \
model.encoder.num_zpressor_layers=6 \
model.encoder.apply_cluster_steps=0 \
model.encoder.use_clstoken=false \
model.encoder.return_depth=true \
wandb.mode=disabled \
checkpointing.pretrained_model=pretrained/depthsplat-dl3dv-baseline-n50-256x448.ckpt \
checkpointing.no_strict_load=false \
test.save_image=false \
test.save_gaussian=false \
test.save_depth=false \
test.save_video=true \
test.compute_scores=true \
test.stablize_camera=false \
wandb.project=depthsplat \
output_dir=checkpoints/test-dl3dv-[context_num]-[cluster_num]-8-n50
"""

# Loop through each context_num and run the command
for i in range(len(context_nums)):
    command = base_command.replace("[context_num]", str(context_nums[i]))
    command = command.replace("[cuda_idx]", str(cuda_idx[i]))
    command = command.replace("[cluster_num]", str(cluster_num[i]))

    # Run the command
    subprocess.Popen(command, shell=True)
    print(f"Started process for context_num = {context_nums[i]}")