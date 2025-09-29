import subprocess

# List of context_num values
context_nums = [36]  # Add your desired context_num values here
cuda_idx = [0]
cluster_num = [6]

# Base command with placeholders for context_num
base_command = """
CUDA_VISIBLE_DEVICES=[cuda_idx] python -m src.main +experiment=re10k \
checkpointing.load=pretrained/mvsplat-re10k-baseline-n200-256x256.ckpt \
mode=test \
wandb.project=mvsplat \
wandb.mode=disabled \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/re10k_evaluation/re10k_ctx_[context_num]v_tgt_8v_n200.json \
test.compute_scores=true \
test.save_image=false \
test.save_gt_image=false \
test.save_input_images=false \
test.save_video=false \
test.stablize_camera=true \
test.dec_chunk_size=8 \
model.encoder.use_cluster=false \
model.encoder.cluster_num=[cluster_num] \
test.output_path=outputs/test/re10k-base-[context_num]-[cluster_num]-8-n200
"""

# Loop through each context_num and run the command
for i in range(len(context_nums)):
    # Replace the [context_num] placeholder with the actual value
    command = base_command.replace("[context_num]", str(context_nums[i]))
    command = command.replace("[cuda_idx]", str(cuda_idx[i]))
    command = command.replace("[cluster_num]", str(cluster_num[i]))

    # Run the command
    subprocess.Popen(command, shell=True)
    print(f"Started process for context_num = {context_nums[i]}")