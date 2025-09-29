import subprocess

# List of context_num values
context_nums = [12]  # Add your desired context_num values here
cuda_idx = [0]
cluster_num = [4]

# Base command with placeholders for context_num

base_command = """
CUDA_VISIBLE_DEVICES=[cuda_idx] python -m src.main +experiment=re10k_12_4_view \
checkpointing.pretrained_model=pretrained/pixelsplat-re10k-zpressor-n200-256x256.ckpt \
mode=test \
wandb.project=pixelsplat \
wandb.mode=disabled \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=[context_num] \
dataset.view_sampler.index_path=assets/re10k_evaluation/re10k_ctx_[context_num]v_tgt_8v_n200.json \
model.encoder.use_cluster=true \
model.encoder.cluster_num=[cluster_num] \
test.output_path=outputs/test/re10k-zpressor-[context_num]-[cluster_num]-8-n200
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