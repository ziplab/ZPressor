import os


# Function to extract the step number from the filename
def extract_step(file_name):
    step_str = file_name.split("-")[1].split("_")[1].replace(".ckpt", "")
    return int(step_str)


def find_latest_ckpt(ckpt_dir):
    # List all files in the directory that end with .ckpt
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    # Check if there are any .ckpt files in the directory
    if not ckpt_files:
        raise ValueError(f"No .ckpt files found in {ckpt_dir}.")
    else:
        # Find the file with the maximum step
        latest_ckpt_file = max(ckpt_files, key=extract_step)

        return ckpt_dir / latest_ckpt_file
