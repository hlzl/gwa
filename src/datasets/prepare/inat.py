# %%
############### PREPARE iNAT18/19 DATASETS ###############
import os
import shutil
import json
from tqdm import tqdm


def move_val(path, val_file: str = "val2018.json"):
    # Check if val directory already exists
    if os.path.exists(os.path.join(path, "val")):
        raise FileExistsError(
            f"Validation directory '{os.path.join(path, 'val')}' already exists."
        )
    os.makedirs(os.path.join(path, "val"))

    # Read in val dataset info file
    with open(os.path.join(path, val_file), "r") as f:
        val_info = json.load(f)

    if val_file == "val2018.json":
        assert len(val_info["images"]) == 24426

    # Move files
    for image in tqdm(
        val_info["images"], desc="Moving validation files", total=len(val_info["images"])
    ):
        dest_path = os.path.join(path, "val", os.path.join(*image["file_name"].split("/")[1:]))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Ensure destination dir exists
        shutil.move(os.path.join(path, image["file_name"]), dest_path)

    print(f"Moved {len(val_info["images"])} files to {os.path.join(path, 'val')}.")

    # Rename train-val directory to train
    train_val_dir = os.path.join(path, image["file_name"].split("/")[0])
    os.rename(train_val_dir, os.path.join(path, "train"))
    print(f"Renamed '{train_val_dir}' to '{os.path.join(path, 'train')}'")
    print("Done!")


# %%
## iNat18 ##
# Download train and val iNat18 dataset in tmux session with annotations
# wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz
# wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz
# wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz
# NOTE: unpack all .targ.gz files with `tar -xzg file.tar.gz`
# Preparation:
# Go through val_2018.json and move files into newly created val directory
path = "/path/to/iNat18"
move_val(path)
