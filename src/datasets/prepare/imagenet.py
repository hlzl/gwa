# %%
############### CREATE IMAGENET DATASETS ###############
import os
import requests
import tarfile


def download_unpack(url, save_dir):
    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_dir, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract the tar.gz file
    with tarfile.open(save_dir, "r:gz") as tar:
        tar.extractall(path=os.path.join(save_dir, url.split("/")[-1].split(".")[0]))


def label_dirs(save_dir, label_dir):
    # Get labels from ILSVRC directory
    name_list = sorted(set([label for label in os.listdir(label_dir)]))
    assert len(name_list) == 1000

    # Get a list of subdirectories with integer names
    subdirs = list(set([d for d in os.listdir(save_dir) if d.isdigit()]))
    subdirs.sort(key=int)

    # Ensure the name list matches the number of directories
    if len(subdirs) != len(name_list):
        raise ValueError("The number of directories and strings in the list must match.")

    # Rename the directories
    for old_name, new_name in zip(subdirs, name_list):
        old_path = os.path.join(save_dir, old_name)
        new_path = os.path.join(save_dir, new_name)
        os.rename(old_path, new_path)


# %%
# Download and unpack ImageNetV2 testset
url = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
save_dir = "/path/to/ImageNetV2"

download_unpack(url, save_dir)

# %%
# Rename directories such that they have proper label
save_dir = "/path/to/ImageNetV2/imagenetv2-matched-frequency"
label_dir = "/path/to/ILSVRC2012/train"  # NOTE: typically directories within .../ILSVRC2012/train
label_dirs(save_dir, label_dir)


# %%
# ImageNet-ReaL labels
# Simply download: https://github.com/google-research/reassessed-imagenet.git
# and adapt path in test.py
