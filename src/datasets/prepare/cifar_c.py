# %%
############### CREATE CIFAR-10-C SEVERITY SUBSETS ###############
import os
import glob
import shutil
import numpy as np


def split_numpy_files(dir, subdirs):
    for version in subdirs:
        for i in range(1, 6):
            directory = os.path.join(dir, version, str(i))
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Directory '{directory}' deleted successfully.")
            os.makedirs(directory)

        # Iterate through all files within dataset version and split files
        for file in glob.glob(os.path.join(dir, version, "*.npy")):
            images = np.load(file)
            assert images.shape[0] == 50000 and images.shape[-1] == 3
            for i in range(1, 6):
                np.save(
                    os.path.join(dir, version, str(i), file.split("/")[-1]),
                    images[(i - 1) * 10000 : i * 10000],
                )
        os.remove(file)
    print(f">> Done splitting .npy files in {dir}!")


def split_labels_file(path):
    images = np.load(path)
    assert images.size == 50000
    for i in range(1, 6):
        np.save(
            os.path.join("/", *path.split("/")[:-1], f"labels_{i}.npy"),
            images[(i - 1) * 10000 : i * 10000],
        )
    print(f">> Done splitting .npy labels file {path}!")


# %%
# NOTE: First get the CIFAR-10-C versions from https://zenodo.org/records/2535967
# `wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar`
# `tar -xvf CIFAR-10-C.tar`
# Then create directories for blur, digital, noise, and weather and put files in
# there according to https://zenodo.org/records/2235448.
# NOTE: In CIFAR-10-C, the first 10,000 images in each .npy are the test set images corrupted at severity 1,
# and the last 10,000 images are the test set images corrupted at severity 5.
# > labels.npy is the label file for all other image files.
# > this script here splits the .npy files into two and moves them in subsequent folders - equivalent to ImageNet-C


# After downloading CIFAR-10-C and creating directories:
directory = "/path/to/CIFAR-10-C"
subdirectories = ["blur", "digital", "noise", "weather"]
split_numpy_files(directory, subdirectories)

# %%
path = "/path/to/CIFAR-10-C/labels.npy"
split_labels_file(path)

# %%
