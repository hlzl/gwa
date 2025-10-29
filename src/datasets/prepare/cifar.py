# %%
############### CREATE CIFAR DATASETS ###############
import os
import numpy as np
import tensorflow as tf

from git import Repo


def train_val_split(y, val_size: float = 0.1):
    # Number of samples per class for validation
    num_classes = y.max() + 1
    val_samples_per_class = int(val_size * y.size / num_classes)

    # Indices for validation and remaining training data
    val_indices = []
    train_indices = []

    for cls in range(num_classes):
        # Get all indices for the current class
        cls_indices = np.where(y == cls)[0]
        # Shuffle the indices
        np.random.shuffle(cls_indices)
        # Split into validation and remaining training
        val_indices.extend(cls_indices[:val_samples_per_class])
        train_indices.extend(cls_indices[val_samples_per_class:])

    # Convert to arrays
    val_indices = np.array(val_indices)
    train_indices = np.array(train_indices)

    # Return train and validation subsets
    return train_indices, val_indices


def get_cifar_numpy(dataset_name, save_dir, val_size: float = 0.1):
    """
    Downloads CIFAR-10 or CIFAR-100 datasets using tf.keras and saves them as .npy files.

    Parameters:
        dataset_name (str): "cifar10" or "cifar100".
        save_dir (str): Directory to save the .npy files.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load the dataset
    if dataset_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset_name == "cifar100":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()

    else:
        raise ValueError("Invalid dataset_name. Use 'cifar10', 'cifar100', or 'ciFAIR'.")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Train-val split
    if val_size > 0.0:
        train_indices, val_indices = train_val_split(y_train, val_size)
        X_train, y_train, X_val, y_val = (
            X_train[train_indices],
            y_train[train_indices],
            X_train[val_indices],
            y_train[val_indices],
        )
        np.save(
            os.path.join(save_dir, f"{dataset_name}_val_data_{int(val_size*100)}pct.npy"), X_val
        )
        np.save(
            os.path.join(save_dir, f"{dataset_name}_val_labels_{int(val_size*100)}pct.npy"),
            np.squeeze(y_val),
        )
    else:
        train_indices = np.array(range(y_train.shape[0]))
        val_indices = None

    # Save the datasets
    np.save(
        os.path.join(save_dir, f"{dataset_name}_train_data_{int((1-val_size)*100)}pct.npy"),
        X_train,
    )
    np.save(
        os.path.join(save_dir, f"{dataset_name}_train_labels_{int((1-val_size)*100)}pct.npy"),
        np.squeeze(y_train),
    )
    np.save(os.path.join(save_dir, f"{dataset_name}_test_data.npy"), X_test)
    np.save(os.path.join(save_dir, f"{dataset_name}_test_labels.npy"), np.squeeze(y_test))

    print(f"{dataset_name.upper()} dataset saved to {save_dir}.")

    return train_indices, val_indices


def get_cifar_n_numpy(url, save_dir, indices=None):
    """
    Get corresponding CIFAR-N labels for CIFAR-10 and CIFAR-100.

    Parameters:
        dataset_name (str): "cifar10" or "cifar100".
        save_dir (str): Directory to save the .npy files.
    """
    # Clone repository if not yet done
    if not os.path.isdir(save_dir):
        Repo.clone_from(url, save_dir)
    else:
        print(f"Directory {save_dir} already exists. Skipping cloning.")

    # Create subdirectory for numpy files
    os.makedirs(f"{save_dir}/numpy", exist_ok=True)

    # CIFAR-10-N
    noise_file = np.load(f"{save_dir}/data/CIFAR-10_human_ordered.npy", allow_pickle=True).item()
    order = np.load(f"{save_dir}/image_order_c10.npy", allow_pickle=True)
    for key in noise_file.keys():
        if "clean" not in key:
            if indices is None:
                noisy_labels = np.array(noise_file.get(key))[np.argsort(order)]
                train_pct = 100
            else:
                noisy_labels = np.array(noise_file.get(key))[np.argsort(order)][indices[0]]
                train_pct = int(len(indices[0]) / len(noise_file.get("clean_label")) * 100)
            np.save(
                os.path.join(
                    save_dir,
                    "numpy",
                    f"cifar10n_{key.replace("_label", "")}_train_labels_{train_pct}pct.npy",
                ),
                noisy_labels,
            )
    # Random labels similar to Zhang et al. (2017)
    np.save(
        os.path.join(save_dir, "numpy", f"cifar10_randint_train_labels_{train_pct}pct.npy"),
        np.random.randint(0, 10, len(noisy_labels)),
    )

    # CIFAR-100-N
    noise_file = np.load(f"{save_dir}/data/CIFAR-100_human_ordered.npy", allow_pickle=True).item()
    order = np.load(f"{save_dir}/image_order_c100.npy", allow_pickle=True)
    for key in noise_file.keys():
        if "clean" not in key:
            if indices is None:
                noisy_labels = np.array(noise_file.get(key))[np.argsort(order)]
                train_pct = 100
            else:
                noisy_labels = np.array(noise_file.get(key))[np.argsort(order)][indices[1]]
                train_pct = int(len(indices[1]) / len(noise_file.get("clean_label")) * 100)
            np.save(
                os.path.join(
                    save_dir,
                    "numpy",
                    f"cifar100n_{key.replace("_label", "")}_train_labels_{train_pct}pct.npy",
                ),
                np.array(noisy_labels),
            )
    # Random labels similar to Zhang et al. (2017)
    np.save(
        os.path.join(save_dir, "numpy", f"cifar100_randint_train_labels_{train_pct}pct.npy"),
        np.random.randint(0, 100, len(noisy_labels)),
    )

    print(f"CIFAR-N labels saved to {save_dir}")


# %%
# NOTE: Uncomment the following lines depending on the desired dataset
# # Standard CIFAR-10 / -100
# save_dir = "/path/to/CIFAR-10"
# c10_indices, _ = get_cifar_numpy("cifar10", save_dir, val_size=0.0)

# # %%
# save_dir = "/path/to/CIFAR-100"
# c100_indices, _ = get_cifar_numpy("cifar100", save_dir, val_size=0.0)

# # %%
# # CIFAR-10-100-N
# save_dir = "/path/to/cifar-10-100n"
# get_cifar_n_numpy(
#     "https://github.com/UCSC-REAL/cifar-10-100n", save_dir, (c10_indices, c100_indices)
# )


# Get test set v2 and noisy labels by git cloning the following repos:
# https://github.com/modestyachts/CIFAR-10.1
# https://github.com/UCSC-REAL/cifar-10-100n

# And alternative testset with wget from here (and unzip for .npy files):
# https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip
# https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-100.zip

# CIFAR-10-C, -100-C and -10-P (and unzip for .npy files):
# https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1
# https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1
# https://drive.google.com/drive/folders/1dY1_zeRyjMKdfmMbQ_uK8O1i0MVI9UbZ?usp=sharing

# %%
