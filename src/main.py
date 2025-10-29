import os
import random
import hydra
from omegaconf import DictConfig

import numpy as np
import tensorflow as tf

import datasets
from train import train_loop


@hydra.main(
    version_base=None,
    config_path=f"{os.path.dirname(os.path.dirname(__file__))}/conf",
    config_name="config",
)
def main(cfg: DictConfig):
    # Make deterministic
    seed = cfg.seed
    if seed is None:
        import time

        seed = np.random.randint(0, 1000000)
        seed ^= int(time.time())
        cfg.seed = seed

    random.seed(seed)
    np.random.seed(seed)

    # Create directory for logging and get data
    os.makedirs("logs", exist_ok=True)
    # Hide any GPUs form TensorFlow to prevent it being unavailable to JAX
    tf.config.experimental.set_visible_devices([], "GPU")
    if cfg.dataset.get("numpy_path"):  # numpy files
        train_ds, num_train = datasets.numpy_files(config=cfg, mode="train")
        if cfg.dataset.numpy_path.get("val_data") is not None:
            val_ds, num_val = datasets.numpy_files(config=cfg, mode="val")
        else:
            val_ds, num_val = None, 0
        test_ds, num_test = datasets.numpy_files(config=cfg, mode="test")
    elif cfg.dataset.get("jpeg_dir"):  # directory with jpeg files
        train_ds, num_train, val_ds, num_val = datasets.jpeg_directory(config=cfg, mode="train")
        test_ds, num_test, _, _ = datasets.jpeg_directory(config=cfg, mode="test")
    else:
        raise ValueError("Need to either specify 'numpy_path', or 'jpeg_dir' in dataset config.")

    # Train model
    train_loop(cfg, train_ds, val_ds, test_ds, (num_train, num_val, num_test))


if __name__ == "__main__":
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    main()
