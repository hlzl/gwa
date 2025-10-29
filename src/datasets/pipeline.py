# Forked from https://github.com/google-research/vision_transformer/blob/main/vit_jax/input_pipeline.py
import os
import glob
import random
import jax
import flax
import numpy as np
import tensorflow as tf

from absl import logging

from datasets import augment, transforms


def get_directory_info(directory):
    """Returns information about directory dataset -- see `get_dataset_info()`."""
    jpeg_glob = f"{directory}/*/*.[jJ][pP][eE][gG]"
    jpg_glob = f"{directory}/*/*.[jJ][pP][gG]"
    paths = glob.glob(jpeg_glob) + glob.glob(jpg_glob)
    if len(paths) == 0:  # likely due to hierarchical classes and corresponding dir structure
        logging.info(
            f"Couldn't find any jpeg files within first subdirectories. Checking one level deeper."
        )
        jpeg_glob = f"{directory}/*/*/*.[jJ][pP][eE][gG]"
        jpg_glob = f"{directory}/*/*/*.[jJ][pP][gG]"
        paths = glob.glob(jpeg_glob) + glob.glob(jpg_glob)
    get_classname = lambda path: path.split("/")[-2]
    class_names = sorted(set(map(get_classname, paths)))
    return dict(
        num_examples=len(paths),
        num_classes=len(class_names),
        int2str=lambda id_: class_names[id_],
        examples_glob=[jpeg_glob, jpg_glob],
    )


def jpeg_directory(*, config, mode):
    """Returns dataset as read from specified `directory`."""
    if mode == "log":
        directory = config.dataset.jpeg_dir["train"]
    else:
        directory = config.dataset.jpeg_dir[f"{mode}"]
    if not os.path.isdir(directory):
        raise ValueError(f"Expected to find directories .")
    logging.info(f"Reading dataset from '{directory}'.")

    dataset_info = get_directory_info(directory)
    class_names = [dataset_info["int2str"](id_) for id_ in range(dataset_info["num_classes"])]

    def _pp(path):
        index = tf.strings.to_number(
            tf.strings.substr(
                tf.strings.regex_replace(
                    tf.strings.split(tf.strings.split(path, "/")[-1], "_")[-1], r"\D", ""
                ),
                0,
                9,
            ),
            out_type=tf.uint32,
        )
        return dict(
            image=path,
            label=tf.where(tf.strings.split(path, "/")[-2] == class_names)[0][0],
            id=index,
        )

    # Transforms
    train_ops = [transforms.get_decode_jpeg_and_inception_crop(config.dataset.image_size)]
    if config.dataset.rand_aug > 0:
        train_ops = train_ops + [augment.RandAugment(2, config.dataset.rand_aug).distort]
    train_ops = train_ops + [transforms.get_random_flip_lr(), transforms.get_value_range()]
    val_ops = [
        transforms.get_decode(channels=3, precise=True),
        transforms.get_resize_small(256),  # from https://arxiv.org/pdf/2205.01580
        transforms.get_central_crop(config.dataset.image_size),
        transforms.get_value_range(),
    ]

    # Create datasets
    all_paths = glob.glob(dataset_info["examples_glob"][0]) + glob.glob(
        dataset_info["examples_glob"][1]
    )
    if mode == "train":
        # Train-val split
        random.shuffle(all_paths)
        train_ds = tf.data.Dataset.from_tensor_slices(
            all_paths[: int((1 - config.dataset.jpeg_dir.val_size) * len(all_paths))]
        )
        val_subset = all_paths[int((1 - config.dataset.jpeg_dir.val_size) * len(all_paths)) :]
        if len(val_subset) > 0:
            val_ds = tf.data.Dataset.from_tensor_slices(
                all_paths[int((1 - config.dataset.jpeg_dir.val_size) * len(all_paths)) :]
            )
            val_loader = get_image_data(
                data=val_ds,
                mode="val",
                preprocess=_pp,
                image_decoder=lambda x: tf.io.read_file(x),
                transforms=val_ops,
                num_classes=dataset_info["num_classes"],
                repeats=None,
                batch_size=config.optim.batch,
                shuffle_buffer=1,
                config=config,
            )
        else:
            val_loader = None

        return (
            get_image_data(
                data=train_ds,
                mode="train",
                preprocess=_pp,
                image_decoder=lambda x: tf.io.read_file(x),
                transforms=train_ops,
                num_classes=dataset_info["num_classes"],
                repeats=None,
                batch_size=config.optim.batch,
                shuffle_buffer=min(dataset_info["num_examples"], 250_000),
                config=config,
            ),
            int(dataset_info["num_examples"] * (1 - config.dataset.jpeg_dir.val_size)),
            val_loader,
            int(dataset_info["num_examples"] * config.dataset.jpeg_dir.val_size),
        )
    else:
        data = tf.data.Dataset.from_tensor_slices(
            sorted(all_paths, key=lambda path: os.path.basename(path))
        )  # sort files for ImageNet-ReaL
        return (
            get_image_data(
                data=data,
                mode=mode,
                preprocess=_pp,
                image_decoder=lambda x: tf.io.read_file(x),
                transforms=val_ops,
                num_classes=dataset_info["num_classes"],
                repeats=None,
                batch_size=config.optim.batch,
                shuffle_buffer=dataset_info["num_examples"],
                config=config,
            ),
            dataset_info["num_examples"],
            None,
            None,
        )


def numpy_files(*, config, mode):
    """Returns dataset as read from specified numpy files."""
    if mode == "train" or mode == "log":
        version = "train"
    else:
        version = mode

    if not os.path.exists(config.dataset.numpy_path[f"{version}_data"]):
        raise ValueError(
            f"Expected to find '{version}_data.npy' in {config.dataset.numpy_path[f'{version}_data']}."
        )
    logging.info(f"Reading data from '{config.dataset.numpy_path[f'{version}_data']}'.")
    logging.info(f"Reading labels from '{config.dataset.numpy_path[f'{version}_label']}'.")

    data = np.load(config.dataset.numpy_path[f"{version}_data"])
    labels = np.load(config.dataset.numpy_path[f"{version}_label"])
    if config.dataset.get("train_size"):  # subsample
        data = data[:: int(1 / config.dataset.train_size)]
        labels = labels[:: int(1 / config.dataset.train_size)]
    assert len(labels.shape) == 1, "Expect labels array with integers and only one dimension."
    indices = np.array(range(labels.size))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels, indices))

    def _pp(image, label, index):
        return dict(image=image, label=label, id=index)

    if mode == "train":
        ops = [transforms.get_resize(config.dataset.image_size + 4)]
        if config.dataset.rand_aug > 0:
            ops = ops + [augment.RandAugment(2, config.dataset.rand_aug).distort]
        ops = ops + [
            transforms.get_random_crop(config.dataset.image_size),
            transforms.get_random_flip_lr(),
            transforms.get_value_range(),
        ]
    else:
        ops = [transforms.get_resize(config.dataset.image_size), transforms.get_value_range()]

    return (
        get_image_data(
            data=dataset,
            mode=mode,
            preprocess=_pp,
            image_decoder=lambda x: x,
            transforms=ops,
            num_classes=labels.max() + 1,
            repeats=None,
            batch_size=min(config.optim.batch, data.shape[0]),
            shuffle_buffer=labels.size,
            config=config,
        ),
        labels.size,
    )


def get_image_data(
    *,
    data,
    mode,
    preprocess,
    image_decoder,
    transforms,
    num_classes,
    repeats,
    batch_size,
    shuffle_buffer,
    config,
):
    """Returns dataset for training/eval.

    Args:
      data: tf.data.Dataset to read data from.
      mode: Must be "train" or "test".
      num_classes: Number of classes (used for one-hot encoding).
      preprocess: Optional preprocess function. This function will be applied to the dataset
        just after repeat/shuffling, and before the data augmentation preprocess step is applied.
      image_decoder: Applied to `features['image']` after shuffling. Decoding the image after
        shuffling allows for a larger shuffle buffer.
      transforms: List of data augmentation functions applied right after the image decoder.
      repeats: How many times the dataset should be repeated. For indefinite repeats use None.
      batch_size: Global batch size. Note that the returned dataset will have
        dimensions [num_devices, batch_size / num_devices, ...].
      config: config with image size after cropping (for training) / resizing (for evaluation) etc.
    """

    def _pp(data):
        im = image_decoder(data["image"])
        for transform in transforms:
            im = transform(im)
        label = tf.one_hot(data["label"], num_classes)  # pylint: disable=no-value-for-parameter
        return {"input": im, "target": label, "id": data["id"]}

    if mode == "train":
        data = data.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    data = data.repeat(repeats)
    if preprocess is not None:
        data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
    data = data.map(_pp, tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=True)

    # Reshape data such that it can be sharded easily accross distributed devices
    num_devices = jax.local_device_count()

    def _shard(data):
        data["input"] = tf.reshape(
            data["input"],
            [
                num_devices,
                -1,
                config.dataset.image_size,
                config.dataset.image_size,
                data["input"].shape[-1],
            ],
        )
        data["target"] = tf.reshape(data["target"], [num_devices, -1, num_classes])
        data["id"] = tf.reshape(data["id"], [num_devices, -1])
        return data

    if num_devices is not None:
        data = data.map(_shard, tf.data.experimental.AUTOTUNE)

    return data.prefetch(1)


def prefetch(dataset, n_prefetch):
    """Prefetches data to device and converts to numpy array."""
    ds_iter = iter(dataset)
    ds_iter = map(lambda x: jax.tree.map(lambda t: np.asarray(memoryview(t)), x), ds_iter)
    if n_prefetch:
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter
