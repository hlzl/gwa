import joblib
import matplotlib.pyplot as plt
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from absl import logging
from tqdm import tqdm

import datasets
from training.utils import create_scheduler, create_train_state
from training.functions import train_step, train_step_scale, eval_step
from training.alignment import gwa


def train_loop(cfg, trainloader, valloader, testloader, num_samples):
    """
    Completely standard training. Nothing interesting to see here.
    """
    # Useful constants
    steps_per_epoch = int(num_samples[0] // cfg.optim.batch * cfg.optim.acc_steps)
    samples_per_epoch = num_samples[0] - (num_samples[0] % cfg.optim.batch)
    test_steps = {
        "val": max(1, num_samples[1] // cfg.optim.batch),
        "test": max(0, num_samples[2] // cfg.optim.batch),
    }

    # Function Setup
    scheduler = create_scheduler(cfg)
    dtype = jnp.float32
    if cfg.mixed_precision and jax.local_devices()[0].platform == "gpu":
        dtype = jnp.bfloat16

    # State Setup with sharded initialization for data parallelism
    num_devices = jax.device_count()
    init_key = jax.random.key(cfg.seed)
    mesh = Mesh(np.array(jax.devices()), ("data",))
    init_parallel = shard_map(
        partial(create_train_state, cfg=cfg, dtype=dtype, scheduler=scheduler),
        mesh,
        in_specs=(P(), P("data")),
        out_specs=P(),
        check_rep=False,
    )
    state = init_parallel(
        init_key, jnp.ones((num_devices, cfg.dataset.image_size, cfg.dataset.image_size, 3))
    )

    # Create loss and metric functions
    if dtype == jnp.bfloat16:
        update_step = train_step_scale
    else:
        dtype = jnp.float32
        update_step = train_step

    # JIT main functions
    update_step = jax.jit(
        shard_map(
            partial(
                update_step,
                mixup=cfg.dataset.get("mixup"),
                grad_clip=cfg.optim.get("grad_clip_norm"),
                pmean=(
                    partial(jax.lax.pmean, axis_name="data") if num_devices > 1 else lambda x: x
                ),
            ),
            mesh,
            in_specs=(P(), P("data"), P("data")),
            out_specs=(P(), P("data"), P("data"), P("data")),
            check_rep=False,
        ),
        donate_argnames=("state"),
    )
    test_step = jax.jit(
        shard_map(
            eval_step,
            mesh,
            in_specs=(P(), P("data"), P("data")),
            out_specs=(P("data"), P("data"), P("data")),
            check_rep=False,
        )
    )

    # Training loooop
    epochs = cfg.optim.n_iterations // (num_samples[0] // cfg.optim.batch)
    gwa_values = []
    val_accuracies = []
    for epoch in range(0, epochs):
        # Train
        train_logs = None
        for i, batch in tqdm(
            zip(range(steps_per_epoch), datasets.prefetch(trainloader, n_prefetch=2)),
            leave=False,
            desc=f"Epoch {epoch+1}/{epochs} [train]",
            total=steps_per_epoch,
        ):
            state, loss, acc, alignment = update_step(state, batch["input"], batch["target"])

            # Create logs dict and save per-sample values for batch
            if train_logs is None:
                train_logs = {
                    "train_loss": np.zeros(steps_per_epoch),
                    "train_acc": np.zeros(steps_per_epoch),
                    "alignment": np.zeros(samples_per_epoch),
                }

            train_logs["train_loss"][i] = loss.mean()
            train_logs["train_acc"][i] = acc.mean()
            train_logs["alignment"][
                i * cfg.optim.batch : i * cfg.optim.batch + batch["id"].size
            ] = alignment.reshape(-1)

        # GWA for full dataset across epoch
        gwa_values.append(gwa(train_logs["alignment"], axis=0))

        logging.info(f"EPOCH {epoch+1}:")
        logging.info(
            f"train-loss {(train_logs['train_loss'].mean()):.3f}, "
            f"train-acc {(train_logs['train_acc'].mean()*100):.2f}, "
            f"gwa {gwa_values[epoch]:.4f}"
        )

        # Validation and Testing
        test_logs = {}
        for ds, dataloader in enumerate([valloader, testloader]):
            if dataloader is not None:
                name = ["val", "test"][ds]
                test_logs[f"{name}_loss"] = np.zeros(test_steps[name])
                test_logs[f"{name}_acc"] = np.zeros(test_steps[name])

                for i, batch in tqdm(
                    zip(range(test_steps[name]), datasets.prefetch(dataloader, n_prefetch=2)),
                    leave=False,
                    desc=f"Epoch {epoch+1}/{epochs} [{name}]",
                    total=test_steps[name],
                ):
                    loss, acc, _ = test_step(state, batch["input"], batch["target"])
                    test_logs[f"{name}_loss"][i] = loss.mean()
                    test_logs[f"{name}_acc"][i] = acc.mean()

                # Console output
                logging.info(
                    f"{name}-loss {test_logs[f"{name}_loss"].mean():.3f}, "
                    f"{name}-accuracy {(test_logs[f"{name}_acc"].mean()*100):.2f} "
                )
                if name == "val":
                    val_accuracies.append(test_logs[f"{name}_acc"].mean())

        # Save logs
        joblib.dump({**train_logs, **test_logs}, f"logs/jax_metrics_{epoch}.joblib")

    # Plot normalized values over time
    val_acc = np.array(val_accuracies)
    val_acc = (val_acc - val_acc.min()) / abs(val_acc - val_acc.min()).max()
    plt.plot(val_acc, label="Val Acc")
    gwa_scores = np.array(gwa_values)
    gwa_scores = (gwa_scores - gwa_scores.min()) / abs(gwa_scores - gwa_scores.min()).max()
    plt.plot(gwa_scores, label="GWA")
    plt.legend()
    plt.savefig("logs/jax_acc-vs-gwa.png")

    logging.info(">> Training finished.")
