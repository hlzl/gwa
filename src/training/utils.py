from chex import Array, ArrayTree, PRNGKey
from typing import Union
from flax.training import train_state, dynamic_scale

import os
import joblib
import hydra
import numpy as np
import jax
import jax.numpy as jnp
import flax
import optax

import networks  # required for hydra
from .checkpoint import load_pretrained


class TrainState(train_state.TrainState):
    dynamic_scale: dynamic_scale.DynamicScale
    mixup_rng: Union[PRNGKey, Array]
    dropout_rng: Union[PRNGKey, Array]
    noise_rng: Union[PRNGKey, Array]

    def split_rngs(self, batch_size) -> tuple[ArrayTree, ArrayTree]:
        """Create keys for forward pass by splitting from main rng.
        This is useful for generating a different random number for each device.

        Args:
            rng: The random number generator.

        Returns:
            main_rng: Main random number generator, same across devices - requires updateing state with
            keys: New random numbers, different for each device index along the axis.
        """
        mixup_rng, mixup_key = jax.random.split(self.mixup_rng, 2)
        dropout_rng, dropout_key = jax.random.split(self.dropout_rng, 2)
        noise_rng, noise_key = jax.random.split(self.noise_rng, 2)

        # Create new state with by calling state.replace(**main_rngs)
        main_rngs = {
            "mixup_rng": mixup_rng,
            "dropout_rng": dropout_rng,
            "noise_rng": noise_rng,
        }

        # Split into batch size and vary for each device
        keys = {
            "mixup": jax.random.split(
                jax.random.fold_in(mixup_key, jax.process_index()), batch_size
            ),
            "dropout": jax.random.split(
                jax.random.fold_in(dropout_key, jax.process_index()), batch_size
            ),
            "noise": jax.random.split(
                jax.random.fold_in(noise_key, jax.process_index()), batch_size
            ),
        }

        return main_rngs, keys

    def unwrap_keys(self):
        """Convert PRNG keys to arrays."""
        return {
            "mixup_rng": jax.random.key_data(self.mixup_rng),
            "dropout_rng": jax.random.key_data(self.dropout_rng),
            "noise_rng": jax.random.key_data(self.noise_rng),
        }

    def wrap_keys(self):
        """Convert arrays to PRNG keys."""
        return {
            "mixup_rng": jax.random.wrap_key_data(self.mixup_rng),
            "dropout_rng": jax.random.wrap_key_data(self.dropout_rng),
            "noise_rng": jax.random.wrap_key_data(self.noise_rng),
        }


def create_scheduler(cfg):
    if cfg.optim.scheduler is None:
        return {"learning_rate": lambda _: cfg.optim.lr}
    elif cfg.optim.scheduler._target_ == "piecewise_schedule" and cfg.optim.scheduler.get(
        "increase_steps"
    ):
        warmup_schedule = optax.linear_schedule(
            init_value=cfg.optim.scheduler.init_value,
            end_value=cfg.optim.lr,
            transition_steps=cfg.optim.scheduler.increase_steps,
        )
        decay_schedule = optax.cosine_decay_schedule(
            init_value=cfg.optim.lr,
            decay_steps=cfg.optim.scheduler.decrease_steps,
        )

        # Combine the two schedules to get piecewise schedule
        return {
            "learning_rate": optax.join_schedules(
                schedules=[warmup_schedule, decay_schedule],
                boundaries=[cfg.optim.scheduler.increase_steps],
            )
        }
    else:
        return {
            "learning_rate": hydra.utils.instantiate(
                cfg.optim.scheduler, decay_steps=cfg.optim.n_iterations + 1
            )
        }


def create_train_state(key, x, cfg, dtype, scheduler):
    """Create initial training state."""
    init_key, mixup_key, dropout_key, noise_key = jax.random.split(key, 4)

    ### 1. Setup model and parameters ###
    model = hydra.utils.instantiate(cfg.arch.model, dtype=dtype)
    params = model.init({"params": init_key}, x)["params"]

    ### 2. Setup optimizer ###
    grad_scale = None
    tx = hydra.utils.instantiate(cfg.optim.fn, **scheduler)

    # Mixed precision
    if dtype == jnp.bfloat16:
        grad_scale = dynamic_scale.DynamicScale()
        if "adam" in cfg.optim.fn._target_:
            tx = hydra.utils.instantiate(cfg.optim.fn, **scheduler, mu_dtype=dtype)

    # SGD with weight decay
    if "adam" not in cfg.optim.fn._target_ and cfg.optim.get("weight_decay"):
        tx = optax.chain(optax.add_decayed_weights(cfg.optim.weight_decay), tx)

    # Accumulating gradients
    if cfg.optim.acc_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=cfg.optim.acc_steps)

    ### 3. Create state - either from scratch or checkpoint ###
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=grad_scale,
        mixup_rng=mixup_key,
        dropout_rng=dropout_key,
        noise_rng=noise_key,
    )

    # Load weights
    if cfg.loading.dir is not None:
        if "i21k" in cfg.loading.dir:  # Google pre-trained weights
            params = load_pretrained(
                pretrained_path=cfg.loading.dir,
                init_params=params,
                model_config={
                    "representation_size": None,
                    "classifier": cfg.arch.model.classifier,
                },
            )
            state = state.replace(params=params)
        else:  # local checkpoints
            loaded_state = flax.serialization.from_state_dict(
                state,
                joblib.load(
                    os.path.join(cfg.loading.dir, "checkpoints", f"ckpts_{cfg.loading.epoch}")
                ),
            )
            if cfg.loading.resume_training is True:
                del state
                state = loaded_state
            else:
                state = state.replace(params=loaded_state.params)

    del params  # not needed anymore
    return state


def prefetch(dataset, num_devices: int = 1, n_prefetch: int = 2):
    """Prefetches data to device while also converting input to numpy array."""
    ds_iter = iter(dataset)
    ds_iter = map(
        lambda x: jax.tree_util.tree_map(
            lambda t: np.reshape(t.numpy(), (num_devices, -1) + t.shape[1:]),
            x,
        ),
        ds_iter,
    )
    return flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
