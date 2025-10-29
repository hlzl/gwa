import jax
import jax.numpy as jnp
import optax
from functools import partial

import datasets
from .alignment import logits_and_latents, head_alignment


def accuracy(logits, y, one_hot: bool = True):
    if one_hot:
        y = jnp.argmax(y, axis=-1)
    return jnp.mean(jnp.argmax(logits, axis=-1) == y)


def softmax_ce(params, state, input, targets):
    logits, latent_embeddings = logits_and_latents(params, state.apply_fn, input)
    loss = optax.softmax_cross_entropy(logits.astype(jnp.float32), targets).mean()
    acc = accuracy(logits, targets, one_hot=True)
    return loss, (acc, logits, latent_embeddings)


## Full precision
def backprop(state, x, y):
    (loss, aux), grads = jax.value_and_grad(softmax_ce, has_aux=True)(state.params, state, x, y)

    alignment = head_alignment(state.params["head"]["kernel"], aux[1], aux[2], y)
    return grads, loss, aux[0], alignment


def train_step(state, x, y, pmean, mixup: float = 0.0, grad_clip: float = None):
    main_keys, rngs = state.split_rngs(x.shape[1])
    x, y = datasets.utils.mixup(mixup, jnp.squeeze(x), jnp.squeeze(y), rngs)

    # Compute gradients, loss and alignment
    grads, loss, acc, alignment = backprop(state, x, y)

    with jax.named_scope("sync_gradients"):
        grads = jax.tree_util.tree_map(pmean, grads)

    if grad_clip is not None:
        clip_ratio = jnp.minimum(1.0, grad_clip / optax.global_norm(grads))
        grads = jax.tree_util.tree_map(lambda x: clip_ratio * x, grads)

    # Apply gradients
    new_state = state.apply_gradients(grads=grads)
    return (
        new_state.replace(**main_keys),
        jnp.expand_dims(loss, axis=0),
        jnp.expand_dims(acc, axis=0),
        jnp.expand_dims(alignment, axis=0),
    )


## Mixed precision
def backprop_scale(state, x, y):
    dynamic_scale, is_fin, (loss, aux), grads = state.dynamic_scale.value_and_grad(
        softmax_ce, has_aux=True
    )(state.params, state, x, y)

    alignment = head_alignment(state.params["head"]["kernel"], aux[1], aux[2], y)
    return grads, loss, aux[0], (dynamic_scale, is_fin), alignment


def train_step_scale(state, x, y, pmean, mixup: float = 0.0, grad_clip: float = None):
    main_keys, rngs = state.split_rngs(x.shape[1])
    x, y = datasets.utils.mixup(mixup, jnp.squeeze(x), jnp.squeeze(y), rngs)

    # Compute gradients, loss and alignment
    grads, loss, acc, scaler, alignment = backprop_scale(state, x, y)

    with jax.named_scope("sync_gradients"):
        grads = jax.tree_util.tree_map(pmean, grads)

    if grad_clip is not None:
        clip_ratio = jnp.minimum(1.0, grad_clip / optax.global_norm(grads))
        grads = jax.tree_util.tree_map(lambda x: clip_ratio * x, grads)

    # Apply gradients
    new_state = state.apply_gradients(grads=grads)

    # Update state
    new_state = new_state.replace(
        opt_state=jax.tree_util.tree_map(
            partial(jnp.where, jnp.all(scaler[1])),
            new_state.opt_state,
            state.opt_state,
        ),
        params=jax.tree_util.tree_map(
            partial(jnp.where, jnp.all(scaler[1])),
            new_state.params,
            state.params,
        ),
        dynamic_scale=scaler[0],
        **main_keys,
    )

    return (
        new_state,
        jnp.expand_dims(loss, axis=0),
        jnp.expand_dims(acc, axis=0),
        jnp.expand_dims(alignment, axis=0),
    )


def eval_step(state, x, y):
    # Compute loss and accuracy
    loss, aux = softmax_ce(state.params, state, jnp.squeeze(x), jnp.squeeze(y))

    return jnp.expand_dims(loss, axis=0), jnp.expand_dims(aux[0], axis=0), aux[1]
