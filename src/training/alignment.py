import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import kurtosis


def gwa(data, axis=1):
    return np.mean(data, axis=axis) / (kurtosis(data, axis=axis) + 1.2)


def head_alignment(params, logits, latent_embeddings, labels):
    """Compute per-sample alignment using closed form of cross-entropy loss"""
    diff = jax.nn.softmax(logits, axis=-1) - jnp.squeeze(labels)

    # Numerator: - x_b^T (W^T diff_b) for each sample b
    v = diff @ params.T  # B x D
    dot_product = -(latent_embeddings * v).sum(axis=-1)  # B
    # Denominator: ||grad|| * ||W||
    grad_norm = jnp.linalg.norm(latent_embeddings, axis=-1) * jnp.linalg.norm(diff, axis=-1)  # B
    alignment_scores = dot_product / (grad_norm * jnp.linalg.norm(params) + 1e-12)

    return alignment_scores


def logits_and_latents(params, forward, images):
    # Compute latent embeddings without gradient
    logits, latent_embeddings = forward(
        {"params": params}, jnp.squeeze(images), mutable=["latent_embeddings"]
    )
    return logits, jnp.squeeze(latent_embeddings["latent_embeddings"]["x"][0])
