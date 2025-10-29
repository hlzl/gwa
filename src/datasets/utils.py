import jax
from chex import Array


def mixup(
    mixup_alpha, images: Array, labels: Array, rngs, num_classes: int = None
) -> tuple[Array, Array]:
    """ADAPTED FOR INT LABELS"""
    if mixup_alpha is not None and mixup_alpha > 0.0:
        ratio = jax.random.beta(rngs["mixup"][0], *(mixup_alpha,) * 2)
        randperm = jax.random.permutation(rngs["mixup"][0], images.shape[0])
        images = ratio * images + (1 - ratio) * images[randperm]
        if num_classes:
            labels = jax.numpy.eye(num_classes)[labels]
        labels = ratio * labels + (1 - ratio) * labels[randperm]

    return images, labels
