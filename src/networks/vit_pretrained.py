# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar

import flax.linen as nn
import jax.numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

T = TypeVar("T")


def weight_standardize(w, axis, eps):
    """Subtracts mean and divides by standard deviation."""
    w = w - jnp.mean(w, axis=axis)
    w = w / (jnp.std(w, axis=axis) + eps)
    return w


class StdConv(nn.Conv):
    """Convolution with weight standardization."""

    def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
        param = super().param(name, init_fn, *init_args)
        if name == "kernel":
            param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
        return param


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x):
        return x


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""

    features: int
    strides: Sequence[int] = (1, 1)
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        needs_projection = x.shape[-1] != self.features * 4 or self.strides != (1, 1)

        residual = x
        if needs_projection:
            residual = StdConv(
                features=self.features * 4,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=False,
                name="conv_proj",
                dtype=self.dtype,
                param_dtype=self.dtype,
            )(residual)
            residual = nn.GroupNorm(name="gn_proj")(residual)

        y = StdConv(
            features=self.features,
            kernel_size=(1, 1),
            use_bias=False,
            name="conv1",
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(x)
        y = nn.GroupNorm(name="gn1")(y)
        y = nn.relu(y)
        y = StdConv(
            features=self.features,
            kernel_size=(3, 3),
            strides=self.strides,
            use_bias=False,
            name="conv2",
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(y)
        y = nn.GroupNorm(name="gn2")(y)
        y = nn.relu(y)
        y = StdConv(
            features=self.features * 4,
            kernel_size=(1, 1),
            use_bias=False,
            name="conv3",
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(y)

        y = nn.GroupNorm(name="gn3", scale_init=nn.initializers.zeros)(y)
        y = nn.relu(residual + y)
        return y


class ResNetStage(nn.Module):
    """A ResNet stage."""

    block_size: Sequence[int]
    nout: int
    first_stride: Sequence[int]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = ResidualUnit(self.nout, strides=self.first_stride, name="unit1", dtype=self.dtype)(x)
        for i in range(1, self.block_size):
            x = ResidualUnit(self.nout, strides=(1, 1), name=f"unit{i + 1}", dtype=self.dtype)(x)
        return x


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape, self.param_dtype)
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(  # pytype: disable=wrong-arg-types
            inputs
        )
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(  # pytype: disable=wrong-arg-types
            x
        )
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(y, deterministic=deterministic)

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = True
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, *, train):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        if self.add_position_embedding:
            x = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
                name="posembed_input",
                param_dtype=self.dtype,
            )(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoderblock_{lyr}",
                num_heads=self.num_heads,
                dtype=self.dtype,
            )(x, deterministic=not train)
        encoded = nn.LayerNorm(name="encoder_norm")(x)

        return encoded


class VisionTransformer(nn.Module):
    """VisionTransformer."""

    mlp_dim: int
    num_layers: int
    num_heads: int
    num_classes: int

    patch_size: int
    hidden_size: int

    attention_dropout_rate: float
    dropout_rate: float

    name: Optional[str] = None
    encoder: Type[nn.Module] = Encoder
    resnet: Optional[Any] = None

    representation_size: Optional[int] = None
    classifier: str = "token"
    head_bias_init: float = 0.0

    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, train: bool = False):

        x = inputs
        # (Possibly partial) ResNet root.
        if self.resnet is not None:
            width = int(64 * self.resnet.width_factor)

            # Root block.
            x = StdConv(
                features=width,
                kernel_size=(7, 7),
                strides=(2, 2),
                use_bias=False,
                name="conv_root",
                dtype=self.dtype,
                param_dtype=self.dtype,
            )(x)
            x = nn.GroupNorm(name="gn_root")(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

            # ResNet stages.
            if self.resnet.num_layers:
                x = ResNetStage(
                    block_size=self.resnet.num_layers[0],
                    nout=width,
                    first_stride=(1, 1),
                    name="block1",
                    dtype=self.dtype,
                )(x)
                for i, block_size in enumerate(self.resnet.num_layers[1:], 1):
                    x = ResNetStage(
                        block_size=block_size,
                        nout=width * 2**i,
                        first_stride=(2, 2),
                        name=f"block{i + 1}",
                        dtype=self.dtype,
                    )(x)

        # We can merge s2d+emb into a single conv; it's the same.
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="embedding",
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(x)

        # Here, x is a grid of embeddings.

        # (Possibly partial) Transformer.
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # If we want to add a class token, add it here.
        if self.classifier in ["token", "token_unpooled"]:
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c))
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        x = self.encoder(
            self.num_layers,
            self.mlp_dim,
            self.num_heads,
            self.dropout_rate,
            self.attention_dropout_rate,
            dtype=self.dtype,
            name="Transformer",
        )(x, train=train)

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
        elif self.classifier in ["unpooled", "token_unpooled"]:
            pass
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        if self.representation_size is not None:
            x = nn.Dense(
                features=self.representation_size,
                name="pre_logits",
                dtype=self.dtype,
                param_dtype=self.dtype,
            )(x)
            x = nn.tanh(x)
        else:
            x = IdentityLayer(name="pre_logits")(x)

        if self.num_classes:
            self.sow("latent_embeddings", "x", x)
            x = nn.Dense(
                features=self.num_classes,
                name="head",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(self.head_bias_init),
                dtype=self.dtype,
                param_dtype=self.dtype,
            )(x)
        return x
