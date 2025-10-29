# Forked from @https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class GRN(nn.Module):
    dim: int

    def setup(self):
        self.gamma = self.param("gamma", nn.initializers.constant(0.0), (1, 1, 1, self.dim))
        self.beta = self.param("beta", nn.initializers.constant(0.0), (1, 1, 1, self.dim))

    def __call__(self, x):
        Gx = jnp.linalg.norm(abs(x) + 1e-9, ord="fro", axis=(1, 2), keepdims=True)
        Nx = Gx / (jnp.mean(Gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    dim: int
    kernel_size: int
    dtype: Any = jnp.float32

    def setup(self):
        self.dwconv = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            feature_group_count=self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.truncated_normal(stddev=0.02),
        )
        self.norm = nn.LayerNorm(epsilon=1e-6)
        self.pwconv1 = nn.Dense(
            features=4 * self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.truncated_normal(stddev=0.02),
        )
        self.act = nn.gelu
        self.grn = GRN(4 * self.dim)
        self.pwconv2 = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.truncated_normal(stddev=0.02),
        )

    def __call__(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class ConvNeXtV2(nn.Module):
    in_chans: int = 3
    num_classes: int = 1000
    depths: tuple = (3, 3, 9, 3)
    dims: tuple = (96, 192, 384, 768)
    downsample: int = 4
    head_init_scale: float = 1.0
    dtype: Any = jnp.float32

    def setup(self):
        self.downsample_layers = [nn.Sequential(
            [
                nn.Conv(
                    features=self.dims[0],
                    kernel_size=(self.downsample, self.downsample),
                    strides=(self.downsample, self.downsample),
                    dtype=self.dtype,
                    kernel_init=jax.nn.initializers.truncated_normal(stddev=0.02),
                    use_bias=False,
                ),
                nn.LayerNorm(epsilon=1e-6),
            ]
        )] + [
            nn.Sequential(
                [
                    nn.LayerNorm(epsilon=1e-6),
                    nn.Conv(
                        features=self.dims[i + 1],
                        kernel_size=(2, 2),
                        strides=(2, 2),
                        dtype=self.dtype,
                        kernel_init=jax.nn.initializers.truncated_normal(stddev=0.02),
                    ),
                ]
            )
            for i in range(3)
        ]

        kernel_size = 3 if self.downsample <= 2 else 5
        self.stages = [
            nn.Sequential(
                [
                    Block(dim=self.dims[i], kernel_size=kernel_size, dtype=self.dtype)
                    for _ in range(self.depths[i])
                ]
            )
            for i in range(len(self.dims))
        ]

        self.norm = nn.LayerNorm(epsilon=1e-6)
        self.head = nn.Dense(
            features=self.num_classes,
            kernel_init=jax.nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.constant(0.0),
            dtype=jnp.float32,
        )

    def __call__(self, x, **kwargs):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x= self.stages[i](x)
        x = self.norm(x.mean(axis=(1, 2)))  # Global average pooling (N, H, W, C) -> (N, C)

        self.sow("latent_embeddings", "x", x)
        x = self.head(x)
        return x


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
