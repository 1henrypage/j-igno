# src/components/encoder.py

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Tuple, Callable

from .activation import FunActivation
from .fcn import FCNet
from .cnn import CNNet1d, CNNet2d


class EncoderFCNet(nn.Module):
    """Fully connected encoder"""
    layers_list: list[int]
    activation: str | Callable
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if isinstance(self.activation, str):
            self.act_fn = FunActivation()(self.activation)
        else:
            self.act_fn = self.activation

        self.net = FCNet(
            layers_list=self.layers_list,
            activation=self.act_fn,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode input to latent space

        Args:
            x: Input (n_batch, n_points, in_size)

        Returns:
            beta: Latent code (n_batch, latent_dim)
        """
        # Flatten spatial dimensions: (n_batch, n_points, in_size) -> (n_batch, n_points*in_size)
        x = x.reshape(x.shape[0], -1)
        # Encode: (n_batch, n_points*in_size) -> (n_batch, latent_dim)
        x = self.net(x)
        return x


class EncoderFCNet_VAE(nn.Module):
    """Variational autoencoder with reparameterization trick"""
    layers_list: list
    activation: str | Callable
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if isinstance(self.activation, str):
            self.act_fn = FunActivation()(self.activation)
        else:
            self.act_fn = self.activation

        # Two separate networks for mu and log_var
        self.net_mu = FCNet(
            layers_list=self.layers_list,
            activation=self.act_fn,
            dtype=self.dtype
        )
        self.net_log_var = FCNet(
            layers_list=self.layers_list,
            activation=self.act_fn,
            dtype=self.dtype
        )

    def reparam(self, rng: jax.Array, mu: jax.Array, log_var: jax.Array) -> jax.Array:
        """Reparameterization trick

        Args:
            rng: PRNG key
            mu: Mean
            log_var: Log variance

        Returns:
            Sample from N(mu, exp(log_var))
        """
        std = jnp.exp(0.5 * log_var)
        eps = random.normal(rng, log_var.shape)
        return mu + std * eps

    @nn.compact
    def __call__(
            self,
            x: jax.Array,
            rng: jax.Array = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Encode with VAE

        Args:
            x: Input (n_batch, n_points, in_size)
            rng: PRNG key for sampling

        Returns:
            beta: Sample (n_batch, n_latent)
            mu: Mean (n_batch, n_latent)
            log_var: Log variance (n_batch, n_latent)
        """
        # Flatten: (n_batch, n_points, in_size) -> (n_batch, n_points*in_size)
        x = x.reshape(x.shape[0], -1)

        # Get mu and log_var
        mu = self.net_mu(x)
        log_var = self.net_log_var(x)

        # Sample using reparameterization
        if rng is not None:
            beta = self.reparam(rng, mu, log_var)
        else:
            # During inference, just use mean
            beta = mu

        return beta, mu, log_var


class EncoderCNNet1d(nn.Module):
    """1D CNN encoder"""
    conv_arch: list[int]
    fc_arch: list[int]
    activation_conv: str | Callable
    activation_fc: str | Callable
    kernel_size: int = 5
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_channel = self.conv_arch[0]
        self.net = CNNet1d(
            conv_arch=self.conv_arch,
            fc_arch=self.fc_arch,
            activation_conv=self.activation_conv,
            activation_fc=self.activation_fc,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode 1D input

        Args:
            x: Input (n_batch, mesh_size, in_channel)

        Returns:
            beta: Latent code (n_batch, latent_dim)
        """
        # Note: JAX Conv expects (batch, length, channels)
        # PyTorch expects (batch, channels, length)
        # Input is already in correct format for JAX
        x = self.net(x)
        return x


class EncoderCNNet2d(nn.Module):
    """2D CNN encoder"""
    conv_arch: list[int]
    fc_arch: list[int]
    activation_conv: str | Callable
    activation_fc: str | Callable
    nx_size: int
    ny_size: int
    kernel_size: Tuple[int, int] = (5, 5)
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_channel = self.conv_arch[0]
        self.net = CNNet2d(
            conv_arch=self.conv_arch,
            fc_arch=self.fc_arch,
            activation_conv=self.activation_conv,
            activation_fc=self.activation_fc,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode 2D input

        Args:
            x: Input (n_batch, ny*nx, in_channel)

        Returns:
            beta: Latent code (n_batch, latent_dim)
        """
        # Reshape to 2D grid
        # Input: (n_batch, ny*nx, in_channel)
        # Output for Conv2d: (n_batch, ny, nx, in_channel)
        x = x.reshape(-1, self.ny_size, self.nx_size, self.in_channel)
        x = self.net(x)
        return x


class EncoderCNNet2dTanh(nn.Module):
    """2D CNN encoder with tanh output"""
    conv_arch: list[int]
    fc_arch: list[int]
    activation_conv: str | Callable
    activation_fc: str | Callable
    nx_size: int
    ny_size: int
    kernel_size: Tuple[int, int] = (5, 5)
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = EncoderCNNet2d(
            conv_arch=self.conv_arch,
            fc_arch=self.fc_arch,
            activation_conv=self.activation_conv,
            activation_fc=self.activation_fc,
            nx_size=self.nx_size,
            ny_size=self.ny_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode with tanh output

        Args:
            x: Input (n_batch, ny*nx, in_channel)

        Returns:
            beta: Latent code in [-1, 1] (n_batch, latent_dim)
        """
        beta = self.encoder(x)
        return jnp.tanh(beta)