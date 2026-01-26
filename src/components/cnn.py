# src/components/cnn.py

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Callable
from .activation import FunActivation
from .fcn import FCNet


class CNNPure1d(nn.Module):
    """1D CNN with optimized convolutions"""
    conv_arch: list
    activation: str | Callable = 'Tanh'
    kernel_size: int = 5
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if isinstance(self.activation, str):
            self.act_fn = FunActivation()(self.activation)
        else:
            self.act_fn = self.activation

        # Create conv layers
        self.conv_layers = [
            nn.Conv(
                features=out_channels,
                kernel_size=(self.kernel_size,),
                strides=(self.stride,),
                dtype=self.dtype,
                name=f'conv_{i}'
            )
            for i, (in_channels, out_channels) in enumerate(
                zip(self.conv_arch[:-1], self.conv_arch[1:])
            )
        ]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass

        Args:
            x: Input (batch, length, channels)

        Returns:
            Output (batch, new_length, new_channels)
        """
        for conv in self.conv_layers:
            x = conv(x)
            x = self.act_fn(x)
        return x


class CNNet1d(nn.Module):
    """1D CNN + FC network"""
    conv_arch: list
    fc_arch: list
    activation_conv: str | Callable = 'Tanh'
    activation_fc: str | Callable = 'Tanh'
    kernel_size: int = 5
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_net = CNNPure1d(
            conv_arch=self.conv_arch,
            activation=self.activation_conv,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dtype=self.dtype
        )

        self.fc_net = FCNet(
            layers_list=self.fc_arch,
            activation=self.activation_fc,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass

        Args:
            x: Input (batch, length, channels)

        Returns:
            Output (batch, fc_arch[-1])
        """
        x = self.conv_net(x)
        # Flatten all except batch dimension
        x = x.reshape(x.shape[0], -1)
        x = self.fc_net(x)
        return x


class CNNPure2d(nn.Module):
    """2D CNN with optimized convolutions"""
    conv_arch: list
    activation: str | Callable = 'Tanh'
    kernel_size: Tuple[int, int] = (3, 3)
    stride: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if isinstance(self.activation, str):
            self.act_fn = FunActivation()(self.activation)
        else:
            self.act_fn = self.activation

        # Create conv layers
        self.conv_layers = [
            nn.Conv(
                features=out_channels,
                kernel_size=self.kernel_size,
                strides=(self.stride, self.stride),
                dtype=self.dtype,
                name=f'conv_{i}'
            )
            for i, (in_channels, out_channels) in enumerate(
                zip(self.conv_arch[:-1], self.conv_arch[1:])
            )
        ]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass

        Args:
            x: Input (batch, height, width, channels)

        Returns:
            Output (batch, new_height, new_width, new_channels)
        """
        for conv in self.conv_layers:
            x = conv(x)
            x = self.act_fn(x)
        return x


class CNNet2d(nn.Module):
    """2D CNN + FC network"""
    conv_arch: list
    fc_arch: list
    activation_conv: str | Callable = 'Tanh'
    activation_fc: str | Callable = 'Tanh'
    kernel_size: Tuple[int, int] = (5, 5)
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_net = CNNPure2d(
            conv_arch=self.conv_arch,
            activation=self.activation_conv,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dtype=self.dtype
        )

        self.fc_net = FCNet(
            layers_list=self.fc_arch,
            activation=self.activation_fc,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass

        Args:
            x: Input (batch, height, width, channels)

        Returns:
            Output (batch, fc_arch[-1])
        """
        x = self.conv_net(x)
        # Flatten all except batch dimension
        x = x.reshape(x.shape[0], -1)
        x = self.fc_net(x)
        return x