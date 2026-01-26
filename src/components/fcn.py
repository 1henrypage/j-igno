# src/components/fcn.py

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Callable
from .activation import FunActivation


class FCNet(nn.Module):
    """Fully connected network with activation functions

    Optimized with JIT compilation through Flax.
    """
    layers_list: List[int]
    activation: str | Callable = 'Tanh'
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Get activation function
        if isinstance(self.activation, str):
            self.act_fn = FunActivation()(self.activation)
        else:
            self.act_fn = self.activation

        # Create layers
        self.layers = [
            nn.Dense(out_features, dtype=self.dtype, name=f'layer_{i}')
            for i, (in_features, out_features) in enumerate(
                zip(self.layers_list[:-1], self.layers_list[1:])
            )
        ]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # All layers except last get activation
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_fn(x)

        # Last layer without activation
        x = self.layers[-1](x)
        return x