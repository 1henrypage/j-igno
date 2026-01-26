# src/components/mon.py
# Multi-Output DeepONet architectures converted to JAX/Flax
# Optimized with vmap and jit

import jax
import jax.numpy as jnp
from jax import vmap
from flax import linen as nn
from typing import Callable, List
from functools import partial

from .activation import FunActivation


class MultiONetBatch(nn.Module):
    """Multi-Output Operator Network with batch processing

    Implements the DeepONet architecture with trunk and branch networks.
    Optimized with vmap for parallel computation.
    """
    in_size_x: int
    in_size_a: int
    trunk_layers: List[int]
    branch_layers: List[int]
    activation_trunk: str | Callable = 'SiLU_Sin'
    activation_branch: str | Callable = 'SiLU'
    sum_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.sum_layers < len(self.branch_layers)
        self.l = self.sum_layers

        # Activation functions
        if isinstance(self.activation_trunk, str):
            self.act_trunk = FunActivation()(self.activation_trunk)
        else:
            self.act_trunk = self.activation_trunk

        if isinstance(self.activation_branch, str):
            self.act_branch = FunActivation()(self.activation_branch)
        else:
            self.act_branch = self.activation_branch

        # Trunk network layers
        self.fc_trunk_in = nn.Dense(self.trunk_layers[0], dtype=self.dtype, name='trunk_in')
        self.trunk_net = [
            nn.Dense(hidden, dtype=self.dtype, name=f'trunk_{i}')
            for i, hidden in enumerate(self.trunk_layers[1:])
        ]

        # Branch network layers
        self.fc_branch_in = nn.Dense(self.branch_layers[0], dtype=self.dtype, name='branch_in')
        self.branch_net = [
            nn.Dense(hidden, dtype=self.dtype, name=f'branch_{i}')
            for i, hidden in enumerate(self.branch_layers[1:])
        ]

        # Learnable weights and bias for final layers
        # These will be initialized in the first call
        self.w_init = lambda key, shape: jnp.zeros(shape)
        self.b_init = lambda key, shape: jnp.zeros(shape)

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        """Forward pass

        Args:
            x: Spatial coordinates (n_batch, n_mesh, dx)
            a: Latent/branch input (n_batch, latent_size)

        Returns:
            Output field (n_batch, n_mesh)
        """
        assert x.shape[0] == a.shape[0], "Batch sizes must match"

        # Initialize weights and bias
        w = [self.param(f'w_{i}', self.w_init, (1,)) for i in range(self.l)]
        b = self.param('b', self.b_init, (1,))

        # Trunk network - operates on spatial coordinates
        # (n_batch, n_mesh, dx) -> (n_batch, n_mesh, hidden)
        x_trunk = self.act_trunk(self.fc_trunk_in(x))

        # Branch network - operates on latent codes
        # (n_batch, latent_size) -> (n_batch, hidden)
        a_branch = self.act_branch(self.fc_branch_in(a))

        # Process through intermediate layers
        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))

        # Sum over final layers with learnable weights
        out = 0.
        for net_t, net_b, weight in zip(
                self.trunk_net[-self.l:],
                self.branch_net[-self.l:],
                w
        ):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))
            # Einstein summation: (batch, n_mesh, hidden) * (batch, hidden) -> (batch, n_mesh)
            out = out + jnp.einsum('bnh,bh->bn', x_trunk, a_branch) * weight[0]

        # Average and add bias
        out = out / self.l + b[0]

        return out


class MultiONetBatch_X(nn.Module):
    """Multi-Input & Multi-Output Operator Network

    Extended version for multiple input/output channels.
    """
    in_size_x: int
    in_size_a: int
    latent_size: int
    out_size: int
    trunk_layers: List[int]
    branch_layers: List[int]
    activation_trunk: str | Callable = 'SiLU_Sin'
    activation_branch: str | Callable = 'SiLU'
    sum_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.sum_layers < len(self.branch_layers)
        self.l = self.sum_layers

        # Activation functions
        if isinstance(self.activation_trunk, str):
            self.act_trunk = FunActivation()(self.activation_trunk)
        else:
            self.act_trunk = self.activation_trunk

        if isinstance(self.activation_branch, str):
            self.act_branch = FunActivation()(self.activation_branch)
        else:
            self.act_branch = self.activation_branch

        # Trunk network
        self.fc_trunk_in = nn.Dense(self.trunk_layers[0], dtype=self.dtype, name='trunk_in')
        self.trunk_net = [
            nn.Dense(hidden, dtype=self.dtype, name=f'trunk_{i}')
            for i, hidden in enumerate(self.trunk_layers[1:])
        ]

        # Branch network
        self.fc_branch_in = nn.Dense(self.branch_layers[0], dtype=self.dtype, name='branch_in')
        self.branch_net = [
            nn.Dense(hidden, dtype=self.dtype, name=f'branch_{i}')
            for i, hidden in enumerate(self.branch_layers[1:])
        ]

        # Output layer
        self.fc_out = nn.Dense(self.out_size, dtype=self.dtype, name='out')

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        """Forward pass

        Args:
            x: Spatial coordinates (n_batch, n_mesh, dx)
            a: Latent input (n_batch, latent_size, da)

        Returns:
            Output field (n_batch, n_mesh, out_size)
        """
        assert x.shape[0] == a.shape[0], "Batch sizes must match"

        # Trunk network
        x_trunk = self.act_trunk(self.fc_trunk_in(x))

        # Branch network
        a_branch = self.act_branch(self.fc_branch_in(a))

        # Process through intermediate layers
        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))

        # Sum over final layers
        out = 0.
        for net_t, net_b in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))
            # (batch, n_mesh, hidden) * (batch, latent_size, hidden) -> (batch, n_mesh, latent_size)
            out = out + jnp.einsum('bnh,bmh->bnm', x_trunk, a_branch)

        # Output layer: (batch, n_mesh, latent_size) -> (batch, n_mesh, out_size)
        out = self.fc_out(out / self.l)

        return out


class MultiONetCartesianProd(nn.Module):
    """Multi-Output Operator Network with Cartesian product structure

    More efficient for cases where trunk and branch can be evaluated separately
    and combined via outer product.
    """
    in_size_x: int
    in_size_a: int
    trunk_layers: List[int]
    branch_layers: List[int]
    activation_trunk: str | Callable = 'SiLU_Sin'
    activation_branch: str | Callable = 'SiLU'
    sum_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.sum_layers < len(self.branch_layers)
        self.l = self.sum_layers

        # Activation functions
        if isinstance(self.activation_trunk, str):
            self.act_trunk = FunActivation()(self.activation_trunk)
        else:
            self.act_trunk = self.activation_trunk

        if isinstance(self.activation_branch, str):
            self.act_branch = FunActivation()(self.activation_branch)
        else:
            self.act_branch = self.activation_branch

        # Trunk network
        self.fc_trunk_in = nn.Dense(self.trunk_layers[0], dtype=self.dtype, name='trunk_in')
        self.trunk_net = [
            nn.Dense(hidden, dtype=self.dtype, name=f'trunk_{i}')
            for i, hidden in enumerate(self.trunk_layers[1:])
        ]

        # Branch network
        self.fc_branch_in = nn.Dense(self.branch_layers[0], dtype=self.dtype, name='branch_in')
        self.branch_net = [
            nn.Dense(hidden, dtype=self.dtype, name=f'branch_{i}')
            for i, hidden in enumerate(self.branch_layers[1:])
        ]

        self.w_init = lambda key, shape: jnp.zeros(shape)
        self.b_init = lambda key, shape: jnp.zeros(shape)

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        """Forward pass with Cartesian product

        Args:
            x: Spatial coordinates (mesh_size, dx) - NO batch dimension
            a: Latent input (n_batch, latent_size)

        Returns:
            Output field (n_batch, mesh_size)
        """
        # Initialize weights and bias
        w = [self.param(f'w_{i}', self.w_init, (1,)) for i in range(self.l)]
        b = self.param('b', self.b_init, (1,))

        # Trunk network - NO batch dimension
        # (mesh_size, dx) -> (mesh_size, hidden)
        x_trunk = self.act_trunk(self.fc_trunk_in(x))

        # Branch network - HAS batch dimension
        # (n_batch, latent_size) -> (n_batch, hidden)
        a_branch = self.act_branch(self.fc_branch_in(a))

        # Process through intermediate layers
        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))

        # Sum over final layers - Cartesian product
        out = 0.
        for net_t, net_b, weight in zip(
                self.trunk_net[-self.l:],
                self.branch_net[-self.l:],
                w
        ):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))
            # (batch, hidden) * (mesh, hidden) -> (batch, mesh)
            out = out + jnp.einsum('bh,mh->bm', a_branch, x_trunk) * weight[0]

        # Average and add bias
        out = out / self.l + b[0]

        return out


class MultiONetCartesianProd_X(nn.Module):
    """Multi-Input & Multi-Output with Cartesian product structure"""
    in_size_x: int
    in_size_a: int
    latent_size: int
    out_size: int
    trunk_layers: List[int]
    branch_layers: List[int]
    activation_trunk: str | Callable = 'SiLU_Sin'
    activation_branch: str | Callable = 'SiLU'
    sum_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.sum_layers < len(self.branch_layers)
        self.l = self.sum_layers

        # Activation functions
        if isinstance(self.activation_trunk, str):
            self.act_trunk = FunActivation()(self.activation_trunk)
        else:
            self.act_trunk = self.activation_trunk

        if isinstance(self.activation_branch, str):
            self.act_branch = FunActivation()(self.activation_branch)
        else:
            self.act_branch = self.activation_branch

        # Trunk network
        self.fc_trunk_in = nn.Dense(self.trunk_layers[0], dtype=self.dtype, name='trunk_in')
        self.trunk_net = [
            nn.Dense(hidden, dtype=self.dtype, name=f'trunk_{i}')
            for i, hidden in enumerate(self.trunk_layers[1:])
        ]

        # Branch network
        self.fc_branch_in = nn.Dense(self.branch_layers[0], dtype=self.dtype, name='branch_in')
        self.branch_net = [
            nn.Dense(hidden, dtype=self.dtype, name=f'branch_{i}')
            for i, hidden in enumerate(self.branch_layers[1:])
        ]

        # Output layer
        self.fc_out = nn.Dense(self.out_size, dtype=self.dtype, name='out')

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        """Forward pass with Cartesian product

        Args:
            x: Spatial coordinates (mesh_size, dx) - NO batch dimension
            a: Latent input (n_batch, latent_size, da)

        Returns:
            Output field (n_batch, mesh_size, out_size)
        """
        # Trunk network - NO batch dimension
        x_trunk = self.act_trunk(self.fc_trunk_in(x))

        # Branch network - HAS batch dimension
        a_branch = self.act_branch(self.fc_branch_in(a))

        # Process through intermediate layers
        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))

        # Sum over final layers - Cartesian product
        out = 0.
        for net_t, net_b in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))
            # (batch, latent_size, hidden) * (mesh, hidden) -> (batch, mesh, latent_size)
            out = out + jnp.einsum('bmh,nh->bnm', a_branch, x_trunk)

        # Output layer: (batch, mesh, latent_size) -> (batch, mesh, out_size)
        out = self.fc_out(out / self.l)

        return out


# ============================================================================
# Optimized versions with explicit vmap for trunk network
# ============================================================================

def create_vmapped_trunk_forward(trunk_net, act_fn):
    """Create a vmapped version of trunk network forward pass

    This allows us to efficiently process batched spatial coordinates.
    """

    def trunk_forward(x):
        """Process single spatial coordinate"""
        for layer in trunk_net:
            x = act_fn(layer(x))
        return x

    # Vmap over the mesh dimension (axis 0)
    return vmap(trunk_forward, in_axes=0)


def create_vmapped_branch_forward(branch_net, act_fn):
    """Create a vmapped version of branch network forward pass

    This allows us to efficiently process batched latent codes.
    """

    def branch_forward(a):
        """Process single latent code"""
        for layer in branch_net:
            a = act_fn(layer(a))
        return a

    # Vmap over the batch dimension (axis 0)
    return vmap(branch_forward, in_axes=0)