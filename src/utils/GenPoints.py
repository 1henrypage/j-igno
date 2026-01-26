# src/utils/GenPoints.py

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from typing import Literal, Tuple
from functools import partial


######################################## 1d - Pure functional versions

@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_point_1d_mesh(
        key: jax.Array,
        num_sample: int,
        x_lb: float = 0.,
        x_ub: float = 1.
) -> jnp.ndarray:
    """Generate mesh points in 1D - JIT compatible"""
    return jnp.linspace(x_lb, x_ub, num_sample).reshape(-1, 1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_point_1d_uniform(
        key: jax.Array,
        num_sample: int,
        x_lb: float = 0.,
        x_ub: float = 1.
) -> jnp.ndarray:
    """Generate uniform random points in 1D - JIT compatible"""
    return random.uniform(key, shape=(num_sample, 1), minval=x_lb, maxval=x_ub)


######################################## 2d - Pure functional versions

@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_point_2d_mesh(
        key: jax.Array,
        num_sample: int,
        x_lb: Tuple[float, float] = (0., 0.),
        x_ub: Tuple[float, float] = (1., 1.)
) -> jnp.ndarray:
    """Generate mesh points in 2D - JIT compatible"""
    x_mesh = jnp.linspace(x_lb[0], x_ub[0], num_sample)
    y_mesh = jnp.linspace(x_lb[1], x_ub[1], num_sample)
    x_mesh, y_mesh = jnp.meshgrid(x_mesh, y_mesh)
    return jnp.stack([x_mesh.flatten(), y_mesh.flatten()], axis=1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_point_2d_uniform(
        key: jax.Array,
        num_sample: int,
        x_lb: Tuple[float, float] = (0., 0.),
        x_ub: Tuple[float, float] = (1., 1.)
) -> jnp.ndarray:
    """Generate uniform random points in 2D - JIT compatible"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)
    return random.uniform(key, shape=(num_sample, 2), minval=lb, maxval=ub)


@partial(jax.jit, static_argnums=(1, 4))
def inner_point_sphere_muller(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float,
        dim: int = 2
) -> jnp.ndarray:
    """Generate points inside a sphere using Muller method - JIT compatible"""
    key1, key2 = random.split(key)
    x = random.normal(key1, shape=(num_sample, dim))
    r = random.uniform(key2, shape=(num_sample, 1)) ** 0.5
    x = (x * r) / jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True))
    return x * radius + xc


@partial(jax.jit, static_argnums=(1,))
def inner_point_sphere_mesh(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """Generate points inside a sphere using mesh - JIT compatible"""
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., num_sample),
        jnp.linspace(-1., 1., num_sample)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)
    mask = jnp.linalg.norm(grid, axis=1) < 1.
    x = grid[mask, :]
    return x * radius + xc


@partial(jax.jit, static_argnums=(1, 2, 3))
def boundary_point_2d_mesh(
        key: jax.Array,
        num_each_edge: int,
        x_lb: Tuple[float, float] = (0., 0.),
        x_ub: Tuple[float, float] = (1., 1.)
) -> jnp.ndarray:
    """Generate boundary points using mesh - JIT compatible"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    x_mesh = jnp.linspace(lb[0], ub[0], num_each_edge)
    y_mesh = jnp.linspace(lb[1], ub[1], num_each_edge)

    X_bd = []
    for d in range(2):
        X_lb = jnp.stack([x_mesh, y_mesh], axis=1)
        X_ub = jnp.stack([x_mesh, y_mesh], axis=1)
        X_lb = X_lb.at[:, d].set(lb[d])
        X_ub = X_ub.at[:, d].set(ub[d])
        X_bd.append(X_lb)
        X_bd.append(X_ub)

    return jnp.concatenate(X_bd, axis=0)


@partial(jax.jit, static_argnums=(1, 2, 3))
def boundary_point_2d_uniform(
        key: jax.Array,
        num_each_edge: int,
        x_lb: Tuple[float, float] = (0., 0.),
        x_ub: Tuple[float, float] = (1., 1.)
) -> jnp.ndarray:
    """Generate boundary points uniformly - JIT compatible"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    X_bd = []
    for d in range(2):
        key, subkey1, subkey2 = random.split(key, 3)
        X_lb = random.uniform(subkey1, shape=(num_each_edge, 2), minval=lb, maxval=ub)
        X_ub = random.uniform(subkey2, shape=(num_each_edge, 2), minval=lb, maxval=ub)
        X_lb = X_lb.at[:, d].set(lb[d])
        X_ub = X_ub.at[:, d].set(ub[d])
        X_bd.append(X_lb)
        X_bd.append(X_ub)

    return jnp.concatenate(X_bd, axis=0)


@partial(jax.jit, static_argnums=(1,))
def boundary_point_sphere_muller(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """Generate points on sphere surface using Muller - JIT compatible"""
    x = random.normal(key, shape=(num_sample, 2))
    x = x / jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True))
    return x * radius + xc


@partial(jax.jit, static_argnums=(1,))
def boundary_point_sphere_mesh(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """Generate points on sphere surface using mesh - JIT compatible"""
    theta = jnp.linspace(0., 2. * jnp.pi, num_sample)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    return x * radius + xc


@partial(jax.jit, static_argnums=(1, 4, 5))
def weight_centers(
        key: jax.Array,
        n_center: int,
        x_lb: Tuple[float, float],
        x_ub: Tuple[float, float],
        R_max: float = 1e-4,
        R_min: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate centers of compact support regions - JIT compatible

    Returns:
        xc: size(n_center, 1, 2)
        R: size(n_center, 1, 1)
    """
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    key1, key2 = random.split(key)
    R = random.uniform(key1, shape=(n_center, 1), minval=R_min, maxval=R_max)

    # Generate centers uniformly in adjusted bounds
    lb_adj = lb + R
    ub_adj = ub - R
    xc = random.uniform(key2, shape=(n_center, 2), minval=lb_adj, maxval=ub_adj)

    return xc.reshape(-1, 1, 2), R.reshape(-1, 1, 1)


@partial(jax.jit, static_argnums=(1,))
def integral_grid(
        key: jax.Array,
        n_mesh: int = 9
) -> jnp.ndarray:
    """Meshgrid for calculating integrals in [-1,1]^2 - JIT compatible"""
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., n_mesh),
        jnp.linspace(-1., 1., n_mesh)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)
    mask = jnp.linalg.norm(grid, axis=1) < 1.
    return grid[mask, :]


######################################## Class-based API (backwards compatibility)
# These classes now wrap the pure functional versions

class Point1D:
    """Wrapper class for 1D point generation - maintains stateful key"""

    def __init__(
            self,
            x_lb: float = 0.,
            x_ub: float = 1.,
            dataType=jnp.float32,
            random_seed: int | None = None
    ):
        self.lb = x_lb
        self.ub = x_ub
        self.dtype = dataType
        self.key = random.PRNGKey(random_seed if random_seed is not None else 0)

    def inner_point(
            self,
            num_sample: int = 100,
            method: Literal['mesh', 'uniform'] = 'uniform'
    ) -> jnp.ndarray:
        """Generate points - dispatches to pure functions"""
        if method == 'mesh':
            return inner_point_1d_mesh(
                self.key, num_sample, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            self.key, subkey = random.split(self.key)
            return inner_point_1d_uniform(
                subkey, num_sample, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError


class Point2D:
    """Wrapper class for 2D point generation - maintains stateful key"""

    def __init__(
            self,
            x_lb: list[float] = [0., 0.],
            x_ub: list[float] = [1., 1.],
            dataType=jnp.float32,
            random_seed: int | None = None
    ):
        self.lb = tuple(x_lb)
        self.ub = tuple(x_ub)
        self.dtype = dataType
        self.key = random.PRNGKey(random_seed if random_seed is not None else 0)

    def inner_point(
            self,
            num_sample_or_mesh: int,
            method: Literal['mesh', 'uniform'] = 'uniform'
    ) -> jnp.ndarray:
        """Points inside the domain - dispatches to pure functions"""
        if method == 'mesh':
            return inner_point_2d_mesh(
                self.key, num_sample_or_mesh, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            self.key, subkey = random.split(self.key)
            return inner_point_2d_uniform(
                subkey, num_sample_or_mesh, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError

    def inner_point_sphere(
            self,
            num_sample: int,
            xc: jnp.ndarray,
            radius: float,
            method: Literal['muller', 'mesh'] = 'muller'
    ) -> jnp.ndarray:
        """Points inside a sphere - dispatches to pure functions"""
        self.key, subkey = random.split(self.key)
        if method == 'muller':
            return inner_point_sphere_muller(
                subkey, num_sample, xc, radius
            ).astype(self.dtype)
        elif method == 'mesh':
            return inner_point_sphere_mesh(
                subkey, num_sample, xc, radius
            ).astype(self.dtype)
        else:
            raise NotImplementedError

    def boundary_point(
            self,
            num_each_edge: int,
            method: Literal['mesh', 'uniform'] = 'uniform'
    ) -> jnp.ndarray:
        """Points on the boundary - dispatches to pure functions"""
        if method == 'mesh':
            return boundary_point_2d_mesh(
                self.key, num_each_edge, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            self.key, subkey = random.split(self.key)
            return boundary_point_2d_uniform(
                subkey, num_each_edge, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError

    def boundary_point_sphere(
            self,
            num_sample: int,
            xc: jnp.ndarray,
            radius: float,
            method: Literal['muller', 'mesh'] = 'mesh'
    ) -> jnp.ndarray:
        """Points on sphere surface - dispatches to pure functions"""
        self.key, subkey = random.split(self.key)
        if method == 'muller':
            return boundary_point_sphere_muller(
                subkey, num_sample, xc, radius
            ).astype(self.dtype)
        elif method == 'mesh':
            return boundary_point_sphere_mesh(
                subkey, num_sample, xc, radius
            ).astype(self.dtype)
        else:
            raise NotImplementedError

    def weight_centers(
            self,
            n_center: int,
            R_max: float = 1e-4,
            R_min: float = 1e-4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate centers - dispatches to pure function"""
        # Validation
        if R_max < R_min:
            raise ValueError('R_max should be larger than R_min.')
        lb_arr = jnp.array(self.lb)
        ub_arr = jnp.array(self.ub)
        if (2. * R_max) > jnp.min(ub_arr - lb_arr):
            raise ValueError('R_max is too large.')
        if (R_min) < 1e-4 and self.dtype is jnp.float32:
            raise ValueError('R_min<1e-4 when data_type is jnp.float32!')
        if (R_min) < 1e-10 and self.dtype is jnp.float64:
            raise ValueError('R_min<1e-10 when data_type is jnp.float64!')

        self.key, subkey = random.split(self.key)
        xc, R = weight_centers(subkey, n_center, self.lb, self.ub, R_max, R_min)
        return xc.astype(self.dtype), R.astype(self.dtype)

    def integral_grid(self, n_mesh_or_grid: int = 9) -> jnp.ndarray:
        """Meshgrid for integrals - dispatches to pure function"""
        return integral_grid(self.key, n_mesh_or_grid).astype(self.dtype)

    def showPoint(self, point_dict: dict, title: str = ''):
        """Visualize the generated points"""
        fig = plt.figure(figsize=(9, 6))
        for key in point_dict.keys():
            x = point_dict[key][:, 0]
            y = point_dict[key][:, 1]
            plt.scatter(x, y, label=key)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        plt.show()
