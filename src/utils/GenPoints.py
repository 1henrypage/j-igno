# src/utils/GenPoints.py
# JAX version with fixes:
# 1. Added `key` parameter to class methods for JIT compatibility
# 2. Optional LHS sampling support
# 3. Fixed all API mismatches

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Optional
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
    """Generate points inside a sphere using mesh - JIT compatible

    Note: Returns padded array with fixed size for JIT compatibility.
    Points outside sphere are masked to (0,0) + xc.
    """
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., num_sample),
        jnp.linspace(-1., 1., num_sample)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)

    # For JIT compatibility, we keep all points but zero out those outside
    # If you need variable-size output, call this outside JIT
    mask = jnp.linalg.norm(grid, axis=1, keepdims=True) < 1.
    x = jnp.where(mask, grid, 0.0)
    return x * radius + xc


def inner_point_sphere_mesh_variable(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """Generate points inside sphere - NOT JIT compatible (variable size output)"""
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

    # Bottom edge (y = lb[1])
    bottom = jnp.stack([x_mesh, jnp.full(num_each_edge, lb[1])], axis=1)
    # Top edge (y = ub[1])
    top = jnp.stack([x_mesh, jnp.full(num_each_edge, ub[1])], axis=1)
    # Left edge (x = lb[0])
    left = jnp.stack([jnp.full(num_each_edge, lb[0]), y_mesh], axis=1)
    # Right edge (x = ub[0])
    right = jnp.stack([jnp.full(num_each_edge, ub[0]), y_mesh], axis=1)

    return jnp.concatenate([bottom, top, left, right], axis=0)


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

    keys = random.split(key, 8)

    # Bottom edge (y = lb[1])
    bottom_x = random.uniform(keys[0], shape=(num_each_edge,), minval=lb[0], maxval=ub[0])
    bottom = jnp.stack([bottom_x, jnp.full(num_each_edge, lb[1])], axis=1)

    # Top edge (y = ub[1])
    top_x = random.uniform(keys[1], shape=(num_each_edge,), minval=lb[0], maxval=ub[0])
    top = jnp.stack([top_x, jnp.full(num_each_edge, ub[1])], axis=1)

    # Left edge (x = lb[0])
    left_y = random.uniform(keys[2], shape=(num_each_edge,), minval=lb[1], maxval=ub[1])
    left = jnp.stack([jnp.full(num_each_edge, lb[0]), left_y], axis=1)

    # Right edge (x = ub[0])
    right_y = random.uniform(keys[3], shape=(num_each_edge,), minval=lb[1], maxval=ub[1])
    right = jnp.stack([jnp.full(num_each_edge, ub[0]), right_y], axis=1)

    return jnp.concatenate([bottom, top, left, right], axis=0)


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
def weight_centers_uniform(
        key: jax.Array,
        n_center: int,
        x_lb: Tuple[float, float],
        x_ub: Tuple[float, float],
        R_max: float = 1e-4,
        R_min: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate centers of compact support regions using UNIFORM sampling - JIT compatible

    Returns:
        xc: size(n_center, 1, 2)
        R: size(n_center, 1, 1)
    """
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    key1, key2 = random.split(key)
    R = random.uniform(key1, shape=(n_center, 1), minval=R_min, maxval=R_max)

    # Adjust bounds based on radius
    lb_adj = lb + R
    ub_adj = ub - R

    # Uniform sampling in adjusted bounds
    xc = random.uniform(key2, shape=(n_center, 2), minval=0., maxval=1.)
    xc = xc * (ub_adj - lb_adj) + lb_adj

    return xc.reshape(-1, 1, 2), R.reshape(-1, 1, 1)


def weight_centers_lhs(
        key: jax.Array,
        n_center: int,
        x_lb: Tuple[float, float],
        x_ub: Tuple[float, float],
        R_max: float = 1e-4,
        R_min: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate centers using Latin Hypercube Sampling - NOT JIT compatible

    LHS provides better space coverage than uniform sampling.
    Use this for better training results, but call outside JIT.

    Returns:
        xc: size(n_center, 1, 2)
        R: size(n_center, 1, 1)
    """
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    key1, key2, key3, key4, key5 = random.split(key, 5)
    R = random.uniform(key1, shape=(n_center, 1), minval=R_min, maxval=R_max)

    # Adjust bounds based on radius
    lb_adj = lb + R
    ub_adj = ub - R

    # Latin Hypercube Sampling implementation
    # Divide each dimension into n_center equal intervals
    # Place one sample in each interval, then shuffle

    # For dimension 0
    intervals_0 = jnp.arange(n_center)
    perm_0 = random.permutation(key2, intervals_0)
    u_0 = random.uniform(key3, shape=(n_center,))
    samples_0 = (perm_0 + u_0) / n_center  # in [0, 1]

    # For dimension 1
    intervals_1 = jnp.arange(n_center)
    perm_1 = random.permutation(key4, intervals_1)
    u_1 = random.uniform(key5, shape=(n_center,))
    samples_1 = (perm_1 + u_1) / n_center  # in [0, 1]

    # Stack and scale to bounds
    xc_unit = jnp.stack([samples_0, samples_1], axis=1)  # (n_center, 2) in [0,1]^2
    xc = xc_unit * (ub_adj - lb_adj) + lb_adj

    return xc.reshape(-1, 1, 2), R.reshape(-1, 1, 1)


# Alias for backward compatibility - defaults to uniform (JIT-compatible)
weight_centers = weight_centers_uniform


@partial(jax.jit, static_argnums=(1,))
def integral_grid(
        key: jax.Array,
        n_mesh: int = 9
) -> jnp.ndarray:
    """Meshgrid for calculating integrals in [-1,1]^2 - JIT compatible

    Note: Returns fixed-size array. Points outside unit circle are zeroed.
    """
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., n_mesh),
        jnp.linspace(-1., 1., n_mesh)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)

    # For JIT compatibility, keep fixed size but mask invalid points
    mask = jnp.linalg.norm(grid, axis=1, keepdims=True) < 1.
    return jnp.where(mask, grid, 0.0)


def integral_grid_variable(
        key: jax.Array,
        n_mesh: int = 9
) -> jnp.ndarray:
    """Meshgrid for integrals - NOT JIT compatible (variable size)"""
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., n_mesh),
        jnp.linspace(-1., 1., n_mesh)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)
    mask = jnp.linalg.norm(grid, axis=1) < 1.
    return grid[mask, :]


######################################## Class-based API (backwards compatibility)
# These classes now wrap the pure functional versions
# FIXED: Added `key` parameter to all methods for JIT compatibility

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
            method: Literal['mesh', 'uniform'] = 'uniform',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Generate points - dispatches to pure functions

        Args:
            num_sample: Number of points to generate
            method: 'mesh' or 'uniform'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'mesh':
            return inner_point_1d_mesh(
                key, num_sample, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            return inner_point_1d_uniform(
                key, num_sample, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")


class Point2D:
    """Wrapper class for 2D point generation - maintains stateful key

    FIXED: All methods now accept optional `key` parameter for JIT compatibility.
    When called from JIT-compiled code, pass the key explicitly.
    """

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
            method: Literal['mesh', 'uniform'] = 'uniform',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Points inside the domain - dispatches to pure functions

        Args:
            num_sample_or_mesh: Number of points or mesh size
            method: 'mesh' or 'uniform'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'mesh':
            return inner_point_2d_mesh(
                key, num_sample_or_mesh, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            return inner_point_2d_uniform(
                key, num_sample_or_mesh, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def inner_point_sphere(
            self,
            num_sample: int,
            xc: jnp.ndarray,
            radius: float,
            method: Literal['muller', 'mesh'] = 'muller',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Points inside a sphere - dispatches to pure functions

        Args:
            num_sample: Number of points
            xc: Center of sphere
            radius: Radius of sphere
            method: 'muller' or 'mesh'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'muller':
            return inner_point_sphere_muller(
                key, num_sample, xc, radius
            ).astype(self.dtype)
        elif method == 'mesh':
            return inner_point_sphere_mesh(
                key, num_sample, xc, radius
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def boundary_point(
            self,
            num_each_edge: int,
            method: Literal['mesh', 'uniform'] = 'uniform',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Points on the boundary - dispatches to pure functions

        Args:
            num_each_edge: Number of points per edge
            method: 'mesh' or 'uniform'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'mesh':
            return boundary_point_2d_mesh(
                key, num_each_edge, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            return boundary_point_2d_uniform(
                key, num_each_edge, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def boundary_point_sphere(
            self,
            num_sample: int,
            xc: jnp.ndarray,
            radius: float,
            method: Literal['muller', 'mesh'] = 'mesh',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Points on sphere surface - dispatches to pure functions

        Args:
            num_sample: Number of points
            xc: Center of sphere
            radius: Radius of sphere
            method: 'muller' or 'mesh'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'muller':
            return boundary_point_sphere_muller(
                key, num_sample, xc, radius
            ).astype(self.dtype)
        elif method == 'mesh':
            return boundary_point_sphere_mesh(
                key, num_sample, xc, radius
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def weight_centers(
            self,
            n_center: int,
            R_max: float = 1e-4,
            R_min: float = 1e-4,
            key: Optional[jax.Array] = None,
            use_lhs: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate centers of compact support regions

        FIXED: Now accepts `key` parameter for JIT compatibility.

        Args:
            n_center: Number of centers
            R_max: Maximum radius
            R_min: Minimum radius
            key: Optional PRNG key. If None, uses internal state.
            use_lhs: If True, use Latin Hypercube Sampling (NOT JIT compatible)
                     If False, use uniform sampling (JIT compatible)

        Returns:
            xc: Centers, shape (n_center, 1, 2)
            R: Radii, shape (n_center, 1, 1)
        """
                # Get key
        if key is None:
            self.key, key = random.split(self.key)

        # Choose sampling method
        if use_lhs:
            xc, R = weight_centers_lhs(key, n_center, self.lb, self.ub, R_max, R_min)
        else:
            xc, R = weight_centers_uniform(key, n_center, self.lb, self.ub, R_max, R_min)

        return xc.astype(self.dtype), R.astype(self.dtype)

    def integral_grid(
            self,
            n_mesh_or_grid: int = 9,
            key: Optional[jax.Array] = None,
            variable_size: bool = True
    ) -> jnp.ndarray:
        """Meshgrid for calculating integrals in [-1,1]^2

        Args:
            n_mesh_or_grid: Mesh size
            key: Optional PRNG key (not used for mesh, but kept for API consistency)
            variable_size: If True, return only points inside unit circle (NOT JIT compatible)
                          If False, return fixed-size array with zeros outside (JIT compatible)
        """
        if key is None:
            key = self.key  # Don't update state since this is deterministic

        if variable_size:
            return integral_grid_variable(key, n_mesh_or_grid).astype(self.dtype)
        else:
            return integral_grid(key, n_mesh_or_grid).astype(self.dtype)

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


if __name__ == '__main__':
    # Test the implementation
    key = random.PRNGKey(42)

    # Test pure functional API
    key, subkey = random.split(key)
    xc, R = weight_centers_uniform(subkey, 100, (0., 0.), (1., 1.), 1e-4, 1e-4)
    print(f"Uniform centers shape: {xc.shape}, R shape: {R.shape}")

    key, subkey = random.split(key)
    xc_lhs, R_lhs = weight_centers_lhs(subkey, 100, (0., 0.), (1., 1.), 1e-4, 1e-4)
    print(f"LHS centers shape: {xc_lhs.shape}, R shape: {R_lhs.shape}")

    # Test class API
    demo = Point2D(random_seed=42)

    # With internal state
    xc1, R1 = demo.weight_centers(n_center=50)
    print(f"Class API (internal key): {xc1.shape}")

    # With explicit key (JIT compatible)
    key, subkey = random.split(key)
    xc2, R2 = demo.weight_centers(n_center=50, key=subkey)
    print(f"Class API (explicit key): {xc2.shape}")

    # Visualize
    grid = demo.integral_grid(n_mesh_or_grid=20, variable_size=True)
    demo.showPoint({
        'uniform_centers': xc.reshape(-1, 2),
        'lhs_centers': xc_lhs.reshape(-1, 2),
        'grid': grid
    }, title='Point Generation Comparison')
