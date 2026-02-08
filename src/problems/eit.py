# src/problems/eit.py

"""
Electrical Impedance Tomography (EIT) problem in JAX.

PDE: -div(a * grad(u)) = 0  (NO source term)

Key differences from Darcy:
- Coefficient is analytic: a(x,y) = Sum_k{c_mask[k] * exp(c_val[k] * sin(k*π*x) * sin(k*π*y))}
- Boundary condition encoding: 20 one-hot patterns (g_l ∈ {1,...,20})
- EIT-specific mollifier: u = u_raw * sin(πx)*sin(πy) + cos(2π*(x*cos(θ) + y*sin(θ)))
- Training data has NO u_sol (only a, boundary condition g_l)
- Beta has 28 dims: 8 from coefficient + 20 from one-hot g_l
- Normalizing flow is only on the 8-dim coefficient latent space
"""

import jax
import jax.numpy as jnp
import h5py
from jax import random, grad, jit, vmap, value_and_grad, lax
from flax import linen as nn
import numpy as np
from functools import partial
from typing import Dict, List, Literal, Tuple, Any, Optional
from pathlib import Path

from src.components.encoder import EncoderCNNet2dTanh
from src.components.nf import RealNVP
from src.problems import ProblemInstance, register_problem
from src.utils.GenPoints import Point2D
from src.utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from src.utils.misc_utils import np2jax
from src.utils.solver_utils import get_model


# ============================================================================
# Pure functions for EIT-specific computations (JIT-compilable)
# ============================================================================

@jit
def one_hot_g_l(g_l: jax.Array) -> jax.Array:
    """One-hot encode boundary condition label

    Args:
        g_l: Boundary condition labels (batch, 1) with values in {1, ..., 20}

    Returns:
        One-hot encoding (batch, 20)
    """
    # Convert from 1-indexed to 0-indexed
    g_l_idx = (g_l.squeeze(-1).astype(jnp.int32) - 1)
    return jax.nn.one_hot(g_l_idx, 20)


@jit
def compute_analytic_a(
        x: jax.Array,
        coe_val: jax.Array,
        coe_mask: jax.Array
) -> jax.Array:
    """Compute coefficient analytically from sinusoidal basis

    a(x,y) = Sum_{k=1}^4 { c_mask[k] * exp(c_val[k] * sin(k*π*x) * sin(k*π*y)) }

    Args:
        x: Coordinates (n_points, 2) or (batch, n_points, 2)
        coe_val: Coefficient values (4,) or (batch, 4)
        coe_mask: Coefficient masks (4,) or (batch, 4)

    Returns:
        Coefficient values (n_points, 1) or (batch, n_points, 1)
    """
    pi = jnp.pi

    # Handle both batched and unbatched cases
    if x.ndim == 2:
        # Unbatched: x (n_points, 2), coe_val (4,), coe_mask (4,)
        x_coord = x[:, 0]  # (n_points,)
        y_coord = x[:, 1]  # (n_points,)

        a = jnp.zeros_like(x_coord)  # (n_points,)
        for k in range(1, 5):
            term = jnp.exp(
                coe_val[k-1] * jnp.sin(k * pi * x_coord) * jnp.sin(k * pi * y_coord)
            )
            a = a + coe_mask[k-1] * term

        return a[..., None]  # (n_points, 1)

    else:
        # Batched: x (batch, n_points, 2), coe_val (batch, 4), coe_mask (batch, 4)
        x_coord = x[..., 0]  # (batch, n_points)
        y_coord = x[..., 1]  # (batch, n_points)

        a = jnp.zeros_like(x_coord)  # (batch, n_points)
        for k in range(1, 5):
            term = jnp.exp(
                coe_val[:, k-1:k] * jnp.sin(k * pi * x_coord) * jnp.sin(k * pi * y_coord)
            )
            a = a + coe_mask[:, k-1:k] * term

        return a[..., None]  # (batch, n_points, 1)


@jit
def eit_boundary_function(x: jax.Array, g_l_scalar: float) -> jax.Array:
    """Compute EIT boundary condition function g(x, y)

    g(x, y) = cos(2π * (x*cos(θ) + y*sin(θ)))
    where θ = π * l / 20

    Args:
        x: Coordinates (..., 2)
        g_l_scalar: Boundary condition index (scalar, 1-indexed)

    Returns:
        Boundary values (...)
    """
    theta = jnp.pi * g_l_scalar / 20.0
    proj = x[..., 0] * jnp.cos(theta) + x[..., 1] * jnp.sin(theta)
    return jnp.cos(2.0 * jnp.pi * proj)


@jit
def mollifier_eit(u_raw: jax.Array, x: jax.Array, g_l: jax.Array) -> jax.Array:
    """Apply EIT mollifier with boundary condition

    u = u_raw * sin(πx) * sin(πy) + g(x, y)

    Args:
        u_raw: Raw network output (batch, n_points) or (n_points,)
        x: Coordinates (batch, n_points, 2) or (n_points, 2)
        g_l: Boundary condition labels (batch, 1) or scalar

    Returns:
        Mollified solution (..., 1)
    """
    pi = jnp.pi
    sin_term = jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1])

    # Handle batched vs unbatched g_l
    if jnp.ndim(g_l) == 2:
        # Batched: g_l (batch, 1)
        def apply_boundary(i):
            g_l_scalar = g_l[i, 0]
            g_vals = eit_boundary_function(x[i], g_l_scalar)
            return u_raw[i] * sin_term[i] + g_vals

        result = vmap(lambda i: apply_boundary(i))(jnp.arange(g_l.shape[0]))
    else:
        # Unbatched: g_l is scalar
        g_vals = eit_boundary_function(x, g_l)
        result = u_raw * sin_term + g_vals

    return result[..., None]


def compute_u_and_grad_eit(
        params_u: Any,
        model_u: nn.Module,
        x: jax.Array,
        beta: jax.Array,
        g_l_scalar: float
) -> Tuple[jax.Array, jax.Array]:
    """Compute u and its gradient w.r.t. x for EIT (per-point version)

    The boundary condition g(x,y) depends on x, so we must compute it
    INSIDE the per-point function for correct autodiff.

    Args:
        params_u: Parameters for u network
        model_u: U network module
        x: Coordinates (n_points, 2)
        beta: Latent code (latent_dim,) - includes both beta_a and one-hot g_l
        g_l_scalar: Boundary condition scalar (for computing g)

    Returns:
        u: Solution values (n_points,)
        du_dx: Gradients (n_points, 2)
    """

    def u_at_point(x_single):
        """Evaluate u at a single point with EIT mollifier"""
        x_batch = x_single[None, None, :]  # (1, 1, 2)
        beta_batch = beta[None, :]  # (1, latent_dim)

        u_val = model_u.apply({'params': params_u}, x_batch, beta_batch)
        u_val = u_val[0, 0]

        # Apply EIT mollifier
        pi = jnp.pi
        sin_term = jnp.sin(pi * x_single[0]) * jnp.sin(pi * x_single[1])
        g_val = eit_boundary_function(x_single, g_l_scalar)

        return u_val * sin_term + g_val

    def value_and_grad_at_point(x_single):
        val = u_at_point(x_single)
        grad_val = jax.grad(u_at_point)(x_single)
        return val, grad_val

    u_vals, du_vals = vmap(value_and_grad_at_point)(x)
    return u_vals, du_vals


def compute_pde_residual_eit_single_sample(
        params_u: Any,
        model_u: nn.Module,
        beta_u: jax.Array,
        g_l_scalar: float,
        xc: jax.Array,
        R: jax.Array,
        int_grid: jax.Array,
        v: jax.Array,
        dv_dr: jax.Array,
        a_vals: jax.Array,
        n_grid: int
) -> jax.Array:
    """Compute PDE residual for single EIT sample

    Weak formulation: ∫ a * grad(u) · grad(v) dx = 0 (no source term!)

    Args:
        params_u: Parameters for u network
        model_u: U network module
        beta_u: Latent code (28-dim: 8 from a + 20 from g_l)
        g_l_scalar: Boundary condition scalar
        xc, R, int_grid, v, dv_dr: Test function parameters
        a_vals: Coefficient values (nc*n_grid, 1) with stop_gradient
        n_grid: Number of grid points per center

    Returns:
        residual: PDE residual (nc,)
    """
    nc = xc.shape[0]

    # Transform integration grid to physical space
    x = int_grid[None, :, :] * R + xc  # (nc, n_grid, 2)
    x_flat = x.reshape(-1, 2)  # (nc*n_grid, 2)

    # Compute test function gradients in physical space
    dv = (dv_dr[None, :, :] / R).reshape(-1, 2)  # (nc*n_grid, 2)
    v_flat = jnp.tile(v, (nc, 1, 1)).reshape(-1, 1)  # (nc*n_grid, 1)

    # Compute u and grad(u) at all collocation points
    def compute_for_center(center_idx):
        x_center = x[center_idx]
        u_vals, du_vals = compute_u_and_grad_eit(params_u, model_u, x_center, beta_u, g_l_scalar)
        return u_vals, du_vals

    u_all, du_all = vmap(compute_for_center)(jnp.arange(nc))
    du_flat = du_all.reshape(-1, 2)  # (nc*n_grid, 2)

    # EIT residual: ∫ a * grad(u) · grad(v) dx = 0
    # (no source term!)
    integrand = jnp.sum(a_vals * du_flat * dv, axis=-1)  # (nc*n_grid,)
    residual = integrand.reshape(nc, n_grid).mean(axis=-1)  # (nc,)

    return residual


@register_problem("eit")
class EIT(ProblemInstance):
    """Electrical Impedance Tomography problem in JAX"""

    # Model hyperparameters
    BETA_SIZE_A = 8  # Latent dim for coefficient
    BETA_SIZE_G = 20  # One-hot dim for boundary condition
    BETA_SIZE_U = 28  # Total: 8 + 20
    HIDDEN_SIZE = 100
    NF_NUM_FLOWS = 3
    NF_HIDDEN_DIM = 56

    def __init__(
            self,
            seed: int,
            dtype: jnp.dtype = jnp.float32,
            train_data_path: str = None,
            test_data_path: str = None,
    ):
        super().__init__(
            seed=seed,
            dtype=dtype,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
        )

        # Current boundary condition (for inversion)
        self._current_g_l = None

        # Load data
        print("Loading data...")
        if self.train_data_path:
            self.train_data, self.gridx_train = self._load_data(self.train_data_path, is_train=True)
            print(f"  Train: a={self.train_data['a'].shape}, g_l={self.train_data['g_l'].shape}")

        if self.test_data_path:
            self.test_data, self.gridx_test = self._load_data(self.test_data_path, is_train=False)
            print(f"  Test: a={self.test_data['a'].shape}, u={self.test_data.get('u', 'N/A')}")

        # Setup grids & test functions
        print("Setting up grids and test functions...")

        self.genPoint = Point2D(
            x_lb=[0., 0.],
            x_ub=[1., 1.],
            random_seed=self.seed
        )

        int_grid, v, dv_dr = TestFun_ParticleWNN(
            fun_type='Wendland',
            dim=2,
            n_mesh_or_grid=9,
        ).get_testFun()

        self.int_grid = int_grid
        self.v = v
        self.dv_dr = dv_dr
        self.n_grid = int_grid.shape[0]

        print(f"  int_grid: {self.int_grid.shape}, v: {self.v.shape}")

        # Build models
        print("Building models...")
        self.models = self._build_models()

        print("Problem initialized (parameters not yet initialized)")

    def _load_data(self, path: str, is_train: bool) -> Tuple[Dict, jax.Array]:
        """Load data from mat file

        Training data: coe_val, coe_mask, g_l (NO u_sol, NO X/Y)
        Test data: coe_val, coe_mask, g_l, u_sol, X, Y
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        data = h5py.File(path, mode='r')

        # Load coefficient parameters
        coe_val = np2jax(np.array(data["coe_val"]).T, self.dtype)  # (N, 4)
        coe_mask = np2jax(np.array(data["coe_mask"]).T, self.dtype)  # (N, 4)
        g_l = np2jax(np.array(data["g_l"]).T, self.dtype)  # (N, 1)

        # Try to load mesh
        if 'X' in data and 'Y' in data:
            X, Y = np.array(data['X']).T, np.array(data['Y']).T
            mesh = np2jax(np.vstack([X.ravel(), Y.ravel()]).T, self.dtype)
            gridx = mesh.reshape(-1, 2)
        else:
            # Generate default 32x32 grid
            print(f"  Generating default 32x32 grid")
            x_1d = np.linspace(0, 1, 32)
            y_1d = np.linspace(0, 1, 32)
            X, Y = np.meshgrid(x_1d, y_1d)
            mesh = np2jax(np.vstack([X.ravel(), Y.ravel()]).T, self.dtype)
            gridx = mesh.reshape(-1, 2)

        ndata = coe_val.shape[0]

        # Compute analytic coefficient at grid points
        x_grid = jnp.tile(gridx[None, :, :], (ndata, 1, 1))  # (N, n_points, 2)
        a = compute_analytic_a(x_grid, coe_val, coe_mask)  # (N, n_points, 1)

        result_dict = {
            'a': a,
            'x': x_grid,
            'g_l': g_l,
            'coe_val': coe_val,
            'coe_mask': coe_mask,
        }

        # Load u_sol if available (test data)
        if 'u_sol' in data or 'sol' in data:
            u_key = 'u_sol' if 'u_sol' in data else 'sol'
            u = np2jax(np.array(data[u_key]).T, self.dtype)
            u = u.reshape(ndata, -1, 1)
            result_dict['u'] = u

        return result_dict, gridx

    def _build_models(self) -> Dict[str, nn.Module]:
        """Build all models"""
        net_type = 'MultiONetBatch'

        # Encoder for coefficient (CNN)
        conv_arch = [1, 64, 64, 64, 64]
        fc_arch = [64, 64, self.BETA_SIZE_A]
        model_enc = EncoderCNNet2dTanh(
            conv_arch=conv_arch,
            fc_arch=fc_arch,
            activation_conv='SiLU',
            activation_fc='SiLU',
            nx_size=32,
            ny_size=32,
            kernel_size=(3, 3),
            stride=2,
            dtype=self.dtype
        )

        # Decoder for u (beta_in=28: 8 from a + 20 from g_l)
        trunk_layers = [self.HIDDEN_SIZE] * 5
        branch_layers = [self.HIDDEN_SIZE] * 5

        model_u = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE_U,  # 28
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=4,
            dtype=self.dtype
        )

        # Decoder for a (beta_in=8: only coefficient latent)
        model_a = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE_A,  # 8
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=4,
            dtype=self.dtype
        )

        # Normalizing flow (only on 8-dim coefficient space)
        model_nf = RealNVP(
            dim=self.BETA_SIZE_A,  # 8
            num_flows=self.NF_NUM_FLOWS,
            hidden_dim=self.NF_HIDDEN_DIM,
        )

        return {
            'enc': model_enc,
            'u': model_u,
            'a': model_a,
            'nf': model_nf,
        }

    # =========================================================================
    # Model configuration methods for trainer
    # =========================================================================

    def get_sample_inputs(self, batch_size: int) -> Dict[str, Dict[str, jax.Array]]:
        """Return sample inputs for each model for initialization"""
        if self.train_data:
            sample_a = self.train_data['a'][:batch_size]
            sample_x = self.train_data['x'][:batch_size]
        else:
            sample_a = self.test_data['a'][:batch_size]
            sample_x = self.test_data['x'][:batch_size]

        sample_beta_a = jnp.ones((batch_size, self.BETA_SIZE_A), dtype=self.dtype)
        sample_beta_u = jnp.ones((batch_size, self.BETA_SIZE_U), dtype=self.dtype)

        return {
            'enc': {'x': sample_a},
            'u': {'x': sample_x, 'a': sample_beta_u},
            'a': {'x': sample_x, 'a': sample_beta_a},
            'nf': {'x': sample_beta_a},
        }

    def get_weight_decay_groups(self) -> Dict[str, bool]:
        """Return dict mapping model name -> whether it should have weight decay"""
        return {
            'enc': False,
            'u': True,
            'a': True,
            'nf': False,
        }

    # =========================================================================
    # Training loss methods - OVERRIDE compute_training_losses
    # =========================================================================

    def compute_training_losses(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jax.Array],
            rng: jax.Array,
            loss_weights: Any
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Compute training losses (override to handle EIT specifics)

        Training data has NO u_sol, only coefficient and boundary condition.
        """
        a = batch['a']
        x = batch['x']
        g_l = batch['g_l']
        coe_val = batch['coe_val']
        coe_mask = batch['coe_mask']
        n_batch = a.shape[0]

        # Encode coefficient -> beta_a (8-dim)
        beta_a = self.models['enc'].apply({'params': params['enc']}, a)

        # Combine with one-hot boundary condition -> beta_u (28-dim)
        g_l_onehot = one_hot_g_l(g_l)  # (batch, 20)
        beta_u = jnp.concatenate([beta_a, g_l_onehot], axis=-1)  # (batch, 28)

        # PDE loss
        nc = 100
        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(
            n_center=nc,
            R_max=1e-4,
            R_min=1e-4,
            key=subkey
        )

        x_grid = self.int_grid[None, :, :] * R + xc
        x_flat = x_grid.reshape(-1, 2)

        def compute_residual_for_sample(beta_u_single, g_l_single, coe_val_single, coe_mask_single):
            # Compute analytic coefficient at collocation points
            a_vals = compute_analytic_a(x_flat, coe_val_single, coe_mask_single)
            a_vals = lax.stop_gradient(a_vals)

            return compute_pde_residual_eit_single_sample(
                params['u'],
                self.models['u'],
                beta_u_single,
                g_l_single[0],  # Scalar
                xc, R,
                self.int_grid, self.v, self.dv_dr,
                a_vals,
                self.n_grid
            )

        residuals = vmap(compute_residual_for_sample)(beta_u, g_l, coe_val, coe_mask)
        residuals_squared = residuals ** 2
        mse_loss = jnp.mean(residuals_squared)

        residuals_flat = residuals_squared.reshape(-1)
        residuals_sorted = jnp.sort(residuals_flat)[::-1]
        top_k_loss = jnp.sum(residuals_sorted[:nc * 10])

        loss_pde = mse_loss + top_k_loss

        # Data loss: coefficient reconstruction only (no u during training)
        a_pred = self.models['a'].apply({'params': params['a']}, x, beta_a)
        loss_data = self.get_loss(a_pred, a.squeeze(-1))

        # NF loss
        loss_nf = self.models['nf'].apply(
            {'params': params['nf']},
            lax.stop_gradient(beta_a),
            method=self.models['nf'].loss
        )

        # Total loss
        total_loss = (
            loss_weights.pde * loss_pde +
            loss_weights.data * loss_data +
            loss_nf
        )

        # Metrics
        metrics = {
            'loss': total_loss,
            'loss_pde': loss_pde,
            'loss_data': loss_data,
            'loss_nf': loss_nf,
        }

        return total_loss, metrics

    def compute_eval_metrics(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jax.Array],
            rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Compute evaluation metrics (override to handle EIT specifics)

        During training eval, we only have coefficient (no u_sol).
        """
        a = batch['a']
        x = batch['x']

        # Encode
        beta_a = self.models['enc'].apply({'params': params['enc']}, a)

        # Predict coefficient
        a_pred = self.models['a'].apply({'params': params['a']}, x, beta_a)
        a_pred = a_pred[..., None]

        # Compute error
        a_error = self.get_error(a_pred, a)

        # NF loss
        loss_nf = self.models['nf'].apply(
            {'params': params['nf']},
            lax.stop_gradient(beta_a),
            method=self.models['nf'].loss
        )

        return {
            'a_error': jnp.mean(a_error),
            'loss_nf': loss_nf,
        }

    def loss_pde(
            self,
            params: Dict[str, Any],
            a: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Not used - compute_training_losses is overridden"""
        raise NotImplementedError("Use compute_training_losses instead")

    def loss_data(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Not used - compute_training_losses is overridden"""
        raise NotImplementedError("Use compute_training_losses instead")

    def error(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Not used - compute_eval_metrics is overridden"""
        raise NotImplementedError("Use compute_eval_metrics instead")

    # =========================================================================
    # From-beta methods (for inversion)
    # =========================================================================

    def loss_pde_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """PDE loss from beta (uses decoded coefficient and current g_l)

        Args:
            params: Model parameters
            beta: Latent codes for coefficient (batch, 8)
            rng: PRNG key

        Returns:
            PDE loss (scalar)
        """
        nc = 100
        n_batch = beta.shape[0]

        # Combine beta_a with current g_l
        if self._current_g_l is None:
            raise RuntimeError("Must call prepare_observations first to set _current_g_l")

        g_l_onehot = one_hot_g_l(self._current_g_l)
        beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)

        # Generate collocation points
        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(
            n_center=nc,
            R_max=1e-4,
            R_min=1e-4,
            key=subkey
        )

        x_grid = self.int_grid[None, :, :] * R + xc
        x_flat = x_grid.reshape(-1, 2)
        x_batch = jnp.tile(x_flat[None, :, :], (n_batch, 1, 1))

        # Decode coefficient
        a_decoded = self.models['a'].apply({'params': params['a']}, x_batch, beta)
        a_decoded = a_decoded[..., None]
        a_decoded = lax.stop_gradient(a_decoded)

        def compute_residual_for_sample(beta_u_single, a_single, g_l_single):
            return compute_pde_residual_eit_single_sample(
                params['u'],
                self.models['u'],
                beta_u_single,
                g_l_single[0],
                xc, R,
                self.int_grid, self.v, self.dv_dr,
                a_single,
                self.n_grid
            )

        residuals = vmap(compute_residual_for_sample)(beta_u, a_decoded, self._current_g_l)
        loss_per_sample = jnp.linalg.norm(residuals, axis=1)
        return jnp.mean(loss_per_sample)

    def loss_data_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        """Data loss from beta

        Args:
            params: Model parameters
            beta: Latent codes for coefficient (batch, 8)
            x: Coordinates (batch, n_points, 2)
            target: Target values (batch, n_points, 1)
            target_type: 'a' or 'u'

        Returns:
            Data loss (scalar)
        """
        if target_type == 'a':
            pred = self.models['a'].apply({'params': params['a']}, x, beta)
            target_flat = target.squeeze(-1) if target.ndim == 3 else target
            return self.get_loss(pred, target_flat)

        elif target_type == 'u':
            if self._current_g_l is None:
                raise RuntimeError("Must call prepare_observations first to set _current_g_l")

            g_l_onehot = one_hot_g_l(self._current_g_l)
            beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)

            pred = self.models['u'].apply({'params': params['u']}, x, beta_u)
            pred = pred[..., None] if pred.ndim == 2 else pred
            pred = mollifier_eit(pred.squeeze(-1), x, self._current_g_l)

            # Relative loss per sample
            loss_per_sample = jnp.linalg.norm(pred - target, axis=1) / jnp.linalg.norm(target, axis=1)
            return jnp.mean(loss_per_sample)

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def error_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        """Error from beta"""
        if target_type == 'u':
            if self._current_g_l is None:
                raise RuntimeError("Must call prepare_observations first to set _current_g_l")

            g_l_onehot = one_hot_g_l(self._current_g_l)
            beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)

            pred = self.models['u'].apply({'params': params['u']}, x, beta_u)
            pred = pred[..., None] if pred.ndim == 2 else pred
            pred = mollifier_eit(pred.squeeze(-1), x, self._current_g_l)

        elif target_type == 'a':
            pred = self.models['a'].apply({'params': params['a']}, x, beta)
            if target.ndim == 3 and target.shape[-1] == 1:
                target = target.squeeze(-1)
            pred = pred[..., None] if pred.ndim == 2 else pred

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return self.get_error(pred, target)

    def predict_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Predict u and a from beta

        Args:
            params: Model parameters
            beta: Latent codes for coefficient (batch, 8)
            x: Coordinates (batch, n_points, 2)

        Returns:
            Dict with 'u_pred' and 'a_pred'
        """
        if self._current_g_l is None:
            raise RuntimeError("Must call prepare_observations first to set _current_g_l")

        g_l_onehot = one_hot_g_l(self._current_g_l)
        beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)

        u_pred = self.models['u'].apply({'params': params['u']}, x, beta_u)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        u_pred = mollifier_eit(u_pred.squeeze(-1), x, self._current_g_l)

        a_pred = self.models['a'].apply({'params': params['a']}, x, beta)
        a_pred = a_pred[..., None]

        return {
            'u_pred': u_pred,
            'a_pred': a_pred,
        }

    # =========================================================================
    # NF-related methods (for Bayesian inference)
    # =========================================================================

    def sample_latent_from_nf(
            self,
            params: Dict[str, Any],
            num_samples: int,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Sample from NF prior (8-dim coefficient space)"""
        nf = self.models['nf']
        return nf.apply(
            {'params': params['nf']},
            rng,
            num_samples,
            method=nf.sample
        )

    def log_prob_latent(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log p(beta) using NF (8-dim coefficient space)"""
        nf = self.models['nf']
        return nf.apply(
            {'params': params['nf']},
            beta,
            method=nf.log_prob
        )

    # =========================================================================
    # Observation methods
    # =========================================================================

    def prepare_observations(
            self,
            sample_indices: List[int],
            obs_indices: jnp.ndarray,
            snr_db: float = None,
            rng: jax.Array = None
    ) -> Dict[str, jnp.ndarray]:
        """Prepare observations for test samples

        Also sets self._current_g_l for inversion context.
        """
        a_true = jnp.stack([self.test_data['a'][i] for i in sample_indices])
        x_full = jnp.stack([self.test_data['x'][i] for i in sample_indices])
        g_l = jnp.stack([self.test_data['g_l'][i] for i in sample_indices])

        # Store current g_l for inversion methods
        self._current_g_l = g_l

        # If u_sol exists in test data, use it
        if 'u' in self.test_data:
            u_true = jnp.stack([self.test_data['u'][i] for i in sample_indices])
            u_obs = u_true[:, obs_indices, :]

            if snr_db is not None and rng is not None:
                u_obs = self.add_noise_snr(u_obs, snr_db, rng)
        else:
            # No u_sol in data
            u_true = None
            u_obs = None

        x_obs = x_full[:, obs_indices, :]

        result = {
            'x_full': x_full,
            'x_obs': x_obs,
            'a_true': a_true,
            'g_l': g_l,
        }

        if u_true is not None:
            result['u_true'] = u_true
            result['u_obs'] = u_obs

        return result

    def get_n_test_samples(self) -> int:
        return len(self.test_data['a'])

    def get_n_points(self) -> int:
        return self.test_data['x'].shape[1]

    def get_batch_keys(self) -> List[str]:
        """Return list of keys needed for training batches"""
        return ['a', 'x', 'g_l', 'coe_val', 'coe_mask']
