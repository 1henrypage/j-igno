# src/problems/darcy_piecewise.py

"""
Darcy Flow Piecewise problem in JAX.

PDE: -div(a * grad(u)) = f (with mixed formulation using stress variables)

Key differences from continuous:
- Piecewise constant coefficients {5, 10} on 29x29 grid
- Binary cross-entropy loss for coefficient reconstruction
- Nearest-neighbor lookup instead of RBF interpolation
- UnitGaussianNormalizer for input normalization
- Mixed formulation: s = a * grad(u), with separate s1/s2 decoders
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
import optax

from src.components.encoder import EncoderFCNetTanh
from src.components.nf import RealNVP
from src.problems import ProblemInstance, register_problem
from src.utils.GenPoints import Point2D
from src.utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from src.utils.misc_utils import np2jax
from src.utils.solver_utils import get_model


# ============================================================================
# Pure functions for PDE computation (JIT-compilable)
# ============================================================================

@jit
def mollifier(u: jax.Array, x: jax.Array) -> jax.Array:
    """Apply mollifier boundary condition (same as continuous)

    Args:
        u: Solution (batch, n_points) or (n_points,)
        x: Coordinates (batch, n_points, 2) or (n_points, 2)

    Returns:
        Mollified solution with extra dimension: (..., 1)
    """
    pi = jnp.pi
    result = u * jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1])
    return result[..., None]


@jit
def mollifier_no_expand(u: jax.Array, x: jax.Array) -> jax.Array:
    """Apply mollifier without dimension expansion (for internal gradient computation)

    Args:
        u: Solution values
        x: Coordinates

    Returns:
        Mollified solution with same shape as input u
    """
    pi = jnp.pi
    return u * jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1])


@jit
def get_high_res_a(x_query: jax.Array, a_coe: jax.Array, res: int = 29) -> jax.Array:
    """Nearest-neighbor lookup on res×res grid for piecewise constant coefficients

    Args:
        x_query: Query coordinates (..., 2) in [0, 1]²
        a_coe: Coefficient values (..., res²) - flattened grid
        res: Grid resolution (default 29)

    Returns:
        Coefficient values at query points (..., 1)
    """
    delta = 1.0 / (res - 1)
    x_loc = jnp.floor(x_query[..., 0] / delta + 0.5).astype(jnp.int32)
    y_loc = jnp.floor(x_query[..., 1] / delta + 0.5).astype(jnp.int32)

    # Clip to valid range [0, res-1]
    x_loc = jnp.clip(x_loc, 0, res - 1)
    y_loc = jnp.clip(y_loc, 0, res - 1)

    loc = y_loc * res + x_loc
    return jnp.take_along_axis(a_coe, loc[..., None], axis=-1)


@jit
def a_sample(a_prob_sigmoid: jax.Array, n_samples: int = 50) -> jax.Array:
    """Convert sigmoid probabilities to coefficient values via soft thresholding

    Implements Gumbel-softmax-like averaging for differentiable sampling.
    Converts probabilities ∈ [0,1] to coefficient values ≈ {5, 10}.

    Args:
        a_prob_sigmoid: Sigmoid probabilities (batch, n_points, 1) or (...,)
        n_samples: Number of samples to average (default 50)

    Returns:
        Coefficient approximations: sigmoid(avg_logits) * 5 + 5
    """
    # This is a simplified version - just use the sigmoid directly
    # Map [0, 1] → [5, 10]
    return a_prob_sigmoid * 5.0 + 5.0


def compute_u_and_grad(
        params_u: Any,
        model_u: nn.Module,
        x: jax.Array,
        beta: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Compute u and its gradient w.r.t. x (per-point version, same as continuous)

    Args:
        params_u: Parameters for u network
        model_u: U network module
        x: Coordinates (n_points, 2)
        beta: Latent code (latent_dim,)

    Returns:
        u: Solution values (n_points,)
        du_dx: Gradients (n_points, 2)
    """

    def u_at_point(x_single):
        """Evaluate u at a single point"""
        x_batch = x_single[None, None, :]
        beta_batch = beta[None, :]
        u_val = model_u.apply({'params': params_u}, x_batch, beta_batch)
        u_val = u_val[0, 0]
        u_val = u_val * jnp.sin(jnp.pi * x_single[0]) * jnp.sin(jnp.pi * x_single[1])
        return u_val

    def value_and_grad_at_point(x_single):
        val = u_at_point(x_single)
        grad_val = jax.grad(u_at_point)(x_single)
        return val, grad_val

    u_vals, du_vals = vmap(value_and_grad_at_point)(x)
    return u_vals, du_vals


def compute_pde_residual_piecewise_single_sample(
        params_u: Any,
        params_s1: Any,
        params_s2: Any,
        model_u: nn.Module,
        model_s1: nn.Module,
        model_s2: nn.Module,
        beta: jax.Array,
        xc: jax.Array,
        R: jax.Array,
        int_grid: jax.Array,
        v: jax.Array,
        dv_dr: jax.Array,
        a_vals: jax.Array,
        n_grid: int
) -> jax.Array:
    """Compute PDE residual for single sample using mixed formulation

    Mixed formulation:
        res1 = ||s - a * grad(u)||²  (constitutive relation)
        res2 = (∫ s · grad(v) - ∫ f * v)²  (weak PDE)

    Args:
        params_u, params_s1, params_s2: Parameters for networks
        model_u, model_s1, model_s2: Network modules
        beta: Latent code (latent_dim,)
        xc: Center points (nc, 1, 2)
        R: Radius (nc, 1, 1)
        int_grid: Integration grid (n_grid, 2)
        v: Test function values (n_grid, 1)
        dv_dr: Test function gradients (n_grid, 2)
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
        u_vals, du_vals = compute_u_and_grad(params_u, model_u, x_center, beta)
        return u_vals, du_vals

    u_all, du_all = vmap(compute_for_center)(jnp.arange(nc))
    # u_all: (nc, n_grid), du_all: (nc, n_grid, 2)

    u_flat = u_all.reshape(-1, 1)  # (nc*n_grid, 1)
    du_flat = du_all.reshape(-1, 2)  # (nc*n_grid, 2)

    # Compute stress variables s = (s1, s2) at collocation points
    # Repeat beta for all points
    beta_batch = jnp.tile(beta[None, :], (nc * n_grid, 1))

    s1_vals = model_s1.apply({'params': params_s1}, x_flat[None, :, :], beta_batch[None, :, :])
    s2_vals = model_s2.apply({'params': params_s2}, x_flat[None, :, :], beta_batch[None, :, :])
    s1_vals = s1_vals[0]  # (nc*n_grid,)
    s2_vals = s2_vals[0]  # (nc*n_grid,)
    s_vals = jnp.stack([s1_vals, s2_vals], axis=-1)  # (nc*n_grid, 2)

    # res1: ||s - a * grad(u)||²
    a_grad_u = a_vals * du_flat  # (nc*n_grid, 2)
    res1 = jnp.sum((s_vals - a_grad_u) ** 2, axis=-1)  # (nc*n_grid,)
    res1 = res1.reshape(nc, n_grid).mean(axis=-1)  # (nc,)

    # res2: (∫ s · grad(v) - ∫ f * v)²
    f = 10.0  # Source term
    left = jnp.sum(s_vals * dv, axis=-1)  # (nc*n_grid,)
    left = left.reshape(nc, n_grid).mean(axis=-1)  # (nc,)

    right = (f * v_flat[:, 0]).reshape(nc, n_grid).mean(axis=-1)  # (nc,)

    res2 = (left - right) ** 2  # (nc,)

    # Combine: mean(res1) + mean(res2) * sqrt(nc)
    nc_float = jnp.float32(nc)
    residual = res1 + res2 * jnp.sqrt(nc_float)

    return residual


@register_problem("darcy_piecewise")
class DarcyPiecewise(ProblemInstance):
    """Darcy flow with piecewise constant coefficients in JAX"""

    # Model hyperparameters
    BETA_SIZE = 8
    HIDDEN_SIZE_U = 100
    HIDDEN_SIZE_A = 256
    HIDDEN_SIZE_S = 100
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

        # Normalization stats (will be set during data loading)
        self.a_mean = None
        self.a_std = None
        self.a_coe_train = None  # Store for PDE loss

        # Load data
        print("Loading data...")
        if self.train_data_path:
            self.train_data, self.gridx_train = self._load_data(self.train_data_path, is_train=True)
            print(f"  Train: a={self.train_data['a'].shape}, u={self.train_data['u'].shape}")

        if self.test_data_path:
            self.test_data, self.gridx_test = self._load_data(self.test_data_path, is_train=False)
            print(f"  Test: a={self.test_data['a'].shape}, u={self.test_data['u'].shape}")

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
            n_mesh_or_grid=7,  # Different from continuous (9)
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

        For piecewise, data contains:
        - coe: Piecewise constant coefficients (N, 29, 29) with values {5, 10}
        - u: Solution (N, 29, 29)
        - X, Y: Mesh grids
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        data = h5py.File(path, mode='r')

        # Load coefficient (29x29 grid, piecewise constant)
        coe = np2jax(np.array(data["coe"]).T, self.dtype)  # (N, 29, 29)
        u = np2jax(np.array(data["sol"]).T, self.dtype)

        X, Y = np.array(data['X']).T, np.array(data['Y']).T
        mesh = np2jax(np.vstack([X.ravel(), Y.ravel()]).T, self.dtype)
        gridx = mesh.reshape(-1, 2)

        ndata = coe.shape[0]

        # Flatten coefficient to (N, 841)
        a_coe = coe.reshape(ndata, -1)  # (N, 841)

        # Compute normalization stats from training data
        if is_train:
            self.a_mean = jnp.mean(a_coe, axis=0, keepdims=True)  # (1, 841)
            self.a_std = jnp.std(a_coe, axis=0, keepdims=True) + 1e-8  # (1, 841)
            self.a_coe_train = a_coe  # Store for PDE loss
            print(f"  Normalization: mean range [{jnp.min(self.a_mean):.3f}, {jnp.max(self.a_mean):.3f}]")
            print(f"                 std range [{jnp.min(self.a_std):.3f}, {jnp.max(self.a_std):.3f}]")

        # Normalize coefficient for encoder input
        if self.a_mean is not None and self.a_std is not None:
            a_normalized = (a_coe - self.a_mean) / self.a_std
        else:
            a_normalized = a_coe  # Test data before train data loaded

        a = a_normalized.reshape(ndata, -1, 1)  # (N, 841, 1) for encoder
        x = jnp.tile(gridx[None, :, :], (ndata, 1, 1))
        u = u.reshape(ndata, -1, 1)

        return {
            'a': a,
            'u': u,
            'x': x,
            'a_coe': a_coe,  # Store raw coefficient for nearest-neighbor lookup
        }, gridx

    def _encode(self, a_coe: jax.Array) -> jax.Array:
        """Normalize coefficient for encoder input

        Args:
            a_coe: Raw coefficient (batch, 841)

        Returns:
            Normalized coefficient (batch, 841, 1)
        """
        a_normalized = (a_coe - self.a_mean) / self.a_std
        return a_normalized[..., None]

    def _build_models(self) -> Dict[str, nn.Module]:
        """Build all models"""
        net_type = 'MultiONetBatch'

        # Encoder: FCNet with tanh
        layers_list = [841, 512, 256, self.BETA_SIZE]
        model_enc = EncoderFCNetTanh(
            layers_list=layers_list,
            activation='SiLU',
            dtype=self.dtype
        )

        # Decoder for u
        trunk_layers = [self.HIDDEN_SIZE_U] * 5
        branch_layers = [self.HIDDEN_SIZE_U] * 5

        model_u = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=4,
            dtype=self.dtype
        )

        # Decoder for a (outputs logits for binary classification)
        trunk_layers_a = [self.HIDDEN_SIZE_A] * 5
        branch_layers_a = [self.HIDDEN_SIZE_A] * 5

        model_a = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers_a,
            branch_layers=branch_layers_a,
            activation_trunk='SiLU_Sin',
            activation_branch='SiLU_Id',
            net_type=net_type,
            sum_layers=4,
            dtype=self.dtype
        )

        # Decoders for stress variables s1, s2
        trunk_layers_s = [self.HIDDEN_SIZE_S] * 5
        branch_layers_s = [self.HIDDEN_SIZE_S] * 5

        model_s1 = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers_s,
            branch_layers=branch_layers_s,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=4,
            dtype=self.dtype
        )

        model_s2 = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers_s,
            branch_layers=branch_layers_s,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=4,
            dtype=self.dtype
        )

        # Normalizing flow
        model_nf = RealNVP(
            dim=self.BETA_SIZE,
            num_flows=self.NF_NUM_FLOWS,
            hidden_dim=self.NF_HIDDEN_DIM,
        )

        return {
            'enc': model_enc,
            'u': model_u,
            'a': model_a,
            's1': model_s1,
            's2': model_s2,
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

        sample_beta = jnp.ones((batch_size, self.BETA_SIZE), dtype=self.dtype)

        return {
            'enc': {'x': sample_a},
            'u': {'x': sample_x, 'a': sample_beta},
            'a': {'x': sample_x, 'a': sample_beta},
            's1': {'x': sample_x, 'a': sample_beta},
            's2': {'x': sample_x, 'a': sample_beta},
            'nf': {'x': sample_beta},
        }

    def get_weight_decay_groups(self) -> Dict[str, bool]:
        """Return dict mapping model name -> whether it should have weight decay"""
        return {
            'enc': False,
            'u': True,
            'a': True,
            's1': True,
            's2': True,
            'nf': False,
        }

    # =========================================================================
    # Training loss methods (encode a -> beta first)
    # =========================================================================

    def loss_pde(
            self,
            params: Dict[str, Any],
            a: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """PDE residual loss during training

        Uses nearest-neighbor lookup on TRUE coefficient field.

        Args:
            params: Model parameters
            a: Normalized coefficient field (batch, n_points, 1)
            rng: PRNG key

        Returns:
            PDE loss (scalar)
        """
        nc = 100
        n_batch = a.shape[0]

        # Encode a -> beta
        beta = self.models['enc'].apply({'params': params['enc']}, a)

        # Generate collocation points
        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(
            n_center=nc,
            R_max=1e-4,
            R_min=1e-4,
            key=subkey
        )

        # Transform grid
        x = self.int_grid[None, :, :] * R + xc  # (nc, n_grid, 2)
        x_flat = x.reshape(-1, 2)  # (nc*n_grid, 2)

        # Get coefficient values via nearest-neighbor lookup
        # We need to map from batch to the corresponding a_coe
        # During training, we get batch indices implicitly
        def get_a_for_batch(batch_idx, a_coe_sample):
            """Get coefficient at collocation points for one sample"""
            a_vals = get_high_res_a(x_flat, a_coe_sample, res=29)  # (nc*n_grid, 1)
            return lax.stop_gradient(a_vals)  # Stop gradient on true coefficient

        # We need a_coe for the current batch
        # This is tricky - we need to pass it separately
        # For now, assume we have access to it via get_batch_data
        # This will be handled in compute_training_losses

        # Simplified: just return 0 for now - will override compute_training_losses
        return jnp.array(0.0, dtype=self.dtype)

    def compute_training_losses(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jax.Array],
            rng: jax.Array,
            loss_weights: Any
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Compute training losses (override to handle a_coe)

        Args:
            params: Model parameters
            batch: Batch data with 'a', 'u', 'x', 'a_coe'
            rng: PRNG key
            loss_weights: Loss weight configuration

        Returns:
            total_loss, metrics_dict
        """
        a = batch['a']
        u = batch['u']
        x = batch['x']
        a_coe = batch['a_coe']
        n_batch = a.shape[0]

        # Encode
        beta = self.models['enc'].apply({'params': params['enc']}, a)

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

        def compute_residual_for_sample(beta_single, a_coe_single):
            a_vals = get_high_res_a(x_flat, a_coe_single, res=29)
            a_vals = lax.stop_gradient(a_vals)
            return compute_pde_residual_piecewise_single_sample(
                params['u'], params['s1'], params['s2'],
                self.models['u'], self.models['s1'], self.models['s2'],
                beta_single,
                xc, R,
                self.int_grid, self.v, self.dv_dr,
                a_vals,
                self.n_grid
            )

        residuals = vmap(compute_residual_for_sample)(beta, a_coe)
        residuals_squared = residuals ** 2
        mse_loss = jnp.mean(residuals_squared)

        residuals_flat = residuals_squared.reshape(-1)
        residuals_sorted = jnp.sort(residuals_flat)[::-1]
        top_k_loss = jnp.sum(residuals_sorted[:nc * 10])

        loss_pde = mse_loss + top_k_loss

        # Data loss: Binary cross-entropy for coefficient
        a_logits = self.models['a'].apply({'params': params['a']}, x, beta)  # (batch, n_points)

        # Target: (a_true - 5) / 5 maps {5, 10} → {0, 1}
        a_true_unnorm = a_coe  # (batch, 841)
        a_target = (a_true_unnorm - 5.0) / 5.0  # (batch, 841)

        # BCE loss
        loss_data_a = jnp.mean(optax.sigmoid_binary_cross_entropy(a_logits, a_target))

        # Data loss for u: standard MSE
        u_pred = self.models['u'].apply({'params': params['u']}, x, beta)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        u_pred = mollifier(u_pred.squeeze(-1), x)
        loss_data_u = self.get_loss(u_pred, u)

        loss_data = loss_data_a + loss_data_u

        # NF loss
        loss_nf = self.models['nf'].apply(
            {'params': params['nf']},
            lax.stop_gradient(beta),
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
            'loss_data_a': loss_data_a,
            'loss_data_u': loss_data_u,
            'loss_nf': loss_nf,
        }

        return total_loss, metrics

    def loss_data(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Data reconstruction loss (not used - overridden in compute_training_losses)"""
        raise NotImplementedError("Use compute_training_losses instead")

    def error(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute error on solution

        Args:
            params: Model parameters
            x: Coordinates (batch, n_points, 2)
            a: Normalized coefficient field (batch, n_points, 1)
            u: Solution field (batch, n_points, 1)

        Returns:
            Error (batch,)
        """
        # Encode
        beta = self.models['enc'].apply({'params': params['enc']}, a)

        # Predict solution
        u_pred = self.models['u'].apply({'params': params['u']}, x, beta)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        u_pred = mollifier(u_pred.squeeze(-1), x)

        # Compute error
        return self.get_error(u_pred, u)

    # =========================================================================
    # From-beta methods (for inversion)
    # =========================================================================

    def loss_pde_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """PDE loss from beta (uses decoded coefficient)

        Args:
            params: Model parameters
            beta: Latent codes (batch, latent_dim)
            rng: PRNG key

        Returns:
            PDE loss (scalar)
        """
        nc = 100
        n_batch = beta.shape[0]

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

        # Decode coefficient: logits → sigmoid → {5, 10}
        a_logits = self.models['a'].apply({'params': params['a']}, x_batch, beta)
        a_prob = jax.nn.sigmoid(a_logits)
        a_decoded = a_sample(a_prob[..., None])  # (n_batch, nc*n_grid, 1)
        a_decoded = lax.stop_gradient(a_decoded)  # Stop gradient

        def compute_residual_for_sample(beta_single, a_single):
            return compute_pde_residual_piecewise_single_sample(
                params['u'], params['s1'], params['s2'],
                self.models['u'], self.models['s1'], self.models['s2'],
                beta_single,
                xc, R,
                self.int_grid, self.v, self.dv_dr,
                a_single,
                self.n_grid
            )

        residuals = vmap(compute_residual_for_sample)(beta, a_decoded)
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
            beta: Latent codes (batch, latent_dim)
            x: Coordinates (batch, n_points, 2)
            target: Target values (batch, n_points, 1)
            target_type: 'a' or 'u'

        Returns:
            Data loss (scalar)
        """
        if target_type == 'a':
            # For 'a': use BCE loss
            a_logits = self.models['a'].apply({'params': params['a']}, x, beta)

            # Target should be raw coefficient values, map to {0, 1}
            # Assuming target is (batch, n_points, 1) with values {5, 10}
            if target.ndim == 3:
                target = target.squeeze(-1)
            a_target = (target - 5.0) / 5.0

            return jnp.mean(optax.sigmoid_binary_cross_entropy(a_logits, a_target))

        elif target_type == 'u':
            pred = self.models['u'].apply({'params': params['u']}, x, beta)
            pred = pred[..., None] if pred.ndim == 2 else pred
            pred = mollifier(pred.squeeze(-1), x)
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
        """Error from beta

        Args:
            params: Model parameters
            beta: Latent codes (batch, latent_dim)
            x: Coordinates (batch, n_points, 2)
            target: Target values (batch, n_points, 1)
            target_type: 'a' or 'u'

        Returns:
            Error (batch,)
        """
        if target_type == 'u':
            pred = self.models['u'].apply({'params': params['u']}, x, beta)
            pred = pred[..., None] if pred.ndim == 2 else pred
            pred = mollifier(pred.squeeze(-1), x)

        elif target_type == 'a':
            # For 'a': decode logits → sigmoid → {5, 10}
            a_logits = self.models['a'].apply({'params': params['a']}, x, beta)
            a_prob = jax.nn.sigmoid(a_logits)
            pred = a_sample(a_prob[..., None])

            # Ensure target has right shape
            if target.ndim == 2:
                target = target[..., None]

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
            beta: Latent codes (batch, latent_dim)
            x: Coordinates (batch, n_points, 2)

        Returns:
            Dict with 'u_pred' and 'a_pred'
        """
        u_pred = self.models['u'].apply({'params': params['u']}, x, beta)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        u_pred = mollifier(u_pred.squeeze(-1), x)

        a_logits = self.models['a'].apply({'params': params['a']}, x, beta)
        a_prob = jax.nn.sigmoid(a_logits)
        a_pred = a_sample(a_prob[..., None])

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
        """Sample from NF prior"""
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
        """Compute log p(beta) using NF"""
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
        """Prepare observations for test samples"""
        a_true_raw = jnp.stack([self.test_data['a_coe'][i] for i in sample_indices])
        u_true = jnp.stack([self.test_data['u'][i] for i in sample_indices])
        x_full = jnp.stack([self.test_data['x'][i] for i in sample_indices])

        # For a_true, we need the raw values (not normalized)
        a_true = a_true_raw[..., None]  # (batch, 841, 1)

        x_obs = x_full[:, obs_indices, :]
        u_obs = u_true[:, obs_indices, :]

        if snr_db is not None and rng is not None:
            u_obs = self.add_noise_snr(u_obs, snr_db, rng)

        return {
            'x_full': x_full,
            'x_obs': x_obs,
            'u_obs': u_obs,
            'u_true': u_true,
            'a_true': a_true,
        }

    def get_n_test_samples(self) -> int:
        return len(self.test_data['a'])

    def get_n_points(self) -> int:
        return self.test_data['x'].shape[1]

    def get_batch_keys(self) -> List[str]:
        """Return list of keys needed for training batches"""
        return ['a', 'u', 'x', 'a_coe']

    # =========================================================================
    # Checkpoint override for normalization stats
    # =========================================================================

    def save_checkpoint_metadata(self) -> Dict[str, Any]:
        """Save normalization stats with checkpoint"""
        return {
            'a_mean': np.array(self.a_mean) if self.a_mean is not None else None,
            'a_std': np.array(self.a_std) if self.a_std is not None else None,
        }

    def load_checkpoint_metadata(self, metadata: Dict[str, Any]):
        """Load normalization stats from checkpoint"""
        if 'a_mean' in metadata and metadata['a_mean'] is not None:
            self.a_mean = np2jax(metadata['a_mean'], self.dtype)
        if 'a_std' in metadata and metadata['a_std'] is not None:
            self.a_std = np2jax(metadata['a_std'], self.dtype)
