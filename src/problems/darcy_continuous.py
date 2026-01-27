# src/problems/darcy_continuous.py
# Darcy Flow in JAX with fixes:
# 1. FIXED: Mollifier now adds dimension like PyTorch version
# 2. FIXED: weight_centers called with key parameter
# 3. IMPROVED: More efficient batched gradient computation option

"""
Darcy Flow Continuous problem in JAX.

PDE: -div(a * grad(u)) = f

Key implementation details:
- Weak formulation with test functions
- Proper JAX autodiff for computing grad(u)
- RBF interpolation for coefficient field during training
- Decoded coefficient field during inversion
- All functions JIT-compilable
"""

import jax
import jax.numpy as jnp
import h5py
from jax import random, grad, jit, vmap, value_and_grad
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
from src.utils.RBFInterpolatorMesh import RBFInterpolator
from src.utils.solver_utils import get_model


# ============================================================================
# Pure functions for PDE computation (JIT-compilable)
# ============================================================================

@jit
def mollifier(u: jax.Array, x: jax.Array) -> jax.Array:
    """Apply mollifier boundary condition

    FIXED: Now adds trailing dimension to match PyTorch behavior.

    Args:
        u: Solution (batch, n_points) or (n_points,)
        x: Coordinates (batch, n_points, 2) or (n_points, 2)

    Returns:
        Mollified solution with extra dimension: (..., 1)
    """
    pi = jnp.pi
    result = u * jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1])
    return result[..., None]  # FIXED: Add trailing dimension


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


def compute_u_and_grad(
        params_u: Any,
        model_u: nn.Module,
        x: jax.Array,
        beta: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Compute u and its gradient w.r.t. x (per-point version)

    This computes gradients point-by-point using vmap.

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
        """Evaluate u at a single point

        Args:
            x_single: (2,)

        Returns:
            u_val: scalar
        """
        # Network expects (batch, n_points, 2) and (batch, latent_dim)
        x_batch = x_single[None, None, :]  # (1, 1, 2)
        beta_batch = beta[None, :]  # (1, latent_dim)

        # Evaluate network
        u_val = model_u.apply({'params': params_u}, x_batch, beta_batch)
        u_val = u_val[0, 0]  # Extract scalar

        # Apply mollifier (without dimension expansion for gradient computation)
        u_val = u_val * jnp.sin(jnp.pi * x_single[0]) * jnp.sin(jnp.pi * x_single[1])

        return u_val

    # Compute both value and gradient at each point
    def value_and_grad_at_point(x_single):
        """Get value and gradient at single point"""
        val = u_at_point(x_single)
        grad_val = jax.grad(u_at_point)(x_single)
        return val, grad_val

    # Apply to all points via vmap
    u_vals, du_vals = vmap(value_and_grad_at_point)(x)

    return u_vals, du_vals


def compute_u_and_grad_batched(
        params_u: Any,
        model_u: nn.Module,
        x: jax.Array,
        beta: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Compute u and its gradient w.r.t. x (BATCHED version - more efficient)

    This version is more efficient as it evaluates the network once for all points,
    then computes gradients using jacfwd.

    Args:
        params_u: Parameters for u network
        model_u: U network module
        x: Coordinates (n_points, 2)
        beta: Latent code (latent_dim,)

    Returns:
        u: Solution values (n_points,)
        du_dx: Gradients (n_points, 2)
    """
    n_points = x.shape[0]

    def u_fn(x_all):
        """Evaluate u at all points

        Args:
            x_all: (n_points, 2)

        Returns:
            u_vals: (n_points,)
        """
        x_batch = x_all[None, :, :]  # (1, n_points, 2)
        beta_batch = beta[None, :]    # (1, latent_dim)

        u = model_u.apply({'params': params_u}, x_batch, beta_batch)
        u = u[0]  # (n_points,) or (n_points, 1)
        if u.ndim == 2:
            u = u[:, 0]

        # Apply mollifier
        u = mollifier_no_expand(u, x_all)
        return u

    # Compute values
    u_vals = u_fn(x)

    # Compute Jacobian: du/dx has shape (n_points, n_points, 2)
    # We only need the diagonal: du[i]/dx[i]
    jac = jax.jacfwd(u_fn)(x)  # (n_points, n_points, 2)

    # Extract diagonal
    du_vals = jnp.diagonal(jac, axis1=0, axis2=1).T  # (n_points, 2)

    return u_vals, du_vals


def compute_pde_residual_single_sample(
        params_u: Any,
        model_u: nn.Module,
        beta: jax.Array,
        xc: jax.Array,
        R: jax.Array,
        int_grid: jax.Array,
        v: jax.Array,
        dv_dr: jax.Array,
        a_vals: jax.Array,
        n_grid: int,
        use_batched_grad: bool = False
) -> jax.Array:
    """Compute PDE residual for single sample

    Weak formulation: ∫ a * grad(u) · grad(v) dx = ∫ f * v dx

    Args:
        params_u: Parameters for u network
        model_u: U network module
        beta: Latent code (latent_dim,)
        xc: Center points (nc, 1, 2)
        R: Radius (nc, 1, 1)
        int_grid: Integration grid (n_grid, 2)
        v: Test function values (n_grid, 1)
        dv_dr: Test function gradients (n_grid, 2)
        a_vals: Coefficient values (nc*n_grid, 1)
        n_grid: Number of grid points per center
        use_batched_grad: If True, use more efficient batched gradient computation

    Returns:
        residual: PDE residual (nc,)
    """
    nc = xc.shape[0]

    # Transform integration grid to physical space
    # x: (nc, n_grid, 2)
    x = int_grid[None, :, :] * R + xc
    x_flat = x.reshape(-1, 2)  # (nc*n_grid, 2)

    # Compute test function gradients in physical space
    # dv/dx = (1/R) * dv/dr
    dv = (dv_dr[None, :, :] / R).reshape(-1, 2)  # (nc*n_grid, 2)
    v_flat = jnp.tile(v, (nc, 1, 1)).reshape(-1, 1)  # (nc*n_grid, 1)

    # Compute u and grad(u) at all collocation points
    if use_batched_grad:
        # More efficient but may have numerical differences
        def compute_for_center_batched(center_idx):
            x_center = x[center_idx]  # (n_grid, 2)
            u_vals, du_vals = compute_u_and_grad_batched(params_u, model_u, x_center, beta)
            return u_vals, du_vals
        u_all, du_all = vmap(compute_for_center_batched)(jnp.arange(nc))
    else:
        # Original per-point version
        def compute_for_center(center_idx):
            x_center = x[center_idx]  # (n_grid, 2)
            u_vals, du_vals = compute_u_and_grad(params_u, model_u, x_center, beta)
            return u_vals, du_vals
        u_all, du_all = vmap(compute_for_center)(jnp.arange(nc))

    # u_all: (nc, n_grid), du_all: (nc, n_grid, 2)

    # Flatten
    u_flat = u_all.reshape(-1, 1)  # (nc*n_grid, 1)
    du_flat = du_all.reshape(-1, 2)  # (nc*n_grid, 2)

    # Source term f = 10
    f = 10.0 * jnp.ones_like(u_flat)

    # Left side: ∫ a * grad(u) · grad(v) dx
    # (nc*n_grid, 1) * (nc*n_grid, 2) * (nc*n_grid, 2) -> (nc*n_grid,)
    left = jnp.sum(a_vals * du_flat * dv, axis=-1)
    left = left.reshape(nc, n_grid).mean(axis=-1)  # (nc,)

    # Right side: ∫ f * v dx
    right = (f * v_flat).reshape(nc, n_grid).mean(axis=-1)  # (nc,)

    # Residual
    residual = (left - right) ** 2

    return residual


@register_problem("darcy_continuous")
class DarcyContinuous(ProblemInstance):
    """Darcy flow problem in JAX with proper PDE loss"""

    # Model hyperparameters
    BETA_SIZE = 18
    HIDDEN_SIZE = 100
    NF_NUM_FLOWS = 3
    NF_HIDDEN_DIM = 56

    def __init__(
            self,
            seed: int,
            dtype: jnp.dtype = jnp.float32,
            train_data_path: str = None,
            test_data_path: str = None,
            use_batched_grad: bool = False  # Option for efficiency
    ):
        super().__init__(
            seed=seed,
            dtype=dtype,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
        )

        self.use_batched_grad = use_batched_grad

        # Load data
        print("Loading data...")
        if self.train_data_path:
            self.train_data, self.gridx_train = self._load_data(self.train_data_path)
            print(f"  Train: a={self.train_data['a'].shape}, u={self.train_data['u'].shape}")

            # Initialize RBF interpolator (JAX version)
            self.fun_a = RBFInterpolator(
                x_mesh=self.gridx_train,
                kernel='gaussian',
                eps=25.,
                smoothing=0.,
                degree=6,
            )

        if self.test_data_path:
            self.test_data, self.gridx_test = self._load_data(self.test_data_path)
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

    def _load_data(self, path: str) -> Tuple[Dict, jax.Array]:
        """Load data from npy directory"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")


        data = h5py.File(path, mode='r')

        a = np2jax(np.array(data["coeff"]).T, self.dtype)
        u = np2jax(np.array(data["sol"]).T, self.dtype)

        X, Y = np.array(data['X']).T, np.array(data['Y']).T
        mesh = np2jax(np.vstack([X.ravel(), Y.ravel()]).T, self.dtype)
        gridx = mesh.reshape(-1, 2)

        ndata = a.shape[0]
        a = a.reshape(ndata, -1, 1)
        x = jnp.tile(gridx[None, :, :], (ndata, 1, 1))
        u = u.reshape(ndata, -1, 1)

        return {'a': a, 'u': u, 'x': x}, gridx

    def _build_models(self) -> Dict[str, nn.Module]:
        """Build all models"""
        net_type = 'MultiONetBatch'

        # Encoder
        conv_arch = [1, 64, 64, 64]
        fc_arch = [64 * 2 * 2, 128, 64, self.BETA_SIZE]
        model_enc = EncoderCNNet2dTanh(
            conv_arch=conv_arch,
            fc_arch=fc_arch,
            activation_conv='SiLU',
            activation_fc='SiLU',
            nx_size=29,
            ny_size=29,
            kernel_size=(3, 3),
            stride=2,
            dtype=self.dtype
        )

        # Decoders
        trunk_layers = [self.HIDDEN_SIZE] * 6
        branch_layers = [self.HIDDEN_SIZE] * 6

        model_a = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=5,
            dtype=self.dtype
        )

        model_u = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=5,
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
            'nf': model_nf,
        }

    # =========================================================================
    # Training loss methods (encode a -> beta first)
    # Uses TRUE coefficient via RBF interpolator
    # =========================================================================

    def loss_pde(
            self,
            params: Dict[str, Any],
            a: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """PDE residual loss during training
        Uses RBF interpolator on TRUE coefficient field.
        Args:
            params: Model parameters
            a: Coefficient field (batch, n_points, 1)
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
            key=subkey  # FIXED: Added key parameter
        )

        # Compute coefficient values at collocation points using RBF
        # Transform grid
        x = self.int_grid[None, :, :] * R + xc  # (nc, n_grid, 2)
        x_flat = jnp.tile(x.reshape(-1, 2)[None, :, :], (n_batch, 1, 1))  # (n_batch, nc*n_grid, 2)

        # Interpolate coefficient (batched)
        a_vals = self.fun_a(x_flat, a)  # (n_batch, nc*n_grid, 1)

        # Compute PDE residual for each sample
        def compute_residual_for_sample(beta_single, a_single):
            """Compute residual for one sample"""
            return compute_pde_residual_single_sample(
                params['u'],
                self.models['u'],
                beta_single,
                xc,
                R,
                self.int_grid,
                self.v,
                self.dv_dr,
                a_single,
                self.n_grid,
                use_batched_grad=self.use_batched_grad
            )

        # Vmap over batch
        residuals = vmap(compute_residual_for_sample)(beta, a_vals)
        # residuals: (n_batch, nc)

        # Combine residuals
        # Use same strategy as PyTorch: MSE + top-k sorting
        mse_loss = jnp.mean(residuals)

        # Sort and take top 10*nc largest residuals
        residuals_flat = residuals.reshape(-1)
        residuals_sorted = jnp.sort(residuals_flat)[::-1]  # Descending
        top_k_loss = jnp.sum(residuals_sorted[:nc * 10])

        return mse_loss + top_k_loss

    def loss_data(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Data reconstruction loss

        Encodes a -> beta, then reconstructs coefficient.

        Args:
            params: Model parameters
            x: Coordinates (batch, n_points, 2)
            a: Coefficient field (batch, n_points, 1)
            u: Solution field (batch, n_points, 1) - not used

        Returns:
            Data loss (scalar)
        """
        # Encode
        beta = self.models['enc'].apply({'params': params['enc']}, a)

        # Reconstruct coefficient
        a_pred = self.models['a'].apply({'params': params['a']}, x, beta)

        # Compute loss
        a_target = a.squeeze(-1) if a.ndim == 3 else a
        return self.get_loss(a_pred, a_target)

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
            a: Coefficient field (batch, n_points, 1)
            u: Solution field (batch, n_points, 1)

        Returns:
            Error (batch,)
        """
        # Encode
        beta = self.models['enc'].apply({'params': params['enc']}, a)

        # Predict solution
        u_pred = self.models['u'].apply({'params': params['u']}, x, beta)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        u_pred = mollifier(u_pred.squeeze(-1), x)  # FIXED: Now adds dimension

        # Compute error
        return self.get_error(u_pred, u)

    # =========================================================================
    # From-beta methods (for inversion)
    # Uses DECODED coefficient
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
        # FIXED: Pass key explicitly
        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(
            n_center=nc,
            R_max=1e-4,
            R_min=1e-4,
            key=subkey  # FIXED: Added key parameter
        )

        # Transform grid
        x = self.int_grid[None, :, :] * R + xc  # (nc, n_grid, 2)
        x_flat = jnp.tile(x.reshape(-1, 2)[None, :, :], (n_batch, 1, 1))  # (n_batch, nc*n_grid, 2)

        # Decode coefficient (not RBF interpolation!)
        a_decoded = self.models['a'].apply({'params': params['a']}, x_flat, beta)
        a_decoded = a_decoded[..., None]  # (n_batch, nc*n_grid, 1)

        # Compute PDE residual for each sample
        def compute_residual_for_sample(beta_single, a_single):
            """Compute residual for one sample"""
            return compute_pde_residual_single_sample(
                params['u'],
                self.models['u'],
                beta_single,
                xc,
                R,
                self.int_grid,
                self.v,
                self.dv_dr,
                a_single,
                self.n_grid,
                use_batched_grad=self.use_batched_grad
            )

        # Vmap over batch
        residuals = vmap(compute_residual_for_sample)(beta, a_decoded)
        # residuals: (n_batch, nc)

        # Mean residual over centers, then mean over batch
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
            pred = self.models['a'].apply({'params': params['a']}, x, beta)
            target_flat = target.squeeze(-1) if target.ndim == 3 else target
            return self.get_loss(pred, target_flat)

        elif target_type == 'u':
            pred = self.models['u'].apply({'params': params['u']}, x, beta)
            pred = pred[..., None] if pred.ndim == 2 else pred
            pred = mollifier(pred.squeeze(-1), x)  # FIXED: Now adds dimension
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
            pred = mollifier(pred.squeeze(-1), x)  # FIXED: Now adds dimension

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
            beta: Latent codes (batch, latent_dim)
            x: Coordinates (batch, n_points, 2)

        Returns:
            Dict with 'u_pred' and 'a_pred'
        """
        u_pred = self.models['u'].apply({'params': params['u']}, x, beta)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        u_pred = mollifier(u_pred.squeeze(-1), x)  # FIXED: Now adds dimension

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
        """Sample from NF prior

        Args:
            params: Model parameters (must include 'nf')
            num_samples: Number of samples to draw
            rng: PRNG key

        Returns:
            beta: Latent samples (num_samples, latent_dim)
        """
        nf = self.models['nf']
        return nf.apply(
            {'params': params['nf']},
            num_samples,
            rng,
            method=nf.sample
        )

    def log_prob_latent(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log p(beta) using NF

        For MCMC sampling, this gives the prior log-probability.

        Args:
            params: Model parameters (must include 'nf')
            beta: Latent samples (batch, latent_dim)

        Returns:
            log_prob: (batch,) log probability values
        """
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

        Args:
            sample_indices: Indices into test set
            obs_indices: Observation point indices
            snr_db: SNR for noise (None for clean)
            rng: PRNG key for noise

        Returns:
            Dict with observation data
        """
        # Stack samples
        a_true = jnp.stack([self.test_data['a'][i] for i in sample_indices])
        u_true = jnp.stack([self.test_data['u'][i] for i in sample_indices])
        x_full = jnp.stack([self.test_data['x'][i] for i in sample_indices])

        # Extract observations
        x_obs = x_full[:, obs_indices, :]
        u_obs = u_true[:, obs_indices, :]

        # Add noise if specified
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
