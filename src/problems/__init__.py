# src/problems/__init__.py
# JAX version - refactored for model-agnostic trainer

"""
Problem instances for JAX.

Each problem:
- Loads data
- Builds models (Flax modules)
- Defines JIT-compiled loss functions
- Handles checkpoints
- Provides sample inputs and weight decay configuration for trainer
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Callable, Literal, List, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn

from src.utils.Losses import MyError, MyLoss
from src.utils.misc_utils import setup_seed

_PROBLEM_REGISTRY: Dict[str, Type['ProblemInstance']] = {}


def register_problem(name: str) -> Callable:
    """Decorator to register problem class."""

    def decorator(cls: Type['ProblemInstance']) -> Type['ProblemInstance']:
        _PROBLEM_REGISTRY[name] = cls
        return cls

    return decorator


def get_problem_class(name: str) -> Type['ProblemInstance']:
    if name not in _PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem: '{name}'. Available: {list(_PROBLEM_REGISTRY.keys())}")
    return _PROBLEM_REGISTRY[name]


def list_problems() -> list:
    return list(_PROBLEM_REGISTRY.keys())


class ProblemInstance(ABC):
    """
    Abstract base for PDE problems in JAX.

    Key principles:
    - Models are Flax modules (Python objects)
    - Parameters stored separately (functional)
    - Loss functions are JIT-compiled pure functions
    - No mutable state during computation

    Training uses:
    - loss_pde(params, a, rng): Encodes a -> beta, then computes PDE loss
    - loss_data(params, x, a, u): Encodes a -> beta, then computes data loss
    - NF loss computed on stop_gradient(beta) in trainer

    Inversion uses:
    - loss_pde_from_beta(params, beta, rng): PDE loss directly from beta
    - loss_data_from_beta(params, beta, x, target, target_type): Data loss from beta
    - sample_latent_from_nf(params, num_samples, rng): Sample from NF prior
    - log_prob_latent(params, beta): Compute log p(beta) for MCMC
    """

    def __init__(
            self,
            seed: int,
            dtype: jnp.dtype = jnp.float32,
            train_data_path: str = None,
            test_data_path: str = None,
    ) -> None:
        self.rng = setup_seed(seed)
        self.seed = seed
        self.dtype = dtype
        self.run_dir: Optional[Path] = None

        # Data paths
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        # Models (Flax modules - not yet initialized)
        self.models: Dict[str, nn.Module] = {}

        # Parameters (initialized during setup)
        self.params: Dict[str, Any] = {}

        # Data (JAX arrays)
        self.train_data: Dict[str, jnp.ndarray] = {}
        self.test_data: Dict[str, jnp.ndarray] = {}

        # Loss/error functions
        self.get_loss = None
        self.get_error = None
        self.init_error()
        self.init_loss()

    # =========================================================================
    # Model building and configuration (NEW - for model-agnostic trainer)
    # =========================================================================

    @abstractmethod
    def _build_models(self) -> Dict[str, nn.Module]:
        """Build ALL models for this problem.

        Returns:
            Dict with keys like 'enc', 'u', 'a', 'nf' (problem-specific)
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_inputs(self, batch_size: int) -> Dict[str, Dict[str, jax.Array]]:
        """Return sample inputs for each model for initialization.

        This allows each problem to define its own model structure without
        the trainer needing to know about specific model names or input shapes.

        Args:
            batch_size: Batch size to use for sample inputs

        Returns:
            Dict mapping model name -> dict of sample inputs
            Example for DarcyContinuous:
                {
                    'enc': {'x': sample_a},
                    'u': {'x': sample_x, 'a': sample_beta},
                    'a': {'x': sample_x, 'a': sample_beta},
                    'nf': {'x': sample_beta},
                }
        """
        raise NotImplementedError

    @abstractmethod
    def get_weight_decay_groups(self) -> Dict[str, bool]:
        """Return dict mapping model name -> whether it should have weight decay.

        This allows each problem to specify which models should have weight decay
        without hardcoding in the trainer.

        Returns:
            Dict mapping model name -> True if should have weight decay
            Example:
                {'enc': False, 'u': True, 'a': True, 'nf': False}
        """
        raise NotImplementedError

    def get_batch_keys(self) -> List[str]:
        """Return keys to extract from train_data for batching.

        Override if problem needs different data keys.

        Returns:
            List of keys to extract from train_data/test_data
        """
        return ['a', 'u', 'x']

    def initialize_models(
            self,
            sample_inputs: Dict[str, Dict[str, jax.Array]]
    ) -> None:
        """Initialize model parameters

        Args:
            sample_inputs: Dict mapping model names to their sample inputs
        """
        self.rng, *init_rngs = random.split(self.rng, len(self.models) + 1)

        for (name, model), init_rng in zip(self.models.items(), init_rngs):
            if name in sample_inputs:
                variables = model.init(init_rng, **sample_inputs[name])
                self.params[name] = variables['params']
                print(f"  Initialized {name}: {self._count_params(self.params[name]):,} params")
            else:
                raise ValueError(f"No sample input for model '{name}'")

    def _count_params(self, params) -> int:
        """Count parameters in pytree"""
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    # =========================================================================
    # Training interface (NEW - for model-agnostic trainer)
    # =========================================================================

    def compute_training_losses(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jnp.ndarray],
            rng: jax.Array,
            loss_weights: Any
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute all training losses for a batch.

        This is the main training interface that the trainer calls.
        Default implementation uses loss_pde, loss_data, and NF loss.
        Override for problems with different loss structure.

        Args:
            params: All model parameters
            batch: Dict with batch data (keys from get_batch_keys())
            rng: PRNG key
            loss_weights: Loss weight configuration (has .pde, .data attributes)

        Returns:
            (total_loss, metrics_dict) where metrics_dict has individual losses
        """
        # Default implementation for IGNO-style training
        a = batch['a']
        u = batch['u']
        x = batch['x']

        # Encode a -> beta (needed for NF loss)
        beta = self.models['enc'].apply({'params': params['enc']}, a)

        # PDE loss
        loss_pde = self.loss_pde(params, a, rng)

        # Data/reconstruction loss
        loss_data = self.loss_data(params, x, a, u)

        # NF loss on DETACHED beta (key insight: gradients don't flow to encoder)
        beta_detached = jax.lax.stop_gradient(beta)
        loss_nf = self.models['nf'].apply(
            {'params': params['nf']},
            beta_detached,
            method=self.models['nf'].loss
        )

        # Total loss
        total_loss = loss_weights.pde * loss_pde + loss_weights.data * loss_data + loss_nf

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
            batch: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Compute evaluation metrics for a batch.

        Args:
            params: All model parameters
            batch: Dict with batch data

        Returns:
            Dict with evaluation metrics
        """
        a = batch['a']
        u = batch['u']
        x = batch['x']

        # Compute error
        error = self.error(params, x, a, u)

        # Compute NF loss
        beta = self.models['enc'].apply({'params': params['enc']}, a)
        nf_loss = self.models['nf'].apply(
            {'params': params['nf']},
            beta,
            method=self.models['nf'].loss
        )

        return {
            'error': jnp.mean(error),
            'nf_loss': nf_loss,
        }

    def run_diagnostics(
            self,
            epoch: int,
            params: Dict[str, Any],
            sample_data: Dict[str, jnp.ndarray],
            logger: Callable[[str, float, int], None]
    ) -> None:
        """Run problem-specific diagnostics.

        Default implementation: NF health check.
        Override for problems needing different diagnostics.

        Args:
            epoch: Current epoch
            params: Model parameters
            sample_data: Sample data for diagnostics (e.g., first batch)
            logger: Logging function (tag, value, step) -> None
        """
        self._run_nf_diagnostics(epoch, params, sample_data['a'], logger)

    def _run_nf_diagnostics(
        self,
        epoch: int,
        params: Dict[str, Any],
        a_sample: jnp.ndarray,
        logger: Callable[[str, float, int], None]
    ) -> None:
        """Detailed monitoring of NF health

        Checks:
        - Z-space statistics (should be ~N(0,1))
        - Dead dimensions (std < 0.1)
        - Invertibility (roundtrip error: beta -> z -> beta_rec)
        - Inverse sampling check (z -> beta from prior samples)
        - Beta hypercube violations (values outside [-1, 1])
        - Log-det stability
        """
        nf = self.models['nf']
        enc = self.models['enc']

        # Get latents from encoder
        beta = enc.apply({'params': params['enc']}, a_sample)

        # Forward through NF: beta -> z
        z_out, log_det_fwd = nf.apply(
            {'params': params['nf']},
            beta,
            method=nf.__call__
        )

        # Stats for Z-space (should be close to N(0,1))
        z_mean_total = float(jnp.mean(z_out))
        z_std_total = float(jnp.std(z_out))
        z_std_per_dim = jnp.std(z_out, axis=0)
        dead_dims = int(jnp.sum(z_std_per_dim < 0.1))
        exploding_dims = int(jnp.sum(z_std_per_dim > 5.0))

        # Invertibility check: beta -> z -> beta_rec
        beta_rec, _ = nf.apply(
            {'params': params['nf']},
            z_out,
            method=nf.inverse
        )
        rec_err = float(jnp.mean(jnp.abs(beta - beta_rec)))

        # === NEW: Inverse sampling check (z ~ N(0,1) -> beta) ===
        # Sample fresh z from standard normal and map to beta space
        self.rng, sample_key = random.split(self.rng)
        n_samples = a_sample.shape[0]
        z_prior = random.normal(sample_key, (n_samples, beta.shape[1]))

        beta_sampled, log_det_inv = nf.apply(
            {'params': params['nf']},
            z_prior,
            method=nf.inverse
        )

        # === NEW: Hypercube analysis for encoded betas ===
        beta_min = float(jnp.min(beta))
        beta_max = float(jnp.max(beta))
        beta_outside_hypercube = jnp.abs(beta) > 1.0
        n_violations_enc = int(jnp.sum(beta_outside_hypercube))
        frac_violations_enc = float(jnp.mean(beta_outside_hypercube))

        # Per-dimension analysis for encoded betas
        dims_with_violations_enc = int(jnp.sum(jnp.any(beta_outside_hypercube, axis=0)))

        # === NEW: Hypercube analysis for sampled betas ===
        beta_sampled_min = float(jnp.min(beta_sampled))
        beta_sampled_max = float(jnp.max(beta_sampled))
        beta_sampled_outside = jnp.abs(beta_sampled) > 1.0
        n_violations_sampled = int(jnp.sum(beta_sampled_outside))
        frac_violations_sampled = float(jnp.mean(beta_sampled_outside))

        # Per-dimension analysis for sampled betas
        dims_with_violations_sampled = int(jnp.sum(jnp.any(beta_sampled_outside, axis=0)))

        # Log-det analysis
        ldj_mean = float(jnp.mean(log_det_fwd))
        ldj_std = float(jnp.std(log_det_fwd))
        ldj_inv_mean = float(jnp.mean(log_det_inv))
        ldj_inv_std = float(jnp.std(log_det_inv))

        # Logging to TensorBoard
        logger("nf_health/z_mean_avg", z_mean_total, epoch)
        logger("nf_health/z_std_avg", z_std_total, epoch)
        logger("nf_health/dead_dims", float(dead_dims), epoch)
        logger("nf_health/exploding_dims", float(exploding_dims), epoch)
        logger("nf_health/rec_error_abs", rec_err, epoch)
        logger("nf_health/log_det_mean", ldj_mean, epoch)
        logger("nf_health/log_det_std", ldj_std, epoch)

        # NEW: Hypercube metrics
        logger("nf_health/beta_enc_min", beta_min, epoch)
        logger("nf_health/beta_enc_max", beta_max, epoch)
        logger("nf_health/beta_enc_violations_frac", frac_violations_enc, epoch)
        logger("nf_health/beta_enc_dims_with_violations", float(dims_with_violations_enc), epoch)

        logger("nf_health/beta_sampled_min", beta_sampled_min, epoch)
        logger("nf_health/beta_sampled_max", beta_sampled_max, epoch)
        logger("nf_health/beta_sampled_violations_frac", frac_violations_sampled, epoch)
        logger("nf_health/beta_sampled_dims_with_violations", float(dims_with_violations_sampled), epoch)

        logger("nf_health/log_det_inv_mean", ldj_inv_mean, epoch)
        logger("nf_health/log_det_inv_std", ldj_inv_std, epoch)

        # Console output
        print(f"  [NF Health] Roundtrip Err: {rec_err:.2e} | "
              f"Dead Dims: {dead_dims}/{z_out.shape[1]} | "
              f"Exploding Dims: {exploding_dims}/{z_out.shape[1]}")
        print(f"  [NF Health] Z-Space: Mean={z_mean_total:.3f}, Std={z_std_total:.3f} | "
              f"LogDet: mean={ldj_mean:.2f}, std={ldj_std:.2f}")
        print(f"  [NF Health] Beta (enc): range=[{beta_min:.3f}, {beta_max:.3f}] | "
              f"Outside [-1,1]: {frac_violations_enc*100:.1f}% ({dims_with_violations_enc}/{beta.shape[1]} dims)")
        print(f"  [NF Health] Beta (sampled): range=[{beta_sampled_min:.3f}, {beta_sampled_max:.3f}] | "
              f"Outside [-1,1]: {frac_violations_sampled*100:.1f}% ({dims_with_violations_sampled}/{beta.shape[1]} dims)")

        # Warnings
        if dead_dims > (z_out.shape[1] // 2):
            print("  ⚠️ WARNING: High number of dead dimensions detected. "
                  "Flow might be collapsing.")
        if exploding_dims > 0:
            print("  ⚠️ WARNING: Exploding dimensions detected. "
                  "Check for numerical instability.")
        if rec_err > 1e-3:
            print("  ⚠️ WARNING: Poor invertibility. "
                  "Check for vanishing/exploding gradients in NF.")
        if abs(z_mean_total) > 0.5:
            print("  ⚠️ WARNING: Z-space mean far from 0. "
                  "Flow may not be mapping to standard normal.")
        if z_std_total < 0.5 or z_std_total > 2.0:
            print("  ⚠️ WARNING: Z-space std far from 1. "
                  "Flow may not be mapping to standard normal.")

        # NEW: Hypercube warnings
        if frac_violations_enc > 0.1:
            print(f"  ⚠️ WARNING: {frac_violations_enc*100:.1f}% of encoded betas outside [-1,1]. "
                  "Encoder may need regularization or output activation.")
        if frac_violations_sampled > 0.1:
            print(f"  ⚠️ WARNING: {frac_violations_sampled*100:.1f}% of NF-sampled betas outside [-1,1]. "
                  "Flow prior may not match encoder distribution well.")
        if abs(beta_sampled_max) > 3.0 or abs(beta_sampled_min) > 3.0:
            print(f"  ⚠️ WARNING: Sampled betas have extreme values (range [{beta_sampled_min:.2f}, {beta_sampled_max:.2f}]). "
                  "NF may be producing out-of-distribution samples.")

    # =========================================================================
    # Checkpoint management
    # =========================================================================

    def save_checkpoint(
            self,
            path: Path,
            epoch: int,
            opt_states: Optional[Dict] = None,
            metric: Optional[float] = None,
            metric_name: Optional[str] = None,
            extra: Optional[Dict] = None,
    ) -> Path:
        """Save checkpoint"""
        from src.utils.solver_utils import save_checkpoint as save_ckpt

        state = {
            'params': self.params,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
        }

        if opt_states is not None:
            state['opt_states'] = opt_states

        metadata = {}
        if metric is not None:
            metadata['metric'] = float(metric)
            metadata['metric_name'] = metric_name
        if extra is not None:
            metadata.update(extra)

        save_ckpt(path, state, metadata)
        return path

    def load_checkpoint(
            self,
            path: Path,
            models_to_load: Optional[List[str]] = None,
            load_opt_states: bool = False,
    ) -> Dict[str, Any]:
        """Load checkpoint into parameters"""
        from src.utils.solver_utils import load_checkpoint as load_ckpt

        state, metadata = load_ckpt(path)

        print(f"Loading checkpoint: {path}")

        # Load parameters
        to_load = models_to_load if models_to_load else list(self.models.keys())
        for name in to_load:
            if name in state['params']:
                self.params[name] = state['params'][name]
                print(f"  Loaded {name}")

        result = {'epoch': state.get('epoch', 0), **metadata}

        if load_opt_states and 'opt_states' in state:
            result['opt_states'] = state['opt_states']

        return result

    # =========================================================================
    # NF-related methods (for Bayesian inference)
    # =========================================================================

    def sample_latent_from_nf(
            self,
            params: Dict[str, Any],
            num_samples: int,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Sample from NF prior.

        Args:
            params: Model parameters (must include 'nf')
            num_samples: Number of samples to draw
            rng: PRNG key

        Returns:
            beta: Latent samples (num_samples, latent_dim)
        """
        nf = self.models['nf']
        # Note: nf.sample signature is (rng, num_samples)
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
        """Compute log p(beta) using NF.

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
    # Core loss methods (JIT-compiled pure functions)
    # =========================================================================

    @abstractmethod
    def loss_pde(
            self,
            params: Dict[str, Any],
            a: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Compute PDE residual loss (JIT-compilable)

        First encodes a -> beta, then computes PDE loss.
        Used during training.

        Args:
            params: All model parameters
            a: Coefficient field (batch, n_points, 1)
            rng: PRNG key for collocation point sampling

        Returns:
            PDE loss (scalar)
        """
        raise NotImplementedError

    @abstractmethod
    def loss_data(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute data fitting loss (JIT-compilable)

        First encodes a -> beta, then computes reconstruction loss.
        Used during training.

        Args:
            params: All model parameters
            x: Coordinates (batch, n_points, dim)
            a: Coefficient field (batch, n_points, 1)
            u: Solution field (batch, n_points, 1)

        Returns:
            Data loss (scalar)
        """
        raise NotImplementedError

    @abstractmethod
    def error(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute error metric (JIT-compilable)

        Args:
            params: All model parameters
            x: Coordinates
            a: Coefficient field
            u: Solution field (target)

        Returns:
            Error (batch,) or scalar
        """
        raise NotImplementedError

    # =========================================================================
    # From-beta methods (for inversion - skip encoding step)
    # =========================================================================

    @abstractmethod
    def loss_pde_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Compute PDE loss from beta directly (skips encoding).

        Used during inversion when optimizing beta directly.
        Uses DECODED coefficient field from decoder.

        Args:
            params: All model parameters
            beta: Latent representation (batch, latent_dim)
            rng: PRNG key for collocation points

        Returns:
            PDE loss (scalar)
        """
        raise NotImplementedError

    @abstractmethod
    def loss_data_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        """Compute data loss from beta directly (skips encoding).

        Used during inversion when optimizing beta directly.

        Args:
            params: All model parameters
            beta: Latent representation (batch, latent_dim)
            x: Coordinates (batch, n_points, dim)
            target: Target values (batch, n_points, 1)
            target_type: 'a' for coefficient, 'u' for solution

        Returns:
            Data loss (scalar)
        """
        raise NotImplementedError

    @abstractmethod
    def error_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        """Compute error from beta directly.

        Args:
            params: All model parameters
            beta: Latent representation (batch, latent_dim)
            x: Coordinates
            target: Target values
            target_type: 'a' or 'u'

        Returns:
            Error metric
        """
        raise NotImplementedError

    def predict_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Predict u and a from beta.

        Args:
            params: All model parameters
            beta: Latent representation (batch, latent_dim)
            x: Coordinates (batch, n_points, dim)

        Returns:
            Dict with 'u_pred' and 'a_pred'
        """
        raise NotImplementedError

    # =========================================================================
    # Observation utilities
    # =========================================================================

    def add_noise_snr(
            self,
            signal: jnp.ndarray,
            snr_db: float,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Add Gaussian noise to achieve target SNR.

        SNR = 10 * log10(signal_power / noise_power)

        Args:
            signal: Clean signal
            snr_db: Target SNR in decibels
            rng: PRNG key

        Returns:
            Noisy signal
        """
        signal_power = jnp.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = jnp.sqrt(noise_power)
        return signal + random.normal(rng, signal.shape) * noise_std

    def sample_observation_indices(
            self,
            n_total: int,
            n_obs: int,
            method: str = "random",
            rng: jax.Array = None
    ) -> jnp.ndarray:
        """Sample observation point indices.

        Args:
            n_total: Total number of grid points
            n_obs: Number of observations to sample
            method: "random" or "grid"
            rng: PRNG key (required for random)

        Returns:
            Sorted indices array
        """

        if method == "random":
            if rng is None:
                raise ValueError("rng required for random sampling")
            indices = random.choice(rng, n_total, (n_obs,), replace=False)
            return jnp.sort(indices)
        elif method == "grid":
            step = max(1, n_total // n_obs)
            return jnp.arange(0, n_total, step)[:n_obs]
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    @abstractmethod
    def prepare_observations(
            self,
            sample_indices: List[int],
            obs_indices: jnp.ndarray,
            snr_db: float = None,
            rng: jax.Array = None
    ) -> Dict[str, jnp.ndarray]:
        """Prepare observation data for test samples.

        Args:
            sample_indices: Indices into test set
            obs_indices: Observation point indices
            snr_db: SNR for noise (None for clean)
            rng: PRNG key for noise

        Returns:
            Dict with x_full, x_obs, u_obs, u_true, a_true
        """
        raise NotImplementedError

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_train_data(self) -> Dict[str, jnp.ndarray]:
        return self.train_data

    def get_test_data(self) -> Dict[str, jnp.ndarray]:
        return self.test_data

    def init_error(
            self,
            err_type: str = 'lp_rel',
            d: int = 2,
            p: int = 2
    ) -> None:
        self.get_error = MyError(d=d, p=p)(err_type)

    def init_loss(self, loss_type: str = 'mse_org') -> None:
        self.get_loss = MyLoss()(loss_type)

    def pre_train_check(self) -> None:
        if self.get_loss is None or self.get_error is None:
            self.init_loss()
            self.init_error()

    @abstractmethod
    def get_n_test_samples(self) -> int:
        """Get number of test samples"""
        raise NotImplementedError

    @abstractmethod
    def get_n_points(self) -> int:
        """Get number of grid points per sample"""
        raise NotImplementedError


def create_problem(config, load_train_data: bool = True) -> ProblemInstance:
    """Create problem from config

    Args:
        config: TrainingConfig
        load_train_data: If False, only load test data (for evaluation)

    Returns:
        Initialized ProblemInstance
    """
    problem_cls = get_problem_class(config.problem.type)
    return problem_cls(
        seed=config.seed,
        train_data_path=config.problem.train_data if load_train_data else None,
        test_data_path=config.problem.test_data,
    )


# Auto-register problems
from src.problems.darcy_continuous import DarcyContinuous
