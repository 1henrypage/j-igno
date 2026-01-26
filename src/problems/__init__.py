# src/problems/__init__.py
# JAX version - streamlined without standardization
# Fixes:
# 1. Added sample_latent_from_nf and log_prob_latent methods back (needed for Bayesian inference)
# 2. Better documentation

"""
Problem instances for JAX.

Each problem:
- Loads data
- Builds models (Flax modules)
- Defines JIT-compiled loss functions
- Handles checkpoints
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
    # Model building
    # =========================================================================

    @abstractmethod
    def _build_models(self) -> Dict[str, nn.Module]:
        """Build ALL models for this problem.

        Returns:
            Dict with keys like 'enc', 'u', 'a', 'nf'
        """
        raise NotImplementedError

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
    # These are needed for MCMC sampling even without standardization
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
