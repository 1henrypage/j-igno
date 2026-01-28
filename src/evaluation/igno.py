# src/evaluation/igno.py
"""
IGNO-style gradient-based inversion in JAX.

Given sparse noisy observations, optimize beta to minimize
PDE loss + data loss, starting from NF sample.

Supports batched inversion for processing multiple samples simultaneously.
"""
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
from tqdm import trange
from typing import Dict, Any, Optional
from functools import partial

from src.problems import ProblemInstance
from src.solver.config import InversionConfig


class IGNOInverter:
    """
    IGNO gradient-based inversion in JAX.

    Process:
    1. Sample z ~ N(0, I) for each sample in batch
    2. Pass through NF inverse to get initial betas
    3. Optimize betas via gradient descent on: w_pde * L_pde + w_data * L_data
    4. Return optimized betas

    Supports batched inversion where multiple independent samples are
    optimized in parallel for significant speedup.

    Note: In JAX, all models are already "frozen" since we use functional
    updates - we simply don't update their parameters during inversion.
    """

    def __init__(self, problem: ProblemInstance, rng: jax.Array):
        """
        Initialize inverter.

        Args:
            problem: ProblemInstance with trained models and params
            rng: PRNG key for sampling
        """
        self.problem = problem
        self.rng = rng

        # Store frozen params (we won't update these)
        self.frozen_params = problem.params

    def invert(
            self,
            x_obs: jnp.ndarray,
            u_obs: jnp.ndarray,
            x_full: jnp.ndarray,
            config: InversionConfig,
            verbose: bool = False,
    ) -> jnp.ndarray:
        """
        Run gradient-based inversion on one or more samples.

        Supports batched inversion where multiple independent samples are
        optimized simultaneously for significant speedup.

        Args:
            x_obs: Observation coordinates (batch, n_obs, 2)
            u_obs: Noisy observations (batch, n_obs, 1)
            x_full: Full grid coordinates (batch, n_points, 2) - for PDE loss
            config: Inversion configuration
            verbose: Print progress

        Returns:
            Optimized beta (batch, latent_dim)
        """
        batch_size = x_obs.shape[0]

        # Initialize: sample from NF prior
        self.rng, sample_rng = random.split(self.rng)
        beta_init = self.problem.sample_latent_from_nf(
            params=self.frozen_params,
            num_samples=batch_size,
            rng=sample_rng
        )

        # Create optimizer
        optimizer = self._create_optimizer(config)
        opt_state = optimizer.init(beta_init)

        # Get loss weights
        weights = config.loss_weights

        # Create JIT-compiled update step
        @jit
        def loss_fn(beta, rng_key):
            """Compute inversion loss."""
            # PDE loss (uses decoded coefficient)
            loss_pde = self.problem.loss_pde_from_beta(
                self.frozen_params, beta, rng_key
            )

            # Data loss (predicted u at obs points vs observed u)
            loss_data = self.problem.loss_data_from_beta(
                self.frozen_params, beta, x_obs, u_obs, target_type='u'
            )

            # Total loss
            total_loss = weights.pde * loss_pde + weights.data * loss_data

            return total_loss, {'loss_pde': loss_pde, 'loss_data': loss_data}

        @jit
        def update_step(beta, opt_state, rng_key):
            """Single optimization step."""
            (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(beta, rng_key)
            updates, new_opt_state = optimizer.update(grads, opt_state, beta)
            new_beta = optax.apply_updates(beta, updates)
            return new_beta, new_opt_state, loss, aux

        # Optimization loop
        beta = beta_init

        iterator = trange(config.epochs, desc="Inverting", disable=not verbose)
        for epoch in iterator:
            self.rng, step_rng = random.split(self.rng)
            beta, opt_state, loss, aux = update_step(beta, opt_state, step_rng)

            if verbose and (epoch + 1) % 100 == 0:
                iterator.set_postfix({
                    'loss': f'{float(loss):.4f}',
                    'pde': f'{float(aux["loss_pde"]):.4f}',
                    'data': f'{float(aux["loss_data"]):.4f}',
                })

        return beta

    def _create_optimizer(self, config: InversionConfig) -> optax.GradientTransformation:
        """Create optimizer for inversion."""
        opt_cfg = config.optimizer

        # Base learning rate
        lr = opt_cfg.lr

        # Create schedule if specified
        if config.scheduler is not None and config.scheduler.type is not None:
            sched_cfg = config.scheduler

            if sched_cfg.type == 'StepLR':
                # For inversion, step_size is already in optimizer steps (not epochs)
                schedule = optax.exponential_decay(
                    init_value=lr,
                    transition_steps=sched_cfg.step_size,
                    decay_rate=sched_cfg.gamma,
                    staircase=True
                )
                lr = schedule
            elif sched_cfg.type == 'CosineAnnealing':
                schedule = optax.cosine_decay_schedule(
                    init_value=lr,
                    decay_steps=config.epochs,
                    alpha=sched_cfg.eta_min / lr if sched_cfg.eta_min else 0.0
                )
                lr = schedule

        # Create optimizer
        OPTIMIZERS = {
            'Adam': optax.adam,
            'AdamW': optax.adamw,
            'SGD': optax.sgd,
        }

        opt_type = opt_cfg.type
        if opt_type not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        return OPTIMIZERS[opt_type](learning_rate=lr)
