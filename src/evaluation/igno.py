# src/evaluation/igno.py
"""
IGNO-style gradient-based inversion in JAX.
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
    """

    def __init__(self, problem: ProblemInstance, rng: jax.Array):
        self.problem = problem
        self.rng = rng
        self.frozen_params = problem.params

    def invert(
            self,
            x_obs: jnp.ndarray,
            u_obs: jnp.ndarray,
            x_full: jnp.ndarray,
            config: InversionConfig,
            verbose: bool = False,
    ) -> jnp.ndarray:
        batch_size = x_obs.shape[0]

        # Initialize: sample from NF prior
        # Initialize at mode: z=0 is the mode of Beta(α,α)
        z_mode = jnp.zeros((batch_size, self.problem.BETA_SIZE))
        beta_init, _ = self.problem.models['nf'].apply(
            {'params': self.frozen_params['nf']},
            z_mode,
            method=self.problem.models['nf'].inverse
        )
        # self.rng, sample_rng = random.split(self.rng)
        # beta_init = self.problem.sample_latent_from_nf(
        #     params=self.frozen_params,
        #     num_samples=batch_size,
        #     rng=sample_rng
        # )

        # DEBUG: Initial beta stats
        print("=" * 60)
        print("JAX DEBUG - INVERSION START")
        print("=" * 60)
        print(f"Initial beta shape: {beta_init.shape}")
        print(f"Initial beta mean: {float(jnp.mean(beta_init)):.6f}")
        print(f"Initial beta std: {float(jnp.std(beta_init)):.6f}")
        print(f"Initial beta min: {float(jnp.min(beta_init)):.6f}")
        print(f"Initial beta max: {float(jnp.max(beta_init)):.6f}")
        print(f"Initial beta: {beta_init}")

        # Create optimizer
        optimizer = self._create_optimizer(config)
        opt_state = optimizer.init(beta_init)

        # Get loss weights
        weights = config.loss_weights

        print(f"\nLoss weights: pde={weights.pde}, data={weights.data}")
        print(f"Optimizer: {config.optimizer.type}, lr={config.optimizer.lr}")

        # Create loss function (not JIT for debugging)
        def loss_fn(beta, rng_key):
            """Compute inversion loss."""
            loss_pde = self.problem.loss_pde_from_beta(
                self.frozen_params, beta, rng_key
            )
            loss_data = self.problem.loss_data_from_beta(
                self.frozen_params, beta, x_obs, u_obs, target_type='u'
            )
            total_loss = weights.pde * loss_pde + weights.data * loss_data
            return total_loss, {'loss_pde': loss_pde, 'loss_data': loss_data}

        # DEBUG: Compute initial losses
        self.rng, init_rng = random.split(self.rng)
        init_loss, init_aux = loss_fn(beta_init, init_rng)
        print(f"\nInitial losses (epoch 0):")
        print(f"  loss_pde: {float(init_aux['loss_pde']):.6f}")
        print(f"  loss_data: {float(init_aux['loss_data']):.6f}")
        print(f"  total: {float(init_loss):.6f}")
        print("=" * 60)

        # JIT compile update step
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

            # DEBUG: Print at specific epochs
            if epoch in [0, 1, 10, 50, 100, 250, 499]:
                print(f"\nEpoch {epoch}:")
                print(f"  loss_pde: {float(aux['loss_pde']):.6f}")
                print(f"  loss_data: {float(aux['loss_data']):.6f}")
                print(f"  total: {float(loss):.6f}")
                print(f"  beta mean: {float(jnp.mean(beta)):.6f}, std: {float(jnp.std(beta)):.6f}")

            if verbose and (epoch + 1) % 100 == 0:
                iterator.set_postfix({
                    'loss': f'{float(loss):.4f}',
                    'pde': f'{float(aux["loss_pde"]):.4f}',
                    'data': f'{float(aux["loss_data"]):.4f}',
                })

        # DEBUG: Final stats
        print("\n" + "=" * 60)
        print("JAX DEBUG - INVERSION END")
        print("=" * 60)
        print(f"Final beta mean: {float(jnp.mean(beta)):.6f}")
        print(f"Final beta std: {float(jnp.std(beta)):.6f}")
        print(f"Final beta[0, :5]: {beta[0, :5]}")

        # Compute final losses
        self.rng, final_rng = random.split(self.rng)
        final_loss, final_aux = loss_fn(beta, final_rng)
        print(f"Final loss_pde: {float(final_aux['loss_pde']):.6f}")
        print(f"Final loss_data: {float(final_aux['loss_data']):.6f}")
        print("=" * 60)

        return beta

    def _create_optimizer(self, config: InversionConfig) -> optax.GradientTransformation:
        """Create optimizer for inversion."""
        opt_cfg = config.optimizer
        lr = opt_cfg.lr

        if config.scheduler is not None and config.scheduler.type is not None:
            sched_cfg = config.scheduler

            if sched_cfg.type == 'StepLR':
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

        OPTIMIZERS = {
            'Adam': optax.adam,
            'AdamW': optax.adamw,
            'SGD': optax.sgd,
        }

        opt_type = opt_cfg.type
        if opt_type not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        return OPTIMIZERS[opt_type](learning_rate=lr)
