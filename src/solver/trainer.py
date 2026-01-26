# src/solver/trainer.py
# JAX trainer with JIT-compiled training step
# Fixes:
# 1. Updated to work with fixed GenPoints (key passing)
# 2. Improved NF diagnostics
# 3. Better documentation

"""
IGNO Trainer for JAX.

Key differences from PyTorch:
- Training step is JIT-compiled
- Uses functional updates (no mutable state)
- Optimizer states handled via optax
- NF trained with stop_gradient (JAX equivalent of .detach())
"""

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import time
from typing import Optional, Dict, Any, Tuple
from tqdm import trange
from datetime import datetime
from pathlib import Path
from functools import partial

from tensorboardX import SummaryWriter

from src.solver.config import TrainingConfig
from src.problems import ProblemInstance
from src.utils.solver_utils import create_train_state, create_data_batches


class IGNOTrainer:
    """
    JAX trainer for IGNO: encoder + decoders + NF jointly.

    Training loop is JIT-compiled for maximum performance.
    NF trained with stop_gradient on latents (prevents gradients to encoder).
    """

    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.dtype = problem.dtype

        # Training state (will be created in setup)
        self.train_state: Optional[Dict] = None
        self.writer: Optional[SummaryWriter] = None

        # Directories
        self.run_dir: Optional[Path] = None
        self.weights_dir: Optional[Path] = None
        self.tb_dir: Optional[Path] = None

    def _create_run_dir(self, config: TrainingConfig) -> Path:
        """Create timestamped run directory"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = Path(config.artifact_root) / f"{timestamp}_{config.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _setup_directories(self) -> None:
        """Setup directories for weights and tensorboard"""
        if self.run_dir is None:
            raise RuntimeError("run_dir must be set")

        self.weights_dir = self.run_dir / "weights"
        self.tb_dir = self.run_dir / "tensorboard"

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def _setup_tensorboard(self) -> None:
        """Initialize TensorBoard writer"""
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def _log(self, tag: str, value: float, step: int) -> None:
        """Log to TensorBoard"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def setup(self, config: TrainingConfig, pretrained_path: Optional[Path] = None) -> None:
        """Setup trainer and initialize models"""
        self.problem.pre_train_check()

        # Create run directory
        self.run_dir = self._create_run_dir(config)
        self.problem.run_dir = self.run_dir

        # Setup directories and tensorboard
        self._setup_directories()
        self._setup_tensorboard()

        # Initialize model parameters if not already done
        if not self.problem.params:
            print("Initializing model parameters...")
            # Create sample inputs for initialization
            sample_a = self.problem.train_data['a'][:1]
            sample_x = self.problem.train_data['x'][:1]
            sample_beta = jnp.ones((1, self.problem.BETA_SIZE))

            sample_inputs = {
                'enc': {'x': sample_a},
                'u': {'x': sample_x, 'a': sample_beta},
                'a': {'x': sample_x, 'a': sample_beta},
                'nf': {'x': sample_beta},
            }
            self.problem.initialize_models(sample_inputs)

        # Load pretrained if specified
        ckpt_path = pretrained_path or config.get_pretrained_path()
        if ckpt_path:
            if not ckpt_path.exists():
                raise RuntimeError("Couldn't find pretrained checkpoint")
            print(f"Loading pretrained: {ckpt_path}")
            self.problem.load_checkpoint(ckpt_path)

        # Create training state
        cfg = config.training
        num_steps = cfg.epochs * (len(self.problem.train_data['a']) // cfg.batch_size)

        # Sample inputs for optimizer initialization
        sample_a = self.problem.train_data['a'][:cfg.batch_size]
        sample_x = self.problem.train_data['x'][:cfg.batch_size]
        sample_u = self.problem.train_data['u'][:cfg.batch_size]

        sample_inputs = {
            'enc': {'x': sample_a},
            'u': {'x': sample_x, 'a': jnp.ones((cfg.batch_size, self.problem.BETA_SIZE))},
            'a': {'x': sample_x, 'a': jnp.ones((cfg.batch_size, self.problem.BETA_SIZE))},
            'nf': {'x': jnp.ones((cfg.batch_size, self.problem.BETA_SIZE))},
        }

        self.train_state = create_train_state(
            models=self.problem.models,
            rng=self.problem.rng,
            sample_inputs=sample_inputs,
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            num_steps=num_steps
        )

        # Update problem params with initialized params
        self.problem.params = self.train_state['params']

        # Save config
        config.save(self.run_dir / "config.yaml")
        print(f"Run directory: {self.run_dir}")

    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        """Train IGNO with JIT-compiled training loop"""
        cfg = config.training
        train_data = self.problem.get_train_data()
        test_data = self.problem.get_test_data()

        # Create JIT-compiled training and evaluation functions
        train_step_fn = self._create_train_step(cfg)
        eval_step_fn = self._create_eval_step()

        t_start = time.time()
        best_error = float('inf')

        print("\n" + "=" * 60)
        print("IGNO Joint Training (encoder + decoders + NF)")
        print("=" * 60)
        print(f"Loss weights: pde={cfg.loss_weights.pde}, data={cfg.loss_weights.data}")
        print(f"Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}")
        print("=" * 60 + "\n")

        for epoch in trange(cfg.epochs, desc="Training"):
            # Create batches for this epoch
            self.train_state['rng'], data_rng = random.split(self.train_state['rng'])
            train_batches, n_train = create_data_batches(
                train_data['a'], train_data['u'], train_data['x'],
                batch_size=cfg.batch_size,
                shuffle=True,
                rng=data_rng
            )

            # Training loop
            loss_sum = 0.
            pde_sum = 0.
            data_sum = 0.
            nf_sum = 0.

            for batch_a, batch_u, batch_x in train_batches:
                # Get RNG for this step
                self.train_state['rng'], step_rng = random.split(self.train_state['rng'])

                new_params, new_opt_states, metrics = train_step_fn(
                    self.train_state['params'],
                    self.train_state['opt_states'],
                    batch_a,
                    batch_u,
                    batch_x,
                    step_rng
                )
                self.train_state['params'] = new_params
                self.train_state['opt_states'] = new_opt_states

                loss_sum += float(metrics['loss'])
                pde_sum += float(metrics['loss_pde'])
                data_sum += float(metrics['loss_data'])
                nf_sum += float(metrics['loss_nf'])

            # Update problem params
            self.problem.params = self.train_state['params']

            # Evaluation
            test_batches, n_test = create_data_batches(
                test_data['a'], test_data['u'], test_data['x'],
                batch_size=cfg.batch_size,
                shuffle=False
            )

            error_sum = 0.
            test_nf_sum = 0.

            for batch_a, batch_u, batch_x in test_batches:
                metrics = eval_step_fn(
                    self.train_state['params'],
                    batch_a,
                    batch_u,
                    batch_x
                )
                error_sum += float(metrics['error'])
                test_nf_sum += float(metrics['nf_loss'])

            # Compute averages
            avg_loss = loss_sum / n_train
            avg_error = error_sum / n_test
            avg_nf = nf_sum / n_train
            avg_test_nf = test_nf_sum / n_test

            # Logging
            self._log("train/loss", avg_loss, epoch)
            self._log("train/pde", pde_sum / n_train, epoch)
            self._log("train/data", data_sum / n_train, epoch)
            self._log("train/nf", avg_nf, epoch)
            self._log("test/error", avg_error, epoch)
            self._log("test/nf", avg_test_nf, epoch)

            # Save best checkpoint
            if avg_error < best_error:
                best_error = avg_error
                self.problem.save_checkpoint(
                    self.weights_dir / 'best.pt',
                    epoch=epoch,
                    opt_states=self.train_state['opt_states'],
                    metric=float(avg_error),
                    metric_name='error'
                )

            # Print progress
            if (epoch + 1) % cfg.epoch_show == 0:
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Loss: {avg_loss:.4f} (pde={pde_sum / n_train:.4f}, "
                      f"data={data_sum / n_train:.4f}, nf={avg_nf:.4f})")
                print(f"  Test Error: {avg_error:.4f}, Test NF NLL: {avg_test_nf:.4f}")

                # NF diagnostics
                self._run_nf_diagnostics(epoch, train_data['a'][:cfg.batch_size])

            self.train_state['step'] += 1

        # Save last checkpoint
        self.problem.save_checkpoint(
            self.weights_dir / 'last.pt',
            epoch=cfg.epochs - 1,
            opt_states=self.train_state['opt_states']
        )

        total_time = time.time() - t_start
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best error: {best_error:.4f}")
        print(f"Checkpoints saved to: {self.weights_dir}")

        return {
            'best_error': best_error,
            'time': total_time,
        }

    def _create_train_step(self, cfg):
        """Create JIT-compiled training step"""
        problem = self.problem
        weights = cfg.loss_weights

        # Keep optimizers outside JIT - access via closure
        optimizers = self.train_state['optimizers']

        @jit
        def train_step(params, opt_states, batch_a, batch_u, batch_x, rng):
            """Single training step (JIT-compiled)

            Args:
                params: Model parameters dict
                opt_states: Optimizer states dict
                batch_a: Coefficient field (batch, n_points, 1)
                batch_u: Solution field (batch, n_points, 1)
                batch_x: Coordinates (batch, n_points, 2)
                rng: PRNG key

            Returns:
                (new_params, new_opt_states, metrics)
            """
            rng_pde, rng_enc = random.split(rng)

            def loss_fn(params_dict):
                """Combined loss function"""
                beta = problem.models['enc'].apply(
                    {'params': params_dict['enc']},
                    batch_a
                )
                loss_pde = problem.loss_pde(params_dict, batch_a, rng_pde)
                loss_data = problem.loss_data(params_dict, batch_x, batch_a, batch_u)

                beta_detached = jax.lax.stop_gradient(beta)
                loss_nf = problem.models['nf'].apply(
                    {'params': params_dict['nf']},
                    beta_detached,
                    method=problem.models['nf'].loss
                )

                total_loss = weights.pde * loss_pde + weights.data * loss_data + loss_nf
                return total_loss, {
                    'loss': total_loss,
                    'loss_pde': loss_pde,
                    'loss_data': loss_data,
                    'loss_nf': loss_nf,
                }

            (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(params)

            new_params = {}
            new_opt_states = {}
            for name in params.keys():
                updates, new_opt_state = optimizers[name].update(
                    grads[name],
                    opt_states[name],
                    params[name]
                )
                new_params[name] = optax.apply_updates(params[name], updates)
                new_opt_states[name] = new_opt_state

            return new_params, new_opt_states, metrics

        return train_step

    def _create_eval_step(self):
        """Create JIT-compiled evaluation step"""
        problem = self.problem

        @jit
        def eval_step(params, batch_a, batch_u, batch_x):
            """Single evaluation step (JIT-compiled)"""
            # Compute error
            error = problem.error(params, batch_x, batch_a, batch_u)

            # Compute NF loss
            beta = problem.models['enc'].apply({'params': params['enc']}, batch_a)
            nf_loss = problem.models['nf'].apply(
                {'params': params['nf']},
                beta,
                method=problem.models['nf'].loss
            )

            return {
                'error': jnp.mean(error),
                'nf_loss': nf_loss,
            }

        return eval_step

    def _run_nf_diagnostics(self, epoch: int, a_sample: jnp.ndarray) -> None:
        """Detailed monitoring of NF health

        Checks:
        - Z-space statistics (should be ~N(0,1))
        - Dead dimensions (std < 0.1)
        - Invertibility (roundtrip error)
        - Log-det stability
        """
        nf = self.problem.models['nf']
        enc = self.problem.models['enc']
        params = self.train_state['params']

        # Get latents
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

        # Log-det analysis
        ldj_mean = float(jnp.mean(log_det_fwd))
        ldj_std = float(jnp.std(log_det_fwd))

        # Logging to TensorBoard
        self._log("nf_health/z_mean_avg", z_mean_total, epoch)
        self._log("nf_health/z_std_avg", z_std_total, epoch)
        self._log("nf_health/dead_dims", float(dead_dims), epoch)
        self._log("nf_health/exploding_dims", float(exploding_dims), epoch)
        self._log("nf_health/rec_error_abs", rec_err, epoch)
        self._log("nf_health/log_det_mean", ldj_mean, epoch)
        self._log("nf_health/log_det_std", ldj_std, epoch)

        # Console output
        print(f"  [NF Health] Roundtrip Err: {rec_err:.2e} | "
              f"Dead Dims: {dead_dims}/{z_out.shape[1]} | "
              f"Exploding Dims: {exploding_dims}/{z_out.shape[1]}")
        print(f"  [NF Health] Z-Space: Mean={z_mean_total:.3f}, Std={z_std_total:.3f} | "
              f"LogDet: mean={ldj_mean:.2f}, std={ldj_std:.2f}")

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

    def close(self) -> None:
        """Cleanup resources"""
        if self.writer:
            self.writer.close()
            self.writer = None
