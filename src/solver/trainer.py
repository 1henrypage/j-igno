# src/solver/trainer.py
"""
IGNO Trainer for JAX

This trainer works with any ProblemInstance that implements the required interface:
- get_sample_inputs(batch_size) -> sample inputs for model initialization
- get_weight_decay_groups() -> which models get weight decay
- get_batch_keys() -> which data keys to batch
- compute_training_losses(params, batch, rng, loss_weights) -> losses
- compute_eval_metrics(params, batch) -> evaluation metrics
- run_diagnostics(epoch, params, sample_data, logger) -> diagnostics

"""

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import time
from typing import Optional, Dict, Any, Tuple, List
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
    JAX trainer for IGNO - Model Agnostic Version.

    Works with any ProblemInstance that implements the required interface.
    Training loop is JIT-compiled for maximum performance.
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

        cfg = config.training

        # Get sample inputs from problem (not hardcoded!)
        sample_inputs = self.problem.get_sample_inputs(cfg.batch_size)

        # Initialize model parameters if not already done
        if not self.problem.params:
            print("Initializing model parameters...")
            self.problem.initialize_models(sample_inputs)

        # Load pretrained if specified
        ckpt_path = pretrained_path or config.get_pretrained_path()
        if ckpt_path:
            if not ckpt_path.exists():
                raise RuntimeError("Couldn't find pretrained checkpoint")
            print(f"Loading pretrained: {ckpt_path}")
            self.problem.load_checkpoint(ckpt_path)

        # Calculate number of steps
        batch_keys = self.problem.get_batch_keys()
        first_key = batch_keys[0]
        n_train_samples = len(self.problem.train_data[first_key])
        batches_per_epoch = (n_train_samples + cfg.batch_size - 1) // cfg.batch_size
        num_steps = cfg.epochs * batches_per_epoch

        # Get weight decay groups from problem (not hardcoded!)
        weight_decay_groups = self.problem.get_weight_decay_groups()

        # Create training state
        self.train_state = create_train_state(
            models=self.problem.models,
            rng=self.problem.rng,
            sample_inputs=sample_inputs,
            weight_decay_groups=weight_decay_groups,
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            num_steps=num_steps,
            epochs=cfg.epochs,  # FIXED: Pass epochs for scheduler conversion
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

        # Get batch keys from problem
        batch_keys = self.problem.get_batch_keys()

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
        print(f"Models: {list(self.problem.models.keys())}")
        print(f"Weight decay groups: {self.problem.get_weight_decay_groups()}")
        print("=" * 60 + "\n")

        for epoch in trange(cfg.epochs, desc="Training"):
            # Create batches for this epoch
            self.train_state['rng'], data_rng = random.split(self.train_state['rng'])

            # Get arrays in order of batch_keys
            train_arrays = [train_data[k] for k in batch_keys]
            train_batches, n_train = create_data_batches(
                *train_arrays,
                batch_size=cfg.batch_size,
                shuffle=True,
                rng=data_rng
            )

            # Training loop
            loss_sum = 0.
            pde_sum = 0.
            data_sum = 0.
            nf_sum = 0.

            for batch_tuple in train_batches:
                # Convert tuple to dict
                batch = {k: v for k, v in zip(batch_keys, batch_tuple)}

                # Get RNG for this step
                self.train_state['rng'], step_rng = random.split(self.train_state['rng'])

                new_params, new_opt_states, metrics = train_step_fn(
                    self.train_state['params'],
                    self.train_state['opt_states'],
                    batch,
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
            test_arrays = [test_data[k] for k in batch_keys]
            test_batches, n_test = create_data_batches(
                *test_arrays,
                batch_size=cfg.batch_size,
                shuffle=False
            )

            error_sum = 0.
            test_nf_sum = 0.

            for batch_tuple in test_batches:
                batch = {k: v for k, v in zip(batch_keys, batch_tuple)}
                metrics = eval_step_fn(
                    self.train_state['params'],
                    batch
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

            # Print progress and run diagnostics
            if (epoch + 1) % cfg.epoch_show == 0:
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Loss: {avg_loss:.4f} (pde={pde_sum / n_train:.4f}, "
                      f"data={data_sum / n_train:.4f}, nf={avg_nf:.4f})")
                print(f"  Test Error: {avg_error:.4f}, Test NF NLL: {avg_test_nf:.4f}")

                # Run problem-specific diagnostics
                # Get first batch for diagnostics
                first_batch = {k: train_data[k][:cfg.batch_size] for k in batch_keys}
                self.problem.run_diagnostics(
                    epoch=epoch,
                    params=self.train_state['params'],
                    sample_data=first_batch,
                    logger=self._log
                )

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
        def train_step(params, opt_states, batch, rng):
            """Single training step (JIT-compiled)

            Args:
                params: Model parameters dict
                opt_states: Optimizer states dict
                batch: Dict with batch data (keys from get_batch_keys())
                rng: PRNG key

            Returns:
                (new_params, new_opt_states, metrics)
            """

            def loss_fn(params_dict):
                """Combined loss function - delegates to problem"""
                total_loss, metrics = problem.compute_training_losses(
                    params_dict, batch, rng, weights
                )
                return total_loss, metrics

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
        def eval_step(params, batch):
            """Single evaluation step (JIT-compiled)

            Args:
                params: Model parameters dict
                batch: Dict with batch data

            Returns:
                Dict with evaluation metrics
            """
            return problem.compute_eval_metrics(params, batch)

        return eval_step

    def close(self) -> None:
        """Cleanup resources"""
        if self.writer:
            self.writer.close()
            self.writer = None
