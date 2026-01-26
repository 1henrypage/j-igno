#!/usr/bin/env python3
"""
Main entry point for IGNO training (JAX version).

Trains encoder + decoders + NF jointly in a single phase.
Models are owned by ProblemInstance. Trainer orchestrates training.

Usage:
    python training.py --config configs/training/example_train.yaml
    python training.py --config configs/training/example_train.yaml --epochs 5000
    python training.py --config configs/training/example_train.yaml --pretrained path/to/checkpoint.pkl

JAX-specific options:
    python training.py --config configs/training/example_train.yaml --platform cpu
    python training.py --config configs/training/example_train.yaml --platform gpu
    python training.py --config configs/training/example_train.yaml --disable-jit
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any


# Set JAX platform before importing JAX
def set_platform(platform: str):
    """Set JAX platform (must be called before importing JAX)"""
    if platform:
        os.environ['JAX_PLATFORMS'] = platform


def main():
    parser = argparse.ArgumentParser(description='Train IGNO models (JAX)')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained checkpoint')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--dry-run', action='store_true', help='Print config and exit')

    # JAX-specific options
    parser.add_argument('--platform', type=str, choices=['cpu', 'gpu', 'tpu'],
                        help='JAX platform (cpu, gpu, tpu)')
    parser.add_argument('--disable-jit', action='store_true',
                        help='Disable JIT compilation (for debugging)')
    parser.add_argument('--debug-nans', action='store_true',
                        help='Enable NaN debugging (slower)')

    args = parser.parse_args()

    # Set platform before importing JAX
    if args.platform:
        set_platform(args.platform)

    # Now import JAX and other modules
    import jax

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        print("JIT compilation disabled")

    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
        print("NaN debugging enabled")

    # Print JAX info
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    # Import project modules after JAX setup
    sys.path.insert(0, str(Path(__file__).parent))
    from src.solver.config import TrainingConfig
    from src.solver.trainer import IGNOTrainer
    from src.problems import create_problem, ProblemInstance

    # Load config
    config = TrainingConfig.load(args.config)

    # Apply CLI overrides
    if args.pretrained:
        config.pretrained = {'path': args.pretrained}
    if args.seed:
        config.seed = args.seed
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.optimizer.lr = args.lr

    # Print config summary
    print(f"\n{'=' * 60}")
    print(f"IGNO Training (JAX)")
    print(f"{'=' * 60}")
    print(f"Config: {args.config}")
    print(f"Seed: {config.seed}")
    print(f"Problem: {config.problem.type}")
    print(f"  Train data: {config.problem.train_data}")
    print(f"  Test data: {config.problem.test_data}")
    print(f"Training:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Optimizer: {config.training.optimizer.type}, lr={config.training.optimizer.lr}")
    print(f"  Loss weights: pde={config.training.loss_weights.pde}, data={config.training.loss_weights.data}")
    if config.pretrained:
        print(f"Pretrained: {config.pretrained.get('path')}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("[DRY RUN] Config loaded successfully. Exiting.")
        return

    # Run training
    results = run_training(config, pretrained_path=Path(args.pretrained) if args.pretrained else None)

    print(f"\nTraining complete!")
    print(f"  Best error: {results['best_error']:.6f}")
    print(f"  Time: {results['time']:.1f}s")


def run_training(
        config: 'TrainingConfig',
        problem: Optional['ProblemInstance'] = None,
        pretrained_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run IGNO training.

    Args:
        config: Training configuration
        problem: Optional pre-created problem instance
        pretrained_path: Optional path to pretrained checkpoint

    Returns:
        Results dict with metrics
    """
    # Import here to allow platform selection before import
    from src.solver.trainer import IGNOTrainer
    from src.problems import create_problem

    if problem is None:
        print(f"Creating problem: {config.problem.type}")
        problem = create_problem(config)

    trainer = IGNOTrainer(problem)
    trainer.setup(config, pretrained_path=pretrained_path)

    try:
        results = trainer.train(config)
    finally:
        trainer.close()

    print(f"\nDone. Results saved to: {trainer.run_dir}")
    return results


if __name__ == '__main__':
    main()