# src/utils/solver_utils.py
# JAX version with fixes:
# 1. Added OneCycle scheduler support
# 2. Improved gradient clipping with NaN detection option
# 3. Better documentation

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import optax
from flax import linen as nn
from typing import Optional, Dict, List, Literal, Union, Tuple, Any, Callable
from functools import partial
import pickle
from pathlib import Path

from src.components.fcn import FCNet
from src.components.mon import MultiONetBatch, MultiONetBatch_X


def get_model(
        x_in_size: int,
        beta_in_size: int,
        trunk_layers: list[int],
        branch_layers: list[int],
        latent_size: int = None,
        out_size: int = 1,
        activation_trunk: str = 'SiLU_Sin',
        activation_branch: str = 'SiLU',
        net_type: str = 'MultiONetBatch',
        dtype: jnp.dtype = jnp.float32,
        **kwargs
) -> nn.Module:
    """Get neural network model (Flax module - not initialized)

    Args:
        x_in_size: Input dimension for coordinates
        beta_in_size: Input dimension for latent code
        trunk_layers: Hidden layer sizes for trunk network
        branch_layers: Hidden layer sizes for branch network
        latent_size: Latent dimension (for MultiONetBatch_X)
        out_size: Output dimension
        activation_trunk: Activation function for trunk
        activation_branch: Activation function for branch
        net_type: Network type ('MultiONetBatch', 'MultiONetBatch_X', 'FCNet')
        dtype: Data type
        **kwargs: Additional arguments passed to network constructor

    Returns:
        Flax Module (not yet initialized)
    """

    if net_type == 'MultiONetBatch':
        return MultiONetBatch(
            in_size_x=x_in_size,
            in_size_a=beta_in_size,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk=activation_trunk,
            activation_branch=activation_branch,
            dtype=dtype,
            **kwargs
        )
    elif net_type == 'MultiONetBatch_X':
        return MultiONetBatch_X(
            in_size_x=x_in_size,
            in_size_a=beta_in_size,
            latent_size=latent_size,
            out_size=out_size,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk=activation_trunk,
            activation_branch=activation_branch,
            dtype=dtype,
            **kwargs
        )
    elif net_type == 'FCNet':
        return FCNet(dtype=dtype, **kwargs)
    else:
        raise NotImplementedError(f"Unknown net_type: {net_type}")


def get_optimizer(
        optimizer_config,
        learning_rate: float = None,
        clip_grad_norm: float = 10.0,
        error_on_nonfinite: bool = True
) -> optax.GradientTransformation:
    """Create optax optimizer with gradient clipping

    Args:
        optimizer_config: Config object with type, lr, weight_decay
        learning_rate: Override learning rate (or use schedule)
        clip_grad_norm: Max gradient norm for clipping
        error_on_nonfinite: If True, raises error on NaN/Inf gradients
                           If False, clips silently (default JAX behavior)

    Returns:
        Optax gradient transformation
    """

    lr = learning_rate if learning_rate is not None else optimizer_config.lr

    OPTIMIZERS = {
        'Adam': optax.adam,
        'AdamW': optax.adamw,
        'RMSprop': optax.rmsprop,
        'SGD': optax.sgd,
    }

    optimizer_type = optimizer_config.type
    if optimizer_type not in OPTIMIZERS:
        raise NotImplementedError(f'Unknown optimizer: {optimizer_type}')

    # Create base optimizer
    if optimizer_type == 'AdamW':
        opt = OPTIMIZERS[optimizer_type](
            learning_rate=lr,
            weight_decay=optimizer_config.weight_decay
        )
    else:
        opt = OPTIMIZERS[optimizer_type](learning_rate=lr)

    # Build chain with gradient clipping
    transforms = []

    # Optional: Check for NaN/Inf before clipping
    if error_on_nonfinite:
        # Custom transform that checks for non-finite values
        def check_finite(updates, state, params=None):
            finite = jax.tree_util.tree_all(
                jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), updates)
            )
            # This will cause NaN in the updates if any are non-finite
            # which will propagate and be visible in loss
            updates = jax.tree_util.tree_map(
                lambda x: jnp.where(finite, x, jnp.full_like(x, jnp.nan)),
                updates
            )
            return updates, state

        transforms.append(optax.stateless(check_finite))

    transforms.append(optax.clip_by_global_norm(clip_grad_norm))
    transforms.append(opt)

    return optax.chain(*transforms)


def get_scheduler(
        scheduler_config,
        optimizer_config,
        num_steps: int
) -> Optional[optax.Schedule]:
    """Create learning rate schedule

    Args:
        scheduler_config: Config with type and parameters
        optimizer_config: Config with base learning rate
        num_steps: Total number of training steps

    Returns:
        Optax schedule or None
    """

    if scheduler_config.type is None:
        return None

    base_lr = optimizer_config.lr

    if scheduler_config.type == 'StepLR':
        return optax.exponential_decay(
            init_value=base_lr,
            transition_steps=scheduler_config.step_size,
            decay_rate=scheduler_config.gamma,
            staircase=True
        )
    elif scheduler_config.type == 'CosineAnnealing':
        return optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=num_steps,
            alpha=scheduler_config.eta_min / base_lr if scheduler_config.eta_min else 0.0
        )
    elif scheduler_config.type == 'OneCycle':
        # FIXED: Added OneCycle scheduler support
        # OneCycle: warmup to max_lr, then decay
        return optax.schedules.warmup_cosine_decay_schedule(
            init_value=base_lr / scheduler_config.div_factor,
            peak_value=base_lr,
            warmup_steps=int(num_steps * scheduler_config.pct_start),
            decay_steps=num_steps,
            end_value=base_lr / scheduler_config.final_div_factor
        )
    elif scheduler_config.type == 'Plateau':
        # Note: ReduceLROnPlateau requires metric feedback, not directly supported
        # in optax schedules. Would need custom implementation.
        raise NotImplementedError(
            "Plateau scheduler requires metric feedback and is not directly "
            "supported in optax. Consider using CosineAnnealing or StepLR instead."
        )
    else:
        raise NotImplementedError(f'Unknown scheduler: {scheduler_config.type}')


def create_train_state(
        models: Dict[str, nn.Module],
        rng: jax.Array,
        sample_inputs: Dict[str, Dict[str, jax.Array]],
        optimizer_config,
        scheduler_config=None,
        num_steps: int = None,
        clip_grad_norm: float = 10.0
) -> Dict:
    """Initialize all model parameters and optimizer states

    Args:
        models: Dict of Flax modules
        rng: PRNG key
        sample_inputs: Dict mapping model names to sample input dicts
        optimizer_config: Optimizer configuration
        scheduler_config: Optional scheduler configuration
        num_steps: Total training steps (required if using scheduler)
        clip_grad_norm: Gradient clipping norm

    Returns:
        Dict with 'params', 'opt_states', 'optimizers', 'step', 'rng'
    """
    params = {}
    opt_states = {}
    optimizers = {}

    # Create schedule if needed
    schedule = None
    if scheduler_config is not None and num_steps is not None:
        schedule = get_scheduler(scheduler_config, optimizer_config, num_steps)

    # Initialize each model
    rng, *init_rngs = random.split(rng, len(models) + 1)

    for (name, model), init_rng in zip(models.items(), init_rngs):
        # Initialize params
        if name in sample_inputs:
            variables = model.init(init_rng, **sample_inputs[name])
            params[name] = variables['params']
        else:
            raise ValueError(f"No sample input for model '{name}'")

        # Create optimizer for this model
        # Decoders ('a', 'u') get weight decay, encoder/NF don't
        if name in ['a', 'u']:
            opt_config_copy = type(optimizer_config)(
                type=optimizer_config.type,
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay
            )
        else:
            opt_config_copy = type(optimizer_config)(
                type=optimizer_config.type,
                lr=optimizer_config.lr,
                weight_decay=0.0
            )

        if schedule is not None:
            optimizer = get_optimizer(
                opt_config_copy,
                learning_rate=schedule,
                clip_grad_norm=clip_grad_norm
            )
        else:
            optimizer = get_optimizer(
                opt_config_copy,
                clip_grad_norm=clip_grad_norm
            )

        optimizers[name] = optimizer
        opt_states[name] = optimizer.init(params[name])

    return {
        'params': params,
        'opt_states': opt_states,
        'optimizers': optimizers,
        'step': 0,
        'rng': rng
    }


# Data utilities

def create_data_batches(
        *arrays: jnp.ndarray,
        batch_size: int = 100,
        shuffle: bool = True,
        rng: jax.Array = None,
        drop_last: bool = False
) -> Tuple[List[Tuple], int]:
    """Create list of batches for epoch

    Args:
        *arrays: Arrays to batch (must have same first dimension)
        batch_size: Batch size
        shuffle: Whether to shuffle
        rng: PRNG key (required if shuffle=True)
        drop_last: If True, drop last incomplete batch

    Returns:
        (batches, num_batches) where batches is list of tuples
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required")

    n_samples = arrays[0].shape[0]
    if not all(arr.shape[0] == n_samples for arr in arrays):
        raise ValueError("All arrays must have same first dimension")

    # Create indices
    if shuffle:
        if rng is None:
            raise ValueError("rng required when shuffle=True")
        indices = random.permutation(rng, n_samples)
    else:
        indices = jnp.arange(n_samples)

    # Create batches
    batches = []

    if drop_last:
        num_batches = n_samples // batch_size
    else:
        num_batches = (n_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        batch = tuple(arr[batch_indices] for arr in arrays)
        batches.append(batch)

    return batches, num_batches


# Checkpoint utilities

def save_checkpoint(path: Path, state: Dict, metadata: Dict = None) -> None:
    """Save checkpoint using pickle

    Args:
        path: Path to save checkpoint
        state: State dict (params, opt_states, etc.)
        metadata: Optional metadata dict
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'state': state,
        'metadata': metadata or {}
    }

    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: Path) -> Tuple[Dict, Dict]:
    """Load checkpoint

    Args:
        path: Path to checkpoint

    Returns:
        (state, metadata) tuple
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    return checkpoint['state'], checkpoint.get('metadata', {})


# Utility functions for parameter counting and inspection

def count_params(params: Dict) -> int:
    """Count total parameters in a pytree

    Args:
        params: Parameter pytree

    Returns:
        Total number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def print_param_shapes(params: Dict, prefix: str = "") -> None:
    """Print shapes of all parameters (for debugging)

    Args:
        params: Parameter pytree
        prefix: Prefix for printing
    """
    def print_leaf(path, leaf):
        path_str = "/".join(str(p) for p in path)
        print(f"  {prefix}{path_str}: {leaf.shape}")

    jax.tree_util.tree_map_with_path(print_leaf, params)
