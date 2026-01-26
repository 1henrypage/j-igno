"""
Training and evaluation configuration for JAX.

Simplified structure:
- Single training phase (encoder + decoders + NF jointly)
- Evaluation config for inversion methods
"""
from dataclasses import dataclass, asdict, field, fields
from typing import Literal, Optional, List, Dict, Any, Type, TypeVar
from pathlib import Path
import yaml
import numpy as np
import jax.numpy as jnp


def _serialize(obj: Any) -> Any:
    """Serialize objects for YAML output"""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, jnp.ndarray):
        return float(obj) if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    return obj


T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        if data is None:
            data = {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))


# =============================================================================
# Problem Config
# =============================================================================

@dataclass
class ProblemConfig(BaseConfig):
    """Problem type and data paths."""
    type: str = None
    train_data: Optional[str] = None
    test_data: Optional[str] = None


# =============================================================================
# Training Components
# =============================================================================

@dataclass
class OptimizerConfig(BaseConfig):
    type: Literal['Adam', 'AdamW', 'RMSprop', 'SGD'] = 'Adam'
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig(BaseConfig):
    type: Optional[Literal['StepLR', 'CosineAnnealing', 'OneCycle']] = None
    # StepLR params
    step_size: Optional[int] = None
    gamma: Optional[float] = None
    # CosineAnnealing params
    eta_min: Optional[float] = None
    # OneCycle params
    pct_start: Optional[float] = 0.3
    div_factor: Optional[float] = 25.0
    final_div_factor: Optional[float] = 1e4


@dataclass
class LossWeights(BaseConfig):
    pde: float = 1.0
    data: float = 1.0


# =============================================================================
# Training Config (Joint: encoder + decoders + NF)
# =============================================================================

@dataclass
class IGNOTrainingConfig(BaseConfig):
    """
    Config for joint IGNO training (encoder + decoders + NF).

    The NF is trained jointly with encoder/decoders using detached latents,
    as per the original IGNO paper implementation.
    """
    epochs: int = 10000
    batch_size: int = 50
    epoch_show: int = 50

    # Loss weights for PDE and data terms
    # Note: NF loss doesn't need a weight since gradients don't flow to encoder/decoders
    loss_weights: LossWeights = field(default_factory=LossWeights)

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'IGNOTrainingConfig':
        if data is None:
            return cls()
        return cls(
            epochs=data.get('epochs', 10000),
            batch_size=data.get('batch_size', 50),
            epoch_show=data.get('epoch_show', 50),
            loss_weights=LossWeights.from_dict(data.get('loss_weights', {})),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


# =============================================================================
# Evaluation Configs
# =============================================================================

@dataclass
class InversionConfig(BaseConfig):
    """Config for gradient-based inversion."""
    epochs: int = 500
    loss_weights: LossWeights = None
    optimizer: OptimizerConfig = None
    scheduler: SchedulerConfig = None

    @classmethod
    def from_dict(cls, data: dict) -> 'InversionConfig':
        if data is None:
            return cls()
        return cls(
            epochs=data.get('epochs', 500),
            loss_weights=LossWeights.from_dict(data.get('loss_weights', {})),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


@dataclass
class EvaluationConfig(BaseConfig):
    """Config for evaluation/inversion."""
    method: Literal['igno', 'mcmc'] = 'igno'

    # Batch size for parallel inversion
    batch_size: int = 200

    # Observation setup
    n_obs: int = 100
    obs_sampling: Literal['random', 'grid'] = 'random'

    # Noise (None for clean)
    snr_db: Optional[float] = None

    # Inversion params (for IGNO method)
    inversion: InversionConfig = field(default_factory=InversionConfig)

    # Output
    results_dir: str = "results"

    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationConfig':
        if data is None:
            return cls()
        return cls(
            method=data.get('method', 'igno'),
            batch_size=data.get('batch_size', 200),
            n_obs=data.get('n_obs', 100),
            obs_sampling=data.get('obs_sampling', 'random'),
            snr_db=data.get('snr_db'),
            inversion=InversionConfig.from_dict(data.get('inversion', {})),
            results_dir=data.get('results_dir', 'results'),
        )


# =============================================================================
# Main Config
# =============================================================================

@dataclass
class TrainingConfig(BaseConfig):
    """Main config for training and evaluation."""
    run_name: str = None
    artifact_root: str = "runs"
    seed: int = 10086

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    pretrained: Optional[Dict[str, Any]] = None

    # Training config (joint encoder + decoders + NF)
    training: IGNOTrainingConfig = field(default_factory=IGNOTrainingConfig)

    # Evaluation config
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        if data is None:
            return cls()

        problem_data = data.get('problem', {})
        if isinstance(problem_data, str):
            problem_data = {'type': problem_data}

        return cls(
            run_name=data.get('run_name'),
            artifact_root=data.get('artifact_root', 'runs'),
            seed=data.get('seed', 10086),
            problem=ProblemConfig.from_dict(problem_data),
            pretrained=data.get('pretrained'),
            training=IGNOTrainingConfig.from_dict(data.get('training', {})),
            evaluation=EvaluationConfig.from_dict(data.get('evaluation', {})),
        )

    @classmethod
    def load(cls, path: Path) -> 'TrainingConfig':
        with open(Path(path), 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_pretrained_path(self) -> Optional[Path]:
        if self.pretrained is None:
            return None
        path = Path(self.pretrained.get('path', ''))
        if not path.exists():
            raise RuntimeError("Pretrained path doesn't exist.")
        if path.suffix == '.pt' or path.suffix == '.pkl':
            return path
        # Default to best.pt in weights directory
        checkpoint = self.pretrained.get('checkpoint', 'best.pt')
        return path / 'weights' / checkpoint
