
# src/evaluation/__init__.py
"""
Evaluation module for inverse problems in JAX.

Components:
- IGNOInverter: Gradient-based inversion using trained IGNO models
- Metrics: RMSE, relative L2, cross-correlation for evaluation
"""

from src.evaluation.igno import IGNOInverter
from src.evaluation.metrics import (
    rmse,
    relative_l2,
    cross_correlation,
    compute_all_metrics,
    aggregate_metrics,
)

__all__ = [
    'IGNOInverter',
    'rmse',
    'relative_l2',
    'cross_correlation',
    'compute_all_metrics',
    'aggregate_metrics',
]