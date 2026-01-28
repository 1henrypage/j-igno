# src/evaluation/metrics.py
"""
Evaluation metrics for inverse problems in JAX.

From IGNO paper Section 4.5:
- Relative RMSE (Eq. 14): For continuous coefficients
- Cross-correlation indicator I_corr (Eq. 15): For discontinuous targets

Also includes standard relative L2.
"""
import jax.numpy as jnp
from typing import Dict, List

from utils.Losses import MyError


def rmse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """
    Relative Root Mean Square Error (IGNO Eq. 14).

    RMSE = sqrt(sum((a_rec - a_true)^2) / sum(a_true^2))

    This is normalized by the energy of the true signal.
    """
    pred_flat = pred.flatten().astype(jnp.float32)
    true_flat = true.flatten().astype(jnp.float32)

    numerator = jnp.sum((pred_flat - true_flat) ** 2)
    denominator = jnp.sum(true_flat ** 2)

    # Handle zero denominator
    safe_denom = jnp.where(denominator < 1e-10, 1e-10, denominator)
    result = jnp.sqrt(numerator / safe_denom)

    return float(result)


def relative_l2(pred: jnp.ndarray, true: jnp.ndarray, p: int = 2) -> float:
    """
    Relative L_p error

    rel_L2 = ||pred - true||_p / ||true||_p
    """
    error_fn = MyError(d=2, p=p, size_average=True, reduction=True)
    return float(error_fn.Lp_rel(pred, true))


def cross_correlation(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """
    Cross-correlation indicator I_corr (IGNO Eq. 15).

    For discontinuous targets, measures morphological similarity.

    I_corr = sum(a_true_scaled^2 * a_rec_scaled^2) /
             (sqrt(sum(a_true_scaled^2)) * sqrt(sum(a_rec_scaled^2)))

    Where a_scaled denotes coefficients rescaled to [0, 1].

    I_corr ranges from 0 to 1, with values close to 1 indicating
    strong morphological agreement.
    """
    pred_flat = pred.flatten().astype(jnp.float32)
    true_flat = true.flatten().astype(jnp.float32)

    # Rescale to [0, 1]
    def rescale_01(x):
        x_min, x_max = jnp.min(x), jnp.max(x)
        range_val = x_max - x_min
        # Handle constant arrays
        safe_range = jnp.where(range_val < 1e-10, 1.0, range_val)
        return jnp.where(range_val < 1e-10, jnp.zeros_like(x), (x - x_min) / safe_range)

    pred_scaled = rescale_01(pred_flat)
    true_scaled = rescale_01(true_flat)

    # Compute I_corr from Eq. 15
    true_sq = true_scaled ** 2
    pred_sq = pred_scaled ** 2

    numerator = jnp.sum(true_sq * pred_sq)
    denominator = jnp.sqrt(jnp.sum(true_sq)) * jnp.sqrt(jnp.sum(pred_sq))

    # Handle zero denominator
    safe_denom = jnp.where(denominator < 1e-10, 1e-10, denominator)

    return float(numerator / safe_denom)


def compute_all_metrics(pred: jnp.ndarray, true: jnp.ndarray) -> Dict[str, float]:
    """
    Compute all metrics at once.

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        Dictionary with all metrics
    """
    return {
        'rmse': rmse(pred, true),
        'relative_l2': relative_l2(pred, true),
        'cross_correlation': cross_correlation(pred, true),
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple samples.

    Args:
        metrics_list: List of metric dicts from compute_all_metrics

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    result = {}

    for key in keys:
        values = [m[key] for m in metrics_list]
        n = len(values)
        mean_val = sum(values) / n
        std_val = (sum((v - mean_val) ** 2 for v in values) / n) ** 0.5

        result[key] = {
            'mean': mean_val,
            'std': std_val,
            'min': min(values),
            'max': max(values),
        }

    return result