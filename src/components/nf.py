# src/components/nf.py
# Neural Spline Flow - Optimized version with explicit handling of control flow

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
from flax import linen as nn
from typing import Tuple
from functools import partial
import math


# ============================================================================
# Rational Quadratic Spline Functions (Pure functional, JIT-compilable)
# ============================================================================

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


@jit
def searchsorted(bin_locations: jax.Array, inputs: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Binary search for bin indices"""
    bin_locations = bin_locations.at[..., -1].add(eps)
    return jnp.sum(inputs[..., None] >= bin_locations, axis=-1) - 1


@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def unconstrained_RQS(
        inputs: jax.Array,
        unnormalized_widths: jax.Array,
        unnormalized_heights: jax.Array,
        unnormalized_derivatives: jax.Array,
        inverse: bool = False,
        tail_bound: float = 3.,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE
) -> Tuple[jax.Array, jax.Array]:
    """Unconstrained rational quadratic spline transformation"""

    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # Pad derivatives
    unnormalized_derivatives = jnp.pad(unnormalized_derivatives, ((0, 0), (1, 1)), mode='constant')
    constant = jnp.log(jnp.exp(1 - min_derivative) - 1)
    unnormalized_derivatives = unnormalized_derivatives.at[:, 0].set(constant)
    unnormalized_derivatives = unnormalized_derivatives.at[:, -1].set(constant)

    # Apply RQS to all points (we'll mask later)
    outputs_inside, logabsdet_inside = RQS(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    # Use jnp.where for conditional (JIT-friendly)
    outputs = jnp.where(inside_intvl_mask, outputs_inside, inputs)
    logabsdet = jnp.where(inside_intvl_mask, logabsdet_inside, 0.)

    return outputs, logabsdet


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def RQS(
        inputs: jax.Array,
        unnormalized_widths: jax.Array,
        unnormalized_heights: jax.Array,
        unnormalized_derivatives: jax.Array,
        inverse: bool = False,
        left: float = 0.,
        right: float = 1.,
        bottom: float = 0.,
        top: float = 1.,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE
) -> Tuple[jax.Array, jax.Array]:
    """Rational quadratic spline transformation"""

    num_bins = unnormalized_widths.shape[-1]

    # Compute widths
    widths = jax.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = jnp.cumsum(widths, axis=-1)
    cumwidths = jnp.pad(cumwidths, ((0, 0), (1, 0)), mode='constant', constant_values=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths = cumwidths.at[:, 0].set(left)
    cumwidths = cumwidths.at[:, -1].set(right)
    widths = cumwidths[:, 1:] - cumwidths[:, :-1]

    # Compute derivatives
    derivatives = min_derivative + jax.nn.softplus(unnormalized_derivatives)

    # Compute heights
    heights = jax.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = jnp.cumsum(heights, axis=-1)
    cumheights = jnp.pad(cumheights, ((0, 0), (1, 0)), mode='constant', constant_values=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights = cumheights.at[:, 0].set(bottom)
    cumheights = cumheights.at[:, -1].set(top)
    heights = cumheights[:, 1:] - cumheights[:, :-1]

    # Find bin indices - use lax.cond for the branch
    def get_bin_idx_forward():
        return searchsorted(cumwidths, inputs)[..., None]

    def get_bin_idx_inverse():
        return searchsorted(cumheights, inputs)[..., None]

    # Note: inverse is a static argument, so this branch happens at trace time
    bin_idx = get_bin_idx_inverse() if inverse else get_bin_idx_forward()

    # Gather values for selected bins
    input_cumwidths = jnp.take_along_axis(cumwidths, bin_idx, axis=-1)[..., 0]
    input_bin_widths = jnp.take_along_axis(widths, bin_idx, axis=-1)[..., 0]
    input_cumheights = jnp.take_along_axis(cumheights, bin_idx, axis=-1)[..., 0]
    delta = heights / widths
    input_delta = jnp.take_along_axis(delta, bin_idx, axis=-1)[..., 0]
    input_derivatives = jnp.take_along_axis(derivatives, bin_idx, axis=-1)[..., 0]
    input_derivatives_plus_one = jnp.take_along_axis(derivatives[:, 1:], bin_idx, axis=-1)[..., 0]
    input_heights = jnp.take_along_axis(heights, bin_idx, axis=-1)[..., 0]

    # Compute transformation - inverse is static, so this branch happens at trace time
    if inverse:
        a = (((inputs - input_cumheights) *
              (input_derivatives + input_derivatives_plus_one - 2 * input_delta) +
              input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives -
             (inputs - input_cumheights) *
             (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b ** 2 - 4 * a * c
        discriminant = jnp.maximum(discriminant, 0.)

        root = (2 * c) / (-b - jnp.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) *
                                     theta_one_minus_theta)
        derivative_numerator = input_delta ** 2 * (
                input_derivatives_plus_one * root ** 2 +
                2 * input_delta * theta_one_minus_theta +
                input_derivatives * (1 - root) ** 2
        )
        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta ** 2 + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) *
                                     theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta ** 2 * (
                input_derivatives_plus_one * theta ** 2 +
                2 * input_delta * theta_one_minus_theta +
                input_derivatives * (1 - theta) ** 2
        )
        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)
        return outputs, logabsdet


# ============================================================================
# Neural Network Components
# ============================================================================

class FCNN(nn.Module):
    """2-hidden-layer feedforward network with SiLU activation"""
    in_dim: int
    out_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


# ============================================================================
# Helper function for single RQS transformation (for vmap)
# ============================================================================

@partial(jit, static_argnums=(4, 5))
def apply_rqs_single(
        u: jax.Array,
        w: jax.Array,
        h: jax.Array,
        d: jax.Array,
        inverse: bool,
        tail_bound: float
) -> Tuple[jax.Array, jax.Array]:
    """Apply RQS to a single sample (to be vmapped)"""
    # Add batch dimension for compatibility with unconstrained_RQS
    u_batch = u[None, :]
    w_batch = w[None, :]
    h_batch = h[None, :]
    d_batch = d[None, :]

    result, logdet = unconstrained_RQS(u_batch, w_batch, h_batch, d_batch, inverse, tail_bound)

    # Remove batch dimension
    return result[0], logdet[0]


# ============================================================================
# Neural Spline Flow Coupling Layer
# ============================================================================

class NSF_CL(nn.Module):
    """Neural Spline Flow coupling layer - optimized with vmap"""
    dim: int
    hidden_dim: int
    K: int = 5
    B: float = 3.0

    def setup(self):
        self.f1 = FCNN(self.dim // 2, (3 * self.K - 1) * (self.dim - self.dim // 2), self.hidden_dim)
        self.f2 = FCNN(self.dim - self.dim // 2, (3 * self.K - 1) * (self.dim // 2), self.hidden_dim)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Forward transformation

        Args:
            x: Input (batch, dim)

        Returns:
            z: Output (batch, dim)
            log_det: Log determinant (batch,)
        """
        log_det = jnp.zeros(x.shape[0])
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]

        # Transform upper given lower
        out = self.f1(lower).reshape(-1, self.dim - self.dim // 2, 3 * self.K - 1)
        W, H, D = jnp.split(out, [self.K, 2*self.K], axis=2)

        # Vectorize over batch dimension
        upper, ld = vmap(
            partial(apply_rqs_single, inverse=False, tail_bound=self.B),
            in_axes=(0, 0, 0, 0)
        )(upper, W, H, D)
        log_det = log_det + jnp.sum(ld, axis=1)

        # Transform lower given upper
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = jnp.split(out, [self.K, 2*self.K], axis=2)

        lower, ld = vmap(
            partial(apply_rqs_single, inverse=False, tail_bound=self.B),
            in_axes=(0, 0, 0, 0)
        )(lower, W, H, D)
        log_det = log_det + jnp.sum(ld, axis=1)

        return jnp.concatenate([lower, upper], axis=1), log_det

    def inverse(self, z: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Inverse transformation

        Args:
            z: Input (batch, dim)

        Returns:
            x: Output (batch, dim)
            log_det: Log determinant (batch,)
        """
        log_det = jnp.zeros(z.shape[0])
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]

        # Inverse transform lower given upper
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = jnp.split(out, [self.K, 2*self.K], axis=2)

        lower, ld = vmap(
            partial(apply_rqs_single, inverse=True, tail_bound=self.B),
            in_axes=(0, 0, 0, 0)
        )(lower, W, H, D)
        log_det = log_det + jnp.sum(ld, axis=1)

        # Inverse transform upper given lower
        out = self.f1(lower).reshape(-1, self.dim - self.dim // 2, 3 * self.K - 1)
        W, H, D = jnp.split(out, [self.K, 2*self.K], axis=2)

        upper, ld = vmap(
            partial(apply_rqs_single, inverse=True, tail_bound=self.B),
            in_axes=(0, 0, 0, 0)
        )(upper, W, H, D)
        log_det = log_det + jnp.sum(ld, axis=1)

        return jnp.concatenate([lower, upper], axis=1), log_det


# ============================================================================
# Main RealNVP Class
# ============================================================================

class RealNVP(nn.Module):
    """Normalizing Flow using Neural Spline Flows

    Maps between latent space β ∈ [-1,1]^d and standard Gaussian z ~ N(0,I).
    Optimized with vmap and jit.
    """
    dim: int
    num_flows: int = 3
    hidden_dim: int = 64
    K: int = 5
    B: float = 3.0

    def setup(self):
        self.flows = [
            NSF_CL(self.dim, self.hidden_dim, self.K, self.B)
            for _ in range(self.num_flows)
        ]

        # Fixed permutation (reversal - its own inverse)
        self.perm = jnp.arange(self.dim - 1, -1, -1, dtype=jnp.int32)
        self.log_2pi = jnp.log(2.0 * jnp.pi)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Forward: data x -> latent z

        Args:
            x: Data samples (batch, dim)

        Returns:
            z: Latent samples (batch, dim)
            log_det: Log determinant (batch,)
        """
        log_det_total = jnp.zeros(x.shape[0])

        # Python loop unrolls at trace time - this is fine
        for i, flow in enumerate(self.flows):
            x, log_det = flow(x)
            log_det_total = log_det_total + log_det
            # Static control flow - i is known at trace time
            if i < len(self.flows) - 1:
                x = x[:, self.perm]

        return x, log_det_total

    def inverse(self, z: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Inverse: latent z -> data x

        Args:
            z: Latent samples (batch, dim)

        Returns:
            x: Data samples (batch, dim)
            log_det: Log determinant (batch,)
        """
        log_det_total = jnp.zeros(z.shape[0])

        # Python loop unrolls at trace time - this is fine
        for i, flow in enumerate(reversed(self.flows)):
            # Static control flow - i is known at trace time
            if i > 0:
                z = z[:, self.perm]
            z, log_det = flow.inverse(z)
            log_det_total = log_det_total + log_det

        return z, log_det_total

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log p(x) using change of variables

        Args:
            x: Data samples (batch, dim)

        Returns:
            log_prob: Log probability (batch,)
        """
        z, log_det = self(x)
        log_pz = -0.5 * (z ** 2 + self.log_2pi).sum(axis=1)
        return log_pz + log_det

    def loss(self, x: jax.Array) -> jax.Array:
        """Negative log-likelihood loss

        Args:
            x: Data samples (batch, dim)

        Returns:
            loss: Scalar loss
        """
        return -self.log_prob(x).mean()

    def sample(self, rng: jax.Array, num_samples: int) -> jax.Array:
        """Sample from the model

        Args:
            rng: PRNG key
            num_samples: Number of samples

        Returns:
            samples: Generated samples (num_samples, dim)
        """
        z = random.normal(rng, (num_samples, self.dim))
        x, _ = self.inverse(z)
        return x
