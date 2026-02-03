"""
JAX Numerical Diagnostics for IGNO
Run: python debug_precision.py
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import hashlib

print("=" * 60)
print("JAX NUMERICAL DIAGNOSTICS")
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default matmul precision: {jax.config.jax_default_matmul_precision}")
print(f"x64 enabled: {jax.config.jax_enable_x64}")
print()

key = jax.random.PRNGKey(42)

# =============================================================================
# Test 1: Random number generation
# =============================================================================
print("TEST 1: Random number generation")
samples = jax.random.normal(key, (5,))
print(f"  normal(key=42, shape=5): {samples}")
h = hashlib.md5(np.array(samples).tobytes()).hexdigest()[:12]
print(f"  hash: {h}")
print()

# =============================================================================
# Test 2: Basic autodiff
# =============================================================================
print("TEST 2: Basic autodiff")
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
g = grad(lambda x: jnp.sum(x**2))(x)
print(f"  grad(sum(x^2)) at [1,2,3]: {g}")
print()

# =============================================================================
# Test 3: Second-order gradients (PDE-like)
# =============================================================================
print("TEST 3: Second-order gradients")

def pde_like_loss(params):
    # Simulates: derivative of derivative (laplacian-ish)
    x = jnp.linspace(0, 1, 50)
    u = jnp.sin(params[0] * x) * jnp.exp(-params[1] * x)
    du = jnp.gradient(u, x[1] - x[0])
    d2u = jnp.gradient(du, x[1] - x[0])
    return jnp.mean(d2u**2)

params = jnp.array([3.14, 0.5], dtype=jnp.float32)
loss_val = pde_like_loss(params)
grad_val = grad(pde_like_loss)(params)
print(f"  params: {params}")
print(f"  loss: {loss_val:.10f}")
print(f"  grad: {grad_val}")
print()

# =============================================================================
# Test 4: Matrix operations at different scales
# =============================================================================
print("TEST 4: Matrix operations (varying scales)")
key, k1, k2 = jax.random.split(key, 3)

for scale in [1e-3, 1.0, 1e3]:
    A = jax.random.normal(k1, (64, 64)) * scale
    B = jax.random.normal(k2, (64, 64)) * scale
    C = A @ B
    print(f"  scale={scale:.0e}: matmul sum={jnp.sum(C):.10f}, std={jnp.std(C):.10f}")
print()

# =============================================================================
# Test 5: Softmax stability (like normalizing flow)
# =============================================================================
print("TEST 5: Softmax/log-sum-exp stability")
key, k1 = jax.random.split(key)
logits = jax.random.normal(k1, (100,)) * 10  # large logits
logsumexp = jax.scipy.special.logsumexp(logits)
softmax_sum = jnp.sum(jax.nn.softmax(logits))
print(f"  logsumexp: {logsumexp:.10f}")
print(f"  softmax sum (should be 1.0): {softmax_sum:.10f}")
print()

# =============================================================================
# Test 6: Chained operations (like encoder -> flow -> decoder)
# =============================================================================
print("TEST 6: Chained linear layers (encoder-like)")
key, *keys = jax.random.split(key, 5)

def mlp_forward(x, keys):
    for i, k in enumerate(keys):
        W = jax.random.normal(k, (x.shape[-1], 32 if i < len(keys)-1 else 8)) * 0.1
        x = x @ W
        if i < len(keys) - 1:
            x = jax.nn.gelu(x)
    return x

x_in = jax.random.normal(keys[0], (16, 64))
out = mlp_forward(x_in, keys[1:])
print(f"  input shape: {x_in.shape}, output shape: {out.shape}")
print(f"  output mean: {jnp.mean(out):.10f}")
print(f"  output std: {jnp.std(out):.10f}")
print(f"  output hash: {hashlib.md5(np.array(out).tobytes()).hexdigest()[:12]}")
print()

# =============================================================================
# Test 7: Gradient through chained ops
# =============================================================================
print("TEST 7: Gradient through MLP")

def mlp_loss(x, keys):
    return jnp.mean(mlp_forward(x, keys)**2)

g = grad(mlp_loss)(x_in, keys[1:])
print(f"  grad shape: {g.shape}")
print(f"  grad mean: {jnp.mean(g):.10f}")
print(f"  grad std: {jnp.std(g):.10f}")
print(f"  grad hash: {hashlib.md5(np.array(g).tobytes()).hexdigest()[:12]}")
print()

# =============================================================================
# Test 8: JIT compilation consistency
# =============================================================================
print("TEST 8: JIT consistency")

@jit
def jitted_computation(x):
    return jnp.sum(jnp.sin(x) * jnp.cos(x**2))

x = jax.random.normal(jax.random.PRNGKey(123), (1000,))
result1 = jitted_computation(x)
result2 = jitted_computation(x)  # second call uses cached
print(f"  result (call 1): {result1:.10f}")
print(f"  result (call 2): {result2:.10f}")
print(f"  match: {result1 == result2}")
print()

print("=" * 60)
print("DONE - Compare outputs between environments")
print("=" * 60)
