# Bayesian IGNO: Technical Specification

## 1. Recap: What IGNO Currently Does

### 1.1 The Architecture (Frozen After Training)

After training, you have these **frozen** components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINED IGNO MODEL                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Coefficient Encoder: E_θβ1 : a → β₁ ∈ ℝ^{d_β1}                │
│  (CNN + FFCN, Tanh output → β₁ ∈ [-1,1]^d)                     │
│                                                                 │
│  Boundary Encoder: E_θβ2 : g → β₂ ∈ ℝ^{d_β2}                   │
│  (One-hot for operator-based, or omitted for solution-based)   │
│                                                                 │
│  Solution Decoder: G_θu : β = (β₁, β₂) → u(x)                  │
│  (MultiONet architecture)                                       │
│                                                                 │
│  Coefficient Decoder: G_θa : β₁ → a(x)                         │
│  (MultiONet architecture)                                       │
│                                                                 │
│  Normalizing Flow: F_θNF : β₁ ↔ z ∈ ℝ^{d_β1}                   │
│  (RealNVP, 3 coupling layers)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Current Inversion: Point Estimation

Given noisy observations `u_obs = {u(x₁), ..., u(x_m)}`, IGNO solves:

```
β₁* = arg min F(β₁)
        β₁
```

Where the objective function is:

```
F(β₁) = F_data(β₁) + F_pde(β₁)

      = ρ_data/m · Σᵢ |G_θu(β₁)(xᵢ) - u_obs(xᵢ)|²     ← data mismatch
      
      + ρ_pde · ||R(G_θu(β₁), G_θa(β₁))||²            ← PDE residual
```

**Current Algorithm (deterministic):**
```python
# Initialize from flow
z = sample N(0, I)
β₁ = F_θNF_inverse(z)

# Gradient descent
for step in range(200-500):
    loss = F_data(β₁) + F_pde(β₁)
    β₁ = β₁ - lr * ∇_{β₁} loss

# Output single point estimate
a_recovered = G_θa(β₁)
```

**The problem:** This gives ONE answer. No uncertainty quantification.

---

## 2. The Bayesian Reformulation

### 2.1 Key Insight: F(β₁) IS a Negative Log-Posterior

The loss function F(β₁) can be interpreted probabilistically:

```
F(β₁) = -log p(β₁ | u_obs) + const
```

Let's derive this properly.

### 2.2 The Probabilistic Model

**Likelihood (data term):**

The observations are noisy measurements of the true solution:
```
u_obs(xᵢ) = u_true(xᵢ) + εᵢ,    εᵢ ~ N(0, σ²_data)
```

Given β₁, the predicted solution is `G_θu(β₁)(x)`. The likelihood is:

```
p(u_obs | β₁) = ∏ᵢ N(u_obs(xᵢ) | G_θu(β₁)(xᵢ), σ²_data)

             = (2πσ²_data)^{-m/2} exp(-1/(2σ²_data) · Σᵢ|G_θu(β₁)(xᵢ) - u_obs(xᵢ)|²)
```

Taking negative log:
```
-log p(u_obs | β₁) = m/2 · log(2πσ²_data) + 1/(2σ²_data) · Σᵢ|G_θu(β₁)(xᵢ) - u_obs(xᵢ)|²
```

**This is exactly F_data(β₁) up to constants, with ρ_data = 1/(2σ²_data)!**


**Physics Prior (PDE term):**

The PDE residual acts as a "soft constraint" or regularizing prior. We can interpret it as:

```
p_pde(β₁) ∝ exp(-ρ_pde · ||R(G_θu(β₁), G_θa(β₁))||²)
```

This says: "β₁ values that better satisfy the PDE are more probable."


**Latent Space Prior (from Normalizing Flow):**

The normalizing flow gives us an **explicit, learned prior** over β₁:

```
p_{β₁}(β₁) = p_z(F_θNF(β₁)) · |det ∂F_θNF/∂β₁|
```

Where `p_z(z) = N(z | 0, I)` is the standard Gaussian.

In log form:
```
log p_{β₁}(β₁) = log p_z(F_θNF(β₁)) + log|det ∂F_θNF/∂β₁|
               = -1/2 ||F_θNF(β₁)||² - d/2·log(2π) + log|det J_F(β₁)|
```

### 2.3 The Full Posterior

By Bayes' theorem:

```
p(β₁ | u_obs) ∝ p(u_obs | β₁) · p_pde(β₁) · p_{β₁}(β₁)
```

Taking negative log:

```
-log p(β₁ | u_obs) = -log p(u_obs | β₁) - log p_pde(β₁) - log p_{β₁}(β₁) + const

                   = 1/(2σ²_data) · Σᵢ|G_θu(β₁)(xᵢ) - u_obs(xᵢ)|²   ← F_data
                   + ρ_pde · ||R(G_θu(β₁), G_θa(β₁))||²              ← F_pde  
                   + 1/2 ||F_θNF(β₁)||² - log|det J_F(β₁)|           ← flow prior
                   + const
```

**This is the unnormalized negative log-posterior!**

Define:
```
U(β₁) := F_data(β₁) + F_pde(β₁) + F_prior(β₁)

where F_prior(β₁) = 1/2 ||F_θNF(β₁)||² - log|det J_F(β₁)|
```

Then: `p(β₁ | u_obs) ∝ exp(-U(β₁))`

---

## 3. The MCMC Algorithm

### 3.1 Goal

Instead of finding `β₁* = arg min U(β₁)`, we want to **sample** from `p(β₁ | u_obs) ∝ exp(-U(β₁))`.

This gives us a collection of samples `{β₁⁽¹⁾, β₁⁽²⁾, ..., β₁⁽ᴺ⁾}` that represent the full posterior.

### 3.2 Why Hamiltonian Monte Carlo (HMC)?

Basic Metropolis-Hastings would work but is slow in moderate dimensions (d_β1 ~ 16-64).

**HMC is ideal here because:**
1. β₁ is continuous and real-valued ✓
2. U(β₁) is differentiable (autodiff through decoders) ✓
3. Latent space is smooth and well-conditioned ✓
4. Dimension is moderate (not thousands) ✓

### 3.3 HMC Intuition

Think of β₁ as the position of a particle on a surface where height = U(β₁).

HMC introduces auxiliary "momentum" variables `p ∈ ℝ^{d_β1}` and simulates Hamiltonian dynamics:

```
H(β₁, p) = U(β₁) + 1/2 p^T M^{-1} p
           ↑         ↑
        potential  kinetic energy
        energy     (M = mass matrix)
```

The joint distribution is:
```
p(β₁, p) ∝ exp(-H(β₁, p)) = exp(-U(β₁)) · exp(-1/2 p^T M^{-1} p)
```

Marginalizing over p gives us `p(β₁) ∝ exp(-U(β₁))` — exactly what we want!

### 3.4 HMC Algorithm for Bayesian IGNO

```python
def hmc_sample_igno(
    u_obs,           # Observations: {(x_i, u_obs_i)}
    G_θu,            # Frozen solution decoder
    G_θa,            # Frozen coefficient decoder  
    F_θNF,           # Frozen normalizing flow
    ρ_data,          # Data weight
    ρ_pde,           # PDE weight
    n_samples,       # Number of posterior samples
    n_leapfrog,      # Leapfrog steps per sample (L)
    step_size,       # Leapfrog step size (ε)
    M=None           # Mass matrix (default: identity)
):
    """
    Sample from p(β₁ | u_obs) using HMC.
    """
    
    d = dim(β₁)
    if M is None:
        M = I_d  # Identity mass matrix
    M_inv = inverse(M)
    
    # === Define the potential energy (negative log-posterior) ===
    def U(β₁):
        # Data term
        u_pred = G_θu(β₁)  # Evaluate at sensor locations
        F_data = ρ_data * sum((u_pred(x_i) - u_obs_i)² for (x_i, u_obs_i) in observations)
        
        # PDE term
        a_pred = G_θa(β₁)
        residual = compute_pde_residual(u_pred, a_pred)  # Weak or strong form
        F_pde = ρ_pde * ||residual||²
        
        # Flow prior term
        z = F_θNF(β₁)                    # Forward pass through flow
        log_det_J = F_θNF.log_det_jacobian(β₁)
        F_prior = 0.5 * ||z||² - log_det_J
        
        return F_data + F_pde + F_prior
    
    # === Gradient of potential energy ===
    def grad_U(β₁):
        return autograd(U, β₁)  # PyTorch/JAX autodiff
    
    # === Leapfrog integrator ===
    def leapfrog(β₁, p, ε, L):
        β₁ = β₁.clone()
        p = p.clone()
        
        # Half step for momentum
        p = p - 0.5 * ε * grad_U(β₁)
        
        # Full steps
        for _ in range(L - 1):
            β₁ = β₁ + ε * M_inv @ p
            p = p - ε * grad_U(β₁)
        
        # Final position step
        β₁ = β₁ + ε * M_inv @ p
        
        # Half step for momentum
        p = p - 0.5 * ε * grad_U(β₁)
        
        return β₁, -p  # Negate momentum for reversibility
    
    # === Initialize from flow prior ===
    z_init = sample(N(0, I_d))
    β₁_current = F_θNF.inverse(z_init)
    
    samples = []
    accepts = 0
    
    # === Main HMC loop ===
    for i in range(n_samples):
        # Sample momentum
        p_current = sample(N(0, M))
        
        # Current Hamiltonian
        H_current = U(β₁_current) + 0.5 * p_current @ M_inv @ p_current
        
        # Leapfrog integration
        β₁_proposed, p_proposed = leapfrog(β₁_current, p_current, step_size, n_leapfrog)
        
        # Proposed Hamiltonian
        H_proposed = U(β₁_proposed) + 0.5 * p_proposed @ M_inv @ p_proposed
        
        # Metropolis acceptance
        log_accept_prob = H_current - H_proposed
        
        if log(uniform(0, 1)) < log_accept_prob:
            β₁_current = β₁_proposed
            accepts += 1
        
        samples.append(β₁_current.clone())
    
    print(f"Acceptance rate: {accepts / n_samples:.2%}")
    return samples
```

### 3.5 From β₁ Samples to Coefficient Field Samples

Once you have posterior samples `{β₁⁽¹⁾, ..., β₁⁽ᴺ⁾}`, getting coefficient field samples is trivial:

```python
def get_coefficient_samples(β₁_samples, G_θa, grid_points):
    """
    Transform β₁ samples to coefficient field samples.
    """
    a_samples = []
    for β₁ in β₁_samples:
        a = G_θa(β₁)  # Evaluate decoder at grid points
        a_samples.append(a)
    return a_samples
```

---

## 4. What You Can Compute from Posterior Samples

### 4.1 Point Estimates

**Posterior Mean (Bayes estimator):**
```python
a_mean = (1/N) * sum(G_θa(β₁⁽ⁱ⁾) for i in 1..N)
```

**Maximum A Posteriori (MAP):**
```python
i_map = argmin(U(β₁⁽ⁱ⁾) for i in 1..N)
a_map = G_θa(β₁_samples[i_map])
```

Note: MAP ≈ current IGNO point estimate (but sampled version)

### 4.2 Uncertainty Quantification

**Posterior Standard Deviation (per pixel):**
```python
a_std = sqrt((1/N) * sum((G_θa(β₁⁽ⁱ⁾) - a_mean)² for i in 1..N))
```

This gives you a spatial map of uncertainty!

**Credible Intervals:**
```python
# For each spatial location x:
a_values_at_x = [G_θa(β₁⁽ⁱ⁾)(x) for i in 1..N]
a_lower_95 = percentile(a_values_at_x, 2.5)
a_upper_95 = percentile(a_values_at_x, 97.5)
```

### 4.3 Visualizations

1. **Mean ± Std field:** Show recovered coefficient with uncertainty bands
2. **Sample gallery:** Display 9-16 posterior samples to show variability
3. **Marginal histograms:** For specific locations of interest
4. **Correlation analysis:** How do uncertainties at different locations covary?

---

## 5. Practical Considerations

### 5.1 Tuning HMC

| Parameter | Starting Value | How to Tune |
|-----------|---------------|-------------|
| `step_size` (ε) | 0.01 | Target 65-80% acceptance rate |
| `n_leapfrog` (L) | 10-50 | Longer = less correlated samples |
| `n_samples` | 1000-5000 | Until ESS > 100 per dimension |
| `n_warmup` | 500-1000 | Discard these (burn-in) |

### 5.2 Mass Matrix Adaptation

The mass matrix M should approximate the inverse posterior covariance. Options:

1. **Identity (simplest):** M = I
2. **Diagonal adaptation:** Estimate variance per dimension during warmup
3. **Full adaptation:** Use empirical covariance from warmup samples

### 5.3 Using NUTS (No U-Turn Sampler)

NUTS automatically tunes `n_leapfrog` and `step_size`. Highly recommended.

Available in:
- **NumPyro** (JAX-based, fast)
- **PyMC** (PyTensor-based)
- **BlackJAX** (JAX, minimal)

### 5.4 Computational Cost

| Step | Cost |
|------|------|
| One U(β₁) evaluation | 1 forward pass through G_θu + G_θa + F_θNF |
| One ∇U(β₁) evaluation | 1 backward pass (autodiff) |
| One leapfrog step | 1 grad_U evaluation |
| One HMC sample | L leapfrog steps = L grad_U evaluations |
| Full posterior (N samples) | N × L × (forward + backward pass) |

**Rough estimate:** If point estimation takes 300 iterations × 1 forward/backward each = 300 passes, then 2000 HMC samples × 20 leapfrog steps = 40,000 passes ≈ **130× slower**.

But you get full uncertainty quantification, not just a point estimate!

---

## 6. Complete Algorithm Summary

```
╔══════════════════════════════════════════════════════════════════╗
║           BAYESIAN IGNO: POSTERIOR SAMPLING ALGORITHM            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  INPUT:                                                          ║
║    • Observations: u_obs = {(x_i, u_obs_i)}_{i=1}^m              ║
║    • Trained IGNO: G_θu, G_θa, F_θNF (all frozen)               ║
║    • Hyperparameters: ρ_data, ρ_pde                              ║
║                                                                  ║
║  DEFINE POSTERIOR:                                               ║
║    U(β₁) = ρ_data · ||G_θu(β₁) - u_obs||²     [data]            ║
║          + ρ_pde  · ||R(G_θu(β₁), G_θa(β₁))||² [physics]        ║
║          + ½||F_θNF(β₁)||² - log|det J_F(β₁)|  [flow prior]     ║
║                                                                  ║
║    p(β₁ | u_obs) ∝ exp(-U(β₁))                                  ║
║                                                                  ║
║  SAMPLE (HMC/NUTS):                                              ║
║    1. Initialize: z ~ N(0,I), β₁⁰ = F_θNF⁻¹(z)                  ║
║    2. Warmup: Run sampler, adapt step size & mass matrix        ║
║    3. Sample: Collect {β₁⁽¹⁾, ..., β₁⁽ᴺ⁾}                       ║
║                                                                  ║
║  OUTPUT:                                                         ║
║    • Coefficient samples: a⁽ⁱ⁾ = G_θa(β₁⁽ⁱ⁾)                    ║
║    • Posterior mean: ā = (1/N) Σᵢ a⁽ⁱ⁾                          ║
║    • Posterior std:  σ_a = sqrt((1/N) Σᵢ (a⁽ⁱ⁾ - ā)²)           ║
║    • Credible intervals: [a_2.5%, a_97.5%] per location         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 7. Comparison: Point Estimation vs Bayesian

| Aspect | Current IGNO | Bayesian IGNO |
|--------|--------------|---------------|
| Output | Single β₁* | Distribution p(β₁ \| u_obs) |
| Uncertainty | None | Full posterior std, credible intervals |
| Computational cost | ~300 iterations | ~2000-5000 samples × ~20 steps |
| Initialization | Flow sample | Flow sample (same!) |
| Multimodality | Gets stuck in one mode | Can explore multiple modes |
| Use case | Fast point estimate | Decision-making under uncertainty |

---

## 8. Key Mathematical Relationships

### The Flow Prior Density

Given the normalizing flow `F_θNF : β₁ → z`, the density of β₁ is:

```
p_{β₁}(β₁) = p_z(z) · |det(∂z/∂β₁)|

where z = F_θNF(β₁)
```

For RealNVP with coupling layers, the log-determinant is cheap to compute:

```
log|det J| = Σ_layers Σ_dims s_layer(input)
```

where `s(·)` is the scale network in each coupling layer.

### Why the Prior Term Was Missing in Original IGNO

Original IGNO does `β₁* = arg min [F_data + F_pde]` without the flow prior term.

This is equivalent to assuming a **uniform prior** over β₁ (improper, but okay for optimization).

For proper Bayesian inference, we **must** include the flow prior because:
1. It regularizes toward the learned latent distribution
2. It makes the posterior normalizable
3. It provides the correct probabilistic interpretation

### Noise Model and ρ_data

If you believe observation noise is σ_data, then:

```
ρ_data = 1 / (2 σ²_data)
```

Larger ρ_data = smaller assumed noise = tighter likelihood = narrower posterior.
