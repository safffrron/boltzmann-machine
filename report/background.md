# Background

## Energy-Based Models

### Fundamentals

Energy-Based Models (EBMs) define a probability distribution over data **x** through an energy function E(**x**; θ):

```
p(x; θ) = exp(-E(x; θ)) / Z(θ)
```

where Z(θ) = ∫ exp(-E(**x**; θ)) d**x** is the intractable partition function. The energy function can be any parameterized function (neural network) that outputs a scalar "energy" for each input.

**Key Properties**:
- Lower energy → higher probability
- Flexible: any architecture can parameterize E(**x**; θ)
- Unnormalized: don't need to compute Z(θ) explicitly for many operations
- Challenging: sampling and gradient computation require MCMC

### Learning Objective

The maximum likelihood objective for EBMs is:

```
max_θ ∑_i log p(x_i; θ) = max_θ ∑_i [-E(x_i; θ) - log Z(θ)]
```

The gradient with respect to parameters θ is:

```
∇_θ log p(x; θ) = -∇_θ E(x; θ) + E_p(x')[∇_θ E(x'; θ)]
```

This has two terms:
1. **Positive phase**: Push down energy of observed data
2. **Negative phase**: Push up energy of samples from the model

The negative phase requires sampling from p(**x**; θ), which is computationally challenging.

## Restricted Boltzmann Machines

### Architecture

RBMs are a specific class of EBMs with:
- **Visible units** v ∈ {0,1}^n_v (observed data)
- **Hidden units** h ∈ {0,1}^n_h (latent features)
- **Bipartite structure**: no visible-visible or hidden-hidden connections

Energy function:
```
E(v, h) = -v^T W h - b^T v - c^T h
```

where W is the weight matrix, b and c are bias vectors.

### Inference

Due to the bipartite structure, inference is tractable:

```
p(h_j=1|v) = σ(∑_i W_ij v_i + c_j)
p(v_i=1|h) = σ(∑_j W_ij h_j + b_i)
```

where σ is the sigmoid function. This enables efficient Gibbs sampling.

### Gibbs Sampling

Starting from data **v**^(0):
1. Sample h^(t) ~ p(h|v^(t))
2. Sample v^(t+1) ~ p(v|h^(t))
3. Repeat

After sufficient steps, samples approximate p(**v**).

## Contrastive Divergence

### Motivation

Exact maximum likelihood learning requires:
```
∇_θ log Z(θ) = E_p(x)[∇_θ E(x; θ)]
```

This expectation requires sampling from the model distribution, typically needing very long MCMC chains (thousands of steps).

### CD-k Algorithm

Hinton (2002) proposed Contrastive Divergence as an efficient approximation:

**Algorithm: CD-k for RBM**
```
1. Initialize v^(0) = x_data (from training batch)
2. For t = 1 to k:
     Sample h^(t) ~ p(h | v^(t-1))
     Sample v^(t) ~ p(v | h^(t))
3. Compute gradient:
   ∇_θ L ≈ -∇_θ E(v^(0), h^(0)) + ∇_θ E(v^(k), h^(k))
```

**Key Insight**: Initialize chains from data, not random noise. After just k steps, the samples are "contrastive" to the data, providing useful gradient signal.

### Theoretical Justification

CD minimizes a different objective than maximum likelihood:
```
CD_k minimizes: KL(p_data || p_∞) - KL(p_k || p_∞)
```

where p_k is the distribution after k Gibbs steps starting from data.

**Bias-Variance Trade-off**:
- Small k: Fast, but biased gradient estimates
- Large k: More accurate, but higher variance and computational cost
- k=∞: Exact gradient, but intractable

## Persistent Contrastive Divergence

### Motivation

Standard CD reinitializes chains from data at each iteration. PCD (Tieleman, 2008) maintains persistent "fantasy particles" across iterations.

### PCD Algorithm

```
1. Maintain buffer of M persistent samples {v_1, ..., v_M}
2. At each training iteration:
   a. Sample minibatch from buffer
   b. Run k Gibbs steps on these samples
   c. Update parameters using these samples for negative phase
   d. Store updated samples back to buffer
3. Periodically reinitialize a small fraction (e.g., 5%) of buffer
```

### Advantages

1. **Better Mixing**: Chains continue from where they left off, avoiding reinitialization
2. **Lower Variance**: Samples better approximate p(x; θ_t)
3. **Asymptotic Correctness**: As k→∞, converges to true gradient

### Disadvantages

1. **Memory**: Must store buffer of samples (thousands)
2. **Stale Gradients**: Samples lag behind current parameters
3. **Initialization Sensitivity**: Initial buffer quality matters

## Langevin Dynamics for Continuous EBMs

### SGLD (Stochastic Gradient Langevin Dynamics)

For continuous data, we use Langevin dynamics instead of Gibbs sampling:

```
x_{t+1} = x_t - ε∇_x E(x_t) + √(2ε) z_t
```

where:
- ε is the step size
- z_t ~ N(0, I) is noise
- ∇_x E(x_t) is the gradient of energy w.r.t. input

### Properties

1. **Convergence**: As ε→0 and t→∞, samples converge to p(x)
2. **Flexibility**: Works for any differentiable E(x)
3. **Stability**: Requires careful tuning of ε and noise scale

### CD with Langevin

For CD-k with Langevin:
1. Initialize x^(0) from data or noise
2. Run k Langevin steps
3. Use x^(k) as negative samples
4. Update model parameters

## Related Work

### Early Energy-Based Models

- **Hopfield Networks** (1982): Binary associative memory
- **Boltzmann Machines** (1985): Fully connected, intractable
- **RBMs** (Smolensky, 1986): Bipartite structure enables tractable inference
- **DBNs** (Hinton et al., 2006): Stack of RBMs, enabled deep learning

### Modern Energy-Based Models

- **Convolutional EBMs** (Ngiam et al., 2011): Apply CNNs to energy modeling
- **Deep EBMs** (Du & Mordatch, 2019): Modern architectures with SGLD
- **Neural EBMs** (Nijkamp et al., 2019): Short-run MCMC for training
- **JEM** (Grathwohl et al., 2020): Joint energy-based modeling and classification

### Training Algorithms

- **CD-k** (Hinton, 2002): Foundational work
- **PCD** (Tieleman, 2008): Persistent chains
- **Fast PCD** (Tieleman & Hinton, 2009): Adaptive learning rates
- **Parallel Tempering** (Desjardins et al., 2010): Multiple temperature chains
- **Score Matching** (Hyvärinen, 2005): Alternative to maximum likelihood

### Theoretical Analysis

- **Convergence of CD** (Bengio & Delalleau, 2009): Conditions for convergence
- **Bias of CD** (Fischer & Igel, 2010): Systematic study of CD bias
- **PCD Theory** (Breuleux et al., 2011): Theoretical foundations
- **MCMC for EBMs** (Song & Kingma, 2021): Modern perspective

## Research Gaps

Despite extensive work, several questions remain:

1. **Practical Guidelines**: Limited systematic comparison of CD-k values
2. **Compute-Quality Trade-offs**: Few studies quantify efficiency vs accuracy
3. **Modern Architectures**: Most CD studies focus on RBMs, not deep EBMs
4. **MCMC Diagnostics**: Mixing properties often not rigorously analyzed
5. **Reproducibility**: Many results hard to reproduce due to implementation details

Our work addresses these gaps through systematic experimentation with modern evaluation metrics.

## Key Concepts for This Work

### Bias-Variance Trade-off in CD-k

- **k=1**: Very fast, highly biased, high variance in gradients
- **k=5**: Good balance for many applications
- **k=10-20**: Lower bias, but increased computational cost
- **k→∞**: Unbiased but impractical

### Mixing Time

Time for MCMC chain to reach stationary distribution:
- Measured via autocorrelation function
- Critical for sample quality
- Varies with model, data, and sampling procedure

### Energy Gap

Difference between energy of real data and generated samples:
- E_gap = E[E(x_real)] - E[E(x_fake)]
- Ideally converges to 0
- Indicates model has learned the distribution

### Sample Quality Metrics

- **FID**: Measures distribution similarity via Inception features
- **Inception Score**: Measures quality and diversity
- **LPIPS**: Perceptual similarity/diversity
- **Reconstruction Error**: For models with explicit reconstruction
- **Log-Likelihood**: Gold standard but often intractable (use AIS approximation)

These concepts form the foundation for our experimental analysis in subsequent sections.