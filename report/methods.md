# Methods

## Model Architectures

### Restricted Boltzmann Machine (RBM)

**Architecture**:
```
Visible units:  n_v = 784 (28×28 MNIST)
Hidden units:   n_h = 256
Parameters:     ~200K
```

**Energy Function**:
```
E(v, h; θ) = -v^T W h - b^T v - c^T h
```

where θ = {W ∈ ℝ^(784×256), b ∈ ℝ^784, c ∈ ℝ^256}

**Implementation Details**:
- Binary visible and hidden units: v_i, h_j ∈ {0, 1}
- Sigmoid activation: p(h_j=1|v) = σ(∑_i W_ij v_i + c_j)
- Weight initialization: W ~ N(0, 0.01)
- Bias initialization: b = 0, c = 0

**Rationale**: RBMs provide a well-understood testbed with tractable inference, enabling direct analysis of CD-k effects without confounding architectural complexity.

### Convolutional Energy-Based Model (Conv-EBM)

**Architecture**:
```
Input:          32×32×3 (CIFAR-10 RGB)
Base channels:  64
Layers:         3 residual stages
Parameters:     ~4M (small), ~1M (tiny)
Output:         Scalar energy
```

**Network Structure**:
```
ConvEBM(
  (conv1): Conv2d(3, 64, 3×3, stride=1, padding=1)
  (bn1): BatchNorm2d(64)
  (layer1): ResBlock(64, 64)  × 2  # 32×32
  (layer2): ResBlock(64, 128) × 2  # 16×16  
  (layer3): ResBlock(128, 256) × 2 # 8×8
  (avgpool): AdaptiveAvgPool2d(1×1)
  (fc): Linear(256, 1)
)
```

**Residual Block**:
```
ResBlock(in_c, out_c):
  conv1: Conv(in_c, out_c, 3×3) → BN → Swish
  conv2: Conv(out_c, out_c, 3×3) → BN
  shortcut: Conv(in_c, out_c, 1×1) if needed
  output: Swish(conv2 + shortcut)
```

**Energy Function**:
```
E(x; θ) = fc(GlobalAvgPool(ConvNet(x)))
```

**Regularization**:
- Spectral normalization on all conv and linear layers
- Helps training stability and prevents mode collapse
- L2 regularization: λ = 0.001 on energies

**Rationale**: Modern architecture with residual connections and batch normalization, representative of current deep learning practice. Small size (4M params) enables training on Kaggle GPUs.

## Training Procedures

### RBM Training with CD-k

**Algorithm**:
```python
for epoch in epochs:
    for minibatch v_data:
        # Positive phase
        h_pos ~ p(h | v_data)
        
        # Negative phase (CD-k)
        v_neg = v_data
        for k_steps:
            h_neg ~ p(h | v_neg)
            v_neg ~ p(v | h_neg)
        
        # Update
        ΔW = lr * (v_data ⊗ h_pos - v_neg ⊗ h_neg)
        W += momentum * v_W + ΔW
```

**Hyperparameters**:
- Learning rate: 0.01 (initial), scheduled decay
- Momentum: 0.5 → 0.9 (annealed at epoch 5)
- Weight decay: 0.0001
- Batch size: 128
- Epochs: 30

**CD-k values**: k ∈ {1, 5, 10, 20}

### RBM Training with PCD

**Algorithm**:
```python
# Initialize persistent buffer
buffer = initialize_random(size=5000)

for epoch in epochs:
    for minibatch v_data:
        # Positive phase
        h_pos ~ p(h | v_data)
        
        # Negative phase (PCD-k)
        v_neg = sample_from_buffer(batch_size)
        for k_steps:
            h_neg ~ p(h | v_neg)
            v_neg ~ p(v | h_neg)
        
        # Update
        ΔW = lr * (v_data ⊗ h_pos - v_neg ⊗ h_neg)
        W += momentum * v_W + ΔW
        
        # Update buffer
        update_buffer(v_neg)
```

**PCD-specific hyperparameters**:
- Buffer size: 5000 samples
- Reinitialize: 5% of buffer per epoch
- Otherwise same as CD

### Conv-EBM Training with CD-k

**Algorithm (SGLD-based CD)**:
```python
for epoch in epochs:
    for minibatch x_data:
        # Positive phase: energy of real data
        E_pos = model(x_data)
        
        # Negative phase: sample via Langevin
        x_neg = initialize_uniform(-1, 1)
        x_neg.requires_grad = True
        
        for k_steps:
            E = model(x_neg)
            grad = autograd.grad(E.sum(), x_neg)[0]
            
            # Langevin update
            x_neg = x_neg - step_size * grad + noise * randn()
            x_neg = clamp(x_neg, -1, 1)
        
        # Contrastive loss
        E_neg = model(x_neg.detach())
        loss = E_pos.mean() - E_neg.mean()
        
        # Add regularization
        loss += 0.001 * (E_pos**2 + E_neg**2).mean()
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(params, max_norm=0.1)
        optimizer.step()
```

**Hyperparameters**:
- Optimizer: Adam(lr=0.0001, β1=0.0, β2=0.999)
- Langevin steps: k ∈ {5, 10, 20}
- Step size: ε = 0.01
- Noise scale: σ = 0.005
- Gradient clip: 0.01 (during Langevin), 0.1 (for model)
- Batch size: 64
- Epochs: 50

### Conv-EBM Training with PCD

**Algorithm**:
```python
# Initialize replay buffer
buffer = ReplayBuffer(size=5000, shape=(3,32,32))

for epoch in epochs:
    for minibatch x_data:
        E_pos = model(x_data)
        
        # Sample from replay buffer
        x_neg = buffer.sample(batch_size, reinit_prob=0.05)
        
        # Langevin dynamics
        for k_steps:
            x_neg = langevin_step(model, x_neg)
        
        E_neg = model(x_neg.detach())
        loss = E_pos.mean() - E_neg.mean() + reg
        
        optimize(loss)
        
        # Update buffer
        buffer.add(x_neg.detach())
```

**PCD-specific**:
- Buffer size: 5000 samples
- Reinit probability: 5%
- Otherwise same as CD

## Evaluation Metrics

### For RBMs (MNIST)

**1. Reconstruction Error**
```
RE = (1/N) ∑_i ||v_i - p(v|h)||²
```
where h ~ p(h|v_i), measuring how well model reconstructs data.

**2. Free Energy**
```
F(v) = -log ∑_h exp(-E(v,h))
```
Lower free energy on data indicates better fit.

**3. Pseudo-Likelihood**
```
PL(v) = ∑_i p(v_i | v_{-i})
```
Tractable approximation to likelihood.

**4. AIS Log-Likelihood**
Annealed Importance Sampling estimate:
- 100 chains, 1000 annealing steps
- Provides unbiased estimate of log p(v)
- Computationally expensive, run on subset

**5. MCMC Diagnostics**
- **Autocorrelation**: ρ(τ) = Corr(E_t, E_{t+τ})
- **Mixing time**: τ_mix = min{τ : |ρ(τ)| < 0.1}
- **ESS**: Effective sample size from chain

### For Conv-EBMs (CIFAR-10)

**1. Fréchet Inception Distance (FID)**
```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g))
```
where μ, Σ are mean and covariance of Inception features.
- Lower is better
- Measures distribution similarity

**2. Inception Score (IS)**
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```
- Higher is better
- Measures quality and diversity
- Computed with 10 splits

**3. LPIPS Diversity**
```
Diversity = E_{x,x'~p_g}[LPIPS(x, x')]
```
- Uses AlexNet perceptual features
- Higher indicates more diverse samples
- 1000 random pairs

**4. Energy Statistics**
```
E_gap = |E[E(x_real)] - E[E(x_fake)]|
```
- Should decrease during training
- Indicates distribution matching

**5. MCMC Diagnostics**
Same as RBMs, tracking energy autocorrelation during sampling.

## Sampling Procedures

### For RBMs

**Generation**:
```python
def generate_samples(rbm, n_samples, k_steps=1000):
    v = bernoulli(0.5, size=(n_samples, 784))
    
    for _ in range(k_steps):
        h = sample_h(v)
        v = sample_v(h)
    
    return v
```

Long chains (k=1000) ensure high-quality samples for evaluation.

### For Conv-EBMs

**Generation via Long Langevin Chains**:
```python
def generate_samples(model, n_samples, k_steps=200):
    x = uniform(-1, 1, size=(n_samples, 3, 32, 32))
    
    for _ in range(k_steps):
        x = langevin_step(model, x, 
                         step_size=0.01, 
                         noise=0.005)
    
    return x
```

Longer chains for evaluation (k=200) vs training (k=5-20).

## Implementation Details

### Software Stack
- **PyTorch** 2.0: Deep learning framework
- **torchvision**: Dataset loading and transforms
- **NumPy** 1.24: Numerical computations
- **Matplotlib/Seaborn**: Visualization
- **pytorch-fid**: FID computation
- **lpips**: Perceptual metrics

### Hardware
- **Kaggle GPU**: NVIDIA T4 (16GB) or P100 (16GB)
- **CPU**: Variable (4-8 cores)
- **RAM**: 16-32GB

### Computational Budget
- **RBM experiments**: 2-7 hours each
- **Conv-EBM experiments**: 4-7 hours each
- **Total compute**: ~60-80 GPU hours
- All within Kaggle's free tier limits

### Reproducibility
- Seeds: 42 for all experiments
- Deterministic operations where possible
- Code available: [GitHub link]
- Configs in appendix

## Statistical Analysis

### Comparison Methodology

1. **Within-model comparison**: Compare CD-k values for same architecture
2. **Cross-method comparison**: Compare CD vs PCD
3. **Cross-architecture comparison**: Compare RBM vs Conv-EBM trends

### Metrics Aggregation

- Final values: Average over last 5 epochs
- Error bars: Standard deviation where applicable
- Trends: Moving average with window=5

### Significance Testing

Due to computational constraints, we run single seeds but report:
- Consistent trends across k values
- Qualitative sample quality assessment
- Multiple complementary metrics

## Ablation Studies

### Parameters Investigated

**Langevin Step Size** (Conv-EBM):
- Values: {0.001, 0.005, 0.01, 0.02, 0.05}
- Fixed: k=20, noise=0.005

**Langevin Noise Scale** (Conv-EBM):
- Values: {0.001, 0.005, 0.01, 0.02}
- Fixed: k=20, step_size=0.01

**PCD Buffer Size** (Conv-EBM):
- Values: {1000, 3000, 5000, 10000}
- Fixed: k=10, other params default

### Analysis Approach

For each ablation:
1. Train model with varied parameter
2. Measure final FID and training time
3. Plot parameter vs quality
4. Identify optimal range

This methodology enables systematic investigation of CD-k effects while maintaining reproducibility and computational feasibility.