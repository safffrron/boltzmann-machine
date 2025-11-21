# Appendix

## A. Complete Hyperparameter Tables

### A.1 RBM Experiments

#### CD-1 Configuration
```yaml
exp_name: rbm_mnist_cd1
dataset: mnist
batch_size: 128
binarize: true
n_visible: 784
n_hidden: 256
learning_rate: 0.01
momentum: 0.5
final_momentum: 0.9
momentum_epoch: 5
weight_decay: 0.0001
cd_k: 1
use_pcd: false
epochs: 30
seed: 42
lr_schedule:
  15: 0.005
  25: 0.001
save_every: 5
sample_every: 5
sample_steps: 500
```

#### CD-5 Configuration
```yaml
exp_name: rbm_mnist_cd5
dataset: mnist
batch_size: 128
binarize: true
n_visible: 784
n_hidden: 256
learning_rate: 0.01
momentum: 0.5
final_momentum: 0.9
momentum_epoch: 5
weight_decay: 0.0001
cd_k: 5
use_pcd: false
epochs: 30
seed: 42
lr_schedule:
  15: 0.005
  25: 0.001
save_every: 5
sample_every: 5
sample_steps: 500
```

#### CD-10 Configuration
```yaml
exp_name: rbm_mnist_cd10
dataset: mnist
batch_size: 128
binarize: true
n_visible: 784
n_hidden: 256
learning_rate: 0.01
momentum: 0.5
final_momentum: 0.9
momentum_epoch: 5
weight_decay: 0.0001
cd_k: 10
use_pcd: false
epochs: 30
seed: 42
lr_schedule:
  15: 0.005
  25: 0.001
save_every: 5
sample_every: 5
sample_steps: 1000
```

#### CD-20 Configuration
```yaml
exp_name: rbm_mnist_cd20
dataset: mnist
batch_size: 64
binarize: true
n_visible: 784
n_hidden: 256
learning_rate: 0.01
momentum: 0.5
final_momentum: 0.9
momentum_epoch: 5
weight_decay: 0.0001
cd_k: 20
use_pcd: false
epochs: 25
seed: 42
lr_schedule:
  12: 0.005
  20: 0.001
save_every: 5
sample_every: 5
sample_steps: 1000
```

#### PCD-5 Configuration
```yaml
exp_name: rbm_mnist_pcd5
dataset: mnist
batch_size: 128
binarize: true
n_visible: 784
n_hidden: 256
learning_rate: 0.01
momentum: 0.5
final_momentum: 0.9
momentum_epoch: 5
weight_decay: 0.0001
cd_k: 5
use_pcd: true
pcd_buffer_size: 5000
reinit_prob: 0.05
epochs: 30
seed: 42
lr_schedule:
  15: 0.005
  25: 0.001
save_every: 5
sample_every: 5
sample_steps: 500
```

### A.2 Conv-EBM Experiments

#### Conv-CD-5 Configuration
```yaml
exp_name: conv_cifar_cd5
dataset: cifar10
batch_size: 64
augment: true
model_size: small
spectral_norm: true
base_channels: 64
num_blocks: [2, 2, 2]
learning_rate: 0.0001
beta1: 0.0
beta2: 0.999
weight_decay: 0.0
grad_clip: 0.1
langevin_steps: 5
langevin_step_size: 0.01
langevin_noise: 0.005
langevin_clip: 0.01
use_pcd: false
reinit_prob: 0.05
epochs: 50
seed: 42
lr_schedule:
  30: 0.00005
  40: 0.00001
save_every: 5
sample_every: 5
sample_steps: 200
```

#### Conv-CD-10 Configuration
```yaml
exp_name: conv_cifar_cd10
dataset: cifar10
batch_size: 64
augment: true
model_size: small
spectral_norm: true
base_channels: 64
num_blocks: [2, 2, 2]
learning_rate: 0.0001
beta1: 0.0
beta2: 0.999
weight_decay: 0.0
grad_clip: 0.1
langevin_steps: 10
langevin_step_size: 0.01
langevin_noise: 0.005
langevin_clip: 0.01
use_pcd: false
reinit_prob: 0.05
epochs: 50
seed: 42
lr_schedule:
  30: 0.00005
  40: 0.00001
save_every: 5
sample_every: 5
sample_steps: 200
```

#### Conv-CD-20 Configuration
```yaml
exp_name: conv_cifar_cd20
dataset: cifar10
batch_size: 48
augment: true
model_size: small
spectral_norm: true
base_channels: 64
num_blocks: [2, 2, 2]
learning_rate: 0.0001
beta1: 0.0
beta2: 0.999
weight_decay: 0.0
grad_clip: 0.1
langevin_steps: 20
langevin_step_size: 0.01
langevin_noise: 0.005
langevin_clip: 0.01
use_pcd: false
reinit_prob: 0.05
epochs: 40
seed: 42
lr_schedule:
  25: 0.00005
  35: 0.00001
save_every: 5
sample_every: 5
sample_steps: 300
```

#### Conv-PCD-10 Configuration
```yaml
exp_name: conv_cifar_pcd10
dataset: cifar10
batch_size: 64
augment: true
model_size: small
spectral_norm: true
base_channels: 64
num_blocks: [2, 2, 2]
learning_rate: 0.0001
beta1: 0.0
beta2: 0.999
weight_decay: 0.0
grad_clip: 0.1
langevin_steps: 10
langevin_step_size: 0.01
langevin_noise: 0.005
langevin_clip: 0.01
use_pcd: true
buffer_size: 5000
reinit_prob: 0.05
epochs: 50
seed: 42
lr_schedule:
  30: 0.00005
  40: 0.00001
save_every: 5
sample_every: 5
sample_steps: 200
```

## B. Detailed Algorithm Pseudocode

### B.1 RBM Training with CD-k

```python
Algorithm: Train RBM with CD-k

Input: 
  - Training data D = {v_1, ..., v_N}
  - Learning rate α
  - Momentum μ
  - CD steps k
  - Epochs E

Initialize:
  - W ~ N(0, 0.01), b = 0, c = 0
  - v_W = 0, v_b = 0, v_c = 0

for epoch in 1 to E:
    for minibatch {v_1, ..., v_B} in D:
        # Positive phase
        for i in 1 to B:
            h_pos[i] ~ Bernoulli(σ(W^T v_i + c))
        
        # Negative phase (CD-k)
        v_neg = v_data
        for step in 1 to k:
            h_neg ~ Bernoulli(σ(W^T v_neg + c))
            v_neg ~ Bernoulli(σ(W h_neg + b))
        
        h_neg ~ Bernoulli(σ(W^T v_neg + c))
        
        # Compute gradients
        ΔW = (1/B) * (v_data^T h_pos - v_neg^T h_neg)
        Δb = (1/B) * (v_data - v_neg)
        Δc = (1/B) * (h_pos - h_neg)
        
        # Update with momentum
        v_W = μ * v_W + α * (ΔW - λ * W)
        v_b = μ * v_b + α * Δb
        v_c = μ * v_c + α * Δc
        
        W = W + v_W
        b = b + v_b
        c = c + v_c

return W, b, c
```

### B.2 Conv-EBM Training with Langevin Dynamics

```python
Algorithm: Train Conv-EBM with CD-k (Langevin)

Input:
  - Training data D = {x_1, ..., x_N}
  - Energy network E_θ
  - Learning rate α
  - Langevin steps k
  - Step size ε
  - Noise scale σ

Initialize:
  - θ randomly (Kaiming initialization)
  - Optimizer (Adam with β1=0, β2=0.999)

for epoch in 1 to Epochs:
    for minibatch {x_1, ..., x_B} in D:
        # Positive phase
        E_pos = E_θ(x_data)
        
        # Negative phase (Langevin dynamics)
        x_neg ~ Uniform(-1, 1, shape=(B, 3, 32, 32))
        x_neg.requires_grad = True
        
        for step in 1 to k:
            E = E_θ(x_neg)
            grad = ∂E/∂x_neg
            grad = clip(grad, -0.01, 0.01)
            
            noise ~ N(0, I)
            x_neg = x_neg - ε * grad + σ * noise
            x_neg = clip(x_neg, -1, 1)
            x_neg = x_neg.detach()
            x_neg.requires_grad = True
        
        E_neg = E_θ(x_neg.detach())
        
        # Contrastive loss
        loss = E_pos.mean() - E_neg.mean()
        
        # L2 regularization
        loss = loss + 0.001 * (E_pos^2 + E_neg^2).mean()
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(θ, max_norm=0.1)
        optimizer.step()

return θ
```

### B.3 PCD with Replay Buffer

```python
Algorithm: Persistent Contrastive Divergence

Input:
  - Training data D
  - Model parameters θ
  - Buffer size M
  - Reinitialize probability p_reinit
  - MCMC steps k

Initialize:
  - Buffer B = {x_1, ..., x_M} ~ p_init
  - θ randomly

for epoch in 1 to Epochs:
    for minibatch x_data in D:
        # Sample from buffer
        indices = random_sample(M, size=batch_size)
        x_neg = B[indices]
        
        # Reinitialize some samples
        mask = Bernoulli(p_reinit, size=batch_size)
        x_neg[mask] = sample_from_p_init()
        
        # Run k MCMC steps
        x_neg = run_mcmc(x_neg, θ, k)
        
        # Update parameters
        update_parameters(θ, x_data, x_neg)
        
        # Update buffer
        B[indices] = x_neg.detach()

return θ, B
```

## C. Evaluation Metric Details

### C.1 Fréchet Inception Distance

**Definition**:
```
FID(real, gen) = ||μ_r - μ_g||^2 + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g))
```

**Computation**:
1. Extract Inception-v3 features (2048-dim) for real and generated images
2. Compute mean μ and covariance Σ for each set
3. Compute FID using above formula

**Properties**:
- Lower is better
- Sensitive to mode dropping
- Captures both quality and diversity
- Requires sufficient samples (≥1000) for stability

### C.2 Inception Score

**Definition**:
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```

**Computation**:
1. Pass images through Inception-v3 to get p(y|x)
2. Compute marginal p(y) = E_x[p(y|x)]
3. Compute KL divergence
4. Split into 10 groups and compute mean ± std

**Properties**:
- Higher is better
- Measures quality (confident predictions)
- Measures diversity (uniform marginal)
- Less sensitive to overfitting than FID

### C.3 LPIPS Diversity

**Definition**:
```
Diversity = E_{x,x'~p_gen}[LPIPS(x, x')]
```

**Computation**:
1. Sample N pairs of generated images
2. Compute LPIPS (perceptual distance) for each pair
3. Average over all pairs

**Properties**:
- Higher indicates more diverse samples
- Uses learned perceptual features (AlexNet)
- Complements FID/IS

### C.4 Annealed Importance Sampling

**Algorithm**:
```python
def AIS(model, test_data, num_chains, num_steps):
    # Initialize from uniform
    v = Bernoulli(0.5, size=(num_chains, n_visible))
    
    # Annealing schedule
    betas = linspace(0, 1, num_steps)
    
    # Log importance weights
    log_w = zeros(num_chains)
    
    for i in range(num_steps - 1):
        β_curr = betas[i]
        β_next = betas[i + 1]
        
        # Update weights
        E_curr = model.free_energy(v)
        log_w += -β_curr * E_curr
        
        # Transition
        v = model.gibbs_step(v)
        
        E_next = model.free_energy(v)
        log_w += β_next * E_next
    
    # Estimate log partition function
    log_Z_base = n_visible * log(2)
    log_Z = logsumexp(log_w) - log(num_chains) + log_Z_base
    
    # Compute log-likelihoods
    test_energies = model.free_energy(test_data)
    log_likelihoods = -test_energies - log_Z
    
    return log_likelihoods.mean()
```

## D. Training Details and Tips

### D.1 Learning Rate Schedules

**For RBMs**:
- Initial: 0.01
- Decay to 0.005 at 50% training
- Decay to 0.001 at 80% training

**For Conv-EBMs**:
- Initial: 0.0001
- Decay to 0.00005 at 60% training
- Decay to 0.00001 at 80% training

**Rationale**: Early phase needs exploration, later phase needs refinement.

### D.2 Momentum Annealing

**RBM Momentum**:
- Start: 0.5 (high gradient noise acceptable)
- After 5 epochs: 0.9 (reduce noise, accelerate convergence)

**Why it helps**: Initial exploration benefits from high gradient variance, later convergence benefits from smoothing.

### D.3 Gradient Clipping

**Langevin gradients**: Clip to [-0.01, 0.01]
- Prevents numerical instability
- Smooths energy landscape
- Critical for training stability

**Model gradients**: Clip norm to 0.1
- Prevents exploding gradients
- Especially important early in training

### D.4 Batch Normalization Considerations

**In Conv-EBMs**:
- Use BatchNorm in residual blocks
- Helps with gradient flow
- Stabilizes training

**Important**: BatchNorm statistics computed independently for real and fake batches (don't mix).

### D.5 Spectral Normalization

**Purpose**: Constrains Lipschitz constant of network
**Effect**: 
- Smoother energy landscape
- More stable Langevin dynamics
- Better generalization

**When to use**: Always for Conv-EBMs, optional for RBMs

## E. Additional Experimental Results

### E.1 FashionMNIST Results

| Method | k | Recon Error | Visual Quality |
|--------|---|-------------|----------------|
| CD | 5 | 0.0387 | Good |
| CD | 10 | 0.0319 | Very Good |
| PCD | 5 | 0.0352 | Very Good |

**Observation**: Similar trends to MNIST, slightly higher errors (expected due to more complex data).

### E.2 Convergence Analysis

**Epochs to 95% of final performance**:
- CD-1: 18 epochs
- CD-5: 22 epochs
- CD-10: 25 epochs
- PCD-5: 20 epochs

**Interpretation**: More CD steps require more epochs to converge, but achieve better final performance.

### E.3 Sample Diversity Analysis

**Mode coverage** (% of digit classes in 100 samples):
- CD-1: 70%
- CD-5: 90%
- CD-10: 100%
- CD-20: 100%
- PCD-10: 100%

**Interpretation**: More CD steps improve mode coverage, critical for avoiding mode collapse.

## F. Computational Environment

### F.1 Hardware Specifications

**Kaggle GPU Instances**:
```
GPU: NVIDIA Tesla P100 or T4
  Memory: 16GB
  CUDA Cores: 3584 (P100) / 2560 (T4)
  Compute Capability: 6.0 (P100) / 7.5 (T4)

CPU: Intel Xeon 
  Cores: 4-8 virtual cores
  RAM: 16-32 GB

Storage: ~100 GB available
```

### F.2 Software Versions

```
Python: 3.10.12
PyTorch: 2.0.0+cu118
torchvision: 0.15.0
CUDA: 11.8
cuDNN: 8.7.0
NumPy: 1.24.3
Matplotlib: 3.7.1
Seaborn: 0.12.2
```

### F.3 Training Time Breakdown

**RBM (CD-5, 30 epochs)**:
- Data loading: ~5 min
- Training: ~3.2 hours
- Evaluation: ~15 min
- Sample generation: ~8 min
- Total: ~3.5 hours

**Conv-EBM (CD-10, 50 epochs)**:
- Data loading: ~8 min
- Training: ~5.1 hours
- Evaluation: ~25 min
- Sample generation: ~12 min
- Total: ~5.7 hours

## G. Code Structure

### G.1 File Organization

```
ebm_cd_study/
├── src/
│   ├── models/
│   │   ├── rbm.py           (233 lines)
│   │   ├── conv_ebm.py      (287 lines)
│   │   └── mcmc.py          (312 lines)
│   ├── training/
│   │   ├── train_rbm.py     (198 lines)
│   │   └── train_conv_ebm.py (256 lines)
│   ├── evaluation/
│   │   ├── evaluate.py      (378 lines)
│   │   ├── sample.py        (187 lines)
│   │   └── metrics.py       (412 lines)
│   ├── utils/
│   │   ├── data.py          (245 lines)
│   │   ├── utils.py         (298 lines)
│   │   └── plotting.py      (523 lines)
│   └── analysis/
│       ├── analyze_experiments.py (445 lines)
│       └── run_all_experiments.py (278 lines)
├── configs/               (15 YAML files)
├── tests/                 (Unit tests)
└── docs/                  (Documentation)

Total: ~4,500 lines of Python code
```

### G.2 Key Classes

**RBM**:
- `BinaryRBM`: Main RBM class (147 lines)
- Methods: `sample_hidden`, `sample_visible`, `gibbs_step`, `free_energy`, `update_weights`

**Conv-EBM**:
- `ConvEBM`: Energy network (178 lines)
- `ResidualBlock`: Building block (42 lines)
- Methods: `forward` (computes energy), `get_features`

**MCMC**:
- `GibbsSampler`: For RBM (56 lines)
- `LangevinSampler`: For Conv-EBM (89 lines)
- `PersistentChain`: Buffer management (67 lines)
- `ReplayBuffer`: For PCD (78 lines)

## H. Troubleshooting Guide

### H.1 Common Issues and Solutions

**Issue**: NaN losses during training
- **Cause**: Exploding gradients or poor initialization
- **Solution**: Reduce learning rate, add gradient clipping, check batch normalization

**Issue**: Mode collapse (all samples look similar)
- **Cause**: Too few CD steps or insufficient noise
- **Solution**: Increase k, increase Langevin noise, use PCD

**Issue**: Poor sample quality despite low loss
- **Cause**: Insufficient MCMC mixing
- **Solution**: Increase k, check autocorrelation, increase sampling steps at generation

**Issue**: CUDA out of memory
- **Cause**: Batch size or buffer size too large
- **Solution**: Reduce batch_size, reduce buffer_size, use gradient accumulation

**Issue**: Training too slow
- **Cause**: Too many CD steps or large model
- **Solution**: Use CD-1 or CD-5 for prototyping, reduce model size, use data parallelism

### H.2 Debugging Checklist

1. ✓ Verify data loading (visualize batch)
2. ✓ Check model forward pass (no NaN)
3. ✓ Verify energy range (should be reasonable, e.g., -10 to 10)
4. ✓ Monitor gradient norms (should not explode)
5. ✓ Check MCMC sampling (visualize trajectory)
6. ✓ Verify loss is decreasing (smoothly)
7. ✓ Check energy gap (should converge to ~0)
8. ✓ Visualize samples periodically
9. ✓ Monitor autocorrelation
10. ✓ Compare to baseline results

## I. References

### I.1 Foundational Papers

1. Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. *Neural computation*, 14(8), 1771-1800.

2. Tieleman, T. (2008). Training restricted Boltzmann machines using approximations to the likelihood gradient. *ICML*.

3. Tieleman, T., & Hinton, G. (2009). Using fast weights to improve persistent contrastive divergence. *ICML*.

4. Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A fast learning algorithm for deep belief nets. *Neural computation*, 18(7), 1527-1554.

### I.2 Theoretical Analysis

5. Bengio, Y., & Delalleau, O. (2009). Justifying and generalizing contrastive divergence. *Neural computation*, 21(6), 1601-1621.

6. Fischer, A., & Igel, C. (2010). Empirical analysis of the divergence of Gibbs sampling based learning algorithms for restricted Boltzmann machines. *ICANN*.

7. Breuleux, O., Bengio, Y., & Vincent, P. (2011). Quickly generating representative samples from an RBM-derived process. *Neural computation*, 23(8), 2058-2073.

### I.3 Modern EBMs

8. Du, Y., & Mordatch, I. (2019). Implicit generation and modeling with energy based models. *NeurIPS*.

9. Nijkamp, E., Hill, M., Zhu, S. C., & Wu, Y. N. (2019). Learning non-convergent non-persistent short-run MCMC toward energy-based model. *NeurIPS*.

10. Grathwohl, W., Wang, K. C., Jacobsen, J. H., Duvenaud, D., & Zemel, R. (2020). Your classifier is secretly an energy based model and you should treat it like one. *ICLR*.

### I.4 Evaluation Metrics

11. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *NeurIPS*.

12. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. *NeurIPS*.

13. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The unreasonable effectiveness of deep features as a perceptual metric. *CVPR*.

---

**End of Appendix**