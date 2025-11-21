# Experiments

## Overview

Our experimental study comprises three main sets:
- **Set A**: RBM experiments on MNIST (CD vs PCD, various k)
- **Set B**: Conv-EBM experiments on CIFAR-10 (CD vs PCD, various k)
- **Set C**: Ablation studies on key hyperparameters

All experiments are designed to run within Kaggle's computational constraints while providing meaningful insights into the CD-k trade-off.

## Datasets

### MNIST

**Description**: Handwritten digits dataset
- **Training**: 60,000 images
- **Test**: 10,000 images
- **Size**: 28×28 grayscale
- **Classes**: 10 (digits 0-9)

**Preprocessing**:
```python
# For RBM
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x > 0.5).float())  # Binarize
])
```

**Rationale**: Standard benchmark for RBMs, binary structure suits Bernoulli units, well-studied enabling comparison with literature.

### FashionMNIST

**Description**: Fashion product images
- **Training**: 60,000 images
- **Test**: 10,000 images
- **Size**: 28×28 grayscale
- **Classes**: 10 (clothing items)

**Usage**: Additional validation set for RBM, more complex than MNIST while maintaining same format.

### CIFAR-10

**Description**: Natural images dataset
- **Training**: 50,000 images
- **Test**: 10,000 images
- **Size**: 32×32 RGB
- **Classes**: 10 (airplane, automobile, bird, etc.)

**Preprocessing**:
```python
# Training
transform_train = Compose([
    RandomHorizontalFlip(),
    RandomCrop(32, padding=4),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])

# Testing
transform_test = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

**Rationale**: Standard benchmark for image generation, color images test Conv-EBM capabilities, challenging distribution for energy-based models.

## Experiment Set A: RBM on MNIST

### A1: CD-k Sweep

**Objective**: Measure how reconstruction quality scales with CD steps.

**Configurations**:

| Experiment | k | Method | Epochs | Batch Size | LR |
|------------|---|--------|--------|------------|-----|
| RBM-CD-1 | 1 | CD | 30 | 128 | 0.01 |
| RBM-CD-5 | 5 | CD | 30 | 128 | 0.01 |
| RBM-CD-10 | 10 | CD | 30 | 128 | 0.01 |
| RBM-CD-20 | 20 | CD | 25 | 64 | 0.01 |

**Config File**: `rbm_mnist_cd5.yaml` (example)
```yaml
exp_name: rbm_mnist_cd5
dataset: mnist
batch_size: 128
binarize: true
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
```

**Expected Runtime**: 2-6 hours per experiment

**Metrics**:
- Reconstruction error vs epoch
- Free energy vs epoch
- Final reconstruction error
- Training time

### A2: PCD Comparison

**Objective**: Compare PCD against CD for different k values.

**Configurations**:

| Experiment | k | Method | Buffer Size | Epochs |
|------------|---|--------|-------------|--------|
| RBM-PCD-1 | 1 | PCD | 5000 | 30 |
| RBM-PCD-5 | 5 | PCD | 5000 | 30 |
| RBM-PCD-10 | 10 | PCD | 5000 | 30 |

**Key Differences from CD**:
- Maintains persistent sample buffer
- 5% reinitialization per epoch
- Slightly larger memory footprint

**Expected Runtime**: 3-7 hours per experiment

**Metrics**:
- Same as A1, plus:
- CD vs PCD reconstruction error comparison
- Buffer sample quality over time

### A3: MCMC Diagnostics

**Objective**: Analyze mixing properties of MCMC chains.

**Procedure**:
1. After training, generate samples with long chains (k=1000)
2. Track energy at each step
3. Compute autocorrelation function
4. Estimate mixing time and ESS

**Analysis**:
- Autocorrelation ρ(τ) for τ ∈ [0, 100]
- Mixing time τ_mix (first time |ρ(τ)| < 0.1)
- Effective sample size
- Compare across CD-k values

### A4: Log-Likelihood Estimation

**Objective**: Estimate model likelihood using AIS.

**Procedure**:
- Select subset of 1000 test images
- Run AIS with 100 chains, 1000 annealing steps
- Compute bits per dimension
- Compare across CD-k values

**Note**: Computationally expensive, run selectively.

### A5: Qualitative Analysis

**Procedure**:
1. Generate 100 samples after training (k=1000 Gibbs steps)
2. Visualize as 10×10 grid
3. Assess visual quality: diversity, mode coverage, artifacts
4. Compare generations across CD-k values

## Experiment Set B: Conv-EBM on CIFAR-10

### B1: CD-k Sweep

**Objective**: Measure FID and IS scaling with Langevin steps.

**Configurations**:

| Experiment | k | Method | Epochs | Batch Size | LR |
|------------|---|--------|--------|------------|-----|
| Conv-Tiny-CD-5 | 5 | CD | 30 | 128 | 0.0001 |
| Conv-CD-5 | 5 | CD | 50 | 64 | 0.0001 |
| Conv-CD-10 | 10 | CD | 50 | 64 | 0.0001 |
| Conv-CD-20 | 20 | CD | 40 | 48 | 0.0001 |

**Config File**: `conv_cifar_cd10.yaml` (example)
```yaml
exp_name: conv_cifar_cd10
dataset: cifar10
batch_size: 64
augment: true
model_size: small
spectral_norm: true
learning_rate: 0.0001
beta1: 0.0
beta2: 0.999
weight_decay: 0.0
langevin_steps: 10
langevin_step_size: 0.01
langevin_noise: 0.005
langevin_clip: 0.01
grad_clip: 0.1
use_pcd: false
epochs: 50
seed: 42
```

**Expected Runtime**: 4-7 hours per experiment

**Metrics**:
- FID score (compute on 1000 generated samples)
- Inception Score with 10 splits
- LPIPS diversity
- Energy gap vs epoch
- Training loss curves

### B2: PCD Comparison

**Objective**: Evaluate PCD for continuous EBMs.

**Configurations**:

| Experiment | k | Method | Buffer Size | Epochs |
|------------|---|--------|-------------|--------|
| Conv-PCD-5 | 5 | PCD | 5000 | 50 |
| Conv-PCD-10 | 10 | PCD | 5000 | 50 |

**Expected Runtime**: 5-7 hours per experiment

**Metrics**:
- Same as B1
- CD vs PCD FID comparison
- Training stability (loss variance)

### B3: Sample Quality Over Training

**Objective**: Track sample quality evolution.

**Procedure**:
- Generate 64 samples every 5 epochs
- Save as grid images
- Compute FID at checkpoints
- Visualize quality progression

**Analysis**:
- When do recognizable structures appear?
- Does quality saturate or keep improving?
- Are there mode collapse issues?

### B4: Energy Landscape Analysis

**Objective**: Understand learned energy function.

**Procedure**:
1. Compute energy for test set (real data)
2. Generate samples and compute their energy
3. Plot energy histograms
4. Analyze energy gap over training

**Analysis**:
- Do real and fake energy distributions overlap?
- How does energy gap change with CD-k?
- Is the model assigning low energy to realistic images?

## Experiment Set C: Ablation Studies

### C1: Langevin Step Size

**Objective**: Find optimal step size for Langevin dynamics.

**Setup**:
- Base: Conv-EBM, k=20, 30 epochs
- Vary: step_size ∈ {0.001, 0.005, 0.01, 0.02, 0.05}
- Fixed: noise=0.005, other params default

**Analysis**:
- FID vs step size
- Training stability
- Sample quality
- Optimal range identification

### C2: Langevin Noise Scale

**Objective**: Determine appropriate noise injection.

**Setup**:
- Base: Conv-EBM, k=20, 30 epochs
- Vary: noise ∈ {0.001, 0.005, 0.01, 0.02}
- Fixed: step_size=0.01, other params default

**Analysis**:
- FID vs noise scale
- Sample diversity
- Chain mixing
- Noise-quality trade-off

### C3: PCD Buffer Size

**Objective**: Evaluate memory-quality trade-off.

**Setup**:
- Base: Conv-EBM, k=10, PCD, 30 epochs
- Vary: buffer_size ∈ {1000, 3000, 5000, 10000}
- Fixed: other params default

**Analysis**:
- FID vs buffer size
- Memory consumption
- Training time
- Diminishing returns point

### C4: Number of Sampling Steps

**Objective**: Quality vs computation during generation.

**Setup**:
- Trained model: Conv-CD-10
- Vary generation steps: {10, 50, 100, 200, 500}
- Measure FID for each

**Analysis**:
- FID vs sampling steps
- When does quality saturate?
- Minimum steps for acceptable quality

## Computational Resources

### Total Budget

| Resource | Amount |
|----------|--------|
| GPU Hours | ~70 hours |
| Experiments | 20 total |
| Storage | ~10 GB |
| RAM | 16-32 GB |

### Per-Experiment Breakdown

**RBM Experiments** (7 total):
- Average: 3-5 hours each
- Total: ~30 GPU hours

**Conv-EBM Experiments** (8 total):
- Average: 5-6 hours each
- Total: ~40 GPU hours

**Ablation Studies** (5 total):
- Average: 2-3 hours each
- Total: ~12 GPU hours

### Kaggle Optimization

All experiments designed to:
- Fit in 15GB GPU memory (T4/P100)
- Complete within 9-12 hour session limits
- Use efficient data loading (4 workers)
- Checkpoint frequently (every 5 epochs)
- Generate analysis plots automatically

## Reproducibility

### Seeds and Determinism

```python
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
```

### Version Control

- PyTorch: 2.0.0
- CUDA: 11.8
- Python: 3.10

### Code Availability

Complete codebase with:
- All model implementations
- Training scripts
- Evaluation scripts
- Configuration files
- Plotting utilities
- README with setup instructions

### Data Availability

- MNIST: Downloaded via torchvision
- CIFAR-10: Downloaded via torchvision
- Preprocessing code included
- Train/test splits: standard

### Expected Outputs

Each experiment produces:
- Model checkpoints (.pt files)
- Training logs (JSON)
- Generated samples (images)
- Evaluation metrics (JSON)
- Visualization plots (PNG)

All results organized in experiment-specific directories for easy comparison and analysis.

## Timeline

**Week 1**: Setup and RBM-CD experiments
**Week 2**: RBM-PCD experiments and evaluation
**Week 3**: Conv-EBM-CD experiments
**Week 4**: Conv-EBM-PCD and ablation studies
**Week 5**: Analysis and report writing

Total duration: 4-5 weeks for complete study.