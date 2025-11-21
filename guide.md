# Complete Experiments Guide

Comprehensive guide for running all EBM experiments and generating analysis.

## � Table of Contents

1. [Quick Start](#quick-start)
2. [Experiment Sets](#experiment-sets)
3. [Running Individual Experiments](#running-individual-experiments)
4. [Running Complete Experiment Suite](#running-complete-experiment-suite)
5. [Generating Analysis](#generating-analysis)
6. [Expected Results](#expected-results)
7. [Troubleshooting](#troubleshooting)

---

## � Quick Start

### Option 1: Run Everything (Long)

```bash
# This will run all experiments and generate analysis
# WARNING: Takes 20-40 hours total on Kaggle
python src/run_all_experiments.py
```

### Option 2: Quick Test (5 minutes)

```bash
# Test that everything works
python src/run_all_experiments.py --quick
```

### Option 3: Run Specific Set

```bash
# Only RBM experiments (8-15 hours)
python src/run_all_experiments.py --rbm-only

# Only Conv-EBM experiments (12-25 hours)
python src/run_all_experiments.py --conv-only
```

---

## � Experiment Sets

### Set A: RBM on MNIST

**Goal**: Compare CD-k vs PCD-k for different k values

| Experiment | CD Steps | Method | Time | GPU Memory | Config |
|------------|----------|--------|------|------------|--------|
| RBM-CD-1 | 1 | CD | 2-3h | ~2GB | `rbm_mnist_cd1.yaml` |
| RBM-CD-5 | 5 | CD | 3-4h | ~2GB | `rbm_mnist_cd5.yaml` |
| RBM-CD-10 | 10 | CD | 4-5h | ~2-3GB | `rbm_mnist_cd10.yaml` |
| RBM-CD-20 | 20 | CD | 5-6h | ~3GB | `rbm_mnist_cd20.yaml` |
| RBM-PCD-1 | 1 | PCD | 3-4h | ~3GB | `rbm_mnist_pcd1.yaml` |
| RBM-PCD-5 | 5 | PCD | 4-5h | ~3-4GB | `rbm_mnist_pcd5.yaml` |
| RBM-PCD-10 | 10 | PCD | 5-7h | ~4GB | `rbm_mnist_pcd10.yaml` |

**Metrics Measured**:
- Reconstruction error
- Pseudo-likelihood
- AIS log-likelihood
- Free energy
- MCMC autocorrelation
- Mixing time
- Effective sample size

### Set B: Conv-EBM on CIFAR-10

**Goal**: Compare CD-k and PCD-k with Langevin dynamics

| Experiment | Langevin Steps | Method | Time | GPU Memory | Config |
|------------|---------------|--------|------|------------|--------|
| Conv-Tiny | 5 | CD | 2-3h | ~2-3GB | `conv_cifar_tiny_cd5.yaml` |
| Conv-CD-5 | 5 | CD | 4-5h | ~4-5GB | `conv_cifar_cd5.yaml` |
| Conv-CD-10 | 10 | CD | 5-6h | ~4-5GB | `conv_cifar_cd10.yaml` |
| Conv-CD-20 | 20 | CD | 6-7h | ~4-5GB | `conv_cifar_cd20.yaml` |
| Conv-PCD-5 | 5 | PCD | 5-6h | ~6GB | `conv_cifar_pcd5.yaml` |
| Conv-PCD-10 | 10 | PCD | 6-7h | ~6GB | `conv_cifar_pcd10.yaml` |

**Metrics Measured**:
- FID (Fréchet Inception Distance)
- Inception Score
- LPIPS diversity
- Energy statistics (real vs generated)
- MCMC autocorrelation
- Training loss curves

### Set C: Ablation Studies

**Goal**: Investigate hyperparameter sensitivity

| Study | Parameter Varied | Values Tested | Config |
|-------|------------------|---------------|--------|
| Step Size | `langevin_step_size` | 0.001, 0.005, 0.01, 0.02, 0.05 | `conv_cifar_ablation_step.yaml` |
| Noise | `langevin_noise` | 0.001, 0.005, 0.01, 0.02 | `conv_cifar_ablation_noise.yaml` |
| Buffer Size | `buffer_size` | 1000, 3000, 5000, 10000 | `conv_cifar_ablation_buffer.yaml` |

---

## � Running Individual Experiments

### RBM Training

```bash
# Train RBM with CD-5
python src/train_rbm.py --config configs/rbm_mnist_cd5.yaml

# Monitor progress
tail -f results/rbm_mnist_cd5_*/logs/*.log
```

### Conv-EBM Training

```bash
# Train Conv-EBM with CD-10
python src/train_conv_ebm.py --config configs/conv_cifar_cd10.yaml

# Monitor progress
tail -f results/conv_cifar_cd10_*/logs/*.log
```

### Generate Samples

```bash
# Generate samples from trained RBM
python src/sample.py \
    --checkpoint results/rbm_mnist_cd5_*/checkpoints/rbm_best.pt \
    --model_type rbm \
    --num_samples 100 \
    --num_steps 1000 \
    --output samples/rbm_cd5

# Generate samples from trained Conv-EBM
python src/sample.py \
    --checkpoint results/conv_cifar_cd10_*/checkpoints/conv_ebm_best.pt \
    --model_type conv_ebm \
    --config configs/conv_cifar_cd10.yaml \
    --num_samples 100 \
    --num_steps 200 \
    --output samples/conv_cd10
```

### Run Evaluation

```bash
# Evaluate RBM
python src/evaluate.py \
    --checkpoint results/rbm_mnist_cd5_*/checkpoints/rbm_best.pt \
    --model_type rbm \
    --num_samples 1000 \
    --output evaluation/rbm_cd5

# Evaluate Conv-EBM
python src/evaluate.py \
    --checkpoint results/conv_cifar_cd10_*/checkpoints/conv_ebm_best.pt \
    --model_type conv_ebm \
    --config configs/conv_cifar_cd10.yaml \
    --num_samples 1000 \
    --metrics fid is lpips \
    --output evaluation/conv_cd10
```

---

## � Generating Analysis

### Automatic Analysis

```bash
# Analyze all results and generate plots
python src/analyze_experiments.py \
    --results_dir ./results \
    --output_dir ./analysis
```

This generates:

**RBM Analysis** (`analysis/rbm/`):
- `rbm_recon_error_vs_cd.png` - Reconstruction error comparison
- `rbm_cd_vs_pcd.png` - CD vs PCD comparison
- `rbm_autocorrelations.png` - MCMC mixing comparison
- `*_training.png` - Individual training curves

**Conv-EBM Analysis** (`analysis/conv_ebm/`):
- `conv_fid_vs_cd.png` - FID score comparison
- `conv_is_vs_cd.png` - Inception Score comparison
- `conv_cd_vs_pcd_fid.png` - CD vs PCD comparison
- `conv_energy_gap_vs_cd.png` - Energy gap analysis
- `conv_compute_vs_quality.png` - Efficiency analysis
- `*_training.png` - Individual training curves

**Summary**:
- `summary_report.txt` - Text summary of all experiments

### Custom Plotting

```python
from plotting import *
import json

# Load specific experiment results
with open('results/rbm_mnist_cd5_*/logs/metrics.json') as f:
    metrics = json.load(f)

# Plot custom comparison
plot_cd_comparison(
    {1: 0.05, 5: 0.03, 10: 0.025, 20: 0.022},
    'custom_plot.png',
    metric_name='reconstruction_error',
    title='Custom Comparison'
)
```

---

## � Expected Results

### RBM on MNIST

| Metric | CD-1 | CD-5 | CD-10 | CD-20 | PCD-5 | PCD-10 |
|--------|------|------|-------|-------|-------|--------|
| Recon Error | ~0.05 | ~0.03 | ~0.025 | ~0.022 | ~0.028 | ~0.023 |
| Mixing Time | ~15 | ~10 | ~8 | ~7 | ~8 | ~6 |
| Training Time | 2-3h | 3-4h | 4-5h | 5-6h | 4-5h | 5-7h |

**Key Findings**:
- Reconstruction error decreases with more CD steps
- PCD shows slightly better mixing with fewer steps
- Diminishing returns beyond CD-10 for MNIST
- CD-5 offers good balance of speed and quality

### Conv-EBM on CIFAR-10

| Metric | CD-5 | CD-10 | CD-20 | PCD-5 | PCD-10 |
|--------|------|-------|-------|-------|--------|
| FID | ~45 | ~40 | ~38 | ~42 | ~37 |
| IS | ~5.5 | ~6.0 | ~6.5 | ~5.8 | ~6.3 |
| Energy Gap | ~8 | ~6 | ~5 | ~7 | ~5 |
| Training Time | 4-5h | 5-6h | 6-7h | 5-6h | 6-7h |

**Key Findings**:
- More Langevin steps improve sample quality
- PCD provides more stable training
- CD-10 to CD-20 shows good improvement
- FID around 35-40 is reasonable for small models

---

## �️ Troubleshooting

### Problem: CUDA Out of Memory

**Solutions**:

```yaml
# In config file, reduce:
batch_size: 32          # from 64
buffer_size: 3000       # from 5000 (for PCD)
n_hidden: 128          # from 256 (for RBM)
model_size: tiny       # from small (for Conv-EBM)
```

### Problem: Training Too Slow

**Solutions**:

```yaml
# Reduce training:
epochs: 20             # from 30-50
cd_k: 5               # from 10 or 20
langevin_steps: 10    # from 20
sample_steps: 200     # from 500
```

### Problem: NaN Losses

**Solutions**:

```yaml
# For RBM:
learning_rate: 0.005   # from 0.01
weight_decay: 0.0001   # add regularization

# For Conv-EBM:
learning_rate: 0.00005  # from 0.0001
langevin_step_size: 0.005  # from 0.01
grad_clip: 0.05        # from 0.1
```

### Problem: Poor Sample Quality

**Diagnosis**:
1. Check energy gap - should decrease during training
2. Check autocorrelation - should decay quickly
3. Check training curves - should be stable

**Solutions**:
- Increase CD/Langevin steps
- Use PCD instead of CD
- Train for more epochs
- Adjust learning rate
- Increase model capacity

### Problem: Samples Look Like Noise

**Solutions**:
```yaml
# Increase sampling quality:
sample_steps: 1000     # from 500
langevin_step_size: 0.01  # proper step size
langevin_noise: 0.005  # not too high
```

---

## � Tips for Success

### 1. Start Small
- Always run quick test first
- Use tiny model for initial experiments
- Start with CD-1 or CD-5

### 2. Monitor Progress
```bash
# Watch training live
watch -n 1 'tail -20 results/*/logs/*.log'

# Check GPU usage
watch -n 1 nvidia-smi
```

### 3. Save Checkpoints Frequently
```yaml
save_every: 5  # Save every 5 epochs
```

### 4. Compare Incrementally
- Run CD-1, then CD-5, compare
- If CD-5 better, try CD-10
- Don't jump straight to CD-20

### 5. Resource Management
- Clear GPU cache between experiments:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```
- Close other applications on Kaggle
- Download results periodically

---

## � Recommended Workflow

### Week 1: Setup & Quick Experiments
1. Setup environment
2. Run quick test (5 min)
3. Run RBM-CD-1 (2-3h)
4. Run Conv-Tiny (2-3h)
5. Verify results

### Week 2: RBM Experiments
1. Run RBM-CD-5 (3-4h)
2. Run RBM-CD-10 (4-5h)
3. Run RBM-PCD-5 (4-5h)
4. Analyze RBM results
5. Generate RBM plots

### Week 3: Conv-EBM Experiments
1. Run Conv-CD-5 (4-5h)
2. Run Conv-CD-10 (5-6h)
3. Run Conv-CD-20 (6-7h)
4. Analyze Conv-EBM results
5. Generate comparison plots

### Week 4: Final Analysis
1. Run PCD experiments
2. Run ablation studies
3. Generate all analysis plots
4. Write final report
5. Compile results

**Total Estimated Time**: 40-60 hours of compute across 4 weeks

---

## � Success Criteria

Your experiments are successful if:

1. ✅ All training runs complete without errors
2. ✅ Generated samples look reasonable (not random noise)
3. ✅ Metrics show expected trends (better with more CD steps)
4. ✅ Energy gap decreases during training
5. ✅ Autocorrelation shows mixing is happening
6. ✅ CD vs PCD comparison shows meaningful differences
7. ✅ All plots generate correctly
8. ✅ Summary report contains all experiments

---

## � Next Steps

After completing experiments:

1. **Review Results**: Check all generated plots
2. **Write Report**: Use findings to write analysis
3. **Compare Literature**: How do results compare to papers?
4. **Extend**: Try different architectures or datasets
5. **Share**: Publish code and results

Good luck! �