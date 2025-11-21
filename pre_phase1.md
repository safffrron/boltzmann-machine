# Energy-Based Models: Contrastive Divergence Analysis

A comprehensive experimental package investigating the role of Contrastive Divergence (CD) steps on sample quality in Energy-Based Models (EBMs).

## Project Structure

```
ebm_cd_study/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── configs/                     # YAML configuration files
│   ├── rbm_mnist_cd1.yaml
│   ├── rbm_mnist_cd5.yaml
│   ├── rbm_mnist_cd20.yaml
│   ├── rbm_mnist_cd100.yaml
│   ├── rbm_mnist_pcd1.yaml
│   ├── rbm_mnist_pcd5.yaml
│   ├── rbm_mnist_pcd20.yaml
│   ├── conv_cifar_cd1.yaml
│   ├── conv_cifar_cd5.yaml
│   ├── conv_cifar_cd20.yaml
│   ├── conv_cifar_pcd1.yaml
│   ├── conv_cifar_pcd20.yaml
│   └── conv_cifar_sgld.yaml
├── src/                         # Source code
│   ├── data.py                  # Dataset loaders
│   ├── utils.py                 # Utility functions
│   ├── metrics.py               # Evaluation metrics
│   ├── plotting.py              # Visualization code
│   ├── rbm.py                   # RBM implementation
│   ├── conv_ebm.py              # Convolutional EBM
│   ├── mcmc.py                  # MCMC sampling methods
│   ├── train_rbm.py             # RBM training script
│   ├── train_conv_ebm.py        # Conv-EBM training script
│   ├── sample.py                # Sampling script
│   └── evaluate.py              # Evaluation script
├── plots/                       # Generated plots
├── results/                     # Experiment results
│   ├── checkpoints/            # Model checkpoints
│   ├── logs/                   # Training logs
│   └── samples/                # Generated samples
└── report/                      # Research report
    ├── introduction.md
    ├── background.md
    ├── methods.md
    ├── experiments.md
    ├── results.md
    ├── analysis.md
    ├── conclusion.md
    └── appendix.md
```

## Installation

### Local Setup

```bash
# Clone or download the project
cd ebm_cd_study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Kaggle Setup

```python
# Upload the entire folder as a dataset or use this in a Kaggle notebook
!pip install -q pyyaml lpips pytorch-fid

# Copy source files
import sys
sys.path.append('/kaggle/input/ebm-cd-study/src')
```

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.65.0
scipy>=1.10.0
pillow>=9.5.0
lpips>=0.1.4
pytorch-fid>=0.3.0
tensorboard>=2.13.0
```

## Quick Start

### 1. Train RBM on MNIST with CD-5

```bash
python src/train_rbm.py --config configs/rbm_mnist_cd5.yaml
```

### 2. Train Conv-EBM on CIFAR-10 with CD-20

```bash
python src/train_conv_ebm.py --config configs/conv_cifar_cd20.yaml
```

### 3. Generate Samples

```bash
python src/sample.py --checkpoint results/checkpoints/rbm_cd5_best.pt \
                     --model_type rbm \
                     --num_samples 100 \
                     --output plots/rbm_cd5_samples.png
```

### 4. Evaluate Model

```bash
python src/evaluate.py --checkpoint results/checkpoints/conv_ebm_cd20.pt \
                       --model_type conv_ebm \
                       --config configs/conv_cifar_cd20.yaml \
                       --metrics fid is lpips
```

## Experiment Sets

### Set A: RBM on MNIST
Compare CD-k vs PCD-k with k ∈ {1, 5, 20, 100}

```bash
# Run all RBM experiments
for config in configs/rbm_*.yaml; do
    python src/train_rbm.py --config $config
done
```

### Set B: Conv-EBM on CIFAR-10
Compare CD-k and PCD-k for convolutional models

```bash
# Run all Conv-EBM experiments
for config in configs/conv_*.yaml; do
    python src/train_conv_ebm.py --config $config
done
```

### Set C: Ablation Studies
Investigate hyperparameter sensitivity (step size, buffer size, noise)

```bash
# Modify configs and run ablations
python src/train_conv_ebm.py --config configs/conv_cifar_cd20.yaml \
                             --override langevin_step_size=0.01
```

## Key Features

- **Modular Design**: Clean separation of models, training, and evaluation
- **Reproducible**: Seed control and deterministic operations
- **Configurable**: All hyperparameters in YAML files
- **Comprehensive Metrics**: FID, IS, LPIPS, AIS, autocorrelation, ESS
- **Visualization**: Automatic plot generation for all experiments
- **Memory Efficient**: Optimized for Kaggle/Colab (single GPU, 16GB RAM)

## Memory Considerations for Kaggle

Our experiments are designed to run within Kaggle's constraints:
- **RBM experiments**: ~2-4GB GPU memory, 2-4 hours training
- **Conv-EBM experiments**: ~8-12GB GPU memory, 6-12 hours training
- Batch sizes adjusted for P100/T4 GPUs
- Gradient accumulation for large models

## Results Structure

After running experiments, results are organized as:

```
results/
├── checkpoints/
│   ├── rbm_cd1_best.pt
│   ├── rbm_cd5_best.pt
│   └── conv_ebm_cd20_best.pt
├── logs/
│   ├── rbm_cd1_log.json
│   └── tensorboard/
└── samples/
    ├── rbm_cd1_epoch_50.png
    └── conv_ebm_cd20_samples.png
```

## Visualization

Generate all plots:

```bash
python src/plotting.py --results_dir results/ --output_dir plots/
```

This creates:
- Training curves (loss, energy)
- FID vs CD-k comparison
- Sample grids
- MCMC diagnostics (autocorrelation, ESS)
- Ablation study plots

## Citation

If you use this code, please cite:

```bibtex
@software{ebm_cd_analysis,
  title={Analysis of Energy-Based Models: Contrastive Divergence Steps},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ebm-cd-study}
}
```

## License

MIT License - See LICENSE file for details

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config files
- Reduce `buffer_size` for PCD
- Use gradient accumulation: set `accumulation_steps: 2`

### Slow Training
- Reduce `num_epochs` for quick tests
- Use CD-1 or CD-5 instead of CD-20 initially
- Reduce `langevin_steps` for Conv-EBM

### NaN Losses
- Reduce `learning_rate`
- Reduce `langevin_step_size`
- Increase `langevin_noise`
- Check data normalization

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].

## Acknowledgments

This project implements techniques from:
- Hinton (2002) - Training Products of Experts
- Tieleman (2008) - Training RBMs using Persistent CD
- Du & Mordatch (2019) - Implicit Generation and Modeling with Energy
- Nijkamp et al. (2019) - Learning Non-Convergent Short-Run MCMC