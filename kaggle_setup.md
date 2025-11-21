# Kaggle Setup Guide for EBM Experiments

Complete guide to running Energy-Based Model experiments on Kaggle with limited GPU resources.

## � Table of Contents
1. [Initial Setup](#initial-setup)
2. [Running Experiments](#running-experiments)
3. [Memory Management](#memory-management)
4. [Expected Training Times](#expected-training-times)
5. [Troubleshooting](#troubleshooting)

---

## � Initial Setup

### Step 1: Create New Kaggle Notebook

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click "Code" → "New Notebook"
3. **Enable GPU**: Settings → Accelerator → GPU T4 x2 (or P100)
4. **Enable Internet**: Settings → Internet → On

### Step 2: Install Dependencies

```python
# Cell 1: Install required packages
!pip install -q pyyaml lpips pytorch-fid tensorboard
```

### Step 3: Upload Source Code

**Option A: Upload as Dataset**
1. Create a new dataset on Kaggle
2. Upload all `.py` files from `src/` folder
3. Add dataset to your notebook

**Option B: Clone from GitHub**
```python
# Cell 2: Clone repository (if available)
!git clone https://github.com/yourusername/ebm-cd-study.git
import sys
sys.path.append('/kaggle/working/ebm-cd-study/src')
```

**Option C: Copy-Paste Code**
```python
# Cell 2: Create directory structure
!mkdir -p src configs results/checkpoints results/logs plots

# Then copy-paste each .py file into cells using %%writefile
```

### Step 4: Setup Directory Structure

```python
# Cell 3: Create directories
import os

os.makedirs('data', exist_ok=True)
os.makedirs('results/checkpoints', exist_ok=True)
os.makedirs('results/logs', exist_ok=True)
os.makedirs('results/samples', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('configs', exist_ok=True)

print("✓ Directory structure created")
```

---

## � Running Experiments

### Quick Test (5 minutes)

```python
# Cell 4: Quick test to verify everything works
import sys
sys.path.append('/kaggle/working/src')

from train_rbm import train_rbm

test_config = {
    'exp_name': 'quick_test',
    'dataset': 'mnist',
    'batch_size': 128,
    'binarize': True,
    'n_hidden': 128,  # Small for testing
    'learning_rate': 0.01,
    'momentum': 0.5,
    'final_momentum': 0.9,
    'momentum_epoch': 2,
    'weight_decay': 0.0001,
    'cd_k': 1,
    'use_pcd': False,
    'epochs': 2,  # Just 2 epochs
    'seed': 42,
    'save_every': 1,
    'sample_every': 1,
    'sample_steps': 100,
    'output_dir': '/kaggle/working/results',
    'data_dir': '/kaggle/working/data'
}

train_rbm(test_config)
print("✓ Quick test completed!")
```

### Experiment 1: RBM with CD-1 (2-3 hours)

```python
# Cell 5: Write config file
%%writefile configs/rbm_mnist_cd1.yaml
exp_name: rbm_mnist_cd1
dataset: mnist
batch_size: 128
binarize: true
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
output_dir: /kaggle/working/results
data_dir: /kaggle/working/data
```

```python
# Cell 6: Run training
import sys
sys.path.append('/kaggle/working/src')

from train_rbm import train_rbm
from utils import load_config

config = load_config('configs/rbm_mnist_cd1.yaml')
train_rbm(config)
```

### Experiment 2: RBM with CD-5 (3-4 hours)

```python
# Cell 7: CD-5 config
%%writefile configs/rbm_mnist_cd5.yaml
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
save_every: 5
sample_every: 5
sample_steps: 500
output_dir: /kaggle/working/results
data_dir: /kaggle/working/data
```

```python
# Cell 8: Run CD-5
config = load_config('configs/rbm_mnist_cd5.yaml')
train_rbm(config)
```

### Experiment 3: RBM with PCD-5 (4-5 hours)

```python
# Cell 9: PCD-5 config
%%writefile configs/rbm_mnist_pcd5.yaml
exp_name: rbm_mnist_pcd5
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
use_pcd: true
pcd_buffer_size: 5000
epochs: 30
seed: 42
lr_schedule:
  15: 0.005
  25: 0.001
save_every: 5
sample_every: 5
sample_steps: 500
output_dir: /kaggle/working/results
data_dir: /kaggle/working/data
```

```python
# Cell 10: Run PCD-5
config = load_config('configs/rbm_mnist_pcd5.yaml')
train_rbm(config)
```

---

## � Memory Management

### Monitor GPU Memory

```python
# Cell: Check GPU memory usage
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print("No GPU available")

print_gpu_memory()
```

### Clear Memory Between Experiments

```python
# Cell: Clear memory before starting new experiment
import torch
import gc

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Garbage collection
gc.collect()

print("✓ Memory cleared")
print_gpu_memory()
```

### Reduce Batch Size if OOM

If you encounter "CUDA out of memory" errors:

```python
# Modify config before running
config['batch_size'] = 64  # Reduced from 128
config['pcd_buffer_size'] = 3000  # Reduced from 5000
```

---

## ⏱️ Expected Training Times

| Experiment | Config | CD Steps | GPU Memory | Training Time | Kaggle-Ready |
|------------|--------|----------|------------|---------------|--------------|
| RBM CD-1 | 256 hidden | 1 | ~2GB | 2-3 hours | ✅ Yes |
| RBM CD-5 | 256 hidden | 5 | ~2GB | 3-4 hours | ✅ Yes |
| RBM CD-10 | 256 hidden | 10 | ~2-3GB | 4-5 hours | ✅ Yes |
| RBM CD-20 | 256 hidden | 20 | ~3GB | 5-6 hours | ⚠️ Tight |
| RBM PCD-1 | 256 hidden + buffer | 1 | ~3GB | 3-4 hours | ✅ Yes |
| RBM PCD-5 | 256 hidden + buffer | 5 | ~3-4GB | 4-5 hours | ✅ Yes |
| RBM PCD-10 | 256 hidden + buffer | 10 | ~4GB | 5-7 hours | ⚠️ Tight |

**Kaggle Limits:**
- GPU T4: ~15GB memory, 9-12 hour runtime
- GPU P100: ~16GB memory, 9-12 hour runtime

---

## � Troubleshooting

### Problem 1: CUDA Out of Memory

**Solution:**
```python
# Reduce batch size
config['batch_size'] = 64  # or even 32

# Reduce buffer size for PCD
config['pcd_buffer_size'] = 3000

# Reduce hidden units
config['n_hidden'] = 128
```

### Problem 2: Training Too Slow

**Solution:**
```python
# Reduce epochs
config['epochs'] = 20

# Reduce CD steps
config['cd_k'] = 5  # instead of 10 or 20

# Reduce sampling steps
config['sample_steps'] = 300
```

### Problem 3: Notebook Timeout

**Strategy: Run Multiple Short Sessions**

```python
# Save checkpoint every few epochs
config['save_every'] = 3

# Resume from checkpoint
from utils import load_checkpoint
checkpoint = load_checkpoint(
    '/kaggle/working/results/.../checkpoints/rbm_epoch_15.pt',
    model=rbm
)
start_epoch = checkpoint['epoch'] + 1
```

### Problem 4: Data Download Issues

**Solution:**
```python
# Download datasets manually first
from torchvision import datasets

# This will cache the data
datasets.MNIST(root='/kaggle/working/data', train=True, download=True)
datasets.MNIST(root='/kaggle/working/data', train=False, download=True)
```

### Problem 5: Import Errors

**Solution:**
```python
# Add paths explicitly
import sys
sys.path.insert(0, '/kaggle/working/src')
sys.path.insert(0, '/kaggle/working')

# Or use absolute imports
from src.train_rbm import train_rbm
```

---

## � Visualizing Results

### View Generated Samples

```python
# Cell: Display samples
from IPython.display import Image, display
import os

# Find latest samples
samples_dir = '/kaggle/working/results/rbm_mnist_cd5_*/samples'
latest_sample = sorted(os.listdir(samples_dir))[-1]

display(Image(os.path.join(samples_dir, latest_sample)))
```

### Plot Training Curves

```python
# Cell: Plot metrics
import json
import matplotlib.pyplot as plt

# Load metrics
with open('/kaggle/working/results/.../logs/metrics.json', 'r') as f:
    metrics = json.load(f)

# Plot
epochs = [int(k) for k in metrics.keys()]
recon_errors = [metrics[str(e)]['reconstruction_error'] for e in epochs]

plt.figure(figsize=(10, 5))
plt.plot(epochs, recon_errors)
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Error')
plt.title('Training Progress')
plt.grid(True)
plt.show()
```

---

## � Tips for Success

1. **Start Small**: Always run the quick test first (2-3 minutes)
2. **Monitor Memory**: Check GPU memory usage regularly
3. **Save Often**: Set `save_every: 5` to avoid losing progress
4. **Use Persistent Output**: Save to `/kaggle/working/` not `/tmp/`
5. **Download Results**: Download checkpoints and samples before session ends
6. **Multiple Sessions**: Break long experiments into multiple sessions
7. **Compare CD-k**: Run CD-1, CD-5, CD-10 to see the tradeoff

---

## � Saving Results

### Download Checkpoints

```python
# Cell: Prepare results for download
import shutil

# Compress results
shutil.make_archive(
    '/kaggle/working/experiment_results',
    'zip',
    '/kaggle/working/results'
)

print("✓ Results compressed: experiment_results.zip")
print("Download from: Files → experiment_results.zip")
```

### Export to Dataset

1. Click "Data" → "New Dataset"
2. Upload `/kaggle/working/results` folder
3. Version your results
4. Use in future notebooks

---

## ✅ Recommended Workflow

**Session 1: Setup & Quick Test (30 min)**
- Setup environment
- Run quick test
- Verify everything works

**Session 2: CD-1 Baseline (3 hours)**
- Run RBM with CD-1
- Analyze results
- Download checkpoint

**Session 3: CD-5 Experiment (4 hours)**
- Run RBM with CD-5
- Compare with CD-1
- Download results

**Session 4: PCD-5 Experiment (5 hours)**
- Run RBM with PCD-5
- Compare CD vs PCD
- Compile final results

**Total: ~12 hours across 4 sessions**

---

## � Next Steps

After completing RBM experiments:
1. Move to Conv-EBM experiments (Phase 3)
2. Run evaluation metrics (FID, IS, LPIPS)
3. Generate comparison plots
4. Write analysis report

Good luck! �