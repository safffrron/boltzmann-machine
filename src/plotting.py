"""
Visualization utilities for EBM experiments.

Basic plotting functions needed for training and evaluation.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import torchvision.utils as vutils

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None
):
    """
    Save a grid of images.
    
    Args:
        images: Images tensor [N, C, H, W]
        save_path: Path to save grid
        nrow: Number of images per row
        normalize: Whether to normalize images
        value_range: Range for normalization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Make grid
    grid = vutils.make_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=2,
        pad_value=1
    )
    
    # Convert to numpy and save
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_training_curves(
    metrics: dict,
    save_path: str,
    title: str = "Training Curves"
):
    """
    Plot training curves.
    
    Args:
        metrics: Dictionary of metric_name -> {step: value}
        save_path: Path to save plot
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        steps = sorted(values.keys())
        vals = [values[s] for s in steps]
        
        axes[idx].plot(steps, vals, linewidth=2)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric_name.replace('_', ' ').title())
        axes[idx].set_title(metric_name.replace('_', ' ').title())
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_energy_histogram(
    energies_real: np.ndarray,
    energies_fake: np.ndarray,
    save_path: str,
    title: str = "Energy Distribution"
):
    """
    Plot histogram of energies.
    
    Args:
        energies_real: Energies of real data
        energies_fake: Energies of generated samples
        save_path: Path to save plot
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(energies_real, bins=50, alpha=0.6, label='Real Data', density=True)
    plt.hist(energies_fake, bins=50, alpha=0.6, label='Generated', density=True)
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_autocorrelation(
    autocorr: np.ndarray,
    save_path: str,
    title: str = "Energy Autocorrelation",
    max_lag: Optional[int] = None
):
    """
    Plot autocorrelation function.
    
    Args:
        autocorr: Autocorrelation values
        save_path: Path to save plot
        title: Plot title
        max_lag: Maximum lag to plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if max_lag is not None:
        autocorr = autocorr[:max_lag]
    
    plt.figure(figsize=(10, 6))
    plt.plot(autocorr, linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.3, label='Threshold (0.1)')
    plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_weight_filters(
    weights: torch.Tensor,
    save_path: str,
    num_filters: int = 64,
    img_shape: Tuple[int, int] = (28, 28),
    title: str = "RBM Weight Filters"
):
    """
    Visualize RBM weight filters.
    
    Args:
        weights: Weight matrix [n_visible, n_hidden]
        save_path: Path to save plot
        num_filters: Number of filters to show
        img_shape: Shape to reshape filters to
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    weights = weights.cpu().numpy()
    n_hidden = min(num_filters, weights.shape[1])
    
    # Reshape filters
    filters = weights[:, :n_hidden].T.reshape(n_hidden, *img_shape)
    
    # Normalize each filter
    filters_norm = []
    for f in filters:
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f = (f - f_min) / (f_max - f_min)
        filters_norm.append(f)
    filters = np.array(filters_norm)
    
    # Plot grid
    nrow = int(np.ceil(np.sqrt(n_hidden)))
    ncol = int(np.ceil(n_hidden / nrow))
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
    axes = axes.flatten()
    
    for idx in range(n_hidden):
        axes[idx].imshow(filters[idx], cmap='gray')
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(n_hidden, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: str,
    num_samples: int = 8
):
    """
    Plot original vs reconstructed images.
    
    Args:
        original: Original images [N, C, H, W]
        reconstructed: Reconstructed images [N, C, H, W]
        save_path: Path to save plot
        num_samples: Number of samples to show
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    num_samples = min(num_samples, original.size(0))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*2, 4))
    
    for idx in range(num_samples):
        # Original
        img_orig = original[idx].cpu().numpy()
        if img_orig.shape[0] == 1:
            img_orig = img_orig.squeeze()
            axes[0, idx].imshow(img_orig, cmap='gray')
        else:
            img_orig = img_orig.transpose(1, 2, 0)
            axes[0, idx].imshow(img_orig)
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_title('Original', fontsize=12)
        
        # Reconstructed
        img_recon = reconstructed[idx].cpu().numpy()
        if img_recon.shape[0] == 1:
            img_recon = img_recon.squeeze()
            axes[1, idx].imshow(img_recon, cmap='gray')
        else:
            img_recon = img_recon.transpose(1, 2, 0)
            axes[1, idx].imshow(img_recon)
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_title('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_sampling_trajectory(
    samples: List[torch.Tensor],
    save_path: str,
    num_chains: int = 5,
    title: str = "Sampling Trajectory"
):
    """
    Plot MCMC sampling trajectory.
    
    Args:
        samples: List of sample tensors at different steps
        save_path: Path to save plot
        num_chains: Number of chains to visualize
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    num_steps = len(samples)
    num_chains = min(num_chains, samples[0].size(0))
    
    fig, axes = plt.subplots(num_chains, num_steps, 
                            figsize=(num_steps*1.5, num_chains*1.5))
    
    if num_chains == 1:
        axes = axes.reshape(1, -1)
    
    for chain_idx in range(num_chains):
        for step_idx, sample_batch in enumerate(samples):
            img = sample_batch[chain_idx].cpu().numpy()
            
            if img.ndim == 3:  # [C, H, W]
                if img.shape[0] == 1:
                    img = img.squeeze()
                    axes[chain_idx, step_idx].imshow(img, cmap='gray')
                else:
                    img = img.transpose(1, 2, 0)
                    axes[chain_idx, step_idx].imshow(img)
            else:  # Flattened
                size = int(np.sqrt(img.shape[0]))
                img = img.reshape(size, size)
                axes[chain_idx, step_idx].imshow(img, cmap='gray')
            
            axes[chain_idx, step_idx].axis('off')
            
            if chain_idx == 0:
                axes[chain_idx, step_idx].set_title(f'Step {step_idx}', 
                                                     fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_cd_comparison(
    results_dict: dict,
    save_path: str,
    metric_name: str = 'fid',
    title: str = 'CD-k Comparison'
):
    """
    Compare metrics across different CD-k values.
    
    Args:
        results_dict: Dict mapping CD-k to metric value
        save_path: Path to save plot
        metric_name: Name of metric being compared
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cd_values = sorted(results_dict.keys())
    metric_values = [results_dict[k] for k in cd_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(cd_values, metric_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('CD-k Steps', fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_compute_vs_quality(
    cd_steps: list,
    training_times: list,
    quality_scores: list,
    save_path: str,
    quality_metric: str = 'FID'
):
    """
    Plot compute cost vs quality tradeoff.
    
    Args:
        cd_steps: List of CD-k values
        training_times: List of training times (hours)
        quality_scores: List of quality scores
        save_path: Path to save plot
        quality_metric: Name of quality metric
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training time vs CD-k
    ax1.plot(cd_steps, training_times, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('CD-k Steps', fontsize=12)
    ax1.set_ylabel('Training Time (hours)', fontsize=12)
    ax1.set_title('Training Time vs CD-k', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Quality vs CD-k
    better_lower = (quality_metric.lower() == 'fid')
    color = 'green' if not better_lower else 'red'
    ax2.plot(cd_steps, quality_scores, 'o-', linewidth=2, markersize=8, color=color)
    ax2.set_xlabel('CD-k Steps', fontsize=12)
    ax2.set_ylabel(f'{quality_metric} Score', fontsize=12)
    ax2.set_title(f'{quality_metric} vs CD-k', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Testing plotting functions...")
    
    # Test image grid
    images = torch.randn(16, 1, 28, 28)
    save_image_grid(images, './test_plots/grid.png')
    print("✓ Image grid saved")
    
    # Test training curves
    metrics = {
        'loss': {i: np.random.rand() for i in range(20)},
        'energy': {i: np.random.rand() * 10 for i in range(20)}
    }
    plot_training_curves(metrics, './test_plots/curves.png')
    print("✓ Training curves saved")
    
    # Test autocorrelation
    autocorr = np.exp(-np.arange(100) / 20)
    plot_autocorrelation(autocorr, './test_plots/autocorr.png')
    print("✓ Autocorrelation plot saved")
    
    # Test CD comparison
    cd_results = {1: 45.2, 5: 38.5, 10: 35.1, 20: 33.8}
    plot_cd_comparison(cd_results, './test_plots/cd_comparison.png')
    print("✓ CD comparison saved")
    
    print("\n✅ All plotting tests passed!")