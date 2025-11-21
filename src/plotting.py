"""
Complete visualization suite for EBM experiments.

Generates all plots for analysis and report.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import torchvision.utils as vutils
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


# ============================================================================
# Basic Plotting
# ============================================================================

def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None
):
    """Save a grid of images."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    grid = vutils.make_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=2,
        pad_value=1
    )
    
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    plt.figure(figsize=(12, 12))
    if grid_np.shape[2] == 1:
        plt.imshow(grid_np.squeeze(), cmap='gray')
    else:
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
    """Plot training curves from metrics dict."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        steps = sorted(values.keys())
        vals = [values[s] for s in steps]
        
        axes[idx].plot(steps, vals, linewidth=2)
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_title(metric_name.replace('_', ' ').title(), fontsize=13)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# Comparison Plots
# ============================================================================

def plot_cd_comparison(
    results_dict: dict,
    save_path: str,
    metric_name: str = 'fid',
    title: str = 'CD-k Comparison',
    ylabel: Optional[str] = None,
    log_scale: bool = False
):
    """
    Compare metrics across different CD-k values.
    
    Args:
        results_dict: Dict mapping CD-k to metric value
        save_path: Path to save plot
        metric_name: Name of metric
        title: Plot title
        ylabel: Y-axis label (auto if None)
        log_scale: Use log scale for y-axis
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cd_values = sorted(results_dict.keys())
    metric_values = [results_dict[k] for k in cd_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(cd_values, metric_values, 'o-', linewidth=2.5, markersize=10, color='steelblue')
    plt.xlabel('CD-k Steps', fontsize=13)
    plt.ylabel(ylabel or metric_name.upper(), fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    # Annotate points
    for cd, val in zip(cd_values, metric_values):
        plt.annotate(f'{val:.2f}', (cd, val), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_cd_vs_pcd_comparison(
    cd_results: dict,
    pcd_results: dict,
    save_path: str,
    metric_name: str = 'FID',
    title: str = 'CD vs PCD Comparison'
):
    """
    Compare CD-k vs PCD-k performance.
    
    Args:
        cd_results: Dict of CD-k -> metric value
        pcd_results: Dict of PCD-k -> metric value
        save_path: Path to save plot
        metric_name: Name of metric
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get common k values
    k_values = sorted(set(cd_results.keys()) & set(pcd_results.keys()))
    
    cd_vals = [cd_results[k] for k in k_values]
    pcd_vals = [pcd_results[k] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, cd_vals, 'o-', linewidth=2.5, markersize=10, 
             label='CD-k', color='steelblue')
    plt.plot(k_values, pcd_vals, 's-', linewidth=2.5, markersize=10, 
             label='PCD-k', color='coral')
    
    plt.xlabel('Number of Steps (k)', fontsize=13)
    plt.ylabel(metric_name, fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_compute_vs_quality(
    cd_steps: list,
    training_times: list,
    quality_scores: list,
    save_path: str,
    quality_metric: str = 'FID',
    better_lower: bool = True
):
    """
    Plot compute cost vs quality tradeoff.
    
    Args:
        cd_steps: List of CD-k values
        training_times: List of training times (hours)
        quality_scores: List of quality scores
        save_path: Path to save plot
        quality_metric: Name of quality metric
        better_lower: Whether lower is better for quality metric
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training time vs CD-k
    ax1.plot(cd_steps, training_times, 'o-', linewidth=2.5, markersize=10, color='steelblue')
    ax1.set_xlabel('CD-k Steps', fontsize=12)
    ax1.set_ylabel('Training Time (hours)', fontsize=12)
    ax1.set_title('Training Time vs CD-k', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Quality vs CD-k
    color = 'green' if better_lower else 'red'
    ax2.plot(cd_steps, quality_scores, 'o-', linewidth=2.5, markersize=10, color=color)
    ax2.set_xlabel('CD-k Steps', fontsize=12)
    ax2.set_ylabel(f'{quality_metric} Score', fontsize=12)
    ax2.set_title(f'{quality_metric} vs CD-k', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Efficiency: Quality per hour
    efficiency = np.array(quality_scores) / np.array(training_times)
    if better_lower:
        efficiency = 1 / efficiency  # Invert for "lower is better" metrics
    
    ax3.plot(cd_steps, efficiency, 'o-', linewidth=2.5, markersize=10, color='purple')
    ax3.set_xlabel('CD-k Steps', fontsize=12)
    ax3.set_ylabel('Quality per Hour', fontsize=12)
    ax3.set_title('Training Efficiency', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# Energy and MCMC Diagnostics
# ============================================================================

def plot_energy_histogram(
    energies_real: np.ndarray,
    energies_fake: np.ndarray,
    save_path: str,
    title: str = "Energy Distribution"
):
    """Plot histogram of energies for real and generated samples."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(energies_real, bins=50, alpha=0.6, label='Real Data', 
             density=True, color='steelblue')
    plt.hist(energies_fake, bins=50, alpha=0.6, label='Generated', 
             density=True, color='coral')
    
    # Add mean lines
    plt.axvline(energies_real.mean(), color='steelblue', 
                linestyle='--', linewidth=2, alpha=0.8)
    plt.axvline(energies_fake.mean(), color='coral', 
                linestyle='--', linewidth=2, alpha=0.8)
    
    plt.xlabel('Energy', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Real: μ={energies_real.mean():.2f}, σ={energies_real.std():.2f}\n"
    stats_text += f"Gen: μ={energies_fake.mean():.2f}, σ={energies_fake.std():.2f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_autocorrelation(
    autocorr: np.ndarray,
    save_path: str,
    title: str = "Energy Autocorrelation",
    max_lag: Optional[int] = None
):
    """Plot autocorrelation function."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if max_lag is not None:
        autocorr = autocorr[:max_lag]
    
    plt.figure(figsize=(10, 6))
    plt.plot(autocorr, linewidth=2.5, color='steelblue')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, 
                linewidth=2, label='Threshold (±0.1)')
    plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.xlabel('Lag', fontsize=13)
    plt.ylabel('Autocorrelation', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.3, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_multiple_autocorrelations(
    autocorr_dict: dict,
    save_path: str,
    title: str = "Autocorrelation Comparison",
    max_lag: Optional[int] = None
):
    """
    Plot multiple autocorrelation functions for comparison.
    
    Args:
        autocorr_dict: Dict mapping label to autocorrelation array
        save_path: Path to save plot
        title: Plot title
        max_lag: Maximum lag to plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(autocorr_dict)))
    
    for (label, autocorr), color in zip(autocorr_dict.items(), colors):
        if max_lag is not None:
            autocorr = autocorr[:max_lag]
        plt.plot(autocorr, linewidth=2.5, label=label, color=color)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    
    plt.xlabel('Lag', fontsize=13)
    plt.ylabel('Autocorrelation', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.3, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# RBM-specific Plots
# ============================================================================

def plot_weight_filters(
    weights: torch.Tensor,
    save_path: str,
    num_filters: int = 64,
    img_shape: Tuple[int, int] = (28, 28),
    title: str = "RBM Weight Filters"
):
    """Visualize RBM weight filters."""
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
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*1.5, nrow*1.5))
    axes = axes.flatten()
    
    for idx in range(n_hidden):
        axes[idx].imshow(filters[idx], cmap='gray')
        axes[idx].axis('off')
    
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
    """Plot original vs reconstructed images."""
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
            axes[0, idx].set_title('Original', fontsize=12, fontweight='bold')
        
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
            axes[1, idx].set_title('Reconstructed', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# Sampling Trajectories
# ============================================================================

def plot_sampling_trajectory(
    samples: List[torch.Tensor],
    save_path: str,
    num_chains: int = 5,
    title: str = "Sampling Trajectory"
):
    """Plot MCMC sampling trajectory."""
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
            
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.squeeze()
                    axes[chain_idx, step_idx].imshow(img, cmap='gray')
                else:
                    img = img.transpose(1, 2, 0)
                    axes[chain_idx, step_idx].imshow(np.clip(img, 0, 1))
            else:
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


def plot_energy_trajectory(
    energy_history: list,
    save_path: str,
    title: str = "Energy During Sampling"
):
    """Plot energy evolution during sampling."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(energy_history, linewidth=2, color='steelblue')
    plt.xlabel('Sampling Step', fontsize=13)
    plt.ylabel('Energy', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    window = min(20, len(energy_history) // 10)
    if window > 1:
        moving_avg = np.convolve(energy_history, 
                                np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(energy_history)), moving_avg, 
                linewidth=2.5, color='coral', alpha=0.8, 
                label=f'Moving Avg (window={window})')
        plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# Ablation Study Plots
# ============================================================================

def plot_ablation_study(
    param_values: list,
    metric_values: list,
    save_path: str,
    param_name: str = 'Parameter',
    metric_name: str = 'Metric',
    title: str = 'Ablation Study'
):
    """
    Plot ablation study results.
    
    Args:
        param_values: List of parameter values tested
        metric_values: List of corresponding metric values
        save_path: Path to save plot
        param_name: Name of parameter being varied
        metric_name: Name of metric being measured
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, metric_values, 'o-', linewidth=2.5, 
             markersize=10, color='steelblue')
    
    plt.xlabel(param_name, fontsize=13)
    plt.ylabel(metric_name, fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate best value
    best_idx = np.argmin(metric_values)
    plt.plot(param_values[best_idx], metric_values[best_idx], 
             'r*', markersize=20, label='Best')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# Summary Dashboard
# ============================================================================

def create_experiment_dashboard(
    experiment_results: dict,
    save_path: str,
    title: str = "Experiment Dashboard"
):
    """
    Create comprehensive dashboard for experiment results.
    
    Args:
        experiment_results: Dict containing all experiment metrics
        save_path: Path to save dashboard
        title: Dashboard title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Placeholder plots - customize based on available data
    # This is a template that can be filled with actual experiment data
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Testing comprehensive plotting functions...")
    
    os.makedirs('./test_plots', exist_ok=True)
    
    # Test CD comparison
    cd_results = {1: 45.2, 5: 38.5, 10: 35.1, 20: 33.8}
    plot_cd_comparison(cd_results, './test_plots/cd_comparison.png', 
                       metric_name='fid', title='FID vs CD-k')
    print("✓ CD comparison plot saved")
    
    # Test CD vs PCD
    pcd_results = {1: 42.0, 5: 36.2, 10: 33.5}
    plot_cd_vs_pcd_comparison(cd_results, pcd_results, 
                              './test_plots/cd_vs_pcd.png')
    print("✓ CD vs PCD comparison saved")
    
    # Test compute vs quality
    plot_compute_vs_quality(
        [1, 5, 10, 20],
        [2.0, 3.0, 4.0, 5.5],
        [45.2, 38.5, 35.1, 33.8],
        './test_plots/compute_vs_quality.png'
    )
    print("✓ Compute vs quality plot saved")
    
    # Test multiple autocorrelations
    autocorr_dict = {
        'CD-1': np.exp(-np.arange(50) / 5),
        'CD-5': np.exp(-np.arange(50) / 10),
        'CD-10': np.exp(-np.arange(50) / 15)
    }
    plot_multiple_autocorrelations(autocorr_dict, 
                                   './test_plots/multiple_autocorr.png')
    print("✓ Multiple autocorrelation plot saved")
    
    print("\n✅ All plotting tests passed!")