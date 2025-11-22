"""
Evaluation metrics for Energy-Based Models.

Includes FID, Inception Score, LPIPS diversity, AIS log-likelihood,
and MCMC diagnostics (autocorrelation, ESS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from typing import Tuple, Optional, List
import warnings

try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    warnings.warn("pytorch-fid not installed. FID computation will be unavailable.")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    warnings.warn("lpips not installed. LPIPS diversity will be unavailable.")


# ============================================================================
# Fréchet Inception Distance (FID)
# ============================================================================

class FIDCalculator:
    """Calculate Fréchet Inception Distance between two sets of images."""
    
    def __init__(self, device: torch.device, dims: int = 2048):
        """
        Initialize FID calculator.
        
        Args:
            device: Device to run calculations on
            dims: Dimensionality of Inception features
        """
        if not HAS_FID:
            raise ImportError("pytorch-fid is required for FID calculation")
        
        self.device = device
        self.dims = dims
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx]).to(device)
        self.inception.eval()
    
    def compute_statistics(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance of Inception features.
        
        Args:
            images: Batch of images [N, C, H, W] in range [-1, 1]
            
        Returns:
            Tuple of (mean, covariance)
        """
        with torch.no_grad():
            features = self.inception(images)[0]
            
            if features.size(2) != 1 or features.size(3) != 1:
                features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
            
            features = features.squeeze(3).squeeze(2).cpu().numpy()
        
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def calculate_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Calculate FID between two distributions.
        
        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Small value for numerical stability
            
        Returns:
            FID score
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# ============================================================================
# Inception Score (FIXED VERSION)
# ============================================================================

class InceptionScoreCalculator:
    """Calculate Inception Score for generated images."""
    
    def __init__(self, device: torch.device, splits: int = 10):
        """
        Initialize IS calculator.
        
        Args:
            device: Device to run calculations on
            splits: Number of splits for computing std
        """
        if not HAS_FID:
            raise ImportError("pytorch-fid is required for IS calculation")
        
        self.device = device
        self.splits = splits

        # Use 2048-dim pool features (supported by your InceptionV3)
        dim = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dim]
        self.inception = InceptionV3([block_idx]).to(device)
        self.inception.eval()

        # Add a small linear head for 1000-class logits
        self.fc = nn.Linear(dim, 1000).to(device)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def calculate_inception_score(self, images: torch.Tensor, batch_size: int = 32):
        """
        Calculate Inception Score.
        
        Args:
            images: Generated images [N, C, H, W]
            batch_size: Batch size for processing
            
        Returns:
            Tuple (mean IS, std IS)
        """
        n_images = images.size(0)
        preds = []

        for i in range(0, n_images, batch_size):
            batch = images[i:i+batch_size].to(self.device)

            with torch.no_grad():
                # Extract 2048-dim features
                feats = self.inception(batch)[0]
                feats = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze()

                # Convert to logits
                logits = self.fc(feats)

                # Probabilities
                p = F.softmax(logits, dim=1)

            preds.append(p.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Compute IS for each split
        split_scores = []
        split_size = n_images // self.splits

        for k in range(self.splits):
            part = preds[k * split_size:(k + 1) * split_size, :]
            py = np.mean(part, axis=0)

            kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
            split_scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

        return float(np.mean(split_scores)), float(np.std(split_scores))


# ============================================================================
# LPIPS Diversity
# ============================================================================

class LPIPSDiversity:
    """Calculate diversity using LPIPS perceptual distance."""
    
    def __init__(self, device: torch.device, net: str = 'alex'):
        """
        Initialize LPIPS calculator.
        
        Args:
            device: Device to run calculations on
            net: Network to use ('alex', 'vgg', 'squeeze')
        """
        if not HAS_LPIPS:
            raise ImportError("lpips is required for diversity calculation")
        
        self.device = device
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.lpips_fn.eval()
    
    def compute_diversity(
        self,
        images: torch.Tensor,
        num_pairs: int = 1000
    ) -> float:
        """
        Compute average pairwise LPIPS distance.
        
        Args:
            images: Generated images [N, C, H, W]
            num_pairs: Number of random pairs to sample
            
        Returns:
            Average LPIPS distance
        """
        n = images.size(0)
        num_pairs = min(num_pairs, n * (n - 1) // 2)
        
        distances = []
        for _ in range(num_pairs):
            i, j = np.random.choice(n, size=2, replace=False)
            img1 = images[i:i+1].to(self.device)
            img2 = images[j:j+1].to(self.device)
            
            with torch.no_grad():
                dist = self.lpips_fn(img1, img2)
            distances.append(dist.item())
        
        return np.mean(distances)


# ============================================================================
# Annealed Importance Sampling (AIS) for Log-Likelihood
# ============================================================================

def ais_log_likelihood(
    model,
    test_data: torch.Tensor,
    num_chains: int = 100,
    num_steps: int = 1000,
    device: torch.device = torch.device('cpu')
) -> float:
    """
    Estimate log-likelihood using Annealed Importance Sampling.
    
    Args:
        model: RBM model with free_energy method
        test_data: Test data [N, D]
        num_chains: Number of AIS chains
        num_steps: Number of annealing steps
        device: Device to run on
        
    Returns:
        Average log-likelihood estimate
    """
    n_samples = test_data.size(0)
    vis_dim = test_data.size(1)
    
    # Initialize from uniform distribution
    v = torch.rand(num_chains, vis_dim).to(device)
    v = (v > 0.5).float()
    
    # Annealing schedule
    betas = torch.linspace(0, 1, num_steps).to(device)
    
    # Log importance weights
    log_weights = torch.zeros(num_chains).to(device)
    
    for i in range(len(betas) - 1):
        beta_curr = betas[i]
        beta_next = betas[i + 1]
        
        # Update weights
        with torch.no_grad():
            energy_curr = model.free_energy(v)
            log_weights += -beta_curr * energy_curr
        
        # Gibbs transition
        v = model.gibbs_step(v)
        
        with torch.no_grad():
            energy_next = model.free_energy(v)
            log_weights += beta_next * energy_next
    
    # Compute log partition function estimate
    log_z_base = vis_dim * np.log(2)  # Base partition function (uniform)
    log_z_est = torch.logsumexp(log_weights, dim=0) - np.log(num_chains) + log_z_base
    
    # Compute log-likelihood for test data
    with torch.no_grad():
        test_energies = model.free_energy(test_data.to(device))
    
    log_likelihoods = -test_energies - log_z_est
    
    return log_likelihoods.mean().item()


# ============================================================================
# MCMC Diagnostics
# ============================================================================

def compute_autocorrelation(
    samples: np.ndarray,
    max_lag: Optional[int] = None
) -> np.ndarray:
    """
    Compute autocorrelation function.
    
    Args:
        samples: Time series samples [T, ...]
        max_lag: Maximum lag to compute (default: T//2)
        
    Returns:
        Autocorrelation values [max_lag]
    """
    samples = samples.reshape(samples.shape[0], -1)
    
    if max_lag is None:
        max_lag = len(samples) // 2
    
    # Center the data
    samples_centered = samples - samples.mean(axis=0, keepdims=True)
    
    # Compute autocorrelation
    autocorr = np.zeros(max_lag)
    variance = np.sum(samples_centered[0] ** 2)
    
    for lag in range(max_lag):
        if lag < len(samples):
            autocorr[lag] = np.sum(
                samples_centered[:-lag or None] * samples_centered[lag:]
            ) / variance
        else:
            autocorr[lag] = 0
    
    return autocorr


def effective_sample_size(samples: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Compute effective sample size using autocorrelation.
    
    Args:
        samples: MCMC samples [T, ...]
        max_lag: Maximum lag for autocorrelation
        
    Returns:
        Effective sample size
    """
    n = len(samples)
    autocorr = compute_autocorrelation(samples, max_lag)
    
    # Integrated autocorrelation time
    tau_int = 0.5 + np.sum(autocorr[1:])
    
    # Effective sample size
    ess = n / (2 * tau_int)
    
    return max(1.0, ess)


def energy_autocorrelation(
    energies: np.ndarray,
    max_lag: Optional[int] = None
) -> np.ndarray:
    """
    Compute autocorrelation of energy values.
    
    Args:
        energies: Energy time series [T]
        max_lag: Maximum lag
        
    Returns:
        Autocorrelation function
    """
    return compute_autocorrelation(energies.reshape(-1, 1), max_lag)


def mixing_time(autocorr: np.ndarray, threshold: float = 0.1) -> int:
    """
    Estimate mixing time from autocorrelation.
    
    Args:
        autocorr: Autocorrelation function
        threshold: Threshold for considering mixed
        
    Returns:
        Mixing time (number of steps)
    """
    below_threshold = np.where(np.abs(autocorr) < threshold)[0]
    if len(below_threshold) > 0:
        return below_threshold[0]
    return len(autocorr)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    reduction: str = 'mean'
) -> float:
    """
    Compute reconstruction error.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        reduction: How to reduce ('mean' or 'sum')
        
    Returns:
        Reconstruction error
    """
    error = F.mse_loss(reconstructed, original, reduction=reduction)
    return error.item()


def bits_per_dimension(log_likelihood: float, dim: int) -> float:
    """
    Convert log-likelihood to bits per dimension.
    
    Args:
        log_likelihood: Log-likelihood value
        dim: Data dimensionality
        
    Returns:
        Bits per dimension
    """
    return -log_likelihood / (dim * np.log(2))


if __name__ == "__main__":
    print("Testing metrics...")
    
    # Test autocorrelation
    samples = np.random.randn(1000, 10)
    autocorr = compute_autocorrelation(samples, max_lag=100)
    print(f"Autocorrelation shape: {autocorr.shape}")
    print(f"Autocorrelation at lag 0: {autocorr[0]:.3f}")
    
    # Test ESS
    ess = effective_sample_size(samples)
    print(f"Effective sample size: {ess:.2f}")
    
    # Test mixing time
    mix_time = mixing_time(autocorr, threshold=0.1)
    print(f"Mixing time: {mix_time}")
    
    print("\nMetrics test passed!")