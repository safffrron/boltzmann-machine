"""
MCMC sampling methods for Energy-Based Models.

Includes Gibbs sampling for RBMs and Langevin dynamics for Conv-EBMs.
Optimized for Kaggle with memory-efficient implementations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


# ============================================================================
# Gibbs Sampling for RBM
# ============================================================================

class GibbsSampler:
    """Gibbs sampler for RBM models."""
    
    def __init__(self, rbm, device: torch.device = torch.device('cpu')):
        """
        Initialize Gibbs sampler.
        
        Args:
            rbm: RBM model
            device: Device to run on
        """
        self.rbm = rbm
        self.device = device
    
    def sample(
        self,
        num_samples: int,
        num_steps: int = 1000,
        init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples using Gibbs sampling.
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of Gibbs steps
            init: Initial state (random if None)
            
        Returns:
            Samples [num_samples, n_visible]
        """
        if init is None:
            # Random initialization
            v = torch.bernoulli(
                torch.ones(num_samples, self.rbm.n_visible) * 0.5
            ).to(self.device)
        else:
            v = init.to(self.device)
        
        # Run Gibbs sampling
        for _ in range(num_steps):
            v = self.rbm.gibbs_step(v)
        
        return v
    
    def sample_chain(
        self,
        num_samples: int,
        num_steps: int = 1000,
        save_every: int = 10
    ) -> List[torch.Tensor]:
        """
        Generate samples and save intermediate states.
        
        Args:
            num_samples: Number of samples
            num_steps: Total number of steps
            save_every: Save every N steps
            
        Returns:
            List of samples at different steps
        """
        v = torch.bernoulli(
            torch.ones(num_samples, self.rbm.n_visible) * 0.5
        ).to(self.device)
        
        chain = [v.cpu().clone()]
        
        for step in range(num_steps):
            v = self.rbm.gibbs_step(v)
            if (step + 1) % save_every == 0:
                chain.append(v.cpu().clone())
        
        return chain


# ============================================================================
# Persistent Contrastive Divergence
# ============================================================================

class PersistentChain:
    """
    Persistent chain for PCD training.
    
    Memory-efficient implementation for Kaggle.
    """
    
    def __init__(
        self,
        num_chains: int,
        dim: int,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize persistent chains.
        
        Args:
            num_chains: Number of persistent chains (kept small for memory)
            dim: Dimension of each chain
            device: Device to store chains
        """
        self.num_chains = num_chains
        self.dim = dim
        self.device = device
        
        # Initialize chains randomly
        self.chains = torch.bernoulli(
            torch.ones(num_chains, dim) * 0.5
        ).to(device)
    
    def get_samples(self, batch_size: int) -> torch.Tensor:
        """
        Get samples from persistent chains.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            Samples from chains
        """
        if batch_size <= self.num_chains:
            # Return subset
            indices = torch.randperm(self.num_chains)[:batch_size]
            return self.chains[indices]
        else:
            # Repeat chains if needed
            repeats = (batch_size + self.num_chains - 1) // self.num_chains
            samples = self.chains.repeat(repeats, 1)[:batch_size]
            return samples
    
    def update(self, new_samples: torch.Tensor):
        """
        Update persistent chains with new samples.
        
        Args:
            new_samples: New samples to update chains
        """
        batch_size = new_samples.size(0)
        if batch_size >= self.num_chains:
            # Replace all chains
            self.chains = new_samples[:self.num_chains].detach()
        else:
            # Replace random subset
            indices = torch.randperm(self.num_chains)[:batch_size]
            self.chains[indices] = new_samples.detach()
    
    def reset(self):
        """Reset chains to random state."""
        self.chains = torch.bernoulli(
            torch.ones(self.num_chains, self.dim) * 0.5
        ).to(self.device)


# ============================================================================
# Langevin Dynamics for Conv-EBM
# ============================================================================

class LangevinSampler:
    """
    Langevin dynamics sampler for continuous EBMs.
    
    Uses SGLD (Stochastic Gradient Langevin Dynamics).
    Optimized for Kaggle with gradient clipping and stability measures.
    """
    
    def __init__(
        self,
        step_size: float = 0.01,
        noise_scale: float = 0.005,
        clip_grad: float = 0.01,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize Langevin sampler.
        
        Args:
            step_size: Step size for gradient descent (reduced for stability)
            noise_scale: Scale of injected noise (reduced for Kaggle)
            clip_grad: Gradient clipping value
            device: Device to run on
        """
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.clip_grad = clip_grad
        self.device = device
    
    def sample(
        self,
        energy_fn,
        init_samples: torch.Tensor,
        num_steps: int = 20,  # Reduced for Kaggle
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Sample using Langevin dynamics.
        
        Args:
            energy_fn: Function that computes energy E(x)
            init_samples: Initial samples [batch_size, ...]
            num_steps: Number of Langevin steps (reduced for speed)
            return_trajectory: Return entire trajectory
            
        Returns:
            Final samples or trajectory
        """
        x = init_samples.clone().to(self.device)
        x.requires_grad = True
        
        trajectory = [x.detach().cpu().clone()] if return_trajectory else None
        
        for _ in range(num_steps):
            # Ensure x requires grad for Langevin updates
            x = x.detach()
            x.requires_grad_(True)

            # Compute energy and gradients
            energy = energy_fn(x).sum()
            grad = torch.autograd.grad(energy, x, create_graph=False)[0]

            
            # Clip gradients for stability
            if self.clip_grad is not None:
                grad = torch.clamp(grad, -self.clip_grad, self.clip_grad)
            
            # Langevin update: x = x - step_size * ∇E(x) + noise
            noise = torch.randn_like(x) * self.noise_scale
            x = x - self.step_size * grad + noise
            
            # Clamp to valid range
            x = torch.clamp(x, -1, 1)
            x = x.detach()
            x.requires_grad = True
            
            if return_trajectory:
                trajectory.append(x.detach().cpu().clone())
        
        x = x.detach()
        
        if return_trajectory:
            return torch.stack(trajectory)
        return x
    
    def sample_with_diagnostics(
        self,
        energy_fn,
        init_samples: torch.Tensor,
        num_steps: int = 20
    ) -> Tuple[torch.Tensor, List[float], List[float]]:
        """
        Sample with diagnostic information.
        
        Args:
            energy_fn: Energy function
            init_samples: Initial samples
            num_steps: Number of steps
            
        Returns:
            Tuple of (samples, energy_trajectory, grad_norm_trajectory)
        """
        x = init_samples.clone().to(self.device)
        x.requires_grad = True
        
        energy_traj = []
        grad_norm_traj = []
        
        for _ in range(num_steps):
            # Compute energy
            energy = energy_fn(x)
            energy_traj.append(energy.mean().item())
            
            # Compute gradients
            grad = torch.autograd.grad(
                energy.sum(), x, create_graph=False
            )[0]
            grad_norm_traj.append(grad.norm().item())
            
            # Clip gradients
            if self.clip_grad is not None:
                grad = torch.clamp(grad, -self.clip_grad, self.clip_grad)
            
            # Update
            noise = torch.randn_like(x) * self.noise_scale
            x = x - self.step_size * grad + noise
            x = torch.clamp(x, -1, 1)
            x = x.detach()
            x.requires_grad = True
        
        return x.detach(), energy_traj, grad_norm_traj


class ReplayBuffer:
    """
    Replay buffer for storing negative samples.
    
    Memory-efficient for Kaggle: stores limited samples.
    """
    
    def __init__(
        self,
        buffer_size: int,
        sample_shape: Tuple[int, ...],
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum buffer size (keep small for Kaggle)
            sample_shape: Shape of each sample (C, H, W)
            device: Device to store buffer
        """
        self.buffer_size = buffer_size
        self.sample_shape = sample_shape
        self.device = device
        
        # Initialize empty buffer
        self.buffer = torch.FloatTensor(
            buffer_size, *sample_shape
        ).uniform_(-1, 1).to(device)
        self.current_size = 0
    
    def sample(self, batch_size: int, reinit_prob: float = 0.05) -> torch.Tensor:
        """
        Sample from replay buffer.
        
        Args:
            batch_size: Number of samples to return
            reinit_prob: Probability of reinitializing samples
            
        Returns:
            Samples from buffer [batch_size, C, H, W]
        """
        if self.current_size < batch_size:
            # Not enough samples, return random
            return torch.FloatTensor(
                batch_size, *self.sample_shape
            ).uniform_(-1, 1).to(self.device)
        
        # Sample from buffer
        indices = torch.randint(
            0, min(self.current_size, self.buffer_size), (batch_size,)
        )
        samples = self.buffer[indices].clone()
        
        # Randomly reinitialize some samples
        reinit_mask = torch.rand(batch_size) < reinit_prob
        n_reinit = reinit_mask.sum().item()
        if n_reinit > 0:
            samples[reinit_mask] = torch.FloatTensor(
                n_reinit, *self.sample_shape
            ).uniform_(-1, 1).to(self.device)
        
        return samples
    
    def add(self, samples: torch.Tensor):
        """
        Add samples to buffer.
        
        Args:
            samples: Samples to add [batch_size, C, H, W]
        """
        batch_size = samples.size(0)
        
        if self.current_size + batch_size <= self.buffer_size:
            # Add to buffer
            self.buffer[self.current_size:self.current_size + batch_size] = samples
            self.current_size += batch_size
        else:
            # Replace random samples
            indices = torch.randint(0, self.buffer_size, (batch_size,))
            self.buffer[indices] = samples
            self.current_size = self.buffer_size


def initialize_samples(
    batch_size: int,
    shape: Tuple[int, ...],
    method: str = 'uniform',
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Initialize samples for MCMC.
    
    Args:
        batch_size: Number of samples
        shape: Shape of each sample
        method: Initialization method ('uniform', 'gaussian', 'bernoulli')
        device: Device
        
    Returns:
        Initialized samples
    """
    if method == 'uniform':
        samples = torch.FloatTensor(batch_size, *shape).uniform_(-1, 1)
    elif method == 'gaussian':
        samples = torch.randn(batch_size, *shape)
    elif method == 'bernoulli':
        samples = torch.bernoulli(torch.ones(batch_size, *shape) * 0.5)
        samples = samples * 2 - 1  # Map to [-1, 1]
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return samples.to(device)


if __name__ == "__main__":
    print("Testing MCMC samplers...")
    
    # Test Persistent Chain
    print("\n1. Testing Persistent Chain...")
    chain = PersistentChain(num_chains=100, dim=784)
    samples = chain.get_samples(batch_size=32)
    print(f"Chain samples shape: {samples.shape}")
    chain.update(samples)
    print("✅ Persistent chain test passed")
    
    # Test Langevin Sampler
    print("\n2. Testing Langevin Sampler...")
    sampler = LangevinSampler(step_size=0.01, noise_scale=0.005)
    
    # Dummy energy function
    def dummy_energy(x):
        return (x ** 2).sum(dim=[1, 2, 3])
    
    init = torch.randn(8, 3, 32, 32)
    samples = sampler.sample(dummy_energy, init, num_steps=10)
    print(f"Langevin samples shape: {samples.shape}")
    print("✅ Langevin sampler test passed")
    
    # Test Replay Buffer
    print("\n3. Testing Replay Buffer...")
    buffer = ReplayBuffer(buffer_size=1000, sample_shape=(3, 32, 32))
    samples = buffer.sample(batch_size=16)
    print(f"Buffer samples shape: {samples.shape}")
    buffer.add(samples)
    print(f"Buffer size: {buffer.current_size}")
    print("✅ Replay buffer test passed")
    
    print("\n✅ All MCMC tests passed!")