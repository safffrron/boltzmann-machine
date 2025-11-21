"""
Restricted Boltzmann Machine (RBM) implementation.

Optimized for Kaggle environments with limited GPU memory.
Binary visible and hidden units.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class BinaryRBM(nn.Module):
    """
    Binary Restricted Boltzmann Machine.
    
    Energy function: E(v,h) = -v^T W h - b^T v - c^T h
    
    Optimized for Kaggle: Small hidden layer, efficient sampling.
    """
    
    def __init__(
        self,
        n_visible: int = 784,
        n_hidden: int = 256,  # Reduced from 512 for Kaggle
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        use_cuda: bool = True
    ):
        """
        Initialize RBM.
        
        Args:
            n_visible: Number of visible units (e.g., 784 for MNIST)
            n_hidden: Number of hidden units (kept small for Kaggle)
            learning_rate: Learning rate for weight updates
            momentum: Momentum for weight updates
            weight_decay: L2 regularization
            use_cuda: Whether to use CUDA if available
        """
        super(BinaryRBM, self).__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize weights and biases
        # Using smaller initial weights for stability
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(n_visible))  # Visible bias
        self.b_h = nn.Parameter(torch.zeros(n_hidden))   # Hidden bias
        
        # Velocity for momentum (not parameters)
        self.register_buffer('W_velocity', torch.zeros_like(self.W))
        self.register_buffer('b_v_velocity', torch.zeros_like(self.b_v))
        self.register_buffer('b_h_velocity', torch.zeros_like(self.b_h))
        
        # Device
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def sample_hidden(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units.
        
        Args:
            v: Visible units [batch_size, n_visible]
            
        Returns:
            Tuple of (hidden probabilities, hidden samples)
        """
        # Activation: a_h = sigmoid(v @ W + b_h)
        activation = torch.mm(v, self.W) + self.b_h
        prob_h = torch.sigmoid(activation)
        
        # Sample from Bernoulli
        sample_h = torch.bernoulli(prob_h)
        
        return prob_h, sample_h
    
    def sample_visible(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units.
        
        Args:
            h: Hidden units [batch_size, n_hidden]
            
        Returns:
            Tuple of (visible probabilities, visible samples)
        """
        # Activation: a_v = sigmoid(h @ W^T + b_v)
        activation = torch.mm(h, self.W.t()) + self.b_v
        prob_v = torch.sigmoid(activation)
        
        # Sample from Bernoulli
        sample_v = torch.bernoulli(prob_v)
        
        return prob_v, sample_v
    
    def gibbs_step(self, v: torch.Tensor) -> torch.Tensor:
        """
        Perform one Gibbs sampling step: v -> h -> v'
        
        Args:
            v: Visible units
            
        Returns:
            New visible samples
        """
        _, h = self.sample_hidden(v)
        _, v_new = self.sample_visible(h)
        return v_new
    
    def gibbs_sampling(
        self,
        v: torch.Tensor,
        k: int = 1,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Perform k steps of Gibbs sampling.
        
        Args:
            v: Initial visible units
            k: Number of Gibbs steps
            return_probs: Return probabilities instead of samples
            
        Returns:
            Sampled visible units after k steps
        """
        v_sample = v
        
        for _ in range(k):
            _, h_sample = self.sample_hidden(v_sample)
            prob_v, v_sample = self.sample_visible(h_sample)
        
        if return_probs:
            return prob_v
        return v_sample
    
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute free energy of visible units.
        
        F(v) = -b_v^T v - sum(softplus(v @ W + b_h))
        
        Args:
            v: Visible units [batch_size, n_visible]
            
        Returns:
            Free energy [batch_size]
        """
        v_term = torch.mv(v, self.b_v)
        wx_b = torch.mm(v, self.W) + self.b_h
        h_term = torch.sum(F.softplus(wx_b), dim=1)
        
        return -v_term - h_term
    
    def contrastive_divergence(
        self,
        v_pos: torch.Tensor,
        k: int = 1,
        persistent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Contrastive Divergence (CD-k) update.
        
        Args:
            v_pos: Positive phase visible data [batch_size, n_visible]
            k: Number of Gibbs steps
            persistent: Persistent chain for PCD (None for CD)
            
        Returns:
            Tuple of (positive gradient, negative gradient, reconstruction)
        """
        batch_size = v_pos.size(0)
        
        # Positive phase
        prob_h_pos, _ = self.sample_hidden(v_pos)
        
        # Negative phase
        if persistent is not None:
            # Persistent CD (PCD): Continue from persistent chain
            v_neg = persistent
        else:
            # Standard CD: Start from data
            v_neg = v_pos
        
        # Run k steps of Gibbs sampling
        v_neg = self.gibbs_sampling(v_neg, k=k)
        prob_h_neg, _ = self.sample_hidden(v_neg)
        
        # Compute gradients (expectations)
        positive_grad = torch.mm(v_pos.t(), prob_h_pos) / batch_size
        negative_grad = torch.mm(v_neg.t(), prob_h_neg) / batch_size
        
        return positive_grad, negative_grad, v_neg
    
    def update_weights(
        self,
        v_pos: torch.Tensor,
        k: int = 1,
        persistent: Optional[torch.Tensor] = None
    ) -> Tuple[float, torch.Tensor]:
        """
        Update RBM parameters using CD-k.
        
        Args:
            v_pos: Positive phase data
            k: Number of CD steps
            persistent: Persistent chain for PCD
            
        Returns:
            Tuple of (reconstruction error, negative samples for PCD)
        """
        batch_size = v_pos.size(0)
        
        # Compute gradients
        pos_grad, neg_grad, v_neg = self.contrastive_divergence(v_pos, k, persistent)
        
        # Bias gradients
        pos_bias_v = torch.mean(v_pos, dim=0)
        neg_bias_v = torch.mean(v_neg, dim=0)
        
        prob_h_pos, _ = self.sample_hidden(v_pos)
        prob_h_neg, _ = self.sample_hidden(v_neg)
        pos_bias_h = torch.mean(prob_h_pos, dim=0)
        neg_bias_h = torch.mean(prob_h_neg, dim=0)
        
        # Update with momentum and weight decay
        dW = self.learning_rate * (pos_grad - neg_grad - self.weight_decay * self.W)
        db_v = self.learning_rate * (pos_bias_v - neg_bias_v)
        db_h = self.learning_rate * (pos_bias_h - neg_bias_h)
        
        # Apply momentum
        self.W_velocity.mul_(self.momentum).add_(dW)
        self.b_v_velocity.mul_(self.momentum).add_(db_v)
        self.b_h_velocity.mul_(self.momentum).add_(db_h)
        
        # Update parameters
        self.W.data.add_(self.W_velocity)
        self.b_v.data.add_(self.b_v_velocity)
        self.b_h.data.add_(self.b_h_velocity)
        
        # Compute reconstruction error
        with torch.no_grad():
            v_recon = self.gibbs_sampling(v_pos, k=1, return_probs=True)
            recon_error = F.mse_loss(v_recon, v_pos).item()
        
        return recon_error, v_neg.detach()
    
    def reconstruct(self, v: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Reconstruct visible units.
        
        Args:
            v: Input visible units
            k: Number of Gibbs steps
            
        Returns:
            Reconstructed visible units
        """
        return self.gibbs_sampling(v, k=k, return_probs=True)
    
    def generate_samples(
        self,
        num_samples: int,
        k: int = 1000,
        return_chain: bool = False
    ) -> torch.Tensor:
        """
        Generate samples from the model using Gibbs sampling.
        
        Args:
            num_samples: Number of samples to generate
            k: Number of Gibbs steps
            return_chain: Return entire chain instead of final samples
            
        Returns:
            Generated samples [num_samples, n_visible] or chain
        """
        # Initialize from random
        v = torch.bernoulli(torch.ones(num_samples, self.n_visible) * 0.5).to(self.device)
        
        if return_chain:
            chain = [v.cpu()]
        
        # Run Gibbs sampling
        for i in range(k):
            v = self.gibbs_step(v)
            if return_chain and (i % 10 == 0):
                chain.append(v.cpu())
        
        if return_chain:
            return torch.stack(chain)
        
        return v
    
    def pseudo_likelihood(self, v: torch.Tensor) -> float:
        """
        Compute pseudo-likelihood (approximation to likelihood).
        
        Args:
            v: Visible units
            
        Returns:
            Average pseudo-likelihood
        """
        batch_size = v.size(0)
        
        # Randomly flip one bit per sample
        i = torch.randint(0, self.n_visible, (batch_size,))
        v_flip = v.clone()
        v_flip[torch.arange(batch_size), i] = 1 - v_flip[torch.arange(batch_size), i]
        
        # Compute free energies
        fe = self.free_energy(v)
        fe_flip = self.free_energy(v_flip)
        
        # Pseudo-likelihood
        pl = torch.mean(self.n_visible * torch.log(torch.sigmoid(fe_flip - fe)))
        
        return pl.item()
    
    def get_weights_visualization(self) -> torch.Tensor:
        """
        Get weight matrix for visualization.
        
        Returns:
            Weight matrix reshaped for visualization
        """
        return self.W.detach().cpu()


if __name__ == "__main__":
    print("Testing RBM implementation...")
    
    # Create small RBM for testing
    rbm = BinaryRBM(
        n_visible=784,
        n_hidden=128,
        learning_rate=0.01,
        use_cuda=False
    )
    
    print(f"RBM created with {rbm.n_visible} visible and {rbm.n_hidden} hidden units")
    print(f"Device: {rbm.device}")
    print(f"Number of parameters: {sum(p.numel() for p in rbm.parameters())}")
    
    # Test forward pass
    batch_size = 32
    v = torch.bernoulli(torch.ones(batch_size, 784) * 0.5)
    
    # Test sampling
    prob_h, sample_h = rbm.sample_hidden(v)
    print(f"\nHidden probabilities shape: {prob_h.shape}")
    print(f"Hidden samples shape: {sample_h.shape}")
    
    # Test Gibbs sampling
    v_recon = rbm.gibbs_sampling(v, k=5)
    print(f"Reconstructed visible shape: {v_recon.shape}")
    
    # Test CD update
    recon_error, v_neg = rbm.update_weights(v, k=1)
    print(f"Reconstruction error: {recon_error:.4f}")
    
    # Test generation
    samples = rbm.generate_samples(num_samples=10, k=100)
    print(f"Generated samples shape: {samples.shape}")
    
    # Test free energy
    energy = rbm.free_energy(v)
    print(f"Free energy shape: {energy.shape}")
    
    print("\n✅ RBM tests passed!")