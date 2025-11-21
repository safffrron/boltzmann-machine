"""
Data loading utilities for EBM experiments.

Handles MNIST, FashionMNIST, and CIFAR-10 datasets with appropriate
preprocessing for RBMs and convolutional EBMs.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional


def get_mnist_loaders(
    batch_size: int = 128,
    binarize: bool = True,
    data_dir: str = './data',
    num_workers: int = 4,
    train: bool = True
) -> DataLoader:
    """
    Get MNIST data loader for RBM training.
    
    Args:
        batch_size: Batch size for training
        binarize: Whether to binarize images (for RBM)
        data_dir: Directory to store/load data
        num_workers: Number of worker processes
        train: Whether to load training or test set
        
    Returns:
        DataLoader for MNIST
    """
    transform_list = [transforms.ToTensor()]
    
    if binarize:
        # Binarize with threshold 0.5
        transform_list.append(transforms.Lambda(lambda x: (x > 0.5).float()))
    else:
        # Normalize to [0, 1]
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )
    
    return loader


def get_fashion_mnist_loaders(
    batch_size: int = 128,
    binarize: bool = True,
    data_dir: str = './data',
    num_workers: int = 4,
    train: bool = True
) -> DataLoader:
    """
    Get FashionMNIST data loader for RBM training.
    
    Args:
        batch_size: Batch size for training
        binarize: Whether to binarize images (for RBM)
        data_dir: Directory to store/load data
        num_workers: Number of worker processes
        train: Whether to load training or test set
        
    Returns:
        DataLoader for FashionMNIST
    """
    transform_list = [transforms.ToTensor()]
    
    if binarize:
        transform_list.append(transforms.Lambda(lambda x: (x > 0.5).float()))
    else:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )
    
    return loader


def get_cifar10_loaders(
    batch_size: int = 128,
    augment: bool = True,
    data_dir: str = './data',
    num_workers: int = 4,
    train: bool = True
) -> DataLoader:
    """
    Get CIFAR-10 data loader for Conv-EBM training.
    
    Args:
        batch_size: Batch size for training
        augment: Whether to apply data augmentation
        data_dir: Directory to store/load data
        num_workers: Number of worker processes
        train: Whether to load training or test set
        
    Returns:
        DataLoader for CIFAR-10
    """
    if train and augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )
    
    return loader


class UnlabeledDataset(Dataset):
    """Wrapper to remove labels from dataset (for EBM training)."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, _ = self.dataset[idx]
        return data


def get_data_loader(
    dataset_name: str,
    batch_size: int = 128,
    data_dir: str = './data',
    num_workers: int = 4,
    train: bool = True,
    **kwargs
) -> DataLoader:
    """
    Generic data loader getter.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fashion_mnist', 'cifar10')
        batch_size: Batch size
        data_dir: Data directory
        num_workers: Number of workers
        train: Whether to load training or test set
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        DataLoader for specified dataset
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return get_mnist_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            num_workers=num_workers,
            train=train,
            **kwargs
        )
    elif dataset_name == 'fashion_mnist':
        return get_fashion_mnist_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            num_workers=num_workers,
            train=train,
            **kwargs
        )
    elif dataset_name == 'cifar10':
        return get_cifar10_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            num_workers=num_workers,
            train=train,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def binarize_batch(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Binarize a batch of images.
    
    Args:
        x: Input tensor
        threshold: Binarization threshold
        
    Returns:
        Binarized tensor
    """
    return (x > threshold).float()


def add_noise(x: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    
    Args:
        x: Input tensor
        noise_level: Standard deviation of noise
        
    Returns:
        Noisy tensor
    """
    noise = torch.randn_like(x) * noise_level
    return torch.clamp(x + noise, 0, 1)


def get_data_statistics(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of dataset.
    
    Args:
        loader: DataLoader
        
    Returns:
        Tuple of (mean, std)
    """
    mean = 0.
    std = 0.
    total = 0
    
    for data in loader:
        if isinstance(data, (list, tuple)):
            data = data[0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total += batch_samples
    
    mean /= total
    std /= total
    
    return mean, std


def prepare_rbm_batch(x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """
    Prepare batch for RBM training.
    
    Args:
        x: Input batch [B, C, H, W] or [B, D]
        flatten: Whether to flatten spatial dimensions
        
    Returns:
        Processed batch
    """
    if flatten and x.dim() == 4:
        # Flatten spatial dimensions
        return x.view(x.size(0), -1)
    return x


def prepare_conv_ebm_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Prepare batch for Conv-EBM training.
    
    Args:
        x: Input batch [B, C, H, W]
        
    Returns:
        Processed batch (keep spatial structure)
    """
    # Ensure proper normalization
    if x.min() >= 0 and x.max() <= 1:
        # Normalize to [-1, 1]
        x = x * 2 - 1
    return x


def sample_data_batch(loader: DataLoader, num_samples: int) -> torch.Tensor:
    """
    Sample a batch of data from loader.
    
    Args:
        loader: DataLoader
        num_samples: Number of samples to get
        
    Returns:
        Batch of samples
    """
    samples = []
    for data in loader:
        if isinstance(data, (list, tuple)):
            data = data[0]
        samples.append(data)
        if sum(s.size(0) for s in samples) >= num_samples:
            break
    
    samples = torch.cat(samples, dim=0)[:num_samples]
    return samples


if __name__ == "__main__":
    # Test data loaders
    print("Testing MNIST loader...")
    mnist_loader = get_mnist_loaders(batch_size=64, binarize=True)
    batch = next(iter(mnist_loader))
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    print(f"MNIST batch shape: {batch.shape}")
    print(f"MNIST value range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    print("\nTesting CIFAR-10 loader...")
    cifar_loader = get_cifar10_loaders(batch_size=64)
    batch = next(iter(cifar_loader))
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    print(f"CIFAR-10 batch shape: {batch.shape}")
    print(f"CIFAR-10 value range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    print("\nData loaders test passed!")