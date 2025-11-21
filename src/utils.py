"""
Utility functions for EBM training and evaluation.

Includes config loading, logging, checkpointing, and general helpers.
"""

import os
import json
import yaml
import torch
import random
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of experiment
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get torch device.
    
    Args:
        gpu_id: GPU ID to use (None for auto-select)
        
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cuda')
    return torch.device('cpu')


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    save_path: str,
    additional_info: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (can be None)
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        additional_info: Additional information to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    # Only save optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics = {}
        self.log_file = os.path.join(
            log_dir, 
            f"{experiment_name}_metrics.json"
        )
        os.makedirs(log_dir, exist_ok=True)
        
    def log(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics at a step.
        
        Args:
            step: Training step/epoch
            metrics: Dictionary of metric names and values
        """
        if step not in self.metrics:
            self.metrics[step] = {}
        self.metrics[step].update(metrics)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metric(self, metric_name: str) -> Dict[int, float]:
        """
        Get all values for a specific metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Dictionary mapping step to value
        """
        return {
            step: metrics.get(metric_name) 
            for step, metrics in self.metrics.items() 
            if metric_name in metrics
        }
    
    def load(self, log_file: Optional[str] = None):
        """
        Load metrics from file.
        
        Args:
            log_file: Path to log file (uses default if None)
        """
        if log_file is None:
            log_file = self.log_file
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.metrics = json.load(f)
                # Convert string keys back to ints
                self.metrics = {
                    int(k): v for k, v in self.metrics.items()
                }


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_exp_dir(base_dir: str, exp_name: str) -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        exp_name: Name of experiment
        
    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    return exp_dir


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def format_time(seconds: float) -> str:
    """
    Format seconds into readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def ensure_dir(directory: str):
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_dict_to_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save to
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_dict_from_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if should stop early.
        
        Args:
            score: Current metric score
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print(f"Random number after seed: {torch.rand(1).item():.4f}")
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test metrics logger
    logger = MetricsLogger("./test_logs", "test_exp")
    logger.log(0, {"loss": 1.5, "accuracy": 0.6})
    logger.log(1, {"loss": 1.2, "accuracy": 0.7})
    print(f"Logged metrics: {logger.metrics}")
    
    # Test average meter
    meter = AverageMeter()
    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    print(f"Average: {meter.avg:.2f}")
    
    print("\nUtility functions test passed!")