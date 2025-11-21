"""
Training script for Restricted Boltzmann Machine.

Optimized for Kaggle: Small models, efficient memory usage, fast training.
Supports both CD-k and PCD-k training.
"""

import os
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm

from rbm import BinaryRBM
from mcmc import PersistentChain
from data import get_data_loader
from utils import (
    load_config, set_seed, get_device, save_checkpoint,
    MetricsLogger, AverageMeter, create_exp_dir
)
from plotting import save_image_grid


def train_rbm(config: dict):
    """
    Train RBM with CD-k or PCD-k.
    
    Args:
        config: Configuration dictionary
    """
    # Setup
    set_seed(config.get('seed', 42))
    device = get_device(config.get('gpu_id'))
    
    # Create experiment directory
    exp_dir = create_exp_dir(
        config.get('output_dir', './results'),
        config.get('exp_name', 'rbm')
    )
    print(f"Experiment directory: {exp_dir}")
    
    # Initialize logger
    logger = MetricsLogger(
        log_dir=os.path.join(exp_dir, 'logs'),
        experiment_name=config['exp_name']
    )
    
    # Load data
    print("Loading data...")
    train_loader = get_data_loader(
        dataset_name=config['dataset'],
        batch_size=config['batch_size'],
        binarize=config.get('binarize', True),
        data_dir=config.get('data_dir', './data'),
        train=True
    )
    
    # Get data dimension
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    n_visible = sample_batch.view(sample_batch.size(0), -1).size(1)
    print(f"Visible dimension: {n_visible}")
    
    # Initialize RBM
    print("Initializing RBM...")
    rbm = BinaryRBM(
        n_visible=n_visible,
        n_hidden=config['n_hidden'],
        learning_rate=config['learning_rate'],
        momentum=config.get('momentum', 0.5),
        weight_decay=config.get('weight_decay', 0.0001),
        use_cuda=(device.type == 'cuda')
    )
    
    print(f"RBM: {n_visible} visible, {config['n_hidden']} hidden units")
    print(f"Parameters: {sum(p.numel() for p in rbm.parameters()):,}")
    print(f"Device: {device}")
    
    # Initialize persistent chain for PCD
    persistent_chain = None
    if config.get('use_pcd', False):
        print(f"Using PCD with buffer size: {config['pcd_buffer_size']}")
        persistent_chain = PersistentChain(
            num_chains=config['pcd_buffer_size'],
            dim=n_visible,
            device=device
        )
    
    # Training loop
    print(f"\nTraining for {config['epochs']} epochs...")
    print(f"CD-k steps: {config['cd_k']}")
    
    best_loss = float('inf')
    cd_k = config['cd_k']
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Metrics
        recon_error_meter = AverageMeter()
        free_energy_meter = AverageMeter()
        
        # Adjust learning rate
        if epoch in config.get('lr_schedule', {}):
            rbm.learning_rate = config['lr_schedule'][epoch]
            print(f"Learning rate updated to {rbm.learning_rate}")
        
        # Adjust momentum
        if epoch == config.get('momentum_epoch', 5):
            rbm.momentum = config.get('final_momentum', 0.9)
            print(f"Momentum updated to {rbm.momentum}")
        
        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, data in enumerate(pbar):
            if isinstance(data, (list, tuple)):
                data = data[0]
            
            # Flatten and move to device
            v = data.view(data.size(0), -1).to(device)
            
            # Get persistent samples if using PCD
            persistent = None
            if persistent_chain is not None:
                persistent = persistent_chain.get_samples(v.size(0))
            
            # Update weights
            recon_error, v_neg = rbm.update_weights(
                v_pos=v,
                k=cd_k,
                persistent=persistent
            )
            
            # Update persistent chain
            if persistent_chain is not None:
                persistent_chain.update(v_neg)
            
            # Track metrics
            recon_error_meter.update(recon_error, v.size(0))
            
            with torch.no_grad():
                free_energy = rbm.free_energy(v).mean().item()
            free_energy_meter.update(free_energy, v.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'recon': f'{recon_error_meter.avg:.4f}',
                'energy': f'{free_energy_meter.avg:.2f}'
            })
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.log(epoch, {
            'reconstruction_error': recon_error_meter.avg,
            'free_energy': free_energy_meter.avg,
            'epoch_time': epoch_time
        })
        
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Recon Error: {recon_error_meter.avg:.4f}, "
              f"Free Energy: {free_energy_meter.avg:.2f}, "
              f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = recon_error_meter.avg < best_loss
        if is_best:
            best_loss = recon_error_meter.avg
        
        if (epoch + 1) % config.get('save_every', 5) == 0 or is_best:
            checkpoint_path = os.path.join(
                exp_dir, 'checkpoints',
                f"rbm_epoch_{epoch+1}.pt" if not is_best else "rbm_best.pt"
            )
            save_checkpoint(
                model=rbm,
                optimizer=None,
                epoch=epoch,
                loss=recon_error_meter.avg,
                save_path=checkpoint_path,
                additional_info={
                    'config': config,
                    'free_energy': free_energy_meter.avg
                }
            )
            if is_best:
                print(f"✓ Saved best model (recon error: {best_loss:.4f})")
        
        # Generate samples periodically
        if (epoch + 1) % config.get('sample_every', 10) == 0:
            print("Generating samples...")
            with torch.no_grad():
                samples = rbm.generate_samples(
                    num_samples=64,
                    k=config.get('sample_steps', 1000)
                )
                
                # Save grid
                grid_path = os.path.join(
                    exp_dir, 'samples',
                    f"samples_epoch_{epoch+1}.png"
                )
                
                # Reshape for visualization
                if config['dataset'] in ['mnist', 'fashion_mnist']:
                    samples = samples.view(-1, 1, 28, 28)
                
                save_image_grid(samples.cpu(), grid_path, nrow=8)
                print(f"Saved samples to {grid_path}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best reconstruction error: {best_loss:.4f}")
    print(f"Results saved to: {exp_dir}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Train RBM')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if args.gpu_id is not None:
        config['gpu_id'] = args.gpu_id
    
    # Train
    train_rbm(config)


if __name__ == "__main__":
    # Example usage for testing
    test_config = {
        'exp_name': 'rbm_mnist_cd5_test',
        'dataset': 'mnist',
        'batch_size': 64,
        'binarize': True,
        'n_hidden': 256,
        'learning_rate': 0.01,
        'momentum': 0.5,
        'final_momentum': 0.9,
        'momentum_epoch': 5,
        'weight_decay': 0.0001,
        'cd_k': 5,
        'use_pcd': False,
        'epochs': 3,  # Short test
        'seed': 42,
        'save_every': 1,
        'sample_every': 1,
        'sample_steps': 100,
        'output_dir': './test_results',
        'data_dir': './data'
    }
    
    print("Running quick test...")
    train_rbm(test_config)