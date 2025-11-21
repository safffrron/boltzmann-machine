"""
Training script for Convolutional Energy-Based Models.

Optimized for Kaggle: Fast training with Langevin dynamics.
Supports CD-k and PCD with replay buffer.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from conv_ebm import build_conv_ebm
from mcmc import LangevinSampler, ReplayBuffer, initialize_samples
from data import get_data_loader, prepare_conv_ebm_batch
from utils import (
    load_config, set_seed, get_device, save_checkpoint,
    MetricsLogger, AverageMeter, create_exp_dir
)
from plotting import save_image_grid


def compute_cd_loss(
    model: nn.Module,
    data_batch: torch.Tensor,
    sampler: LangevinSampler,
    replay_buffer: ReplayBuffer = None,
    cd_steps: int = 20,
    reinit_prob: float = 0.05
) -> tuple:
    """
    Compute contrastive divergence loss.
    
    Args:
        model: Energy model
        data_batch: Real data [batch_size, C, H, W]
        sampler: Langevin sampler
        replay_buffer: Replay buffer for PCD
        cd_steps: Number of Langevin steps
        reinit_prob: Probability to reinitialize samples
        
    Returns:
        Tuple of (loss, negative_samples, pos_energy, neg_energy)
    """
    batch_size = data_batch.size(0)
    
    # Positive phase: energy of real data
    pos_energy = model(data_batch)
    
    # Negative phase: sample from model
    if replay_buffer is not None:
        # PCD: use replay buffer
        neg_samples = replay_buffer.sample(batch_size, reinit_prob=reinit_prob)
    else:
        # CD: initialize from noise
        neg_samples = initialize_samples(
            batch_size,
            data_batch.shape[1:],
            method='uniform',
            device=data_batch.device
        )
    
    # Run Langevin dynamics
    neg_samples = sampler.sample(
        energy_fn=model,
        init_samples=neg_samples,
        num_steps=cd_steps
    )
    
    # Compute energy of negative samples
    neg_energy = model(neg_samples.detach())
    
    # Contrastive divergence loss
    # Minimize: E(real) - E(fake) = E(real) + E(fake) with targets flipped
    loss_pos = pos_energy.mean()
    loss_neg = neg_energy.mean()
    loss = loss_pos - loss_neg
    
    # Add L2 regularization on energies (helps stability)
    reg_loss = 0.001 * (pos_energy ** 2).mean() + 0.001 * (neg_energy ** 2).mean()
    loss = loss + reg_loss
    
    return loss, neg_samples, loss_pos.item(), loss_neg.item()


def train_conv_ebm(config: dict):
    """
    Train Convolutional EBM.
    
    Args:
        config: Configuration dictionary
    """
    # Setup
    set_seed(config.get('seed', 42))
    device = get_device(config.get('gpu_id'))
    
    # Create experiment directory
    exp_dir = create_exp_dir(
        config.get('output_dir', './results'),
        config.get('exp_name', 'conv_ebm')
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
        augment=config.get('augment', True),
        data_dir=config.get('data_dir', './data'),
        train=True
    )
    
    # Build model
    print("Building model...")
    model = build_conv_ebm(
        model_size=config.get('model_size', 'small'),
        input_channels=3,
        spectral_norm=config.get('spectral_norm', True)
    ).to(device)
    
    print(f"Model: {config.get('model_size', 'small')} ConvEBM")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(config.get('beta1', 0.0), config.get('beta2', 0.999)),
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    # Initialize Langevin sampler
    sampler = LangevinSampler(
        step_size=config['langevin_step_size'],
        noise_scale=config['langevin_noise'],
        clip_grad=config.get('langevin_clip', 0.01),
        device=device
    )
    
    # Initialize replay buffer for PCD
    replay_buffer = None
    if config.get('use_pcd', False):
        print(f"Using PCD with buffer size: {config['buffer_size']}")
        replay_buffer = ReplayBuffer(
            buffer_size=config['buffer_size'],
            sample_shape=(3, 32, 32),
            device=device
        )
    
    # Training loop
    print(f"\nTraining for {config['epochs']} epochs...")
    print(f"Langevin steps: {config['langevin_steps']}")
    
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        model.train()
        
        # Metrics
        loss_meter = AverageMeter()
        pos_energy_meter = AverageMeter()
        neg_energy_meter = AverageMeter()
        
        # Adjust learning rate
        if epoch in config.get('lr_schedule', {}):
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr_schedule'][epoch]
            print(f"Learning rate updated to {config['lr_schedule'][epoch]}")
        
        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, data in enumerate(pbar):
            if isinstance(data, (list, tuple)):
                data = data[0]
            
            # Move to device and normalize
            data = prepare_conv_ebm_batch(data.to(device))
            
            # Compute CD loss
            loss, neg_samples, pos_energy, neg_energy = compute_cd_loss(
                model=model,
                data_batch=data,
                sampler=sampler,
                replay_buffer=replay_buffer,
                cd_steps=config['langevin_steps'],
                reinit_prob=config.get('reinit_prob', 0.05)
            )
            
            # Update replay buffer
            if replay_buffer is not None:
                replay_buffer.add(neg_samples.detach())
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            if config.get('grad_clip', None):
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            
            # Track metrics
            loss_meter.update(loss.item(), data.size(0))
            pos_energy_meter.update(pos_energy, data.size(0))
            neg_energy_meter.update(neg_energy, data.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'E_pos': f'{pos_energy_meter.avg:.2f}',
                'E_neg': f'{neg_energy_meter.avg:.2f}'
            })
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.log(epoch, {
            'loss': loss_meter.avg,
            'pos_energy': pos_energy_meter.avg,
            'neg_energy': neg_energy_meter.avg,
            'energy_gap': pos_energy_meter.avg - neg_energy_meter.avg,
            'epoch_time': epoch_time
        })
        
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Loss: {loss_meter.avg:.4f}, "
              f"E_pos: {pos_energy_meter.avg:.2f}, "
              f"E_neg: {neg_energy_meter.avg:.2f}, "
              f"Gap: {pos_energy_meter.avg - neg_energy_meter.avg:.2f}, "
              f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = loss_meter.avg < best_loss
        if is_best:
            best_loss = loss_meter.avg
        
        if (epoch + 1) % config.get('save_every', 5) == 0 or is_best:
            checkpoint_path = os.path.join(
                exp_dir, 'checkpoints',
                f"conv_ebm_epoch_{epoch+1}.pt" if not is_best else "conv_ebm_best.pt"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=loss_meter.avg,
                save_path=checkpoint_path,
                additional_info={
                    'config': config,
                    'pos_energy': pos_energy_meter.avg,
                    'neg_energy': neg_energy_meter.avg
                }
            )
            if is_best:
                print(f"✓ Saved best model (loss: {best_loss:.4f})")
        
        # Generate samples periodically
        if (epoch + 1) % config.get('sample_every', 10) == 0:
            print("Generating samples...")
            model.eval()

            # Sample using long Langevin chain (needs gradients wrt x!)
            num_samples = 64
            samples = initialize_samples(
                num_samples,
                (3, 32, 32),
                method='uniform',
                device=device
            )

            samples = sampler.sample(
                energy_fn=model,
                init_samples=samples,
                num_steps=config.get('sample_steps', 200)
            )

            # From here on, we don't need grads anymore
            with torch.no_grad():
                # Denormalize for visualization
                samples_vis = (samples + 1) / 2
                samples_vis = torch.clamp(samples_vis, 0, 1)

                # Save grid
                grid_path = os.path.join(
                    exp_dir, 'samples',
                    f"samples_epoch_{epoch+1}.png"
                )
                save_image_grid(samples_vis.cpu(), grid_path, nrow=8, normalize=False)
                print(f"Saved samples to {grid_path}")

            model.train()

    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Results saved to: {exp_dir}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Train Convolutional EBM')
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
    train_conv_ebm(config)


if __name__ == "__main__":
    # Example usage for testing
    test_config = {
        'exp_name': 'conv_ebm_test',
        'dataset': 'cifar10',
        'batch_size': 64,
        'augment': True,
        'model_size': 'tiny',  # Use tiny for quick test
        'spectral_norm': True,
        'learning_rate': 0.0001,
        'beta1': 0.0,
        'beta2': 0.999,
        'weight_decay': 0.0,
        'langevin_steps': 20,
        'langevin_step_size': 0.01,
        'langevin_noise': 0.005,
        'langevin_clip': 0.01,
        'grad_clip': 0.1,
        'use_pcd': False,
        'reinit_prob': 0.05,
        'epochs': 2,  # Short test
        'seed': 42,
        'save_every': 1,
        'sample_every': 1,
        'sample_steps': 100,
        'output_dir': './test_results',
        'data_dir': './data'
    }
    
    print("Running quick test...")
    train_conv_ebm(test_config)