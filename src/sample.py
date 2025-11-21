"""
Sampling script for trained EBM models.

Generate samples from RBM or Conv-EBM for evaluation.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from rbm import BinaryRBM
from conv_ebm import build_conv_ebm
from mcmc import LangevinSampler, initialize_samples
from utils import load_config, load_checkpoint, get_device
from plotting import save_image_grid, plot_sampling_trajectory


def sample_rbm(
    checkpoint_path: str,
    num_samples: int = 100,
    num_steps: int = 1000,
    output_dir: str = './samples',
    save_trajectory: bool = False,
    device: torch.device = None
):
    """
    Generate samples from trained RBM.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to generate
        num_steps: Number of Gibbs steps
        output_dir: Directory to save samples
        save_trajectory: Whether to save sampling trajectory
        device: Device to use
    """
    if device is None:
        device = get_device()
    
    print(f"Loading RBM from {checkpoint_path}...")
    
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create RBM
    n_visible = config.get('n_visible', 784)
    n_hidden = config.get('n_hidden', 256)
    
    rbm = BinaryRBM(
        n_visible=n_visible,
        n_hidden=n_hidden,
        use_cuda=(device.type == 'cuda')
    )
    
    # Load weights
    rbm.load_state_dict(checkpoint['model_state_dict'])
    rbm.eval()
    
    print(f"RBM loaded: {n_visible} visible, {n_hidden} hidden units")
    print(f"Generating {num_samples} samples with {num_steps} Gibbs steps...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    with torch.no_grad():
        if save_trajectory:
            # Save trajectory
            samples = rbm.generate_samples(
                num_samples=min(16, num_samples),
                k=num_steps,
                return_chain=True
            )
            
            # Select keyframes
            n_frames = min(10, samples.size(0))
            indices = np.linspace(0, samples.size(0)-1, n_frames, dtype=int)
            trajectory = [samples[i] for i in indices]
            
            # Save trajectory
            traj_path = os.path.join(output_dir, 'sampling_trajectory.png')
            plot_sampling_trajectory(trajectory, traj_path, num_chains=5)
            print(f"Saved trajectory to {traj_path}")
        
        # Generate final samples
        samples = rbm.generate_samples(num_samples=num_samples, k=num_steps)
        
        # Reshape for visualization
        img_size = int(np.sqrt(n_visible))
        samples = samples.view(-1, 1, img_size, img_size)
        
        # Save grid
        grid_path = os.path.join(output_dir, f'rbm_samples_{num_samples}.png')
        save_image_grid(samples.cpu(), grid_path, nrow=10)
        print(f"Saved samples to {grid_path}")
        
        # Save individual samples as numpy
        samples_np = samples.cpu().numpy()
        np_path = os.path.join(output_dir, f'rbm_samples_{num_samples}.npy')
        np.save(np_path, samples_np)
        print(f"Saved numpy samples to {np_path}")
    
    return samples


def sample_conv_ebm(
    checkpoint_path: str,
    config_path: str,
    num_samples: int = 100,
    num_steps: int = 200,
    output_dir: str = './samples',
    save_trajectory: bool = False,
    device: torch.device = None
):
    """
    Generate samples from trained Conv-EBM.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_samples: Number of samples to generate
        num_steps: Number of Langevin steps
        output_dir: Directory to save samples
        save_trajectory: Whether to save sampling trajectory
        device: Device to use
    """
    if device is None:
        device = get_device()
    
    print(f"Loading Conv-EBM from {checkpoint_path}...")
    
    # Load config
    config = load_config(config_path)
    
    # Build model
    model = build_conv_ebm(
        model_size=config.get('model_size', 'small'),
        input_channels=3,
        spectral_norm=config.get('spectral_norm', True)
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    
    print(f"Model loaded: {config.get('model_size', 'small')} ConvEBM")
    print(f"Generating {num_samples} samples with {num_steps} Langevin steps...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Langevin sampler
    sampler = LangevinSampler(
        step_size=config.get('langevin_step_size', 0.01),
        noise_scale=config.get('langevin_noise', 0.005),
        clip_grad=config.get('langevin_clip', 0.01),
        device=device
    )
    
    # Generate samples in batches
    batch_size = 64
    all_samples = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating"):
            batch_num = min(batch_size, num_samples - i)
            
            # Initialize samples
            init_samples = initialize_samples(
                batch_num,
                (3, 32, 32),
                method='uniform',
                device=device
            )
            
            # Run Langevin dynamics
            if save_trajectory and i == 0:
                # Save trajectory for first batch
                samples = sampler.sample(
                    energy_fn=model,
                    init_samples=init_samples[:16],
                    num_steps=num_steps,
                    return_trajectory=True
                )
                
                # Select keyframes
                n_frames = min(10, samples.size(0))
                indices = np.linspace(0, samples.size(0)-1, n_frames, dtype=int)
                trajectory = [samples[i] for i in indices]
                
                # Denormalize
                trajectory = [(t + 1) / 2 for t in trajectory]
                
                # Save trajectory
                traj_path = os.path.join(output_dir, 'sampling_trajectory.png')
                plot_sampling_trajectory(trajectory, traj_path, num_chains=5)
                print(f"\nSaved trajectory to {traj_path}")
                
                # Continue with rest of samples
                samples = sampler.sample(
                    energy_fn=model,
                    init_samples=init_samples,
                    num_steps=num_steps
                )
            else:
                samples = sampler.sample(
                    energy_fn=model,
                    init_samples=init_samples,
                    num_steps=num_steps
                )
            
            # Denormalize to [0, 1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            all_samples.append(samples.cpu())
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    
    # Save grid
    grid_path = os.path.join(output_dir, f'conv_ebm_samples_{num_samples}.png')
    save_image_grid(all_samples, grid_path, nrow=10, normalize=False)
    print(f"Saved samples to {grid_path}")
    
    # Save as numpy
    samples_np = all_samples.numpy()
    np_path = os.path.join(output_dir, f'conv_ebm_samples_{num_samples}.npy')
    np.save(np_path, samples_np)
    print(f"Saved numpy samples to {np_path}")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Generate samples from EBM')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rbm', 'conv_ebm'],
                       help='Type of model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (required for conv_ebm)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of MCMC steps (default: 1000 for RBM, 200 for Conv-EBM)')
    parser.add_argument('--output', type=str, default='./samples',
                       help='Output directory')
    parser.add_argument('--trajectory', action='store_true',
                       help='Save sampling trajectory')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device(args.gpu_id)
    
    # Generate samples
    if args.model_type == 'rbm':
        num_steps = args.num_steps if args.num_steps else 1000
        sample_rbm(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            num_steps=num_steps,
            output_dir=args.output,
            save_trajectory=args.trajectory,
            device=device
        )
    elif args.model_type == 'conv_ebm':
        if args.config is None:
            raise ValueError("--config is required for conv_ebm")
        
        num_steps = args.num_steps if args.num_steps else 200
        sample_conv_ebm(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_samples=args.num_samples,
            num_steps=num_steps,
            output_dir=args.output,
            save_trajectory=args.trajectory,
            device=device
        )


if __name__ == "__main__":
    main()