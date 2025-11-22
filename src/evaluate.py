"""
Evaluation script for trained EBM models.

Computes FID, Inception Score, LPIPS diversity, and MCMC diagnostics.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

from rbm import BinaryRBM
from conv_ebm import build_conv_ebm
from mcmc import LangevinSampler, initialize_samples
from data import get_data_loader, sample_data_batch
from utils import load_config, load_checkpoint, get_device, save_dict_to_json
from metrics import (
    FIDCalculator, InceptionScoreCalculator, LPIPSDiversity,
    ais_log_likelihood, compute_autocorrelation, effective_sample_size,
    energy_autocorrelation, mixing_time
)
from sample import sample_rbm, sample_conv_ebm


def evaluate_rbm(
    checkpoint_path: str,
    output_dir: str = './evaluation',
    num_samples: int = 1000,
    device: torch.device = None
):
    """
    Evaluate trained RBM.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for results
        num_samples: Number of samples for evaluation
        device: Device to use
    """
    if device is None:
        device = get_device()
    
    print("="*60)
    print("RBM EVALUATION")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    n_visible = config.get('n_visible', 784)
    n_hidden = config.get('n_hidden', 256)
    
    rbm = BinaryRBM(
        n_visible=n_visible,
        n_hidden=n_hidden,
        use_cuda=(device.type == 'cuda')
    )
    rbm.load_state_dict(checkpoint['model_state_dict'])
    rbm.eval()
    
    print(f"✓ Model loaded: {n_visible} visible, {n_hidden} hidden")
    
    # Load test data
    print("\nLoading test data...")
    test_loader = get_data_loader(
        dataset_name=config.get('dataset', 'mnist'),
        batch_size=100,
        binarize=True,
        train=False
    )
    
    test_data = sample_data_batch(test_loader, num_samples)
    test_data = test_data.view(-1, n_visible).to(device)
    print(f"✓ Loaded {test_data.size(0)} test samples")
    
    # 1. Reconstruction error
    print("\n1. Computing reconstruction error...")
    with torch.no_grad():
        recon = rbm.reconstruct(test_data[:100], k=1)
        recon_error = torch.mean((recon - test_data[:100]) ** 2).item()
    results['reconstruction_error'] = recon_error
    print(f"   Reconstruction error: {recon_error:.6f}")
    
    # 2. Pseudo-likelihood
    print("\n2. Computing pseudo-likelihood...")
    with torch.no_grad():
        pl = rbm.pseudo_likelihood(test_data[:100])
    results['pseudo_likelihood'] = pl
    print(f"   Pseudo-likelihood: {pl:.4f}")
    
    # 3. AIS log-likelihood (slow, use small sample)
    print("\n3. Computing AIS log-likelihood...")
    print("   (This may take a few minutes...)")
    try:
        ll = ais_log_likelihood(
            rbm,
            test_data[:100],
            num_chains=50,
            num_steps=500,
            device=device
        )
        results['ais_log_likelihood'] = ll
        print(f"   AIS log-likelihood: {ll:.4f}")
    except Exception as e:
        print(f"   Warning: AIS failed with error: {e}")
        results['ais_log_likelihood'] = None
    
    # 4. MCMC diagnostics - sample energy trajectory
    print("\n4. Computing MCMC diagnostics...")
    print("   Generating samples and tracking energy...")
    
    energies = []
    v = torch.bernoulli(torch.ones(10, n_visible) * 0.5).to(device)
    
    with torch.no_grad():
        for step in tqdm(range(500), desc="   Sampling"):
            v = rbm.gibbs_step(v)
            if step % 5 == 0:
                energy = rbm.free_energy(v).mean().item()
                energies.append(energy)
    
    energies = np.array(energies)
    
    # Autocorrelation
    autocorr = energy_autocorrelation(energies, max_lag=50)
    results['autocorrelation'] = autocorr.tolist()
    
    # Mixing time
    mix_time = mixing_time(autocorr, threshold=0.1)
    results['mixing_time'] = int(mix_time)
    print(f"   Mixing time: {mix_time} steps")
    
    # Effective sample size
    ess = effective_sample_size(energies.reshape(-1, 1))
    results['effective_sample_size'] = float(ess)
    print(f"   Effective sample size: {ess:.2f}")
    
    # Save results
    results_path = os.path.join(output_dir, 'rbm_evaluation.json')
    save_dict_to_json(results, results_path)
    print(f"\n✓ Results saved to {results_path}")
    
    return results


def evaluate_conv_ebm(
    checkpoint_path: str,
    config_path: str,
    output_dir: str = './evaluation',
    num_samples: int = 1000,
    compute_fid: bool = True,
    compute_is: bool = True,
    compute_lpips: bool = True,
    device: torch.device = None
):
    """
    Evaluate trained Conv-EBM.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Output directory for results
        num_samples: Number of samples for evaluation
        compute_fid: Whether to compute FID
        compute_is: Whether to compute Inception Score
        compute_lpips: Whether to compute LPIPS diversity
        device: Device to use
    """
    if device is None:
        device = get_device()
    
    print("="*60)
    print("CONV-EBM EVALUATION")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Load config
    config = load_config(config_path)
    
    # Load model
    print("\nLoading model...")
    model = build_conv_ebm(
        model_size=config.get('model_size', 'small'),
        input_channels=3,
        spectral_norm=config.get('spectral_norm', True)
    ).to(device)
    
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    print(f"✓ Model loaded: {config.get('model_size', 'small')} ConvEBM")
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples...")
    samples = sample_conv_ebm(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        num_samples=num_samples,
        num_steps=200,
        output_dir=output_dir,
        device=device
    )
    
    # Normalize to [-1, 1] for FID/IS
    samples_normalized = samples * 2 - 1
    
    # Load real data
    print("\nLoading real data...")
    test_loader = get_data_loader(
        dataset_name=config.get('dataset', 'cifar10'),
        batch_size=100,
        augment=False,
        train=False
    )
    
    real_data = sample_data_batch(test_loader, num_samples)
    print(f"✓ Loaded {real_data.size(0)} real samples")
    
    # 1. FID Score
    if compute_fid:
        print("\n1. Computing FID...")
        try:
            fid_calc = FIDCalculator(device=device, dims=2048)
            
            # Compute statistics for real data
            print("   Computing real data statistics...")
            mu_real, sigma_real = fid_calc.compute_statistics(real_data.to(device))
            
            # Compute statistics for generated data
            print("   Computing generated data statistics...")
            mu_gen, sigma_gen = fid_calc.compute_statistics(samples_normalized.to(device))
            
            # Compute FID
            fid = fid_calc.calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
            results['fid'] = float(fid)
            print(f"   FID: {fid:.4f}")
        except Exception as e:
            print(f"   Warning: FID computation failed: {e}")
            results['fid'] = None
    
    # 2. Inception Score
    if compute_is:
        print("\n2. Computing Inception Score...")
        try:
            is_calc = InceptionScoreCalculator(device=device, splits=10)
            # Resize to 299x299
            samples_resized = F.interpolate(
                samples_normalized,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )

            is_mean, is_std = is_calc.calculate_inception_score(
                samples_resized.to(device),
                batch_size=32
            )

            results['inception_score_mean'] = float(is_mean)
            results['inception_score_std'] = float(is_std)
            print(f"   Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        except Exception as e:
            print(f"   Warning: IS computation failed: {e}")
            results['inception_score_mean'] = None
            results['inception_score_std'] = None
    
    # 3. LPIPS Diversity
    if compute_lpips:
        print("\n3. Computing LPIPS diversity...")
        try:
            lpips_calc = LPIPSDiversity(device=device, net='alex')
            samples_lpips = F.interpolate(
                samples_normalized,
                size=(64, 64),      # Recommended input resolution for LPIPS
                mode='bilinear',
                align_corners=False
            )

            diversity = lpips_calc.compute_diversity(
                samples_normalized.to(device),
                num_pairs=min(1000, num_samples * (num_samples - 1) // 2)
            )
            results['lpips_diversity'] = float(diversity)
            print(f"   LPIPS diversity: {diversity:.4f}")
        except Exception as e:
            print(f"   Warning: LPIPS computation failed: {e}")
            results['lpips_diversity'] = None
    
    # 4. Energy statistics
    print("\n4. Computing energy statistics...")
    with torch.no_grad():
        real_energies = model(real_data.to(device)[:500]).cpu().numpy()
        gen_energies = model(samples_normalized.to(device)[:500]).cpu().numpy()
    
    results['real_energy_mean'] = float(real_energies.mean())
    results['real_energy_std'] = float(real_energies.std())
    results['gen_energy_mean'] = float(gen_energies.mean())
    results['gen_energy_std'] = float(gen_energies.std())
    results['energy_gap'] = float(real_energies.mean() - gen_energies.mean())
    
    print(f"   Real energy: {real_energies.mean():.2f} ± {real_energies.std():.2f}")
    print(f"   Generated energy: {gen_energies.mean():.2f} ± {gen_energies.std():.2f}")
    print(f"   Energy gap: {results['energy_gap']:.2f}")
    
    # 5. MCMC diagnostics
    print("\n5. Computing MCMC diagnostics...")
    print("   Sampling trajectory and computing autocorrelation...")

    sampler = LangevinSampler(
        step_size=config.get('langevin_step_size', 0.01),
        noise_scale=config.get('langevin_noise', 0.005),
        clip_grad=config.get('langevin_clip', 0.01),
        device=device
    )

    init_samples = initialize_samples(10, (3, 32, 32), device=device)
    energies = []

    x = init_samples

    for step in tqdm(range(200), desc="   Sampling"):
        # Compute energy WITHOUT grad
        with torch.no_grad():
            energies.append(model(x).mean().item())

        # Langevin sampling WITH grad
        x = sampler.sample(
            energy_fn=model,
            init_samples=x,
            num_steps=1
        )

    
    # Autocorrelation
    autocorr = energy_autocorrelation(energies, max_lag=50)
    results['autocorrelation'] = autocorr.tolist()
    
    # Mixing time
    mix_time = mixing_time(autocorr, threshold=0.1)
    results['mixing_time'] = int(mix_time)
    print(f"   Mixing time: {mix_time} steps")
    
    # Effective sample size
    ess = effective_sample_size(energies.reshape(-1, 1))
    results['effective_sample_size'] = float(ess)
    print(f"   Effective sample size: {ess:.2f}")
    
    # Save results
    results_path = os.path.join(output_dir, 'conv_ebm_evaluation.json')
    save_dict_to_json(results, results_path)
    print(f"\n✓ Results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate EBM')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rbm', 'conv_ebm'],
                       help='Type of model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (required for conv_ebm)')
    parser.add_argument('--output', type=str, default='./evaluation',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples for evaluation')
    parser.add_argument('--metrics', nargs='+', 
                       default=['fid', 'is', 'lpips'],
                       help='Metrics to compute (for conv_ebm)')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    device = get_device(args.gpu_id)
    
    if args.model_type == 'rbm':
        evaluate_rbm(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            num_samples=args.num_samples,
            device=device
        )
    elif args.model_type == 'conv_ebm':
        if args.config is None:
            raise ValueError("--config is required for conv_ebm")
        
        evaluate_conv_ebm(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            output_dir=args.output,
            num_samples=args.num_samples,
            compute_fid='fid' in args.metrics,
            compute_is='is' in args.metrics,
            compute_lpips='lpips' in args.metrics,
            device=device
        )


if __name__ == "__main__":
    main()