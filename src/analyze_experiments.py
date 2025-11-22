"""
Comprehensive analysis script for EBM experiments.

Loads results from multiple experiments and generates comparison plots.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from plotting import (
    plot_cd_comparison, plot_cd_vs_pcd_comparison,
    plot_compute_vs_quality, plot_multiple_autocorrelations,
    plot_ablation_study, plot_training_curves
)
from utils import load_dict_from_json


def load_experiment_results(results_dir: str, pattern: str = "*") -> Dict:
    """
    Load all experiment results from directory.
    
    Args:
        results_dir: Directory containing experiment results
        pattern: Pattern to match experiment folders
        
    Returns:
        Dictionary mapping experiment name to results
    """
    results = {}
    results_path = Path(results_dir)
    
    for exp_dir in results_path.glob(pattern):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # Load metrics
        metrics_file = exp_dir / 'logs' / f'{exp_name}_metrics.json'
        if metrics_file.exists():
            results[exp_name] = {
                'metrics': load_dict_from_json(str(metrics_file))
            }
        
        # Load evaluation results if available
        eval_file_rbm = exp_dir / 'evaluation' / 'rbm_evaluation.json'
        eval_file_conv = exp_dir / 'evaluation' / 'conv_ebm_evaluation.json'
        
        if eval_file_rbm.exists():
            results[exp_name]['evaluation'] = load_dict_from_json(str(eval_file_rbm))
        elif eval_file_conv.exists():
            results[exp_name]['evaluation'] = load_dict_from_json(str(eval_file_conv))
    
    return results


def extract_cd_k_from_name(exp_name: str) -> int:
    """Extract CD-k value from experiment name."""
    if 'cd1' in exp_name.lower():
        return 1
    elif 'cd5' in exp_name.lower():
        return 5
    elif 'cd10' in exp_name.lower():
        return 10
    elif 'cd20' in exp_name.lower():
        return 20
    elif 'cd30' in exp_name.lower():
        return 30
    elif 'cd100' in exp_name.lower():
        return 100
    return None


def is_pcd_experiment(exp_name: str) -> bool:
    """Check if experiment uses PCD."""
    return 'pcd' in exp_name.lower()


def analyze_rbm_experiments(results: Dict, output_dir: str):
    """
    Analyze RBM experiments and generate comparison plots.
    
    Args:
        results: Dictionary of experiment results
        output_dir: Directory to save plots
    """
    print("\n" + "="*60)
    print("ANALYZING RBM EXPERIMENTS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate CD and PCD experiments
    cd_experiments = {}
    pcd_experiments = {}
    
    for exp_name, exp_data in results.items():
        if 'rbm' not in exp_name.lower():
            continue
        
        cd_k = extract_cd_k_from_name(exp_name)
        if cd_k is None:
            continue
        
        if is_pcd_experiment(exp_name):
            pcd_experiments[cd_k] = exp_data
        else:
            cd_experiments[cd_k] = exp_data
    
    if not cd_experiments and not pcd_experiments:
        print("No RBM experiments found!")
        return
    
    # 1. Reconstruction error vs CD-k
    print("\n1. Plotting reconstruction error vs CD-k...")
    if cd_experiments:
        recon_errors_cd = {}
        for k, data in cd_experiments.items():
            if 'metrics' in data:
                # Get final reconstruction error
                epochs = sorted([int(e) for e in data['metrics'].keys()])
                if epochs:
                    final_epoch = str(epochs[-1])
                    recon_errors_cd[k] = data['metrics'][final_epoch].get('reconstruction_error', None)
        
        if recon_errors_cd:
            recon_errors_cd = {k: v for k, v in recon_errors_cd.items() if v is not None}
            plot_cd_comparison(
                recon_errors_cd,
                os.path.join(output_dir, 'rbm_recon_error_vs_cd.png'),
                metric_name='reconstruction_error',
                title='RBM: Reconstruction Error vs CD-k',
                ylabel='Reconstruction Error'
            )
            print(f"   ✓ Saved: rbm_recon_error_vs_cd.png")
    
    # 2. CD vs PCD comparison
    print("\n2. Comparing CD vs PCD...")
    if cd_experiments and pcd_experiments:
        recon_cd = {}
        recon_pcd = {}
        
        for k, data in cd_experiments.items():
            if 'metrics' in data:
                epochs = sorted([int(e) for e in data['metrics'].keys()])
                if epochs:
                    final = str(epochs[-1])
                    recon_cd[k] = data['metrics'][final].get('reconstruction_error', None)
        
        for k, data in pcd_experiments.items():
            if 'metrics' in data:
                epochs = sorted([int(e) for e in data['metrics'].keys()])
                if epochs:
                    final = str(epochs[-1])
                    recon_pcd[k] = data['metrics'][final].get('reconstruction_error', None)
        
        recon_cd = {k: v for k, v in recon_cd.items() if v is not None}
        recon_pcd = {k: v for k, v in recon_pcd.items() if v is not None}
        
        if recon_cd and recon_pcd:
            plot_cd_vs_pcd_comparison(
                recon_cd, recon_pcd,
                os.path.join(output_dir, 'rbm_cd_vs_pcd.png'),
                metric_name='Reconstruction Error',
                title='RBM: CD vs PCD Comparison'
            )
            print(f"   ✓ Saved: rbm_cd_vs_pcd.png")
    
    # 3. Training curves for all experiments
    print("\n3. Plotting training curves...")
    for exp_name, exp_data in results.items():
        if 'rbm' not in exp_name.lower():
            continue
        if 'metrics' not in exp_data:
            continue
        
        metrics_to_plot = {}
        for metric in ['reconstruction_error', 'free_energy']:
            metric_values = {}
            for epoch, vals in exp_data['metrics'].items():
                if metric in vals:
                    metric_values[int(epoch)] = vals[metric]
            if metric_values:
                metrics_to_plot[metric] = metric_values
        
        if metrics_to_plot:
            safe_name = exp_name.replace('/', '_')
            plot_training_curves(
                metrics_to_plot,
                os.path.join(output_dir, f'{safe_name}_training.png'),
                title=f'{exp_name} - Training Curves'
            )
            print(f"   ✓ Saved: {safe_name}_training.png")
    
    # 4. Autocorrelation comparison
    print("\n4. Comparing autocorrelations...")
    autocorr_dict = {}
    for exp_name, exp_data in results.items():
        if 'rbm' not in exp_name.lower():
            continue
        if 'evaluation' in exp_data and 'autocorrelation' in exp_data['evaluation']:
            cd_k = extract_cd_k_from_name(exp_name)
            if cd_k:
                label = f"{'PCD' if is_pcd_experiment(exp_name) else 'CD'}-{cd_k}"
                autocorr_dict[label] = np.array(exp_data['evaluation']['autocorrelation'])
    
    if autocorr_dict:
        plot_multiple_autocorrelations(
            autocorr_dict,
            os.path.join(output_dir, 'rbm_autocorrelations.png'),
            title='RBM: Autocorrelation Comparison',
            max_lag=50
        )
        print(f"   ✓ Saved: rbm_autocorrelations.png")
    
    print("\n✅ RBM analysis complete!")


def analyze_conv_ebm_experiments(results: Dict, output_dir: str):
    """
    Analyze Conv-EBM experiments and generate comparison plots.
    
    Args:
        results: Dictionary of experiment results
        output_dir: Directory to save plots
    """
    print("\n" + "="*60)
    print("ANALYZING CONV-EBM EXPERIMENTS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate CD and PCD experiments
    cd_experiments = {}
    pcd_experiments = {}
    
    for exp_name, exp_data in results.items():
        if 'conv' not in exp_name.lower() and 'cifar' not in exp_name.lower():
            continue
        
        cd_k = extract_cd_k_from_name(exp_name)
        if cd_k is None:
            continue
        
        if is_pcd_experiment(exp_name):
            pcd_experiments[cd_k] = exp_data
        else:
            cd_experiments[cd_k] = exp_data
    
    if not cd_experiments and not pcd_experiments:
        print("No Conv-EBM experiments found!")
        return
    
    # 1. FID vs CD-k
    print("\n1. Plotting FID vs CD-k...")
    if cd_experiments:
        fid_scores = {}
        for k, data in cd_experiments.items():
            if 'evaluation' in data and 'fid' in data['evaluation']:
                fid_scores[k] = data['evaluation']['fid']
        
        if fid_scores:
            plot_cd_comparison(
                fid_scores,
                os.path.join(output_dir, 'conv_fid_vs_cd.png'),
                metric_name='fid',
                title='Conv-EBM: FID vs CD-k',
                ylabel='FID Score (lower is better)'
            )
            print(f"   ✓ Saved: conv_fid_vs_cd.png")
    
    # 2. Inception Score vs CD-k
    print("\n2. Plotting Inception Score vs CD-k...")
    if cd_experiments:
        is_scores = {}
        for k, data in cd_experiments.items():
            if 'evaluation' in data and 'inception_score_mean' in data['evaluation']:
                is_scores[k] = data['evaluation']['inception_score_mean']
        
        if is_scores:
            plot_cd_comparison(
                is_scores,
                os.path.join(output_dir, 'conv_is_vs_cd.png'),
                metric_name='inception_score',
                title='Conv-EBM: Inception Score vs CD-k',
                ylabel='Inception Score (higher is better)'
            )
            print(f"   ✓ Saved: conv_is_vs_cd.png")
    
    # 3. CD vs PCD comparison (FID)
    print("\n3. Comparing CD vs PCD (FID)...")
    if cd_experiments and pcd_experiments:
        fid_cd = {}
        fid_pcd = {}
        
        for k, data in cd_experiments.items():
            if 'evaluation' in data and 'fid' in data['evaluation']:
                fid_cd[k] = data['evaluation']['fid']
        
        for k, data in pcd_experiments.items():
            if 'evaluation' in data and 'fid' in data['evaluation']:
                fid_pcd[k] = data['evaluation']['fid']
        
        if fid_cd and fid_pcd:
            plot_cd_vs_pcd_comparison(
                fid_cd, fid_pcd,
                os.path.join(output_dir, 'conv_cd_vs_pcd_fid.png'),
                metric_name='FID Score',
                title='Conv-EBM: CD vs PCD (FID)'
            )
            print(f"   ✓ Saved: conv_cd_vs_pcd_fid.png")
    
    # 4. Energy gap vs CD-k
    print("\n4. Plotting energy gap vs CD-k...")
    energy_gaps = {}
    for exp_name, exp_data in results.items():
        if 'conv' not in exp_name.lower() and 'cifar' not in exp_name.lower():
            continue
        
        cd_k = extract_cd_k_from_name(exp_name)
        if cd_k and not is_pcd_experiment(exp_name):
            if 'evaluation' in exp_data and 'energy_gap' in exp_data['evaluation']:
                energy_gaps[cd_k] = abs(exp_data['evaluation']['energy_gap'])
    
    if energy_gaps:
        plot_cd_comparison(
            energy_gaps,
            os.path.join(output_dir, 'conv_energy_gap_vs_cd.png'),
            metric_name='energy_gap',
            title='Conv-EBM: Energy Gap vs CD-k',
            ylabel='Energy Gap (|E_real - E_gen|)'
        )
        print(f"   ✓ Saved: conv_energy_gap_vs_cd.png")
    
    # 5. Training curves
    print("\n5. Plotting training curves...")
    for exp_name, exp_data in results.items():
        if 'conv' not in exp_name.lower() and 'cifar' not in exp_name.lower():
            continue
        if 'metrics' not in exp_data:
            continue
        
        metrics_to_plot = {}
        for metric in ['loss', 'pos_energy', 'neg_energy', 'energy_gap']:
            metric_values = {}
            for epoch, vals in exp_data['metrics'].items():
                if metric in vals:
                    metric_values[int(epoch)] = vals[metric]
            if metric_values:
                metrics_to_plot[metric] = metric_values
        
        if metrics_to_plot:
            safe_name = exp_name.replace('/', '_')
            plot_training_curves(
                metrics_to_plot,
                os.path.join(output_dir, f'{safe_name}_training.png'),
                title=f'{exp_name} - Training Curves'
            )
            print(f"   ✓ Saved: {safe_name}_training.png")
    
    # 6. Compute vs quality tradeoff
    print("\n6. Plotting compute vs quality tradeoff...")
    if cd_experiments:
        cd_vals = sorted(cd_experiments.keys())
        
        # Estimate training times (you can replace with actual times)
        training_times = [k * 0.5 for k in cd_vals]  # Rough estimate
        
        fid_scores_list = []
        for k in cd_vals:
            if 'evaluation' in cd_experiments[k] and 'fid' in cd_experiments[k]['evaluation']:
                fid_scores_list.append(cd_experiments[k]['evaluation']['fid'])
            else:
                fid_scores_list.append(None)
        
        # Filter out None values
        valid_indices = [i for i, fid in enumerate(fid_scores_list) if fid is not None]
        if valid_indices:
            cd_vals_filtered = [cd_vals[i] for i in valid_indices]
            times_filtered = [training_times[i] for i in valid_indices]
            fid_filtered = [fid_scores_list[i] for i in valid_indices]
            
            plot_compute_vs_quality(
                cd_vals_filtered,
                times_filtered,
                fid_filtered,
                os.path.join(output_dir, 'conv_compute_vs_quality.png'),
                quality_metric='FID',
                better_lower=True
            )
            print(f"   ✓ Saved: conv_compute_vs_quality.png")
    
    print("\n✅ Conv-EBM analysis complete!")


def analyze_ablation_studies(results: Dict, output_dir: str):
    """
    Analyze ablation study results.
    
    Args:
        results: Dictionary of experiment results
        output_dir: Directory to save plots
    """
    print("\n" + "="*60)
    print("ANALYZING ABLATION STUDIES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for ablation experiments
    ablation_experiments = {
        'step_size': [],
        'noise': [],
        'buffer': []
    }
    
    for exp_name, exp_data in results.items():
        if 'ablation' not in exp_name.lower():
            continue
        
        if 'step' in exp_name.lower():
            ablation_experiments['step_size'].append((exp_name, exp_data))
        elif 'noise' in exp_name.lower():
            ablation_experiments['noise'].append((exp_name, exp_data))
        elif 'buffer' in exp_name.lower():
            ablation_experiments['buffer'].append((exp_name, exp_data))
    
    # Plot each ablation study
    for study_type, experiments in ablation_experiments.items():
        if not experiments:
            continue
        
        print(f"\nAnalyzing {study_type} ablation...")
        # This would require extracting parameter values from experiment names
        # or config files - placeholder for now
        print(f"   Found {len(experiments)} experiments")
    
    print("\n✅ Ablation analysis complete!")


def generate_summary_report(results: Dict, output_dir: str):
    """Generate text summary report."""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EBM EXPERIMENTS SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total experiments: {len(results)}\n\n")
        
        # RBM experiments
        rbm_exps = [name for name in results.keys() if 'rbm' in name.lower()]
        f.write(f"RBM Experiments: {len(rbm_exps)}\n")
        for name in sorted(rbm_exps):
            f.write(f"  - {name}\n")
        f.write("\n")
        
        # Conv-EBM experiments
        conv_exps = [name for name in results.keys() 
                    if 'conv' in name.lower() or 'cifar' in name.lower()]
        f.write(f"Conv-EBM Experiments: {len(conv_exps)}\n")
        for name in sorted(conv_exps):
            f.write(f"  - {name}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze EBM experiments')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                       help='Directory to save analysis plots')
    parser.add_argument('--pattern', type=str, default='*',
                       help='Pattern to match experiment folders')
    
    args = parser.parse_args()
    
    print("Loading experiment results...")
    results = load_experiment_results(args.results_dir, args.pattern)
    print(f"Loaded {len(results)} experiments")
    
    if not results:
        print("No experiments found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze RBM experiments
    analyze_rbm_experiments(results, os.path.join(args.output_dir, 'rbm'))
    
    # Analyze Conv-EBM experiments
    analyze_conv_ebm_experiments(results, os.path.join(args.output_dir, 'conv_ebm'))
    
    # Analyze ablation studies
    analyze_ablation_studies(results, os.path.join(args.output_dir, 'ablations'))
    
    # Generate summary report
    generate_summary_report(results, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"All plots saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()