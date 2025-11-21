"""
Master script to run all EBM experiments sequentially.

Useful for running complete experiment sets on Kaggle.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from train_rbm import train_rbm
from train_conv_ebm import train_conv_ebm
from utils import load_config
from sample import sample_rbm, sample_conv_ebm
from evaluate import evaluate_rbm, evaluate_conv_ebm
from analyze_experiments import (
    load_experiment_results, analyze_rbm_experiments,
    analyze_conv_ebm_experiments, generate_summary_report
)


def run_experiment_set(
    config_files: list,
    model_type: str,
    output_base: str = './results',
    run_evaluation: bool = True,
    run_sampling: bool = True
):
    """
    Run a set of experiments sequentially.
    
    Args:
        config_files: List of config file paths
        model_type: 'rbm' or 'conv_ebm'
        output_base: Base output directory
        run_evaluation: Whether to run evaluation after training
        run_sampling: Whether to generate samples after training
    """
    print("\n" + "="*80)
    print(f"RUNNING {len(config_files)} {model_type.upper()} EXPERIMENTS")
    print("="*80 + "\n")
    
    results = []
    
    for idx, config_file in enumerate(config_files, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {idx}/{len(config_files)}: {Path(config_file).name}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # Load config
            config = load_config(config_file)
            
            # Ensure output directory is set
            if 'output_dir' not in config:
                config['output_dir'] = output_base
            
            # Train model
            print("\n--- TRAINING ---")
            if model_type == 'rbm':
                train_rbm(config)
            elif model_type == 'conv_ebm':
                train_conv_ebm(config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Find checkpoint
            exp_name = config['exp_name']
            checkpoint_dir = Path(output_base) / f"{exp_name}_*" / "checkpoints"
            checkpoint_paths = list(Path(output_base).glob(f"{exp_name}_*/checkpoints/"))
            
            if checkpoint_paths:
                checkpoint_dir = checkpoint_paths[0]
                best_checkpoint = checkpoint_dir / f"{model_type}_best.pt"
                
                if best_checkpoint.exists():
                    # Generate samples
                    if run_sampling:
                        print("\n--- GENERATING SAMPLES ---")
                        sample_dir = checkpoint_dir.parent / 'evaluation'
                        
                        if model_type == 'rbm':
                            sample_rbm(
                                str(best_checkpoint),
                                num_samples=100,
                                num_steps=1000,
                                output_dir=str(sample_dir)
                            )
                        else:
                            sample_conv_ebm(
                                str(best_checkpoint),
                                config_file,
                                num_samples=100,
                                num_steps=200,
                                output_dir=str(sample_dir)
                            )
                    
                    # Run evaluation
                    if run_evaluation:
                        print("\n--- EVALUATION ---")
                        eval_dir = checkpoint_dir.parent / 'evaluation'
                        
                        if model_type == 'rbm':
                            evaluate_rbm(
                                str(best_checkpoint),
                                output_dir=str(eval_dir),
                                num_samples=1000
                            )
                        else:
                            evaluate_conv_ebm(
                                str(best_checkpoint),
                                config_file,
                                output_dir=str(eval_dir),
                                num_samples=1000
                            )
            
            elapsed = time.time() - start_time
            print(f"\n✓ Experiment completed in {elapsed/3600:.2f} hours")
            
            results.append({
                'config': config_file,
                'status': 'success',
                'time': elapsed
            })
            
        except Exception as e:
            print(f"\n✗ Experiment failed with error: {e}")
            results.append({
                'config': config_file,
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SET SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    total_time = sum(r.get('time', 0) for r in results)
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/3600:.2f} hours")
    
    return results


def run_rbm_experiments(
    quick_mode: bool = False,
    output_dir: str = './results'
):
    """
    Run all RBM experiments.
    
    Args:
        quick_mode: If True, run only CD-1 and CD-5 for quick testing
        output_dir: Output directory
    """
    if quick_mode:
        configs = [
            'configs/rbm_mnist_cd1.yaml',
            'configs/rbm_mnist_cd5.yaml'
        ]
    else:
        configs = [
            'configs/rbm_mnist_cd1.yaml',
            'configs/rbm_mnist_cd5.yaml',
            'configs/rbm_mnist_cd10.yaml',
            'configs/rbm_mnist_cd20.yaml',
            'configs/rbm_mnist_pcd1.yaml',
            'configs/rbm_mnist_pcd5.yaml',
            'configs/rbm_mnist_pcd10.yaml'
        ]
    
    # Filter existing configs
    configs = [c for c in configs if os.path.exists(c)]
    
    return run_experiment_set(
        configs,
        model_type='rbm',
        output_base=output_dir,
        run_evaluation=True,
        run_sampling=True
    )


def run_conv_ebm_experiments(
    quick_mode: bool = False,
    output_dir: str = './results'
):
    """
    Run all Conv-EBM experiments.
    
    Args:
        quick_mode: If True, run only tiny model for quick testing
        output_dir: Output directory
    """
    if quick_mode:
        configs = [
            'configs/conv_cifar_tiny_cd5.yaml'
        ]
    else:
        configs = [
            'configs/conv_cifar_cd5.yaml',
            'configs/conv_cifar_cd10.yaml',
            'configs/conv_cifar_cd20.yaml',
            'configs/conv_cifar_pcd5.yaml',
            'configs/conv_cifar_pcd10.yaml'
        ]
    
    # Filter existing configs
    configs = [c for c in configs if os.path.exists(c)]
    
    return run_experiment_set(
        configs,
        model_type='conv_ebm',
        output_base=output_dir,
        run_evaluation=True,
        run_sampling=True
    )


def run_all_experiments(
    quick_mode: bool = False,
    rbm_only: bool = False,
    conv_only: bool = False,
    output_dir: str = './results',
    analysis_dir: str = './analysis'
):
    """
    Run all experiments and generate analysis.
    
    Args:
        quick_mode: Quick mode for testing
        rbm_only: Run only RBM experiments
        conv_only: Run only Conv-EBM experiments
        output_dir: Output directory for results
        analysis_dir: Output directory for analysis
    """
    print("\n" + "="*80)
    print("RUNNING COMPLETE EBM EXPERIMENT SUITE")
    print("="*80)
    print(f"\nMode: {'QUICK TEST' if quick_mode else 'FULL EXPERIMENTS'}")
    print(f"Output directory: {output_dir}")
    print(f"Analysis directory: {analysis_dir}\n")
    
    overall_start = time.time()
    
    # Run RBM experiments
    if not conv_only:
        print("\n" + "█"*80)
        print("PHASE 1: RBM EXPERIMENTS")
        print("█"*80 + "\n")
        run_rbm_experiments(quick_mode, output_dir)
    
    # Run Conv-EBM experiments
    if not rbm_only:
        print("\n" + "█"*80)
        print("PHASE 2: CONV-EBM EXPERIMENTS")
        print("█"*80 + "\n")
        run_conv_ebm_experiments(quick_mode, output_dir)
    
    # Generate analysis
    print("\n" + "█"*80)
    print("PHASE 3: ANALYSIS AND VISUALIZATION")
    print("█"*80 + "\n")
    
    print("Loading all experiment results...")
    results = load_experiment_results(output_dir)
    
    os.makedirs(analysis_dir, exist_ok=True)
    
    if not rbm_only:
        analyze_rbm_experiments(results, os.path.join(analysis_dir, 'rbm'))
    
    if not conv_only:
        analyze_conv_ebm_experiments(results, os.path.join(analysis_dir, 'conv_ebm'))
    
    generate_summary_report(results, analysis_dir)
    
    # Final summary
    total_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nTotal runtime: {total_time/3600:.2f} hours")
    print(f"Results: {output_dir}")
    print(f"Analysis: {analysis_dir}")
    print("\nNext steps:")
    print("  1. Review generated plots in the analysis directory")
    print("  2. Check summary_report.txt for overview")
    print("  3. Use analyze_experiments.py for custom analysis")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete EBM experiment suite'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: run minimal experiments for testing')
    parser.add_argument('--rbm-only', action='store_true',
                       help='Run only RBM experiments')
    parser.add_argument('--conv-only', action='store_true',
                       help='Run only Conv-EBM experiments')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--analysis-dir', type=str, default='./analysis',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    run_all_experiments(
        quick_mode=args.quick,
        rbm_only=args.rbm_only,
        conv_only=args.conv_only,
        output_dir=args.output_dir,
        analysis_dir=args.analysis_dir
    )


if __name__ == "__main__":
    main()