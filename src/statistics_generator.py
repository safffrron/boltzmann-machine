
import pandas as pd
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from utils import load_config, get_device
from sample import sample_rbm, sample_conv_ebm
from evaluate import evaluate_rbm, evaluate_conv_ebm
from analyze_experiments import (
    load_experiment_results, 
    analyze_rbm_experiments,
    analyze_conv_ebm_experiments,
    generate_summary_report
)

results_dir = './results'

# List all experiment folders
experiments = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
print(f"Found {len(experiments)} experiments:")
for exp in sorted(experiments):
    print(f"  - {exp}")

# Check what's in each experiment
print("\nExperiment contents:")
for exp in sorted(experiments)[:3]:  # Show first 3 as example
    exp_path = os.path.join(results_dir, exp)
    contents = os.listdir(exp_path)
    print(f"\n{exp}:")
    print(f"  {contents}")

results = load_experiment_results(results_dir)
print(f"Loaded {len(results)} experiments\n")


# summary_data = []

# for exp_name, exp_data in results.items():
#     row = {'Experiment': exp_name}
    
#     # Extract CD-k value
#     if 'cd1' in exp_name.lower():
#         row['CD-k'] = 1
#     elif 'cd5' in exp_name.lower():
#         row['CD-k'] = 5
#     elif 'cd10' in exp_name.lower():
#         row['CD-k'] = 10
#     elif 'cd20' in exp_name.lower():
#         row['CD-k'] = 20
#     else:
#         row['CD-k'] = 'Unknown'
    
#     # Method
#     row['Method'] = 'PCD' if 'pcd' in exp_name.lower() else 'CD'
    
#     # Model type
#     if 'rbm' in exp_name.lower():
#         row['Model'] = 'RBM'
#     elif 'conv' in exp_name.lower():
#         row['Model'] = 'Conv-EBM'
#     else:
#         row['Model'] = 'Unknown'
    
#     # Extract final metrics
#     if 'evaluation' in exp_data:
#         eval_data = exp_data['evaluation']
        
#         # RBM metrics
#         if 'reconstruction_error' in eval_data:
#             row['Recon Error'] = f"{eval_data['reconstruction_error']:.4f}"
#         if 'mixing_time' in eval_data:
#             row['Mixing Time'] = eval_data['mixing_time']
#         if 'effective_sample_size' in eval_data:
#             row['ESS'] = f"{eval_data['effective_sample_size']:.1f}"
        
#         # Conv-EBM metrics
#         if 'fid' in eval_data:
#             row['FID'] = f"{eval_data['fid']:.2f}"
#         if 'inception_score_mean' in eval_data:
#             row['IS'] = f"{eval_data['inception_score_mean']:.2f}"
#         if 'lpips_diversity' in eval_data:
#             row['LPIPS'] = f"{eval_data['lpips_diversity']:.3f}"
    
#     # Training time from metrics
#     if 'metrics' in exp_data:
#         epochs = sorted([int(k) for k in exp_data['metrics'].keys()])
#         if epochs:
#             final_epoch = str(epochs[-1])
#             if 'epoch_time' in exp_data['metrics'][final_epoch]:
#                 total_time = sum(
#                     exp_data['metrics'][str(e)].get('epoch_time', 0) 
#                     for e in epochs
#                 ) / 3600  # Convert to hours
#                 row['Time (h)'] = f"{total_time:.1f}"
    
#     summary_data.append(row)

# # Create DataFrame
# df = pd.DataFrame(summary_data)

# # Sort by Model and CD-k
# df = df.sort_values(['Model', 'CD-k'])

# print("\n" + "="*60)
# print("EXPERIMENT SUMMARY TABLE")
# print("="*60)
# print(df.to_string(index=False))

# # Save to CSV
# csv_path = os.path.join(analysis_dir, 'experiment_summary.csv')
# df.to_csv(csv_path, index=False)
# print(f"\n✓ Summary table saved to: {csv_path}")