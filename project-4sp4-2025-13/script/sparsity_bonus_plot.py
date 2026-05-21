#!/usr/bin/env python3
"""
Plot comparing two sparsity implementations: Magnitude-based vs SparseGPT
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to the repository root (parent of this script dir)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'nn_cpu.json'
DEFAULT_PLOT_PATH = REPO_ROOT / 'plots' / 'sparsity_comparison.png'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparing two sparsity implementations: Magnitude-based vs SparseGPT",
    )
    parser.add_argument(
        "log_path",
        type=Path,
        nargs="?",
        default=DEFAULT_LOG_PATH,
        help="Path to benchmark JSON (default: logs/nn_cpu.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Output path for the plot image (default: plots/sparsity_comparison.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.log_path.exists():
        raise FileNotFoundError(f"Benchmark log not found: {args.log_path}")
    # Read the benchmark data
    with args.log_path.open('r') as f:
        data = json.load(f)

    # Extract the relevant benchmark entries
    magnitude_data_90 = None
    magnitude_data_60 = None
    sparsegpt_data = None

    for benchmark in data['benchmarks']:
        if benchmark['run_type'] == 'aggregate' and benchmark['aggregate_name'] == 'mean':
            if 'BM_SPARSITY_COMPARISON_MAGNITUDE/32/32/90' in benchmark['name']:
                magnitude_data_90 = benchmark
            elif 'BM_SPARSITY_COMPARISON_MAGNITUDE/32/32/60' in benchmark['name']:
                magnitude_data_60 = benchmark
            elif 'BM_SPARSITY_COMPARISON_SPARSEGPT/32/32' in benchmark['name']:
                sparsegpt_data = benchmark

    # Ensure required benchmark entries exist
    if magnitude_data_90 is None:
        raise ValueError("Missing magnitude 90% benchmark entry (BM_SPARSITY_COMPARISON_MAGNITUDE/32/32/90)")
    if magnitude_data_60 is None:
        raise ValueError("Missing magnitude 60% benchmark entry (BM_SPARSITY_COMPARISON_MAGNITUDE/32/32/60)")
    if sparsegpt_data is None:
        raise ValueError("Missing SparseGPT benchmark entry (BM_SPARSITY_COMPARISON_SPARSEGPT/32/32)")

    # Prepare data for plotting
    sparsity_methods = ["Magnitude-based_90%",'Magnitude-based_60%', 'SparseGPT_90%']
    accuracies = [magnitude_data_90['Accuracy'], magnitude_data_60['Accuracy'], sparsegpt_data['Accuracy']]

    # Convert runtime from microseconds to milliseconds
    magnitudes_runtime_ms_90 = magnitude_data_90['real_time'] / 1000
    magnitudes_runtime_ms_60 = magnitude_data_60['real_time'] / 1000
    sparsegpt_runtime_ms = sparsegpt_data['real_time'] / 1000
    runtimes = [magnitudes_runtime_ms_90, magnitudes_runtime_ms_60, sparsegpt_runtime_ms]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Type of Sparsity vs Accuracy
    colors = ['#DC143C', '#DC143C', '#1E90FF']  # Red for magnitude-based, blue for SparseGPT
    ax1.bar(sparsity_methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sparsity Method', fontsize=12, fontweight='bold')
    ax1.set_title('Sparsity Method vs Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (method, acc) in enumerate(zip(sparsity_methods, accuracies)):
        ax1.text(i, acc + 1, f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 2: Type of Sparsity vs Runtime
    ax2.bar(sparsity_methods, runtimes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sparsity Method', fontsize=12, fontweight='bold')
    ax2.set_title('Sparsity Method vs Runtime', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (method, runtime) in enumerate(zip(sparsity_methods, runtimes)):
        ax2.text(i, runtime + 10, f'{runtime:.2f} ms', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

    # Print summary statistics
    print("SPARSITY COMPARISON SUMMARY")
    print("="*60)
    print(f"\nMagnitude-based Sparsity (90%):")
    print(f"  Accuracy:  {magnitude_data_90['Accuracy']:.2f}%")
    print(f"  Runtime:   {magnitudes_runtime_ms_90:.2f} ms")
    print(f"\nMagnitude-based Sparsity (60%):")
    print(f"  Accuracy:  {magnitude_data_60['Accuracy']:.2f}%")
    print(f"  Runtime:   {magnitudes_runtime_ms_60:.2f} ms")
    print(f"\nSparseGPT:")
    print(f"  Accuracy:  {sparsegpt_data['Accuracy']:.2f}%")
    print(f"  Runtime:   {sparsegpt_runtime_ms:.2f} ms")
    print(f"\nDifferences:")
    print(f"  Accuracy gain (SparseGPT vs Magnitude 90%): {sparsegpt_data['Accuracy'] - magnitude_data_90['Accuracy']:.2f}%")
    print(f"  Accuracy gain (SparseGPT vs Magnitude 60%): {sparsegpt_data['Accuracy'] - magnitude_data_60['Accuracy']:.2f}%")
    print(f"  Runtime overhead (SparseGPT vs Magnitude 60%): {sparsegpt_runtime_ms - magnitudes_runtime_ms_60:.2f} ms ({((sparsegpt_runtime_ms - magnitudes_runtime_ms_60) / magnitudes_runtime_ms_60 * 100):.2f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
