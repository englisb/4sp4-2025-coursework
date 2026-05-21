#!/usr/bin/env python3
"""
Heat Equation Performance Analysis
Single file for all benchmark visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def load_benchmark_data(filepath):
    """Load and parse JSON benchmark data"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        # print(f"Error loading {filepath}: {e}")
        return None


def extract_timing_results(benchmarks, pattern):
    """Extract timing measurements for matching benchmark patterns"""
    results = {}

    for bench in benchmarks:
        name = bench.get('name', '')
        if not name.startswith(pattern):
            continue

        parts = name.split('/')
        if len(parts) < 2:
            continue

        try:
            grid_size = int(parts[1])

            # Handle different time units properly
            time_unit = bench.get('time_unit', 's')
            real_time = bench.get('real_time', 0)

            if time_unit == 'ms':
                time_seconds = real_time / 1000.0
            elif time_unit == 'us':
                time_seconds = real_time / 1000000.0
            elif time_unit == 'ns':
                time_seconds = real_time / 1000000000.0
            else:
                time_seconds = real_time

            # Keep minimum time for each grid size
            if grid_size not in results or time_seconds < results[grid_size]:
                results[grid_size] = time_seconds

        except (ValueError, IndexError):
            continue

    return results


def generate_performance_comparison():
    """Create dual-panel performance comparison with cache analysis"""
    data = load_benchmark_data("logs/lab02-part2.json")
    if not data:
        return

    benchmarks = data.get('benchmarks', [])

    # Extract performance data
    normal_times = extract_timing_results(benchmarks, 'BM_STENCIL/')
    tiled_times = extract_timing_results(benchmarks, 'BM_STENCIL_TILED/')

    # Find overlapping grid sizes
    common_sizes = sorted(set(normal_times.keys()) & set(tiled_times.keys()))
    if not common_sizes:
        # print("No overlapping grid sizes found")
        return

    # Convert to arrays for plotting
    sizes = np.array(common_sizes)
    normal_ms = np.array([normal_times[s] * 1000 for s in common_sizes])
    tiled_ms = np.array([tiled_times[s] * 1000 for s in common_sizes])

    # Calculate speedup relative to L1 baseline (smallest size)
    baseline_time = normal_times[common_sizes[0]]
    normal_speedup = baseline_time / \
        np.array([normal_times[s] for s in common_sizes])
    tiled_speedup = baseline_time / \
        np.array([tiled_times[s] for s in common_sizes])

    # Create dual subplot figure
    fig, (ax_time, ax_speedup) = plt.subplots(2, 1, figsize=(12, 10))

    # Define cache boundary approximations
    l2_boundary = 512
    l3_boundary = 2048

    # Upper panel: Execution times
    ax_time.loglog(sizes, normal_ms, 'o-', label='Normal',
                   color='#1f77b4', linewidth=2.5, markersize=8)
    ax_time.loglog(sizes, tiled_ms, 's-', label='Tiled',
                   color='#ff7f0e', linewidth=2.5, markersize=8)

    # Highlight L1 baseline point
    ax_time.scatter([sizes[0]], [normal_ms[0]], color='#ffcc00', s=120,
                    zorder=5, label='L1 baseline', edgecolor='black', linewidth=1)

    # Add cache region shading
    ax_time.axvspan(0, 128, color='#fff2cc', alpha=0.6, zorder=0)
    ax_time.axvspan(128, l2_boundary, color='#d4edda', alpha=0.4, zorder=0)
    ax_time.axvspan(l2_boundary, l3_boundary,
                    color='#cce7ff', alpha=0.4, zorder=0)
    ax_time.axvspan(l3_boundary, max(sizes)*1.5,
                    color='#f0f0f0', alpha=0.4, zorder=0)

    # Cache level annotations
    ax_time.text(64, max(normal_ms)*0.7, 'L1', ha='center', fontsize=14,
                 color='#b8860b', fontweight='bold')
    ax_time.text(320, max(normal_ms)*0.7, 'L2', ha='center', fontsize=14,
                 color='#228b22', fontweight='bold')
    ax_time.text(1280, max(normal_ms)*0.7, 'L3', ha='center', fontsize=14,
                 color='#4682b4', fontweight='bold')
    ax_time.text(l3_boundary*1.5, max(normal_ms)*0.7, 'Memory', ha='center',
                 fontsize=14, color='#808080', fontweight='bold')

    # Cache boundary lines
    ax_time.axvline(l2_boundary, color='green',
                    linestyle='--', alpha=0.7, linewidth=2)
    ax_time.axvline(l3_boundary, color='blue',
                    linestyle='--', alpha=0.7, linewidth=2)

    # Grid size annotations
    for i, size in enumerate(sizes):
        ax_time.text(size, min(normal_ms[i], tiled_ms[i])*0.7, f'n={size}',
                     ha='center', va='top', fontsize=9, rotation=90, alpha=0.7)

    ax_time.set_xlabel('Grid Size (n)', fontsize=12)
    ax_time.set_ylabel('Time (ms)', fontsize=12)
    ax_time.set_title('Heat Equation — Time & Speedup vs Grid Size',
                      fontsize=14, fontweight='bold')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=11)
    ax_time.set_xlim(sizes[0]*0.8, sizes[-1]*1.2)

    # Lower panel: Speedup analysis
    ax_speedup.semilogx(sizes, tiled_speedup, 's-',
                        color='#ff7f0e', linewidth=2.5, markersize=8)

    # L1 baseline point
    ax_speedup.scatter([sizes[0]], tiled_speedup[0], color='#ffcc00', s=120,
                       zorder=5, edgecolor='black', linewidth=1)

    # Cache region shading (matching upper panel)
    ax_speedup.axvspan(0, 128, color='#fff2cc', alpha=0.6, zorder=0)
    ax_speedup.axvspan(128, l2_boundary, color='#d4edda', alpha=0.4, zorder=0)
    ax_speedup.axvspan(l2_boundary, l3_boundary,
                       color='#cce7ff', alpha=0.4, zorder=0)
    ax_speedup.axvspan(l3_boundary, max(sizes)*1.5,
                       color='#f0f0f0', alpha=0.4, zorder=0)

    # Cache annotations
    ax_speedup.text(320, max(tiled_speedup)*0.8, 'L2', ha='center',
                    fontsize=14, color='#228b22', fontweight='bold')
    ax_speedup.text(1280, max(tiled_speedup)*0.8, 'L3', ha='center',
                    fontsize=14, color='#4682b4', fontweight='bold')
    ax_speedup.text(l3_boundary*1.5, max(tiled_speedup)*0.8, 'Memory',
                    ha='center', fontsize=14, color='#808080', fontweight='bold')

    # Boundary lines
    ax_speedup.axvline(l2_boundary, color='green',
                       linestyle='--', alpha=0.7, linewidth=2)
    ax_speedup.axvline(l3_boundary, color='blue',
                       linestyle='--', alpha=0.7, linewidth=2)
    ax_speedup.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax_speedup.set_xlabel('Grid Size (n)', fontsize=12)
    ax_speedup.set_ylabel('Speedup vs L1', fontsize=12)
    ax_speedup.grid(True, alpha=0.3)
    ax_speedup.set_xlim(sizes[0]*0.8, sizes[-1]*1.2)

    # Configure x-axis with powers of 2
    power_ticks = [2**i for i in range(6, 13) if 2**i <= max(sizes)*1.2]
    ax_time.set_xticks(power_ticks)
    ax_time.set_xticklabels([f'2^{int(np.log2(t))}' for t in power_ticks])
    ax_speedup.set_xticks(power_ticks)
    ax_speedup.set_xticklabels([f'2^{int(np.log2(t))}' for t in power_ticks])

    plt.tight_layout()

    # Save output
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/plot2_speedup.png', dpi=150, bbox_inches='tight')
    plt.close()
    # print("Performance comparison saved to plots/plot2_speedup.png")


def generate_speedup_analysis():
    """Create focused speedup comparison plot"""
    data = load_benchmark_data("logs/lab02-part2.json")
    if not data:
        return

    benchmarks = data.get('benchmarks', [])

    # Extract timing data
    normal_times = extract_timing_results(benchmarks, 'BM_STENCIL/')
    tiled_times = extract_timing_results(benchmarks, 'BM_STENCIL_TILED/')

    # Find common grid sizes
    common_sizes = sorted(set(normal_times.keys()) & set(tiled_times.keys()))
    if not common_sizes:
        return

    # Calculate speedup ratios
    grid_sizes = []
    speedups = []

    for size in common_sizes:
        speedup = normal_times[size] / tiled_times[size]
        grid_sizes.append(size)
        speedups.append(speedup)

    # Create the speedup plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot speedup curve
    ax.plot(grid_sizes, speedups, 'o-',
            color='#1f77b4', linewidth=3, markersize=8)

    # Add speedup value annotations
    for size, speedup in zip(grid_sizes, speedups):
        ax.annotate(f'{speedup:.2f}x', (size, speedup),
                    textcoords="offset points", xytext=(0, 15), ha='center',
                    fontsize=12, fontweight='bold')

    # Configure plot styling
    ax.set_xlabel('Grid Size n', fontsize=14)
    ax.set_ylabel('Speedup (BM_STENCIL / BM_STENCIL_TILED)', fontsize=14)
    ax.set_title('Speedup vs Grid Size | steps=500 | tile=32',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.4)

    # Set appropriate axis limits
    y_margin = (max(speedups) - min(speedups)) * 0.1
    ax.set_ylim(min(speedups) - y_margin, max(speedups) + y_margin)
    ax.set_xlim(min(grid_sizes) * 0.95, max(grid_sizes) * 1.05)

    # Configure x-axis ticks
    ax.set_xticks(grid_sizes)
    ax.set_xticklabels([str(size) for size in grid_sizes])

    # Styling
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    # Save output
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/plot3_tile_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()

    # print("Speedup analysis saved to plots/plot3_tile_sweep.png")

    # Print summary results
    # print("\nSpeedup Analysis Results:")
    # print("=" * 40)
    # for size, speedup in zip(grid_sizes, speedups):
    #     print(f"Grid {size}×{size}: {speedup:.2f}x speedup")


def plot_cache_hierarchy_analysis(json_path):
    """
    Plot cache hierarchy analysis showing performance transitions
    at L1, L2, and L3 cache boundaries.
    """
    # Load benchmark data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract data from benchmark results
    sizes = []
    times = []
    bandwidths = []

    for benchmark in data['benchmarks']:
        if 'BM_COPY' in benchmark['name'] and 'median' in benchmark['name']:
            # Extract array size from benchmark name
            # Format: "BM_COPY/1024/1/iterations:1/repeats:3_median"
            parts = benchmark['name'].split('/')
            size = int(parts[1])
            sizes.append(size)

            # Get median time in microseconds
            times.append(benchmark['cpu_time'])

            # Calculate bandwidth (bytes per second)
            bytes_per_op = size * 8  # 8 bytes per double
            # Convert microseconds to seconds, handle zero time
            cpu_time_us = benchmark['cpu_time']
            if cpu_time_us > 0:
                bandwidth = bytes_per_op / (cpu_time_us * 1e-6)
            else:
                bandwidth = 0  # Handle zero time case
            bandwidths.append(bandwidth)

    # Convert to numpy arrays for easier manipulation
    sizes = np.array(sizes)
    times = np.array(times)
    bandwidths = np.array(bandwidths)

    # Sort by size
    sort_idx = np.argsort(sizes)
    sizes = sizes[sort_idx]
    times = times[sort_idx]
    bandwidths = bandwidths[sort_idx]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Time per operation vs Array Size
    ax1.loglog(sizes, times, 'b-o', markersize=4, linewidth=2)
    ax1.set_xlabel('Array Size (elements)')
    ax1.set_ylabel('Time per Operation (μs)')
    ax1.set_title('Cache Hierarchy Analysis: Time vs Array Size')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bandwidth vs Array Size
    ax2.loglog(sizes, bandwidths, 'r-s', markersize=4, linewidth=2)
    ax2.set_xlabel('Array Size (elements)')
    ax2.set_ylabel('Bandwidth (bytes/sec)')
    ax2.set_title('Cache Hierarchy Analysis: Bandwidth vs Array Size')
    ax2.grid(True, alpha=0.3)

    # Estimate cache boundaries by finding performance cliffs
    # Look for significant increases in time or decreases in bandwidth
    # Handle zero values in time ratios
    time_ratios = np.zeros(len(times) - 1)
    bandwidth_ratios = np.zeros(len(bandwidths) - 1)

    for i in range(len(times) - 1):
        if times[i] > 0 and times[i+1] > 0:
            time_ratios[i] = times[i+1] / times[i]
        else:
            time_ratios[i] = 1.0  # No change if zero time

    for i in range(len(bandwidths) - 1):
        if bandwidths[i] > 0 and bandwidths[i+1] > 0:
            bandwidth_ratios[i] = bandwidths[i] / bandwidths[i+1]
        else:
            bandwidth_ratios[i] = 1.0  # No change if zero bandwidth

    # Find significant performance drops (threshold > 1.5x)
    time_cliffs = np.where(time_ratios > 1.5)[0]
    bandwidth_cliffs = np.where(bandwidth_ratios > 1.5)[0]

    # Mark potential cache boundaries
    for cliff_idx in time_cliffs:
        if cliff_idx < len(sizes) - 1:
            ax1.axvline(x=sizes[cliff_idx + 1],
                        color='green', linestyle='--', alpha=0.7)
            ax2.axvline(x=sizes[cliff_idx + 1],
                        color='green', linestyle='--', alpha=0.7)

    # Add annotations for estimated cache sizes
    estimated_caches = []
    for cliff_idx in time_cliffs:
        if cliff_idx < len(sizes) - 1:
            cache_size_bytes = sizes[cliff_idx + 1] * 8  # Convert to bytes
            estimated_caches.append(cache_size_bytes)

    # Label estimated cache sizes
    cache_labels = ['L1', 'L2', 'L3']
    for i, (cliff_idx, label) in enumerate(zip(time_cliffs[:3], cache_labels)):
        if cliff_idx < len(sizes) - 1:
            cache_size_bytes = sizes[cliff_idx + 1] * 8
            ax1.annotate(f'{label} (~{cache_size_bytes/1024:.0f}KB)',
                         xy=(sizes[cliff_idx + 1], times[cliff_idx + 1]),
                         xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # Create plots directory if it doesn't exist
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    plt.savefig("./plots/plot1.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary of estimated cache sizes
    print("Cache Hierarchy Analysis Summary:")
    print("=" * 40)
    print(
        f"Tested range: {sizes[0]*8/1024:.0f} KB to {sizes[-1]*8/1024/1024:.0f} MB")
    print(f"Data points: {len(sizes)}")
    print()

    for i, cache_size_bytes in enumerate(estimated_caches[:3]):
        cache_label = ['L1', 'L2', 'L3'][i]
        print(f"{cache_label} Cache: ~{cache_size_bytes/1024:.0f} KB")

    if len(estimated_caches) == 0:
        print("No clear cache boundaries detected in this range.")
        print("This may indicate:")
        print("- All data fits in L1 cache")
        print("- Need larger test sizes")
        print("- Performance counters not available")

    return estimated_caches


def plot_google_benchmark_median(json_path):
    """
    Main function to plot cache hierarchy analysis.
    """
    return plot_cache_hierarchy_analysis(json_path)


if __name__ == "__main__":
    # print("Generating performance analysis plots...")

    # Execute all analysis functions
    generate_performance_comparison()
    generate_speedup_analysis()

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "./logs/lab02-part1.json"

    plot_google_benchmark_median(json_path)
