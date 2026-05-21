import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_google_benchmark_median(json_path):
    # Open and parse the JSON benchmark results
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract benchmark results
    benchmarks = data.get('benchmarks', [])
    
    # Separate N-body and Cholesky benchmarks
    nbody_baseline = {}  # Dictionary: N -> [times]
    nbody_vectorized = {}  # Dictionary: N -> [times]
    forces_baseline = {}  # Dictionary: N -> [times]
    forces_vectorized = {}  # Dictionary: N -> [times]
    positions_baseline = {}  # Dictionary: N -> [times]
    positions_vectorized = {}  # Dictionary: N -> [times]
    cholesky_baseline = []
    cholesky_vectorized = []
    
    for bench in benchmarks:
        name = bench['name']
        
        # Skip aggregate results, only use individual runs
        if 'median' in name or 'mean' in name or 'stddev' in name:
            continue
            
        time = bench.get('real_time', bench.get('cpu_time', 0))
        
        # Extract N value from benchmark name (format: BM_NBODY/N/1)
        if 'BM_NBODY' in name or 'BM_FORCES' in name or 'BM_POSITIONS' in name:
            try:
                parts = name.split('/')
                N = int(parts[1]) if len(parts) > 1 else 1000  # Default N=1000
            except:
                N = 1000
            
            if 'BM_FORCES_VEC' in name:
                if N not in forces_vectorized:
                    forces_vectorized[N] = []
                forces_vectorized[N].append(time)
            elif 'BM_FORCES' in name:
                if N not in forces_baseline:
                    forces_baseline[N] = []
                forces_baseline[N].append(time)
            elif 'BM_POSITIONS_VEC' in name:
                if N not in positions_vectorized:
                    positions_vectorized[N] = []
                positions_vectorized[N].append(time)
            elif 'BM_POSITIONS' in name:
                if N not in positions_baseline:
                    positions_baseline[N] = []
                positions_baseline[N].append(time)
            elif 'BM_NBODY_VEC' in name:
                if N not in nbody_vectorized:
                    nbody_vectorized[N] = []
                nbody_vectorized[N].append(time)
            elif 'BM_NBODY' in name:
                if N not in nbody_baseline:
                    nbody_baseline[N] = []
                nbody_baseline[N].append(time)
        elif 'BM_CHOLESKY_VEC' in name:
            try:
                size = int(name.split('/')[1])
            except:
                size = 64
            cholesky_vectorized.append((size, time))
        elif 'BM_CHOLESKY' in name and 'VEC' not in name:
            try:
                size = int(name.split('/')[1])
            except:
                size = 64
            cholesky_baseline.append((size, time))
    
    # Create plots folder if it doesn't exist
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    
    # Plot 1: N-body Performance Comparison
    if nbody_baseline and nbody_vectorized:
        plot_nbody_comparison(nbody_baseline, nbody_vectorized)
    
    # Plot 2: Force Calculation Performance Comparison
    if forces_baseline and forces_vectorized:
        plot_forces_comparison(forces_baseline, forces_vectorized)
    
    # Plot 4: Position Update Performance Comparison
    if positions_baseline and positions_vectorized:
        plot_positions_comparison(positions_baseline, positions_vectorized)

    # Plot 5: Cholesky Performance Comparison
    if cholesky_baseline and cholesky_vectorized:
        plot_cholesky_performance(cholesky_baseline, cholesky_vectorized)

def plot_nbody_comparison(baseline_dict, vectorized_dict):
    """Plot N-body baseline vs vectorized performance comparison with speedup"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Get sorted N values and calculate median times for each N
    N_values = sorted(set(list(baseline_dict.keys()) + list(vectorized_dict.keys())))
    
    baseline_medians = []
    vectorized_medians = []
    speedups = []
    
    for N in N_values:
        if N in baseline_dict and N in vectorized_dict:
            baseline_med = np.median(baseline_dict[N])
            vectorized_med = np.median(vectorized_dict[N])
            baseline_medians.append(baseline_med)
            vectorized_medians.append(vectorized_med)
            speedups.append(baseline_med / vectorized_med)
        elif N in baseline_dict:
            baseline_medians.append(np.median(baseline_dict[N]))
            vectorized_medians.append(None)
            speedups.append(None)
        elif N in vectorized_dict:
            baseline_medians.append(None)
            vectorized_medians.append(np.median(vectorized_dict[N]))
            speedups.append(None)
    
    # Plot runtime lines on primary y-axis
    line1 = ax1.plot(N_values, baseline_medians, 'o-', label='Scalar N-Body', 
                     linewidth=2.5, markersize=10, color='#1f77b4', markeredgewidth=2, 
                     markeredgecolor='white')
    line2 = ax1.plot(N_values, vectorized_medians, 's-', label='Vectorized N-Body (AVX)', 
                     linewidth=2.5, markersize=10, color='#ff7f0e', markeredgewidth=2,
                     markeredgecolor='white')
    
    ax1.set_xlabel('N (number of particles)', fontsize=13)
    ax1.set_ylabel('Runtime (μs)', fontsize=13)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Create secondary y-axis for speedup
    ax2 = ax1.twinx()
    line3 = ax2.plot(N_values, speedups, '^--', label='Speedup (scalar / vec)', 
                     linewidth=2.5, markersize=10, color='#2ca02c', markeredgewidth=2,
                     markeredgecolor='white')
    ax2.set_ylabel('Speedup (X)', fontsize=13)
    ax2.tick_params(axis='y', labelsize=11)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.title('N-Body Runtime vs N (100 steps)', fontsize=15, pad=15)
    plt.tight_layout()
    plt.savefig("./plots/nbody_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    avg_speedup = np.mean([s for s in speedups if s is not None])
    print(f"✓ N-body plot saved: Vectorized implementation achieves {avg_speedup:.2f}x average speedup over baseline")

def plot_forces_comparison(baseline_dict, vectorized_dict):
    """Plot force calculation baseline vs vectorized performance comparison with speedup"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Get sorted N values and calculate median times for each N
    N_values = sorted(set(list(baseline_dict.keys()) + list(vectorized_dict.keys())))
    
    baseline_medians = []
    vectorized_medians = []
    speedups = []
    
    for N in N_values:
        if N in baseline_dict and N in vectorized_dict:
            baseline_med = np.median(baseline_dict[N])
            vectorized_med = np.median(vectorized_dict[N])
            baseline_medians.append(baseline_med)
            vectorized_medians.append(vectorized_med)
            speedups.append(baseline_med / vectorized_med)
        elif N in baseline_dict:
            baseline_medians.append(np.median(baseline_dict[N]))
            vectorized_medians.append(None)
            speedups.append(None)
        elif N in vectorized_dict:
            baseline_medians.append(None)
            vectorized_medians.append(np.median(vectorized_dict[N]))
            speedups.append(None)
    
    # Plot runtime lines on primary y-axis
    line1 = ax1.plot(N_values, baseline_medians, 'o-', label='Scalar Force Calculation', 
                     linewidth=2.5, markersize=10, color='#1f77b4', markeredgewidth=2, 
                     markeredgecolor='white')
    line2 = ax1.plot(N_values, vectorized_medians, 's-', label='Vectorized Force Calculation (AVX)', 
                     linewidth=2.5, markersize=10, color='#ff7f0e', markeredgewidth=2,
                     markeredgecolor='white')
    
    ax1.set_xlabel('N (number of particles)', fontsize=13)
    ax1.set_ylabel('Runtime (μs)', fontsize=13)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Create secondary y-axis for speedup
    ax2 = ax1.twinx()
    line3 = ax2.plot(N_values, speedups, '^--', label='Speedup (scalar / vec)', 
                     linewidth=2.5, markersize=10, color='#2ca02c', markeredgewidth=2,
                     markeredgecolor='white')
    ax2.set_ylabel('Speedup (X)', fontsize=13)
    ax2.tick_params(axis='y', labelsize=11)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.title('Force Calculation Runtime vs N', fontsize=15, pad=15)
    plt.tight_layout()
    plt.savefig("./plots/forces_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    avg_speedup = np.mean([s for s in speedups if s is not None])
    print(f"✓ Force calculation plot saved: Vectorized implementation achieves {avg_speedup:.2f}x average speedup over baseline")

def plot_positions_comparison(baseline_dict, vectorized_dict):
    """Plot position update baseline vs vectorized performance comparison with speedup"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Get sorted N values and calculate median times for each N
    N_values = sorted(set(list(baseline_dict.keys()) + list(vectorized_dict.keys())))
    
    baseline_medians = []
    vectorized_medians = []
    speedups = []
    
    for N in N_values:
        if N in baseline_dict and N in vectorized_dict:
            baseline_med = np.median(baseline_dict[N])
            vectorized_med = np.median(vectorized_dict[N])
            baseline_medians.append(baseline_med)
            vectorized_medians.append(vectorized_med)
            speedups.append(baseline_med / vectorized_med)
        elif N in baseline_dict:
            baseline_medians.append(np.median(baseline_dict[N]))
            vectorized_medians.append(None)
            speedups.append(None)
        elif N in vectorized_dict:
            baseline_medians.append(None)
            vectorized_medians.append(np.median(vectorized_dict[N]))
            speedups.append(None)
    
    # Plot runtime lines on primary y-axis
    line1 = ax1.plot(N_values, baseline_medians, 'o-', label='Scalar Position Update', 
                     linewidth=2.5, markersize=10, color='#1f77b4', markeredgewidth=2, 
                     markeredgecolor='white')
    line2 = ax1.plot(N_values, vectorized_medians, 's-', label='Vectorized Position Update (AVX)', 
                     linewidth=2.5, markersize=10, color='#ff7f0e', markeredgewidth=2,
                     markeredgecolor='white')
    
    ax1.set_xlabel('N (number of particles)', fontsize=13)
    ax1.set_ylabel('Runtime (μs)', fontsize=13)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Create secondary y-axis for speedup
    ax2 = ax1.twinx()
    line3 = ax2.plot(N_values, speedups, '^--', label='Speedup (scalar / vec)', 
                     linewidth=2.5, markersize=10, color='#2ca02c', markeredgewidth=2,
                     markeredgecolor='white')
    ax2.set_ylabel('Speedup (X)', fontsize=13)
    ax2.tick_params(axis='y', labelsize=11)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.title('Position Update Runtime vs N', fontsize=15, pad=15)
    plt.tight_layout()
    plt.savefig("./plots/positions_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    avg_speedup = np.mean([s for s in speedups if s is not None])
    print(f"✓ Position update plot saved: Vectorized implementation achieves {avg_speedup:.2f}x average speedup over baseline")

def plot_cholesky_performance(baseline_data, vectorized_data):
    """Plot Cholesky performance vs matrix size and GFLOP/s as separate figures"""
    # Sort by matrix size and calculate medians for each size
    baseline_dict = {}
    vectorized_dict = {}
    
    for size, time in baseline_data:
        size = int(size) if isinstance(size, str) else size
        if size not in baseline_dict:
            baseline_dict[size] = []
        baseline_dict[size].append(time)
    
    for size, time in vectorized_data:
        size = int(size) if isinstance(size, str) else size
        if size not in vectorized_dict:
            vectorized_dict[size] = []
        vectorized_dict[size].append(time)
    
    # Calculate medians
    baseline_sizes = sorted(baseline_dict.keys())
    baseline_medians = [np.median(baseline_dict[s]) for s in baseline_sizes]
    
    vectorized_sizes = sorted(vectorized_dict.keys())
    vectorized_medians = [np.median(vectorized_dict[s]) for s in vectorized_sizes]
    
    # Figure 1: Runtime scaling
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_sizes, baseline_medians, 'o-', label='Baseline', 
            linewidth=2, markersize=8, color='#3498db')
    plt.plot(vectorized_sizes, vectorized_medians, 's-', label='Vectorized (AVX)', 
            linewidth=2, markersize=8, color='#e74c3c')
    
    plt.ylabel("Execution Time (μs)", fontsize=12)
    plt.xlabel("Matrix Size", fontsize=12)
    plt.title("Cholesky Decomposition: Runtime Scaling", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("./plots/cholesky_runtime.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate GFLOP/s
    # Cholesky requires n³/3 FLOPs for n×n matrix
    baseline_gflops = [(size**3/3) / (1e3 * time) for size, time in zip(baseline_sizes, baseline_medians)]
    vectorized_gflops = [(size**3/3) / (1e3 * time) for size, time in zip(vectorized_sizes, vectorized_medians)]
    
    # Figure 2: GFLOP/s
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_sizes, baseline_gflops, 'o-', label='Baseline', 
            linewidth=2, markersize=8, color='#3498db')
    plt.plot(vectorized_sizes, vectorized_gflops, 's-', label='Vectorized (AVX)', 
            linewidth=2, markersize=8, color='#e74c3c')
    
    plt.ylabel("Performance (GFLOP/s)", fontsize=12)
    plt.xlabel("Matrix Size", fontsize=12)
    plt.title("Cholesky Decomposition: Computational Throughput", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("./plots/cholesky_gflops.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate speedups for each size
    speedups = [baseline_medians[i] / vectorized_medians[i] 
                for i in range(min(len(baseline_medians), len(vectorized_medians)))]
    
    # Figure 3: Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_sizes, speedups, '^-', 
            label='Vectorized vs Baseline', 
            linewidth=2, markersize=8, color='#2ecc71')
    
    # Add horizontal line at y=1 to show baseline reference
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.ylabel("Speedup Factor (×)", fontsize=12)
    plt.xlabel("Matrix Size", fontsize=12)
    plt.title("Cholesky Decomposition: Vectorization Speedup", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("./plots/cholesky_speedup.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate statistics
    avg_speedup = np.mean(speedups)
    peak_speedup = max(speedups)
    peak_base_gflops = max(baseline_gflops)
    peak_vec_gflops = max(vectorized_gflops)
    
    print("✓ Cholesky plots saved:")
    print(f"  - Speedup analysis:")
    print(f"    * Average: {avg_speedup:.2f}×")
    print(f"    * Peak:   {peak_speedup:.2f}× (at n={baseline_sizes[speedups.index(peak_speedup)]})")
    print("  - Peak performance (GFLOP/s):")
    print(f"    * Baseline:   {peak_base_gflops:.1f}")
    print(f"    * Vectorized: {peak_vec_gflops:.1f}")

# Example usage: the input should stay the same. The output plot should be saved in the plots folder.
plot_google_benchmark_median("./logs/lab03.json")