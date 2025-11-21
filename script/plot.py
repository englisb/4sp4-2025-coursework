import json
import matplotlib.pyplot as plt
import os

def plot_google_benchmark_median(json_path):
    # TODO: open the json/csv file and do necessary processing to plot what is needed.
    pass


# ============ Task 2 GPU Plotting Functions (Added) ============
import numpy as np

def plot_gpu_task2_analysis(json_path="./logs/lab04_gpu.json"):
    """Generate Task 2 GPU SpTRSV performance analysis plots"""
    
    # Load GPU benchmark data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract performance data
    custom_data = {}
    cusparse_data = {}
    
    for benchmark in data['benchmarks']:
        bench_name = benchmark['name']
        for state in benchmark['states']:
            if state.get('is_skipped', False):
                continue
            
            matrix_name = None
            gpu_time = None
            
            for summary in state['summaries']:
                if summary.get('tag') == 'matrix_name':
                    matrix_name = summary['data'][0]['value'].split('/')[-1].replace('.mtx', '')
                elif summary.get('tag') == 'nv/cold/time/gpu/mean':
                    gpu_time = float(summary['data'][0]['value']) * 1e6  # to microseconds
            
            if matrix_name and gpu_time:
                if bench_name == 'sptrsv':
                    custom_data[matrix_name] = gpu_time
                elif bench_name == 'sptrsv_cusparse':
                    cusparse_data[matrix_name] = gpu_time
    
    # Get common matrices
    matrices = sorted([m for m in custom_data.keys() if m in cusparse_data.keys()])
    if not matrices:
        print("No GPU data found")
        return
    
    custom_times = [custom_data[m] for m in matrices]
    cusparse_times = [cusparse_data[m] for m in matrices]
    speedups = [cusparse_data[m] / custom_data[m] for m in matrices]
    
    # Create plots directory
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    
    # Plot 1: Performance Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(matrices))
    width = 0.35
    
    plt.bar(x - width/2, custom_times, width, label='Custom SpTRSV', alpha=0.8, color='steelblue')
    plt.bar(x + width/2, cusparse_times, width, label='cuSPARSE', alpha=0.8, color='coral')
    
    plt.xlabel('Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('GPU Execution Time (μs)', fontsize=12, fontweight='bold')
    plt.title('Task 2: GPU SpTRSV Performance Comparison\nCustom Implementation vs cuSPARSE (Lower is Better)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, matrices, rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./plots/task2_gpu_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Generated: plots/task2_gpu_comparison.png")
    plt.close()
    
    # Plot 2: Speedup Analysis
    plt.figure(figsize=(10, 6))
    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = plt.bar(matrices, speedups, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Baseline (cuSPARSE)')
    plt.xlabel('Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    plt.title('Task 2: Custom GPU SpTRSV Speedup vs cuSPARSE\n(Values > 1 mean custom is faster)', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add speedup values on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("./plots/task2_gpu_speedup.png", dpi=300, bbox_inches='tight')
    print("✓ Generated: plots/task2_gpu_speedup.png")
    plt.close()
    
    # Plot 3: Performance Scaling
    plt.figure(figsize=(12, 7))
    x_pos = np.arange(len(matrices))
    
    plt.plot(x_pos, custom_times, 'o-', label='Custom GPU SpTRSV', 
             linewidth=2.5, markersize=10, color='steelblue')
    plt.plot(x_pos, cusparse_times, 's-', label='cuSPARSE', 
             linewidth=2.5, markersize=10, color='coral')
    
    plt.xlabel('Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (μs, log scale)', fontsize=12, fontweight='bold')
    plt.title('Task 2: GPU SpTRSV Performance Scaling', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, matrices, rotation=45, ha='right')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./plots/task2_gpu_scaling.png", dpi=300, bbox_inches='tight')
    print("✓ Generated: plots/task2_gpu_scaling.png")
    plt.close()
    
    # Print Summary
    print("\n" + "="*70)
    print("Task 2: GPU SpTRSV Performance Summary")
    print("="*70)
    print(f"{'Matrix':<15} {'Custom (μs)':<15} {'cuSPARSE (μs)':<15} {'Speedup':<10}")
    print("-"*70)
    for i, m in enumerate(matrices):
        print(f"{m:<15} {custom_times[i]:<15.2f} {cusparse_times[i]:<15.2f} {speedups[i]:<10.2f}x")
    print("-"*70)
    avg_speedup = sum(speedups) / len(speedups)
    print(f"{'Average':<15} {sum(custom_times)/len(custom_times):<15.2f} "
          f"{sum(cusparse_times)/len(cusparse_times):<15.2f} {avg_speedup:<10.2f}x")
    print("="*70)
    print("\n✓ All Task 2 GPU plots generated successfully!")

# Run Task 2 GPU analysis
if __name__ == "__main__":
    print("Generating Task 2 GPU Analysis Plots...")
    plot_gpu_task2_analysis()