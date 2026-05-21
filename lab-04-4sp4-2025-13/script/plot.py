import json
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

def plot_google_benchmark_median(json_path):
    # TODO: open the json/csv file and do necessary processing to plot what is needed.
    pass


# ============ Task 2 GPU Plotting Functions (Added) ============
import numpy as np

def plot_gpu_task2_analysis(gpu_json_path="./logs/lab04_gpu.json", cpu_json_path="./logs/lab04-cpu.json"):
    """Generate Task 2 GPU SpTRSV performance analysis plots comparing GPU vs CPU Sequential"""
    
    # Load GPU benchmark data
    with open(gpu_json_path, 'r') as f:
        gpu_data = json.load(f)
    
    # Load CPU benchmark data for sequential baseline
    try:
        with open(cpu_json_path, 'r') as f:
            cpu_data = json.load(f)
    except:
        cpu_data = None
    
    # Extract GPU performance data
    parallel_gpu_data = {}
    cusparse_data = {}
    
    for benchmark in gpu_data['benchmarks']:
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
                    parallel_gpu_data[matrix_name] = gpu_time
                elif bench_name == 'sptrsv_cusparse':
                    cusparse_data[matrix_name] = gpu_time
    
    # Extract CPU Sequential baseline data (use median/mean of valid runs)
    cpu_sequential_data = {}
    if cpu_data:
        temp_times = {}
        for benchmark in cpu_data['benchmarks']:
            bench_name = benchmark['name']
            if 'BM_SPTRSV/' not in bench_name or 'OMP' in bench_name or 'MKL' in bench_name:
                continue
            
            # Skip aggregate/mean entries
            if benchmark.get('run_type') == 'aggregate':
                continue
            
            label = benchmark.get('label', '')
            if 'Matrix:' in label:
                matrix_name = label.split('/')[-1].replace('.mtx', '')
                cpu_time = benchmark.get('real_time', 0)  # microseconds
                
                # Filter out outliers (too small values indicate benchmark issues)
                if cpu_time > 1.0:  # Ignore sub-microsecond times
                    if matrix_name not in temp_times:
                        temp_times[matrix_name] = []
                    temp_times[matrix_name].append(cpu_time)
        
        # Calculate median for each matrix
        for matrix_name, times in temp_times.items():
            if times:
                cpu_sequential_data[matrix_name] = sorted(times)[len(times)//2]  # median
    
    # Get common matrices
    matrices = sorted([m for m in parallel_gpu_data.keys() if m in cusparse_data.keys() and m in cpu_sequential_data.keys()])
    if not matrices:
        print("No common GPU and CPU data found")
        return
    
    parallel_gpu_times = [parallel_gpu_data[m] for m in matrices]
    cpu_sequential_times = [cpu_sequential_data[m] / 1e6 for m in matrices]  # Convert to seconds
    parallel_gpu_times_sec = [parallel_gpu_data[m] / 1e6 for m in matrices]  # Convert to seconds
    cusparse_times_sec = [cusparse_data[m] / 1e6 for m in matrices]  # Convert to seconds
    speedups_vs_sequential = [cpu_sequential_data[m] / parallel_gpu_data[m] for m in matrices]
    
    # Create plots directory
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    
    # Plot 1: Performance Comparison Bar Chart
    plt.figure(figsize=(14, 6))
    x = np.arange(len(matrices))
    width = 0.25
    
    plt.bar(x - width, cpu_sequential_times, width, label='Sequential', alpha=0.8, color='steelblue')
    plt.bar(x, parallel_gpu_times_sec, width, label='Level_Sync', alpha=0.8, color='orange')
    plt.bar(x + width, cusparse_times_sec, width, label='Cusparse', alpha=0.8, color='green')
    
    plt.xlabel('Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('GPU Runtime (seconds)', fontsize=12, fontweight='bold')
    plt.title('GPU Runtime Comparison by Matrix', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, matrices, rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./plots/task2_gpu_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Generated: plots/task2_gpu_comparison.png")
    plt.close()
    
    # Plot 2: Speedup vs CPU Sequential
    plt.figure(figsize=(12, 6))
    x = np.arange(len(matrices))
    width = 0.35
    
    # Speedup of GPU parallel vs CPU sequential
    colors = ['green' if s > 1 else 'red' for s in speedups_vs_sequential]
    bars = plt.bar(matrices, speedups_vs_sequential, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (Sequential)')
    plt.xlabel('Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    plt.title('Task 2: GPU Parallel (Level_Sync) Speedup vs CPU Sequential\n(Higher is Better)', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add speedup values on bars
    for bar, speedup in zip(bars, speedups_vs_sequential):
        height = bar.get_height()
        label = f'{speedup:.1f}x'
        plt.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("./plots/task2_gpu_speedup.png", dpi=300, bbox_inches='tight')
    print("✓ Generated: plots/task2_gpu_speedup.png")
    plt.close()
    
    # Plot 3: Performance Scaling
    plt.figure(figsize=(12, 7))
    x_pos = np.arange(len(matrices))
    
    plt.plot(x_pos, cpu_sequential_times, 'o-', label='Sequential', 
             linewidth=2.5, markersize=10, color='steelblue')
    plt.plot(x_pos, parallel_gpu_times_sec, 'o-', label='Level_Sync', 
             linewidth=2.5, markersize=10, color='orange')
    plt.plot(x_pos, cusparse_times_sec, 'o-', label='Cusparse', 
             linewidth=2.5, markersize=10, color='green')
    
    plt.xlabel('Matrix Type', fontsize=12, fontweight='bold')
    plt.ylabel('GPU Runtime (seconds)', fontsize=12, fontweight='bold')
    plt.title('GPU Runtime Scaling by Matrix', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, matrices, rotation=45, ha='right')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./plots/task2_gpu_scaling.png", dpi=300, bbox_inches='tight')
    print("✓ Generated: plots/task2_gpu_scaling.png")
    plt.close()
    
    # Print Summary
    print("\n" + "="*85)
    print("Task 2: GPU SpTRSV Performance Summary")
    print("="*115)
    print(f"{'Matrix':<12} {'Sequential':<12} {'Solve Time':<12} {'Scheduling':<12} {'Accumulated':<12} {'Cusparse':<12} {'Speedup':<15}")
    print(f"{'':12} {'CPU (μs)':<12} {'GPU (μs)':<12} {'Time (μs)':<12} {'Time (μs)':<12} {'(μs)':<12} {'vs CPU':<15}")
    print("-"*115)
    
    # Extract scheduling times if available in GPU JSON
    try:
        with open("./logs/lab04_gpu.json", "r") as f:
            gpu_full_data = json.load(f)
        
        scheduling_times = {}
        num_levels_data = {}
        
        for benchmark in gpu_full_data.get("benchmarks", []):
            # Process sptrsv benchmark (not sequential or cusparse)
            if benchmark["name"] == "sptrsv":
                # Each state corresponds to a different matrix
                for state in benchmark.get("states", []):
                    matrix_key = None
                    
                    # Extract matrix name and timing data from this state's summaries
                    for summary in state.get("summaries", []):
                        tag = summary.get("tag", "")
                        data = summary.get("data", [])
                        
                        if tag == "matrix_name" and data:
                            matrix_name = data[0].get("value", "")
                            matrix_key = matrix_name.split('/')[-1].replace('.mtx', '')
                        elif "Scheduling Time" in tag and data:
                            temp_sched = float(data[0].get("value", "0"))
                            if matrix_key:
                                scheduling_times[matrix_key] = temp_sched
                        elif "Num Levels" in tag and data:
                            temp_levels = int(float(data[0].get("value", "0")))
                            if matrix_key:
                                num_levels_data[matrix_key] = temp_levels
                    
                    # Store temp values if matrix_key was found later
                    if matrix_key and 'temp_sched' in locals():
                        scheduling_times[matrix_key] = temp_sched
                    if matrix_key and 'temp_levels' in locals():
                        num_levels_data[matrix_key] = temp_levels
        
        for i, m in enumerate(matrices):
            speedup = speedups_vs_sequential[i]
            sched = scheduling_times.get(m, 0.0)
            accumulated = parallel_gpu_times[i] + sched
            levels = num_levels_data.get(m, "N/A")
            print(f"{m:<12} {cpu_sequential_times[i]:<12.2f} {parallel_gpu_times[i]:<12.2f} {sched:<12.2f} {accumulated:<12.2f} {cusparse_times_sec[i]:<12.2f} {speedup:<15.2f}x")
    except:
        # Fallback if scheduling data not available
        for i, m in enumerate(matrices):
            speedup = speedups_vs_sequential[i]
            print(f"{m:<12} {cpu_sequential_times[i]:<12.2f} {parallel_gpu_times[i]:<12.2f} {'N/A':<12} {'N/A':<12} {cusparse_times_sec[i]:<12.2f} {speedup:<15.2f}x")
    
    print("-"*115)
    avg_seq = sum(cpu_sequential_times) / len(cpu_sequential_times)
    avg_par = sum(parallel_gpu_times) / len(parallel_gpu_times)
    avg_cusparse = sum(cusparse_times_sec) / len(cusparse_times_sec)
    avg_speedup = sum(speedups_vs_sequential) / len(speedups_vs_sequential)
    print(f"{'Average':<12} {avg_seq:<12.2f} {avg_par:<12.2f} {'--':<12} {'--':<12} {avg_cusparse:<12.2f} {avg_speedup:<15.2f}x")
    print("="*115)
    
    print("\n" + "="*80)
    print("ANALYSIS NOTES:")
    print("="*80)
    print("• Solve Time: Pure GPU kernel execution (what gets benchmarked)")
    print("• Scheduling Time: Level-set preprocessing on CPU (done once)")
    print("• Accumulated Time: Total = Solve + Scheduling")
    print("• Speedup: Based on solve time only (fair comparison)")
    print("• cuSPARSE faster: Expected - highly optimized commercial library")
    print("• Level-set enables parallelism but adds kernel launch overhead")
    print("="*80)
    print("\n✓ All Task 2 GPU plots generated successfully!")

# Run Task 2 GPU analysis
if __name__ == "__main__":
    print("Generating Task 2 GPU Analysis Plots...")
    plot_gpu_task2_analysis()
