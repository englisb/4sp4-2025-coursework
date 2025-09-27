
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def plot_google_benchmark_median(json_path):
    # Load and parse the JSON benchmark data
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Dictionary to store data for each algorithm
    algorithms = {
        'BM_SORT_STD': {'name': 'std::sort', 'data': defaultdict(list), 'color': '#1f77b4'},
        'BM_QUICKSORT': {'name': 'Quick Sort', 'data': defaultdict(list), 'color': '#ff7f0e'},
        'BM_SELECTION_SORT': {'name': 'Selection Sort', 'data': defaultdict(list), 'color': '#2ca02c'},
        'BM_BUBBLE_SORT': {'name': 'Bubble Sort', 'data': defaultdict(list), 'color': '#d62728'}
    }
    
    # Parse benchmark results
    for benchmark in data['benchmarks']:
        name = benchmark['name']
        
        # Extract algorithm name and array size
        for alg_key in algorithms.keys():
            if name.startswith(alg_key):
                # Extract array size (first number after algorithm name)
                size_str = name.split('/')[1]
                array_size = int(size_str)
                
                # Store median time (real_time in microseconds)
                algorithms[alg_key]['data'][array_size].append(benchmark['real_time'])
                break
    
    # Calculate median times for each array size
    for alg_key in algorithms.keys():
        for size in algorithms[alg_key]['data']:
            times = algorithms[alg_key]['data'][size]
            algorithms[alg_key]['data'][size] = np.median(times)
    
    # Create the performance comparison plot
    plt.figure(figsize=(12, 8))
    
    for alg_key, alg_info in algorithms.items():
        if alg_info['data']:
            sizes = sorted(alg_info['data'].keys())
            times = [alg_info['data'][size] for size in sizes]
            
            plt.loglog(sizes, times, 'o-', label=alg_info['name'], 
                      color=alg_info['color'], linewidth=2, markersize=6)
    
    # Add theoretical complexity lines for reference
    if algorithms['BM_SORT_STD']['data']:
        sizes = sorted(algorithms['BM_SORT_STD']['data'].keys())
        
        # O(n log n) reference line
        n_log_n_ref = [size * np.log2(size) * 0.001 for size in sizes]
        plt.loglog(sizes, n_log_n_ref, '--', alpha=0.5, color='gray', 
                  label='O(n log n) reference')
        
        # O(n²) reference line
        n_squared_ref = [size * size * 0.00001 for size in sizes]
        plt.loglog(sizes, n_squared_ref, ':', alpha=0.5, color='red', 
                  label='O(n²) reference')
    
    plt.ylabel("Execution Time (microseconds)", fontsize=12)
    plt.xlabel("Array Size (elements)", fontsize=12)
    plt.title("Sorting Algorithms Performance Comparison\n(Log-Log Scale)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    
    # Save the log-log plot
    plt.savefig("./plots/sorting_performance_loglog.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a linear scale plot for smaller sizes to show detail
    plt.figure(figsize=(12, 8))
    
    for alg_key, alg_info in algorithms.items():
        if alg_info['data']:
            sizes = sorted(alg_info['data'].keys())
            times = [alg_info['data'][size] for size in sizes]
            
            # Only plot up to a reasonable size for linear scale
            max_size_linear = 2048
            linear_sizes = [s for s in sizes if s <= max_size_linear]
            linear_times = [alg_info['data'][s] for s in linear_sizes]
            
            if linear_sizes:
                plt.plot(linear_sizes, linear_times, 'o-', label=alg_info['name'], 
                        color=alg_info['color'], linewidth=2, markersize=6)
    
    plt.ylabel("Execution Time (microseconds)", fontsize=12)
    plt.xlabel("Array Size (elements)", fontsize=12)
    plt.title("Sorting Algorithms Performance Comparison\n(Linear Scale - Small Arrays)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the linear plot
    plt.savefig("./plots/sorting_performance_linear.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a complexity ratio plot
    plt.figure(figsize=(12, 8))
    
    # Use std::sort as baseline for comparison
    std_data = algorithms['BM_SORT_STD']['data']
    if std_data:
        for alg_key, alg_info in algorithms.items():
            if alg_key != 'BM_SORT_STD' and alg_info['data']:
                sizes = []
                ratios = []
                
                for size in sorted(alg_info['data'].keys()):
                    if size in std_data:
                        ratio = alg_info['data'][size] / std_data[size]
                        sizes.append(size)
                        ratios.append(ratio)
                
                if sizes:
                    plt.semilogx(sizes, ratios, 'o-', label=f"{alg_info['name']} / std::sort", 
                               color=alg_info['color'], linewidth=2, markersize=6)
        
        plt.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='std::sort baseline')
    
    plt.ylabel("Performance Ratio (relative to std::sort)", fontsize=12)
    plt.xlabel("Array Size (elements)", fontsize=12)
    plt.title("Sorting Algorithms Performance Ratios\n(Relative to std::sort)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    # Save the ratio plot
    plt.savefig("./plots/sorting_performance_ratios.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated three plots:")
    print("1. ./plots/sorting_performance_loglog.png - Log-log scale comparison")
    print("2. ./plots/sorting_performance_linear.png - Linear scale for small arrays")  
    print("3. ./plots/sorting_performance_ratios.png - Performance ratios relative to std::sort")

# Example usage: the input should stay the same. The output plot should be saved in the plots folder.
plot_google_benchmark_median("./logs/lab01.json")
