import csv
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

def parse_ncu_csv(csv_path):
    """Parse Nsight Compute CSV output and extract metrics per-kernel instance, then map to sizes.

    We group rows by CSV "ID" (one kernel invocation), capture version, grid, block,
    L1/TEX Hit Rate (%) and Memory Throughput (Gbyte/second), then map (grid,block)
    to the known problem sizes {2^10,2^15,2^20,2^25} using ceil(n/256) => grid.
    """
    # Known sizes and their expected grid for block=256
    known_sizes = [1024, 32768, 1048576, 33554432]
    # v1/v2 use 1 element per thread: grid = ceil(n/256)
    # v3 uses 4 elements per thread: grid = ceil(n/(256*4))
    grid_for_size_v1v2 = { 4: 1024, 128: 32768, 4096: 1048576, 131072: 33554432 }
    grid_for_size_v3 = { 1: 1024, 32: 32768, 1024: 1048576, 32768: 33554432 }

    # First pass: collect per-ID info
    per_id = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get('ID')
            if rid is None:
                # Some Nsight versions may not include ID; fall back to Kernel Name as key
                rid = row.get('Kernel Name', '') + '|' + row.get('Context', '') + '|' + row.get('Stream', '')
            entry = per_id.setdefault(rid, {
                'kernel_name': row.get('Kernel Name', ''),
                'version': None,
                'grid': None,
                'block': None,
                'l1_hit_rate': None,
                'mem_throughput_gbps': None,
            })

            kname = entry['kernel_name']
            if entry['version'] is None:
                if 'mulAddKernel_v1' in kname:
                    entry['version'] = 'v1'
                elif 'mulAddKernel_v2' in kname:
                    entry['version'] = 'v2'
                elif 'mulAddKernel_v3' in kname:
                    entry['version'] = 'v3'

            section = row.get('Section Name', '')
            mname = row.get('Metric Name', '')
            munit = row.get('Metric Unit', '')
            mval = row.get('Metric Value', row.get('Avg', '')).strip()

            # Capture launch configuration
            if section == 'Launch Statistics':
                if mname == 'Grid Size':
                    try:
                        # Grid Size is in format "(4, 1, 1)" - extract first number
                        cleaned = mval.replace(',', '')
                        if cleaned.startswith('('):
                            cleaned = cleaned.strip('()').split(',')[0].strip()
                        entry['grid'] = int(cleaned)
                    except:
                        pass
                elif mname == 'Block Size':
                    try:
                        # Block Size is in format "(256, 1, 1)" - extract first number
                        cleaned = mval.replace(',', '')
                        if cleaned.startswith('('):
                            cleaned = cleaned.strip('()').split(',')[0].strip()
                        entry['block'] = int(cleaned)
                    except:
                        pass

            # Capture L1/TEX Hit Rate
            if mname == 'L1/TEX Hit Rate' or 'l1tex__t_sector_hit_rate' in mname:
                try:
                    entry['l1_hit_rate'] = float(mval.replace('%', ''))
                except:
                    pass

            # Prefer absolute Memory Throughput (Gbyte/second) from Memory Workload Analysis
            if mname == 'Memory Throughput' and 'Gbyte/second' in munit:
                try:
                    entry['mem_throughput_gbps'] = float(mval)
                except:
                    pass

    # Second pass: aggregate by (version, size)
    data = defaultdict(lambda: defaultdict(dict))
    for entry in per_id.values():
        version = entry.get('version')
        grid = entry.get('grid')
        block = entry.get('block')
        l1 = entry.get('l1_hit_rate')
        gbps = entry.get('mem_throughput_gbps')

        if version is None:
            continue

        # Infer size using grid for block=256 mapping; use version-specific grid mapping
        size = None
        if isinstance(grid, int):
            # Choose the right grid->size mapping based on kernel version
            if version == 'v3':
                size = grid_for_size_v3.get(grid)
            else:
                size = grid_for_size_v1v2.get(grid)
            
            if size is None and isinstance(block, int) and block > 0:
                # Approximate n ~= grid*block and map to the closest known size by matching grid
                # Compute candidate n for each known size and choose the one with same grid
                for s in known_sizes:
                    if version == 'v3':
                        expected_grid = (s + 1023) // 1024  # 4 elements per thread
                    else:
                        expected_grid = (s + 255) // 256  # 1 element per thread
                    if expected_grid == grid:
                        size = s
                        break

        # If we couldn't infer size, skip aggregation for this entry
        if size is None:
            continue

        if l1 is not None:
            data[version][size]['l1_hit_rate'] = l1
        if gbps is not None:
            data[version][size]['mem_throughput'] = gbps

    return data

def plot_google_benchmark_median(json_path):
    """Plot L1/TEX Hit Rate and Memory Throughput from Nsight Compute CSV."""
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
    
    data = parse_ncu_csv(json_path)
    
    # plots folder does not exist, create it
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    
    # Prepare data
    versions = ['v1', 'v2', 'v3']
    sizes = sorted(set(s for v in data.values() for s in v.keys() if s is not None))
    
    if not sizes:
        sizes = [1024, 32768, 1048576, 33554432]  # 2^10, 2^15, 2^20, 2^25
    
    size_labels = [f'2^{i}' for i in [10, 15, 20, 25]] if len(sizes) == 4 else [str(s) for s in sizes]
    
    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # L1/TEX Hit Rate
    for version in versions:
        hit_rates = [data.get(version, {}).get(size, {}).get('l1_hit_rate', 0) for size in sizes]
        ax1.plot(size_labels, hit_rates, marker='o', linewidth=2, markersize=8, label=f'Kernel {version}')
    
    ax1.set_ylabel("L1/TEX Hit Rate (%)", fontsize=12)
    ax1.set_xlabel("Array Size (n)", fontsize=12)
    ax1.set_title("L1/TEX Cache Hit Rate", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Memory Throughput
    for version in versions:
        throughputs = [data.get(version, {}).get(size, {}).get('mem_throughput', 0) for size in sizes]
        ax2.plot(size_labels, throughputs, marker='s', linewidth=2, markersize=8, label=f'Kernel {version}')
    
    ax2.set_ylabel("Memory Throughput (GB/s)", fontsize=12)
    ax2.set_xlabel("Array Size (n)", fontsize=12)
    ax2.set_title("Memory Throughput", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./plots/combined_metrics.png", dpi=300)
    print("Saved: ./plots/combined_metrics.png")
    
    plt.close('all')
    
    # Print summary
    print("\n" + "="*60)
    print("Profiling Summary")
    print("="*60)
    for version in versions:
        if version in data:
            print(f"\nKernel {version}:")
            for size in sizes:
                metrics = data.get(version, {}).get(size, {})
                if metrics:
                    print(f"  Size {size:>10}:")
                    if 'l1_hit_rate' in metrics:
                        print(f"    L1/TEX Hit Rate: {metrics['l1_hit_rate']:.2f}%")
                    if 'mem_throughput' in metrics:
                        print(f"    Memory Throughput: {metrics['mem_throughput']:.2f} GB/s")

# Example usage: the input should stay the same. The output plot should be saved in the plots folder.
plot_google_benchmark_median("./logs/ncu_results.csv")