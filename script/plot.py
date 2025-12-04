
import json
import matplotlib.pyplot as plt
import os
import sys
import statistics
from collections import defaultdict

def parse_gemm_args(name):
    """Return (m,n,k,t1,t2) parsed robustly from BM_GEMM* name.
    Name can include extra tags (e.g., /threads:1). We extract the first 5 integers after BM_GEMM*.
    Handles BM_GEMM_NAIVE, BM_GEMM_SIMD, BM_GEMM_SIMD_PAR, BM_GEMM_TILED_SIMD_PAR.
    """
    parts = name.split('/')
    if not parts or not parts[0].startswith('BM_GEMM'):
        return None
    nums = []
    for p in parts[1:]:
        # strip possible key:value suffixes
        token = p.split(':')[0]
        try:
            nums.append(int(token))
            # For TILED_SIMD_PAR we need 5 args, others need 3 args, but we'll accept 5 for TILED
            if len(nums) >= 3:
                # If we're at 3 nums and next is not a number (iterations:), we're done
                # For TILED we need to reach 5
                if 'TILED' in parts[0] and len(nums) < 5:
                    continue
                else:
                    break
        except ValueError:
            continue
    # For non-TILED variants, pad with -1 for tile sizes
    if len(nums) == 3 and 'TILED' not in parts[0]:
        nums.append(-1)
        nums.append(-1)
    if len(nums) != 5:
        return None
    m, n, k, t1, t2 = nums
    return (m, n, k, t1, t2)

def parse_spmm_args(name):
    """Return (m,n,k,t1,t2) parsed robustly from BM_SPMM name."""
    parts = name.split('/')
    if not parts or parts[0] != 'BM_SPMM':
        return None
    nums = []
    for p in parts[1:]:
        token = p.split(':')[0]
        try:
            nums.append(int(token))
            if len(nums) == 5:
                break
        except ValueError:
            continue
    if len(nums) != 5:
        return None
    m, n, k, t1, t2 = nums
    return (m, n, k, t1, t2)

def parse_gemv_args(name):
    """Return (m,n) parsed from BM_GEMV* name.
    Handles BM_GEMV_NAIVE, BM_GEMV_SIMD, BM_GEMV_SIMD_PAR.
    """
    parts = name.split('/')
    if not parts or not parts[0].startswith('BM_GEMV'):
        return None
    nums = []
    for p in parts[1:]:
        token = p.split(':')[0]
        try:
            nums.append(int(token))
            if len(nums) == 2:
                break
        except ValueError:
            continue
    if len(nums) != 2:
        return None
    return tuple(nums)

def parse_spmv_args(name):
    """Return (m,n) parsed from BM_SPMV name."""
    parts = name.split('/')
    if not parts or parts[0] != 'BM_SPMV':
        return None
    nums = []
    for p in parts[1:]:
        token = p.split(':')[0]
        try:
            nums.append(int(token))
            if len(nums) == 2:
                break
        except ValueError:
            continue
    if len(nums) != 2:
        return None
    return tuple(nums)


def collect_gemm(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    bench = data.get('benchmarks', [])
    grouped = {}
    for entry in bench:
        name = entry.get('name', '')
        args = parse_gemm_args(name)
        if not args:
            continue
        run_type = entry.get('run_type')
        # Prefer aggregate median if available; fall back to iteration list
        if run_type == 'aggregate' and entry.get('aggregate_name') == 'median':
            m,n,k,t1,t2 = args
            key = (m,n,k,t1,t2)
            grouped.setdefault(key, []).append(entry.get('real_time'))
        elif run_type == 'iteration':
            m,n,k,t1,t2 = args
            key = (m,n,k,t1,t2)
            grouped.setdefault(key, []).append(entry.get('real_time'))
    records = []
    for key, times in grouped.items():
        if not times:
            continue
        med = statistics.median(times)
        m,n,k,t1,t2 = key
        # Determine time unit (default microseconds). Google Benchmark may report ns.
        time_unit = entry_unit = data.get('context', {}).get('time_unit')
        # Fallback per-entry unit if present
        if not time_unit and len(bench) > 0:
            time_unit = bench[0].get('time_unit')
        # Convert to seconds
        unit_scale = 1e-6 if time_unit == 'us' else 1e-9 if time_unit == 'ns' else 1e-6
        records.append({
            'm': m, 'n': n, 'k': k, 't1': t1, 't2': t2,
            'median_time_s': med * unit_scale,
            'gflops': (2.0*m*n*k) / (med * unit_scale) / 1e9  # 2*m*n*k operations
        })
    return records

def collect_spmm(json_path):
    """Collect SPMM benchmark data (similar to GEMM)."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    bench = data.get('benchmarks', [])
    grouped = {}
    for entry in bench:
        name = entry.get('name', '')
        args = parse_spmm_args(name)
        if not args:
            continue
        run_type = entry.get('run_type')
        if run_type == 'aggregate' and entry.get('aggregate_name') == 'median':
            grouped.setdefault(args, []).append(entry.get('real_time'))
        elif run_type == 'iteration':
            grouped.setdefault(args, []).append(entry.get('real_time'))
    
    records = []
    time_unit = data.get('context', {}).get('time_unit', 'us')
    if not time_unit and bench:
        time_unit = bench[0].get('time_unit', 'us')
    unit_scale = 1e-6 if time_unit == 'us' else 1e-9 if time_unit == 'ns' else 1e-6
    
    for key, times in grouped.items():
        if not times:
            continue
        med = statistics.median(times)
        m,n,k,t1,t2 = key
        records.append({
            'm': m, 'n': n, 'k': k, 't1': t1, 't2': t2,
            'median_time_s': med * unit_scale,
            'gflops': (2.0*m*n*k) / (med * unit_scale) / 1e9
        })
    return records

def collect_gemv(json_path):
    """Collect GEMV benchmark data."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    bench = data.get('benchmarks', [])
    grouped = {}
    for entry in bench:
        name = entry.get('name', '')
        args = parse_gemv_args(name)
        if not args:
            continue
        run_type = entry.get('run_type')
        if run_type == 'aggregate' and entry.get('aggregate_name') == 'median':
            grouped.setdefault(args, []).append(entry.get('real_time'))
        elif run_type == 'iteration':
            grouped.setdefault(args, []).append(entry.get('real_time'))
    
    records = []
    time_unit = data.get('context', {}).get('time_unit', 'us')
    if not time_unit and bench:
        time_unit = bench[0].get('time_unit', 'us')
    unit_scale = 1e-6 if time_unit == 'us' else 1e-9 if time_unit == 'ns' else 1e-6
    
    for key, times in grouped.items():
        if not times:
            continue
        med = statistics.median(times)
        m, n = key
        records.append({
            'm': m, 'n': n,
            'median_time_s': med * unit_scale,
            'gflops': (2.0*m*n) / (med * unit_scale) / 1e9  # 2*m*n operations for GEMV
        })
    return records

def collect_spmv(json_path):
    """Collect SPMV benchmark data."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    bench = data.get('benchmarks', [])
    grouped = {}
    for entry in bench:
        name = entry.get('name', '')
        args = parse_spmv_args(name)
        if not args:
            continue
        run_type = entry.get('run_type')
        if run_type == 'aggregate' and entry.get('aggregate_name') == 'median':
            grouped.setdefault(args, []).append(entry.get('real_time'))
        elif run_type == 'iteration':
            grouped.setdefault(args, []).append(entry.get('real_time'))
    
    records = []
    time_unit = data.get('context', {}).get('time_unit', 'us')
    if not time_unit and bench:
        time_unit = bench[0].get('time_unit', 'us')
    unit_scale = 1e-6 if time_unit == 'us' else 1e-9 if time_unit == 'ns' else 1e-6
    
    for key, times in grouped.items():
        if not times:
            continue
        med = statistics.median(times)
        m, n = key
        records.append({
            'm': m, 'n': n,
            'median_time_s': med * unit_scale,
            'gflops': (2.0*m*n) / (med * unit_scale) / 1e9  # 2*m*n operations for SPMV
        })
    return records

def collect_nn(json_path):
    """Collect neural network benchmark data (both dense and sparse).
    Returns dict with 'dense' and 'sparse' keys, each containing sparsity->time/accuracy mappings.
    Also includes 'mkl' key for MKL dense baseline if available.
    """
    if not os.path.exists(json_path):
        return {'dense': {}, 'sparse': {}, 'mkl': {}}
    with open(json_path, 'r') as f:
        data = json.load(f)
    bench = data.get('benchmarks', [])
    
    dense_records = []
    sparse_records = {}  # sparsity -> list of times
    mkl_records = []
    
    time_unit = data.get('context', {}).get('time_unit', 'us')
    if not time_unit and bench:
        time_unit = bench[0].get('time_unit', 'us')
    unit_scale = 1e-6 if time_unit == 'us' else 1e-9 if time_unit == 'ns' else 1e-6
    
    for entry in bench:
        name = entry.get('name', '')
        run_type = entry.get('run_type')
        
        # Dense NN benchmark
        if 'BM_DENSENN' in name and 'MKL' not in name:
            if run_type in ['aggregate', 'iteration']:
                time_val = entry.get('real_time', 0)
                accuracy = entry.get('Accuracy', 0.0)  # Google Benchmark counters
                dense_records.append({
                    'time_s': time_val * unit_scale,
                    'accuracy': accuracy
                })
        
        # MKL Dense NN benchmark
        elif 'BM_DENSENN_MKL' in name:
            if run_type in ['aggregate', 'iteration']:
                time_val = entry.get('real_time', 0)
                accuracy = entry.get('Accuracy', 0.0)
                mkl_records.append({
                    'time_s': time_val * unit_scale,
                    'accuracy': accuracy
                })
        
        # Sparse NN benchmark (CPU with sparsity sweep)
        elif 'BM_SPARSENN' in name:
            # Parse sparsity from Args - format: BM_SPARSENN/32/32/50
            parts = name.split('/')
            if len(parts) >= 4:
                try:
                    sparsity = int(parts[3])
                    if run_type in ['aggregate', 'iteration']:
                        time_val = entry.get('real_time', 0)
                        accuracy = entry.get('Accuracy', 0.0)
                        if sparsity not in sparse_records:
                            sparse_records[sparsity] = []
                        sparse_records[sparsity].append({
                            'time_s': time_val * unit_scale,
                            'accuracy': accuracy
                        })
                except (ValueError, IndexError):
                    pass
        
        # GPU Dense NN benchmark
        elif 'GPU_Dense_NN' in name:
            if run_type in ['aggregate', 'iteration']:
                time_val = entry.get('real_time', 0)
                dense_records.append({
                    'time_s': time_val * unit_scale,
                    'accuracy': 0.0
                })
        
        # GPU Sparse NN benchmark
        elif 'GPU_Sparse_NN' in name:
            # nvbench format may differ - check for sparsity in entry
            sparsity = entry.get('sparsity')  # nvbench axis parameter
            if sparsity is not None:
                if run_type in ['aggregate', 'iteration']:
                    time_val = entry.get('real_time', 0)
                    if sparsity not in sparse_records:
                        sparse_records[sparsity] = []
                    sparse_records[sparsity].append({
                        'time_s': time_val * unit_scale,
                        'accuracy': 0.0  # GPU bench doesn't report accuracy to JSON
                    })
    
    # Aggregate results
    result = {'dense': {}, 'sparse': {}, 'mkl': {}}
    
    if dense_records:
        median_time = statistics.median([r['time_s'] for r in dense_records])
        avg_accuracy = statistics.mean([r['accuracy'] for r in dense_records]) if dense_records[0]['accuracy'] > 0 else 0
        result['dense'] = {
            'time_s': median_time,
            'accuracy': avg_accuracy
        }
    
    if mkl_records:
        median_time = statistics.median([r['time_s'] for r in mkl_records])
        avg_accuracy = statistics.mean([r['accuracy'] for r in mkl_records]) if mkl_records[0]['accuracy'] > 0 else 0
        result['mkl'] = {
            'time_s': median_time,
            'accuracy': avg_accuracy
        }
    
    for sparsity, records in sparse_records.items():
        if records:
            median_time = statistics.median([r['time_s'] for r in records])
            avg_accuracy = statistics.mean([r['accuracy'] for r in records]) if records[0]['accuracy'] > 0 else 0
            result['sparse'][sparsity] = {
                'time_s': median_time,
                'accuracy': avg_accuracy
            }
    
    return result

def plot_nn(nn_data):
    """Plot neural network benchmark results comparing dense vs sparse at different sparsity levels."""
    if not nn_data or (not nn_data.get('dense') and not nn_data.get('sparse') and not nn_data.get('mkl')):
        print("No NN data to plot.")
        return
    
    os.makedirs('./plots', exist_ok=True)
    
    sparse_data = nn_data.get('sparse', {})
    dense_data = nn_data.get('dense', {})
    mkl_data = nn_data.get('mkl', {})
    
    if not sparse_data:
        print("No sparse NN data found for sparsity sweep plot.")
        return
    
    # Sort sparsity levels
    sparsity_levels = sorted(sparse_data.keys())
    
    # Runtime comparison
    plt.figure(figsize=(12, 6))
    
    sparse_times = [sparse_data[sp]['time_s'] * 1e6 for sp in sparsity_levels]  # Convert to µs
    labels = [f"{sp}%" for sp in sparsity_levels]
    
    x_pos = range(len(sparsity_levels))
    width = 0.6
    
    bars = plt.bar(x_pos, sparse_times, width, label='Sparse NN', color='#2ca02c')
    
    # Add dense baseline as horizontal line if available
    if dense_data:
        dense_time = dense_data['time_s'] * 1e6
        plt.axhline(y=dense_time, color='#1f77b4', linestyle='--', linewidth=2, label='Dense NN (Custom)')
        
        # Add speedup annotations against custom dense
        for i, (sp, sparse_t) in enumerate(zip(sparsity_levels, sparse_times)):
            if sparse_t > 0:
                speedup = dense_time / sparse_t
                plt.text(i, sparse_t * 1.05, f"{speedup:.2f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add MKL baseline as horizontal line if available
    if mkl_data:
        mkl_time = mkl_data['time_s'] * 1e6
        plt.axhline(y=mkl_time, color='#d62728', linestyle='-.', linewidth=2, label='Dense NN (MKL)')
        
        # Annotate speedup against MKL at rightmost bar
        if sparse_times:
            best_sparse_time = min(sparse_times)
            best_idx = sparse_times.index(best_sparse_time)
            mkl_speedup = mkl_time / best_sparse_time
            plt.text(best_idx, mkl_time * 0.9, f"Best: {mkl_speedup:.2f}x vs MKL", 
                    ha='center', va='top', fontsize=9, fontweight='bold', color='#d62728')
    
    plt.ylabel('Runtime (µs)')
    plt.xlabel('Sparsity Level')
    plt.title('Sparse Neural Network Performance vs Sparsity Level')
    plt.xticks(x_pos, labels)
    plt.legend(loc='best')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./plots/nn_sparsity_runtime.png', dpi=150)
    print('Saved ./plots/nn_sparsity_runtime.png')
    
    # Accuracy comparison if available
    if any(sparse_data[sp]['accuracy'] > 0 for sp in sparsity_levels):
        plt.figure(figsize=(12, 6))
        
        sparse_accuracies = [sparse_data[sp]['accuracy'] for sp in sparsity_levels]
        
        bars = plt.bar(x_pos, sparse_accuracies, width, label='Sparse NN', color='#ff7f0e')
        
        if dense_data and dense_data.get('accuracy', 0) > 0:
            dense_acc = dense_data['accuracy']
            plt.axhline(y=dense_acc, color='#1f77b4', linestyle='--', linewidth=2, label='Dense NN')
        
        if mkl_data and mkl_data.get('accuracy', 0) > 0:
            mkl_acc = mkl_data['accuracy']
            plt.axhline(y=mkl_acc, color='#d62728', linestyle='-.', linewidth=2, label='Dense NN (MKL)')
        
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Sparsity Level')
        plt.title('Neural Network Accuracy vs Sparsity Level')
        plt.xticks(x_pos, labels)
        plt.ylim([0, 100])
        plt.legend(loc='best')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('./plots/nn_sparsity_accuracy.png', dpi=150)
        print('Saved ./plots/nn_sparsity_accuracy.png')


def plot_gemm(records):
    if not records:
        print("No GEMM records to plot.")
        return
    # Sort by problem size then tile sizes
    records.sort(key=lambda r: (r['m'], r['t1'], r['t2']))
    labels = []
    med_times = []
    gflops = []
    baselines = {}  # baseline median per size (t1=-1,t2=-1)
    for r in records:
        size_label = f"{r['m']}"  # since m=n=k
        tile_label = 'baseline' if r['t1'] < 0 or r['t2'] < 0 else f"{r['t1']}x{r['t2']}"
        labels.append(f"{size_label}-{tile_label}")
        med_times.append(r['median_time_s'] * 1e6)  # plot in microseconds
        gflops.append(r['gflops'])
        if tile_label == 'baseline':
            baselines[r['m']] = r['median_time_s'] * 1e6  # store in microseconds

    os.makedirs('./plots', exist_ok=True)

    # Median time plot
    plt.figure(figsize=(16,6))
    x_pos = range(len(labels))
    bars = plt.bar(x_pos, med_times, width=0.7, color=['#444' if 'baseline' in lbl else '#1f77b4' for lbl in labels])
    plt.ylabel('Median Time (µs)')
    plt.yscale('log')
    plt.xlabel('Problem - XTileSize x YTileSize')
    plt.title('GEMM Median Time vs Tile Size')
    plt.xticks(x_pos, labels, rotation=35, ha='right')
    # annotate speedup vs baseline of same size
    for i, (bar, lbl, t) in enumerate(zip(bars, labels, med_times)):
        size = int(lbl.split('-')[0])
        base = baselines.get(size)
        if base and 'baseline' not in lbl:
            speedup = base / t if t>0 else 0
            plt.text(i, bar.get_height()*1.05, f"{speedup:.2f}x", ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig('./plots/gemm_median.png', dpi=150)
    print('Saved ./plots/gemm_median.png')

    # GFLOPs plot
    plt.figure(figsize=(16,6))
    bars2 = plt.bar(x_pos, gflops, width=0.7, color=['#444' if 'baseline' in lbl else '#2ca02c' for lbl in labels])
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('ProblemSize-Tile')
    plt.title('GEMM Throughput vs Tile Size')
    plt.xticks(x_pos, labels, rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig('./plots/gemm_gflops.png', dpi=150)
    print('Saved ./plots/gemm_gflops.png')

def plot_spmm(records):
    if not records:
        print("No SPMM records to plot.")
        return
    records.sort(key=lambda r: (r['m'], r['t1'], r['t2']))
    labels = []
    med_times = []
    gflops = []
    baselines = {}
    size_times = defaultdict(list)
    for r in records:
        size_label = f"{r['m']}"
        tile_label = 'baseline' if r['t1'] < 0 or r['t2'] < 0 else f"{r['t1']}x{r['t2']}"
        labels.append(f"{size_label}-{tile_label}")
        time_us = r['median_time_s'] * 1e6
        med_times.append(time_us)
        gflops.append(r['gflops'])
        size_times[r['m']].append(time_us)
        if tile_label == 'baseline':
            baselines[r['m']] = time_us

    os.makedirs('./plots', exist_ok=True)

    # Median time plot
    plt.figure(figsize=(16,6))
    x_pos = range(len(labels))
    bars = plt.bar(x_pos, med_times, width=0.7, color=['#444' if 'baseline' in lbl else '#ff7f0e' for lbl in labels])
    plt.ylabel('Median Time (µs)')
    plt.yscale('log')
    plt.xlabel('Problem - XTileSize x YTileSize')
    plt.title('SPMM Median Time vs Tile Size')
    plt.xticks(x_pos, labels, rotation=35, ha='right')
    # annotate speedup vs baseline if present; otherwise use best (fastest) time per size
    best_times = {s: min(ts) for s, ts in size_times.items() if ts}
    for i, (bar, lbl, t) in enumerate(zip(bars, labels, med_times)):
        size = int(lbl.split('-')[0])
        ref = baselines.get(size, best_times.get(size))
        if ref and 'baseline' not in lbl:
            speedup = ref / t if t > 0 else 0
            plt.text(i, bar.get_height()*1.05, f"{speedup:.2f}x", ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig('./plots/spmm_median.png', dpi=150)
    print('Saved ./plots/spmm_median.png')

    # GFLOPs plot
    plt.figure(figsize=(16,6))
    bars2 = plt.bar(x_pos, gflops, width=0.7, color=['#444' if 'baseline' in lbl else '#d62728' for lbl in labels])
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('ProblemSize-Tile')
    plt.title('SPMM Throughput vs Tile Size')
    plt.xticks(x_pos, labels, rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig('./plots/spmm_gflops.png', dpi=150)
    print('Saved ./plots/spmm_gflops.png')

def plot_gemv(records):
    if not records:
        print("No GEMV records to plot.")
        return
    records.sort(key=lambda r: (r['m'], r['n']))
    labels = [f"{r['m']}x{r['n']}" for r in records]
    med_times = [r['median_time_s'] * 1e6 for r in records]
    gflops = [r['gflops'] for r in records]

    os.makedirs('./plots', exist_ok=True)

    # Median time plot
    plt.figure(figsize=(14,6))
    x_pos = range(len(labels))
    plt.bar(x_pos, med_times, width=0.7, color='#9467bd')
    plt.ylabel('Median Time (µs)')
    plt.yscale('log')
    plt.xlabel('Matrix Size (m x n)')
    plt.title('GEMV Median Time vs Problem Size')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./plots/gemv_median.png', dpi=150)
    print('Saved ./plots/gemv_median.png')

    # GFLOPs plot
    plt.figure(figsize=(14,6))
    plt.bar(x_pos, gflops, width=0.7, color='#8c564b')
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Matrix Size')
    plt.title('GEMV Throughput vs Problem Size')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./plots/gemv_gflops.png', dpi=150)
    print('Saved ./plots/gemv_gflops.png')

def plot_spmv(records):
    if not records:
        print("No SPMV records to plot.")
        return
    records.sort(key=lambda r: (r['m'], r['n']))
    labels = [f"{r['m']}x{r['n']}" for r in records]
    med_times = [r['median_time_s'] * 1e6 for r in records]
    gflops = [r['gflops'] for r in records]

    os.makedirs('./plots', exist_ok=True)

    # Median time plot
    plt.figure(figsize=(14,6))
    x_pos = range(len(labels))
    plt.bar(x_pos, med_times, width=0.7, color='#e377c2')
    plt.ylabel('Median Time (µs)')
    plt.yscale('log')
    plt.xlabel('Matrix Size (m x n)')
    plt.title('SPMV Median Time vs Problem Size')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./plots/spmv_median.png', dpi=150)
    print('Saved ./plots/spmv_median.png')

    # GFLOPs plot
    plt.figure(figsize=(14,6))
    plt.bar(x_pos, gflops, width=0.7, color='#7f7f7f')
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Matrix Size')
    plt.title('SPMV Throughput vs Problem Size')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./plots/spmv_gflops.png', dpi=150)
    print('Saved ./plots/spmv_gflops.png')


# ---------------------------------------------------------------------------
# Sparse CPU optimization comparisons (baseline tile vs best tile per size)
# ---------------------------------------------------------------------------
def _group_sparse_baseline_best(records, label_fn):
    """Return ordered labels and per-size baseline/best metrics.

    Baseline: smallest tile area (t1*t2) for that size.
    Best: minimum median runtime for that size.
    label_fn: function mapping record -> string label for plots.
    """
    if not records:
        return [], [], [], []
    by_size = defaultdict(list)
    for r in records:
        by_size[(r['m'], r['n'])].append(r)

    sizes = sorted(by_size.keys())
    labels = [label_fn(m, n) for (m, n) in sizes]
    base_times = []
    best_times = []
    base_gflops = []
    best_gflops = []

    for key in sizes:
         bucket = by_size[key]
         # baseline: smallest tile area (if tiles exist, else just first entry)
         baseline = min(bucket, key=lambda r: max(1, r.get('t1', -1)) * max(1, r.get('t2', -1)))
         best = min(bucket, key=lambda r: r['median_time_s'])
         base_times.append(baseline['median_time_s'] * 1e6)
         best_times.append(best['median_time_s'] * 1e6)
         base_gflops.append(baseline['gflops'])
         best_gflops.append(best['gflops'])    
         
    return labels, base_times, best_times, base_gflops, best_gflops


def plot_spmm_optimization(records):
    """Grouped bars comparing baseline tile vs best tile per size for CPU SPMM."""
    if not records:
        print("No SPMM records to plot for optimization comparison.")
        return

    labels, base_t, best_t, base_g, best_g = _group_sparse_baseline_best(
        records, lambda m, n: f"{m}"  # m==n==k in benchmarks
    )

    x = list(range(len(labels)))
    width = 0.35
    os.makedirs('./plots', exist_ok=True)

    # Runtime plot
    plt.figure(figsize=(12, 8))
    plt.bar([xi - width / 2 for xi in x], base_t, width, label='Baseline Tile', color='#7f7f7f')
    bars = plt.bar([xi + width / 2 for xi in x], best_t, width, label='Best Tile', color='#ff7f0e')
    plt.yscale('log')
    plt.ylabel('Runtime (µs)')
    plt.xlabel('Matrix Size (m=n=k)')
    plt.title('SPMM Runtime by Optimization Strategy (CPU)')
    plt.xticks(x, labels)
    for i, bar in enumerate(bars):
        if best_t[i] > 0:
            speedup = base_t[i] / best_t[i]
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{speedup:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./plots/spmm_optimization_runtime.png', dpi=150)
    print('Saved ./plots/spmm_optimization_runtime.png')

    # Throughput plot
    plt.figure(figsize=(12, 8))
    plt.bar([xi - width / 2 for xi in x], base_g, width, label='Baseline Tile', color='#7f7f7f')
    bars = plt.bar([xi + width / 2 for xi in x], best_g, width, label='Best Tile', color='#d62728')
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Matrix Size (m=n=k)')
    plt.title('SPMM Throughput by Optimization Strategy (CPU)')
    plt.xticks(x, labels)
    for i, bar in enumerate(bars):
        if base_g[i] > 0:
            speedup = best_g[i] / base_g[i]
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{speedup:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./plots/spmm_optimization_gflops.png', dpi=150)
    print('Saved ./plots/spmm_optimization_gflops.png')


def plot_spmv_optimization(records):
    """Grouped bars comparing baseline tile vs best tile per size for CPU SPMV."""
    if not records:
        print("No SPMV records to plot for optimization comparison.")
        return

    labels, base_t, best_t, base_g, best_g = _group_sparse_baseline_best(
        records, lambda m, n: f"{m}x{n}"
    )

    x = list(range(len(labels)))
    width = 0.35
    os.makedirs('./plots', exist_ok=True)

    # Runtime plot
    plt.figure(figsize=(12, 8))
    plt.bar([xi - width / 2 for xi in x], base_t, width, label='Baseline Tile', color='#7f7f7f')
    bars = plt.bar([xi + width / 2 for xi in x], best_t, width, label='Best Tile', color='#17becf')
    plt.yscale('log')
    plt.ylabel('Runtime (µs)')
    plt.xlabel('Matrix Size (m x n)')
    plt.title('SPMV Runtime by Optimization Strategy (CPU)')
    plt.xticks(x, labels, rotation=20, ha='right')
    for i, bar in enumerate(bars):
        if best_t[i] > 0:
            speedup = base_t[i] / best_t[i]
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{speedup:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./plots/spmv_optimization_runtime.png', dpi=150)
    print('Saved ./plots/spmv_optimization_runtime.png')

    # Throughput plot
    plt.figure(figsize=(12, 8))
    plt.bar([xi - width / 2 for xi in x], base_g, width, label='Baseline Tile', color='#7f7f7f')
    bars = plt.bar([xi + width / 2 for xi in x], best_g, width, label='Best Tile', color='#bcbd22')
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Matrix Size (m x n)')
    plt.title('SPMV Throughput by Optimization Strategy (CPU)')
    plt.xticks(x, labels, rotation=20, ha='right')
    for i, bar in enumerate(bars):
        if base_g[i] > 0:
            speedup = best_g[i] / base_g[i]
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{speedup:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./plots/spmv_optimization_gflops.png', dpi=150)
    print('Saved ./plots/spmv_optimization_gflops.png')

def collect_variant_data(json_path):
    """Collect variant-specific records for GEMM/GEMV.
    Returns dicts keyed by size containing technique metrics.
    Techniques: 'naive', 'simd', 'simd_par', 'tiled_best'.
    """
    if not os.path.exists(json_path):
        return {}, {}
    with open(json_path, 'r') as f:
        data = json.load(f)
    bench = data.get('benchmarks', [])
    time_unit = data.get('context', {}).get('time_unit', 'us')
    unit_scale = 1e-6 if time_unit == 'us' else 1e-9 if time_unit == 'ns' else 1e-6

    gemm_variants = defaultdict(lambda: defaultdict(list))  # size m -> technique -> list of (time_s, gflops)
    gemm_tiled = defaultdict(list)  # size m -> list of (t1,t2,time_s,gflops)
    gemv_variants = defaultdict(lambda: defaultdict(list))  # size m -> technique -> list of (time_s, gflops)

    for entry in bench:
        name = entry.get('name', '')
        parts = name.split('/')
        if not parts:
            continue
        run_type = entry.get('run_type')
        if run_type == 'aggregate' and entry.get('aggregate_name') != 'median':
            continue
        real = entry.get('real_time')
        if real is None:
            continue
        time_s = real * unit_scale

        # GEMM variants
        if parts[0] in ('BM_GEMM_NAIVE', 'BM_GEMM_SIMD', 'BM_GEMM_SIMD_PAR', 'BM_GEMM_TILED_SIMD_PAR'):
            # parse first 5 ints after name
            nums = []
            for p in parts[1:]:
                token = p.split(':')[0]
                try:
                    nums.append(int(token))
                except ValueError:
                    pass
            if parts[0] == 'BM_GEMM_TILED_SIMD_PAR':
                if len(nums) < 5:
                    continue
                m,n,k,t1,t2 = nums[:5]
            else:
                if len(nums) < 3:
                    continue
                m,n,k = nums[:3]
                t1=t2=-1
            gflops = (2.0*m*n*k) / time_s / 1e9
            size = m  # square sizes
            if parts[0] == 'BM_GEMM_TILED_SIMD_PAR':
                gemm_tiled[size].append((t1,t2,time_s,gflops))
            else:
                technique = 'naive' if parts[0]=='BM_GEMM_NAIVE' else 'simd' if parts[0]=='BM_GEMM_SIMD' else 'simd_par'
                gemm_variants[size][technique].append((time_s, gflops))

        # GEMV variants
        if parts[0] in ('BM_GEMV_NAIVE', 'BM_GEMV_SIMD', 'BM_GEMV_SIMD_PAR'):
            nums = []
            for p in parts[1:]:
                token = p.split(':')[0]
                try:
                    nums.append(int(token))
                except ValueError:
                    pass
            if len(nums) < 2:
                continue
            m,n = nums[:2]
            gflops = (2.0*m*n) / time_s / 1e9
            size = m
            technique = 'naive' if parts[0]=='BM_GEMV_NAIVE' else 'simd' if parts[0]=='BM_GEMV_SIMD' else 'simd_par'
            gemv_variants[size][technique].append((time_s, gflops))

    # Reduce lists to medians and select best tile per size
    gemm_result = {}
    for size, techs in gemm_variants.items():
        out = {}
        for tname, vals in techs.items():
            if not vals:
                continue
            med_time = statistics.median(v[0] for v in vals)
            med_gflops = statistics.median(v[1] for v in vals)
            out[tname] = {'time_s': med_time, 'gflops': med_gflops}
        # choose best tiled (min time)
        if gemm_tiled.get(size):
            best = min(gemm_tiled[size], key=lambda x: x[2])
            t1,t2,time_s,gflops = best
            out['tiled_best'] = {'time_s': time_s, 'gflops': gflops, 'tile': (t1,t2)}
        gemm_result[size] = out

    gemv_result = {}
    for size, techs in gemv_variants.items():
        out = {}
        for tname, vals in techs.items():
            if not vals:
                continue
            med_time = statistics.median(v[0] for v in vals)
            med_gflops = statistics.median(v[1] for v in vals)
            out[tname] = {'time_s': med_time, 'gflops': med_gflops}
        gemv_result[size] = out

    return gemm_result, gemv_result

def plot_optimization_stacks(json_path):
    """Generate stacked bar plots across sizes for optimization techniques.
    For GEMM, includes naive, simd, simd_par, and best tiled per size (always use best tile even if suboptimal for smaller sizes).
    For GEMV, includes naive, simd, simd_par.
    Sweeps across all available matrix sizes: 64, 128, 256, 512, 1024, 4096.
    Produces both runtime (µs) and throughput (GFLOPs) stacked bars.
    """
    gemm_data, gemv_data = collect_variant_data(json_path)
    os.makedirs('./plots', exist_ok=True)

    # Target sizes for sweep (use these if available in data)
    target_sizes = [64, 128, 256, 512, 1024, 4096]

    # GEMM grouped runtime - sweep across all target sizes
    if gemm_data:
        # Filter to only available sizes from data, in order
        available_sizes = sorted([s for s in target_sizes if s in gemm_data])
        if not available_sizes:
            available_sizes = sorted(gemm_data.keys())
        
        labels = [str(s) for s in available_sizes]
        # Get independent measurements for each technique
        naive = [gemm_data[s].get('naive', {}).get('time_s', 0.0)*1e6 for s in available_sizes]
        simd = [gemm_data[s].get('simd', {}).get('time_s', 0.0)*1e6 for s in available_sizes]
        simd_par = [gemm_data[s].get('simd_par', {}).get('time_s', 0.0)*1e6 for s in available_sizes]
        tiled = [gemm_data[s].get('tiled_best', {}).get('time_s', 0.0)*1e6 for s in available_sizes]
        
        plt.figure(figsize=(14, 10))
        x = range(len(available_sizes))
        width = 0.2
        x_offset = [(x_i - 1.5*width) for x_i in x]
        
        p1 = plt.bar([xi + 0*width for xi in x_offset], naive, width, label='Naive', color='#1f77b4')
        p2 = plt.bar([xi + 1*width for xi in x_offset], simd, width, label='SIMD', color='#2ca02c')
        p3 = plt.bar([xi + 2*width for xi in x_offset], simd_par, width, label='SIMD+OMP', color='#ff7f0e')
        p4 = plt.bar([xi + 3*width for xi in x_offset], tiled, width, label='Best Tiled', color='#d62728')
        
        # Add speedup annotations for best tiled vs naive
        for i in range(len(available_sizes)):
            if naive[i] > 0 and tiled[i] > 0:
                speedup = naive[i] / tiled[i]
                y_pos = tiled[i]
                plt.text(x[i] + 1.5*width, y_pos * 1.15, f"{speedup:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.ylabel('Runtime (µs)')
        plt.xlabel('Matrix Size (m=n=k)')
        plt.title('GEMM Runtime by Optimization Technique (Best Tiling per Size)')
        plt.yscale('log')
        plt.xticks(x, labels)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('./plots/gemm_optimization_stacked_runtime.png', dpi=150)
        print('Saved ./plots/gemm_optimization_stacked_runtime.png')

        # GEMM grouped throughput
        naive_t = [gemm_data[s].get('naive', {}).get('gflops', 0.0) for s in available_sizes]
        simd_t = [gemm_data[s].get('simd', {}).get('gflops', 0.0) for s in available_sizes]
        simd_par_t = [gemm_data[s].get('simd_par', {}).get('gflops', 0.0) for s in available_sizes]
        tiled_t = [gemm_data[s].get('tiled_best', {}).get('gflops', 0.0) for s in available_sizes]
        
        plt.figure(figsize=(14, 10))
        x_offset = [(x_i - 1.5*width) for x_i in x]
        
        p1 = plt.bar([xi + 0*width for xi in x_offset], naive_t, width, label='Naive', color='#1f77b4')
        p2 = plt.bar([xi + 1*width for xi in x_offset], simd_t, width, label='SIMD', color='#2ca02c')
        p3 = plt.bar([xi + 2*width for xi in x_offset], simd_par_t, width, label='SIMD+OMP', color='#ff7f0e')
        p4 = plt.bar([xi + 3*width for xi in x_offset], tiled_t, width, label='Best Tiled', color='#d62728')
        
        # Add speedup annotations for best tiled vs naive (throughput ratio)
        for i in range(len(available_sizes)):
            if naive_t[i] > 0 and tiled_t[i] > 0:
                speedup = tiled_t[i] / naive_t[i]
                y_pos = tiled_t[i]
                plt.text(x[i] + 1.5*width, y_pos * 1.05, f"{speedup:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.ylabel('Throughput (GFLOPs)')
        plt.xlabel('Matrix Size (m=n=k)')
        plt.title('GEMM Throughput by Optimization Technique (Best Tiling per Size)')
        plt.xticks(x, labels)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('./plots/gemm_optimization_stacked_throughput.png', dpi=150)
        print('Saved ./plots/gemm_optimization_stacked_throughput.png')

    # GEMV grouped runtime - sweep across all target sizes
    if gemv_data:
        available_sizes = sorted([s for s in target_sizes if s in gemv_data])
        if not available_sizes:
            available_sizes = sorted(gemv_data.keys())
        
        labels = [str(s) for s in available_sizes]
        naive = [gemv_data[s].get('naive', {}).get('time_s', 0.0)*1e6 for s in available_sizes]
        simd = [gemv_data[s].get('simd', {}).get('time_s', 0.0)*1e6 for s in available_sizes]
        simd_par = [gemv_data[s].get('simd_par', {}).get('time_s', 0.0)*1e6 for s in available_sizes]
        
        plt.figure(figsize=(14, 10))
        x = range(len(available_sizes))
        width = 0.25
        x_offset = [(x_i - 1.0*width) for x_i in x]
        
        p1 = plt.bar([xi + 0*width for xi in x_offset], naive, width, label='Naive', color='#1f77b4')
        p2 = plt.bar([xi + 1*width for xi in x_offset], simd, width, label='SIMD', color='#2ca02c')
        p3 = plt.bar([xi + 2*width for xi in x_offset], simd_par, width, label='SIMD+OMP', color='#ff7f0e')
        
        # Add speedup annotations for SIMD+OMP vs naive
        for i in range(len(available_sizes)):
            if naive[i] > 0 and simd_par[i] > 0:
                speedup = naive[i] / simd_par[i]
                y_pos = simd_par[i]
                plt.text(x[i] + 1.0*width, y_pos * 1.15, f"{speedup:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.ylabel('Runtime (µs)')
        plt.xlabel('Matrix Size (m=n)')
        plt.title('GEMV Runtime by Optimization Technique')
        plt.yscale('log')
        plt.xticks(x, labels)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('./plots/gemv_optimization_stacked_runtime.png', dpi=150)
        print('Saved ./plots/gemv_optimization_stacked_runtime.png')

        naive_t = [gemv_data[s].get('naive', {}).get('gflops', 0.0) for s in available_sizes]
        simd_t = [gemv_data[s].get('simd', {}).get('gflops', 0.0) for s in available_sizes]
        simd_par_t = [gemv_data[s].get('simd_par', {}).get('gflops', 0.0) for s in available_sizes]
        
        plt.figure(figsize=(14, 10))
        x_offset = [(x_i - 1.0*width) for x_i in x]
        
        p1 = plt.bar([xi + 0*width for xi in x_offset], naive_t, width, label='Naive', color='#1f77b4')
        p2 = plt.bar([xi + 1*width for xi in x_offset], simd_t, width, label='SIMD', color='#2ca02c')
        p3 = plt.bar([xi + 2*width for xi in x_offset], simd_par_t, width, label='SIMD+OMP', color='#ff7f0e')
        
        # Add speedup annotations for SIMD+OMP vs naive (throughput ratio)
        for i in range(len(available_sizes)):
            if naive_t[i] > 0 and simd_par_t[i] > 0:
                speedup = simd_par_t[i] / naive_t[i]
                y_pos = simd_par_t[i]
                plt.text(x[i] + 1.0*width, y_pos * 1.05, f"{speedup:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.ylabel('Throughput (GFLOPs)')
        plt.xlabel('Matrix Size (m=n)')
        plt.title('GEMV Throughput by Optimization Technique')
        plt.xticks(x, labels)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('./plots/gemv_optimization_stacked_throughput.png', dpi=150)
        print('Saved ./plots/gemv_optimization_stacked_throughput.png')

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else './logs/project.json'
    
    # Collect and plot all benchmark types
    print("Processing benchmarks from:", path)
    
    gemm_records = collect_gemm(path)
    print(f"Found {len(gemm_records)} GEMM configurations")
    plot_gemm(gemm_records)
    
    spmm_records = collect_spmm(path)
    print(f"Found {len(spmm_records)} SPMM configurations")
    plot_spmm(spmm_records)
    plot_spmm_optimization(spmm_records)
    
    gemv_records = collect_gemv(path)
    print(f"Found {len(gemv_records)} GEMV configurations")
    plot_gemv(gemv_records)
    
    spmv_records = collect_spmv(path)
    print(f"Found {len(spmv_records)} SPMV configurations")
    plot_spmv(spmv_records)
    plot_spmv_optimization(spmv_records)
    
    spmv_records = collect_spmv(path)
    print(f"Found {len(spmv_records)} SPMV configurations")
    plot_spmv(spmv_records)
    
    nn_data = collect_nn(path)
    print(f"Found NN data: Dense={bool(nn_data['dense'])}, Sparse sparsity levels={sorted(nn_data['sparse'].keys())}")
    plot_nn(nn_data)

    # New stacked bar plot comparing optimization techniques across sizes
    plot_optimization_stacks(path)
    
    print("\nAll plots saved to ./plots/")

if __name__ == '__main__':
    main()
