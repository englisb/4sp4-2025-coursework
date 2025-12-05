
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
    """Return (m,n,k,sparsity,t1,t2) parsed from BM_SPMM_* variant names.
    
    New format: BM_SPMM_NAIVE, BM_SPMM_SIMD, BM_SPMM_SIMD_PAR, BM_SPMM_TILED
    Map variant name to t1 code:
    - NAIVE -> t1=-1
    - SIMD -> t1=0
    - SIMD_PAR -> t1=1
    - TILED -> t1=16
    """
    parts = name.split('/')
    if not parts:
        return None
    
    benchmark_name = parts[0]
    if not benchmark_name.startswith('BM_SPMM_'):
        return None
    
    # Extract variant from benchmark name
    variant_part = benchmark_name.replace('BM_SPMM_', '')
    if variant_part == 'NAIVE':
        t1 = -1
    elif variant_part == 'SIMD':
        t1 = 0
    elif variant_part == 'SIMD_PAR':
        t1 = 1
    elif variant_part == 'TILED':
        t1 = 16
    else:
        return None
    
    # Parse arguments (should be m,n,k,sparsity)
    nums = []
    for p in parts[1:]:
        token = p.split(':')[0]
        try:
            nums.append(int(token))
            if len(nums) == 4:
                break
        except ValueError:
            continue
    if len(nums) != 4:
        return None
    
    m, n, k, sparsity = nums
    t2 = 0  # t2 is unused for sparse benchmarks
    return (m, n, k, sparsity, t1, t2)

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
    """Return (m,n,sparsity,t1,t2) parsed from BM_SPMV_* variant names.
    
    New format: BM_SPMV_NAIVE, BM_SPMV_PARALLEL, BM_SPMV_TILED
    Map variant name to t1 code:
    - NAIVE -> t1=-1
    - PARALLEL -> t1=0
    - TILED -> t1=16
    """
    parts = name.split('/')
    if not parts:
        return None
    
    benchmark_name = parts[0]
    if not benchmark_name.startswith('BM_SPMV_'):
        return None
    
    # Extract variant from benchmark name
    variant_part = benchmark_name.replace('BM_SPMV_', '')
    if variant_part == 'NAIVE':
        t1 = -1
    elif variant_part == 'PARALLEL':
        t1 = 0
    elif variant_part == 'TILED':
        t1 = 16
    else:
        return None
    
    # Parse arguments (should be m,n,sparsity)
    nums = []
    for p in parts[1:]:
        token = p.split(':')[0]
        try:
            nums.append(int(token))
            if len(nums) == 3:
                break
        except ValueError:
            continue
    if len(nums) != 3:
        return None
    
    m, n, sparsity = nums
    t2 = 0  # t2 is unused for sparse benchmarks
    return (m, n, sparsity, t1, t2)


def collect_gemm(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is nvbench format (GPU benchmarks)
    if is_nvbench_format(data):
        return collect_gemm_nvbench(data)
    
    # Google Benchmark format (CPU benchmarks)
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
    """Collect SPMM benchmark data with sparsity levels."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is nvbench format (GPU benchmarks)
    if is_nvbench_format(data):
        return collect_spmm_nvbench(data)
    
    # Google Benchmark format (CPU benchmarks)
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
        m,n,k,sparsity,t1,t2 = key
        variant = get_spmm_variant_name(t1, t2)
        records.append({
            'm': m, 'n': n, 'k': k, 'sparsity': sparsity, 't1': t1, 't2': t2,
            'variant': variant,
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
    
    # Check if this is nvbench format (GPU benchmarks)
    if is_nvbench_format(data):
        return collect_gemv_nvbench(data)
    
    # Google Benchmark format (CPU benchmarks)
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
    """Collect SPMV benchmark data with sparsity levels."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is nvbench format (GPU benchmarks)
    if is_nvbench_format(data):
        return collect_spmv_nvbench(data)
    
    # Google Benchmark format (CPU benchmarks)
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
        m, n, sparsity, t1, t2 = key
        variant = get_spmv_variant_name(t1, t2)
        records.append({
            'm': m, 'n': n, 'sparsity': sparsity, 't1': t1, 't2': t2,
            'variant': variant,
            'median_time_s': med * unit_scale,
            'gflops': (2.0*m*n) / (med * unit_scale) / 1e9  # 2*m*n operations for SPMV
        })
    return records

def is_nvbench_format(data):
    """Check if JSON data is in nvbench format (GPU benchmarks) vs Google Benchmark format."""
    return 'meta' in data and 'nvbench' in data.get('meta', {}).get('version', {})

def collect_gemm_nvbench(data):
    """Parse GEMM benchmarks from nvbench format (GPU)."""
    benchmarks = data.get('benchmarks', [])
    grouped = {}
    
    for benchmark in benchmarks:
        bench_name = benchmark.get('name', '')
        if 'GEMM' not in bench_name:
            continue
        
        states = benchmark.get('states', [])
        for state in states:
            if not isinstance(state, dict):
                continue
            # Extract time from summaries - use GPU mean time
            summaries = state.get('summaries', [])
            time_s = None
            for summary in summaries:
                if summary.get('tag') == 'nv/cold/time/gpu/mean':
                    data_val = summary['data'][0]['value']
                    time_s = float(data_val) if isinstance(data_val, str) else data_val
                    break
            if time_s is None:
                for summary in summaries:
                    if summary.get('tag') == 'nv/cold/time/cpu/mean':
                        data_val = summary['data'][0]['value']
                        time_s = float(data_val) if isinstance(data_val, str) else data_val
                        break
            if time_s is None:
                continue
            
            # Extract axis values - nvbench typically uses 'n' for square matrices
            axis_values = {av['name']: int(av['value']) for av in state.get('axis_values', [])}
            n = axis_values.get('n', axis_values.get('m', 0))
            k = axis_values.get('k', n)
            m = axis_values.get('m', n)
            
            # No tiling info in GPU benchmarks
            key = (m, n, k, -1, -1)
            grouped.setdefault(key, []).append(time_s)
    
    records = []
    for key, times in grouped.items():
        if not times:
            continue
        med = statistics.median(times)
        m, n, k, t1, t2 = key
        records.append({
            'm': m, 'n': n, 'k': k, 't1': t1, 't2': t2,
            'median_time_s': med,
            'gflops': (2.0*m*n*k) / med / 1e9
        })
    return records

def collect_spmm_nvbench(data):
    """Parse SPMM benchmarks from nvbench format (GPU)."""
    benchmarks = data.get('benchmarks', [])
    grouped = {}
    
    for benchmark in benchmarks:
        bench_name = benchmark.get('name', '')
        if 'SpMM' not in bench_name:
            continue

        # Capture variant (Baseline/Coalesced/Shared/Combined) from the name
        variant = bench_name.replace('SpMM_', '') if 'SpMM_' in bench_name else bench_name
        
        states = benchmark.get('states', [])
        for state in states:
            summaries = state.get('summaries', [])
            time_s = None
            for summary in summaries:
                if summary.get('tag') == 'nv/cold/time/gpu/mean':
                    data_val = summary['data'][0]['value']
                    time_s = float(data_val) if isinstance(data_val, str) else data_val
                    break
            if time_s is None:
                for summary in summaries:
                    if summary.get('tag') == 'nv/cold/time/cpu/mean':
                        data_val = summary['data'][0]['value']
                        time_s = float(data_val) if isinstance(data_val, str) else data_val
                        break
            if time_s is None:
                continue
            
            axis_values = {av['name']: int(float(av['value'])) for av in state.get('axis_values', [])}
            n = axis_values.get('n', 0)
            k = axis_values.get('k', n)
            # nvbench only provides `n` for square inputs; treat m=n by default
            m = axis_values.get('m', n if n else 0)
            sparsity = int(axis_values.get('sparsity', 0))
            block_size = axis_values.get('block_size', -1)
            
            key = (variant, m, n, k, sparsity, block_size, -1)
            grouped.setdefault(key, []).append(time_s)
    
    records = []
    for key, times in grouped.items():
        if not times:
            continue
        med = statistics.median(times)
        variant, m, n, k, sparsity, block_size, t2 = key
        records.append({
            'variant': variant,
            'm': m, 'n': n, 'k': k, 'sparsity': sparsity, 't1': block_size, 't2': t2,
            'median_time_s': med,
            'gflops': (2.0*m*n*k) / med / 1e9
        })
    return records

def collect_gemv_nvbench(data):
    """Parse GEMV benchmarks from nvbench format (GPU)."""
    benchmarks = data.get('benchmarks', [])
    grouped = {}
    
    for benchmark in benchmarks:
        bench_name = benchmark.get('name', '')
        if 'GEMV' not in bench_name:
            continue
        
        states = benchmark.get('states', [])
        for state in states:
            summaries = state.get('summaries', [])
            time_s = None
            for summary in summaries:
                if summary.get('tag') == 'nv/cold/time/gpu/mean':
                    data_val = summary['data'][0]['value']
                    time_s = float(data_val) if isinstance(data_val, str) else data_val
                    break
            if time_s is None:
                for summary in summaries:
                    if summary.get('tag') == 'nv/cold/time/cpu/mean':
                        data_val = summary['data'][0]['value']
                        time_s = float(data_val) if isinstance(data_val, str) else data_val
                        break
            if time_s is None:
                continue
            
            axis_values = state.get('axis_values') or []
            n = None

            # GEMV uses only 'n' axis for n x n matrix; some nvbench outputs omit axis_values
            for av in axis_values:
                name = av.get('name') if isinstance(av, dict) else None
                try:
                    val = int(av.get('value')) if isinstance(av, dict) else None
                except (ValueError, TypeError):
                    val = None
                if val is None:
                    continue
                if name == 'n':
                    n = val
                elif n is None:
                    n = val  # First axis defaults to n

            # If no axes were provided, fall back to treating GEMV as 1x1 to avoid crash
            if n is None:
                n = 1
            
            if n is None:
                continue
            
            # For GEMV: m=n, n=n (square matrix)
            key = (n, n)
            grouped.setdefault(key, []).append(time_s)
    
    records = []
    for key, times in grouped.items():
        if not times:
            continue
        med = statistics.median(times)
        m, n = key
        records.append({
            'm': m, 'n': n,
            'median_time_s': med,
            'gflops': (2.0*m*n) / med / 1e9
        })
    return records

def collect_spmv_nvbench(data):
    """Parse SPMV benchmarks from nvbench format (GPU)."""
    benchmarks = data.get('benchmarks', [])
    grouped = {}
    
    for benchmark in benchmarks:
        bench_name = benchmark.get('name', '')
        if 'SpMV' not in bench_name:
            continue

        # Capture variant (Baseline/Coalesced/WarpLevel/Combined) from the name
        variant = bench_name.replace('SpMV_', '') if 'SpMV_' in bench_name else bench_name
        
        states = benchmark.get('states', [])
        for state in states:
            summaries = state.get('summaries', [])
            time_s = None
            for summary in summaries:
                if summary.get('tag') == 'nv/cold/time/gpu/mean':
                    data_val = summary['data'][0]['value']
                    time_s = float(data_val) if isinstance(data_val, str) else data_val
                    break
            if time_s is None:
                for summary in summaries:
                    if summary.get('tag') == 'nv/cold/time/cpu/mean':
                        data_val = summary['data'][0]['value']
                        time_s = float(data_val) if isinstance(data_val, str) else data_val
                        break
            if time_s is None:
                continue
            
            axis_values_list = state.get('axis_values') or []
            axis_values = {}
            for av in axis_values_list:
                if not isinstance(av, dict):
                    continue
                name = av.get('name')
                try:
                    val = int(float(av.get('value')))
                except (ValueError, TypeError):
                    continue
                axis_values[name] = val

            n = axis_values.get('n')
            # nvbench often only provides `n`; treat matrices as square so m=n
            if n is None:
                n = 1
            m = axis_values.get('m', n)
            sparsity = int(axis_values.get('sparsity', 0))
            block_size = axis_values.get('block_size', -1)
            
            key = (variant, m, n, sparsity, block_size, -1)
            grouped.setdefault(key, []).append(time_s)
    
    records = []
    for key, times in grouped.items():
        if not times:
            continue
        med = statistics.median(times)
        variant, m, n, sparsity, block_size, t2 = key
        records.append({
            'variant': variant,
            'm': m, 'n': n, 'sparsity': sparsity, 't1': block_size, 't2': t2,
            'median_time_s': med,
            'gflops': (2.0*m*n) / med / 1e9
        })
    return records

def collect_nn_nvbench(json_path):
    """Collect neural network benchmark data from nvbench format (GPU benchmarks).
    Returns dict with 'dense' and 'sparse' keys.
    """
    if not os.path.exists(json_path):
        return {'dense': {}, 'sparse': {}, 'mkl': {}}
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    benchmarks = data.get('benchmarks', [])
    dense_records = []
    sparse_records = defaultdict(list)
    
    for benchmark in benchmarks:
        bench_name = benchmark.get('name', '')
        states = benchmark.get('states', [])
        
        for state in states:
            if not isinstance(state, dict):
                continue
            # Extract time from summaries - use GPU mean time
            summaries = state.get('summaries', [])
            time_s = None
            for summary in summaries:
                if summary.get('tag') == 'nv/cold/time/gpu/mean':
                    try:
                        time_s = float(summary['data'][0]['value'])
                    except (ValueError, TypeError, KeyError, IndexError):
                        time_s = None
                    break
            
            # Fallback to CPU time if GPU time not found
            if time_s is None:
                for summary in summaries:
                    if summary.get('tag') == 'nv/cold/time/cpu/mean':
                        try:
                            time_s = float(summary['data'][0]['value'])
                        except (ValueError, TypeError, KeyError, IndexError):
                            time_s = None
                        break
            
            if time_s is None:
                continue
            
            # Extract axis values (like batch_size, sparsity)
            axis_list = state.get('axis_values') or []
            if not isinstance(axis_list, list):
                axis_list = []
            axis_values = {}
            for av in axis_list:
                if not isinstance(av, dict):
                    continue
                name = av.get('name')
                value = av.get('value')
                if name is None:
                    continue
                try:
                    axis_values[name] = int(value)
                except (ValueError, TypeError):
                    try:
                        axis_values[name] = float(value)
                    except (ValueError, TypeError):
                        continue
            
            if 'GPU_Dense_NN' in bench_name:
                dense_records.append({'time_s': time_s, 'accuracy': 0.0})
            elif 'GPU_Sparse_NN' in bench_name:
                sparsity = axis_values.get('sparsity')
                if sparsity is not None:
                    sparse_records[sparsity].append({'time_s': time_s, 'accuracy': 0.0})
    
    # Aggregate results
    result = {'dense': {}, 'sparse': {}, 'mkl': {}}
    
    if dense_records:
        median_time = statistics.median([float(r['time_s']) for r in dense_records if r.get('time_s') is not None])
        result['dense'] = {
            'time_s': median_time,
            'accuracy': 0.0  # GPU benchmarks don't report accuracy in JSON
        }
    
    for sparsity, records in sparse_records.items():
        if records:
            median_time = statistics.median([float(r['time_s']) for r in records if r.get('time_s') is not None])
            result['sparse'][sparsity] = {
                'time_s': median_time,
                'accuracy': 0.0
            }
    
    return result

def collect_nn(json_path):
    """Collect neural network benchmark data (both dense and sparse).
    Returns dict with 'dense' and 'sparse' keys, each containing sparsity->time/accuracy mappings.
    Also includes 'mkl' key for MKL dense baseline if available.
    Handles both Google Benchmark format (CPU) and nvbench format (GPU).
    """
    if not os.path.exists(json_path):
        return {'dense': {}, 'sparse': {}, 'mkl': {}}
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is nvbench format (GPU benchmarks)
    if is_nvbench_format(data):
        return collect_nn_nvbench(json_path)
    # Google Benchmark format (CPU benchmarks)
    bench = data.get('benchmarks', [])
    
    dense_records = []
    sparse_records = {}  # sparsity -> optimization -> list of times
    mkl_records = []
    
    time_unit = data.get('context', {}).get('time_unit', 'us')
    if not time_unit and bench:
        time_unit = bench[0].get('time_unit', 'us')
    unit_scale = 1e-6 if time_unit == 'us' else 1e-9 if time_unit == 'ns' else 1e-6
    
    def get_opt_label(tile1, tile2):
        """Get optimization technique label from tile parameters."""
        if tile1 < 0:
            return 'Naive'
        elif tile1 == 0:
            return 'SIMD'
        elif tile1 == 1:
            return 'SIMD+Parallel'
        else:
            return f'Tiled+SIMD+Parallel ({tile1}x{tile2})'
    
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
                # Prefer args if present; otherwise parse sparsity from the name tokens
                args = entry.get('args', []) or []
                sparsity = None
                tile1 = -1
                tile2 = 0

                if len(args) >= 4:
                    try:
                        sparsity = int(args[3])
                        if len(args) > 4:
                            tile1 = int(args[4])
                        if len(args) > 5:
                            tile2 = int(args[5])
                    except (ValueError, TypeError):
                        sparsity = None

                # Fallback to parsing from name: BM_SPARSENN/m/n/sparsity
                if sparsity is None:
                    parts = name.split('/')
                    if len(parts) >= 4:
                        try:
                            sparsity = int(parts[3])
                        except (ValueError, TypeError):
                            sparsity = None

                if sparsity is None:
                    continue

                opt_label = get_opt_label(tile1, tile2)

                if run_type in ['aggregate', 'iteration']:
                    time_val = entry.get('real_time', 0)
                    accuracy = entry.get('Accuracy', 0.0)
                    if sparsity not in sparse_records:
                        sparse_records[sparsity] = {}
                    if opt_label not in sparse_records[sparsity]:
                        sparse_records[sparsity][opt_label] = []
                    sparse_records[sparsity][opt_label].append({
                        'time_s': time_val * unit_scale,
                        'accuracy': accuracy
                    })
        
        # Sparsity comparison benchmarks (Magnitude-based pruning)
        elif 'BM_SPARSITY_COMPARISON_MAGNITUDE' in name:
                # Similar structure: args has {m, n, sparsity, tile1, tile2}; fallback to name tokens if missing
                args = entry.get('args', []) or []
                sparsity = None
                tile1 = -1
                tile2 = 0

                if len(args) >= 3:
                    try:
                        sparsity = int(args[2])
                        if len(args) > 3:
                            tile1 = int(args[3])
                        if len(args) > 4:
                            tile2 = int(args[4])
                    except (ValueError, TypeError):
                        sparsity = None

                if sparsity is None:
                    parts = name.split('/')
                    if len(parts) >= 4:
                        try:
                            sparsity = int(parts[3])
                        except (ValueError, TypeError):
                            sparsity = None

                if sparsity is None:
                    continue

                opt_label = get_opt_label(tile1, tile2)

                if run_type in ['aggregate', 'iteration']:
                    time_val = entry.get('real_time', 0)
                    accuracy = entry.get('Accuracy', 0.0)
                    if sparsity not in sparse_records:
                        sparse_records[sparsity] = {}
                    if opt_label not in sparse_records[sparsity]:
                        sparse_records[sparsity][opt_label] = []
                    sparse_records[sparsity][opt_label].append({
                        'time_s': time_val * unit_scale,
                        'accuracy': accuracy
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
    
    # For sparse, keep the optimization technique breakdown
    for sparsity, opt_dict in sparse_records.items():
        result['sparse'][sparsity] = {}
        for opt_label, records in opt_dict.items():
            if records:
                median_time = statistics.median([r['time_s'] for r in records])
                avg_accuracy = statistics.mean([r['accuracy'] for r in records]) if records[0]['accuracy'] > 0 else 0
                result['sparse'][sparsity][opt_label] = {
                    'time_s': median_time,
                    'accuracy': avg_accuracy
                }
    
    return result

def plot_nn(nn_data, suffix=''):
    """Plot neural network benchmark results comparing different sparse optimization techniques."""
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
    
    # Collect all optimization techniques (or default None for single-series nvbench format)
    all_techniques = set()
    for sp in sparsity_levels:
        entry = sparse_data.get(sp)
        if not isinstance(entry, dict):
            continue
        if 'time_s' in entry:
            # nvbench single-series format
            all_techniques.add(None)
        else:
            all_techniques.update(entry.keys())

    if not all_techniques:
        all_techniques = {None}

    all_techniques = sorted(list(all_techniques), key=lambda x: '' if x is None else x)
    
    # Runtime comparison
    plt.figure(figsize=(14, 6))
    
    colors = plt.cm.Set3(range(len(all_techniques)))
    x_pos = range(len(sparsity_levels))
    width = 0.15 if len(all_techniques) > 4 else 0.2
    
    for idx, technique in enumerate(all_techniques):
        times = []
        for sp in sparsity_levels:
            entry = sparse_data.get(sp, {})
            if technique is None:
                # Single-series (nvbench) case: expect a dict with time_s
                if isinstance(entry, dict) and 'time_s' in entry:
                    times.append(entry['time_s'] * 1e6)
                else:
                    times.append(0)
            else:
                if isinstance(entry, dict) and technique in entry:
                    times.append(entry[technique].get('time_s', 0) * 1e6)
                else:
                    times.append(0)
        
        offset = width * (idx - len(all_techniques)/2 + 0.5)
        bars = plt.bar([p + offset for p in x_pos], times, width, label=technique if technique else 'Sparse NN', color=colors[idx])
    
    # Add dense baseline as horizontal line if available
    if dense_data and 'time_s' in dense_data:
        dense_time = dense_data['time_s'] * 1e6
        plt.axhline(y=dense_time, color='#1f77b4', linestyle='--', linewidth=2, label='Dense NN (Custom)')
    
    # Add MKL baseline as horizontal line if available
    if mkl_data and 'time_s' in mkl_data:
        mkl_time = mkl_data['time_s'] * 1e6
        plt.axhline(y=mkl_time, color='#d62728', linestyle='-.', linewidth=2, label='Dense NN (MKL)')
    
    plt.ylabel('Runtime (µs)')
    plt.xlabel('Sparsity Level')
    plt.title('Sparse NN Performance by Optimization Technique vs Sparsity Level')
    plt.xticks(x_pos, [f"{sp}%" for sp in sparsity_levels])
    plt.legend(loc='best', fontsize=9)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./plots/nn_sparsity_runtime{suffix}.png', dpi=150)
    print(f'Saved ./plots/nn_sparsity_runtime{suffix}.png')
    
    # Accuracy comparison if available (plot whenever an accuracy field exists, even if zero)
    has_accuracy = False
    for sp in sparsity_levels:
        entry = sparse_data.get(sp, {})
        if isinstance(entry, dict):
            if 'accuracy' in entry:
                has_accuracy = True
                break
            for opt_label, data in entry.items():
                if isinstance(data, dict) and 'accuracy' in data:
                    has_accuracy = True
                    break
        if has_accuracy:
            break

    if has_accuracy:
        plt.figure(figsize=(14, 6))
        
        for idx, technique in enumerate(all_techniques):
            accuracies = []
            for sp in sparsity_levels:
                entry = sparse_data.get(sp, {})
                if technique is None:
                    if isinstance(entry, dict) and 'accuracy' in entry:
                        accuracies.append(entry.get('accuracy', 0))
                    else:
                        accuracies.append(0)
                else:
                    if isinstance(entry, dict) and technique in entry:
                        accuracies.append(entry[technique].get('accuracy', 0))
                    else:
                        accuracies.append(0)
            
            offset = width * (idx - len(all_techniques)/2 + 0.5)
            bars = plt.bar([p + offset for p in x_pos], accuracies, width, label=technique if technique else 'Sparse NN', color=colors[idx])
        
        if dense_data and dense_data.get('accuracy', 0) > 0:
            dense_acc = dense_data['accuracy']
            plt.axhline(y=dense_acc, color='#1f77b4', linestyle='--', linewidth=2, label='Dense NN')
        
        if mkl_data and mkl_data.get('accuracy', 0) > 0:
            mkl_acc = mkl_data['accuracy']
            plt.axhline(y=mkl_acc, color='#d62728', linestyle='-.', linewidth=2, label='Dense NN (MKL)')
        
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Sparsity Level')
        plt.title('Neural Network Accuracy vs Sparsity Level by Optimization Technique')
        plt.xticks(x_pos, [f"{sp}%" for sp in sparsity_levels])
        plt.ylim([0, 100])
        plt.legend(loc='best', fontsize=9)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./plots/nn_sparsity_accuracy{suffix}.png', dpi=150)
        print(f'Saved ./plots/nn_sparsity_accuracy{suffix}.png')


def plot_gemm(records, suffix=''):
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
    plt.savefig(f'./plots/gemm_median{suffix}.png', dpi=150)
    print(f'Saved ./plots/gemm_median{suffix}.png')

    # GFLOPs plot
    plt.figure(figsize=(16,6))
    bars2 = plt.bar(x_pos, gflops, width=0.7, color=['#444' if 'baseline' in lbl else '#2ca02c' for lbl in labels])
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('ProblemSize-Tile')
    plt.title('GEMM Throughput vs Tile Size')
    plt.xticks(x_pos, labels, rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(f'./plots/gemm_gflops{suffix}.png', dpi=150)
    print(f'Saved ./plots/gemm_gflops{suffix}.png')

def plot_spmm(records, suffix='', is_gpu=False):
    if not records:
        print("No SPMM records to plot.")
        return
    os.makedirs('./plots', exist_ok=True)
    # Group by sparsity level
    by_sparsity = defaultdict(list)
    for r in records:
        by_sparsity[r['sparsity']].append(r)
    
    sparsities = sorted(by_sparsity.keys())
    labels = [f"{s}%" for s in sparsities]
    
    # Get unique variants in consistent order
    if is_gpu:
            variant_order = ['Baseline', 'Coalesced', 'Shared', 'Combined']
            variant_colors = {
                "Baseline": "#1f77b4",              # blue
                "Coalesced": "#2ca02c",                # green
                "Shared": "#ff7f0e",       # orange
                "Combined": "#d62728"  # red
            }
    else:
            variant_order = ["Naive", "SIMD", "SIMD+Parallel", "Tiled+SIMD+Parallel"]
            variant_colors = {
                "Naive": "#1f77b4",              # blue
                "SIMD": "#2ca02c",                # green
                "SIMD+Parallel": "#ff7f0e",       # orange
                "Tiled+SIMD+Parallel": "#d62728"  # red
            }
    variants = []
    for v in variant_order:
        for record_list in by_sparsity.values():
            # print(f"Checking records for sparsity level with {len(record_list)} records for variant {v}")
            for r in record_list:
                if r.get('variant') == v:
                    variants.append(v)
                    # print(f"Variant found: {v}")
                    break
        # if any(r.get('variant') == v for records_list in by_sparsity.values() for r in records_list):
        #     variants.append(v)
        #     print(f"Variant found: {v}")

    title_suffix = 'GPU' if is_gpu else 'CPU'

    # Runtime grouped bar plot
    plt.figure(figsize=(14, 8))
    width = 0.8 / len(variants)
    x = list(range(len(labels)))
    
    legend_added = set()
    for idx, variant in enumerate(variants):
        times = []
        for sparsity in sparsities:
            matching = [r['median_time_s'] * 1e6 for r in by_sparsity[sparsity] if r.get('variant') == variant]
            times.append(matching[0] if matching else 0)
        
        color = variant_colors.get(variant, plt.cm.tab10(idx))
        label = variant if variant not in legend_added else None
        bars = plt.bar([xi + idx * width for xi in x], times, width, label=label, color=color)
        legend_added.add(variant)
        
    
    plt.yscale('log')
    plt.ylabel('Runtime (µs)')
    plt.xlabel('Sparsity Level')
    plt.title(f'SpMM Runtime by Optimization Implementation across Sparsity Levels ({title_suffix})')
    plt.xticks(x, labels)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'./plots/spmm_optimization_runtime{suffix}.png', dpi=150)
    print(f'Saved ./plots/spmm_optimization_runtime{suffix}.png')

    # Throughput grouped bar plot
    plt.figure(figsize=(14, 8))
    legend_added = set()
    
    for idx, variant in enumerate(variants):
        gflops_vals = []
        for sparsity in sparsities:
            matching = [r['gflops'] for r in by_sparsity[sparsity] if r.get('variant') == variant]
            gflops_vals.append(matching[0] if matching else 0)
        
        color = variant_colors.get(variant, plt.cm.tab10(idx))
        label = variant if variant not in legend_added else None
        bars = plt.bar([xi + idx * width for xi in x], gflops_vals, width, 
                   label=label, color=color)
        legend_added.add(variant)
        
    
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Sparsity Level')
    plt.title(f'SpMM Throughput by Optimization Implementation across Sparsity Levels ({title_suffix})')
    plt.xticks(x, labels)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'./plots/spmm_optimization_gflops{suffix}.png', dpi=150)
    print(f'Saved ./plots/spmm_optimization_gflops{suffix}.png')

def plot_gemv(records, suffix=''):
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
    plt.savefig(f'./plots/gemv_median{suffix}.png', dpi=150)
    print(f'Saved ./plots/gemv_median{suffix}.png')

    # GFLOPs plot
    plt.figure(figsize=(14,6))
    plt.bar(x_pos, gflops, width=0.7, color='#8c564b')
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Matrix Size')
    plt.title('GEMV Throughput vs Problem Size')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'./plots/gemv_gflops{suffix}.png', dpi=150)
    print(f'Saved ./plots/gemv_gflops{suffix}.png')

def plot_spmv(records, suffix='', is_gpu=False):
    if not records:
        print("No SPMV records to plot.")
        return
    os.makedirs('./plots', exist_ok=True)

    # Group by sparsity level
    by_sparsity = defaultdict(list)
    for r in records:
        by_sparsity[r['sparsity']].append(r)
    
    sparsities = sorted(by_sparsity.keys())
    labels = [f"{s}%" for s in sparsities]
    
    # Get unique variants in consistent order
    if is_gpu:
            variant_order = ['Baseline', 'Coalesced', 'Shared', 'Combined']
            variant_colors = {
                "Baseline": "#1f77b4",              # blue
                "Coalesced": "#2ca02c",                # green
                "Shared": "#ff7f0e",       # orange
                "Combined": "#d62728"  # red
            }
    else:
            variant_order = ["Naive", "SIMD", "SIMD+Parallel", "Tiled+SIMD+Parallel"]
            variant_colors = {
                "Naive": "#1f77b4",              # blue
                "SIMD": "#2ca02c",                # green
                "SIMD+Parallel": "#ff7f0e",       # orange
                "Tiled+SIMD+Parallel": "#d62728"  # red
            }
    
    variants = []
    for v in variant_order:
        for record_list in by_sparsity.values():
            for r in record_list:
                if r.get('variant') == v:
                    variants.append(v)
                    # print(f"Variant found: {v}")
                    break
        # if any(r.get('variant') == v for records_list in by_sparsity.values() for r in records_list):
        #     variants.append(v)
        #     print(f"Variant found: {v}")
    # Assign distinct colors to each variant

    title_suffix = 'GPU' if is_gpu else 'CPU'
    
    # Runtime grouped bar plot
    plt.figure(figsize=(14, 8))
    width = 0.8 / len(variants)
    x = list(range(len(labels)))
    legend_added = set()
    
    for idx, variant in enumerate(variants):
        times = []
        for sparsity in sparsities:
            matching = [r['median_time_s'] * 1e6 for r in by_sparsity[sparsity] if r.get('variant') == variant]
            times.append(matching[0] if matching else 0)
        
        color = variant_colors.get(variant, plt.cm.tab10(idx))
        label = variant if variant not in legend_added else None
        bars = plt.bar([xi + idx * width for xi in x], times, width, 
                   label=label, color=color)
        legend_added.add(variant)
        
    
    plt.yscale('log')
    plt.ylabel('Runtime (µs)')
    plt.xlabel('Sparsity Level')
    plt.title(f'SpMV Runtime by Optimization Implementation across Sparsity Levels ({title_suffix})')
    plt.xticks(x, labels)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'./plots/spmv_optimization_runtime{suffix}.png', dpi=150)
    print(f'Saved ./plots/spmv_optimization_runtime{suffix}.png')

    # Throughput grouped bar plot
    plt.figure(figsize=(14, 8))
    
    legend_added = set()
    for idx, variant in enumerate(variants):
        gflops_vals = []
        for sparsity in sparsities:
            matching = [r['gflops'] for r in by_sparsity[sparsity] if r.get('variant') == variant]
            gflops_vals.append(matching[0] if matching else 0)
        
        color = variant_colors.get(variant, plt.cm.tab10(idx))
        label = variant if variant not in legend_added else None
        bars = plt.bar([xi + idx * width for xi in x], gflops_vals, width, 
                   label=label, color=color)
        legend_added.add(variant)
        
    
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Sparsity Level')
    plt.title(f'SpMV Throughput by Optimization Implementation across Sparsity Levels ({title_suffix})')
    plt.xticks(x, labels)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'./plots/spmv_optimization_gflops{suffix}.png', dpi=150)
    print(f'Saved ./plots/spmv_optimization_gflops{suffix}.png')


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


def get_spmm_variant_name(t1, t2):
    """Return human-readable name for SPMM optimization variant.
    
    t1 codes: -1=Naive, 0=SIMD, 1=SIMD+Parallel, 16=Tiled+SIMD+Parallel
    Returns implementation name ONLY (no tile sizes, they're just codes not dimensions).
    """
    if t1 < 0:
        return "Naive"
    if t1 == 0:
        return "SIMD"
    if t1 == 1:
        return "SIMD+Parallel"
    if t1 == 16:
        return "Tiled+SIMD+Parallel"
    # Fallback for unexpected codes
    return f"Unknown({t1})"

def get_spmv_variant_name(t1, t2):
    """Return human-readable name for SPMV optimization variant.
    
    t1 codes: -1=Naive, 0=Parallel, 16=Tiled+Parallel
    Returns implementation name ONLY (no tile sizes, they're just codes not dimensions).
    """
    if t1 < 0:
        return "Naive"
    if t1 == 0:
        return "Parallel"
    if t1 == 16:
        return "Tiled+Parallel"
    # Fallback for unexpected codes
    return f"Unknown({t1})"

def plot_spmm_optimization(records, suffix=''):
    """Grouped bars showing different optimization techniques across sparsity levels for CPU SPMM."""
    if not records:
        print("No SPMM records to plot for optimization comparison.")
        return

    # Group by sparsity level
    by_sparsity = defaultdict(list)
    for r in records:
        by_sparsity[r['sparsity']].append(r)
    
    sparsities = sorted(by_sparsity.keys())
    labels = [f"{s}%" for s in sparsities]
    
    # Get unique optimization variants (t1, t2) and sort
    variants = set()
    for records_list in by_sparsity.values():
        for r in records_list:
            variant_label = r.get('variant') or get_spmm_variant_name(r.get('t1', -1), r.get('t2', -1))
            variants.add(variant_label)
    variants = sorted(variants)
    
    os.makedirs('./plots', exist_ok=True)

    # Runtime plot with grouped bars (one per optimization variant)
    plt.figure(figsize=(14, 8))
    width = 0.8 / len(variants)
    x = list(range(len(labels)))
    colors = plt.cm.tab10(range(len(variants)))
    
    legend_added = set()
    legend_added = set()
    for idx, variant_label in enumerate(variants):
        times = []
        for sparsity in sparsities:
            matching = [r['median_time_s'] * 1e6 for r in by_sparsity[sparsity]
                        if (r.get('variant') or get_spmm_variant_name(r.get('t1', -1), r.get('t2', -1))) == variant_label]
            times.append(matching[0] if matching else 0)
        
        label = variant_label if variant_label not in legend_added else None
        bars = plt.bar([xi + idx * width for xi in x], times, width, 
                   label=label, color=colors[idx])
        legend_added.add(variant_label)
        
        # Annotate
        for i, (bar, t) in enumerate(zip(bars, times)):
            if t > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{t:.0f}µs", ha='center', va='bottom', fontsize=7)
    
    plt.yscale('log')
    plt.ylabel('Runtime (µs)')
    plt.xlabel('Sparsity Level')
    plt.title('SPMM Runtime by Optimization Technique across Sparsity Levels (CPU)')
    plt.xticks(x, labels)
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'./plots/spmm_optimization_runtime{suffix}.png', dpi=150)
    print(f'Saved ./plots/spmm_optimization_runtime{suffix}.png')

    # Throughput plot
    plt.figure(figsize=(14, 8))
    x = list(range(len(labels)))
    
    legend_added = set()
    for idx, variant_label in enumerate(variants):
        gflops_vals = []
        for sparsity in sparsities:
            matching = [r['gflops'] for r in by_sparsity[sparsity]
                        if (r.get('variant') or get_spmm_variant_name(r.get('t1', -1), r.get('t2', -1))) == variant_label]
            gflops_vals.append(matching[0] if matching else 0)
        
        label = variant_label if variant_label not in legend_added else None
        bars = plt.bar([xi + idx * width for xi in x], gflops_vals, width, 
                   label=label, color=colors[idx])
        legend_added.add(variant_label)
        
        # Annotate
        for i, (bar, g) in enumerate(zip(bars, gflops_vals)):
            if g > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{g:.1f}", ha='center', va='bottom', fontsize=7)
    
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Sparsity Level')
    plt.title('SPMM Throughput by Optimization Technique across Sparsity Levels (CPU)')
    plt.xticks(x, labels)
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'./plots/spmm_optimization_gflops{suffix}.png', dpi=150)
    print(f'Saved ./plots/spmm_optimization_gflops{suffix}.png')


def plot_spmv_optimization(records, suffix=''):
    """Grouped bars showing different optimization techniques across sparsity levels for CPU SPMV."""
    if not records:
        print("No SPMV records to plot for optimization comparison.")
        return

    # Group by sparsity level
    by_sparsity = defaultdict(list)
    for r in records:
        by_sparsity[r['sparsity']].append(r)
    
    sparsities = sorted(by_sparsity.keys())
    labels = [f"{s}%" for s in sparsities]
    
    # Get unique optimization variants (t1, t2) and sort
    variants = set()
    for records_list in by_sparsity.values():
        for r in records_list:
            variant_label = r.get('variant') or get_spmv_variant_name(r.get('t1', -1), r.get('t2', -1))
            variants.add(variant_label)
    variants = sorted(variants)
    
    os.makedirs('./plots', exist_ok=True)

    # Runtime plot with grouped bars (one per optimization variant)
    plt.figure(figsize=(14, 8))
    width = 0.8 / len(variants)
    x = list(range(len(labels)))
    colors = plt.cm.tab10(range(len(variants)))
    
    for idx, variant_label in enumerate(variants):
        times = []
        for sparsity in sparsities:
            matching = [r['median_time_s'] * 1e6 for r in by_sparsity[sparsity]
                        if (r.get('variant') or get_spmv_variant_name(r.get('t1', -1), r.get('t2', -1))) == variant_label]
            times.append(matching[0] if matching else 0)
        
        label = variant_label if variant_label not in legend_added else None
        bars = plt.bar([xi + idx * width for xi in x], times, width, 
                   label=label, color=colors[idx])
        legend_added.add(variant_label)
        
        # Annotate
        for i, (bar, t) in enumerate(zip(bars, times)):
            if t > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{t:.0f}µs", ha='center', va='bottom', fontsize=7)
    
    plt.yscale('log')
    plt.ylabel('Runtime (µs)')
    plt.xlabel('Sparsity Level')
    plt.title('SPMV Runtime by Optimization Technique across Sparsity Levels (CPU)')
    plt.xticks(x, labels)
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'./plots/spmv_optimization_runtime{suffix}.png', dpi=150)
    print(f'Saved ./plots/spmv_optimization_runtime{suffix}.png')

    # Throughput plot
    plt.figure(figsize=(14, 8))
    x = list(range(len(labels)))
    
    legend_added = set()
    for idx, variant_label in enumerate(variants):
        gflops_vals = []
        for sparsity in sparsities:
            matching = [r['gflops'] for r in by_sparsity[sparsity]
                        if (r.get('variant') or get_spmv_variant_name(r.get('t1', -1), r.get('t2', -1))) == variant_label]
            gflops_vals.append(matching[0] if matching else 0)
        
        label = variant_label if variant_label not in legend_added else None
        bars = plt.bar([xi + idx * width for xi in x], gflops_vals, width, 
                   label=label, color=colors[idx])
        legend_added.add(variant_label)
        
        # Annotate
        for i, (bar, g) in enumerate(zip(bars, gflops_vals)):
            if g > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{g:.1f}", ha='center', va='bottom', fontsize=7)
    
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('Sparsity Level')
    plt.title('SPMV Throughput by Optimization Technique across Sparsity Levels (CPU)')
    plt.xticks(x, labels)
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'./plots/spmv_optimization_gflops{suffix}.png', dpi=150)
    print(f'Saved ./plots/spmv_optimization_gflops{suffix}.png')

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

def plot_optimization_stacks(json_path, suffix=''):
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
        plt.savefig(f'./plots/gemm_optimization_stacked_runtime{suffix}.png', dpi=150)
        print(f'Saved ./plots/gemm_optimization_stacked_runtime{suffix}.png')

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
        plt.savefig(f'./plots/gemm_optimization_stacked_throughput{suffix}.png', dpi=150)
        print(f'Saved ./plots/gemm_optimization_stacked_throughput{suffix}.png')

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
        plt.savefig(f'./plots/gemv_optimization_stacked_runtime{suffix}.png', dpi=150)
        print(f'Saved ./plots/gemv_optimization_stacked_runtime{suffix}.png')

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
        plt.savefig(f'./plots/gemv_optimization_stacked_throughput{suffix}.png', dpi=150)
        print(f'Saved ./plots/gemv_optimization_stacked_throughput{suffix}.png')

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else './logs/project.json'
    
    # Collect and plot all benchmark types
    print("Processing benchmarks from:", path)
    
    # Detect if this is GPU (nvbench) format
    with open(path, 'r') as f:
        data = json.load(f)
    is_gpu = is_nvbench_format(data)
    suffix = '_gpu' if is_gpu else ''
    stem = os.path.splitext(os.path.basename(path))[0]
    nn_suffix = f"_{stem}" if is_gpu else suffix
    
    # Check if this is an NN-only benchmark file
    is_nn_only = 'nn_' in stem.lower()
    
    if not is_nn_only:
        gemm_records = collect_gemm(path)
        print(f"Found {len(gemm_records)} GEMM configurations")
        plot_gemm(gemm_records, suffix)

        spmm_records = collect_spmm(path)
        print(f"Found {len(spmm_records)} SPMM configurations")
        plot_spmm(spmm_records, suffix, is_gpu)
        
        gemv_records = collect_gemv(path)
        print(f"Found {len(gemv_records)} GEMV configurations")
        plot_gemv(gemv_records, suffix)
        
        spmv_records = collect_spmv(path)
        print(f"Found {len(spmv_records)} SPMV configurations")
        plot_spmv(spmv_records, suffix, is_gpu)
    
    nn_data = collect_nn(path)
    print(f"Found NN data: Dense={bool(nn_data['dense'])}, Sparse sparsity levels={sorted(nn_data['sparse'].keys())}")
    plot_nn(nn_data, nn_suffix)

    # New stacked bar plot comparing optimization techniques across sizes
    if not is_nn_only:
        plot_optimization_stacks(path, suffix)
    
    print("\nAll plots saved to ./plots/")

if __name__ == '__main__':
    main()
