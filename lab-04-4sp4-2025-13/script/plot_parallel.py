import json
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple


def _load_benchmark_json(json_path: str) -> Dict:
    """Load a Google Benchmark JSON file and return the parsed dict."""
    with open(json_path, mode="r", encoding="utf-8") as f:
        return json.load(f)


def _extract_time(entry: Dict) -> float:
    """Extract a representative time from a benchmark entry.
    Preference order: median aggregate, mean aggregate, fallback real_time.
    Returned units are microseconds (check time_unit field for conversion).
    """
    time_val = entry.get("real_time", 0.0)    
    return time_val


def _parse_matrix_index(name: str) -> int:
    """Extract matrix index from benchmark name.
    Name pattern examples:
      BM_SPTRSV/0/1/repeats:5
      BM_SPTRSV_OMP/2/1/repeats:5
    We take the first numeric component after the base name.
    """
    parts = name.split('/')
    # parts[0] is benchmark name; next should be matrix index
    for token in parts[1:]:
        if token.isdigit():
            return int(token)
    return -1


def _collect_times(data: Dict, prefix: str) -> Dict[int, List[float]]:
    """Collect times per matrix index for benchmarks whose name starts with prefix."""
    per_matrix: Dict[int, List[float]] = {}
    for entry in data.get("benchmarks", []):
        name = entry.get("name", "")
        if not name.startswith(prefix):
            continue
        idx = _parse_matrix_index(name)
        t_us = _extract_time(entry)
        if idx not in per_matrix:
            per_matrix[idx] = []
        per_matrix[idx].append(t_us)
    return per_matrix


def _reduce_times(per_matrix: Dict[int, List[float]]) -> Tuple[List[int], List[float]]:
    """Reduce list of times per matrix index to a single representative (median of collected)."""
    indices = sorted(per_matrix.keys())
    reduced: List[float] = []
    for idx in indices:
        vals = sorted(per_matrix[idx])
        if not vals:
            reduced.append(0.0)
            continue
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            reduced.append(vals[mid])
        else:
            reduced.append(0.5 * (vals[mid - 1] + vals[mid]))
    return indices, reduced


def plot_sptrsv_seq_vs_omp(json_path: str, output_path: str = "./plots/sptrsv_seq_vs_omp.png") -> None:
    """Plot performance comparison of sequential vs OpenMP SpTRSV (Best Thread Count)."""
    data = _load_benchmark_json(json_path)

    # Sequential data - collect only BM_SPTRSV (not BM_SPTRSV_OMP)
    seq_times_raw = {}
    for entry in data.get("benchmarks", []):
        name = entry.get("name", "")
        # Match BM_SPTRSV/ but NOT BM_SPTRSV_OMP/
        if name.startswith("BM_SPTRSV/") and "OMP" not in name:
            idx = _parse_matrix_index(name)
            t_us = _extract_time(entry)
            if idx not in seq_times_raw:
                seq_times_raw[idx] = []
            seq_times_raw[idx].append(t_us)
    seq_indices, seq_times = _reduce_times(seq_times_raw)
    seq_map = dict(zip(seq_indices, seq_times))

    # OpenMP data: find best thread count per matrix
    # Note: _collect_omp_thread_times is defined below, but available at runtime
    omp_thread_data = _collect_omp_thread_times(data)
    omp_best_map = {}
    omp_best_threads = {}

    for m, t_dict in omp_thread_data.items():
        best_t = -1
        min_time = float('inf')
        for t, times in t_dict.items():
            # Calculate median for this thread count
            sorted_times = sorted(times)
            n = len(sorted_times)
            if n == 0: continue
            mid = n // 2
            med = sorted_times[mid] if n % 2 else 0.5 * (sorted_times[mid-1] + sorted_times[mid])
            
            if med < min_time:
                min_time = med
                best_t = t
        
        if best_t != -1:
            omp_best_map[m] = min_time
            omp_best_threads[m] = best_t

    # Align indices
    all_indices = sorted(set(seq_indices) | set(omp_best_map.keys()))
    seq_aligned = [seq_map.get(i, 0.0) for i in all_indices]
    omp_aligned = [omp_best_map.get(i, 0.0) for i in all_indices]
    best_threads = [omp_best_threads.get(i, 0) for i in all_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(all_indices))
    width = 0.35
    
    rects1 = ax.bar([p - width/2 for p in x], seq_aligned, width=width, label='Sequential', color='#1f77b4')
    rects2 = ax.bar([p + width/2 for p in x], omp_aligned, width=width, label='OpenMP (Best)', color='#ff7f0e')

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Matrix {i}" for i in all_indices])
    ax.set_ylabel("Median Runtime (µs)")
    ax.set_xlabel("Matrix Index")
    ax.set_title("Sparse Triangular Solve: Sequential vs OpenMP (Best Config)")
    ax.set_yscale('log')  # Use logarithmic scale for y-axis
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_sptrsv_speedup(json_path: str, output_path: str = "./plots/sptrsv_speedup.png") -> None:
    """Optional speedup plot (sequential_time / omp_time)."""
    data = _load_benchmark_json(json_path)
    
    # Sequential data - collect only BM_SPTRSV (not BM_SPTRSV_OMP)
    seq_times_raw = {}
    for entry in data.get("benchmarks", []):
        name = entry.get("name", "")
        if name.startswith("BM_SPTRSV/") and "OMP" not in name:
            idx = _parse_matrix_index(name)
            t_us = _extract_time(entry)
            if idx not in seq_times_raw:
                seq_times_raw[idx] = []
            seq_times_raw[idx].append(t_us)
    
    omp_times_raw = _collect_times(data, "BM_SPTRSV_OMP")
    seq_indices, seq_times = _reduce_times(seq_times_raw)
    omp_indices, omp_times = _reduce_times(omp_times_raw)
    all_indices = sorted(set(seq_indices) | set(omp_indices))
    seq_map = dict(zip(seq_indices, seq_times))
    omp_map = dict(zip(omp_indices, omp_times))
    speedups = [seq_map.get(i, 0.0) / omp_map.get(i, 1e-9) for i in all_indices]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(all_indices)), speedups, color='#2ca02c')
    plt.xticks(range(len(all_indices)), [f"Matrix {i}" for i in all_indices])
    plt.ylabel("Speedup (Seq / OMP)")
    plt.xlabel("Matrix Index")
    plt.title("SpTRSV Speedup from OpenMP")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

# ---- Thread scaling additions ----
def _parse_matrix_thread(name: str):
    # Expected: BM_SPTRSV_OMP/<matrix>/<threads>/repeats:...
    parts = name.split('/')
    if len(parts) < 3 or not parts[0].startswith('BM_SPTRSV_OMP'):
        return -1, -1
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return -1, -1

def _collect_omp_thread_times(data: Dict) -> Dict[int, Dict[int, List[float]]]:
    result: Dict[int, Dict[int, List[float]]] = {}
    for entry in data.get('benchmarks', []):
        name = entry.get('name', '')
        m, t = _parse_matrix_thread(name)
        if m < 0:
            continue
        time_us = _extract_time(entry)
        result.setdefault(m, {}).setdefault(t, []).append(time_us)
    return result

def _median(vals: List[float]) -> float:
    if not vals: return 0.0
    s = sorted(vals)
    mid = len(s)//2
    return s[mid] if len(s)%2 else 0.5*(s[mid-1]+s[mid])

def plot_omp_thread_scaling(json_path: str, out_dir: str = './plots') -> None:
    if not os.path.exists(json_path):
        return
    data = _load_benchmark_json(json_path)
    collected = _collect_omp_thread_times(data)
    if not collected:
        return
    os.makedirs(out_dir, exist_ok=True)

    # Runtime vs threads
    import math
    plt.figure(figsize=(10,6))
    for m in sorted(collected.keys()):
        threads = sorted(collected[m].keys())
        med_times = [_median(collected[m][th]) for th in threads]
        plt.plot(threads, med_times, marker='o', label=f'Matrix {m}')
    plt.xscale('log', base=2)
    plt.xlabel('Threads (log2)')
    plt.ylabel('Median Runtime (µs)')
    plt.title('SpTRSV OpenMP Thread Scaling')
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sptrsv_omp_thread_scaling.png'))
    plt.close()

    # Speedup curves baseline=1 if present else smallest
    plt.figure(figsize=(10,6))
    for m in sorted(collected.keys()):
        threads = sorted(collected[m].keys())
        med_times = [_median(collected[m][th]) for th in threads]
        baseline_time = None
        if 1 in collected[m]:
            baseline_time = _median(collected[m][1])
        else:
            baseline_time = med_times[0]
        speedups = [baseline_time / t if t>0 else 0.0 for t in med_times]
        plt.plot(threads, speedups, marker='o', label=f'Matrix {m}')
    plt.xscale('log', base=2)
    plt.xlabel('Threads (log2)')
    plt.ylabel('Speedup (baseline/time)')
    plt.title('SpTRSV OpenMP Speedup vs Threads')
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sptrsv_omp_speedup_threads.png'))
    plt.close()


def plot_levelset_mean_times(json_path: str = "./logs/lab04-cpu.json", out_dir: str = "./plots"):
    """Extract and plot mean level-set computation times for each matrix."""
    if not os.path.exists(json_path):
        return
    data = _load_benchmark_json(json_path)
    
    # Extract level-set benchmark data
    levelset_data = {}  # matrix_name -> {threads -> {mean, num_levels}}
    
    for entry in data.get("benchmarks", []):
        name = entry.get("name", "")
        if not name.startswith("BM_SPTRSV_OMP_LEVELSET"):
            continue
        
        # Extract matrix name and thread count from label
        label = entry.get("label", "")
        if "Matrix:" in label and "Threads:" in label:
            parts = label.split(":")
            # parts = ["Matrix", "matrix_name", "Threads", "thread_count"]
            matrix_name = parts[1]
            thread_count = int(parts[3])
            
            if "mean_level_time_us" in entry:
                if matrix_name not in levelset_data:
                    levelset_data[matrix_name] = {}
                
                levelset_data[matrix_name][thread_count] = {
                    'mean': entry["mean_level_time_us"],
                    'num_levels': entry.get("num_levels", 0),
                    'min': entry.get("min_level_time_us", 0),
                    'max': entry.get("max_level_time_us", 0)
                }
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot 1: Mean level-set time for each matrix (averaged across threads)
    plt.figure(figsize=(12, 6))
    matrix_names = sorted(levelset_data.keys())
    means_by_matrix = []
    
    for matrix in matrix_names:
        thread_data = levelset_data[matrix]
        mean_across_threads = sum(d['mean'] for d in thread_data.values()) / len(thread_data)
        means_by_matrix.append(mean_across_threads)
    
    plt.bar(matrix_names, means_by_matrix, alpha=0.7, edgecolor='black')
    plt.xlabel('Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Level-Set Time (μs)', fontsize=12, fontweight='bold')
    plt.title('Mean Level-Set Computation Time per Matrix (OpenMP)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levelset_mean_time_by_matrix.png'), dpi=300)
    plt.close()
    
    # Plot 2: Mean level-set time vs thread count for each matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, matrix in enumerate(sorted(levelset_data.keys())):
        if idx >= 4:
            break
        ax = axes[idx]
        thread_data = levelset_data[matrix]
        
        threads = sorted(thread_data.keys())
        means = [thread_data[t]['mean'] for t in threads]
        
        ax.plot(threads, means, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Level-Set Time (μs)', fontsize=11, fontweight='bold')
        ax.set_title(f'{matrix}', fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(threads)
        ax.set_xticklabels(threads)
    
    plt.suptitle('Mean Level-Set Computation Time vs Thread Count', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levelset_mean_time_by_threads.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Default usage: produce comparison and speedup plots from CPU benchmark output
    cpu_json = "./logs/lab04-cpu.json"
    if os.path.exists(cpu_json):
        plot_sptrsv_seq_vs_omp(cpu_json)
        plot_sptrsv_speedup(cpu_json)
        plot_omp_thread_scaling(cpu_json)
        plot_levelset_mean_times(cpu_json)
    else:
        print(f"Benchmark JSON not found at {cpu_json}. Run benchmarks first.")