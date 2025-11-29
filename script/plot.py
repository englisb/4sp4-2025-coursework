
import json
import matplotlib.pyplot as plt
import os
import sys
import statistics

def parse_gemm_args(name):
    """Return (m,n,k,t1,t2) parsed robustly from BM_GEMM name.
    Name can include extra tags (e.g., /threads:1). We extract the first 5 integers after BM_GEMM.
    """
    parts = name.split('/')
    if not parts or parts[0] != 'BM_GEMM':
        return None
    nums = []
    for p in parts[1:]:
        # strip possible key:value suffixes
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
    plt.figure(figsize=(12,6))
    bars = plt.bar(labels, med_times, color=['#444' if 'baseline' in lbl else '#1f77b4' for lbl in labels])
    plt.ylabel('Median Time (µs)')
    plt.xlabel('ProblemSize-Tile')
    plt.title('GEMM Median Time vs Tile Size')
    plt.xticks(rotation=35, ha='right')
    # annotate speedup vs baseline of same size
    for bar, lbl, t in zip(bars, labels, med_times):
        size = int(lbl.split('-')[0])
        base = baselines.get(size)
        if base and 'baseline' not in lbl:
            speedup = base / t if t>0 else 0
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, f"{speedup:.2f}x", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('./plots/gemm_median.png', dpi=150)
    print('Saved ./plots/gemm_median.png')

    # GFLOPs plot
    plt.figure(figsize=(12,6))
    bars2 = plt.bar(labels, gflops, color=['#444' if 'baseline' in lbl else '#2ca02c' for lbl in labels])
    plt.ylabel('Throughput (GFLOPs)')
    plt.xlabel('ProblemSize-Tile')
    plt.title('GEMM Throughput vs Tile Size')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig('./plots/gemm_gflops.png', dpi=150)
    print('Saved ./plots/gemm_gflops.png')

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else './logs/project.json'
    records = collect_gemm(path)
    plot_gemm(records)

if __name__ == '__main__':
    main()
