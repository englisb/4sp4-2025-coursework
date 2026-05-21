import json
import matplotlib.pyplot as plt
#!/usr/bin/env python3
"""Styled plotting for sorting benchmarks.

This script reads Google Benchmark JSON (the file produced with
--benchmark_format=json) and creates publication-style plots:
- Median real time (log-log) with friendly labels and markers
- Median L1 data-cache loads (log-log)

The script keeps raw measurements (no standardization) and computes
medians across repeated runs.

Usage: python3 script/plot.py ./logs/lab01.json --outdir ./plots
"""
#!/usr/bin/env python3
"""Styled plotting for sorting benchmarks.

This script reads Google Benchmark JSON (the file produced with
--benchmark_format=json) and creates publication-style plots:
- Median real time (log-log) with friendly labels and markers
- Median L1 data-cache loads (log-log)

The script keeps raw measurements (no standardization) and computes
medians across repeated runs.

Usage: python3 script/plot.py ./logs/lab01.json --outdir ./plots
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid", context="talk")


def load_benchmarks(json_path):
    """Load Google Benchmark JSON and return mappings for metrics.

    Returns two nested dicts:
      times[alg][size] -> list of real_time (microseconds)
      l1loads[alg][size] -> list of L1-dcache-loads (counts)

    No normalization is applied; raw values are kept.
    """
    with open(json_path, 'r') as fh:
        data = json.load(fh)

    times = defaultdict(lambda: defaultdict(list))
    l1loads = defaultdict(lambda: defaultdict(list))

    for bench in data.get('benchmarks', []):
        full_name = bench.get('name', '')
        parts = full_name.split('/')
        if not parts:
            continue
        alg_token = parts[0]

        # extract an integer size from the name (first numeric segment)
        size = None
        for p in parts[1:]:
            if p.isdigit():
                size = int(p)
                break
            # sometimes size is like '16384' with other tags; try stripping non-digits
            try:
                cleaned = ''.join(ch for ch in p if ch.isdigit())
                if cleaned:
                    size = int(cleaned)
                    break
            except Exception:
                pass

        if size is None:
            continue

        # real_time and L1-dcache-loads may be present per iteration
        rt = bench.get('real_time')
        if rt is None:
            rt = bench.get('cpu_time')
        if rt is not None:
            times[alg_token][size].append(float(rt))

        l1 = bench.get('L1-dcache-loads')
        if l1 is not None:
            try:
                l1loads[alg_token][size].append(float(l1))
            except Exception:
                pass

    return times, l1loads


def median_by_size(values):
    """Convert values[alg][size] = list(...) into medians per size."""
    med = {}
    for alg, sizes in values.items():
        med[alg] = {}
        for size, vlist in sizes.items():
            if not vlist:
                continue
            med[alg][size] = float(np.median(vlist))
    return med


def pretty_name(token):
    mapping = {
        'BM_SORT_STD': 'STD Sort',
        'BM_QUICKSORT': 'Quick Sort',
        'BM_SELECTION_SORT': 'Selection Sort',
        'BM_BUBBLE_SORT': 'Bubble Sort'
    }
    return mapping.get(token, token)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def style_for_token(token):
    # colors and markers chosen to match the provided figures
    palette = {
        'BM_SORT_STD': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'BM_QUICKSORT': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
        'BM_SELECTION_SORT': {'color': '#e24a33', 'marker': 's', 'linestyle': '--'},
        'BM_BUBBLE_SORT': {'color': '#ffbf00', 'marker': 'D', 'linestyle': ':'}
    }
    return palette.get(token, {'color': None, 'marker': 'o', 'linestyle': '-'})


def plot_performance_and_l1(medians_time, medians_l1, outdir='./plots'):
    ensure_dir(outdir)

    # prepare tokens in a preferred order for consistent legend
    preferred = ['BM_SORT_STD', 'BM_QUICKSORT', 'BM_SELECTION_SORT', 'BM_BUBBLE_SORT']
    tokens = [t for t in preferred if t in medians_time]
    # if others exist, append them
    for t in sorted(medians_time.keys()):
        if t not in tokens:
            tokens.append(t)

    # global style
    plt.rcParams.update({'axes.titlesize': 20, 'axes.titleweight': 'bold', 'font.size': 12})

    # Plot 1: Median Real Time (log-log)
    fig, ax = plt.subplots(figsize=(14, 6))
    for token in tokens:
        data = medians_time.get(token, {})
        if not data:
            continue
        sizes = sorted(data.keys())
        times = [data[s] for s in sizes]
        style = style_for_token(token)
        ax.plot(sizes, times, marker=style['marker'], linestyle=style['linestyle'],
                color=style['color'], label=pretty_name(token), linewidth=2, markersize=6)

    # set scales and labels
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    try:
        left = min(min(sorted(medians_time[t].keys())) for t in tokens if medians_time.get(t))
        ax.set_xlim(left=left)
    except Exception:
        pass
    ax.set_xlabel('Array Size (n)', fontsize=14)
    ax.set_ylabel('Real Time (µs)', fontsize=14)
    ax.set_title('Sorting Algorithm Performance (Median Real Time)')
    ax.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.25)
    ax.legend(loc='upper left', fontsize=12)

    # bottom-centered caption box (rounded)
    caption = 'Quick/STD scale ≈ n·log n and are fastest at large n; selection/bubble grow quadratically.'
    fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.95))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out1 = os.path.join(outdir, 'sorting_performance_median_realtime.png')
    fig.savefig(out1, dpi=300)
    plt.close(fig)

    # Plot 2: Median L1 Data-Cache Loads vs Array Size (log-log)
    if medians_l1:
        tokens_l1 = [t for t in preferred if t in medians_l1]
        for t in sorted(medians_l1.keys()):
            if t not in tokens_l1:
                tokens_l1.append(t)

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        for token in tokens_l1:
            data = medians_l1.get(token, {})
            if not data:
                continue
            sizes = sorted(data.keys())
            vals = [data[s] for s in sizes]
            style = style_for_token(token)
            ax2.plot(sizes, vals, marker=style['marker'], linestyle=style['linestyle'],
                     color=style['color'], label=pretty_name(token), linewidth=2, markersize=6)

        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        try:
            left2 = min(min(sorted(medians_l1[t].keys())) for t in tokens_l1 if medians_l1.get(t))
            ax2.set_xlim(left=left2)
        except Exception:
            pass
        ax2.set_xlabel('Array Size (n)', fontsize=14)
        ax2.set_ylabel('Median L1-dcache-loads (count)', fontsize=14)
        ax2.set_title('Median L1 Data-Cache Loads vs Array Size')
        ax2.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.25)
        ax2.legend(loc='upper left', fontsize=12)

        caption2 = 'L1-load counts rise ≈ n·log n for STD/Quick and ≈ n^2 for Selection/Bubble, mirroring their work.'
        fig2.text(0.5, 0.02, caption2, ha='center', va='bottom', fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.95))

        fig2.tight_layout(rect=[0, 0.06, 1, 1])
        out2 = os.path.join(outdir, 'median_l1_dcache_loads.png')
        fig2.savefig(out2, dpi=300)
        plt.close(fig2)

    print('Wrote plots to', outdir)


def process_benchmarks(LOG):
    print("Reading", LOG)
    data = json.load(open(LOG))
    benchmarks = data.get("benchmarks", [])

    rows = []
    for b in benchmarks:
        name = b.get("name", "")
        time = b.get("real_time", b.get("cpu_time", None))
        stddev = b.get("stddev", 0.0)
        iterations = b.get("iterations", 1)
        repetitions = b.get("repetitions", 1)
        size = None
        if "args" in b and isinstance(b["args"], list) and len(b["args"]) > 0:
            try:
                size = int(b["args"][0])
            except Exception:
                size = b["args"][0]
        rows.append({
            "bench": name,
            "algo": name.split('/')[0],
            "size": size,
            "real_time": time,
            "stddev": stddev,
            "iterations": iterations,
            "repetitions": repetitions,
            "time_unit": b.get("time_unit", "us"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No benchmark entries found in " + str(LOG))

    # Group summary per algo & size
    summary = df.groupby(["algo", "size"]).real_time.agg(["mean", "median", "std", "count"]).reset_index()
    if "mean" in summary.columns:
        summary = summary.rename(columns={"mean": "mean_time"})

    return summary


def main():
    parser = argparse.ArgumentParser(description='Styled plotting for sorting benchmarks')
    parser.add_argument('json', nargs='?', default='./logs/lab01.json')
    parser.add_argument('--outdir', default='./plots')
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print('Benchmark JSON not found:', args.json)
        return

    times, l1 = load_benchmarks(args.json)
    med_time = median_by_size(times)
    med_l1 = median_by_size(l1)
    plot_performance_and_l1(med_time, med_l1, outdir=args.outdir)

    # Call process_benchmarks to integrate functionality
    summary = process_benchmarks(args.json)


if __name__ == '__main__':
    main()