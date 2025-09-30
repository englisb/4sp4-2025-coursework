#!/usr/bin/env python3
"""Comprehensive plotting for sorting benchmarks.

This script reads Google Benchmark JSON (the file produced with
--benchmark_format=json) and creates publication-style plots:
- Median real time (log-log) with friendly labels and markers
- Median L1 data-cache loads (log-log)
- Mean time vs size plots
- Boxplot distributions
- Relative variability analysis
- CSV output for raw data and summaries

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

# Set seaborn style for better plots
sns.set(style="whitegrid", context="talk")


def load_benchmarks(json_path):
    """Load Google Benchmark JSON and return mappings for metrics and DataFrame.

    Returns:
      times[alg][size] -> list of real_time (microseconds)
      l1loads[alg][size] -> list of L1-dcache-loads (counts)
      df -> pandas DataFrame with raw benchmark data
    """
    with open(json_path, 'r') as fh:
        data = json.load(fh)

    times = defaultdict(lambda: defaultdict(list))
    l1loads = defaultdict(lambda: defaultdict(list))
    rows = []

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

        # Collect data for DataFrame
        time_val = bench.get("real_time", bench.get("cpu_time", None))
        stddev = bench.get("stddev", 0.0)
        iterations = bench.get("iterations", 1)
        repetitions = bench.get("repetitions", 1)
        
        rows.append({
            "bench": full_name,
            "algo": alg_token,
            "size": size,
            "real_time": time_val,
            "stddev": stddev,
            "iterations": iterations,
            "repetitions": repetitions,
            "time_unit": bench.get("time_unit", "us"),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return times, l1loads, df


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


def create_csv_outputs(df, outdir='./plots'):
    """Create CSV files with raw data and summary statistics."""
    ensure_dir(outdir)
    
    if df.empty:
        print("No data available for CSV output")
        return
    
    # Save raw table
    raw_csv = os.path.join(outdir, 'bench_raw.csv')
    df.to_csv(raw_csv, index=False)
    print(f"Wrote {raw_csv}")
    
    # Group summary per algo & size
    summary = df.groupby(["algo", "size"]).real_time.agg(["mean", "median", "std", "count"]).reset_index()
    # Rename columns to stable names if present
    if "mean" in summary.columns:
        summary = summary.rename(columns={"mean": "mean_time"})
    if "std" in summary.columns:
        summary = summary.rename(columns={"std": "std_time"})
    # Ensure numeric types and fill missing std with 0
    summary["mean_time"] = pd.to_numeric(summary.get("mean_time", summary.get("mean", pd.Series())), errors='coerce')
    summary["std_time"] = pd.to_numeric(summary.get("std_time", summary.get("std", pd.Series())), errors='coerce').fillna(0.0)
    
    # Save summary
    summary_csv = os.path.join(outdir, 'bench_summary.csv')
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")
    
    return summary


def plot_additional_analysis(df, summary, outdir='./plots'):
    """Create additional analysis plots from plot_benchmarks.py functionality."""
    ensure_dir(outdir)
    
    if df.empty or summary.empty:
        print("No data available for additional analysis plots")
        return
    
    # Plot 1: mean time vs size (log-log)
    plt.figure(figsize=(9,6))
    for algo, group in summary.groupby("algo"):
        # drop rows with missing size
        gs = group.dropna(subset=["size"])
        if gs.empty:
            continue
        style = style_for_token(algo)
        plt.plot(gs["size"].astype(float), gs["mean_time"], 
                marker=style['marker'], color=style['color'], label=pretty_name(algo))
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Input size")
    plt.ylabel(f"Mean time ({df['time_unit'].iat[0]})")
    plt.title("Mean execution time vs input size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_mean = os.path.join(outdir, 'mean_time_vs_size.png')
    plt.savefig(out_mean, dpi=150)
    plt.close()
    print(f"Wrote {out_mean}")

    # Plot 2: boxplot of run times by algorithm (all sizes)
    plt.figure(figsize=(10,6))
    order = sorted(df["algo"].unique())
    sns.boxplot(x="algo", y="real_time", data=df, order=order)
    plt.ylabel(f"Real time ({df['time_unit'].iat[0]})")
    plt.title("Distribution of run times per algorithm (all sizes)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_box = os.path.join(outdir, 'boxplot_algo_allsizes.png')
    plt.savefig(out_box, dpi=150)
    plt.close()
    print(f"Wrote {out_box}")

    # Plot 3: relative variability (std/mean) vs size per algorithm
    # compute relative standard deviation robustly
    summary["rel_std"] = 0.0
    if "std_time" in summary.columns and "mean_time" in summary.columns:
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = summary["std_time"] / summary["mean_time"]
        rel = rel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        summary["rel_std"] = rel
    
    plt.figure(figsize=(9,6))
    for algo, group in summary.groupby("algo"):
        g = group.dropna(subset=["size"])
        if g.empty:
            continue
        style = style_for_token(algo)
        plt.plot(g["size"].astype(float), g["rel_std"], 
                marker=style['marker'], color=style['color'], label=pretty_name(algo))
    plt.xscale("log", base=2)
    plt.xlabel("Input size")
    plt.ylabel("Relative std (std/mean)")
    plt.title("Relative variability vs input size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_var = os.path.join(outdir, 'relative_variability_vs_size.png')
    plt.savefig(out_var, dpi=150)
    plt.close()
    print(f"Wrote {out_var}")


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

    print('Wrote median performance plots to', outdir)


def plot_google_benchmark_median(json_path):
    """Legacy function maintained for compatibility."""
    main_with_args(json_path, './plots')


def main_with_args(json_path, outdir):
    """Main function that can be called with arguments."""
    if not os.path.exists(json_path):
        print('Benchmark JSON not found:', json_path)
        return

    print(f"Reading benchmark data from {json_path}")
    
    # Load all data
    times, l1, df = load_benchmarks(json_path)
    
    if df.empty:
        print("No benchmark entries found in the JSON file")
        return
    
    # Create CSV outputs
    summary = create_csv_outputs(df, outdir)
    
    # Create median-based plots (original functionality)
    med_time = median_by_size(times)
    med_l1 = median_by_size(l1)
    plot_performance_and_l1(med_time, med_l1, outdir=outdir)
    
    # Create additional analysis plots (from plot_benchmarks.py)
    if summary is not None:
        plot_additional_analysis(df, summary, outdir)
    
    # Print summary to console
    if summary is not None and not summary.empty:
        print("\nTop summary rows:")
        print(summary.sort_values(["algo","size"]).head(20).to_string(index=False))
    
    print(f"\nAll outputs written to {outdir}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive plotting for sorting benchmarks')
    parser.add_argument('json', nargs='?', default='./logs/lab01.json',
                       help='Path to Google Benchmark JSON file')
    parser.add_argument('--outdir', default='./plots',
                       help='Output directory for plots and CSV files')
    args = parser.parse_args()

    main_with_args(args.json, args.outdir)


if __name__ == '__main__':
    main()
