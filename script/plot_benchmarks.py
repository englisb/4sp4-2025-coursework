#!/usr/bin/env python3
# script/plot_benchmarks.py
# Read Google Benchmark JSON and produce summary CSV + PNG plots.
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", context="talk")

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "lab01.json"
OUTP = ROOT / "plots"
OUTP.mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(parents=True, exist_ok=True)

print("Reading", LOG)
data = json.load(open(LOG))
benchmarks = data.get("benchmarks", [])

rows = []
for b in benchmarks:
    name = b.get("name", "")
    # real_time is in time unit specified by 'time_unit' (usually us)
    time = b.get("real_time", b.get("cpu_time", None))
    stddev = b.get("stddev", 0.0)
    iterations = b.get("iterations", 1)
    repetitions = b.get("repetitions", 1)
    # size usually stored in args list
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

# Save raw table
df.to_csv(ROOT / "logs" / "bench_raw.csv", index=False)

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
summary.to_csv(ROOT / "logs" / "bench_summary.csv", index=False)
print("Wrote logs/bench_summary.csv")

# Plot 1: mean time vs size (log-log)
plt.figure(figsize=(9,6))
for algo, group in summary.groupby("algo"):
    # drop rows with missing size
    gs = group.dropna(subset=["size"])
    if gs.empty:
        continue
    plt.plot(gs["size"].astype(float), gs["mean_time"], marker="o", label=algo)
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Input size")
plt.ylabel(f"Mean time ({df['time_unit'].iat[0]})")
plt.title("Mean execution time vs input size")
plt.legend()
plt.tight_layout()
plt.savefig(OUTP / "mean_time_vs_size.png", dpi=150)
plt.close()
print("Wrote", OUTP / "mean_time_vs_size.png")

# Plot 2: boxplot of run times by algorithm (all sizes)
plt.figure(figsize=(10,6))
order = sorted(df["algo"].unique())
sns.boxplot(x="algo", y="real_time", data=df, order=order)
plt.ylabel(f"Real time ({df['time_unit'].iat[0]})")
plt.title("Distribution of run times per algorithm (all sizes)")
plt.tight_layout()
plt.savefig(OUTP / "boxplot_algo_allsizes.png", dpi=150)
plt.close()
print("Wrote", OUTP / "boxplot_algo_allsizes.png")

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
    plt.plot(g["size"].astype(float), g["rel_std"], marker="o", label=algo)
plt.xscale("log", base=2)
plt.xlabel("Input size")
plt.ylabel("Relative std (std/mean)")
plt.title("Relative variability vs input size")
plt.legend()
plt.tight_layout()
plt.savefig(OUTP / "relative_variability_vs_size.png", dpi=150)
plt.close()
print("Wrote", OUTP / "relative_variability_vs_size.png")

# Optional: quick numeric summary printed to console
print("\nTop summary rows:")
print(summary.sort_values(["algo","size"]).head(20).to_string(index=False))
print("\nDone.") 
