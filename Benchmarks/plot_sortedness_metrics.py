#!/usr/bin/env python3
"""
plot_sortedness_metrics.py

Plots the "sortedness" metrics produced by GeneratorUtils.hpp's
computeSortednessMetrics() / writeSortednessMetricsJSON(), as written by
reordering_benchmarks.cpp to sortedness_metrics.json.

JSON layout:
  { "datasets": [ { "name": ..., "lag1Autocorrelation": ..., ... }, ... ] }

For each metric, produces a grouped bar chart with one bar per dataset.

Usage:
    python plot_sortedness_metrics.py [--results PATH] [--out DIR]

Defaults:
    --results  …/EncodingsPlayground/Benchmarks/results/reordering/all_with_shuffle/sortedness_metrics.json
    --out      …/EncodingsPlayground/Benchmarks/plots/sortedness/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Metrics to plot: (json_field, display_name, y_label)
# ---------------------------------------------------------------------------
METRICS = [
    ("lag1Autocorrelation", "Lag-1 Autocorrelation (4.1)", "rho_1"),
    ("runLengthEntropyNormalized", "Run-Length Entropy Score (4.2)", "S_RLE"),
    ("compressionRatioDelta", "Zstd Compression Ratio Delta (4.3)", "CR_sorted / CR_original - 1"),
    ("normalizedInversions", "Normalized Inversions (4.4)", "tau_dist"),
    ("approximateEntropy", "Approximate Entropy (4.5)", "ApEn"),
    ("cardinality", "Cardinality (4.6)", "# distinct values"),
    ("skewness", "Skewness (4.6)", "skew"),
    ("mutualInformationAdjacent", "Mutual Information, adjacent rows (4.7)", "MI (bits)"),
]


def load_datasets(json_path: Path) -> list[dict]:
    with json_path.open() as f:
        data = json.load(f)
    return data["datasets"]


def plot_metric(datasets: list[dict], field: str, title: str, ylabel: str, out_dir: Path):
    names = [d["name"] for d in datasets]
    values = [d.get(field, 0.0) for d in datasets]

    if field == "compressionRatioDelta":
        # Skip datasets where zstd wasn't available at build time.
        filtered = [(n, v, d) for n, v, d in zip(names, values, datasets)
                    if d.get("compressionAvailable", False)]
        if not filtered:
            print(f"  Skipping {title}: no dataset has compressionAvailable=true")
            return
        names, values, _ = zip(*filtered)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4.5))
    ax.bar(range(len(names)), values, color="tab:blue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path = out_dir / f"{field}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    here = Path(__file__).parent
    default_results = (
        here / "results" / "reordering" / "all_with_shuffle" / "sortedness_metrics.json"
    )
    default_out = here / "plots" / "sortedness"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=default_results)
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()

    if not args.results.exists():
        print(f"Error: results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(args.results)
    print(f"Loaded {len(datasets)} dataset(s) from {args.results}")

    for field, title, ylabel in METRICS:
        plot_metric(datasets, field, title, ylabel, args.out)


if __name__ == "__main__":
    main()
