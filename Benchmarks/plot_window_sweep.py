#!/usr/bin/env python3
"""
plot_window_sweep.py

Plots the effect of the window-size parameter (W) on compression and access
performance for WindowedSort and BWT reorderers paired with AutoSubIntSplit.

For each dataset found in the JSON results:
  - One figure is produced, with one subplot per metric.
  - x-axis: window size W (log₂ scale).
  - y-axis: the measured metric value (scaled as noted in the units label).
  - Lines: one per reorderer type (WSort, BWT).
  - Horizontal dashed reference lines for window-less baselines
    (AutoSubIntSplit, Sort|AutoSubIntSplit).

JSON layout (as produced by BenchmarkOutput.hpp):
  top-level fields: encoderName, datasetName, dataSize, isComposed, metrics
  metrics fields:   timing, memory, randomAccess, customMetrics, ...

Usage:
    python plot_window_sweep.py [--results PATH] [--out DIR]

Defaults:
    --results  …/EncodingsPlayground/Benchmarks/results/reordering/all/reordering_results.json
    --out      …/EncodingsPlayground/Benchmarks/plots/window_sweep/
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Metrics to plot:
#   (json_dot_path, display_name, y_axis_units, lower_is_better, scale)
#
# json_dot_path is a dot-separated path into the full result record.
#   e.g. "metrics.memory.compressionRatio" navigates rec["metrics"]["memory"]["compressionRatio"]
# scale is multiplied onto the raw value before plotting (use 1e-6 to convert ns → ms).
# ---------------------------------------------------------------------------
METRICS = [
    # Compression
    ("metrics.memory.compressionRatio",
     "Compression ratio", "ratio (lower = better)", True, 1.0),

    # Encode / decode time (convert ns → ms)
    ("metrics.timing.encodeTime_ns",
     "Encode time", "ms (lower = better)", True, 1e-6),
    ("metrics.timing.decodeBulkTime_ns",
     "Bulk decode time", "ms (lower = better)", True, 1e-6),

    # Access latency (ns → ms)
    ("metrics.randomAccess.averageRandomAccessTime_ns",
     "Avg random-access latency", "ms (lower = better)", True, 1e-6),
    ("metrics.randomAccess.averageRangeAccessTime_ns",
     "Avg range-access latency", "ms (lower = better)", True, 1e-6),

    # Permutation index overhead
    ("metrics.customMetrics.permutation_bytes",
     "Permutation index size", "bytes (lower = better)", True, 1.0),
    ("metrics.customMetrics.permutation_pct_of_encoded",
     "Permutation / encoded data", "% of encoded size (lower = better)", True, 1.0),
    ("metrics.customMetrics.permutation_pct_of_uncompressed",
     "Permutation / raw data", "% of raw size (lower = better)", True, 1.0),

    # Reordering-layer timing (ns → ms)
    ("metrics.customMetrics.reorder_encode_time_ns",
     "Reorder time (encode)", "ms (lower = better)", True, 1e-6),
    ("metrics.customMetrics.unreorder_decode_all_time_ns",
     "Unreorder time (decode-all)", "ms (lower = better)", True, 1e-6),
    ("metrics.customMetrics.perm_lookup_decode_at_ns",
     "Perm lookup (decodeAt, accumulated)", "ns accum (lower = better)", True, 1.0),
    ("metrics.customMetrics.perm_lookup_decode_range_ns",
     "Perm lookup (decodeRange, accumulated)", "ns accum (lower = better)", True, 1.0),
]

# Auxiliary fields collected for decomposition but not plotted directly
AUX_FIELDS = [
    ("metrics.memory.originalSize",            1.0),
    ("metrics.randomAccess.randomAccessCount", 1.0),
    ("metrics.randomAccess.rangeQueryCount",   1.0),
]

# Reorderer prefixes recognised for the windowed sweep
WINDOWED_REORDERERS = ["WSort", "BWT"]

# Baselines plotted as horizontal reference lines; key = exact encoderName
BASELINES = {
    "AutoSubIntSplit":      "AutoSubIntSplit (no reorder)",
    "Sort|AutoSubIntSplit": "Sort (full) | AutoSubIntSplit",
}

_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
REORDERER_COLOURS = {r: _COLOURS[i % len(_COLOURS)] for i, r in enumerate(WINDOWED_REORDERERS)}
BASELINE_COLOURS  = {k: _COLOURS[len(WINDOWED_REORDERERS) + i] for i, k in enumerate(BASELINES)}
BASELINE_STYLES   = ["--", "-."]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nested_get(d: dict, dotted_key: str):
    """Traverse a nested dict by a dot-separated key path; return None if missing."""
    for k in dotted_key.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def parse_windowed(enc_name: str):
    """
    Parse "WSort64|AutoSubIntSplit" or "BWT128|AutoSubIntSplit".
    Returns (prefix, W, inner_name) or None.
    """
    m = re.match(r"^([A-Za-z]+)(\d+)\|(.+)$", enc_name)
    if m and m.group(1) in WINDOWED_REORDERERS:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


def load_results(json_path: Path) -> list[dict]:
    with json_path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    raise ValueError(f"Unexpected JSON structure in {json_path}")


# ---------------------------------------------------------------------------
# Permutation-index decomposition
#
# DECOMPOSABLE maps a metric path to a function (data_dict -> float|None) that
# returns the "inner codec only" value (i.e. with the permutation index
# contribution removed).  Values in data_dict are already scaled.
# ---------------------------------------------------------------------------

def _noindex_ratio(d: dict):
    """compressionRatio without the permutation index blob."""
    ratio    = d.get("metrics.memory.compressionRatio")
    perm_b   = d.get("metrics.customMetrics.permutation_bytes")
    orig_b   = d.get("metrics.memory.originalSize")
    if ratio is None or perm_b is None or not orig_b:
        return None
    result = ratio - perm_b / orig_b
    return result if result > 0 else None


def _noindex_subtract(total_key: str, sub_key: str):
    """Factory: returns a function that subtracts sub_key from total_key."""
    def fn(d: dict):
        total = d.get(total_key)
        sub   = d.get(sub_key)
        if total is None or sub is None:
            return None
        result = total - sub
        return result if result >= 0 else None
    return fn


def _noindex_divide_sub(total_key: str, sub_key: str, divisor_key: str,
                        sub_scale: float = 1.0):
    """Factory: subtracts (sub_key * sub_scale) / divisor_key from total_key."""
    def fn(d: dict):
        total = d.get(total_key)
        sub   = d.get(sub_key)
        n     = d.get(divisor_key)
        if total is None or sub is None or not n:
            return None
        result = total - (sub * sub_scale) / n
        return result if result >= 0 else None
    return fn


DECOMPOSABLE = {
    "metrics.memory.compressionRatio":
        _noindex_ratio,
    "metrics.timing.encodeTime_ns":
        _noindex_subtract(
            "metrics.timing.encodeTime_ns",
            "metrics.customMetrics.reorder_encode_time_ns"),
    "metrics.timing.decodeBulkTime_ns":
        _noindex_subtract(
            "metrics.timing.decodeBulkTime_ns",
            "metrics.customMetrics.unreorder_decode_all_time_ns"),
    # perm_lookup fields are stored in raw ns (scale=1.0); averageRandomAccessTime
    # is stored in ms (scale=1e-6), so divide the lookup by 1e6 to match units.
    "metrics.randomAccess.averageRandomAccessTime_ns":
        _noindex_divide_sub(
            "metrics.randomAccess.averageRandomAccessTime_ns",
            "metrics.customMetrics.perm_lookup_decode_at_ns",
            "metrics.randomAccess.randomAccessCount",
            sub_scale=1e-6),
    "metrics.randomAccess.averageRangeAccessTime_ns":
        _noindex_divide_sub(
            "metrics.randomAccess.averageRangeAccessTime_ns",
            "metrics.customMetrics.perm_lookup_decode_range_ns",
            "metrics.randomAccess.rangeQueryCount",
            sub_scale=1e-6),
}


# ---------------------------------------------------------------------------
# Format sweep plot (x-axis = permutation format, categorical)
# ---------------------------------------------------------------------------

# Ordered from most to least space-efficient (left to right = smaller overhead)
FORMAT_ORDER = [
    "InverseEliasFano", "ValueGrouped",
    "DeltaZstd", "DeltaLZ4", "DeltaBitPacked",
    "FlatBitPacked",
    "ChunkRelativeZstd", "ChunkRelativeLZ4", "ChunkRelative",
]

# Regex: "Sort[FormatName]|AutoSubIntSplit" or "WSort256[FormatName]|AutoSubIntSplit"
_FMT_SWEEP_RE = re.compile(r"^(Sort|WSort\d+)\[([^\]]+)\]\|(.+)$")


def parse_format_sweep(enc_name: str):
    """Return (reorderer_prefix, fmt_name, inner) or None."""
    m = _FMT_SWEEP_RE.match(enc_name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def plot_format_sweep(all_results: list, out_dir: Path):
    """
    Plot permutation-format comparison for Sort and WSort256 reorderers.
    x-axis: permutation format (categorical, ordered by compressibility).
    y-axis: each metric.
    One figure per dataset.
    """
    # Organise: fmtData[dataset][reorderer_prefix][fmt_name][metric_path] = value
    fmtData: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for rec in all_results:
        enc_name = rec.get("encoderName", "")
        dataset  = rec.get("datasetName", rec.get("metrics", {}).get("generatorName", "unknown"))
        parsed = parse_format_sweep(enc_name)
        if not parsed:
            continue
        prefix, fmt_name, _ = parsed
        for mpath, _, _, _, scale in METRICS:
            raw = nested_get(rec, mpath)
            if isinstance(raw, (int, float)):
                fmtData[dataset][prefix][fmt_name][mpath] = raw * scale
        for mpath, scale in AUX_FIELDS:
            raw = nested_get(rec, mpath)
            if isinstance(raw, (int, float)):
                fmtData[dataset][prefix][fmt_name][mpath] = raw * scale

    if not fmtData:
        print("  (no format-sweep records found)")
        return

    n_metrics = len(METRICS)
    n_cols    = min(4, n_metrics)
    n_rows    = math.ceil(n_metrics / n_cols)

    for dataset, prefix_data in fmtData.items():
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(5.5 * n_cols, 4.0 * n_rows),
                                  squeeze=False)
        fig.suptitle(f"Permutation format comparison — {dataset}",
                     fontsize=13, fontweight="bold")

        # Determine which formats are present and their display order
        all_fmts: set = set()
        for pd in prefix_data.values():
            all_fmts.update(pd.keys())
        ordered_fmts = [f for f in FORMAT_ORDER if f in all_fmts]
        ordered_fmts += sorted(all_fmts - set(FORMAT_ORDER))  # any unknown formats at end
        x_ticks = range(len(ordered_fmts))

        for mi, (mpath, mname, munits, lower_better, *_) in enumerate(METRICS):
            ax = axes[mi // n_cols][mi % n_cols]

            decompfn = DECOMPOSABLE.get(mpath)
            for pi, (prefix, colour) in enumerate(zip(sorted(prefix_data.keys()),
                                                       _COLOURS)):
                fmt_map = prefix_data[prefix]
                ys = [fmt_map.get(f, {}).get(mpath) for f in ordered_fmts]
                xs = [i for i, v in enumerate(ys) if v is not None]
                ys_valid = [v for v in ys if v is not None]
                if not xs:
                    continue
                bar_xs = [x + pi * 0.35 for x in xs]
                if decompfn:
                    ys_inner = [decompfn(fmt_map.get(f, {})) for f in ordered_fmts]
                    # Pair inner values with valid positions
                    inner_valid = [ys_inner[i] for i in xs]
                    # Fall back to solid bar if inner is unavailable
                    if any(v is not None for v in inner_valid):
                        bottoms   = [v if v is not None else y
                                     for v, y in zip(inner_valid, ys_valid)]
                        overheads = [y - b for y, b in zip(ys_valid, bottoms)]
                        ax.bar(bar_xs, bottoms, width=0.32, color=colour,
                               alpha=0.8, label=prefix)
                        ax.bar(bar_xs, overheads, width=0.32, bottom=bottoms,
                               color=colour, alpha=0.4, hatch="/",
                               label=f"{prefix} (index)")
                        continue
                ax.bar(bar_xs, ys_valid,
                       width=0.32, color=colour, alpha=0.8, label=prefix)

            ax.set_xticks([x + 0.175 for x in x_ticks])
            ax.set_xticklabels(ordered_fmts, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel(munits, fontsize=8)
            ax.set_title(mname, fontsize=10)
            ax.legend(fontsize=7)
            ax.set_ylim(bottom=0)
            ax.grid(True, axis="y", linestyle=":", alpha=0.4)
            arrow = "↓" if lower_better else "↑"
            ax.annotate(arrow, xy=(1, 0), xycoords="axes fraction",
                         xytext=(-4, 4), textcoords="offset points",
                         ha="right", va="bottom", fontsize=9, color="gray")

        for mi in range(n_metrics, n_rows * n_cols):
            axes[mi // n_cols][mi % n_cols].set_visible(False)

        fig.tight_layout()
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset)
        out_path = out_dir / f"format_sweep_{safe}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main():
    here = Path(__file__).parent
    default_results = (
        here / "results" / "reordering" / "all" / "reordering_results.json"
    )
    default_out = here / "plots" / "window_sweep"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=default_results)
    ap.add_argument("--out",     type=Path, default=default_out)
    args = ap.parse_args()

    if not args.results.exists():
        print(f"Error: results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    all_results = load_results(args.results)
    print(f"Loaded {len(all_results)} benchmark records from {args.results}")

    # -----------------------------------------------------------------
    # Organise data by dataset name
    #   datasets[dataset][prefix][W][metric_path]   = scaled_value
    #   baselines[dataset][enc_name][metric_path]   = scaled_value
    # -----------------------------------------------------------------
    datasets:  dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    baselines: dict = defaultdict(lambda: defaultdict(dict))

    for rec in all_results:
        enc_name  = rec.get("encoderName", "")
        dataset   = rec.get("datasetName", rec.get("metrics", {}).get("generatorName", "unknown"))

        parsed = parse_windowed(enc_name)
        if parsed:
            prefix, W, _ = parsed
            for mpath, _, _, _, scale in METRICS:
                raw = nested_get(rec, mpath)
                if isinstance(raw, (int, float)):
                    datasets[dataset][prefix][W][mpath] = raw * scale
            for mpath, scale in AUX_FIELDS:
                raw = nested_get(rec, mpath)
                if isinstance(raw, (int, float)):
                    datasets[dataset][prefix][W][mpath] = raw * scale
            continue

        if enc_name in BASELINES:
            for mpath, _, _, _, scale in METRICS:
                raw = nested_get(rec, mpath)
                if isinstance(raw, (int, float)):
                    baselines[dataset][enc_name][mpath] = raw * scale
            for mpath, scale in AUX_FIELDS:
                raw = nested_get(rec, mpath)
                if isinstance(raw, (int, float)):
                    baselines[dataset][enc_name][mpath] = raw * scale

    if not datasets:
        print("No windowed-sweep records found. "
              "Have you run reordering_benchmarks yet?", file=sys.stderr)
        print("Encoder names seen:", [r.get("encoderName") for r in all_results], file=sys.stderr)
        sys.exit(1)

    print(f"Found sweep data for {len(datasets)} dataset(s): {list(datasets.keys())}")

    # -----------------------------------------------------------------
    # Window-sweep figures
    # -----------------------------------------------------------------
    n_metrics = len(METRICS)
    n_cols    = min(4, n_metrics)
    n_rows    = math.ceil(n_metrics / n_cols)

    for dataset, reorderer_data in datasets.items():
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(5.5 * n_cols, 4.0 * n_rows),
                                  squeeze=False)
        fig.suptitle(
            f"Window-size sweep — {dataset}\n"
            r"Solid line = total (with index)  ·  dashed = inner codec only  ·  shaded = index overhead",
            fontsize=12, fontweight="bold")

        for mi, (mpath, mname, munits, lower_better, *_) in enumerate(METRICS):
            ax = axes[mi // n_cols][mi % n_cols]

            decompfn = DECOMPOSABLE.get(mpath)
            for prefix, colour in REORDERER_COLOURS.items():
                if prefix not in reorderer_data:
                    continue
                w_data = reorderer_data[prefix]
                pairs  = sorted((W, d.get(mpath)) for W, d in w_data.items())
                pairs  = [(W, v) for W, v in pairs if v is not None]
                if not pairs:
                    continue
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=5,
                        color=colour, label=prefix)

                if decompfn:
                    ys_inner = [decompfn(w_data[W]) for W, _ in pairs]
                    ys_inner_plot = [v if v is not None else float("nan")
                                     for v in ys_inner]
                    if any(v == v for v in ys_inner_plot):  # at least one non-nan
                        ax.plot(xs, ys_inner_plot, linestyle="--", linewidth=1.5,
                                markersize=0, color=colour, alpha=0.6)
                        ax.fill_between(xs, ys_inner_plot, ys,
                                        alpha=0.12, color=colour)

            for bi, (benc, blabel) in enumerate(BASELINES.items()):
                bdata = baselines.get(dataset, {}).get(benc, {})
                bval  = bdata.get(mpath)
                if bval is not None:
                    ax.axhline(bval, linestyle=BASELINE_STYLES[bi],
                               color=BASELINE_COLOURS[benc], linewidth=1.5,
                               label=blabel, alpha=0.8)
                    if decompfn:
                        bval_inner = decompfn(bdata)
                        if bval_inner is not None:
                            ax.axhline(bval_inner, linestyle=BASELINE_STYLES[bi],
                                       color=BASELINE_COLOURS[benc], linewidth=1.0,
                                       alpha=0.45, label=f"{blabel} (inner)")
                            ax.axhspan(bval_inner, bval, alpha=0.06,
                                       color=BASELINE_COLOURS[benc])

            ax.set_xlabel("Window size W", fontsize=9)
            ax.set_ylabel(munits, fontsize=8)
            ax.set_title(mname, fontsize=10)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: str(int(x)) if x == int(x) else f"{x:.0f}"))
            ax.legend(fontsize=7, loc="best")
            ax.set_ylim(bottom=0)
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
            arrow = "↓" if lower_better else "↑"
            ax.annotate(arrow, xy=(1, 0), xycoords="axes fraction",
                         xytext=(-4, 4), textcoords="offset points",
                         ha="right", va="bottom", fontsize=9, color="gray")

        for mi in range(n_metrics, n_rows * n_cols):
            axes[mi // n_cols][mi % n_cols].set_visible(False)

        fig.tight_layout()
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset)
        out_path = args.out / f"window_sweep_{safe}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # -----------------------------------------------------------------
    # Format-sweep figures
    # -----------------------------------------------------------------
    print("\nGenerating format-sweep plots...")
    plot_format_sweep(all_results, args.out)

    print("Done.")


if __name__ == "__main__":
    main()
