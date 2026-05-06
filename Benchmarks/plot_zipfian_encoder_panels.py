#!/usr/bin/env python3
"""
Create six benchmark comparison plots for 10 encoders across five Zipfian datasets.

Each output figure contains 5 dataset subplots (Zipfian alpha 0 -> 2 left-to-right),
with a compact capability table below each subplot.

Metrics plotted:
  1) encode time
  2) bulk decode time
  3) random-access decode time
  4) compression rate (transformed to 1 / compressionRatio)
  5) encode peak heap memory
  6) decode bulk peak heap memory

Notes for decode memory plot:
- Bar height = memory.decodeBulkPeakHeapBytes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# Required bar order
ENCODER_ORDER = [
    "Raw",
    "AdaptiveFOR",
    "FreqNoIndex",
    "FreqPerTierBitmaps",
    "FreqTierTagArray",
    "FreqEliasFano",
    "Dictionary",
    "RawBitPacked",
    "Zstd",
    "OpenZL",
]

DISPLAY_NAME = {
    "Raw": "Raw",
    "AdaptiveFOR": "AdaptiveFOR",
    "FreqNoIndex": "FreqNoIndex",
    "FreqPerTierBitmaps": "FreqPerTierBitmaps",
    "FreqTierTagArray": "FreqTierTagArray",
    "FreqEliasFano": "FreqEliasFano",
    "Dictionary": "Dictionary",
    "RawBitPacked": "BitPrefix",
    "Zstd": "Zstd (level 3)",
    "OpenZL": "OpenZL",
}

# Distinct color family for Freq* only (single-hue gradient). AdaptiveFOR stays outside this family.
ENCODER_COLORS = {
    "Raw": "#7f7f7f",
    "AdaptiveFOR": "#b4241f",
    "FreqNoIndex": "#cfe1f2",
    "FreqPerTierBitmaps": "#8bb9df",
    "FreqTierTagArray": "#4e8fc8",
    "FreqEliasFano": "#1f5e9e",
    "Dictionary": "#2ca02c",
    "RawBitPacked": "#9467bd",
    "Zstd": "#bcbd22",
    "OpenZL": "#17becf",
}

FREQ_ENCODERS = {"FreqNoIndex", "FreqPerTierBitmaps", "FreqTierTagArray", "FreqEliasFano"}
FOCUS_ENCODERS = FREQ_ENCODERS | {"AdaptiveFOR"}

DATASET_ORDER = ["Zipfian0", "Zipfian0.5", "Zipfian1", "Zipfian1.5", "Zipfian2"]
DATASET_TITLE = {
    "Zipfian0": "Zipfian α=0",
    "Zipfian0.5": "Zipfian α=0.5",
    "Zipfian1": "Zipfian α=1",
    "Zipfian1.5": "Zipfian α=1.5",
    "Zipfian2": "Zipfian α=2",
}


def _safe_get(d: dict[str, Any], *keys: str, default: float = np.nan) -> float:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def load_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_lookup(results: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Pick one row per (encoder, dataset), preferring largest dataSize then latest timestamp."""
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in results.get("results", []):
        encoder = row.get("encoderName")
        dataset = row.get("datasetName")
        if encoder is None or dataset is None:
            continue
        key = (encoder, dataset)
        prev = lookup.get(key)
        if prev is None:
            lookup[key] = row
            continue

        prev_size = int(prev.get("dataSize", 0))
        this_size = int(row.get("dataSize", 0))
        if this_size > prev_size:
            lookup[key] = row
            continue
        if this_size == prev_size:
            prev_ts = str(prev.get("metrics", {}).get("timestamp", ""))
            this_ts = str(row.get("metrics", {}).get("timestamp", ""))
            if this_ts > prev_ts:
                lookup[key] = row
    return lookup


def extract_metric(
    lookup: dict[tuple[str, str], dict[str, Any]],
    metric_fn: Callable[[dict[str, Any]], float],
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for dataset in DATASET_ORDER:
        ds_vals: list[float] = []
        for enc in ENCODER_ORDER:
            row = lookup.get((enc, dataset))
            if row is None:
                ds_vals.append(np.nan)
            else:
                ds_vals.append(metric_fn(row))
        values[dataset] = ds_vals
    return values


def capability_table_rows() -> list[tuple[str, list[str]]]:
    random_access = ["✗" if e in {"Zstd", "OpenZL"} else "✓" for e in ENCODER_ORDER]
    reorders = ["✓" if e in FREQ_ENCODERS else "✗" for e in ENCODER_ORDER]
    index_mapping = ["✗" if e == "FreqNoIndex" else "✓" for e in ENCODER_ORDER]
    return [
        ("Random access", random_access),
        ("Reorders", reorders),
        ("Positional map", index_mapping),
    ]


def format_ylabel(unit_kind: str) -> str:
    if unit_kind == "time_ms":
        return "Time (ms)"
    if unit_kind == "time_ns":
        return "Time (ns)"
    if unit_kind == "ratio":
        return "Compression rate (1 / compressionRatio)"
    if unit_kind == "memory_mb":
        return "Memory (MB)"
    return "Value"


def format_bar_label(unit_kind: str, value: float) -> str:
    if not np.isfinite(value):
        return ""

    if unit_kind == "time_ms":
        if value >= 100:
            return f"{value:.0f}ms"
        if value >= 10:
            return f"{value:.1f}ms"
        return f"{value:.2f}ms"

    if unit_kind == "time_ns":
        if value >= 1000:
            return f"{value:,.0f}ns"
        return f"{value:.0f}ns"

    if unit_kind == "ratio":
        if value >= 10:
            return f"{value:.1f}x"
        return f"{value:.2f}x"

    if unit_kind == "memory_mb":
        if value >= 100:
            return f"{value:.0f}MB"
        if value >= 10:
            return f"{value:.1f}MB"
        return f"{value:.2f}MB"

    return f"{value:.2f}"


def make_metric_figure(
    output_path: Path,
    title: str,
    values_by_dataset: dict[str, list[float]],
    unit_kind: str,
) -> None:
    # 2x5 layout: top bars, bottom capability table
    fig = plt.figure(figsize=(36, 10), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=5, height_ratios=[4.35, 1.35], wspace=0.46, hspace=0.30)

    cap_rows = capability_table_rows()
    x = np.arange(len(ENCODER_ORDER))

    if unit_kind == "time_ms":
        converted = {k: np.array(v, dtype=float) / 1_000_000.0 for k, v in values_by_dataset.items()}
    elif unit_kind == "time_ns":
        converted = {k: np.array(v, dtype=float) for k, v in values_by_dataset.items()}
    elif unit_kind == "memory_mb":
        converted = {k: np.array(v, dtype=float) / (1024.0 * 1024.0) for k, v in values_by_dataset.items()}
    else:
        converted = {k: np.array(v, dtype=float) for k, v in values_by_dataset.items()}

    all_vals = np.concatenate([arr for arr in converted.values()])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size > 0:
        y_max = float(np.max(finite))
        y_top = (y_max * 1.12) if y_max > 0 else 1.0
    else:
        y_top = 1.0

    for col, dataset in enumerate(DATASET_ORDER):
        ax = fig.add_subplot(gs[0, col])
        ax_tbl = fig.add_subplot(gs[1, col])

        vals = converted[dataset]

        for i, enc in enumerate(ENCODER_ORDER):
            color = ENCODER_COLORS[enc]
            lw = 2.2 if enc in FOCUS_ENCODERS else 0.8
            edge = "black" if enc in FOCUS_ENCODERS else "#333333"
            ax.bar(
                x[i],
                vals[i],
                color=color,
                edgecolor=edge,
                linewidth=lw,
                zorder=3,
            )

        # Add labels where practical (avoid unreadable clutter on very short bars).
        for i, v in enumerate(vals):
            if not np.isfinite(v) or v <= 0:
                continue
            # "Where possible": label only when bar has enough visual height.
            if v < (0.035 * y_top):
                continue
            label = format_bar_label(unit_kind, float(v))
            if not label:
                continue
            ax.text(
                x[i],
                v + (0.01 * y_top),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
                clip_on=True,
                zorder=4,
            )

        ax.set_title(DATASET_TITLE[dataset], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAME[e] for e in ENCODER_ORDER], rotation=90, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax.set_xlim(-0.6, len(ENCODER_ORDER) - 0.4)
        ax.set_ylim(0, y_top)
        ax.tick_params(axis="y", labelleft=True)

        if col == 0:
            ax.set_ylabel(format_ylabel(unit_kind), fontsize=11)

        # table subplot
        ax_tbl.axis("off")
        cell_text = [row_vals for _, row_vals in cap_rows]
        row_labels = [name for name, _ in cap_rows]
        table = ax_tbl.table(
            cellText=cell_text,
            rowLabels=row_labels,
            loc="center",
            cellLoc="center",
            rowLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.26)

        for (r, c), cell in table.get_celld().items():
            txt = cell.get_text().get_text().strip()
            if txt == "✓":
                cell.get_text().set_color("#1e9b3f")
                cell.get_text().set_fontweight("bold")
            elif txt == "✗":
                cell.get_text().set_color("#c62828")
                cell.get_text().set_fontweight("bold")

    # Common legend at top of each figure
    legend_handles = [
        Patch(
            facecolor=ENCODER_COLORS[e],
            edgecolor="black" if e in FOCUS_ENCODERS else "#333333",
            linewidth=2.2 if e in FOCUS_ENCODERS else 0.8,
            label=DISPLAY_NAME[e],
        )
        for e in ENCODER_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.99),
        frameon=True,
        fontsize=10,
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.04)
    fig.subplots_adjust(top=0.82, bottom=0.07)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 6 Zipfian encoder comparison plots with per-dataset subplots.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Benchmarks/results/benchmark_results_empty_trace.json"),
        help="Input benchmark JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Benchmarks/plots/zipfian_encoder_comparison"),
        help="Directory where generated plots are written.",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    lookup = build_lookup(results)

    encode_time_vals = extract_metric(
        lookup,
        lambda r: _safe_get(r, "metrics", "timing", "encodeTime_ns"),
    )
    decode_bulk_time_vals = extract_metric(
        lookup,
        lambda r: _safe_get(r, "metrics", "timing", "decodeBulkTime_ns"),
    )
    decode_random_time_vals = extract_metric(
        lookup,
        lambda r: _safe_get(r, "metrics", "randomAccess", "averageRandomAccessTime_ns"),
    )
    compression_rate_vals = extract_metric(
        lookup,
        lambda r: (
            1.0 / _safe_get(r, "metrics", "memory", "compressionRatio")
            if _safe_get(r, "metrics", "memory", "compressionRatio") > 0
            else np.nan
        ),
    )
    encode_mem_vals = extract_metric(
        lookup,
        lambda r: _safe_get(r, "metrics", "memory", "encodePeakHeapBytes"),
    )
    decode_mem_vals = extract_metric(
        lookup,
        lambda r: _safe_get(r, "metrics", "memory", "decodeBulkPeakHeapBytes"),
    )

    out = args.output_dir

    dataset_title_portion = "by Encoder and Zipfian Dataset (#Uniques = 1M, #Values = 10M)"

    make_metric_figure(
        out / "encode_time.png",
        f"Encode Time {dataset_title_portion}",
        encode_time_vals,
        unit_kind="time_ms",
    )
    make_metric_figure(
        out / "decode_bulk_time.png",
        f"Bulk Decode Time {dataset_title_portion}",
        decode_bulk_time_vals,
        unit_kind="time_ms",
    )
    make_metric_figure(
        out / "decode_random_access_time.png",
        f"Random Access Decode Time {dataset_title_portion}",
        decode_random_time_vals,
        unit_kind="time_ns",
    )
    make_metric_figure(
        out / "compression_rate.png",
        f"Compression Rate (Higher is Better) {dataset_title_portion}",
        compression_rate_vals,
        unit_kind="ratio",
    )
    make_metric_figure(
        out / "encode_peak_memory.png",
        f"Encode Peak Memory {dataset_title_portion}",
        encode_mem_vals,
        unit_kind="memory_mb",
    )
    make_metric_figure(
        out / "decode_bulk_peak_memory.png",
        f"Decode Bulk Peak Memory {dataset_title_portion}",
        decode_mem_vals,
        unit_kind="memory_mb",
    )

    print(f"Wrote 6 plots to: {out.resolve()}")


if __name__ == "__main__":
    main()
