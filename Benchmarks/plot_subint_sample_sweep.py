#!/usr/bin/env python3
"""Plot AutoSubIntSplit sweep results (sample size vs compression/timing).

Expected CSV columns from sweep_subint_samples.cpp:
    sampleSize,encodeTime_ns,selectionTime_ns,compressionRatio,
    bitsPerElement,compressedSize,uncompressedSize

Usage:
    python plot_subint_sample_sweep.py --csv subint_sample_sweep.csv --out sweep_plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        raise ValueError("No rows found in CSV")
    # Ensure structured array even for a single row
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def _mad_mask(values, threshold=3.5):
    values = np.asarray(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.ones_like(values, dtype=bool)
    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) <= threshold


def _clean_series(samples, compression_x, encode_ms, selection_ms):
    finite_mask = np.isfinite(samples) & np.isfinite(compression_x) & np.isfinite(encode_ms) & np.isfinite(selection_ms)
    positive_mask = (samples > 0) & (compression_x > 0) & (encode_ms >= 0) & (selection_ms >= 0)
    ratio_mask = selection_ms <= encode_ms + 1e-9

    mask = finite_mask & positive_mask & ratio_mask
    if mask.sum() >= 3:
        mask &= _mad_mask(compression_x[mask])
        mask &= _mad_mask(encode_ms[mask])

    # Enforce monotonic non-decreasing compression with increasing samples
    if mask.sum() >= 2:
        idx = np.argsort(samples[mask])
        smp = samples[mask][idx]
        comp = compression_x[mask][idx]
        keep = np.ones_like(comp, dtype=bool)
        best = comp[0]
        for i in range(1, comp.size):
            if comp[i] < best:
                keep[i] = False
            else:
                best = comp[i]
        keep_mask = np.zeros_like(mask)
        keep_mask[np.where(mask)[0][idx]] = keep
        mask &= keep_mask

    return samples[mask], compression_x[mask], encode_ms[mask], selection_ms[mask]


def plot_combined(samples, compression_ratio, encode_ms, selection_ms, out_path, clean_plot=False):
    fig, ax_left = plt.subplots(figsize=(9, 5))
    ax_right = ax_left.twinx()

    encode_only = np.maximum(encode_ms - selection_ms, 0)
    total_encode = encode_ms
    compression_x = np.array([1.0 / r if r > 0 else 0.0 for r in compression_ratio])

    if clean_plot:
        samples, compression_x, encode_ms, selection_ms = _clean_series(
            np.array(samples), compression_x, np.array(encode_ms), np.array(selection_ms)
        )
        encode_only = np.maximum(encode_ms - selection_ms, 0)
        total_encode = encode_ms

    marker = None if clean_plot else "o"
    ax_left.plot(samples, compression_x, marker=marker, color="tab:blue", label="Compression (× smaller)")
    ax_left.set_ylabel("Compression (× smaller than raw, blue)")
    ax_left.yaxis.label.set_color("tab:blue")

    base_color = "red"
    selection_color = "darkred"
    encode_only_color = "lightcoral"

    ax_right.plot(samples, total_encode, marker=marker, linestyle="-", color=base_color, label="Total encode")
    ax_right.plot(samples, selection_ms, marker=marker, linestyle="--", color=selection_color, label="Selection")
    ax_right.plot(samples, encode_only, marker=marker, linestyle=":", color=encode_only_color, label="Encode only")
    ax_right.set_ylabel("Encode time (ms, total/selection/encode in red hues)")
    ax_right.yaxis.label.set_color("darkred")

    ax_left.set_xlabel("Sample size")
    fig.suptitle("Compression vs encode time by sample size", fontsize=12, y=0.98)
    ax_left.set_xscale("log")
    ax_left.grid(True, alpha=0.3)

    lines = ax_left.get_lines() + ax_right.get_lines()
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, fontsize=8, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.93), frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot AutoSubIntSplit sample sweep results")
    parser.add_argument("--csv", required=True, type=Path, help="Path to sweep CSV")
    parser.add_argument("--out", default=Path("sweep_plots"), type=Path, help="Output directory")
    parser.add_argument("--clean-plot", action="store_true",
                        help="Remove anomalous points (MAD-based) and draw lines without markers")
    args = parser.parse_args()

    data = load_csv(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)

    samples = data["sampleSize"].astype(float)
    encode_ms = data["encodeTime_ns"].astype(float) / 1e6
    selection_ms = data["selectionTime_ns"].astype(float) / 1e6
    compression_ratio = data["compressionRatio"].astype(float)
    bits_per_elem = data["bitsPerElement"].astype(float)

    plot_combined(samples, compression_ratio, encode_ms, selection_ms,
                  args.out / "compression_vs_encode_time.png",
                  clean_plot=False)

    if args.clean_plot:
        plot_combined(samples, compression_ratio, encode_ms, selection_ms,
                      args.out / "compression_vs_encode_time_clean.png",
                      clean_plot=True)

    # Optional reference plots (uncomment if needed)
    # plot_series(samples, bits_per_elem,
    #             "Sample size", "Bits per element", "Bits per element vs sample size",
    #             args.out / "bits_per_element_vs_sample_size.png")


if __name__ == "__main__":
    main()
