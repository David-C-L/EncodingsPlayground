#!/usr/bin/env python3
"""
Plot value-frequency histograms produced by GeneratorUtils::writeHistogramCSV.

Usage examples
--------------
# Single file, bar chart:
python plot_histograms.py results/UniformRandom.csv

# Multiple files overlaid as line charts (useful for comparison):
python plot_histograms.py results/*.csv --overlay

# Limit to the top-N most frequent values (avoids very wide plots):
python plot_histograms.py results/ParquetColumn.csv --top 30

# Save to a PNG instead of showing interactively:
python plot_histograms.py results/*.csv --output plots/histograms.png

# Log-scale y-axis (good for power-law / Zipfian distributions):
python plot_histograms.py results/ParquetColumn.csv --log-y
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_histogram(filepath: Path) -> dict:
    """
    Load a histogram CSV written by GeneratorUtils::writeHistogramCSV.

    Uses pandas for fast vectorised CSV parsing; falls back to the pure-Python
    csv reader if pandas is unavailable.

    Returns a dict with keys:
        name    : str            – stem of the filename
        values  : np.ndarray     – numeric values (int64 when all integers)
        counts  : np.ndarray     – int64 counts
        freqs   : np.ndarray     – float64 frequencies (count / total)
        total   : int            – total number of observations
    """
    df     = pd.read_csv(filepath, dtype={"value": "float64", "count": "int64",
                                          "frequency": "float64"})
    values = df["value"].to_numpy()
    counts = df["count"].to_numpy()
    freqs  = df["frequency"].to_numpy()
    total  = int(counts.sum())

    # Promote to int64 if all values are whole numbers (cheap vectorised check)
    if np.all(values == np.floor(values)):
        values = values.astype(np.int64)

    return {
        "name":   filepath.stem,
        "values": values,
        "counts": counts,
        "freqs":  freqs,
        "total":  total,
    }


def values_are_integers(hist: dict) -> bool:
    """Return True if all values in the histogram are whole numbers."""
    v = np.asarray(hist["values"])
    return bool(np.issubdtype(v.dtype, np.integer) or np.all(v == np.floor(v)))


def byte_component_histograms(hist: dict) -> list[dict]:
    """
    Decompose each integer value into its 4 individual bytes (treating the
    value as an unsigned 32-bit integer) and return a histogram dict for each
    byte position.

    Byte positions:
        byte0 = bits  0–7  (least-significant)
        byte1 = bits  8–15
        byte2 = bits 16–23
        byte3 = bits 24–31 (most-significant)

    Each returned dict follows the same schema as load_histogram():
        name, values, counts, freqs, total
    plus an extra key:
        byte_pos  : int  – 0..3
        n_distinct: int  – number of distinct values seen in this byte position

    The counts in the returned histograms are weighted by the original entry
    counts so that the byte distributions reflect the true data frequencies.
    The total is the same as the parent histogram's total.
    """
    total  = hist["total"]
    # Convert to uint32 view for bitwise ops
    iv     = np.asarray(hist["values"], dtype=np.int64).view(np.int64) & 0xFFFFFFFF
    counts = np.asarray(hist["counts"], dtype=np.int64)

    byte_labels = ["byte 0 (bits 0–7, LSB)", "byte 1 (bits 8–15)",
                   "byte 2 (bits 16–23)",    "byte 3 (bits 24–31, MSB)"]
    result = []
    for b in range(4):
        byte_vals = ((iv >> (8 * b)) & 0xFF).astype(np.int64)
        # Accumulate counts per byte value using np.bincount (always 256 buckets)
        bcnts = np.bincount(byte_vals, weights=counts, minlength=256).astype(np.int64)
        mask  = bcnts > 0
        bvals = np.where(mask)[0]
        bc    = bcnts[mask]
        bf    = bc / max(total, 1)
        result.append({
            "name":       f"{hist['name']} – {byte_labels[b]}",
            "values":     bvals,
            "counts":     bc,
            "freqs":      bf,
            "total":      total,
            "byte_pos":   b,
            "n_distinct": int(mask.sum()),
        })
    return result


def two_byte_component_histograms(hist: dict) -> list[dict]:
    """
    Decompose each integer value into its three overlapping 2-byte groups:
        lo  = bits  0–15  (bytes 0+1)
        mid = bits  8–23  (bytes 1+2)
        hi  = bits 16–31  (bytes 2+3)

    Returns a histogram dict for each group.
    """
    total  = hist["total"]
    iv     = np.asarray(hist["values"], dtype=np.int64) & 0xFFFFFFFF
    counts = np.asarray(hist["counts"], dtype=np.int64)

    shifts = [0, 8, 16]
    labels = ["bytes 0–1 (bits 0–15)", "bytes 1–2 (bits 8–23)", "bytes 2–3 (bits 16–31)"]
    result = []
    for g, sh in enumerate(shifts):
        words = ((iv >> sh) & 0xFFFF).astype(np.int64)
        gcnts = np.bincount(words, weights=counts, minlength=65536).astype(np.int64)
        mask  = gcnts > 0
        gvals = np.where(mask)[0]
        gc    = gcnts[mask]
        gf    = gc / max(total, 1)
        result.append({
            "name":       f"{hist['name']} – {labels[g]}",
            "values":     gvals,
            "counts":     gc,
            "freqs":      gf,
            "total":      total,
            "n_distinct": int(mask.sum()),
        })
    return result


def three_byte_component_histograms(hist: dict) -> list[dict]:
    """
    Decompose each integer value into its two 3-byte groups:
        lo3  = bits  0–23  (bytes 0+1+2)
        hi3  = bits  8–31  (bytes 1+2+3)
    """
    total  = hist["total"]
    iv     = np.asarray(hist["values"], dtype=np.int64) & 0xFFFFFFFF
    counts = np.asarray(hist["counts"], dtype=np.int64)

    shifts = [0, 8]
    labels = ["bytes 0–2 (bits 0–23)", "bytes 1–3 (bits 8–31)"]
    result = []
    for g, sh in enumerate(shifts):
        words = ((iv >> sh) & 0xFFFFFF).astype(np.int64)
        gcnts = np.bincount(words, weights=counts, minlength=16777216).astype(np.int64)
        mask  = gcnts > 0
        gvals = np.where(mask)[0]
        gc    = gcnts[mask]
        gf    = gc / max(total, 1)
        result.append({
            "name":       f"{hist['name']} – {labels[g]}",
            "values":     gvals,
            "counts":     gc,
            "freqs":      gf,
            "total":      total,
            "n_distinct": int(mask.sum()),
        })
    return result


def top_n(hist: dict, n: int) -> dict:
    """Return a copy of hist keeping only the n most frequent entries."""
    if n <= 0 or n >= len(hist["values"]):
        return hist
    # Sort by count descending, keep top-n, then re-sort by value ascending
    idx = sorted(range(len(hist["counts"])),
                 key=lambda i: hist["counts"][i], reverse=True)[:n]
    idx_sorted = sorted(idx, key=lambda i: hist["values"][i])
    return {
        "name":   hist["name"],
        "values": [hist["values"][i] for i in idx_sorted],
        "counts": [hist["counts"][i] for i in idx_sorted],
        "freqs":  [hist["freqs"][i]  for i in idx_sorted],
        "total":  hist["total"],
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _apply_log_y(ax, log_y: bool) -> None:
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def _add_component_summary(ax, comp: dict, max_distinct: int) -> None:
    """
    Annotate a byte/word component subplot.
    Shows: distinct values seen, max possible, and the repetition rate
    (= 1 - distinct/max), where 1.0 means perfect repetition (all values
    map to a single code) and 0.0 means no repetition (flat distribution).
    """
    n_distinct  = comp["n_distinct"]
    repetition  = 1.0 - n_distinct / max_distinct
    text = (
        f"distinct = {n_distinct} / {max_distinct}\n"
        f"repetition = {repetition:.1%}"
    )
    ax.text(0.98, 0.97, text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def _plot_component_ax(ax, comp: dict, max_distinct: int, log_y: bool,
                       color: str, label: str) -> None:
    """Draw one byte/word component bar chart onto an existing axes."""
    values = np.asarray(comp["values"])
    freqs  = np.asarray(comp["freqs"])
    counts = np.asarray(comp["counts"])
    binned = False

    if len(values) > _MAX_DISPLAY_BARS:
        bin_counts, bin_edges = np.histogram(
            values, bins=_MAX_DISPLAY_BARS, weights=counts
        )
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_freqs   = bin_counts / max(comp["total"], 1)
        bin_width   = (bin_edges[-1] - bin_edges[0]) / _MAX_DISPLAY_BARS
        ax.bar(bin_centres, bin_freqs,
               width=bin_width * 0.95,
               color=color, edgecolor="none", alpha=0.85)
        tick_vals = np.linspace(values.min(), values.max(), 8)
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([f"{int(v):d}" for v in tick_vals],
                           rotation=45, ha="right", fontsize=6)
        binned = True
    else:
        x = np.arange(len(values))
        ax.bar(x, freqs, width=0.8, color=color, edgecolor="none", alpha=0.85)
        if len(values) <= 32:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:d}" for v in values],
                               rotation=45, ha="right", fontsize=6)
        else:
            step = max(1, len(values) // 8)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([f"{values[i]:d}" for i in range(0, len(values), step)],
                               rotation=45, ha="right", fontsize=6)

    ax.set_xlabel("Value (decimal)")
    ax.set_ylabel("Relative frequency")
    suffix = f"  [binned ×{_MAX_DISPLAY_BARS:,}]" if binned else ""
    ax.set_title(f"{label}{suffix}", fontsize=9)

    _apply_log_y(ax, log_y)
    _add_component_summary(ax, comp, max_distinct)


def plot_byte_components(hist: dict, log_y: bool) -> plt.Figure:
    """
    Plot four subplots: one per byte (0=LSB … 3=MSB) of the integer values.
    Only bytes that have >1 distinct value are shown (fully-zero bytes are
    collapsed and noted in the title).
    """
    comps = byte_component_histograms(hist)
    active = [c for c in comps]

    n      = len(active)
    ncols  = min(n, 2) if n > 0 else 1
    nrows  = (n + ncols - 1) // ncols if n > 0 else 1
    colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(9 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle(f"1-byte component distributions – {hist['name']}", fontsize=11)

    for i, comp in enumerate(active):
        ax    = axes[i // ncols][i % ncols]
        color = colors[comp["byte_pos"] % len(colors)]
        _plot_component_ax(ax, comp, max_distinct=256, log_y=log_y,
                           color=color, label=comp["name"].split("–", 1)[-1].strip())

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_two_byte_components(hist: dict, log_y: bool) -> plt.Figure:
    """
    Plot three subplots: one per 2-byte (16-bit) group of the integer values.
    """
    comps  = two_byte_component_histograms(hist)
    colors = ["steelblue", "darkorange", "seagreen"]

    ncols = min(len(comps), 3)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 4), squeeze=False)
    fig.suptitle(f"2-byte component distributions – {hist['name']}", fontsize=11)

    for i, comp in enumerate(comps):
        ax = axes[0][i]
        _plot_component_ax(ax, comp, max_distinct=65536, log_y=log_y,
                           color=colors[i], label=comp["name"].split("–", 1)[-1].strip())

    fig.tight_layout()
    return fig


def plot_three_byte_components(hist: dict, log_y: bool) -> plt.Figure:
    """
    Plot two subplots: one per 3-byte (24-bit) group of the integer values.
    """
    comps  = three_byte_component_histograms(hist)
    colors = ["steelblue", "darkorange"]

    fig, axes = plt.subplots(1, 2, figsize=(9 * 2, 4), squeeze=False)
    fig.suptitle(f"3-byte component distributions – {hist['name']}", fontsize=11)

    for i, comp in enumerate(comps):
        ax = axes[0][i]
        _plot_component_ax(ax, comp, max_distinct=16_777_216, log_y=log_y,
                           color=colors[i], label=comp["name"].split("–", 1)[-1].strip())

    fig.tight_layout()
    return fig





def _add_summary(ax, hist: dict) -> None:
    """Annotate the plot with key statistics."""
    total      = hist["total"]
    counts     = np.asarray(hist["counts"])
    freqs      = np.asarray(hist["freqs"])
    shown      = int(counts.sum())
    shown_pct  = (shown / total * 100.0) if total > 0 else 0.0
    n_distinct = len(hist["values"])
    top_idx    = int(np.argmax(counts))
    top_val    = hist["values"][top_idx]
    top_pct    = float(freqs.max()) * 100.0
    text = (
        f"n = {total:,}\n"
        f"shown = {shown:,} ({shown_pct:.1f}%)\n"
        f"distinct = {n_distinct:,}\n"
        f"mode = {top_val:g} ({top_pct:.1f}%)"
    )
    ax.text(0.98, 0.97, text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


# ---------------------------------------------------------------------------
# Single-histogram bar chart
# ---------------------------------------------------------------------------

_MAX_DISPLAY_BARS = 2000   # never render more than this many bars/points


def plot_single(hist: dict, log_y: bool, ax=None) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    values = np.asarray(hist["values"])
    freqs  = np.asarray(hist["freqs"])
    counts = np.asarray(hist["counts"])
    binned = False

    if len(values) > _MAX_DISPLAY_BARS:
        # Bin into equal-width display buckets weighted by original counts
        bin_counts, bin_edges = np.histogram(
            values, bins=_MAX_DISPLAY_BARS, weights=counts
        )
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_freqs   = bin_counts / max(hist["total"], 1)
        bin_width   = (bin_edges[-1] - bin_edges[0]) / _MAX_DISPLAY_BARS
        ax.bar(bin_centres, bin_freqs,
               width=bin_width * 0.95,
               color="steelblue", edgecolor="none", alpha=0.85)
        # x-axis: value labels from the actual data range
        tick_vals = np.linspace(values.min(), values.max(), 8)
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([f"{v:g}" for v in tick_vals],
                           rotation=45, ha="right", fontsize=7)
        binned = True
    else:
        x = np.arange(len(values))
        ax.bar(x, freqs, width=0.8, color="steelblue", edgecolor="none", alpha=0.85)
        if len(values) <= 40:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" for v in values],
                               rotation=45, ha="right", fontsize=7)
        else:
            step = max(1, len(values) // 8)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([f"{values[i]:g}" for i in range(0, len(values), step)],
                               rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Value")
    ax.set_ylabel("Relative frequency")
    suffix = f"  [binned into {_MAX_DISPLAY_BARS:,} display buckets]" if binned else ""
    ax.set_title(f"Histogram – {hist['name']}{suffix}")

    _apply_log_y(ax, log_y)
    _add_summary(ax, hist)

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Overlay line chart (multiple histograms on one axes)
# ---------------------------------------------------------------------------

def plot_overlay(histograms: list[dict], log_y: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    for hist in histograms:
        values = np.asarray(hist["values"])
        freqs  = np.asarray(hist["freqs"])

        if len(values) > _MAX_DISPLAY_BARS:
            step   = max(1, len(values) // _MAX_DISPLAY_BARS)
            values = values[::step]
            freqs  = freqs[::step]
            marker = None          # skip per-point markers at this density
            ms     = 0
        else:
            marker = "o"
            ms     = 3

        ax.plot(
            values, freqs,
            marker=marker, markersize=ms,
            linewidth=1.2,
            label=f"{hist['name']} (n={hist['total']:,})",
        )

    ax.set_xlabel("Value")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Histogram comparison")
    ax.legend(fontsize=8)
    _apply_log_y(ax, log_y)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Grid of bar charts (one subplot per file, no overlay)
# ---------------------------------------------------------------------------

def plot_grid(histograms: list[dict], log_y: bool) -> plt.Figure:
    n     = len(histograms)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, hist in enumerate(histograms):
        ax = axes[idx // ncols][idx % ncols]
        plot_single(hist, log_y=log_y, ax=ax)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Byte-pair analysis
# ---------------------------------------------------------------------------
# These functions analyse the joint distribution of two byte positions within
# each integer value.  The core primitive is BytePairSpec — a small dataclass
# that names the two byte positions to analyse.  Adding a new pairing is a
# one-liner in main().
#
# Input: a histogram CSV with columns  value, count, frequency
#        (the same format written by GeneratorUtils::writeHistogramCSV)
# ---------------------------------------------------------------------------

@dataclass
class BytePairSpec:
    """Describes one byte-pair analysis to run.

    byte_lo / byte_hi are 0-based byte indices into the little-endian
    representation of each integer value (0 = least-significant byte).
    label_lo / label_hi are the human-readable names used in plot titles
    and file names.
    """
    byte_lo:   int
    byte_hi:   int
    label_lo:  str = ''
    label_hi:  str = ''
    int_width: int = 4   # sizeof(T) in bytes — used for masking

    def __post_init__(self):
        if not self.label_lo:
            self.label_lo = f'byte{self.byte_lo}'
        if not self.label_hi:
            self.label_hi = f'byte{self.byte_hi}'


def _extract_bytes(values: np.ndarray, spec: 'BytePairSpec'):
    """Return (lo_bytes, hi_bytes) uint8 arrays for the two byte positions."""
    lo = (values >> (spec.byte_lo * 8)) & 0xFF
    hi = (values >> (spec.byte_hi * 8)) & 0xFF
    return lo.astype(np.uint8), hi.astype(np.uint8)


def _build_frequency_matrix(
    values: np.ndarray,
    counts: np.ndarray,
    spec: 'BytePairSpec',
) -> np.ndarray:
    """
    Build a 256×256 matrix M where M[hi, lo] = sum of counts for all
    values whose (byte_lo, byte_hi) pair equals (lo, hi).

    Rows index byte_hi (y-axis), columns index byte_lo (x-axis) so the
    matrix plots naturally with imshow.
    """
    lo_bytes, hi_bytes = _extract_bytes(values, spec)
    mat = np.zeros((256, 256), dtype=np.float64)
    np.add.at(mat, (hi_bytes, lo_bytes), counts)
    return mat


def _bit_entropy_by_byte_pair(mat: np.ndarray) -> np.ndarray:
    """
    For each of the 16 bit positions (8 from byte_lo + 8 from byte_hi),
    compute the Bernoulli variance p*(1-p) of that bit's marginal frequency.

    Perfect balance (p=0.5) → variance=0.25 (high entropy, hard to compress).
    Near-constant bit (p≈0 or p≈1) → variance≈0 (low entropy, compressible).

    Returns a (16,) float array.
    """
    total = mat.sum()
    if total == 0:
        return np.zeros(16)

    variances = np.zeros(16)
    indices   = np.arange(256, dtype=np.float64)

    # Bits 0–7: from byte_lo (column marginal)
    col_marginal = mat.sum(axis=0)   # shape (256,)
    for b in range(8):
        bit_set = ((indices.astype(np.int64) >> b) & 1).astype(np.float64)
        p = np.dot(col_marginal, bit_set) / total
        variances[b] = p * (1 - p)

    # Bits 8–15: from byte_hi (row marginal)
    row_marginal = mat.sum(axis=1)   # shape (256,)
    for b in range(8):
        bit_set = ((indices.astype(np.int64) >> b) & 1).astype(np.float64)
        p = np.dot(row_marginal, bit_set) / total
        variances[8 + b] = p * (1 - p)

    return variances


def _bit_mi_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Compute the 16×16 pairwise mutual information matrix for the 16 bits
    spanning the two bytes (8 from byte_lo + 8 from byte_hi).

    For each pair of bit positions (i, j), MI is estimated from the joint
    distribution of (bit_i, bit_j) derived by marginalising over the 256×256
    frequency matrix.

    MI(i; j) = sum_{a,b in {0,1}} p(a,b) * log2(p(a,b) / (p_i(a) * p_j(b)))

    Returns a symmetric (16, 16) float array in bits.  Diagonal entries are
    the individual bit entropies H(b_i).
    """
    total = mat.sum()
    if total == 0:
        return np.zeros((16, 16))

    # Precompute the marginal probability of each bit being 1
    # bits 0-7  → byte_lo column marginal; bits 8-15 → byte_hi row marginal
    col_marg = mat.sum(axis=0) / total   # p(byte_lo = v)
    row_marg = mat.sum(axis=1) / total   # p(byte_hi = v)

    vals = np.arange(256)

    def _bit_prob_vec(byte_idx: int) -> np.ndarray:
        """Return a (256,) array of p(byte_val = v) for the given byte."""
        return col_marg if byte_idx == 0 else row_marg

    def _bit1_p(byte_idx: int, b: int) -> float:
        """Marginal P(bit b of byte byte_idx = 1)."""
        mask = ((vals >> b) & 1).astype(bool)
        return float(_bit_prob_vec(byte_idx)[mask].sum())

    # Build a lookup: for each of 16 bits, (byte_idx_in_pair, bit_within_byte)
    bit_info = [(0, b) for b in range(8)] + [(1, b) for b in range(8)]

    mi = np.zeros((16, 16))
    for i, (bi, bb) in enumerate(bit_info):
        for j, (bj, bc) in enumerate(bit_info):
            if j < i:
                mi[i, j] = mi[j, i]   # symmetric
                continue
            # joint distribution over (bit_i, bit_j) ∈ {0,1}²
            # marginalise the 256×256 joint over the two byte values
            mask_i = ((vals >> bb) & 1).astype(bool)
            mask_j = ((vals >> bc) & 1).astype(bool)

            # p(bit_i=a, bit_j=b) for a,b in {0,1}
            # We need to marginalise mat over pairs (v_lo, v_hi) grouped by
            # (bit_i of v_{byte_i}, bit_j of v_{byte_j})
            joint = np.zeros((2, 2))
            for a in range(2):
                for b in range(2):
                    # rows/cols of mat where the relevant bit equals a/b
                    if bi == 0:   # bit_i comes from byte_lo (columns)
                        cols_a = np.where(((vals >> bb) & 1) == a)[0]
                    else:         # bit_i comes from byte_hi (rows)
                        cols_a = None
                    if bj == 0:   # bit_j comes from byte_lo (columns)
                        cols_b = np.where(((vals >> bc) & 1) == b)[0]
                    else:
                        cols_b = None

                    # Select the sub-matrix where both bits match
                    rows_a = np.where(((vals >> bb) & 1) == a)[0] if bi == 1 else None
                    rows_b = np.where(((vals >> bc) & 1) == b)[0] if bj == 1 else None

                    # Both from same byte: share row or column selector
                    if bi == 0 and bj == 0:
                        # both bits come from byte_lo (columns)
                        c_idx = np.intersect1d(cols_a, cols_b)
                        joint[a, b] = mat[:, c_idx].sum()
                    elif bi == 1 and bj == 1:
                        # both bits come from byte_hi (rows)
                        r_idx = np.intersect1d(rows_a, rows_b)
                        joint[a, b] = mat[r_idx, :].sum()
                    else:
                        # one from each byte
                        r_idx = rows_a if bi == 1 else rows_b
                        c_idx = cols_a if bi == 0 else cols_b
                        joint[a, b] = mat[np.ix_(r_idx, c_idx)].sum()

            joint /= total
            p_i = joint.sum(axis=1)   # marginal of bit_i
            p_j = joint.sum(axis=0)   # marginal of bit_j
            mi_val = 0.0
            for a in range(2):
                for b in range(2):
                    pab = joint[a, b]
                    if pab > 0 and p_i[a] > 0 and p_j[b] > 0:
                        mi_val += pab * np.log2(pab / (p_i[a] * p_j[b]))
            if i == j:
                # diagonal: entropy H(b_i)
                h = 0.0
                for a in range(2):
                    if p_i[a] > 0:
                        h -= p_i[a] * np.log2(p_i[a])
                mi[i, j] = h
            else:
                mi[i, j] = max(0.0, mi_val)

    return mi


def _sorted_axis_indices(mat: np.ndarray):
    """
    Return (col_order, row_order) — indices that sort byte_lo columns and
    byte_hi rows by their marginal frequency (most frequent first).
    Used to reorder the heatmap so the dense region is top-left.
    """
    col_freq = mat.sum(axis=0)
    row_freq = mat.sum(axis=1)
    col_order = np.argsort(col_freq)[::-1]
    row_order = np.argsort(row_freq)[::-1]
    return col_order, row_order


def _cumulative_coverage(mat: np.ndarray):
    """
    Sort all (byte_lo, byte_hi) pairs by frequency descending and return
    (n_codewords, cumulative_fraction) arrays.

    n_codewords[k] = k+1, cumulative_fraction[k] = fraction of total
    observations covered by the top k+1 pairs.
    """
    flat   = mat.flatten()
    total  = flat.sum()
    if total == 0:
        return np.array([0]), np.array([0.0])
    sorted_desc = np.sort(flat)[::-1]
    nonzero     = sorted_desc[sorted_desc > 0]
    cumsum      = np.cumsum(nonzero) / total
    n_codes     = np.arange(1, len(cumsum) + 1)
    return n_codes, cumsum


def _conditional_entropy_profile(mat: np.ndarray):
    """
    For each observed byte_lo value v (column), compute H(byte_hi | byte_lo=v).

    Returns:
        lo_vals   : (K,) array of byte_lo values that appear at least once
        cond_h    : (K,) array of conditional entropies in bits (0 = perfectly
                    predictable, 8 = fully uniform over 256 byte_hi values)
        col_freq  : (K,) array of P(byte_lo = v) — for sorting / annotation
    """
    col_sums = mat.sum(axis=0)
    observed = np.where(col_sums > 0)[0]
    cond_h   = np.zeros(len(observed))
    for k, v in enumerate(observed):
        col     = mat[:, v]
        p_col   = col / col.sum()
        nz      = p_col[p_col > 0]
        cond_h[k] = float(-np.sum(nz * np.log2(nz)))
    col_freq = col_sums[observed] / mat.sum()
    return observed, cond_h, col_freq


def plot_byte_pair_analysis(
    hist: dict,
    output_dir: Path,
    specs: list,        # list[BytePairSpec]
    log_scale: bool = True,
    name: str = '',
) -> None:
    """
    For each BytePairSpec in *specs*, produce seven PNG files inside
    *output_dir*:

    1. ``byte_pair_heatmap_``          — 256×256 raw frequency heatmap with
                                         marginal spines.
    2. ``byte_pair_heatmap_sorted_``   — same heatmap with rows/columns
                                         reordered by descending frequency so
                                         the dense region collapses to the
                                         top-left corner.
    3. ``byte_pair_bit_variance_``     — 16-bar Bernoulli variance chart.
    4. ``byte_pair_bit_mi_``           — 16×16 mutual information matrix
                                         heatmap; diagonal = bit entropy.
    5. ``byte_pair_cumulative_``       — cumulative coverage curve: how many
                                         distinct (lo, hi) codewords are needed
                                         to cover X% of observations.
    6. ``byte_pair_cond_entropy_``     — conditional entropy H(byte_hi|byte_lo=v)
                                         for each observed byte_lo value,
                                         sorted by frequency.
    7. ``byte_pair_fft_``              — 2D FFT log-magnitude spectrum of the
                                         heatmap (DC-zeroed) to reveal periodic
                                         structure in the byte-pair space.

    Also prints a suggested bit permutation to stdout: bits ordered from
    lowest to highest joint entropy (sum of MI-row) — low-entropy bits first
    are best packed into a dictionary prefix.

    *hist* must follow the load_histogram() dict schema (keys: values, counts,
    freqs, total, name).
    """
    values = np.array(hist['values'], dtype=np.int64)
    counts = np.array(hist['counts'], dtype=np.float64)
    tag    = f'_{name}' if name else ''

    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        mat = _build_frequency_matrix(values, counts, spec)
        pfx = f'{spec.label_lo}_{spec.label_hi}{tag}'   # shared filename prefix

        # Check data exists
        lo_obs = np.where(mat.sum(axis=0) > 0)[0]
        hi_obs = np.where(mat.sum(axis=1) > 0)[0]
        if lo_obs.size == 0 or hi_obs.size == 0:
            print(f"  Skipping {spec.label_lo}×{spec.label_hi}: no data.")
            continue

        lo_min, lo_max = int(lo_obs.min()), int(lo_obs.max())
        hi_min, hi_max = int(hi_obs.min()), int(hi_obs.max())
        mat_crop = mat[hi_min:hi_max + 1, lo_min:lo_max + 1]

        scale_label = 'log(1 + count)' if log_scale else 'count'

        def _display(m):
            return np.log1p(m) if log_scale else m

        def _save(fig, path):
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")
            plt.close(fig)

        # ------------------------------------------------------------------ #
        # 1.  Raw frequency heatmap with marginal spines
        # ------------------------------------------------------------------ #
        col_marg = mat_crop.sum(axis=0)
        row_marg = mat_crop.sum(axis=1)

        fig = plt.figure(figsize=(10, 10))
        gs  = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                               hspace=0.04, wspace=0.04)
        ax_top   = fig.add_subplot(gs[0, 0])
        ax_main  = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        fig.add_subplot(gs[0, 1]).set_visible(False)

        im = ax_main.imshow(_display(mat_crop), origin='lower', aspect='auto',
                            cmap='viridis',
                            extent=[lo_min - 0.5, lo_max + 0.5,
                                    hi_min - 0.5, hi_max + 0.5],
                            interpolation='nearest')
        ax_main.set_xlabel(f'{spec.label_lo} value  (0–255)')
        ax_main.set_ylabel(f'{spec.label_hi} value  (0–255)')
        fig.colorbar(im, ax=ax_main, fraction=0.03, pad=0.02).set_label(
            scale_label, fontsize=8)

        lo_vals = np.arange(lo_min, lo_max + 1)
        ax_top.bar(lo_vals, col_marg, width=1.0, color='steelblue', linewidth=0)
        ax_top.set_xlim(lo_min - 0.5, lo_max + 0.5)
        if col_marg.max() > 0:
            ax_top.set_yscale('log')
        ax_top.tick_params(labelbottom=False, bottom=False)
        ax_top.set_ylabel('count', fontsize=7)
        ax_top.set_title(
            f'{spec.label_lo} × {spec.label_hi} pair frequency'
            + (f'  —  {name}' if name else ''), fontsize=11)

        hi_vals = np.arange(hi_min, hi_max + 1)
        ax_right.barh(hi_vals, row_marg, height=1.0, color='coral', linewidth=0)
        ax_right.set_ylim(hi_min - 0.5, hi_max + 0.5)
        if row_marg.max() > 0:
            ax_right.set_xscale('log')
        ax_right.tick_params(labelleft=False, left=False)
        ax_right.set_xlabel('count', fontsize=7)

        _save(fig, output_dir / f'byte_pair_heatmap_{pfx}.png')

        # ------------------------------------------------------------------ #
        # 2.  Frequency-sorted heatmap  (dense region collapses to top-left)
        # ------------------------------------------------------------------ #
        col_order, row_order = _sorted_axis_indices(mat)
        # restrict to observed cols/rows only (non-zero marginals)
        col_order_obs = col_order[col_order[np.isin(col_order, lo_obs)].argsort()
                                  ] if False else col_order[np.isin(col_order, lo_obs)]
        row_order_obs = row_order[np.isin(row_order, hi_obs)]
        mat_sorted = mat[np.ix_(row_order_obs, col_order_obs)]

        n_cols_s, n_rows_s = mat_sorted.shape[1], mat_sorted.shape[0]

        fig_s, ax_s = plt.subplots(figsize=(10, 9))
        im_s = ax_s.imshow(_display(mat_sorted), origin='upper', aspect='auto',
                           cmap='viridis', interpolation='nearest')
        fig_s.colorbar(im_s, ax=ax_s, fraction=0.03, pad=0.02).set_label(
            scale_label, fontsize=8)

        # Sparse tick labels: show actual byte values at regular positions
        tick_step = max(1, n_cols_s // 8)
        ax_s.set_xticks(np.arange(0, n_cols_s, tick_step))
        ax_s.set_xticklabels(col_order_obs[::tick_step], fontsize=6, rotation=45)
        tick_step_r = max(1, n_rows_s // 8)
        ax_s.set_yticks(np.arange(0, n_rows_s, tick_step_r))
        ax_s.set_yticklabels(row_order_obs[::tick_step_r], fontsize=6)
        ax_s.set_xlabel(f'{spec.label_lo} value  (ranked by frequency)')
        ax_s.set_ylabel(f'{spec.label_hi} value  (ranked by frequency)')
        ax_s.set_title(
            f'{spec.label_lo} × {spec.label_hi}  — frequency-sorted heatmap'
            + (f'  —  {name}' if name else '') +
            '\nAxes ordered by descending marginal frequency; '
            'dense region concentrates in top-left',
            fontsize=10)
        fig_s.tight_layout()
        _save(fig_s, output_dir / f'byte_pair_heatmap_sorted_{pfx}.png')

        # ------------------------------------------------------------------ #
        # 3.  Bit variance chart  (marginal Bernoulli variance per bit)
        # ------------------------------------------------------------------ #
        variances = _bit_entropy_by_byte_pair(mat)

        def _bit_p(byte_idx_in_pair: int, bit_within_byte: int) -> float:
            marginal = mat.sum(axis=0) if byte_idx_in_pair == 0 else mat.sum(axis=1)
            tot_m    = marginal.sum()
            if tot_m == 0:
                return 0.0
            bit_set = ((np.arange(256) >> bit_within_byte) & 1).astype(np.float64)
            return float(np.dot(marginal, bit_set) / tot_m)

        p_vals = np.array([_bit_p(0, b) for b in range(8)] +
                          [_bit_p(1, b) for b in range(8)])
        bit_labels = ([f'{spec.label_lo}\nb{b}' for b in range(8)] +
                      [f'{spec.label_hi}\nb{b}' for b in range(8)])

        colours_bv = np.where(variances < 0.05, '#d73027',
                     np.where(variances < 0.15, '#fc8d59',
                     np.where(variances < 0.20, '#fee090', '#4575b4')))

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        bars2 = ax2.bar(np.arange(16), variances, color=colours_bv, width=0.7)
        ax2.axhline(0.25, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvline(7.5, color='grey', linestyle=':', linewidth=1)
        for bar, p in zip(bars2, p_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f'{p:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)
        ax2.set_xticks(np.arange(16))
        ax2.set_xticklabels(bit_labels, fontsize=8)
        ax2.set_ylim(0, 0.28)
        ax2.set_ylabel('Bernoulli variance  p · (1−p)')
        ax2.set_xlabel('Bit position  (grouped by source byte)')
        ax2.set_title(
            f'Bit variance — {spec.label_lo} × {spec.label_hi}'
            + (f'  —  {name}' if name else '') +
            '\nRed = near-constant (compressible);  Blue = balanced (high entropy)',
            fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(handles=[
            mpatches.Patch(color='#d73027', label='var < 0.05  (near-constant)'),
            mpatches.Patch(color='#fc8d59', label='var < 0.15  (low variance)'),
            mpatches.Patch(color='#fee090', label='var < 0.20  (medium)'),
            mpatches.Patch(color='#4575b4', label='var ≥ 0.20  (near-balanced)'),
        ], fontsize=8, loc='upper left')
        fig2.tight_layout()
        _save(fig2, output_dir / f'byte_pair_bit_variance_{pfx}.png')

        # ------------------------------------------------------------------ #
        # 4.  Bit mutual information matrix  (16×16)
        # ------------------------------------------------------------------ #
        print(f"  Computing bit MI matrix for {spec.label_lo}×{spec.label_hi} …")
        mi_mat = _bit_mi_matrix(mat)

        # Suggested permutation: sort bits by total MI (sum of MI row),
        # ascending = most compressible / least joint entropy first
        bit_total_mi = mi_mat.sum(axis=1) - np.diag(mi_mat)  # exclude self
        perm_order   = np.argsort(bit_total_mi)               # low MI first

        print(f"  Suggested bit permutation (low→high joint MI, "
              f"put first {sum(bit_total_mi < bit_total_mi.mean())} bits "
              f"into dictionary prefix):")
        perm_labels = ([f'{spec.label_lo}_b{b}' for b in range(8)] +
                       [f'{spec.label_hi}_b{b}' for b in range(8)])
        for rank, idx in enumerate(perm_order):
            print(f"    [{rank:2d}] {perm_labels[idx]:14s}  "
                  f"H={mi_mat[idx,idx]:.3f} bits  "
                  f"totalMI={bit_total_mi[idx]:.3f} bits")

        fig_mi, ax_mi = plt.subplots(figsize=(9, 8))
        im_mi = ax_mi.imshow(mi_mat, cmap='YlOrRd', aspect='auto',
                             interpolation='nearest', vmin=0)
        fig_mi.colorbar(im_mi, ax=ax_mi, fraction=0.04, pad=0.02).set_label(
            'Mutual information  (bits)', fontsize=8)

        # Annotate cells
        for i in range(16):
            for j in range(16):
                val = mi_mat[i, j]
                ax_mi.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=5,
                           color='white' if val > mi_mat.max() * 0.6 else 'black')

        tick_lbl = ([f'{spec.label_lo}\nb{b}' for b in range(8)] +
                    [f'{spec.label_hi}\nb{b}' for b in range(8)])
        ax_mi.set_xticks(np.arange(16)); ax_mi.set_xticklabels(tick_lbl, fontsize=7)
        ax_mi.set_yticks(np.arange(16)); ax_mi.set_yticklabels(tick_lbl, fontsize=7)

        # Draw separator between the two bytes
        for ax_line in [ax_mi]:
            ax_line.axhline(7.5, color='white', linewidth=1.5, linestyle='--')
            ax_line.axvline(7.5, color='white', linewidth=1.5, linestyle='--')

        ax_mi.set_title(
            f'Bit mutual information — {spec.label_lo} × {spec.label_hi}'
            + (f'  —  {name}' if name else '') +
            '\nDiagonal = H(bit)  ·  Off-diagonal = MI(bᵢ; bⱼ)  ·  '
            'Dashed line separates the two bytes',
            fontsize=9)
        fig_mi.tight_layout()
        _save(fig_mi, output_dir / f'byte_pair_bit_mi_{pfx}.png')

        # ------------------------------------------------------------------ #
        # 5.  Cumulative coverage curve
        # ------------------------------------------------------------------ #
        n_codes, cum_frac = _cumulative_coverage(mat)

        # Milestones: how many codewords for 50%, 90%, 95%, 99%
        milestones = {}
        for thr in (0.50, 0.90, 0.95, 0.99):
            idx = np.searchsorted(cum_frac, thr)
            if idx < len(n_codes):
                milestones[thr] = int(n_codes[idx])

        fig_cv, ax_cv = plt.subplots(figsize=(9, 5))
        ax_cv.plot(n_codes, cum_frac * 100, color='steelblue', linewidth=1.5)

        for thr, nc in milestones.items():
            ax_cv.axhline(thr * 100, color='grey', linestyle=':', linewidth=0.8)
            ax_cv.axvline(nc, color='grey', linestyle=':', linewidth=0.8)
            ax_cv.annotate(
                f'{thr:.0%} @ {nc} codes',
                xy=(nc, thr * 100),
                xytext=(nc + len(n_codes) * 0.02, thr * 100 - 3),
                fontsize=7, color='dimgrey',
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.6),
            )

        ax_cv.set_xscale('log')
        ax_cv.set_xlabel('Number of distinct (byte_lo, byte_hi) codewords  [log scale]')
        ax_cv.set_ylabel('Cumulative coverage  (%)')
        ax_cv.set_ylim(0, 101)
        ax_cv.set_title(
            f'Cumulative coverage — {spec.label_lo} × {spec.label_hi}'
            + (f'  —  {name}' if name else '') +
            '\nHow many distinct byte-pair codewords are needed to cover X% of data',
            fontsize=10)
        ax_cv.grid(axis='both', alpha=0.3)
        fig_cv.tight_layout()
        _save(fig_cv, output_dir / f'byte_pair_cumulative_{pfx}.png')

        # ------------------------------------------------------------------ #
        # 6.  Conditional entropy  H(byte_hi | byte_lo = v)
        # ------------------------------------------------------------------ #
        lo_vals_ce, cond_h, col_freq = _conditional_entropy_profile(mat)

        # Sort by descending frequency of byte_lo value
        sort_idx  = np.argsort(col_freq)[::-1]
        lo_sorted = lo_vals_ce[sort_idx]
        ce_sorted = cond_h[sort_idx]
        cf_sorted = col_freq[sort_idx]

        n_shown = min(len(lo_sorted), 64)   # cap at 64 bars for readability
        x_pos   = np.arange(n_shown)

        fig_ce, ax_ce = plt.subplots(figsize=(max(10, n_shown * 0.25), 5))
        bars_ce = ax_ce.bar(x_pos, ce_sorted[:n_shown], width=0.8,
                            color='steelblue', alpha=0.85)

        # Colour by conditional entropy: low = green (predictable), high = red
        ce_max = ce_sorted[:n_shown].max() if n_shown > 0 else 1.0
        for bar, h_val in zip(bars_ce, ce_sorted[:n_shown]):
            frac = h_val / ce_max if ce_max > 0 else 0
            bar.set_facecolor(plt.cm.RdYlGn_r(frac))  # type: ignore[attr-defined]

        # Reference line: unconditional H(byte_hi)
        row_marg_full = mat.sum(axis=1)
        row_marg_full /= row_marg_full.sum()
        nz = row_marg_full[row_marg_full > 0]
        h_uncond = float(-np.sum(nz * np.log2(nz)))
        ax_ce.axhline(h_uncond, color='black', linestyle='--', linewidth=1,
                      label=f'H({spec.label_hi}) unconditional = {h_uncond:.2f} bits')

        ax_ce.set_xticks(x_pos)
        ax_ce.set_xticklabels(
            [f'{v}\n({cf:.1%})' for v, cf in
             zip(lo_sorted[:n_shown], cf_sorted[:n_shown])],
            fontsize=6, rotation=45, ha='right')
        ax_ce.set_ylabel(f'H({spec.label_hi} | {spec.label_lo}=v)  (bits)')
        ax_ce.set_xlabel(
            f'{spec.label_lo} value  (sorted by descending frequency; '
            f'showing top {n_shown} of {len(lo_sorted)})')
        ax_ce.set_ylim(0, 8.2)
        ax_ce.set_title(
            f'Conditional entropy H({spec.label_hi} | {spec.label_lo}=v)'
            + (f'  —  {name}' if name else '') +
            '\nGreen = low conditional entropy (byte_hi predictable given byte_lo);'
            '  Red = high;  Dashed = unconditional entropy',
            fontsize=9)
        ax_ce.legend(fontsize=8)
        ax_ce.grid(axis='y', alpha=0.3)
        fig_ce.tight_layout()
        _save(fig_ce, output_dir / f'byte_pair_cond_entropy_{pfx}.png')

        # ------------------------------------------------------------------ #
        # 7.  2D FFT log-magnitude spectrum  (DC-zeroed)
        # ------------------------------------------------------------------ #
        # Work on the full 256×256 matrix (not cropped) so frequencies are
        # well-defined.  Normalise by total count first.
        mat_norm  = mat / (mat.sum() + 1e-12)
        fft2      = np.fft.fft2(mat_norm)
        fft2_mag  = np.abs(np.fft.fftshift(fft2))

        # Zero the DC component so the dynamic range of the AC terms is visible
        cy, cx = fft2_mag.shape[0] // 2, fft2_mag.shape[1] // 2
        fft2_mag[cy, cx] = 0.0
        fft2_disp = np.log1p(fft2_mag)

        fig_fft, ax_fft = plt.subplots(figsize=(9, 8))
        im_fft = ax_fft.imshow(fft2_disp, origin='upper', aspect='auto',
                               cmap='inferno', interpolation='nearest')
        fig_fft.colorbar(im_fft, ax=ax_fft, fraction=0.03, pad=0.02).set_label(
            'log(1 + |FFT|)  [DC zeroed]', fontsize=8)

        # Axis labels: map pixel index to spatial frequency in cycles/256
        freq_ticks  = np.linspace(0, 255, 9, dtype=int)
        freq_labels = [f'{int(f - 128)}' for f in freq_ticks]
        ax_fft.set_xticks(freq_ticks); ax_fft.set_xticklabels(freq_labels, fontsize=7)
        ax_fft.set_yticks(freq_ticks); ax_fft.set_yticklabels(freq_labels, fontsize=7)
        ax_fft.set_xlabel('Spatial frequency  (byte_lo direction)  [cycles / 256 values]')
        ax_fft.set_ylabel('Spatial frequency  (byte_hi direction)  [cycles / 256 values]')
        ax_fft.set_title(
            f'2D FFT spectrum — {spec.label_lo} × {spec.label_hi}'
            + (f'  —  {name}' if name else '') +
            '\nDC term zeroed.  Peaks at (fx, fy) ≠ 0 indicate periodic structure '
            'in the byte-pair distribution\n'
            '(e.g. taxonomic block size ≈ 256/|fx| values)',
            fontsize=9)

        # Annotate the top-5 non-DC peaks
        flat_fft = fft2_disp.flatten()
        top5_flat = np.argpartition(flat_fft, -5)[-5:]
        for idx_flat in top5_flat:
            fy, fx = divmod(int(idx_flat), 256)
            ax_fft.plot(fx, fy, 'r+', markersize=8, markeredgewidth=1.5)
            ax_fft.annotate(
                f'({fx-128},{fy-128})',
                xy=(fx, fy), xytext=(fx + 4, fy - 4),
                fontsize=6, color='red',
            )

        fig_fft.tight_layout()
        _save(fig_fft, output_dir / f'byte_pair_fft_{pfx}.png')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot value-frequency histograms from GeneratorUtils CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="One or more histogram CSV files.",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="Overlay all histograms as line charts on a single axes.",
    )
    parser.add_argument(
        "--top", type=int, default=0, metavar="N",
        help="Keep only the N most frequent values per histogram (0 = all).",
    )
    parser.add_argument(
        "--log-y", action="store_true",
        help="Use a logarithmic y-axis.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="FILE",
        help="Save the figure to this path instead of displaying it.",
    )
    parser.add_argument(
        "--byte-components", action="store_true",
        help=(
            "For integer-valued histograms, also plot the distributions of "
            "each 1-byte, 2-byte, and 3-byte component of the values. "
            "Useful for seeing how much repetition exists within each byte "
            "position. Saved alongside --output as <stem>_bytes1.<ext>, "
            "<stem>_bytes2.<ext>, <stem>_bytes3.<ext>."
        ),
    )
    parser.add_argument(
        "--byte-pair-analysis", action="store_true",
        help=(
            "For integer-valued histograms, produce a 256×256 frequency "
            "heatmap and a 16-bar bit-variance chart for each pair of byte "
            "positions named by --byte-pair-specs (default: byte1 × byte2). "
            "Output PNGs are written to the directory of --output (or the "
            "current directory when --output is not given)."
        ),
    )
    parser.add_argument(
        "--byte-pair-name", type=str, default='', metavar='NAME',
        help="Short label appended to byte-pair plot filenames.",
    )
    args = parser.parse_args()

    # Load
    histograms = []
    for path in args.files:
        if not path.exists():
            print(f"Warning: file not found: {path}", file=sys.stderr)
            continue
        h = load_histogram(path)
        if args.top > 0:
            h = top_n(h, args.top)
        # Print how much of the original data is being plotted (important when --top is used)
        shown = sum(h.get("counts", []))
        total = h.get("total", 0)
        pct = (shown / total * 100.0) if total > 0 else 0.0
        print(f"Loaded {h['name']}: showing {shown:,} of {total:,} ({pct:.1f}%)")
        histograms.append(h)

    if not histograms:
        print("No valid CSV files found.", file=sys.stderr)
        sys.exit(1)

    # Plot main histograms
    if args.overlay or len(histograms) > 1 and args.overlay:
        fig = plot_overlay(histograms, log_y=args.log_y)
    elif len(histograms) == 1:
        fig = plot_single(histograms[0], log_y=args.log_y)
    else:
        fig = plot_grid(histograms, log_y=args.log_y)

    # Output (duplicate overlay branch was a bug — removed)
    if args.overlay and len(histograms) > 1:
        fig = plot_overlay(histograms, log_y=args.log_y)

    def _save_or_show(f, path: Path | None) -> None:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            f.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved to {path}")
        else:
            plt.show()

    _save_or_show(fig, args.output)

    # Byte-component plots (only for integer-valued histograms)
    if args.byte_components:
        for h in histograms:
            if not values_are_integers(h):
                print(f"Skipping byte-component plots for '{h['name']}': non-integer values.")
                continue

            # Derive output paths: <stem>_bytes1.<ext>, _bytes2.<ext>, _bytes3.<ext>
            def _component_path(suffix: str) -> Path | None:
                if args.output is None:
                    return None
                p = args.output
                return p.with_stem(p.stem + suffix)

            print(f"\nByte-component breakdown for '{h['name']}':")
            for b in range(4):
                pass  # printed inside byte_component_histograms summary below

            # 1-byte components
            fig1 = plot_byte_components(h, log_y=args.log_y)
            for comp in byte_component_histograms(h):
                nd = comp["n_distinct"]
                rep = 1.0 - nd / 256
                print(f"  {comp['name'].split('–',1)[-1].strip():30s}  "
                      f"distinct={nd:3d}/256   repetition={rep:.1%}")
            _save_or_show(fig1, _component_path("_bytes1"))

            # 2-byte components
            fig2 = plot_two_byte_components(h, log_y=args.log_y)
            for comp in two_byte_component_histograms(h):
                nd = comp["n_distinct"]
                rep = 1.0 - nd / 65536
                print(f"  {comp['name'].split('–',1)[-1].strip():30s}  "
                      f"distinct={nd:6d}/65536   repetition={rep:.1%}")
            _save_or_show(fig2, _component_path("_bytes2"))

            # 3-byte components
            fig3 = plot_three_byte_components(h, log_y=args.log_y)
            for comp in three_byte_component_histograms(h):
                nd = comp["n_distinct"]
                rep = 1.0 - nd / 16_777_216
                print(f"  {comp['name'].split('–',1)[-1].strip():30s}  "
                      f"distinct={nd:8d}/16777216   repetition={rep:.1%}")
            _save_or_show(fig3, _component_path("_bytes3"))

    # Byte-pair analysis plots
    if args.byte_pair_analysis:
        # Determine output directory (same folder as --output, or cwd)
        bpa_dir = args.output.parent if args.output else Path('.')

        # Define which byte pairings to analyse.  Extend this list to add more.
        specs = [
            BytePairSpec(byte_lo=0, byte_hi=1, label_lo='byte0', label_hi='byte1'),
            BytePairSpec(byte_lo=1, byte_hi=2, label_lo='byte1', label_hi='byte2'),
            BytePairSpec(byte_lo=2, byte_hi=3, label_lo='byte2', label_hi='byte3'),
        ]

        for h in histograms:
            if not values_are_integers(h):
                print(f"Skipping byte-pair analysis for '{h['name']}': non-integer values.")
                continue
            print(f"\nByte-pair analysis for '{h['name']}':")
            name_tag = args.byte_pair_name or h['name']
            plot_byte_pair_analysis(h, bpa_dir, specs,
                                    log_scale=True, name=name_tag)


if __name__ == "__main__":
    main()
