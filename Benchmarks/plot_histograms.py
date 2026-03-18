#!/usr/bin/env python3
"""
Plot value-frequency histograms produced by GeneratorUtils::writeHistogramCSV.

Automatically detects whether values are 32-bit or 64-bit integers and adjusts
all byte-group and byte-pair analyses accordingly.

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

# Byte-group breakdown (sizes 1-3 for int32, 1-7 for int64) + overall:
python plot_histograms.py results/snowflake.csv --byte-components --output plots/out.png

# Full byte-pair analysis (adjacent pairs only, auto-detects int width):
python plot_histograms.py results/snowflake.csv --byte-pair-analysis \
    --byte-pair-name snowflake64 --output plots/out.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV loading + integer-width detection
# ---------------------------------------------------------------------------

def load_histogram(filepath: Path) -> dict:
    """
    Load a histogram CSV written by GeneratorUtils::writeHistogramCSV.

    Returns a dict with keys:
        name      : str        -- stem of the filename
        values    : np.ndarray -- int64 (when integer) or float64
        counts    : np.ndarray -- int64
        freqs     : np.ndarray -- float64
        total     : int
        int_width : int        -- 4 (int32), 8 (int64), or 0 if non-integer
    """
    df     = pd.read_csv(filepath, dtype={"value": "float64", "count": "int64",
                                          "frequency": "float64"})
    values = df["value"].to_numpy()
    counts = df["count"].to_numpy()
    freqs  = df["frequency"].to_numpy()
    total  = int(counts.sum())

    int_width = 0
    if np.all(values == np.floor(values)):
        values    = values.astype(np.int64)
        int_width = _detect_int_width(values)

    return {
        "name":      filepath.stem,
        "values":    values,
        "counts":    counts,
        "freqs":     freqs,
        "total":     total,
        "int_width": int_width,
    }


def _detect_int_width(values: np.ndarray) -> int:
    """Return 4 if all values fit in int32 range, else 8 (int64)."""
    if values.size == 0:
        return 4
    lo, hi = int(values.min()), int(values.max())
    if lo >= -(2**31) and hi <= (2**31 - 1):
        return 4
    return 8


def values_are_integers(hist: dict) -> bool:
    return hist.get("int_width", 0) > 0


# ---------------------------------------------------------------------------
# top-n filter
# ---------------------------------------------------------------------------

def top_n(hist: dict, n: int) -> dict:
    """Return a copy of hist keeping only the n most frequent entries."""
    if n <= 0 or n >= len(hist["values"]):
        return hist
    idx = sorted(range(len(hist["counts"])),
                 key=lambda i: hist["counts"][i], reverse=True)[:n]
    idx_sorted = sorted(idx, key=lambda i: hist["values"][i])
    return {**hist,
            "values": hist["values"][idx_sorted],
            "counts": hist["counts"][idx_sorted],
            "freqs":  hist["freqs"][idx_sorted]}


# ---------------------------------------------------------------------------
# Byte-group component histograms
# ---------------------------------------------------------------------------

def _group_histogram(values: np.ndarray, counts: np.ndarray,
                     total: int, start_byte: int, nbytes: int,
                     name: str) -> dict:
    """
    Extract the `nbytes`-byte group starting at `start_byte` from each value
    and build a weighted frequency histogram.

    For nbytes <= 2: use np.bincount (O(N) + 256/65536 buckets).
    For nbytes  > 2: use np.unique  (avoids gigantic intermediate arrays).
    """
    mask = (1 << (nbytes * 8)) - 1
    grp  = ((values.astype(np.int64) >> (start_byte * 8)) & mask)

    if nbytes <= 2:
        n_bins = 1 << (nbytes * 8)
        bcnts  = np.bincount(grp, weights=counts.astype(np.float64),
                             minlength=n_bins).astype(np.int64)
        present = np.where(bcnts > 0)[0]
        gc      = bcnts[present]
    else:
        uvals, inverse = np.unique(grp, return_inverse=True)
        gc = np.zeros(len(uvals), dtype=np.int64)
        np.add.at(gc, inverse, counts.astype(np.int64))
        present = uvals

    gf = gc / max(total, 1)
    return {
        "name":         name,
        "values":       present.astype(np.int64),
        "counts":       gc,
        "freqs":        gf,
        "total":        total,
        "n_distinct":   int((gc > 0).sum()),
        "max_distinct": int(1 << (nbytes * 8)),
        "group_size":   nbytes,
        "start_byte":   start_byte,
    }


def byte_group_histograms(hist: dict) -> dict:
    """
    Return a dict  group_size -> list[component_dict]  covering all adjacent
    byte groups of size 1 to (int_width - 1).

    int32 (int_width=4): sizes 1, 2, 3   (size 4 = whole value, shown separately)
    int64 (int_width=8): sizes 1, 2, 3, 4, 5, 6, 7

    Each list entry covers one starting-byte position (adjacent groups only,
    so there are  int_width - nbytes + 1  entries per size).
    """
    int_width = hist["int_width"]
    if int_width == 0:
        return {}

    values = np.asarray(hist["values"], dtype=np.int64)
    counts = np.asarray(hist["counts"], dtype=np.int64)
    total  = hist["total"]
    result = {}

    for nbytes in range(1, int_width):                      # 1 .. int_width-1
        entries = []
        for start in range(int_width - nbytes + 1):         # adjacent positions
            end   = start + nbytes - 1
            lo_b  = start * 8
            hi_b  = end * 8 + 7
            label = f"bytes {start}–{end}  (bits {lo_b}–{hi_b})"
            h = _group_histogram(values, counts, total,
                                 start_byte=start, nbytes=nbytes,
                                 name=f"{hist['name']} – {label}")
            entries.append(h)
        result[nbytes] = entries

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_MAX_DISPLAY_BARS = 2000
_BYTE_COLOURS = [
    "steelblue", "darkorange", "seagreen", "mediumpurple",
    "crimson",   "saddlebrown", "teal",    "olive",
]


def _apply_log_y(ax, log_y: bool) -> None:
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def _fmt_val(v) -> str:
    """Decimal for small values, hex for large."""
    iv = int(v)
    if abs(iv) > 0xFFFF:
        return f"0x{iv & 0xFFFFFFFFFFFFFFFF:X}"
    return str(iv)


def _add_data_summary(ax, hist: dict) -> None:
    total      = hist["total"]
    counts     = np.asarray(hist["counts"])
    freqs      = np.asarray(hist["freqs"])
    shown      = int(counts.sum())
    shown_pct  = (shown / total * 100.0) if total > 0 else 0.0
    n_distinct = len(hist["values"])
    if n_distinct == 0:
        return
    top_idx  = int(np.argmax(counts))
    top_val  = hist["values"][top_idx]
    top_pct  = float(freqs.max()) * 100.0
    text = (f"n = {total:,}\nshown = {shown:,} ({shown_pct:.1f}%)\n"
            f"distinct = {n_distinct:,}\nmode = {_fmt_val(top_val)} ({top_pct:.1f}%)")
    ax.text(0.98, 0.97, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def _add_component_summary(ax, comp: dict) -> None:
    nd  = comp["n_distinct"]
    mx  = comp["max_distinct"]
    rep = 1.0 - nd / mx if mx > 0 else 0.0
    ax.text(0.98, 0.97, f"distinct = {nd:,} / {mx:,}\nrepetition = {rep:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def _bar_plot_on_ax(ax, comp: dict, log_y: bool, color: str, title: str,
                    show_summary: bool = True) -> None:
    """Generic binned-bar chart for any component histogram."""
    values = np.asarray(comp["values"])
    freqs  = np.asarray(comp["freqs"])
    counts = np.asarray(comp["counts"])
    total  = comp["total"]
    binned = False

    if len(values) > _MAX_DISPLAY_BARS:
        bin_counts, bin_edges = np.histogram(values, bins=_MAX_DISPLAY_BARS,
                                             weights=counts)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_freqs   = bin_counts / max(total, 1)
        bin_width   = (bin_edges[-1] - bin_edges[0]) / _MAX_DISPLAY_BARS
        ax.bar(bin_centres, bin_freqs, width=bin_width * 0.95,
               color=color, edgecolor="none", alpha=0.85)
        tick_vals = np.linspace(values.min(), values.max(), 8)
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([_fmt_val(v) for v in tick_vals],
                           rotation=45, ha="right", fontsize=6)
        binned = True
    else:
        x = np.arange(len(values))
        ax.bar(x, freqs, width=0.8, color=color, edgecolor="none", alpha=0.85)
        if len(values) <= 32:
            ax.set_xticks(x)
            ax.set_xticklabels([_fmt_val(v) for v in values],
                               rotation=45, ha="right", fontsize=6)
        else:
            step = max(1, len(values) // 8)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([_fmt_val(values[i])
                                 for i in range(0, len(values), step)],
                               rotation=45, ha="right", fontsize=6)

    ax.set_xlabel("Value")
    ax.set_ylabel("Relative frequency")
    suffix = f"  [binned ×{_MAX_DISPLAY_BARS:,}]" if binned else ""
    ax.set_title(f"{title}{suffix}", fontsize=9)
    _apply_log_y(ax, log_y)
    if show_summary and "n_distinct" in comp:
        _add_component_summary(ax, comp)


# ---------------------------------------------------------------------------
# Overall histogram
# ---------------------------------------------------------------------------

def plot_single(hist: dict, log_y: bool, ax=None) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()
    _bar_plot_on_ax(ax, hist, log_y=log_y, color="steelblue",
                    title=f"Histogram – {hist['name']}", show_summary=False)
    _add_data_summary(ax, hist)
    if standalone:
        fig.tight_layout()
    return fig


def plot_overlay(histograms: list, log_y: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for hist in histograms:
        values = np.asarray(hist["values"])
        freqs  = np.asarray(hist["freqs"])
        if len(values) > _MAX_DISPLAY_BARS:
            step = max(1, len(values) // _MAX_DISPLAY_BARS)
            values, freqs = values[::step], freqs[::step]
            marker, ms = None, 0
        else:
            marker, ms = "o", 3
        ax.plot(values, freqs, marker=marker, markersize=ms, linewidth=1.2,
                label=f"{hist['name']} (n={hist['total']:,})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Histogram comparison")
    ax.legend(fontsize=8)
    _apply_log_y(ax, log_y)
    fig.tight_layout()
    return fig


def plot_grid(histograms: list, log_y: bool) -> plt.Figure:
    n = len(histograms)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, hist in enumerate(histograms):
        plot_single(hist, log_y=log_y, ax=axes[idx // ncols][idx % ncols])
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Byte-group component plots
# ---------------------------------------------------------------------------

def plot_byte_group_size(hist: dict, group_size: int,
                         entries: list, log_y: bool) -> plt.Figure:
    """One figure, one subplot per adjacent starting position for this group size."""
    n     = len(entries)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    int_w = hist["int_width"]
    width_label = f"int{int_w * 8}"

    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle(
        f"{group_size}-byte group distributions – {hist['name']}  [{width_label}]",
        fontsize=12)

    for i, comp in enumerate(entries):
        ax    = axes[i // ncols][i % ncols]
        color = _BYTE_COLOURS[comp["start_byte"] % len(_BYTE_COLOURS)]
        label = comp["name"].split("–", 1)[-1].strip()
        _bar_plot_on_ax(ax, comp, log_y=log_y, color=color, title=label)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Byte-pair analysis helpers
# ---------------------------------------------------------------------------

def _build_pair_matrix(values: np.ndarray, counts: np.ndarray,
                       byte_lo: int, byte_hi: int) -> np.ndarray:
    """256×256 frequency matrix M[hi, lo]."""
    lo  = ((values >> (byte_lo * 8)) & 0xFF).astype(np.int64)
    hi  = ((values >> (byte_hi * 8)) & 0xFF).astype(np.int64)
    mat = np.zeros((256, 256), dtype=np.float64)
    np.add.at(mat, (hi, lo), counts.astype(np.float64))
    return mat


def _bit_variance(mat: np.ndarray) -> np.ndarray:
    """16-element Bernoulli variance: bits 0-7 from byte_lo, 8-15 from byte_hi."""
    total = mat.sum()
    if total == 0:
        return np.zeros(16)
    bits  = np.arange(256)
    col   = mat.sum(axis=0)
    row   = mat.sum(axis=1)
    var   = np.zeros(16)
    for b in range(8):
        mask = ((bits >> b) & 1).astype(np.float64)
        p = np.dot(col, mask) / total;  var[b]     = p * (1 - p)
        p = np.dot(row, mask) / total;  var[8 + b] = p * (1 - p)
    return var


def _sorted_axes(mat: np.ndarray):
    return (np.argsort(mat.sum(axis=0))[::-1],
            np.argsort(mat.sum(axis=1))[::-1])


def _cumulative_coverage(mat: np.ndarray):
    flat  = mat.flatten()
    total = flat.sum()
    if total == 0:
        return np.array([0]), np.array([0.0])
    desc    = np.sort(flat)[::-1]
    nonzero = desc[desc > 0]
    cumsum  = np.cumsum(nonzero) / total
    return np.arange(1, len(cumsum) + 1), cumsum


def _conditional_entropy(mat: np.ndarray):
    col_sums = mat.sum(axis=0)
    observed = np.where(col_sums > 0)[0]
    cond_h   = np.zeros(len(observed))
    for k, v in enumerate(observed):
        col = mat[:, v]; p = col / col.sum(); nz = p[p > 0]
        cond_h[k] = float(-np.sum(nz * np.log2(nz)))
    col_freq = col_sums[observed] / mat.sum()
    return observed, cond_h, col_freq


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Byte-pair analysis: six plots per adjacent pair
# ---------------------------------------------------------------------------

def plot_byte_pair_analysis(hist: dict, output_dir: Path,
                             log_scale: bool = True, name: str = "") -> None:
    """
    For every adjacent byte pair in `hist` (int_width - 1 pairs total),
    write six PNGs into output_dir:

      1. byte_pair_heatmap_         -- raw 256x256 heatmap + marginal spines
      2. byte_pair_heatmap_sorted_  -- frequency-sorted heatmap
      3. byte_pair_bit_variance_    -- 16-bar Bernoulli variance chart
      4. byte_pair_cumulative_      -- cumulative coverage curve
      5. byte_pair_cond_entropy_    -- H(byte_hi | byte_lo=v) profile
      6. byte_pair_fft_             -- 2-D FFT log-magnitude spectrum
    """
    int_width = hist["int_width"]
    if int_width == 0:
        print("  Skipping: non-integer values.")
        return

    values = np.asarray(hist["values"], dtype=np.int64)
    counts = np.asarray(hist["counts"], dtype=np.float64)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{name}" if name else ""

    def _disp(m):
        return np.log1p(m) if log_scale else m

    scale_label = "log(1 + count)" if log_scale else "count"

    for b in range(int_width - 1):          # adjacent pairs only
        lbl_lo = f"byte{b}"
        lbl_hi = f"byte{b+1}"
        pfx    = f"{lbl_lo}_{lbl_hi}{tag}"

        mat = _build_pair_matrix(values, counts, b, b + 1)
        lo_obs = np.where(mat.sum(axis=0) > 0)[0]
        hi_obs = np.where(mat.sum(axis=1) > 0)[0]
        if lo_obs.size == 0 or hi_obs.size == 0:
            print(f"  Skipping {lbl_lo}×{lbl_hi}: no data.")
            continue

        lo_min, lo_max = int(lo_obs.min()), int(lo_obs.max())
        hi_min, hi_max = int(hi_obs.min()), int(hi_obs.max())
        mat_crop = mat[hi_min:hi_max + 1, lo_min:lo_max + 1]

        # ── 1. Raw heatmap ────────────────────────────────────────────────
        col_marg = mat_crop.sum(axis=0)
        row_marg = mat_crop.sum(axis=1)

        fig = plt.figure(figsize=(10, 10))
        gs  = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                               hspace=0.04, wspace=0.04)
        ax_top   = fig.add_subplot(gs[0, 0])
        ax_main  = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        fig.add_subplot(gs[0, 1]).set_visible(False)

        im = ax_main.imshow(_disp(mat_crop), origin="lower", aspect="auto",
                            cmap="viridis",
                            extent=[lo_min-.5, lo_max+.5, hi_min-.5, hi_max+.5],
                            interpolation="nearest")
        ax_main.set_xlabel(f"{lbl_lo} value  (0–255)")
        ax_main.set_ylabel(f"{lbl_hi} value  (0–255)")
        fig.colorbar(im, ax=ax_main, fraction=0.03, pad=0.02).set_label(
            scale_label, fontsize=8)

        ax_top.bar(np.arange(lo_min, lo_max+1), col_marg, width=1.0,
                   color="steelblue", linewidth=0)
        ax_top.set_xlim(lo_min-.5, lo_max+.5)
        if col_marg.max() > 0:
            ax_top.set_yscale("log")
        ax_top.tick_params(labelbottom=False, bottom=False)
        ax_top.set_ylabel("count", fontsize=7)
        ax_top.set_title(
            f"{lbl_lo} × {lbl_hi} pair frequency" + (f"  —  {name}" if name else ""),
            fontsize=11)

        ax_right.barh(np.arange(hi_min, hi_max+1), row_marg,
                      height=1.0, color="coral", linewidth=0)
        ax_right.set_ylim(hi_min-.5, hi_max+.5)
        if row_marg.max() > 0:
            ax_right.set_xscale("log")
        ax_right.tick_params(labelleft=False, left=False)
        ax_right.set_xlabel("count", fontsize=7)

        _save_fig(fig, output_dir / f"byte_pair_heatmap_{pfx}.png")

        # ── 2. Frequency-sorted heatmap ───────────────────────────────────
        col_ord, row_ord = _sorted_axes(mat)
        col_obs = col_ord[np.isin(col_ord, lo_obs)]
        row_obs = row_ord[np.isin(row_ord, hi_obs)]
        mat_s   = mat[np.ix_(row_obs, col_obs)]
        nc, nr  = mat_s.shape[1], mat_s.shape[0]

        fig_s, ax_s = plt.subplots(figsize=(10, 9))
        im_s = ax_s.imshow(_disp(mat_s), origin="upper", aspect="auto",
                           cmap="viridis", interpolation="nearest")
        fig_s.colorbar(im_s, ax=ax_s, fraction=0.03, pad=0.02).set_label(
            scale_label, fontsize=8)
        ts = max(1, nc // 8)
        ax_s.set_xticks(np.arange(0, nc, ts))
        ax_s.set_xticklabels(col_obs[::ts], fontsize=6, rotation=45)
        ts = max(1, nr // 8)
        ax_s.set_yticks(np.arange(0, nr, ts))
        ax_s.set_yticklabels(row_obs[::ts], fontsize=6)
        ax_s.set_xlabel(f"{lbl_lo} value  (ranked by frequency)")
        ax_s.set_ylabel(f"{lbl_hi} value  (ranked by frequency)")
        ax_s.set_title(
            f"{lbl_lo} × {lbl_hi} — frequency-sorted heatmap"
            + (f"  —  {name}" if name else "") +
            "\nAxes ordered by descending marginal frequency", fontsize=10)
        fig_s.tight_layout()
        _save_fig(fig_s, output_dir / f"byte_pair_heatmap_sorted_{pfx}.png")

        # ── 3. Bit variance ───────────────────────────────────────────────
        var = _bit_variance(mat)
        tot = mat.sum() + 1e-12
        col_f = mat.sum(axis=0) / tot
        row_f = mat.sum(axis=1) / tot
        bits  = np.arange(256)
        p_vals = np.array(
            [float(np.dot(col_f, ((bits >> b) & 1))) for b in range(8)] +
            [float(np.dot(row_f, ((bits >> b) & 1))) for b in range(8)]
        )
        bit_labels = ([f"{lbl_lo}\nb{b}" for b in range(8)] +
                      [f"{lbl_hi}\nb{b}" for b in range(8)])
        colours_bv = np.where(var < 0.05, "#d73027",
                     np.where(var < 0.15, "#fc8d59",
                     np.where(var < 0.20, "#fee090", "#4575b4")))

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        bars2 = ax2.bar(np.arange(16), var, color=colours_bv, width=0.7)
        ax2.axhline(0.25, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax2.axvline(7.5, color="grey", linestyle=":", linewidth=1)
        for bar, p in zip(bars2, p_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.003, f"{p:.2f}",
                     ha="center", va="bottom", fontsize=7, rotation=45)
        ax2.set_xticks(np.arange(16))
        ax2.set_xticklabels(bit_labels, fontsize=8)
        ax2.set_ylim(0, 0.28)
        ax2.set_ylabel("Bernoulli variance  p · (1−p)")
        ax2.set_xlabel("Bit position  (grouped by source byte)")
        ax2.set_title(
            f"Bit variance — {lbl_lo} × {lbl_hi}"
            + (f"  —  {name}" if name else "") +
            "\nRed = near-constant (compressible);  Blue = balanced (high entropy)",
            fontsize=10)
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend(handles=[
            mpatches.Patch(color="#d73027", label="var < 0.05  (near-constant)"),
            mpatches.Patch(color="#fc8d59", label="var < 0.15  (low variance)"),
            mpatches.Patch(color="#fee090", label="var < 0.20  (medium)"),
            mpatches.Patch(color="#4575b4", label="var ≥ 0.20  (near-balanced)"),
        ], fontsize=8, loc="upper left")
        fig2.tight_layout()
        _save_fig(fig2, output_dir / f"byte_pair_bit_variance_{pfx}.png")

        # ── 4. Cumulative coverage ────────────────────────────────────────
        n_codes, cum_frac = _cumulative_coverage(mat)
        milestones = {}
        for thr in (0.50, 0.90, 0.95, 0.99):
            idx = np.searchsorted(cum_frac, thr)
            if idx < len(n_codes):
                milestones[thr] = int(n_codes[idx])

        fig_cv, ax_cv = plt.subplots(figsize=(9, 5))
        ax_cv.plot(n_codes, cum_frac * 100, color="steelblue", linewidth=1.5)
        for thr, nc in milestones.items():
            ax_cv.axhline(thr * 100, color="grey", linestyle=":", linewidth=0.8)
            ax_cv.axvline(nc, color="grey", linestyle=":", linewidth=0.8)
            ax_cv.annotate(
                f"{thr:.0%} @ {nc} codes",
                xy=(nc, thr * 100),
                xytext=(nc + len(n_codes) * 0.02, thr * 100 - 3),
                fontsize=7, color="dimgrey",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))
        ax_cv.set_xscale("log")
        ax_cv.set_xlabel("Distinct (lo, hi) codewords  [log scale]")
        ax_cv.set_ylabel("Cumulative coverage  (%)")
        ax_cv.set_ylim(0, 101)
        ax_cv.set_title(
            f"Cumulative coverage — {lbl_lo} × {lbl_hi}"
            + (f"  —  {name}" if name else ""), fontsize=10)
        ax_cv.grid(alpha=0.3)
        fig_cv.tight_layout()
        _save_fig(fig_cv, output_dir / f"byte_pair_cumulative_{pfx}.png")

        # ── 5. Conditional entropy ────────────────────────────────────────
        lo_vals_ce, cond_h, col_freq = _conditional_entropy(mat)
        sort_idx  = np.argsort(col_freq)[::-1]
        lo_sorted = lo_vals_ce[sort_idx]
        ce_sorted = cond_h[sort_idx]
        cf_sorted = col_freq[sort_idx]
        n_shown   = min(len(lo_sorted), 64)

        fig_ce, ax_ce = plt.subplots(figsize=(max(10, n_shown * 0.25), 5))
        bars_ce = ax_ce.bar(np.arange(n_shown), ce_sorted[:n_shown],
                            width=0.8, color="steelblue", alpha=0.85)
        ce_max = ce_sorted[:n_shown].max() if n_shown > 0 else 1.0
        for bar, h_val in zip(bars_ce, ce_sorted[:n_shown]):
            frac = h_val / ce_max if ce_max > 0 else 0
            bar.set_facecolor(plt.cm.RdYlGn_r(frac))   # type: ignore

        row_marg_full = mat.sum(axis=1)
        row_marg_full /= row_marg_full.sum()
        nz = row_marg_full[row_marg_full > 0]
        h_uncond = float(-np.sum(nz * np.log2(nz)))
        ax_ce.axhline(h_uncond, color="black", linestyle="--", linewidth=1,
                      label=f"H({lbl_hi}) unconditional = {h_uncond:.2f} bits")
        ax_ce.set_xticks(np.arange(n_shown))
        ax_ce.set_xticklabels(
            [f"{v}\n({cf:.1%})" for v, cf in
             zip(lo_sorted[:n_shown], cf_sorted[:n_shown])],
            fontsize=6, rotation=45, ha="right")
        ax_ce.set_ylabel(f"H({lbl_hi} | {lbl_lo}=v)  (bits)")
        ax_ce.set_xlabel(
            f"{lbl_lo} value  (sorted by frequency; showing top {n_shown})")
        ax_ce.set_ylim(0, 8.2)
        ax_ce.set_title(
            f"Conditional entropy H({lbl_hi} | {lbl_lo}=v)"
            + (f"  —  {name}" if name else ""), fontsize=9)
        ax_ce.legend(fontsize=8)
        ax_ce.grid(axis="y", alpha=0.3)
        fig_ce.tight_layout()
        _save_fig(fig_ce, output_dir / f"byte_pair_cond_entropy_{pfx}.png")

        # ── 6. 2-D FFT ────────────────────────────────────────────────────
        mat_norm = mat / (mat.sum() + 1e-12)
        fft2_mag = np.abs(np.fft.fftshift(np.fft.fft2(mat_norm)))
        cy, cx   = fft2_mag.shape[0] // 2, fft2_mag.shape[1] // 2
        fft2_mag[cy, cx] = 0.0
        fft2_disp = np.log1p(fft2_mag)

        fig_fft, ax_fft = plt.subplots(figsize=(9, 8))
        im_fft = ax_fft.imshow(fft2_disp, origin="upper", aspect="auto",
                               cmap="inferno", interpolation="nearest")
        fig_fft.colorbar(im_fft, ax=ax_fft, fraction=0.03, pad=0.02).set_label(
            "log(1 + |FFT|)  [DC zeroed]", fontsize=8)
        fticks = np.linspace(0, 255, 9, dtype=int)
        ax_fft.set_xticks(fticks)
        ax_fft.set_xticklabels([f"{int(f)-128}" for f in fticks], fontsize=7)
        ax_fft.set_yticks(fticks)
        ax_fft.set_yticklabels([f"{int(f)-128}" for f in fticks], fontsize=7)
        ax_fft.set_xlabel("Spatial frequency  (byte_lo direction)")
        ax_fft.set_ylabel("Spatial frequency  (byte_hi direction)")
        ax_fft.set_title(
            f"2D FFT — {lbl_lo} × {lbl_hi}"
            + (f"  —  {name}" if name else ""), fontsize=9)
        top5 = np.argpartition(fft2_disp.flatten(), -5)[-5:]
        for idx_flat in top5:
            fy, fx = divmod(int(idx_flat), 256)
            ax_fft.plot(fx, fy, "r+", markersize=8, markeredgewidth=1.5)
            ax_fft.annotate(f"({fx-128},{fy-128})", xy=(fx, fy),
                            xytext=(fx+4, fy-4), fontsize=6, color="red")
        fig_fft.tight_layout()
        _save_fig(fig_fft, output_dir / f"byte_pair_fft_{pfx}.png")

        # ── Console summary ───────────────────────────────────────────────
        n_distinct_pairs = int((mat > 0).sum())
        print(f"  {lbl_lo}×{lbl_hi}: {n_distinct_pairs} distinct pairs, "
              f"90%@{milestones.get(0.90,'?')} codes, "
              f"H_uncond={h_uncond:.2f} bits")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot value-frequency histograms (auto-detects int32/int64).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", type=Path,
                        help="One or more histogram CSV files.")
    parser.add_argument("--overlay", action="store_true",
                        help="Overlay all histograms on one axes.")
    parser.add_argument("--top", type=int, default=0, metavar="N",
                        help="Keep only the N most frequent values (0 = all).")
    parser.add_argument("--log-y", action="store_true",
                        help="Log-scale y-axis.")
    parser.add_argument("--output", type=Path, default=None, metavar="FILE",
                        help="Save main figure here instead of displaying.")
    parser.add_argument("--byte-components", action="store_true",
                        help=(
                            "Plot overall histogram plus byte-group distributions "
                            "for every adjacent group of size 1-3 (int32) or 1-7 (int64). "
                            "Saved as <stem>_overall.png, <stem>_bytes1.png, ..."))
    parser.add_argument("--byte-pair-analysis", action="store_true",
                        help=(
                            "Plot adjacent byte-pair analyses (3 pairs for int32, "
                            "7 pairs for int64).  PNGs written to --output's folder."))
    parser.add_argument("--byte-pair-name", type=str, default="", metavar="NAME",
                        help="Short label appended to byte-pair filenames.")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    histograms = []
    for path in args.files:
        if not path.exists():
            print(f"Warning: file not found: {path}", file=sys.stderr)
            continue
        h = load_histogram(path)
        if args.top > 0:
            h = top_n(h, args.top)
        shown = int(h["counts"].sum()); total = h["total"]
        pct   = shown / total * 100.0 if total > 0 else 0.0
        width_str = (f"int{h['int_width']*8}" if h["int_width"] else "float")
        print(f"Loaded {h['name']}: {shown:,} / {total:,} ({pct:.1f}%)  [{width_str}]")
        histograms.append(h)

    if not histograms:
        print("No valid CSV files found.", file=sys.stderr)
        sys.exit(1)

    # ── Main histogram figure ─────────────────────────────────────────────
    if args.overlay and len(histograms) > 1:
        fig = plot_overlay(histograms, log_y=args.log_y)
    elif len(histograms) == 1:
        fig = plot_single(histograms[0], log_y=args.log_y)
    else:
        fig = plot_grid(histograms, log_y=args.log_y)

    def _save_or_show(f, path):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            f.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved to {path}")
            plt.close(f)
        else:
            plt.show()

    _save_or_show(fig, args.output)

    # ── Byte-group component plots ────────────────────────────────────────
    if args.byte_components:
        for h in histograms:
            if not values_are_integers(h):
                print(f"Skipping byte-components for '{h['name']}': non-integer.")
                continue

            int_w     = h["int_width"]
            max_grp   = int_w - 1
            print(f"\nByte-group breakdown for '{h['name']}'  "
                  f"[int{int_w*8}, groups 1–{max_grp}]:")

            # Overall histogram
            overall_path = (args.output.with_stem(args.output.stem + "_overall")
                            if args.output else None)
            _save_or_show(plot_single(h, log_y=args.log_y), overall_path)

            # Per-group-size
            all_groups = byte_group_histograms(h)
            for nbytes, entries in sorted(all_groups.items()):
                for comp in entries:
                    nd  = comp["n_distinct"]
                    mx  = comp["max_distinct"]
                    rep = 1.0 - nd / mx if mx > 0 else 0.0
                    label = comp["name"].split("–", 1)[-1].strip()
                    print(f"  [{nbytes}B] {label:48s}  "
                          f"distinct={nd:12,}/{mx:12,}   repetition={rep:.1%}")

                grp_path = (args.output.with_stem(args.output.stem + f"_bytes{nbytes}")
                            if args.output else None)
                _save_or_show(plot_byte_group_size(h, nbytes, entries, args.log_y),
                              grp_path)

    # ── Byte-pair analysis ────────────────────────────────────────────────
    if args.byte_pair_analysis:
        bpa_dir = args.output.parent if args.output else Path(".")
        for h in histograms:
            if not values_are_integers(h):
                print(f"Skipping byte-pair analysis for '{h['name']}': non-integer.")
                continue
            int_w = h["int_width"]
            print(f"\nByte-pair analysis for '{h['name']}'  "
                  f"[int{int_w*8}, {int_w-1} adjacent pairs]:")
            name_tag = args.byte_pair_name or h["name"]
            plot_byte_pair_analysis(h, bpa_dir, log_scale=True, name=name_tag)


if __name__ == "__main__":
    main()
