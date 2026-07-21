#!/usr/bin/env python3
"""
Plot range-access heatmaps and throughput CDF for the SubIntSplit motivation figure.

Heatmap figures (eight, PDF + PNG each) under plots/motivational_heatmaps/:
  heatmap_{throughput,input_bw}_{raw,interp}_{shared,perplot}.{pdf,png}

  Subplots are arranged left-to-right in ascending compression-ratio order.
  raw    — measured sample grid
  interp — cubic-interpolated continuous surface
  shared — single colorbar shared across all encoder subplots (for comparison)
  perplot — individual colorbar per encoder (reveals internal structure)

CDF figure (one, PDF + PNG):
  throughput_cdf.{pdf,png}
  Empirical CDF of decoded throughput across all (A,B) pairs per encoder.
  Curves coloured by compression ratio (light = low compression, dark = high).

Run from EncodingsPlayground/:
    python3 Benchmarks/plot_heatmap.py [--input path/to/heatmap_benchmark.csv] [--output plots/dir] [--skip-interp]

    --skip-interp skips the cubic-interpolation heatmap figures, which are by
    far the slowest to render (scipy.interpolate.griddata over a 500x500 grid
    per encoder, per metric, per legend mode).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from matplotlib.cm import ScalarMappable, get_cmap
from scipy.interpolate import griddata

DEFAULT_CSV    = Path("Benchmarks/results/heatmap_benchmark.csv")
OUT_DIR        = Path("plots/motivational_heatmaps")
INTERP_RES     = 500   # pixels on each axis for the interpolated grid
HEATMAP_NCOLS  = 4     # subplots per row in grid layout
DATASET_N      = 1_000_000  # must match N in heatmap_benchmark.cpp; used only to
                             # render real element counts in the 2-D colour-key legend


def _encoders_by_compression(df: pd.DataFrame) -> list[str]:
    """Return encoder names sorted ascending by median compression ratio."""
    ratios = (
        df.groupby("encoding")["compression_ratio"]
        .median()
        .sort_values()
    )
    return list(ratios.index)


def _encoder_ratios(df: pd.DataFrame, encoders: list[str]) -> dict[str, float]:
    return {enc: float(df[df["encoding"] == enc]["compression_ratio"].median())
            for enc in encoders}


def _global_norm(df: pd.DataFrame, encoders: list[str], metric: str) -> mcolors.LogNorm:
    parts = []
    for enc in encoders:
        vals = df[df["encoding"] == enc][metric].values
        pos  = vals[np.isfinite(vals) & (vals > 0)]
        if pos.size:
            parts.append(pos)
    combined = np.concatenate(parts)
    return mcolors.LogNorm(vmin=float(combined.min()), vmax=float(combined.max()))


def _local_norm(vals: np.ndarray) -> mcolors.LogNorm:
    pos = vals[np.isfinite(vals) & (vals > 0)]
    return mcolors.LogNorm(vmin=float(pos.min()), vmax=float(pos.max()))


def _label_ax(ax, enc: str, ratio: float) -> None:
    ax.set_title(f"{enc}\n({ratio:.2f}×)", fontsize=8, pad=3)
    ax.set_xlabel(r"Start  $A/N$",  fontsize=7)
    ax.set_ylabel(r"Length  $B/N$", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.plot([0, 1], [1, 0], color="white", lw=0.7, ls="--", alpha=0.5, zorder=3)


def _per_colorbar(ax, im, metric_label: str) -> None:
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(metric_label, fontsize=6)
    cbar.ax.tick_params(labelsize=5)


def _shared_colorbar(fig, axes, cmap: str, norm: mcolors.Normalize,
                     metric_label: str) -> None:
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(),
                 label=metric_label, shrink=0.8, aspect=25, pad=0.02)


def _add_row_compression_lines(fig, axes, nrows: int, ncols: int, n: int) -> None:
    """
    Draw a horizontal rule above each subplot row and label the first row
    "Low Compression" and the last row "High Compression".
    Uses fig.canvas.draw() to read constrained-layout positions before annotating.
    """
    fig.canvas.draw()  # force constrained_layout to finalise positions

    for row in range(nrows):
        n_in_row = min(ncols, n - row * ncols)
        bbox_l = axes[row, 0].get_position()
        bbox_r = axes[row, n_in_row - 1].get_position()

        y  = bbox_l.y1 + 0.014   # just above this row's top edge
        x0 = bbox_l.x0
        x1 = bbox_r.x1

        fig.add_artist(Line2D(
            [x0, x1], [y, y],
            transform=fig.transFigure,
            color="#555555", lw=2.2, clip_on=False, zorder=10,
        ))

        if nrows == 1:
            label = "Low → High Compression"
            ha    = "left"
            xpos  = x0
        elif row == 0:
            label = "Low Compression"
            ha    = "left"
            xpos  = x0
        elif row == nrows - 1:
            label = "High Compression"
            ha    = "left"
            xpos  = x0
        else:
            continue

        fig.text(xpos, y + 0.011, label,
                 ha=ha, va="bottom", fontsize=8.5,
                 color="#555555", style="italic", fontweight="semibold")


def _make_raw_figure(
    df: pd.DataFrame,
    encoders: list[str],
    ratios: dict[str, float],
    metric: str,
    metric_label: str,
    cmap: str,
    out_stem: str,
    shared_legend: bool,
) -> None:
    """Grid of subplots ordered left-to-right, top-to-bottom by ascending compression ratio."""
    global_norm = _global_norm(df, encoders, metric) if shared_legend else None
    n     = len(encoders)
    ncols = min(n, HEATMAP_NCOLS)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.6, nrows * 3.8 + 0.6),
                             squeeze=False, constrained_layout=True)

    for ax, enc in zip(axes.flat, encoders):
        sub = df[df["encoding"] == enc]
        if sub.empty:
            ax.set_visible(False)
            continue

        pivot = sub.pivot_table(values=metric, index="B_frac", columns="A_frac",
                                aggfunc="median")
        vals = pivot.to_numpy()
        norm = global_norm if shared_legend else _local_norm(vals.ravel())
        im   = ax.pcolormesh(pivot.columns.to_numpy(), pivot.index.to_numpy(),
                             vals, cmap=cmap, norm=norm, shading="nearest")
        _label_ax(ax, enc, ratios[enc])
        if not shared_legend:
            _per_colorbar(ax, im, metric_label)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    if shared_legend:
        assert global_norm is not None
        _shared_colorbar(fig, axes, cmap, global_norm, metric_label)

    _add_row_compression_lines(fig, axes, nrows, ncols, n)
    legend_tag = "shared" if shared_legend else "perplot"
    fig.suptitle(f"{metric_label}  (sample grid, {legend_tag} legend)", fontsize=11)
    _save(fig, out_stem)


def _make_interp_figure(
    df: pd.DataFrame,
    encoders: list[str],
    ratios: dict[str, float],
    metric: str,
    metric_label: str,
    cmap: str,
    out_stem: str,
    shared_legend: bool,
) -> None:
    """Grid of subplots ordered left-to-right, top-to-bottom by ascending compression ratio."""
    global_norm = _global_norm(df, encoders, metric) if shared_legend else None
    n     = len(encoders)
    ncols = min(n, HEATMAP_NCOLS)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.6, nrows * 3.8 + 0.6),
                             squeeze=False, constrained_layout=True)

    xi = np.linspace(0.0, 1.0, INTERP_RES)
    yi = np.linspace(0.0, 1.0, INTERP_RES)
    xg, yg = np.meshgrid(xi, yi)
    triangle_mask = (xg + yg) > 1.0

    for ax, enc in zip(axes.flat, encoders):
        sub = df[df["encoding"] == enc]
        if sub.empty:
            ax.set_visible(False)
            continue

        points = sub[["A_frac", "B_frac"]].to_numpy()
        values = sub[metric].to_numpy()
        finite = np.isfinite(values) & (values > 0)
        if finite.sum() < 4:
            ax.set_visible(False)
            continue

        zi = griddata(points[finite], values[finite], (xg, yg), method="cubic")
        zi = np.clip(zi, float(values[finite].min()), None)
        zi[triangle_mask] = np.nan

        norm = global_norm if shared_legend else _local_norm(zi.ravel())
        im   = ax.pcolormesh(xi, yi, zi, cmap=cmap, norm=norm, shading="auto")
        _label_ax(ax, enc, ratios[enc])
        if not shared_legend:
            _per_colorbar(ax, im, metric_label)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    if shared_legend:
        assert global_norm is not None
        _shared_colorbar(fig, axes, cmap, global_norm, metric_label)

    _add_row_compression_lines(fig, axes, nrows, ncols, n)
    legend_tag = "shared" if shared_legend else "perplot"
    fig.suptitle(f"{metric_label}  (cubic interpolation, {legend_tag} legend)", fontsize=11)
    _save(fig, out_stem)


def _make_throughput_cdf(
    df: pd.DataFrame,
    encoders: list[str],
    ratios: dict[str, float],
    out_stem: str = "throughput_cdf",
) -> None:
    """
    Empirical CDF of decoded throughput (elem_Meps) across all (A,B) sample pairs.
    Each encoder's curve is coloured by its compression ratio:
      light (yellow) = low compression, dark (brown) = high compression.
    A curve shifted right = consistently high throughput.
    No dark curve is also shifted right — that is the point.
    """
    ratio_vals = list(ratios.values())
    ratio_norm = mcolors.Normalize(vmin=min(ratio_vals), vmax=max(ratio_vals))
    # Clip the lightest ~35 % of YlOrBr so low-compression curves remain visible on white.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "YlOrBr_vis", get_cmap("YlOrBr")(np.linspace(0.35, 1.0, 256))
    )

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    for enc in encoders:
        vals = df[df["encoding"] == enc]["elem_Meps"].values
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        sorted_vals = np.sort(vals)
        cdf         = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        color       = cmap(ratio_norm(ratios[enc]))
        ax.plot(sorted_vals, cdf, color=color, lw=1.8,
                label=f"{enc}  ({ratios[enc]:.2f}×)")

    ax.set_xscale("log")
    ax.set_xlabel("Decoded throughput  (M elements / s)", fontsize=11)
    ax.set_ylabel(r"$F(t) = P\,(\mathrm{throughput}(A,B) \leq t)$", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8, loc="upper left")

    sm = ScalarMappable(cmap=cmap, norm=ratio_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Compression ratio  (×)", pad=0.02)

    fig.suptitle("Throughput CDF per encoding, coloured by compression ratio", fontsize=12)
    _save(fig, out_stem)


def _dot_colors_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    2-D colour encoding via HSV:
      hue   = A/B ratio, traversing red → magenta → blue
              (backward around the wheel, so green/yellow are never crossed)
      sat   = 0.90 constant — always vivid
      value = scales with A+B magnitude so near-(0,0) points are darker
    Results: pure-A dots are red, pure-B dots are blue, equal-mix dots are magenta.
    No route to brown or olive regardless of overlap or alpha compositing.
    """
    ratio = b / (a + b + 1e-9)          # 0 = pure A, 1 = pure B
    hue   = (1.0 - ratio / 3.0) % 1.0  # red=0.0, magenta=0.833, blue=0.667
    sat   = np.full_like(a, 0.90)
    val   = np.clip(0.55 + 0.45 * (a + b), 0.55, 1.0)
    hsv   = np.column_stack([hue, sat, val])
    return mcolors.hsv_to_rgb(hsv)


def _format_count(x: float) -> str:
    """Human-readable element count, e.g. 750_000 -> '750K', 1_000_000 -> '1M'."""
    if x >= 1_000_000:
        return f"{x / 1_000_000:.3g}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def _format_minmax_note(df: pd.DataFrame, col: str, label: str) -> str:
    """e.g. 'Range Length $\\in$ [16K, 1M]' using the real element-count scale."""
    lo = float(df[col].min()) * DATASET_N
    hi = float(df[col].max()) * DATASET_N
    return rf"{label} $\in$ [{_format_count(lo)}, {_format_count(hi)}]"


def _colored_suptitle(fig, lines: list[list[tuple[str, str, str]]],
                      fontsize: float = 12.5, line_sep: float = 3.0) -> None:
    """
    Render a horizontally packed, multi-coloured, optionally multi-line figure title,
    positioned snugly above the axes. lines: list of lines, each a list of
    (text, color, fontweight) segments concatenated left-to-right; lines are
    stacked vertically and centered. Reserves matching constrained-layout space
    via a blank fig.suptitle() placeholder sized to the same line count/fontsize,
    so there is no large gap between the title and the plot below it.
    """
    fig.suptitle("\n".join(" " for _ in lines), fontsize=fontsize)

    row_boxes = [
        HPacker(
            children=[
                TextArea(text, textprops=dict(color=color, fontsize=fontsize, fontweight=weight))
                for text, color, weight in segs
            ],
            align="baseline", pad=0, sep=0,
        )
        for segs in lines
    ]
    packer = VPacker(children=row_boxes, align="center", pad=0, sep=line_sep)
    ab = AnnotationBbox(packer, (0.5, 0.995), xycoords="figure fraction",
                        box_alignment=(0.5, 1.0), frameon=False, annotation_clip=False)
    fig.add_artist(ab)


def _add_2d_color_legend(ax, dataset_n: int) -> None:
    """
    Standalone colour-key legend placed outside the main axes, showing the
    (Start Offset, Range Length) → colour mapping in real element counts
    rather than normalised [0, 1] fractions.
    """
    ins = ax.inset_axes([1.18, 0.25, 0.16, 0.45])
    res = 120
    A = np.linspace(0.0, 1.0, res)
    B = np.linspace(0.0, 1.0, res)
    Ag, Bg = np.meshgrid(A, B)
    img = _dot_colors_2d(Ag.ravel(), Bg.ravel()).reshape(res, res, 3)
    # grey out the invalid region (Start Offset + Range Length > N — impossible samples)
    img[(Ag + Bg) > 1.0] = [0.90, 0.90, 0.90]

    ins.imshow(img, origin="lower", extent=[0.0, dataset_n, 0.0, dataset_n],
               aspect="auto", interpolation="bilinear")
    ins.plot([0, dataset_n], [dataset_n, 0], color="white", lw=0.8, ls="--", alpha=0.7)

    formatter = mticker.FuncFormatter(lambda x, _: _format_count(x))
    ticks = np.linspace(0, dataset_n, 5)
    ins.set_xticks(ticks)
    ins.set_yticks(ticks)
    ins.xaxis.set_major_formatter(formatter)
    ins.yaxis.set_major_formatter(formatter)

    ins.set_xlabel("Start Offset",  fontsize=8)
    ins.set_ylabel("Range Length",  fontsize=8)
    ins.tick_params(labelsize=7, length=3)
    ins.set_title("Colour Key", fontsize=9.5, pad=5, fontweight="semibold")


def _make_strip_plot(
    df: pd.DataFrame,
    encoders: list[str],
    ratios: dict[str, float],
    color_by: str = "ratio",   # "ratio" | "A_frac" | "B_frac" | "AB"
    out_stem: str = "throughput_strip",
    vertical: bool = False,
) -> None:
    """
    Strip / dot plot of decoded throughput.
    Horizontal (default): one row per encoding (low compression at top, high at
    bottom), throughput on the x-axis.
    Vertical (vertical=True): one column per encoding (low compression at left,
    high at right), throughput on the y-axis.
    Each dot = one (A,B) sample, jittered to reveal density.
    color_by controls dot colour:
      "ratio"  — compression ratio (YlOrBr, consistent with CDF)
      "A_frac" — start position A/N  (white → red)
      "B_frac" — range length  B/N  (white → blue)
      "AB"     — 2-D: R=A_frac, B=B_frac; red=large start, blue=long range, purple=both
    """
    if color_by == "ratio":
        ratio_vals = list(ratios.values())
        cnorm      = mcolors.Normalize(vmin=min(ratio_vals), vmax=max(ratio_vals))
        cmap       = mcolors.LinearSegmentedColormap.from_list(
            "YlOrBr_vis", get_cmap("YlOrBr")(np.linspace(0.35, 1.0, 256)))
        cbar_label = "Compression ratio  (×)"
    elif color_by == "A_frac":
        cnorm      = mcolors.Normalize(vmin=0.0, vmax=1.0)
        cmap       = mcolors.LinearSegmentedColormap.from_list(
            "Reds_vis", get_cmap("Reds")(np.linspace(0.25, 1.0, 256)))
        cbar_label = "Start position  A/N"
    elif color_by == "B_frac":
        cnorm      = mcolors.Normalize(vmin=0.0, vmax=1.0)
        cmap       = mcolors.LinearSegmentedColormap.from_list(
            "Blues_vis", get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
        cbar_label = "Range length  B/N"
    else:  # "AB" — 2-D colouring, no single colormap
        cnorm = cmap = cbar_label = None

    n = len(encoders)
    if vertical:
        fig_w = max(4.5, n * 0.9 + 1.6) + (2.2 if color_by == "AB" else 0.0)
        fig_h = 6.0
    else:
        fig_h = max(3.5, n * 0.65 + 1.4)
        fig_w = 10.5 if color_by == "AB" else 9   # extra width for the outside colour-key legend
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    for i, enc in enumerate(encoders):
        sub  = df[df["encoding"] == enc]
        vals = sub["elem_Meps"].values
        mask = np.isfinite(vals) & (vals > 0)
        vals = vals[mask]
        if vals.size == 0:
            continue
        rng    = np.random.default_rng(i)
        jitter = rng.uniform(-0.22, 0.22, size=vals.size)
        pos    = np.full_like(vals, float(i)) + jitter
        x, y   = (pos, vals) if vertical else (vals, pos)

        if color_by == "ratio":
            assert cmap is not None and cnorm is not None
            ax.scatter(x, y, color=cmap(cnorm(ratios[enc])),
                       s=5, alpha=0.45, linewidths=0, zorder=2)
        elif color_by == "AB":
            dot_c = _dot_colors_2d(sub["A_frac"].values[mask],
                                   sub["B_frac"].values[mask])
            ax.scatter(x, y, color=dot_c,
                       s=5, alpha=0.45, linewidths=0, zorder=2)
        else:
            assert cmap is not None and cnorm is not None
            c_vals = sub[color_by].values[mask]
            ax.scatter(x, y, c=c_vals, cmap=cmap, norm=cnorm,
                       s=5, alpha=0.45, linewidths=0, zorder=2)

    if vertical:
        # index 0 = lowest compression at the left, reading left→right
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(
            [f"{enc}\n({ratios[enc]:.2f}×)" for enc in encoders],
            fontsize=8, rotation=35, ha="right",
        )
        ax.set_yscale("log")
        ax.set_ylabel("Decoded throughput  (M elements / s)", fontsize=11)
        ax.grid(True, axis="y", which="both", ls=":", lw=0.5, alpha=0.6)
        ax.tick_params(axis="x", length=0)

        # Horizontal arrow above the plot: left (low compression) → right (high compression)
        ax.annotate(
            "", xy=(0.99, 1.06), xytext=(0.01, 1.06),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=2.0, mutation_scale=14),
            annotation_clip=False,
        )
        ax.text(0.01, 1.085, "Low Compression",  transform=ax.transAxes,
                ha="left",  va="bottom", fontsize=8, color="#555555", style="italic",
                fontweight="semibold")
        ax.text(0.99, 1.085, "High Compression", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#555555", style="italic",
                fontweight="semibold")
    else:
        # index 0 is lowest compression; invert so it reads top→bottom
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f"{enc}  ({ratios[enc]:.2f}×)" for enc in encoders],
            fontsize=8.5,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Decoded throughput  (M elements / s)", fontsize=11)
        ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
        ax.tick_params(axis="y", length=0)

        # Vertical arrow: top (low compression) → bottom (high compression)
        ax.annotate(
            "", xy=(1.04, 0.02), xytext=(1.04, 0.98),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=2.0, mutation_scale=14),
            annotation_clip=False,
        )
        ax.text(1.06, 0.98, "Low\nCompression",  transform=ax.transAxes,
                ha="left", va="top",    fontsize=8, color="#555555", style="italic",
                fontweight="semibold")
        ax.text(1.06, 0.02, "High\nCompression", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color="#555555", style="italic",
                fontweight="semibold")

    if color_by == "AB":
        _add_2d_color_legend(ax, DATASET_N)
        title_fontsize = 11 if vertical else 12.5
        if vertical:
            lines = [
                [("Range-Decode Throughput across all", "#222222", "normal")],
                [("<", "#222222", "normal"), ("Start Offset", "#cc0000", "bold"),
                 (", ", "#222222", "normal"), ("Range Length", "#0033cc", "bold"),
                 ("> Combinations", "#222222", "normal")],
            ]
        else:
            lines = [[
                ("Range-Decode Throughput across all <", "#222222", "normal"),
                ("Start Offset", "#cc0000", "bold"),
                (", ", "#222222", "normal"),
                ("Range Length", "#0033cc", "bold"),
                ("> Combinations", "#222222", "normal"),
            ]]
        _colored_suptitle(fig, lines, fontsize=title_fontsize)
    else:
        assert cmap is not None and cnorm is not None
        sm = ScalarMappable(cmap=cmap, norm=cnorm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=cbar_label, pad=0.13, shrink=0.8)

        title_text = "Range-Decode Throughput across all <Start Offset, Range Length> Combinations"
        if vertical:
            title_text = "Range-Decode Throughput across all\n<Start Offset, Range Length> Combinations"
        fig.suptitle(title_text, fontsize=11 if vertical else 12.5)

        if color_by == "ratio":
            note = (_format_minmax_note(df, "A_frac", "Start Offset") + "      " +
                    _format_minmax_note(df, "B_frac", "Range Length"))
        elif color_by == "A_frac":
            note = _format_minmax_note(df, "B_frac", "Range Length")
        else:  # B_frac
            note = _format_minmax_note(df, "A_frac", "Start Offset")

        ax.set_title(note, fontsize=8, color="#666666", style="italic", pad=6)

    _save(fig, out_stem)


def _make_box_plot(
    df: pd.DataFrame,
    encoders: list[str],
    ratios: dict[str, float],
    vertical: bool = False,
    out_stem: str = "throughput_box",
) -> None:
    """
    Box-and-whisker plot of decoded throughput per encoder — the aggregated
    counterpart to the compression-ratio strip plot, summarising the same
    per-(A,B)-sample distribution instead of showing every point.
    Box facecolor encodes compression ratio (YlOrBr, consistent with strip/CDF plots).
    """
    ratio_vals = list(ratios.values())
    cnorm = mcolors.Normalize(vmin=min(ratio_vals), vmax=max(ratio_vals))
    cmap  = mcolors.LinearSegmentedColormap.from_list(
        "YlOrBr_vis", get_cmap("YlOrBr")(np.linspace(0.35, 1.0, 256)))

    n    = len(encoders)
    data = []
    for enc in encoders:
        vals = df[df["encoding"] == enc]["elem_Meps"].values
        vals = vals[np.isfinite(vals) & (vals > 0)]
        data.append(vals)

    if vertical:
        fig_w = max(4.5, n * 0.9 + 1.6)
        fig_h = 6.0
    else:
        fig_h = max(3.5, n * 0.65 + 1.4)
        fig_w = 9.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    positions = list(range(n))
    bp = ax.boxplot(
        data, positions=positions, vert=vertical,
        widths=0.6, patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=2, alpha=0.3, markeredgewidth=0),
        medianprops=dict(color="#222222", lw=1.6),
        whiskerprops=dict(color="#555555", lw=1.1),
        capprops=dict(color="#555555", lw=1.1),
        boxprops=dict(lw=1.4, edgecolor="#222222"),
    )
    for patch, enc in zip(bp["boxes"], encoders):
        patch.set_facecolor(cmap(cnorm(ratios[enc])))
        patch.set_alpha(0.95)

    # Some encoders have an IQR too narrow to show the fill colour at all — draw a
    # fixed-size colour swatch at each box's median so compression ratio stays
    # legible regardless of how tight the distribution is.
    for i, (enc, vals) in enumerate(zip(encoders, data)):
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        mx, my = (i, med) if vertical else (med, i)
        ax.scatter([mx], [my], marker="s", s=60, color=cmap(cnorm(ratios[enc])),
                   edgecolor="#222222", linewidth=0.9, zorder=5)

    if vertical:
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{enc}\n({ratios[enc]:.2f}×)" for enc in encoders],
            fontsize=8, rotation=35, ha="right",
        )
        ax.set_yscale("log")
        ax.set_ylabel("Decoded throughput  (M elements / s)", fontsize=11)
        ax.grid(True, axis="y", which="both", ls=":", lw=0.5, alpha=0.6)
        ax.tick_params(axis="x", length=0)

        ax.annotate(
            "", xy=(0.99, 1.06), xytext=(0.01, 1.06),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=2.0, mutation_scale=14),
            annotation_clip=False,
        )
        ax.text(0.01, 1.085, "Low Compression",  transform=ax.transAxes,
                ha="left",  va="bottom", fontsize=8, color="#555555", style="italic",
                fontweight="semibold")
        ax.text(0.99, 1.085, "High Compression", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#555555", style="italic",
                fontweight="semibold")
    else:
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [f"{enc}  ({ratios[enc]:.2f}×)" for enc in encoders],
            fontsize=8.5,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Decoded throughput  (M elements / s)", fontsize=11)
        ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
        ax.tick_params(axis="y", length=0)

        ax.annotate(
            "", xy=(1.04, 0.02), xytext=(1.04, 0.98),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=2.0, mutation_scale=14),
            annotation_clip=False,
        )
        ax.text(1.06, 0.98, "Low\nCompression",  transform=ax.transAxes,
                ha="left", va="top",    fontsize=8, color="#555555", style="italic",
                fontweight="semibold")
        ax.text(1.06, 0.02, "High\nCompression", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color="#555555", style="italic",
                fontweight="semibold")

    sm = ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Compression ratio  (×)", pad=0.13, shrink=0.8)

    title_text = "Range-Decode Throughput Distribution across all <Start Offset, Range Length> Combinations"
    if vertical:
        title_text = "Range-Decode Throughput Distribution across all\n<Start Offset, Range Length> Combinations"
    fig.suptitle(title_text, fontsize=11 if vertical else 12.5)

    note = (_format_minmax_note(df, "A_frac", "Start Offset") + "      " +
            _format_minmax_note(df, "B_frac", "Range Length"))
    ax.set_title(note, fontsize=8, color="#666666", style="italic", pad=6)

    _save(fig, out_stem)


_FIGURES_SAVED = 0


def _save(fig, stem: str) -> None:
    global _FIGURES_SAVED
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_DIR / f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(str(OUT_DIR / f"{stem}.png"), bbox_inches="tight", dpi=150)
    print(f"  Saved {OUT_DIR}/{stem}.{{pdf,png}}")
    plt.close(fig)
    _FIGURES_SAVED += 1


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(
        description="Plot range-access heatmaps, CDF, and strip plots for the "
                     "SubIntSplit motivation figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV,
                        help="Path to heatmap_benchmark.csv (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUT_DIR,
                        help="Directory to write plot files into (default: %(default)s)")
    parser.add_argument("--skip-interp", action="store_true",
                        help="Skip the cubic-interpolation heatmap figures (slowest to render).")
    parser.add_argument("--filter-regressive-encoders", action="store_true",
                        help="Filter out encoders with worse compression than Raw (ratio < 1.0).")
    args = parser.parse_args()
    csv_path = args.input
    OUT_DIR  = args.output

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run heatmap_benchmark first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, encoders: {list(df['encoding'].unique())}")

    encoders = _encoders_by_compression(df)
    ratios   = _encoder_ratios(df, encoders)

    # Drop encoders that expand data beyond what Raw stores uncompressed (ratio < 1.0).
    raw_ratio = ratios.get("Raw", 1.0)
    if args.filter_regressive_encoders:
        encoders  = [enc for enc in encoders if ratios[enc] >= raw_ratio]
    ratios    = {enc: ratios[enc] for enc in encoders}

    print(f"  Encoder order (ascending compression ratio, ≥ Raw filtered in):")
    for enc in encoders:
        print(f"    {enc:20s}  {ratios[enc]:.3f}×")

    for metric, metric_label, cmap in [
        ("elem_Meps",  "Decoded throughput (M elements / s)",               "viridis"),
        ("input_MBps", "Input bandwidth (MB / s read from encoded stream)", "plasma"),
    ]:
        stem = "throughput" if "Meps" in metric else "input_bw"
        for shared in (True, False):
            tag = "shared" if shared else "perplot"
            print(f"\nPlotting {stem} raw ({tag}) ...")
            _make_raw_figure(df, encoders, ratios, metric, metric_label, cmap,
                             f"heatmap_{stem}_raw_{tag}", shared_legend=shared)
            if args.skip_interp:
                print(f"Skipping {stem} interp ({tag}) [--skip-interp]")
            else:
                print(f"Plotting {stem} interp ({tag}) ...")
                _make_interp_figure(df, encoders, ratios, metric, metric_label, cmap,
                                    f"heatmap_{stem}_interp_{tag}", shared_legend=shared)

    print("\nPlotting throughput CDF ...")
    _make_throughput_cdf(df, encoders, ratios)

    strip_variants = [
        ("ratio",  "throughput_strip",       "compression ratio colours"),
        ("A_frac", "throughput_strip_by_A",  "start-position A colours"),
        ("B_frac", "throughput_strip_by_B",  "range-length B colours"),
        ("AB",     "throughput_strip_2D",    "2-D A+B colour"),
    ]
    for color_by, stem, desc in strip_variants:
        for vertical in (False, True):
            orient = "vertical" if vertical else "horizontal"
            out_stem = f"{stem}_vertical" if vertical else stem
            print(f"\nPlotting throughput strip plot ({desc}, {orient}) ...")
            _make_strip_plot(df, encoders, ratios, color_by=color_by,
                             out_stem=out_stem, vertical=vertical)

    for vertical in (False, True):
        orient   = "vertical" if vertical else "horizontal"
        out_stem = "throughput_box_vertical" if vertical else "throughput_box"
        print(f"\nPlotting throughput box plot (compression ratio colours, {orient}) ...")
        _make_box_plot(df, encoders, ratios, vertical=vertical, out_stem=out_stem)

    print(f"\nDone. {_FIGURES_SAVED} figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
