#!/usr/bin/env python3
"""
Plot gather-access (selective row-range) throughput heatmaps.

Consumes Benchmarks/results/bench_decode_gather.csv, the output of
Benchmarks/gather_heatmap_benchmark.cpp, which sweeps sparse reads over
(s0_frac, l, sigma, run_length) and drives them through Codec::decodeGatherInto().

Figures (PDF + PNG each) under plots/gather_heatmaps/<dataset>/:

  gather_{metric}_l_sigma_s0{i}_{shared,perplot}
      Throughput over (span l, selectivity sigma) at a fixed start fraction.
      The sigma = 1 row is the contiguous range-access baseline.

  gather_{metric}_s0_l_sigma{s}_{shared,perplot}
      Throughput over (start fraction, span l) at a fixed selectivity.

  gather_selectivity_curves
      Throughput against selectivity, one line per encoder, faceted by span.
      The most direct read of "what does sparsity cost this encoding".

  gather_skip_fraction
      Share of gather time spent skipping vs materializing, against selectivity,
      for the encoders that report the split (profiling-instrumented codecs only).

  gather_baseline_check
      The sigma = 1 slice against Benchmarks/results/heatmap_benchmark.csv, when
      that file is present. These should coincide; they are the same access.

Cells the driver marked skipped=1 (sequential codecs past --seq-max-k, where a
gather costs one full-payload decode per range) are hatched, not blank — the
absence of a number there is itself the finding.

Run from EncodingsPlayground/:
    python3 Benchmarks/plot_gather_heatmap.py [--input CSV] [--output DIR]
                                              [--metric COL] [--skip-interp]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:  # matplotlib >= 3.9 removed matplotlib.cm.get_cmap
    _get_cmap = matplotlib.colormaps.__getitem__
except AttributeError:
    from matplotlib.cm import get_cmap as _get_cmap

DEFAULT_CSV = Path("Benchmarks/results/bench_decode_gather.csv")
BASELINE_CSV = Path("Benchmarks/results/heatmap_benchmark.csv")
OUT_ROOT = Path("plots/gather_heatmaps")
NCOLS = 4

# Sequential single-hue ramps, truncated away from the near-white end so the
# lightest cells stay visible on paper. One hue per metric, never a rainbow:
# these encode magnitude, and a rainbow would imply category boundaries that
# do not exist in the data.
METRICS = {
    "sel_elem_Meps":  ("Gather throughput (M selected elements / s)", "YlOrBr", 0.35),
    "useful_MBps":    ("Useful bandwidth (MB / s of selected data)",  "Reds",   0.25),
    "span_elem_Meps": ("Span-normalised rate (M span elements / s)",  "Blues",  0.25),
    "input_MBps":     ("Input bandwidth (MB / s from encoded stream)", "Purples", 0.25),
}

_FIGURES_SAVED = 0
_OUT_DIR = OUT_ROOT


def _ramp(name: str, lo: float) -> mcolors.Colormap:
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_vis", _get_cmap(name)(np.linspace(lo, 1.0, 256)))


def _save(fig, stem: str) -> None:
    global _FIGURES_SAVED
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_OUT_DIR / f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(str(_OUT_DIR / f"{stem}.png"), bbox_inches="tight", dpi=150)
    print(f"  Saved {_OUT_DIR}/{stem}.{{pdf,png}}")
    plt.close(fig)
    _FIGURES_SAVED += 1


def _encoders_by_compression(df: pd.DataFrame) -> list[str]:
    """Encoder names in ascending median compression ratio, matching plot_heatmap.py."""
    return list(df.groupby("encoding")["compression_ratio"].median().sort_values().index)


def _ratios(df: pd.DataFrame, encoders: list[str]) -> dict[str, float]:
    return {e: float(df.loc[df["encoding"] == e, "compression_ratio"].median())
            for e in encoders}


def _norm(values: np.ndarray) -> mcolors.Normalize:
    """Log norm over the positive finite values, falling back to linear if degenerate."""
    pos = values[np.isfinite(values) & (values > 0)]
    if pos.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    lo, hi = float(pos.min()), float(pos.max())
    if lo == hi:
        return mcolors.Normalize(vmin=lo * 0.9, vmax=hi * 1.1)
    return mcolors.LogNorm(vmin=lo, vmax=hi)


def _global_norm(df: pd.DataFrame, metric: str) -> mcolors.Normalize:
    return _norm(df[metric].to_numpy(dtype=float))


# ── Heatmap grids ────────────────────────────────────────────────────────────

def _facet_grid(n: int):
    ncols = min(n, NCOLS)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.7, nrows * 3.6 + 0.5),
                             squeeze=False, constrained_layout=True)
    return fig, axes, nrows, ncols


def _hatch_skipped(ax, sub: pd.DataFrame, xcol: str, ycol: str) -> bool:
    """Mark not-viable cells so their absence reads as a result, not missing data."""
    sk = sub[sub["skipped"] == 1]
    if sk.empty:
        return False
    ax.scatter(sk[xcol], sk[ycol], marker="x", s=16,
               color="#3d3d3d", linewidths=0.9, zorder=4)
    return True


def _heatmap_figure(df: pd.DataFrame, encoders: list[str], ratios: dict[str, float],
                    metric: str, label: str, cmap_name: str, cmap_lo: float,
                    xcol: str, ycol: str, xlabel: str, ylabel: str,
                    title: str, stem: str, shared: bool,
                    logx: bool = False, logy: bool = False) -> None:
    cmap = _ramp(cmap_name, cmap_lo)
    gnorm = _global_norm(df, metric) if shared else None
    fig, axes, nrows, ncols = _facet_grid(len(encoders))
    any_skipped = False

    for ax, enc in zip(axes.flat, encoders):
        sub = df[df["encoding"] == enc]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(values=metric, index=ycol, columns=xcol, aggfunc="median")
        vals = pivot.to_numpy(dtype=float)
        norm = gnorm if shared else _norm(vals.ravel())
        im = ax.pcolormesh(pivot.columns.to_numpy(), pivot.index.to_numpy(), vals,
                           cmap=cmap, norm=norm, shading="nearest")
        any_skipped |= _hatch_skipped(ax, sub, xcol, ycol)

        ax.set_title(f"{enc}\n({ratios[enc]:.2f}×)", fontsize=8, pad=3)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if not shared:
            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label(label, fontsize=6)
            cbar.ax.tick_params(labelsize=5)

    for ax in list(axes.flat)[len(encoders):]:
        ax.set_visible(False)

    if shared:
        sm = ScalarMappable(cmap=cmap, norm=gnorm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), label=label,
                     shrink=0.8, aspect=25, pad=0.02)

    handles = []
    if any_skipped:
        handles.append(Line2D([], [], marker="x", ls="none", color="#888888",
                              markersize=5, label="not viable (full-payload decode per range)"))
    if handles:
        fig.legend(handles=handles, loc="lower center", fontsize=7,
                   frameon=False, ncol=1, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=11)
    _save(fig, stem)


# ── Line figures ─────────────────────────────────────────────────────────────

def _selectivity_curves(df: pd.DataFrame, encoders: list[str], ratios: dict[str, float],
                        metric: str, label: str) -> None:
    """Throughput against selectivity, faceted by span. One line per encoder.

    Colour carries compression ratio (a magnitude), so it is a single-hue ramp;
    identity comes from the legend, which is always present.
    """
    spans = sorted(df["l"].unique())
    if len(spans) > 8:
        idx = np.linspace(0, len(spans) - 1, 8).astype(int)
        spans = [spans[i] for i in idx]

    cmap = _ramp("YlOrBr", 0.35)
    rvals = [ratios[e] for e in encoders]
    rnorm = mcolors.LogNorm(vmin=max(min(rvals), 1e-3), vmax=max(rvals))
    # Compression ratios cluster tightly, so the ramp alone cannot separate a
    # dozen encoders. Dash pattern carries identity alongside hue, which also
    # keeps the figure readable in greyscale and under colour-vision deficiency.
    styles = ["-", "--", "-.", ":"]
    style_of = {e: styles[i % len(styles)] for i, e in enumerate(encoders)}

    fig, axes, _, _ = _facet_grid(len(spans))
    for ax, span in zip(axes.flat, spans):
        sub = df[(df["l"] == span) & (df["skipped"] == 0)]
        for enc in encoders:
            e = sub[sub["encoding"] == enc]
            if e.empty:
                continue
            # Group by the sweep cell, but place each point at the selectivity
            # actually delivered — nominal identifies the cell, achieved is the
            # honest x coordinate.
            g = e.groupby("sigma_nominal").agg(
                x=("sigma_achieved", "median"), y=(metric, "median")).sort_values("x")
            ax.plot(g["x"], g["y"], lw=1.6, marker="o", markersize=3,
                    ls=style_of[enc], color=cmap(rnorm(ratios[enc])), label=enc)
        ax.set_title(f"span $l$ = {int(span):,}", fontsize=8)
        ax.set_xlabel(r"achieved selectivity $\sigma$", fontsize=7)
        ax.set_ylabel(label, fontsize=6)
        ax.set_yscale("log")
        ax.set_ylim(bottom=None)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, lw=0.5)

    for ax in list(axes.flat)[len(spans):]:
        ax.set_visible(False)

    # Legend outside the axes grid, so it cannot collide with the bottom row's
    # x-axis labels however many spans are faceted.
    handles = [Line2D([], [], color=cmap(rnorm(ratios[e])), lw=1.8,
                      ls=style_of[e], label=f"{e}  ({ratios[e]:.2f}×)")
               for e in encoders]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=7, frameon=False, title="encoder (compression)",
               title_fontsize=8)
    fig.suptitle(f"{label} against selectivity  (colour = compression ratio)", fontsize=11)
    _save(fig, "gather_selectivity_curves")


def _skip_fraction(df: pd.DataFrame, encoders: list[str]) -> None:
    """Share of gather time spent skipping vs materializing, where the codec reports it."""
    have = [e for e in encoders
            if df.loc[df["encoding"] == e, "gather_skip_ns"].notna().any()
            and df.loc[df["encoding"] == e, "gather_materialize_ns"].notna().any()]
    if not have:
        print("  (no encoder reports the skip/materialize split; "
              "register a profiling codec variant to populate it)")
        return

    fig, axes, _, _ = _facet_grid(len(have))
    for ax, enc in zip(axes.flat, have):
        sub = df[(df["encoding"] == enc) & (df["skipped"] == 0)].copy()
        sub["total"] = sub["gather_skip_ns"].fillna(0) + sub["gather_materialize_ns"].fillna(0)
        sub = sub[sub["total"] > 0]
        if sub.empty:
            ax.set_visible(False)
            continue
        sub["skip_frac"] = sub["gather_skip_ns"].fillna(0) / sub["total"]
        g = sub.groupby("sigma_nominal").agg(
            x=("sigma_achieved", "median"), y=("skip_frac", "median")).sort_values("x")
        ax.fill_between(g["x"], 0, g["y"], color="#8c6d31", alpha=0.85,
                        label="skipping")
        ax.fill_between(g["x"], g["y"], 1, color="#c7c7c7", alpha=0.85,
                        label="materializing")
        ax.set_title(enc, fontsize=8)
        ax.set_xlabel(r"achieved selectivity $\sigma$", fontsize=7)
        ax.set_ylabel("share of gather time", fontsize=7)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)

    for ax in list(axes.flat)[len(have):]:
        ax.set_visible(False)

    fig.legend(handles=[Patch(color="#8c6d31", label="skipping"),
                        Patch(color="#c7c7c7", label="materializing")],
               loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Where gather time goes", fontsize=11)
    _save(fig, "gather_skip_fraction")


def _baseline_check(df: pd.DataFrame, encoders: list[str], baseline: Path) -> None:
    """Overlay the sigma = 1 slice on the contiguous range-access benchmark.

    Both measure the same access — one contiguous range — so the points should
    sit on the line. They do for codecs that override decodeRangeInto. Codecs
    that do not (Raw, AdaptiveBitPrefix, Zstd, OpenZL) sit roughly 2x below it,
    because the base-class decodeRangeInto is decodeRange plus a copy into the
    caller's buffer: that is the cost of the gather API's buffer contract, and
    it applies to every sigma, not just this slice.
    """
    if not baseline.exists():
        print(f"  (no {baseline}; skipping the baseline overlay)")
        return
    base = pd.read_csv(baseline)

    dense = df[(np.isclose(df["sigma_nominal"], 1.0)) & (df["skipped"] == 0)]
    if dense.empty:
        print("  (no sigma = 1 rows; skipping the baseline overlay)")
        return

    shared = [e for e in encoders if e in set(base["encoding"])]
    if not shared:
        print("  (no encoder names in common with the range-access CSV)")
        return

    # Compare against span expressed as a fraction of the stream, not absolute
    # elements: the two drivers may have been run at different N, and a bare
    # median over each grid would compare differently-shaped sweeps.
    n = float(dense["N"].iloc[0])
    fig, axes, _, _ = _facet_grid(len(shared))
    for ax, enc in zip(axes.flat, shared):
        g = (dense[dense["encoding"] == enc]
             .assign(span_frac=lambda d: d["l"] / n)
             .groupby("span_frac")["sel_elem_Meps"].median().sort_index())
        b = (base[base["encoding"] == enc]
             .groupby("B_frac")["elem_Meps"].median().sort_index())
        ax.plot(b.index, b.to_numpy(), lw=1.4, color="#c7c7c7", label="range access")
        ax.plot(g.index, g.to_numpy(), lw=0, marker="o", markersize=4,
                color="#8c6d31", label=r"gather, $\sigma=1$")
        ax.set_title(enc, fontsize=8)
        ax.set_xlabel("span as fraction of stream", fontsize=7)
        ax.set_ylabel("M elements / s", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, lw=0.5)

    for ax in list(axes.flat)[len(shared):]:
        ax.set_visible(False)

    fig.legend(handles=[Line2D([], [], color="#c7c7c7", lw=1.8, label="range access"),
                        Line2D([], [], color="#8c6d31", lw=0, marker="o", markersize=5,
                               label=r"gather, $\sigma=1$")],
               loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(r"$\sigma = 1$ gather vs contiguous range access"
                 "\n(the same access issued two ways; the points should sit on the line)",
                 fontsize=10)
    _save(fig, "gather_baseline_check")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _OUT_DIR
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_CSV)
    p.add_argument("--output", type=Path, default=OUT_ROOT)
    p.add_argument("--baseline", type=Path, default=BASELINE_CSV,
                   help="range-access CSV for the sigma=1 overlay")
    p.add_argument("--metric", default="sel_elem_Meps", choices=sorted(METRICS),
                   help="metric to render (default: gather throughput)")
    p.add_argument("--all-metrics", action="store_true",
                   help="render heatmaps for every metric, not just --metric "
                        "(roughly 4x the figures and 4x the runtime)")
    p.add_argument("--skip-interp", action="store_true",
                   help="accepted for symmetry with plot_heatmap.py; this script "
                        "does not interpolate, so it is a no-op")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: {args.input} not found. Run ./build/bin/gather_heatmap_benchmark first.")

    df = pd.read_csv(args.input)
    if df.empty:
        sys.exit(f"ERROR: {args.input} has no rows.")

    print(f"Loaded {len(df):,} rows from {args.input}")
    for col in ("skipped", "truncated"):
        if col not in df:
            df[col] = 0

    for dataset, ddf in df.groupby("dataset"):
        _OUT_DIR = args.output / str(dataset)
        n = int(ddf["N"].iloc[0])
        print(f"\n══ {dataset}  (N = {n:,}) ══")

        encoders = _encoders_by_compression(ddf)
        ratios = _ratios(ddf, encoders)
        for e in encoders:
            print(f"    {e:22s} {ratios[e]:8.3f}×")

        # Keep the not-viable rows in: their metric columns are empty, so they
        # pivot to NaN and _hatch_skipped marks them. Dropping them here would
        # leave a blank cell that reads as missing data rather than as a result.
        measured = ddf

        # (l × sigma) at each start fraction, and (s0 × l) at each selectivity.
        wanted = METRICS if args.all_metrics else {args.metric: METRICS[args.metric]}
        for metric, (label, cname, clo) in wanted.items():
            if metric not in measured:
                continue
            for i, s0f in enumerate(sorted(measured["s0_frac"].unique())):
                sl = measured[np.isclose(measured["s0_frac"], s0f)]
                if sl.empty:
                    continue
                for shared in (True, False):
                    tag = "shared" if shared else "perplot"
                    _heatmap_figure(
                        sl, encoders, ratios, metric, label, cname, clo,
                        # Index the grid by the nominal sigma: it identifies the
                        # sweep cell.  sigma_achieved differs from it by a
                        # fraction of a percent per cell (integer flooring), which
                        # would split every row into many near-duplicates and
                        # leave the pivot almost entirely NaN.
                        xcol="l", ycol="sigma_nominal",
                        xlabel=r"span $l$ (elements)", ylabel=r"selectivity $\sigma$",
                        title=f"{label}\nstart fraction $s_0$ = {s0f:.2f}",
                        stem=f"gather_{metric}_l_sigma_s0{i}_{tag}",
                        shared=shared, logx=True)

            for j, sig in enumerate(sorted(measured["sigma_nominal"].unique())):
                sl = measured[np.isclose(measured["sigma_nominal"], sig)]
                if sl.empty or sl["s0_frac"].nunique() < 2:
                    continue
                for shared in (True, False):
                    tag = "shared" if shared else "perplot"
                    _heatmap_figure(
                        sl, encoders, ratios, metric, label, cname, clo,
                        xcol="s0_frac", ycol="l",
                        xlabel=r"start fraction $s_0/(N-l)$", ylabel=r"span $l$ (elements)",
                        title=f"{label}\nselectivity $\\sigma$ = {sig:.2f}",
                        stem=f"gather_{metric}_s0_l_sigma{j}_{tag}",
                        shared=shared, logy=True)

        label = METRICS[args.metric][0]
        print("\n  selectivity curves ...")
        _selectivity_curves(ddf, encoders, ratios, args.metric, label)
        print("  skip/materialize split ...")
        _skip_fraction(ddf, encoders)
        print("  sigma = 1 baseline overlay ...")
        _baseline_check(ddf, encoders, args.baseline)

    print(f"\nDone. {_FIGURES_SAVED} figures written under {args.output}/")


if __name__ == "__main__":
    main()
