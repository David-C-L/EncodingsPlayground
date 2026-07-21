#!/usr/bin/env python3
"""
Plot the sweeps produced by cascade_cardinality_experiment: cardinality/bit-width
stats, and separately BlockFrequencyPartitionEncoder's compression rate on the
deepest cascade level's residuals.

The cardinality CSV has two `sweep` values:
  plain_for_depth1 - a single FOR level (no residual cascading), frame size
                      swept across a wide range. One row per (dataset, frame
                      size, role).
  cascading_for     - the residual stream cascaded to varying depth, starting
                       from varying outermost frame sizes. One row per
                       (dataset, startFrameSize, depth, level, role).

The BlockFPE CSV has the same (sweep, dataset, startFrameSize, depth) grid but
one row per config (not per level), reporting the compression ratio
BlockFrequencyPartitionEncoder achieves on the deepest cascade level's
residuals alone.

Usage examples
--------------
# Default: read cascade_cardinality_sweep.csv + cascade_blockfpe_compression.csv,
# write PNGs to ./plots
python plot_cascade_cardinality.py

# Explicit input/output
python plot_cascade_cardinality.py results/sweep.csv --blockfpe-csv results/blockfpe.csv --output-dir plots/

# Show interactively instead of saving
python plot_cascade_cardinality.py sweep.csv --show
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["parentResidualLevel"] = df["parentResidualLevel"].astype("Int64")
    return df


def _dataset_names(df: pd.DataFrame) -> list[str]:
    return sorted(df["dataset"].unique())


def _annotate_min(ax, x: float, y: float, label: str, y_all: pd.Series, log_scale: bool = False) -> None:
    """Place a small boxed label near (x, y), flipping above/below depending on
    whether (x, y) sits in the upper or lower part of the plotted data range --
    avoids the label colliding with the panel title when the point is near the
    top, or with crossing lines when the point is near the middle."""
    if log_scale:
        positive = y_all[y_all > 0]
        lo, hi = math.log(positive.min()), math.log(positive.max())
        yv = math.log(y) if y > 0 else lo
    else:
        lo, hi = y_all.min(), y_all.max()
        yv = y
    frac = (yv - lo) / (hi - lo) if hi > lo else 0.5

    y_offset = -18 if frac > 0.6 else 12
    va = "top" if y_offset < 0 else "bottom"
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(15, y_offset), fontsize=8, va=va,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))


def plot_plain_for_sweep(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Cardinality vs. frame size, depth=1 (no cascading), one row per dataset."""
    sub = df[df["sweep"] == "plain_for_depth1"]
    if sub.empty:
        return

    datasets = _dataset_names(sub)
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.2 * len(datasets)), squeeze=False)

    for row, dataset in enumerate(datasets):
        ds = sub[sub["dataset"] == dataset]
        for col, role in enumerate(["Residual", "Reference"]):
            ax = axes[row][col]
            # x-axis is always the *residual* frame size being swept (startFrameSize),
            # even for the Reference facet, whose own cascade frame is held fixed —
            # what varies there is the length of the array it's cascading (numFrames
            # of the residual level above it), not its own frame size.
            part = ds[ds["role"] == role].sort_values("startFrameSize")
            if part.empty:
                ax.set_visible(False)
                continue
            ax.plot(part["startFrameSize"], part["intraFrameDistinctMean"], marker="o",
                     label="intra-frame distinct (mean)")
            ax.fill_between(part["startFrameSize"], part["intraFrameDistinctMin"], part["intraFrameDistinctMax"],
                             alpha=0.15, label="intra-frame distinct (min-max)")
            ax.plot(part["startFrameSize"], part["globalHllEstimate"], marker="x", linestyle="--",
                     color="gray", label="global cardinality (HLL)")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("residual frame size (swept)")
            ax.set_ylabel("distinct values")
            ax.set_title(f"{dataset} — {role}")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Plain FOR (depth=1): cardinality vs. frame size", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(fig, output_dir / "plain_for_depth1_cardinality.png", show)


def plot_plain_for_bitwidth(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Residual bit width vs. frame size, depth=1 — shows the classic FOR tradeoff."""
    sub = df[(df["sweep"] == "plain_for_depth1") & (df["role"] == "Residual")]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset in _dataset_names(sub):
        part = sub[sub["dataset"] == dataset].sort_values("frameSize")
        ax.plot(part["frameSize"], part["avgBitWidth"], marker="o", label=dataset)

    ax.set_xscale("log", base=2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("frame size")
    ax.set_ylabel("avg residual bit width")
    ax.set_title("Plain FOR (depth=1): residual bit width vs. frame size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, output_dir / "plain_for_depth1_bitwidth.png", show)


def plot_cascading_sweep(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Cardinality vs. cascade level, one row per dataset, one column per role,
    one line per starting frame size (depth = the deepest sweep run, so every
    line shows the fullest cascade for that starting frame size)."""
    sub = df[df["sweep"] == "cascading_for"]
    if sub.empty:
        return

    max_depth = sub["depth"].max()
    sub = sub[sub["depth"] == max_depth]

    datasets = _dataset_names(sub)
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.2 * len(datasets)), squeeze=False)

    for row, dataset in enumerate(datasets):
        ds = sub[sub["dataset"] == dataset]
        for col, role in enumerate(["Residual", "Reference"]):
            ax = axes[row][col]
            part = ds[ds["role"] == role]
            if part.empty:
                ax.set_visible(False)
                continue
            # Residual rows are indexed by their own cascade level. Reference rows
            # all sit at levelIndex=0 (the reference schedule here is depth-1) —
            # the interesting axis for them is *which residual level* produced the
            # reference array being measured, i.e. parentResidualLevel.
            x_col = "levelIndex" if role == "Residual" else "parentResidualLevel"
            for start in sorted(part["startFrameSize"].unique()):
                line = part[part["startFrameSize"] == start].sort_values(x_col)
                ax.plot(line[x_col], line["intraFrameDistinctMean"], marker="o",
                        label=f"start={start}")
            ax.set_yscale("log")
            ax.set_xlabel("residual cascade level" if role == "Reference" else "cascade level")
            ax.set_ylabel("intra-frame distinct (mean)")
            ax.set_title(f"{dataset} — {role}")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, title="outermost frame")

    fig.suptitle(f"Cascading FOR (depth={max_depth}): cardinality vs. cascade level", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(fig, output_dir / "cascading_for_by_level.png", show)


def plot_cascading_depth_comparison(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Final-level (deepest residual level reached) global cardinality vs. depth,
    one line per starting frame size — shows how much deeper cascading helps."""
    sub = df[(df["sweep"] == "cascading_for") & (df["role"] == "Residual")]
    if sub.empty:
        return

    final_level = sub.loc[sub.groupby(["dataset", "startFrameSize", "depth"])["levelIndex"].idxmax()]

    datasets = _dataset_names(final_level)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        ds = final_level[final_level["dataset"] == dataset]
        for start in sorted(ds["startFrameSize"].unique()):
            line = ds[ds["startFrameSize"] == start].sort_values("depth")
            ax.plot(line["depth"], line["globalHllEstimate"], marker="o", label=f"start={start}")
        ax.set_yscale("log")
        ax.set_xlabel("cascade depth")
        ax.set_ylabel("global cardinality (HLL), deepest level")
        ax.set_title(dataset)
        ax.legend(fontsize=8)

    fig.suptitle("Cascading FOR: does more depth reduce cardinality further?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "cascading_for_depth_comparison.png", show)


def plot_blockfpe_plain_for(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """BlockFPE compression ratio vs. frame size, depth=1 (no cascading).
    Fixed at the deepest level's MIN policy (see plot_deepest_policy_effect for
    the MIN/FIRST/MID comparison) so this isn't tripled/overlapping lines."""
    sub = df[(df["sweep"] == "plain_for_depth1") & (df["deepestPolicy"] == "MIN")]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset in _dataset_names(sub):
        part = sub[sub["dataset"] == dataset].sort_values("deepestFrameSize")
        ax.plot(part["deepestFrameSize"], part["compressionRatio"], marker="o", label=dataset)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="break-even (1.0)")
    ax.set_xscale("log", base=2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("frame size")
    ax.set_ylabel("BlockFPE compressed / uncompressed")
    ax.set_title("Plain FOR (depth=1): BlockFPE compression ratio on residuals vs. frame size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, output_dir / "blockfpe_plain_for_depth1.png", show)


def plot_blockfpe_cascading(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """BlockFPE compression ratio vs. cascade depth, one panel per dataset,
    one line per starting (outermost) frame size. Fixed at MIN policy (see
    plot_deepest_policy_effect for the MIN/FIRST/MID comparison)."""
    sub = df[(df["sweep"] == "cascading_for") & (df["deepestPolicy"] == "MIN")]
    if sub.empty:
        return

    datasets = _dataset_names(sub)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.2), squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        ds = sub[sub["dataset"] == dataset]
        for start in sorted(ds["startFrameSize"].unique()):
            line = ds[ds["startFrameSize"] == start].sort_values("depth")
            ax.plot(line["depth"], line["compressionRatio"], marker="o", label=f"start={start}")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("cascade depth")
        ax.set_ylabel("BlockFPE compressed / uncompressed")
        ax.set_title(dataset)
        ax.legend(fontsize=8, title="outermost frame")

    fig.suptitle("Cascading FOR: BlockFPE compression ratio on deepest-level residuals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _finish(fig, output_dir / "blockfpe_cascading_for.png", show)


def plot_deepest_policy_effect(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Does the deepest level's reference policy (MIN/FIRST/MID) change BlockFPE
    compression? Intermediate-level policy is provably inert for residual values
    (telescoping identity — see CascadingFOREncoder.hpp), but the deepest level's
    policy can affect cross-window value alignment, which isn't provable
    analytically. One panel per dataset, cascading_for sweep at the smallest
    starting frame size (so depth=4 reaches the smallest deepest frame size,
    where any policy divergence is most visible), one line per policy."""
    sub = df[df["sweep"] == "cascading_for"]
    if sub.empty:
        return
    start = sub["startFrameSize"].min()
    sub = sub[sub["startFrameSize"] == start]

    datasets = _dataset_names(sub)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.2), squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        ds = sub[sub["dataset"] == dataset]
        for policy in ["MIN", "FIRST", "MID"]:
            line = ds[ds["deepestPolicy"] == policy].sort_values("depth")
            ax.plot(line["depth"], line["compressionRatio"], marker="o", label=policy)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("cascade depth")
        ax.set_ylabel("BlockFPE compressed / uncompressed")
        ax.set_title(dataset)
        ax.legend(fontsize=8, title="deepest-level policy")

    fig.suptitle(f"Does the deepest level's reference policy affect BlockFPE compression? (start={start})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _finish(fig, output_dir / "blockfpe_deepest_policy_effect.png", show)


def plot_cardinality_vs_compression(card_df: pd.DataFrame, blockfpe_df: pd.DataFrame,
                                     output_dir: Path, show: bool) -> None:
    """Scatter: corrected deepest-level global cardinality (HLL) vs. BlockFPE
    compression ratio, one point per (sweep, dataset, startFrameSize, depth)
    config, colored by dataset. This is the direct validation that the
    CascadeCardinalityAnalyzer fix (measuring post-subtraction residuals
    instead of pre-subtraction data) produces a metric that actually predicts
    downstream compressibility."""
    res = card_df[card_df["role"] == "Residual"]
    if res.empty or blockfpe_df.empty:
        return
    deepest = res.loc[res.groupby(["sweep", "dataset", "startFrameSize", "depth"])["levelIndex"].idxmax()]
    bfpe_min = blockfpe_df[blockfpe_df["deepestPolicy"] == "MIN"]
    merged = deepest.merge(bfpe_min, on=["sweep", "dataset", "startFrameSize", "depth"])
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset in _dataset_names(merged):
        part = merged[merged["dataset"] == dataset]
        ax.scatter(part["globalHllEstimate"], part["compressionRatio"], label=dataset, alpha=0.7)
        r = part["globalHllEstimate"].corr(part["compressionRatio"])
        print(f"  corr(globalHllEstimate, compressionRatio) for {dataset}: r={r:.4f}")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("corrected global cardinality at deepest residual level (HLL)")
    ax.set_ylabel("BlockFPE compressed / uncompressed")
    ax.set_title("Corrected global cardinality vs. BlockFPE compression ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, output_dir / "cardinality_vs_compression_validation.png", show)


def plot_blockforfpe_comparison(blockfpe_df: pd.DataFrame, blockforfpe_df: pd.DataFrame,
                                 refonly_df: pd.DataFrame | None,
                                 output_dir: Path, show: bool) -> None:
    """Bar chart: BlockFORFPEEncoder's own auto-selected block size and
    compression ratio (integrated single-block-size FOR+FPE, run directly on
    raw data) vs. the best compression ratio the two-stage CascadingFOREncoder
    + BlockFrequencyPartitionEncoder pipeline achieved anywhere in the sweep
    (both residual and reference streams cascaded), vs. the best compression
    ratio when ONLY the reference stream is cascaded (residual stream fixed at
    a single depth-1 FOR pass — see plot_reference_only_heatmap). All three
    numbers are for the complete, decodable encoding (FOR references + BlockFPE
    residuals) — see cascade_cardinality_experiment.cpp's header comment on why
    the reference stream must always be counted."""
    if blockforfpe_df.empty or blockfpe_df.empty:
        return
    best_two_stage = blockfpe_df.loc[blockfpe_df.groupby("dataset")["compressionRatio"].idxmin()]
    has_refonly = refonly_df is not None and not refonly_df.empty
    if has_refonly:
        best_refonly = refonly_df.loc[refonly_df.groupby("dataset")["compressionRatio"].idxmin()]

    datasets = sorted(blockforfpe_df["dataset"].unique())
    integrated = [blockforfpe_df.set_index("dataset").loc[d, "compressionRatio"] for d in datasets]
    two_stage = [best_two_stage.set_index("dataset").loc[d, "compressionRatio"] for d in datasets]
    # "block="/"frame=" annotations aren't always meaningful (the residual stream
    # and each cascade level's own reference array can each pick a different
    # BlockFPE block size) -- annotate with the winning FOR frame size instead.
    two_stage_frame = [int(best_two_stage.set_index("dataset").loc[d, "deepestFrameSize"]) for d in datasets]
    integrated_block = [int(blockforfpe_df.set_index("dataset").loc[d, "blockSize"]) for d in datasets]

    x = range(len(datasets))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_bars = 3 if has_refonly else 2
    width = 0.8 / n_bars

    def offset(i):
        return (i - (n_bars - 1) / 2) * width

    ax.bar([i + offset(0) for i in x], integrated, width, label="BlockFORFPEEncoder (integrated)")
    ax.bar([i + offset(1) for i in x], two_stage, width,
           label="Both streams cascaded (best over sweep)")
    for i, (ib, tf) in enumerate(zip(integrated_block, two_stage_frame)):
        ax.annotate(f"block={ib}", (i + offset(0), integrated[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7)
        ax.annotate(f"frame={tf}", (i + offset(1), two_stage[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7)

    if has_refonly:
        refonly = [best_refonly.set_index("dataset").loc[d, "compressionRatio"] for d in datasets]
        refonly_frame = [int(best_refonly.set_index("dataset").loc[d, "residualFrameSize"]) for d in datasets]
        ax.bar([i + offset(2) for i in x], refonly, width,
               label="Reference-only cascaded (residual fixed at depth=1)")
        for i, rf in enumerate(refonly_frame):
            ax.annotate(f"frame={rf}", (i + offset(2), refonly[i]), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("compressed / uncompressed")
    ax.set_title("Integrated vs. both-streams-cascaded vs. reference-only-cascaded: best compression ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, output_dir / "blockforfpe_vs_two_stage.png", show)


def plot_reference_only_heatmap(refonly_df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Heatmap: compression ratio vs. (residual frame size, reference-cascade
    depth), best over reference-cascade starting frame size, one panel per
    dataset. Tests whether cascading ONLY the reference stream (keeping the
    residual side at a single depth-1 FOR pass) captures the same benefit as
    cascading both streams -- see cascade_cardinality_experiment.cpp's
    "Reference-only cascade experiment" section."""
    if refonly_df.empty:
        return

    datasets = _dataset_names(refonly_df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)

    residual_sizes = sorted(refonly_df["residualFrameSize"].unique())
    ref_depths = sorted(refonly_df["refDepth"].unique())  # includes 0 = no ref cascading (baseline)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        ds = refonly_df[refonly_df["dataset"] == dataset]
        best = ds.loc[ds.groupby(["residualFrameSize", "refDepth"])["compressionRatio"].idxmin()]
        grid = best.pivot(index="refDepth", columns="residualFrameSize", values="compressionRatio")
        grid = grid.reindex(index=ref_depths, columns=residual_sizes)

        im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap="viridis_r")
        ax.set_xticks(range(len(residual_sizes)))
        ax.set_xticklabels(residual_sizes, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(ref_depths)))
        ax.set_yticklabels(["none" if d == 0 else d for d in ref_depths], fontsize=8)
        ax.set_xlabel("residual frame size (depth=1)")
        ax.set_ylabel("reference cascade depth")
        ax.set_title(dataset, fontsize=10)
        fig.colorbar(im, ax=ax, label="compressed/uncompressed", fraction=0.046, pad=0.04)

    fig.suptitle("Reference-only cascading: compression ratio vs. residual frame size × reference depth",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "reference_only_cascade_heatmap.png", show)


def plot_delta_for_sorting_comparison(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Bar chart: Delta vs. FOR prepass x BlockFPE vs. RawBitPacked leaf x
    unsorted vs. sorted, one panel per dataset. The sorted bar is stacked into
    its payload contribution (compressedBytes - permutationBytes, i.e. what
    sorting achieves on the data itself) and its permutation-overhead
    contribution (permutationBytes) -- both expressed as a fraction of
    uncompressedBytes, so the two segments sum exactly to the sorted
    compressionRatio and "compression without the permutation tax" is directly
    readable without cross-referencing a separate metric -- see
    cascade_cardinality_experiment.cpp's "Delta vs. FOR prepass, with/without
    sorting" section."""
    if df.empty:
        return

    datasets = _dataset_names(df)
    configs = [("FOR", "BlockFPE"), ("FOR", "RawBitPacked"), ("Delta", "BlockFPE"), ("Delta", "RawBitPacked")]
    labels = [f"{p}/{l}" for p, l in configs]

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.8), squeeze=False)

    x = range(len(configs))
    width = 0.35

    for col, dataset in enumerate(datasets):
        ds = df[df["dataset"] == dataset]
        ax = axes[0][col]

        unsorted_vals = []
        sorted_payload = []
        sorted_perm = []
        for prepass, leaf in configs:
            row_unsorted = ds[(ds["prepass"] == prepass) & (ds["leaf"] == leaf) & (ds["sorted"] == 0)]
            row_sorted = ds[(ds["prepass"] == prepass) & (ds["leaf"] == leaf) & (ds["sorted"] == 1)]
            unsorted_vals.append(row_unsorted["compressionRatio"].iloc[0] if not row_unsorted.empty else 0)
            if not row_sorted.empty:
                r = row_sorted.iloc[0]
                payload_ratio = (r["compressedBytes"] - r["permutationBytes"]) / r["uncompressedBytes"]
                perm_ratio = r["permutationBytes"] / r["uncompressedBytes"]
            else:
                payload_ratio, perm_ratio = 0, 0
            sorted_payload.append(payload_ratio)
            sorted_perm.append(perm_ratio)

        ax.bar([i - width / 2 for i in x], unsorted_vals, width, label="unsorted")
        ax.bar([i + width / 2 for i in x], sorted_payload, width, label="sorted (payload)",
               hatch="//", alpha=0.8, color="tab:orange")
        ax.bar([i + width / 2 for i in x], sorted_perm, width, bottom=sorted_payload,
               label="sorted (permutation overhead)", hatch="\\\\", alpha=0.8, color="tab:red")

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.set_ylim(bottom=0)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("compressed / uncompressed")
        ax.set_title(dataset, fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Delta vs. FOR prepass x leaf x sorting: compression ratio, "
                 "with permutation overhead broken out",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "delta_for_sorting_comparison.png", show)


def plot_for_cardinality_decomposition(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plain FOR (no cascading): exact global cardinality of the residual
    stream, the reference stream, and their sum, vs. frame size -- one panel
    per dataset. Deliberately minimal (no bit-width, no HLL, no intra-frame
    stats): residual cardinality shrinks and reference cardinality grows as
    frame size shrinks, so the total traces a U-shape whose minimum is
    annotated directly -- see cascade_cardinality_experiment.cpp's
    "Plain-FOR ... global cardinality decomposition" section."""
    if df.empty:
        return

    datasets = _dataset_names(df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.5), squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        part = df[df["dataset"] == dataset].sort_values("frameSize")

        ax.plot(part["frameSize"], part["residualExactDistinct"], marker="o", label="delta")
        ax.plot(part["frameSize"], part["referenceExactDistinct"], marker="o", label="reference")
        ax.plot(part["frameSize"], part["totalExactDistinct"], marker="o", linewidth=2.5,
                linestyle="--", color="black", label="total")

        best = part.loc[part["totalExactDistinct"].idxmin()]
        ax.axvline(best["frameSize"], color="gray", linestyle=":", linewidth=1)
        all_y = pd.concat([part["residualExactDistinct"], part["referenceExactDistinct"], part["totalExactDistinct"]])
        _annotate_min(ax, best["frameSize"], best["totalExactDistinct"],
                      f"min at frame={int(best['frameSize'])}", all_y)

        ax.set_xscale("log", base=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("frame size")
        ax.set_ylabel("exact distinct count")
        ax.set_title(dataset, fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Plain FOR (no cascading): global cardinality decomposition vs. frame size", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "for_cardinality_decomposition.png", show)


def plot_for_cardinality_decomposition_loglog(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Same data as plot_for_cardinality_decomposition, but with both axes
    log2 -- makes the residual/reference power-law-like trends and the
    total's U-shape comparable in slope regardless of dataset (useful when,
    as for real data, the linear-y view compresses the reference line and
    the total's variation into a sliver near the top of the axis)."""
    if df.empty:
        return

    datasets = _dataset_names(df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.5), squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        part = df[df["dataset"] == dataset].sort_values("frameSize")

        ax.plot(part["frameSize"], part["residualExactDistinct"], marker="o", label="delta")
        ax.plot(part["frameSize"], part["referenceExactDistinct"], marker="o", label="reference")
        ax.plot(part["frameSize"], part["totalExactDistinct"], marker="o", linewidth=2.5,
                linestyle="--", color="black", label="total")

        best = part.loc[part["totalExactDistinct"].idxmin()]
        ax.axvline(best["frameSize"], color="gray", linestyle=":", linewidth=1)
        all_y = pd.concat([part["residualExactDistinct"], part["referenceExactDistinct"], part["totalExactDistinct"]])
        _annotate_min(ax, best["frameSize"], best["totalExactDistinct"],
                      f"min at frame={int(best['frameSize'])}", all_y, log_scale=True)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("frame size")
        ax.set_ylabel("exact distinct count")
        ax.set_title(dataset, fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Plain FOR (no cascading): global cardinality decomposition vs. frame size (log-log)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "for_cardinality_decomposition_loglog.png", show)


def plot_for_cardinality_decomposition_with_delta(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Same as plot_for_cardinality_decomposition, plus a flat reference line
    for the exact cardinality achieved by standard first-order delta encoding
    (no frame-size parameter, so it's constant across the sweep) -- lets FOR's
    best-case total be compared directly against delta encoding's single
    number."""
    if df.empty:
        return

    datasets = _dataset_names(df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.5), squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        part = df[df["dataset"] == dataset].sort_values("frameSize")

        ax.plot(part["frameSize"], part["residualExactDistinct"], marker="o", label="delta (FOR residual)")
        ax.plot(part["frameSize"], part["referenceExactDistinct"], marker="o", label="reference")
        ax.plot(part["frameSize"], part["totalExactDistinct"], marker="o", linewidth=2.5,
                linestyle="--", color="black", label="total")
        ax.axhline(part["deltaExactDistinct"].iloc[0], color="tab:green", linestyle="-.", linewidth=2,
                   label="delta encoding")

        best = part.loc[part["totalExactDistinct"].idxmin()]
        ax.axvline(best["frameSize"], color="gray", linestyle=":", linewidth=1)
        all_y = pd.concat([part["residualExactDistinct"], part["referenceExactDistinct"],
                            part["totalExactDistinct"], part["deltaExactDistinct"]])
        _annotate_min(ax, best["frameSize"], best["totalExactDistinct"],
                      f"min at frame={int(best['frameSize'])}", all_y)

        ax.set_xscale("log", base=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("frame size")
        ax.set_ylabel("exact distinct count")
        ax.set_title(dataset, fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Plain FOR (no cascading) vs. delta encoding: global cardinality vs. frame size", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "for_cardinality_decomposition_with_delta.png", show)


def plot_for_cardinality_decomposition_loglog_with_delta(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Same as plot_for_cardinality_decomposition_loglog, plus a flat
    reference line for standard first-order delta encoding's exact
    cardinality (see plot_for_cardinality_decomposition_with_delta)."""
    if df.empty:
        return

    datasets = _dataset_names(df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.5), squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        part = df[df["dataset"] == dataset].sort_values("frameSize")

        ax.plot(part["frameSize"], part["residualExactDistinct"], marker="o", label="delta (FOR residual)")
        ax.plot(part["frameSize"], part["referenceExactDistinct"], marker="o", label="reference")
        ax.plot(part["frameSize"], part["totalExactDistinct"], marker="o", linewidth=2.5,
                linestyle="--", color="black", label="total")
        ax.axhline(part["deltaExactDistinct"].iloc[0], color="tab:green", linestyle="-.", linewidth=2,
                   label="delta encoding")

        best = part.loc[part["totalExactDistinct"].idxmin()]
        ax.axvline(best["frameSize"], color="gray", linestyle=":", linewidth=1)
        all_y = pd.concat([part["residualExactDistinct"], part["referenceExactDistinct"],
                            part["totalExactDistinct"], part["deltaExactDistinct"]])
        _annotate_min(ax, best["frameSize"], best["totalExactDistinct"],
                      f"min at frame={int(best['frameSize'])}", all_y, log_scale=True)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("frame size")
        ax.set_ylabel("exact distinct count")
        ax.set_title(dataset, fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Plain FOR (no cascading) vs. delta encoding: global cardinality vs. frame size (log-log)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "for_cardinality_decomposition_loglog_with_delta.png", show)


def _bitrange_grid(df: pd.DataFrame, col: str, start_bits: list[int], widths: list[int]) -> pd.DataFrame:
    return df.pivot(index="startBit", columns="width", values=col).reindex(index=start_bits, columns=widths)


def _bitrange_axes_style(axes, widths: list[int], tick_positions: list[int]) -> None:
    for ax in axes:
        ax.set_facecolor("black")  # distinguishes the invalid (start+width>64) triangle from real data
        ax.set_xlabel("bit-range width")
        ax.set_ylabel("start bit (LSB=0)")
        ax.set_xticks(list(tick_positions))
        ax.set_xticklabels([widths[p] for p in tick_positions])
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_positions)


def plot_bitrange_cardinality_heatmap(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Two heatmaps over every contiguous bit-range (start bit x width)
    extractable from a 64-bit value: minimum total exact cardinality achieved
    over the plain-FOR frame-size sweep (left), and the frame size that
    achieves it (right). Cells where start+width > 64 are invalid; shown as
    black (see _bitrange_axes_style) rather than left to blend into the
    figure background -- see cascade_cardinality_experiment.cpp's
    "Bit-range cardinality heatmap" section."""
    if df.empty:
        return

    dataset = df["dataset"].iloc[0]
    start_bits = list(range(64))
    widths = list(range(1, 65))
    tick_positions = list(range(0, 64, 8))

    total_grid = _bitrange_grid(df, "minTotalExactDistinct", start_bits, widths)
    frame_grid = _bitrange_grid(df, "bestFrameSize", start_bits, widths)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    im0 = axes[0].imshow(total_grid.values, aspect="auto", origin="lower", cmap="viridis_r", norm=LogNorm())
    axes[0].set_title("Minimum total exact cardinality\n(best over frame-size sweep)", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(frame_grid.values, aspect="auto", origin="lower", cmap="plasma", norm=LogNorm())
    axes[1].set_title("Frame size achieving the minimum", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    _bitrange_axes_style(axes, widths, tick_positions)

    fig.suptitle(f"{dataset}: bit-range cardinality landscape (start bit x width)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "bitrange_cardinality_heatmap.png", show)


def plot_bitrange_cardinality_change(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Two heatmaps comparing FOR and standard first-order delta encoding's
    effect on cardinality, relative to the raw (untransformed) bit-range:
    log2(transformed/raw) on a diverging scale centered at 0 -- negative
    (blue) means the transform reduces cardinality, positive (red) means it
    makes cardinality worse (e.g. FOR's reference-stream overhead can
    outweigh its residual reduction for already-compact narrow bit-ranges).
    The invalid (start+width>64) triangle is shown as black rather than left
    white, so it isn't confused with genuine "no change" cells (which are
    also white, at the center of the diverging colormap) -- see
    plot_bitrange_cardinality_change_linear for a non-log variant."""
    if df.empty:
        return

    dataset = df["dataset"].iloc[0]
    start_bits = list(range(64))
    widths = list(range(1, 65))
    tick_positions = list(range(0, 64, 8))

    total_grid = _bitrange_grid(df, "minTotalExactDistinct", start_bits, widths)
    raw_grid = _bitrange_grid(df, "rawExactDistinct", start_bits, widths)
    delta_grid = _bitrange_grid(df, "deltaExactDistinct", start_bits, widths)

    for_change = np.log2(total_grid / raw_grid)
    delta_change = np.log2(delta_grid / raw_grid)
    change_abs_max = max(for_change.abs().max().max(), delta_change.abs().max().max())
    change_norm = TwoSlopeNorm(vcenter=0, vmin=-change_abs_max, vmax=change_abs_max)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    im0 = axes[0].imshow(for_change.values, aspect="auto", origin="lower", cmap="RdBu_r", norm=change_norm)
    axes[0].set_title("FOR: log2(minTotal / raw)\n(negative = FOR helps)", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(delta_change.values, aspect="auto", origin="lower", cmap="RdBu_r", norm=change_norm)
    axes[1].set_title("Delta encoding: log2(delta / raw)\n(negative = delta helps)", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    _bitrange_axes_style(axes, widths, tick_positions)

    fig.suptitle(f"{dataset}: cardinality change vs. raw bit-range (log2 ratio)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "bitrange_cardinality_change_log.png", show)


def plot_bitrange_cardinality_change_linear(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Same comparison as plot_bitrange_cardinality_change, but plotting the
    raw ratio (transformed/raw) directly instead of its log2 -- centered at
    1.0 (no change) instead of 0. Trades off resolution on the "helps a lot"
    side (which can span orders of magnitude) for a more literal reading of
    the ratio value itself."""
    if df.empty:
        return

    dataset = df["dataset"].iloc[0]
    start_bits = list(range(64))
    widths = list(range(1, 65))
    tick_positions = list(range(0, 64, 8))

    total_grid = _bitrange_grid(df, "minTotalExactDistinct", start_bits, widths)
    raw_grid = _bitrange_grid(df, "rawExactDistinct", start_bits, widths)
    delta_grid = _bitrange_grid(df, "deltaExactDistinct", start_bits, widths)

    for_ratio = total_grid / raw_grid
    delta_ratio = delta_grid / raw_grid
    ratio_max = max(for_ratio.max().max(), delta_ratio.max().max())
    ratio_min = min(for_ratio.min().min(), delta_ratio.min().min())
    ratio_norm = TwoSlopeNorm(vcenter=1.0, vmin=ratio_min, vmax=ratio_max)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    im0 = axes[0].imshow(for_ratio.values, aspect="auto", origin="lower", cmap="RdBu_r", norm=ratio_norm)
    axes[0].set_title("FOR: minTotal / raw\n(< 1 = FOR helps)", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(delta_ratio.values, aspect="auto", origin="lower", cmap="RdBu_r", norm=ratio_norm)
    axes[1].set_title("Delta encoding: delta / raw\n(< 1 = delta helps)", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    _bitrange_axes_style(axes, widths, tick_positions)

    fig.suptitle(f"{dataset}: cardinality change vs. raw bit-range (linear ratio)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, output_dir / "bitrange_cardinality_change_linear.png", show)


def _finish(fig: plt.Figure, path: Path, show: bool) -> None:
    if show:
        plt.show()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default="cascade_cardinality_sweep.csv",
                        help="cardinality CSV produced by cascade_cardinality_experiment (default: %(default)s)")
    parser.add_argument("--blockfpe-csv", default="cascade_blockfpe_compression.csv",
                        help="BlockFPE compression CSV produced by cascade_cardinality_experiment "
                             "(default: %(default)s; skipped with a note if not found)")
    parser.add_argument("--blockforfpe-csv", default="cascade_blockforfpe_comparison.csv",
                        help="BlockFORFPEEncoder comparison CSV produced by cascade_cardinality_experiment "
                             "(default: %(default)s; skipped with a note if not found)")
    parser.add_argument("--refonly-csv", default="cascade_reference_only_compression.csv",
                        help="reference-only cascade CSV produced by cascade_cardinality_experiment "
                             "(default: %(default)s; skipped with a note if not found)")
    parser.add_argument("--delta-for-sorting-csv", default="cascade_delta_for_sorting_comparison.csv",
                        help="Delta vs. FOR prepass x leaf x sorting comparison CSV produced by "
                             "cascade_cardinality_experiment (default: %(default)s; skipped with a "
                             "note if not found)")
    parser.add_argument("--cardinality-decomposition-csv", default="cascade_for_cardinality_decomposition.csv",
                        help="plain-FOR global cardinality decomposition CSV produced by "
                             "cascade_cardinality_experiment (default: %(default)s; skipped with a "
                             "note if not found)")
    parser.add_argument("--bitrange-heatmap-csv", default="cascade_bitrange_cardinality_heatmap.csv",
                        help="bit-range cardinality heatmap CSV produced by cascade_cardinality_experiment "
                             "(default: %(default)s; skipped with a note if not found)")
    parser.add_argument("--output-dir", default="plots", help="directory for output PNGs (default: %(default)s)")
    parser.add_argument("--show", action="store_true", help="show plots interactively instead of saving")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: {csv_path} not found", file=sys.stderr)
        return 1

    df = load(csv_path)
    output_dir = Path(args.output_dir)

    plot_plain_for_sweep(df, output_dir, args.show)
    plot_plain_for_bitwidth(df, output_dir, args.show)
    plot_cascading_sweep(df, output_dir, args.show)
    plot_cascading_depth_comparison(df, output_dir, args.show)

    blockfpe_csv_path = Path(args.blockfpe_csv)
    blockfpe_df = None
    if blockfpe_csv_path.exists():
        blockfpe_df = pd.read_csv(blockfpe_csv_path)
        plot_blockfpe_plain_for(blockfpe_df, output_dir, args.show)
        plot_blockfpe_cascading(blockfpe_df, output_dir, args.show)
        plot_deepest_policy_effect(blockfpe_df, output_dir, args.show)
        print("Cardinality/compression correlation:")
        plot_cardinality_vs_compression(df, blockfpe_df, output_dir, args.show)
    else:
        print(f"note: {blockfpe_csv_path} not found, skipping BlockFPE compression plots")

    refonly_csv_path = Path(args.refonly_csv)
    refonly_df = None
    if refonly_csv_path.exists():
        refonly_df = pd.read_csv(refonly_csv_path)
        plot_reference_only_heatmap(refonly_df, output_dir, args.show)
    else:
        print(f"note: {refonly_csv_path} not found, skipping reference-only cascade heatmap")

    blockforfpe_csv_path = Path(args.blockforfpe_csv)
    if blockforfpe_csv_path.exists() and blockfpe_df is not None:
        blockforfpe_df = pd.read_csv(blockforfpe_csv_path)
        plot_blockforfpe_comparison(blockfpe_df, blockforfpe_df, refonly_df, output_dir, args.show)
    else:
        print(f"note: {blockforfpe_csv_path} not found, skipping BlockFORFPEEncoder comparison plot")

    delta_for_sorting_csv_path = Path(args.delta_for_sorting_csv)
    if delta_for_sorting_csv_path.exists():
        delta_for_sorting_df = pd.read_csv(delta_for_sorting_csv_path)
        plot_delta_for_sorting_comparison(delta_for_sorting_df, output_dir, args.show)
    else:
        print(f"note: {delta_for_sorting_csv_path} not found, skipping Delta vs. FOR sorting comparison plot")

    cardinality_decomposition_csv_path = Path(args.cardinality_decomposition_csv)
    if cardinality_decomposition_csv_path.exists():
        cardinality_decomposition_df = pd.read_csv(cardinality_decomposition_csv_path)
        plot_for_cardinality_decomposition(cardinality_decomposition_df, output_dir, args.show)
        plot_for_cardinality_decomposition_loglog(cardinality_decomposition_df, output_dir, args.show)
        plot_for_cardinality_decomposition_with_delta(cardinality_decomposition_df, output_dir, args.show)
        plot_for_cardinality_decomposition_loglog_with_delta(cardinality_decomposition_df, output_dir, args.show)
    else:
        print(f"note: {cardinality_decomposition_csv_path} not found, "
              f"skipping plain-FOR cardinality decomposition plot")

    bitrange_heatmap_csv_path = Path(args.bitrange_heatmap_csv)
    if bitrange_heatmap_csv_path.exists():
        bitrange_heatmap_df = pd.read_csv(bitrange_heatmap_csv_path)
        plot_bitrange_cardinality_heatmap(bitrange_heatmap_df, output_dir, args.show)
        plot_bitrange_cardinality_change(bitrange_heatmap_df, output_dir, args.show)
        plot_bitrange_cardinality_change_linear(bitrange_heatmap_df, output_dir, args.show)
    else:
        print(f"note: {bitrange_heatmap_csv_path} not found, skipping bit-range cardinality heatmap plot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
