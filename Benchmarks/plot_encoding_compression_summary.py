#!/usr/bin/env python3
"""
Plot compression-ratio results from a single heatmap_benchmark run.

Produces three figures:
  1. compression_ratios         — bar chart of overall compression ratio per encoder.
  2. autosis_breakdown          — grouped bar chart of each section's own
                                   compression ratio for each AutoSIS* variant.
  3. autosis_breakdown_stacked  — stacked-bytes composition of each AutoSIS*
                                   variant, with each section labeled by its
                                   own compression ratio.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Data (hardcoded from the heatmap_benchmark run pasted by the user)
# ---------------------------------------------------------------------------

N_ELEMENTS = 100_000  # matches the "[SubIntSplit] N=100000" in the benchmark log

TOP_LEVEL_RATIOS = {
    'Raw': 0.99999,
    'RawBitPacked': 1.04915,
    'BlockFPE': 0.969363,
    'BlockFORFPE': 1.12829,
    'AdaptiveBitPrefix': 1.14827,
    'OpenZL': 1.836,
    'Zstd': 1.13089,
    'AutoSIS': 1.60632,
    'AutoSIS_Delta': 1.61907,
    'AutoSIS_DeltaBlockFSE': 1.75411,
    'AutoSIS_OpenZL': 1.8856,
}

# variant -> list of (section_idx, bits, bytes, codec)
AUTOSIS_BREAKDOWN = {
    'AutoSIS': [
        (0, 13, 53361, 'BlockFrequencyPartition'),
        (1, 1, 12514, 'RawBitPacked'),
        (2, 8, 49911, 'BlockFrequencyPartition'),
        (3, 28, 350013, 'RawBitPacked'),
        (4, 8, 30869, 'RunLength'),
        (5, 6, 1302, 'RunLength'),
    ],
    'AutoSIS_Delta': [
        (0, 13, 53361, 'BlockFrequencyPartition'),
        (1, 8, 64152, 'AdaptiveDictionary'),
        (2, 1, 12514, 'RawBitPacked'),
        (3, 25, 312517, 'RawBitPacked'),
        (4, 17, 51512, 'CascadingFOR<l,resLevels=1,refLevels=1>'),
    ],
    'AutoSIS_DeltaBlockFSE': [
        (0, 3, 25771, 'BlockFSE'),
        (1, 4, 5428, 'BlockFSE'),
        (2, 5, 3184, 'BlockFSE'),
        (3, 2, 25010, 'RawBitPacked'),
        (4, 3, 31221, 'BlockFSE'),
        (5, 4, 11636, 'CascadingFOR<l,resLevels=1,refLevels=1>'),
        (6, 1, 1270, 'BlockFSE'),
        (7, 23, 287517, 'RawBitPacked'),
        (8, 4, 40773, 'CascadingFOR<l,resLevels=1,refLevels=1>'),
        (9, 4, 17428, 'CascadingFOR<l,resLevels=1,refLevels=1>'),
        (10, 8, 6706, 'RunLength'),
        (11, 3, 10, 'RawBitPacked'),
    ],
    'AutoSIS_OpenZL': [
        (0, 3, 25919, 'FSE'),
        (1, 9, 12230, 'FSE'),
        (2, 3, 37514, 'RawBitPacked'),
        (3, 7, 28502, 'OpenZL'),
        (4, 32, 317472, 'OpenZL'),
        (5, 10, 2568, 'OpenZL'),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fname: Path, **kwargs):
    plt.savefig(fname, dpi=150, **kwargs)
    print(f"Saved: {fname}")
    pdf_fname = fname.with_suffix('.pdf')
    plt.savefig(pdf_fname, **kwargs)
    print(f"Saved: {pdf_fname}")


def _encoder_colours(names):
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}


def _codec_colours(codec_names):
    """Stable colour per codec name, drawn from tab20 for good separation."""
    cmap = mpl.colormaps['tab20']
    return {name: cmap(i % 20) for i, name in enumerate(sorted(codec_names))}


def _is_light(rgb) -> bool:
    r, g, b = rgb[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 0.6


def _segment_ratio(bits: int, bytes_: int) -> float:
    """Compression ratio for one section, using the same definition as the
    top-level plot: (uncompressed size) / (compressed size). A section's
    uncompressed size is its raw bit-packed size at its own bit width."""
    raw_bytes = bits * N_ELEMENTS / 8
    return raw_bytes / bytes_


def _fmt_ratio(r: float) -> str:
    if r >= 100:
        return f'{r:.0f}×'
    if r >= 10:
        return f'{r:.1f}×'
    return f'{r:.2f}×'


# ---------------------------------------------------------------------------
# Plot 1 — overall compression ratio per encoder
# ---------------------------------------------------------------------------

def plot_compression_ratios(output_dir: Path):
    encoders = sorted(TOP_LEVEL_RATIOS, key=lambda n: TOP_LEVEL_RATIOS[n], reverse=True)
    ratios = [TOP_LEVEL_RATIOS[e] for e in encoders]
    colours = _encoder_colours(encoders)

    fig, ax = plt.subplots(figsize=(max(8, len(encoders) * 0.9), 6))
    bars = ax.bar(encoders, ratios, color=[colours[e] for e in encoders],
                   edgecolor='black', linewidth=0.5)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label='No compression (1×)')

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{ratio:.3f}×', ha='center', va='bottom', fontsize=9)

    ax.set_ylim(bottom=0)
    ax.set_ylabel('Compression ratio (higher = better)')
    ax.set_xlabel('Encoding')
    ax.set_title(f'Compression ratio by encoding (n={N_ELEMENTS:,})')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save_fig(output_dir / 'compression_ratios.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — AutoSIS per-section codec breakdown
# ---------------------------------------------------------------------------

def plot_autosis_breakdown(output_dir: Path):
    variants = list(AUTOSIS_BREAKDOWN.keys())
    all_codecs = {codec for sections in AUTOSIS_BREAKDOWN.values()
                  for _, _, _, codec in sections}
    codec_colours = _codec_colours(all_codecs)

    all_ratios = [_segment_ratio(bits, bytes_)
                  for sections in AUTOSIS_BREAKDOWN.values()
                  for _, bits, bytes_, _ in sections]
    max_ratio = max(all_ratios)

    fig, ax = plt.subplots(figsize=(max(10, len(variants) * 2.8), 7.5))

    seen_codecs = set()
    legend_handles = []
    legend_labels = []
    cluster_half_width = 0.4

    for x, variant in enumerate(variants):
        sections = AUTOSIS_BREAKDOWN[variant]
        n = len(sections)
        bar_width = (2 * cluster_half_width) / n
        offsets = [-cluster_half_width + bar_width * (i + 0.5) for i in range(n)]

        for offset, (_, bits, bytes_, codec) in zip(offsets, sections):
            ratio = _segment_ratio(bits, bytes_)
            colour = codec_colours[codec]
            bar = ax.bar(x + offset, ratio, width=bar_width * 0.92, color=colour,
                         edgecolor='black', linewidth=0.5, zorder=3)
            if codec not in seen_codecs:
                seen_codecs.add(codec)
                legend_handles.append(bar[0])
                legend_labels.append(codec)

            # Skip labeling near-baseline bars in dense clusters to avoid overlap —
            # their color already marks them as ~1x in the codec legend.
            if abs(ratio - 1.0) > 0.25:
                ax.text(x + offset, ratio * 1.08, _fmt_ratio(ratio), ha='center',
                        va='bottom', fontsize=6, rotation=90)

        overall = TOP_LEVEL_RATIOS[variant]
        overall_line = ax.hlines(overall, x - cluster_half_width, x + cluster_half_width,
                                  colors='black', linewidth=1.6, zorder=4,
                                  label='Overall ratio (from compression_ratios plot)')

    no_compression_line = ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1,
                                     alpha=0.7, zorder=2, label='No compression (1×)')

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_yscale('log')
    ax.set_ylim(min(1, min(all_ratios)) * 0.7, max_ratio * 3)
    ax.set_ylabel('Compression ratio per section (log scale, higher = better)')
    ax.set_xlabel('AutoSIS variant')
    ax.set_title('AutoSIS per-section compression ratio\n'
                 '(each bar = one section, using the same ratio definition as the compression ratio plot)')
    ax.grid(axis='y', which='major', alpha=0.3)
    ax.set_axisbelow(True)

    fig.subplots_adjust(right=0.72)
    codec_legend = ax.legend(legend_handles, legend_labels, title='Codec', fontsize=8,
                              loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    ax.add_artist(codec_legend)
    ax.legend(handles=[overall_line, no_compression_line], fontsize=8,
              loc='upper left', bbox_to_anchor=(1.02, 0.35), borderaxespad=0.)

    _save_fig(output_dir / 'autosis_breakdown.png', bbox_inches='tight')
    plt.close(fig)


def _place_outside_labels(ax, x_anchor, y_items, total, min_gap_frac=0.06):
    """Lay out (mid_y, text) callout labels to the right of a bar without
    overlapping, connected back to their segment with a thin leader line."""
    if not y_items:
        return
    y_items = sorted(y_items, key=lambda t: t[0])
    min_gap = total * min_gap_frac
    positions = [mid_y for mid_y, _ in y_items]

    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap:
            positions[i] = positions[i - 1] + min_gap

    max_allowed = total * 1.12
    if positions[-1] > max_allowed:
        positions[-1] = max_allowed
        for i in range(len(positions) - 2, -1, -1):
            if positions[i + 1] - positions[i] < min_gap:
                positions[i] = positions[i + 1] - min_gap

    for (mid_y, text), y_pos in zip(y_items, positions):
        ax.annotate(text, xy=(x_anchor, mid_y), xycoords='data',
                    xytext=(x_anchor + 0.55, y_pos), textcoords='data',
                    fontsize=7, va='center', ha='left',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.6,
                                    shrinkA=0, shrinkB=2))


def plot_autosis_breakdown_stacked(output_dir: Path):
    """Same stacked-bytes composition as the original breakdown chart, but
    every section is labeled with its own compression ratio (same definition
    as the top-level plot) instead of its bit width."""
    variants = list(AUTOSIS_BREAKDOWN.keys())
    all_codecs = {codec for sections in AUTOSIS_BREAKDOWN.values()
                  for _, _, _, codec in sections}
    codec_colours = _codec_colours(all_codecs)
    totals = {v: sum(bytes_ for _, _, bytes_, _ in AUTOSIS_BREAKDOWN[v]) for v in variants}
    max_total = max(totals.values())

    x_step = 1.8
    bar_width = 0.6
    fig, ax = plt.subplots(figsize=(max(12, len(variants) * 4.2), 11))

    seen_codecs = set()
    legend_handles = []
    legend_labels = []

    for i, variant in enumerate(variants):
        x = i * x_step
        sections = AUTOSIS_BREAKDOWN[variant]
        total = totals[variant]
        bottom = 0
        outside_items = []
        for _, bits, bytes_, codec in sections:
            colour = codec_colours[codec]
            bar = ax.bar(x, bytes_, bottom=bottom, width=bar_width, color=colour,
                         edgecolor='black', linewidth=0.5)
            if codec not in seen_codecs:
                seen_codecs.add(codec)
                legend_handles.append(bar[0])
                legend_labels.append(codec)

            ratio = _segment_ratio(bits, bytes_)
            label = _fmt_ratio(ratio)
            mid_y = bottom + bytes_ / 2
            if bytes_ / total > 0.04:
                text_colour = 'black' if _is_light(colour) else 'white'
                ax.text(x, mid_y, label, ha='center', va='center',
                        fontsize=8, color=text_colour, fontweight='bold')
            else:
                outside_items.append((mid_y, label))
            bottom += bytes_

        _place_outside_labels(ax, x + bar_width / 2, outside_items, total)
        ax.text(x, bottom + max_total * 0.015, f'{total / 1024:.1f} KB', ha='center',
                va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks([i * x_step for i in range(len(variants))])
    ax.set_xticklabels(variants, rotation=0, ha='center')
    ax.set_xlim(-1.0, (len(variants) - 1) * x_step + 1.0)
    ax.set_ylim(0, max_total * 1.2)
    ax.set_ylabel('Compressed size (bytes)')
    ax.set_xlabel('AutoSIS variant')
    ax.set_title('AutoSIS per-section codec breakdown\n'
                 '(bar height = compressed bytes; label = that section\'s own compression ratio)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    fig.subplots_adjust(right=0.78)
    ax.legend(legend_handles, legend_labels, title='Codec', fontsize=8,
              loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    _save_fig(output_dir / 'autosis_breakdown_stacked.png', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot compression-ratio results from a heatmap_benchmark run')
    parser.add_argument('-o', '--output', type=Path, default=Path('plots'),
                        help='Output directory for plots (default: plots/)')
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)

    plot_compression_ratios(args.output)
    plot_autosis_breakdown(args.output)
    plot_autosis_breakdown_stacked(args.output)


if __name__ == '__main__':
    main()
