#!/usr/bin/env python3
"""
Plot benchmark results from JSON output.

Produces one horizontal-bar chart per metric so encoders are easy to
compare side-by-side even when there is only a single dataset/size.
All log-scaled axes guard against zero/negative values by falling back
to a linear scale with a warning.
"""

import json
import colorsys
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import argparse

# ---------------------------------------------------------------------------
# Colour palette — one stable colour per encoder name so it is consistent
# across all charts.
# ---------------------------------------------------------------------------
_PALETTE = plt.rcParams['axes.prop_cycle'].by_key()['color']

def _encoder_colours(encoders):
    return {enc: _PALETTE[i % len(_PALETTE)] for i, enc in enumerate(encoders)}


@dataclass(frozen=True)
class GroupRule:
    name: str
    predicate: Callable[[str], bool]
    sort_key: Callable[[str], tuple]


def _default_group_rules() -> List[GroupRule]:
    autosubintsplit_re = re.compile(r'^AutoSubIntSplit(?P<num>\d+)$')
    openzl_re = re.compile(r'^OpenZL(?P<num>\d+)?$')
    subint_re = re.compile(r'^(?P<prefix>.*)SubInt(?P<num>\d+)?$')
    varint_re = re.compile(r'.*VarInt$')
    raw_re = re.compile(r'^Raw')
    trisplit_re = re.compile(r'^TriSplit')
    trisplit_openzlonly_re = re.compile(r'^TriSplitOpenZLOnly(?P<num>\d+)?$')

    def openzl_key(name: str) -> tuple:
        match = openzl_re.match(name)
        num = int(match.group('num')) if match and match.group('num') else -1
        return (num, name)

    def raw_key(name: str) -> tuple:
        return (0 if name == 'Raw' else 1, name)

    def varint_key(name: str) -> tuple:
        return (0 if name == 'VarInt' else 1, name)

    def autosubintsplit_key(name: str) -> tuple:
        match = autosubintsplit_re.match(name)
        num = int(match.group('num')) if match else -1
        exact_first = 0 if match else 1
        return (num, exact_first, name)

    def subint_key(name: str) -> tuple:
        match = subint_re.match(name)
        num = int(match.group('num')) if match and match.group('num') else -1
        exact_name = f'SubInt{num}' if num >= 0 else 'SubInt'
        exact_first = 0 if name == exact_name else 1
        return (num, exact_first, name)

    def trisplit_key(name: str) -> tuple:
        # TriSplitOpenZLOnly should be first, then its numeric variants ascending
        m_only = trisplit_openzlonly_re.match(name)
        if m_only:
            num = int(m_only.group('num')) if m_only.group('num') else -1
            exact_first = 0 if m_only.group('num') is None else 1
            return (0, exact_first, num, name)

        # If TriSplit includes OpenZL-style suffix, order by that numeric
        m = re.search(r'OpenZL(?P<num>\d+)?$', name)
        if m:
            num = int(m.group('num')) if m.group('num') else -1
            return (1, num, name)
        return (2, name)

    return [
        GroupRule(
            name='AutoSubIntSplit',
            predicate=lambda n: n.startswith('AutoSubIntSplit'),
            sort_key=autosubintsplit_key
        ),
        GroupRule(
            name='Raw',
            predicate=lambda n: bool(raw_re.match(n)),
            sort_key=raw_key
        ),
        GroupRule(
            name='SubInt',
            predicate=lambda n: ('SubInt' in n) and (not n.startswith('AutoSubIntSplit')),
            sort_key=subint_key
        ),
        GroupRule(
            name='TriSplit',
            predicate=lambda n: bool(trisplit_re.match(n)),
            sort_key=trisplit_key
        ),
        GroupRule(
            name='OpenZL',
            predicate=lambda n: bool(openzl_re.match(n)),
            sort_key=openzl_key
        ),
        GroupRule(
            name='VarInt',
            predicate=lambda n: bool(varint_re.match(n)),
            sort_key=varint_key
        ),
    ]


def _group_encoders(encoders: Iterable[str], rules: List[GroupRule]) -> List[tuple[str, List[str]]]:
    remaining = list(dict.fromkeys(encoders))
    grouped: List[tuple[str, List[str]]] = []
    for rule in rules:
        matches = [e for e in remaining if rule.predicate(e)]
        if matches:
            ordered = sorted(matches, key=rule.sort_key)
            grouped.append((rule.name, ordered))
            remaining = [e for e in remaining if e not in matches]

    if remaining:
        grouped.append(("Other", sorted(remaining)))
    return grouped


def _grouped_encoder_order(encoders: Iterable[str], rules: List[GroupRule]) -> List[str]:
    grouped = _group_encoders(encoders, rules)
    return [enc for _, encs in grouped for enc in encs]


def _grouped_encoder_colours(encoders: Iterable[str], rules: List[GroupRule]) -> dict:
    grouped = _group_encoders(encoders, rules)

    # Use deliberately separated base hues per group to improve distinction
    # between the darkest shades of each gradient.
    preferred_group_hues = {
        'AutoSubIntSplit': 0.58,  # blue-cyan
        'Raw':             0.00,  # red
        'SubInt':          0.30,  # green
        'TriSplit':        0.78,  # violet
        'OpenZL':          0.12,  # amber
        'VarInt':          0.92,  # magenta-red
        'Other':           0.52,  # fallback teal-ish
    }

    # Additional hues for unforeseen groups, spaced by golden angle.
    golden_angle = 0.61803398875
    colours = {}
    for idx, (group_name, encs) in enumerate(grouped):
        h = preferred_group_hues.get(group_name, (0.07 + idx * golden_angle) % 1.0)
        # Keep base saturation/lightness in a range that preserves group identity
        # even for the darkest shades.
        base_s = 0.82
        base_l = 0.56
        count = max(len(encs), 1)
        for j, enc in enumerate(encs):
            # Vary lightness within group while staying away from near-black,
            # so dark shades across groups remain distinguishable.
            if count == 1:
                l_j = base_l
                s_j = base_s
            else:
                l_min = 0.34
                l_max = 0.80
                l_j = l_min + (l_max - l_min) * (j / (count - 1))
                # Slight saturation variation gives depth without collapsing hues.
                s_min = 0.66
                s_max = 0.92
                s_j = s_min + (s_max - s_min) * (j / (count - 1))
            r, g, b = colorsys.hls_to_rgb(h, l_j, s_j)
            colours[enc] = (r, g, b)
    return colours


def _sorted_encoders(results, group_encoders: bool = False, group_rules: List[GroupRule] | None = None):
    """Alphabetical encoder order but keep Raw at the top for readability."""
    encoders = sorted(set(r['encoderName'] for r in results['results']),
                      key=lambda n: (0, n) if n == 'Raw' else (1, n))
    if not group_encoders:
        return encoders

    rules = group_rules if group_rules is not None else _default_group_rules()
    return _grouped_encoder_order(encoders, rules)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def _pick_size(results):
    """Use the largest data size present."""
    return max(r['dataSize'] for r in results['results'])


def _pick_size_for_dataset(results, dataset):
    """Pick the largest dataSize available for a specific dataset (fallback to global max)."""
    sizes = [r['dataSize'] for r in results['results'] if r['datasetName'] == dataset]
    return max(sizes) if sizes else _pick_size(results)


def _row(results, encoder, dataset, size):
    """Return the single matching result dict, or None."""
    matches = [r for r in results['results']
               if r['encoderName'] == encoder
               and r['datasetName'] == dataset
               and r['dataSize'] == size]
    return matches[0] if matches else None


def _has_custom_metric(results, key: str) -> bool:
    for r in results['results']:
        metrics = r.get('metrics', {})
        custom = metrics.get('customMetrics', {})
        if key in custom:
            return True
    return False


_AUTOSUBINTSPLIT_FORCED_RE = re.compile(r'^AutoSubIntSplit(?P<splits>\d+)$')


def _autosubintsplit_forced_split_count(encoder_name: str):
    match = _AUTOSUBINTSPLIT_FORCED_RE.match(encoder_name or '')
    if not match:
        return None
    return int(match.group('splits'))


def _autosubintsplit_forced_split_rows(results, dataset):
    size = _pick_size_for_dataset(results, dataset)
    rows = []
    for r in results['results']:
        if r.get('datasetName') != dataset or r.get('dataSize') != size:
            continue
        splits = _autosubintsplit_forced_split_count(r.get('encoderName', ''))
        if splits is None:
            continue
        rows.append((splits, r))
    rows.sort(key=lambda item: (item[0], item[1].get('encoderName', '')))
    return size, rows


def _has_autosubintsplit_forced_data(results) -> bool:
    return any(
        _autosubintsplit_forced_split_count(r.get('encoderName', '')) is not None
        for r in results['results']
    )


def _plot_split_sweep_panel(ax, split_counts, values, title, ylabel, *, log_y=False, unit='', fmt='.3g'):
    ax.plot(split_counts, values, marker='o', color='tab:blue', linewidth=1.5, markersize=5)
    ax.set_title(title)
    ax.set_xlabel('Forced split count')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if split_counts:
        ax.set_xticks(split_counts)
        ax.set_xticklabels([str(split) for split in split_counts])
    if log_y and all(v > 0 for v in values):
        ax.set_yscale('log')
        min_positive = min(v for v in values if v > 0)
        max_positive = max(v for v in values if v > 0)
        ax.set_ylim([min(min_positive * 0.5, 1), max_positive * 2])

    for x, y in zip(split_counts, values):
        if y and y > 0:
            label = f'{y:{fmt}} {unit}'.strip()
            ax.annotate(label, (x, y), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=7)
        else:
            ax.annotate('N/A', (x, 0.02), textcoords='offset points', xytext=(0, 0),
                        ha='center', fontsize=7, color='grey')


def plot_autosubintsplit_forced_split_sweep(results, output_dir, colours, source_label, group_encoders=False):
    """Plot AutoSubIntSplit variants that end in a numeric split count.

    The numeric suffix is interpreted as the forced number of splits used by the
    encoder, and each dataset gets a 5-panel summary across those split counts.
    """
    datasets = sorted(set(r['datasetName'] for r in results['results']))

    for dataset in datasets:
        size, rows = _autosubintsplit_forced_split_rows(results, dataset)
        if not rows:
            continue

        split_counts = [splits for splits, _ in rows]
        encode_ms = [row['metrics']['timing'].get('encodeTime_ns', 0) / 1e6 for _, row in rows]
        bulk_decode_ms = [row['metrics']['timing'].get('decodeBulkTime_ns', 0) / 1e6 for _, row in rows]
        random_decode_ns = [row['metrics']['randomAccess'].get('averageRandomAccessTime_ns', 0) for _, row in rows]
        ranged_decode_ms = [row['metrics']['randomAccess'].get('averageRangeAccessTime_ns', 0) / 1e6 for _, row in rows]
        compression_x = []
        for _, row in rows:
            ratio = row['metrics']['memory'].get('compressionRatio', 0)
            compression_x.append(1.0 / ratio if ratio and ratio > 0 else 0)

        fig, axes = plt.subplots(3, 2, figsize=(13, 12))
        fig.suptitle(
            f'AutoSubIntSplit forced-split sweep — {dataset}  (n={size:,})',
            fontsize=13, y=0.98)
        _annotate_source(fig, source_label)

        _plot_split_sweep_panel(
            axes[0, 0], split_counts, encode_ms,
            title='Encode time',
            ylabel='Time (ms)',
            log_y=True,
            unit='ms',
            fmt='.2f',
        )
        _plot_split_sweep_panel(
            axes[0, 1], split_counts, bulk_decode_ms,
            title='Bulk decode time',
            ylabel='Time (ms)',
            log_y=True,
            unit='ms',
            fmt='.2f',
        )
        _plot_split_sweep_panel(
            axes[1, 0], split_counts, random_decode_ns,
            title='Random access time',
            ylabel='Time (ns)',
            log_y=True,
            unit='ns',
            fmt='.0f',
        )
        _plot_split_sweep_panel(
            axes[1, 1], split_counts, ranged_decode_ms,
            title='Ranged access time',
            ylabel='Time (ms)',
            log_y=True,
            unit='ms',
            fmt='.2f',
        )
        _plot_split_sweep_panel(
            axes[2, 0], split_counts, compression_x,
            title='Compression ratio',
            ylabel='Compression (× smaller than raw)',
            log_y=False,
            unit='×',
            fmt='.3f',
        )
        axes[2, 1].axis('off')
        axes[2, 1].text(
            0.5, 0.6,
            'Numeric suffix on AutoSubIntSplitN\nmeans N forced splits',
            ha='center', va='center', fontsize=10, transform=axes[2, 1].transAxes)
        axes[2, 1].grid(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = output_dir / f'autosubintsplit_forced_split_sweep_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig)


def _safe_log_scale(ax, axis='x'):
    """Switch to log scale only when all visible data are positive."""
    setter = ax.set_xscale if axis == 'x' else ax.set_yscale
    getter = ax.get_xlim   if axis == 'x' else ax.get_ylim
    lo, hi = getter()
    if lo > 0 and hi > 0:
        setter('log')


# Encoders considered non-random-access for plotting annotation purposes.
# Per request: names containing OpenZL, VarInt, Delta, LZ4, Huff/Huffman.
_NON_RANDOM_ACCESS_TOKENS = (
    'openzl',
    'varint',
    'delta',
    'lz4',
    'huff',
    'huffman',
    'fse',
)


def _is_random_access_encoder(encoder_name: str) -> bool:
    n = encoder_name.lower()
    return not any(tok in n for tok in _NON_RANDOM_ACCESS_TOKENS)


def _encoder_display_label(encoder_name: str) -> str:
    return f"{encoder_name} [{'RA' if _is_random_access_encoder(encoder_name) else 'No-RA'}]"


def _set_encoder_axis_labels(ax, y, encoders):
    ax.set_yticks(y)
    ax.set_yticklabels([_encoder_display_label(e) for e in encoders])


def _ra_bar_legend_handles():
    return [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='RA: random-access capable'),
        mpatches.Patch(facecolor='lightgray', edgecolor='black', hatch='///', label='No-RA: fallback/sequential access'),
    ]


def _ra_scatter_legend_handles():
    return [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black',
                      markersize=7, label='RA: random-access capable'),
        mlines.Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markeredgecolor='black',
                      markersize=7, label='No-RA: fallback/sequential access'),
    ]


def _hbar(ax, encoders, values, colours, unit='', fmt='.3g', log=True):
    """
    Draw a horizontal bar chart with one bar per encoder.
    Bars with value == 0 are drawn as a thin stub and labelled 'N/A'.
    """
    y = np.arange(len(encoders))
    bar_h = 0.6
    bars = ax.barh(y, values, height=bar_h,
                   color=[colours[e] for e in encoders])

    # Visually mark non-random-access encoders.
    for bar, enc in zip(bars, encoders):
        if not _is_random_access_encoder(enc):
            bar.set_hatch('///')
            bar.set_edgecolor('black')
            bar.set_linewidth(0.6)

    # Value labels
    x_max = max((v for v in values if v and v > 0), default=1)
    for bar, val in zip(bars, values):
        if val and val > 0:
            label = f'{val:{fmt}} {unit}'.strip()
            ax.text(val + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
                    label, va='center', fontsize=8)
        else:
            ax.text(x_max * 0.01, bar.get_y() + bar.get_height() / 2,
                    'N/A', va='center', fontsize=8, color='grey')

    _set_encoder_axis_labels(ax, y, encoders)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    if log:
        positive = [v for v in values if v and v > 0]
        if positive:
            ax.set_xscale('log')

    return bars


def _lighten_color(color, amount=0.35):
    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, l + (1.0 - l) * amount)
    return colorsys.hls_to_rgb(h, l, s)


def _darken_color(color, amount=0.25):
    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, l * (1.0 - amount))
    return colorsys.hls_to_rgb(h, l, s)


def _is_light_color(color) -> bool:
    r, g, b = color
    # Relative luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 0.6


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _annotate_source(fig, source_label: str):
    """Place the source JSON filename at the top-left of the figure."""
    if source_label:
        fig.text(0.01, 0.995, source_label, ha='left', va='top', fontsize=8, alpha=0.7)


def plot_compression(results, output_dir, colours, source_label, group_encoders=False):
    """Inverted compression ratio (higher = better) and bits-per-element."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        inv_ratios = []
        bpe_vals   = []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                ratio = row['metrics']['memory']['compressionRatio']
                inv_ratios.append(1.0 / ratio if ratio > 0 else 0)
                bpe_vals.append(row['metrics']['memory']['bitsPerElement'])
            else:
                inv_ratios.append(0)
                bpe_vals.append(0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(encoders) * 0.7 + 1.5)))
        fig.suptitle(f'Compression — {dataset}  (n={size:,})', fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        _hbar(ax1, encoders, inv_ratios, colours, unit='×', fmt='.3f', log=False)
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
                    label='No compression (1×)')
        ax1.set_xlabel('Compression ratio (higher = better)')
        ax1.set_title('Inverted compression ratio')
        ax1.legend(handles=[
            mlines.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='No compression (1×)'),
            *_ra_bar_legend_handles(),
        ], fontsize=8)

        _hbar(ax2, encoders, bpe_vals, colours, unit='bits', fmt='.2f', log=False)
        ax2.axvline(x=64, color='red', linestyle='--', linewidth=1, alpha=0.6,
                    label='Raw int64 (64 bits)')
        ax2.set_xlabel('Bits per element (lower = better)')
        ax2.set_title('Bits per element')
        ax2.legend(handles=[
            mlines.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Raw int64 (64 bits)'),
            *_ra_bar_legend_handles(),
        ], fontsize=8)

        plt.tight_layout()
        fname = output_dir / f'compression_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_compression_overhead(results, output_dir, colours, source_label, group_encoders=False):
    """Compression ratio normalized to the best (smallest) ratio per dataset/size."""
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        ratios = []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                ratios.append(row['metrics']['memory']['compressionRatio'])
            else:
                ratios.append(0)

        positive = [r for r in ratios if r and r > 0]
        best_ratio = min(positive) if positive else 0
        overhead = [(r / best_ratio) if (r and r > 0 and best_ratio > 0) else 0 for r in ratios]

        fig, ax = plt.subplots(figsize=(10, max(4, len(encoders) * 0.7 + 1.5)))
        fig.suptitle(f'Compression overhead — {dataset}  (n={size:,})', fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        _hbar(ax, encoders, overhead, colours, unit='×', fmt='.3f', log=False)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
                    label='Best compression (1×)')
        ax.set_xlabel('Compression ratio relative to best (lower = better)')
        ax.set_title('Normalized compression ratio (overhead)')
        ax.legend(handles=[
            mlines.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Best compression (1×)'),
            *_ra_bar_legend_handles(),
        ], fontsize=8)

        plt.tight_layout()
        fname = output_dir / f'compression_overhead_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_encode_decode_time(results, output_dir, colours, source_label, group_encoders=False):
    """Encode + bulk decode time in ms."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        enc_ms, enc_tp = [], []
        selection_ms = []
        dec_ms, dec_tp = [], []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                t  = row['metrics']['timing']
                enc_ms.append(t['encodeTime_ns']     / 1e6)
                dec_ms.append(t['decodeBulkTime_ns'] / 1e6)
                enc_tp.append(t['encodeElementsPerSecond']      / 1e6)
                dec_tp.append(t['decodeBulkElementsPerSecond']  / 1e6)
                custom = row.get('metrics', {}).get('customMetrics', {})
                selection_ns = custom.get('selectionTime_ns', 0)
                selection_ms.append(selection_ns / 1e6 if selection_ns else 0)
            else:
                enc_ms.append(0); enc_tp.append(0)
                dec_ms.append(0); dec_tp.append(0)
                selection_ms.append(0)

        h = max(4, len(encoders) * 0.7 + 1.5)
        fig, axes = plt.subplots(1, 2, figsize=(14, h))
        fig.suptitle(f'Encode / Decode time — {dataset}  (n={size:,})', fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        # Encode time plot with optional selection-time stack
        ax = axes[0]
        y = np.arange(len(encoders))
        bar_h = 0.6
        has_selection = any(v > 0 for v in selection_ms)
        x_max = max((v for v in enc_ms if v and v > 0), default=1)
        min_label_width = x_max * 0.05
        label_x_offset = x_max * 0.01
        if has_selection:
            encode_only = [max(e - s, 0) for e, s in zip(enc_ms, selection_ms)]
            selection_colors = [_lighten_color(colours[e], 0.45) for e in encoders]
            encode_colors = [_darken_color(colours[e], 0.2) for e in encoders]
            ax.barh(y, selection_ms, height=bar_h, color=selection_colors,
                    label='Selection', hatch='///', edgecolor='black', linewidth=0.5)
            ax.barh(y, encode_only, left=selection_ms, height=bar_h, color=encode_colors, label='Encode')
            total_vals = enc_ms
            # Label selection and encode-only segments (if non-zero)
            for idx, (sel, enc_only) in enumerate(zip(selection_ms, encode_only)):
                if sel > 0:
                    x_sel = sel * 0.5
                    if sel >= min_label_width:
                        sel_text_color = 'black' if _is_light_color(selection_colors[idx]) else 'white'
                        ax.text(x_sel, y[idx], f'{sel:.1f} ms', va='center', ha='center',
                                fontsize=7, color=sel_text_color, fontweight='bold')
                    else:
                        ax.text(sel + label_x_offset, y[idx], f'{sel:.1f} ms', va='center', ha='left',
                                fontsize=7, color='black', fontweight='bold')
                if enc_only > 0:
                    x_enc = sel + enc_only * 0.5
                    if enc_only >= min_label_width:
                        enc_text_color = 'black' if _is_light_color(encode_colors[idx]) else 'white'
                        ax.text(x_enc, y[idx], f'{enc_only:.1f} ms', va='center', ha='center',
                                fontsize=7, color=enc_text_color, fontweight='bold')
                    else:
                        ax.text(sel + enc_only + label_x_offset, y[idx], f'{enc_only:.1f} ms',
                                va='center', ha='left', fontsize=7, color='black', fontweight='bold')
        else:
            total_vals = enc_ms
            bars = ax.barh(y, total_vals, height=bar_h, color=[colours[e] for e in encoders], label='Encode')
            for bar, enc in zip(bars, encoders):
                if not _is_random_access_encoder(enc):
                    bar.set_hatch('///')
                    bar.set_edgecolor('black')
                    bar.set_linewidth(0.6)
            for idx, val in enumerate(total_vals):
                if val > 0:
                    if val >= min_label_width:
                        enc_text_color = 'black' if _is_light_color(colours[encoders[idx]]) else 'white'
                        ax.text(val * 0.5, y[idx], f'{val:.1f} ms', va='center', ha='center',
                                fontsize=7, color=enc_text_color, fontweight='bold')
                    else:
                        ax.text(val + label_x_offset, y[idx], f'{val:.1f} ms', va='center', ha='left',
                                fontsize=7, color='black', fontweight='bold')

        _set_encoder_axis_labels(ax, y, encoders)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        if any(v > 0 for v in total_vals):
            ax.set_xscale('log')
        ax.set_xlabel('Time (ms, log scale, lower = better)')
        ax.set_title('Encode time')
        ra_leg = ax.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')
        if has_selection:
            main_leg = ax.legend(fontsize=8, loc='lower right')
            ax.add_artist(ra_leg)
            ax.add_artist(main_leg)

        # Decode plot (unchanged)
        ax = axes[1]
        _hbar(ax, encoders, dec_ms, colours, unit='ms', fmt='.1f', log=True)
        ax.set_xlabel('Time (ms, log scale, lower = better)')
        ax.set_title('Bulk decode time')
        ax.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        plt.tight_layout()
        fname = output_dir / f'encode_decode_time_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_random_access(results, output_dir, colours, source_label, group_encoders=False):
    """Average, min, and max single-element random access time (ns)."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        avg_ns, min_ns, max_ns, strided_ns = [], [], [], []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                ra = row['metrics']['randomAccess']
                avg_ns.append(ra.get('averageRandomAccessTime_ns', 0))
                min_ns.append(ra.get('minRandomAccessTime_ns',     0))
                max_ns.append(ra.get('maxRandomAccessTime_ns',     0))
                strided_ns.append(ra.get('averageStridedAccessTime_ns', 0))
            else:
                avg_ns.append(0); min_ns.append(0)
                max_ns.append(0); strided_ns.append(0)

        h = max(4, len(encoders) * 0.7 + 1.5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, h))
        fig.suptitle(f'Random / Strided access — {dataset}  (n={size:,})', fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        # Left: average random access with min/max error bars
        y      = np.arange(len(encoders))
        bar_h  = 0.6
        valid  = [(a, lo, hi) for a, lo, hi in zip(avg_ns, min_ns, max_ns) if a > 0]

        bars = ax1.barh(y, avg_ns, height=bar_h,
                        color=[colours[e] for e in encoders])
        for bar, enc in zip(bars, encoders):
            if not _is_random_access_encoder(enc):
                bar.set_hatch('///')
                bar.set_edgecolor('black')
                bar.set_linewidth(0.6)
        # Label each bar with its value (ns) similar to _hbar helper
        x_max = max((v for v in avg_ns if v and v > 0), default=1)
        for bar, val in zip(bars, avg_ns):
            if val and val > 0:
                ax1.text(val + x_max * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f"{val:.0f} ns",
                         va='center', fontsize=8)
            else:
                ax1.text(x_max * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         'N/A', va='center', fontsize=8, color='grey')
        # # Error bars (xerr = [left_err, right_err])
        # x_lo = [max(0, a - lo) for a, lo in zip(avg_ns, min_ns)]
        # x_hi = [hi - a          for a, hi in zip(avg_ns, max_ns)]
        # ax1.errorbar(avg_ns, y, xerr=[x_lo, x_hi],
        #              fmt='none', color='black', capsize=3, linewidth=1)

        _set_encoder_axis_labels(ax1, y, encoders); ax1.invert_yaxis()
        ax1.set_xlabel('Time (ns, log scale, lower = better)')
        ax1.set_title('Avg random access (bars)')
        ax1.grid(axis='x', alpha=0.3)
        if any(v > 0 for v in avg_ns):
            ax1.set_xscale('log')
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        # Right: average strided access
        _hbar(ax2, encoders, strided_ns, colours, unit='ns', fmt='.0f', log=True)
        ax2.set_xlabel('Time (ns, log scale, lower = better)')
        ax2.set_title('Avg strided access (stride from config)')
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        plt.tight_layout()
        fname = output_dir / f'random_access_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_range_access(results, output_dir, colours, source_label, group_encoders=False):
    """Average range-query time (ms) and implied throughput (M elem/s)."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        avg_ms, range_sz, tp_vals = [], [], []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                ra  = row['metrics']['randomAccess']
                avg_ns  = ra.get('averageRangeAccessTime_ns', 0)
                r_size  = ra.get('averageRangeSize', 0)
                avg_ms.append(avg_ns / 1e6)
                range_sz.append(r_size)
                # throughput = elements decoded per second across one range query
                tp = (r_size / (avg_ns / 1e9)) / 1e6 if avg_ns > 0 else 0
                tp_vals.append(tp)
            else:
                avg_ms.append(0); range_sz.append(0); tp_vals.append(0)

        # Range sizes should all be the same, but take the max for the label
        rng_label = f'{int(max(range_sz)):,}' if range_sz else '?'

        h = max(4, len(encoders) * 0.7 + 1.5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, h))
        fig.suptitle(
            f'Range access (avg range = {rng_label} elems) — {dataset}  (n={size:,})',
            fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        _hbar(ax1, encoders, avg_ms, colours, unit='ms', fmt='.2f', log=True)
        ax1.set_xlabel('Time (ms, log scale, lower = better)')
        ax1.set_title('Avg range-query latency')
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        _hbar(ax2, encoders, tp_vals, colours, unit='M e/s', fmt='.1f', log=True)
        ax2.set_xlabel('Throughput (M elem/s, log scale, higher = better)')
        ax2.set_title('Range-query decode throughput')
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        plt.tight_layout()
        fname = output_dir / f'range_access_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_throughput_summary(results, output_dir, colours, source_label, group_encoders=False):
    """Single summary chart: encode + decode throughput side-by-side."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        enc_tp, dec_tp = [], []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                t = row['metrics']['timing']
                enc_tp.append(t['encodeElementsPerSecond']     / 1e6)
                dec_tp.append(t['decodeBulkElementsPerSecond'] / 1e6)
            else:
                enc_tp.append(0); dec_tp.append(0)

        h  = max(4, len(encoders) * 0.7 + 1.5)
        y  = np.arange(len(encoders))
        bh = 0.35

        fig, ax = plt.subplots(figsize=(10, h))
        fig.suptitle(f'Throughput summary — {dataset}  (n={size:,})', fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        bars_enc = ax.barh(y - bh/2, enc_tp, height=bh,
                           color=[colours[e] for e in encoders], label='Encode', alpha=0.85)
        bars_dec = ax.barh(y + bh/2, dec_tp, height=bh,
                           color=[colours[e] for e in encoders], label='Decode (bulk)',
                           alpha=0.55, hatch='//')
        for b1, b2, enc in zip(bars_enc, bars_dec, encoders):
            if not _is_random_access_encoder(enc):
                b1.set_hatch('///'); b1.set_edgecolor('black'); b1.set_linewidth(0.6)
                b2.set_hatch('xx');  b2.set_edgecolor('black'); b2.set_linewidth(0.6)

        _set_encoder_axis_labels(ax, y, encoders)
        ax.invert_yaxis()
        ax.set_xlabel('Throughput (M elements/sec, log scale, higher = better)')
        ax.grid(axis='x', alpha=0.3)
        if any(v > 0 for v in enc_tp + dec_tp):
            ax.set_xscale('log')

        # Custom legend: colour patches for each encoder, then encode/decode style
        enc_patch  = mpatches.Patch(facecolor='grey', alpha=0.85, label='Encode (solid)')
        dec_patch  = mpatches.Patch(facecolor='grey', alpha=0.55, hatch='//', label='Decode (hatch)')
        handles    = [enc_patch, dec_patch, *_ra_bar_legend_handles()]
        ax.legend(handles=handles, fontsize=8, loc='lower right')

        plt.tight_layout()
        fname = output_dir / f'throughput_summary_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_compression_vs_random_access(results, output_dir, colours, source_label, group_encoders=False):
    """Scatter of compression ratio vs random access time, with Pareto frontier.

    Axes orientation
    ----------------
    X : compression ratio (higher = better → right)
    Y : avg random access time — axis is **inverted** so that lower (faster)
        values appear at the **top**.  The ideal encoder therefore sits in the
        top-right corner.  The global Pareto frontier and the per-group
        frontiers all curve toward that corner.
    """
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    # Helper: compute the Pareto frontier for a list of (compression_x, ra)
    # pairs.  "Better" = higher compression_x AND lower ra.
    # Sort by compression descending; greedily keep points where ra strictly
    # decreases → each successive point is faster for the same or worse compression.
    def _pareto(pts):
        frontier = []
        for cx, ra in sorted(pts, key=lambda t: t[0], reverse=True):
            if not frontier or ra < frontier[-1][1]:
                frontier.append((cx, ra))
        return frontier

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        points = []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if not row:
                continue
            ratio = row['metrics']['memory']['compressionRatio']
            ra = row['metrics']['randomAccess'].get('averageRandomAccessTime_ns', 0)
            if ratio <= 0 or ra <= 0:
                continue
            points.append((enc, 1.0 / ratio, ra))

        if not points:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'Compression vs random access — {dataset}  (n={size:,})', fontsize=13, y=0.98)
        _annotate_source(fig, source_label)

        for enc, ratio, ra in points:
            marker = 'o' if _is_random_access_encoder(enc) else 'X'
            ax.scatter(ratio, ra, color=colours.get(enc, 'tab:blue'), marker=marker, s=45, zorder=3)

        # ── Global Pareto frontier ─────────────────────────────────────────
        global_frontier = _pareto([(cx, ra) for _, cx, ra in points])
        pareto_handle = None
        if len(global_frontier) >= 2:
            line = ax.plot([p[0] for p in global_frontier],
                           [p[1] for p in global_frontier],
                           color='black', linestyle='--', linewidth=1.4,
                           zorder=4, label='Pareto frontier (global)')
            pareto_handle = line[0]

        # ── Per-group Pareto frontiers ─────────────────────────────────────
        group_frontier_handles = []
        group_rules = _default_group_rules()
        groups = _group_encoders([p[0] for p in points], group_rules)
        for group_name, group_encs in groups:
            group_pts = [(cx, ra) for enc, cx, ra in points if enc in group_encs]
            if len(group_pts) < 2:
                continue
            gf = _pareto(group_pts)
            if len(gf) < 2:
                continue
            # Representative colour: use the first encoder in the group.
            rep_color = colours.get(group_encs[0], 'grey')
            ax.plot([p[0] for p in gf], [p[1] for p in gf],
                    color=rep_color, linestyle=':', linewidth=1.0,
                    alpha=0.55, zorder=2)
            group_frontier_handles.append(
                mlines.Line2D([0], [0], color=rep_color, linestyle=':',
                              linewidth=1.0, alpha=0.55, label=group_name))

        # ── Point labels (staggered, overlaps culled) ──────────────────────
        offsets = [(6, 4), (6, -6), (-6, 4), (-6, -6), (10, 0), (-10, 0), (0, 8), (0, -8)]
        texts = []
        for idx, (enc, ratio, ra) in enumerate(points):
            dx, dy = offsets[idx % len(offsets)]
            txt = ax.annotate(enc, (ratio, ra), textcoords="offset points",
                              xytext=(dx, dy), fontsize=7,
                              bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                        ec="none", alpha=0.7))
            texts.append(txt)

        fig.canvas.draw()
        renderer_fn = getattr(fig.canvas, "get_renderer", None)
        renderer = renderer_fn() if callable(renderer_fn) else None
        if renderer is not None:
            placed_bboxes = []
            for txt in texts:
                bbox = txt.get_window_extent(renderer=renderer).expanded(1.05, 1.1)
                if any(bbox.overlaps(prev) for prev in placed_bboxes):
                    txt.set_visible(False)
                else:
                    placed_bboxes.append(bbox)

    # ── Legends ────────────────────────────────────────────────────────
        # Group scatter-dot legend (when --group-encoders is active)
        group_dot_handles = []
        group_dot_labels = []
        if group_encoders:
            for group_name, encs in groups:
                if not encs:
                    continue
                color = colours.get(encs[0], 'tab:blue')
                group_dot_handles.append(
                    mlines.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=6))
                group_dot_labels.append(group_name)

        # Combine global frontier + per-group frontier entries
        frontier_handles = []
        frontier_labels  = []
        if pareto_handle is not None:
            frontier_handles.append(pareto_handle)
            frontier_labels.append('Global Pareto frontier')
        if group_frontier_handles:
            frontier_handles += group_frontier_handles
            frontier_labels  += [h.get_label() for h in group_frontier_handles]

        # Place all legends at the top with aligned top edge and fixed gap.
        legend_specs = []
        if group_dot_handles:
            legend_specs.append(dict(
                handles=group_dot_handles,
                labels=group_dot_labels,
                title='Encoder groups',
                ncol=min(len(group_dot_handles), 4),
                framealpha=0.85,
            ))
        if frontier_handles:
            legend_specs.append(dict(
                handles=frontier_handles,
                labels=frontier_labels,
                title='Pareto frontiers',
                framealpha=0.85,
            ))

        ra_handles = _ra_scatter_legend_handles()
        legend_specs.append(dict(
            handles=ra_handles,
            labels=[h.get_label() for h in ra_handles],
            title='Access capability',
            framealpha=0.85,
        ))

        legend_top_y = 0.94
        legend_x = 0.02
        legend_gap = 0.02
        fig.canvas.draw()
        for spec in legend_specs:
            leg = fig.legend(
                spec['handles'], spec['labels'],
                fontsize=7,
                loc='upper left',
                bbox_to_anchor=(legend_x, legend_top_y),
                bbox_transform=fig.transFigure,
                title=spec.get('title'),
                title_fontsize=7,
                ncol=spec.get('ncol', 1),
                framealpha=spec.get('framealpha', 0.85),
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = leg.get_window_extent(renderer=renderer)
            fig_w_px = fig.get_size_inches()[0] * fig.dpi
            legend_x += (bbox.width / fig_w_px) + legend_gap

        # ── Axes ───────────────────────────────────────────────────────────
        ax.set_xlabel('Compression (× smaller than raw, higher = better →)')
        ax.set_ylabel('Avg random access time (ns) — faster at top ↑')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        # Invert y-axis: low (fast) access time floats to the top so the
        # ideal top-right corner = high compression AND fast access.
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.82])
        fname = output_dir / f'compression_vs_random_access_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_selection_time(results, output_dir, colours, source_label, group_encoders=False):
    """Deprecated: selection time is now rendered in the encode plot."""
    return


# ---------------------------------------------------------------------------
# Memory plots
# ---------------------------------------------------------------------------

def _bytes_to_mb(b):
    return b / (1024 * 1024)


def _has_memory_data(results) -> bool:
    """Return True if at least one result has non-zero memory measurement fields."""
    for r in results['results']:
        m = r.get('metrics', {}).get('memory', {})
        if any(m.get(k, 0) > 0 for k in (
            'encodePeakHeapBytes', 'encodeNetHeapDeltaBytes',
            'decodeBulkPeakHeapBytes', 'decodeBulkNetHeapDeltaBytes',
            'decodeRandomPeakHeapBytes', 'decodeStridedPeakHeapBytes',
            'decodeRangePeakHeapBytes',
        )):
            return True
    return False


def plot_memory_encode_decode(results, output_dir, colours, source_label, group_encoders=False):
    """
    Mirrors plot_encode_decode_time but shows heap memory usage (MB).

    Left panel : encode peak heap vs net heap delta (stacked bar so the
                 difference = internal buffers freed before encode returned).
    Right panel: bulk-decode peak heap vs net heap delta (same layout).
    """
    if not _has_memory_data(results):
        return

    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        enc_peak, enc_net = [], []
        dec_peak, dec_net = [], []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                m = row['metrics']['memory']
                enc_peak.append(_bytes_to_mb(m.get('encodePeakHeapBytes', 0)))
                enc_net.append(_bytes_to_mb(m.get('encodeNetHeapDeltaBytes', 0)))
                dec_peak.append(_bytes_to_mb(m.get('decodeBulkPeakHeapBytes', 0)))
                dec_net.append(_bytes_to_mb(m.get('decodeBulkNetHeapDeltaBytes', 0)))
            else:
                enc_peak.append(0); enc_net.append(0)
                dec_peak.append(0); dec_net.append(0)

        h = max(4, len(encoders) * 0.7 + 1.5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, h))
        fig.suptitle(
            f'Heap memory usage — {dataset}  (n={size:,})',
            fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        def _mem_stacked(ax, encoders, peak_vals, net_vals, title):
            """
            Stacked bar: net (retained) portion + transient (peak - net) portion.
            Peak is the high-water mark; net is what remains allocated after the
            call returns.  The difference shows intermediate buffers that were
            freed before the call returned.
            """
            y = np.arange(len(encoders))
            bar_h = 0.6
            net_clipped   = [max(v, 0) for v in net_vals]
            # Transient portion: peak above net (internal working memory freed by return)
            transient     = [max(p - n, 0) for p, n in zip(peak_vals, net_clipped)]
            net_colors    = [colours[e] for e in encoders]
            trans_colors  = [_lighten_color(colours[e], 0.45) for e in encoders]

            ax.barh(y, net_clipped, height=bar_h, color=net_colors, label='Net retained')
            ax.barh(y, transient, left=net_clipped, height=bar_h,
                    color=trans_colors, hatch='///', edgecolor='black',
                    linewidth=0.5, label='Transient (freed by return)')

            x_max = max((v for v in peak_vals if v > 0), default=1)
            for i, (net, trans, peak) in enumerate(zip(net_clipped, transient, peak_vals)):
                if peak > 0:
                    ax.text(peak + x_max * 0.01,
                            y[i],
                            f'{peak:.2f} MB', va='center', fontsize=8)
                else:
                    ax.text(x_max * 0.01, y[i],
                            'N/A', va='center', fontsize=8, color='grey')

            _set_encoder_axis_labels(ax, y, encoders); ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlabel('Heap memory (MB, lower = better)')
            ax.set_title(title)
            base_leg = ax.legend(fontsize=8, loc='lower right')
            ax.add_artist(base_leg)
            ax.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        _mem_stacked(ax1, encoders, enc_peak, enc_net, 'Encode heap memory')
        _mem_stacked(ax2, encoders, dec_peak, dec_net, 'Bulk decode heap memory')

        plt.tight_layout()
        fname = output_dir / f'memory_encode_decode_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close()


def plot_memory_access(results, output_dir, colours, source_label, group_encoders=False):
    """
    Mirrors plot_random_access / plot_range_access but shows heap memory.

    Left panel : peak heap per decodeAt call (random + strided).
                 For most encoders this will be 0 because decodeAt returns
                 std::optional<T> by value (stack-only).  Non-zero values
                 indicate internal scratch-buffer allocations.
    Right panel: peak working heap for a single decodeRange call (MB).
                 This is the output buffer + any internal temporary buffers
                 alive at the moment decodeRange returns.
    """
    if not _has_memory_data(results):
        return

    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        rand_kb, strided_kb, range_mb = [], [], []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                m = row['metrics']['memory']
                # Random/strided: values are small (bytes), display in KB
                rand_kb.append(m.get('decodeRandomPeakHeapBytes', 0) / 1024)
                strided_kb.append(m.get('decodeStridedPeakHeapBytes', 0) / 1024)
                range_mb.append(_bytes_to_mb(m.get('decodeRangePeakHeapBytes', 0)))
            else:
                rand_kb.append(0); strided_kb.append(0); range_mb.append(0)

        h = max(4, len(encoders) * 0.7 + 1.5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, h))
        fig.suptitle(
            f'Decode access heap memory — {dataset}  (n={size:,})',
            fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        # Left: random vs strided side-by-side (grouped, KB scale)
        y   = np.arange(len(encoders))
        bh  = 0.28
        ax1.barh(y - bh/2, rand_kb,    height=bh,
                 color=[colours[e] for e in encoders], label='Random', alpha=0.9)
        ax1.barh(y + bh/2, strided_kb, height=bh,
                 color=[colours[e] for e in encoders], label='Strided',
                 alpha=0.55, hatch='//')
        _set_encoder_axis_labels(ax1, y, encoders); ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlabel('Peak heap per decodeAt call (KB, lower = better)')
        ax1.set_title('Random / strided access heap\n(0 = stack-only decode, expected for most encoders)')

        import matplotlib.patches as mpatches
        rand_patch    = mpatches.Patch(facecolor='grey', alpha=0.9,  label='Random (solid)')
        strided_patch = mpatches.Patch(facecolor='grey', alpha=0.55, hatch='//', label='Strided (hatch)')
        base_leg = ax1.legend(handles=[rand_patch, strided_patch], fontsize=8, loc='lower right')
        ax1.add_artist(base_leg)
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        # Right: range query (MB)
        _hbar(ax2, encoders, range_mb, colours, unit='MB', fmt='.2f', log=False)
        ax2.set_xlabel('Peak heap for one decodeRange call (MB, lower = better)')
        ax2.set_title('Range-access heap memory\n(output buffer + internal allocations alive at return)')
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        plt.tight_layout()
        fname = output_dir / f'memory_access_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close()


def plot_time_memory_paired(results, output_dir, colours, source_label, group_encoders=False):
    """
    Paired plots that put timing and memory on the same figure for direct
    comparison.

    Produces two output files per dataset:
      time_memory_encode_<dataset>.png  — encode time (ms) vs encode peak heap (MB)
      time_memory_decode_<dataset>.png  — bulk decode time (ms) vs decode peak heap (MB)

    Each figure is a 1×2 grid: left = time bar chart, right = memory bar chart,
    with identical encoder ordering and colour mapping so the eye can directly
    compare the two panels.
    """
    if not _has_memory_data(results):
        return

    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)

        enc_ms, dec_ms = [], []
        enc_peak_mb, dec_peak_mb = [], []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                t = row['metrics']['timing']
                m = row['metrics']['memory']
                enc_ms.append(t['encodeTime_ns']     / 1e6)
                dec_ms.append(t['decodeBulkTime_ns'] / 1e6)
                enc_peak_mb.append(_bytes_to_mb(m.get('encodePeakHeapBytes', 0)))
                dec_peak_mb.append(_bytes_to_mb(m.get('decodeBulkPeakHeapBytes', 0)))
            else:
                enc_ms.append(0); dec_ms.append(0)
                enc_peak_mb.append(0); dec_peak_mb.append(0)

        h = max(4, len(encoders) * 0.7 + 1.5)

        # ── Encode paired ────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, h))
        fig.suptitle(
            f'Encode — time vs memory — {dataset}  (n={size:,})',
            fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        _hbar(ax1, encoders, enc_ms,      colours, unit='ms', fmt='.1f', log=True)
        ax1.set_xlabel('Encode time (ms, log scale, lower = better)')
        ax1.set_title('Encode time')
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        _hbar(ax2, encoders, enc_peak_mb, colours, unit='MB', fmt='.2f', log=False)
        ax2.set_xlabel('Encode peak heap (MB, lower = better)')
        ax2.set_title('Encode peak heap memory')
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        plt.tight_layout()
        fname = output_dir / f'time_memory_encode_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close()

        # ── Decode paired ────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, h))
        fig.suptitle(
            f'Bulk decode — time vs memory — {dataset}  (n={size:,})',
            fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        _hbar(ax1, encoders, dec_ms,      colours, unit='ms', fmt='.1f', log=True)
        ax1.set_xlabel('Decode time (ms, log scale, lower = better)')
        ax1.set_title('Bulk decode time')
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        _hbar(ax2, encoders, dec_peak_mb, colours, unit='MB', fmt='.2f', log=False)
        ax2.set_xlabel('Decode peak heap (MB, lower = better)')
        ax2.set_title('Bulk decode peak heap memory')
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        plt.tight_layout()
        fname = output_dir / f'time_memory_decode_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close()


# ---------------------------------------------------------------------------
# Sub-stream stacked variants (SubIntSplitEncoder profiling variants only)
# ---------------------------------------------------------------------------

def _rows_for_dataset(results, encoders, dataset, size):
    rows = [(enc, _row(results, enc, dataset, size)) for enc in encoders]
    return [(enc, row) for enc, row in rows if row]


def _substream_initials(name):
    cleaned = (name or '').strip()
    if not cleaned:
        return '?'
    parts = re.findall(r'[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+', cleaned)
    if parts:
        initials = ''.join(part[0] for part in parts if part)
    else:
        initials = re.sub(r'[^A-Za-z0-9]+', '', cleaned)[:3]
    initials = initials.upper()
    return initials or '?'


def _substream_segment_label(ss):
    return _substream_initials(ss.get('name', '?'))


def _substream_legend_label(ss):
    name = ss.get('name', '?')
    if not name or name == '?':
        return '?'
    initials = _substream_initials(name)
    bit_width = ss.get('bitWidth', '?')
    if bit_width in (None, '', '?'):
        return f'{initials}: {name}'
    return f'{initials}: {name} ({bit_width}b)'


def _substream_legend_handles(prepared, section_cmap):
    seen = set()
    handles = []
    for _, _, streams, _, _ in prepared:
        for si, ss in enumerate(streams):
            name = ss.get('name', '?')
            if not name or name in seen:
                continue
            seen.add(name)
            handles.append(
                mpatches.Patch(
                    facecolor=section_cmap(si % 20),
                    edgecolor='black',
                    label=_substream_legend_label(ss),
                )
            )
    return handles


def _format_metric_value(value, fmt='.3g', unit=''):
    label = f'{value:{fmt}}'
    return f'{label} {unit}'.rstrip()


def _plot_stacked_substream_panel(
    ax,
    rows,
    colours,
    metric_key,
    aggregate_getter,
    title,
    xlabel,
    unit='',
    fmt='.3g',
    scale=1.0,
    log=False,
    section_cmap=None,
    xmin=None,
    xmax=None,
):
    section_cmap = section_cmap or cm.get_cmap('tab20')
    y = np.arange(len(rows))
    bar_h = 0.72

    prepared = []
    for enc_name, row in rows:
        streams = row.get('metrics', {}).get('subStreamMetrics') or []
        if streams:
            segments = [(ss.get(metric_key, 0) or 0) * scale for ss in streams]
            total = sum(segments)
        else:
            segments = None
            total = (aggregate_getter(row) if row else 0) * scale
        prepared.append((enc_name, row, streams, segments, total))

    x_max = max((total for _, _, _, _, total in prepared if total > 0), default=1.0)
    any_positive = False

    for idx, (enc_name, row, streams, segments, total) in enumerate(prepared):
        if segments is not None:
            left = 0.0
            small_label_idx = 0
            for si, (ss, val) in enumerate(zip(streams, segments)):
                if val <= 0:
                    continue
                any_positive = True
                color = section_cmap(si % 20)
                initials = _substream_initials(ss.get('name', '?'))
                ax.barh(idx, val, left=left, height=bar_h, color=color,
                        edgecolor='black', linewidth=0.4)
                if total > 0 and val / total >= 0.14:
                    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    txt_color = 'white' if lum < 0.5 else 'black'
                    ax.text(left + val * 0.5, idx, initials,
                            va='center', ha='center', fontsize=6, color=txt_color,
                            fontweight='bold')
                else:
                    side = -1 if small_label_idx % 2 == 0 else 1
                    level = small_label_idx // 2
                    small_label_idx += 1
                    edge_y = idx - bar_h / 2 if side < 0 else idx + bar_h / 2
                    text_y = idx + side * (bar_h / 2 + 0.18 + level * 0.12)
                    ax.annotate(
                        initials,
                        xy=(left + val * 0.5, edge_y),
                        xytext=(left + val * 0.5, text_y),
                        textcoords='data',
                        ha='center',
                        va='center',
                        fontsize=6,
                        color='black',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='black', alpha=0.85),
                        arrowprops=dict(arrowstyle='-', color='black', lw=0.6, shrinkA=0, shrinkB=0),
                        annotation_clip=False,
                    )
                left += val

            if total > 0:
                ax.text(total + x_max * 0.01, idx, _format_metric_value(total, fmt, unit),
                        va='center', ha='left', fontsize=8, fontweight='bold')
                if not _is_random_access_encoder(enc_name):
                    ax.barh(idx, total, height=bar_h, left=0.0, facecolor='none',
                            edgecolor='black', hatch='///', linewidth=0.8)
            else:
                ax.text(x_max * 0.01, idx, 'N/A', va='center', ha='left',
                        fontsize=8, color='grey')
        else:
            bar = ax.barh(idx, total, height=bar_h, color=[colours[enc_name]])[0]
            if not _is_random_access_encoder(enc_name):
                bar.set_hatch('///')
                bar.set_edgecolor('black')
                bar.set_linewidth(0.6)
            if total > 0:
                any_positive = True
                ax.text(total + x_max * 0.01, idx, _format_metric_value(total, fmt, unit),
                        va='center', ha='left', fontsize=8, fontweight='bold')
            else:
                ax.text(x_max * 0.01, idx, 'N/A', va='center', ha='left',
                        fontsize=8, color='grey')

    _set_encoder_axis_labels(ax, y, [enc for enc, _ in rows])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    if log and any_positive:
        ax.set_xscale('log')
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)

    return _substream_legend_handles(prepared, section_cmap)


def plot_encode_decode_time_substreams(results, output_dir, colours, source_label, group_encoders=False):
    """Stacked sub-stream breakdown for encode and bulk-decode time."""
    if not _has_substream_data(results):
        return

    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        rows = _rows_for_dataset(results, encoders, dataset, size)
        if not any(row.get('metrics', {}).get('subStreamMetrics') for _, row in rows):
            continue

        h = max(4.5, len(rows) * 0.8 + 1.8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, h))
        fig.suptitle(f'Encode / Decode time — sub-stream stacked — {dataset}  (n={size:,})',
                     fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        substream_handles = _plot_stacked_substream_panel(
            ax1, rows, colours,
            metric_key='encodeTime_ns',
            aggregate_getter=lambda row: row['metrics']['timing']['encodeTime_ns'],
            title='Encode time (stacked by sub-stream)',
            xlabel='Time (ms, log scale, lower = better)',
            unit='ms',
            fmt='.1f',
            scale=1e-6,
            log=True,
        )
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        _plot_stacked_substream_panel(
            ax2, rows, colours,
            metric_key='decodeBulkTime_ns',
            aggregate_getter=lambda row: row['metrics']['timing']['decodeBulkTime_ns'],
            title='Bulk decode time (stacked by sub-stream)',
            xlabel='Time (ms, log scale, lower = better)',
            unit='ms',
            fmt='.1f',
            scale=1e-6,
            log=True,
        )
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        if substream_handles:
            fig.legend(
                handles=substream_handles,
                fontsize=7,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.01),
                ncol=2 if len(substream_handles) > 3 else 1,
                framealpha=0.9,
                title='Subencoder key',
                title_fontsize=7,
            )

        plt.tight_layout(rect=[0, 0.08, 1, 0.93])
        safe_ds = dataset.replace('/', '_').replace(' ', '_')
        fname = output_dir / f'encode_decode_time_substreams_{safe_ds}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig)


def plot_random_access_substreams(results, output_dir, colours, source_label, group_encoders=False):
    """Stacked sub-stream breakdown for average random-access time."""
    if not _has_substream_data(results):
        return

    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        rows = _rows_for_dataset(results, encoders, dataset, size)
        if not any(row.get('metrics', {}).get('subStreamMetrics') for _, row in rows):
            continue

        avg_ns, strided_ns = [], []
        for enc, row in rows:
            ra = row['metrics']['randomAccess']
            avg_ns.append(ra.get('averageRandomAccessTime_ns', 0))
            strided_ns.append(ra.get('averageStridedAccessTime_ns', 0))

        h = max(4.5, len(rows) * 0.8 + 1.8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, h))
        fig.suptitle(f'Random / Strided access — sub-stream stacked — {dataset}  (n={size:,})',
                     fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        substream_handles = _plot_stacked_substream_panel(
            ax1, rows, colours,
            metric_key='decodeAtTime_ns',
            aggregate_getter=lambda row: row['metrics']['randomAccess']['averageRandomAccessTime_ns'],
            title='Avg random access (stacked by sub-stream)',
            xlabel='Time (ns, log scale, lower = better)',
            unit='ns',
            fmt='.0f',
            scale=1.0,
            log=True,
            xmin=1.0,  # Avoid log(0) when no data; will show as "N/A" labels instead of tiny bars
        )
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        _hbar(ax2, encoders, strided_ns, colours, unit='ns', fmt='.0f', log=True)
        ax2.set_xlabel('Time (ns, log scale, lower = better)')
        ax2.set_title('Avg strided access (stride from config)')
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        if substream_handles:
            fig.legend(
                handles=substream_handles,
                fontsize=7,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.01),
                ncol=2 if len(substream_handles) > 3 else 1,
                framealpha=0.9,
                title='Subencoder key',
                title_fontsize=7,
            )

        plt.tight_layout(rect=[0, 0.08, 1, 0.93])
        safe_ds = dataset.replace('/', '_').replace(' ', '_')
        fname = output_dir / f'random_access_substreams_{safe_ds}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig)


def plot_range_access_substreams(results, output_dir, colours, source_label, group_encoders=False):
    """Stacked sub-stream breakdown for average range-query latency."""
    if not _has_substream_data(results):
        return

    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        rows = _rows_for_dataset(results, encoders, dataset, size)
        if not any(row.get('metrics', {}).get('subStreamMetrics') for _, row in rows):
            continue

        avg_ms, range_sz, tp_vals = [], [], []
        for enc, row in rows:
            ra = row['metrics']['randomAccess']
            avg_ns = ra.get('averageRangeAccessTime_ns', 0)
            r_size = ra.get('averageRangeSize', 0)
            avg_ms.append(avg_ns / 1e6)
            range_sz.append(r_size)
            tp_vals.append((r_size / (avg_ns / 1e9)) / 1e6 if avg_ns > 0 else 0)

        rng_label = f'{int(max(range_sz)):,}' if range_sz else '?'
        h = max(4.5, len(rows) * 0.8 + 1.8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, h))
        fig.suptitle(
            f'Range access (avg range = {rng_label} elems) — sub-stream stacked — {dataset}  (n={size:,})',
            fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        substream_handles = _plot_stacked_substream_panel(
            ax1, rows, colours,
            metric_key='decodeRangeTime_ns',
            aggregate_getter=lambda row: row['metrics']['randomAccess']['averageRangeAccessTime_ns'],
            title='Avg range-query latency (stacked by sub-stream)',
            xlabel='Time (ms, log scale, lower = better)',
            unit='ms',
            fmt='.2f',
            scale=1e-6,
            log=True,
        )
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        _hbar(ax2, encoders, tp_vals, colours, unit='M e/s', fmt='.1f', log=True)
        ax2.set_xlabel('Throughput (M elem/s, log scale, higher = better)')
        ax2.set_title('Range-query decode throughput')
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        if substream_handles:
            fig.legend(
                handles=substream_handles,
                fontsize=7,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.01),
                ncol=2 if len(substream_handles) > 3 else 1,
                framealpha=0.9,
                title='Subencoder key',
                title_fontsize=7,
            )

        plt.tight_layout(rect=[0, 0.08, 1, 0.93])
        safe_ds = dataset.replace('/', '_').replace(' ', '_')
        fname = output_dir / f'range_access_substreams_{safe_ds}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig)


def plot_compression_size_substreams(results, output_dir, colours, source_label, group_encoders=False):
    """Stacked sub-stream breakdown for total compressed size."""
    if not _has_substream_data(results):
        return

    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        rows = _rows_for_dataset(results, encoders, dataset, size)
        if not any(row.get('metrics', {}).get('subStreamMetrics') for _, row in rows):
            continue

        bpe_vals = []
        for enc, row in rows:
            bpe_vals.append(row['metrics']['memory']['bitsPerElement'])

        h = max(4.5, len(rows) * 0.8 + 1.8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, h))
        fig.suptitle(f'Compression size — sub-stream stacked — {dataset}  (n={size:,})',
                     fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        substream_handles = _plot_stacked_substream_panel(
            ax1, rows, colours,
            metric_key='encodedBytes',
            aggregate_getter=lambda row: row['metrics']['memory']['encodedSize'],
            title='Total compressed size (stacked by sub-stream)',
            xlabel='Compressed size (MB, log scale, lower = better)',
            unit='MB',
            fmt='.2f',
            scale=1.0 / (1024 * 1024),
            log=True,
        )
        ax1.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        _hbar(ax2, encoders, bpe_vals, colours, unit='bits', fmt='.2f', log=False)
        ax2.set_xlabel('Bits per element (lower = better)')
        ax2.set_title('Bits per element')
        ax2.legend(handles=_ra_bar_legend_handles(), fontsize=8, loc='upper right', title='Access capability')

        if substream_handles:
            fig.legend(
                handles=substream_handles,
                fontsize=7,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.01),
                ncol=2 if len(substream_handles) > 3 else 1,
                framealpha=0.9,
                title='Subencoder key',
                title_fontsize=7,
            )

        plt.tight_layout(rect=[0, 0.08, 1, 0.93])
        safe_ds = dataset.replace('/', '_').replace(' ', '_')
        fname = output_dir / f'compression_size_substreams_{safe_ds}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Sub-stream breakdown (SubIntSplitEncoder profiling variants only)
# ---------------------------------------------------------------------------

def _has_substream_data(results) -> bool:
    return any(
        r.get('metrics', {}).get('subStreamMetrics')
        for r in results['results']
    )


def plot_substream_breakdown(results, output_dir, colours, source_label, group_encoders=False):
    """
    Stacked horizontal bar charts showing how each sub-stream codec contributes
    to compressed size, encode time, bulk-decode time, random-access decode time,
    and range-decode time.  Only emits plots for datasets that have at least one
    encoder with subStreamMetrics present (i.e. a SubIntSplitEncoder profiling
    variant was registered in the benchmark run).
    """
    if not _has_substream_data(results):
        return

    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = _sorted_encoders(results, group_encoders=group_encoders)

    metrics_cfg = [
        ('encodedBytes',       'Compressed size (bytes)',        'Compression per sub-stream',          False),
        ('encodeTime_ns',      'Encode time (ns)',               'Encode time per sub-stream',           True),
        ('decodeBulkTime_ns',  'Bulk decode time (ns)',          'Bulk decode time per sub-stream',      True),
        ('decodeAtTime_ns',    'Avg per-decodeAt time (ns)',     'Random-access time per sub-stream',    True),
        ('decodeRangeTime_ns', 'Avg per-decodeRange time (ns)', 'Range-decode time per sub-stream',     True),
    ]
    section_cmap = cm.get_cmap('tab10')

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)

        # Keep only encoders that have subStreamMetrics for this dataset/size
        rows = [(enc, _row(results, enc, dataset, size)) for enc in encoders]
        rows = [(enc, r) for enc, r in rows
                if r and r.get('metrics', {}).get('subStreamMetrics')]
        if not rows:
            continue

        n_panels = len(metrics_cfg)
        h = max(4, len(rows) * 0.9 + 1.5)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, h))
        fig.suptitle(f'Sub-stream breakdown — {dataset}  (n={size:,})', fontsize=13, y=1.04)
        _annotate_source(fig, source_label)

        for ax, (metric_key, xlabel, title, use_log) in zip(axes, metrics_cfg):
            y_pos = np.arange(len(rows))
            bar_h = 0.6
            any_nonzero = False

            for idx, (enc_name, row) in enumerate(rows):
                streams = row['metrics']['subStreamMetrics']
                total = sum(ss.get(metric_key, 0) or 0 for ss in streams)
                left = 0.0
                for si, ss in enumerate(streams):
                    val = ss.get(metric_key, 0) or 0
                    if val <= 0:
                        continue
                    any_nonzero = True
                    color = section_cmap(si % 10)
                    ax.barh(idx, val, left=left, height=bar_h, color=color,
                            edgecolor='black', linewidth=0.4)
                    if total > 0 and val / total > 0.08:
                        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                        txt_color = 'white' if lum < 0.5 else 'black'
                        label = f"{ss.get('name', '?')}\n({ss.get('bitWidth', '?')}b)"
                        ax.text(left + val * 0.5, idx, label, va='center', ha='center',
                                fontsize=6, color=txt_color, fontweight='bold')
                    left += val

            ax.set_yticks(y_pos)
            ax.set_yticklabels([enc for enc, _ in rows], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            if use_log and any_nonzero:
                ax.set_xscale('log')

        plt.tight_layout()
        safe_ds = dataset.replace('/', '_').replace(' ', '_')
        fname = output_dir / f'substream_breakdown_{safe_ds}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot encoding benchmark results')
    parser.add_argument('input',  type=Path, help='Input JSON benchmark results file')
    parser.add_argument('-o', '--output', type=Path, default=Path('plots'),
                        help='Output directory for plots (default: plots/)')
    parser.add_argument('--group-encoders', action='store_true',
                        help='Group encoders by scheme family (AutoSubIntSplit, Raw, SubInt, TriSplit, OpenZL, VarInt)')
    args = parser.parse_args()

    print(f"Loading results from: {args.input}")
    results = load_results(args.input)

    source_label = f"Source: {args.input.name}"

    args.output.mkdir(exist_ok=True, parents=True)

    # Stable colour map across all charts
    encoders = _sorted_encoders(results, group_encoders=args.group_encoders)
    if args.group_encoders:
        colours = _grouped_encoder_colours(encoders, _default_group_rules())
    else:
        colours = _encoder_colours(encoders)

    print("\nGenerating plots...")
    # ── Existing timing / compression plots ─────────────────────────────
    plot_compression(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    plot_compression_overhead(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    plot_encode_decode_time(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    plot_random_access(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    plot_range_access(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    plot_throughput_summary(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    plot_compression_vs_random_access(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    # selection time (if present) is shown inside the encode plot

    # ── AutoSubIntSplit forced-split sweep ──────────────────────────────
    if _has_autosubintsplit_forced_data(results):
        print("\nAutoSubIntSplitN variants detected — generating forced-split sweep plots...")
        plot_autosubintsplit_forced_split_sweep(
            results, args.output, colours, source_label, group_encoders=args.group_encoders)
    else:
        print("\nNo numbered AutoSubIntSplit variants found — skipping forced-split sweep plots.")

    # ── Sub-stream stacked variants (skipped if no profiling variants present) ──
    if _has_substream_data(results):
        print("\nSub-stream profiling data detected — generating stacked variant plots...")
        plot_encode_decode_time_substreams(results, args.output, colours, source_label, group_encoders=args.group_encoders)
        plot_random_access_substreams(results, args.output, colours, source_label, group_encoders=args.group_encoders)
        plot_range_access_substreams(results, args.output, colours, source_label, group_encoders=args.group_encoders)
        plot_compression_size_substreams(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    else:
        print("\nNo sub-stream profiling data found — skipping sub-stream variant plots.")
        print("  (Register a SubIntSplitEncoder<T, true> or SubIntSplitAutoEncoder<T, true> variant to collect it.)")

    # ── Memory plots (skipped silently if no memory data present) ────────
    if _has_memory_data(results):
        print("\nMemory data detected — generating memory plots...")
        plot_memory_encode_decode(results, args.output, colours, source_label, group_encoders=args.group_encoders)
        plot_memory_access(results, args.output, colours, source_label, group_encoders=args.group_encoders)
        plot_time_memory_paired(results, args.output, colours, source_label, group_encoders=args.group_encoders)
    else:
        print("\nNo memory data found in results — skipping memory plots.")
        print("  (Re-run benchmarks with BenchmarkConfig::measureMemory = true to collect it.)")

    print(f"\n✓ All plots saved to: {args.output}")
    print(f"\nBenchmark Summary:")
    print(f"  Total benchmarks : {len(results['results'])}")
    print(f"  Total duration   : {results['metadata']['totalDuration_s']:.2f} s")
    print(f"  Start            : {results['metadata']['startTime']}")
    print(f"  End              : {results['metadata']['endTime']}")


if __name__ == '__main__':
    main()
