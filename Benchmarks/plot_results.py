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
            sort_key=lambda n: (n,)
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
    group_count = max(len(grouped), 1)
    cmap = cm.get_cmap('tab10', group_count)
    colours = {}
    for idx, (_, encs) in enumerate(grouped):
        group_name = grouped[idx][0]
        if group_name == 'VarInt':
            base = (1.0, 0.55, 0.0, 1.0)  # orange
        else:
            base = cmap(idx)
        h, l, s = colorsys.rgb_to_hls(base[0], base[1], base[2])
        count = max(len(encs), 1)
        for j, enc in enumerate(encs):
            # Vary lightness within group for a gentle gradient
            if count == 1:
                l_j = l
                s_j = s
            else:
                spread = 0.6
                l_min = max(0.08, l - spread / 2)
                l_max = min(0.94, l + spread / 2)
                l_j = l_min + (l_max - l_min) * (j / (count - 1))
                s_min = max(0.25, s * 0.7)
                s_max = min(1.0, s * 1.25)
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


def _safe_log_scale(ax, axis='x'):
    """Switch to log scale only when all visible data are positive."""
    setter = ax.set_xscale if axis == 'x' else ax.set_yscale
    getter = ax.get_xlim   if axis == 'x' else ax.get_ylim
    lo, hi = getter()
    if lo > 0 and hi > 0:
        setter('log')


def _hbar(ax, encoders, values, colours, unit='', fmt='.3g', log=True):
    """
    Draw a horizontal bar chart with one bar per encoder.
    Bars with value == 0 are drawn as a thin stub and labelled 'N/A'.
    """
    y = np.arange(len(encoders))
    bar_h = 0.6
    bars = ax.barh(y, values, height=bar_h,
                   color=[colours[e] for e in encoders])

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

    ax.set_yticks(y)
    ax.set_yticklabels(encoders)
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
        ax1.legend(fontsize=8)

        _hbar(ax2, encoders, bpe_vals, colours, unit='bits', fmt='.2f', log=False)
        ax2.axvline(x=64, color='red', linestyle='--', linewidth=1, alpha=0.6,
                    label='Raw int64 (64 bits)')
        ax2.set_xlabel('Bits per element (lower = better)')
        ax2.set_title('Bits per element')
        ax2.legend(fontsize=8)

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
        ax.legend(fontsize=8)

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
            ax.barh(y, total_vals, height=bar_h, color=[colours[e] for e in encoders], label='Encode')
            for idx, val in enumerate(total_vals):
                if val > 0:
                    if val >= min_label_width:
                        enc_text_color = 'black' if _is_light_color(colours[encoders[idx]]) else 'white'
                        ax.text(val * 0.5, y[idx], f'{val:.1f} ms', va='center', ha='center',
                                fontsize=7, color=enc_text_color, fontweight='bold')
                    else:
                        ax.text(val + label_x_offset, y[idx], f'{val:.1f} ms', va='center', ha='left',
                                fontsize=7, color='black', fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(encoders)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        if any(v > 0 for v in total_vals):
            ax.set_xscale('log')
        ax.set_xlabel('Time (ms, log scale, lower = better)')
        ax.set_title('Encode time')
        if has_selection:
            ax.legend(fontsize=8, loc='lower right')

        # Decode plot (unchanged)
        ax = axes[1]
        _hbar(ax, encoders, dec_ms, colours, unit='ms', fmt='.1f', log=True)
        ax.set_xlabel('Time (ms, log scale, lower = better)')
        ax.set_title('Bulk decode time')

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

        ax1.set_yticks(y); ax1.set_yticklabels(encoders); ax1.invert_yaxis()
        ax1.set_xlabel('Time (ns, log scale, lower = better)')
        ax1.set_title('Avg random access (bars)')
        ax1.grid(axis='x', alpha=0.3)
        if any(v > 0 for v in avg_ns):
            ax1.set_xscale('log')

        # Right: average strided access
        _hbar(ax2, encoders, strided_ns, colours, unit='ns', fmt='.0f', log=True)
        ax2.set_xlabel('Time (ns, log scale, lower = better)')
        ax2.set_title('Avg strided access (stride from config)')

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

        _hbar(ax2, encoders, tp_vals, colours, unit='M e/s', fmt='.1f', log=True)
        ax2.set_xlabel('Throughput (M elem/s, log scale, higher = better)')
        ax2.set_title('Range-query decode throughput')

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

        ax.barh(y - bh/2, enc_tp, height=bh,
                color=[colours[e] for e in encoders], label='Encode', alpha=0.85)
        ax.barh(y + bh/2, dec_tp, height=bh,
                color=[colours[e] for e in encoders], label='Decode (bulk)',
                alpha=0.55, hatch='//')

        ax.set_yticks(y)
        ax.set_yticklabels(encoders)
        ax.invert_yaxis()
        ax.set_xlabel('Throughput (M elements/sec, log scale, higher = better)')
        ax.grid(axis='x', alpha=0.3)
        if any(v > 0 for v in enc_tp + dec_tp):
            ax.set_xscale('log')

        # Custom legend: colour patches for each encoder, then encode/decode style
        import matplotlib.patches as mpatches
        enc_patch  = mpatches.Patch(facecolor='grey', alpha=0.85, label='Encode (solid)')
        dec_patch  = mpatches.Patch(facecolor='grey', alpha=0.55, hatch='//', label='Decode (hatch)')
        handles    = [enc_patch, dec_patch]
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
            ax.scatter(ratio, ra, color=colours.get(enc, 'tab:blue'), s=40, zorder=3)

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

        legend_y = 0.94
        if group_dot_handles:
            fig.legend(group_dot_handles, group_dot_labels, fontsize=7,
                       loc='upper center', bbox_to_anchor=(0.5, legend_y),
                       ncol=min(len(group_dot_handles), 4),
                       title='Encoder groups')
            legend_y -= 0.05

        # Combine global frontier + per-group frontier entries
        frontier_handles = []
        frontier_labels  = []
        if pareto_handle is not None:
            frontier_handles.append(pareto_handle)
            frontier_labels.append('Global Pareto frontier')
        if group_frontier_handles:
            frontier_handles += group_frontier_handles
            frontier_labels  += [h.get_label() for h in group_frontier_handles]
        if frontier_handles:
            fig.legend(frontier_handles, frontier_labels, fontsize=7,
                       loc='lower left', bbox_to_anchor=(0.08, 0.08),
                       title='Pareto frontiers', title_fontsize=7,
                       framealpha=0.85)

        # ── Axes ───────────────────────────────────────────────────────────
        ax.set_xlabel('Compression (× smaller than raw, higher = better →)')
        ax.set_ylabel('Avg random access time (ns) — faster at top ↑')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        # Invert y-axis: low (fast) access time floats to the top so the
        # ideal top-right corner = high compression AND fast access.
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.88])
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

            ax.set_yticks(y); ax.set_yticklabels(encoders); ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlabel('Heap memory (MB, lower = better)')
            ax.set_title(title)
            ax.legend(fontsize=8, loc='lower right')

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
        ax1.set_yticks(y); ax1.set_yticklabels(encoders); ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlabel('Peak heap per decodeAt call (KB, lower = better)')
        ax1.set_title('Random / strided access heap\n(0 = stack-only decode, expected for most encoders)')

        import matplotlib.patches as mpatches
        rand_patch    = mpatches.Patch(facecolor='grey', alpha=0.9,  label='Random (solid)')
        strided_patch = mpatches.Patch(facecolor='grey', alpha=0.55, hatch='//', label='Strided (hatch)')
        ax1.legend(handles=[rand_patch, strided_patch], fontsize=8, loc='lower right')

        # Right: range query (MB)
        _hbar(ax2, encoders, range_mb, colours, unit='MB', fmt='.2f', log=False)
        ax2.set_xlabel('Peak heap for one decodeRange call (MB, lower = better)')
        ax2.set_title('Range-access heap memory\n(output buffer + internal allocations alive at return)')

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

        _hbar(ax2, encoders, enc_peak_mb, colours, unit='MB', fmt='.2f', log=False)
        ax2.set_xlabel('Encode peak heap (MB, lower = better)')
        ax2.set_title('Encode peak heap memory')

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

        _hbar(ax2, encoders, dec_peak_mb, colours, unit='MB', fmt='.2f', log=False)
        ax2.set_xlabel('Decode peak heap (MB, lower = better)')
        ax2.set_title('Bulk decode peak heap memory')

        plt.tight_layout()
        fname = output_dir / f'time_memory_decode_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close()


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
