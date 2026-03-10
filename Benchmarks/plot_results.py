#!/usr/bin/env python3
"""
Plot benchmark results from JSON output.

Produces one horizontal-bar chart per metric so encoders are easy to
compare side-by-side even when there is only a single dataset/size.
All log-scaled axes guard against zero/negative values by falling back
to a linear scale with a warning.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_compression(results, output_dir, colours):
    """Inverted compression ratio (higher = better) and bits-per-element."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))

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
        fig.suptitle(f'Compression — {dataset}  (n={size:,})', fontsize=13)

        _hbar(ax1, encoders, inv_ratios, colours, unit='×', fmt='.3f', log=False)
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
                    label='No compression (1×)')
        ax1.set_xlabel('Compression ratio (higher = better)')
        ax1.set_title('Inverted compression ratio')
        ax1.legend(fontsize=8)

        _hbar(ax2, encoders, bpe_vals, colours, unit='bits', fmt='.2f', log=False)
        ax2.axvline(x=32, color='red', linestyle='--', linewidth=1, alpha=0.6,
                    label='Raw int32 (32 bits)')
        ax2.set_xlabel('Bits per element (lower = better)')
        ax2.set_title('Bits per element')
        ax2.legend(fontsize=8)

        plt.tight_layout()
        fname = output_dir / f'compression_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_encode_decode_time(results, output_dir, colours):
    """Encode + bulk decode time in ms, with throughput (M elem/s) on a twin axis."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))

    for dataset in datasets:
        size = _pick_size_for_dataset(results, dataset)
        enc_ms, enc_tp = [], []
        dec_ms, dec_tp = [], []
        for enc in encoders:
            row = _row(results, enc, dataset, size)
            if row:
                t  = row['metrics']['timing']
                enc_ms.append(t['encodeTime_ns']     / 1e6)
                dec_ms.append(t['decodeBulkTime_ns'] / 1e6)
                enc_tp.append(t['encodeElementsPerSecond']      / 1e6)
                dec_tp.append(t['decodeBulkElementsPerSecond']  / 1e6)
            else:
                enc_ms.append(0); enc_tp.append(0)
                dec_ms.append(0); dec_tp.append(0)

        h = max(4, len(encoders) * 0.7 + 1.5)
        fig, axes = plt.subplots(1, 2, figsize=(14, h))
        fig.suptitle(f'Encode / Decode time — {dataset}  (n={size:,})', fontsize=13)

        for ax, times, tps, title, tp_label in [
            (axes[0], enc_ms, enc_tp, 'Encode time', 'Encode throughput (M elem/s)'),
            (axes[1], dec_ms, dec_tp, 'Bulk decode time', 'Decode throughput (M elem/s)'),
        ]:
            _hbar(ax, encoders, times, colours, unit='ms', fmt='.1f', log=True)
            ax.set_xlabel('Time (ms, log scale, lower = better)')
            ax.set_title(title)

            # Twin axis for throughput
            ax2 = ax.twiny()
            y   = np.arange(len(encoders))
            ax2.plot(tps, y, marker='D', linestyle='none',
                     color='black', markersize=5, label=tp_label)
            ax2.set_xlabel(tp_label, fontsize=8)
            ax2.tick_params(axis='x', labelsize=7)
            _safe_log_scale(ax2, 'x')
            ax2.legend(fontsize=7, loc='lower right')

        plt.tight_layout()
        fname = output_dir / f'encode_decode_time_{dataset.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close()


def plot_random_access(results, output_dir, colours):
    """Average, min, and max single-element random access time (ns)."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))

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
        fig.suptitle(f'Random / Strided access — {dataset}  (n={size:,})', fontsize=13)

        # Left: average random access with min/max error bars
        y      = np.arange(len(encoders))
        bar_h  = 0.6
        valid  = [(a, lo, hi) for a, lo, hi in zip(avg_ns, min_ns, max_ns) if a > 0]

        bars = ax1.barh(y, avg_ns, height=bar_h,
                        color=[colours[e] for e in encoders])
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


def plot_range_access(results, output_dir, colours):
    """Average range-query time (ms) and implied throughput (M elem/s)."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))

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
            fontsize=13)

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


def plot_throughput_summary(results, output_dir, colours):
    """Single summary chart: encode + decode throughput side-by-side."""
    size     = _pick_size(results)
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))

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
        fig.suptitle(f'Throughput summary — {dataset}  (n={size:,})', fontsize=13)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot encoding benchmark results')
    parser.add_argument('input',  type=Path, help='Input JSON benchmark results file')
    parser.add_argument('-o', '--output', type=Path, default=Path('plots'),
                        help='Output directory for plots (default: plots/)')
    args = parser.parse_args()

    print(f"Loading results from: {args.input}")
    results = load_results(args.input)

    args.output.mkdir(exist_ok=True, parents=True)

    # Stable colour map across all charts
    encoders = sorted(set(r['encoderName'] for r in results['results']))
    colours  = _encoder_colours(encoders)

    print("\nGenerating plots...")
    plot_compression(results, args.output, colours)
    plot_encode_decode_time(results, args.output, colours)
    plot_random_access(results, args.output, colours)
    plot_range_access(results, args.output, colours)
    plot_throughput_summary(results, args.output, colours)

    print(f"\n✓ All plots saved to: {args.output}")
    print(f"\nBenchmark Summary:")
    print(f"  Total benchmarks : {len(results['results'])}")
    print(f"  Total duration   : {results['metadata']['totalDuration_s']:.2f} s")
    print(f"  Start            : {results['metadata']['startTime']}")
    print(f"  End              : {results['metadata']['endTime']}")


if __name__ == '__main__':
    main()
