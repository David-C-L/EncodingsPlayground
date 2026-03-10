#!/usr/bin/env python3
"""
Visualization script for vector encoding benchmarks.
Generates plots comparing compression ratios, throughput, latency, and reconstruction errors
across different encoders, dimensions, and data generators.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results(filename='vector_benchmark_results.json'):
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['benchmarks']

def plot_compression_by_dimension(results, output_dir='plots/vector'):
    """Plot compression ratios grouped by dimension."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group by generator type
    generators = sorted(set(r['generator'] for r in results))
    dimensions = sorted(set(r['dimension'] for r in results))
    encoders = sorted(set(r['encoder'] for r in results))
    
    # Use largest vector count for comparison
    max_count = max(r['vector_count'] for r in results)
    
    for gen in generators:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter results for this generator and max vector count
        filtered = [r for r in results 
                   if r['generator'] == gen and r['vector_count'] == max_count]
        
        # Group by encoder
        x_pos = np.arange(len(dimensions))
        width = 0.2
        
        for i, encoder in enumerate(encoders):
            ratios = []
            errors = []
            
            for dim in dimensions:
                matching = [r for r in filtered 
                           if r['encoder'] == encoder and r['dimension'] == dim]
                if matching:
                    ratios.append(matching[0]['compression_ratio']['mean'])
                    errors.append(matching[0]['compression_ratio']['stddev'])
                else:
                    ratios.append(0)
                    errors.append(0)
            
            ax.bar(x_pos + i * width, ratios, width, 
                   label=encoder, yerr=errors, capsize=3)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Compression Ratio')
        ax.set_title(f'Compression Ratio by Dimension - {gen} ({max_count:,} vectors)')
        ax.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
        ax.set_xticklabels(dimensions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/compression_by_dimension_{gen}.png', dpi=300)
        plt.close()

def plot_compression_by_count(results, output_dir='plots/vector'):
    """Plot compression ratios vs vector count."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generators = sorted(set(r['generator'] for r in results))
    dimensions = sorted(set(r['dimension'] for r in results))
    encoders = sorted(set(r['encoder'] for r in results))
    
    for dim in dimensions:
        for gen in generators:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            filtered = [r for r in results 
                       if r['dimension'] == dim and r['generator'] == gen]
            
            for encoder in encoders:
                enc_results = [r for r in filtered if r['encoder'] == encoder]
                enc_results.sort(key=lambda x: x['vector_count'])
                
                counts = [r['vector_count'] for r in enc_results]
                ratios = [r['compression_ratio']['mean'] for r in enc_results]
                errors = [r['compression_ratio']['stddev'] for r in enc_results]
                
                ax.errorbar(counts, ratios, yerr=errors, marker='o', 
                           label=encoder, capsize=3, linewidth=2)
            
            ax.set_xlabel('Vector Count')
            ax.set_ylabel('Compression Ratio')
            ax.set_title(f'Compression Ratio Scaling - D={dim}, {gen}')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/compression_scaling_D{dim}_{gen}.png', dpi=300)
            plt.close()

def plot_throughput_comparison(results, output_dir='plots/vector'):
    """Plot encode and decode throughput."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dimensions = sorted(set(r['dimension'] for r in results))
    encoders = sorted(set(r['encoder'] for r in results))
    
    # Use largest vector count
    max_count = max(r['vector_count'] for r in results)
    
    # Plot for Unit generator (representative)
    filtered = [r for r in results 
               if r['generator'] == 'Unit' and r['vector_count'] == max_count]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x_pos = np.arange(len(dimensions))
    width = 0.2
    
    # Encode throughput
    for i, encoder in enumerate(encoders):
        throughputs = []
        errors = []
        
        for dim in dimensions:
            matching = [r for r in filtered 
                       if r['encoder'] == encoder and r['dimension'] == dim]
            if matching:
                throughputs.append(matching[0]['encode_throughput_mbps']['mean'])
                errors.append(matching[0]['encode_throughput_mbps']['stddev'])
            else:
                throughputs.append(0)
                errors.append(0)
        
        ax1.bar(x_pos + i * width, throughputs, width, 
               label=encoder, yerr=errors, capsize=3)
    
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Throughput (MB/s)')
    ax1.set_title(f'Encode Throughput - Unit Vectors ({max_count:,})')
    ax1.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
    ax1.set_xticklabels(dimensions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Decode throughput
    for i, encoder in enumerate(encoders):
        throughputs = []
        errors = []
        
        for dim in dimensions:
            matching = [r for r in filtered 
                       if r['encoder'] == encoder and r['dimension'] == dim]
            if matching:
                throughputs.append(matching[0]['decode_throughput_mbps']['mean'])
                errors.append(matching[0]['decode_throughput_mbps']['stddev'])
            else:
                throughputs.append(0)
                errors.append(0)
        
        ax2.bar(x_pos + i * width, throughputs, width, 
               label=encoder, yerr=errors, capsize=3)
    
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Throughput (MB/s)')
    ax2.set_title(f'Decode Throughput - Unit Vectors ({max_count:,})')
    ax2.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
    ax2.set_xticklabels(dimensions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_comparison.png', dpi=300)
    plt.close()

def plot_latency_comparison(results, output_dir='plots/vector'):
    """Plot encode and decode latency."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dimensions = sorted(set(r['dimension'] for r in results))
    encoders = sorted(set(r['encoder'] for r in results))
    vector_counts = sorted(set(r['vector_count'] for r in results))
    
    for count in vector_counts:
        filtered = [r for r in results 
                   if r['generator'] == 'Unit' and r['vector_count'] == count]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x_pos = np.arange(len(dimensions))
        width = 0.2
        
        # Encode latency
        for i, encoder in enumerate(encoders):
            latencies = []
            errors = []
            
            for dim in dimensions:
                matching = [r for r in filtered 
                           if r['encoder'] == encoder and r['dimension'] == dim]
                if matching:
                    latencies.append(matching[0]['encode_latency_ms']['mean'])
                    errors.append(matching[0]['encode_latency_ms']['stddev'])
                else:
                    latencies.append(0)
                    errors.append(0)
            
            ax1.bar(x_pos + i * width, latencies, width, 
                   label=encoder, yerr=errors, capsize=3)
        
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title(f'Encode Latency - Unit Vectors ({count:,})')
        ax1.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
        ax1.set_xticklabels(dimensions)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Decode latency
        for i, encoder in enumerate(encoders):
            latencies = []
            errors = []
            
            for dim in dimensions:
                matching = [r for r in filtered 
                           if r['encoder'] == encoder and r['dimension'] == dim]
                if matching:
                    latencies.append(matching[0]['decode_latency_ms']['mean'])
                    errors.append(matching[0]['decode_latency_ms']['stddev'])
                else:
                    latencies.append(0)
                    errors.append(0)
            
            ax2.bar(x_pos + i * width, latencies, width, 
                   label=encoder, yerr=errors, capsize=3)
        
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title(f'Decode Latency - Unit Vectors ({count:,})')
        ax2.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
        ax2.set_xticklabels(dimensions)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/latency_comparison_{count}.png', dpi=300)
        plt.close()

def plot_error_metrics(results, output_dir='plots/vector'):
    """Plot reconstruction error metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dimensions = sorted(set(r['dimension'] for r in results))
    encoders = sorted(set(r['encoder'] for r in results))
    generators = sorted(set(r['generator'] for r in results))
    
    # Use largest vector count
    max_count = max(r['vector_count'] for r in results)
    
    for gen in generators:
        filtered = [r for r in results 
                   if r['generator'] == gen and r['vector_count'] == max_count]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x_pos = np.arange(len(dimensions))
        width = 0.2
        
        # Mean Absolute Error
        for i, encoder in enumerate(encoders):
            errors = [r['errors']['mean_absolute_error'] for r in filtered
                     if r['encoder'] == encoder and r['dimension'] in dimensions]
            errors_sorted = [next((r['errors']['mean_absolute_error'] for r in filtered
                                  if r['encoder'] == encoder and r['dimension'] == dim), 0)
                           for dim in dimensions]
            ax1.bar(x_pos + i * width, errors_sorted, width, label=encoder)
        
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title(f'Mean Absolute Error - {gen}')
        ax1.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
        ax1.set_xticklabels(dimensions)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Mean Euclidean Distance
        for i, encoder in enumerate(encoders):
            distances = [next((r['errors']['mean_euclidean_distance'] for r in filtered
                             if r['encoder'] == encoder and r['dimension'] == dim), 0)
                        for dim in dimensions]
            ax2.bar(x_pos + i * width, distances, width, label=encoder)
        
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Mean Euclidean Distance')
        ax2.set_title(f'Mean Euclidean Distance - {gen}')
        ax2.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
        ax2.set_xticklabels(dimensions)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Mean Cosine Similarity
        for i, encoder in enumerate(encoders):
            similarities = [next((r['errors']['mean_cosine_similarity'] for r in filtered
                                if r['encoder'] == encoder and r['dimension'] == dim), 0)
                           for dim in dimensions]
            ax3.bar(x_pos + i * width, similarities, width, label=encoder)
        
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Mean Cosine Similarity')
        ax3.set_title(f'Mean Cosine Similarity - {gen}')
        ax3.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
        ax3.set_xticklabels(dimensions)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Mean Angle Error (in degrees)
        for i, encoder in enumerate(encoders):
            angles = [next((r['errors']['mean_angle_error_rad'] * 180 / np.pi for r in filtered
                          if r['encoder'] == encoder and r['dimension'] == dim), 0)
                     for dim in dimensions]
            ax4.bar(x_pos + i * width, angles, width, label=encoder)
        
        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Mean Angle Error (degrees)')
        ax4.set_title(f'Mean Angle Error - {gen}')
        ax4.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
        ax4.set_xticklabels(dimensions)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_metrics_{gen}.png', dpi=300)
        plt.close()

def plot_compression_vs_error_tradeoff(results, output_dir='plots/vector'):
    """Plot compression ratio vs reconstruction error tradeoff."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dimensions = sorted(set(r['dimension'] for r in results))
    generators = sorted(set(r['generator'] for r in results))
    
    # Use largest vector count
    max_count = max(r['vector_count'] for r in results)
    
    for dim in dimensions:
        for gen in generators:
            filtered = [r for r in results 
                       if r['dimension'] == dim and r['generator'] == gen 
                       and r['vector_count'] == max_count]
            
            if not filtered:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            encoders = sorted(set(r['encoder'] for r in filtered))
            colors = cm.Set2(np.linspace(0, 1, len(encoders)))
            
            for encoder, color in zip(encoders, colors):
                enc_data = [r for r in filtered if r['encoder'] == encoder]
                
                for r in enc_data:
                    compression = r['compression_ratio']['mean']
                    error = r['errors']['mean_euclidean_distance']
                    
                    ax.scatter(compression, error, s=200, c=[color], 
                             label=encoder, alpha=0.7, edgecolors='black', linewidth=1.5)
                    
                    # Add encoder name as annotation
                    ax.annotate(encoder.replace('SphericalEncoder', 'Sph').replace('_Normalized', '_N'),
                              (compression, error), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
            
            ax.set_xlabel('Compression Ratio')
            ax.set_ylabel('Mean Euclidean Distance Error')
            ax.set_title(f'Compression vs Error Tradeoff - D={dim}, {gen} ({max_count:,} vectors)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Remove duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/tradeoff_D{dim}_{gen}.png', dpi=300)
            plt.close()

def plot_heatmap_best_encoder(results, output_dir='plots/vector'):
    """Heatmap showing best encoder for each (dimension, generator) combination."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dimensions = sorted(set(r['dimension'] for r in results))
    generators = sorted(set(r['generator'] for r in results))
    encoders = sorted(set(r['encoder'] for r in results))
    
    # Use largest vector count
    max_count = max(r['vector_count'] for r in results)
    
    # Create matrix for best compression ratio
    compression_matrix = np.zeros((len(generators), len(dimensions)))
    best_encoder_matrix = np.empty((len(generators), len(dimensions)), dtype=object)
    
    for i, gen in enumerate(generators):
        for j, dim in enumerate(dimensions):
            filtered = [r for r in results 
                       if r['dimension'] == dim and r['generator'] == gen
                       and r['vector_count'] == max_count]
            
            if filtered:
                best = max(filtered, key=lambda x: x['compression_ratio']['mean'])
                compression_matrix[i, j] = best['compression_ratio']['mean']
                best_encoder_matrix[i, j] = best['encoder']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(compression_matrix, cmap='YlGnBu', aspect='auto')
    
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_yticks(np.arange(len(generators)))
    ax.set_xticklabels(dimensions)
    ax.set_yticklabels(generators)
    
    # Add text annotations
    for i in range(len(generators)):
        for j in range(len(dimensions)):
            encoder_short = best_encoder_matrix[i, j].replace('SphericalEncoder', 'Sph')
            encoder_short = encoder_short.replace('_Normalized', '_N')
            text = ax.text(j, i, f'{encoder_short}\n{compression_matrix[i, j]:.1f}x',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(f'Best Encoder by Compression Ratio ({max_count:,} vectors)')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Generator Type')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Compression Ratio', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_best_encoder.png', dpi=300)
    plt.close()

def generate_summary_table(results, output_dir='plots/vector'):
    """Generate a summary table with key metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Use largest vector count
    max_count = max(r['vector_count'] for r in results)
    
    with open(f'{output_dir}/summary_table.txt', 'w') as f:
        f.write("=" * 120 + "\n")
        f.write(f"VECTOR ENCODING BENCHMARK SUMMARY ({max_count:,} vectors)\n")
        f.write("=" * 120 + "\n\n")
        
        dimensions = sorted(set(r['dimension'] for r in results))
        generators = sorted(set(r['generator'] for r in results))
        
        for dim in dimensions:
            f.write(f"\nDIMENSION {dim}\n")
            f.write("-" * 120 + "\n")
            
            for gen in generators:
                f.write(f"\n  Generator: {gen}\n")
                f.write("  " + "-" * 116 + "\n")
                
                filtered = [r for r in results 
                           if r['dimension'] == dim and r['generator'] == gen
                           and r['vector_count'] == max_count]
                
                if not filtered:
                    continue
                
                f.write(f"  {'Encoder':<40} {'Comp.Ratio':<12} {'Enc MB/s':<12} "
                       f"{'Dec MB/s':<12} {'MAE':<12} {'Cosine Sim':<12} "
                       f"{'Mag Rel Err':<14} {'Mag Abs Err':<14}\n")
                f.write("  " + "-" * 130 + "\n")
                
                for r in sorted(filtered, key=lambda x: -x['compression_ratio']['mean']):
                    mag_rel = r['errors'].get('mean_magnitude_relative_error', float('nan'))
                    mag_abs = r['errors'].get('mean_magnitude_absolute_error', float('nan'))
                    mag_rel_str = f"{mag_rel:.3e}" if mag_rel and mag_rel > 0 else "N/A"
                    mag_abs_str = f"{mag_abs:.3e}" if mag_abs and mag_abs > 0 else "N/A"
                    f.write(f"  {r['encoder']:<40} "
                           f"{r['compression_ratio']['mean']:>6.2f}±{r['compression_ratio']['stddev']:>4.2f} "
                           f"{r['encode_throughput_mbps']['mean']:>8.1f}±{r['encode_throughput_mbps']['stddev']:>3.1f} "
                           f"{r['decode_throughput_mbps']['mean']:>8.1f}±{r['decode_throughput_mbps']['stddev']:>3.1f} "
                           f"{r['errors']['mean_absolute_error']:>10.2e}  "
                           f"{r['errors']['mean_cosine_similarity']:>10.6f}  "
                           f"{mag_rel_str:>12}  "
                           f"{mag_abs_str:>12}\n")
        
        f.write("\n" + "=" * 120 + "\n")

def plot_magnitude_errors(results, output_dir='plots/vector'):
    """
    Plot magnitude-specific round-trip errors for encoders that store magnitudes.

    Uses the mean_magnitude_relative_error and mean_magnitude_absolute_error fields
    which are only non-zero for SphericalEncoder_Normalized_* variants.  Encoders
    that do not store magnitudes are silently skipped so the axes remain uncluttered.

    Three subplots per generator:
        1. Mean magnitude relative error   (log scale) — shows precision loss as a fraction
        2. Mean magnitude absolute error   (log scale) — shows absolute scale of the loss
        3. Max  magnitude absolute error   (log scale) — shows worst-case behaviour
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    generators   = sorted(set(r['generator']  for r in results))
    dimensions   = sorted(set(r['dimension']  for r in results))
    encoders_all = sorted(set(r['encoder']    for r in results))
    max_count    = max(r['vector_count']       for r in results)

    # Only keep encoders that actually have magnitude errors in at least one result
    def has_mag_error(enc):
        return any(
            r.get('errors', {}).get('mean_magnitude_relative_error', 0.0) > 0
            for r in results if r['encoder'] == enc
        )
    encoders = [e for e in encoders_all if has_mag_error(e)]

    if not encoders:
        print("  (no magnitude-error data found — skipping magnitude error plots)")
        return

    colors = cm.tab10(np.linspace(0, 0.9, len(encoders)))
    x_pos  = np.arange(len(dimensions))
    width  = 0.8 / max(len(encoders), 1)

    for gen in generators:
        filtered = [r for r in results
                    if r['generator'] == gen and r['vector_count'] == max_count]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f'Magnitude Round-Trip Errors — {gen} (D varies, N={max_count:,})',
            fontsize=13, fontweight='bold'
        )

        for i, encoder in enumerate(encoders):
            rel_errs  = []
            abs_errs  = []
            max_errs  = []

            for dim in dimensions:
                match = next((r for r in filtered
                              if r['encoder'] == encoder and r['dimension'] == dim), None)
                rel_errs.append(match['errors'].get('mean_magnitude_relative_error', 0) if match else 0)
                abs_errs.append(match['errors'].get('mean_magnitude_absolute_error', 0) if match else 0)
                max_errs.append(match['errors'].get('max_magnitude_absolute_error',  0) if match else 0)

            label  = encoder
            offset = x_pos + i * width

            ax1.bar(offset, rel_errs, width, label=label, color=colors[i])
            ax2.bar(offset, abs_errs, width, label=label, color=colors[i])
            ax3.bar(offset, max_errs, width, label=label, color=colors[i])

        tick_offset = width * (len(encoders) - 1) / 2
        for ax, title, ylabel in [
            (ax1, 'Mean Magnitude Relative Error',  'Relative Error  |m̂−m| / m'),
            (ax2, 'Mean Magnitude Absolute Error',  'Absolute Error  |m̂−m|'),
            (ax3, 'Max  Magnitude Absolute Error',  'Absolute Error  |m̂−m|'),
        ]:
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Dimension', fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xticks(x_pos + tick_offset)
            ax.set_xticklabels(dimensions)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/magnitude_errors_{gen}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_encoder_comparison_by_generator(results, output_dir='plots/vector'):
    """
    Compare encoders across different data generators at largest dimension and dataset size.
    
    Creates a comprehensive comparison showing:
    - Compression ratios
    - Encode/decode throughput
    - Reconstruction errors
    
    All at the largest dimension (768) and largest dataset size (1,000,000).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get largest dimension and vector count
    max_dim = max(r['dimension'] for r in results)
    max_count = max(r['vector_count'] for r in results)
    
    # Filter to largest dimension and count
    filtered = [r for r in results 
               if r['dimension'] == max_dim and r['vector_count'] == max_count]
    
    if not filtered:
        print(f"Warning: No results found for dimension={max_dim}, count={max_count}")
        return
    
    generators = sorted(set(r['generator'] for r in filtered))
    encoders = sorted(set(r['encoder'] for r in filtered))
    
    # Create a 2x3 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    
    x_pos = np.arange(len(generators))
    width = 0.18
    colors = cm.Set2(np.linspace(0, 1, len(encoders)))
    
    # 1. Compression Ratio
    for i, encoder in enumerate(encoders):
        ratios = []
        errors = []
        for gen in generators:
            matching = [r for r in filtered 
                       if r['encoder'] == encoder and r['generator'] == gen]
            if matching:
                ratios.append(matching[0]['compression_ratio']['mean'])
                errors.append(matching[0]['compression_ratio']['stddev'])
            else:
                ratios.append(0)
                errors.append(0)
        
        ax1.bar(x_pos + i * width, ratios, width, 
               label=encoder, yerr=errors, capsize=3, color=colors[i])
    
    ax1.set_xlabel('Data Generator', fontweight='bold')
    ax1.set_ylabel('Compression Ratio', fontweight='bold')
    ax1.set_title(f'Compression Ratio by Generator (D={max_dim}, N={max_count:,})', 
                  fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
    ax1.set_xticklabels(generators, rotation=15, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Encode Throughput
    for i, encoder in enumerate(encoders):
        throughputs = []
        errors = []
        for gen in generators:
            matching = [r for r in filtered 
                       if r['encoder'] == encoder and r['generator'] == gen]
            if matching:
                throughputs.append(matching[0]['encode_throughput_mbps']['mean'])
                errors.append(matching[0]['encode_throughput_mbps']['stddev'])
            else:
                throughputs.append(0)
                errors.append(0)
        
        ax2.bar(x_pos + i * width, throughputs, width, 
               label=encoder, yerr=errors, capsize=3, color=colors[i])
    
    ax2.set_xlabel('Data Generator', fontweight='bold')
    ax2.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax2.set_title(f'Encode Throughput by Generator (D={max_dim}, N={max_count:,})', 
                  fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
    ax2.set_xticklabels(generators, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Decode Throughput
    for i, encoder in enumerate(encoders):
        throughputs = []
        errors = []
        for gen in generators:
            matching = [r for r in filtered 
                       if r['encoder'] == encoder and r['generator'] == gen]
            if matching:
                throughputs.append(matching[0]['decode_throughput_mbps']['mean'])
                errors.append(matching[0]['decode_throughput_mbps']['stddev'])
            else:
                throughputs.append(0)
                errors.append(0)
        
        ax3.bar(x_pos + i * width, throughputs, width, 
               label=encoder, yerr=errors, capsize=3, color=colors[i])
    
    ax3.set_xlabel('Data Generator', fontweight='bold')
    ax3.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax3.set_title(f'Decode Throughput by Generator (D={max_dim}, N={max_count:,})', 
                  fontweight='bold', fontsize=12)
    ax3.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
    ax3.set_xticklabels(generators, rotation=15, ha='right')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Mean Euclidean Distance Error (log scale)
    for i, encoder in enumerate(encoders):
        errors_euc = []
        for gen in generators:
            matching = [r for r in filtered 
                       if r['encoder'] == encoder and r['generator'] == gen]
            if matching:
                errors_euc.append(matching[0]['errors']['mean_euclidean_distance'])
            else:
                errors_euc.append(0)
        
        ax4.bar(x_pos + i * width, errors_euc, width, 
               label=encoder, color=colors[i])
    
    ax4.set_xlabel('Data Generator', fontweight='bold')
    ax4.set_ylabel('Mean Euclidean Distance', fontweight='bold')
    ax4.set_title(f'Reconstruction Error by Generator (D={max_dim}, N={max_count:,})', 
                  fontweight='bold', fontsize=12)
    ax4.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
    ax4.set_xticklabels(generators, rotation=15, ha='right')
    ax4.legend(loc='upper left')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Mean magnitude relative error (log scale) — only non-zero for Normalized encoders
    for i, encoder in enumerate(encoders):
        mag_rel = []
        for gen in generators:
            matching = [r for r in filtered
                        if r['encoder'] == encoder and r['generator'] == gen]
            val = matching[0]['errors'].get('mean_magnitude_relative_error', 0) if matching else 0
            mag_rel.append(val if val > 0 else np.nan)

        ax5.bar(x_pos + i * width, mag_rel, width,
                label=encoder, color=colors[i])

    ax5.set_xlabel('Data Generator', fontweight='bold')
    ax5.set_ylabel('Mean |m̂−m| / m', fontweight='bold')
    ax5.set_title(f'Magnitude Relative Error by Generator (D={max_dim}, N={max_count:,})',
                  fontweight='bold', fontsize=12)
    ax5.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
    ax5.set_xticklabels(generators, rotation=15, ha='right')
    ax5.legend(loc='upper left')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Mean magnitude absolute error (log scale)
    for i, encoder in enumerate(encoders):
        mag_abs = []
        for gen in generators:
            matching = [r for r in filtered
                        if r['encoder'] == encoder and r['generator'] == gen]
            val = matching[0]['errors'].get('mean_magnitude_absolute_error', 0) if matching else 0
            mag_abs.append(val if val > 0 else np.nan)

        ax6.bar(x_pos + i * width, mag_abs, width,
                label=encoder, color=colors[i])

    ax6.set_xlabel('Data Generator', fontweight='bold')
    ax6.set_ylabel('Mean |m̂−m|', fontweight='bold')
    ax6.set_title(f'Magnitude Absolute Error by Generator (D={max_dim}, N={max_count:,})',
                  fontweight='bold', fontsize=12)
    ax6.set_xticks(x_pos + width * (len(encoders) - 1) / 2)
    ax6.set_xticklabels(generators, rotation=15, ha='right')
    ax6.legend(loc='upper left')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/encoder_comparison_by_generator.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional detailed comparison table
    with open(f'{output_dir}/encoder_comparison_table.txt', 'w') as f:
        f.write("=" * 140 + "\n")
        f.write(f"ENCODER COMPARISON ACROSS GENERATORS (D={max_dim}, N={max_count:,})\n")
        f.write("=" * 140 + "\n\n")
        
        for gen in generators:
            f.write(f"\nGenerator: {gen}\n")
            f.write("-" * 140 + "\n")
            f.write(f"{'Encoder':<35} {'Comp.Ratio':<14} {'Enc MB/s':<14} {'Dec MB/s':<14} "
                   f"{'Euclidean Err':<16} {'Cosine Sim':<14}\n")
            f.write("-" * 140 + "\n")
            
            gen_results = [r for r in filtered if r['generator'] == gen]
            gen_results.sort(key=lambda x: -x['compression_ratio']['mean'])
            
            for r in gen_results:
                f.write(f"{r['encoder']:<35} "
                       f"{r['compression_ratio']['mean']:>6.2f}±{r['compression_ratio']['stddev']:>4.2f}  "
                       f"{r['encode_throughput_mbps']['mean']:>8.1f}±{r['encode_throughput_mbps']['stddev']:>3.1f}  "
                       f"{r['decode_throughput_mbps']['mean']:>8.1f}±{r['decode_throughput_mbps']['stddev']:>3.1f}  "
                       f"{r['errors']['mean_euclidean_distance']:>14.6e}  "
                       f"{r['errors']['mean_cosine_similarity']:>12.8f}\n")
        
        f.write("\n" + "=" * 140 + "\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot vector encoding benchmark results")
    parser.add_argument('--results', type=str, default='vector_benchmark_results.json',
                        help='Path to benchmark results JSON file')
    args = parser.parse_args()
    print("Loading benchmark results...")
    results = load_results(args.results)

    print(f"Loaded {len(results)} benchmark results")
    print(f"Dimensions: {sorted(set(r['dimension'] for r in results))}")
    print(f"Vector counts: {sorted(set(r['vector_count'] for r in results))}")
    print(f"Generators: {sorted(set(r['generator'] for r in results))}")
    print(f"Encoders: {sorted(set(r['encoder'] for r in results))}")
    
    print("\nGenerating plots...")
    
    output_dir = 'plots/vector'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("  - Compression by dimension...")
    plot_compression_by_dimension(results, output_dir)
    
    print("  - Compression scaling...")
    plot_compression_by_count(results, output_dir)
    
    print("  - Throughput comparison...")
    plot_throughput_comparison(results, output_dir)
    
    print("  - Latency comparison...")
    plot_latency_comparison(results, output_dir)
    
    print("  - Error metrics...")
    plot_error_metrics(results, output_dir)
    
    print("  - Compression vs error tradeoff...")
    plot_compression_vs_error_tradeoff(results, output_dir)
    
    print("  - Best encoder heatmap...")
    plot_heatmap_best_encoder(results, output_dir)
    
    print("  - Magnitude errors...")
    plot_magnitude_errors(results, output_dir)

    print("  - Encoder comparison by generator...")
    plot_encoder_comparison_by_generator(results, output_dir)
    
    print("  - Summary table...")
    generate_summary_table(results, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")
    print(f"Summary table saved to {output_dir}/summary_table.txt")

if __name__ == '__main__':
    main()
