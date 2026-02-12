#!/usr/bin/env python3
"""
Plot benchmark results from JSON output
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(filepath):
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_compression_ratios(results, output_dir):
    """Plot compression ratios for different encoders and datasets"""
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))
    
    # Group by dataset size (use largest size for clearest differences)
    max_size = max(r['dataSize'] for r in results['results'])
    
    data = {encoder: [] for encoder in encoders}
    
    for encoder in encoders:
        for dataset in datasets:
            matching = [r for r in results['results'] 
                       if r['encoderName'] == encoder 
                       and r['datasetName'] == dataset
                       and r['dataSize'] == max_size]
            
            if matching:
                ratio = matching[0]['metrics']['memory']['compressionRatio']
                # Invert ratio to show "how many times smaller"
                inverted_ratio = 1.0 / ratio if ratio > 0 else 0
                data[encoder].append(inverted_ratio)
            else:
                data[encoder].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(encoders)
    
    for i, encoder in enumerate(encoders):
        offset = (i - len(encoders)/2) * width + width/2
        bars = ax.bar(x + offset, data[encoder], width, label=encoder)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}x',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Compression Ratio (times smaller than raw)')
    ax.set_title(f'Compression Ratios (n={max_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'compression_ratios.png', dpi=300)
    print(f"Saved: {output_dir / 'compression_ratios.png'}")
    plt.close()


def plot_encode_throughput(results, output_dir):
    """Plot encoding throughput"""
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))
    
    max_size = max(r['dataSize'] for r in results['results'])
    
    data = {encoder: [] for encoder in encoders}
    
    for encoder in encoders:
        for dataset in datasets:
            matching = [r for r in results['results'] 
                       if r['encoderName'] == encoder 
                       and r['datasetName'] == dataset
                       and r['dataSize'] == max_size]
            
            if matching:
                throughput = matching[0]['metrics']['timing']['encodeElementsPerSecond']
                data[encoder].append(throughput / 1e6)  # Convert to M elem/s
            else:
                data[encoder].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(encoders)
    
    for i, encoder in enumerate(encoders):
        offset = (i - len(encoders)/2) * width + width/2
        ax.bar(x + offset, data[encoder], width, label=encoder)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Throughput (M elements/sec)')
    ax.set_title(f'Encoding Throughput (n={max_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'encode_throughput.png', dpi=300)
    print(f"Saved: {output_dir / 'encode_throughput.png'}")
    plt.close()


def plot_decode_throughput(results, output_dir):
    """Plot decoding throughput"""
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))
    
    max_size = max(r['dataSize'] for r in results['results'])
    
    data = {encoder: [] for encoder in encoders}
    
    for encoder in encoders:
        for dataset in datasets:
            matching = [r for r in results['results'] 
                       if r['encoderName'] == encoder 
                       and r['datasetName'] == dataset
                       and r['dataSize'] == max_size]
            
            if matching:
                throughput = matching[0]['metrics']['timing']['decodeBulkElementsPerSecond']
                data[encoder].append(throughput / 1e6)  # Convert to M elem/s
            else:
                data[encoder].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(encoders)
    
    for i, encoder in enumerate(encoders):
        offset = (i - len(encoders)/2) * width + width/2
        ax.bar(x + offset, data[encoder], width, label=encoder)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Throughput (M elements/sec)')
    ax.set_title(f'Decoding Throughput (n={max_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decode_throughput.png', dpi=300)
    print(f"Saved: {output_dir / 'decode_throughput.png'}")
    plt.close()


def plot_scaling(results, output_dir):
    """Plot how performance scales with data size"""
    encoders = sorted(set(r['encoderName'] for r in results['results']))
    
    # Pick one representative dataset
    dataset = 'Sequential'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for encoder in encoders:
        matching = [r for r in results['results'] 
                   if r['encoderName'] == encoder 
                   and r['datasetName'] == dataset]
        
        matching.sort(key=lambda x: x['dataSize'])
        
        sizes = [r['dataSize'] for r in matching]
        encode_times = [r['metrics']['timing']['encodeTime_ns'] / 1e6 for r in matching]  # ms
        decode_times = [r['metrics']['timing']['decodeBulkTime_ns'] / 1e6 for r in matching]  # ms
        
        ax1.plot(sizes, encode_times, marker='o', label=encoder)
        ax2.plot(sizes, decode_times, marker='o', label=encoder)
    
    ax1.set_xlabel('Data Size (elements)')
    ax1.set_ylabel('Encoding Time (ms)')
    ax1.set_title(f'Encoding Scaling ({dataset})')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Data Size (elements)')
    ax2.set_ylabel('Decoding Time (ms)')
    ax2.set_title(f'Decoding Scaling ({dataset})')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling.png', dpi=300)
    print(f"Saved: {output_dir / 'scaling.png'}")
    plt.close()


def plot_random_access(results, output_dir):
    """Plot random access performance"""
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))
    
    max_size = max(r['dataSize'] for r in results['results'])
    
    data = {encoder: [] for encoder in encoders}
    
    for encoder in encoders:
        for dataset in datasets:
            matching = [r for r in results['results'] 
                       if r['encoderName'] == encoder 
                       and r['datasetName'] == dataset
                       and r['dataSize'] == max_size]
            
            if matching:
                avg_time = matching[0]['metrics']['randomAccess']['averageRandomAccessTime_ns']
                data[encoder].append(avg_time if avg_time > 0 else None)
            else:
                data[encoder].append(None)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(encoders)
    
    for i, encoder in enumerate(encoders):
        offset = (i - len(encoders)/2) * width + width/2
        # Filter out None values
        plot_data = [d if d is not None else 0 for d in data[encoder]]
        bars = ax.bar(x + offset, plot_data, width, label=encoder)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average Random Access Time (ns)')
    ax.set_title(f'Random Access Performance (n={max_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'random_access.png', dpi=300)
    print(f"Saved: {output_dir / 'random_access.png'}")
    plt.close()


def plot_bits_per_element(results, output_dir):
    """Plot bits per element for different encoders"""
    datasets = sorted(set(r['datasetName'] for r in results['results']))
    encoders = sorted(set(r['encoderName'] for r in results['results']))
    
    max_size = max(r['dataSize'] for r in results['results'])
    
    data = {encoder: [] for encoder in encoders}
    
    for encoder in encoders:
        for dataset in datasets:
            matching = [r for r in results['results'] 
                       if r['encoderName'] == encoder 
                       and r['datasetName'] == dataset
                       and r['dataSize'] == max_size]
            
            if matching:
                bpe = matching[0]['metrics']['memory']['bitsPerElement']
                data[encoder].append(bpe)
            else:
                data[encoder].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(encoders)
    
    for i, encoder in enumerate(encoders):
        offset = (i - len(encoders)/2) * width + width/2
        ax.bar(x + offset, data[encoder], width, label=encoder)
    
    # Add baseline line (32 bits for int32_t)
    ax.axhline(y=32, color='r', linestyle='--', alpha=0.5, label='Uncompressed (32 bits)')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Bits per Element')
    ax.set_title(f'Storage Efficiency (n={max_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bits_per_element.png', dpi=300)
    print(f"Saved: {output_dir / 'bits_per_element.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot encoding benchmark results')
    parser.add_argument('input', type=Path, help='Input JSON file with benchmark results')
    parser.add_argument('-o', '--output', type=Path, default=Path('plots'),
                       help='Output directory for plots (default: plots/)')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    
    # Create output directory
    args.output.mkdir(exist_ok=True, parents=True)
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_compression_ratios(results, args.output)
    plot_encode_throughput(results, args.output)
    plot_decode_throughput(results, args.output)
    plot_scaling(results, args.output)
    plot_random_access(results, args.output)
    plot_bits_per_element(results, args.output)
    
    print(f"\n✓ All plots saved to: {args.output}")
    
    # Print summary
    print(f"\nBenchmark Summary:")
    print(f"  Total benchmarks: {len(results['results'])}")
    print(f"  Total duration: {results['metadata']['totalDuration_s']:.2f} seconds")
    print(f"  Start time: {results['metadata']['startTime']}")
    print(f"  End time: {results['metadata']['endTime']}")


if __name__ == '__main__':
    main()
