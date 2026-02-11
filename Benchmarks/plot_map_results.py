#!/usr/bin/env python3
"""
Plot map benchmark results

Generates visualizations for map encoding performance across different strategies.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filename='map_benchmark_results.json'):
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
        # Extract the results array from the JSON structure
        if isinstance(data, dict) and 'results' in data:
            return data['results']
        return data

def plot_compression_by_encoder(results, output_dir='.', data_size=1000):
    """Plot compression ratios grouped by encoder (for a specific data size)"""
    # Filter results for the specified data size
    filtered_results = [r for r in results if r['dataSize'] == data_size]
    
    encoders = {}
    
    for result in filtered_results:
        encoder = result['encoderName']
        generator = result['datasetName']
        ratio = result['metrics']['memory']['compressionRatio']
        
        if encoder not in encoders:
            encoders[encoder] = {'generators': [], 'ratios': []}
        
        encoders[encoder]['generators'].append(generator)
        encoders[encoder]['ratios'].append(ratio)
    
    # Get unique generators in consistent order
    all_generators = sorted(set(r['datasetName'] for r in filtered_results))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(all_generators))
    width = 0.15
    
    for i, (encoder, data) in enumerate(encoders.items()):
        # Create mapping from generator to ratio
        gen_ratio_map = dict(zip(data['generators'], data['ratios']))
        # Get ratios in the same order as all_generators
        ratios = [gen_ratio_map.get(gen, 0) for gen in all_generators]
        
        offset = (i - len(encoders) / 2) * width
        ax.bar(x + offset, ratios, width, label=encoder)
    
    ax.set_xlabel('Data Generator')
    ax.set_ylabel('Compression Ratio')
    ax.set_title(f'Map Encoding: Compression Ratio by Encoder and Generator (n={data_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(all_generators, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'map_compression_by_encoder.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/map_compression_by_encoder.png")

def plot_encode_decode_times(results, output_dir='.', data_size=1000):
    """Plot encode and decode times (for a specific data size)"""
    # Filter results for the specified data size
    filtered_results = [r for r in results if r['dataSize'] == data_size]
    
    encoders = {}
    
    for result in filtered_results:
        encoder = result['encoderName']
        generator = result['datasetName']
        # Convert nanoseconds to milliseconds
        encode_time = result['metrics']['timing']['encodeTime_ns'] / 1_000_000
        decode_time = result['metrics']['timing']['decodeBulkTime_ns'] / 1_000_000
        
        key = f"{encoder}_{generator}"
        encoders[key] = {
            'encoder': encoder,
            'generator': generator,
            'encode': encode_time,
            'decode': decode_time
        }
    
    # Get unique generators in consistent order
    all_generators = sorted(set(r['datasetName'] for r in filtered_results))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Encode times
    encode_data = {}
    for key, data in encoders.items():
        encoder = data['encoder']
        if encoder not in encode_data:
            encode_data[encoder] = {}
        encode_data[encoder][data['generator']] = data['encode']
    
    x = np.arange(len(all_generators))
    width = 0.15
    
    for i, (encoder, gen_times) in enumerate(encode_data.items()):
        times = [gen_times.get(gen, 0) for gen in all_generators]
        offset = (i - len(encode_data) / 2) * width
        ax1.bar(x + offset, times, width, label=encoder)
    
    ax1.set_xlabel('Data Generator')
    ax1.set_ylabel('Encode Time (ms)')
    ax1.set_title(f'Encode Performance (n={data_size})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_generators, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Decode times
    decode_data = {}
    for key, data in encoders.items():
        encoder = data['encoder']
        if encoder not in decode_data:
            decode_data[encoder] = {}
        decode_data[encoder][data['generator']] = data['decode']
    
    for i, (encoder, gen_times) in enumerate(decode_data.items()):
        times = [gen_times.get(gen, 0) for gen in all_generators]
        offset = (i - len(decode_data) / 2) * width
        ax2.bar(x + offset, times, width, label=encoder)
    
    ax2.set_xlabel('Data Generator')
    ax2.set_ylabel('Decode Time (ms)')
    ax2.set_title(f'Decode Performance (n={data_size})')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_generators, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'map_encode_decode_times.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/map_encode_decode_times.png")

def plot_throughput_comparison(results, output_dir='.', data_size=1000):
    """Plot encode and decode throughput (for a specific data size)"""
    # Filter results for the specified data size
    filtered_results = [r for r in results if r['dataSize'] == data_size]
    
    encoders = {}
    
    for result in filtered_results:
        encoder = result['encoderName']
        generator = result['datasetName']
        
        # Throughput already calculated in the JSON (elements per second)
        encode_throughput = result['metrics']['timing'].get('encodeThroughputMBps', 0)
        decode_throughput = result['metrics']['timing'].get('decodeBulkThroughputMBps', 0)
        
        key = f"{encoder}_{generator}"
        encoders[key] = {
            'encoder': encoder,
            'generator': generator,
            'encode_throughput': encode_throughput,
            'decode_throughput': decode_throughput
        }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group by encoder
    throughput_data = {}
    for key, data in encoders.items():
        encoder = data['encoder']
        if encoder not in throughput_data:
            throughput_data[encoder] = {'generators': [], 'encode': [], 'decode': []}
        throughput_data[encoder]['generators'].append(data['generator'])
        throughput_data[encoder]['encode'].append(data['encode_throughput'])
        throughput_data[encoder]['decode'].append(data['decode_throughput'])
    
    # Plot encode throughput
    x = np.arange(len(next(iter(throughput_data.values()))['generators']))
    width = 0.08
    
    for i, (encoder, data) in enumerate(throughput_data.items()):
        offset = (i - len(throughput_data) / 2) * width
        ax.bar(x + offset, data['encode'], width, label=f'{encoder} (encode)', alpha=0.8)
    
    ax.set_xlabel('Data Generator')
    ax.set_ylabel('Throughput (MB/s)')
    ax.set_title('Map Encoding: Encode Throughput')
    ax.set_xticks(x)
    ax.set_xticklabels(next(iter(throughput_data.values()))['generators'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'map_throughput.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/map_throughput.png")

def plot_space_time_tradeoff(results, output_dir='.', data_size=1000):
    """Plot space-time tradeoff (compression vs speed) for a specific data size"""
    # Filter results for the specified data size
    filtered_results = [r for r in results if r['dataSize'] == data_size]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compression vs Encode Time
    encoders = set(r['encoderName'] for r in filtered_results)
    colors = plt.cm.Set3(np.linspace(0, 1, len(encoders)))
    encoder_colors = dict(zip(encoders, colors))
    
    for result in results:
        encoder = result['encoderName']
        compression = result['metrics']['memory']['compressionRatio']
        encode_time = result['metrics']['timing']['encodeTime_ns'] / 1_000_000  # Convert to ms
        
        ax1.scatter(compression, encode_time,
                   label=encoder, color=encoder_colors[encoder], s=100, alpha=0.6)
    
    ax1.set_xlabel('Compression Ratio')
    ax1.set_ylabel('Encode Time (ms)')
    ax1.set_title('Compression vs Encode Speed')
    ax1.grid(True, alpha=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    # Compression vs Decode Time
    for result in results:
        encoder = result['encoderName']
        compression = result['metrics']['memory']['compressionRatio']
        decode_time = result['metrics']['timing']['decodeBulkTime_ns'] / 1_000_000  # Convert to ms
        
        ax2.scatter(compression, decode_time,
                   label=encoder, color=encoder_colors[encoder], s=100, alpha=0.6)
    
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('Decode Time (ms)')
    ax2.set_title('Compression vs Decode Speed')
    ax2.grid(True, alpha=0.3)
    
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'map_space_time_tradeoff.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/map_space_time_tradeoff.png")

def plot_best_encoder_by_generator(results, output_dir='.', data_size=1000):
    """Show which encoder is best for each generator type (for a specific data size)"""
    # Filter results for the specified data size
    filtered_results = [r for r in results if r['dataSize'] == data_size]
    
    generators = {}
    
    for result in filtered_results:
        generator = result['datasetName']
        encoder = result['encoderName']
        ratio = result['metrics']['memory']['compressionRatio']
        
        if generator not in generators:
            generators[generator] = []
        
        generators[generator].append({
            'encoder': encoder,
            'ratio': ratio,
            'encode_time': result['metrics']['timing']['encodeTime_ns'] / 1_000_000,
            'decode_time': result['metrics']['timing']['decodeBulkTime_ns'] / 1_000_000
        })
    
    # Find best compression for each generator
    fig, ax = plt.subplots(figsize=(12, 6))
    
    gen_names = []
    best_encoders = []
    best_ratios = []
    
    for gen_name, encoder_results in generators.items():
        best = min(encoder_results, key=lambda x: x['ratio'])
        gen_names.append(gen_name)
        best_encoders.append(best['encoder'])
        best_ratios.append(best['ratio'])
    
    colors = plt.cm.Set2(np.arange(len(gen_names)))
    bars = ax.bar(gen_names, best_ratios, color=colors)
    
    # Add encoder names on bars
    for bar, encoder in zip(bars, best_encoders):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               encoder, ha='center', va='bottom', rotation=45, fontsize=8)
    
    ax.set_xlabel('Data Generator')
    ax.set_ylabel('Best Compression Ratio')
    ax.set_title('Best Encoder for Each Data Pattern')
    ax.set_xticklabels(gen_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'map_best_encoder_by_generator.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/map_best_encoder_by_generator.png")

def plot_heatmap(results, output_dir='.', data_size=1000):
    """Create heatmap of compression ratios (for a specific data size)"""
    # Filter results for the specified data size
    filtered_results = [r for r in results if r['dataSize'] == data_size]
    
    # Build matrix
    encoders = sorted(set(r['encoderName'] for r in filtered_results))
    generators = sorted(set(r['datasetName'] for r in filtered_results))
    
    matrix = np.zeros((len(encoders), len(generators)))
    
    for result in results:
        i = encoders.index(result['encoderName'])
        j = generators.index(result['datasetName'])
        matrix[i, j] = result['metrics']['memory']['compressionRatio']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(generators)))
    ax.set_yticks(np.arange(len(encoders)))
    ax.set_xticklabels(generators, rotation=45, ha='right')
    ax.set_yticklabels(encoders)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Compression Ratio', rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(len(encoders)):
        for j in range(len(generators)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Compression Ratio Heatmap: Encoders vs Generators')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'map_compression_heatmap.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/map_compression_heatmap.png")

def main():
    print("Loading map benchmark results...")
    results = load_results()
    
    print(f"Found {len(results)} benchmark results\n")
    
    # Use data_size=100000 for clearer visualizations
    # You can change this to 1000 or 10000 to see results for smaller datasets
    data_size = 100000
    
    print(f"Generating plots for data size = {data_size}...")
    plot_compression_by_encoder(results, data_size=data_size)
    plot_encode_decode_times(results, data_size=data_size)
    plot_throughput_comparison(results, data_size=data_size)
    plot_space_time_tradeoff(results, data_size=data_size)
    plot_best_encoder_by_generator(results, data_size=data_size)
    plot_heatmap(results, data_size=data_size)
    
    print("\nAll plots generated successfully!")
    print("\nGenerated plots:")
    print("  1. map_compression_by_encoder.png - Compression ratios grouped by encoder")
    print("  2. map_encode_decode_times.png - Encode/decode performance")
    print("  3. map_throughput.png - Throughput comparison")
    print("  4. map_space_time_tradeoff.png - Compression vs speed tradeoffs")
    print("  5. map_best_encoder_by_generator.png - Best encoder for each pattern")
    print("  6. map_compression_heatmap.png - Heatmap of all combinations")

if __name__ == '__main__':
    main()
