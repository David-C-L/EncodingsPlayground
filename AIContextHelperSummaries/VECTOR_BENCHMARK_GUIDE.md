# Vector Encoding Benchmark Guide

## Overview

The vector encoding benchmark (`vector_benchmarks.cpp`) compares the performance of different encoding strategies for high-dimensional floating-point vectors. This is particularly relevant for machine learning embeddings, scientific computing, and data compression applications.

## Benchmark Configuration

### Dimensions Tested
- **96**: Small embedding size (e.g., BERT-tiny, compact models)
- **384**: Medium embedding size (e.g., sentence transformers)
- **768**: Large embedding size (e.g., BERT-base, GPT-2)

### Dataset Sizes
- **1,000**: Small datasets for quick testing
- **100,000**: Medium datasets (typical batch processing)
- **1,000,000**: Large datasets (production scale)

### Data Generators
1. **Unit**: Unit-norm vectors (magnitude = 1.0)
   - Common in: Normalized embeddings, directional data
   - Generated using: Marsaglia method (Gaussian sampling on hypersphere)

2. **NonUnit_Small**: Non-unit vectors with magnitude range [0.1, 2.0]
   - Common in: Raw embeddings before normalization
   - Relatively tight magnitude distribution

3. **NonUnit_Wide**: Non-unit vectors with magnitude range [0.001, 1000.0]
   - Common in: Scientific data, unnormalized features
   - Wide dynamic range (6 orders of magnitude)

### Encoders Compared

#### 1. SphericalEncoder
- **Approach**: Converts Cartesian coordinates to spherical coordinates
- **Compression**: Exploits reduced entropy in angular representation
- **Best for**: Unit vectors or data with low magnitude variation
- **Properties**: Lossless (within floating-point precision)

#### 2. SphericalEncoder_Normalized
- **Approach**: Normalizes vectors, encodes magnitudes separately
- **Compression**: Spherical encoding + magnitude compression
- **Best for**: Non-unit vectors with varying magnitudes
- **Properties**: Lossless magnitude + lossless angles

#### 3. RawEncoder
- **Approach**: Stores vectors as-is with minimal overhead
- **Compression**: None (baseline)
- **Best for**: Benchmarking, when speed > size
- **Properties**: Fast, supports random access

#### 4. ZstdEncoder
- **Approach**: General-purpose Zstd compression on float stream
- **Compression**: Dictionary + entropy coding
- **Best for**: General purpose, patterns in data
- **Properties**: Good all-around performance

## Metrics Collected

### Compression Metrics
- **Compression Ratio**: Uncompressed size / compressed size
- **Compressed Size**: Actual bytes used
- **Statistics**: Mean, standard deviation over 3 iterations

### Performance Metrics
- **Encode Throughput**: MB/s during compression
- **Decode Throughput**: MB/s during decompression
- **Encode Latency**: Milliseconds to compress
- **Decode Latency**: Milliseconds to decompress

### Error Metrics
All encoders in this benchmark are lossless, but floating-point precision limits apply:

1. **Mean Absolute Error**: Average per-element absolute difference
2. **Max Absolute Error**: Maximum per-element absolute difference
3. **Mean Relative Error**: Average relative error (normalized by magnitude)
4. **Mean Euclidean Distance**: Average L2 distance between vectors
5. **Max Euclidean Distance**: Maximum L2 distance
6. **Mean Cosine Similarity**: Average angular similarity (1.0 = perfect)
7. **Mean Angle Error**: Average angular difference in radians

## Visualization Outputs

### 1. Compression by Dimension
**Files**: `compression_by_dimension_{generator}.png`

Shows compression ratio vs dimension for each generator type. Helps identify:
- Which encoders scale better with dimensionality
- Generator-specific compression patterns

### 2. Compression Scaling
**Files**: `compression_scaling_D{dim}_{generator}.png`

Shows how compression ratio changes with dataset size. Useful for:
- Understanding if larger datasets compress better
- Identifying fixed overhead vs per-element costs

### 3. Throughput Comparison
**File**: `throughput_comparison.png`

Side-by-side encode and decode throughput for Unit vectors. Shows:
- Speed/performance tradeoffs
- Asymmetry between encode and decode

### 4. Latency Comparison
**Files**: `latency_comparison_{count}.png`

Log-scale latency plots for different dataset sizes. Helps identify:
- Absolute time costs
- Scalability characteristics

### 5. Error Metrics
**Files**: `error_metrics_{generator}.png`

Four-panel comparison of reconstruction errors:
- Mean Absolute Error
- Mean Euclidean Distance
- Mean Cosine Similarity
- Mean Angle Error

All on log scale for better visibility of small errors.

### 6. Compression vs Error Tradeoff
**Files**: `tradeoff_D{dim}_{generator}.png`

Scatter plot showing compression ratio vs reconstruction error. Ideal encoders are:
- Top-right: High compression, low error
- Bottom-left: Low compression, high error (avoid)

### 7. Best Encoder Heatmap
**File**: `heatmap_best_encoder.png`

Shows which encoder achieves best compression for each (dimension, generator) combination. Quick reference for encoder selection.

### 8. **NEW: Encoder Comparison by Generator**
**File**: `encoder_comparison_by_generator.png`

**Purpose**: Direct comparison of all encoders across different data generators at the largest scale.

**Configuration**: 
- Dimension: 768 (largest tested)
- Dataset size: 1,000,000 vectors (largest tested)
- Generators: All three (Unit, NonUnit_Small, NonUnit_Wide)

**Four Panels**:

1. **Top-Left: Compression Ratio**
   - Shows which encoder compresses best for each generator
   - Higher is better
   - Identifies generator-specific strengths

2. **Top-Right: Encode Throughput**
   - Compression speed in MB/s
   - Higher is better
   - Shows performance cost of compression

3. **Bottom-Left: Decode Throughput**
   - Decompression speed in MB/s
   - Higher is better
   - Critical for read-heavy workloads

4. **Bottom-Right: Reconstruction Error**
   - Mean Euclidean distance (log scale)
   - Lower is better
   - All should be near floating-point precision

**Use Cases**:
- **Quick encoder selection**: See all metrics at a glance
- **Generator sensitivity**: Identify which encoders handle different data types well
- **Tradeoff analysis**: Balance compression, speed, and accuracy
- **Production decisions**: Choose encoder for specific data characteristics

**Companion File**: `encoder_comparison_table.txt`
- Detailed numeric breakdown by generator
- Sorted by compression ratio
- Includes error bars (standard deviations)
- Easy to copy into reports/papers

## Summary Table

**File**: `summary_table.txt`

Text-based summary with all metrics organized by dimension and generator. Includes:
- Compression ratio ± stddev
- Encode throughput ± stddev
- Decode throughput ± stddev
- Mean absolute error
- Cosine similarity

## Running the Benchmark

### Quick Test (Small Dataset)
```bash
# Build
cd debugBuild
cmake --build . --target vector_benchmarks -j8

# Run
cd ../Benchmarks
../debugBuild/bin/vector_benchmarks
```

**Expected time**: ~5-10 minutes for full benchmark

### Generate Plots
```bash
cd Benchmarks
python3 plot_vector_results.py
```

**Output**: `plots/vector/` directory with all visualizations

## Interpreting Results

### Expected Patterns

1. **Unit Vectors**:
   - SphericalEncoder should excel (low entropy in angles)
   - Normalization adds overhead, not beneficial
   - ZstdEncoder decent but not specialized

2. **NonUnit_Small**:
   - SphericalEncoder_Normalized competitive
   - Magnitude range narrow → good compression
   - SphericalEncoder without normalization may struggle

3. **NonUnit_Wide**:
   - Wide magnitude range challenges all encoders
   - SphericalEncoder_Normalized handles magnitudes separately
   - ZstdEncoder may struggle with wide dynamic range

### Anomalies to Investigate

- **Very low compression (< 1.1x)**: Data may be random or already compressed
- **Very high error (> 1e-5)**: Potential bug in encode/decode
- **Asymmetric throughput**: Normal - decode often faster than encode
- **Dimension independence**: Suggests per-vector overhead dominates

## Performance Tips

### For Maximum Compression
- Use SphericalEncoder for unit vectors
- Use SphericalEncoder_Normalized for non-unit vectors
- Consider Zstd compression level tuning (currently level 3)

### For Maximum Speed
- Use RawEncoder (no compression overhead)
- Pre-normalize data if using spherical encoding
- Use larger batch sizes to amortize overhead

### For Balanced Performance
- Start with ZstdEncoder (good all-around)
- Profile with your specific data
- Consider dimension-specific tuning

## Future Enhancements

Potential additions to the benchmark:

1. **More Encoders**:
   - Quantization (8-bit, 4-bit)
   - PCA/SVD dimensionality reduction
   - Product quantization
   - Huffman coding on angle bins

2. **More Generators**:
   - Clustered data (k-means centers)
   - Sparse vectors (many zeros)
   - Correlated dimensions
   - Real embeddings (BERT, GPT, etc.)

3. **More Dimensions**:
   - Very small: 32, 64
   - Very large: 1024, 2048, 4096

4. **More Metrics**:
   - Memory usage during encode/decode
   - Random access performance (decode single vector)
   - Batch decode efficiency
   - Progressive decoding support

5. **Parallel Encoding**:
   - Multi-threaded compression
   - SIMD optimizations
   - GPU encoding

## Related Documentation

- `IMPLEMENTATION_SUMMARY.md`: Overall encoding framework
- `BIT_PARENTHESES_ENCODER.md`: Structural encoding for trees
- `MAP_ENCODING_GUIDE.md`: Map encoding strategies
- `SPHERICAL_ENCODER.md`: Detailed spherical coordinate math

## Citation

If using these benchmarks in research:

```bibtex
@misc{vector_encoding_benchmark,
  title={High-Dimensional Vector Encoding Benchmark},
  author={EncodingsPlayground},
  year={2026},
  note={Comparison of spherical, dictionary, and raw encoding for embeddings}
}
```
