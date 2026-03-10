# Benchmarking Framework Guide

## Overview

The EncodingsPlayground benchmarking framework provides comprehensive performance measurement for encoding schemes. It measures:

- **Timing**: Encoding, bulk decoding, random access (shuffled, strided, range queries)
- **Memory**: Original size, encoded size, compression ratio, bits per element
- **Accuracy**: Lossless verification, mismatch detection
- **Throughput**: Elements/second and MB/s

## Quick Start

### Basic Usage

```cpp
#include "benchmark/BenchmarkRunner.hpp"
#include "benchmark/BenchmarkOutput.hpp"
#include "generators/CommonGenerators.hpp"
#include "encoders/RawEncoder.hpp"

using namespace encodings::benchmark;
using namespace encodings::generators;
using namespace encodings::encoders;

int main() {
    // Configure
    BenchmarkConfig config;
    config.dataSizes = {1000, 10000, 100000};
    config.iterations = 10;
    
    // Create runner
    BenchmarkRunner<int32_t> runner(config);
    
    // Register encoders
    runner.registerEncoder("Raw", std::make_shared<RawEncoder<int32_t>>());
    
    // Add datasets
    runner.addDataset("Sequential", 
        std::make_shared<SequentialGenerator<int32_t>>(0, 1));
    
    // Run
    auto results = runner.runAll();
    
    // Output
    TableFormatter::printSummaryTable(results);
    saveBenchmarkResults(results, "results.json");
    
    return 0;
}
```

### Configuration Options

```cpp
struct BenchmarkConfig {
    // Data sizes to test
    std::vector<size_t> dataSizes{1000, 10000, 100000};
    
    // Number of measurement iterations (for statistics)
    size_t iterations{10};
    
    // Warmup runs (not measured, for cache warming)
    size_t warmupRuns{3};
    
    // Random access testing
    bool testRandomAccess{true};
    size_t randomAccessSamples{100};  // Number of random reads
    
    // Strided access testing
    bool testStridedAccess{true};
    size_t stridedAccessSamples{100};
    size_t stride{10};  // Access every Nth element
    
    // Range query testing
    bool testRangeAccess{true};
    size_t rangeQueryCount{10};
    std::vector<size_t> rangeSizes{10, 100, 1000};
    
    // Validation
    bool validateCorrectness{true};
    bool validateRandomAccess{true};
    
    // Output
    bool verboseOutput{false};
    std::string outputPath{"Benchmarks/results"};
};
```

## Benchmark Metrics

### TimingMetrics

```cpp
struct TimingMetrics {
    nanoseconds encodeTime;                      // Total encoding time
    nanoseconds decodeBulkTime;                  // Decode all elements
    nanoseconds decodeRandomAccessTime;          // Random access time
    nanoseconds decodeStridedAccessTime;         // Strided access time
    nanoseconds decodeRangeAccessTime;           // Range query time
    
    // Statistics (if multiple iterations)
    std::optional<nanoseconds> encodeTimeStdDev;
    std::optional<nanoseconds> decodeBulkTimeStdDev;
    
    // Throughput
    double encodeElementsPerSecond;
    double decodeBulkElementsPerSecond;
    double encodeThroughputMBps;
    double decodeBulkThroughputMBps;
};
```

### MemoryMetrics

```cpp
struct MemoryMetrics {
    size_t originalSize;   // Original data size in bytes
    size_t encodedSize;    // Encoded data size in bytes
    
    double compressionRatio() const {
        return static_cast<double>(originalSize) / encodedSize;
    }
    
    double bitsPerElement() const {
        return (encodedSize * 8.0) / elementCount;
    }
};
```

### RandomAccessMetrics

```cpp
struct RandomAccessMetrics {
    // Random (shuffled) access
    nanoseconds averageRandomAccessTime;
    nanoseconds minRandomAccessTime;
    nanoseconds maxRandomAccessTime;
    size_t randomAccessCount;
    
    // Strided access
    nanoseconds averageStridedAccessTime;
    size_t stridedAccessCount;
    size_t stride;
    
    // Range queries
    nanoseconds averageRangeAccessTime;
    size_t rangeQueryCount;
    size_t averageRangeSize;
};
```

## Data Generators

The framework includes several built-in generators:

### UniformRandomGenerator
```cpp
// Random uniform distribution
auto gen = std::make_shared<UniformRandomGenerator<int32_t>>(
    0,      // min value
    10000   // max value
);
```

### SequentialGenerator
```cpp
// Monotonically increasing values
auto gen = std::make_shared<SequentialGenerator<int32_t>>(
    0,  // start value
    1   // step
);
```

### RepetitiveGenerator
```cpp
// Repetitive runs (good for RLE testing)
auto gen = std::make_shared<RepetitiveGenerator<int32_t>>(
    20,     // run length
    0,      // min value
    100     // max value
);
```

### ZipfianGenerator
```cpp
// Skewed distribution (good for dictionary encoding)
auto gen = std::make_shared<ZipfianGenerator<int32_t>>(
    100,    // cardinality (number of unique values)
    1.5     // exponent (higher = more skewed)
);
```

### ConstantGenerator
```cpp
// All same value (extreme RLE case)
auto gen = std::make_shared<ConstantGenerator<int32_t>>(42);
```

### NearlySortedGenerator
```cpp
// Mostly sorted with some noise
auto gen = std::make_shared<NearlySortedGenerator<int32_t>>(
    0,      // start
    1,      // step
    0.05    // noise fraction (5% shuffled)
);
```

### NormalGenerator
```cpp
// Floating-point normal distribution
auto gen = std::make_shared<NormalGenerator<double>>(
    0.0,    // mean
    1.0     // standard deviation
);
```

## Output Formats

### Console Output

#### Summary Table
```cpp
TableFormatter::printSummaryTable(results);
```

Output:
```
========================================================================================================================
BENCHMARK SUMMARY
========================================================================================================================

Encoder             Dataset             Size        Enc (ms)       Dec (ms)       Ratio       Bits/Elem      
------------------------------------------------------------------------------------------------------------------------
Raw                 Sequential          100000      1.234          0.987          1.00        32.00          
RLE                 Sequential          100000      2.345          1.234          15.23       2.10           
Delta               Sequential          100000      1.876          1.123          12.45       2.57           
```

#### Detailed Table
```cpp
TableFormatter::printDetailedTable(results);
```

Output:
```
Encoder: RLE | Dataset: Sequential | Size: 100000
----------------------------------------------------------------------------
TIMING:
  Encode:        2.345 ms  (42.67 M elem/s)
  Decode (bulk): 1.234 ms  (81.03 M elem/s)
  Random access: 45 ns/read  (min: 32 ns, max: 78 ns)

MEMORY:
  Original:      400000 bytes
  Encoded:       26267 bytes
  Compression:   15.23x
  Bits/element:  2.10

ACCURACY:
  Lossless:      Yes
```

### JSON Output

```cpp
saveBenchmarkResults(results, "benchmark_results.json");
```

JSON structure:
```json
{
  "metadata": {
    "startTime": "2024-01-15 14:30:22",
    "endTime": "2024-01-15 14:35:45",
    "totalDuration_s": 323.45,
    "totalBenchmarks": 90
  },
  "config": {
    "dataSizes": [1000, 10000, 100000],
    "iterations": 10,
    "warmupRuns": 3,
    "randomAccessSamples": 100,
    ...
  },
  "results": [
    {
      "encoderName": "RLE",
      "datasetName": "Sequential",
      "dataSize": 100000,
      "metrics": {
        "timing": {
          "encodeTime_ns": 2345678,
          "decodeBulkTime_ns": 1234567,
          "encodeElementsPerSecond": 42670000,
          ...
        },
        "memory": {
          "originalSize": 400000,
          "encodedSize": 26267,
          "compressionRatio": 15.23,
          "bitsPerElement": 2.10
        },
        ...
      }
    },
    ...
  ]
}
```

## Visualization

Use the included Python script to generate plots:

```bash
python3 Benchmarks/plot_results.py benchmark_results.json -o plots/
```

This generates:
- `compression_ratios.png` - Bar chart of compression for each encoder × dataset
- `encode_throughput.png` - Encoding throughput comparison
- `decode_throughput.png` - Decoding throughput comparison
- `scaling.png` - How performance scales with data size
- `random_access.png` - Random access performance
- `bits_per_element.png` - Storage efficiency

## Advanced Usage

### Composed Encoders

```cpp
// Create composed encoder: Delta -> RLE
auto delta = std::make_shared<DeltaEncoder<int32_t>>();
auto rle = std::make_shared<RunLengthEncoder<int32_t>>();
auto composed = composeEncoders(delta, rle);

runner.registerEncoder("Delta->RLE", composed);
```

The framework will automatically track:
- Individual layer metrics
- Final composed metrics
- Property inheritance

### Custom Data Generators

```cpp
template<typename T>
class MyCustomGenerator : public core::DataGenerator<T> {
public:
    std::vector<T> generate(size_t count) override {
        std::vector<T> data;
        // Your generation logic
        return data;
    }
    
    void reset() override {
        // Reset internal state
    }
};

runner.addDataset("Custom", std::make_shared<MyCustomGenerator<int32_t>>());
```

### Programmatic Result Analysis

```cpp
auto results = runner.runAll();

// Find best compression
double bestRatio = 0.0;
std::string bestEncoder;

for (const auto& result : results.results) {
    double ratio = result.metrics.memory.compressionRatio();
    if (ratio > bestRatio) {
        bestRatio = ratio;
        bestEncoder = result.encoderName;
    }
}

std::cout << "Best: " << bestEncoder << " (" << bestRatio << "x)\n";
```

### Filter Results

```cpp
// Only results for a specific dataset
auto sequential = std::ranges::views::filter(results.results,
    [](const auto& r) { return r.datasetName == "Sequential"; });

// Only large datasets
auto large = std::ranges::views::filter(results.results,
    [](const auto& r) { return r.dataSize >= 100000; });
```

## Building and Running

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Run benchmarks
./bin/run_benchmarks

# Results saved to: Benchmarks/results/benchmark_results.json

# Generate plots
python3 ../Benchmarks/plot_results.py \
    Benchmarks/results/benchmark_results.json \
    -o plots/
```

## Performance Tips

1. **Use Release builds**: `-DCMAKE_BUILD_TYPE=Release` enables optimizations
2. **Sufficient iterations**: Use 10+ iterations for stable statistics
3. **Warmup runs**: 3-5 warmup runs prevent cold-cache effects
4. **System state**: Close other applications, disable CPU frequency scaling
5. **Large datasets**: Use 100k+ elements for meaningful throughput measurements

## Interpreting Results

### Compression Ratio
- **> 10x**: Excellent (RLE on repetitive data, Delta on sequential)
- **3-10x**: Good (Dictionary on low-cardinality)
- **1-3x**: Moderate
- **< 1x**: Expansion (encoder overhead > compression gains)

### Throughput
- **> 100 M elem/s**: Excellent (Raw, simple encoders)
- **10-100 M elem/s**: Good (Most encoders)
- **1-10 M elem/s**: Moderate (Complex encoders, dictionary)
- **< 1 M elem/s**: Poor (Should investigate)

### Random Access
- **< 100 ns**: Excellent (O(1) lookups)
- **100-1000 ns**: Good (Binary search, small overhead)
- **> 1000 ns**: Poor (May need optimization)

## Troubleshooting

**Q: Validation failures?**
- Check encoder correctness with `test_encoders`
- Ensure data type matches encoder constraints
- Verify composed encoders are compatible

**Q: Inconsistent timings?**
- Increase warmup runs
- Close background applications
- Check for thermal throttling

**Q: Memory metrics seem wrong?**
- Verify `sizeof(T)` matches data type
- Check for memory alignment overhead
- Review encoder format specifications

**Q: Random access shows 0 ns?**
- Encoder doesn't support random access
- Check `properties().has(EncodingProperty::RandomAccess)`

## Next Steps

- Implement additional encoders (BitPacking, Frame-of-Reference, etc.)
- Add multi-threaded benchmarking
- Support composite types (arrays, maps)
- Add statistical significance testing
- Create automated regression testing
