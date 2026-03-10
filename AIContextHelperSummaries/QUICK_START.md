# Quick Start Guide

## Building the Project

```bash
# Create build directory
mkdir build && cd build

# Configure (use Release for accurate benchmarks!)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build everything
cmake --build .

# Or build just the benchmarks
cmake --build . --target run_benchmarks
```

## Running Benchmarks

```bash
# Run the benchmark suite
./bin/run_benchmarks

# Results are automatically saved to:
# - Console output (summary tables)
# - Benchmarks/results/benchmark_results.json
```

## Generating Plots

```bash
# Install Python dependencies
pip install -r ../Benchmarks/requirements.txt

# Generate plots from benchmark results
python3 ../Benchmarks/plot_results.py \
    Benchmarks/results/benchmark_results.json \
    -o plots/

# Plots will be saved in the plots/ directory:
# - compression_ratios.png
# - encode_throughput.png
# - decode_throughput.png
# - scaling.png
# - random_access.png
# - bits_per_element.png
```

## Example Output

After running benchmarks, you'll see:

```
=== Encoding Playground Benchmark Suite ===

Running 72 benchmark configurations...
[1/72] Benchmarking: Raw on Sequential (n=1000)
...
[72/72] Benchmarking: Dictionary on Constant (n=100000)

Completed 72 benchmarks in 0.29 seconds

========================================================
BENCHMARK SUMMARY
========================================================
Encoder    Dataset      Size    Enc(ms)  Dec(ms)  Ratio  Bits/Elem
--------------------------------------------------------
Raw        Sequential   100k    0.075    0.047    1.00   32.00
RLE        Constant     100k    0.037    0.053    0.00   0.00  (!!)
Delta      Sequential   100k    0.155    0.191    1.00   32.00
Dictionary Repetitive   100k    0.236    0.153    0.25   8.02
...
```

## Interpreting Results

### Compression Ratio
- **RLE on Constant data**: ~1000x compression!
- **Dictionary on low-cardinality**: ~4-12x compression
- **Delta on random data**: ~1x (no benefit)

### Throughput
- **Raw encoder**: Fastest (5-7 billion elements/second)
- **RLE/Delta**: Fast (500M-2B elements/second)
- **Dictionary**: Moderate (100M-500M elements/second)

### Best Use Cases
- **Sequential/monotonic data**: Delta encoder
- **Repetitive data**: RLE encoder  
- **Low cardinality**: Dictionary encoder
- **Random data**: Consider no compression

## Documentation

- **[BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md)** - Comprehensive benchmarking documentation
- **[ENCODERS_GUIDE.md](ENCODERS_GUIDE.md)** - Encoder implementation details
- **[COMPOSITION_GUIDE.md](COMPOSITION_GUIDE.md)** - Encoder composition
- **[README.md](README.md)** - Project overview

## Customizing Benchmarks

Edit `Benchmarks/run_benchmarks.cpp` to:

1. **Add/remove encoders**:
```cpp
runner.registerEncoder("MyEncoder", 
    std::make_shared<MyEncoder<int32_t>>());
```

2. **Add/remove datasets**:
```cpp
runner.addDataset("MyData",
    std::make_shared<MyGenerator<int32_t>>());
```

3. **Change benchmark configuration**:
```cpp
BenchmarkConfig config;
config.dataSizes = {10000, 100000, 1000000};  // Larger sizes
config.iterations = 20;                        // More iterations
config.randomAccessSamples = 500;              // More samples
```

4. **Rebuild and run**:
```bash
cmake --build . --target run_benchmarks
./bin/run_benchmarks
```

## Testing Encoders

```bash
# Run the encoder tests
./Source/encoders/test_encoders

# All tests should pass:
✓ RawEncoder correctness tests passed
✓ RunLengthEncoder correctness tests passed  
✓ DeltaEncoder correctness tests passed
✓ DictionaryEncoder correctness tests passed
```

## Next Steps

1. **Implement your own encoder** - See ENCODERS_GUIDE.md
2. **Create custom data generators** - See BENCHMARKING_GUIDE.md
3. **Analyze results** - Use the Python plots
4. **Optimize** - Profile and improve encoders
5. **Share results** - JSON output is portable

## Troubleshooting

**Q: Benchmarks are too slow?**
- Reduce `dataSizes` in config
- Reduce `iterations` and `randomAccessSamples`
- Build in Release mode (`-DCMAKE_BUILD_TYPE=Release`)

**Q: Results are inconsistent?**
- Increase `warmupRuns` (default: 3)
- Close other applications
- Check for thermal throttling

**Q: Plots don't generate?**
- Install matplotlib: `pip install matplotlib numpy`
- Check that JSON file exists
- Verify Python 3.x is installed

**Q: Compilation errors?**
- Ensure C++23 compiler (GCC 13+, Clang 16+)
- Check CMake version (3.28+)
- Clean build: `rm -rf build && mkdir build`

## Performance Tips

1. **Always use Release builds** for benchmarking
2. **Pin CPU frequency** to avoid throttling
3. **Run multiple times** and average results
4. **Test on representative data** from your use case
5. **Monitor system resources** during benchmarks

Happy benchmarking! 🚀
