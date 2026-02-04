# EncodingsPlayground

A C++23 playground for benchmarking encoding schemes across various data types and structures.

## Overview

This project provides a flexible framework for:
- Testing different encoding strategies (delta, run-length, dictionary, bit-packing, etc.)
- Measuring write speed, compression ratio, and read performance (bulk + random access)
- Supporting primitives (8-64 bit ints, 32-64 bit floats, strings, bools) and composites (arrays, maps)
- Generating test data with various distributions and patterns

## Architecture

### Core Components

1. **Type System** (`Source/core/DataType.hpp`)
   - Enumeration of supported data types
   - Type concepts and traits for compile-time checking
   - Type-to-enum mappings

2. **Encoding Interface** (`Source/encodings/Encoder.hpp`)
   - `Encoder<T>`: Abstract interface for encoding data
   - `Decoder<T>`: Abstract interface for decoding data
   - `Codec<T>`: Combined encoder/decoder for convenience

3. **Data Generation** (`Source/generators/DataGenerator.hpp`)
   - `DataGenerator<T>`: Abstract interface for generating test data
   - Configurable generation strategies (uniform, normal, zipfian, etc.)

4. **Benchmark Framework** (`Source/benchmark/BenchmarkMetrics.hpp`)
   - `BenchmarkMetrics`: Comprehensive performance measurements
   - `BenchmarkConfig`: Configuration for benchmark runs
   - Timing, memory, accuracy, and random access metrics

5. **Encoded Data** (`Source/encodings/EncodedData.hpp`)
   - Container for encoded bytes with metadata
   - Compression ratio calculations
   - Encoding scheme information

## Building

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Build Options

- `-DBUILD_BENCHMARKS=ON/OFF` - Build benchmark executables (default: ON)
- `-DBUILD_TESTS=ON/OFF` - Build unit tests (default: ON)
- `-DENABLE_SANITIZERS=ON/OFF` - Enable ASan/UBSan (default: OFF)
- `-DCMAKE_BUILD_TYPE=Release` - Enable optimizations for accurate benchmarking

## Requirements

- C++23 compatible compiler (GCC 13+, Clang 16+, MSVC 19.36+)
- CMake 3.28+

## Project Structure

```
EncodingsPlayground/
├── Source/
│   ├── core/              # Core types (DataType, concepts)
│   ├── encodings/         # Encoding/decoding interfaces, properties, composition
│   ├── generators/        # Data generator interface and implementations
│   ├── encoders/          # Concrete encoding implementations
│   │   ├── RawEncoder.hpp           # Baseline (no compression)
│   │   ├── RunLengthEncoder.hpp     # RLE for repetitive data
│   │   ├── DeltaEncoder.hpp         # Delta for monotonic data
│   │   ├── DictionaryEncoder.hpp    # Dictionary for low cardinality
│   │   └── test_encoders.cpp        # Test suite
│   └── benchmark/         # Benchmarking framework and metrics
│       ├── BenchmarkMetrics.hpp     # Metrics structures
│       ├── BenchmarkRunner.hpp      # Main benchmarking engine
│       └── BenchmarkOutput.hpp      # JSON and table formatters
├── Benchmarks/
│   ├── run_benchmarks.cpp # Example benchmark suite
│   ├── plot_results.py    # Python plotting script
│   └── results/           # Benchmark output data
├── Tests/                 # Unit tests (TODO)
├── BENCHMARKING_GUIDE.md  # Detailed benchmarking documentation
├── COMPOSITION_GUIDE.md   # Guide to encoder composition
├── ENCODERS_GUIDE.md      # Encoder implementations guide
└── CMakeLists.txt
```

## Quick Start: Running Benchmarks

```bash
# Build in release mode (important for accurate measurements!)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Run the benchmark suite
./bin/run_benchmarks

# Results saved to: Benchmarks/results/benchmark_results.json

# Generate plots (requires matplotlib)
cd ..
python3 Benchmarks/plot_results.py \
    build/Benchmarks/results/benchmark_results.json \
    -o plots/
```

See **[BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md)** for comprehensive documentation.

## Extensibility

The framework is designed for easy extension:

### Using the Implemented Encoders

```cpp
#include "encoders/DeltaEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/DictionaryEncoder.hpp"

// Time series data
DeltaEncoder<int64_t> deltaEnc;
std::vector<int64_t> timestamps = {1000, 1001, 1002, 1003};
auto encoded = deltaEnc.encode(timestamps);

// Repetitive data
RunLengthEncoder<int32_t> rleEnc;
std::vector<int32_t> data = {1, 1, 1, 2, 2, 2, 3, 3};
auto encoded2 = rleEnc.encode(data);

// Low cardinality data
DictionaryEncoder<std::string> dictEnc;
std::vector<std::string> categories = {"A", "B", "A", "C", "B"};
auto encoded3 = dictEnc.encode(categories);
```

### Adding a New Encoder

```cpp
#include "encodings/Encoder.hpp"

template<typename T>
class MyCustomEncoder : public encodings::Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
        // Your encoding logic
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        // Your decoding logic
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        // Random access logic
    }
    
    std::string name() const override { return "MyCustomEncoder"; }
    bool supportsRandomAccess() const override { return true; }
};
```

### Adding a New Generator

```cpp
#include "generators/DataGenerator.hpp"

template<typename T>
class MyCustomGenerator : public encodings::core::DataGenerator<T> {
public:
    std::vector<T> generate(size_t count) override {
        // Your generation logic
    }
    
    std::string name() const override { return "MyCustomGenerator"; }
    void reset() override { /* Reset state */ }
};
```

## License

TBD

## Author

David C-L
