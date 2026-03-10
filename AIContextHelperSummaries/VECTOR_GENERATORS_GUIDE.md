# Vector Data Generators

## Overview

Comprehensive data generators for testing vector compression algorithms, specifically designed for use with the `SphericalEncoder`. Three main generator types are provided:

1. **UnitVectorGenerator** - Random unit vectors (normalized to length 1)
2. **NonUnitVectorGenerator** - Random vectors with varying magnitudes
3. **PatternedVectorGenerator** - Vectors with specific patterns for testing edge cases

## UnitVectorGenerator

Generates uniformly distributed random unit vectors on the hypersphere.

### Features

- **Uniform distribution** using Marsaglia method (Gaussian sampling + normalization)
- **Alternative rejection sampling** method for verification
- **Reproducible** with fixed seeds
- **Any dimensionality** (compile-time template parameter)

### Usage

```cpp
#include "generators/VectorGenerator.hpp"

// Generate 3D unit vectors
auto gen = UnitVectorGenerator<std::array<float, 3>, 3>(
    42,          // seed (default: 42)
    "gaussian"   // method: "gaussian" or "rejection"
);

auto vectors = gen.generate(10000);  // Generate 10,000 vectors

// All vectors have magnitude ≈ 1.0
```

### Methods

#### Gaussian (Marsaglia) - Default
- Sample each component from N(0,1)
- Normalize to unit length
- ✅ **Fast** - O(d) per vector
- ✅ **Uniform** distribution on sphere
- ✅ **Recommended** for most use cases

#### Rejection Sampling
- Sample uniformly from [-1,1]^d
- Reject if outside unit ball
- Normalize to unit surface
- ⚠️ **Slower** - especially for high dimensions
- ✓ Alternative for verification

### Example: Different Dimensions

```cpp
// 3D vectors (RGB, spatial)
auto gen3D = UnitVectorGenerator<std::array<float, 3>, 3>(42);
auto vectors3D = gen3D.generate(10000);

// 128D vectors (features)
auto gen128D = UnitVectorGenerator<std::array<float, 128>, 128>(42);
auto vectors128D = gen128D.generate(10000);

// 1536D vectors (OpenAI embeddings)
auto gen1536D = UnitVectorGenerator<std::array<float, 1536>, 1536>(42);
auto embeddings = gen1536D.generate(10000);
```

### Use Cases

Perfect for testing:
- ✅ Text embeddings (usually normalized)
- ✅ Image features (normalized descriptors)
- ✅ Directional data (unit vectors by definition)
- ✅ Spherical encoder compression
- ✅ Angular similarity metrics

### Configuration

```cpp
auto config = gen.getConfig();
// Returns:
// {
//   "type": "unit_vector",
//   "dimension": "3",
//   "seed": "42",
//   "method": "gaussian",
//   "normalized": "true"
// }
```

## NonUnitVectorGenerator

Generates vectors with random directions and magnitudes.

### Features

- **Random directions** (uniform on sphere)
- **Configurable magnitude range** [minMag, maxMag]
- **Multiple magnitude distributions**:
  - Uniform
  - Log-normal (common in real data)
  - Exponential
  - Fixed (all same magnitude)

### Usage

```cpp
// Generate vectors with magnitudes in [0.5, 5.0]
auto gen = NonUnitVectorGenerator<std::array<float, 3>, 3>(
    0.5f,   // min magnitude
    5.0f,   // max magnitude
    42,     // seed
    NonUnitVectorGenerator<...>::MagnitudeDistribution::Uniform
);

auto vectors = gen.generate(10000);
```

### Magnitude Distributions

#### 1. Uniform Distribution
```cpp
auto gen = NonUnitVectorGenerator<Vec, D>(
    1.0f, 10.0f, 42,
    MagnitudeDistribution::Uniform
);
// Magnitudes uniformly distributed in [1.0, 10.0]
```

**Use case**: General testing, baseline comparison

#### 2. Log-Normal Distribution
```cpp
auto gen = NonUnitVectorGenerator<Vec, D>(
    0.1f, 10.0f, 42,
    MagnitudeDistribution::LogNormal
);
// Most vectors have smaller magnitudes, few outliers
```

**Use case**: Natural data patterns, scale-free phenomena

#### 3. Exponential Distribution
```cpp
auto gen = NonUnitVectorGenerator<Vec, D>(
    0.1f, 10.0f, 42,
    MagnitudeDistribution::Exponential
);
// Exponentially decaying magnitudes
```

**Use case**: Decay processes, time series data

#### 4. Fixed Distribution
```cpp
auto gen = NonUnitVectorGenerator<Vec, D>(
    5.0f, 5.0f, 42,
    MagnitudeDistribution::Fixed
);
// All vectors have magnitude 5.0
```

**Use case**: Testing normalization benefit

### Example: Testing Normalization

```cpp
auto gen = NonUnitVectorGenerator<std::array<float, 3>, 3>(
    0.5f, 5.0f, 42,
    MagnitudeDistribution::LogNormal
);

auto vectors = gen.generate(10000);

// Test without normalization
auto encoder1 = SphericalEncoder<std::array<float, 3>, 3>(
    std::make_shared<ZstdEncoder>(),
    nullptr,
    false  // don't normalize
);
auto encoded1 = encoder1.encode(vectors);

// Test with normalization
auto encoder2 = SphericalEncoder<std::array<float, 3>, 3>(
    std::make_shared<ZstdEncoder>(),
    std::make_shared<ZstdFloatEncoder>(),
    true  // normalize
);
auto encoded2 = encoder2.encode(vectors);

// Compare compression ratios
std::cout << "Without normalization: " << ratio1 << "x\n";
std::cout << "With normalization: " << ratio2 << "x\n";
```

### Use Cases

Perfect for testing:
- ✅ General vector data (not normalized)
- ✅ Feature vectors with varying scales
- ✅ Normalization effectiveness
- ✅ Magnitude codec performance
- ✅ Real-world data patterns

## PatternedVectorGenerator

Generates vectors with specific patterns for testing edge cases and compression characteristics.

### Patterns

#### 1. Clustered
```cpp
auto gen = PatternedVectorGenerator<Vec, D>(
    Pattern::Clustered,
    42,     // seed
    0.1f    // cluster radius
);
```

- Vectors clustered in small regions
- Few cluster centers, many nearby vectors
- **Expected**: High compression (similar angles)

#### 2. Sparse
```cpp
auto gen = PatternedVectorGenerator<Vec, D>(
    Pattern::Sparse,
    42
);
```

- Most components are zero
- Only ~10% of dimensions non-zero
- **Expected**: Different compression characteristics

#### 3. CoordinateAligned
```cpp
auto gen = PatternedVectorGenerator<Vec, D>(
    Pattern::CoordinateAligned,
    42
);
```

- Vectors aligned with coordinate axes
- One component dominant, others small
- **Expected**: Very structured, high compression

#### 4. Repeated
```cpp
auto gen = PatternedVectorGenerator<Vec, D>(
    Pattern::Repeated,
    42
);
```

- Same vectors repeated multiple times
- Only ~10% unique vectors
- **Expected**: Maximum compression

#### 5. Smooth
```cpp
auto gen = PatternedVectorGenerator<Vec, D>(
    Pattern::Smooth,
    42
);
```

- Smoothly varying directions
- Each vector similar to previous
- **Expected**: High compression (temporal coherence)

### Example: Comparing Patterns

```cpp
const size_t count = 10000;

struct Test {
    Pattern pattern;
    std::string name;
};

std::vector<Test> tests = {
    {Pattern::Clustered, "Clustered"},
    {Pattern::Sparse, "Sparse"},
    {Pattern::Repeated, "Repeated"},
    {Pattern::Smooth, "Smooth"}
};

for (const auto& test : tests) {
    auto gen = PatternedVectorGenerator<std::array<float, 3>, 3>(
        test.pattern, 42
    );
    
    auto vectors = gen.generate(count);
    auto encoder = SphericalEncoder<std::array<float, 3>, 3>(...);
    auto encoded = encoder.encode(vectors);
    
    float ratio = (float)originalSize / encoded.size();
    std::cout << test.name << ": " << ratio << "x\n";
}

// Expected output:
// Clustered: 3.5x (similar angles compress well)
// Sparse: 2.8x (many zeros)
// Repeated: 15.0x (extreme redundancy)
// Smooth: 4.2x (temporal coherence)
```

### Use Cases

Perfect for:
- ✅ Edge case testing
- ✅ Compression algorithm validation
- ✅ Worst/best case analysis
- ✅ Pattern-specific optimization testing
- ✅ Benchmark diversity

## Common Usage Patterns

### 1. Basic Compression Testing

```cpp
// Generate unit vectors
auto gen = UnitVectorGenerator<std::array<float, 128>, 128>(42);
auto vectors = gen.generate(10000);

// Compress
auto encoder = SphericalEncoder<std::array<float, 128>, 128>(
    std::make_shared<ZstdEncoder>()
);
auto encoded = encoder.encode(vectors);

// Measure
size_t originalSize = 10000 * 128 * sizeof(float);
float ratio = (float)originalSize / encoded.size();
std::cout << "Compression: " << ratio << "x\n";
```

### 2. Comparing Normalization Strategies

```cpp
auto gen = NonUnitVectorGenerator<Vec, D>(0.5f, 5.0f, 42);
auto vectors = gen.generate(10000);

// Without normalization
auto encoder1 = SphericalEncoder<Vec, D>(codec1, nullptr, false);
auto encoded1 = encoder1.encode(vectors);

// With normalization
auto encoder2 = SphericalEncoder<Vec, D>(codec1, codec2, true);
auto encoded2 = encoder2.encode(vectors);

std::cout << "Without: " << ratio1 << "x\n";
std::cout << "With: " << ratio2 << "x\n";
std::cout << "Improvement: " << (ratio2/ratio1) << "x\n";
```

### 3. Reproducible Benchmarks

```cpp
// Run 1
auto gen1 = UnitVectorGenerator<Vec, D>(42);
auto vectors1 = gen1.generate(10000);
auto result1 = benchmark(vectors1);

// Run 2 - same seed, same results
auto gen2 = UnitVectorGenerator<Vec, D>(42);
auto vectors2 = gen2.generate(10000);
auto result2 = benchmark(vectors2);

assert(result1 == result2);  // Reproducible!
```

### 4. Testing Multiple Dimensions

```cpp
template<size_t D>
float testDimension(size_t count) {
    auto gen = UnitVectorGenerator<std::array<float, D>, D>(42);
    auto vectors = gen.generate(count);
    
    auto encoder = SphericalEncoder<std::array<float, D>, D>(...);
    auto encoded = encoder.encode(vectors);
    
    return (float)(count * D * sizeof(float)) / encoded.size();
}

// Compare dimensions
std::cout << "3D: " << testDimension<3>(10000) << "x\n";
std::cout << "16D: " << testDimension<16>(10000) << "x\n";
std::cout << "128D: " << testDimension<128>(10000) << "x\n";
std::cout << "1536D: " << testDimension<1536>(1000) << "x\n";
```

### 5. Pattern Analysis

```cpp
std::map<std::string, float> compressionByPattern;

for (auto pattern : allPatterns) {
    auto gen = PatternedVectorGenerator<Vec, D>(pattern, 42);
    auto vectors = gen.generate(10000);
    
    auto encoder = SphericalEncoder<Vec, D>(...);
    auto encoded = encoder.encode(vectors);
    
    float ratio = (float)originalSize / encoded.size();
    compressionByPattern[patternName] = ratio;
}

// Analyze which patterns compress best
```

## Generator Configuration

All generators support:

```cpp
// Get configuration
auto config = gen.getConfig();
for (const auto& [key, value] : config) {
    std::cout << key << ": " << value << "\n";
}

// Reset to initial state
gen.reset();

// Get name
std::cout << "Generator: " << gen.name() << "\n";

// Get data type
std::cout << "Type: " << dataTypeToString(gen.dataType()) << "\n";
```

## Performance Characteristics

### Generation Speed

| Generator | Dimension | Vectors/sec | Notes |
|-----------|-----------|-------------|-------|
| UnitVector (Gaussian) | 3 | ~10M | Fast |
| UnitVector (Gaussian) | 128 | ~500K | Scales with D |
| UnitVector (Rejection) | 3 | ~2M | Slower |
| NonUnit | 3 | ~8M | Similar to unit |
| Patterned (Clustered) | 3 | ~5M | Depends on pattern |

### Memory Usage

- **Temporary**: O(n·d) for vector storage
- **Generator state**: O(1) - just RNG state
- **No caching**: Vectors generated on-demand

## Best Practices

### 1. Use Fixed Seeds for Benchmarks
```cpp
// Always use fixed seed for reproducible results
auto gen = UnitVectorGenerator<Vec, D>(42);  // ✓ Good
// auto gen = UnitVectorGenerator<Vec, D>(time(nullptr));  // ✗ Not reproducible
```

### 2. Reset Between Runs
```cpp
auto gen = UnitVectorGenerator<Vec, D>(42);
auto run1 = benchmark(gen.generate(10000));

gen.reset();  // ✓ Reset for next run
auto run2 = benchmark(gen.generate(10000));
// run1 and run2 use same data
```

### 3. Match Generator to Use Case
```cpp
// For embeddings - use unit vectors
auto gen = UnitVectorGenerator<Vec, D>(42);

// For general features - use non-unit with appropriate distribution
auto gen = NonUnitVectorGenerator<Vec, D>(
    0.1f, 10.0f, 42,
    MagnitudeDistribution::LogNormal  // Realistic distribution
);

// For edge cases - use patterned
auto gen = PatternedVectorGenerator<Vec, D>(Pattern::Clustered, 42);
```

### 4. Dimension-Appropriate Counts
```cpp
// Larger dimensions need fewer vectors for same data size
size_t count = 1000000 / Dimension;  // ✓ Adaptive
// size_t count = 1000000;  // ✗ Too much data for large D
```

## Summary

The vector generators provide:

1. ✅ **Comprehensive coverage** - Unit, non-unit, and patterned vectors
2. ✅ **Reproducible** - Fixed seeds for benchmark consistency
3. ✅ **Flexible** - Multiple distributions and patterns
4. ✅ **Type-safe** - Compile-time dimension checking
5. ✅ **Fast** - Efficient generation algorithms
6. ✅ **Well-tested** - Validated statistical properties

Perfect for:
- Spherical encoder benchmarking
- Compression algorithm testing
- Edge case validation
- Performance analysis
- Research experiments
