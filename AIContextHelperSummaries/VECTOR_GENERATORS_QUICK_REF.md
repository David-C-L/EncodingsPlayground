# Vector Generators - Quick Reference

## Summary

Three comprehensive data generators for testing vector compression:

1. **UnitVectorGenerator** - Unit vectors (magnitude = 1.0)
2. **NonUnitVectorGenerator** - Arbitrary magnitude vectors
3. **PatternedVectorGenerator** - Special patterns for edge cases

## Quick Start

### Unit Vectors
```cpp
#include "generators/VectorGenerator.hpp"

// 3D unit vectors
auto gen = UnitVectorGenerator<std::array<float, 3>, 3>(42);
auto vectors = gen.generate(10000);

// Test with spherical encoder
auto encoder = SphericalEncoder<std::array<float, 3>, 3>(
    std::make_shared<ZstdEncoder>()
);
auto encoded = encoder.encode(vectors);
```

### Non-Unit Vectors
```cpp
// Vectors with magnitudes in [0.5, 5.0]
auto gen = NonUnitVectorGenerator<std::array<float, 3>, 3>(
    0.5f,   // min magnitude
    5.0f,   // max magnitude  
    42,     // seed
    NonUnitVectorGenerator<...>::MagnitudeDistribution::Uniform
);

auto vectors = gen.generate(10000);

// Test with and without normalization
auto encoder1 = SphericalEncoder<...>(..., nullptr, false);  // No normalize
auto encoder2 = SphericalEncoder<...>(..., codec, true);     // Normalize
```

### Patterned Vectors
```cpp
// Clustered vectors
auto gen = PatternedVectorGenerator<std::array<float, 3>, 3>(
    PatternedVectorGenerator<...>::Pattern::Clustered,
    42,     // seed
    0.1f    // cluster radius
);

auto vectors = gen.generate(10000);
```

## Generator Comparison

| Generator | Normalized | Magnitude | Use Case |
|-----------|-----------|-----------|----------|
| UnitVector | ✓ Yes | Always 1.0 | Embeddings, directional data |
| NonUnit | ✗ No | Configurable | General vectors, testing normalization |
| Patterned | ✓ Yes | Always 1.0 | Edge cases, compression analysis |

## UnitVectorGenerator

**Template**: `UnitVectorGenerator<T, Dimension>`

**Constructor**:
```cpp
UnitVectorGenerator(
    uint32_t seed = 42,
    const std::string& method = "gaussian"  // "gaussian" or "rejection"
)
```

**Methods**:
- `generate(size_t count)` - Generate vectors
- `reset()` - Reset RNG state
- `name()` - Get generator name
- `getConfig()` - Get configuration

**Best For**:
- Text embeddings (OpenAI, BERT, etc.)
- Normalized feature vectors
- Directional data (angles, orientations)
- Spherical encoding tests

## NonUnitVectorGenerator

**Template**: `NonUnitVectorGenerator<T, Dimension>`

**Constructor**:
```cpp
NonUnitVectorGenerator(
    float minMagnitude = 0.1f,
    float maxMagnitude = 10.0f,
    uint32_t seed = 42,
    MagnitudeDistribution dist = MagnitudeDistribution::Uniform
)
```

**Magnitude Distributions**:
- `Uniform` - Uniform distribution in [min, max]
- `LogNormal` - Log-normal (realistic, common in nature)
- `Exponential` - Exponential decay
- `Fixed` - All vectors same magnitude

**Best For**:
- Testing normalization effectiveness
- General vector data (not normalized)
- Magnitude codec testing
- Real-world data simulation

## PatternedVectorGenerator

**Template**: `PatternedVectorGenerator<T, Dimension>`

**Constructor**:
```cpp
PatternedVectorGenerator(
    Pattern pattern,
    uint32_t seed = 42,
    float clusterRadius = 0.1f
)
```

**Patterns**:
- `Clustered` - Vectors in small regions (high compression)
- `Sparse` - Many zero components
- `CoordinateAligned` - Aligned with axes
- `Repeated` - Same vectors repeated (maximum compression)
- `Smooth` - Smoothly varying (temporal coherence)

**Best For**:
- Edge case testing
- Worst/best case analysis
- Algorithm validation
- Pattern-specific optimization

## Common Tasks

### Compare Dimensions
```cpp
template<size_t D>
void test() {
    auto gen = UnitVectorGenerator<std::array<float, D>, D>(42);
    auto vectors = gen.generate(10000);
    auto encoder = SphericalEncoder<std::array<float, D>, D>(...);
    auto encoded = encoder.encode(vectors);
    // Measure compression...
}

test<3>();
test<16>();
test<128>();
test<1536>();
```

### Test Normalization
```cpp
auto gen = NonUnitVectorGenerator<Vec, D>(0.5f, 5.0f, 42);
auto vectors = gen.generate(10000);

// Without normalization
auto encoded1 = encoder1.encode(vectors);

// With normalization
auto encoded2 = encoder2.encode(vectors);

// Compare ratios
```

### Benchmark Reproducibility
```cpp
auto gen = UnitVectorGenerator<Vec, D>(42);  // Fixed seed

// Run 1
auto vectors1 = gen.generate(10000);
auto result1 = benchmark(vectors1);

// Run 2 - reset for same data
gen.reset();
auto vectors2 = gen.generate(10000);
auto result2 = benchmark(vectors2);

assert(result1 == result2);  // Reproducible!
```

### Pattern Analysis
```cpp
std::map<std::string, float> results;

for (auto pattern : {Clustered, Sparse, Repeated, Smooth}) {
    auto gen = PatternedVectorGenerator<Vec, D>(pattern, 42);
    auto vectors = gen.generate(10000);
    auto encoded = encoder.encode(vectors);
    results[patternName] = compressionRatio;
}

// Analyze which patterns compress best
```

## Expected Compression Ratios

### Unit Vectors (with SphericalEncoder)
| Dimension | Typical Ratio | Notes |
|-----------|--------------|-------|
| 3D | 2.0-2.5x | Small dimensions |
| 16D | 2.5-3.0x | Medium dimensions |
| 128D | 3.0-4.0x | High dimensions |
| 1536D | 3.5-4.5x | Very high dimensions |

### Non-Unit Vectors
- **Without normalization**: 1.5-2.5x (depends on magnitude distribution)
- **With normalization**: Similar to unit vectors + magnitude overhead

### Patterned Vectors
- **Clustered**: 3.0-5.0x (similar angles)
- **Sparse**: 2.0-3.0x (many zeros)
- **CoordinateAligned**: 4.0-6.0x (structured)
- **Repeated**: 10.0-20.0x (extreme redundancy)
- **Smooth**: 3.5-5.5x (temporal coherence)

## Files

```
Source/generators/
├── DataGenerator.hpp       # Base class
└── VectorGenerator.hpp     # Vector generators

Tests/
├── test_vector_generators.cpp      # Comprehensive tests
└── VECTOR_GENERATORS_GUIDE.md      # Full documentation
```

## Building and Running

```bash
cd build
cmake ..
make test_vector_generators

./bin/test_vector_generators
```

## Key Features

✅ **Type-safe** - Compile-time dimension checking  
✅ **Reproducible** - Fixed seeds for consistent results  
✅ **Flexible** - Multiple distributions and patterns  
✅ **Fast** - Efficient generation (millions of vectors/sec)  
✅ **Well-documented** - Clear usage examples  
✅ **Tested** - Validated statistical properties  

## Integration with SphericalEncoder

Perfect combination:
```cpp
// Generate data
auto gen = UnitVectorGenerator<std::array<float, 128>, 128>(42);
auto vectors = gen.generate(10000);

// Encode
auto encoder = SphericalEncoder<std::array<float, 128>, 128>(
    std::make_shared<ZstdEncoder>()
);
auto encoded = encoder.encode(vectors);

// Decode and verify
auto decoded = encoder.decodeAll(encoded);

// Measure
size_t originalSize = vectors.size() * 128 * sizeof(float);
float ratio = (float)originalSize / encoded.size();
float maxError = computeMaxError(vectors, decoded);

std::cout << "Compression: " << ratio << "x\n";
std::cout << "Error: " << maxError << "\n";
```

## Summary

The vector generators provide everything needed to:
- ✅ Test spherical encoder performance
- ✅ Compare compression strategies  
- ✅ Validate accuracy
- ✅ Benchmark different dimensions
- ✅ Analyze edge cases
- ✅ Reproduce results

Use `UnitVectorGenerator` for most tests, `NonUnitVectorGenerator` to test normalization, and `PatternedVectorGenerator` for edge cases!
