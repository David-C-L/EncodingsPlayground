# TPC-H Lineitem L_PARTKEY Generator

## Overview

Generates realistic `L_PARTKEY` values for the TPC-H lineitem table, following the TPC-H benchmark specification.

## TPC-H Background

In TPC-H:
- The **lineitem** table is the largest table (contains order line items)
- **L_PARTKEY** is a foreign key referencing the **part** table's primary key
- Cardinality: `SF × 200,000` unique part keys (where SF = scale factor)
- Type: 32-bit integer identifier
- Range: `[1, SF × 200,000]`

## Generators

### 1. TPCHLineitemPartKeyGenerator

Basic generator with configurable distribution patterns.

**Features:**
- Uniform distribution (default)
- Zipfian distribution (models popular items)
- Configurable scale factor
- Reproducible (seeded)

**Constructor:**
```cpp
TPCHLineitemPartKeyGenerator(
    double scaleFactor = 1.0,     // TPC-H scale factor
    int64_t seed = 42,             // Random seed
    double zipfExponent = 0.3      // Skew (0=uniform, 1.0=Zipfian)
)
```

**Example Usage:**
```cpp
#include "generators/TPCHLineitemPartKeyGenerator.hpp"

// Uniform distribution, SF=1.0 (200K parts)
TPCHLineitemPartKeyGenerator gen1(1.0, 42, 0.0);
auto uniformKeys = gen1.generate(1000000);

// Realistic skew, SF=1.0
TPCHLineitemPartKeyGenerator gen2(1.0, 42, 0.3);
auto skewedKeys = gen2.generate(1000000);

// Small scale for testing, SF=0.01 (2K parts)
TPCHLineitemPartKeyGenerator gen3(0.01, 42, 0.0);
auto testKeys = gen3.generate(10000);
```

### 2. TPCHLineitemPartKeyClusteredGenerator

Generator with temporal clustering to simulate real-world access patterns where certain parts are popular during specific periods.

**Features:**
- Temporal clustering (hot parts change over time)
- Configurable cluster size
- 60% access to "hot" parts, 40% to any part
- 70% from current cluster, 30% random

**Constructor:**
```cpp
TPCHLineitemPartKeyClusteredGenerator(
    double scaleFactor = 1.0,      // TPC-H scale factor
    int64_t seed = 42,              // Random seed
    size_t clusterSize = 100,       // Items per cluster
    size_t numHotParts = 1000       // Number of "hot" parts
)
```

**Example Usage:**
```cpp
// Clustered access pattern
TPCHLineitemPartKeyClusteredGenerator gen(1.0, 42, 100, 1000);
auto clusteredKeys = gen.generate(1000000);
```

## Distribution Characteristics

### Uniform (zipfExponent = 0.0)
```
10,000 values, SF=0.01 (2,000 parts):
  Unique values: ~1,991 (99.5%)
  Most frequent: ~14 times (0.14%)
```
**Best for:** Testing, baseline performance

### Mild Zipfian (zipfExponent = 0.3)
```
10,000 values, SF=1.0 (200,000 parts):
  Unique values: ~9,714 (97%)
  Most frequent: ~4 times (0.04%)
```
**Best for:** Realistic TPC-H workloads

### Strong Zipfian (zipfExponent = 1.0)
```
10,000 values, SF=1.0 (200,000 parts):
  Unique values: ~4,703 (47%)
  Top item: 777 times (7.77%)
  Top 5: ~18% of all accesses
```
**Best for:** Modeling highly skewed workloads
**Note:** Slow for large cardinalities due to inverse transform sampling

### Clustered
```
10,000 values, SF=1.0 (200,000 parts):
  Unique values: ~2,022 (20%)
  Top 5: ~1.4% of accesses
  Temporal locality: ~70% from recent clusters
```
**Best for:** Simulating time-series or session-based access patterns

## Scale Factors

| Scale Factor | Parts | Database Size | Use Case |
|--------------|-------|---------------|----------|
| 0.01 | 2,000 | ~10 MB | Quick testing |
| 0.1 | 20,000 | ~100 MB | Development |
| 1.0 | 200,000 | ~1 GB | Standard benchmark |
| 10.0 | 2,000,000 | ~10 GB | Large-scale testing |

## Performance Considerations

- **Uniform & Clustered**: O(1) per value
- **Zipfian (zipf < 0.5)**: O(cardinality) per value (fast for SF ≤ 1.0)
- **Zipfian (zipf ≥ 1.0)**: O(cardinality) per value (slow for large SF)

For large-scale Zipfian generation, consider:
1. Pre-generate a lookup table
2. Use rejection sampling
3. Use the clustered generator instead

## Integration with Encoders

```cpp
#include "generators/TPCHLineitemPartKeyGenerator.hpp"
#include "encoders/DictionaryEncoder.hpp"
#include "encoders/DeltaEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"

// Generate data
TPCHLineitemPartKeyGenerator gen(1.0, 42, 0.3);
auto partKeys = gen.generate(1000000);

// Test with different encoders
DictionaryEncoder<int32_t> dictEncoder;
auto dictEncoded = dictEncoder.encode(partKeys);
std::cout << "Dictionary: " << dictEncoded.metadata().compressionRatio() << "x\n";

RunLengthEncoder<int32_t> rleEncoder;
auto rleEncoded = rleEncoder.encode(partKeys);
std::cout << "RLE: " << rleEncoded.metadata().compressionRatio() << "x\n";
```

## Expected Compression Results

| Generator | Dictionary | RLE | Delta | Notes |
|-----------|------------|-----|-------|-------|
| Uniform | ✅ Excellent | ❌ Poor | ❌ Poor | Low cardinality relative to data size |
| Zipfian 0.3 | ✅ Very Good | ⚠️ Fair | ❌ Poor | Repeated popular items |
| Zipfian 1.0 | ✅ Excellent | ✅ Good | ❌ Poor | Highly repetitive |
| Clustered | ✅ Excellent | ✅ Very Good | ⚠️ Fair | Temporal locality |

## Recommendations

1. **For general benchmarking**: Use `zipfExponent=0.3` (realistic skew)
2. **For dictionary encoder testing**: Use `zipfExponent=1.0` (high repetition)
3. **For RLE testing**: Use clustered generator (temporal locality)
4. **For baseline testing**: Use `zipfExponent=0.0` (uniform)
5. **For quick tests**: Use `SF=0.01` (small cardinality)
6. **For production-like tests**: Use `SF=1.0` or higher

## See Also

- `RandomMinMaxGenerator.hpp` - Generic random number generator
- `DictionaryEncoder.hpp` - Ideal for low-cardinality foreign keys
- `RunLengthEncoder.hpp` - Good for clustered access patterns
