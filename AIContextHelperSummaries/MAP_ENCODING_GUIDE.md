# Map Encoding Strategies

This document describes the 6 specialized map encoding strategies implemented in the EncodingsPlayground framework.

## Overview

Maps (key-value pairs) can be encoded using various strategies depending on the data characteristics. Each strategy makes different tradeoffs between compression ratio, encode/decode speed, and suitability for different access patterns.

All strategies encode map sizes using **Run-Length Encoding (RLE)** since many maps often have the same size, making RLE very effective.

## Encoding Strategies

### 1. MapDictSeparate - Separate Dictionary Encoding

**Class:** `MapDictSeparateEncoder<K, V>`

**Strategy:** Encode all keys with one dictionary encoder, encode all values with another dictionary encoder.

**Format:**
```
[size_len][key_len][value_len][sizes][keys][values]
```

**Best For:**
- Maps where both keys and values have low cardinality
- When keys and values have independent patterns
- General-purpose encoding with good compression

**Example Use Case:**
```cpp
auto keyEncoder = std::make_shared<DictionaryEncoder<int32_t>>();
auto valueEncoder = std::make_shared<DictionaryEncoder<int32_t>>();
auto encoder = std::make_shared<MapDictSeparateEncoder<int32_t, int32_t>>(
    keyEncoder, valueEncoder);
```

**Properties:**
- ✅ Lossless
- ✅ Good compression for low-cardinality data
- ✅ Independent key/value encoding
- ❌ No random access
- ❌ Requires full decode for access

---

### 2. MapDictTogether - Pair Dictionary Encoding

**Class:** `MapDictTogetherEncoder<K, V>`

**Strategy:** Encode entire (key, value) pairs together as a single dictionary.

**Format:**
```
[size_len][pair_len][sizes][pairs]
```

**Best For:**
- Maps where specific key-value combinations repeat
- When the correlation between keys and values matters
- Capturing full map entry patterns

**Example Use Case:**
```cpp
auto pairEncoder = std::make_shared<DictionaryEncoder<std::pair<int32_t, int32_t>>>();
auto encoder = std::make_shared<MapDictTogetherEncoder<int32_t, int32_t>>(pairEncoder);
```

**Properties:**
- ✅ Lossless
- ✅ Excellent compression when pairs repeat
- ✅ Preserves key-value correlation
- ❌ Lower compression if pairs don't repeat
- ❌ Requires pair encoder support

---

### 3. MapDeltaRLEDict - Delta+RLE Keys, Dictionary Values

**Class:** `MapDeltaRLEDictEncoder<K, V>`

**Strategy:** Use Delta+RLE encoding for keys (optimal for sequential/monotonic keys), dictionary encoding for values.

**Format:**
```
[size_len][key_len][value_len][sizes][delta_rle_keys][dict_values]
```

**Best For:**
- Maps with sequential or monotonic keys (e.g., 1, 2, 3, 4, ...)
- Time-series data with sequential timestamps
- Low-cardinality values with sequential keys

**Example Use Case:**
```cpp
auto keyEncoder = std::make_shared<DeltaEncoder<int32_t>>();
auto valueEncoder = std::make_shared<DictionaryEncoder<int32_t>>();
auto encoder = std::make_shared<MapDeltaRLEDictEncoder<int32_t, int32_t>>(
    keyEncoder, valueEncoder);
```

**Properties:**
- ✅ Lossless
- ✅ Excellent compression for sequential keys
- ✅ Good compression for low-cardinality values
- ✅ Specialized for time-series patterns
- ❌ Poor compression for random keys

---

### 4. MapColumnarPairsDict - Columnar Layout with Pairs

**Class:** `MapColumnarPairsDictEncoder<K, V>`

**Strategy:** Transform to columnar layout (all keys together, all values together), then encode as dictionary.

**Format:**
```
[size_len][key_len][value_len][num_keys][sizes][keys][values]
```

**Best For:**
- Analytical workloads needing columnar access
- When you want to access all keys or all values separately
- Better CPU cache locality for bulk operations

**Example Use Case:**
```cpp
auto keyEncoder = std::make_shared<DictionaryEncoder<int32_t>>();
auto valueEncoder = std::make_shared<DictionaryEncoder<int32_t>>();
auto encoder = std::make_shared<MapColumnarPairsDictEncoder<int32_t, int32_t>>(
    keyEncoder, valueEncoder);
```

**Properties:**
- ✅ Lossless
- ✅ Columnar layout
- ✅ Better cache locality
- ✅ Can access keys/values separately
- ❌ Slightly larger header

---

### 5. MapColumnarSeparateDict - Columnar Separate Dictionary

**Class:** `MapColumnarSeparateDictEncoder<K, V>`

**Strategy:** Store all keys in one column, all values in another column. Both are dictionary encoded separately.

**Format:**
```
[size_len][key_len][value_len][sizes][key_column][value_column]
```

**Best For:**
- Columnar analytics requiring separate key/value access
- OLAP-style queries
- When you want to scan all keys or all values independently

**Example Use Case:**
```cpp
auto keyEncoder = std::make_shared<DictionaryEncoder<int32_t>>();
auto valueEncoder = std::make_shared<DictionaryEncoder<int32_t>>();
auto encoder = std::make_shared<MapColumnarSeparateDictEncoder<int32_t, int32_t>>(
    keyEncoder, valueEncoder);
```

**Properties:**
- ✅ Lossless
- ✅ Columnar layout
- ✅ True column separation
- ✅ Optimal for analytical queries
- ✅ SIMD-friendly for bulk scans

---

### 6. MapColumnarMixed - Columnar with Mixed Encodings

**Class:** `MapColumnarMixedEncoder<K, V>`

**Strategy:** Columnar layout with Delta+RLE for keys (sequential) and Dictionary for values (low-cardinality).

**Format:**
```
[size_len][key_len][value_len][sizes][delta_rle_key_column][dict_value_column]
```

**Best For:**
- Time-series data with sequential keys and categorical values
- Combining benefits of both Delta and Dictionary
- Analytical workloads with specific key patterns

**Example Use Case:**
```cpp
auto keyEncoder = std::make_shared<DeltaEncoder<int32_t>>();
auto valueEncoder = std::make_shared<DictionaryEncoder<int32_t>>();
auto encoder = std::make_shared<MapColumnarMixedEncoder<int32_t, int32_t>>(
    keyEncoder, valueEncoder);
```

**Properties:**
- ✅ Lossless
- ✅ Columnar layout
- ✅ Best compression for sequential keys + low-cardinality values
- ✅ Optimal for time-series analytics
- ✅ Specialized for common patterns

---

## Choosing the Right Strategy

### Decision Tree

```
Do you have sequential/monotonic keys?
├─ YES → Use MapDeltaRLEDict or MapColumnarMixed
└─ NO
   └─ Do you need columnar access?
      ├─ YES → Use MapColumnarSeparateDict or MapColumnarPairsDict
      └─ NO
         └─ Do key-value pairs repeat as combinations?
            ├─ YES → Use MapDictTogether
            └─ NO → Use MapDictSeparate
```

### Performance Characteristics

| Strategy | Compression | Encode Speed | Decode Speed | Columnar | Random Access |
|----------|-------------|--------------|--------------|----------|---------------|
| MapDictSeparate | Good | Fast | Fast | ❌ | ❌ |
| MapDictTogether | Excellent* | Fast | Fast | ❌ | ❌ |
| MapDeltaRLEDict | Excellent** | Medium | Medium | ❌ | ❌ |
| MapColumnarPairsDict | Good | Fast | Fast | ✅ | Partial |
| MapColumnarSeparateDict | Good | Fast | Fast | ✅ | ✅ |
| MapColumnarMixed | Excellent** | Medium | Medium | ✅ | Partial |

\* When pairs repeat  
\** For sequential keys

---

## Data Generators

The framework includes specialized map generators for testing:

### 1. SequentialKeyMapGenerator
Generates maps with sequential integer keys.
```cpp
auto gen = std::make_shared<SequentialKeyMapGenerator<int32_t>>(
    5,    // minMapSize
    20,   // maxMapSize
    0     // startKey
);
```

### 2. LowCardinalityKeyMapGenerator
Generates maps with keys drawn from a small set.
```cpp
auto gen = std::make_shared<LowCardinalityKeyMapGenerator<int32_t, int32_t>>(
    10,   // numUniqueKeys
    5,    // minMapSize
    20    // maxMapSize
);
```

### 3. LowCardinalityValueMapGenerator
Generates maps with values drawn from a small set.
```cpp
auto gen = std::make_shared<LowCardinalityValueMapGenerator<int32_t, int32_t>>(
    5,    // numUniqueValues
    5,    // minMapSize
    15    // maxMapSize
);
```

### 4. ConstantSizeMapGenerator
All maps have the same size (optimal for RLE size encoding).
```cpp
auto gen = std::make_shared<ConstantSizeMapGenerator<int32_t, int32_t>>(
    10    // mapSize
);
```

### 5. VaryingSizeMapGenerator
Maps with widely varying sizes.
```cpp
auto gen = std::make_shared<VaryingSizeMapGenerator<int32_t, int32_t>>(
    1,    // minMapSize
    50    // maxMapSize
);
```

### 6. ColumnarOptimizedMapGenerator
Sequential keys + low-cardinality values (optimal for columnar mixed encoding).
```cpp
auto gen = std::make_shared<ColumnarOptimizedMapGenerator<int32_t>>(
    5,    // numUniqueValues
    10,   // minMapSize
    20    // maxMapSize
);
```

---

## Running Map Benchmarks

### Build
```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make map_benchmarks
```

### Run
```bash
./bin/map_benchmarks
```

### Generate Plots
```bash
cd Benchmarks
python3 plot_map_results.py
```

### Output
- `map_benchmark_results.json` - Raw benchmark data
- `map_compression_by_encoder.png` - Compression ratios by encoder
- `map_encode_decode_times.png` - Performance comparison
- `map_throughput.png` - Throughput analysis
- `map_space_time_tradeoff.png` - Compression vs speed
- `map_best_encoder_by_generator.png` - Best encoder per pattern
- `map_compression_heatmap.png` - Full matrix visualization

---

## Implementation Details

### Map Size Encoding

All strategies use the same RLE-based size encoding:

```
[num_runs, (size, count)...]
```

For example, if you have 100 maps:
- 30 maps of size 5
- 50 maps of size 10
- 20 maps of size 5

The encoding would be:
```
[3, (5, 30), (10, 50), (5, 20)]
```

This is extremely efficient when many maps have the same size.

### Memory Layout

**Row-oriented (MapDictSeparate, MapDeltaRLEDict):**
```
Map 1: K1 V1 K2 V2 K3 V3
Map 2: K4 V4 K5 V5
...
Encoded as: [sizes][K1 K2 K3 K4 K5...][V1 V2 V3 V4 V5...]
```

**Columnar (MapColumnar* variants):**
```
Key Column:   [K1 K2 K3 K4 K5...]
Value Column: [V1 V2 V3 V4 V5...]
Sizes:        [3, 2, ...]
```

---

## Extension Points

### Custom Key/Value Encoders

You can plug in any encoder that implements `Codec<T>`:

```cpp
// Custom float encoder for values
auto customValueEncoder = std::make_shared<MyFloatEncoder>();
auto encoder = std::make_shared<MapDictSeparateEncoder<int32_t, float>>(
    keyEncoder, customValueEncoder);
```

### Composed Encodings

For even better compression, chain encoders:

```cpp
// Delta encode, then RLE the deltas
auto delta = std::make_shared<DeltaEncoder<int32_t>>();
auto rle = std::make_shared<RunLengthEncoder<int32_t>>();
auto composed = std::make_shared<ComposedEncoder<int32_t, DeltaEncoder<int32_t>, 
                                                   RunLengthEncoder<uint8_t>>>(delta, rle);
```

---

## References

- [QUICK_START.md](../QUICK_START.md) - Getting started guide
- [BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md) - Benchmarking framework details
- [ENCODER_GUIDE.md](ENCODER_GUIDE.md) - Creating custom encoders
