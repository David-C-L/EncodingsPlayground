# Encoder Implementations Guide

## Overview

Four complete encoder implementations with full random access support:

1. **RawEncoder** - Baseline (no compression)
2. **RunLengthEncoder** - For repetitive data (integral types only)
3. **DeltaEncoder** - For monotonic/correlated data (integral types only)
4. **DictionaryEncoder** - For low-cardinality data (all types)

All encoders support the full `Codec<T>` interface with:
- ✅ `encode()` - Encode data
- ✅ `decodeAll()` - Bulk decode
- ✅ `decodeAt()` - O(1) or O(log n) random access
- ✅ `decodeRange()` - Efficient range queries

## 1. RawEncoder

**Purpose**: Baseline for comparison, no compression

**Supports**: All types (`int8_t` through `int64_t`, `uint8_t` through `uint64_t`, `float`, `double`, `bool`, `std::string`, composite types)

**Format**:
```
[element_count (8 bytes), raw_bytes...]
```

**Properties**:
- Random Access: ✅ O(1)
- Lossless: ✅
- Preserves Order: ✅
- Compression: None (baseline)
- Memory Overhead: Low

**Best For**: Comparison baseline, incompressible data

**Example**:
```cpp
RawEncoder<int32_t> encoder;
std::vector<int32_t> data = {1, 2, 3, 4, 5};
auto encoded = encoder.encode(data);
auto value = encoder.decodeAt(encoded, 2);  // Returns 3
```

## 2. RunLengthEncoder

**Purpose**: Compress sequences of repeated values

**Supports**: Integral types only (`int8_t` through `int64_t`, `uint8_t` through `uint64_t`)

**Format** (as specified):
```
[num_runs (8 bytes),
 size_of_run_starts_in_bytes (8 bytes),
 size_of_run_values_in_bytes (8 bytes),
 run_starts (num_runs * sizeof(size_t)),
 run_values (num_runs * sizeof(T))]
```

**Properties**:
- Random Access: ✅ O(log n) via binary search
- Lossless: ✅
- Preserves Order: ✅
- Compression: Excellent for repetitive data
- Memory Overhead: Low

**Best For**:
- Sparse data with long runs
- Categorical data with repeated values
- Boolean arrays
- Segmented constant data

**Example**:
```cpp
RunLengthEncoder<int32_t> encoder;
std::vector<int32_t> data = {1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3};
// 12 values, 3 runs
auto encoded = encoder.encode(data);
auto value = encoder.decodeAt(encoded, 5);  // Returns 2 (in second run)
```

**Random Access**: Binary search on run_starts to find which run contains the index, then lookup value.

## 3. DeltaEncoder

**Purpose**: Compress sequences by storing differences

**Supports**: Integral types only (`int8_t` through `int64_t`, `uint8_t` through `uint64_t`)

**Format** (as specified):
```
[size_of_deltas_in_bytes (8 bytes),
 start_value (sizeof(T) bytes),
 deltas ((n-1) * sizeof(T) bytes)]
```

**Properties**:
- Random Access: ✅ O(n) - must accumulate deltas from start
- Lossless: ✅
- Preserves Order: ✅
- Optimized for Sorted: ✅
- Compression: Excellent for monotonic sequences
- Memory Overhead: Low

**Best For**:
- Timestamps (monotonically increasing)
- Auto-incrementing IDs
- Sorted sequences
- Sensor readings with small changes
- Financial tick data

**Example**:
```cpp
DeltaEncoder<int64_t> encoder;
std::vector<int64_t> timestamps = {1000, 1001, 1002, 1003, 1004};
// Stored as: start=1000, deltas=[1, 1, 1, 1]
auto encoded = encoder.encode(timestamps);
auto value = encoder.decodeAt(encoded, 3);  // Accumulates: 1000+1+1+1 = 1003
```

**Random Access**: Accumulates deltas from index 0 to target index. Less efficient than RLE for random access, but excellent compression for sorted data.

## 4. DictionaryEncoder

**Purpose**: Replace repeated values with compact keys

**Supports**: All types (primitives, `std::string`, composite types)

**Format** (as specified):
```
[size_of_dict (8 bytes),
 size_of_dict_keys_in_bytes (8 bytes),
 dict_entries (size_of_dict * sizeof(T)),  // Just values, keys are indices
 keys (num_elements * sizeof(KeyType))]
```

**Key Type Selection** (automatic):
- `uint8_t` if dict size ≤ 256
- `uint16_t` if dict size ≤ 65536
- `uint32_t` otherwise

**Properties**:
- Random Access: ✅ O(1)
- Lossless: ✅
- Preserves Order: ✅
- Dictionary Based: ✅
- Requires Full Data: ✅ (must build dictionary first)
- Memory Overhead: High (dictionary storage)

**Best For**:
- Low cardinality data (few unique values)
- Categorical data (status codes, categories)
- Enum-like data
- String columns with repetition
- Foreign key references

**Example**:
```cpp
DictionaryEncoder<std::string> encoder;
std::vector<std::string> data = {"apple", "banana", "apple", "cherry", "banana"};
// Dictionary: [0:"apple", 1:"banana", 2:"cherry"]
// Keys: [0, 1, 0, 2, 1]
auto encoded = encoder.encode(data);
auto value = encoder.decodeAt(encoded, 2);  // Returns "apple"
```

**Random Access**: Direct lookup: keys[index] gives dictionary index, then lookup in dictionary.

## Performance Characteristics

### Compression Ratio (for 100 int32_t values = 400 bytes)

**Monotonic Data** (100, 101, 102, ...):
- Raw: ~408 bytes (1.02x)
- RLE: ~1200 bytes (3.0x) ❌ Worst
- **Delta: ~408 bytes (1.02x) ✅ Best** - Deltas all = 1
- Dictionary: ~500 bytes (1.25x)

**Repetitive Data** (10 runs of 10 identical values):
- Raw: ~408 bytes (1.02x)
- **RLE: ~200 bytes (0.5x) ✅ Best**
- Delta: ~408 bytes (1.02x)
- Dictionary: ~150 bytes (0.375x) ✅✅ Even better if only 10 unique

**Low Cardinality** (cycling through 0-4):
- Raw: ~408 bytes (1.02x)
- RLE: ~800 bytes (2.0x) - Alternating
- Delta: ~408 bytes (1.02x)
- **Dictionary: ~120 bytes (0.3x) ✅ Best**

### Random Access Speed

1. **RawEncoder**: O(1) - Direct offset
2. **DictionaryEncoder**: O(1) - Direct key lookup + dictionary lookup
3. **RunLengthEncoder**: O(log n) - Binary search on runs
4. **DeltaEncoder**: O(n) - Must accumulate deltas

### Encoding Speed

1. **RawEncoder**: Fastest (memcpy)
2. **DeltaEncoder**: Fast (single pass, subtraction)
3. **RunLengthEncoder**: Fast (single pass, comparison)
4. **DictionaryEncoder**: Slower (hash map + sorting)

## Usage Patterns

### Time Series Data
```cpp
DeltaEncoder<int64_t> timestampEncoder;
RawEncoder<double> valueEncoder;  // Or DeltaEncoder if smooth

// Or compose:
auto composed = composeEncoders<int64_t>(
    DeltaEncoder<int64_t>{},
    RunLengthEncoder<uint8_t>{}  // Further compress deltas
);
```

### Categorical Data
```cpp
DictionaryEncoder<std::string> encoder;
// Perfect for: status codes, user agents, categories, etc.
```

### Sparse/Segmented Data
```cpp
RunLengthEncoder<uint8_t> encoder;
// Perfect for: bitmaps, sparse matrices, segmented data
```

### Mixed Data (Columnar)
```cpp
// Transform row-based to column-based first
struct Row { int64_t id; string category; double value; };
vector<Row> rows;

// Encode each column with optimal encoder
DeltaEncoder<int64_t> idEncoder;
DictionaryEncoder<string> categoryEncoder;
RawEncoder<double> valueEncoder;
```

## Testing

Build and run the test suite:

```bash
cd build
cmake ..
cmake --build .
./Source/encoders/test_encoders
```

The test suite validates:
- ✅ Correctness (encode/decode round-trip)
- ✅ Random access (decodeAt, decodeRange)
- ✅ Format compliance (header structure)
- ✅ Compression ratios
- ✅ Edge cases (empty, single element)

## Implementation Details

### String Handling (DictionaryEncoder)

Strings are variable-length, so dictionary entries store:
```
[string_length (8 bytes), string_data (length bytes)]
```

### Empty Data Handling

All encoders handle empty input gracefully:
- Return minimal valid encoding (headers only)
- Element count = 0
- Decoding returns empty vector

### Metadata

All encoders populate `EncodedData::metadata()`:
- `encodingName`: Encoder name
- `dataType`: Type being encoded
- `elementCount`: Original element count
- `compressedSize`: Encoded size
- `uncompressedSize`: Original size
- `supportsRandomAccess`: true for all
- `customMetadata`: Encoder-specific stats

### Type Safety

- C++23 concepts enforce type constraints
- `RunLengthEncoder` and `DeltaEncoder` require `IntegralType`
- `DictionaryEncoder` supports all types
- Compile-time errors for invalid type combinations

## Future Optimizations

1. **Bit-packing for DeltaEncoder**: Pack small deltas into fewer bits
2. **Indexed access for DeltaEncoder**: Store checkpoints every N elements
3. **Adaptive DictionaryEncoder**: Choose optimal key type dynamically
4. **SIMD**: Vectorize delta accumulation and run-length search
5. **Compression**: Apply general compression to encoded bytes
