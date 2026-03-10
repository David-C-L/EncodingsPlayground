# Codec Composition Guide

## Overview

The EncodingsPlayground framework now supports **two types of composition**:

1. **Sequential Composition**: Chain codecs together (e.g., Delta → RLE → BitPacking)
2. **Structural Composition**: Apply different codecs to different parts of composite types (e.g., different encoders for map keys vs values)

## 1. Sequential Composition

### What It Does
Applies codecs in sequence: the output of one becomes the input to the next.

### When To Use
- **Layer compression strategies**: Delta encoding to reduce magnitude, then RLE to compress repeated values
- **Multi-stage optimization**: Bit-packing after delta encoding for time-series data
- **Generic compression**: Apply any codec, then use general-purpose compression (LZ4, Zstd)

### Example: Delta → RLE

```cpp
#include "encodings/ComposedEncoder.hpp"

// Create individual encoders
DeltaEncoder<int32_t> delta;
RLEEncoder<uint8_t> rle;  // Works on byte output

// Compose them
auto composed = composeEncoders<int32_t>(delta, rle);

// Use like any other codec
std::vector<int32_t> data = {100, 101, 102, 103, 104};
auto encoded = composed.encode(data);
auto decoded = composed.decodeAll(encoded);

// Check properties
std::cout << composed.name();  // "Delta | RLE"
auto props = composed.properties();
// Properties are intersected: only lossless if BOTH are lossless
```

### Three-Way Composition

```cpp
DeltaEncoder<int64_t> delta;
RLEEncoder<uint8_t> rle;
BitPackingEncoder<uint8_t> bitpack;

auto tripleComposed = composeEncoders<int64_t>(delta, rle, bitpack);
// Name: "Delta | RLE | BitPacking"
```

### Property Propagation

When composing encoders, properties are combined conservatively:

- **RandomAccess**: TRUE only if ALL support it (otherwise SequentialOnly)
- **Lossless**: TRUE only if ALL are lossless (otherwise Lossy)
- **PreservesOrder**: TRUE only if ALL preserve order
- **RequiresFullData**: TRUE if ANY requires full data
- **MemoryOverhead**: Takes the worst case (High if any is High)

## 2. Structural Composition

### What It Does
Applies different codecs to different components of structured data.

### When To Use
- **Heterogeneous data**: Different types benefit from different encodings
- **Columnar-style encoding**: Encode struct fields independently
- **Optimize per-field**: Use domain knowledge (timestamps → delta, IDs → dictionary)

### Example: Pair Vector with Different Encoders

```cpp
#include "encodings/ComposedEncoder.hpp"

// Different encoders for keys and values
DictionaryEncoder<std::string> keyEncoder;  // Strings → dictionary
DeltaEncoder<int32_t> valueEncoder;         // Integers → delta

auto pairEncoder = encodePairs<std::string, int32_t>(keyEncoder, valueEncoder);

// Use on vector of pairs
std::vector<std::pair<std::string, int32_t>> data = {
    {"user1", 100},
    {"user2", 101},
    {"user1", 102}
};

auto encoded = pairEncoder.encode(data);
// Internally: keys and values are encoded separately, then combined
```

### Example: Map Encoder

```cpp
// Encode maps with different strategies for keys vs values
DeltaEncoder<int64_t> timestampEncoder;      // Monotonic timestamps
CustomEncoder<double> measurementEncoder;    // Measurement values

MapEncoder<int64_t, double, DeltaEncoder<int64_t>, CustomEncoder<double>>
    mapEncoder(timestampEncoder, measurementEncoder);

// Use on vector of maps
std::vector<std::map<int64_t, double>> timeSeries;
// Each map: timestamp → measurement

auto encoded = mapEncoder.encode(timeSeries);
```

## 3. Hybrid Composition

You can combine both approaches!

### Example: Structural + Sequential

```cpp
// Step 1: Structural composition for a map
DeltaEncoder<int64_t> keyEncoder;
DeltaEncoder<double> valueEncoder;

MapEncoder<int64_t, double, DeltaEncoder<int64_t>, DeltaEncoder<double>>
    mapEncoder(keyEncoder, valueEncoder);

// Step 2: Apply RLE to the byte output
RLEEncoder<uint8_t> rle;

// Combine: Map encoding → RLE on bytes
auto fullyComposed = composeEncoders<std::map<int64_t, double>>(mapEncoder, rle);

// This gives you:
// 1. Optimized encoding per field (delta for timestamps, delta for values)
// 2. Additional compression layer (RLE on the encoded bytes)
```

## Real-World Use Cases

### 1. Time Series Data

```cpp
// Data: vector of (timestamp, value) pairs
// Timestamps are monotonic → delta encoding
// Values are smooth → delta encoding
// Result may have patterns → RLE

DeltaEncoder<int64_t> timestampEnc;
DeltaEncoder<double> valueEnc;
auto structuralEnc = encodePairs<int64_t, double>(timestampEnc, valueEnc);

RLEEncoder<uint8_t> rle;
auto final = composeEncoders<std::pair<int64_t, double>>(structuralEnc, rle);
```

### 2. Sparse Data (Map-heavy)

```cpp
// Data: vector of maps, many repeated keys
// Keys → dictionary (reduce cardinality)
// Values → delta (if numeric/correlated)
// Byte output → compression

DictionaryEncoder<std::string> keyEnc;
DeltaEncoder<int32_t> valueEnc;
MapEncoder<std::string, int32_t, ...> mapEnc(keyEnc, valueEnc);

ZstdEncoder<uint8_t> compression;
auto final = composeEncoders<std::map<...>>(mapEnc, compression);
```

### 3. Columnar-Style Encoding

```cpp
// Instead of encoding Row{id, name, score, timestamp}
// Transform to: {Vector<id>, Vector<name>, Vector<score>, Vector<timestamp>}
// Each column gets optimal encoding:

// Pseudo-code for columnar transformation:
struct Row { int64_t id; string name; float score; int64_t timestamp; };
vector<Row> rows;

// Transform to columns
auto [ids, names, scores, timestamps] = toColumns(rows);

// Encode each column optimally
DeltaEncoder<int64_t> idEnc;
DictionaryEncoder<string> nameEnc;
FloatEncoder<float> scoreEnc;
DeltaEncoder<int64_t> timeEnc;

auto encodedIds = idEnc.encode(ids);
auto encodedNames = nameEnc.encode(names);
auto encodedScores = scoreEnc.encode(scores);
auto encodedTimestamps = timeEnc.encode(timestamps);

// Combine (custom composite encoder)
```

## Implementation Notes

### EncodedData Format

**Sequential Composition:**
- First encoder outputs EncodedData
- Second encoder treats those bytes as input
- Metadata tracks the composition chain

**Structural Composition:**
- Components encoded separately
- Header stores sizes/offsets
- Metadata tracks per-component encodings

### Random Access Considerations

- **Sequential composition**: Random access is tricky
  - Need to decode through all layers
  - Current implementation decodes all, then indexes
  - Future: Could add indexed layers

- **Structural composition**: Easier random access
  - Access independent components
  - Example: decoding one map doesn't require decoding others

### Memory Efficiency

- **Sequential**: Intermediate representations needed during decode
- **Structural**: Can decode components independently
- **Trade-off**: More layers = more overhead vs better compression

## API Summary

```cpp
// Sequential composition
auto composed = composeEncoders<T>(encoder1, encoder2);
auto triple = composeEncoders<T>(encoder1, encoder2, encoder3);

// Structural composition - Pairs
auto pairEnc = encodePairs<K, V>(keyEncoder, valueEncoder);

// Structural composition - Maps
MapEncoder<K, V, KeyEnc, ValEnc> mapEnc(keyEnc, valEnc);

// Properties
composed.name();        // "Enc1 | Enc2"
composed.properties();  // Merged properties
composed.encodingType(); // EncodingType::Composed or ::Structural
```

## Future Extensions

1. **Adaptive composition**: Auto-select best composition based on data
2. **Indexed layers**: Support true random access in composed codecs
3. **Parallel encode/decode**: Independent components in parallel
4. **Columnar transformers**: Automatic row→column transformation
5. **Metadata preservation**: Better tracking of composition layers
6. **Optimizations**: Skip redundant layers, fuse operations

## Performance Tips

1. **Order matters**: Delta before RLE is usually better than RLE before Delta
2. **Profile first**: Measure before adding complexity
3. **Simple is often best**: Don't over-compose
4. **Consider data**: Match encoding to data characteristics
5. **Benchmark compositions**: Different orders can have different results
