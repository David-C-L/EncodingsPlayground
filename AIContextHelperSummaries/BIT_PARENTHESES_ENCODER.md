# BitParenthesesEncoder

## Overview

The `BitParenthesesEncoder` implements a encoding scheme where each integral value `n` is represented as `n` ones followed by a zero, creating a balanced parentheses (bit sequence) representation. The resulting bit sequence is then compressed using Run-Length Encoding.

## How It Works

### Encoding Process

1. **Bit-Parentheses Conversion**: Each value `n` becomes `n` ones followed by one zero
   ```
   [3, 5, 4] → [1,1,1,0, 1,1,1,1,1,0, 1,1,1,1,0]
   ```

2. **RLE Compression**: The bit sequence is compressed using run-length encoding
   - Store run starts (bit positions)
   - Store run values (0 or 1)

3. **Boundary Tracking**: Store where each element ends for random access

### Format

```
[num_elements (8 bytes)]
[num_bit_runs (8 bytes)]
[bit_run_starts (num_bit_runs * 8 bytes)]
[bit_run_values (num_bit_runs * 1 byte)]
[element_boundaries (num_elements * 8 bytes)]
```

## Features

- ✅ **Lossless**: Perfect reconstruction
- ✅ **Random Access**: O(log n) access via binary search on boundaries
- ✅ **Range Queries**: Efficient `decodeRange()` implementation
- ✅ **Composable**: Can be used in encoder chains

## Performance Characteristics

### Time Complexity
- **Encode**: O(sum of values) - must create bit for each count
- **Decode All**: O(n + sum of values)
- **Decode At**: O(k) where k is the number of RLE runs
- **Decode Range**: O(m × k) where m is range size

### Space Complexity
- **Best Case**: Small repeated values (e.g., [1,1,1,1,1])
- **Worst Case**: Large values (each value contributes many bits)
- **Overhead**: 
  - 16 bytes header
  - ~9 bytes per RLE run
  - 8 bytes per element (boundaries)

## When to Use

### ✅ Good For:
- Map sizes (typically 1-20 entries)
- Array lengths (small values)
- Small counters or frequencies
- Data with values 1-20 and repetition

### ❌ Avoid For:
- Large values (>100)
- No repetition in the data
- General purpose compression

### ⚠️ Reality Check

**In practice, this encoder performs WORSE than alternatives** due to high overhead:

```
Test Case: 1000 map sizes (values 1-10)
Raw Encoder:        4,008 bytes  (baseline)
RunLength Encoder: 12,024 bytes  (0.33x)
BitParentheses:    26,016 bytes  (0.15x) ← WORST
```

## Why Use It Then?

This encoder is primarily **educational and theoretical**:

1. **Succinct Data Structures**: In specialized data structures with rank/select operations, bit-parentheses can be efficient
2. **Composition**: Might be useful as part of a larger encoding pipeline
3. **Specific Patterns**: May work for very specific data patterns not tested here

## Implementation Details

### Random Access

Uses precomputed element boundaries to quickly jump to any element:

```cpp
size_t prevBoundary = (index == 0) ? 0 : elementBoundaries[index - 1];
size_t boundary = elementBoundaries[index];
size_t onesCount = countOnesInRange(prevBoundary, boundary - 1);
```

### RLE Bit Counting

Efficiently counts ones without materializing the full bit vector:

```cpp
size_t countOnesInRange(start, end) {
    // Find overlapping RLE runs
    // Sum up ones in those runs
    // Return count
}
```

## Example Usage

```cpp
#include "encoders/BitParenthesesEncoder.hpp"

BitParenthesesEncoder<int32_t> encoder;

// Encode map sizes
std::vector<int32_t> mapSizes = {3, 5, 4, 2, 7};
auto encoded = encoder.encode(mapSizes);

// Random access
auto size = encoder.decodeAt(encoded, 2);  // Returns 4

// Range access
auto range = encoder.decodeRange(encoded, 1, 3);  // Returns [5, 4]

// Decode all
auto decoded = encoder.decodeAll(encoded);  // Returns [3, 5, 4, 2, 7]
```

## Comparison with Alternatives

| Encoder | Best For | Compression | Speed |
|---------|----------|-------------|-------|
| Raw | No compression needed | 1.0x | ⚡⚡⚡ |
| RunLength | Highly repetitive data | 3-10x | ⚡⚡ |
| BitParentheses | Small values (1-20), theoretical | 0.1-0.3x ⚠️ | ⚡ |

## Future Improvements

To make this encoder practical:

1. **Remove boundary storage**: Use rank/select on the bit vector instead
2. **Better bit compression**: Use bit-packing instead of RLE
3. **Hybrid approach**: Only use for very small values, fallback for large
4. **Wavelet tree**: Use a wavelet tree representation for better compression

## Conclusion

While theoretically interesting, the `BitParenthesesEncoder` has high overhead that makes it impractical for most use cases. Consider using:

- **RunLengthEncoder** for repetitive data
- **DictionaryEncoder** for low-cardinality data
- **DeltaEncoder** for sequential/incremental data

This encoder serves primarily as an educational example of succinct data structure techniques.
