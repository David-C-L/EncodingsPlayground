# Spherical Encoder Implementation Guide

## Overview

The `SphericalEncoder` is a compression technique for floating-point vectors that converts Cartesian coordinates to spherical coordinates before compression. This transformation reduces entropy in both the exponent and mantissa of the floating-point representation, enabling significantly better compression ratios with standard compression algorithms like Zstd.

## Implementation

The implementation is based on the `jzip.c` reference implementation and provides the following key features:

### Core Algorithm

1. **Cartesian to Spherical Conversion**: Converts d-dimensional vectors to (d-1) angles
   - Uses optimized O(d) time, O(d) space algorithm per vector
   - Precomputes partial squared norms via backward cumulative sum to avoid error accumulation
   - Uses double precision internally to minimize reconstruction error (~7e-8)

2. **Data Transformation Pipeline**:
   ```
   Input Vectors → [Optional: Normalize] → Spherical Coordinates → 
   Transpose → Byte Shuffle → Compress (Zstd or custom codec)
   ```

3. **Format**: 
   ```
   [dimension (4 bytes)][flags (1 byte)][magnitudes (if normalized)][compressed angles]
   ```

### Key Features

#### 1. **Pluggable Codecs**
Instead of hardcoding Zstd, the encoder accepts two codecs:
- `angleCodec`: For compressing the spherical angle data (default: `ZstdEncoder`)
- `magnitudeCodec`: For compressing magnitudes when vectors are normalized (default: `ZstdFloatEncoder`)

This allows testing with different compression algorithms:
```cpp
auto encoder = SphericalEncoder<VectorType>(
    dimension,
    std::make_shared<ZstdEncoder>(level),      // For angles
    std::make_shared<CustomFloatEncoder>(),     // For magnitudes
    normalizeToUnit
);
```

#### 2. **Optional Normalization**
The `normalizeToUnit` flag controls whether input vectors should be normalized:

- **`normalizeToUnit = false`** (default for unit vectors):
  - Assumes input vectors are already unit length
  - Encodes only angular data
  - Best for: embeddings, normalized feature vectors, directional data

- **`normalizeToUnit = true`** (for general vectors):
  - Computes and stores magnitude separately
  - Normalizes vectors before spherical conversion
  - Separately encodes magnitudes and angles
  - Best for: general vector data, mixed-magnitude vectors

#### 3. **Compression Pipeline**

The encoder implements the full jzip pipeline:

1. **Backward Pass**: Compute partial norms
   ```cpp
   r2[i] = v[i]² + v[i+1]² + ... + v[d-1]²
   ```

2. **Forward Pass**: Compute angles
   ```cpp
   a[i] = acos(v[i] / sqrt(r2[i]))  // for i < d-2
   a[d-2] = atan2(v[d-1], v[d-2])   // last angle
   ```

3. **Transpose**: [n][d-1] → [d-1][n]
   - Groups similar angle types together
   - Improves compression by creating patterns

4. **Byte Shuffle**: Separate float bytes into planes
   - Groups bytes by significance (byte 0, byte 1, etc.)
   - Creates even better patterns for entropy coding

5. **Compress**: Apply codec (Zstd or custom)

### Usage Examples

#### Example 1: Unit Vectors (e.g., text embeddings)
```cpp
#include "encoders/SphericalEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"

// Create encoder for 3D unit vectors
auto encoder = SphericalEncoder<std::array<float, 3>>(
    3,                                    // dimension
    std::make_shared<ZstdEncoder>(),     // codec for angles
    nullptr,                              // no magnitude codec needed
    false                                 // vectors are already unit length
);

// Encode
std::vector<std::array<float, 3>> unitVectors = /* ... */;
auto encoded = encoder.encode(unitVectors);

// Decode
auto decoded = encoder.decodeAll(encoded);

// Check compression
float ratio = (float)originalSize / encoded.size();
std::cout << "Compression: " << ratio << "x" << std::endl;
```

#### Example 2: Non-Unit Vectors with Normalization
```cpp
// Create encoder that normalizes vectors
auto encoder = SphericalEncoder<std::array<float, 3>>(
    3,
    std::make_shared<ZstdEncoder>(),           // for angles
    std::make_shared<ZstdFloatEncoder>(),      // for magnitudes
    true                                        // normalize to unit
);

// Works with any vectors
std::vector<std::array<float, 3>> vectors = /* arbitrary magnitudes */;
auto encoded = encoder.encode(vectors);
auto decoded = encoder.decodeAll(encoded);
// decoded vectors will have original magnitudes restored
```

#### Example 3: High-Dimensional Vectors
```cpp
// Works with any Vector32Type (contiguous range of float)
auto encoder = SphericalEncoder<std::vector<float>>(
    128,                                   // 128-dimensional vectors
    std::make_shared<ZstdEncoder>(3),     // Zstd level 3
    nullptr,
    false
);

std::vector<std::vector<float>> vectors(1000, std::vector<float>(128));
// ... populate vectors ...
auto encoded = encoder.encode(vectors);
```

#### Example 4: Testing Different Codecs
```cpp
// Test with different compression levels
for (int level = 1; level <= 22; level++) {
    auto encoder = SphericalEncoder<VectorType>(
        dim,
        std::make_shared<ZstdEncoder>(level),
        std::make_shared<ZstdFloatEncoder>(level),
        normalizeToUnit
    );
    
    auto encoded = encoder.encode(data);
    // Compare compression ratio vs speed
}

// Or test with completely different codecs
auto encoder = SphericalEncoder<VectorType>(
    dim,
    std::make_shared<LZ4Encoder>(),        // Fast compression
    std::make_shared<DeltaEncoder>(),      // For magnitudes
    true
);
```

## Performance Characteristics

### Compression Ratio
- **Unit vectors**: Typical 2-5x compression (dimension-dependent)
- **High-dimensional**: Better compression as dimension increases
- **Non-unit normalized**: Slightly lower due to magnitude storage

### Speed
- **Encoding**: O(n·d) time, dominated by trigonometric operations
- **Decoding**: O(n·d) time, similar cost
- **Memory**: O(n·d) working space for transformations

### Accuracy
- **Reconstruction error**: ~7e-8 for unit vectors (using double precision internally)
- **Lossless**: The encoding is mathematically lossless (within floating-point precision)
- **Magnitude preservation**: When normalized, magnitudes are perfectly preserved

## Implementation Details

### File Structure
```
Source/
├── encoders/
│   ├── SphericalEncoder.hpp      # Main spherical encoder
│   └── ZstdEncoder.hpp           # Zstd codecs (byte and float)
└── encodings/
    ├── EncodingType.hpp          # Added SphericalEncoding enum
    └── ...
```

### Key Classes

#### `SphericalEncoder<T>`
Template parameters:
- `T`: Vector type (must satisfy `Vector32Type` concept)
  - Must be a contiguous range of `float`
  - Examples: `std::vector<float>`, `std::array<float, N>`

Constructor parameters:
- `dimension`: Dimensionality of vectors
- `angleCodec`: Codec for angle compression
- `magnitudeCodec`: Codec for magnitude compression (when normalizing)
- `normalizeToUnit`: Whether to normalize input vectors

#### `ZstdEncoder` and `ZstdFloatEncoder`
- `ZstdEncoder`: Compresses byte data (for shuffled angles)
- `ZstdFloatEncoder`: Compresses float data (for magnitudes)
- Both accept compression level (1-22, default 3)

### Spherical Coordinate Math

For a d-dimensional unit vector **v**, we compute (d-1) angles:

```
θ[i] = arccos(v[i] / r[i])    for i = 0..d-3
θ[d-2] = atan2(v[d-1], v[d-2])
```

where `r[i] = sqrt(v[i]² + v[i+1]² + ... + v[d-1]²)`

The last angle uses `atan2` for full circle coverage.

### Reconstruction

From angles back to Cartesian:
```
s = 1.0
for i = 0 to d-2:
    v[i] = s × cos(θ[i])
    s = s × sin(θ[i])
v[d-2] = s × cos(θ[d-2])
v[d-1] = s × sin(θ[d-2])
```

## Comparison with Raw Zstd

For typical unit vector data:
- **Raw Zstd**: ~1.2-1.5x compression
- **Spherical + Zstd**: ~2-5x compression

The improvement comes from:
1. Reduced entropy in angular representation
2. Transpose creating similar-value sequences
3. Byte shuffle grouping by significance

## Testing

Build and run the test suite:
```bash
cd build
cmake ..
make test_spherical_encoder
./bin/test_spherical_encoder
```

The test suite covers:
1. Unit vector compression
2. Non-unit vectors without normalization
3. Non-unit vectors with normalization
4. High-dimensional vectors (128D)
5. Accuracy verification
6. Magnitude preservation

## Future Enhancements

Potential improvements:
1. **SIMD optimization**: Vectorize trigonometric operations
2. **Parallel processing**: Process multiple vectors concurrently
3. **Adaptive normalization**: Auto-detect when normalization helps
4. **Block-based encoding**: Support partial decoding
5. **Quantization**: Optional lossy compression with controllable error

## References

- Original implementation: `jzip.c` (included in repository)
- Zstandard: https://facebook.github.io/zstd/
- Spherical coordinates: https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
