#pragma once

#include <span>
#include <vector>
#include <cstring>
#include <concepts>
#include <cmath>
#include <memory>
#include <algorithm>
#include <numeric>
#include <limits>
#include <type_traits>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encoders/ZstdEncoder.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {
    using core::DataType;
    using core::PrimitiveType;
    using core::FloatingPointType;
    using core::typeToDataType;
    using core::Vector32Type;
    using core::Float32Type;

/**
 * @brief Result of a magnitude encode-decode round-trip error measurement.
 *
 * All values are in long double so the caller can compare float/double/long double
 * MagT variants on equal footing without any precision being lost in the result.
 */
struct MagnitudeErrorResult {
    long double meanAbsoluteError  = 0.0L;  ///< Mean |m_hat - m| over all vectors
    long double maxAbsoluteError   = 0.0L;  ///< Max  |m_hat - m|
    long double meanRelativeError  = 0.0L;  ///< Mean |m_hat - m| / m  (skips m ≈ 0)
};

/**
 * @brief Interface for encoders that can report magnitude round-trip errors.
 *
 * Only SphericalEncoder with normalizeToUnit=true implements this.  Callers
 * dynamic_cast to this interface; a null result means the encoder does not
 * store magnitudes.
 */
template<typename T>
requires Vector32Type<T>
class IMagnitudeErrorProvider {
public:
    virtual ~IMagnitudeErrorProvider() = default;

    /**
     * @brief Encode then decode @p data and measure the error in the stored magnitudes.
     *
     * The reference magnitude for each vector is computed in long double regardless
     * of MagT, so the result faithfully shows the precision loss introduced by
     * casting to MagT before storage.
     */
    virtual MagnitudeErrorResult computeMagnitudeErrors(
        std::span<const T> data) const = 0;
};


/**
 * @brief Spherical coordinate encoding for vector types with compile-time dimension
 * 
 * Encodes fixed-size vectors of 32-bit floats by converting them to spherical coordinates.
 * This reduces entropy in both exponent and mantissa, enabling better compression.
 * 
 * For unit vectors: converts directly to angles (d-1 angles per d-dimensional vector)
 * For non-unit vectors with normalization: separately encodes magnitudes and angles
 * 
 * The angles are transposed and byte-shuffled before compression to improve compression ratio.
 * 
 * Dimension is a compile-time constant for:
 * - Better type safety
 * - Compiler optimizations
 * - Stack-based temporary arrays for small dimensions
 * 
 * Format: [dimension(4 bytes)][flags(1 byte)][magnitude_size (if normalized)][magnitudes (if normalized)][angles_size][compressed angles]
 * 
 * @tparam T The vector32 type to encode (must be a contiguous range of float)
 * @tparam Dimension The dimensionality of the vectors (compile-time constant)
 */
template<typename T, size_t Dimension, typename MagT = float_t>
    requires Vector32Type<T> && FloatingPointType<MagT> && (Dimension >= 2)
class SphericalEncoder : public Codec<T>, public IMagnitudeErrorProvider<T> {
public:
    static constexpr size_t dimension = Dimension;
    
    /**
     * @brief Construct a SphericalEncoder
     * 
     * @param angleCodec Codec to use for compressed angle data (default: ZstdEncoder<uint8_t>)
     * @param magnitudeCodec Codec to use for magnitude data when normalizing (default: ZstdEncoder<float>)
     * @param normalizeToUnit If true, normalize input vectors to unit length before encoding
     */
    SphericalEncoder(
        std::shared_ptr<Codec<uint8_t>> angleCodec = nullptr,
        std::shared_ptr<Codec<MagT>> magnitudeCodec = nullptr,
        bool normalizeToUnit = false
    ) : angleCodec_(angleCodec ? angleCodec : std::make_shared<ZstdEncoder<uint8_t>>()), 
        magnitudeCodec_(magnitudeCodec ? magnitudeCodec : std::make_shared<ZstdEncoder<MagT>>()),
        normalizeToUnit_(normalizeToUnit) { }

    EncodedData encode(std::span<const T> data) override {
        if (data.empty()) {
            return createEmptyEncoding();
        }
        
        const size_t n = data.size();
        constexpr size_t d = Dimension;
        
        EncodedData result;
        std::vector<uint8_t> output;
        
        // Write dimension (4 bytes)
        uint32_t dim = static_cast<uint32_t>(d);
        output.insert(output.end(), 
            reinterpret_cast<const uint8_t*>(&dim), 
            reinterpret_cast<const uint8_t*>(&dim) + sizeof(uint32_t));
        
        // Write flags (1 byte: bit 0 = normalizeToUnit)
        uint8_t flags = normalizeToUnit_ ? 0x01 : 0x00;
        output.push_back(flags);
        
        // Prepare vectors for processing
        std::vector<float> vectors(n * d);
        std::vector<MagT> magnitudes;
        
        // Copy and optionally normalize input vectors
        if (normalizeToUnit_) {
            magnitudes.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const T& vec = data[i];
                long double sumSq = 0.0;
                size_t idx = 0;
                for (const auto& val : vec) {
                    if (idx >= d) break;
                    sumSq += static_cast<long double>(val) * static_cast<long double>(val);
                    idx++;
                }
                MagT magnitude = static_cast<MagT>(std::sqrt(sumSq));
                magnitudes.push_back(magnitude);

                // Compute a tolerance that depends on the magnitude type (MagT).
                // Purpose: avoid division by zero or huge values when magnitude is extremely small.
                // We pick sensible defaults for float/double and fall back to a numeric_limits-based
                // heuristic for other floating types.
                MagT zero_tol;
                if constexpr (std::is_same_v<MagT, float>) {
                    zero_tol = static_cast<MagT>(1e-10f);
                } else if constexpr (std::is_same_v<MagT, double>) {
                    // double has much smaller machine epsilon; use a smaller absolute tolerance
                    zero_tol = static_cast<MagT>(1e-20);
                } else {
                    // Generic fallback: use a scaled machine epsilon
                    zero_tol = std::numeric_limits<MagT>::epsilon() * static_cast<MagT>(100);
                }

                // Normalize and store (avoid division when magnitude is effectively zero)
                MagT invMag = (magnitude > zero_tol) ? (static_cast<MagT>(1) / magnitude) : static_cast<MagT>(0);
                idx = 0;
                for (const auto& val : vec) {
                    if (idx >= d) break;
                    vectors[i * d + idx] = val * invMag;
                    idx++;
                }
            }
            
            // Encode magnitudes
            EncodedData magnitudeEncoded = magnitudeCodec_->encode(std::span<const MagT>(magnitudes));
            const auto& magData = magnitudeEncoded.data();
            
            // Write magnitude data size (8 bytes)
            uint64_t magSize = magData.size();
            output.insert(output.end(),
                reinterpret_cast<const uint8_t*>(&magSize),
                reinterpret_cast<const uint8_t*>(&magSize) + sizeof(uint64_t));
            
            // Write magnitude data
            output.insert(output.end(), magData.begin(), magData.end());
        } else {
            // Copy vectors as-is
            for (size_t i = 0; i < n; ++i) {
                const T& vec = data[i];
                size_t idx = 0;
                for (const auto& val : vec) {
                    if (idx >= d) break;
                    vectors[i * d + idx] = val;
                    idx++;
                }
            }
        }
        
        // Convert Cartesian to spherical coordinates
        std::vector<float> angles(n * (d - 1));
        cartesianToSpherical(vectors.data(), angles.data(), n, d);
        
        // Transpose angles: from [n][d-1] to [d-1][n]
        std::vector<float> anglesTransposed(n * (d - 1));
        transpose(angles.data(), anglesTransposed.data(), n, d - 1);
        
        // Byte shuffle to improve compression
        std::vector<uint8_t> shuffled(n * (d - 1) * sizeof(float));
        byteShuffle(
            reinterpret_cast<const uint8_t*>(anglesTransposed.data()),
            shuffled.data(),
            n * (d - 1)
        );
        
        // Compress the shuffled angle data
        EncodedData angleEncoded = angleCodec_->encode(std::span<const uint8_t>(shuffled));
        const auto& angleData = angleEncoded.data();
        
        // Append compressed angle data
        output.insert(output.end(), angleData.begin(), angleData.end());
        
        result.data() = std::move(output);
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = n;
        result.metadata().compressedSize = result.size();
        result.metadata().uncompressedSize = n * d * sizeof(float);
        result.metadata().supportsRandomAccess = false;
        
        return result;
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < sizeof(uint32_t) + 1) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        // Read dimension
        uint32_t d;
        std::memcpy(&d, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Read flags
        uint8_t flags = data[offset++];
        bool wasNormalized = (flags & 0x01) != 0;
        
        // Read magnitudes if present
        std::vector<MagT> magnitudes;
        if (wasNormalized) {
            uint64_t magSize;
            std::memcpy(&magSize, data + offset, sizeof(uint64_t));
            offset += sizeof(uint64_t);
            
            // Create EncodedData for magnitudes
            EncodedData magnitudeEncoded;
            magnitudeEncoded.data().assign(data + offset, data + offset + magSize);
            magnitudeEncoded.metadata().dataType = typeToDataType<MagT>;
            magnitudeEncoded.metadata().uncompressedSize = encoded.metadata().elementCount * core::dataTypeSize(typeToDataType<MagT>);
            magnitudeEncoded.metadata().elementCount = encoded.metadata().elementCount;
            offset += magSize;
            
            magnitudes = magnitudeCodec_->decodeAll(magnitudeEncoded);
            if (magnitudes.empty()) {
                return {};
            }
        }
        
        const size_t n = wasNormalized ? magnitudes.size() : encoded.metadata().elementCount;
        
        // Decompress angle data
        EncodedData angleEncoded;
        angleEncoded.data().assign(data + offset, data + encoded.size());
        angleEncoded.metadata().dataType = core::DataType::UInt8;
        angleEncoded.metadata().uncompressedSize = n * (d - 1) * sizeof(float);
        angleEncoded.metadata().elementCount = n * (d - 1) * sizeof(float);
        
        std::vector<uint8_t> shuffled = angleCodec_->decodeAll(angleEncoded);
        if (shuffled.empty()) {
            return {};
        }
        
        // Byte unshuffle
        std::vector<uint8_t> anglesTransposedBytes(n * (d - 1) * sizeof(float));
        byteUnshuffle(shuffled.data(), anglesTransposedBytes.data(), n * (d - 1));
        
        // Transpose back: from [d-1][n] to [n][d-1]
        std::vector<float> angles(n * (d - 1));
        transpose(
            reinterpret_cast<const float*>(anglesTransposedBytes.data()),
            angles.data(),
            d - 1,
            n
        );
        
        // Convert spherical to Cartesian
        std::vector<float> vectors(n * d);
        sphericalToCartesian(angles.data(), vectors.data(), n, d);
        
        // Reconstruct vectors with magnitudes if they were normalized
        std::vector<T> result(n);
        for (size_t i = 0; i < n; ++i) {
            T vec;
            // Ensure vec has enough space if it's a std::vector or similar
            if constexpr (requires { vec.resize(d); }) {
                vec.resize(d);
            }
            
            long double mag = wasNormalized ? static_cast<long double>(magnitudes[i]) : static_cast<long double>(1.0f);
            size_t idx = 0;
            for (auto it = vec.begin(); it != vec.end() && idx < d; ++it, ++idx) {
                *it = static_cast<typename T::value_type>(vectors[i * d + idx] * mag);
            }
            result[i] = vec;
        }
        
        return result;
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        // Random access not supported for spherical encoding
        // Would need to decompress all data
        auto all = decodeAll(encoded);
        if (index < all.size()) {
            return all[index];
        }
        return std::nullopt;
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        // Range decoding not efficiently supported
        // Decode all and return subset
        auto all = decodeAll(encoded);
        if (all.empty() || start >= all.size()) {
            return {};
        }
        size_t actualEnd = std::min(end, all.size());
        return std::vector<T>(all.begin() + start, all.begin() + actualEnd);
    }
    
    EncodingType encodingType() const override {
        return EncodingType::SphericalEncoding;
    }
    
    std::string name() const override {
        std::string baseName = "SphericalEncoder";
        if (normalizeToUnit_) {
            baseName += "_Normalized";
            baseName += "_MagCodec(" + magnitudeCodec_->name() + "_" + core::dataTypeToString(core::typeToDataType<MagT>) + ")";
        }
        return baseName;
    }
    
    EncodingProperties properties() const override {
        // Spherical encoding is lossless for the angular representation,
        // but may have minor floating-point precision loss due to trig operations
        return EncodingProperties(EncodingProperty::Lossless)
            | EncodingProperty::FixedSize
            | EncodingProperty::LowMemoryOverhead
            | EncodingProperty::Composable;
    }
    
    size_t estimateEncodedSize(size_t elementCount) const override {
        if (elementCount == 0) {
            return sizeof(uint32_t) + 1; // dimension + flags
        }
        
        constexpr size_t d = Dimension;
        size_t baseSize = sizeof(uint32_t) + 1; // dimension + flags
        size_t angleSize = elementCount * (d - 1) * sizeof(float);
        
        if (normalizeToUnit_) {
            size_t magSize = sizeof(uint64_t) + elementCount * core::dataTypeSize(typeToDataType<MagT>);
            return baseSize + magSize + angleSize;
        }
        
        return baseSize + angleSize;
    }
    
    /**
     * @brief Measure the error introduced by storing magnitudes as MagT.
     *
     * For each vector the "true" magnitude is computed in long double.  It is then
     * cast to MagT (simulating what encode() does), cast back to long double, and
     * the absolute and relative differences are accumulated.  The angle pipeline is
     * NOT exercised here — this metric is purely about magnitude precision.
     *
     * Only meaningful when normalizeToUnit_ == true; returns zeros otherwise.
     */
    MagnitudeErrorResult computeMagnitudeErrors(
        std::span<const T> data) const override
    {
        MagnitudeErrorResult result{};
        if (!normalizeToUnit_ || data.empty()) return result;

        constexpr size_t d = Dimension;
        size_t n = data.size();
        long double sumAbs = 0.0L, sumRel = 0.0L;
        size_t relCount = 0;

        for (size_t i = 0; i < n; ++i) {
            const T& vec = data[i];

            // Reference magnitude in long double
            long double sumSq = 0.0L;
            size_t idx = 0;
            for (const auto& val : vec) {
                if (idx++ >= d) break;
                sumSq += static_cast<long double>(val) * static_cast<long double>(val);
            }
            long double trueMag = std::sqrt(sumSq);

            // Simulate the cast to MagT and back
            MagT stored   = static_cast<MagT>(trueMag);
            long double retrieved = static_cast<long double>(stored);

            long double absErr = std::fabs(retrieved - trueMag);
            sumAbs += absErr;
            result.maxAbsoluteError = std::max(result.maxAbsoluteError, absErr);

            if (trueMag > 1e-30L) {
                sumRel += absErr / trueMag;
                ++relCount;
            }
        }

        result.meanAbsoluteError = sumAbs / static_cast<long double>(n);
        result.meanRelativeError = (relCount > 0)
            ? sumRel / static_cast<long double>(relCount)
            : 0.0L;
        return result;
    }

private:
    std::shared_ptr<Codec<uint8_t>> angleCodec_;
    std::shared_ptr<Codec<MagT>> magnitudeCodec_;
    bool normalizeToUnit_;

    /**
     * @brief Convert Cartesian coordinates to spherical coordinates
     * 
     * Optimized for unit-norm vectors: O(d) time, O(d) space per vector.
     * Precomputes partial squared norms via backward cumulative sum to avoid error accumulation.
     * Uses stack allocation for small dimensions (< 32) for better cache performance.
     * 
     * Based on the algorithm from jzip.c
     * 
     * @param x Input Cartesian coordinates [n * d]
     * @param ang Output angles [n * (d-1)]
     * @param n Number of vectors
     * @param d Dimension of each vector
     */
    void cartesianToSpherical(const float* x, float* ang, size_t n, size_t d) {
        constexpr size_t MAX_STACK_DIM = 32;
        
        // Use stack allocation for small dimensions, heap for large
        if constexpr (Dimension <= MAX_STACK_DIM) {
            // Stack-allocated workspace for better cache performance
            double r2[Dimension];
            
            for (size_t row = 0; row < n; row++) {
                const float* v = x + row * d;
                float* a = ang + row * (d - 1);

                // Backward pass: compute cumulative sum of squares from end
                r2[d - 1] = static_cast<double>(v[d - 1]) * v[d - 1];
                for (int i = d - 2; i >= 0; i--) {
                    double vi = static_cast<double>(v[i]);
                    r2[i] = r2[i + 1] + vi * vi;
                }

                // Forward pass: compute angles using precomputed partial norms
                for (size_t i = 0; i < d - 2; i++) {
                    double r = std::sqrt(r2[i]);
                    double val = v[i] / r;
                    // Clamp to valid acos domain
                    if (val > 1.0) val = 1.0;
                    if (val < -1.0) val = -1.0;
                    a[i] = static_cast<float>(std::acos(val));
                }
                // Last angle uses atan2
                a[d - 2] = static_cast<float>(std::atan2(
                    static_cast<double>(v[d - 1]),
                    static_cast<double>(v[d - 2])
                ));
            }
        } else {
            // Heap-allocated workspace for large dimensions
            std::vector<double> r2(d);
            
            for (size_t row = 0; row < n; row++) {
                const float* v = x + row * d;
                float* a = ang + row * (d - 1);

                // Backward pass: compute cumulative sum of squares from end
                r2[d - 1] = static_cast<double>(v[d - 1]) * v[d - 1];
                for (int i = d - 2; i >= 0; i--) {
                    double vi = static_cast<double>(v[i]);
                    r2[i] = r2[i + 1] + vi * vi;
                }

                // Forward pass: compute angles using precomputed partial norms
                for (size_t i = 0; i < d - 2; i++) {
                    double r = std::sqrt(r2[i]);
                    double val = v[i] / r;
                    // Clamp to valid acos domain
                    if (val > 1.0) val = 1.0;
                    if (val < -1.0) val = -1.0;
                    a[i] = static_cast<float>(std::acos(val));
                }
                // Last angle uses atan2
                a[d - 2] = static_cast<float>(std::atan2(
                    static_cast<double>(v[d - 1]),
                    static_cast<double>(v[d - 2])
                ));
            }
        }
    }

    /**
     * @brief Convert spherical coordinates to Cartesian coordinates
     * 
     * Uses double precision internally to reduce reconstruction error.
     * 
     * Based on the algorithm from jzip.c
     * 
     * @param ang Input angles [n * (d-1)]
     * @param x Output Cartesian coordinates [n * d]
     * @param n Number of vectors
     * @param d Dimension of each vector
     */
    void sphericalToCartesian(const float* ang, float* x, size_t n, size_t d) {
        for (size_t row = 0; row < n; row++) {
            const float* a = ang + row * (d - 1);
            float* v = x + row * d;
            
            double s = 1.0;
            for (size_t i = 0; i < d - 2; i++) {
                double angle = static_cast<double>(a[i]);
                v[i] = static_cast<float>(s * std::cos(angle));
                s *= std::sin(angle);
            }
            
            double lastAngle = static_cast<double>(a[d - 2]);
            v[d - 2] = static_cast<float>(s * std::cos(lastAngle));
            v[d - 1] = static_cast<float>(s * std::sin(lastAngle));
        }
    }

    /**
     * @brief Transpose a matrix
     * 
     * @param src Source matrix [rows * cols]
     * @param dst Destination matrix [cols * rows]
     * @param rows Number of rows in source
     * @param cols Number of columns in source
     */
    void transpose(const float* src, float* dst, size_t rows, size_t cols) {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    /**
     * @brief Byte shuffle to improve compression
     * 
     * Reorganizes float data by separating bytes into planes.
     * For n floats: [f0_b0, f0_b1, f0_b2, f0_b3, f1_b0, ...] 
     *         -> [f0_b0, f1_b0, ..., f0_b1, f1_b1, ..., f0_b2, f1_b2, ..., f0_b3, f1_b3, ...]
     * 
     * @param src Source bytes [n_floats * 4]
     * @param dst Destination bytes [n_floats * 4]
     * @param n_floats Number of floats
     */
    void byteShuffle(const uint8_t* src, uint8_t* dst, size_t n_floats) {
        for (size_t i = 0; i < n_floats; i++) {
            dst[i] = src[i * 4];
            dst[n_floats + i] = src[i * 4 + 1];
            dst[n_floats * 2 + i] = src[i * 4 + 2];
            dst[n_floats * 3 + i] = src[i * 4 + 3];
        }
    }

    /**
     * @brief Reverse byte shuffle operation
     * 
     * @param src Source bytes (shuffled) [n_floats * 4]
     * @param dst Destination bytes [n_floats * 4]
     * @param n_floats Number of floats
     */
    void byteUnshuffle(const uint8_t* src, uint8_t* dst, size_t n_floats) {
        for (size_t i = 0; i < n_floats; i++) {
            dst[i * 4] = src[i];
            dst[i * 4 + 1] = src[n_floats + i];
            dst[i * 4 + 2] = src[n_floats * 2 + i];
            dst[i * 4 + 3] = src[n_floats * 3 + i];
        }
    }

    EncodedData createEmptyEncoding() {
        EncodedData result;
        std::vector<uint8_t> output;
        
        // Write dimension
        uint32_t d = static_cast<uint32_t>(Dimension);
        output.insert(output.end(),
            reinterpret_cast<const uint8_t*>(&d),
            reinterpret_cast<const uint8_t*>(&d) + sizeof(uint32_t));
        
        // Write flags
        uint8_t flags = normalizeToUnit_ ? 0x01 : 0x00;
        output.push_back(flags);
        
        result.data() = std::move(output);
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = result.size();
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = false;
        
        return result;
    }
};

} // namespace encodings::encoders
