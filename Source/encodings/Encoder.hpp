#pragma once

#include <span>
#include <vector>
#include <memory>
#include <string>
#include <concepts>
#include "core/DataType.hpp"
#include "EncodedData.hpp"
#include "EncodingProperty.hpp"
#include "EncodingType.hpp"

namespace encodings {

    using core::DataType;
    using core::PrimitiveType;
    using core::MapType;
    using core::typeToDataType;

/**
 * @brief Abstract base class for encoding strategies
 * 
 * @tparam TIn The type of data to encode (e.g., int32_t, std::string)
 * @tparam TOut The type of data produced by the encoder (default is uint8_t for byte-oriented encodings)
 */
template<typename TIn, typename TOut = uint8_t>
class Encoder {
public:
    virtual ~Encoder() = default;
    
    /**
     * @brief Encode a span of data elements
     * 
     * @param data Input data to encode (span is a lightweight view, no copy of data occurs)
     * @return EncodedData containing compressed bytes and metadata
     * 
     * @note std::span is passed by value following C++ standard library conventions.
     *       The underlying data is not copied; span is merely a view (pointer + size).
     */
    virtual EncodedBuffer<TOut> encode(std::span<const TIn> data) = 0;
    
    /**
     * @brief Get the encoding type of this encoding scheme
     */
    virtual EncodingType encodingType() const = 0;

    /**
     * @brief Get the name of this encoding scheme
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get the properties of this encoding scheme
     * 
     * This provides a rich set of metadata about the encoding's capabilities
     * and characteristics. Prefer using this over individual boolean methods.
     */
    virtual EncodingProperties properties() const = 0;
    
    /**
     * @brief Get the data type this encoder handles
     */
    virtual DataType dataType() const {
        if constexpr (PrimitiveType<TIn>) {
            return typeToDataType<TIn>;
        } else if constexpr (core::MapType<TIn>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }

    /**
     * @brief Get the data type this encoder produces
     */
    virtual DataType encodedType() const {
        if constexpr (PrimitiveType<TOut>) {
            return typeToDataType<TOut>;
        } else if constexpr (core::MapType<TOut>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }
    
    /**
     * @brief Estimate the encoded size without actually encoding
     * Useful for pre-allocation
     * 
     * @param elementCount Number of elements to encode
     * @return Estimated size in bytes (0 if cannot estimate)
     */
    virtual size_t estimateEncodedSize(size_t /* elementCount */) const {
        return 0; // Default: cannot estimate
    }
};

/**
 * @brief Abstract base class for decoding strategies
 * 
 * @tparam T The type of data to decode
 */
template<typename TIn, typename TOut = uint8_t>
class Decoder {
public:
    virtual ~Decoder() = default;
    
    /**
     * @brief Decode all data at once (bulk read)
     * 
     * @param encoded The encoded data to decode
     * @return Vector containing all decoded elements
     */
    virtual std::vector<TIn> decodeAll(const EncodedBuffer<TOut>& encoded) = 0;
    
    /**
     * @brief Decode a single element at a specific index (random access)
     * 
     * @param encoded The encoded data
     * @param index Index of the element to decode
     * @return The decoded element, or std::nullopt if index is out of bounds
     */
    virtual std::optional<TIn> decodeAt(const EncodedBuffer<TOut>& encoded, size_t index) = 0;
    
    /**
     * @brief Decode a range of elements (batch random access)
     * 
     * @param encoded The encoded data
     * @param start Starting index (inclusive)
     * @param end Ending index (exclusive)
     * @return Vector containing decoded elements in the range
     */
    virtual std::vector<TIn> decodeRange(const EncodedBuffer<TOut>& encoded, size_t start, size_t end) = 0;
    
    /**
     * @brief Get the encoding type of this decoding scheme (should match encoder)
     */
    virtual EncodingType encodingType() const = 0;

    /**
     * @brief Get the name of this decoding scheme (should match encoder)
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Get the properties of this decoding scheme
     * Should match the encoder's properties
     */
    virtual EncodingProperties properties() const = 0;
    
    /**
     * @brief Get the data type this decoder handles
     */
    virtual DataType dataType() const {
        if constexpr (PrimitiveType<TIn>) {
            return typeToDataType<TIn>;
        } else if constexpr (core::MapType<TIn>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }

    /**
     * @brief Get the data type this decoder expects as input (encoded type)
     */
    virtual DataType encodedType() const {
        if constexpr (PrimitiveType<TOut>) {
            return typeToDataType<TOut>;
        } else if constexpr (core::MapType<TOut>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }
};

/**
 * @brief Combined encoder/decoder interface for convenience
 * 
 * Many encoding schemes naturally implement both encoding and decoding,
 * so this class provides a unified interface.
 * 
 * @tparam T The type of data to encode/decode
 */
template<typename TIn, typename TOut = uint8_t>
class Codec : public Encoder<TIn, TOut>, public Decoder<TIn, TOut> {
public:
    virtual ~Codec() = default;
    
    // Inherit from both Encoder and Decoder
    using Encoder<TIn, TOut>::encode;
    using Decoder<TIn, TOut>::decodeAll;
    using Decoder<TIn, TOut>::decodeAt;
    using Decoder<TIn, TOut>::decodeRange;

    // Per-sub-stream decode profiling — overridden by SubIntSplitEncoder<T, true> only.
    // Default implementations return empty / no-op so all other codecs are unaffected.

    /// Per-section decode times (ns) from the most recent decodeAll() call.
    /// Index matches subStreamEncodeMetrics order in the encoded buffer's metadata.
    virtual std::vector<int64_t> subStreamBulkDecodeTimeNs()   const { return {}; }

    /// Per-section accumulated ns across decodeAt() calls since the last reset.
    virtual std::vector<int64_t> subStreamDecodeAtAccumNs()    const { return {}; }

    /// Per-section accumulated ns across decodeRange() calls since the last reset.
    virtual std::vector<int64_t> subStreamDecodeRangeAccumNs() const { return {}; }

    /// Zero the decodeAt accumulator — call before each random-access benchmark loop.
    virtual void resetSubStreamDecodeAtAccum()    {}

    /// Zero the decodeRange accumulator — call before each range-access benchmark loop.
    virtual void resetSubStreamDecodeRangeAccum() {}

    // Single encodingType() implementation for both interfaces
    EncodingType encodingType() const override = 0;

    // Single name() implementation for both interfaces
    std::string name() const override = 0;
    
    // Single properties() implementation for both interfaces
    EncodingProperties properties() const override = 0;
    
    // Single dataType() implementation for both interfaces
    DataType dataType() const override {
        if constexpr (PrimitiveType<TIn>) {
            return typeToDataType<TIn>;
        } else if constexpr (core::MapType<TIn>) {
            return DataType::Map;
        } else if constexpr (core::Vector32Type<TIn>) {
            return DataType::Vector32;
        } else {
            return DataType::Array; // Default for composite types
        }
    }

    // Single encodedType() implementation for both interfaces
    DataType encodedType() const override {
        if constexpr (PrimitiveType<TOut>) {
            return typeToDataType<TOut>;
        } else if constexpr (core::MapType<TOut>) {
            return DataType::Map;
        } else if constexpr (core::Vector32Type<TOut>) {
            return DataType::Vector32;
        } else {
            return DataType::Array; // Default for composite types
        }
    }
};

} // namespace encodings
