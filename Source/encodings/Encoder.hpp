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
 * @tparam T The type of data to encode (e.g., int32_t, std::string)
 */
template<typename T>
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
    virtual EncodedData encode(std::span<const T> data) = 0;
    
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
        if constexpr (PrimitiveType<T>) {
            return typeToDataType<T>;
        } else if constexpr (core::MapType<T>) {
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
template<typename T>
class Decoder {
public:
    virtual ~Decoder() = default;
    
    /**
     * @brief Decode all data at once (bulk read)
     * 
     * @param encoded The encoded data to decode
     * @return Vector containing all decoded elements
     */
    virtual std::vector<T> decodeAll(const EncodedData& encoded) = 0;
    
    /**
     * @brief Decode a single element at a specific index (random access)
     * 
     * @param encoded The encoded data
     * @param index Index of the element to decode
     * @return The decoded element, or std::nullopt if index is out of bounds
     */
    virtual std::optional<T> decodeAt(const EncodedData& encoded, size_t index) = 0;
    
    /**
     * @brief Decode a range of elements (batch random access)
     * 
     * @param encoded The encoded data
     * @param start Starting index (inclusive)
     * @param end Ending index (exclusive)
     * @return Vector containing decoded elements in the range
     */
    virtual std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) = 0;
    
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
        if constexpr (PrimitiveType<T>) {
            return typeToDataType<T>;
        } else if constexpr (core::MapType<T>) {
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
template<typename T>
class Codec : public Encoder<T>, public Decoder<T> {
public:
    virtual ~Codec() = default;
    
    // Inherit from both Encoder and Decoder
    using Encoder<T>::encode;
    using Decoder<T>::decodeAll;
    using Decoder<T>::decodeAt;
    using Decoder<T>::decodeRange;

    // Single encodingType() implementation for both interfaces
    EncodingType encodingType() const override {
        return Encoder<T>::encodingType();
    }

    // Single name() implementation for both interfaces
    std::string name() const override = 0;
    
    // Single properties() implementation for both interfaces
    EncodingProperties properties() const override = 0;
    
    // Single dataType() implementation for both interfaces
    DataType dataType() const override {
        if constexpr (PrimitiveType<T>) {
            return typeToDataType<T>;
        } else {
            return DataType::Array;
        }
    }
};

} // namespace encodings
