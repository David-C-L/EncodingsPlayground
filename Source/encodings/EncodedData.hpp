#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <optional>
#include "core/DataType.hpp"

namespace encodings {

    using core::DataType;

/**
 * @brief Metadata about an encoding scheme
 */
struct EncodingMetadata {
    std::string encodingName;
    DataType dataType;
    size_t elementCount;
    size_t compressedSize;
    size_t uncompressedSize;
    bool supportsRandomAccess;
    
    DataType encodedType = DataType::UInt8; // Default to byte-oriented encoding, can be overridden for other encodings
    // Additional encoding-specific metadata can be stored here
    std::map<std::string, std::string> customMetadata;
    
    /**
     * @brief Calculate compression ratio
     */
    double compressionRatio() const {
        return uncompressedSize > 0 
            ? static_cast<double>(compressedSize) / uncompressedSize 
            : 0.0;
    }
    
    /**
     * @brief Calculate space savings percentage
     */
    double spaceSavingsPercent() const {
        return uncompressedSize > 0
            ? (1.0 - compressionRatio()) * 100.0
            : 0.0;
    }
};

/**
 * @brief Container for encoded data with metadata
 * 
 * This class holds the raw encoded bytes along with metadata about
 * the encoding scheme used and the original data characteristics.
 */
template<typename TOut>
class EncodedBuffer {
public:
    EncodedBuffer() = default;

    explicit EncodedBuffer(std::vector<TOut> data, EncodingMetadata metadata)
        : data_(std::move(data)), metadata_(std::move(metadata)) {}
    
    // Access to encoded bytes
    const std::vector<TOut>& data() const { return data_; }
    std::vector<TOut>& data() { return data_; }
    
    // Access to metadata
    const EncodingMetadata& metadata() const { return metadata_; }
    EncodingMetadata& metadata() { return metadata_; }
    
    // Size information
    size_t size() const {
        auto typeSize = core::dataTypeSize(core::typeToDataType<TOut>);
        return data_.size() * typeSize;
    }
    bool empty() const { return data_.empty(); }
    
    // Utility methods
    void reserve(size_t capacity) { data_.reserve(capacity); }
    void clear() { data_.clear(); }
    
private:
    std::vector<TOut> data_;
    EncodingMetadata metadata_;
};

using EncodedData = EncodedBuffer<uint8_t>;

} // namespace encodings
