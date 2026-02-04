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
class EncodedData {
public:
    EncodedData() = default;
    
    explicit EncodedData(std::vector<uint8_t> data, EncodingMetadata metadata)
        : data_(std::move(data)), metadata_(std::move(metadata)) {}
    
    // Access to encoded bytes
    const std::vector<uint8_t>& data() const { return data_; }
    std::vector<uint8_t>& data() { return data_; }
    
    // Access to metadata
    const EncodingMetadata& metadata() const { return metadata_; }
    EncodingMetadata& metadata() { return metadata_; }
    
    // Size information
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    
    // Utility methods
    void reserve(size_t capacity) { data_.reserve(capacity); }
    void clear() { data_.clear(); }
    
private:
    std::vector<uint8_t> data_;
    EncodingMetadata metadata_;
};

} // namespace encodings
