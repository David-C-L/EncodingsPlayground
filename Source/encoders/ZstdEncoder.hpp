#pragma once

#include <span>
#include <vector>
#include <cstring>
#include <zstd.h>
#include <algorithm>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

// Helper traits to distinguish encoder types
template<typename T>
concept ZstdPrimitiveType = PrimitiveType<T> && (!std::is_same_v<T, std::string>);

template<typename T>
concept ZstdVectorType = core::Vector32Type<T>;

/**
 * @brief Zstd encoder for primitive types
 * 
 * This encoder uses the Zstandard compression algorithm to compress data.
 *
 * Format: [Zstd compressed data]
 * 
 */
template <typename T>
requires ZstdPrimitiveType<T>
class ZstdEncoder : public Codec<T> {
public:

    ZstdEncoder(int32_t level = ZSTD_CLEVEL_DEFAULT) : level_(level) {}

    EncodedData encode(std::span<const T> data) override {
        EncodedData result;

        auto dataType = this->dataType();
        size_t bytesSize = core::dataTypeSize(dataType) * data.size();
        const size_t bound = ZSTD_compressBound(bytesSize);
        std::vector<uint8_t> output(bound);

        size_t csize = ZSTD_compress(
            output.data(), bound, data.data(), bytesSize, level_
        );

        if (ZSTD_isError(csize)) {
            // Handle compression error (e.g., log or throw)
            return {};
        }

        output.resize(csize);
        result.data() = std::move(output);
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = data.size();
        result.metadata().compressedSize = output.size();
        result.metadata().uncompressedSize = bytesSize;
        result.metadata().supportsRandomAccess = false;
        
        return result;
    }

    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() == 0) {
            return {};
        }
        
        const size_t expectedByteSize = encoded.metadata().uncompressedSize;
        const size_t expectedElemCount = expectedByteSize / sizeof(T);
        
        std::vector<uint8_t> decompressed(expectedByteSize);
        size_t dsize = ZSTD_decompress(
            decompressed.data(), expectedByteSize, 
            encoded.data().data(), encoded.size()
        );
        
        if (ZSTD_isError(dsize) || dsize != expectedByteSize) {
            return {};
        }
        
        // Convert bytes back to T
        std::vector<T> result(expectedElemCount);
        std::memcpy(result.data(), decompressed.data(), expectedByteSize);
        
        return result;
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        // Zstd doesn't support random access - must decompress all
        auto all = decodeAll(encoded);
        if (all.empty() || index >= all.size()) {
            return std::nullopt;
        }
        return all[index];
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        // Zstd doesn't support efficient range decoding - must decompress all
        auto all = decodeAll(encoded);
        if (all.empty() || start >= all.size()) {
            return {};
        }
        size_t actualEnd = std::min(end, all.size());
        return std::vector<T>(all.begin() + start, all.begin() + actualEnd);
    }
    
    EncodingType encodingType() const override {
        return EncodingType::Zstd;
    }
    
    std::string name() const override {
        return "Zstd" + std::to_string(level_);
    }
    
    EncodingProperties properties() const override {
        return EncodingProperty::Lossless;
    }
    
    size_t estimateEncodedSize(size_t elementCount) const override {
        return ZSTD_compressBound(elementCount * sizeof(T));
    }
    
private:

    int32_t level_;
};

/**
 * @brief ZstdEncoder for Vector32Type (e.g., std::vector<float>)
 * 
 * Treats vectors as contiguous streams of floats and compresses them with Zstd.
 * 
 * Format: [vector_count (8 bytes), dimension (8 bytes), Zstd compressed float data]
 * 
 * @tparam T The vector type (must satisfy Vector32Type)
 */
template<typename T>
    requires ZstdVectorType<T>
class ZstdVectorEncoder : public Codec<T> {
public:
    ZstdVectorEncoder(int32_t level = ZSTD_CLEVEL_DEFAULT) : level_(level) {}

    EncodedData encode(std::span<const T> data) override {
        using namespace core;
        
        EncodedData result;
        
        if (data.empty()) {
            // Empty encoding
            result.data().resize(2 * sizeof(size_t));
            size_t zero = 0;
            std::memcpy(result.data().data(), &zero, sizeof(size_t));
            std::memcpy(result.data().data() + sizeof(size_t), &zero, sizeof(size_t));
            
            result.metadata().encodingName = name();
            result.metadata().dataType = DataType::Float32;
            result.metadata().elementCount = 0;
            result.metadata().compressedSize = result.data().size();
            result.metadata().uncompressedSize = 0;
            result.metadata().supportsRandomAccess = false;
            
            return result;
        }
        
        const size_t vectorCount = data.size();
        const size_t dimension = data[0].size();
        const size_t floatCount = vectorCount * dimension;
        const size_t floatBytes = floatCount * sizeof(float);
        const size_t headerSize = 2 * sizeof(size_t);
        
        // Flatten vectors into contiguous float array
        std::vector<float> flatData;
        flatData.reserve(floatCount);
        for (const auto& vec : data) {
            for (const auto& val : vec) {
                flatData.push_back(val);
            }
        }
        
        // Compress the flat float data
        const size_t bound = ZSTD_compressBound(floatBytes);
        std::vector<uint8_t> compressed(headerSize + bound);
        
        // Write header
        std::memcpy(compressed.data(), &vectorCount, sizeof(size_t));
        std::memcpy(compressed.data() + sizeof(size_t), &dimension, sizeof(size_t));
        
        // Compress
        size_t csize = ZSTD_compress(
            compressed.data() + headerSize, bound,
            flatData.data(), floatBytes,
            level_
        );
        
        if (ZSTD_isError(csize)) {
            return {};
        }
        
        compressed.resize(headerSize + csize);
        result.data() = std::move(compressed);
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = DataType::Float32;
        result.metadata().elementCount = vectorCount;
        result.metadata().compressedSize = result.data().size();
        result.metadata().uncompressedSize = floatBytes;
        result.metadata().supportsRandomAccess = false;
        
        return result;
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < 2 * sizeof(size_t)) {
            return {};
        }
        
        // Read header
        size_t vectorCount;
        size_t dimension;
        std::memcpy(&vectorCount, encoded.data().data(), sizeof(size_t));
        std::memcpy(&dimension, encoded.data().data() + sizeof(size_t), sizeof(size_t));
        
        if (vectorCount == 0) {
            return {};
        }
        
        const size_t headerSize = 2 * sizeof(size_t);
        const size_t floatCount = vectorCount * dimension;
        const size_t floatBytes = floatCount * sizeof(float);
        
        // Decompress float data
        std::vector<float> flatData(floatCount);
        size_t dsize = ZSTD_decompress(
            flatData.data(), floatBytes,
            encoded.data().data() + headerSize, encoded.size() - headerSize
        );
        
        if (ZSTD_isError(dsize) || dsize != floatBytes) {
            return {};
        }
        
        // Reconstruct vectors
        std::vector<T> result;
        result.reserve(vectorCount);
        
        for (size_t i = 0; i < vectorCount; ++i) {
            T vec;
            if constexpr (requires { vec.resize(dimension); }) {
                vec.resize(dimension);
            }
            
            // Copy dimension floats
            std::memcpy(vec.data(), flatData.data() + i * dimension, dimension * sizeof(float));
            
            result.push_back(std::move(vec));
        }
        
        return result;
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        // Zstd doesn't support random access - must decompress all
        auto all = decodeAll(encoded);
        if (all.empty() || index >= all.size()) {
            return std::nullopt;
        }
        return all[index];
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        // Zstd doesn't support efficient range decoding - must decompress all
        auto all = decodeAll(encoded);
        if (all.empty() || start >= all.size()) {
            return {};
        }
        size_t actualEnd = std::min(end, all.size());
        return std::vector<T>(all.begin() + start, all.begin() + actualEnd);
    }
    
    EncodingType encodingType() const override {
        return EncodingType::Zstd;
    }
    
    std::string name() const override {
        return "Zstd" + std::to_string(level_);
    }
    
    EncodingProperties properties() const override {
        return EncodingProperty::Lossless;
    }
    
    size_t estimateEncodedSize(size_t vectorCount) const override {
        // Estimate assuming some typical dimension (e.g., 128)
        constexpr size_t typicalDimension = 128;
        return 2 * sizeof(size_t) + ZSTD_compressBound(vectorCount * typicalDimension * sizeof(float));
    }
    
private:
    int32_t level_;
};

} // namespace encodings::encoders
