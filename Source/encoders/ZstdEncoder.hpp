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

namespace encodings::encoders {

/**
 * @brief Zstd encoder for byte data
 * 
 * This encoder uses the Zstandard compression algorithm to compress data.
 *
 * Format: [Zstd compressed data]
 * 
 */
template <typename T>
requires PrimitiveType<T> && (!std::is_same_v<T, std::string>)
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

} // namespace encodings::encoders
