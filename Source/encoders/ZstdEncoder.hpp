#pragma once

#include <span>
#include <vector>
#include <cstring>
#include <stdexcept>
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
 * Format:
 * - BlockSize == 0 (default): [Zstd compressed data] — single frame over the whole span.
 * - BlockSize != 0: [uint64 blockCount] followed by, for each block,
 *   [uint64 elems][uint64 compSize][compressed block bytes]. Splitting into
 *   independently-compressed blocks lets decodeAt/decodeRange skip blocks that
 *   don't overlap the requested index/range instead of always decompressing
 *   the whole payload.
 */
template <typename T, size_t BlockSize = 0>
requires ZstdPrimitiveType<T>
class ZstdEncoder : public Codec<T> {
public:

    ZstdEncoder(int32_t level = ZSTD_CLEVEL_DEFAULT) : level_(level) {}

    EncodedData encode(std::span<const T> data) override {
        EncodedData result;

        const size_t bytesSize = core::dataTypeSize(this->dataType()) * data.size();

        std::vector<uint8_t> buffer;

        if constexpr (BlockSize == 0) {
            const size_t bound = ZSTD_compressBound(bytesSize);
            buffer.resize(bound);

            size_t csize = ZSTD_compress(
                buffer.data(), bound, data.data(), bytesSize, level_
            );

            if (ZSTD_isError(csize)) {
                // Handle compression error (e.g., log or throw)
                return {};
            }

            buffer.resize(csize);
        } else {
            const size_t totalElems = data.size();
            const size_t blockCount = (totalElems + BlockSize - 1) / BlockSize;
            appendUint64(buffer, static_cast<uint64_t>(blockCount));

            size_t offset = 0;
            for (size_t b = 0; b < blockCount; ++b) {
                const size_t elems = std::min(BlockSize, totalElems - offset);
                const auto blockSpan = data.subspan(offset, elems);
                std::vector<uint8_t> compressed = compressBlock(blockSpan);

                appendUint64(buffer, static_cast<uint64_t>(elems));
                appendUint64(buffer, static_cast<uint64_t>(compressed.size()));
                buffer.insert(buffer.end(), compressed.begin(), compressed.end());

                offset += elems;
            }
        }

        result.data() = std::move(buffer);

        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = data.size();
        result.metadata().compressedSize = result.data().size();
        result.metadata().uncompressedSize = bytesSize;
        result.metadata().supportsRandomAccess = (BlockSize != 0);

        return result;
    }

    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() == 0) {
            return {};
        }

        const size_t expectedByteSize = encoded.metadata().uncompressedSize;
        const size_t expectedElemCount = expectedByteSize / sizeof(T);

        if constexpr (BlockSize == 0) {
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
        } else {
            const uint8_t* p = encoded.data().data();
            const uint8_t* end = p + encoded.data().size();
            if (static_cast<size_t>(end - p) < sizeof(uint64_t)) {
                throw std::runtime_error("ZstdEncoder: corrupted block header (count)");
            }
            const uint64_t blockCount = readUint64(p);

            std::vector<T> out;
            out.reserve(expectedElemCount);

            for (uint64_t b = 0; b < blockCount; ++b) {
                if (static_cast<size_t>(end - p) < 2 * sizeof(uint64_t)) {
                    throw std::runtime_error("ZstdEncoder: corrupted block header (sizes)");
                }
                const uint64_t elems = readUint64(p);
                const uint64_t compSize = readUint64(p);
                if (static_cast<size_t>(end - p) < compSize) {
                    throw std::runtime_error("ZstdEncoder: corrupted block payload");
                }
                std::vector<T> block = decompressSingle(p, static_cast<size_t>(compSize), static_cast<size_t>(elems));
                out.insert(out.end(), block.begin(), block.end());
                p += compSize;
            }

            if (out.size() != expectedElemCount) {
                throw std::runtime_error("ZstdEncoder: decoded element count mismatch");
            }
            return out;
        }
    }

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const size_t bytes = encoded.metadata().uncompressedSize;
        if (bytes == 0) return std::nullopt;
        const size_t N = bytes / sizeof(T);
        if (index >= N) return std::nullopt;

        if constexpr (BlockSize == 0) {
            auto all = decodeAll(encoded);
            if (all.empty() || index >= all.size()) return std::nullopt;
            return all[index];
        } else {
            const uint8_t* p = encoded.data().data();
            const uint8_t* end = p + encoded.data().size();
            if (static_cast<size_t>(end - p) < sizeof(uint64_t)) {
                throw std::runtime_error("ZstdEncoder: corrupted block header (count)");
            }
            const uint64_t blockCount = readUint64(p);

            uint64_t base = 0;
            for (uint64_t b = 0; b < blockCount; ++b) {
                if (static_cast<size_t>(end - p) < 2 * sizeof(uint64_t)) {
                    throw std::runtime_error("ZstdEncoder: corrupted block header (sizes)");
                }
                const uint64_t elems = readUint64(p);
                const uint64_t compSize = readUint64(p);
                if (static_cast<size_t>(end - p) < compSize) {
                    throw std::runtime_error("ZstdEncoder: corrupted block payload");
                }
                if (index < base + elems) {
                    std::vector<T> block = decompressSingle(p, static_cast<size_t>(compSize), static_cast<size_t>(elems));
                    return block[static_cast<size_t>(index - base)];
                }
                base += elems;
                p += compSize;
            }
            return std::nullopt;
        }
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        const size_t bytes = encoded.metadata().uncompressedSize;
        if (bytes == 0) return {};
        const size_t N = bytes / sizeof(T);
        if (start >= N) return {};
        end = std::min(end, N);

        if constexpr (BlockSize == 0) {
            auto all = decodeAll(encoded);
            if (all.empty() || start >= all.size()) return {};
            size_t actualEnd = std::min(end, all.size());
            return std::vector<T>(all.begin() + static_cast<ptrdiff_t>(start),
                                  all.begin() + static_cast<ptrdiff_t>(actualEnd));
        } else {
            const uint8_t* p = encoded.data().data();
            const uint8_t* readEnd = p + encoded.data().size();
            if (static_cast<size_t>(readEnd - p) < sizeof(uint64_t)) {
                throw std::runtime_error("ZstdEncoder: corrupted block header (count)");
            }
            const uint64_t blockCount = readUint64(p);

            std::vector<T> out;
            out.reserve(end - start);
            uint64_t base = 0;
            for (uint64_t b = 0; b < blockCount && base < end; ++b) {
                if (static_cast<size_t>(readEnd - p) < 2 * sizeof(uint64_t)) {
                    throw std::runtime_error("ZstdEncoder: corrupted block header (sizes)");
                }
                const uint64_t elems = readUint64(p);
                const uint64_t compSize = readUint64(p);
                if (static_cast<size_t>(readEnd - p) < compSize) {
                    throw std::runtime_error("ZstdEncoder: corrupted block payload");
                }

                const uint64_t blockEnd = base + elems;
                if (blockEnd > start && base < end) {
                    // Overlaps requested range
                    std::vector<T> block = decompressSingle(p, static_cast<size_t>(compSize), static_cast<size_t>(elems));
                    const size_t localStart = static_cast<size_t>(std::max<uint64_t>(start, base) - base);
                    const size_t localEnd = static_cast<size_t>(std::min<uint64_t>(end, blockEnd) - base);
                    out.insert(out.end(), block.begin() + static_cast<ptrdiff_t>(localStart),
                                          block.begin() + static_cast<ptrdiff_t>(localEnd));
                }

                base = blockEnd;
                p += compSize;
            }

            return out;
        }
    }

    EncodingType encodingType() const override {
        return EncodingType::Zstd;
    }

    std::string name() const override {
        if constexpr (BlockSize == 0) {
            return "Zstd" + std::to_string(level_);
        } else {
            return "Zstd" + std::to_string(level_) + "_b" + std::to_string(BlockSize);
        }
    }

    EncodingProperties properties() const override {
        EncodingProperties props = EncodingProperty::Lossless;
        if constexpr (BlockSize != 0) {
            props.add(EncodingProperty::RandomAccess);
        }
        return props;
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        return ZSTD_compressBound(elementCount * sizeof(T));
    }

private:

    std::vector<uint8_t> compressBlock(std::span<const T> data) const {
        const size_t bytesSize = core::dataTypeSize(this->dataType()) * data.size();
        const size_t bound = ZSTD_compressBound(bytesSize);
        std::vector<uint8_t> output(bound);

        size_t csize = ZSTD_compress(
            output.data(), bound, data.data(), bytesSize, level_
        );

        if (ZSTD_isError(csize)) {
            throw std::runtime_error("ZstdEncoder: block compress failed");
        }

        output.resize(csize);
        return output;
    }

    std::vector<T> decompressSingle(const uint8_t* data, size_t size, size_t elemCount) const {
        const size_t expectedByteSize = elemCount * sizeof(T);
        std::vector<uint8_t> decompressed(expectedByteSize);
        size_t dsize = ZSTD_decompress(decompressed.data(), expectedByteSize, data, size);

        if (ZSTD_isError(dsize) || dsize != expectedByteSize) {
            throw std::runtime_error("ZstdEncoder: block decompress failed");
        }

        std::vector<T> result(elemCount);
        std::memcpy(result.data(), decompressed.data(), expectedByteSize);
        return result;
    }

    static void appendUint64(std::vector<uint8_t>& buf, uint64_t v) {
        uint8_t tmp[sizeof(uint64_t)];
        std::memcpy(tmp, &v, sizeof(uint64_t));
        buf.insert(buf.end(), tmp, tmp + sizeof(uint64_t));
    }

    static uint64_t readUint64(const uint8_t*& p) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(uint64_t));
        p += sizeof(uint64_t);
        return v;
    }

    int32_t level_;
};

// Convenience factory
template <typename T, size_t BlockSize = 0>
std::shared_ptr<ZstdEncoder<T, BlockSize>> makeZstdEncoder(int32_t level = ZSTD_CLEVEL_DEFAULT) {
    return std::make_shared<ZstdEncoder<T, BlockSize>>(level);
}

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
