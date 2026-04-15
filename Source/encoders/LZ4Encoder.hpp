#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

#if defined(__has_include)
#  if __has_include(<lz4.h>)
#    include <lz4.h>
#    define HAVE_LZ4 1
#  endif
#endif

namespace encodings::encoders {

// LZ4 (fast mode) codec for integral/trivially-copyable numeric streams.
//
// Wire format:
//   [8 bytes] elementCount
//   [8 bytes] uncompressedBytes
//   [8 bytes] compressedBytes
//   [compressedBytes] payload from LZ4_compress_default
//
// Note: A high-compression variant (LZ4HC) can be added as a separate codec.
template <typename T>
    requires std::is_integral_v<T>
class LZ4Encoder final : public Codec<T> {
public:
    using EncodedData = EncodedBuffer<uint8_t>;

    static constexpr size_t kHeaderBytes = 8 + 8 + 8;

    EncodedData encode(std::span<const T> data) override {
#ifndef HAVE_LZ4
        throw std::runtime_error("LZ4Encoder: LZ4 headers not found. Install liblz4-dev and rebuild.");
#else
        EncodedData out;
        out.metadata().encodingName = name();
        out.metadata().dataType = this->dataType();
        out.metadata().elementCount = data.size();
        out.metadata().supportsRandomAccess = true; // fallback decodeAt/decodeRange via decodeAll for benchmarking compatibility

        const uint64_t elementCount = static_cast<uint64_t>(data.size());
        const uint64_t uncompressedBytes = static_cast<uint64_t>(data.size() * sizeof(T));

        std::vector<uint8_t> bytes;
        bytes.resize(kHeaderBytes);
        std::memcpy(bytes.data() + 0, &elementCount, sizeof(uint64_t));
        std::memcpy(bytes.data() + 8, &uncompressedBytes, sizeof(uint64_t));

        if (data.empty()) {
            const uint64_t compressedBytes = 0;
            std::memcpy(bytes.data() + 16, &compressedBytes, sizeof(uint64_t));
            out.data() = std::move(bytes);
            out.metadata().compressedSize = out.data().size();
            out.metadata().uncompressedSize = 0;
            return out;
        }

        if (uncompressedBytes > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("LZ4Encoder: input too large for LZ4_compress_default (int size limit)");
        }

        const int srcSize = static_cast<int>(uncompressedBytes);
        const int maxDst = LZ4_compressBound(srcSize);
        if (maxDst <= 0) {
            throw std::runtime_error("LZ4Encoder: invalid LZ4_compressBound result");
        }

        const size_t payloadOffset = bytes.size();
        bytes.resize(payloadOffset + static_cast<size_t>(maxDst));

        const int written = LZ4_compress_default(
            reinterpret_cast<const char*>(data.data()),
            reinterpret_cast<char*>(bytes.data() + payloadOffset),
            srcSize,
            maxDst);

        if (written <= 0) {
            throw std::runtime_error("LZ4Encoder: compression failed");
        }

        const uint64_t compressedBytes = static_cast<uint64_t>(written);
        // IMPORTANT: bytes may have reallocated during resize() above, so always
        // re-address header fields through bytes.data() instead of a cached pointer.
        std::memcpy(bytes.data() + 16, &compressedBytes, sizeof(uint64_t));

        bytes.resize(payloadOffset + static_cast<size_t>(written));

        out.data() = std::move(bytes);
        out.metadata().compressedSize = out.data().size();
        out.metadata().uncompressedSize = static_cast<size_t>(uncompressedBytes);
        return out;
#endif
    }

    std::vector<T> decodeAll(const EncodedData& encoded) override {
#ifndef HAVE_LZ4
        throw std::runtime_error("LZ4Encoder: LZ4 headers not found. Install liblz4-dev and rebuild.");
#else
        const auto& buf = encoded.data();
        if (buf.size() < kHeaderBytes) {
            throw std::runtime_error("LZ4Encoder::decodeAll: buffer too small for header");
        }

        const uint8_t* p = buf.data();
        uint64_t elementCount = 0;
        uint64_t uncompressedBytes = 0;
        uint64_t compressedBytes = 0;
        std::memcpy(&elementCount, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        std::memcpy(&uncompressedBytes, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        std::memcpy(&compressedBytes, p, sizeof(uint64_t)); p += sizeof(uint64_t);

        if (uncompressedBytes == 0 || elementCount == 0) {
            return {};
        }
        if (uncompressedBytes % sizeof(T) != 0) {
            throw std::runtime_error("LZ4Encoder::decodeAll: uncompressed byte count is not aligned to element size");
        }

        const size_t payloadOffset = kHeaderBytes;
        if (payloadOffset + static_cast<size_t>(compressedBytes) > buf.size()) {
            throw std::runtime_error("LZ4Encoder::decodeAll: payload exceeds buffer size");
        }
        if (payloadOffset + static_cast<size_t>(compressedBytes) != buf.size()) {
            throw std::runtime_error("LZ4Encoder::decodeAll: header compressed size does not match buffer length");
        }

        if (uncompressedBytes > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            compressedBytes > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("LZ4Encoder::decodeAll: size exceeds LZ4 int API limits");
        }

        std::vector<uint8_t> raw(static_cast<size_t>(uncompressedBytes));
        const int decoded = LZ4_decompress_safe(
            reinterpret_cast<const char*>(buf.data() + payloadOffset),
            reinterpret_cast<char*>(raw.data()),
            static_cast<int>(compressedBytes),
            static_cast<int>(uncompressedBytes));

        if (decoded < 0 || static_cast<uint64_t>(decoded) != uncompressedBytes) {
            throw std::runtime_error("LZ4Encoder::decodeAll: decompression failed or size mismatch");
        }

        std::vector<T> out(static_cast<size_t>(uncompressedBytes / sizeof(T)));
        std::memcpy(out.data(), raw.data(), static_cast<size_t>(uncompressedBytes));

        if (out.size() != static_cast<size_t>(elementCount)) {
            throw std::runtime_error("LZ4Encoder::decodeAll: element count mismatch");
        }

        return out;
#endif
    }

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        auto all = decodeAll(encoded);
        if (index >= all.size()) return std::nullopt;
        return all[index];
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (start >= end) return {};
        auto all = decodeAll(encoded);
        if (start >= all.size()) return {};
        end = std::min(end, all.size());
        return std::vector<T>(all.begin() + static_cast<std::ptrdiff_t>(start),
                              all.begin() + static_cast<std::ptrdiff_t>(end));
    }

    EncodingType encodingType() const override { return EncodingType::LZ4; }

    std::string name() const override { return "LZ4"; }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::Lossless)
             | EncodingProperty::PreservesOrder
             | EncodingProperty::RequiresFullData
             | EncodingProperty::VariableSize
             | EncodingProperty::RandomAccess;
    }
};

} // namespace encodings::encoders
