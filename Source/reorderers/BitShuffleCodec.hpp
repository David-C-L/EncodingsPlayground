#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "Reorderer.hpp"  // for ReorderableType concept

namespace encodings::reorderers {

// ---------------------------------------------------------------------------
// BitShuffleCodec<T, InnerCodec>
//
// Transposes the N × (8*sizeof(T)) bit-matrix: bit b of element i goes to
// position b*N + i.  Each bit-plane becomes a contiguous byte run, making
// entropy coders and byte-level compressors (LZ4, Zstd) much more effective
// for data with correlated bit patterns (timestamps, Snowflake IDs, etc.).
//
// This is NOT a Reorderer<T>; it is a Codec<T, uint8_t> in its own right.
// The transform is its own inverse (bitshuffle ∘ bitshuffle = identity).
//
// Wire format:
//   [N:8][bits_per_elem:1][inner codec bytes]
//
// decodeAt(i): reassemble element i by reading bit b from plane b's byte at
// position (b*N + i)/8, bit (b*N + i)%8.  O(bits_per_elem) per access.
// ---------------------------------------------------------------------------

template <ReorderableType T>
class BitShuffleCodec : public encodings::Codec<T, uint8_t> {
public:
    static constexpr uint32_t kBitsPerElem = static_cast<uint32_t>(sizeof(T) * 8);

    explicit BitShuffleCodec(std::shared_ptr<encodings::Codec<T, uint8_t>> innerCodec)
        : inner_(std::move(innerCodec)) {}

    // -----------------------------------------------------------------------
    // Encode: bitshuffle → inner codec
    // -----------------------------------------------------------------------

    encodings::EncodedData encode(std::span<const T> data) override {
        const size_t N = data.size();

        // Bitshuffle: transpose N × kBitsPerElem bit matrix
        std::vector<uint8_t> shuffled = bitshuffle(data);

        // Inner codec encodes the shuffled bytes reinterpreted as T values.
        // We store shuffled bytes directly + the inner codec wraps them.
        // For simplicity: inner codec receives the shuffled bytes wrapped as T[].
        // Better approach: store shuffled bytes + pass to a byte-level codec.
        // Here we store [N:8][bits:1][shuffled bytes] and skip inner for now
        // (caller can chain with LZ4/Zstd via the full pipeline).
        // TODO: integrate inner codec when byte-level Codec<uint8_t,uint8_t> exists.

        std::vector<uint8_t> out;
        out.reserve(9 + shuffled.size());
        for (int b = 0; b < 8; ++b) out.push_back(static_cast<uint8_t>(N >> (8 * b)));
        out.push_back(static_cast<uint8_t>(kBitsPerElem));
        out.insert(out.end(), shuffled.begin(), shuffled.end());

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::typeToDataType<T>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(T);
        meta.supportsRandomAccess = true; // O(kBitsPerElem) per element
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // -----------------------------------------------------------------------
    // Decode all: bitdeshuffle (same transform, self-inverse)
    // -----------------------------------------------------------------------

    std::vector<T> decodeAll(const encodings::EncodedData& encoded) override {
        const auto& raw = encoded.data();
        const size_t N  = parseN(raw);
        std::span<const uint8_t> shuffled(raw.data() + 9, raw.size() - 9);
        return bitdeshuffle(shuffled, N);
    }

    // -----------------------------------------------------------------------
    // Random access: reconstruct one element from bit-planes. O(kBitsPerElem).
    // -----------------------------------------------------------------------

    std::optional<T> decodeAt(const encodings::EncodedData& encoded, size_t index) override {
        const auto& raw = encoded.data();
        const size_t N  = parseN(raw);
        if (index >= N) return std::nullopt;

        const uint8_t* planes = raw.data() + 9;
        using U = std::make_unsigned_t<T>;
        U result = 0;
        for (uint32_t b = 0; b < kBitsPerElem; ++b) {
            const size_t bitPos  = b * N + index;
            const size_t bytePos = bitPos / 8;
            const uint32_t bitOff = static_cast<uint32_t>(bitPos % 8);
            const uint8_t  bit   = (planes[bytePos] >> bitOff) & 1u;
            result |= static_cast<U>(bit) << b;
        }
        return static_cast<T>(result);
    }

    std::vector<T> decodeRange(const encodings::EncodedData& encoded,
                               size_t start, size_t end) override {
        const auto& raw = encoded.data();
        const size_t N  = parseN(raw);
        end = std::min(end, N);
        if (start >= end) return {};

        const uint8_t* planes = raw.data() + 9;
        using U = std::make_unsigned_t<T>;
        const size_t len = end - start;
        std::vector<T> result(len);
        for (size_t ri = 0; ri < len; ++ri) {
            const size_t index = start + ri;
            U val = 0;
            for (uint32_t b = 0; b < kBitsPerElem; ++b) {
                const size_t  bitPos  = b * N + index;
                const size_t  bytePos = bitPos / 8;
                const uint32_t bitOff = static_cast<uint32_t>(bitPos % 8);
                val |= static_cast<U>((planes[bytePos] >> bitOff) & 1u) << b;
            }
            result[ri] = static_cast<T>(val);
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Codec interface
    // -----------------------------------------------------------------------

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::ReorderingEncoding;
    }

    std::string name() const override { return "BitShuffle"; }

    encodings::EncodingProperties properties() const override {
        return encodings::EncodingProperties(encodings::EncodingProperty::Lossless)
             | encodings::EncodingProperty::RandomAccess
             | encodings::EncodingProperty::Composable
             | encodings::EncodingProperty::Vectorizable
             | encodings::EncodingProperty::LowMemoryOverhead;
    }

private:
    // -----------------------------------------------------------------------
    // Bit-matrix transpose: N × kBitsPerElem → kBitsPerElem × N
    // Element i, bit b → plane b, position i.
    // -----------------------------------------------------------------------

    static std::vector<uint8_t> bitshuffle(std::span<const T> data) {
        const size_t N = data.size();
        // Output: kBitsPerElem planes, each of ceil(N/8) bytes
        const size_t planeBytes = (N + 7) / 8;
        std::vector<uint8_t> out(kBitsPerElem * planeBytes, 0);

        using U = std::make_unsigned_t<T>;
        for (size_t i = 0; i < N; ++i) {
            const U v = static_cast<U>(data[i]);
            for (uint32_t b = 0; b < kBitsPerElem; ++b) {
                const size_t  bitPos  = b * N + i;
                const size_t  bytePos = bitPos / 8;
                const uint32_t bitOff = static_cast<uint32_t>(bitPos % 8);
                out[bytePos] |= static_cast<uint8_t>(((v >> b) & 1u) << bitOff);
            }
        }
        return out;
    }

    static std::vector<T> bitdeshuffle(std::span<const uint8_t> planes, size_t N) {
        using U = std::make_unsigned_t<T>;
        std::vector<T> out(N);
        for (size_t i = 0; i < N; ++i) {
            U val = 0;
            for (uint32_t b = 0; b < kBitsPerElem; ++b) {
                const size_t  bitPos  = b * N + i;
                const size_t  bytePos = bitPos / 8;
                const uint32_t bitOff = static_cast<uint32_t>(bitPos % 8);
                val |= static_cast<U>((planes[bytePos] >> bitOff) & 1u) << b;
            }
            out[i] = static_cast<T>(val);
        }
        return out;
    }

    static size_t parseN(const std::vector<uint8_t>& raw) {
        size_t N = 0;
        for (int b = 0; b < 8; ++b) N |= static_cast<size_t>(raw[b]) << (8 * b);
        return N;
    }

    std::shared_ptr<encodings::Codec<T, uint8_t>> inner_;
};

// Factory (inner codec optional — nullptr means no additional compression)
template <ReorderableType T>
std::shared_ptr<BitShuffleCodec<T>> makeBitShuffleCodec(
    std::shared_ptr<encodings::Codec<T, uint8_t>> inner = nullptr) {
    return std::make_shared<BitShuffleCodec<T>>(std::move(inner));
}

} // namespace encodings::reorderers
