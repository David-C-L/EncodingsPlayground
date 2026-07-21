#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <string>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

    using core::DataType;
    using core::typeToDataType;
    using core::IntegralType;

// =============================================================================
//  VarInt codec — LEB128 with optional ZigZag pre-processing
// =============================================================================
//
//  Encoding
//  --------
//  Each integer is encoded as a sequence of 7-bit groups in little-endian order
//  (least-significant group first).  The most-significant bit of each byte is
//  set to 1 for every byte except the last, which acts as a continuation flag:
//
//    value 300 = 0b1_0010_1100
//      byte 0 : 1_010_1100  (0xAC)  — continuation bit set, low 7 bits
//      byte 1 : 0_000_0010  (0x02)  — final byte, high bits
//
//  This scheme (LEB128) gives:
//    • 1 byte  for values 0–127
//    • 2 bytes for values 128–16383
//    • …up to 10 bytes for full 64-bit values
//
//  ZigZag pre-processing  (for signed types, on by default)
//  ---------------------------------------------------------
//  Maps signed integers bijectively onto unsigned integers so that small
//  negative numbers also encode in few bytes:
//
//    encodeZigZag(n) = (n << 1) ^ (n >> (bits-1))
//    decodeZigZag(u) = (u >> 1) ^ -(u & 1)
//
//    0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, …
//
//  Without ZigZag a signed -1 encodes as the full-width unsigned max (10 bytes
//  for int64_t), making the codec counterproductive for negative values.
//
//  Random access
//  -------------
//  The VarInt stream is NOT randomly addressable without a seek structure.
//  This implementation uses a linear skip-scan from the stream start to find
//  element[i] — O(i) decode work.
//
//  Alternatives and their trade-offs:
//    (a) Skip-scan from start [CURRENT]
//          Overhead : none (no index stored)
//          decodeAt : O(i) — scans and discards i elements
//          decodeRange(s,e) : O(s + (e-s)) — scan to s then decode
//
//    (b) Full offset index
//          Overhead : ~8 bytes per element (one uint64_t file-offset per value)
//          decodeAt : O(1) — direct seek to byte, then decode one value
//          Best when random access is the dominant use-case.
//
//    (c) Chunked index every K elements
//          Overhead : 8 * (n/K) bytes
//          decodeAt : O(K) — seek to nearest chunk boundary, skip up to K values
//          Good compromise; K=128 or K=256 is typical.
//
//  Wire format
//  -----------
//    [ num_elements  : 8 bytes  (uint64_t, little-endian)         ]
//    [ flags         : 1 byte   (bit 0 = zigzag_enabled)          ]
//    [ varint stream : variable (1–10 bytes per element)          ]
//
// =============================================================================

namespace detail {

    // ---- unsigned → LEB128 ------------------------------------------------
    //  Writes into a raw pre-allocated buffer at *p and advances p.
    //  Caller must ensure (sizeof(U)*8/7 + 1) bytes are available.

    template<typename U>
    requires std::unsigned_integral<U>
    inline void writeLEB128(uint8_t*& p, U value) {
        while (value >= 0x80u) {
            *p++ = static_cast<uint8_t>(0x80u | (value & 0x7Fu));
            value >>= 7;
        }
        *p++ = static_cast<uint8_t>(value);
    }

    // ---- LEB128 → unsigned ------------------------------------------------
    //  Returns the decoded value; advances ptr past the consumed bytes.
    //  Does NOT bounds-check — callers must ensure enough bytes remain.

    template<typename U>
    requires std::unsigned_integral<U>
    inline U readLEB128(const uint8_t*& ptr) {
        U result = 0;
        unsigned shift = 0;
        while (true) {
            const uint8_t byte = *ptr++;
            result |= static_cast<U>(byte & 0x7Fu) << shift;
            if (!(byte & 0x80u)) break;
            shift += 7;
        }
        return result;
    }

    // ---- ZigZag encode/decode  --------------------------------------------
    //  Operates in the unsigned domain matching the bit-width of T.

    template<typename T>
    requires std::signed_integral<T>
    inline auto zigzagEncode(T val) -> std::make_unsigned_t<T> {
        using U = std::make_unsigned_t<T>;
        // Arithmetic right shift fills with sign bits; avoids UB on signed shift.
        return (static_cast<U>(val) << 1) ^ static_cast<U>(val >> (sizeof(T) * 8 - 1));
    }

    template<typename U>
    requires std::unsigned_integral<U>
    inline auto zigzagDecode(U val) -> std::make_signed_t<U> {
        using S = std::make_signed_t<U>;
        return static_cast<S>((val >> 1) ^ (~(val & 1u) + 1u));
    }

    // ---- Map T → its unsigned counterpart ---------------------------------

    template<typename T>
    requires IntegralType<T>
    using UnsignedOf = std::make_unsigned_t<T>;

} // namespace detail

// =============================================================================

/**
 * @brief Configuration for VarIntEncoder.
 *
 * @param useZigZag  Apply ZigZag pre-processing before LEB128 encoding.
 *                   Defaults to true for signed types (so that small negative
 *                   numbers still encode in few bytes) and false for unsigned
 *                   types (where ZigZag would waste a bit for every value).
 */
template<typename T>
requires IntegralType<T>
struct VarIntConfig {
    /// If true, signed values are ZigZag-mapped before LEB128; ignored for
    /// unsigned types (unsigned values never need ZigZag).
    bool useZigZag = std::is_signed_v<T>;
};

// =============================================================================

/**
 * @brief LEB128 variable-length integer codec.
 *
 * Encodes each element as a sequence of 7-bit continuation bytes so that small
 * values consume fewer bytes than their fixed-width representation.  For signed
 * types ZigZag mapping is applied first (configurable) to ensure that small
 * negative values also compress well.
 *
 * @tparam T  Any integral type (int8_t … int64_t, uint8_t … uint64_t).
 */
template<typename T>
requires IntegralType<T>
class VarIntEncoder : public Codec<T> {
public:
    using UnsignedT = detail::UnsignedOf<T>;

    explicit VarIntEncoder(VarIntConfig<T> cfg = {}) : cfg_(cfg) {}

    // =========================================================================
    //  encode
    // =========================================================================

    EncodedData encode(std::span<const T> data) override {
        const uint64_t numElements = data.size();
        const uint8_t  flags       = cfg_.useZigZag ? 0x01u : 0x00u;

        // Worst-case: ceil(bits/7) bytes per element (5 for int32, 10 for int64).
        // We write directly into the final buffer with a raw pointer — no separate
        // stream vector, no copy, no push_back overhead.
        constexpr size_t maxBytesPerElement = (sizeof(T) * 8 + 6) / 7;
        constexpr size_t headerSize = sizeof(uint64_t) + sizeof(uint8_t);
        const size_t worstCaseSize = headerSize + data.size() * maxBytesPerElement;

        EncodedData result;
        result.data().resize(worstCaseSize);
        uint8_t* base = result.data().data();
        uint8_t* p    = base;

        // Write header
        std::memcpy(p, &numElements, sizeof(uint64_t)); p += sizeof(uint64_t);
        *p++ = flags;

        // Hoist the runtime useZigZag branch outside the loop by dispatching
        // to one of two concrete lambda bodies.  The compiler can then inline
        // and optimise each path independently.
        if constexpr (std::is_signed_v<T>) {
            if (cfg_.useZigZag) {
                for (const T val : data)
                    detail::writeLEB128(p, detail::zigzagEncode(val));
            } else {
                for (const T val : data)
                    detail::writeLEB128(p, static_cast<UnsignedT>(val));
            }
        } else {
            for (const T val : data)
                detail::writeLEB128(p, static_cast<UnsignedT>(val));
        }

        // Trim the buffer to the actual number of bytes written
        const size_t actualSize = static_cast<size_t>(p - base);
        result.data().resize(actualSize);

        result.metadata().encodingName        = name();
        result.metadata().dataType            = this->dataType();
        result.metadata().elementCount        = data.size();
        result.metadata().compressedSize      = actualSize;
        result.metadata().uncompressedSize    = data.size() * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["zigzag"] = cfg_.useZigZag ? "true" : "false";

        return result;
    }

    // =========================================================================
    //  decodeAll
    // =========================================================================

    std::vector<T> decodeAll(const EncodedData& encoded) override {
        const auto [numElements, streamPtr] = readHeader(encoded);
        if (!streamPtr) return {};

        std::vector<T> out;
        out.reserve(numElements);

        const uint8_t* p = streamPtr;
        for (uint64_t i = 0; i < numElements; ++i) {
            out.push_back(decodeOne(p));
        }
        return out;
    }

    // =========================================================================
    //  decodeAt  — O(index) skip-scan from start of stream
    // =========================================================================
    //
    //  See wire-format comment at the top for alternatives (full index → O(1),
    //  chunked index → O(K)) and their storage trade-offs.

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const auto [numElements, streamPtr] = readHeader(encoded);
        if (!streamPtr || index >= numElements) return std::nullopt;

        const uint8_t* p = streamPtr;

        // Skip elements [0, index)
        for (size_t i = 0; i < index; ++i) skipOne(p);

        return decodeOne(p);
    }

    // =========================================================================
    //  decodeRange  — O(start + (end-start)) skip-scan from start of stream
    // =========================================================================

    std::vector<T> decodeRange(const EncodedData& encoded,
                               size_t start, size_t end) override {
        const auto [numElements, streamPtr] = readHeader(encoded);
        if (!streamPtr || start >= numElements) return {};

        end = std::min(end, static_cast<size_t>(numElements));
        if (start >= end) return {};

        const uint8_t* p = streamPtr;

        // Skip elements [0, start)
        for (size_t i = 0; i < start; ++i) skipOne(p);

        std::vector<T> out;
        out.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            out.push_back(decodeOne(p));
        }
        return out;
    }

    // =========================================================================
    //  decodeGatherInto — single forward pass over the whole RowRangeList
    // =========================================================================
    //
    //  The default fallback (independent decodeRangeInto() per range) restarts
    //  the skip-scan from byte 0 for every range. Since ranges arrive here as
    //  one ascending, non-overlapping list, a single forward pass (local byte
    //  cursor + local logical-index cursor) visits each byte at most once.

    void decodeGatherInto(const EncodedData& encoded,
                         const RowRangeList& ranges,
                         T* dst, size_t n) override {
        if (ranges.empty()) {
            if (n != 0) throw std::runtime_error("VarIntEncoder::decodeGatherInto: decoded size mismatch");
            return;
        }
        const auto [numElements, streamPtr] = readHeader(encoded);
        if (!streamPtr) {
            if (n != 0) throw std::runtime_error("VarIntEncoder::decodeGatherInto: empty stream, n!=0");
            return;
        }

        const uint8_t* p = streamPtr;
        size_t lastIdx = 0;
        size_t off = 0;
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            const size_t begin = r.begin;
            const size_t end   = std::min(r.end, static_cast<size_t>(numElements));
            if (begin >= end) continue;

            while (lastIdx < begin) { skipOne(p); ++lastIdx; }
            while (lastIdx < end)   { dst[off++] = decodeOne(p); ++lastIdx; }
        }
        if (off != n) throw std::runtime_error("VarIntEncoder::decodeGatherInto: decoded size mismatch");
    }

    // =========================================================================
    //  Metadata
    // =========================================================================

    EncodingType encodingType() const override {
        return EncodingType::VarIntEncoding;
    }

    std::string name() const override {
        if constexpr (std::is_signed_v<T>) {
            return cfg_.useZigZag ? "VarInt(ZigZag)" : "VarInt(Signed)";
        } else {
            return "VarInt(Unsigned)";
        }
    }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::SequentialOnly)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::VariableSize
            | EncodingProperty::StreamingFriendly
            | EncodingProperty::ImmutableOnly
            | EncodingProperty::Composable
            | EncodingProperty::RandomAccess
             /* We support random access via skip-scan, but it's O(n) — consider removing this if we want to be strict about it. */
            | EncodingProperty::FastSkip;
    }

    /// Estimate: assumes values are small enough to fit in ceil(bits/7)/2 bytes
    /// on average — a very rough heuristic. Exact size depends on the data.
    size_t estimateEncodedSize(size_t elementCount) const override {
        constexpr size_t headerSize = sizeof(uint64_t) + sizeof(uint8_t);
        // Optimistic average: roughly 2 bytes per element for 32-bit types,
        // 4 bytes for 64-bit types.
        constexpr size_t avgBytesPerElement = sizeof(T) <= 4 ? 2 : 4;
        return headerSize + elementCount * avgBytesPerElement;
    }

private:
    VarIntConfig<T> cfg_;

    // -------------------------------------------------------------------------
    //  Header helpers
    // -------------------------------------------------------------------------

    struct HeaderResult {
        uint64_t       numElements;
        const uint8_t* streamPtr;   // nullptr on failure
    };

    HeaderResult readHeader(const EncodedData& encoded) const {
        constexpr size_t headerSize = sizeof(uint64_t) + sizeof(uint8_t);
        if (encoded.size() < headerSize) return {0, nullptr};

        const uint8_t* p = encoded.data().data();

        uint64_t numElements;
        std::memcpy(&numElements, p, sizeof(uint64_t));
        p += sizeof(uint64_t);

        // flags byte — zigzag bit validated against config (informational only;
        // we trust the config passed at construction time)
        // const uint8_t flags = *p;
        p += sizeof(uint8_t);

        return {numElements, p};
    }

    // -------------------------------------------------------------------------
    //  Single-element decode; advances ptr
    // -------------------------------------------------------------------------

    inline T decodeOne(const uint8_t*& p) const {
        const UnsignedT u = detail::readLEB128<UnsignedT>(p);
        if constexpr (std::is_signed_v<T>) {
            return cfg_.useZigZag
                ? detail::zigzagDecode(u)
                : static_cast<T>(u);
        } else {
            return static_cast<T>(u);
        }
    }

    // -------------------------------------------------------------------------
    //  Skip one encoded element without decoding; advances ptr
    // -------------------------------------------------------------------------

    static inline void skipOne(const uint8_t*& p) {
        while (*p++ & 0x80u) { /* continuation bit sereadHeadert — keep scanning */ }
    }
};

} // namespace encodings::encoders
