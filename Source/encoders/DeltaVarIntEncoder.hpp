#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <string>
#include <optional>
#include <type_traits>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"
#include "encoders/VarIntEncoder.hpp"   // reuses detail::writeLEB128 / readLEB128 / zigzag*

namespace encodings::encoders {

    using core::IntegralType;

// =============================================================================
//  DeltaVarIntEncoder — Delta pre-pass followed by LEB128 (VarInt) encoding
// =============================================================================
//
//  Motivation
//  ----------
//  Plain VarInt (LEB128) only saves space when the absolute values being
//  encoded are small.  For datasets whose values are large but *locally
//  correlated* (e.g. sorted IDs, time-series, spatial keys), the deltas
//  between consecutive elements are typically far smaller than the values
//  themselves — making VarInt highly effective on the delta stream.
//
//  Algorithm
//  ---------
//  Encode:
//    1. Store data[0] as a fixed-width "anchor" (sizeof(T) bytes).
//    2. For each subsequent element, compute delta_i = data[i] - data[i-1].
//    3. ZigZag-map each delta (signed → unsigned) so that small negative deltas
//       also encode in few bytes.
//    4. LEB128-encode each ZigZag-mapped delta into the output stream.
//
//  Decode:
//    1. Read the anchor.
//    2. For each element, LEB128-decode a ZigZag delta and accumulate.
//
//  ZigZag default
//  --------------
//  ZigZag is always applied to the deltas regardless of whether T is signed or
//  unsigned, because deltas can be negative even when T is unsigned (e.g. an
//  unsorted uint32_t column).  The config flag lets you disable it if you know
//  all deltas are non-negative (e.g. a strictly monotone-increasing column).
//
//  Random access
//  -------------
//  Uses skip-scan from the stream start — O(index) work.
//  See VarIntEncoder.hpp for a discussion of O(1) and O(K) alternatives.
//
//  Wire format
//  -----------
//    [ num_elements  : 8 bytes  (uint64_t)                         ]
//    [ flags         : 1 byte   (bit 0 = zigzag_enabled)           ]
//    [ anchor        : sizeof(T) bytes  (data[0], fixed-width)     ]
//    [ delta stream  : variable (1–10 LEB128 bytes per delta)      ]
//
// =============================================================================

/**
 * @brief Configuration for DeltaVarIntEncoder.
 *
 * @param useZigZag  ZigZag-map each delta before LEB128.  Defaults to true
 *                   because deltas can be negative even for unsigned T.
 *                   Disable only when you know all deltas are ≥ 0.
 */
template<typename T>
requires IntegralType<T>
struct DeltaVarIntConfig {
    bool useZigZag = true;
};

// =============================================================================

/**
 * @brief Fused Delta + LEB128 VarInt codec for integral types.
 *
 * Encodes consecutive differences (deltas) as LEB128 variable-length integers,
 * with optional ZigZag mapping so that small negative deltas also compress
 * well.  Ideal for sorted or slowly-varying integer sequences.
 *
 * @tparam T  Any integral type (int8_t … int64_t, uint8_t … uint64_t).
 */
template<typename T>
requires IntegralType<T>
class DeltaVarIntEncoder : public Codec<T> {
public:
    // The signed counterpart of T — used for the delta type so that subtraction
    // of unsigned values wraps correctly and ZigZag handles the sign.
    using SignedT   = std::make_signed_t<T>;
    using UnsignedT = std::make_unsigned_t<T>;

    explicit DeltaVarIntEncoder(DeltaVarIntConfig<T> cfg = {}) : cfg_(cfg) {}

    // =========================================================================
    //  encode
    // =========================================================================

    EncodedData encode(std::span<const T> data) override {
        if (data.empty())   return makeEmpty();
        if (data.size() == 1) return makeSingle(data[0]);

        const uint64_t numElements = static_cast<uint64_t>(data.size());
        const uint8_t  flags       = cfg_.useZigZag ? 0x01u : 0x00u;

        // Worst-case buffer: header + anchor + (n-1) deltas at max LEB128 width.
        // SignedT is the same width as T, so max LEB128 bytes = ceil(bits/7).
        constexpr size_t maxDeltaBytes = (sizeof(T) * 8 + 6) / 7;
        constexpr size_t headerSize    = sizeof(uint64_t) + sizeof(uint8_t) + sizeof(T);
        const size_t     worstCase     = headerSize + (data.size() - 1) * maxDeltaBytes;

        EncodedData result;
        result.data().resize(worstCase);
        uint8_t* base = result.data().data();
        uint8_t* p    = base;

        // --- header ---
        std::memcpy(p, &numElements, sizeof(uint64_t)); p += sizeof(uint64_t);
        *p++ = flags;

        // --- anchor: first value stored verbatim ---
        std::memcpy(p, &data[0], sizeof(T)); p += sizeof(T);

        // --- delta stream ---
        // Hoist the useZigZag branch outside the loop so each path is a tight,
        // branch-free inner loop the compiler can auto-vectorise.
        if (cfg_.useZigZag) {
            for (size_t i = 1; i < data.size(); ++i) {
                // Compute delta in the signed domain so wrap-around is defined.
                const SignedT delta = static_cast<SignedT>(data[i])
                                    - static_cast<SignedT>(data[i - 1]);
                detail::writeLEB128(p, detail::zigzagEncode(delta));
            }
        } else {
            for (size_t i = 1; i < data.size(); ++i) {
                const SignedT delta = static_cast<SignedT>(data[i])
                                    - static_cast<SignedT>(data[i - 1]);
                detail::writeLEB128(p, static_cast<UnsignedT>(delta));
            }
        }

        const size_t actualSize = static_cast<size_t>(p - base);
        result.data().resize(actualSize);

        result.metadata().encodingName         = name();
        result.metadata().dataType             = this->dataType();
        result.metadata().elementCount         = data.size();
        result.metadata().compressedSize       = actualSize;
        result.metadata().uncompressedSize     = data.size() * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["zigzag"] = cfg_.useZigZag ? "true" : "false";

        return result;
    }

    // =========================================================================
    //  decodeAll
    // =========================================================================

    std::vector<T> decodeAll(const EncodedData& encoded) override {
        const auto [numElements, anchor, streamPtr] = readHeader(encoded);
        if (!streamPtr) return {};

        std::vector<T> out;
        out.reserve(numElements);
        out.push_back(anchor);

        const uint8_t* p = streamPtr;
        T current = anchor;
        for (uint64_t i = 1; i < numElements; ++i) {
            current += decodeDelta(p);
            out.push_back(current);
        }
        return out;
    }

    // =========================================================================
    //  decodeAt  — O(index) skip-scan
    // =========================================================================

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const auto [numElements, anchor, streamPtr] = readHeader(encoded);
        if (!streamPtr || index >= numElements) return std::nullopt;
        if (index == 0) return anchor;

        const uint8_t* p = streamPtr;
        T current = anchor;
        for (size_t i = 1; i <= index; ++i) {
            current += decodeDelta(p);
        }
        return current;
    }

    // =========================================================================
    //  decodeRange  — O(start + (end-start)) skip-scan
    // =========================================================================

    std::vector<T> decodeRange(const EncodedData& encoded,
                               size_t start, size_t end) override {
        const auto [numElements, anchor, streamPtr] = readHeader(encoded);
        if (!streamPtr || start >= numElements) return {};

        end = std::min(end, static_cast<size_t>(numElements));
        if (start >= end) return {};

        const uint8_t* p = streamPtr;
        T current = anchor;

        // Advance to the element just before `start`
        for (size_t i = 1; i <= start && i < numElements; ++i) {
            current += decodeDelta(p);
        }

        std::vector<T> out;
        out.reserve(end - start);

        // If start == 0 the anchor is our first output and we need to collect
        // it before entering the delta loop.
        if (start == 0) {
            out.push_back(anchor);
            // Decode remaining elements in [1, end)
            for (size_t i = 1; i < end; ++i) {
                current += decodeDelta(p);
                out.push_back(current);
            }
        } else {
            out.push_back(current);
            for (size_t i = start + 1; i < end; ++i) {
                current += decodeDelta(p);
                out.push_back(current);
            }
        }
        return out;
    }

    // =========================================================================
    //  decodeGatherInto — single forward pass over the whole RowRangeList
    // =========================================================================
    //
    //  Same rationale as VarIntEncoder: the default fallback restarts the
    //  delta-accumulation scan from the anchor for every range. A single
    //  forward pass carries the running accumulator (`current`), the byte
    //  cursor, and the logical-index cursor as locals across the whole list.

    void decodeGatherInto(const EncodedData& encoded,
                         const RowRangeList& ranges,
                         T* dst, size_t n) override {
        if (ranges.empty()) {
            if (n != 0) throw std::runtime_error("DeltaVarIntEncoder::decodeGatherInto: decoded size mismatch");
            return;
        }
        const auto [numElements, anchor, streamPtr] = readHeader(encoded);
        if (!streamPtr) {
            if (n != 0) throw std::runtime_error("DeltaVarIntEncoder::decodeGatherInto: empty stream, n!=0");
            return;
        }

        const uint8_t* p = streamPtr;
        T current = anchor;
        size_t lastIdx = 0;   // next element index not yet produced/skipped (0 == anchor position)
        size_t off = 0;

        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            const size_t begin = r.begin;
            const size_t end   = std::min(r.end, static_cast<size_t>(numElements));
            if (begin >= end) continue;

            // Skip forward to `begin`, accumulating (discarding) deltas. Index 0
            // is the anchor (no delta consumed); indices >= 1 each consume one.
            while (lastIdx < begin) {
                if (lastIdx > 0) current += decodeDelta(p);
                ++lastIdx;
            }
            // Emit [begin, end).
            while (lastIdx < end) {
                if (lastIdx == 0) {
                    dst[off++] = anchor;
                } else {
                    current += decodeDelta(p);
                    dst[off++] = current;
                }
                ++lastIdx;
            }
        }
        if (off != n) throw std::runtime_error("DeltaVarIntEncoder::decodeGatherInto: decoded size mismatch");
    }

    // =========================================================================
    //  Metadata
    // =========================================================================

    EncodingType encodingType() const override {
        return EncodingType::DeltaVarIntEncoding;
    }

    std::string name() const override {
        return cfg_.useZigZag ? "DeltaVarInt(ZigZag)" : "DeltaVarInt";
    }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::SequentialOnly)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::DeltaBased
            | EncodingProperty::VariableSize
            | EncodingProperty::StreamingFriendly
            | EncodingProperty::OptimizedForSorted
            | EncodingProperty::ImmutableOnly
            | EncodingProperty::Composable
            | EncodingProperty::RandomAccess
             /* We support random access via skip-scan, but it's O(n) — consider removing this if we want to be strict about it. */
            | EncodingProperty::FastSkip;
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        constexpr size_t headerSize = sizeof(uint64_t) + sizeof(uint8_t) + sizeof(T);
        if (elementCount == 0) return headerSize;
        // Conservative estimate: 2 bytes per delta (fits when |delta| < 16384).
        return headerSize + (elementCount - 1) * 2;
    }

private:
    DeltaVarIntConfig<T> cfg_;

    // -------------------------------------------------------------------------
    //  Header layout
    // -------------------------------------------------------------------------

    struct HeaderResult {
        uint64_t       numElements;
        T              anchor;
        const uint8_t* streamPtr;   // nullptr on parse failure
    };

    HeaderResult readHeader(const EncodedData& encoded) const {
        constexpr size_t minSize = sizeof(uint64_t) + sizeof(uint8_t) + sizeof(T);
        if (encoded.size() < minSize) return {0, T{}, nullptr};

        const uint8_t* p = encoded.data().data();

        uint64_t numElements;
        std::memcpy(&numElements, p, sizeof(uint64_t)); p += sizeof(uint64_t);

        // flags — reserved; actual zigzag behaviour comes from cfg_
        p += sizeof(uint8_t);

        T anchor;
        std::memcpy(&anchor, p, sizeof(T)); p += sizeof(T);

        return {numElements, anchor, p};
    }

    // -------------------------------------------------------------------------
    //  Decode one delta from the stream and return the value to add to current
    // -------------------------------------------------------------------------

    inline T decodeDelta(const uint8_t*& p) const {
        const UnsignedT u = detail::readLEB128<UnsignedT>(p);
        if (cfg_.useZigZag) {
            // zigzagDecode returns SignedT; cast to T for accumulation
            return static_cast<T>(detail::zigzagDecode(u));
        } else {
            return static_cast<T>(u);
        }
    }

    // -------------------------------------------------------------------------
    //  Edge-case helpers
    // -------------------------------------------------------------------------

    EncodedData makeEmpty() {
        constexpr size_t headerSize = sizeof(uint64_t) + sizeof(uint8_t) + sizeof(T);
        EncodedData result;
        result.data().resize(headerSize, 0u);   // numElements=0, flags=0, anchor=0

        result.metadata().encodingName         = name();
        result.metadata().dataType             = this->dataType();
        result.metadata().elementCount         = 0;
        result.metadata().compressedSize       = headerSize;
        result.metadata().uncompressedSize     = 0;
        result.metadata().supportsRandomAccess = true;
        return result;
    }

    EncodedData makeSingle(T value) {
        constexpr size_t headerSize = sizeof(uint64_t) + sizeof(uint8_t) + sizeof(T);
        EncodedData result;
        result.data().resize(headerSize);

        uint8_t* p = result.data().data();
        const uint64_t one = 1;
        std::memcpy(p, &one,   sizeof(uint64_t)); p += sizeof(uint64_t);
        *p++ = cfg_.useZigZag ? 0x01u : 0x00u;
        std::memcpy(p, &value, sizeof(T));

        result.metadata().encodingName         = name();
        result.metadata().dataType             = this->dataType();
        result.metadata().elementCount         = 1;
        result.metadata().compressedSize       = headerSize;
        result.metadata().uncompressedSize     = sizeof(T);
        result.metadata().supportsRandomAccess = true;
        return result;
    }
};

} // namespace encodings::encoders
