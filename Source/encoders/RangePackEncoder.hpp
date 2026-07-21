#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "encoders/ISectionCodecIntegral.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

// =============================================================================
//  RangePackSectionCodec<TIn>
//
//  Global Frame-of-Reference + narrowest-width repack, composed with any inner
//  section codec factory. Mirrors OpenZL's `range_pack` transform: a single
//  min/max scan over the whole input, subtract the minimum, then hand the
//  shifted residuals to an inner codec constructed at the NARROWEST bit width
//  that can hold the shifted range — not the section's original nominal
//  width. This is what lets composing RangePack with e.g.
//  FrequencyPartitionEncoder or BlockFrequencyPartitionEncoder produce a
//  genuinely smaller encoding: those encoders store dictionary/fallback
//  values at a fixed sizeof(T) width, so narrowing T is what actually saves
//  bytes — a same-width value shift alone would not (dict/fallback entries
//  are stored at fixed width regardless of the magnitude of the values they
//  hold).
//
//  Composable with ANY inner section-codec factory of the shape
//  `(uint8_t bits) -> shared_ptr<ISectionCodecIntegral<TIn>>` — every
//  `makeXSection<TIn>` factory in SubIntEncodingUtils.hpp's detail_trisplit
//  namespace already fits this shape, so RangePack can be composed with any
//  of them via makeRangePackSection() below, not just FrequencyPartitionEncoder
//  / BlockFrequencyPartitionEncoder.
//
//  Random access — preserved when the inner codec supports it
//  -------------------------------------------------------------
//  decodeAt/decodeRange reconstruct the inner codec (deterministically, from
//  the stored narrowedBits — decoding never needs the original data) and
//  delegate directly to its decodeAt/decodeRange when it reports
//  EncodingProperty::RandomAccess, falling back to a full decodeAll() + index
//  otherwise (mirroring FOREncoder::decodeAt, FOREncoder.hpp:253-275).
//
//  Wire format (uint8_t buffer)
//  ----------------------------
//  [ 0 ..  7]                 N            (uint64_t) — element count
//  [ 8 .. 8+sizeof(TIn)-1]    minVal       (TIn)      — global minimum subtracted from all values
//  [ next 1 byte]             narrowedBits (uint8_t)  — bit width the inner codec was constructed at
//  [ next 8 bytes]            innerBytes   (uint64_t) — byte length of the inner-encoded payload
//  [ remaining ]              inner-encoded payload
// =============================================================================

template <typename TIn = uint64_t>
    requires (std::is_same_v<TIn, uint64_t> || std::is_same_v<TIn, uint32_t>)
class RangePackSectionCodec final : public ISectionCodecIntegral<TIn> {
public:
    using InnerFactory = std::function<std::shared_ptr<ISectionCodecIntegral<TIn>>(uint8_t)>;

    RangePackSectionCodec(InnerFactory innerFactory, encodings::EncodingType selfType)
        : innerFactory_(std::move(innerFactory)), selfType_(selfType) {
        if (!innerFactory_) {
            throw std::invalid_argument("RangePackSectionCodec: innerFactory must not be null");
        }
    }

    EncodedBuffer<uint8_t> encode(std::span<const TIn> data) override {
        const size_t N = data.size();
        if (N == 0) return makeEmpty();

        const TIn minVal = *std::min_element(data.begin(), data.end());
        const TIn maxVal = *std::max_element(data.begin(), data.end());
        const TIn range  = static_cast<TIn>(maxVal - minVal);
        const uint8_t narrowedBits = (range == 0)
            ? uint8_t{1}
            : static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(range)));

        std::vector<TIn> residuals(N);
        for (size_t i = 0; i < N; ++i)
            residuals[i] = static_cast<TIn>(data[i] - minVal);

        auto innerCodec = innerFactory_(narrowedBits);
        auto innerEncoded = innerCodec->encode(std::span<const TIn>(residuals.data(), residuals.size()));
        const auto& innerBytes = innerEncoded.data();

        std::vector<uint8_t> out;
        out.reserve(headerSize() + innerBytes.size());
        appendLE<uint64_t>(out, static_cast<uint64_t>(N));
        appendLE<TIn>(out, minVal);
        out.push_back(narrowedBits);
        appendLE<uint64_t>(out, static_cast<uint64_t>(innerBytes.size()));
        out.insert(out.end(), innerBytes.begin(), innerBytes.end());

        EncodingMetadata meta;
        meta.encodingName         = name();
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(TIn);
        meta.supportsRandomAccess = innerCodec->properties().has(EncodingProperty::RandomAccess);

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    std::vector<TIn> decodeAll(const EncodedBuffer<uint8_t>& enc) override {
        const ParsedHeader h = parseHeader(enc);
        if (h.N == 0) return {};
        auto innerCodec = innerFactory_(h.narrowedBits);
        auto residuals = innerCodec->decodeAll(innerView(enc, h));
        if (residuals.size() != h.N) {
            throw std::runtime_error("RangePackSectionCodec::decodeAll: inner decode size mismatch");
        }
        std::vector<TIn> result(h.N);
        for (size_t i = 0; i < h.N; ++i)
            result[i] = static_cast<TIn>(residuals[i] + h.minVal);
        return result;
    }

    std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t idx) override {
        const ParsedHeader h = parseHeader(enc);
        if (idx >= h.N) return std::nullopt;

        auto innerCodec = innerFactory_(h.narrowedBits);
        const auto& view = innerView(enc, h);

        if (innerCodec->properties().has(EncodingProperty::RandomAccess)) {
            auto v = innerCodec->decodeAt(view, idx);
            if (!v) {
                throw std::runtime_error("RangePackSectionCodec::decodeAt: inner codec returned no value for valid index");
            }
            return static_cast<TIn>(*v + h.minVal);
        }

        auto all = innerCodec->decodeAll(view);
        if (all.size() != h.N) {
            throw std::runtime_error("RangePackSectionCodec::decodeAt: inner decode size mismatch");
        }
        return static_cast<TIn>(all[idx] + h.minVal);
    }

    std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) override {
        const ParsedHeader h = parseHeader(enc);
        end = std::min(end, h.N);
        if (start >= end) return {};

        auto innerCodec = innerFactory_(h.narrowedBits);
        const auto& view = innerView(enc, h);
        const size_t count = end - start;
        std::vector<TIn> result;
        result.reserve(count);

        if (innerCodec->properties().has(EncodingProperty::RandomAccess)) {
            auto slice = innerCodec->decodeRange(view, start, end);
            if (slice.size() != count) {
                throw std::runtime_error("RangePackSectionCodec::decodeRange: inner range decode size mismatch");
            }
            for (size_t i = 0; i < count; ++i)
                result.push_back(static_cast<TIn>(slice[i] + h.minVal));
            return result;
        }

        auto all = innerCodec->decodeAll(view);
        if (all.size() != h.N) {
            throw std::runtime_error("RangePackSectionCodec::decodeRange: inner decode size mismatch");
        }
        for (size_t i = start; i < end; ++i)
            result.push_back(static_cast<TIn>(all[i] + h.minVal));
        return result;
    }

    void decodeAllInto(const EncodedBuffer<uint8_t>& enc, TIn* dst, size_t n) override {
        auto all = decodeAll(enc);
        if (all.size() != n) {
            throw std::runtime_error("RangePackSectionCodec::decodeAllInto: size mismatch");
        }
        std::copy(all.begin(), all.end(), dst);
    }

    void decodeRangeInto(const EncodedBuffer<uint8_t>& enc,
                          size_t start, size_t end, TIn* dst, size_t n) override {
        auto slice = decodeRange(enc, start, end);
        if (slice.size() != n) {
            throw std::runtime_error("RangePackSectionCodec::decodeRangeInto: size mismatch");
        }
        std::copy(slice.begin(), slice.end(), dst);
    }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        auto props = EncodingProperties{}
            .add(EP::Lossless)
            .add(EP::PreservesOrder)
            .add(EP::Composable);
        // Probe the inner factory at a representative width to report whether
        // random access is actually preserved; the real decodeAt/decodeRange
        // above always re-check the specific instantiated inner codec at call
        // time regardless of what this generic self-description claims, so
        // this is informational only, not a correctness dependency.
        if (innerFactory_(8)->properties().has(EP::RandomAccess)) {
            props.add(EP::RandomAccess);
        }
        return props;
    }

    std::string name() const override {
        return "RangePack+" + std::string(encodings::encodingTypeToString(selfType_));
    }

private:
    struct ParsedHeader {
        size_t  N;
        TIn     minVal;
        uint8_t narrowedBits;
        size_t  innerOffset;
        size_t  innerBytes;
    };

    static size_t headerSize() { return 8 + sizeof(TIn) + 1 + 8; }

    template <typename T>
    static void appendLE(std::vector<uint8_t>& buf, T value) {
        const size_t off = buf.size();
        buf.resize(off + sizeof(T));
        std::memcpy(buf.data() + off, &value, sizeof(T));
    }

    template <typename T>
    static T readLE(const uint8_t* p) {
        T v;
        std::memcpy(&v, p, sizeof(T));
        return v;
    }

    static ParsedHeader parseHeader(const EncodedBuffer<uint8_t>& enc) {
        const uint8_t* p = enc.data().data();
        const uint64_t N            = readLE<uint64_t>(p); p += 8;
        const TIn      minVal       = readLE<TIn>(p);       p += sizeof(TIn);
        const uint8_t  narrowedBits = *p;                    p += 1;
        const uint64_t innerBytes   = readLE<uint64_t>(p);   p += 8;
        const size_t innerOffset = static_cast<size_t>(p - enc.data().data());
        return ParsedHeader{
            static_cast<size_t>(N), minVal, narrowedBits, innerOffset, static_cast<size_t>(innerBytes)
        };
    }

    // Builds a lightweight EncodedBuffer view over just the inner payload
    // bytes, so the inner codec's own decode functions can operate on it
    // directly. Cached by (basePtr,totalSize) identity to avoid re-copying on
    // repeated decodeAt/decodeRange calls, mirroring FOREncoder's
    // getCachedResidualView (Source/encoders/FOREncoder.hpp:380-401).
    const EncodedBuffer<uint8_t>& innerView(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h) const {
        const uint8_t* basePtr = enc.data().data();
        const size_t totalSize = enc.data().size();

        if (cachedOuterPtr_ != basePtr || cachedOuterSize_ != totalSize) {
            std::vector<uint8_t> payload(h.innerBytes);
            if (h.innerBytes > 0) {
                std::memcpy(payload.data(), basePtr + h.innerOffset, h.innerBytes);
            }
            EncodingMetadata meta;
            meta.elementCount     = h.N;
            meta.compressedSize   = h.innerBytes;
            meta.uncompressedSize = h.N * sizeof(TIn);
            cachedInnerView_ = encodings::EncodedData(std::move(payload), std::move(meta));
            cachedOuterPtr_  = basePtr;
            cachedOuterSize_ = totalSize;
        }
        return cachedInnerView_;
    }

    static EncodedBuffer<uint8_t> makeEmpty() {
        std::vector<uint8_t> out(headerSize(), 0u);
        EncodingMetadata meta;
        meta.elementCount         = 0;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = 0;
        meta.supportsRandomAccess = true;
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    InnerFactory innerFactory_;
    encodings::EncodingType selfType_;

    mutable const uint8_t* cachedOuterPtr_{nullptr};
    mutable size_t cachedOuterSize_{0};
    mutable EncodedBuffer<uint8_t> cachedInnerView_;
};

// ---------------------------------------------------------------------------
// Generic factory — compose RangePack with any inner section-codec factory.
// ---------------------------------------------------------------------------

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeRangePackSection(
    typename RangePackSectionCodec<SectionCodecTIn>::InnerFactory innerFactory,
    encodings::EncodingType selfType) {
    return std::make_shared<RangePackSectionCodec<SectionCodecTIn>>(std::move(innerFactory), selfType);
}

} // namespace encodings::encoders
