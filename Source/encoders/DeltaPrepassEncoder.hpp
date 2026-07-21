#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "core/DataType.hpp"
#include "encoders/FOREncoder.hpp"          // detail_for::writeLE/readLE
#include "encoders/RawBitPackedEncoder.hpp" // default leaf codec
#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

// =============================================================================
//  DeltaPrepassEncoder<TIn>
//
//  First-order delta prepass with a pluggable leaf Codec for the delta stream,
//  mirroring CascadingFOREncoder's leaf-encoder pattern so Delta and FOR can be
//  compared as two prepass strategies feeding identical downstream compressors.
//
//  Internal arithmetic is always done in int64_t regardless of TIn, matching
//  CascadingFOREncoder's convention.
//
//  Wire format
//  -----------
//    [0 .. 7]  N        (uint64_t)
//    [8 .. 15] anchor   (int64_t)               -- only present if N > 0
//    [16 .. ]  leafEncoder->encode(deltas)      -- deltas.size() == N-1
//
//  where anchor = data[0] and delta[i] = data[i] - data[i-1] for i in [1, N).
//  No delta[0] is synthesized -- the leaf stream has exactly N-1 elements.
//
//  Random access
//  -------------
//  Reconstruction is an inherently sequential prefix sum, and (as with
//  CascadingFOREncoder) this is a research/comparison tool whose primary
//  consumers are encode()-driven ratio comparisons. decodeAt/decodeRange
//  cache the full decodeAll() result keyed by buffer identity (pointer+size)
//  rather than implementing a true O(index) skip scheme. properties()
//  deliberately does NOT claim RandomAccess -- unlike DeltaVarIntEncoder,
//  which claims RandomAccess despite the same sequential prefix-sum decode;
//  that inconsistency is not repeated here.
// =============================================================================

struct DeltaPrepassConfig {
    std::shared_ptr<Codec<int64_t, uint8_t>> leafEncoder =
        std::make_shared<RawBitPackedEncoder<int64_t>>();
};

template <typename TIn>
class DeltaPrepassEncoder : public Codec<TIn, uint8_t> {
    static_assert(std::is_integral_v<TIn> && std::is_signed_v<TIn>,
                  "DeltaPrepassEncoder: TIn must be a signed integral type");

    static constexpr size_t kHeaderSizeNonEmpty = 16; // N (8) + anchor (8)
    static constexpr size_t kHeaderSizeEmpty    = 8;  // N (8) only

public:
    explicit DeltaPrepassEncoder(DeltaPrepassConfig cfg = {}) : cfg_(std::move(cfg)) {
        if (!cfg_.leafEncoder)
            throw std::invalid_argument("DeltaPrepassEncoder: leafEncoder must not be null");
    }

    // ---- Encoder::encode ---------------------------------------------------

    EncodedBuffer<uint8_t> encode(std::span<const TIn> data) override {
        const size_t N = data.size();
        if (N == 0) return makeEmpty();

        std::vector<int64_t> wide(N);
        for (size_t i = 0; i < N; ++i) wide[i] = static_cast<int64_t>(data[i]);

        const int64_t anchor = wide[0];
        std::vector<int64_t> deltas(N - 1);
        for (size_t i = 1; i < N; ++i) deltas[i - 1] = wide[i] - wide[i - 1];

        std::vector<uint8_t> leafBytes = cfg_.leafEncoder->encode(std::span<const int64_t>(deltas)).data();

        std::vector<uint8_t> out;
        out.reserve(kHeaderSizeNonEmpty + leafBytes.size());
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(N));
        detail_for::writeLE<int64_t>(out, anchor);
        out.insert(out.end(), leafBytes.begin(), leafBytes.end());

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::core::typeToDataType<TIn>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(TIn);
        meta.supportsRandomAccess = false; // see class-level doc: decodeAt caches full decodeAll()

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // ---- Decoder::decodeAll -------------------------------------------------

    std::vector<TIn> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        if (encoded.data().size() < kHeaderSizeEmpty) return {};
        const uint8_t* p = encoded.data().data();
        const uint64_t N = detail_for::readLE<uint64_t>(p); p += 8;
        if (N == 0) return {};

        if (encoded.data().size() < kHeaderSizeNonEmpty)
            throw std::runtime_error("DeltaPrepassEncoder::decodeAll: truncated header");
        const int64_t anchor = detail_for::readLE<int64_t>(p); p += 8;

        std::span<const uint8_t> leafSpan(p, encoded.data().size() - kHeaderSizeNonEmpty);
        std::vector<uint8_t> leafCopy(leafSpan.begin(), leafSpan.end());
        EncodedBuffer<uint8_t> leafBuf(std::move(leafCopy), encodings::EncodingMetadata{});
        std::vector<int64_t> deltas = cfg_.leafEncoder->decodeAll(leafBuf);

        if (deltas.size() != N - 1)
            throw std::runtime_error("DeltaPrepassEncoder::decodeAll: leaf decode size mismatch");

        std::vector<TIn> out(static_cast<size_t>(N));
        int64_t running = anchor;
        out[0] = static_cast<TIn>(running);
        for (size_t i = 1; i < N; ++i) {
            running += deltas[i - 1];
            out[i] = static_cast<TIn>(running);
        }
        return out;
    }

    // ---- Decoder::decodeAt / decodeRange ------------------------------------
    // See class-level doc: cache the full decode, keyed by buffer identity.

    std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        refreshCache(encoded);
        if (index >= cachedDecoded_.size()) return std::nullopt;
        return cachedDecoded_[index];
    }

    std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        refreshCache(encoded);
        end = std::min(end, cachedDecoded_.size());
        if (start >= end) return {};
        return std::vector<TIn>(cachedDecoded_.begin() + static_cast<ptrdiff_t>(start),
                                 cachedDecoded_.begin() + static_cast<ptrdiff_t>(end));
    }

    // ---- Identity ------------------------------------------------------------

    EncodingType encodingType() const override { return EncodingType::DeltaPrepassEncoding; }

    std::string name() const override {
        return "DeltaPrepass<" + std::string(typeid(TIn).name()) +
               ",leaf=" + cfg_.leafEncoder->name() + ">";
    }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        return EncodingProperties{}
            .add(EP::Lossless)
            .add(EP::PreservesOrder)
            .add(EP::DeltaBased)
            .add(EP::Composable)
            .add(EP::OptimizedForSorted);
    }

private:
    void refreshCache(const EncodedBuffer<uint8_t>& encoded) {
        const uint8_t* ptr = encoded.data().data();
        const size_t sz = encoded.data().size();
        if (cachedPtr_ != ptr || cachedSize_ != sz) {
            cachedDecoded_ = decodeAll(encoded);
            cachedPtr_ = ptr;
            cachedSize_ = sz;
        }
    }

    static EncodedBuffer<uint8_t> makeEmpty() {
        std::vector<uint8_t> out(kHeaderSizeEmpty, 0);
        encodings::EncodingMetadata meta;
        meta.encodingName         = "DeltaPrepass(empty)";
        meta.elementCount         = 0;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = 0;
        meta.supportsRandomAccess = false;
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    DeltaPrepassConfig cfg_;

    const uint8_t* cachedPtr_{nullptr};
    size_t cachedSize_{0};
    std::vector<TIn> cachedDecoded_;
};

} // namespace encodings::encoders
