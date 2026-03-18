#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include <vector>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

// =============================================================================
//  FOREncoder<TIn, TOut, FrameSize>
//
//  Frame-of-Reference (FOR) encoder.
//
//  The input stream is partitioned into non-overlapping frames of exactly
//  FrameSize elements.  Within each frame a reference value r is chosen
//  (see ReferencePolicy).  Every element e is stored as the residual
//
//      residual = e - r
//
//  which must fit in TOut.  If any residual overflows TOut, encode() throws.
//
//  Constraints
//  -----------
//  - TIn  must be a signed integral type (int8_t … int64_t).
//  - TOut must be a signed integral type strictly narrower than TIn.
//  - FrameSize must be a power of two ≥ 1.
//
//  These are all enforced with static_assert so violations are a compile
//  error, not a runtime error.
//
//  Random access — O(1)
//  ---------------------
//  decodeAt(encoded, i):
//    frame_idx  = i >> Log2FrameSize          // compiler: constant shift
//    intra_idx  = i  & (FrameSize - 1)        // compiler: constant mask
//    ref        = refVals [frame_idx]          // one indexed load
//    residual   = residuals[i]                 // one indexed load
//    result     = static_cast<TIn>(ref + residual)
//
//  Because FrameSize is a compile-time constant the compiler replaces every
//  division and modulo with a shift and a mask — no runtime divides at all.
//
//  Wire format (uint8_t buffer)
//  ----------------------------
//  [ 0 ..  7]  N               (uint64_t) — element count
//  [ 8 .. 15]  FrameSize_enc   (uint64_t) — FrameSize baked in for safety
//  [16 .. 23]  num_frames      (uint64_t) — ceil(N / FrameSize)
//  [24 .. 31]  ref_bytes       (uint64_t) — num_frames * sizeof(TIn)
//  [32 .. 39]  res_bytes       (uint64_t) — N          * sizeof(TOut)
//  [40 .. 40 + ref_bytes - 1]  reference values  (TIn[num_frames], little-endian)
//  [40 + ref_bytes .. end]     residual  values  (TOut[N],         little-endian)
//
//  ReferencePolicy
//  ---------------
//  MIN   – minimum of the frame        (residuals ≥ 0, signed TOut still works)
//  FIRST – first element of the frame  (cheaper, good for locally-monotone data)
//  MID   – (min + max) / 2             (minimises residual magnitude, best for
//                                       symmetric distributions)
//
// =============================================================================

enum class FORReferencePolicy : uint8_t {
    MIN   = 0,
    FIRST = 1,
    MID   = 2,
};

// ---------------------------------------------------------------------------
// FORConfig — passed to the constructor
// ---------------------------------------------------------------------------

template <typename TIn, typename TOut>
struct FORConfig {
    FORReferencePolicy policy = FORReferencePolicy::MIN;
    std::shared_ptr<Codec<TOut, uint8_t>> subEncoder{};
};

// ---------------------------------------------------------------------------
// Helper: read / write little-endian integers from/to a byte buffer
// ---------------------------------------------------------------------------

namespace detail_for {

template <typename T>
void writeLE(std::vector<uint8_t>& buf, T value) {
    const size_t off = buf.size();
    buf.resize(off + sizeof(T));
    std::memcpy(buf.data() + off, &value, sizeof(T));
}

template <typename T>
T readLE(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

} // namespace detail_for

// ---------------------------------------------------------------------------
// FOREncoder
// ---------------------------------------------------------------------------

/**
 * @brief Frame-of-Reference encoder.
 *
 * @tparam TIn        Input signed integral type  (e.g. int64_t)
 * @tparam TOut       Output signed integral type (e.g. int16_t) — must be
 *                    strictly narrower than TIn.
 * @tparam FrameSize  Number of elements per frame — must be a power of two.
 */
template <
    typename TIn,
    typename TOut,
    size_t   FrameSize = 128
>
class FOREncoder : public Codec<TIn, uint8_t> {
    // ---- Compile-time constraints ----------------------------------------
    static_assert(std::is_integral_v<TIn>  && std::is_signed_v<TIn>,
                  "FOREncoder: TIn must be a signed integral type");
    static_assert(std::is_integral_v<TOut> && std::is_signed_v<TOut>,
                  "FOREncoder: TOut must be a signed integral type");
    static_assert(sizeof(TOut) <= sizeof(TIn),
                  "FOREncoder: TOut must be no larger than TIn");
    static_assert(FrameSize >= 1,
                  "FOREncoder: FrameSize must be at least 1");
    static_assert((FrameSize & (FrameSize - 1)) == 0,
                  "FOREncoder: FrameSize must be a power of two");

    // Derived compile-time constant — bit count for the shift/mask trick
    static constexpr size_t Log2FrameSize = std::countr_zero(FrameSize);
    static constexpr size_t FrameMask     = FrameSize - 1;

    // Residual range for TOut
    static constexpr TIn kOutMin = static_cast<TIn>(std::numeric_limits<TOut>::min());
    static constexpr TIn kOutMax = static_cast<TIn>(std::numeric_limits<TOut>::max());

public:
    explicit FOREncoder(FORConfig<TIn, TOut> cfg = {})
        : cfg_(cfg)
    {
        if (!cfg_.subEncoder) {
            throw std::invalid_argument("FOREncoder: FORConfig.subEncoder must not be null");
        }
    }

    // ---- Encoder::encode -------------------------------------------------

    EncodedBuffer<uint8_t> encode(std::span<const TIn> data) override {
        const size_t N = data.size();
        if (N == 0) {
            return makeEmpty(N);
        }

        const size_t numFrames = (N + FrameSize - 1) >> Log2FrameSize;

        std::vector<TIn>  refVals(numFrames);
        std::vector<TOut> residuals(N);

        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f << Log2FrameSize;                    // f * FrameSize
            const size_t hi = std::min(lo + FrameSize, N);           // exclusive

            const TIn ref = computeRef(data, lo, hi);
            refVals[f] = ref;

            for (size_t i = lo; i < hi; ++i) {
                const TIn residual = data[i] - ref;
                if (residual < kOutMin || residual > kOutMax) {
                    throw std::overflow_error(
                        "FOREncoder::encode: residual " + std::to_string(residual) +
                        " at index " + std::to_string(i) +
                        " does not fit in TOut (range [" +
                        std::to_string(kOutMin) + ", " +
                        std::to_string(kOutMax) + "])");
                }
                residuals[i] = static_cast<TOut>(residual);
            }
        }

    const auto residualEncoded = cfg_.subEncoder->encode(std::span<const TOut>(residuals.data(), residuals.size()));

    // ---- Serialise ---------------------------------------------------
        const size_t refBytes = numFrames * sizeof(TIn);
    const size_t resBytes = residualEncoded.data().size();

        std::vector<uint8_t> out;
        out.reserve(40 + refBytes + resBytes);

        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(N));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(FrameSize));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(numFrames));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(refBytes));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(resBytes));

        // Reference values (TIn[] — original width)
        const size_t refOff = out.size();
        out.resize(refOff + refBytes);
        std::memcpy(out.data() + refOff, refVals.data(), refBytes);

    // Residuals (sub-encoded byte stream)
        const size_t resOff = out.size();
        out.resize(resOff + resBytes);
    std::memcpy(out.data() + resOff, residualEncoded.data().data(), resBytes);

        encodings::EncodingMetadata meta;
        meta.encodingName     = name();
        meta.dataType         = encodings::core::typeToDataType<TIn>;
        meta.elementCount     = N;
        meta.compressedSize   = out.size();
        meta.uncompressedSize = N * sizeof(TIn);
        meta.supportsRandomAccess = true;

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // ---- Decoder::decodeAll ---------------------------------------------

    std::vector<TIn> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        const ParsedHeader h = parseHeader(encoded);
        if (h.N == 0) return {};

        const auto& residualEncoded = getCachedResidualView(encoded, h);
        auto residualDecoded = cfg_.subEncoder->decodeAll(residualEncoded);
        if (residualDecoded.size() != h.N) {
            throw std::runtime_error("FOREncoder::decodeAll: residual decode size mismatch");
        }

        std::vector<TIn> result(h.N);
        const TIn*  refs = reinterpret_cast<const TIn* >(encoded.data().data() + h.refOffset);

        for (size_t i = 0; i < h.N; ++i) {
            const size_t f = i >> Log2FrameSize;
            result[i] = static_cast<TIn>(refs[f] + static_cast<TIn>(residualDecoded[i]));
        }
        return result;
    }

    // ---- Decoder::decodeAt ----------------------------------------------

    std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        const ParsedHeader h = parseHeader(encoded);
        if (index >= h.N) return std::nullopt;

        const uint8_t* base = encoded.data().data();
        const TIn  ref      = detail_for::readLE<TIn> (base + h.refOffset + (index >> Log2FrameSize) * sizeof(TIn));

        const auto& residualEncoded = getCachedResidualView(encoded, h);

        if (cfg_.subEncoder->properties().has(EncodingProperty::RandomAccess)) {
            auto residualAt = cfg_.subEncoder->decodeAt(residualEncoded, index);
            if (!residualAt.has_value()) {
                throw std::runtime_error("FOREncoder::decodeAt: sub-encoder returned no value for valid index");
            }
            return static_cast<TIn>(ref + static_cast<TIn>(*residualAt));
        }

        auto residualDecoded = cfg_.subEncoder->decodeAll(residualEncoded);
        if (residualDecoded.size() != h.N) {
            throw std::runtime_error("FOREncoder::decodeAt: residual decode size mismatch");
        }
        return static_cast<TIn>(ref + static_cast<TIn>(residualDecoded[index]));
    }

    // ---- Decoder::decodeRange -------------------------------------------

    std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& encoded,
                                 size_t start, size_t end) override {
        const ParsedHeader h = parseHeader(encoded);
        end = std::min(end, h.N);
        if (start >= end) return {};

        const size_t count = end - start;
        std::vector<TIn> result;
        result.reserve(count);

        const uint8_t* base = encoded.data().data();
        const TIn*  refs = reinterpret_cast<const TIn* >(base + h.refOffset);

        const auto& residualEncoded = getCachedResidualView(encoded, h);

        if (cfg_.subEncoder->properties().has(EncodingProperty::RandomAccess)) {
            auto residualSlice = cfg_.subEncoder->decodeRange(residualEncoded, start, end);
            if (residualSlice.size() != count) {
                throw std::runtime_error("FOREncoder::decodeRange: sub-encoder range decode size mismatch");
            }
            for (size_t i = 0; i < count; ++i) {
                result.push_back(static_cast<TIn>(refs[(start + i) >> Log2FrameSize] + static_cast<TIn>(residualSlice[i])));
            }
            return result;
        }

        auto residualDecoded = cfg_.subEncoder->decodeAll(residualEncoded);
        if (residualDecoded.size() != h.N) {
            throw std::runtime_error("FOREncoder::decodeRange: residual decode size mismatch");
        }

        for (size_t i = start; i < end; ++i) {
            result.push_back(static_cast<TIn>(refs[i >> Log2FrameSize] + static_cast<TIn>(residualDecoded[i])));
        }
        return result;
    }

    // ---- Identity -------------------------------------------------------

    EncodingType encodingType() const override {
        return EncodingType::FrameOfReference;
    }

    std::string name() const override {
        return "FOR<" + std::string(typeid(TIn).name()) +
               "→" + std::string(typeid(TOut).name()) +
               ",F=" + std::to_string(FrameSize) + "," +
               policyName() + ">";
    }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        return EncodingProperties{}
            .add(EP::RandomAccess)
            .add(EP::Lossless)
            .add(EP::PreservesOrder)
            .add(EP::FixedSize)
            .add(EP::DeltaBased)
            .add(EP::Composable)
            .add(EP::LowMemoryOverhead);
    }

private:
    // ---- Parsed header (lazy, reconstructed from the byte buffer) --------

    struct ParsedHeader {
        size_t  N;
        size_t  numFrames;
        size_t  resBytes;
        size_t  refOffset;   // byte offset of ref values in encoded buffer
        size_t  resOffset;   // byte offset of residuals in encoded buffer
    };

    static ParsedHeader parseHeader(const EncodedBuffer<uint8_t>& encoded) {
        const uint8_t* p = encoded.data().data();

        const uint64_t N         = detail_for::readLE<uint64_t>(p);      p += 8;
        const uint64_t frameSzE  = detail_for::readLE<uint64_t>(p);      p += 8;
        const uint64_t numFrames = detail_for::readLE<uint64_t>(p);      p += 8;
    const uint64_t refBytes  = detail_for::readLE<uint64_t>(p);      p += 8;
    const uint64_t resBytes  = detail_for::readLE<uint64_t>(p);      p += 8;

        if (frameSzE != FrameSize) {
            throw std::runtime_error(
                "FOREncoder::decode: FrameSize mismatch — encoded with " +
                std::to_string(frameSzE) + ", decoder has FrameSize=" +
                std::to_string(FrameSize));
        }

        const size_t refOffset = static_cast<size_t>(p - encoded.data().data());
        const size_t resOffset = refOffset + static_cast<size_t>(refBytes);

        return {
            .N         = static_cast<size_t>(N),
            .numFrames = static_cast<size_t>(numFrames),
            .resBytes  = static_cast<size_t>(resBytes),
            .refOffset = refOffset,
            .resOffset = resOffset,
        };
    }

    const EncodedBuffer<uint8_t>& getCachedResidualView(const EncodedBuffer<uint8_t>& encoded,
                                                        const ParsedHeader& h) const {
        const uint8_t* basePtr = encoded.data().data();
        const size_t totalSize = encoded.data().size();

        if (cachedOuterPtr_ != basePtr || cachedOuterSize_ != totalSize) {
            std::vector<uint8_t> residualPayload(h.resBytes);
            std::memcpy(residualPayload.data(), basePtr + h.resOffset, h.resBytes);

            encodings::EncodingMetadata residualMeta;
            residualMeta.elementCount = h.N;
            residualMeta.compressedSize = h.resBytes;
            residualMeta.uncompressedSize = h.N * sizeof(TOut);
            residualMeta.supportsRandomAccess = cfg_.subEncoder->properties().has(EncodingProperty::RandomAccess);

            cachedResidualView_ = encodings::EncodedData(std::move(residualPayload), std::move(residualMeta));
            cachedOuterPtr_ = basePtr;
            cachedOuterSize_ = totalSize;
        }

        return cachedResidualView_;
    }

    // ---- Reference computation ------------------------------------------

    TIn computeRef(std::span<const TIn> data, size_t lo, size_t hi) const {
        switch (cfg_.policy) {
            case FORReferencePolicy::FIRST:
                return data[lo];

            case FORReferencePolicy::MIN:
                return *std::min_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                         data.begin() + static_cast<ptrdiff_t>(hi));

            case FORReferencePolicy::MID: {
                const TIn mn = *std::min_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                                  data.begin() + static_cast<ptrdiff_t>(hi));
                const TIn mx = *std::max_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                                  data.begin() + static_cast<ptrdiff_t>(hi));
                // Avoid overflow when computing midpoint of potentially large values
                return mn + static_cast<TIn>((static_cast<int64_t>(mx) - static_cast<int64_t>(mn)) / 2);
            }
        }
        return data[lo]; // unreachable
    }

    const char* policyName() const noexcept {
        switch (cfg_.policy) {
            case FORReferencePolicy::FIRST: return "FIRST";
            case FORReferencePolicy::MIN:   return "MIN";
            case FORReferencePolicy::MID:   return "MID";
        }
        return "?";
    }

    // ---- Empty result helper --------------------------------------------

    static EncodedBuffer<uint8_t> makeEmpty(size_t N) {
        std::vector<uint8_t> out(40, 0); // header only, all zeros
        encodings::EncodingMetadata meta;
        meta.encodingName     = "FOR<empty>";
        meta.elementCount     = N;
        meta.compressedSize   = out.size();
        meta.uncompressedSize = 0;
        meta.supportsRandomAccess = true;
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    FORConfig<TIn, TOut> cfg_;

    // Cache of extracted sub-encoded residual payload keyed by outer encoded buffer identity.
    mutable const uint8_t* cachedOuterPtr_{nullptr};
    mutable size_t cachedOuterSize_{0};
    mutable EncodedBuffer<uint8_t> cachedResidualView_;
};

// ---------------------------------------------------------------------------
// Convenience factory
// ---------------------------------------------------------------------------

/**
 * @brief Create a FOREncoder with the given FrameSize and config.
 *
 * Example:
 *   auto enc = makeFOR<128, int64_t, int16_t>();
 *   auto enc = makeFOR<256, int64_t, int8_t>({.policy = FORReferencePolicy::MID});
 */
template <size_t FrameSize, typename TIn, typename TOut>
std::shared_ptr<FOREncoder<TIn, TOut, FrameSize>>
makeFOR(FORConfig<TIn, TOut> cfg = {}) {
    return std::make_shared<FOREncoder<TIn, TOut, FrameSize>>(cfg);
}

} // namespace encodings::encoders
