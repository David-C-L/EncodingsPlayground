#pragma once

#include <algorithm>
#include <cstdint>
#ifdef __AVX2__
#  include <immintrin.h>
#endif
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <limits>
#include <utility>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encoders/ISectionCodecIntegral.hpp"
#include "encoders/DictionaryEncoder.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/OpenZLEncoder.hpp"
#include "encoders/LZ4Encoder.hpp"
#include "encoders/FSEEncoder.hpp"
#include "encoders/AdaptiveFOREncoder.hpp"
#include "encoders/AdaptiveDictionaryEncoder.hpp"
#include "encoders/AdaptiveFramedBitPrefixEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/HuffmanEncoder.hpp"
#include "encoders/BWTSectionEncoder.hpp"
#include "encoders/selectors/SubStreamReordererType.hpp"
#include "encoders/FrequencyPartitionEncoder.hpp"
#include "encoders/BlockFrequencyPartitionEncoder.hpp"
#include "encoders/BlockFSEEncoder.hpp"
#include "encoders/BlockFORFPEEncoder.hpp"
#include "encoders/MainlyConstantEncoder.hpp"
#include "encoders/FOREncoder.hpp"
#include "encoders/RangePackEncoder.hpp"
#include "encoders/CascadingFOREncoder.hpp"
#include "encoders/BitSplitOrder.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"

namespace encodings::encoders {

// ISectionCodecIntegral lives in its own header (encoders/ISectionCodecIntegral.hpp)
// so new section-codec-level classes (e.g. RangePackEncoder.hpp) can depend on
// it without creating a circular include with this file.

// Parallelism settings shared by SubIntSplitConfigIntegral and SubIntSplitAutoEncoderConfig.
// Disabled by default — zero overhead when enabled==false.
struct SubIntSplitParallelismConfig {
    bool enabled{false};
    // Maximum threads to use. 0 means std::thread::hardware_concurrency().
    size_t maxThreads{0};
};

// Generic config for sub-integer splits (up to 64 sections, sum(bits)=64)
template <typename TIn = uint64_t>
    requires (std::is_same_v<TIn, uint64_t> || std::is_same_v<TIn, uint32_t>)
struct SubIntSplitConfigIntegral {
    constexpr static size_t MAX_BITS = sizeof(TIn) * 8;
    std::vector<uint8_t> bits;  // per-section bit widths
    BitSplitOrder order{BitSplitOrder::LSB_TO_MSB};
    std::vector<std::shared_ptr<ISectionCodecIntegral<TIn>>> codecs; // parallel to bits

    // Optional thread parallelism across sections (encode) and chunks (decode).
    // Disabled by default — zero overhead and identical behaviour when disabled.
    using ParallelismConfig = SubIntSplitParallelismConfig;
    ParallelismConfig parallelism{};

    size_t splitCount() const { return bits.size(); }

    // Validate invariants: sizes match, sum(bits)==MAX_BITS, each bit>0, codecs not null, <=MAX_BITS splits
    void validate() const {
        if (bits.size() != codecs.size()) {
            throw std::invalid_argument("SubIntSplitConfigIntegral: bits and codecs size mismatch");
        }
        if (bits.empty()) {
            throw std::invalid_argument("SubIntSplitConfigIntegral: no splits configured");
        }
        if (bits.size() > MAX_BITS) {
            throw std::invalid_argument("SubIntSplitConfigIntegral: split count cannot exceed " + std::to_string(MAX_BITS));
        }
        uint16_t totalBits = 0;
        for (size_t i = 0; i < bits.size(); ++i) {
            const uint8_t b = bits[i];
            if (b == 0 || b > MAX_BITS) {
                throw std::invalid_argument("SubIntSplitConfigIntegral: each split must have bits in [1," + std::to_string(MAX_BITS) + "]");
            }
            if (!codecs[i]) {
                throw std::invalid_argument("SubIntSplitConfigIntegral: codec entry must not be null");
            }
            totalBits = static_cast<uint16_t>(totalBits + b);
        }
        if (totalBits != MAX_BITS) {
            throw std::invalid_argument("SubIntSplitConfigIntegral: sum of split bits must equal " + std::to_string(MAX_BITS));
        }
    }

    // Options that control how section codecs are constructed inside fromSegments().
    // Does not affect the runtime behaviour of the config itself.
    struct SectionBuildOptions {
        bool fpeSectionParallelDecode{false};
    };

    // Build from selector SegmentPlan results (auto-infers order if not provided).
    static SubIntSplitConfigIntegral fromSegments(const std::vector<selectors::SegmentPlan>& segments,
                                                  SectionBuildOptions opts = {});
    static SubIntSplitConfigIntegral fromSegments(const std::vector<selectors::SegmentPlan>& segments,
                                                  BitSplitOrder orderHint,
                                                  SectionBuildOptions opts = {});
};

using SubIntSplitConfig64 = SubIntSplitConfigIntegral<uint64_t>;
using SubIntSplitConfig32 = SubIntSplitConfigIntegral<uint32_t>;

template <typename T, typename SectionCodecTIn = uint64_t>
        requires (std::is_integral_v<T> && (std::is_same_v<SectionCodecTIn, uint64_t> || std::is_same_v<SectionCodecTIn, uint32_t>))
class TypedSectionCodecAdapter final : public ISectionCodecIntegral<SectionCodecTIn> {
    constexpr static size_t MAX_BITS_SECTION_CODEC = sizeof(SectionCodecTIn) * 8;
    constexpr static size_t MAX_BITS = sizeof(T) * 8;

public:
    explicit TypedSectionCodecAdapter(std::shared_ptr<Codec<T, uint8_t>> impl, uint8_t bits)
        : impl_(std::move(impl)), bits_(bits) {
        if (!impl_) {
            throw std::invalid_argument("TypedSectionCodecAdapter: impl must not be null");
        }
        if (bits_ == 0 || bits_ > MAX_BITS_SECTION_CODEC) {
            throw std::invalid_argument("TypedSectionCodecAdapter: bits must be in [1," + std::to_string(MAX_BITS_SECTION_CODEC) + "]");
        }
        if (bits_ > MAX_BITS) {
            throw std::invalid_argument("TypedSectionCodecAdapter: bits exceed underlying type width: " + std::to_string(MAX_BITS));
        }
        mask_ = bits_ == MAX_BITS_SECTION_CODEC
            ? static_cast<SectionCodecTIn>(std::numeric_limits<SectionCodecTIn>::max())
            : ((SectionCodecTIn{1} << bits_) - 1);
    }

    EncodedBuffer<uint8_t> encode(std::span<const SectionCodecTIn> data) override {
        // Validate input branch-free
        if (std::any_of(data.begin(), data.end(), [this](SectionCodecTIn v) {
            return (v & ~mask_) != 0; })) [[unlikely]] {
            throw std::overflow_error("Section value does not fit in configured bit width");
        }

        // Narrow input branch-free
        std::vector<T> narrowed(data.size());
        std::transform(data.begin(), data.end(), narrowed.begin(), [](SectionCodecTIn v) {
            return static_cast<T>(v);
        });
        return impl_->encode(std::span<const T>(narrowed.data(), narrowed.size()));
    }

    std::vector<SectionCodecTIn> decodeAll(const EncodedBuffer<uint8_t>& enc) override {
        auto vals = impl_->decodeAll(enc);
        // reserve+push_back writes each element exactly once with no preceding
        // zero-init pass — vector(n) would memset n*sizeof(SectionCodecTIn) bytes
        // that are immediately overwritten by the cast/copy loop.
        std::vector<SectionCodecTIn> widened;
        widened.reserve(vals.size());
        for (const T v : vals) widened.push_back(static_cast<SectionCodecTIn>(v));
        return widened;
    }

    std::optional<SectionCodecTIn> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t idx) override {
        auto v = impl_->decodeAt(enc, idx);
        if (!v) return std::nullopt;
        return static_cast<SectionCodecTIn>(*v);
    }

    std::vector<SectionCodecTIn> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) override {
        auto vals = impl_->decodeRange(enc, start, end);
        // Same as decodeAll: avoid the zero-init of vector(n) that would be
        // immediately overwritten.
        std::vector<SectionCodecTIn> widened;
        widened.reserve(vals.size());
        for (const T v : vals) widened.push_back(static_cast<SectionCodecTIn>(v));
        return widened;
    }

    void decodeAllInto(const EncodedBuffer<uint8_t>& enc,
                        SectionCodecTIn* dst, size_t n) override {
        auto vals = impl_->decodeAll(enc);
        if (vals.size() != n) [[unlikely]]
            throw std::runtime_error("TypedSectionCodecAdapter::decodeAllInto: size mismatch");
        if constexpr (sizeof(T) == sizeof(SectionCodecTIn)) {
            std::memcpy(dst, vals.data(), n * sizeof(T));
        } else {
            for (size_t i = 0; i < n; ++i)
                dst[i] = static_cast<SectionCodecTIn>(vals[i]);
        }
    }

    void decodeRangeInto(const EncodedBuffer<uint8_t>& enc,
                          size_t start, size_t end,
                          SectionCodecTIn* dst, size_t n) override {
        auto vals = impl_->decodeRange(enc, start, end);
        if (vals.size() != n) [[unlikely]]
            throw std::runtime_error("TypedSectionCodecAdapter::decodeRangeInto: size mismatch");
        if constexpr (sizeof(T) == sizeof(SectionCodecTIn)) {
            std::memcpy(dst, vals.data(), n * sizeof(T));
        } else {
            for (size_t i = 0; i < n; ++i)
                dst[i] = static_cast<SectionCodecTIn>(vals[i]);
        }
    }

    // Optimised fused overrides: use a persistent, zero-init-free scratch buffer to
    // avoid a per-call heap allocation.  impl_->decodeAllInto() writes narrow T values
    // directly into the scratch; the accumulate loop then widens and shifts into acc.
    // Thread-safety: sectionScratch_ is accessed without synchronisation — safe while
    // each SubIntSplitEncoder instance is used from a single thread.
    void decodeAllAndAccumulate(const EncodedBuffer<uint8_t>& enc,
                                SectionCodecTIn* acc, size_t n,
                                uint8_t shift, bool isFirst) override {
        ensureScratch(n);
        impl_->decodeAllInto(enc, sectionScratch_.get(), n);
        accumulateScratch(acc, n, shift, isFirst);
    }

    void decodeRangeAndAccumulate(const EncodedBuffer<uint8_t>& enc,
                                  size_t start, size_t end,
                                  SectionCodecTIn* acc, size_t n,
                                  uint8_t shift, bool isFirst) override {
        ensureScratch(n);
        impl_->decodeRangeInto(enc, start, end, sectionScratch_.get(), n);
        accumulateScratch(acc, n, shift, isFirst);
    }

    EncodingProperties properties() const override { return impl_->properties(); }
    std::string name() const override { return impl_->name(); }

private:
    void ensureScratch(size_t n) const {
        if (scratchSize_ < n) {
            sectionScratch_ = std::make_unique_for_overwrite<T[]>(n);
            scratchSize_ = n;
        }
    }

    void accumulateScratch(SectionCodecTIn* acc, size_t n, uint8_t shift, bool isFirst) const {
        const T* src = sectionScratch_.get();
#ifdef __AVX2__
        // AVX2 widening paths: process 4 elements per iteration for all narrow T→64-bit
        // and narrow T→32-bit combinations.  The scalar tail handles the remainder.
        if constexpr (sizeof(SectionCodecTIn) == 8) {
            // --- 64-bit output ---
            const __m128i vshift = _mm_cvtsi64_si128(static_cast<int64_t>(shift));

            if constexpr (sizeof(T) == 1) {
                // uint8 → uint64: _mm256_cvtepu8_epi64 takes lowest 4 bytes of __m128i
                size_t i = 0;
                for (; i + 4 <= n; i += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(src + i + 32), _MM_HINT_T1);
                    _mm_prefetch(reinterpret_cast<const char*>(acc + i + 32), _MM_HINT_T1);
                    int32_t tmp; std::memcpy(&tmp, src + i, 4);
                    __m256i vs = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(tmp));
                    if (shift) vs = _mm256_sll_epi64(vs, vshift);
                    if (isFirst) {
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), vs);
                    } else {
                        __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_or_si256(vd, vs));
                    }
                }
                for (; i < n; ++i)
                    isFirst ? (acc[i]  = static_cast<SectionCodecTIn>(src[i]) << shift)
                            : (acc[i] |= static_cast<SectionCodecTIn>(src[i]) << shift);
                return;
            } else if constexpr (sizeof(T) == 2) {
                // uint16 → uint64: _mm256_cvtepu16_epi64 takes lowest 8 bytes of __m128i
                size_t i = 0;
                for (; i + 4 <= n; i += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(src + i + 32), _MM_HINT_T1);
                    _mm_prefetch(reinterpret_cast<const char*>(acc + i + 32), _MM_HINT_T1);
                    __m256i vs = _mm256_cvtepu16_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i)));
                    if (shift) vs = _mm256_sll_epi64(vs, vshift);
                    if (isFirst) {
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), vs);
                    } else {
                        __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_or_si256(vd, vs));
                    }
                }
                for (; i < n; ++i)
                    isFirst ? (acc[i]  = static_cast<SectionCodecTIn>(src[i]) << shift)
                            : (acc[i] |= static_cast<SectionCodecTIn>(src[i]) << shift);
                return;
            } else if constexpr (sizeof(T) == 4) {
                // uint32 → uint64: _mm256_cvtepu32_epi64 takes 128-bit register (4×32)
                size_t i = 0;
                for (; i + 4 <= n; i += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(src + i + 32), _MM_HINT_T1);
                    _mm_prefetch(reinterpret_cast<const char*>(acc + i + 32), _MM_HINT_T1);
                    __m256i vs = _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
                    if (shift) vs = _mm256_sll_epi64(vs, vshift);
                    if (isFirst) {
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), vs);
                    } else {
                        __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_or_si256(vd, vs));
                    }
                }
                for (; i < n; ++i)
                    isFirst ? (acc[i]  = static_cast<SectionCodecTIn>(src[i]) << shift)
                            : (acc[i] |= static_cast<SectionCodecTIn>(src[i]) << shift);
                return;
            }
            // sizeof(T)==8: no widening, same-width path falls through to scalar below
        } else if constexpr (sizeof(SectionCodecTIn) == 4) {
            // --- 32-bit output ---
            const __m128i vshift = _mm_cvtsi32_si128(static_cast<int32_t>(shift));

            if constexpr (sizeof(T) == 1) {
                // uint8 → uint32: _mm256_cvtepu8_epi32 takes lowest 8 bytes of __m128i
                size_t i = 0;
                for (; i + 8 <= n; i += 8) {
                    _mm_prefetch(reinterpret_cast<const char*>(src + i + 64), _MM_HINT_T1);
                    _mm_prefetch(reinterpret_cast<const char*>(acc + i + 64), _MM_HINT_T1);
                    int64_t tmp; std::memcpy(&tmp, src + i, 8);
                    __m256i vs = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(tmp));
                    if (shift) vs = _mm256_sll_epi32(vs, vshift);
                    if (isFirst) {
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), vs);
                    } else {
                        __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_or_si256(vd, vs));
                    }
                }
                for (; i < n; ++i)
                    isFirst ? (acc[i]  = static_cast<SectionCodecTIn>(src[i]) << shift)
                            : (acc[i] |= static_cast<SectionCodecTIn>(src[i]) << shift);
                return;
            } else if constexpr (sizeof(T) == 2) {
                // uint16 → uint32: _mm256_cvtepu16_epi32 takes 128-bit register (8×16)
                size_t i = 0;
                for (; i + 8 <= n; i += 8) {
                    _mm_prefetch(reinterpret_cast<const char*>(src + i + 64), _MM_HINT_T1);
                    _mm_prefetch(reinterpret_cast<const char*>(acc + i + 64), _MM_HINT_T1);
                    __m256i vs = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
                    if (shift) vs = _mm256_sll_epi32(vs, vshift);
                    if (isFirst) {
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), vs);
                    } else {
                        __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_or_si256(vd, vs));
                    }
                }
                for (; i < n; ++i)
                    isFirst ? (acc[i]  = static_cast<SectionCodecTIn>(src[i]) << shift)
                            : (acc[i] |= static_cast<SectionCodecTIn>(src[i]) << shift);
                return;
            }
            // sizeof(T)==4: same-width, falls through to scalar
        }
#endif // __AVX2__
        // Scalar fallback: same-width T==SectionCodecTIn, or non-AVX2 build.
        if (isFirst) {
            for (size_t i = 0; i < n; ++i)
                acc[i] = static_cast<SectionCodecTIn>(src[i]) << shift;
        } else {
            for (size_t i = 0; i < n; ++i)
                acc[i] |= static_cast<SectionCodecTIn>(src[i]) << shift;
        }
    }

    std::shared_ptr<Codec<T, uint8_t>> impl_;
    uint8_t bits_{};
    SectionCodecTIn mask_{};
    mutable std::unique_ptr<T[]> sectionScratch_;
    mutable size_t scratchSize_{0};
};

// Helper to choose underlying unsigned type based on bit width
inline constexpr uint8_t chooseTypeBits(uint8_t bits) {
    if (bits <= 8) return 8;
    if (bits <= 16) return 16;
    if (bits <= 32) return 32;
    return 64;
}

template <typename T, typename SectionCodecTIn = uint64_t>
std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodec(std::shared_ptr<Codec<T, uint8_t>> codec, uint8_t bits) {
    return std::make_shared<TypedSectionCodecAdapter<T, SectionCodecTIn>>(std::move(codec), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodecForBits(std::shared_ptr<Codec<uint8_t, uint8_t>> c, uint8_t bits) {
    return makeSectionCodec<uint8_t, SectionCodecTIn>(std::move(c), bits);
}
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodecForBits(std::shared_ptr<Codec<uint16_t, uint8_t>> c, uint8_t bits) {
    return makeSectionCodec<uint16_t, SectionCodecTIn>(std::move(c), bits);
}
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodecForBits(std::shared_ptr<Codec<uint32_t, uint8_t>> c, uint8_t bits) {
    return makeSectionCodec<uint32_t, SectionCodecTIn>(std::move(c), bits);
}
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodecForBits(std::shared_ptr<Codec<uint64_t, uint8_t>> c, uint8_t bits) {
    return makeSectionCodec<uint64_t, SectionCodecTIn>(std::move(c), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodecForBits(std::shared_ptr<Codec<int8_t, uint8_t>> c, uint8_t bits) {
    return makeSectionCodec<int8_t, SectionCodecTIn>(std::move(c), bits);
}
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodecForBits(std::shared_ptr<Codec<int16_t, uint8_t>> c, uint8_t bits) {
    return makeSectionCodec<int16_t, SectionCodecTIn>(std::move(c), bits);
}
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodecForBits(std::shared_ptr<Codec<int32_t, uint8_t>> c, uint8_t bits) {
    return makeSectionCodec<int32_t, SectionCodecTIn>(std::move(c), bits);
}
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeSectionCodecForBits(std::shared_ptr<Codec<int64_t, uint8_t>> c, uint8_t bits) {
    return makeSectionCodec<int64_t, SectionCodecTIn>(std::move(c), bits);
}

// ---------------------------------------------------------------------------
// Generic section factory helpers (shared across encoders)
// ---------------------------------------------------------------------------

namespace detail_trisplit {

inline constexpr uint64_t maxValueForBits(uint8_t bits) {
    return bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
}

inline constexpr uint8_t chooseSignedWidthForBits(uint8_t bits) {
    const uint64_t maxVal = maxValueForBits(bits);
    if (maxVal <= static_cast<uint64_t>(std::numeric_limits<int8_t>::max()))  return 8;
    if (maxVal <= static_cast<uint64_t>(std::numeric_limits<int16_t>::max())) return 16;
    if (maxVal <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) return 32;
    return 64;
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeDictionarySection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<DictionaryEncoder<uint8_t>>(), bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<DictionaryEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<DictionaryEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<DictionaryEncoder<uint64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeAdaptiveDictionarySection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case  8: return makeSectionCodecForBits<SectionCodecTIn>(
                     std::make_shared<AdaptiveDictionaryEncoder<uint8_t>>(),  bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(
                     std::make_shared<AdaptiveDictionaryEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(
                     std::make_shared<AdaptiveDictionaryEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(
                     std::make_shared<AdaptiveDictionaryEncoder<uint64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeRawSection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RawEncoder<uint8_t>>(), bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RawEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RawEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RawEncoder<uint64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeRawBitPackedSection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RawBitPackedEncoder<uint8_t>>(), bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RawBitPackedEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RawBitPackedEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RawBitPackedEncoder<uint64_t>>(), bits);
    }
}

template <size_t BlockSize = 0, typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeOpenZLSection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<OpenZLCodec<uint8_t, BlockSize>>(), bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<OpenZLCodec<uint16_t, BlockSize>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<OpenZLCodec<uint32_t, BlockSize>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<OpenZLCodec<uint64_t, BlockSize>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeAdaptiveFORSection(uint8_t bits) {
    // Use the narrowest signed type that fits the section's value range so that
    // AdaptiveFOR's ref storage (sizeof(TIn) per frame) matches the cost model's
    // storageWidthBits(bits)/8 estimate.  Using int64_t unconditionally causes a
    // 4× ref-size mismatch for small bit widths, corrupting adjacent heap allocations.
    const uint8_t w = chooseSignedWidthForBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<AdaptiveFOREncoder<int8_t>>(),  bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<AdaptiveFOREncoder<int16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<AdaptiveFOREncoder<int32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<AdaptiveFOREncoder<int64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeRLESection(uint8_t bits) {
    // RunLengthEncoder stores (runStarts[], runValues[]) and supports O(log R) random access
    // via binary search on the runStarts array — suitable for sub-streams with long constant runs.
    // Type-switch matches storageWidthBits() used in RLECostModel so cost estimates are accurate.
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RunLengthEncoder<uint8_t>>(), bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RunLengthEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RunLengthEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<RunLengthEncoder<uint64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeAdaptiveFramedBitPrefixSection(uint8_t bits) {
    // Residual width is chosen internally; bits only gates the narrowing adapter.
    return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<AdaptiveFramedBitPrefixEncoder<int64_t>>(), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeHuffmanSection(uint8_t bits) {
    // HuffmanEncoder is sequential-only (no random access); instantiated at the
    // storage type matching chooseTypeBits() / storageWidthBits() so cost estimates are accurate.
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<HuffmanEncoder<uint8_t>>(),  bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<HuffmanEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<HuffmanEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<HuffmanEncoder<uint64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeLZ4Section(uint8_t bits) {
    // LZ4 is block-compressed and effectively sequential; decodeAt/decodeRange
    // fall back to decodeAll in the codec for benchmark API compatibility.
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<LZ4Encoder<uint8_t>>(), bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<LZ4Encoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<LZ4Encoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<LZ4Encoder<uint64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeFSESection(uint8_t bits) {
    // FSE is entropy coding and sequential; decodeAt/decodeRange in the codec
    // are decodeAll fallbacks for benchmark API compatibility.
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FSEEncoder<uint8_t>>(), bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FSEEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FSEEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FSEEncoder<uint64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeFrequencyPartitionSection(
    uint8_t bits, bool parallelDecode = false) {
    const FrequencyPartitionConfig fpeCfg{.parallelDecode = parallelDecode};
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FrequencyPartitionEncoder<uint8_t>>(fpeCfg),  bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FrequencyPartitionEncoder<uint16_t>>(fpeCfg), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FrequencyPartitionEncoder<uint32_t>>(fpeCfg), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FrequencyPartitionEncoder<uint64_t>>(fpeCfg), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeBlockFrequencyPartitionSection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFrequencyPartitionEncoder<uint8_t>>(),  bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFrequencyPartitionEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFrequencyPartitionEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFrequencyPartitionEncoder<uint64_t>>(), bits);
    }
}

// RangePack composed with FrequencyPartitionEncoding / BlockFrequencyPartitionEncoding
// (type-narrowed — see encoders/RangePackEncoder.hpp for why a same-width
// value shift alone would not help these two encoders). The outer 'bits'
// parameter is accepted only for signature uniformity with every other
// makeXSection(uint8_t bits) factory; RangePackSectionCodec derives the real
// narrowed width from the data itself at encode() time.
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeRangePackFrequencyPartitionSection(uint8_t /*bits*/) {
    return makeRangePackSection<SectionCodecTIn>(
        [](uint8_t narrowedBits) { return makeFrequencyPartitionSection<SectionCodecTIn>(narrowedBits); },
        encodings::EncodingType::RangePackFrequencyPartitionEncoding);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeRangePackBlockFrequencyPartitionSection(uint8_t /*bits*/) {
    return makeRangePackSection<SectionCodecTIn>(
        [](uint8_t narrowedBits) { return makeBlockFrequencyPartitionSection<SectionCodecTIn>(narrowedBits); },
        encodings::EncodingType::RangePackBlockFrequencyPartitionEncoding);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeBlockFSESection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case  8: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFSEEncoder<uint8_t>>(),  bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFSEEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFSEEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFSEEncoder<uint64_t>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeBlockFrequencyPartitionFORSection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case  8: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFrequencyPartitionEncoder<uint8_t,  FORPrepass::GlobalFOR>>(), bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFrequencyPartitionEncoder<uint16_t, FORPrepass::GlobalFOR>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFrequencyPartitionEncoder<uint32_t, FORPrepass::GlobalFOR>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFrequencyPartitionEncoder<uint64_t, FORPrepass::GlobalFOR>>(), bits);
    }
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeBlockFORFPESection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case  8: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFORFPEEncoder<uint8_t>>(),  bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFORFPEEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFORFPEEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<BlockFORFPEEncoder<uint64_t>>(), bits);
    }
}

// Flat (maxDepth=0) MainlyConstant section for SubIntSplit — no recursive overhead.
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeMainlyConstantSection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    switch (w) {
        case 8:  return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<MainlyConstantEncoder<uint8_t>>(),  bits);
        case 16: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<MainlyConstantEncoder<uint16_t>>(), bits);
        case 32: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<MainlyConstantEncoder<uint32_t>>(), bits);
        default: return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<MainlyConstantEncoder<uint64_t>>(), bits);
    }
}

template <size_t FrameSize = 512, typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>> makeFORSection(uint8_t bits,
                                                FORReferencePolicy policy = FORReferencePolicy::MIN) {
    const uint8_t w = chooseSignedWidthForBits(bits);

    if (w == 8) {
        using T = int8_t;
        FORConfig<T, T> cfg{.policy = policy, .subEncoder = std::make_shared<RawEncoder<T>>()};
    return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FOREncoder<T, T, FrameSize>>(cfg), bits);
    }
    if (w == 16) {
        using T = int16_t;
        FORConfig<T, T> cfg{.policy = policy, .subEncoder = std::make_shared<RawEncoder<T>>()};
    return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FOREncoder<T, T, FrameSize>>(cfg), bits);
    }
    if (w == 32) {
        using T = int32_t;
        FORConfig<T, T> cfg{.policy = policy, .subEncoder = std::make_shared<RawEncoder<T>>()};
    return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FOREncoder<T, T, FrameSize>>(cfg), bits);
    }
    {
        using T = int64_t;
        FORConfig<T, T> cfg{.policy = policy, .subEncoder = std::make_shared<RawEncoder<T>>()};
        return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<FOREncoder<T, T, FrameSize>>(cfg), bits);
    }
}

// CascadingFOREncoder, standalone. Internal cascade arithmetic is always
// int64_t regardless of the section's own T (per CascadingFOREncoder's own
// design), so there is exactly one instantiation needed here, not a width
// switch like every other makeXSection factory -- but `bits` is still passed
// straight through to makeSectionCodecForBits (NOT hardcoded to 64), since
// TypedSectionCodecAdapter uses it to validate incoming values actually fit in
// the section's declared width, exactly like every other factory.
// Default schedule mirrors makeFORSection<512,...>'s frame-size convention
// (512 is the FrameSize used by every existing makeFORSection<512,...> call
// site in this file), with a small reference cascade so the cascade machinery
// itself is exercised rather than degenerating to a single flat FOR pass.
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORSection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule  = { {512, FORReferencePolicy::MIN} };
    cfg.referenceSchedule = { {64,  FORReferencePolicy::MIN} };
    // residualLeafEncoder / referenceLeafEncoder left at CascadingFORConfig's
    // own defaults (RawBitPackedEncoder<int64_t>).
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

// CascadingFOR composed with BlockFrequencyPartitionEncoder on the residual
// stream ("BlockFrequencyPartitionEncoder on the residuals and CascadingFOR on
// the references, then eventually bitpacking") -- same schedule as
// makeCascadingFORSection above, but with a BlockFrequencyPartitionEncoder
// residual leaf instead of the default RawBitPackedEncoder; referenceLeafEncoder
// stays default (plain bitpacked).
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORBlockFrequencyPartitionSection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::MIN} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

// CascadingFOR composed with a genuine entropy coder (FSE/BlockFSE/Huffman) on
// the residual stream, same "1 level of CascadingFOR + bitpacked leaf" schedule
// on the references as the other makeCascadingFOR*Section factories above.
//
// RANDOM ACCESS CAVEAT: unlike runStarts (sparse, R << N) or the per-frame
// reference array (shrinks by frameSize each level), the RESIDUAL stream is
// always exactly N-sized regardless of cascade depth -- residual arrays
// preserve full N-length indexing at every level (see CascadingFOREncoder.hpp's
// class doc). So composing with a residual leaf that lacks genuine RandomAccess
// does NOT get the same "safe because sparse" argument that motivated
// RunLengthEncoder's runStarts composition:
//   - makeCascadingFORFSESection / makeCascadingFORHuffmanSection: FSEEncoder
//     and HuffmanEncoder do NOT have genuine random access (their properties()
//     no longer claims otherwise -- see FSEEncoder.hpp/HuffmanEncoder.hpp fixes).
//     CascadingFOREncoder::properties() correctly reports RandomAccess=false for
//     these two compositions, and decodeAt/decodeRange fall back to a one-time
//     cached decodeAll() of the FULL N-element residual stream -- i.e. an O(N)
//     first touch, same cost a flat standalone FSEEncoding/HuffmanEncoding
//     section already has today. These two are registered for OFFLINE
//     comparison in the oracle DP (to see whether entropy coding could help
//     compression at all here) -- NOT as safe candidates for real random-access
//     deployment.
//   - makeCascadingFORBlockFSESection: BlockFSEEncoder genuinely has O(blockSize)
//     decodeAt (confirmed via its own properties()/docstring), so this
//     composition DOES preserve real random access end-to-end.
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORFSESection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::MIN} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<FSEEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORBlockFSESection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::MIN} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<BlockFSEEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORHuffmanSection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::MIN} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<HuffmanEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

// PREV-policy siblings of the three makeCascadingFOR*Section factories above
// (FSE/BlockFSE/Huffman) plus PREV-composed FrequencyPartitionEncoder/
// BlockFrequencyPartitionEncoder leaves: residualSchedule uses
// FORReferencePolicy::PREV (consecutive-element delta, bounded to this
// frame -- see FOREncoder.hpp's FORReferencePolicy::PREV doc) instead of MIN.
// referenceSchedule stays MIN -- the reference array stores raw data[lo]
// anchors, not deltas, so PREV doesn't apply there.
//
// Motivation (root-caused this session against the real Twitter Snowflake
// dataset): bits[22..49] (28 bits) is stuck at plain BitPacking -- every
// registered MIN-policy encoding fails to beat it, while MANUALLY
// byte-plane-transposing the segment (4 separate 8-bit-wide arrays), THEN
// delta-coding WITHIN each plane, THEN Huffman/FSE-coding each plane
// separately, reproduced OpenZL's ~10.6% external-tool win almost exactly.
//
// IMPORTANT (empirically verified, do not skip when evaluating this
// composition): PREV applied directly to the WHOLE 28-bit value (as these
// factories do -- there is no byte-plane transpose here) does NOT reproduce
// that win. A real sweep against the actual bits[22..49] column (10M rows)
// showed every leaf performing 2-3x WORSE than plain BitPacking (Huffman:
// ~81MB, BlockFSE: ~108MB, FrequencyPartition: ~71MB,
// BlockFrequencyPartition: ~80MB, all vs BitPacking's 35MB; FSE couldn't even
// build a table -- residual cardinality ~6M exceeds its ~1M-symbol limit).
// This is because the FULL-VALUE delta stream is still close to injective:
// only 2 of the segment's 4 underlying bytes have real delta-friendly
// structure, and mixing all 4 bytes' deltas into one value swamps that
// structure with the other two (near-random) bytes' noise. Frame size barely
// matters for this ill-fitting case (best vs 512/64 default differed by only
// 0.16-0.30% across all 5 leaves), which is itself informative -- there is no
// "correct" tuning to rescue a fundamentally mismatched transform granularity.
//
// These compositions remain genuinely useful for segments where the FULL
// VALUE (not just some of its bytes) is delta-friendly -- confirmed via this
// session's own test suite (test_prev_reference_policy.cpp's monotone/
// drifting-counter cases show real, large wins). Closing the bits[22..49]
// gap specifically requires an actual byte-plane transpose reorderer
// (splitting a section into its native byte-planes before any of these
// factories run on each plane independently) -- a separate, not-yet-built
// follow-up; PREV alone is necessary but not sufficient for that gap.
//
// Frame size 512/64 kept at the existing MIN-policy siblings' convention --
// confirmed empirically (not a placeholder) that finer tuning doesn't matter
// for the mismatched full-value case above, and no delta-friendly-full-value
// dataset was available in this session to tune against instead.
//
// CascadingFORPrevFrequencyPartitionEncoding has no existing MIN-policy
// sibling (only the Block variant was ever wired up as a Cascading
// composition) -- registered directly in its PREV form since that's the
// combination actually motivated by the root-cause investigation above.
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORPrevFSESection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::PREV} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<FSEEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORPrevBlockFSESection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::PREV} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<BlockFSEEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORPrevHuffmanSection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::PREV} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<HuffmanEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORPrevFrequencyPartitionSection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::PREV} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<FrequencyPartitionEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORPrevBlockFrequencyPartitionSection(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {512, FORReferencePolicy::PREV} };
    cfg.referenceSchedule   = { {64,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>();
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

// Dedicated schedule for CascadingFOR applied to a runStarts array specifically
// (as opposed to makeCascadingFORSection's 512/64, tuned for dense per-element
// VALUE streams). Empirically validated against the actual runStarts arrays of
// the two real segments RunLengthCascadingFOREncoding wins on in the Twitter
// Snowflake benchmark (bits[51..58]: 658,849 runs; bits[59..63]: 6,057 runs):
// a frame-size sweep (64/128/256/512/1024/2048/4096 residual x 8/16/32/64/128
// reference) showed 128/16 is best-or-near-best on BOTH -- 1,164,411 vs the
// old 512/64 default's 1,238,787 bytes (-6.0%) on the first, and 14,699 vs
// 16,099 bytes (-8.7%) on the second (not quite the absolute best there --
// 64/8 gets 14,067 -- but 128/16 is the better single compromise across both).
// This makes intuitive sense: runStarts values grow at a roughly constant
// rate (~N/numRuns apart on average), so a SMALL frame keeps the local
// reference tight relative to that growth; a dense value stream doesn't have
// this property, which is why makeCascadingFORSection's own 512/64 default
// stays reasonable there (only ~0.2-0.3% off the empirical optimum when swept
// the same way against real bits[0..12]/[14..21]/[22..50] value streams).
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeCascadingFORSectionForRunStarts(uint8_t bits) {
    CascadingFORConfig cfg;
    cfg.residualSchedule  = { {128, FORReferencePolicy::MIN} };
    cfg.referenceSchedule = { {16,  FORReferencePolicy::MIN} };
    return makeSectionCodecForBits<SectionCodecTIn>(
        std::make_shared<CascadingFOREncoder<int64_t>>(std::move(cfg)), bits);
}

// RunLengthEncoding with its runStarts array (absolute run-start positions,
// monotonically increasing, R entries where R = number of runs, R << N)
// compressed via CascadingFOR instead of stored at a fixed 8 bytes/run --
// runStarts is exactly the "sparse, aggregate-level stream" a FOR-cascade is
// safe to apply to without sacrificing random access (see RunLengthConfig's
// doc in RunLengthEncoder.hpp). runValues stays raw. The runStartsFactory
// lambda is uint64_t-typed regardless of the section's own T, so it is reused
// unchanged across all four width branches below.
template <typename SectionCodecTIn = uint64_t>
inline std::shared_ptr<ISectionCodecIntegral<SectionCodecTIn>>
makeRLECascadingFORSection(uint8_t bits) {
    const uint8_t w = chooseTypeBits(bits);
    auto runStartsFactory = [](uint8_t startsBits) {
        return makeCascadingFORSectionForRunStarts<uint64_t>(startsBits);
    };
    switch (w) {
        case 8: {
            RunLengthConfig<uint8_t> cfg;
            cfg.runStartsFactory = runStartsFactory;
            return makeSectionCodecForBits<SectionCodecTIn>(
                std::make_shared<RunLengthEncoder<uint8_t>>(std::move(cfg)), bits);
        }
        case 16: {
            RunLengthConfig<uint16_t> cfg;
            cfg.runStartsFactory = runStartsFactory;
            return makeSectionCodecForBits<SectionCodecTIn>(
                std::make_shared<RunLengthEncoder<uint16_t>>(std::move(cfg)), bits);
        }
        case 32: {
            RunLengthConfig<uint32_t> cfg;
            cfg.runStartsFactory = runStartsFactory;
            return makeSectionCodecForBits<SectionCodecTIn>(
                std::make_shared<RunLengthEncoder<uint32_t>>(std::move(cfg)), bits);
        }
        default: {
            RunLengthConfig<uint64_t> cfg;
            cfg.runStartsFactory = runStartsFactory;
            return makeSectionCodecForBits<SectionCodecTIn>(
                std::make_shared<RunLengthEncoder<uint64_t>>(std::move(cfg)), bits);
        }
    }
}

} // namespace detail_trisplit

// ---------------------------------------------------------------------------
// Helper: build a typed Codec<T, uint8_t> from an EncodingType tag.
// Used by the BWT wrapping path in fromSegments() to create an inner codec
// that the BWTSectionEncoder can delegate to.
// FOR/AdaptiveFOR are excluded (they require signed types and are not beneficial
// in combination with BWT, which already handles structured data well).
// ---------------------------------------------------------------------------
namespace detail_trisplit {

template <typename T>
    requires std::is_unsigned_v<T>
inline std::shared_ptr<encodings::Codec<T, uint8_t>>
makeTypedSectionCodec(encodings::EncodingType enc) {
    switch (enc) {
        case encodings::EncodingType::DictionaryEncoding:
            return std::make_shared<AdaptiveDictionaryEncoder<T>>();
            // return std::make_shared<DictionaryEncoder<T>>();
        case encodings::EncodingType::AdaptiveDictionaryEncoding:
            return std::make_shared<AdaptiveDictionaryEncoder<T>>();
        case encodings::EncodingType::BlockFrequencyPartitionEncoding:
            return std::make_shared<BlockFrequencyPartitionEncoder<T>>();
        case encodings::EncodingType::BlockFrequencyPartitionFOREncoding:
            return std::make_shared<BlockFrequencyPartitionEncoder<T, FORPrepass::GlobalFOR>>();
        case encodings::EncodingType::BlockFSEEncoding:
            return std::make_shared<BlockFSEEncoder<T>>();
        case encodings::EncodingType::BitPacking:
            return std::make_shared<RawBitPackedEncoder<T>>();
        case encodings::EncodingType::RunLengthEncoding:
            return std::make_shared<RunLengthEncoder<T>>();
        case encodings::EncodingType::HuffmanEncoding:
            return std::make_shared<HuffmanEncoder<T>>();
        case encodings::EncodingType::LZ4:
            return std::make_shared<LZ4Encoder<T>>();
        case encodings::EncodingType::FSEEncoding:
            return std::make_shared<FSEEncoder<T>>();
        case encodings::EncodingType::RawEncoding:
        default:
            return std::make_shared<RawEncoder<T>>();
    }
}

} // namespace detail_trisplit

// ---------------------------------------------------------------------------
// SubIntSplitConfigIntegral factories from SegmentPlan
// ---------------------------------------------------------------------------
namespace SubIntSplitConfigIntegralFactory {

inline BitSplitOrder inferOrderFromSegments(const std::vector<selectors::SegmentPlan>& segments) {
    if (segments.empty()) {
        throw std::invalid_argument("SubIntSplitConfigIntegral::fromSegments: no segments provided");
    }
    // Assume either ascending (LSB->MSB) or descending (MSB->LSB) by bitStart.
    bool ascending = true;
    bool descending = true;
    for (size_t i = 1; i < segments.size(); ++i) {
        if (segments[i].bitStart < segments[i - 1].bitStart) ascending = false;
        if (segments[i].bitStart > segments[i - 1].bitStart) descending = false;
    }
    if (ascending && !descending) return BitSplitOrder::LSB_TO_MSB;
    if (descending && !ascending) return BitSplitOrder::MSB_TO_LSB;
    // Fallback: determine from first vs last bitStart
    return (segments.front().bitStart <= segments.back().bitStart)
               ? BitSplitOrder::LSB_TO_MSB
               : BitSplitOrder::MSB_TO_LSB;
}

} // namespace SubIntSplitConfigIntegralFactory

// ---------------------------------------------------------------------------
// SubIntSplitConfigIntegral static builders
// ---------------------------------------------------------------------------

template <typename TIn>
    requires (std::is_same_v<TIn, uint64_t> || std::is_same_v<TIn, uint32_t>)
inline SubIntSplitConfigIntegral<TIn> SubIntSplitConfigIntegral<TIn>::fromSegments(
    const std::vector<selectors::SegmentPlan>& segments,
    SectionBuildOptions opts) {
    const BitSplitOrder inferred = SubIntSplitConfigIntegralFactory::inferOrderFromSegments(segments);
    return fromSegments(segments, inferred, opts);
}

template <typename TIn>
    requires (std::is_same_v<TIn, uint64_t> || std::is_same_v<TIn, uint32_t>)
inline SubIntSplitConfigIntegral<TIn> SubIntSplitConfigIntegral<TIn>::fromSegments(
    const std::vector<selectors::SegmentPlan>& segments,
    BitSplitOrder orderHint,
    SectionBuildOptions opts) {
    if (segments.empty()) {
        throw std::invalid_argument("SubIntSplitConfigIntegral::fromSegments: no segments provided");
    }

    // std::cout << opts.fpeSectionParallelDecode << std::endl;

    // Copy and sort based on order hint to enforce contiguity check.
    std::vector<selectors::SegmentPlan> sorted = segments;
    if (orderHint == BitSplitOrder::LSB_TO_MSB) {
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.bitStart < b.bitStart;
        });
    } else {
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.bitStart > b.bitStart;
        });
    }

    SubIntSplitConfigIntegral<TIn> cfg;
    cfg.order = orderHint;
    cfg.bits.reserve(sorted.size());
    cfg.codecs.reserve(sorted.size());

    constexpr int maxBitIndex = static_cast<int>(SubIntSplitConfigIntegral<TIn>::MAX_BITS - 1);
    constexpr int fullWidthEnd = static_cast<int>(SubIntSplitConfigIntegral<TIn>::MAX_BITS);
    int boundary = (orderHint == BitSplitOrder::LSB_TO_MSB) ? 0 : maxBitIndex;

    for (const auto& seg : sorted) {
        if (seg.bitStart < 0 || seg.bitEnd < seg.bitStart || seg.bitEnd > maxBitIndex) {
            throw std::invalid_argument("SubIntSplitConfigIntegral::fromSegments: invalid bit range");
        }

        if (orderHint == BitSplitOrder::LSB_TO_MSB) {
            if (seg.bitStart != boundary) {
                throw std::invalid_argument("SubIntSplitConfigIntegral::fromSegments: segments are not contiguous (LSB->MSB)");
            }
        } else { // MSB_TO_LSB
            if (seg.bitEnd != boundary) {
                throw std::invalid_argument("SubIntSplitConfigIntegral::fromSegments: segments are not contiguous (MSB->LSB)");
            }
        }

        const uint8_t width = static_cast<uint8_t>(seg.bitEnd - seg.bitStart + 1);
        cfg.bits.push_back(width);

        // BWT wrapping: build BWTSectionEncoder(innerCodec) before the section adapter.
        if (seg.reorderer == encodings::encoders::selectors::SubStreamReordererType::BWT512) {
            const uint8_t w = chooseTypeBits(width);  // chooseTypeBits is in encodings::encoders
            // makeSectionCodecForBits is in encodings::encoders (not detail_trisplit)
            if (w <= 8) {
                auto inner = detail_trisplit::makeTypedSectionCodec<uint8_t>(seg.encoding);
                cfg.codecs.push_back(makeSectionCodecForBits<TIn>(
                    std::make_shared<BWTSectionEncoder<uint8_t, 512>>(inner), width));
            } else if (w <= 16) {
                auto inner = detail_trisplit::makeTypedSectionCodec<uint16_t>(seg.encoding);
                cfg.codecs.push_back(makeSectionCodecForBits<TIn>(
                    std::make_shared<BWTSectionEncoder<uint16_t, 512>>(inner), width));
            } else if (w <= 32) {
                auto inner = detail_trisplit::makeTypedSectionCodec<uint32_t>(seg.encoding);
                cfg.codecs.push_back(makeSectionCodecForBits<TIn>(
                    std::make_shared<BWTSectionEncoder<uint32_t, 512>>(inner), width));
            } else {
                auto inner = detail_trisplit::makeTypedSectionCodec<uint64_t>(seg.encoding);
                cfg.codecs.push_back(makeSectionCodecForBits<TIn>(
                    std::make_shared<BWTSectionEncoder<uint64_t, 512>>(inner), width));
            }
            boundary = (orderHint == BitSplitOrder::LSB_TO_MSB)
                           ? (seg.bitEnd + 1)
                           : (seg.bitStart - 1);
            continue;
        }

        // Map EncodingType to concrete section codec.
        switch (seg.encoding) {
            case encodings::EncodingType::RawEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRawSection<TIn>(width));
                break;
            case encodings::EncodingType::DictionaryEncoding:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveDictionarySection<TIn>(width));
                // cfg.codecs.push_back(detail_trisplit::makeDictionarySection<TIn>(width));
                break;
            case encodings::EncodingType::AdaptiveDictionaryEncoding:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveDictionarySection<TIn>(width));
                break;
            case encodings::EncodingType::FrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeFORSection<512, TIn>(width));
                break;
            case encodings::EncodingType::AdaptiveFrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveFORSection<TIn>(width));
                break;
            case encodings::EncodingType::OpenZL:
                cfg.codecs.push_back(detail_trisplit::makeOpenZLSection<0, TIn>(width));
                break;
            case encodings::EncodingType::BitPacking:
                cfg.codecs.push_back(detail_trisplit::makeRawBitPackedSection<TIn>(width));
                break;
            case encodings::EncodingType::AdaptiveFramedBitPrefix:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveFramedBitPrefixSection<TIn>(width));
                break;
            case encodings::EncodingType::RunLengthEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRLESection<TIn>(width));
                break;
            case encodings::EncodingType::HuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeHuffmanSection<TIn>(width));
                break;
            case encodings::EncodingType::LZ4:
                cfg.codecs.push_back(detail_trisplit::makeLZ4Section<TIn>(width));
                break;
            case encodings::EncodingType::FSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeFSESection<TIn>(width));
                break;
            case encodings::EncodingType::FrequencyPartitionEncoding:
                // cfg.codecs.push_back(detail_trisplit::makeBlockFrequencyPartitionSection<TIn>(width));
                cfg.codecs.push_back(detail_trisplit::makeFrequencyPartitionSection<TIn>(
                    width, opts.fpeSectionParallelDecode));
                break;
            case encodings::EncodingType::BlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFrequencyPartitionSection<TIn>(width));
                break;
            case encodings::EncodingType::BlockFORFPEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFORFPESection<TIn>(width));
                break;
            case encodings::EncodingType::BlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFSESection<TIn>(width));
                break;
            case encodings::EncodingType::MainlyConstantEncoding:
                cfg.codecs.push_back(detail_trisplit::makeMainlyConstantSection<TIn>(width));
                break;
            case encodings::EncodingType::RangePackFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRangePackFrequencyPartitionSection<TIn>(width));
                break;
            case encodings::EncodingType::RangePackBlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRangePackBlockFrequencyPartitionSection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORSection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORBlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORBlockFrequencyPartitionSection<TIn>(width));
                break;
            case encodings::EncodingType::RunLengthCascadingFOREncoding:
                cfg.codecs.push_back(detail_trisplit::makeRLECascadingFORSection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORFSESection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORBlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORBlockFSESection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORHuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORHuffmanSection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevFSESection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevBlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevBlockFSESection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevHuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevHuffmanSection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevFrequencyPartitionSection<TIn>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevBlockFrequencyPartitionSection<TIn>(width));
                break;
            default:
                throw std::invalid_argument("SubIntSplitConfigIntegral::fromSegments: unsupported encoding type for section");
        }

        boundary = (orderHint == BitSplitOrder::LSB_TO_MSB)
                       ? (seg.bitEnd + 1)
                       : (seg.bitStart - 1);
    }

    // Ensure full width coverage
    const int expectedEnd = (orderHint == BitSplitOrder::LSB_TO_MSB) ? fullWidthEnd : -1;
    if (boundary != expectedEnd) {
        throw std::invalid_argument("SubIntSplitConfigIntegral::fromSegments: segments do not cover full width");
    }

    cfg.validate();
    return cfg;
}

} // namespace encodings::encoders
