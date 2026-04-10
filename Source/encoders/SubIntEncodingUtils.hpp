#pragma once

#include <algorithm>
#include <cstdint>
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
#include "encoders/DictionaryEncoder.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/OpenZLEncoder.hpp"
#include "encoders/AdaptiveFOREncoder.hpp"
#include "encoders/AdaptiveFramedBitPrefixEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/FOREncoder.hpp"
#include "encoders/BitSplitOrder.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"

namespace encodings::encoders {

// ---------------------------------------------------------------------------
// Section codec type-erased adapter
// ---------------------------------------------------------------------------

template <typename TIn = uint64_t>
    requires (std::is_same_v<TIn, uint64_t> || std::is_same_v<TIn, uint32_t>)
class ISectionCodecIntegral {
public:
    virtual ~ISectionCodecIntegral() = default;

    virtual EncodedBuffer<uint8_t> encode(std::span<const TIn> data) = 0;
    virtual std::vector<TIn> decodeAll(const EncodedBuffer<uint8_t>& enc) = 0;
    virtual std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t idx) = 0;
    virtual std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) = 0;
    virtual EncodingProperties properties() const = 0;
    virtual std::string name() const = 0;
};

// Generic config for sub-integer splits (up to 64 sections, sum(bits)=64)
template <typename TIn = uint64_t>
    requires (std::is_same_v<TIn, uint64_t> || std::is_same_v<TIn, uint32_t>)
struct SubIntSplitConfigIntegral {
    constexpr static size_t MAX_BITS = sizeof(TIn) * 8;
    std::vector<uint8_t> bits;  // per-section bit widths
    BitSplitOrder order{BitSplitOrder::LSB_TO_MSB};
    std::vector<std::shared_ptr<ISectionCodecIntegral<TIn>>> codecs; // parallel to bits

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

    // Build from selector SegmentPlan results (auto-infers order if not provided).
    static SubIntSplitConfigIntegral fromSegments(const std::vector<selectors::SegmentPlan>& segments);
    static SubIntSplitConfigIntegral fromSegments(const std::vector<selectors::SegmentPlan>& segments,
                                            BitSplitOrder orderHint);
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
    }

    EncodedBuffer<uint8_t> encode(std::span<const SectionCodecTIn> data) override {
        std::vector<T> narrowed;
        narrowed.reserve(data.size());
        const uint64_t mask = bits_ == MAX_BITS_SECTION_CODEC
            ? static_cast<uint64_t>(std::numeric_limits<SectionCodecTIn>::max())
            : ((uint64_t{1} << bits_) - 1);
        for (SectionCodecTIn v : data) {
            if ((v & ~mask) != 0) {
                throw std::overflow_error("Section value does not fit in configured bit width");
            }
            if constexpr (std::is_signed_v<T>) {
                constexpr uint64_t signedMax = static_cast<uint64_t>(std::numeric_limits<T>::max());
                if (v > signedMax) {
                    throw std::overflow_error("Section value exceeds signed range for underlying type");
                }
            }
            narrowed.push_back(static_cast<T>(v));
        }
        return impl_->encode(std::span<const T>(narrowed.data(), narrowed.size()));
    }

    std::vector<SectionCodecTIn> decodeAll(const EncodedBuffer<uint8_t>& enc) override {
        auto vals = impl_->decodeAll(enc);
        std::vector<SectionCodecTIn> widened;
        widened.reserve(vals.size());
        for (T v : vals) widened.push_back(static_cast<SectionCodecTIn>(v));
        return widened;
    }

    std::optional<SectionCodecTIn> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t idx) override {
        auto v = impl_->decodeAt(enc, idx);
        if (!v) return std::nullopt;
        return static_cast<SectionCodecTIn>(*v);
    }

    std::vector<SectionCodecTIn> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) override {
        auto vals = impl_->decodeRange(enc, start, end);
        std::vector<SectionCodecTIn> widened;
        widened.reserve(vals.size());
        for (T v : vals) widened.push_back(static_cast<SectionCodecTIn>(v));
        return widened;
    }

    EncodingProperties properties() const override { return impl_->properties(); }
    std::string name() const override { return impl_->name(); }

private:
    std::shared_ptr<Codec<T, uint8_t>> impl_;
    uint8_t bits_{};
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
    // Residual width is chosen internally; bits only gates the narrowing adapter.
    return makeSectionCodecForBits<SectionCodecTIn>(std::make_shared<AdaptiveFOREncoder<int64_t>>(), bits);
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
inline SubIntSplitConfigIntegral<TIn> SubIntSplitConfigIntegral<TIn>::fromSegments(const std::vector<selectors::SegmentPlan>& segments) {
    const BitSplitOrder inferred = SubIntSplitConfigIntegralFactory::inferOrderFromSegments(segments);
    return fromSegments(segments, inferred);
}

template <typename TIn>
    requires (std::is_same_v<TIn, uint64_t> || std::is_same_v<TIn, uint32_t>)
inline SubIntSplitConfigIntegral<TIn> SubIntSplitConfigIntegral<TIn>::fromSegments(
    const std::vector<selectors::SegmentPlan>& segments,
    BitSplitOrder orderHint) {
    if (segments.empty()) {
        throw std::invalid_argument("SubIntSplitConfigIntegral::fromSegments: no segments provided");
    }

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

        // Map EncodingType to concrete section codec.
        switch (seg.encoding) {
            case encodings::EncodingType::RawEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRawSection<TIn>(width));
                break;
            case encodings::EncodingType::DictionaryEncoding:
                cfg.codecs.push_back(detail_trisplit::makeDictionarySection<TIn>(width));
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
