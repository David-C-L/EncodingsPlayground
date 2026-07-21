#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"

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

    // Decode directly into a caller-supplied buffer, eliminating the intermediate
    // vector allocation that decodeAll / decodeRange would otherwise return.
    // Implementations must write exactly n elements; a size mismatch is a hard error.
    virtual void decodeAllInto(const EncodedBuffer<uint8_t>& enc, TIn* dst, size_t n) = 0;
    virtual void decodeRangeInto(const EncodedBuffer<uint8_t>& enc,
                                  size_t start, size_t end,
                                  TIn* dst, size_t n) = 0;

    // Fused decode-and-accumulate: decode and OR-shift directly into acc, avoiding
    // a separate wide tmp buffer.  isFirst=true: pure write (acc[i] = v << shift);
    // isFirst=false: read-modify-write (acc[i] |= v << shift).
    // Default implementation allocates a temporary via decodeAllInto; override in
    // TypedSectionCodecAdapter (and any other concrete codec) for the hot path.
    virtual void decodeAllAndAccumulate(const EncodedBuffer<uint8_t>& enc,
                                        TIn* acc, size_t n,
                                        uint8_t shift, bool isFirst) {
        std::vector<TIn> tmp(n);
        decodeAllInto(enc, tmp.data(), n);
        if (isFirst) {
            for (size_t i = 0; i < n; ++i) acc[i] = tmp[i] << shift;
        } else {
            for (size_t i = 0; i < n; ++i) acc[i] |= tmp[i] << shift;
        }
    }

    virtual void decodeRangeAndAccumulate(const EncodedBuffer<uint8_t>& enc,
                                          size_t start, size_t end,
                                          TIn* acc, size_t n,
                                          uint8_t shift, bool isFirst) {
        std::vector<TIn> tmp(n);
        decodeRangeInto(enc, start, end, tmp.data(), n);
        if (isFirst) {
            for (size_t i = 0; i < n; ++i) acc[i] = tmp[i] << shift;
        } else {
            for (size_t i = 0; i < n; ++i) acc[i] |= tmp[i] << shift;
        }
    }

    virtual EncodingProperties properties() const = 0;
    virtual std::string name() const = 0;
};

} // namespace encodings::encoders
