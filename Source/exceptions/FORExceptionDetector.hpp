#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "exceptions/ExceptionDetector.hpp"

namespace encodings::exceptions {

// ---------------------------------------------------------------------------
// FORExceptionDetector<T>
//
// Metric for a single-global-reference, frame-of-reference-style pattern:
//   reference   = min(data)
//   metric[i]   = bits needed to represent (data[i] - reference)
//
// This is an exact match for RawBitPackedEncoder<T>'s own encode-time
// computation (base = minVal, bitWidth = bit_width(range) -- see
// RawBitPackedEncoder.hpp), so pairing ExceptionEncoder<T> with a
// RawBitPackedEncoder<T> inner codec gives an exact (not approximate)
// correspondence between this detector's metric and the inner codec's real
// per-value cost at any candidate bound: once exceptions are pulled out,
// RawBitPackedEncoder computes its own (tighter) min/bitWidth purely from
// the compacted conforming subset -- no coordination between detector and
// inner codec is required beyond "which indices are exceptions".
//
// Named "FOR" (matching the exception-handling design doc's Axis 3) since
// "reference + narrow residual width" is the general frame-of-reference
// idea. This deliberately does not replicate FOREncoder<T>'s per-frame
// referencing (see FOREncoder.hpp) -- that framed variant is a candidate
// for a future dedicated detector, not this one.
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_integral_v<T>
class FORExceptionDetector : public ExceptionDetector<T> {
public:
    std::vector<uint32_t> computeMetric(std::span<const T> data) const override {
        std::vector<uint32_t> metric(data.size());
        if (data.empty()) {
            return metric;
        }
        using U = std::make_unsigned_t<T>;
        const T reference = *std::min_element(data.begin(), data.end());
        const U refU = static_cast<U>(reference);
        for (size_t i = 0; i < data.size(); ++i) {
            const U residual = static_cast<U>(static_cast<U>(data[i]) - refU);
            metric[i] = residual == 0 ? 0u : static_cast<uint32_t>(std::bit_width(residual));
        }
        return metric;
    }

    std::string name() const override { return "FORExceptionDetector"; }
};

} // namespace encodings::exceptions
