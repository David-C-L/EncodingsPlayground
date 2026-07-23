#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace encodings::exceptions {

// ---------------------------------------------------------------------------
// ExceptionMetricUtils
//
// Shared cost-minimizing bound sweep, reused by every "numeric metric"
// ExceptionDetector (FOR, BitPacking, ...) so bound selection is written
// once instead of duplicated per encoder. See Axis 2 of the design doc
// (~/.claude/plans/could-you-analyse-the-misty-widget.md): only the metric
// (ExceptionDetector::computeMetric) is per-encoding-family code; this sweep
// is the shared piece.
// ---------------------------------------------------------------------------
class ExceptionMetricUtils {
public:
    struct BoundChoice {
        uint32_t bound{0};
        size_t exceptionCount{0};
        double cost{std::numeric_limits<double>::infinity()};
    };

    // Finds the bound b in [0, maxBound] minimizing:
    //
    //   cost(b) = conformingBitsPerValue(b) * numConforming(b)
    //           + exceptionBitsPerValue     * numExceptions(b)
    //           + bitmapOverheadBits
    //
    // where numConforming(b)/numExceptions(b) partition metric.size() values
    // by whether metric[i] <= b, and conformingBitsPerValue(b) == b
    // (approximating a single-global-reference bit-packed/FOR-style inner
    // codec's real per-value cost at that bound -- see
    // FORExceptionDetector.hpp's correspondence with RawBitPackedEncoder).
    //
    // This is an O(numValues + maxBound) histogram sweep over the REAL
    // metric values passed in (computed by the detector from the real full
    // data) -- not an estimate carried over from the DP's sample.
    static BoundChoice sweepMinCostBound(
        const std::vector<uint32_t>& metric,
        uint32_t maxBound,
        double exceptionBitsPerValue,
        double bitmapOverheadBits) {
        BoundChoice best;
        if (metric.empty()) {
            best.bound = 0;
            best.exceptionCount = 0;
            best.cost = bitmapOverheadBits;
            return best;
        }

        // Histogram of metric values in [0, maxBound]; values above maxBound
        // (should not occur when maxBound == the full bit width of T, since
        // no residual can need more bits than the type itself) are clamped
        // into the last bucket defensively.
        std::vector<size_t> countAtExactly(static_cast<size_t>(maxBound) + 1, 0);
        for (uint32_t m : metric) {
            const uint32_t idx = m < maxBound ? m : maxBound;
            ++countAtExactly[idx];
        }

        const size_t numValues = metric.size();
        size_t cumulativeConforming = 0;
        for (uint32_t b = 0; b <= maxBound; ++b) {
            cumulativeConforming += countAtExactly[b];
            const size_t numConforming = cumulativeConforming;
            const size_t numExceptions = numValues - numConforming;
            const double cost = static_cast<double>(b) * static_cast<double>(numConforming)
                               + exceptionBitsPerValue * static_cast<double>(numExceptions)
                               + bitmapOverheadBits;
            if (cost < best.cost) {
                best.cost = cost;
                best.bound = b;
                best.exceptionCount = numExceptions;
            }
        }
        return best;
    }
};

} // namespace encodings::exceptions
