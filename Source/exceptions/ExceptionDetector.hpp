#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace encodings::exceptions {

// ---------------------------------------------------------------------------
// ExceptionDetector<T>
//
// Defines only the per-encoding-family "metric": a per-value number
// characterizing how well each value fits the pattern the paired inner
// codec exploits (e.g. residual bits needed relative to a computed
// reference, for FrameOfReference/BitPacking-style codecs). Computed from
// the real, full segment data at encode time -- never from a sample.
//
// Every existing per-segment encoder in this codebase (FOREncoder,
// AdaptiveFOREncoder, RawBitPackedEncoder, AdaptiveFramedBitPrefixEncoder,
// MainlyConstantEncoder, RunLengthEncoder) recomputes its packing
// parameters from real full data at encode() time, never from the DP's
// sample -- see "Resolved: is the exception threshold sample-derived or
// exact?" in the exception-handling design doc
// (~/.claude/plans/could-you-analyse-the-misty-widget.md). Detectors follow
// the same rule.
//
// Bound selection (which metric values become exceptions) is deliberately
// NOT part of this interface -- it is a single shared sweep
// (ExceptionMetricUtils::sweepMinCostBound) reused by every numeric-metric
// detector, so adding a new pattern only requires writing computeMetric(),
// not a whole bespoke per-encoder "choosePlan"-style search.
// ---------------------------------------------------------------------------
template <typename T>
class ExceptionDetector {
public:
    virtual ~ExceptionDetector() = default;

    // Per-value metric computed from the real, full segment data. Higher
    // values mean "less conforming" (e.g. more bits needed to represent this
    // value under the pattern this detector characterizes). Must return
    // exactly data.size() values.
    virtual std::vector<uint32_t> computeMetric(std::span<const T> data) const = 0;

    virtual std::string name() const = 0;
};

} // namespace encodings::exceptions
