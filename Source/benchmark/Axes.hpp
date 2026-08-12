#pragma once

// Sweep-axis construction.
//
// Extracted from gather_heatmap_benchmark.cpp so every driver spaces its axes
// the same way.  Two drivers sweeping "16 log-spaced spans" must produce the
// same 16 spans, or their results cannot be placed on a common plot.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace encodings::benchmark {

/// `steps` values from `lo` to `hi` inclusive, spaced evenly in log2.
///
/// Collisions after rounding are removed, so the result is strictly increasing
/// and MAY BE SHORTER THAN `steps` — at small `lo` with many steps, several
/// exponents round to the same integer.  Callers that report an axis length
/// should report `result.size()`, not `steps`.
inline std::vector<size_t> logSpaced(size_t lo, size_t hi, size_t steps) {
    std::vector<size_t> out;
    if (steps == 0) return out;
    out.reserve(steps);
    if (steps == 1 || lo >= hi) { out.push_back(lo); return out; }
    const double lgLo = std::log2(static_cast<double>(lo));
    const double lgHi = std::log2(static_cast<double>(hi));
    for (size_t i = 0; i < steps; ++i) {
        const double f = static_cast<double>(i) / static_cast<double>(steps - 1);
        out.push_back(static_cast<size_t>(std::llround(std::exp2(lgLo + f * (lgHi - lgLo)))));
    }
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

/// `steps` values from `lo` to `hi` inclusive, spaced evenly.
///
/// With `steps == 1` the single value is `hi`, not `lo`: a one-step axis means
/// "just the endpoint" (e.g. selectivity 1.0, the contiguous baseline), which is
/// the degenerate case callers actually want.
inline std::vector<double> linSpaced(double lo, double hi, size_t steps) {
    std::vector<double> out;
    if (steps == 0) return out;
    out.reserve(steps);
    if (steps == 1) { out.push_back(hi); return out; }
    for (size_t i = 0; i < steps; ++i) {
        const double f = static_cast<double>(i) / static_cast<double>(steps - 1);
        out.push_back(lo + f * (hi - lo));
    }
    return out;
}

} // namespace encodings::benchmark
