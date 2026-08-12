#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace encodings::benchmark {

/**
 * @brief Order statistics over a set of timing samples (nanoseconds).
 *
 * All helpers take the sample vector by reference and are permitted to reorder
 * it — they use std::nth_element rather than a full sort, so a sequence of
 * calls on the same vector is still O(n) each.
 */

/// Value at the given quantile in [0, 1], by nearest-rank on the sample set.
/// Returns 0 for an empty sample set.
inline int64_t percentileOf(std::vector<int64_t>& samples, double quantile) {
    if (samples.empty()) return 0;
    const double clamped = std::clamp(quantile, 0.0, 1.0);
    size_t idx = static_cast<size_t>(clamped * static_cast<double>(samples.size() - 1));
    idx = std::min(idx, samples.size() - 1);
    auto nth = samples.begin() + static_cast<std::ptrdiff_t>(idx);
    std::nth_element(samples.begin(), nth, samples.end());
    return *nth;
}

inline int64_t medianOf(std::vector<int64_t>& samples) {
    return percentileOf(samples, 0.5);
}

inline int64_t minOf(const std::vector<int64_t>& samples) {
    if (samples.empty()) return 0;
    return *std::min_element(samples.begin(), samples.end());
}

/// Median, p90 and min of one timed point, computed in a single pass of calls.
struct TimingSummary {
    int64_t medianNs{0};
    int64_t p90Ns{0};
    int64_t minNs{0};
};

inline TimingSummary summarize(std::vector<int64_t>& samples) {
    TimingSummary s;
    s.minNs    = minOf(samples);
    s.medianNs = percentileOf(samples, 0.5);
    s.p90Ns    = percentileOf(samples, 0.9);
    return s;
}

}  // namespace encodings::benchmark
