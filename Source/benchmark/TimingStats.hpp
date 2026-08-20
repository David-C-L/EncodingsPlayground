#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
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

/**
 * @brief Moment statistics over a real-valued sample set.
 *
 * Deliberately separate from TimingSummary: timings are summarized by order
 * statistics (median/p90/min), which are robust to the occasional descheduled
 * iteration, whereas derived quantities like compression ratios and throughputs
 * have no such outliers and are reported as mean +/- stddev.  Using one for the
 * other would either throw away the spread or report a mean that one preempted
 * iteration can dominate.
 *
 * stddev is the *sample* standard deviation (Bessel-corrected, n-1), and is 0
 * for a single sample.
 */
struct MomentSummary {
    double mean{0.0};
    double stddev{0.0};
    double min{0.0};
    double max{0.0};

    static MomentSummary compute(const std::vector<double>& values) {
        MomentSummary s;
        if (values.empty()) return s;

        s.mean = std::accumulate(values.begin(), values.end(), 0.0)
               / static_cast<double>(values.size());
        s.min = *std::min_element(values.begin(), values.end());
        s.max = *std::max_element(values.begin(), values.end());

        if (values.size() > 1) {
            double sqSum = 0.0;
            for (double v : values) sqSum += (v - s.mean) * (v - s.mean);
            s.stddev = std::sqrt(sqSum / static_cast<double>(values.size() - 1));
        }
        return s;
    }
};

}  // namespace encodings::benchmark
