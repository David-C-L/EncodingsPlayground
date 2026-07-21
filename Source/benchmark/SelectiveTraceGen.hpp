#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

#include "encodings/RowRange.hpp"

namespace encodings::benchmark {

/**
 * @brief Parameters for generating a TableScan-style selective/filtered row-range
 *        trace, with two independent knobs:
 *
 *  - selectivity: overall fraction of rows that survive, in (0, 1].
 *  - meanRunLength: average length of a surviving contiguous run before the
 *    next gap, independent of selectivity — this is the "clumpiness" knob
 *    that determines how much skip latency matters at a given selectivity
 *    (e.g. selectivity=0.5 with meanRunLength=1 is maximally scattered, worst
 *    case for skip cost; selectivity=0.5 with meanRunLength=10000 is a few
 *    huge blocks, best case for a contiguous-span decode).
 */
struct SelectiveTraceParams {
    double selectivity{0.5};
    double meanRunLength{8.0};
    uint64_t seed{42};
    size_t maxRanges{0};  // 0 = unbounded (cover full n)
};

/**
 * @brief Generates an ascending, non-overlapping RowRangeList over [0, n)
 *        using geometrically-distributed run/gap lengths parameterized by
 *        SelectiveTraceParams. Deterministic for a given (n, params).
 *
 * Alternates kept ("run") and skipped ("gap") spans so that, in expectation,
 * E[run]/(E[run]+E[gap]) == selectivity and E[run] == meanRunLength, which
 * pins down E[gap] = meanRunLength * (1/selectivity - 1).
 */
inline RowRangeList makeSelectiveTrace(size_t n, const SelectiveTraceParams& p) {
    RowRangeList ranges;
    if (n == 0 || p.selectivity <= 0.0) return ranges;
    if (p.selectivity >= 1.0) {
        ranges.push_back({0, n});
        return ranges;
    }

    const double meanGapLength = p.meanRunLength * (1.0 / p.selectivity - 1.0);
    std::mt19937_64 rng(p.seed);
    std::geometric_distribution<size_t> runDist(1.0 / std::max(1.0, p.meanRunLength));
    std::geometric_distribution<size_t> gapDist(1.0 / std::max(1.0, meanGapLength));

    size_t pos = 0;
    while (pos < n) {
        size_t runLen = std::min(runDist(rng) + 1, n - pos);  // +1: geometric can return 0
        ranges.push_back({pos, pos + runLen});
        pos += runLen;
        if (pos >= n) break;
        if (p.maxRanges && ranges.size() >= p.maxRanges) break;

        size_t gapLen = std::min(gapDist(rng) + 1, n - pos);
        pos += gapLen;
    }
    return ranges;
}

}  // namespace encodings::benchmark
