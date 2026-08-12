#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/SelectiveTraceGen.hpp"
#include "encodings/RowRange.hpp"

namespace encodings::benchmark {

/**
 * @brief Gap placement model for a gather access.
 *
 * UniformDeterministic — fixed-period alternation of equal-length runs and
 *   equal-length gaps, no randomness at all.  Fully reproducible, and the
 *   cleanest way to isolate the effect of selectivity from the effect of run
 *   placement.  Its weakness is that the period (run + gap) is constant, so it
 *   can resonate with a codec's internal block structure (block sizes, rank
 *   sample strides) and produce banding in a sweep that is an artifact of the
 *   trace rather than a property of the encoding.  Cross-check any banding
 *   against Geometric before believing it.
 *
 * Geometric — geometrically distributed run and gap lengths, delegating to
 *   makeSelectiveTrace().  Maximum entropy for a given mean run length, so it
 *   has no fixed period to alias with, and it is the same model the
 *   BenchmarkRunner selective-access phase uses, which makes numbers from this
 *   driver directly comparable with the main benchmark suite's output.
 */
enum class GapModel { UniformDeterministic, Geometric };

inline std::string gapModelName(GapModel m) {
    return m == GapModel::Geometric ? "geometric" : "uniform";
}

/**
 * @brief One point in the gather access space.
 *
 * The access always spans exactly [start, start + span); selectivity controls
 * how densely that span is populated, never how wide it is.  This holds even
 * at runCount == 1, where the trace is a single run of floor(sigma*span) at
 * `start` followed by a trailing gap — deliberately, so that a fixed span means
 * the same thing at every selectivity and the cells of a sweep stay comparable.
 */
struct GatherAccessParams {
    size_t   start{0};        ///< s0, absolute element index into the stream
    size_t   span{0};         ///< l, window width in elements
    double   selectivity{1.0};///< sigma in (0, 1]: fraction of the span that is read
    size_t   runLength{8};    ///< target length of each contiguous run, in elements
    GapModel gapModel{GapModel::UniformDeterministic};
    uint64_t seed{42};        ///< only consulted by GapModel::Geometric
    size_t   maxRanges{0};    ///< 0 = unbounded; otherwise a hard cap on the range count
};

/**
 * @brief A materialized gather trace plus the structural facts about it.
 *
 * Every quantity here is measured from the trace that was actually built, not
 * from the parameters that were requested.  The two differ because each step of
 * the construction floors, and the drift is largest at small spans and low
 * selectivities — so plots must use `selectivityAchieved` and `runLengthActual`
 * rather than the nominal request, which only identifies the sweep cell.
 */
struct GatherTrace {
    RowRangeList ranges;
    size_t selectedRows{0};        ///< sum of range sizes
    size_t rangeCountNominal{0};   ///< k as implied by (span, sigma, runLength), before capping/merging
    size_t rangeCount{0};          ///< ranges.size(): after maxRanges capping and adjacency merging
    size_t runLengthActual{0};     ///< length of each run as placed (uniform model); mean run length (geometric)
    size_t gapLength{0};           ///< gap between runs as placed (uniform model); mean gap (geometric)
    double selectivityAchieved{0.0};  ///< selectedRows / span
    bool   clamped{false};         ///< maxRanges bound was hit: the uniform model coarsened to
                                   ///< longer runs at the same selectivity, while the geometric
                                   ///< model stopped early and left the tail of the span uncovered

    /**
     * @brief Expand `ranges` into the flat ascending row list some APIs require.
     *
     * Ranges are the right representation for *building* a trace and for
     * reporting its structure (rangeCount, runLengthActual), but not every
     * consumer takes them: nimble's read path expresses a selective read as
     * RowSet = folly::Range<const vector_size_t*>, a sorted array of row
     * numbers.  Keeping this conversion here means a driver body can serve both
     * representations without the trace generator itself knowing about either.
     *
     * `out` is cleared and filled with exactly `selectedRows` ascending indices.
     * Call it OUTSIDE any timed region — it allocates and costs one pass.
     */
    void expandToRows(std::vector<int32_t>& out) const {
        out.clear();
        out.reserve(selectedRows);
        for (const auto& r : ranges)
            for (size_t i = r.begin; i < r.end; ++i)
                out.push_back(static_cast<int32_t>(i));
    }
};

/// Range count implied by a (span, selectivity, runLength) triple, before any
/// maxRanges capping. Always at least 1 for a non-empty span.
inline size_t impliedRangeCount(size_t span, double selectivity, size_t runLength) {
    if (span == 0) return 0;
    const size_t rl       = std::max<size_t>(1, runLength);
    const size_t selected = static_cast<size_t>(
        std::floor(std::clamp(selectivity, 0.0, 1.0) * static_cast<double>(span)));
    return std::max<size_t>(1, selected / rl);
}

namespace detail {

/// Collapse adjacent or overlapping ranges in an ascending list. Only fires as
/// the gap length reaches zero, i.e. as selectivity approaches 1.
inline void mergeAdjacent(RowRangeList& ranges) {
    if (ranges.size() < 2) return;
    RowRangeList merged;
    merged.reserve(ranges.size());
    merged.push_back(ranges.front());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].begin <= merged.back().end) {
            merged.back().end = std::max(merged.back().end, ranges[i].end);
        } else {
            merged.push_back(ranges[i]);
        }
    }
    ranges.swap(merged);
}

inline void finalize(GatherTrace& t, size_t span) {
    detail::mergeAdjacent(t.ranges);
    t.rangeCount   = t.ranges.size();
    t.selectedRows = 0;
    for (const auto& r : t.ranges) t.selectedRows += r.size();
    t.selectivityAchieved = span > 0
        ? static_cast<double>(t.selectedRows) / static_cast<double>(span)
        : 0.0;
}

}  // namespace detail

/**
 * @brief Build the gather trace for one access, deterministic in (streamLength, params).
 *
 * Returns an ascending, non-overlapping RowRangeList contained in
 * [start, start + span), clipped to the stream.
 */
inline GatherTrace buildGatherTrace(size_t streamLength, const GatherAccessParams& p) {
    GatherTrace t;
    if (streamLength == 0 || p.start >= streamLength) return t;

    const size_t start = p.start;
    const size_t span  = std::min(p.span, streamLength - start);
    if (span == 0) return t;

    const double sigma = std::clamp(p.selectivity, 0.0, 1.0);
    if (sigma <= 0.0) return t;

    const size_t runLength = std::max<size_t>(1, p.runLength);

    if (p.gapModel == GapModel::Geometric) {
        SelectiveTraceParams sp;
        sp.selectivity   = sigma;
        sp.meanRunLength = static_cast<double>(runLength);
        sp.seed          = p.seed;
        sp.maxRanges     = p.maxRanges;
        t.ranges = makeSelectiveTrace(span, sp);
        for (auto& r : t.ranges) {
            r.begin += start;
            r.end   += start;
        }
        t.rangeCountNominal = impliedRangeCount(span, sigma, runLength);
        t.runLengthActual   = runLength;
        t.gapLength = static_cast<size_t>(
            std::llround(static_cast<double>(runLength) * (1.0 / std::max(sigma, 1e-9) - 1.0)));
        detail::finalize(t, span);
        // The generator stops once it has emitted maxRanges runs, leaving the
        // tail of the span unread — so a clamp here really does lose coverage.
        t.clamped = p.maxRanges != 0 && t.rangeCount >= p.maxRanges;
        return t;
    }

    // ── Uniform deterministic model ──────────────────────────────────────────
    const size_t selectedTarget = static_cast<size_t>(
        std::floor(sigma * static_cast<double>(span)));
    if (selectedTarget == 0) return t;

    size_t k = std::max<size_t>(1, selectedTarget / runLength);
    t.rangeCountNominal = k;
    if (p.maxRanges != 0 && k > p.maxRanges) {
        k = p.maxRanges;
        t.clamped = true;  // coarsens to longer runs; selectivity and span are preserved
    }

    // Place the runs by exact proportional partition rather than by a single
    // floored (range_len, gap_len) stride.  A fixed stride floors k times over,
    // so it systematically under-delivers selectivity — at sigma = 1 it would
    // leave the span short of full coverage and quietly break the claim that the
    // sigma = 1 slice *is* the contiguous range-access baseline.  Partitioning
    // instead spends the rounding remainder across the runs, which pins the
    // selected total at exactly floor(sigma*span) and the covered span at
    // exactly `span`, at the cost of run lengths varying by at most one element.
    const size_t gapTotal = span - selectedTarget;
    auto selBefore = [&](size_t i) { return selectedTarget * i / k; };
    auto gapBefore = [&](size_t i) { return k > 1 ? gapTotal * i / (k - 1) : size_t{0}; };

    t.ranges.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const size_t begin = start + selBefore(i)     + gapBefore(i);
        const size_t end   = start + selBefore(i + 1) + gapBefore(i);
        if (end > begin) t.ranges.push_back({begin, end});
    }

    detail::finalize(t, span);
    // Representative structural lengths; individual runs may differ by one.
    t.runLengthActual = t.rangeCount > 0 ? t.selectedRows / t.rangeCount : 0;
    t.gapLength       = k > 1 ? gapTotal / (k - 1) : 0;
    // A cap that merged away is not a cap: once the runs are adjacent (sigma = 1)
    // the trace is the canonical contiguous range whatever k was, so reporting it
    // as clamped would mislabel the baseline slice.
    if (t.rangeCount == 1 && t.selectedRows == selectedTarget) t.clamped = false;
    return t;
}

}  // namespace encodings::benchmark
