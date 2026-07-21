#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "encoders/CascadingFOREncoder.hpp"
#include "encoders/selectors/MetricCollector.hpp"

namespace encodings::encoders::analysis {

// =============================================================================
//  analyzeCascade
//
//  Pure analysis utility, deliberately decoupled from CascadingFOREncoder's
//  encode()/decode() hot path: it replays the identical recursive frame
//  partitioning CascadingFOREncoder uses (via the shared
//  computeCascadeFrameReference() helper) but only computes statistics — it
//  never serializes, so cardinality sweeps over many (depth, frame-size)
//  combinations don't pay any encode/decode cost.
//
//  One CascadeLevelStats is emitted per (role, level) — for Reference
//  entries, `parentResidualLevel` records which residual level's own
//  reference array this came from (see CascadingFOREncoder.hpp's
//  "Breaking the regress" note: a reference cascade's own internal reference
//  arrays are not further analyzed/cascaded, mirroring the encoder).
// =============================================================================

struct CascadeLevelStats {
    CascadeStreamRole role{CascadeStreamRole::Residual};
    size_t levelIndex{0};
    std::optional<size_t> parentResidualLevel; // set only for Reference entries

    size_t frameSize{0};
    size_t numFrames{0};
    size_t numElements{0};

    size_t globalExactDistinct{0};
    bool   globalDistinctCapped{false};
    double globalHllEstimate{0.0};

    size_t intraFrameDistinctMin{0};
    size_t intraFrameDistinctMax{0};
    double intraFrameDistinctMean{0.0};
    double intraFrameDistinctMedian{0.0};
    std::vector<uint32_t> intraFrameDistinctPerFrame; // full detail, for offline plotting

    uint8_t maxBitWidth{0};
    double  avgBitWidth{0.0};
};

namespace detail {

// std::hash<int64_t> is the identity function under libstdc++ (and several
// other standard libraries), so MetricCollector's HLL estimator — which
// hashes each value via std::hash<T> — degenerates badly for small-magnitude
// integers: exactly the common case for FOR residuals/references. Mixing
// with this bijective 64-bit avalanche step (SplitMix64's finalizer) before
// handing values to MetricCollector fixes HLL accuracy while leaving exact
// distinct/frequency counts unchanged, since a bijection preserves distinctness.
inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

inline CascadeLevelStats collectLevelStats(std::span<const int64_t> data, size_t frameSize,
                                            CascadeStreamRole role, size_t levelIndex,
                                            std::optional<size_t> parentResidualLevel) {
    const size_t N = data.size();
    const size_t numFrames = (N + frameSize - 1) / frameSize;

    CascadeLevelStats stats;
    stats.role = role;
    stats.levelIndex = levelIndex;
    stats.parentResidualLevel = parentResidualLevel;
    stats.frameSize = frameSize;
    stats.numFrames = numFrames;
    stats.numElements = N;

    // Global cardinality: reuse MetricCollector's single-pass exact + HLL computation.
    // Values are pre-mixed (see mix64() above) so the HLL estimate is reliable
    // even for small-magnitude residual/reference streams.
    std::vector<int64_t> asVec(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        asVec[i] = static_cast<int64_t>(mix64(static_cast<uint64_t>(data[i])));
    }
    selectors::MetricCollector<int64_t> collector;
    const auto metrics = collector.compute(asVec, static_cast<selectors::MetricFlags>(selectors::MetricFlag::FreqStats));
    stats.globalExactDistinct  = metrics.uniqueCount;
    stats.globalDistinctCapped = metrics.uniqueCountCapped;
    stats.globalHllEstimate    = metrics.hllEstimatedCardinality;

    // Intra-frame distinct counts + bit-width span — no existing MetricCollector
    // equivalent since its per-frame stats are tied to a fixed compile-time
    // candidate array, not an arbitrary runtime frame size (see CascadingFOREncoder.hpp).
    stats.intraFrameDistinctPerFrame.resize(numFrames);
    ankerl::unordered_dense::set<int64_t> seen;
    seen.reserve(std::min(frameSize, N));

    uint8_t maxBits = 0;
    double  bitSum  = 0.0;

    for (size_t f = 0; f < numFrames; ++f) {
        const size_t lo = f * frameSize;
        const size_t hi = std::min(lo + frameSize, N);

        seen.clear();
        int64_t frameMin = data[lo];
        int64_t frameMax = data[lo];
        for (size_t i = lo; i < hi; ++i) {
            seen.insert(data[i]);
            frameMin = std::min(frameMin, data[i]);
            frameMax = std::max(frameMax, data[i]);
        }
        stats.intraFrameDistinctPerFrame[f] = static_cast<uint32_t>(seen.size());

        const uint64_t span = static_cast<uint64_t>(frameMax) - static_cast<uint64_t>(frameMin);
        const uint8_t bits = span == 0 ? 0 : static_cast<uint8_t>(64 - std::countl_zero(span));
        maxBits = std::max(maxBits, bits);
        bitSum += bits;
    }

    stats.maxBitWidth = maxBits;
    stats.avgBitWidth = numFrames > 0 ? bitSum / static_cast<double>(numFrames) : 0.0;

    if (!stats.intraFrameDistinctPerFrame.empty()) {
        const auto& v = stats.intraFrameDistinctPerFrame;
        stats.intraFrameDistinctMin = *std::min_element(v.begin(), v.end());
        stats.intraFrameDistinctMax = *std::max_element(v.begin(), v.end());
        const double sum = std::accumulate(v.begin(), v.end(), 0.0);
        stats.intraFrameDistinctMean = sum / static_cast<double>(v.size());

        std::vector<uint32_t> sorted(v.begin(), v.end());
        std::sort(sorted.begin(), sorted.end());
        const size_t mid = sorted.size() / 2;
        stats.intraFrameDistinctMedian = (sorted.size() % 2 == 0)
            ? (static_cast<double>(sorted[mid - 1]) + static_cast<double>(sorted[mid])) / 2.0
            : static_cast<double>(sorted[mid]);
    }

    return stats;
}

// Forward declarations: residual-role and reference-role recursion call each other.
inline void analyzeReferenceLevel(std::span<const int64_t> data, const CascadingFORConfig& cfg,
                                   size_t level, size_t parentResidualLevel,
                                   std::vector<CascadeLevelStats>& out);

inline void analyzeResidualLevel(std::span<const int64_t> data, const CascadingFORConfig& cfg,
                                  size_t level, std::vector<CascadeLevelStats>& out) {
    if (level == cfg.residualSchedule.size()) return; // leaf: no further structure to report

    const size_t frameSize = cfg.residualSchedule[level].frameSize;
    const FORReferencePolicy policy = cfg.residualSchedule[level].policy;

    const size_t N = data.size();
    const size_t numFrames = (N + frameSize - 1) / frameSize;
    std::vector<int64_t> refs(numFrames);
    std::vector<int64_t> residuals(N);
    for (size_t f = 0; f < numFrames; ++f) {
        const size_t lo = f * frameSize;
        const size_t hi = std::min(lo + frameSize, N);
        const int64_t ref = computeCascadeFrameReference(data, lo, hi, policy);
        refs[f] = ref;
        for (size_t i = lo; i < hi; ++i) residuals[i] = data[i] - ref;
    }

    // Stats are collected on `residuals` (post-subtraction), not `data`
    // (pre-subtraction): intra-frame distinct counts are shift-invariant so
    // this doesn't change them, but global cardinality is NOT shift-invariant
    // across windows using different per-window constants, so this is the
    // difference between reporting the array this level actually hands
    // onward (correct) vs. an almost-raw predecessor array (wrong — see
    // CascadingFOREncoder.hpp's telescoping-identity note for why this
    // matters most at the deepest level, which is exactly what
    // computeDeepestResiduals()/the leaf encoder receive).
    out.push_back(collectLevelStats(residuals, frameSize, CascadeStreamRole::Residual, level, std::nullopt));

    analyzeReferenceLevel(std::span<const int64_t>(refs), cfg, 0, level, out);
    analyzeResidualLevel(std::span<const int64_t>(residuals), cfg, level + 1, out);
}

inline void analyzeReferenceLevel(std::span<const int64_t> data, const CascadingFORConfig& cfg,
                                   size_t level, size_t parentResidualLevel,
                                   std::vector<CascadeLevelStats>& out) {
    if (level == cfg.referenceSchedule.size()) return; // leaf: stored directly, nothing further to report

    const size_t frameSize = cfg.referenceSchedule[level].frameSize;
    const FORReferencePolicy policy = cfg.referenceSchedule[level].policy;

    const size_t N = data.size();
    const size_t numFrames = (N + frameSize - 1) / frameSize;
    std::vector<int64_t> residuals(N);
    for (size_t f = 0; f < numFrames; ++f) {
        const size_t lo = f * frameSize;
        const size_t hi = std::min(lo + frameSize, N);
        const int64_t ref = computeCascadeFrameReference(data, lo, hi, policy);
        for (size_t i = lo; i < hi; ++i) residuals[i] = data[i] - ref;
    }

    // See analyzeResidualLevel: stats reflect the post-subtraction array.
    out.push_back(collectLevelStats(residuals, frameSize, CascadeStreamRole::Reference, level, parentResidualLevel));

    // This level's own reference array is stored directly by the encoder (not
    // cascaded again — see CascadingFOREncoder.hpp), so there is no ref-of-ref
    // structure to analyze here; only the residual-style continuation recurses.
    analyzeReferenceLevel(std::span<const int64_t>(residuals), cfg, level + 1, parentResidualLevel, out);
}

} // namespace detail

inline std::vector<CascadeLevelStats> analyzeCascade(std::span<const int64_t> data,
                                                      const CascadingFORConfig& cfg) {
    std::vector<CascadeLevelStats> out;
    if (data.empty()) return out;
    std::vector<int64_t> wide(data.begin(), data.end());
    detail::analyzeResidualLevel(std::span<const int64_t>(wide), cfg, 0, out);
    return out;
}

} // namespace encodings::encoders::analysis
