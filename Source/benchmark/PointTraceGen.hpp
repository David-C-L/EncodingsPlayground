#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "benchmark/SelectiveTraceGen.hpp"
#include "encodings/RowRange.hpp"

namespace encodings::benchmark {

/**
 * @brief Index distribution for a point-lookup (decodeAt) access.
 *
 * Uniform — a shuffled permutation of [0, streamLength), truncated to `probes`.
 *   This is exactly what BenchmarkRunner's random-access phase does, and it is
 *   spelled the same way here on purpose: the point driver's numbers have to
 *   remain comparable with the existing suite's `averageRandomAccessTime`, and a
 *   different notion of "random" (say, sampling with replacement) would shift
 *   the distinct-index count and therefore the cache behaviour without changing
 *   anything about the codec.
 *
 * Zipf — the only genuinely new pattern here, and the one worth explaining.
 *   Under Uniform over a multi-million-element stream every probe is a cache and
 *   TLB miss, so a point lookup measures memory latency with a little decode
 *   attached; codecs whose per-lookup work differs by a factor of several come
 *   out indistinguishable.  A skewed distribution keeps the hot set small enough
 *   that the codec's own auxiliary structures — dictionary, tier tags, rank
 *   samples — stay resident, which is what isolates index-lookup *work* from raw
 *   memory latency.  That separation is the essential control when comparing
 *   FrequencyPartition's TierTagArray (one indirection into a large array, cheap
 *   per lookup but only while the array is cached) against EliasFano (strictly
 *   more instructions per lookup, over a footprint small enough to stay cached
 *   at any stream length).  Ranked at both ends of the skew, those two cross;
 *   measured at Uniform alone, they merely look equal.
 *
 * GeometricClustered — probes drawn from the surviving rows of a selective
 *   trace, i.e. the *same* geometric run/gap model the gather driver and the
 *   BenchmarkRunner selective phase use (see below).  Answers "what if the point
 *   lookups follow a filter's clustering rather than the whole stream".
 *
 * Strided — fixed-stride walk, no randomness.  Reproducible and prefetcher-
 *   friendly, and the pattern real strided scans produce.  Carries the same
 *   aliasing caveat as GatherTraceGen's GapModel::UniformDeterministic: a stride
 *   that resonates with a codec's block size, tier width or rank-sample stride
 *   concentrates every probe into the same phase of that structure and produces
 *   banding across a sweep which is a property of the trace, not of the
 *   encoding.  Any banding seen under Strided must be cross-checked against
 *   Uniform before it is believed.
 *
 * SequentialProbe — 0, 1, 2, ... through the point API.  Not a realistic access
 *   pattern; it is the per-call overhead floor.  It bounds how much of a point
 *   lookup is the lookup and how much is the call plus bounds checks, which is
 *   the number needed to interpret every other pattern in this enum.
 */
enum class PointPattern { Uniform, Zipf, GeometricClustered, Strided, SequentialProbe };

inline const char* pointPatternName(PointPattern p) {
    switch (p) {
        case PointPattern::Uniform:            return "uniform";
        case PointPattern::Zipf:               return "zipf";
        case PointPattern::GeometricClustered: return "geometric_clustered";
        case PointPattern::Strided:            return "strided";
        case PointPattern::SequentialProbe:    return "sequential";
    }
    return "unknown";
}

/**
 * @brief One point in the point-access space.
 *
 * `probes` is the batch size, not a property of the stream: point lookups run in
 * tens of nanoseconds, so a driver must time a batch and divide (CONVENTIONS
 * section 3).  Every other field is consulted by a subset of the patterns only,
 * and the unused ones are deliberately left at their defaults rather than
 * validated — a sweep that varies `stride` across Zipf cells is describing one
 * trace, and the label() string is what records which knobs actually mattered.
 */
struct PointTraceParams {
    size_t       streamLength{};
    size_t       probes{1u << 16};
    PointPattern pattern{PointPattern::Uniform};
    double       zipfTheta{1.0};   ///< 0 = uniform, ~1 = classic Zipf, > 1 = sharply hot
    size_t       meanRunLength{8}; ///< GeometricClustered only
    double       selectivity{0.1}; ///< GeometricClustered only: fixes the mean gap length
    size_t       stride{97};       ///< Strided only; prime by default, see PointPattern
    uint64_t     seed{42};
    bool         ascending{false}; ///< sort the probe list; see PointTrace
};

/**
 * @brief A materialized probe list plus the structural facts about it.
 *
 * As in GatherTrace, every quantity here is MEASURED from the list that was
 * actually built rather than derived from the request.  For point traces the
 * drift is not a rounding artifact but the whole point: Zipf with theta = 1.2
 * over ten million rows may touch a few thousand distinct indices, and Strided
 * over a stream length sharing a factor with the stride revisits a short cycle.
 * Neither is visible in the parameters, both dominate the result, so the driver
 * reports these fields per row and plots against them.
 *
 * `distinctFraction` (= distinctIndices / probes) is the diagnostic that tells a
 * reader whether a row measures the codec or the cache.  Near 1.0 nearly every
 * probe is a fresh line and the number is dominated by memory latency; near 0
 * the hot set is resident and the number is dominated by the codec's per-lookup
 * work.  A comparison between two codecs is only a comparison of their lookup
 * paths at the low end, and the fraction is what proves the reader is there.
 *
 * `footprintSpan` (max - min + 1) separates a small hot set that is *clustered*
 * from one that is merely small: Zipf and GeometricClustered can agree on
 * distinctIndices while spanning six orders of magnitude of address range apart,
 * and TLB behaviour follows the span, not the count.
 */
struct PointTrace {
    std::vector<size_t> indices;
    size_t distinctIndices{};
    size_t footprintSpan{};
    double distinctFraction{};

    std::string label() const;
};

namespace detail {

/**
 * @brief A seeded bijection on [0, 2^bits), used to scramble Zipf ranks.
 *
 * Composed of operations that are each individually invertible on a fixed width
 * — multiply by an odd constant, and x ^= x >> s — so the composition is a
 * permutation, never a hash with collisions.  That property is what the Zipf
 * pattern needs: rank r must map to a distinct index, or two ranks would
 * collapse into one probe and the realized distribution would no longer be the
 * requested one.
 */
struct PowerOfTwoBijection {
    uint64_t mask{};
    unsigned bits{};
    uint64_t oddA{}, oddB{};

    uint64_t apply(uint64_t x) const {
        const unsigned s = std::max(1u, bits / 2);
        x = (x * oddA) & mask;
        x ^= x >> s;
        x = (x * oddB) & mask;
        x ^= x >> s;
        return x & mask;
    }
};

inline PowerOfTwoBijection makeBijection(size_t n, uint64_t seed) {
    PowerOfTwoBijection b;
    b.bits = 1;
    while (b.bits < 64 && (uint64_t{1} << b.bits) < static_cast<uint64_t>(n)) ++b.bits;
    b.mask = b.bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << b.bits) - 1;
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);
    b.oddA = rng() | 1ull;
    b.oddB = rng() | 1ull;
    return b;
}

/// Cycle-walk the bijection until it lands inside [0, n). Terminates because the
/// map is a permutation of a superset of [0, n): iterating from any point walks a
/// cycle, and every cycle contains at least one in-range element.
inline size_t scrambleIntoRange(const PowerOfTwoBijection& b, uint64_t x, size_t n) {
    for (unsigned guard = 0; guard < 64; ++guard) {
        x = b.apply(x);
        if (x < static_cast<uint64_t>(n)) return static_cast<size_t>(x);
    }
    return static_cast<size_t>(x % static_cast<uint64_t>(n));
}

/**
 * @brief Zipf rank sampler over [1, n] with the standard zeta-normalized form.
 *
 * The O(n) zeta sum is paid once at construction and no per-element table is
 * kept, which matters because a point sweep instantiates this at every stream
 * length in the sweep and a cumulative-probability table over ten million rows
 * would itself evict the payload under test.
 */
struct ZipfSampler {
    double zetaN{}, alpha{}, eta{}, theta{}, halfPow{};
    size_t n{};

    ZipfSampler(size_t count, double requestedTheta) : n(count) {
        // The closed form's alpha = 1/(1-theta) diverges at exactly theta == 1,
        // which is also the most natural value to request.  Nudging off the pole
        // keeps a caller's `--zipf-theta 1.0` working; the distribution differs
        // from true theta = 1 by less than the run-to-run spread of a timed cell.
        theta = std::abs(requestedTheta - 1.0) < 1e-3 ? 0.999 : requestedTheta;
        for (size_t i = 1; i <= n; ++i) zetaN += 1.0 / std::pow(static_cast<double>(i), theta);
        halfPow = std::pow(0.5, theta);
        const double zeta2 = 1.0 + halfPow;
        alpha = 1.0 / (1.0 - theta);
        eta   = (1.0 - std::pow(2.0 / static_cast<double>(n), 1.0 - theta)) /
              (1.0 - zeta2 / zetaN);
    }

    /// Returns a 0-based rank; rank 0 is the hottest.
    size_t next(std::mt19937_64& rng) const {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const double u  = u01(rng);
        const double uz = u * zetaN;
        if (uz < 1.0) return 0;
        if (uz < 1.0 + halfPow) return 1;
        const double r = static_cast<double>(n) * std::pow(eta * u - eta + 1.0, alpha);
        if (!(r >= 0.0)) return 0;  // NaN from an extreme theta collapses to the hot row
        const size_t rank = static_cast<size_t>(r);
        return rank >= n ? n - 1 : rank;
    }
};

inline void finalizePointTrace(PointTrace& t) {
    if (t.indices.empty()) return;
    std::vector<size_t> sorted(t.indices);
    std::sort(sorted.begin(), sorted.end());
    t.footprintSpan   = sorted.back() - sorted.front() + 1;
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    t.distinctIndices = sorted.size();
    t.distinctFraction =
        static_cast<double>(t.distinctIndices) / static_cast<double>(t.indices.size());
}

}  // namespace detail

/**
 * @brief Build the probe list for one point access, deterministic in `p`.
 *
 * Every index in the result is in [0, streamLength).  Ascending order is a
 * post-processing sort rather than a property of the generators, so that
 * `ascending` compares probe *order* at a fixed probe *set* — sorting a Uniform
 * trace must not change its distinctFraction, or the ordered and unordered rows
 * would not be measuring the same thing.
 */
inline PointTrace buildPointTrace(const PointTraceParams& p) {
    PointTrace t;
    if (p.streamLength == 0 || p.probes == 0) return t;

    const size_t n = p.streamLength;
    std::mt19937_64 rng(p.seed);
    t.indices.reserve(p.probes);

    switch (p.pattern) {
        case PointPattern::Uniform: {
            // A permutation truncated to `probes`, matching the existing harness.
            // Beyond streamLength probes there is no permutation left to truncate,
            // so further probes come from re-shuffled passes: the distinct-index
            // count saturates at streamLength and distinctFraction falls below 1,
            // which is the honest report of what a probes > N request can be.
            std::vector<size_t> perm(n);
            std::iota(perm.begin(), perm.end(), size_t{0});
            std::shuffle(perm.begin(), perm.end(), rng);
            while (t.indices.size() < p.probes) {
                const size_t take = std::min(n, p.probes - t.indices.size());
                t.indices.insert(t.indices.end(), perm.begin(),
                                 perm.begin() + static_cast<std::ptrdiff_t>(take));
                if (t.indices.size() < p.probes) std::shuffle(perm.begin(), perm.end(), rng);
            }
            break;
        }

        case PointPattern::Zipf: {
            // Ranks are skewed, then scrambled through a bijection so the hot set
            // is spread over the whole index space instead of sitting in the first
            // few hundred rows.  Without the scramble a hot set of 1000 ranks
            // would also be a hot set of one or two codec blocks, and "small
            // working set" would be inseparable from "few blocks touched".
            const detail::PowerOfTwoBijection bij = detail::makeBijection(n, p.seed);
            const double theta = std::clamp(p.zipfTheta, 0.0, 5.0);
            if (theta <= 0.0 || n < 4) {
                std::uniform_int_distribution<size_t> pick(0, n - 1);
                for (size_t i = 0; i < p.probes; ++i) t.indices.push_back(pick(rng));
            } else {
                const detail::ZipfSampler zipf(n, theta);
                for (size_t i = 0; i < p.probes; ++i)
                    t.indices.push_back(detail::scrambleIntoRange(bij, zipf.next(rng), n));
            }
            break;
        }

        case PointPattern::GeometricClustered: {
            // REUSE, not re-derivation: the clustering comes from
            // makeSelectiveTrace(), the same geometric run/gap model behind
            // GatherTraceGen's GapModel::Geometric and BenchmarkRunner's
            // selective phase.  A second notion of clustering local to this file
            // would make point and gather rows incomparable at equal
            // (selectivity, meanRunLength) — which is precisely the comparison
            // the two drivers exist to support ("is it cheaper to point-read the
            // surviving rows or to gather them?").  So the ranges are produced
            // there and only expanded into probe indices here.
            SelectiveTraceParams sp;
            sp.selectivity   = std::clamp(p.selectivity, 1e-9, 1.0);
            sp.meanRunLength = static_cast<double>(std::max<size_t>(1, p.meanRunLength));
            sp.seed          = p.seed;
            const RowRangeList ranges = makeSelectiveTrace(n, sp);

            std::vector<size_t> rows;
            size_t surviving = 0;
            for (const auto& r : ranges) surviving += r.size();
            if (surviving == 0) return t;
            rows.reserve(surviving);
            for (const auto& r : ranges)
                for (size_t i = r.begin; i < r.end; ++i) rows.push_back(i);

            // Probes are drawn from the surviving set with replacement.  The
            // clustering therefore constrains the FOOTPRINT of the access, not
            // the order in which it is walked; walking the runs in order would
            // conflate clustering with sequentiality, which `ascending` exists to
            // vary independently.
            std::uniform_int_distribution<size_t> pick(0, rows.size() - 1);
            for (size_t i = 0; i < p.probes; ++i) t.indices.push_back(rows[pick(rng)]);
            break;
        }

        case PointPattern::Strided: {
            // Start offset is seeded so a sweep over strides does not always begin
            // in the same block; the walk itself is deterministic.  When
            // gcd(stride, n) != 1 the walk closes into a cycle shorter than n and
            // revisits it, which shows up as a low distinctIndices rather than as
            // a silent change of meaning.
            const size_t stride = std::max<size_t>(1, p.stride % std::max<size_t>(1, n));
            size_t idx = rng() % n;
            for (size_t i = 0; i < p.probes; ++i) {
                t.indices.push_back(idx);
                idx = (idx + stride) % n;
            }
            break;
        }

        case PointPattern::SequentialProbe: {
            // Wraps rather than stopping at n: the overhead floor must be
            // measurable at any batch size, independent of stream length.
            for (size_t i = 0; i < p.probes; ++i) t.indices.push_back(i % n);
            break;
        }
    }

    if (p.ascending) std::sort(t.indices.begin(), t.indices.end());
    detail::finalizePointTrace(t);
    return t;
}

inline std::string PointTrace::label() const {
    std::string s = "probes=" + std::to_string(indices.size()) +
                    " distinct=" + std::to_string(distinctIndices) + " span=" +
                    std::to_string(footprintSpan);
    // Three decimals is enough to read the codec-vs-cache question off the label;
    // the exact value belongs in the result row, not in a log line.
    const long milli = std::lround(std::clamp(distinctFraction, 0.0, 1.0) * 1000.0);
    std::string frac = std::to_string(milli % 1000);
    frac.insert(frac.begin(), 3 - frac.size(), '0');
    s += " distinct_frac=" + std::to_string(milli / 1000) + "." + frac;
    return s;
}

}  // namespace encodings::benchmark
