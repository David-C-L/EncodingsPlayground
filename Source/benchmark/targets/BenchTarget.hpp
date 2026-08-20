#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "encodings/RowRange.hpp"

namespace encodings::benchmark {

/**
 * @brief What a target can do, as opposed to what it was asked to do.
 *
 * `randomAccess` and `fastSkip` mirror the codec's declared properties and gate
 * whether a driver runs a cell at all.  `overridesRangeInto` / `overridesGather`
 * record whether the target implements those paths natively or inherits a
 * fallback loop, which is the difference between "this codec's gather is slow"
 * and "this codec has no gather and the base class is looping decodeRangeInto".
 *
 * `nativePointRead` is the one that changes how a number may be *compared*: see
 * the note on emulation below.
 */
struct TargetCaps {
    bool randomAccess{};
    bool fastSkip{};
    bool nativePointRead{};
    bool overridesRangeInto{};
    bool overridesGather{};
};

/**
 * @brief Profiling counters read back after a measured operation.
 *
 * -1 means "not reported", never "zero" — a codec with no distinct skip phase
 * must reach the result row as a typed null (CONVENTIONS section 6), and the
 * only way to keep that distinction is to refuse to encode it as a number.
 * Empty vectors mean the target has no sub-stream decomposition to report.
 */
struct TargetProfile {
    int64_t gatherSkipNs{-1};
    int64_t gatherMaterializeNs{-1};
    std::vector<int64_t> subStreamBulkNs;
    std::vector<int64_t> subStreamPointNs;
    std::vector<int64_t> subStreamRangeNs;
};

/**
 * @brief The port seam: everything a driver body is allowed to know about a codec.
 *
 * A concept, not a virtual base class.  The drivers measure operations that run
 * in tens of nanoseconds per call, and a virtual dispatch inserted between the
 * clock and the work under test would be a measurable fraction of exactly the
 * quantity being reported.  Zero added indirection is therefore a hard
 * requirement, which rules out an abstract interface and leaves a concept
 * satisfied by inline pass-throughs.
 *
 * (a) WHY THE NAMES ARE NIMBLE'S.  materializeAll / materializeRange /
 *     skipThenMaterialize / pointRead are nimble's vocabulary, not this repo's
 *     decode*.  They map onto nimble's Encoding::materialize(), its
 *     skip() + materialize() streaming pair, and readWithVisitor() respectively.
 *     Naming the seam after this repo instead would cost nothing today and a
 *     great deal later: the port would become a rename of every driver body plus
 *     an argument about semantics at each call site (does decodeRange re-seek, or
 *     continue from where the last call left off?).  Adopting the target
 *     vocabulary up front forces that argument to be settled once, here, where
 *     the mapping is written down — and reduces the port to writing one adapter.
 *
 * (b) native() IS THE POINT AT WHICH THE SEAM ADMITS IT IS NOT TOTAL.  The
 *     concept is the COMMON surface, not the ONLY surface.  It deliberately does
 *     not express cost-model estimates, index-type selection, sub-stream layout,
 *     or nimble's selective-reader stack, and it should not grow to: a seam that
 *     covers everything is a second copy of both APIs.  native() returns the
 *     underlying object by reference with no common type across targets, so a
 *     driver that needs a repo-specific capability reaches through it in a
 *     clearly-marked place, and nimble's own harnesses stay reachable from a
 *     driver written against this concept.
 *
 * (c) EMULATED CAPABILITIES MUST BE LABELLED.  nimble has no decodeAt: a point
 *     read there is a one-row visitor read, which pays visitor setup and a
 *     one-element buffer per probe.  A target in that position reports
 *     capabilities().nativePointRead == false, and a driver MUST propagate that
 *     into its output (`emulated_point_read`), because the alternative is a table
 *     in which a native O(1) lookup and an emulated one sit in the same column
 *     and differ by a constant nobody can see.  The same rule applies to any
 *     future emulated path: emulation is permitted, silent emulation is not.
 */
template <typename Target>
concept BenchTargetC = requires(Target&                                   t,
                                std::span<const typename Target::Elem>    src,
                                typename Target::Elem*                    dst,
                                const RowRangeList&                       ranges,
                                size_t                                    n,
                                size_t                                    index,
                                size_t                                    begin,
                                size_t                                    end) {
    typename Target::Elem;

    { t.name() };

    // Encode is part of the seam because bench_encode measures it; decode
    // drivers call it exactly once per artifact (CONVENTIONS section 2).
    { t.encode(src) };

    // The payload as bytes is what compression ratios divide and what cache
    // eviction flushes, so it is a span of bytes rather than of elements.
    { t.payloadBytes() } -> std::same_as<std::span<const std::byte>>;

    // All three materialize paths write into a caller-owned buffer: output
    // buffers are hoisted out of the timed region, so no allocating variant of
    // any of them belongs in the seam at all.
    { t.materializeAll(dst, n) };
    { t.materializeRange(begin, end, dst, n) };
    { t.skipThenMaterialize(ranges, dst, n) };

    { t.pointRead(index) };

    { t.capabilities() } -> std::same_as<TargetCaps>;

    // Profiling is reset immediately before a measured region and read
    // immediately after, so counters describe one cell and not a whole sweep.
    { t.resetProfiling() };
    { t.profile() } -> std::same_as<TargetProfile>;

    // Codec-internal structures (index tables, dictionaries, rank samples) for
    // the cold-all cache state.  A target that cannot enumerate them returns an
    // empty list, which forces the driver's documented fallback to an LLC thrash
    // rather than letting it claim a clean payload-cold measurement.
    { t.internalBuffers() };

    // Intentionally unconstrained: see (b).
    { t.native() };
};

}  // namespace encodings::benchmark
