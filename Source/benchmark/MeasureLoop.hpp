#pragma once

// The one timed loop every driver goes through.
//
// Implements section 3 of Benchmarks/drivers/CONVENTIONS.md.  The loop is shared
// rather than copied per driver because each of its rules was a wrong number
// once: an un-clobbered sink let the optimizer delete the decode; eviction inside
// the t0/t1 window charged a multi-megabyte cache thrash to decode throughput;
// a mean over five iterations moved by 3x when one of them was descheduled.
//
// What this file deliberately does NOT do: allocate the output buffer.  Sinks are
// hoisted into the caller and reused across iterations, because a fresh
// std::vector per iteration charges a heap allocation plus a zero-fill to the
// codec, and for a fast bulk decode that is most of the measurement.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "benchmark/CachePolicy.hpp"
#include "benchmark/TimingStats.hpp"

namespace encodings::benchmark {

/// Force the compiler to treat `p`'s pointee as read and all memory as clobbered.
///
/// Without this, a decode whose result is never inspected is dead code and a
/// sufficiently good optimizer removes it — the classic "my codec decodes 400
/// GB/s" result.  The empty asm is cheaper than any real consumption of the sink
/// (no reduction loop, no volatile store per element) and does not perturb the
/// measured code, only the optimizer's liberty to delete it.
inline void clobber(const void* p) { asm volatile("" : : "r"(p) : "memory"); }

struct MeasureSpec {
    size_t iterations{5};
    /// Untimed iterations run with no eviction, purely to fault in the sink's
    /// pages, resolve lazily-parsed headers and let any per-codec cache
    /// (SubIntSplit's parsed header, FPE's index cache) reach steady state.
    /// Without them the first timed iteration measures one-off setup, which is a
    /// real cost but not the one any of these drivers reports.
    size_t warmup{2};
};

struct MeasureResult {
    TimingSummary time;
    /// Cache preparation cost, summarized separately.  It is not part of `time`
    /// and must not be added to it: a driver reports it as `evict_ns` so a reader
    /// can tell a slow decode from an expensive eviction.
    TimingSummary evict;
    size_t iterationsRun{};
};

/// Run `fn` under the measurement contract.  `fn` must decode into a sink the
/// caller already owns; `targets.sink` is what gets clobbered afterwards.
template <typename Fn>
MeasureResult measure(const MeasureSpec& spec,
                      CacheController& controller,
                      const EvictionTargets& targets,
                      Fn&& fn) {
    using Clock = std::chrono::high_resolution_clock;

    for (size_t i = 0; i < spec.warmup; ++i) {
        fn();
        clobber(targets.sink.data());
    }

    std::vector<int64_t> timeSamples;
    std::vector<int64_t> evictSamples;
    timeSamples.reserve(spec.iterations);
    evictSamples.reserve(spec.iterations);

    for (size_t i = 0; i < spec.iterations; ++i) {
        // Outside the window, always — for Hot this is the read-touch that makes
        // "hot" mean resident, and for the cold states it is the eviction whose
        // cost belongs in evict_ns.
        controller.prepare(targets);

        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();

        clobber(targets.sink.data());
        timeSamples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        evictSamples.push_back(controller.lastEvictNs());
    }

    MeasureResult r;
    r.time  = summarize(timeSamples);
    r.evict = summarize(evictSamples);
    r.iterationsRun = spec.iterations;
    return r;
}

/// Cost of the timing calls themselves, over `iterations` empty timed regions.
/// A ~50ns operation timed with two clock reads is mostly clock; drivers report
/// this as a calibration row.
inline TimingSummary measureClockOverhead(size_t iterations) {
    using Clock = std::chrono::high_resolution_clock;
    std::vector<int64_t> samples;
    samples.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        const auto t0 = Clock::now();
        // Empty on purpose: the quantity wanted is exactly one now()-pair, the
        // same pair that wraps a point read, including the barrier it implies.
        const auto t1 = Clock::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    return summarize(samples);
}

}  // namespace encodings::benchmark
