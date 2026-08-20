#pragma once

// The single implementation of every access-shape measurement.
//
// Four drivers (bench_decode_bulk / _range / _gather / _point) measure four
// access shapes over the same seam, and before this header each carried its own
// copy of the loop.  They had already drifted: one flushed the sink and one did
// not, one reset the profiling accumulators per iteration and one per cell, one
// timed each point probe individually.  Those are not stylistic differences —
// they change the number.  So the loop lives here once and a driver's body is
// reduced to choosing a shape and reporting the result.
//
// Everything this class does that is not obvious follows from CONVENTIONS
// sections 3 and 5, and each rule is spelled out at the place it is enforced:
//
//   * caller-owned sinks, hoisted (the constructor takes no buffer at all);
//   * EvictionTargets assembled once per call, outside the timed region;
//   * a profiling window that covers the timed iterations and nothing else;
//   * point probes timed as a BATCH, because a single probe is the same order as
//     the clock read that would measure it.

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark/CachePolicy.hpp"
#include "benchmark/MeasureLoop.hpp"
#include "benchmark/PointTraceGen.hpp"
#include "benchmark/targets/BenchTarget.hpp"
#include "encodings/RowRange.hpp"

namespace encodings::benchmark {

/// Per-probe cost of a batch-timed point measurement.
///
/// Free function rather than a field on MeasureResult: the division is only
/// meaningful for `points()`, and a `nsPerProbe` member on every result would
/// invite a driver to divide a gather time by a range count and call it the same
/// quantity.
inline double nsPerProbe(const MeasureResult& r, size_t probes) {
    return probes == 0 ? 0.0
                       : static_cast<double>(r.time.medianNs) / static_cast<double>(probes);
}

template <typename Target>
    requires BenchTargetC<Target>
class DecodeHarness {
public:
    using Elem = typename Target::Elem;

    /// Borrows both: the target owns the artifact and outlives the harness, and
    /// the controller is shared across a sweep so its Auto-resolved method is
    /// latched once and labels every row it produced.
    DecodeHarness(Target& target, CacheController& cache) : target_(&target), cache_(&cache) {}

    /// Full materialization of `sink.size()` elements.
    MeasureResult bulk(std::span<Elem> sink, const MeasureSpec& spec) {
        const size_t n = sink.size();
        return run(sink, n, spec, [this, sink, n] { target_->materializeAll(sink.data(), n); });
    }

    /// One contiguous [begin, end).
    MeasureResult range(size_t begin, size_t end, std::span<Elem> sink, const MeasureSpec& spec) {
        if (end < begin) throw std::invalid_argument("DecodeHarness::range: end < begin");
        const size_t n = end - begin;
        require(sink, n, "range");
        return run(sink, n, spec, [this, begin, end, sink, n] {
            target_->materializeRange(begin, end, sink.data(), n);
        });
    }

    /// An ordered range list with gaps to skip.  `n` is the total selected count,
    /// which the caller already knows from the trace — recomputing it here would
    /// be a second sum over the ranges per call for no new information.
    MeasureResult gather(const RowRangeList& ranges, std::span<Elem> sink, size_t n,
                         const MeasureSpec& spec) {
        require(sink, n, "gather");
        return run(sink, n, spec, [this, &ranges, sink, n] {
            target_->skipThenMaterialize(ranges, sink.data(), n);
        });
    }

    /// A batch of individual lookups, timed as ONE region.
    ///
    /// Not per probe, deliberately.  A `decodeAt` on this box runs in tens of ns
    /// and a `high_resolution_clock::now()` pair costs ~50 ns, so a per-probe
    /// timing reports mostly clock and ranks codecs by how well their cost hides
    /// inside that constant.  The batch is the measurable quantity; the derived
    /// per-probe figure comes from nsPerProbe(), and a driver reports the clock
    /// calibration from measureClockOverhead() next to it so a reader can see how
    /// much of one probe the clock would have been.
    ///
    /// Each value is accumulated into a volatile sink INSIDE the batch (a
    /// discarded optional is dead code the optimizer may delete outright) and the
    /// batch is clobbered once at the end by measure().
    MeasureResult points(const PointTrace& trace, const MeasureSpec& spec) {
        // The volatile accumulator doubles as the eviction/clobber sink: there is
        // no output buffer for a point read, and pointing EvictionTargets at
        // nothing would silently turn cold-all into cold-payload.
        const std::span<std::byte> sink(
            reinterpret_cast<std::byte*>(const_cast<Elem*>(&pointSink_)), sizeof(Elem));
        return runRaw(sink, spec, [this, &trace] {
            Elem acc{};
            for (size_t index : trace.indices) {
                const auto v = target_->pointRead(index);
                if (v) acc = static_cast<Elem>(acc + *v);
            }
            pointSink_ = acc;
        });
    }

    /// Counters for the most recent measured call, and nothing else.
    ///
    /// The window is opened after the warmup iterations and closed after the last
    /// timed one (see run()), so a codec that accumulates across calls reports the
    /// timed phase, and one that zeroes per call reports its final iteration.
    /// Either way no warmup and no previous cell leaks in — a stale gather
    /// counter surviving into the next cell would be reported as that cell's skip
    /// time, which is worse than reporting nothing.
    TargetProfile profile() const { return lastProfile_; }

    /// Round-trip check, run before any measurement.
    ///
    /// Returns false with a human-readable reason rather than throwing: a codec
    /// that fails here is excluded and recorded while the rest of the sweep runs
    /// (CONVENTIONS section 5), and an exception would take the whole run with it.
    bool validate(std::span<const Elem> reference, std::string& whyNot) {
        const size_t n = reference.size();
        if (n == 0) {
            whyNot = "empty reference";
            return false;
        }

        scratch_.assign(n, Elem{});
        target_->materializeAll(scratch_.data(), n);
        for (size_t i = 0; i < n; ++i) {
            if (scratch_[i] != reference[i]) {
                whyNot = std::format("materializeAll mismatch at row {}: got {}, expected {}",
                                     i, static_cast<int64_t>(scratch_[i]),
                                     static_cast<int64_t>(reference[i]));
                return false;
            }
        }

        const size_t span  = n / 4 > 0 ? n / 4 : n;
        const size_t begin = (n - span) / 3;
        return validateGatherEqualsRange(begin, begin + span, whyNot);
    }

    /// The sigma = 1 identity: a gather over one full-width range must equal the
    /// contiguous range read.
    ///
    /// This is the claim that makes a gather row at sigma = 1 comparable with the
    /// range driver at all, so it lives in the harness and every decode driver
    /// that validates gets the same assertion.  Exposed separately from
    /// validate() because a gather driver checks it at every span of its own axis,
    /// where validate() checks one representative window.
    bool validateGatherEqualsRange(size_t begin, size_t end, std::string& whyNot) {
        if (end <= begin) return true;
        const size_t span = end - begin;
        const RowRangeList full{{begin, end}};

        scratch_.assign(span, Elem{});
        target_->skipThenMaterialize(full, scratch_.data(), span);
        scratch2_.assign(span, Elem{});
        target_->materializeRange(begin, end, scratch2_.data(), span);
        for (size_t i = 0; i < span; ++i) {
            if (scratch_[i] != scratch2_[i]) {
                whyNot = std::format(
                    "sigma=1 gather != materializeRange at offset {} of [{}, {}): {} vs {}",
                    i, begin, end, static_cast<int64_t>(scratch_[i]),
                    static_cast<int64_t>(scratch2_[i]));
                return false;
            }
        }
        return true;
    }

    /// Value-level check of one gather trace against the reference stream.
    ///
    /// Separate from validate() because it is per-cell: the gather driver walks
    /// its own (span, sigma) axes so that the traces it validates are the traces
    /// it will measure, and a fixed trace here would leave the sweep's actual
    /// range structure unchecked.
    bool validateGather(const RowRangeList& ranges, std::span<const Elem> reference,
                        std::string& whyNot) {
        size_t selected = 0;
        for (const auto& r : ranges) selected += r.size();
        if (selected == 0) return true;

        scratch_.assign(selected, Elem{});
        target_->skipThenMaterialize(ranges, scratch_.data(), selected);

        size_t off = 0;
        for (const auto& r : ranges) {
            for (size_t i = r.begin; i < r.end; ++i, ++off) {
                if (i >= reference.size()) {
                    whyNot = std::format("trace row {} is past the end of the stream ({})", i,
                                         reference.size());
                    return false;
                }
                if (scratch_[off] != reference[i]) {
                    whyNot = std::format("gather mismatch at row {}: got {}, expected {}", i,
                                         static_cast<int64_t>(scratch_[off]),
                                         static_cast<int64_t>(reference[i]));
                    return false;
                }
            }
        }
        return true;
    }

private:
    static void require(std::span<const Elem> sink, size_t n, const char* what) {
        if (sink.size() < n)
            throw std::invalid_argument(std::format(
                "DecodeHarness::{}: sink holds {} elements, {} needed — sinks are "
                "caller-owned and must be sized for the widest cell of the sweep",
                what, sink.size(), n));
    }

    template <typename Fn>
    MeasureResult run(std::span<Elem> sink, size_t n, const MeasureSpec& spec, Fn&& fn) {
        // Only the part of the sink this call writes: cooling (or warming) the
        // whole sweep-sized buffer would charge the harness's buffer reuse to the
        // codec, and for a narrow cell that is most of the measurement.
        return runRaw(std::span<std::byte>(reinterpret_cast<std::byte*>(sink.data()),
                                           n * sizeof(Elem)),
                      spec, std::forward<Fn>(fn));
    }

    /// The one place any of this is timed.
    ///
    /// The eviction targets are assembled here, once, before the first iteration:
    /// internalBuffers() allocates a vector and a cold-all state would otherwise
    /// pay that allocation inside the per-iteration eviction it was meant to
    /// prepare.
    ///
    /// The warmup and timed phases are two measure() calls rather than one so the
    /// profiling window can open between them.  A single call with spec.warmup
    /// would fold the untimed iterations into the counters, and for a codec whose
    /// accumulators span calls that inflates the reported skip time by the warmup
    /// count with nothing in the row to reveal it.
    template <typename Fn>
    MeasureResult runRaw(std::span<std::byte> sink, const MeasureSpec& spec, Fn&& fn) {
        EvictionTargets targets;
        targets.payload       = target_->payloadBytes();
        targets.sink          = sink;
        targets.codecInternal = target_->internalBuffers();

        if (spec.warmup > 0) {
            MeasureSpec warm;
            warm.iterations = 0;
            warm.warmup     = spec.warmup;
            measure(warm, *cache_, targets, fn);
        }

        MeasureSpec timed;
        timed.iterations = spec.iterations;
        timed.warmup     = 0;

        target_->resetProfiling();
        MeasureResult r = measure(timed, *cache_, targets, std::forward<Fn>(fn));
        lastProfile_    = target_->profile();
        return r;
    }

    Target*          target_;
    CacheController* cache_;
    TargetProfile    lastProfile_{};
    /// Reused across calls so validation does not allocate per cell.
    std::vector<Elem> scratch_, scratch2_;
    volatile Elem     pointSink_{};
};

}  // namespace encodings::benchmark
