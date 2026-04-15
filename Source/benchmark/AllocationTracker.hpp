#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Allocation-level peak tracking for the benchmark runner.
//
// Motivation
// ----------
// mallinfo2()-based approaches only see the *net* heap state at discrete
// sample points.  They miss allocations that are both made and freed within a
// single call (e.g. SubIntSplitEncoder::decodeAt calling decodeAll() on a
// non-random-access section codec, allocating an ~8 MB std::vector<T> just to
// retrieve one value, then immediately freeing it).
//
// This module hooks operator new / operator delete (via allocation_hooks.cpp
// which must be linked into the benchmark binary) to maintain a running
// live-bytes counter.  A ScopedAllocationTrack RAII guard records the
// baseline live bytes at construction and exposes the peak above that
// baseline observed during its lifetime — capturing intra-call transient
// allocations that mallinfo2 would never see.
//
// Design notes
// ------------
// * liveBytes is a global running total of all live heap bytes in the process.
//   It is updated on every operator new / delete, unconditionally.
// * Peak tracking is gated behind a bool so that the (modest) CAS on peak
//   only fires while a ScopedAllocationTrack is alive.
// * Using int64_t for liveBytes avoids unsigned wrap-around if pre-existing
//   allocations made before a tracking scope starts are freed inside it.
// * malloc_usable_size(ptr) is used in allocation_hooks.cpp so that alloc and
//   free see the same size (glibc may round up the requested size internally).
// ---------------------------------------------------------------------------

namespace encodings::benchmark::alloc_tracking {

/// Running total of live heap bytes across the entire process lifetime.
/// Incremented in operator new, decremented in operator delete.
extern std::atomic<int64_t> liveBytes;

/// Maximum value of liveBytes observed since the current ScopedAllocationTrack
/// was constructed.  Only updated while trackingPeak is true.
extern std::atomic<int64_t> peakLiveBytes;

/// True while a ScopedAllocationTrack is alive.
extern std::atomic<bool> trackingPeak;

} // namespace encodings::benchmark::alloc_tracking


namespace encodings::benchmark {

/// Called from operator new override in allocation_hooks.cpp.
/// usableSize should be malloc_usable_size(ptr) — the actual block size.
inline void recordAlloc(std::size_t usableSize) noexcept {
    namespace T = alloc_tracking;
    int64_t cur =
        T::liveBytes.fetch_add(static_cast<int64_t>(usableSize),
                               std::memory_order_relaxed) +
        static_cast<int64_t>(usableSize);

    if (T::trackingPeak.load(std::memory_order_relaxed)) {
        int64_t prev = T::peakLiveBytes.load(std::memory_order_relaxed);
        while (cur > prev &&
               !T::peakLiveBytes.compare_exchange_weak(
                   prev, cur, std::memory_order_relaxed)) {}
    }
}

/// Called from operator delete override in allocation_hooks.cpp.
/// usableSize should be malloc_usable_size(ptr) measured BEFORE free.
inline void recordFree(std::size_t usableSize) noexcept {
    alloc_tracking::liveBytes.fetch_sub(static_cast<int64_t>(usableSize),
                                        std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------

/**
 * @brief RAII guard that activates allocation peak tracking.
 *
 * Construct just before the code under measurement; call peakAboveBaseline()
 * (or let the destructor run) when done.  The returned value is the highest
 * number of concurrently-live bytes *above the construction-time baseline*
 * observed during the guard's lifetime.
 *
 * This correctly captures allocations that are freed before the measured
 * function returns — something that a mallinfo2() before/after snapshot
 * cannot do.
 *
 * Example:
 *   {
 *     ScopedAllocationTrack track;
 *     encoder->decodeAt(encoded, idx);   // may internally alloc & free
 *     peak = track.peakAboveBaseline();  // non-zero if any heap was used
 *   }
 */
class ScopedAllocationTrack {
public:
    ScopedAllocationTrack() noexcept {
        namespace T = alloc_tracking;
        baseline_ = T::liveBytes.load(std::memory_order_acquire);
        T::peakLiveBytes.store(baseline_, std::memory_order_release);
        T::trackingPeak.store(true, std::memory_order_release);
    }

    ~ScopedAllocationTrack() noexcept { stop(); }

    // Non-copyable, non-movable.
    ScopedAllocationTrack(const ScopedAllocationTrack&)            = delete;
    ScopedAllocationTrack& operator=(const ScopedAllocationTrack&) = delete;

    /**
     * @brief Stop tracking and return peak heap bytes above the baseline.
     *        Safe to call multiple times; subsequent calls return the same value.
     */
    std::size_t stop() noexcept {
        bool was = alloc_tracking::trackingPeak.exchange(
            false, std::memory_order_acq_rel);
        (void)was;
        int64_t p = alloc_tracking::peakLiveBytes.load(std::memory_order_acquire);
        return p > baseline_ ? static_cast<std::size_t>(p - baseline_) : 0;
    }

    /// Peek without stopping — safe to call while tracking is still active.
    std::size_t peakAboveBaseline() const noexcept {
        int64_t p = alloc_tracking::peakLiveBytes.load(std::memory_order_acquire);
        return p > baseline_ ? static_cast<std::size_t>(p - baseline_) : 0;
    }

private:
    int64_t baseline_{0};
};

} // namespace encodings::benchmark
