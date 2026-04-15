#pragma once

#include <cstddef>
#include <atomic>
#include <thread>
#include <chrono>

#ifdef __linux__
#include <malloc.h>
#endif

namespace encodings::benchmark {

/**
 * @brief Returns the number of bytes currently in use on the heap.
 *
 * Uses glibc mallinfo2 (>= glibc 2.33) on Linux, falling back to mallinfo
 * on older toolchains. Returns 0 on unsupported platforms.
 */
inline size_t currentHeapBytes() {
#if defined(__linux__)
#  if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
    return static_cast<size_t>(mallinfo2().uordblks);
#  else
    // mallinfo uses int fields (wraps at ~4 GB); acceptable for the fallback path.
    return static_cast<size_t>(static_cast<unsigned int>(mallinfo().uordblks));
#  endif
#else
    return 0;
#endif
}

/**
 * @brief RAII helper that tracks the peak heap allocation above a baseline in
 *        a background thread.
 *
 * Construct just before the operation you want to measure; call stop() (or let
 * the destructor run) once the operation is complete. The returned value is the
 * highest number of bytes above the construction-time heap level observed
 * during the operation.
 *
 * The sampling interval (default 1 ms) is a trade-off:
 *   - Too small  → mallinfo2 lock contention slows the measured operation.
 *   - Too large  → short allocation spikes may be missed.
 * At 1 ms, encode/decode operations on 10 M elements (which typically take
 * tens to hundreds of ms) receive ≥ 50 samples, giving good peak coverage with
 * negligible overhead.
 */
class PeakHeapTracker {
public:
    explicit PeakHeapTracker(
            std::chrono::microseconds interval = std::chrono::microseconds{1000})
        : interval_(interval)
        , running_(true)
        , baseline_(currentHeapBytes())
        , peak_(baseline_)
    {
        thread_ = std::thread([this] {
            while (running_.load(std::memory_order_acquire)) {
                size_t cur  = currentHeapBytes();
                size_t prev = peak_.load(std::memory_order_relaxed);
                while (cur > prev &&
                       !peak_.compare_exchange_weak(
                           prev, cur, std::memory_order_relaxed)) {}
                std::this_thread::sleep_for(interval_);
            }
        });
    }

    ~PeakHeapTracker() { stop(); }

    // Non-copyable, non-movable.
    PeakHeapTracker(const PeakHeapTracker&)            = delete;
    PeakHeapTracker& operator=(const PeakHeapTracker&) = delete;

    /**
     * @brief Stop sampling and return the peak heap bytes above the baseline
     *        observed since construction. Safe to call multiple times.
     */
    size_t stop() {
        bool was = running_.exchange(false, std::memory_order_acq_rel);
        if (was) thread_.join();
        size_t p = peak_.load(std::memory_order_acquire);
        return p > baseline_ ? p - baseline_ : 0;
    }

private:
    std::chrono::microseconds interval_;
    std::atomic<bool>         running_;
    size_t                    baseline_;
    std::atomic<size_t>       peak_;
    std::thread               thread_;
};

} // namespace encodings::benchmark
