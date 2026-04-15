/**
 * allocation_hooks.cpp
 *
 * Overrides the six standard global operator new / operator delete forms so
 * that every heap allocation in the benchmark binary is counted.  This feeds
 * the AllocationTracker used by BenchmarkRunner::measureMemoryUsage() to
 * capture intra-call peak allocations that a mallinfo2() before/after snapshot
 * would miss (e.g. a decodeAt() that allocates and immediately frees an 8 MB
 * section buffer internally).
 *
 * Linked into run_benchmarks only — not into the library.
 *
 * Implementation notes
 * --------------------
 * * malloc_usable_size(ptr) is used instead of the requested size so that
 *   recordAlloc() and recordFree() see the same block size.  glibc may round
 *   up the requested size; using usable size keeps liveBytes consistent.
 * * The size is captured BEFORE std::free() in every delete override, which
 *   is the only valid window to call malloc_usable_size.
 * * On non-Linux platforms malloc_usable_size is absent; we fall back to 0
 *   (allocation tracking is disabled but everything still links and runs).
 */

#include "benchmark/AllocationTracker.hpp"
#include <cstdlib>
#include <new>

#ifdef __linux__
#include <malloc.h>
static inline std::size_t usable(void* p) noexcept {
    return p ? malloc_usable_size(p) : 0;
}
#else
static inline std::size_t usable(void*) noexcept { return 0; }
#endif

// ---------------------------------------------------------------------------
// Definition of the three extern atomics declared in AllocationTracker.hpp
// ---------------------------------------------------------------------------
namespace encodings::benchmark::alloc_tracking {
    std::atomic<int64_t> liveBytes{0};
    std::atomic<int64_t> peakLiveBytes{0};
    std::atomic<bool>    trackingPeak{false};
}

// ---------------------------------------------------------------------------
// operator new overrides
// ---------------------------------------------------------------------------

void* operator new(std::size_t n) {
    void* p = std::malloc(n);
    if (!p) throw std::bad_alloc{};
    encodings::benchmark::recordAlloc(usable(p));
    return p;
}

void* operator new[](std::size_t n) {
    void* p = std::malloc(n);
    if (!p) throw std::bad_alloc{};
    encodings::benchmark::recordAlloc(usable(p));
    return p;
}

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    void* p = std::malloc(n);
    if (p) encodings::benchmark::recordAlloc(usable(p));
    return p;
}

void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    void* p = std::malloc(n);
    if (p) encodings::benchmark::recordAlloc(usable(p));
    return p;
}

// ---------------------------------------------------------------------------
// operator delete overrides
// Size is captured BEFORE std::free() — the only valid time to call
// malloc_usable_size.
// ---------------------------------------------------------------------------

void operator delete(void* p) noexcept {
    if (p) { std::size_t sz = usable(p); std::free(p); encodings::benchmark::recordFree(sz); }
}

void operator delete[](void* p) noexcept {
    if (p) { std::size_t sz = usable(p); std::free(p); encodings::benchmark::recordFree(sz); }
}

void operator delete(void* p, std::size_t) noexcept {
    if (p) { std::size_t sz = usable(p); std::free(p); encodings::benchmark::recordFree(sz); }
}

void operator delete[](void* p, std::size_t) noexcept {
    if (p) { std::size_t sz = usable(p); std::free(p); encodings::benchmark::recordFree(sz); }
}

void operator delete(void* p, const std::nothrow_t&) noexcept {
    if (p) { std::size_t sz = usable(p); std::free(p); encodings::benchmark::recordFree(sz); }
}

void operator delete[](void* p, const std::nothrow_t&) noexcept {
    if (p) { std::size_t sz = usable(p); std::free(p); encodings::benchmark::recordFree(sz); }
}
