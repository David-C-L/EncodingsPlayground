#pragma once

#include <cstddef>
#include <atomic>

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

// PeakHeapTracker was removed: it sampled mallinfo2 from a background thread and
// was documented as the mechanism behind the "peak" memory fields, but it was
// never instantiated anywhere. The memory pass is entirely ScopedAllocationTrack
// (see AllocationTracker.hpp), which intercepts operator new/delete and so also
// catches transient buffers freed before the call returns -- something a sampling
// thread would miss. Keeping a dead alternative implementation next to the real
// one only invited reading the wrong one.


} // namespace encodings::benchmark
