#pragma once

// Cache-state control for decode measurements.
//
// Implements section 8 of Benchmarks/drivers/CONVENTIONS.md.  A decode number is
// meaningless without the cache state it was taken in: the same codec differs by
// an order of magnitude between "payload already in L2" and "payload in DRAM",
// and which of those a naive loop measures depends on the payload size, the
// iteration count and what the *previous* cell left in the cache.  This header
// makes the state an explicit, recorded, verifiable input.
//
// Two invariants drive every design decision below:
//
//   1. A cold state that cannot be delivered is an error, never a silent
//      downgrade.  A row labelled cold-payload that was actually hot is worse
//      than a missing row, because it is indistinguishable from a real result.
//   2. Whatever `Auto` resolves to is recorded and reportable, so a plot can
//      never mix clflush rows with LLC-thrash rows under one series.  The two
//      methods do not measure the same thing (see LlcThrash below).
//
// Deliberately absent: madvise(MADV_DONTNEED).  It looks like the obvious way to
// cool a buffer, and it is a data-corruption bug here.  Encoded payloads live in
// private anonymous heap memory; MADV_DONTNEED discards those pages outright and
// the next access faults in *fresh zero pages*, so the payload silently becomes
// a run of zeros.  The codec then decodes garbage at full speed and the run looks
// fast rather than broken — nothing detects it unless validation happens to run
// after eviction.  There is no madvise call anywhere in this file, on purpose.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fstream>
#include <sstream>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#define ENCODINGS_BENCH_HAVE_CLFLUSH 1
#include <immintrin.h>
#else
#define ENCODINGS_BENCH_HAVE_CLFLUSH 0
#endif

namespace encodings::benchmark {

/// What the caches are expected to hold when the timed region starts.
///
/// Hot is an *active* state, not the absence of one — see CacheController::warm.
/// ColdPayload cools only the encoded bytes; ColdAll additionally cools the
/// output sink and whatever codec-internal structures the codec is willing to
/// expose through Decoder::internalBuffers().
enum class CacheState { Hot, ColdPayload, ColdAll };

/// How cooling is performed.  Auto picks per state and payload size, and the
/// choice is latched and reported (CacheController::effectivePolicy).
enum class EvictMethod { Auto, Clflush, LlcThrash, None };

inline const char* cacheStateName(CacheState s) {
    switch (s) {
        case CacheState::Hot:         return "hot";
        case CacheState::ColdPayload: return "cold-payload";
        case CacheState::ColdAll:     return "cold-all";
    }
    return "hot";
}

inline const char* evictMethodName(EvictMethod m) {
    switch (m) {
        case EvictMethod::Auto:      return "auto";
        case EvictMethod::Clflush:   return "clflush";
        case EvictMethod::LlcThrash: return "llc-thrash";
        case EvictMethod::None:      return "none";
    }
    return "auto";
}

/// Detected cache geometry.  Never hardcoded: the LLC size decides both the
/// eviction method and the payload sizes at which hot and cold can differ at
/// all, so a wrong constant produces a sweep whose interesting region is off the
/// end of the axis.
struct CacheTopology {
    size_t l1dBytes{};
    size_t l2Bytes{};
    size_t llcBytes{};
    size_t lineBytes{64};

    /// Reads /sys/devices/system/cpu/cpu0/cache/index*/.
    ///
    /// Only cpu0 is consulted: heterogeneous-core machines would need per-core
    /// detection, but a benchmark pinned to one core is the intended use and a
    /// cpu0 reading is then the honest one.  Levels are taken as
    /// (level, type) pairs so the L1 *instruction* cache is not mistaken for
    /// L1d, and the largest level seen becomes the LLC — that is what "last"
    /// means, and it avoids assuming the LLC is L3 (some parts stop at L2,
    /// others have an L4).  Unreadable sysfs leaves the field 0 and the
    /// consumers fall back rather than inventing a number.
    static CacheTopology detect() {
        CacheTopology t;
        for (int idx = 0; idx < 16; ++idx) {
            const std::string dir = "/sys/devices/system/cpu/cpu0/cache/index"
                                  + std::to_string(idx) + "/";
            const std::string levelStr = readSysfs(dir + "level");
            if (levelStr.empty()) continue;

            const int    level = std::atoi(levelStr.c_str());
            const std::string type = readSysfs(dir + "type");
            const size_t bytes = parseSize(readSysfs(dir + "size"));
            const size_t line  = static_cast<size_t>(std::atoi(readSysfs(dir + "coherency_line_size").c_str()));
            if (line > 0) t.lineBytes = line;
            if (bytes == 0) continue;

            const bool isInstruction = (type.rfind("Instruction", 0) == 0);
            if (level == 1 && !isInstruction)      t.l1dBytes = bytes;
            else if (level == 2 && !isInstruction) t.l2Bytes  = bytes;
            if (!isInstruction && level >= 2 && bytes > t.llcBytes) t.llcBytes = bytes;
        }
        return t;
    }

    /// Payload-byte targets for a cache-policy sweep, straddling the LLC.
    ///
    /// Hot and cold are the same measurement once the payload no longer fits in
    /// the LLC — a 4x-LLC payload is cold on every iteration whatever the policy
    /// says, and a payload inside L1d is hot again by the end of the first
    /// iteration no matter how hard it was flushed.  The interesting structure
    /// is therefore entirely at the boundary, and a sweep that does not cross it
    /// reports one regime twice.  The last point is deliberately past the LLC so
    /// the flat "always cold" asymptote is visible and the drop can be located.
    std::vector<size_t> defaultWorkingSetTargets() const {
        const size_t l1  = l1dBytes ? l1dBytes : size_t{48} * 1024;
        const size_t l2  = l2Bytes  ? l2Bytes  : size_t{1} * 1024 * 1024;
        const size_t llc = llcBytes ? llcBytes : size_t{16} * 1024 * 1024;
        std::vector<size_t> out{l1 / 4, l2 / 2, llc / 2, llc * 4};
        for (auto& v : out) if (v == 0) v = 4096;
        return out;
    }

    std::string describe() const {
        std::ostringstream os;
        os << "l1d=" << humanBytes(l1dBytes)
           << " l2=" << humanBytes(l2Bytes)
           << " llc=" << humanBytes(llcBytes)
           << " line=" << lineBytes << "B";
        return os.str();
    }

    /// Powers-of-two rendering, because that is how cache sizes are specified
    /// and "1.25MiB" must not be printed as "1MB".
    static std::string humanBytes(size_t bytes) {
        std::ostringstream os;
        if (bytes >= (size_t{1} << 20) && bytes % (size_t{1} << 20) == 0)
            os << (bytes >> 20) << "MiB";
        else if (bytes >= (size_t{1} << 20))
            os << (static_cast<double>(bytes) / 1048576.0) << "MiB";
        else if (bytes >= 1024 && bytes % 1024 == 0)
            os << (bytes >> 10) << "KiB";
        else
            os << bytes << "B";
        return os.str();
    }

private:
    static std::string readSysfs(const std::string& path) {
        std::ifstream in(path);
        if (!in) return {};
        std::string line;
        std::getline(in, line);
        return line;
    }

    /// sysfs writes sizes as "48K", "1280K", "18432K" — the suffix is binary.
    static size_t parseSize(const std::string& s) {
        if (s.empty()) return 0;
        size_t value = 0;
        size_t i = 0;
        for (; i < s.size() && s[i] >= '0' && s[i] <= '9'; ++i)
            value = value * 10 + static_cast<size_t>(s[i] - '0');
        if (i < s.size()) {
            if (s[i] == 'K' || s[i] == 'k') value *= 1024;
            else if (s[i] == 'M' || s[i] == 'm') value *= 1024 * 1024;
            else if (s[i] == 'G' || s[i] == 'g') value *= 1024 * 1024 * 1024;
        }
        return value;
    }
};

struct CachePolicy {
    CacheState  state{CacheState::Hot};
    EvictMethod method{EvictMethod::Auto};
    /// Payload size above which Auto stops using clflush.  0 => derive from the
    /// detected LLC.  clflush costs one instruction per 64 bytes of payload and
    /// its cost is charged to evict_ns, not to the measurement; but past a few
    /// LLC-fulls it dominates wall-clock time for no extra coldness, since a
    /// payload that large cannot be resident anyway.
    size_t clflushMaxBytes{0};
    /// Thrash buffer size as a multiple of the LLC.  Must exceed 1 by a real
    /// margin: a buffer exactly LLC-sized leaves a pseudo-LRU cache holding a
    /// large fraction of the victim data.
    double thrashMultiple{2.0};
    /// true => an undeliverable cold state throws.  false => the caller has
    /// explicitly accepted a weaker state than the label claims.
    bool strict{true};
};

/// The memory a cold state applies to.
///
/// `payload` is the encoded bytes.  `sink` is the decode output buffer, cooled
/// only for ColdAll — cooling it for ColdPayload would charge the write-allocate
/// misses of an empty buffer to the codec, which is a property of the harness's
/// buffer reuse, not of the encoding.  `codecInternal` comes from
/// Decoder::internalBuffers() and is what makes ColdAll meaningful for
/// index-carrying codecs.
struct EvictionTargets {
    std::span<const std::byte> payload;
    std::span<std::byte>       sink;
    std::vector<std::span<const std::byte>> codecInternal;
};

class CacheController {
public:
    CacheController(CachePolicy policy, CacheTopology topo)
        : policy_(policy), topo_(topo) {
        requestedMethod_ = policy.method;
        if (topo_.lineBytes == 0) topo_.lineBytes = 64;
        if (policy_.clflushMaxBytes == 0)
            policy_.clflushMaxBytes = (topo_.llcBytes ? topo_.llcBytes : size_t{16} << 20) * 4;

        policy_.method = resolveMethod(policy_.method);

        // The thrash buffer is allocated once, here.  Allocating it per
        // iteration would put a multi-megabyte mmap + first-touch page-fault
        // storm inside the eviction path, which shows up as evict_ns noise and,
        // worse, as page-table churn that outlives the eviction and perturbs the
        // timed region it was supposed to precede.
        if (policy_.method == EvictMethod::LlcThrash) {
            const double base = static_cast<double>(topo_.llcBytes ? topo_.llcBytes
                                                                   : size_t{16} << 20);
            const double mult = policy_.thrashMultiple > 1.0 ? policy_.thrashMultiple : 2.0;
            thrash_.assign(static_cast<size_t>(base * mult), std::byte{1});
        }
    }

    /// Read-touch every target into a volatile accumulator.
    ///
    /// This is why Hot is an active state: without it, "hot" means "whatever the
    /// previous iteration happened to leave behind", which depends on the
    /// iteration order and on the *previous cell's* footprint, and makes the
    /// first timed iteration of every cell a different measurement from the
    /// rest.  Touching at line granularity is enough — the point is residency,
    /// not reading every byte.
    void warm(const EvictionTargets& t) {
        const auto t0 = Clock::now();
        uint64_t acc = 0;
        acc += touch(t.payload);
        acc += touch(std::span<const std::byte>(t.sink.data(), t.sink.size()));
        for (const auto& b : t.codecInternal) acc += touch(b);
        volatileSink_ = acc;
        lastEvictNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
    }

    /// Per-iteration cache preparation.  Called by measure() strictly outside the
    /// t0/t1 window; its own cost is available as lastEvictNs().
    void prepare(const EvictionTargets& t) {
        latchSizeDependentMethod(t.payload.size());
        if (policy_.state == CacheState::Hot) { warm(t); return; }

        const auto t0 = Clock::now();
        switch (policy_.method) {
            case EvictMethod::Clflush:   flushTargets(t); break;
            case EvictMethod::LlcThrash: thrash();        break;
            case EvictMethod::None:      break;
            case EvictMethod::Auto:      break;  // unreachable: resolved in ctor
        }
        lastEvictNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
    }

    int64_t lastEvictNs() const { return lastEvictNs_; }

    /// The policy after Auto resolution — this, not the requested policy, is what
    /// a driver stamps into its `evict_method` column.
    const CachePolicy& effectivePolicy() const { return policy_; }

    std::string describe() const {
        std::string s = std::string(cacheStateName(policy_.state)) + "/"
                      + evictMethodName(policy_.method);
        if (policy_.method == EvictMethod::LlcThrash) {
            s += "(llc=" + CacheTopology::humanBytes(topo_.llcBytes)
               + ",buf=" + CacheTopology::humanBytes(thrash_.size()) + ")";
        } else if (policy_.method == EvictMethod::Clflush) {
            s += "(line=" + std::to_string(topo_.lineBytes) + "B)";
        }
        return s;
    }

private:
    using Clock = std::chrono::high_resolution_clock;

    /// Platform- and state-dependent part of Auto resolution.  The size-dependent
    /// part cannot happen here (no payload yet) and is latched on first prepare().
    EvictMethod resolveMethod(EvictMethod requested) const {
        if (policy_.state == CacheState::Hot) {
            // Hot performs no eviction, so any requested method would be a lie
            // in the output column.  Normalising to None keeps `evict_method`
            // honest for the hot rows.
            return EvictMethod::None;
        }

        if (requested == EvictMethod::Auto) {
            // ColdPayload defaults to clflush because it is the only method that
            // evicts *only* the encoded bytes: everything else in the process
            // (sink, index tables, the decoder's own working set) stays where the
            // previous iteration left it, which is exactly the isolation the
            // cold-payload row claims.
            //
            // ColdAll defaults to LlcThrash because codec-internal structures are
            // not reliably addressable from outside — internalBuffers() defaults
            // to empty and several codecs cannot enumerate their index without
            // restructuring, so a clflush-based cold-all would quietly leave the
            // index hot, which is the single structure the reordering-index
            // contribution is about.
            if (policy_.state == CacheState::ColdPayload) {
#if ENCODINGS_BENCH_HAVE_CLFLUSH
                return EvictMethod::Clflush;
#else
                return EvictMethod::LlcThrash;
#endif
            }
            return EvictMethod::LlcThrash;
        }

#if !ENCODINGS_BENCH_HAVE_CLFLUSH
        if (requested == EvictMethod::Clflush) {
            if (policy_.strict)
                throw std::runtime_error(
                    "CacheController: clflush requested but this is not an x86 target; "
                    "use --evict-method llc-thrash (a coarser cold state) rather than "
                    "reporting a cold label for a hot measurement");
            return EvictMethod::LlcThrash;
        }
#endif
        if (requested == EvictMethod::None && policy_.strict)
            throw std::runtime_error(
                std::string("CacheController: cache state '") + cacheStateName(policy_.state)
                + "' requested with evict-method 'none' — that combination measures a hot "
                  "cache under a cold label.  Pass strict=false only if the intent is to "
                  "quantify the eviction cost itself");
        return requested;
    }

    /// Escalate Auto from clflush to LlcThrash for payloads large enough that
    /// per-line flushing costs more than it buys.  Latched on the first payload
    /// seen: a controller whose method changed halfway through a sweep would emit
    /// rows from two different measurements under one `evict_method` value, which
    /// is the failure this whole class exists to prevent.
    void latchSizeDependentMethod(size_t payloadBytes) {
        if (sizeLatched_) {
            if (policy_.strict && requestedMethod_ == EvictMethod::Auto
                    && policy_.method == EvictMethod::Clflush
                    && payloadBytes > policy_.clflushMaxBytes)
                throw std::runtime_error(
                    "CacheController: payload grew past clflushMaxBytes after the eviction "
                    "method was latched; construct one controller per payload size so the "
                    "reported evict_method matches every row it labels");
            return;
        }
        sizeLatched_ = true;
        if (requestedMethod_ == EvictMethod::Auto
                && policy_.method == EvictMethod::Clflush
                && payloadBytes > policy_.clflushMaxBytes) {
            policy_.method = EvictMethod::LlcThrash;
            if (thrash_.empty()) {
                const double base = static_cast<double>(topo_.llcBytes ? topo_.llcBytes
                                                                       : size_t{16} << 20);
                const double mult = policy_.thrashMultiple > 1.0 ? policy_.thrashMultiple : 2.0;
                thrash_.assign(static_cast<size_t>(base * mult), std::byte{1});
            }
        }
    }

    void flushTargets(const EvictionTargets& t) {
        flushRange(t.payload);
        if (policy_.state == CacheState::ColdAll) {
            flushRange(std::span<const std::byte>(t.sink.data(), t.sink.size()));
            for (const auto& b : t.codecInternal) flushRange(b);
        }
#if ENCODINGS_BENCH_HAVE_CLFLUSH
        // clflush is not ordered with respect to later loads by itself; without
        // the fence the codec's first read can be satisfied from a line the flush
        // has not yet retired.
        _mm_mfence();
#endif
    }

    void flushRange(std::span<const std::byte> r) {
#if ENCODINGS_BENCH_HAVE_CLFLUSH
        const std::byte* p   = r.data();
        const std::byte* end = p + r.size();
        for (; p < end; p += topo_.lineBytes) _mm_clflush(p);
#else
        (void)r;
#endif
    }

    /// Stream a buffer larger than the LLC at line stride.
    ///
    /// This is a *coarse* instrument, and cold-all rows must be read as a lower
    /// bound on hotness rather than a clean payload-cold measurement: the sweep
    /// also flushes the TLB (it touches thousands of pages), retrains the branch
    /// predictors and hardware prefetchers, and pushes out the decoder's own
    /// stack and code lines.  The alternative — leaving the index hot — biases in
    /// the direction that flatters the contribution being measured, which is the
    /// worse of the two errors.
    void thrash() {
        uint64_t acc = 0;
        const size_t stride = topo_.lineBytes;
        for (size_t i = 0; i < thrash_.size(); i += stride)
            acc += static_cast<uint64_t>(thrash_[i]);
        volatileSink_ = acc;
    }

    uint64_t touch(std::span<const std::byte> r) const {
        uint64_t acc = 0;
        const size_t stride = topo_.lineBytes;
        for (size_t i = 0; i < r.size(); i += stride)
            acc += static_cast<uint64_t>(r[i]);
        return acc;
    }

    CachePolicy   policy_;
    CacheTopology topo_;
    /// The method as asked for, kept because only `Auto` may be escalated later:
    /// an explicitly requested clflush stays clflush or fails, it is never
    /// quietly turned into a thrash behind the label.
    EvictMethod   requestedMethod_{EvictMethod::Auto};
    std::vector<std::byte> thrash_;
    volatile uint64_t      volatileSink_{0};
    int64_t lastEvictNs_{0};
    bool    sizeLatched_{false};
};

}  // namespace encodings::benchmark
