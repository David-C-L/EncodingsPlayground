// Full-materialization decode throughput.
//
// The simplest access shape and the one every other driver is read against: the
// whole column, decoded into a caller-owned sink through the target's
// materializeAll() (Codec::decodeAllInto()).  A bulk row is the ceiling a range
// or gather row is compared to, so it has to be free of the two errors that
// inflate it.
//
// FIRST, THE SINK IS HOISTED.  The code this driver replaces called decodeAll()
// inside the timed loop, which allocates and zero-fills an 80 MB vector at
// N = 10M and charges both to decode throughput — at that size the zero-fill
// alone is a DRAM-bandwidth-bound write of the entire output, i.e. the same order
// as the decode it is hiding inside.  Everything timed here goes through
// DecodeHarness::bulk with one sink for the whole sweep.
//
// SECOND, PEAK HEAP IS NOT MEASURED IN THE TIMED LOOP.  ScopedAllocationTrack
// puts a CAS on every operator new, so an instrumented iteration is not a valid
// timing sample; the heap figure comes from one extra untimed decode of the
// already-cached artifact (never a re-encode — see CONVENTIONS section 2).
//
// THE WORKING-SET AXIS IS WHAT MAKES --cache-state MEAN ANYTHING.  Hot and cold
// are the same measurement once the payload no longer fits in the LLC: on this
// box the LLC is 18 MiB, so at N = 10M an int64 payload is already past it and
// both states measure a DRAM read.  A sweep that only reports N = 10M therefore
// reports one regime under two labels.  --working-set-targets takes payload-byte
// targets (default: CacheTopology::defaultWorkingSetTargets(), derived from the
// DETECTED cache sizes), picks for each the N whose payload comes closest, and
// records the achieved payload_bytes and payload_vs_llc so the crossover is
// visible in the output rather than assumed.
//
// Sweeps encoders x datasets x (--sizes and the resolved working-set N) x
// --cache-state.  --cache-state is repeatable so hot and cold-payload rows can
// come from one process and one artifact; a single occurrence behaves exactly as
// CONVENTIONS section 4 specifies.

#include "benchmark/AllocationTracker.hpp"
#include "benchmark/ArtifactCache.hpp"
#include "benchmark/CachePolicy.hpp"
#include "benchmark/Cli.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/DecodeHarness.hpp"
#include "benchmark/MeasureLoop.hpp"
#include "benchmark/ResultWriter.hpp"
#include "benchmark/RunManifest.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "benchmark/targets/PlaygroundTarget.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#ifdef __linux__
#include <malloc.h>
#endif

using namespace encodings;
using namespace encodings::benchmark;

namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);

// ─── Configuration ───────────────────────────────────────────────────────────

/// Every field is bound to a flag by ArgParser, so --help prints these values as
/// its defaults and cannot drift from them.
struct SweepConfig {
    size_t n{10'000'000};
    std::vector<size_t> sizes;               // empty → {n}
    std::vector<size_t> workingSetTargets;   // empty → topology default
    size_t maxN{10'000'000};
    size_t probeN{1'000'000};
    bool   skipWorkingSet{false};
    bool   skipPeakHeap{false};
    size_t iterations{5};
    size_t warmup{2};
    uint64_t seed{42};
    std::vector<std::string> cacheStates;    // empty → {hot}
    EvictMethod evictMethod{EvictMethod::Auto};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_decode_bulk.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Sweep axes:")
        .opt("n", cfg.n, "stream length in elements, used when --sizes is not given")
        .list("sizes", cfg.sizes, "explicit N axis (empty = just --n)")
        .list("working-set-targets", cfg.workingSetTargets,
              "payload-byte targets; empty = derived from the detected cache sizes")
        .flag("skip-working-set", cfg.skipWorkingSet, "drop the working-set axis entirely")
        .opt("max-n", cfg.maxN, "cap on any resolved N; a target needing more is truncated")
        .opt("probe-n", cfg.probeN, "N of the bytes-per-element probe encode");

    args.group("Measurement:")
        .opt("iterations", cfg.iterations, "timed repetitions per point")
        .opt("warmup", cfg.warmup, "untimed repetitions per point")
        .opt("seed", cfg.seed, "recorded in the manifest; all randomness derives from it")
        .flag("skip-peak-heap", cfg.skipPeakHeap,
              "omit the extra untimed decode that measures decode_peak_heap_bytes")
        // Repeatable, unlike the other drivers': hot and cold-payload of the SAME
        // artifact is the comparison the working-set axis exists to make, and
        // running it as two processes would re-encode and re-generate the data in
        // between (generated datasets reseed on reset(), so the second process
        // would not measure the same stream).
        .repeated("cache-state", cfg.cacheStates,
                  "hot | cold-payload | cold-all; repeat to sweep (empty = hot)")
        .enumOpt("evict-method", cfg.evictMethod,
                 {{"auto", EvictMethod::Auto},
                  {"clflush", EvictMethod::Clflush},
                  {"llc-thrash", EvictMethod::LlcThrash},
                  {"none", EvictMethod::None}},
                 "how a cold state is produced");

    args.group("Selection and output:")
        .repeated("dataset", cfg.datasetFilters, "only datasets whose name contains SUBSTR")
        .repeated("encoder", cfg.encoderFilters, "only encoders whose name contains SUBSTR")
        .flag("validate", cfg.validate, "round-trip check before measuring")
        .flag("dry-run", cfg.dryRun, "print the preflight summary and exit")
        .opt("output", cfg.output, "result file")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

bool parseCacheState(const std::string& text, CacheState& out) {
    if (text == "hot")          { out = CacheState::Hot;         return true; }
    if (text == "cold-payload") { out = CacheState::ColdPayload; return true; }
    if (text == "cold-all")     { out = CacheState::ColdAll;     return true; }
    return false;
}

// ─── Size plan ───────────────────────────────────────────────────────────────

/// One N to measure, and where it came from.
///
/// `targetBytes == 0` marks a point from the explicit --sizes axis: it has no
/// payload target, so `working_set_target_bytes` must reach the file as a null
/// rather than a 0 that a reader could take for "targeted an empty payload".
struct SizePoint {
    size_t n{};
    size_t targetBytes{0};
    bool   truncated{false};   ///< the target needed more than --max-n
};

/// Resolve a payload-byte target to an N, by measuring bytes/element and
/// correcting once.
///
/// A search would be exact and is not worth it: each probe is a full encode, and
/// for an AutoSIS codec that re-runs the cost-model DP.  Two encodes per target
/// get within a few percent for every codec here, and the row reports the
/// ACHIEVED payload_bytes, so the axis is honest at whatever precision the model
/// reached (CONVENTIONS section 6: report what was achieved, not what was asked).
size_t resolveN(size_t targetBytes, double bytesPerElem, size_t maxN) {
    if (bytesPerElem <= 0.0) return maxN;
    const double wanted = static_cast<double>(targetBytes) / bytesPerElem;
    const size_t n = static_cast<size_t>(std::llround(std::max(1024.0, wanted)));
    return std::min(n, maxN);
}

// ─── Result schema ───────────────────────────────────────────────────────────

/// The column list, in file order.  Names from `driver` through `skipped` are the
/// per-row commons CONVENTIONS section 6 requires of every driver; the rest are
/// this driver's own.  Timings and rates are nullable and a skipped cell leaves
/// them null — never 0, which would plot as an infinitely fast codec.
std::vector<ColumnSpec> bulkColumns() {
    return {
        stringCol("driver"),
        stringCol("dataset"),
        stringCol("encoding"),
        stringCol("family"),
        stringCol("variant"),
        intCol("is_sequential"),
        intCol("fast_skip"),
        intCol("random_access"),
        intCol("N"),
        intCol("seed"),
        stringCol("cache_state"),
        stringCol("evict_method"),
        intCol("evict_ns"),
        intCol("payload_bytes"),
        doubleCol("compression_ratio"),
        intCol("iterations"),
        intCol("warmup"),
        stringCol("contract"),
        stringCol("size_source"),
        intCol("working_set_target_bytes"),
        intCol("llc_bytes"),
        doubleCol("payload_vs_llc"),
        intCol("time_ns"),
        intCol("time_p90_ns"),
        intCol("time_min_ns"),
        doubleCol("decode_Meps"),
        doubleCol("decode_MBps"),
        intCol("decode_peak_heap_bytes"),
        intCol("substream_count"),
        // Variable arity: a SubIntSplit plan has as many sections as its bit
        // split, so a column per section would need the widest plan of the sweep
        // known before the schema is built, and would leave every other row with
        // nulls whose position carried the meaning.  One '|'-joined field keeps
        // the row width fixed and the arity self-describing.
        stringCol("substream_bulk_ns"),
        intCol("truncated"),
        intCol("skipped"),
    };
}

std::string joinNs(const std::vector<int64_t>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += '|';
        out += std::to_string(values[i]);
    }
    return out;
}

// ─── Peak heap ───────────────────────────────────────────────────────────────

/// Peak heap above baseline for ONE untimed materializeAll.
///
/// Untimed and outside the measured loop, on the artifact the sweep already
/// holds: the tracker's CAS per allocation would corrupt a timing sample, and
/// re-encoding to recover the figure would report a different encode's heap.  The
/// sink is already allocated and warm, so what this attributes to the codec is
/// exactly its own transient allocation (SubIntSplit's per-section buffers, a
/// dictionary expansion, Zstd's window) rather than the harness's output buffer.
///
/// malloc_trim first so the baseline is the allocator's floor rather than free
/// blocks a previous decode left behind, which would otherwise absorb the peak.
size_t measurePeakHeap(PlaygroundTarget<Elem>& target, std::span<Elem> sink) {
#ifdef __linux__
    malloc_trim(0);
#endif
    ScopedAllocationTrack track;
    target.materializeAll(sink.data(), sink.size());
    const size_t peak = track.stop();
    clobber(sink.data());
    return peak;
}

// ─── Preflight ───────────────────────────────────────────────────────────────

void printPreflight(const SweepConfig& cfg,
                    const std::vector<size_t>& sizes,
                    const std::vector<size_t>& targets,
                    const std::vector<CacheState>& states,
                    const std::vector<EncoderEntry<Elem>>& encoders,
                    const std::vector<DatasetEntry<Elem>>& datasets,
                    const CacheTopology& topo) {
    std::cout << "\n── Sweep configuration ──────────────────────────────────────\n"
              << "  element size   " << kElemSize << " bytes (int64_t)\n"
              << "  N axis         ";
    for (size_t i = 0; i < sizes.size(); ++i)
        std::cout << (i ? ", " : "") << sizes[i];
    std::cout << "\n  cache          " << topo.describe() << "\n"
              << "  working set    ";
    if (targets.empty()) {
        std::cout << "(disabled)";
    } else {
        for (size_t i = 0; i < targets.size(); ++i)
            std::cout << (i ? ", " : "") << CacheTopology::humanBytes(targets[i]);
        std::cout << "  payload targets, N resolved per (encoder, dataset)";
    }
    std::cout << "\n  cache states   ";
    for (size_t i = 0; i < states.size(); ++i)
        std::cout << (i ? ", " : "") << cacheStateName(states[i]);
    std::cout << "\n  evict method   " << evictMethodName(cfg.evictMethod)
              << "  (resolved per cell, so the latched method matches the row it labels)\n"
              << "  max N          " << cfg.maxN << "\n"
              << "  probe N        " << cfg.probeN << "\n"
              << "  peak heap      " << (cfg.skipPeakHeap ? "off" : "on, one untimed pass per cell")
              << "\n  iterations     " << cfg.iterations << " timed, " << cfg.warmup << " warmup\n";

    // Where hot and cold can differ at all, stated in payload bytes so a reader
    // can see before running whether the sweep straddles the boundary.
    if (topo.llcBytes != 0 && !targets.empty()) {
        std::cout << "\n── Working-set targets against the detected LLC ──────────────\n"
                  << "   target        x LLC   regime\n";
        for (size_t t : targets) {
            const double x = static_cast<double>(t) / static_cast<double>(topo.llcBytes);
            std::cout << "  " << std::setw(10) << CacheTopology::humanBytes(t)
                      << std::setw(9) << std::fixed << std::setprecision(3) << x << "  "
                      << (x < 0.5 ? "resident: hot and cold must differ"
                                  : (x < 1.5 ? "at the boundary" : "past LLC: both states are DRAM"))
                      << "\n";
        }
        std::cout << std::defaultfloat;
    }

    const size_t points = (sizes.size() + targets.size()) * states.size();
    std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
              << "  cells          " << points << " per (encoder, dataset)\n"
              << "  encoders       " << encoders.size() << "\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  result rows    " << points * encoders.size() * datasets.size() << "\n"
              << "  decode calls   "
              << points * encoders.size() * datasets.size() * (cfg.iterations + cfg.warmup)
              << "\n\n";
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_decode_bulk",
                   "Sweep full-materialization decode throughput over "
                   "(encoder, dataset, N, cache state), with an N axis derived from "
                   "payload-byte working-set targets.");
    bindArgs(args, cfg);
    switch (args.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    cfg.iterations = std::max<size_t>(1, cfg.iterations);
    cfg.maxN       = std::max<size_t>(1024, cfg.maxN);
    cfg.probeN     = std::clamp<size_t>(cfg.probeN, 1024, cfg.maxN);

    std::vector<size_t> sizes = cfg.sizes.empty() ? std::vector<size_t>{cfg.n} : cfg.sizes;
    for (auto& s : sizes) s = std::clamp<size_t>(s, 1, cfg.maxN);
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    const CacheTopology topo = CacheTopology::detect();
    std::vector<size_t> targets;
    if (!cfg.skipWorkingSet) {
        targets = cfg.workingSetTargets.empty() ? topo.defaultWorkingSetTargets()
                                                : cfg.workingSetTargets;
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    }

    std::vector<CacheState> states;
    if (cfg.cacheStates.empty()) {
        states.push_back(CacheState::Hot);
    } else {
        for (const auto& name : cfg.cacheStates) {
            CacheState s{};
            if (!parseCacheState(name, s)) {
                std::cerr << "ERROR: unknown --cache-state '" << name
                          << "' (expected hot|cold-payload|cold-all)\n";
                return 1;
            }
            states.push_back(s);
        }
    }

    auto encoders = applyFilters(allEncoders(), cfg.encoderFilters);
    auto datasets = applyFilters(int64Datasets(), cfg.datasetFilters);
    // A filter that matches nothing is an error, not an empty result file.
    if (encoders.empty()) { std::cerr << "ERROR: no encoders match --encoder filters\n"; return 1; }
    if (datasets.empty()) { std::cerr << "ERROR: no datasets match --dataset filters\n"; return 1; }

    // Constructed here only so --dry-run reports the policy that would have been
    // used and rejects an undeliverable cold state before any work; the
    // controllers that measure are built per cell (see below).
    for (CacheState s : states) {
        CachePolicy probe;
        probe.state  = s;
        probe.method = cfg.evictMethod;
        try {
            CacheController(probe, topo);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return 1;
        }
    }

    printPreflight(cfg, sizes, targets, states, encoders, datasets, topo);
    if (cfg.dryRun) {
        std::cout << "Dry run: no measurements taken.\n";
        return 0;
    }

    RunManifest manifest = RunManifest::capture("bench_decode_bulk", args.argvEcho());
    manifest.seed = cfg.seed;
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    for (const auto& e : encoders) manifest.encoders.push_back(e.name);
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    {
        std::string stateList;
        for (size_t i = 0; i < states.size(); ++i)
            stateList += (i ? "," : "") + std::string(cacheStateName(states[i]));
        manifest.extra["cache_states"] = stateList;
    }
    manifest.extra["evict_method_requested"] = evictMethodName(cfg.evictMethod);
    manifest.extra["peak_heap_pass"] = cfg.skipPeakHeap ? "off" : "on";
    // Before the sweep, so a killed run still has provenance.
    manifest.writeSidecar(cfg.output);

    ResultWriter writer(cfg.output, bulkColumns(), cfg.format);

    // One sink for the whole run, grown to the widest cell and never shrunk.
    // Hoisted out of the timed region: a fresh vector per timed call charges a
    // heap allocation plus an 80 MB zero-fill to decode throughput, which is the
    // defect this driver replaces.
    std::vector<Elem> sink;
    std::vector<std::string> validationFailures;

    DatasetCache<Elem> dataCache;
    ArtifactCache<Elem> artifacts;

    for (auto& ds : datasets) {
        std::cout << "══ Dataset: " << ds.name << " ══\n";

        for (auto& enc : encoders) {
            // ── Resolve this encoder's N axis ────────────────────────────────
            // Payload size per element is a property of (encoder, dataset), so a
            // byte target maps to a different N for every codec; that is the
            // point of targeting bytes rather than elements, since the cache does
            // not care how many elements the bytes decode to.
            std::vector<SizePoint> points;
            for (size_t n : sizes) points.push_back(SizePoint{n, 0, false});

            if (!targets.empty()) {
                std::cout << "  [" << enc.name << "] probing bytes/element at N="
                          << cfg.probeN << "..." << std::flush;
                double bytesPerElem = 0.0;
                try {
                    const auto probeHandle = dataCache.materialize(ds, cfg.probeN);
                    const auto& probe = artifacts.get(enc, probeHandle, EncodeMeasurement::None);
                    bytesPerElem = static_cast<double>(probe.payloadBytes)
                                 / static_cast<double>(probeHandle.n);
                } catch (const std::exception& e) {
                    std::cout << "\n";
                    std::cerr << "  WARNING: [" << enc.name << "] probe failed on '" << ds.name
                              << "': " << e.what() << "; working-set axis dropped\n";
                }
                std::cout << " " << bytesPerElem << " B/elem\n";

                for (size_t target : targets) {
                    if (bytesPerElem <= 0.0) continue;
                    size_t n = resolveN(target, bytesPerElem, cfg.maxN);
                    // One correction step: the first estimate comes from a
                    // different N, and compression ratio is not perfectly
                    // size-independent (block headers, dictionary reuse).
                    try {
                        const auto h = dataCache.materialize(ds, n);
                        const auto& a = artifacts.get(enc, h, EncodeMeasurement::None);
                        const double achieved = static_cast<double>(a.payloadBytes);
                        if (achieved > 0.0
                                && std::abs(achieved - static_cast<double>(target))
                                       / static_cast<double>(target) > 0.05) {
                            n = resolveN(target, achieved / static_cast<double>(h.n), cfg.maxN);
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "  WARNING: [" << enc.name << "] cannot size target "
                                  << CacheTopology::humanBytes(target) << " on '" << ds.name
                                  << "': " << e.what() << "\n";
                        continue;
                    }
                    // Truncated when the target needed more elements than --max-n
                    // allows: the cell is still measured and reported, with the
                    // achieved payload_bytes short of the target and truncated=1,
                    // because "this codec cannot reach a 4x-LLC payload within the
                    // element budget" is a result.
                    points.push_back(SizePoint{n, target, n >= cfg.maxN});
                }
            }

            std::sort(points.begin(), points.end(),
                      [](const SizePoint& a, const SizePoint& b) { return a.n < b.n; });

            for (const SizePoint& point : points) {
                typename DatasetCache<Elem>::Handle handle;
                try {
                    handle = dataCache.materialize(ds, point.n);
                } catch (const std::exception& e) {
                    // A source that cannot yield N elements is a skipped cell with
                    // a warning, not the end of the sweep (CONVENTIONS section 9).
                    std::cerr << "  WARNING: dataset '" << ds.name << "' cannot yield "
                              << point.n << " elements: " << e.what() << "\n";
                    continue;
                }
                const size_t n = handle.n;

                const auto& artifact = artifacts.get(enc, handle, EncodeMeasurement::None);
                const size_t payloadBytes = artifact.payloadBytes;
                const double ratio        = artifact.compressionRatio;
                const bool fastSkip       = artifact.fastSkip;
                const bool randomAccess   = artifact.randomAccess;

                PlaygroundTarget<Elem> target(*enc.codec);
                // A copy of the payload, not a second encode: the target owns the
                // bytes it decodes from and that the cache controller evicts.
                target.adopt(artifact.encoded);

                std::cout << "  [" << enc.name << "] N=" << n << " → " << payloadBytes
                          << " B (ratio " << ratio << "x)";
                if (point.targetBytes != 0)
                    std::cout << "  target " << CacheTopology::humanBytes(point.targetBytes);
                std::cout << std::flush;

                if (sink.size() < n) sink.assign(n, Elem{});
                const std::span<Elem> cellSink(sink.data(), n);

                bool excluded = false;
                if (cfg.validate) {
                    // A controller is needed to build a harness at all; validation
                    // does no timing, so a hot one is the cheapest correct choice.
                    CachePolicy vp;
                    vp.state = CacheState::Hot;
                    CacheController vcache(vp, topo);
                    DecodeHarness<PlaygroundTarget<Elem>> vharness(target, vcache);
                    std::string whyNot;
                    if (!vharness.validate(handle.data, whyNot)) {
                        // Drop the pair rather than aborting: one broken codec must
                        // not cost the sweep of the others.  Reported again in the
                        // summary, and sets the exit code.
                        std::cout << "\n";
                        std::cerr << "  [" << enc.name << "] EXCLUDED at N=" << n << ": "
                                  << whyNot << "\n";
                        validationFailures.push_back(enc.name + " on " + ds.name + " at N="
                                                     + std::to_string(n) + ": " + whyNot);
                        excluded = true;
                    }
                }
                if (excluded) continue;

                size_t peakHeap = 0;
                bool havePeakHeap = false;
                if (!cfg.skipPeakHeap) {
                    peakHeap = measurePeakHeap(target, cellSink);
                    havePeakHeap = true;
                }

                for (CacheState state : states) {
                    CachePolicy policy;
                    policy.state  = state;
                    policy.method = cfg.evictMethod;
                    // One controller per cell, deliberately.  Auto latches
                    // clflush-vs-thrash on the first payload it sees and refuses to
                    // relatch, so a controller shared across an N axis that crosses
                    // clflushMaxBytes would either throw mid-sweep or label two
                    // different methods with one name.
                    std::unique_ptr<CacheController> cache;
                    try {
                        cache = std::make_unique<CacheController>(policy, topo);
                    } catch (const std::exception& e) {
                        std::cerr << "\n  ERROR: " << e.what() << "\n";
                        return 1;
                    }

                    DecodeHarness<PlaygroundTarget<Elem>> harness(target, *cache);
                    MeasureSpec spec;
                    spec.iterations = cfg.iterations;
                    spec.warmup     = cfg.warmup;
                    const MeasureResult r = harness.bulk(cellSink, spec);
                    const TargetProfile prof = harness.profile();

                    const double timeNs = static_cast<double>(r.time.medianNs);
                    const double meps = timeNs > 0.0
                        ? static_cast<double>(n) / timeNs * 1e3 : 0.0;
                    const double mbps = timeNs > 0.0
                        ? static_cast<double>(n * kElemSize) / timeNs * 1e3 : 0.0;

                    auto row = writer.row();
                    row.set("driver", "bench_decode_bulk")
                        .set("dataset", ds.name)
                        .set("encoding", enc.name)
                        .set("family", enc.family)
                        .set("variant", enc.variant)
                        .set("is_sequential", enc.isSequential)
                        .set("fast_skip", fastSkip)
                        .set("random_access", randomAccess)
                        .set("N", n)
                        .set("seed", cfg.seed)
                        .set("cache_state", cacheStateName(cache->effectivePolicy().state))
                        .set("evict_method", evictMethodName(cache->effectivePolicy().method))
                        .set("evict_ns", r.evict.medianNs)
                        .set("payload_bytes", payloadBytes)
                        .set("compression_ratio", ratio)
                        .set("iterations", cfg.iterations)
                        .set("warmup", cfg.warmup)
                        // Which call was timed.  bench_decode_range carries the same
                        // column with "range_into", so the two files can be
                        // concatenated without losing what produced each row.
                        .set("contract", "all_into")
                        .set("size_source", point.targetBytes != 0 ? "working-set" : "sizes")
                        .setIf(point.targetBytes != 0, "working_set_target_bytes",
                               point.targetBytes)
                        .set("llc_bytes", topo.llcBytes)
                        .setIf(topo.llcBytes != 0, "payload_vs_llc",
                               static_cast<double>(payloadBytes)
                                   / static_cast<double>(topo.llcBytes ? topo.llcBytes : 1))
                        .set("time_ns", r.time.medianNs)
                        .set("time_p90_ns", r.time.p90Ns)
                        .set("time_min_ns", r.time.minNs)
                        .set("decode_Meps", meps)
                        .set("decode_MBps", mbps)
                        // Null, not 0, when the pass was skipped: a codec that
                        // genuinely allocates nothing reports 0 and must stay
                        // distinguishable from one that was never measured.
                        .setIf(havePeakHeap, "decode_peak_heap_bytes", peakHeap)
                        // Only the profiling instantiations (SIS_*_Prof, AutoSIS_*_Prof)
                        // report a sub-stream split; everything else leaves both null.
                        .setIf(!prof.subStreamBulkNs.empty(), "substream_count",
                               prof.subStreamBulkNs.size())
                        .setIf(!prof.subStreamBulkNs.empty(), "substream_bulk_ns",
                               joinNs(prof.subStreamBulkNs))
                        .set("truncated", point.truncated)
                        .set("skipped", false);
                    writer.write(std::move(row));
                }
                std::cout << "  done.\n";
                writer.flush();  // keep a long sweep inspectable while it runs
            }

            // The payload and the codec's internal state go before the next encoder
            // is timed, so its cache-state measurements are not polluted by this
            // one's working set.
            artifacts.evict(enc.name);
        }
        // Buffers are held for the whole dataset so every encoder measures the same
        // stream: generated sources reseed from random_device on reset(), so
        // re-materialising a (dataset, N) mid-sweep would silently change the data.
        dataCache.releaseAll();
    }

    writer.close();
    std::cout << "\nResults written to: " << std::filesystem::absolute(cfg.output) << std::endl;

    int exitCode = 0;
    if (!validationFailures.empty()) {
        std::cerr << "\n" << validationFailures.size()
                  << " encoder/dataset pair(s) failed validation and were excluded:\n";
        for (const auto& f : validationFailures) std::cerr << "  - " << f << "\n";
        exitCode = 2;
    }

    manifest.finishedAtIso = detail::isoNow();
    manifest.exitCode      = exitCode;
    manifest.writeSidecar(cfg.output);
    return exitCode;
}
