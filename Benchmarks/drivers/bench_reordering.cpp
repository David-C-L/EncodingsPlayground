// Reorderer x permutation format x inner codec.
//
// A reordering codec buys compression on the values by paying for a permutation:
// bytes to store it and time to invert it on every read.  This driver measures
// both halves of that trade over the four access shapes the other decode drivers
// each own one of (bulk, one contiguous range, a gather, a batch of point
// lookups), because the permutation cost falls on them very differently — a bulk
// read inverts the permutation once for N rows, a point lookup inverts it for
// one.
//
// WHY IT REPLACES reordering_benchmarks.cpp
//
// The old driver could not produce a usable number for two independent reasons.
// It pointed at Datasets/replay_data/pong_actions.parquet, which does not exist,
// so ParquetColumnGenerator threw during startup and no measurement ever ran
// (FINDINGS section 8).  And ReorderingCodec overrode none of decodeAllInto,
// decodeRangeInto or decodeGatherInto, so every access it timed ran Decoder's
// allocate-a-vector-and-copy fallback: the whole reordering dimension was being
// measured through a path that had nothing to do with reordering.  Both are fixed
// — the datasets come from registry/DatasetRegistry.hpp, and the overrides are
// implemented in ReorderingCodec — and --compare-fallback measures the two paths
// against each other so the difference is a reported number rather than a claim.
//
// THE PERMUTATION-FORMAT AXIS IS THE POINT
//
// FlatBitPacked and ChunkRelative answer "where did original row i go?" with a
// bit extraction.  DeltaBitPacked, DeltaZstd, DeltaLZ4, ValueGrouped and
// InverseEliasFano can only answer it by unpacking the entire permutation, which
// is O(N) per access.  perm_random_access_capable carries that distinction from
// permFormatSupportsRandomAccess, and the point-lookup rows are what turn it into
// a cost: a sequential format is measured with far fewer probes (--probes-seq)
// precisely because each probe unpacks the whole permutation, and `probes` is on
// every row so ns_per_probe stays comparable across that difference.
//
// Because a resident unpacked permutation would turn that per-access cost into a
// one-off and quietly change the answer, residency is a REGISTERED VARIANT
// (perm_cache_mode = per-call | resident), not a hidden optimisation: the two
// rows bound the same format from both ends.  internalBuffers() reports the
// resident permutation, so --cache-state cold-all actually cools it instead of
// measuring it warm (the trap FINDINGS section 6 records for SubIntSplit).
//
// COLUMN SEMANTICS worth stating once:
//   * unreorder_decode_all_ns is the codec's own timing of its LAST inversion,
//     i.e. per call.
//   * perm_lookup_decode_at_ns and perm_lookup_decode_range_ns are accumulators
//     over the timed iterations, divided here by `iterations` so every row is
//     per call.  For a point row a "call" is the whole probe batch.
//   * perm_invert_pct_of_access is the derived share of the measured access spent
//     inverting the permutation.  Under profiling it is an UPPER bound: the
//     accumulator is built from steady_clock pairs costing tens of ns each
//     (FINDINGS section 7), which is the same order as a single forwardAt.  The
//     +noprof registry entries exist to bound that instrumentation cost, and the
//     profiling_enabled column says which side of it a row is on.
//
// Everything timed goes through DecodeHarness; this file contains no timing code
// of its own.  See Benchmarks/drivers/CONVENTIONS.md, and section 3a before
// believing any absolute number.

#include "benchmark/ArtifactCache.hpp"
#include "benchmark/CachePolicy.hpp"
#include "benchmark/Cli.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/DecodeHarness.hpp"
#include "benchmark/GatherTraceGen.hpp"
#include "benchmark/MeasureLoop.hpp"
#include "benchmark/PointTraceGen.hpp"
#include "benchmark/ResultWriter.hpp"
#include "benchmark/RunManifest.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "benchmark/registry/ReorderingRegistry.hpp"
#include "benchmark/targets/PlaygroundTarget.hpp"
#include "generators/GeneratorUtils.hpp"
#include "generators/ShuffledGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace encodings;
using namespace encodings::benchmark;

namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);

// ─── Configuration ───────────────────────────────────────────────────────────

struct SweepConfig {
    size_t n{1'000'000};
    double rangeFrac{0.25};
    double sigma{0.1};
    size_t runLength{8};
    size_t maxRanges{65536};
    size_t seqMaxK{64};
    size_t probes{1u << 14};
    /// A probe on a sequential permutation format unpacks the whole permutation,
    /// so the same probe count would run for hours; the count is what makes the
    /// two comparable, and it is on every row.
    size_t probesSeq{8};
    size_t sortednessSample{2'000'000};
    uint64_t seed{42};
    size_t iterations{5};
    size_t warmup{2};
    CacheState  cacheState{CacheState::Hot};
    EvictMethod evictMethod{EvictMethod::Auto};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    bool compareFallback{false};
    bool forceFallback{false};
    bool noSortedness{false};
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_reordering.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Access shapes:")
        .opt("n", cfg.n, "stream length in elements")
        .opt("range-frac", cfg.rangeFrac, "contiguous range width, as a fraction of N")
        .opt("sigma", cfg.sigma, "gather selectivity inside the range window")
        .opt("run-length", cfg.runLength, "gather run length in elements")
        .opt("max-ranges", cfg.maxRanges, "cap on the gather range count, 0 = unbounded")
        .opt("seq-max-k", cfg.seqMaxK,
             "emit gather cells above this k as not viable for sequential codecs")
        .opt("probes", cfg.probes, "point lookups per batch, random-access permutation formats")
        .opt("probes-seq", cfg.probesSeq,
             "point lookups per batch when a probe must unpack the whole permutation");

    args.group("Measurement:")
        .opt("iterations", cfg.iterations, "timed repetitions per cell")
        .opt("warmup", cfg.warmup, "untimed repetitions per cell")
        .opt("seed", cfg.seed, "all randomness derives from this")
        .enumOpt("cache-state", cfg.cacheState,
                 {{"hot", CacheState::Hot},
                  {"cold-payload", CacheState::ColdPayload},
                  {"cold-all", CacheState::ColdAll}},
                 "cache state at the start of each timed iteration")
        .enumOpt("evict-method", cfg.evictMethod,
                 {{"auto", EvictMethod::Auto},
                  {"clflush", EvictMethod::Clflush},
                  {"llc-thrash", EvictMethod::LlcThrash},
                  {"none", EvictMethod::None}},
                 "how a cold state is produced");

    args.group("Selection and output:")
        .repeated("dataset", cfg.datasetFilters, "only datasets whose name contains SUBSTR")
        .repeated("encoder", cfg.encoderFilters, "only encoders whose name contains SUBSTR")
        .flag("compare-fallback", cfg.compareFallback,
              "measure each access twice, with and without the *Into overrides")
        .flag("force-fallback", cfg.forceFallback,
              "measure only Decoder's allocate-and-copy fallback path")
        .flag("no-sortedness", cfg.noSortedness, "skip the per-dataset sortedness sidecar")
        .opt("sortedness-sample", cfg.sortednessSample, "elements sampled for sortedness metrics")
        .flag("validate", cfg.validate, "round-trip and override-vs-fallback checks first")
        .flag("dry-run", cfg.dryRun, "print the preflight summary and exit")
        .opt("output", cfg.output, "result file")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

void resolveConfig(SweepConfig& cfg) {
    cfg.n          = std::max<size_t>(2, cfg.n);
    cfg.rangeFrac  = std::clamp(cfg.rangeFrac, 1e-6, 1.0);
    cfg.sigma      = std::clamp(cfg.sigma, 1e-6, 1.0);
    cfg.runLength  = std::max<size_t>(1, cfg.runLength);
    cfg.iterations = std::max<size_t>(1, cfg.iterations);
    cfg.probes     = std::max<size_t>(1, cfg.probes);
    cfg.probesSeq  = std::max<size_t>(1, cfg.probesSeq);
}

// ─── Datasets ────────────────────────────────────────────────────────────────

/// The registry's int64 sources, plus a shuffled TwitterSnowflake.
///
/// A reordering study needs a sorted/shuffled PAIR of the same column: the
/// permutation of an already-ordered column is near-identity and its formats all
/// look cheap, while the same values shuffled produce the unstructured
/// permutation the formats actually differ on.  The old driver had the pair
/// commented out; it is built here from the registry entry rather than from a
/// second hardcoded path, so it inherits the registry's "missing file is a
/// warning, not an exception" behaviour.
std::vector<DatasetEntry<Elem>> reorderingDatasets() {
    std::vector<DatasetEntry<Elem>> d = int64Datasets();
    for (const auto& entry : d) {
        if (entry.name != "TwitterSnowflake") continue;
        DatasetEntry<Elem> shuffled;
        shuffled.name = "TwitterSnowflakeShuffled";
        shuffled.generator =
            std::make_shared<encodings::datagen::ShuffledGenerator<Elem>>(entry.generator);
        shuffled.fileBacked = entry.fileBacked;
        shuffled.path = entry.path;
        d.push_back(std::move(shuffled));
        break;
    }
    return d;
}

// ─── Result schema ───────────────────────────────────────────────────────────

std::vector<ColumnSpec> reorderingColumns() {
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
        // Reordering axes.
        stringCol("reorderer"),
        stringCol("inner_codec"),
        stringCol("perm_format"),
        intCol("perm_random_access_capable"),
        stringCol("perm_cache_mode"),
        intCol("profiling_enabled"),
        intCol("into_overrides"),
        // Access shape.
        stringCol("access"),
        intCol("access_elems"),
        doubleCol("sigma_nominal"),
        doubleCol("sigma_achieved"),
        intCol("k_nominal"),
        intCol("k_actual"),
        intCol("probes"),
        doubleCol("ns_per_probe"),
        doubleCol("clock_overhead_ns"),
        // Timings.
        intCol("time_ns"),
        intCol("time_p90_ns"),
        intCol("time_min_ns"),
        doubleCol("elem_Meps"),
        // Permutation cost.
        intCol("permutation_bytes"),
        doubleCol("permutation_pct_of_encoded"),
        intCol("reorder_encode_ns"),
        intCol("unreorder_decode_all_ns"),
        intCol("perm_lookup_decode_at_ns"),
        intCol("perm_lookup_decode_range_ns"),
        doubleCol("perm_invert_pct_of_access"),
        intCol("internal_buffer_bytes"),
        intCol("truncated"),
        intCol("skipped"),
        stringCol("skip_reason"),
    };
}

// ─── Cell description ────────────────────────────────────────────────────────

enum class Access { Bulk, Range, Gather, Point };

const char* accessName(Access a) {
    switch (a) {
        case Access::Bulk:   return "bulk";
        case Access::Range:  return "range";
        case Access::Gather: return "gather";
        case Access::Point:  return "point";
    }
    return "unknown";
}

/// Everything about one measured cell that does not come from the codec.
struct AccessPlan {
    Access access{Access::Bulk};
    size_t begin{0}, end{0};       ///< range access
    GatherTrace gather;            ///< gather access
    PointTrace  points;            ///< point access
    size_t elems{0};               ///< elements the access produces
    bool viable{true};
    std::string whyNot;
};

std::vector<AccessPlan> buildPlans(const SweepConfig& cfg, size_t n, bool isSequential,
                                   bool permRandomAccess, bool artifactRandomAccess) {
    std::vector<AccessPlan> plans;

    {
        AccessPlan p;
        p.access = Access::Bulk;
        p.begin = 0;
        p.end = n;
        p.elems = n;
        plans.push_back(std::move(p));
    }

    const size_t span = std::clamp<size_t>(
        static_cast<size_t>(std::llround(cfg.rangeFrac * static_cast<double>(n))), 1, n);
    const size_t begin = (n - span) / 3;
    {
        AccessPlan p;
        p.access = Access::Range;
        p.begin = begin;
        p.end = begin + span;
        p.elems = span;
        plans.push_back(std::move(p));
    }

    {
        GatherAccessParams gp;
        gp.start       = begin;
        gp.span        = span;
        gp.selectivity = cfg.sigma;
        gp.runLength   = cfg.runLength;
        gp.gapModel    = GapModel::UniformDeterministic;
        gp.seed        = cfg.seed;
        gp.maxRanges   = cfg.maxRanges;

        AccessPlan p;
        p.access = Access::Gather;
        p.gather = buildGatherTrace(n, gp);
        p.elems  = p.gather.selectedRows;
        // Same rule as bench_decode_gather: a codec with no seek pays a full
        // inversion per range, and past a few dozen ranges one cell runs for
        // minutes.  "Not viable at this range count" is a result, so the row is
        // emitted with null timings rather than dropped.
        if (isSequential && p.gather.rangeCount > cfg.seqMaxK) {
            p.viable = false;
            p.whyNot = "sequential codec, k=" + std::to_string(p.gather.rangeCount) +
                       " > seq-max-k";
        }
        if (p.elems == 0) {
            p.viable = false;
            p.whyNot = "empty gather trace";
        }
        plans.push_back(std::move(p));
    }

    {
        PointTraceParams pp;
        pp.streamLength = n;
        // A sequential permutation format unpacks the whole permutation per probe.
        // Measuring it with the same probe count as a bit extraction would take
        // hours; measuring it not at all would leave the format axis with no point
        // number, which is the number that decides whether those formats are usable
        // for point access at all.
        pp.probes  = permRandomAccess ? cfg.probes : cfg.probesSeq;
        pp.pattern = PointPattern::Uniform;
        pp.seed    = cfg.seed;

        AccessPlan p;
        p.access = Access::Point;
        p.points = buildPointTrace(pp);
        p.elems  = p.points.indices.size();
        if (!artifactRandomAccess) {
            p.viable = false;
            p.whyNot = "codec reports no random access; decodeAt is a full decode per probe";
        }
        plans.push_back(std::move(p));
    }

    return plans;
}

// ─── Preflight ───────────────────────────────────────────────────────────────

void printPreflight(const SweepConfig& cfg,
                    const std::vector<EncoderEntry<Elem>>& encoders,
                    const std::vector<DatasetEntry<Elem>>& datasets,
                    const CacheController& cache) {
    const size_t span = std::clamp<size_t>(
        static_cast<size_t>(std::llround(cfg.rangeFrac * static_cast<double>(cfg.n))), 1, cfg.n);
    GatherAccessParams gp;
    gp.start       = (cfg.n - span) / 3;
    gp.span        = span;
    gp.selectivity = cfg.sigma;
    gp.runLength   = cfg.runLength;
    gp.gapModel    = GapModel::UniformDeterministic;
    gp.seed        = cfg.seed;
    gp.maxRanges   = cfg.maxRanges;
    const GatherTrace trace = buildGatherTrace(cfg.n, gp);

    std::cout << "\n── Sweep configuration ──────────────────────────────────────\n"
              << "  N              " << cfg.n << " elements ("
              << (cfg.n * kElemSize / (1024 * 1024)) << " MiB raw)\n"
              << "  range          [" << gp.start << ", " << (gp.start + span) << ")  span "
              << span << " (" << cfg.rangeFrac << " of N)\n"
              << "  gather         sigma " << cfg.sigma << " nominal, "
              << trace.selectivityAchieved << " achieved; k=" << trace.rangeCount
              << ", run " << trace.runLengthActual << ", " << trace.selectedRows << " rows"
              << (trace.clamped ? "  (clamped by --max-ranges)" : "") << "\n"
              << "  point probes   " << cfg.probes << " (random-access formats), "
              << cfg.probesSeq << " (formats needing a full unpack per probe)\n"
              << "  cache          " << cache.describe() << "\n"
              << "  iterations     " << cfg.iterations << " timed, " << cfg.warmup << " warmup\n"
              << "  into path      "
              << (cfg.compareFallback ? "override AND base-class fallback (two rows per access)"
                                      : (cfg.forceFallback ? "base-class fallback only"
                                                           : "override only"))
              << "\n";

    std::cout << "\n── Registered codecs ────────────────────────────────────────\n"
              << "  " << std::left << std::setw(42) << "encoding" << std::setw(22) << "reorderer"
              << std::setw(20) << "perm format" << std::setw(6) << "RA" << std::setw(10) << "cache"
              << "prof\n";
    for (const auto& e : encoders) {
        const auto* spec = reorderingSpec(e.name);
        std::cout << "  " << std::left << std::setw(42) << e.name.substr(0, 41);
        if (spec == nullptr) {
            std::cout << "(not a reordering entry)\n";
            continue;
        }
        std::cout << std::setw(22) << spec->reorderer
                  << std::setw(20)
                  << (spec->positionPermutation
                          ? encodings::reorderers::PermutationStore::formatName(spec->permFormat)
                          : "-")
                  << std::setw(6) << (spec->permRandomAccess ? "yes" : "no")
                  << std::setw(10) << encodings::reorderers::permResidencyToString(spec->residency)
                  << (spec->profiling ? "on" : "off") << "\n";
    }
    std::cout << std::right;

    const size_t pathsPerAccess = cfg.compareFallback ? 2 : 1;
    const size_t cells = encoders.size() * datasets.size() * 4 * pathsPerAccess;
    std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
              << "  accesses       4 (bulk, range, gather, point)\n"
              << "  encoders       " << encoders.size() << "\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  rows           " << cells << "\n"
              << "  decode calls   " << cells * (cfg.iterations + cfg.warmup) << "\n\n";
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Round-trip, plus the check that only exists in this driver: the *Into
/// overrides must agree with the base-class fallback element for element, on the
/// same codec and the same artifact.  The overrides are new code on the path every
/// row is measured through, and an override that is fast because it drops rows
/// would otherwise be reported as a speedup.
bool validateEncoder(DecodeHarness<PlaygroundTarget<Elem>>& harness,
                     PlaygroundTarget<Elem>& target,
                     std::span<const Elem> reference,
                     const std::vector<AccessPlan>& plans,
                     const ReorderingSpec* spec,
                     std::string& whyNot) {
    if (!harness.validate(reference, whyNot)) return false;

    for (const auto& plan : plans) {
        if (!plan.viable) continue;
        if (plan.access == Access::Gather &&
            !harness.validateGather(plan.gather.ranges, reference, whyNot)) {
            return false;
        }
        if (plan.access == Access::Point) {
            for (size_t index : plan.points.indices) {
                const auto v = target.pointRead(index);
                if (!v || *v != reference[index]) {
                    whyNot = "pointRead(" + std::to_string(index) + ") disagrees with the stream";
                    return false;
                }
            }
        }
    }

    if (spec == nullptr || !spec->setBypassIntoOverrides) return true;

    const size_t n = reference.size();
    std::vector<Elem> viaOverride(n, Elem{-1}), viaFallback(n, Elem{-2});
    target.materializeAll(viaOverride.data(), n);
    spec->setBypassIntoOverrides(true);
    target.materializeAll(viaFallback.data(), n);
    spec->setBypassIntoOverrides(false);
    if (viaOverride != viaFallback) {
        whyNot = "decodeAllInto override disagrees with the base-class fallback";
        return false;
    }

    for (const auto& plan : plans) {
        if (!plan.viable || plan.access != Access::Gather) continue;
        std::vector<Elem> a(plan.elems, Elem{-1}), b(plan.elems, Elem{-2});
        target.skipThenMaterialize(plan.gather.ranges, a.data(), plan.elems);
        spec->setBypassIntoOverrides(true);
        target.skipThenMaterialize(plan.gather.ranges, b.data(), plan.elems);
        spec->setBypassIntoOverrides(false);
        if (a != b) {
            whyNot = "decodeGatherInto override disagrees with the base-class fallback";
            return false;
        }
    }
    return true;
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_reordering",
                   "Sweep reorderer x permutation format x inner codec over the four "
                   "decode access shapes.");
    bindArgs(args, cfg);
    switch (args.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    resolveConfig(cfg);
    if (cfg.forceFallback && cfg.compareFallback) {
        std::cerr << "ERROR: --force-fallback and --compare-fallback are mutually exclusive\n";
        return 1;
    }

    // reorderingFamily() must be built before reorderingSpec() can answer, since
    // the specs are recorded as the entries are constructed.
    auto encoders = applyFilters(reorderingFamily(), cfg.encoderFilters);
    auto datasets = applyFilters(reorderingDatasets(), cfg.datasetFilters);
    if (encoders.empty()) { std::cerr << "ERROR: no encoders match --encoder filters\n"; return 1; }
    if (datasets.empty()) { std::cerr << "ERROR: no datasets match --dataset filters\n"; return 1; }

    const CacheTopology topo = CacheTopology::detect();
    CachePolicy policy;
    policy.state  = cfg.cacheState;
    policy.method = cfg.evictMethod;
    std::unique_ptr<CacheController> cache;
    try {
        cache = std::make_unique<CacheController>(policy, topo);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    printPreflight(cfg, encoders, datasets, *cache);
    if (cfg.dryRun) {
        std::cout << "Dry run: no measurements taken.\n";
        return 0;
    }

    const TimingSummary clock = measureClockOverhead(4096);

    RunManifest manifest = RunManifest::capture("bench_reordering", args.argvEcho());
    manifest.seed = cfg.seed;
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    for (const auto& e : encoders) manifest.encoders.push_back(e.name);
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    manifest.extra["cache_state"]        = cacheStateName(cache->effectivePolicy().state);
    manifest.extra["evict_method"]       = evictMethodName(cache->effectivePolicy().method);
    manifest.extra["clock_overhead_ns"]  = std::to_string(clock.medianNs);
    manifest.extra["into_path"]          = cfg.compareFallback ? "both"
                                          : (cfg.forceFallback ? "fallback" : "override");
    manifest.writeSidecar(cfg.output);

    ResultWriter writer(cfg.output, reorderingColumns(), cfg.format);

    std::vector<Elem> sink(cfg.n);
    std::vector<std::string> validationFailures;

    DatasetCache<Elem> dataCache;
    ArtifactCache<Elem> artifacts;

    // Sortedness is a property of the DATA, not of any codec, so it is computed
    // once per dataset and written to its own sidecar rather than repeated on
    // every row.  The cardinality it reports is also the viability test for MTF,
    // whose cost scales with the alphabet it has to shift per element.
    std::vector<std::pair<std::string, encodings::generators::SortednessMetrics>> sortedness;

    for (auto& ds : datasets) {
        std::cout << "══ Dataset: " << ds.name << " ══\n"
                  << "  loading " << cfg.n << " elements..." << std::flush;
        typename DatasetCache<Elem>::Handle handle;
        try {
            handle = dataCache.materialize(ds, cfg.n);
        } catch (const std::exception& e) {
            std::cout << "\n";
            std::cerr << "  WARNING: dataset '" << ds.name << "' skipped: " << e.what() << "\n";
            continue;
        }
        const size_t n = handle.n;
        std::cout << " got " << n << ".\n";

        size_t cardinality = 0;
        if (!cfg.noSortedness) {
            std::cout << "  sortedness metrics..." << std::flush;
            const size_t take = std::min(cfg.sortednessSample, n);
            const std::vector<Elem> sample(handle.data.begin(),
                                           handle.data.begin() + static_cast<ptrdiff_t>(take));
            auto metrics = encodings::generators::computeSortednessMetrics(sample, ds.name);
            cardinality = metrics.cardinality;
            std::cout << " cardinality " << cardinality << ", inversions "
                      << std::fixed << std::setprecision(4) << metrics.normalizedInversions
                      << std::defaultfloat << "\n";
            sortedness.emplace_back(ds.name, std::move(metrics));
        }

        for (auto& enc : encoders) {
            const ReorderingSpec* spec = reorderingSpec(enc.name);
            if (spec == nullptr) {
                std::cerr << "  [" << enc.name << "] no spec registered; skipped\n";
                continue;
            }

            // Encode-side viability, checked before encoding: BWT's forward
            // transform and MTF's alphabet shuffling are superlinear in N and in
            // cardinality respectively, so an unviable cell must be reported
            // without paying for the encode that would prove it.
            std::string encodeWhyNot;
            if (spec->maxViableN != 0 && n > spec->maxViableN) {
                encodeWhyNot = "encode is superlinear in N; N=" + std::to_string(n) + " > " +
                               std::to_string(spec->maxViableN);
            } else if (spec->maxViableCardinality != 0 && cardinality > spec->maxViableCardinality) {
                encodeWhyNot = "encode is superlinear in cardinality; " +
                               std::to_string(cardinality) + " > " +
                               std::to_string(spec->maxViableCardinality);
            }

            const auto emitRow = [&](const AccessPlan& plan, bool intoOverrides,
                                     const std::optional<MeasureResult>& measured,
                                     const EncodedArtifact<Elem>* artifact,
                                     const std::string& whyNot) {
                auto row = writer.row();
                row.set("driver", "bench_reordering")
                    .set("dataset", ds.name)
                    .set("encoding", enc.name)
                    .set("family", enc.family)
                    .set("variant", enc.variant)
                    .set("is_sequential", enc.isSequential)
                    .set("N", n)
                    .set("seed", cfg.seed)
                    .set("cache_state", cacheStateName(cache->effectivePolicy().state))
                    .set("evict_method", evictMethodName(cache->effectivePolicy().method))
                    .set("iterations", cfg.iterations)
                    .set("warmup", cfg.warmup)
                    .set("reorderer", spec->reorderer)
                    .set("inner_codec", spec->innerCodec)
                    .setIf(spec->positionPermutation, "perm_format",
                           encodings::reorderers::PermutationStore::formatName(spec->permFormat))
                    .set("perm_random_access_capable", spec->permRandomAccess)
                    .set("perm_cache_mode",
                         encodings::reorderers::permResidencyToString(spec->residency))
                    .set("profiling_enabled", spec->profiling)
                    .set("into_overrides", intoOverrides)
                    .set("access", accessName(plan.access))
                    .set("access_elems", plan.elems)
                    .set("clock_overhead_ns", static_cast<double>(clock.medianNs))
                    .set("truncated", plan.access == Access::Gather && plan.gather.clamped)
                    .set("skipped", !whyNot.empty());

                if (plan.access == Access::Gather) {
                    row.set("sigma_nominal", cfg.sigma)
                        .set("sigma_achieved", plan.gather.selectivityAchieved)
                        .set("k_nominal", plan.gather.rangeCountNominal)
                        .set("k_actual", plan.gather.rangeCount);
                } else {
                    row.setNull("sigma_nominal").setNull("sigma_achieved")
                        .setNull("k_nominal").setNull("k_actual");
                }
                row.setIf(plan.access == Access::Point, "probes", plan.points.indices.size());

                if (artifact != nullptr) {
                    row.set("fast_skip", artifact->fastSkip)
                        .set("random_access", artifact->randomAccess)
                        .set("payload_bytes", artifact->payloadBytes)
                        .set("compression_ratio", artifact->compressionRatio);
                    const auto metric = [&](const char* key) -> std::optional<double> {
                        const auto it = artifact->encodeCustomMetrics.find(key);
                        return it == artifact->encodeCustomMetrics.end()
                                   ? std::nullopt
                                   : std::optional<double>(it->second);
                    };
                    const auto permBytes = metric("permutation_bytes");
                    const auto permPct   = metric("permutation_pct_of_encoded");
                    const auto reorderNs = metric("reorder_encode_time_ns");
                    row.setIf(permBytes.has_value(), "permutation_bytes",
                              static_cast<int64_t>(permBytes.value_or(0)))
                        .setIf(permPct.has_value(), "permutation_pct_of_encoded",
                               permPct.value_or(0.0))
                        .setIf(reorderNs.has_value(), "reorder_encode_ns",
                               static_cast<int64_t>(reorderNs.value_or(0)));
                } else {
                    row.setNull("fast_skip").setNull("random_access").setNull("payload_bytes")
                        .setNull("compression_ratio").setNull("permutation_bytes")
                        .setNull("permutation_pct_of_encoded").setNull("reorder_encode_ns");
                }

                if (!whyNot.empty() || !measured.has_value()) {
                    row.set("skip_reason", whyNot)
                        .setNull("evict_ns").setNull("time_ns").setNull("time_p90_ns")
                        .setNull("time_min_ns").setNull("elem_Meps").setNull("ns_per_probe")
                        .setNull("unreorder_decode_all_ns").setNull("perm_lookup_decode_at_ns")
                        .setNull("perm_lookup_decode_range_ns")
                        .setNull("perm_invert_pct_of_access").setNull("internal_buffer_bytes");
                    writer.write(std::move(row));
                    return;
                }

                const MeasureResult& r = *measured;
                const double timeNs = static_cast<double>(r.time.medianNs);
                const double meps = timeNs > 0.0
                    ? static_cast<double>(plan.elems) / timeNs * 1e3 : 0.0;

                // Per call, not per iteration-sum: the codec accumulates these
                // across the timed iterations and the harness opened the window at
                // the first of them.
                const auto perCall = [&](int64_t accum) -> std::optional<int64_t> {
                    if (accum < 0) return std::nullopt;
                    return accum / static_cast<int64_t>(cfg.iterations);
                };
                const int64_t unreorderNs = enc.codec->unreorderDecodeAllTimeNs();
                const auto atNs    = perCall(enc.codec->permLookupDecodeAtAccumNs());
                const auto rangeNs = perCall(enc.codec->permLookupDecodeRangeAccumNs());

                // The share of THIS access that went on inverting the permutation:
                // whichever of the three the access shape actually used.
                std::optional<double> invertNs;
                if (plan.access == Access::Bulk && unreorderNs >= 0)
                    invertNs = static_cast<double>(unreorderNs);
                else if (plan.access == Access::Point && atNs.has_value())
                    invertNs = static_cast<double>(*atNs);
                else if ((plan.access == Access::Range || plan.access == Access::Gather) &&
                         rangeNs.has_value())
                    invertNs = static_cast<double>(*rangeNs);

                row.set("skip_reason", "")
                    .set("evict_ns", r.evict.medianNs)
                    .set("time_ns", r.time.medianNs)
                    .set("time_p90_ns", r.time.p90Ns)
                    .set("time_min_ns", r.time.minNs)
                    .set("elem_Meps", meps)
                    .setIf(plan.access == Access::Point, "ns_per_probe",
                           nsPerProbe(r, plan.points.indices.size()))
                    .setIf(unreorderNs >= 0, "unreorder_decode_all_ns", unreorderNs)
                    .setIf(atNs.has_value(), "perm_lookup_decode_at_ns", atNs.value_or(0))
                    .setIf(rangeNs.has_value(), "perm_lookup_decode_range_ns",
                           rangeNs.value_or(0))
                    .setIf(invertNs.has_value() && timeNs > 0.0, "perm_invert_pct_of_access",
                           invertNs.value_or(0.0) / (timeNs > 0.0 ? timeNs : 1.0) * 100.0);
                writer.write(std::move(row));
            };

            if (!encodeWhyNot.empty()) {
                std::cout << "  [" << enc.name << "] not viable: " << encodeWhyNot << "\n";
                for (const auto& plan : buildPlans(cfg, n, enc.isSequential,
                                                   spec->permRandomAccess, false)) {
                    emitRow(plan, !cfg.forceFallback, std::nullopt, nullptr, encodeWhyNot);
                }
                writer.flush();
                continue;
            }

            std::cout << "  [" << enc.name << "] encoding..." << std::flush;
            const EncodedArtifact<Elem>* artifact = nullptr;
            try {
                artifact = &artifacts.get(enc, handle, EncodeMeasurement::None);
            } catch (const std::exception& e) {
                std::cout << "\n";
                std::cerr << "  [" << enc.name << "] encode FAILED: " << e.what() << "\n";
                validationFailures.push_back(enc.name + " on " + ds.name + ": encode failed: " +
                                             e.what());
                continue;
            }
            std::cout << " " << artifact->payloadBytes << " B (ratio "
                      << artifact->compressionRatio << "x)\n";

            PlaygroundTarget<Elem> target(*enc.codec);
            target.adopt(artifact->encoded);
            target.declareRangeIntoOverride(true);
            DecodeHarness<PlaygroundTarget<Elem>> harness(target, *cache);

            const auto plans = buildPlans(cfg, n, enc.isSequential, spec->permRandomAccess,
                                          artifact->randomAccess);

            if (cfg.validate) {
                std::cout << "  [" << enc.name << "] validating..." << std::flush;
                std::string whyNot;
                if (!validateEncoder(harness, target, handle.data, plans, spec, whyNot)) {
                    std::cout << "\n";
                    std::cerr << "  [" << enc.name << "] EXCLUDED: " << whyNot << "\n";
                    validationFailures.push_back(enc.name + " on " + ds.name + ": " + whyNot);
                    artifacts.evict(enc.name);
                    continue;
                }
                std::cout << " ok\n";
            }

            std::cout << "  [" << enc.name << "] measuring" << std::flush;
            for (const auto& plan : plans) {
                if (!plan.viable) {
                    emitRow(plan, !cfg.forceFallback, std::nullopt, artifact, plan.whyNot);
                    continue;
                }

                // Both paths measured back to back on one artifact when asked, so
                // the only difference between the two rows is the code path.
                std::vector<bool> paths;
                if (cfg.compareFallback)      paths = {true, false};
                else if (cfg.forceFallback)   paths = {false};
                else                          paths = {true};

                for (bool useOverride : paths) {
                    if (spec->setBypassIntoOverrides) spec->setBypassIntoOverrides(!useOverride);

                    MeasureSpec msp;
                    msp.iterations = cfg.iterations;
                    msp.warmup     = cfg.warmup;

                    MeasureResult r;
                    switch (plan.access) {
                        case Access::Bulk:
                            r = harness.bulk(std::span<Elem>(sink.data(), n), msp);
                            break;
                        case Access::Range:
                            r = harness.range(plan.begin, plan.end, std::span<Elem>(sink), msp);
                            break;
                        case Access::Gather:
                            r = harness.gather(plan.gather.ranges, std::span<Elem>(sink),
                                               plan.elems, msp);
                            break;
                        case Access::Point:
                            r = harness.points(plan.points, msp);
                            break;
                    }
                    emitRow(plan, useOverride, r, artifact, "");
                    std::cout << "." << std::flush;
                }
                if (spec->setBypassIntoOverrides) spec->setBypassIntoOverrides(cfg.forceFallback);
            }
            std::cout << " done\n";

            // internal_buffer_bytes is deliberately not on the rows above: the
            // codec's resident state grows as the accesses run, so one number per
            // (encoder, dataset) taken after them all is the honest report of what
            // cold-all had to cool.
            size_t internalBytes = 0;
            for (const auto& b : target.internalBuffers()) internalBytes += b.size();
            std::cout << "  [" << enc.name << "] decoder-resident state: " << internalBytes
                      << " B in " << target.internalBuffers().size() << " span(s)\n";

            writer.flush();
            artifacts.evict(enc.name);
        }
    }

    writer.close();

    if (!sortedness.empty()) {
        std::filesystem::path sortednessPath = cfg.output;
        sortednessPath += ".sortedness.json";
        encodings::generators::writeSortednessMetricsJSON(sortedness, sortednessPath);
        std::cout << "\nSortedness metrics: " << std::filesystem::absolute(sortednessPath) << "\n";

        std::cout << "\n── Sortedness ───────────────────────────────────────────────\n"
                  << "  " << std::left << std::setw(28) << "dataset" << std::setw(10) << "lag1AC"
                  << std::setw(10) << "S_RLE" << std::setw(10) << "invNorm" << std::setw(14)
                  << "cardinality" << "zstdDelta\n";
        for (const auto& [name, m] : sortedness) {
            std::cout << "  " << std::left << std::setw(28) << name.substr(0, 27)
                      << std::setw(10) << std::fixed << std::setprecision(3)
                      << m.lag1Autocorrelation
                      << std::setw(10) << m.runLengthEntropyNormalized
                      << std::setw(10) << m.normalizedInversions
                      << std::setw(14) << m.cardinality;
            if (m.compressionAvailable) std::cout << m.compressionRatioDelta;
            else std::cout << "n/a";
            std::cout << "\n";
        }
        std::cout << std::defaultfloat << std::right;
    }

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
