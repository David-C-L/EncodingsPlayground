// Gather-access throughput heatmap.
//
// bench_decode_range sweeps contiguous range accesses (start, length) and so
// only ever measures a dense read.  This driver sweeps the *sparse* access that
// Nimble's TableScan actually issues after filter pushdown: a list of surviving
// row ranges with gaps that a codec must skip rather than materialize, driven
// through the target's skipThenMaterialize() (Codec::decodeGatherInto()).
//
// The swept space is (s0_frac, l, sigma, run_length):
//
//   s0_frac  start of the access window, as a fraction of (N - l)
//   l        width of the window in elements
//   sigma    selectivity — fraction of the window that is read
//   run_len  length of each contiguous run inside the window (4th axis,
//            single-valued by default so the default sweep is 3-D)
//
// The range count k is implied by (l, sigma, run_len) and is recorded, not swept.
// The sigma = 1 slice degenerates to a single contiguous range and reproduces the
// range-access baseline; --validate checks that against a direct range decode.
//
// Reading the sigma = 1 row against bench_decode_range: the two issue the same
// access but not the same call.  skipThenMaterialize() must write into a
// caller-owned buffer, so it goes through decodeRangeInto(), and for a codec that
// does not override that (Raw, AdaptiveFramedBitPrefix, Zstd, OpenZL) the base
// class implements it as decodeRange() plus a copy — one extra materialization the
// range-access driver never pays.  Those encoders therefore read as roughly 2x
// slower here at sigma = 1 even though the decode work is identical.  That gap is
// the cost of the gather API's buffer contract, not a measurement artifact, and it
// disappears for codecs that override decodeRangeInto (RawBitPacked, BlockFPE,
// BlockFORFPE, FPE, SubIntSplit).
//
// Everything measured here goes through DecodeHarness::gather, and every column
// is named rather than positional — see Benchmarks/drivers/CONVENTIONS.md for the
// contract this driver is the reference implementation of.

#include "benchmark/Axes.hpp"
#include "benchmark/ArtifactCache.hpp"
#include "benchmark/CachePolicy.hpp"
#include "benchmark/Cli.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/DecodeHarness.hpp"
#include "benchmark/GatherTraceGen.hpp"
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
    size_t lMin{1024};
    size_t lMax{0};            // 0 → n / 8, resolved after parsing
    size_t nL{16};
    size_t nS0{8};
    double sigmaMin{0.1};
    size_t nSigma{10};
    size_t minRangeSize{64 / kElemSize};   // one cache line's worth of elements
    size_t runMax{0};          // 0 → minRangeSize (single-valued 4th axis)
    size_t nRun{1};
    GapModel gapModel{GapModel::UniformDeterministic};
    uint64_t seed{42};
    size_t maxRanges{65536};
    size_t seqMaxK{64};
    size_t iterations{5};
    size_t warmup{2};
    CacheState  cacheState{CacheState::Hot};
    EvictMethod evictMethod{EvictMethod::Auto};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_decode_gather.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Sweep axes:")
        .opt("n", cfg.n, "total stream length in elements")
        .opt("l-min", cfg.lMin, "smallest access span")
        .opt("l-max", cfg.lMax, "largest access span (0 = N/8)")
        .opt("n-l", cfg.nL, "span steps, log-spaced")
        .opt("n-s0", cfg.nS0, "start-fraction steps, linear in [0,1]")
        .opt("sigma-min", cfg.sigmaMin, "lowest selectivity")
        .opt("n-sigma", cfg.nSigma, "selectivity steps, linear up to 1.0")
        .opt("min-range-size", cfg.minRangeSize, "run length in elements")
        .opt("run-max", cfg.runMax, "largest run length for the 4th axis (0 = min-range-size)")
        .opt("n-run", cfg.nRun, "run-length steps, log-spaced");

    args.group("Trace construction:")
        .enumOpt("gap-model", cfg.gapModel,
                 {{"uniform", GapModel::UniformDeterministic},
                  {"geometric", GapModel::Geometric}},
                 "gap placement model")
        .opt("seed", cfg.seed, "geometric-model seed")
        .opt("max-ranges", cfg.maxRanges, "cap on k per access, 0 = unbounded")
        .opt("seq-max-k", cfg.seqMaxK, "emit sequential-encoder cells above this k as not viable");

    args.group("Measurement:")
        .opt("iterations", cfg.iterations, "timed repetitions per point")
        .opt("warmup", cfg.warmup, "untimed repetitions per point")
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
        .flag("validate", cfg.validate, "run correctness and sigma=1 baseline checks first")
        .flag("dry-run", cfg.dryRun, "print the preflight summary and exit")
        .opt("output", cfg.output, "result file")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

/// Resolve defaults that depend on other flags, then sanity-check.
void resolveConfig(SweepConfig& cfg) {
    if (cfg.minRangeSize == 0) cfg.minRangeSize = 1;
    if (cfg.lMax == 0)  cfg.lMax  = std::max<size_t>(cfg.lMin, cfg.n / 8);
    if (cfg.runMax == 0) cfg.runMax = cfg.minRangeSize;
    cfg.lMin   = std::clamp<size_t>(cfg.lMin, 1, cfg.n);
    cfg.lMax   = std::clamp<size_t>(cfg.lMax, cfg.lMin, cfg.n);
    cfg.runMax = std::max(cfg.runMax, cfg.minRangeSize);
    cfg.nL     = std::max<size_t>(1, cfg.nL);
    cfg.nS0    = std::max<size_t>(1, cfg.nS0);
    cfg.nSigma = std::max<size_t>(1, cfg.nSigma);
    cfg.nRun   = std::max<size_t>(1, cfg.nRun);
    cfg.iterations = std::max<size_t>(1, cfg.iterations);
    cfg.sigmaMin   = std::clamp(cfg.sigmaMin, 1e-6, 1.0);
}

// ─── Result schema ───────────────────────────────────────────────────────────

/// The column list, in file order.
///
/// The names from `dataset` through `skipped` are consumed by
/// Benchmarks/plot_gather_heatmap.py and must not be renamed; the rest are the
/// per-row columns CONVENTIONS section 6 requires of every driver.  Timings and
/// rates are nullable and a not-viable cell leaves them null — never 0, which
/// would plot as a real measurement of an infinitely fast codec.
std::vector<ColumnSpec> gatherColumns() {
    return {
        stringCol("driver"),
        stringCol("dataset"),
        stringCol("encoding"),
        stringCol("family"),
        stringCol("variant"),
        intCol("is_sequential"),
        intCol("fast_skip"),
        intCol("random_access"),
        stringCol("gap_model"),
        intCol("seed"),
        intCol("N"),
        stringCol("cache_state"),
        stringCol("evict_method"),
        intCol("evict_ns"),
        intCol("payload_bytes"),
        doubleCol("compression_ratio"),
        intCol("iterations"),
        intCol("warmup"),
        doubleCol("s0_frac"),
        intCol("s0"),
        intCol("l"),
        doubleCol("sigma_nominal"),
        doubleCol("sigma_achieved"),
        intCol("run_length_nominal"),
        intCol("run_length_actual"),
        intCol("k_nominal"),
        intCol("k_actual"),
        intCol("selected_elems"),
        intCol("span_elems"),
        intCol("time_ns"),
        intCol("time_p90_ns"),
        intCol("time_min_ns"),
        doubleCol("sel_elem_Meps"),
        doubleCol("span_elem_Meps"),
        doubleCol("useful_MBps"),
        doubleCol("input_MBps"),
        intCol("gather_skip_ns"),
        intCol("gather_materialize_ns"),
        intCol("truncated"),
        intCol("skipped"),
    };
}

// ─── Preflight ───────────────────────────────────────────────────────────────

void printPreflight(const SweepConfig& cfg,
                    const std::vector<size_t>& spans,
                    const std::vector<double>& s0Fracs,
                    const std::vector<double>& sigmas,
                    const std::vector<size_t>& runs,
                    const std::vector<EncoderEntry<Elem>>& encoders,
                    const std::vector<DatasetEntry<Elem>>& datasets,
                    const CacheController& cache) {
    const size_t lMax = spans.back();

    std::cout << "\n── Sweep configuration ──────────────────────────────────────\n"
              << "  N              " << cfg.n << " elements (" << (cfg.n * kElemSize / (1024 * 1024)) << " MiB raw)\n"
              << "  element size   " << kElemSize << " bytes (int64_t)\n"
              << "  span l         " << spans.front() << " … " << lMax
              << "  (" << spans.size() << " log-spaced steps)\n"
              << "  start s0_frac  " << s0Fracs.front() << " … " << s0Fracs.back()
              << "  (" << s0Fracs.size() << " linear steps of N-l)\n"
              << "  selectivity    " << sigmas.front() << " … " << sigmas.back()
              << "  (" << sigmas.size() << " linear steps)\n"
              << "  run length     " << runs.front() << " … " << runs.back()
              << "  (" << runs.size() << " step" << (runs.size() == 1 ? "" : "s") << ")\n"
              << "  gap model      " << gapModelName(cfg.gapModel)
              << (cfg.gapModel == GapModel::Geometric ? "  (seed " + std::to_string(cfg.seed) + ")" : "") << "\n"
              << "  max ranges     " << cfg.maxRanges << "\n"
              << "  seq max k      " << cfg.seqMaxK << "\n"
              << "  cache          " << cache.describe() << "\n"
              << "  iterations     " << cfg.iterations << " timed, " << cfg.warmup << " warmup\n";

    // Range structure at l_max, so the user can sanity-check the construction
    // before committing to a full sweep.
    std::cout << "\n── Implied range structure at l = " << lMax
              << ", run_length = " << cfg.minRangeSize << " ─────────\n"
              << "   sigma       k    run_len    gap_len   sigma_achieved  note\n";
    for (double sigma : sigmas) {
        GatherAccessParams p;
        p.start        = 0;
        p.span         = lMax;
        p.selectivity  = sigma;
        p.runLength    = cfg.minRangeSize;
        p.gapModel     = cfg.gapModel;
        p.seed         = cfg.seed;
        p.maxRanges    = cfg.maxRanges;
        const GatherTrace t = buildGatherTrace(cfg.n, p);

        std::cout << "  " << std::fixed << std::setprecision(3) << std::setw(6) << sigma
                  << std::setw(9) << t.rangeCount
                  << std::setw(11) << t.runLengthActual
                  << std::setw(11) << t.gapLength
                  << std::setw(17) << std::setprecision(4) << t.selectivityAchieved
                  << "  ";
        if (t.rangeCount == 1)   std::cout << "contiguous (range-access baseline)";
        else if (t.clamped)      std::cout << "clamped by --max-ranges";
        std::cout << "\n";
    }
    std::cout << std::defaultfloat;

    const size_t points = spans.size() * s0Fracs.size() * sigmas.size() * runs.size();
    std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
              << "  grid points    " << points << " per (encoder, dataset)\n"
              << "  encoders       " << encoders.size() << "\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  CSV rows       " << points * encoders.size() * datasets.size() << "\n"
              << "  gather calls   "
              << points * encoders.size() * datasets.size() * (cfg.iterations + cfg.warmup) << "\n\n";
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Checks that (a) the codec round-trips the stream, (b) a gather returns exactly
/// the values at the selected indices for every (span, sigma) the sweep will
/// measure, and (c) the sigma = 1 slice is identical to a plain range decode — the
/// claim that makes the sigma = 1 row of the heatmap the range-access baseline.
///
/// The traces validated are built with the same parameters the sweep uses, so a
/// codec that only fails at a particular range structure is still caught.
bool validateEncoder(DecodeHarness<PlaygroundTarget<Elem>>& harness,
                     std::span<const Elem> reference,
                     const SweepConfig& cfg,
                     bool isSequential,
                     const std::vector<size_t>& spans,
                     const std::vector<double>& sigmas,
                     std::string& whyNot) {
    if (!harness.validate(reference, whyNot)) return false;

    for (size_t span : spans) {
        const size_t spanClamped = std::min(span, reference.size());
        const size_t s0 = (reference.size() > spanClamped) ? (reference.size() - spanClamped) / 3 : 0;
        for (double sigma : sigmas) {
            GatherAccessParams p;
            p.start       = s0;
            p.span        = spanClamped;
            p.selectivity = sigma;
            p.runLength   = cfg.minRangeSize;
            p.gapModel    = cfg.gapModel;
            p.seed        = cfg.seed;
            p.maxRanges   = cfg.maxRanges;
            const GatherTrace t = buildGatherTrace(reference.size(), p);
            if (t.selectedRows == 0) continue;
            // Same viability rule as the sweep: validating a cell the sweep will
            // not measure would cost minutes per span for a sequential codec.
            if (isSequential && t.rangeCount > cfg.seqMaxK) continue;
            if (!harness.validateGather(t.ranges, reference, whyNot)) return false;
        }

        if (!harness.validateGatherEqualsRange(s0, s0 + spanClamped, whyNot)) return false;
    }
    return true;
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_decode_gather",
                   "Sweep gather (selective row-range) decode throughput over "
                   "(s0_frac, l, sigma, run_length).");
    bindArgs(args, cfg);
    switch (args.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    resolveConfig(cfg);

    const auto spans   = logSpaced(cfg.lMin, cfg.lMax, cfg.nL);
    const auto runs    = logSpaced(cfg.minRangeSize, cfg.runMax, cfg.nRun);
    const auto s0Fracs = linSpaced(0.0, 1.0, cfg.nS0);
    const auto sigmas  = linSpaced(cfg.sigmaMin, 1.0, cfg.nSigma);

    auto encoders = applyFilters(allEncoders(), cfg.encoderFilters);
    auto datasets = applyFilters(int64Datasets(), cfg.datasetFilters);
    // A filter that matches nothing is an error, not an empty result file.
    if (encoders.empty()) { std::cerr << "ERROR: no encoders match --encoder filters\n"; return 1; }
    if (datasets.empty()) { std::cerr << "ERROR: no datasets match --dataset filters\n"; return 1; }

    const CacheTopology topo = CacheTopology::detect();
    CachePolicy policy;
    policy.state  = cfg.cacheState;
    policy.method = cfg.evictMethod;
    // Constructing the controller resolves Auto and rejects a cold state this
    // machine cannot deliver — before the preflight, so --dry-run reports the
    // policy that would actually have been used.
    std::unique_ptr<CacheController> cache;
    try {
        cache = std::make_unique<CacheController>(policy, topo);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    printPreflight(cfg, spans, s0Fracs, sigmas, runs, encoders, datasets, *cache);
    if (cfg.dryRun) {
        std::cout << "Dry run: no measurements taken.\n";
        return 0;
    }

    RunManifest manifest = RunManifest::capture("bench_decode_gather", args.argvEcho());
    manifest.seed = cfg.seed;
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    for (const auto& e : encoders) manifest.encoders.push_back(e.name);
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    manifest.extra["cache_state"]  = cacheStateName(cache->effectivePolicy().state);
    manifest.extra["evict_method"] = evictMethodName(cache->effectivePolicy().method);
    manifest.extra["gap_model"]    = gapModelName(cfg.gapModel);
    // Before the sweep, so a killed run still has provenance.
    manifest.writeSidecar(cfg.output);

    ResultWriter writer(cfg.output, gatherColumns(), cfg.format);

    // One sink for the whole run, sized for the sigma = 1 worst case.  Hoisting it
    // out of the timed loop keeps allocation out of the measurement — the
    // pre-refactor range driver allocated a fresh vector on every timed call and
    // so charged allocation to decode throughput.
    std::vector<Elem> sink(spans.back() + 1);
    std::vector<std::string> validationFailures;

    DatasetCache<Elem> dataCache;
    ArtifactCache<Elem> artifacts;

    for (auto& ds : datasets) {
        std::cout << "══ Dataset: " << ds.name << " ══\n"
                  << "  loading " << cfg.n << " elements..." << std::flush;
        typename DatasetCache<Elem>::Handle handle;
        try {
            handle = dataCache.materialize(ds, cfg.n);
        } catch (const std::exception& e) {
            // A source that cannot yield N elements is a filtered-out dataset with
            // a warning, not the end of the sweep (CONVENTIONS section 9).
            std::cout << "\n";
            std::cerr << "  WARNING: dataset '" << ds.name << "' skipped: " << e.what() << "\n";
            continue;
        }
        const size_t n = handle.n;
        std::cout << " got " << n << ".\n";
        if (n < spans.back()) {
            std::cerr << "  WARNING: dataset yielded " << n << " elements, less than l_max="
                      << spans.back() << "; spans are clipped to the stream.\n";
        }

        for (auto& enc : encoders) {
            std::cout << "  [" << enc.name << "] encoding..." << std::flush;
            // EncodeMeasurement::None: exactly one encode() per (encoder, dataset,
            // N).  For an AutoSIS codec a second encode would re-run cost-model
            // selection and could measure a different plan than the one reported.
            const auto& artifact = artifacts.get(enc, handle, EncodeMeasurement::None);
            const size_t payloadBytes = artifact.payloadBytes;
            const double ratio        = artifact.compressionRatio;
            const bool fastSkip       = artifact.fastSkip;
            const bool randomAccess   = artifact.randomAccess;

            PlaygroundTarget<Elem> target(*enc.codec);
            // A copy of the payload, not a second encode: the cache holds one
            // artifact at a time and the target owns the bytes it decodes from and
            // that the cache controller evicts.
            target.adopt(artifact.encoded);

            std::cout << " " << payloadBytes << " B (ratio " << ratio << "x"
                      << (fastSkip ? ", FastSkip" : "") << ")\n";

            DecodeHarness<PlaygroundTarget<Elem>> harness(target, *cache);

            if (cfg.validate) {
                std::cout << "  [" << enc.name << "] validating..." << std::flush;
                std::string whyNot;
                if (!validateEncoder(harness, handle.data, cfg, enc.isSequential, spans, sigmas,
                                     whyNot)) {
                    // Drop the encoder rather than aborting: one codec with a broken
                    // decode should not cost a multi-hour sweep of the others.  The
                    // failure is reported again in the summary and sets the exit code.
                    std::cout << "\n";
                    std::cerr << "  [" << enc.name << "] EXCLUDED from the sweep: " << whyNot
                              << "\n";
                    validationFailures.push_back(enc.name + " on " + ds.name + ": " + whyNot);
                    continue;
                }
                std::cout << " ok\n";
            }

            std::cout << "  [" << enc.name << "] sweeping..." << std::flush;
            size_t emitted = 0, measured = 0;

            for (size_t runLength : runs) {
                for (size_t span : spans) {
                    const size_t spanClamped = std::min(span, n);
                    for (double s0Frac : s0Fracs) {
                        const size_t s0 = static_cast<size_t>(
                            std::llround(s0Frac * static_cast<double>(n - spanClamped)));
                        for (double sigma : sigmas) {
                            GatherAccessParams p;
                            p.start       = s0;
                            p.span        = spanClamped;
                            p.selectivity = sigma;
                            p.runLength   = runLength;
                            p.gapModel    = cfg.gapModel;
                            p.seed        = cfg.seed;
                            p.maxRanges   = cfg.maxRanges;
                            const GatherTrace t = buildGatherTrace(n, p);
                            if (t.selectedRows == 0) continue;

                            // A sequential codec has no gather override, so it pays a
                            // full-payload decode per range.  Past a few dozen ranges a
                            // single point runs for minutes, so emit the cell as
                            // not-viable rather than either measuring it or dropping it —
                            // that is a real property of the encoding under gather, and
                            // the plots should show it as such.
                            const bool skipped =
                                enc.isSequential && t.rangeCount > cfg.seqMaxK;

                            auto row = writer.row();
                            row.set("driver", "bench_decode_gather")
                                .set("dataset", ds.name)
                                .set("encoding", enc.name)
                                .set("family", enc.family)
                                .set("variant", enc.variant)
                                .set("is_sequential", enc.isSequential)
                                .set("fast_skip", fastSkip)
                                .set("random_access", randomAccess)
                                .set("gap_model", gapModelName(cfg.gapModel))
                                .set("seed", cfg.seed)
                                .set("N", n)
                                .set("cache_state", cacheStateName(cache->effectivePolicy().state))
                                .set("evict_method", evictMethodName(cache->effectivePolicy().method))
                                .set("payload_bytes", payloadBytes)
                                .set("compression_ratio", ratio)
                                .set("iterations", cfg.iterations)
                                .set("warmup", cfg.warmup)
                                .set("s0_frac", s0Frac)
                                .set("s0", s0)
                                .set("l", spanClamped)
                                .set("sigma_nominal", sigma)
                                .set("sigma_achieved", t.selectivityAchieved)
                                .set("run_length_nominal", runLength)
                                .set("run_length_actual", t.runLengthActual)
                                .set("k_nominal", t.rangeCountNominal)
                                .set("k_actual", t.rangeCount)
                                .set("selected_elems", t.selectedRows)
                                .set("span_elems", spanClamped)
                                .set("truncated", t.clamped)
                                .set("skipped", skipped);

                            if (skipped) {
                                // Emitted, not dropped: "not viable at this range count"
                                // is a result, and a missing row is indistinguishable
                                // from a crash.  Every measured quantity is a typed
                                // null — a 0 here would plot as an infinitely fast
                                // codec.
                                row.setNull("evict_ns")
                                    .setNull("time_ns")
                                    .setNull("time_p90_ns")
                                    .setNull("time_min_ns")
                                    .setNull("sel_elem_Meps")
                                    .setNull("span_elem_Meps")
                                    .setNull("useful_MBps")
                                    .setNull("input_MBps")
                                    .setNull("gather_skip_ns")
                                    .setNull("gather_materialize_ns");
                                writer.write(std::move(row));
                                ++emitted;
                                continue;
                            }

                            MeasureSpec spec;
                            spec.iterations = cfg.iterations;
                            spec.warmup     = cfg.warmup;
                            const MeasureResult r = harness.gather(
                                t.ranges, std::span<Elem>(sink), t.selectedRows, spec);
                            const TargetProfile prof = harness.profile();

                            const double timeNs = static_cast<double>(r.time.medianNs);
                            const double selElemMeps  = timeNs > 0.0
                                ? static_cast<double>(t.selectedRows) / timeNs * 1e3 : 0.0;
                            const double spanElemMeps = timeNs > 0.0
                                ? static_cast<double>(spanClamped) / timeNs * 1e3 : 0.0;
                            const double usefulMBps   = timeNs > 0.0
                                ? static_cast<double>(t.selectedRows * kElemSize) / timeNs * 1e3 : 0.0;
                            // Compressed-input bandwidth, same model as
                            // bench_decode_range: a sequential codec must read the
                            // whole payload, a random-access one reads a proportional
                            // slice of it.
                            const double inputBytes = enc.isSequential
                                ? static_cast<double>(payloadBytes)
                                : static_cast<double>(t.selectedRows)
                                      * static_cast<double>(payloadBytes)
                                      / static_cast<double>(n);
                            const double inputMBps = timeNs > 0.0 ? inputBytes / timeNs * 1e3 : 0.0;

                            row.set("evict_ns", r.evict.medianNs)
                                .set("time_ns", r.time.medianNs)
                                .set("time_p90_ns", r.time.p90Ns)
                                .set("time_min_ns", r.time.minNs)
                                .set("sel_elem_Meps", selElemMeps)
                                .set("span_elem_Meps", spanElemMeps)
                                .set("useful_MBps", usefulMBps)
                                .set("input_MBps", inputMBps)
                                // -1 from the target means the codec has no distinct
                                // skip phase; that reaches the file as a null, never as
                                // a sentinel a reader could average.
                                .setIf(prof.gatherSkipNs >= 0, "gather_skip_ns", prof.gatherSkipNs)
                                .setIf(prof.gatherMaterializeNs >= 0, "gather_materialize_ns",
                                       prof.gatherMaterializeNs);
                            writer.write(std::move(row));
                            ++emitted;
                            ++measured;
                        }
                    }
                }
            }

            writer.flush();  // keep a long sweep inspectable while it runs
            // The payload and the codec's internal state go before the next
            // encoder is timed, so its cache-state measurements are not polluted
            // by this one's working set.
            artifacts.evict(enc.name);
            std::cout << " " << emitted << " rows (" << measured << " measured, "
                      << (emitted - measured) << " not viable).\n";
        }
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
