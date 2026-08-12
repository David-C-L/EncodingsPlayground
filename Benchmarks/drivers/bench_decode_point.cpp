// Point-lookup (decodeAt) latency.
//
// This driver replaces the random-access phase of BenchmarkRunner, which took 100
// shuffled lookups, hot only, and had three defects that are the reason this file
// exists.  Each is prevented structurally here rather than by care:
//
//   (1) It bracketed one ~50 ns decodeAt with two clock reads.  A
//       high_resolution_clock pair costs ~50 ns on this box, so the reported
//       number was mostly apparatus and ranked codecs by how well their cost hid
//       inside a constant.  Here the whole batch of --probes lookups is one timed
//       region (DecodeHarness::points), `ns_per_probe` is derived by division, and
//       a calibration row per run carries `clock_overhead_ns` so a reader can
//       check that the apparatus is negligible at the batch length used.
//   (2) It discarded the returned std::optional, making the lookup dead code the
//       optimizer is entitled to delete.  DecodeHarness::points accumulates every
//       value into a volatile sink inside the batch and clobbers it after.
//   (3) It early-`return`ed for codecs without random access, emitting a zero row
//       indistinguishable from an infinitely fast one.  Here a codec lacking
//       EncodingProperty::RandomAccess emits `viable=0` with every timing null.
//
// The swept space is (encoder, dataset, PointPattern, probes, cache_state) plus
// the pattern-specific knob of each pattern — see PointTraceGen.hpp for what each
// pattern is for, in particular why Zipf exists (it keeps a codec's auxiliary
// index resident, which is what separates index *work* from memory latency) and
// what the Strided aliasing caveat is.
//
// The two cold contracts in `cold_contract` are different measurements, not two
// settings of one:
//
//   cold-first-probe  evict, time exactly ONE probe, repeat over different
//                     indices.  The true cold lookup latency.  `evict_ns` exceeds
//                     `time_ns` by orders of magnitude here — one LLC thrash per
//                     nanosecond-scale probe — and that ratio is reported rather
//                     than hidden, because it is what says the number is a
//                     latency and not a throughput.  It is also the one contract
//                     where the clock pair is a real fraction of the measurement,
//                     so its rows must be read against the calibration row.
//   cold-batch        evict once, then time a batch of B probes.  Amortised
//                     first-touch, which converges to hot as B grows — hence it
//                     is swept over --probes, and the convergence is visible in
//                     the output.
//
// Hot needs no contract of its own: CacheState::Hot read-touches the payload
// before every timed iteration, and the warmup iterations run the real probe
// sequence, so exactly the lines the batch touches are the lines that are warm.

#include "benchmark/ArtifactCache.hpp"
#include "benchmark/CachePolicy.hpp"
#include "benchmark/Cli.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/DecodeHarness.hpp"
#include "benchmark/MeasureLoop.hpp"
#include "benchmark/PointTraceGen.hpp"
#include "benchmark/ResultWriter.hpp"
#include "benchmark/RunManifest.hpp"
#include "benchmark/TimingStats.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "benchmark/targets/PlaygroundTarget.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace encodings;
using namespace encodings::benchmark;

namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);

/// Empty timed regions used for the clock calibration.  Large enough that its own
/// median is stable to a nanosecond, cheap enough to pay unconditionally: the
/// calibration is what licenses every ns_per_probe in the file, so it is not
/// something a run may be configured out of.
constexpr size_t kClockCalibrationIterations = 20'000;

/// Probes used by the viability pre-check, whose only job is to reject a cell that
/// would run for minutes.  Small enough to be free, large enough that the
/// projection is not one outlier.
constexpr size_t kProbeCostSample = 256;

/// Which cold measurement a row reports.  Hot is not one of these: it is a
/// property of the cache state, and the column reads "hot" there.
enum class ColdContract { FirstProbe, Batch, Both };

// ─── Configuration ───────────────────────────────────────────────────────────

/// Every field is bound to a flag by ArgParser, so --help prints these values as
/// its defaults and cannot drift from them.
struct SweepConfig {
    size_t n{10'000'000};
    /// Swept, because cold-batch converges to hot as the batch grows and that
    /// convergence is the only way to read the amortisation off the file.
    std::vector<size_t> probes{1024, 16384, 262144};
    std::vector<std::string> patternNames;   ///< empty → every pattern
    std::vector<std::string> zipfThetaText;  ///< empty → {1.0}; parsed after parse()
    std::vector<size_t> meanRunLengths{8};
    std::vector<size_t> strides{97};
    double selectivity{0.1};
    bool   ascending{false};
    uint64_t seed{42};
    size_t iterations{5};
    size_t warmup{2};
    size_t maxCellMs{2000};
    CacheState   cacheState{CacheState::Hot};
    EvictMethod  evictMethod{EvictMethod::Auto};
    ColdContract coldContract{ColdContract::Both};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_decode_point.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Sweep axes:")
        .opt("n", cfg.n, "total stream length in elements")
        .list("probes", cfg.probes, "batch sizes in lookups")
        .repeated("pattern", cfg.patternNames,
                  "only these index patterns "
                  "[uniform|zipf|geometric_clustered|strided|sequential]")
        .repeated("zipf-theta", cfg.zipfThetaText, "skew for the zipf pattern (repeatable)")
        .list("mean-run-length", cfg.meanRunLengths, "run length for geometric_clustered")
        .list("stride", cfg.strides, "stride for the strided pattern");

    args.group("Trace construction:")
        .opt("selectivity", cfg.selectivity, "surviving fraction for geometric_clustered")
        .flag("ascending", cfg.ascending, "sort each probe list, holding the probe set fixed")
        .opt("seed", cfg.seed, "all trace randomness derives from this");

    args.group("Measurement:")
        .opt("iterations", cfg.iterations, "timed repetitions per cell")
        .opt("warmup", cfg.warmup, "untimed repetitions per cell")
        .opt("max-cell-ms", cfg.maxCellMs, "emit cells projected slower than this as not viable")
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
                 "how a cold state is produced")
        .enumOpt("cold-contract", cfg.coldContract,
                 {{"first-probe", ColdContract::FirstProbe},
                  {"batch", ColdContract::Batch},
                  {"both", ColdContract::Both}},
                 "which cold measurement(s) to take; ignored when --cache-state hot");

    args.group("Selection and output:")
        .repeated("dataset", cfg.datasetFilters, "only datasets whose name contains SUBSTR")
        .repeated("encoder", cfg.encoderFilters, "only encoders whose name contains SUBSTR")
        .flag("validate", cfg.validate, "round-trip and per-probe value checks first")
        .flag("dry-run", cfg.dryRun, "print the preflight summary and exit")
        .opt("output", cfg.output, "result file")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

const char* coldContractName(ColdContract c) {
    switch (c) {
        case ColdContract::FirstProbe: return "cold-first-probe";
        case ColdContract::Batch:      return "cold-batch";
        case ColdContract::Both:       return "both";
    }
    return "unknown";
}

/// The five patterns, resolved from --pattern by exact name.
///
/// Exact rather than substring: "sequential" is a substring of nothing else here,
/// but a future pattern name that contains another would silently widen every
/// existing invocation of this driver.
bool resolvePatterns(const std::vector<std::string>& names, std::vector<PointPattern>& out) {
    const PointPattern all[] = {PointPattern::Uniform, PointPattern::Zipf,
                                PointPattern::GeometricClustered, PointPattern::Strided,
                                PointPattern::SequentialProbe};
    if (names.empty()) {
        out.assign(std::begin(all), std::end(all));
        return true;
    }
    for (const std::string& want : names) {
        bool found = false;
        for (PointPattern p : all) {
            if (want == pointPatternName(p)) {
                if (std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p);
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "ERROR: unknown --pattern '" << want << "'\n";
            return false;
        }
    }
    return true;
}

bool parseThetas(const std::vector<std::string>& text, std::vector<double>& out) {
    if (text.empty()) {
        out.push_back(1.0);
        return true;
    }
    for (const std::string& s : text) {
        double v = 0.0;
        const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
        if (res.ec != std::errc{} || res.ptr != s.data() + s.size()) {
            std::cerr << "ERROR: --zipf-theta '" << s << "' is not a number\n";
            return false;
        }
        out.push_back(v);
    }
    return true;
}

void resolveConfig(SweepConfig& cfg) {
    if (cfg.probes.empty()) cfg.probes.push_back(1u << 14);
    if (cfg.meanRunLengths.empty()) cfg.meanRunLengths.push_back(8);
    if (cfg.strides.empty()) cfg.strides.push_back(97);
    for (size_t& p : cfg.probes) p = std::max<size_t>(1, p);
    std::sort(cfg.probes.begin(), cfg.probes.end());
    cfg.iterations  = std::max<size_t>(1, cfg.iterations);
    cfg.n           = std::max<size_t>(4, cfg.n);
    cfg.selectivity = std::clamp(cfg.selectivity, 1e-9, 1.0);
}

/// The theta the generator will actually sample with.
///
/// PointTraceGen nudges a theta within 1e-3 of 1.0 to 0.999 because the closed
/// form's alpha = 1/(1-theta) diverges at exactly 1 — and 1.0 is the struct
/// default, so the nominal and the achieved value differ on the most common
/// invocation of this driver.  The nudge is read back out of ZipfSampler rather
/// than restated here: `theta` there is a function of the request alone, so a
/// two-element sampler is enough to observe it, and a copy of the rule in this
/// file would be free to drift from the generator it claims to describe.
double achievedTheta(double requested, size_t streamLength) {
    const double clamped = std::clamp(requested, 0.0, 5.0);
    // The two escapes buildPointTrace takes before it ever builds a sampler; in
    // both the realized distribution is uniform, i.e. theta 0.
    if (clamped <= 0.0 || streamLength < 4) return 0.0;
    return detail::ZipfSampler(2, clamped).theta;
}

// ─── Sweep cells ─────────────────────────────────────────────────────────────

/// One (pattern, knob, probes) point, with its trace already built.
///
/// Traces are built once per stream length and reused across every encoder, for
/// the same reason DatasetCache exists: a Uniform trace is a shuffle of N, and
/// rebuilding it per encoder would both cost more than the measurements and — for
/// patterns whose realized structure depends on the draw — compare two encoders on
/// two different probe lists.
struct PointCell {
    PointTraceParams params;
    PointTrace       trace;
};

/// The knob axis that applies to a pattern, and only that one.  Sweeping stride
/// across Zipf cells would multiply the row count while describing one trace.
std::vector<PointCell> buildCells(const SweepConfig& cfg,
                                  const std::vector<PointPattern>& patterns,
                                  const std::vector<double>& thetas,
                                  size_t n) {
    std::vector<PointCell> cells;
    for (PointPattern pattern : patterns) {
        for (size_t probes : cfg.probes) {
            PointTraceParams base;
            base.streamLength = n;
            base.probes       = probes;
            base.pattern      = pattern;
            base.selectivity  = cfg.selectivity;
            base.seed         = cfg.seed;
            base.ascending    = cfg.ascending;

            switch (pattern) {
                case PointPattern::Zipf:
                    for (double theta : thetas) {
                        PointTraceParams p = base;
                        p.zipfTheta        = theta;
                        cells.push_back({p, buildPointTrace(p)});
                    }
                    break;
                case PointPattern::GeometricClustered:
                    for (size_t runLength : cfg.meanRunLengths) {
                        PointTraceParams p = base;
                        p.meanRunLength    = runLength;
                        cells.push_back({p, buildPointTrace(p)});
                    }
                    break;
                case PointPattern::Strided:
                    for (size_t stride : cfg.strides) {
                        PointTraceParams p = base;
                        p.stride           = stride;
                        cells.push_back({p, buildPointTrace(p)});
                    }
                    break;
                case PointPattern::Uniform:
                case PointPattern::SequentialProbe:
                    cells.push_back({base, buildPointTrace(base)});
                    break;
            }
        }
    }
    return cells;
}

/// The contracts a row will be produced under, given the cache state.
std::vector<const char*> resolveContracts(const SweepConfig& cfg, CacheState effective) {
    if (effective == CacheState::Hot) return {"hot"};
    switch (cfg.coldContract) {
        case ColdContract::FirstProbe: return {"cold-first-probe"};
        case ColdContract::Batch:      return {"cold-batch"};
        case ColdContract::Both:       return {"cold-first-probe", "cold-batch"};
    }
    return {"cold-batch"};
}

// ─── Result schema ───────────────────────────────────────────────────────────

/// The column list, in file order.
///
/// The names required of every driver by CONVENTIONS section 6 are all present;
/// the point-specific ones are the measured trace facts and the two labels
/// (`cold_contract`, `emulated_point_read`) without which two rows in this file
/// are not comparable.  Every timing is nullable and a non-viable cell leaves
/// them null — a 0 would plot as an infinitely fast lookup.
std::vector<ColumnSpec> pointColumns() {
    return {
        stringCol("driver"),
        stringCol("dataset"),
        stringCol("encoding"),
        stringCol("family"),
        stringCol("variant"),
        intCol("is_sequential"),
        intCol("fast_skip"),
        intCol("random_access"),
        intCol("emulated_point_read"),
        intCol("N"),
        intCol("seed"),
        stringCol("cache_state"),
        stringCol("evict_method"),
        stringCol("cold_contract"),
        stringCol("row_kind"),
        stringCol("pattern"),
        intCol("ascending"),
        intCol("probes_nominal"),
        intCol("probes_timed"),
        doubleCol("zipf_theta_nominal"),
        doubleCol("zipf_theta_actual"),
        intCol("mean_run_length"),
        doubleCol("selectivity"),
        intCol("stride"),
        intCol("distinct_indices"),
        doubleCol("distinct_fraction"),
        intCol("footprint_span"),
        intCol("payload_bytes"),
        doubleCol("compression_ratio"),
        intCol("iterations"),
        intCol("warmup"),
        intCol("evict_ns"),
        intCol("batch_ns"),
        intCol("time_ns"),
        intCol("time_p90_ns"),
        intCol("time_min_ns"),
        doubleCol("ns_per_probe"),
        doubleCol("clock_overhead_ns"),
        intCol("viable"),
        intCol("truncated"),
        intCol("skipped"),
    };
}

/// The knob columns, filled only for the pattern that consults them.
///
/// A stride reported on a Zipf row would be read as a swept parameter of that row;
/// it is a struct default the generator never looked at, so it is a typed null.
void setKnobs(ResultWriter::Row& row, const PointTraceParams& p, size_t n) {
    const bool isZipf    = p.pattern == PointPattern::Zipf;
    const bool isCluster = p.pattern == PointPattern::GeometricClustered;
    const bool isStride  = p.pattern == PointPattern::Strided;
    row.setIf(isZipf, "zipf_theta_nominal", p.zipfTheta)
        .setIf(isZipf, "zipf_theta_actual", achievedTheta(p.zipfTheta, n))
        .setIf(isCluster, "mean_run_length", p.meanRunLength)
        .setIf(isCluster, "selectivity", p.selectivity)
        .setIf(isStride, "stride", p.stride);
}

// ─── Preflight ───────────────────────────────────────────────────────────────

void printPreflight(const SweepConfig& cfg,
                    const std::vector<PointCell>& cells,
                    const std::vector<const char*>& contracts,
                    const std::vector<EncoderEntry<Elem>>& encoders,
                    const std::vector<DatasetEntry<Elem>>& datasets,
                    const CacheController& cache,
                    const TimingSummary& clock) {
    std::cout << "\n── Sweep configuration ──────────────────────────────────────\n"
              << "  N              " << cfg.n << " elements ("
              << (cfg.n * kElemSize / (1024 * 1024)) << " MiB raw)\n"
              << "  element size   " << kElemSize << " bytes (int64_t)\n"
              << "  probes         ";
    for (size_t p : cfg.probes) std::cout << p << " ";
    std::cout << "\n  probe order    " << (cfg.ascending ? "ascending (sorted)" : "as generated")
              << "\n  cache          " << cache.describe() << "\n  contracts      ";
    for (const char* c : contracts) std::cout << c << " ";
    std::cout << "\n  iterations     " << cfg.iterations << " timed, " << cfg.warmup << " warmup\n"
              << "  max cell       " << cfg.maxCellMs << " ms projected\n"
              << "  clock overhead " << clock.medianNs << " ns median (p90 " << clock.p90Ns
              << ", min " << clock.minNs << ") over " << kClockCalibrationIterations
              << " empty timed regions\n";

    // The ACHIEVED structure of every trace, not the requested one: a Zipf cell
    // that touches 300 of ten million rows and a Uniform cell that touches all of
    // them are measuring different things, and this table is where that is caught
    // before a sweep runs rather than after.
    std::cout << "\n── Achieved trace structure ─────────────────────────────────\n"
              << "  pattern               knob      probes   distinct  distinct_frac        span\n";
    for (const PointCell& c : cells) {
        std::string knob = "-";
        switch (c.params.pattern) {
            case PointPattern::Zipf:
                knob = "theta=" + std::to_string(achievedTheta(c.params.zipfTheta, cfg.n));
                knob.resize(std::min<size_t>(knob.size(), 11));
                break;
            case PointPattern::GeometricClustered:
                knob = "run=" + std::to_string(c.params.meanRunLength);
                break;
            case PointPattern::Strided:
                knob = "stride=" + std::to_string(c.params.stride);
                break;
            default: break;
        }
        std::cout << "  " << std::left << std::setw(22) << pointPatternName(c.params.pattern)
                  << std::setw(10) << knob << std::right << std::setw(9) << c.params.probes
                  << std::setw(11) << c.trace.distinctIndices << std::setw(16) << std::fixed
                  << std::setprecision(6) << c.trace.distinctFraction << std::defaultfloat
                  << std::setw(12) << c.trace.footprintSpan << "\n";
    }
    std::cout << std::left << std::setw(0) << std::right;

    size_t totalProbes = 0;
    for (const PointCell& c : cells) totalProbes += c.trace.indices.size();
    const size_t pairs = encoders.size() * datasets.size();
    std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
              << "  cells          " << cells.size() << " per (encoder, dataset, contract)\n"
              << "  encoders       " << encoders.size() << "\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  CSV rows       " << cells.size() * pairs * contracts.size() << " + 1 calibration\n"
              << "  point reads    up to "
              << totalProbes * pairs * contracts.size() * (cfg.iterations + cfg.warmup)
              << " (cold-first-probe times one probe per iteration, so its share is far lower)\n\n";
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Round-trip, then the claim this driver actually depends on: that pointRead(i)
/// returns row i of the ORIGINAL stream.
///
/// The bulk check alone does not establish that.  A codec that stores rows in a
/// permuted order can materialize correctly and still answer a positional query in
/// its internal order — FPE's NoIndex variant is exactly that — and without this
/// check its rows would be the fastest in the table and meaningless.  The indices
/// checked come from the traces the sweep will measure, so a codec that only fails
/// in the region a pattern concentrates on is still caught.
bool validateEncoder(DecodeHarness<PlaygroundTarget<Elem>>& harness,
                     PlaygroundTarget<Elem>& target,
                     std::span<const Elem> reference,
                     const std::vector<PointCell>& cells,
                     std::string& whyNot) {
    if (!harness.validate(reference, whyNot)) return false;

    const size_t n = reference.size();
    // Endpoints first: an off-by-one in a positional index shows up there and
    // nowhere in a sampled interior.
    for (size_t index : {size_t{0}, n / 2, n - 1}) {
        const auto v = target.pointRead(index);
        if (!v) {
            whyNot = std::format("pointRead({}) returned no value", index);
            return false;
        }
        if (*v != reference[index]) {
            whyNot = std::format("pointRead({}) = {}, expected {}", index,
                                 static_cast<int64_t>(*v),
                                 static_cast<int64_t>(reference[index]));
            return false;
        }
    }

    // A cap per cell, not the whole trace: at 262144 probes per cell this is the
    // difference between a validation pass and a second full sweep.
    constexpr size_t kPerCell = 2048;
    for (const PointCell& c : cells) {
        const size_t step = std::max<size_t>(1, c.trace.indices.size() / kPerCell);
        for (size_t i = 0; i < c.trace.indices.size(); i += step) {
            const size_t index = c.trace.indices[i];
            const auto   v     = target.pointRead(index);
            if (!v || *v != reference[index]) {
                whyNot = std::format("pointRead({}) under pattern {} = {}, expected {}", index,
                                     pointPatternName(c.params.pattern),
                                     v ? static_cast<int64_t>(*v) : 0,
                                     static_cast<int64_t>(reference[index]));
                return false;
            }
        }
    }
    return true;
}

// ─── Measurement ─────────────────────────────────────────────────────────────

/// cold-first-probe: evict, time ONE probe, repeat over different indices.
///
/// Built out of `iterations` single-probe calls to the harness rather than by
/// giving the harness a one-element trace and `iterations` repetitions, because
/// repeating the same index would measure the second touch of a line the eviction
/// was supposed to have removed.  The indices are drawn spread across the cell's
/// own trace, so the contract measures the cold cost of the pattern under test and
/// not of a uniform one.
///
/// The per-iteration samples are summarized here with the same summarize() every
/// other row uses, so a cold-first-probe row's median/p90/min mean what they mean
/// elsewhere.
MeasureResult measureFirstProbe(DecodeHarness<PlaygroundTarget<Elem>>& harness,
                                const PointTrace& trace,
                                const SweepConfig& cfg) {
    PointTrace single;
    single.indices.resize(1);

    std::vector<int64_t> timeSamples, evictSamples;
    timeSamples.reserve(cfg.iterations);
    evictSamples.reserve(cfg.iterations);

    const size_t step = std::max<size_t>(1, trace.indices.size() / cfg.iterations);
    for (size_t i = 0; i < cfg.iterations; ++i) {
        single.indices[0] = trace.indices[(i * step) % trace.indices.size()];

        MeasureSpec spec;
        spec.iterations = 1;
        // Warmup only on the first sub-call: it faults in the sink page and
        // resolves any lazily-parsed header, which is a one-off cost and not the
        // cold lookup being reported.  Repeating it per iteration would warm the
        // codec's structures immediately before an eviction meant to cool them.
        spec.warmup = (i == 0) ? cfg.warmup : 0;

        const MeasureResult r = harness.points(single, spec);
        timeSamples.push_back(r.time.medianNs);
        evictSamples.push_back(r.evict.medianNs);
    }

    MeasureResult out;
    out.time          = summarize(timeSamples);
    out.evict         = summarize(evictSamples);
    out.iterationsRun = cfg.iterations;
    return out;
}

/// Projected wall time of a cell, from a short untimed sample.
///
/// A codec may declare RandomAccess and still implement decodeAt in O(N) — at
/// 262144 probes over ten million rows that is hours in one cell, and a sweep that
/// discovers this after four hours has produced nothing.  The projection is
/// deliberately crude: it only has to separate "nanoseconds" from "minutes".
double projectCellMs(DecodeHarness<PlaygroundTarget<Elem>>& harness,
                     const PointTrace& trace,
                     size_t probesTimed,
                     size_t iterations) {
    PointTrace sample;
    const size_t take = std::min(kProbeCostSample, trace.indices.size());
    sample.indices.assign(trace.indices.begin(),
                          trace.indices.begin() + static_cast<std::ptrdiff_t>(take));

    MeasureSpec spec;
    spec.iterations = 1;
    // One warmup iteration is mandatory here, not politeness.  Without it the
    // projection is the first work ever done on a freshly adopted artifact and
    // pays first-touch page faults on tens of megabytes plus the codec's lazy
    // header parse — enough to overestimate a 40 ns lookup by two orders of
    // magnitude and reject a perfectly viable cell.  That happened: identical
    // sweeps differed only in whether --validate had incidentally warmed the
    // payload first, and the unvalidated ones emitted skipped=1 for every hot
    // uniform cell.
    spec.warmup = 1;
    const MeasureResult r = harness.points(sample, spec);
    const double perProbeNs = nsPerProbe(r, take);
    return perProbeNs * static_cast<double>(probesTimed) * static_cast<double>(iterations) / 1e6;
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_decode_point",
                   "Sweep point-lookup (decodeAt) latency over "
                   "(pattern, probes, cache state), batch-timed.");
    bindArgs(args, cfg);
    switch (args.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    resolveConfig(cfg);

    std::vector<PointPattern> patterns;
    if (!resolvePatterns(cfg.patternNames, patterns)) return 1;
    std::vector<double> thetas;
    if (!parseThetas(cfg.zipfThetaText, thetas)) return 1;

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
    const CacheState effectiveState = cache->effectivePolicy().state;
    const std::vector<const char*> contracts = resolveContracts(cfg, effectiveState);

    // Measured before any codec is loaded, so the calibration is not taken while a
    // multi-megabyte payload is resident and competing for the same caches.
    const TimingSummary clock = measureClockOverhead(kClockCalibrationIterations);

    std::vector<PointCell> cells = buildCells(cfg, patterns, thetas, cfg.n);
    printPreflight(cfg, cells, contracts, encoders, datasets, *cache, clock);
    if (cfg.dryRun) {
        std::cout << "Dry run: no measurements taken.\n";
        return 0;
    }

    RunManifest manifest = RunManifest::capture("bench_decode_point", args.argvEcho());
    manifest.seed = cfg.seed;
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    for (const auto& e : encoders) manifest.encoders.push_back(e.name);
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    manifest.extra["cache_state"]       = cacheStateName(effectiveState);
    manifest.extra["evict_method"]      = evictMethodName(cache->effectivePolicy().method);
    manifest.extra["cold_contract"]     = coldContractName(cfg.coldContract);
    manifest.extra["clock_overhead_ns"] = std::to_string(clock.medianNs);
    // Before the sweep, so a killed run still has provenance.
    manifest.writeSidecar(cfg.output);

    ResultWriter writer(cfg.output, pointColumns(), cfg.format);

    // The calibration row, once per run and first in the file.
    //
    // It is a row rather than a log line because the question it answers — "how
    // much of this ns_per_probe is the clock?" — is asked of the data long after
    // the run, and a number that only exists in a terminal cannot be asked.  It
    // carries the clock's own median/p90/min in the timing columns; probes_timed
    // is 1 because one empty timed region is what was measured.
    {
        auto row = writer.row();
        row.set("driver", "bench_decode_point")
            .set("dataset", "-")
            .set("encoding", "__clock__")
            .set("family", "calibration")
            .set("variant", "")
            .set("is_sequential", false)
            .set("fast_skip", false)
            .set("random_access", false)
            .set("emulated_point_read", false)
            .set("N", cfg.n)
            .set("seed", cfg.seed)
            .set("cache_state", cacheStateName(effectiveState))
            .set("evict_method", evictMethodName(cache->effectivePolicy().method))
            .set("cold_contract", "-")
            .set("row_kind", "calibration")
            .set("pattern", "-")
            .set("ascending", cfg.ascending)
            .set("probes_nominal", kClockCalibrationIterations)
            .set("probes_timed", size_t{1})
            .setNull("zipf_theta_nominal")
            .setNull("zipf_theta_actual")
            .setNull("mean_run_length")
            .setNull("selectivity")
            .setNull("stride")
            .setNull("distinct_indices")
            .setNull("distinct_fraction")
            .setNull("footprint_span")
            .setNull("payload_bytes")
            .setNull("compression_ratio")
            .set("iterations", kClockCalibrationIterations)
            .set("warmup", size_t{0})
            .setNull("evict_ns")
            .set("batch_ns", clock.medianNs)
            .set("time_ns", clock.medianNs)
            .set("time_p90_ns", clock.p90Ns)
            .set("time_min_ns", clock.minNs)
            .set("ns_per_probe", static_cast<double>(clock.medianNs))
            .set("clock_overhead_ns", static_cast<double>(clock.medianNs))
            .set("viable", true)
            .set("truncated", false)
            .set("skipped", false);
        writer.write(std::move(row));
        writer.flush();
    }

    std::vector<std::string> validationFailures;
    DatasetCache<Elem> dataCache;
    ArtifactCache<Elem> artifacts;
    size_t cellsForN = cfg.n;

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
        // Indices must be in range for the stream actually obtained, so a dataset
        // that yields a different length gets its own traces rather than clamped
        // ones — clamping would fold a tail of probes onto the last row and change
        // the distinct count without saying so.
        if (n != cellsForN) {
            cells     = buildCells(cfg, patterns, thetas, n);
            cellsForN = n;
        }

        for (auto& enc : encoders) {
            std::cout << "  [" << enc.name << "] encoding..." << std::flush;
            // EncodeMeasurement::None: exactly one encode() per (encoder, dataset,
            // N).  For an AutoSIS codec a second encode would re-run cost-model
            // selection and could measure a different plan than the one reported.
            const auto& artifact = artifacts.get(enc, handle, EncodeMeasurement::None);
            const size_t payloadBytes = artifact.payloadBytes;
            const double ratio        = artifact.compressionRatio;
            const bool   fastSkip     = artifact.fastSkip;
            const bool   randomAccess = artifact.randomAccess;

            PlaygroundTarget<Elem> target(*enc.codec);
            // A copy of the payload, not a second encode: the cache holds one
            // artifact at a time and the target owns the bytes it decodes from and
            // that the cache controller evicts.
            target.adopt(artifact.encoded);
            const bool emulatedPointRead = !target.capabilities().nativePointRead;

            std::cout << " " << payloadBytes << " B (ratio " << ratio << "x"
                      << (randomAccess ? "" : ", no random access") << ")\n";

            DecodeHarness<PlaygroundTarget<Elem>> harness(target, *cache);

            if (cfg.validate && randomAccess) {
                std::cout << "  [" << enc.name << "] validating..." << std::flush;
                std::string whyNot;
                if (!validateEncoder(harness, target, handle.data, cells, whyNot)) {
                    // Drop the encoder rather than aborting: one codec with a broken
                    // lookup should not cost the whole sweep.  The failure is
                    // reported again in the summary and sets the exit code.
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

            for (const char* contract : contracts) {
                const bool firstProbe = std::string_view(contract) == "cold-first-probe";
                for (const PointCell& c : cells) {
                    if (c.trace.indices.empty()) continue;
                    // cold-first-probe times exactly one probe per iteration, so the
                    // probes axis selects the index pool there and nothing else;
                    // probes_timed is what a reader divides by.
                    const size_t probesTimed = firstProbe ? 1 : c.trace.indices.size();

                    auto row = writer.row();
                    row.set("driver", "bench_decode_point")
                        .set("dataset", ds.name)
                        .set("encoding", enc.name)
                        .set("family", enc.family)
                        .set("variant", enc.variant)
                        .set("is_sequential", enc.isSequential)
                        .set("fast_skip", fastSkip)
                        .set("random_access", randomAccess)
                        // True on this target (decodeAt is a real entry point), false
                        // on a port: nimble has no such call and must label a one-row
                        // visitor read, or a native O(1) lookup and an emulated one
                        // end up in the same column differing by a constant nobody
                        // can see.
                        .set("emulated_point_read", emulatedPointRead)
                        .set("N", n)
                        .set("seed", cfg.seed)
                        .set("cache_state", cacheStateName(effectiveState))
                        .set("evict_method", evictMethodName(cache->effectivePolicy().method))
                        .set("cold_contract", contract)
                        .set("row_kind", "measurement")
                        .set("pattern", pointPatternName(c.params.pattern))
                        .set("ascending", cfg.ascending)
                        .set("probes_nominal", c.params.probes)
                        .set("probes_timed", probesTimed)
                        // Measured from the list that was built, never derived from
                        // the request: distinct_fraction is what tells a reader
                        // whether a row reflects the codec or the cache.
                        .set("distinct_indices", c.trace.distinctIndices)
                        .set("distinct_fraction", c.trace.distinctFraction)
                        .set("footprint_span", c.trace.footprintSpan)
                        .set("payload_bytes", payloadBytes)
                        .set("compression_ratio", ratio)
                        .set("iterations", cfg.iterations)
                        .set("warmup", cfg.warmup)
                        .set("clock_overhead_ns", static_cast<double>(clock.medianNs))
                        // The probe list ran out of permutation and repeated a pass:
                        // distinct_indices saturates at N and the row is a weaker
                        // statement than the same one below N probes.
                        .set("truncated", c.trace.indices.size() > n);
                    setKnobs(row, c.params, n);

                    const auto emitUnmeasured = [&row](bool viable) {
                        row.set("viable", viable)
                            .set("skipped", true)
                            .setNull("evict_ns")
                            .setNull("batch_ns")
                            .setNull("time_ns")
                            .setNull("time_p90_ns")
                            .setNull("time_min_ns")
                            .setNull("ns_per_probe");
                    };

                    // A codec without RandomAccess has no point-lookup path at all.
                    // The pre-refactor runner returned early here and left a zero
                    // row; a zero is indistinguishable from an infinitely fast
                    // lookup, so this is an emitted row that says viable=0 and
                    // carries no timings whatsoever.
                    if (!randomAccess) {
                        emitUnmeasured(false);
                        writer.write(std::move(row));
                        ++emitted;
                        continue;
                    }

                    const double projectedMs =
                        projectCellMs(harness, c.trace, probesTimed, cfg.iterations);
                    if (cfg.maxCellMs > 0 &&
                        projectedMs > static_cast<double>(cfg.maxCellMs)) {
                        // Viable but deliberately not measured: an O(N) decodeAt is a
                        // real property of the encoding, and the row records that it
                        // was too slow to run rather than dropping the cell.
                        emitUnmeasured(true);
                        writer.write(std::move(row));
                        ++emitted;
                        continue;
                    }

                    MeasureSpec spec;
                    spec.iterations = cfg.iterations;
                    spec.warmup     = cfg.warmup;
                    const MeasureResult r =
                        firstProbe ? measureFirstProbe(harness, c.trace, cfg)
                                   : harness.points(c.trace, spec);

                    row.set("viable", true)
                        .set("skipped", false)
                        // evict_ns is reported next to time_ns and never folded into
                        // it: under cold-first-probe one eviction precedes one
                        // nanosecond-scale probe, so this column is orders of
                        // magnitude larger than the measurement and must stay
                        // visible for the row to be readable at all.
                        .set("evict_ns", r.evict.medianNs)
                        // batch_ns and time_ns are the same measured quantity: the
                        // whole timed region.  Both are emitted because time_ns is
                        // the name CONVENTIONS requires and batch_ns is the name that
                        // stops a reader from mistaking it for a per-probe figure.
                        .set("batch_ns", r.time.medianNs)
                        .set("time_ns", r.time.medianNs)
                        .set("time_p90_ns", r.time.p90Ns)
                        .set("time_min_ns", r.time.minNs)
                        .set("ns_per_probe", nsPerProbe(r, probesTimed));
                    writer.write(std::move(row));
                    ++emitted;
                    ++measured;
                }
            }

            writer.flush();  // keep a long sweep inspectable while it runs
            // The payload and the codec's internal state go before the next encoder
            // is timed, so its cache-state measurements are not polluted by this
            // one's working set.
            artifacts.evict(enc.name);
            std::cout << " " << emitted << " rows (" << measured << " measured, "
                      << (emitted - measured) << " not measured).\n";
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
