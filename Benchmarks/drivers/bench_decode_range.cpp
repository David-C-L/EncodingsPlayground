// Contiguous-range decode throughput heatmap.
//
// Sweeps one dense access [A, A+B) over the (A_frac, B_frac) triangle — start
// position against range length, the two axes a projection push-down actually
// controls — through the target's materializeRange() (Codec::decodeRangeInto()).
// This is the port of Benchmarks/heatmap_benchmark.cpp onto the shared benchmark
// core, and it fixes two defects of that driver:
//
//   * it called the ALLOCATING decodeRange() inside the timed loop, so every
//     timed call paid a fresh std::vector — allocation plus zero-fill — charged
//     to decode throughput.  Everything here goes through DecodeHarness::range
//     with one sink hoisted out of the timed region;
//   * its N, GRID, WARMUP and MEASURE were constexpr, so changing the grid meant
//     recompiling and no result file recorded which grid produced it.  They are
//     flags, and the manifest records the resolved values.
//
// The grid, the column names and the input-bandwidth model are preserved
// deliberately: Benchmarks/plot_heatmap.py consumes `encoding`, `A_frac`,
// `B_frac`, `elem_Meps`, `input_MBps` and `compression_ratio` by name, and pivots
// on the two fractions, so renaming or re-spacing any of them silently changes
// the figure rather than breaking the script.
//
// READING THIS AGAINST bench_decode_gather.  The gather driver's sigma = 1 slice
// issues the same access as a B_frac cell here, but not the same call: gather
// must write into a caller-owned buffer, so it goes through decodeGatherInto()
// which the base class implements as decodeRange() plus a copy for any codec that
// does not override decodeRangeInto().  Those codecs read as roughly 2x slower
// there — one extra materialization, the cost of the gather API's buffer
// contract, not a measurement artifact (see bench_decode_gather.cpp's header).
// The `contract` column names which call each row timed so the comparison is
// made against a recorded fact instead of an assumption about the file.

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
#include <format>
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
    size_t n{1'000'000};
    size_t grid{64};
    size_t iterations{5};
    size_t warmup{2};
    uint64_t seed{42};
    size_t seqMaxN{0};        // 0 → no viability gate
    CacheState  cacheState{CacheState::Hot};
    EvictMethod evictMethod{EvictMethod::Auto};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_decode_range.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Sweep axes:")
        .opt("n", cfg.n, "total stream length in elements")
        .opt("grid", cfg.grid,
             "A and B fractions sampled per axis; ~grid^2/2 cells fall in the triangle");

    args.group("Measurement:")
        .opt("iterations", cfg.iterations, "timed repetitions per cell")
        .opt("warmup", cfg.warmup, "untimed repetitions per cell")
        .opt("seed", cfg.seed, "recorded in the manifest; all randomness derives from it")
        .opt("seq-max-n", cfg.seqMaxN,
             "emit sequential-encoder cells above this N as not viable (0 = no gate)")
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
        .flag("validate", cfg.validate, "run correctness checks before measuring")
        .flag("dry-run", cfg.dryRun, "print the preflight summary and exit")
        .opt("output", cfg.output, "result file")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

// ─── Grid ────────────────────────────────────────────────────────────────────

/// The fractional grid, unchanged from heatmap_benchmark.cpp.
///
///   A_frac in { 0, 1/grid, ..., (grid-1)/grid }   start of the range
///   B_frac in { 1/grid, 2/grid, ..., 1 }          length of the range
///
/// Valid cells satisfy A_frac + B_frac <= 1, so the sampled region is the lower
/// triangle and B_frac = 1 exists only at A_frac = 0 — the full-column read that
/// bench_decode_bulk must agree with.  It is NOT built with Axes::linSpaced: the
/// two axes are offset by one step from each other (A starts at 0, B at 1/grid),
/// which is what keeps every cell a non-empty range, and plot_heatmap.py pivots on
/// the exact fraction values, so respacing the axis moves every cell of every
/// existing figure.
struct Grid {
    std::vector<double> aFracs, bFracs;
};

Grid buildGrid(size_t grid) {
    Grid g;
    g.aFracs.reserve(grid);
    g.bFracs.reserve(grid);
    for (size_t i = 0; i < grid; ++i)
        g.aFracs.push_back(static_cast<double>(i) / static_cast<double>(grid));
    for (size_t j = 1; j <= grid; ++j)
        g.bFracs.push_back(static_cast<double>(j) / static_cast<double>(grid));
    return g;
}

/// The (A, B) a fraction pair resolves to at this N, rounded exactly as the
/// pre-port driver rounded it.  B is clamped so the range never runs off the end
/// and is never empty; both the nominal fractions and the resolved elements reach
/// the row, because the grid is indexed by the nominal value while the throughput
/// is only honest against the achieved one (CONVENTIONS section 6).
struct Cell {
    size_t a{}, b{};
};

Cell resolveCell(double aFrac, double bFrac, size_t n) {
    Cell c;
    c.a = static_cast<size_t>(std::llround(aFrac * static_cast<double>(n)));
    c.b = std::max<size_t>(1, static_cast<size_t>(std::llround(bFrac * static_cast<double>(n))));
    if (c.a >= n) c.a = n - 1;
    if (c.a + c.b > n) c.b = n - c.a;
    return c;
}

// ─── Result schema ───────────────────────────────────────────────────────────

/// The column list, in file order.
///
/// `encoding`, `A_frac`, `B_frac`, `time_ns`, `elem_Meps`, `input_MBps`,
/// `compression_ratio` and `is_sequential` are the columns
/// Benchmarks/plot_heatmap.py reads and MUST NOT be renamed or dropped; the rest
/// are the per-row commons CONVENTIONS section 6 requires.  Timings and rates are
/// nullable and a not-viable cell leaves them null — never 0, which would plot as
/// a real measurement of an infinitely fast codec.
std::vector<ColumnSpec> rangeColumns() {
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
        // Which call was timed.  bench_decode_gather's sigma = 1 rows measure the
        // same ACCESS through decodeGatherInto, and for a codec without a
        // decodeRangeInto override that costs one extra materialization; naming the
        // contract per row is what lets the two files be compared without knowing
        // which driver wrote which half.
        stringCol("contract"),
        doubleCol("A_frac"),
        doubleCol("B_frac"),
        intCol("A"),
        intCol("B"),
        intCol("time_ns"),
        intCol("time_p90_ns"),
        intCol("time_min_ns"),
        doubleCol("elem_Meps"),
        doubleCol("input_MBps"),
        doubleCol("useful_MBps"),
        intCol("truncated"),
        intCol("skipped"),
    };
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Checks that the codec round-trips the stream and that materializeRange returns
/// exactly the reference values at a spread of grid cells.
///
/// Not every cell: at grid = 64 that is ~2000 value-by-value comparisons of up to
/// N elements per encoder, which costs more than the sweep.  The subset is the
/// widest range at every A_frac plus the narrowest at every A_frac, which is where
/// off-by-one seeking errors live — a codec that decodes the interior correctly
/// but mis-seeks at a block boundary fails at one of those.
bool validateEncoder(DecodeHarness<PlaygroundTarget<Elem>>& harness,
                     PlaygroundTarget<Elem>& target,
                     std::span<const Elem> reference,
                     const Grid& grid,
                     std::vector<Elem>& scratch,
                     std::string& whyNot) {
    if (!harness.validate(reference, whyNot)) return false;

    const size_t n = reference.size();
    for (double aFrac : grid.aFracs) {
        for (double bFrac : {grid.bFracs.front(), 1.0 - aFrac}) {
            if (aFrac + bFrac > 1.0 + 1e-9) continue;
            const Cell c = resolveCell(aFrac, bFrac, n);
            if (c.b == 0) continue;
            scratch.assign(c.b, Elem{});
            target.materializeRange(c.a, c.a + c.b, scratch.data(), c.b);
            for (size_t i = 0; i < c.b; ++i) {
                if (scratch[i] != reference[c.a + i]) {
                    whyNot = std::format(
                        "materializeRange mismatch at row {} of [{}, {}): got {}, expected {}",
                        c.a + i, c.a, c.a + c.b, static_cast<int64_t>(scratch[i]),
                        static_cast<int64_t>(reference[c.a + i]));
                    return false;
                }
            }
        }
    }
    return true;
}

// ─── Preflight ───────────────────────────────────────────────────────────────

void printPreflight(const SweepConfig& cfg,
                    const Grid& grid,
                    size_t cellCount,
                    const std::vector<EncoderEntry<Elem>>& encoders,
                    const std::vector<DatasetEntry<Elem>>& datasets,
                    const CacheController& cache) {
    std::cout << "\n── Sweep configuration ──────────────────────────────────────\n"
              << "  N              " << cfg.n << " elements ("
              << (cfg.n * kElemSize / (1024 * 1024)) << " MiB raw)\n"
              << "  element size   " << kElemSize << " bytes (int64_t)\n"
              << "  grid           " << cfg.grid << " x " << cfg.grid
              << "  (A_frac " << grid.aFracs.front() << " … " << grid.aFracs.back()
              << ", B_frac " << grid.bFracs.front() << " … " << grid.bFracs.back() << ")\n"
              << "  cache          " << cache.describe() << "\n"
              << "  seq max N      " << cfg.seqMaxN << (cfg.seqMaxN == 0 ? "  (no gate)" : "")
              << "\n  iterations     " << cfg.iterations << " timed, " << cfg.warmup
              << " warmup\n";

    // The achieved structure, not the requested one: the number of cells in the
    // triangle after clamping is what the run costs, and a degenerate grid shows
    // up here rather than after an hour of measuring.
    std::cout << "\n── Achieved grid at N = " << cfg.n << " ──────────────────────────\n"
              << "  cells in triangle " << cellCount << " of "
              << (grid.aFracs.size() * grid.bFracs.size()) << " sampled\n"
              << "  smallest range    " << resolveCell(0.0, grid.bFracs.front(), cfg.n).b
              << " elements\n"
              << "  largest range     " << resolveCell(0.0, 1.0, cfg.n).b
              << " elements (the full-column cell bench_decode_bulk must match)\n";

    std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
              << "  encoders       " << encoders.size() << "\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  result rows    " << cellCount * encoders.size() * datasets.size() << "\n"
              << "  range calls    "
              << cellCount * encoders.size() * datasets.size() * (cfg.iterations + cfg.warmup)
              << "\n\n";
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_decode_range",
                   "Sweep contiguous-range decode throughput over the (A_frac, B_frac) "
                   "triangle of start position and range length.");
    bindArgs(args, cfg);
    switch (args.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    cfg.n          = std::max<size_t>(1, cfg.n);
    cfg.grid       = std::max<size_t>(1, cfg.grid);
    cfg.iterations = std::max<size_t>(1, cfg.iterations);

    const Grid grid = buildGrid(cfg.grid);
    size_t cellCount = 0;
    for (double aFrac : grid.aFracs)
        for (double bFrac : grid.bFracs)
            if (aFrac + bFrac <= 1.0 + 1e-9) ++cellCount;

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

    printPreflight(cfg, grid, cellCount, encoders, datasets, *cache);
    if (cfg.dryRun) {
        std::cout << "Dry run: no measurements taken.\n";
        return 0;
    }

    RunManifest manifest = RunManifest::capture("bench_decode_range", args.argvEcho());
    manifest.seed = cfg.seed;
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    for (const auto& e : encoders) manifest.encoders.push_back(e.name);
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    manifest.extra["cache_state"]  = cacheStateName(cache->effectivePolicy().state);
    manifest.extra["evict_method"] = evictMethodName(cache->effectivePolicy().method);
    // The grid was a constexpr in the pre-port driver, so no result file recorded
    // the geometry it came from; it is provenance now.
    manifest.extra["grid"]     = std::to_string(cfg.grid);
    manifest.extra["contract"] = "range_into";
    // Before the sweep, so a killed run still has provenance.
    manifest.writeSidecar(cfg.output);

    ResultWriter writer(cfg.output, rangeColumns(), cfg.format);

    // One sink for the whole run, sized for the B_frac = 1 cell.  Hoisting it out
    // of the timed loop is the fix this port exists for: the pre-port driver
    // called the allocating decodeRange() on every timed call and charged a heap
    // allocation plus a zero-fill of up to N elements to decode throughput.
    std::vector<Elem> sink(cfg.n);
    std::vector<Elem> scratch;
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
                if (!validateEncoder(harness, target, handle.data, grid, scratch, whyNot)) {
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

            std::cout << "  [" << enc.name << "] sweeping " << cfg.grid << "x" << cfg.grid
                      << " fractional grid..." << std::flush;
            size_t emitted = 0, measured = 0;

            // A sequential codec re-reads the whole payload per range, so a large-N
            // sweep of it runs for minutes per cell.  Above the gate the cell is
            // emitted as not-viable rather than measured or dropped — that is a real
            // property of the encoding under range access.
            const bool notViable = enc.isSequential && cfg.seqMaxN != 0 && n > cfg.seqMaxN;

            for (double aFrac : grid.aFracs) {
                for (double bFrac : grid.bFracs) {
                    if (aFrac + bFrac > 1.0 + 1e-9) continue;
                    const Cell c = resolveCell(aFrac, bFrac, n);
                    if (c.b == 0) continue;

                    auto row = writer.row();
                    row.set("driver", "bench_decode_range")
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
                        .set("payload_bytes", payloadBytes)
                        .set("compression_ratio", ratio)
                        .set("iterations", cfg.iterations)
                        .set("warmup", cfg.warmup)
                        .set("contract", "range_into")
                        .set("A_frac", aFrac)
                        .set("B_frac", bFrac)
                        .set("A", c.a)
                        .set("B", c.b)
                        // The requested length was clipped to the stream: the row is
                        // still measured, at the achieved B.
                        .set("truncated", c.b < static_cast<size_t>(
                                              std::llround(bFrac * static_cast<double>(n))))
                        .set("skipped", notViable);

                    if (notViable) {
                        // Emitted, not dropped: "not viable at this N" is a result,
                        // and a missing row is indistinguishable from a crash.
                        row.setNull("evict_ns")
                            .setNull("time_ns")
                            .setNull("time_p90_ns")
                            .setNull("time_min_ns")
                            .setNull("elem_Meps")
                            .setNull("input_MBps")
                            .setNull("useful_MBps");
                        writer.write(std::move(row));
                        ++emitted;
                        continue;
                    }

                    MeasureSpec spec;
                    spec.iterations = cfg.iterations;
                    spec.warmup     = cfg.warmup;
                    const MeasureResult r =
                        harness.range(c.a, c.a + c.b, std::span<Elem>(sink), spec);

                    const double timeNs = static_cast<double>(r.time.medianNs);
                    const double elemMeps = timeNs > 0.0
                        ? static_cast<double>(c.b) / timeNs * 1e3 : 0.0;
                    const double usefulMBps = timeNs > 0.0
                        ? static_cast<double>(c.b * kElemSize) / timeNs * 1e3 : 0.0;
                    // Input-bandwidth model, unchanged from heatmap_benchmark.cpp and
                    // shared with bench_decode_gather:
                    //   a sequential encoder must read the entire compressed payload
                    //   on every call, while a random-access one reads O(B) bytes,
                    //   i.e. a proportional slice of the payload.
                    // It is a MODEL, not a measurement — nothing here counts bytes
                    // actually touched — so it is the honest denominator only to the
                    // extent that the is_sequential flag on the row is right, which
                    // is why that flag is derived from the codec's properties rather
                    // than hand-written per driver.
                    const double inputBytes = enc.isSequential
                        ? static_cast<double>(payloadBytes)
                        : static_cast<double>(c.b) * static_cast<double>(payloadBytes)
                              / static_cast<double>(n);
                    const double inputMBps = timeNs > 0.0 ? inputBytes / timeNs * 1e3 : 0.0;

                    row.set("evict_ns", r.evict.medianNs)
                        .set("time_ns", r.time.medianNs)
                        .set("time_p90_ns", r.time.p90Ns)
                        .set("time_min_ns", r.time.minNs)
                        .set("elem_Meps", elemMeps)
                        .set("input_MBps", inputMBps)
                        .set("useful_MBps", usefulMBps);
                    writer.write(std::move(row));
                    ++emitted;
                    ++measured;
                }
            }

            writer.flush();  // keep a long sweep inspectable while it runs
            // The payload and the codec's internal state go before the next
            // encoder is timed, so its cache-state measurements are not polluted
            // by this one's working set.
            artifacts.evict(enc.name);
            std::cout << " " << emitted << " cells (" << measured << " measured, "
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
