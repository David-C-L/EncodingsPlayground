// Gather-access throughput heatmap.
//
// heatmap_benchmark.cpp sweeps contiguous range accesses (start, length) and so
// only ever measures a dense read.  This driver sweeps the *sparse* access that
// Nimble's TableScan actually issues after filter pushdown: a list of surviving
// row ranges with gaps that a codec must skip rather than materialize, driven
// through Codec::decodeGatherInto().
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
// range-access baseline; --validate checks that against decodeRange() directly.
//
// Reading the sigma = 1 row against heatmap_benchmark.cpp: the two issue the same
// access but not the same call.  decodeGatherInto() must write into a caller-owned
// buffer, so it goes through decodeRangeInto(), and for a codec that does not
// override that (Raw, AdaptiveFramedBitPrefix, Zstd, OpenZL) the base class
// implements it as decodeRange() plus a copy — one extra materialization the
// range-access driver never pays.  Those encoders therefore read as roughly 2x
// slower here at sigma = 1 even though the decode work is identical.  That gap is
// the cost of the gather API's buffer contract, not a measurement artifact, and it
// disappears for codecs that override decodeRangeInto (RawBitPacked, BlockFPE,
// BlockFORFPE, FPE, SubIntSplit).

#include "benchmark/Axes.hpp"
#include "benchmark/GatherTraceGen.hpp"
#include "benchmark/TimingStats.hpp"
#include "generators/CommonGenerators.hpp"
#include "generators/ParquetColumnGenerator.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/AdaptiveFramedBitPrefixEncoder.hpp"
#include "encoders/BlockFrequencyPartitionEncoder.hpp"
#include "encoders/FrequencyPartitionEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/OpenZLEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"
#include "encoders/SubIntSplitEncoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define GATHER_HAVE_CLFLUSH 1
#endif

using namespace encodings;
using namespace encodings::generators;
using namespace encodings::encoders;
using encodings::benchmark::GapModel;
using encodings::benchmark::GatherAccessParams;
using encodings::benchmark::GatherTrace;
using encodings::benchmark::buildGatherTrace;
using encodings::benchmark::impliedRangeCount;
using encodings::benchmark::linSpaced;
using encodings::benchmark::logSpaced;

namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);

// ─── Configuration ───────────────────────────────────────────────────────────

enum class FlushMode { None, Clflush };

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
    FlushMode flush{FlushMode::None};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/gather_heatmap_benchmark.csv"};
};

std::string flushModeName(FlushMode m) { return m == FlushMode::Clflush ? "clflush" : "none"; }

void printUsage() {
    std::cout <<
        "Usage: gather_heatmap_benchmark [options]\n"
        "\n"
        "Sweep axes:\n"
        "  --n N                total stream length in elements        (default 10000000)\n"
        "  --l-min N            smallest access span                   (default 1024)\n"
        "  --l-max N            largest access span                    (default N/8)\n"
        "  --n-l N              span steps, log-spaced                 (default 16)\n"
        "  --n-s0 N             start-fraction steps, linear in [0,1]  (default 8)\n"
        "  --sigma-min F        lowest selectivity                     (default 0.1)\n"
        "  --n-sigma N          selectivity steps, linear up to 1.0    (default 10)\n"
        "  --min-range-size N   run length in elements                 (default 8)\n"
        "  --run-max N          largest run length for the 4th axis    (default = min-range-size)\n"
        "  --n-run N            run-length steps, log-spaced           (default 1)\n"
        "\n"
        "Trace construction:\n"
        "  --gap-model MODE     uniform | geometric                    (default uniform)\n"
        "  --seed N             geometric-model seed                   (default 42)\n"
        "  --max-ranges N       cap on k per access, 0 = unbounded     (default 65536)\n"
        "  --seq-max-k N        skip sequential encoders above this k  (default 64)\n"
        "\n"
        "Measurement:\n"
        "  --iterations N       timed repetitions per point            (default 5)\n"
        "  --warmup N           untimed repetitions per point          (default 2)\n"
        "  --flush MODE         none | clflush                         (default none)\n"
        "\n"
        "Selection and output:\n"
        "  --dataset SUBSTR     only datasets whose name contains SUBSTR (repeatable)\n"
        "  --encoder SUBSTR     only encoders whose name contains SUBSTR (repeatable)\n"
        "  --validate           run correctness and sigma=1 baseline checks first\n"
        "  --dry-run            print the preflight summary and exit\n"
        "  --output PATH        CSV output path\n"
        "  --help, -h           this message\n";
}

/// Returns false if the program should exit; `ok` distinguishes --help from a parse error.
bool parseArgs(int argc, char** argv, SweepConfig& cfg, bool& ok) {
    ok = true;
    auto need = [&](int i) {
        if (i + 1 >= argc) {
            std::cerr << "ERROR: " << argv[i] << " requires a value\n";
            ok = false;
            return false;
        }
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        try {
            if (arg == "--help" || arg == "-h") { printUsage(); return false; }
            else if (arg == "--n"              && need(i)) cfg.n            = std::stoull(argv[++i]);
            else if (arg == "--l-min"          && need(i)) cfg.lMin         = std::stoull(argv[++i]);
            else if (arg == "--l-max"          && need(i)) cfg.lMax         = std::stoull(argv[++i]);
            else if (arg == "--n-l"            && need(i)) cfg.nL           = std::stoull(argv[++i]);
            else if (arg == "--n-s0"           && need(i)) cfg.nS0          = std::stoull(argv[++i]);
            else if (arg == "--sigma-min"      && need(i)) cfg.sigmaMin     = std::stod(argv[++i]);
            else if (arg == "--n-sigma"        && need(i)) cfg.nSigma       = std::stoull(argv[++i]);
            else if (arg == "--min-range-size" && need(i)) cfg.minRangeSize = std::stoull(argv[++i]);
            else if (arg == "--run-max"        && need(i)) cfg.runMax       = std::stoull(argv[++i]);
            else if (arg == "--n-run"          && need(i)) cfg.nRun         = std::stoull(argv[++i]);
            else if (arg == "--seed"           && need(i)) cfg.seed         = std::stoull(argv[++i]);
            else if (arg == "--max-ranges"     && need(i)) cfg.maxRanges    = std::stoull(argv[++i]);
            else if (arg == "--seq-max-k"      && need(i)) cfg.seqMaxK      = std::stoull(argv[++i]);
            else if (arg == "--iterations"     && need(i)) cfg.iterations   = std::stoull(argv[++i]);
            else if (arg == "--warmup"         && need(i)) cfg.warmup       = std::stoull(argv[++i]);
            else if (arg == "--output"         && need(i)) cfg.output       = argv[++i];
            else if (arg == "--dataset"        && need(i)) cfg.datasetFilters.emplace_back(argv[++i]);
            else if (arg == "--encoder"        && need(i)) cfg.encoderFilters.emplace_back(argv[++i]);
            else if (arg == "--validate")                  cfg.validate     = true;
            else if (arg == "--dry-run")                   cfg.dryRun       = true;
            else if (arg == "--gap-model" && need(i)) {
                const std::string m = argv[++i];
                if      (m == "uniform")   cfg.gapModel = GapModel::UniformDeterministic;
                else if (m == "geometric") cfg.gapModel = GapModel::Geometric;
                else { std::cerr << "ERROR: unknown gap model '" << m << "'\n"; ok = false; return false; }
            }
            else if (arg == "--flush" && need(i)) {
                const std::string m = argv[++i];
                if      (m == "none")    cfg.flush = FlushMode::None;
                else if (m == "clflush") cfg.flush = FlushMode::Clflush;
                else { std::cerr << "ERROR: unknown flush mode '" << m << "'\n"; ok = false; return false; }
            }
            else if (!ok) {
                return false;  // need() already reported
            }
            else {
                std::cerr << "ERROR: unknown argument '" << arg << "' (try --help)\n";
                ok = false;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "ERROR: bad value for " << arg << ": " << e.what() << "\n";
            ok = false;
            return false;
        }
        if (!ok) return false;
    }

    // Resolve defaults that depend on other flags, then sanity-check.
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
    return true;
}

// ─── Measurement primitives ──────────────────────────────────────────────────

/// Keep the sink observable so the optimizer cannot elide the gather.
inline void clobber(const void* p) {
    asm volatile("" : : "r"(p) : "memory");
}

/// Evict a byte span from the data caches.
///
/// Deliberately NOT madvise(MADV_DONTNEED): on private anonymous heap memory
/// that discards the pages and they fault back in *zeroed*, which corrupts the
/// encoded payload rather than merely cooling it.
void flushSpan(const void* base, size_t bytes) {
#ifdef GATHER_HAVE_CLFLUSH
    const char* p   = static_cast<const char*>(base);
    const char* end = p + bytes;
    for (; p < end; p += 64) _mm_clflush(p);
    _mm_mfence();
#else
    (void)base;
    (void)bytes;
#endif
}

// ─── Registries ──────────────────────────────────────────────────────────────

struct EncoderEntry {
    std::string name;
    std::shared_ptr<Codec<Elem>> codec;
    // true  → decodeGatherInto must read the full compressed payload per range
    //         (Zstd, OpenZL: no override, so the base-class per-range fallback runs)
    // false → reads O(selected) bytes from the encoded payload
    bool isSequential;
};

struct DatasetEntry {
    std::string name;
    std::shared_ptr<datagen::DataGenerator<Elem>> generator;
};

std::vector<EncoderEntry> buildEncoders() {
    std::vector<EncoderEntry> e;
    e.push_back({"Raw",               std::make_shared<RawEncoder<Elem>>(),                     false});
    e.push_back({"RawBitPacked",      std::make_shared<RawBitPackedEncoder<Elem>>(),            false});
    e.push_back({"BlockFPE",          std::make_shared<BlockFrequencyPartitionEncoder<Elem>>(), false});
    e.push_back({"BlockFORFPE",       std::make_shared<BlockFORFPEEncoder<Elem>>(),             false});
    e.push_back({"AdaptiveBitPrefix", std::make_shared<AdaptiveFramedBitPrefixEncoder<Elem>>(), false});
    // FPE with a positional index is the codec with a genuine skip fast path, so
    // it is the one that populates the skip/materialize split columns.
    e.push_back({"FPE_PerTierBitmaps",
        std::make_shared<FrequencyPartitionEncoder<Elem, FreqPartIndexType::PerTierBitmaps>>(), false});
    e.push_back({"AutoSIS",
        makeSubIntSplitEncoderManual<Elem>(
            {13, 1, 8, 28, 8, 6},
            {EncodingType::BlockFrequencyPartitionEncoding,
             EncodingType::BitPacking,
             EncodingType::BlockFrequencyPartitionEncoding,
             EncodingType::BitPacking,
             EncodingType::RunLengthCascadingFOREncoding,
             EncodingType::RunLengthEncoding}),
        false});
    // Profiling instantiation: only SubIntSplitEncoder<T, true> reports the
    // gatherSkipTimeNs()/gatherMaterializeTimeNs() split.
    e.push_back({"AutoSIS_Prof",
        makeSubIntSplitEncoderManualProf<Elem>(
            {13, 1, 8, 28, 8, 6},
            {EncodingType::BlockFrequencyPartitionEncoding,
             EncodingType::BitPacking,
             EncodingType::BlockFrequencyPartitionEncoding,
             EncodingType::BitPacking,
             EncodingType::RunLengthCascadingFOREncoding,
             EncodingType::RunLengthEncoding}),
        false});
    e.push_back({"AutoSIS_Delta",
        makeSubIntSplitEncoderManual<Elem>(
            {13, 8, 1, 25, 17},
            {EncodingType::BlockFrequencyPartitionEncoding,
             EncodingType::AdaptiveDictionaryEncoding,
             EncodingType::BitPacking,
             EncodingType::BitPacking,
             EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding}),
        false});
#ifdef HAVE_OPENZL
    e.push_back({"OpenZL", makeOpenZLCodec<Elem>(), true});
#endif
    e.push_back({"Zstd", std::make_shared<ZstdEncoder<Elem>>(), true});
    return e;
}

std::vector<DatasetEntry> buildDatasets() {
    // Only int64 sources: the driver is instantiated on Codec<int64_t>, matching
    // every other benchmark in this suite.  The TPCH orderkey and iNaturalist
    // species-id parquets are int32 columns and are therefore excluded; picking
    // them up needs a runSweep<T>() template plus an int32 encoder registry.
    const std::filesystem::path kRoot =
        "/home/david/Documents/PhD/symbol-store/MetaNimbleProject/EncodingsPlayground/Datasets";

    std::vector<DatasetEntry> d;
    d.push_back({"TwitterSnowflake",
        std::make_shared<ParquetColumnGenerator<Elem>>(
            kRoot / "TwitterSnowflake/tweet_ids.parquet", "tweet_id")});
    d.push_back({"Sequential",
        std::make_shared<SequentialGenerator<Elem>>(0, 1)});
    d.push_back({"Zipfian1.0",
        std::make_shared<ZipfianGenerator<Elem>>(1'000'000, 1.0)});
    d.push_back({"UniformRandom",
        std::make_shared<UniformRandomGenerator<Elem>>(0, (Elem{1} << 40))});
    return d;
}

template <typename Entry>
std::vector<Entry> applyFilters(std::vector<Entry> in, const std::vector<std::string>& filters) {
    if (filters.empty()) return in;
    std::vector<Entry> out;
    for (auto& e : in) {
        for (const auto& f : filters) {
            if (e.name.find(f) != std::string::npos) { out.push_back(std::move(e)); break; }
        }
    }
    return out;
}

// ─── Preflight ───────────────────────────────────────────────────────────────

void printPreflight(const SweepConfig& cfg,
                    const std::vector<size_t>& spans,
                    const std::vector<double>& s0Fracs,
                    const std::vector<double>& sigmas,
                    const std::vector<size_t>& runs,
                    const std::vector<EncoderEntry>& encoders,
                    const std::vector<DatasetEntry>& datasets) {
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
              << "  gap model      " << encodings::benchmark::gapModelName(cfg.gapModel)
              << (cfg.gapModel == GapModel::Geometric ? "  (seed " + std::to_string(cfg.seed) + ")" : "") << "\n"
              << "  max ranges     " << cfg.maxRanges << "\n"
              << "  seq max k      " << cfg.seqMaxK << "\n"
              << "  flush          " << flushModeName(cfg.flush) << "\n"
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

/// Checks that (a) a gather returns exactly the values at the selected indices,
/// and (b) the sigma = 1 slice is byte-identical to a plain decodeRange — the
/// claim that makes the sigma = 1 row of the heatmap the range-access baseline.
bool validateEncoder(EncoderEntry& entry,
                     const EncodedData& encoded,
                     const std::vector<Elem>& reference,
                     const SweepConfig& cfg,
                     const std::vector<size_t>& spans,
                     const std::vector<double>& sigmas) {
    std::vector<Elem> sink(spans.back() + 1);
    bool allOk = true;

    for (size_t span : spans) {
        const size_t s0 = (reference.size() > span) ? (reference.size() - span) / 3 : 0;
        for (double sigma : sigmas) {
            GatherAccessParams p;
            p.start       = s0;
            p.span        = span;
            p.selectivity = sigma;
            p.runLength   = cfg.minRangeSize;
            p.gapModel    = cfg.gapModel;
            p.seed        = cfg.seed;
            p.maxRanges   = cfg.maxRanges;
            const GatherTrace t = buildGatherTrace(reference.size(), p);
            if (t.selectedRows == 0) continue;
            if (entry.isSequential && t.rangeCount > cfg.seqMaxK) continue;

            entry.codec->decodeGatherInto(encoded, t.ranges, sink.data(), t.selectedRows);

            size_t off = 0;
            for (const auto& r : t.ranges) {
                for (size_t i = r.begin; i < r.end; ++i, ++off) {
                    if (sink[off] != reference[i]) {
                        std::cerr << "  FAIL [" << entry.name << "] gather mismatch at row " << i
                                  << " (l=" << span << ", sigma=" << sigma << "): got " << sink[off]
                                  << ", expected " << reference[i] << "\n";
                        allOk = false;
                        goto nextSigma;  // one report per (span, sigma) is enough
                    }
                }
            }
        nextSigma:;
        }

        // sigma = 1 must agree with the contiguous range read.
        const RowRangeList full{{s0, s0 + span}};
        entry.codec->decodeGatherInto(encoded, full, sink.data(), span);
        const auto viaRange = entry.codec->decodeRange(encoded, s0, s0 + span);
        if (viaRange.size() != span || !std::equal(viaRange.begin(), viaRange.end(), sink.begin())) {
            std::cerr << "  FAIL [" << entry.name << "] sigma=1 gather != decodeRange at l=" << span << "\n";
            allOk = false;
        }
    }
    return allOk;
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    bool ok = true;
    if (!parseArgs(argc, argv, cfg, ok)) return ok ? 0 : 1;

    const auto spans   = logSpaced(cfg.lMin, cfg.lMax, cfg.nL);
    const auto runs    = logSpaced(cfg.minRangeSize, cfg.runMax, cfg.nRun);
    const auto s0Fracs = linSpaced(0.0, 1.0, cfg.nS0);
    const auto sigmas  = linSpaced(cfg.sigmaMin, 1.0, cfg.nSigma);

    auto encoders = applyFilters(buildEncoders(), cfg.encoderFilters);
    auto datasets = applyFilters(buildDatasets(), cfg.datasetFilters);
    if (encoders.empty()) { std::cerr << "ERROR: no encoders match --encoder filters\n"; return 1; }
    if (datasets.empty()) { std::cerr << "ERROR: no datasets match --dataset filters\n"; return 1; }

    printPreflight(cfg, spans, s0Fracs, sigmas, runs, encoders, datasets);
    if (cfg.dryRun) {
        std::cout << "Dry run: no measurements taken.\n";
        return 0;
    }

#ifndef GATHER_HAVE_CLFLUSH
    if (cfg.flush == FlushMode::Clflush)
        std::cerr << "WARNING: --flush clflush requested but this is not an x86-64 build; "
                     "flushing is a no-op.\n";
#endif

    std::filesystem::create_directories(
        cfg.output.parent_path().empty() ? "." : cfg.output.parent_path());
    std::ofstream csv(cfg.output);
    if (!csv) {
        std::cerr << "ERROR: cannot open " << cfg.output << " for writing\n";
        return 1;
    }
    csv << "dataset,encoding,is_sequential,fast_skip,gap_model,flush_mode,seed,"
           "N,s0_frac,s0,l,sigma_nominal,sigma_achieved,"
           "run_length_nominal,run_length_actual,k_nominal,k_actual,"
           "selected_elems,span_elems,compressed_bytes,compression_ratio,"
           "time_ns,time_p90_ns,time_min_ns,"
           "sel_elem_Meps,span_elem_Meps,useful_MBps,input_MBps,"
           "gather_skip_ns,gather_materialize_ns,truncated,skipped\n";

    // One sink for the whole run, sized for the sigma = 1 worst case.  Hoisting
    // it out of the timed loop keeps allocation out of the measurement — the
    // range-access driver's `(void)decodeRange(...)` allocates a fresh vector on
    // every timed call and so charges allocation to decode throughput.
    std::vector<Elem> sink(spans.back() + 1);
    std::vector<std::string> validationFailures;

    for (auto& ds : datasets) {
        std::cout << "══ Dataset: " << ds.name << " ══\n"
                  << "  loading " << cfg.n << " elements..." << std::flush;
        std::vector<Elem> data = ds.generator->generate(cfg.n);
        const size_t n = data.size();
        std::cout << " got " << n << ".\n";
        if (n < spans.back()) {
            std::cerr << "  WARNING: dataset yielded " << n << " elements, less than l_max="
                      << spans.back() << "; spans are clipped to the stream.\n";
        }

        for (auto& enc : encoders) {
            std::cout << "  [" << enc.name << "] encoding..." << std::flush;
            enc.codec->reset();
            const EncodedData encoded = enc.codec->encode(std::span<const Elem>(data.data(), n));
            const size_t compressedBytes = encoded.metadata().compressedSize;
            const double ratio = compressedBytes > 0
                ? static_cast<double>(n * kElemSize) / static_cast<double>(compressedBytes)
                : 0.0;
            const bool fastSkip = enc.codec->properties().has(EncodingProperty::FastSkip);
            std::cout << " " << compressedBytes << " B (ratio " << ratio << "x"
                      << (fastSkip ? ", FastSkip" : "") << ")\n";

            if (cfg.validate) {
                std::cout << "  [" << enc.name << "] validating..." << std::flush;
                if (!validateEncoder(enc, encoded, data, cfg, spans, sigmas)) {
                    // Drop the encoder rather than aborting: one codec with a broken
                    // decode should not cost a multi-hour sweep of the others.  The
                    // failure is reported again in the summary and sets the exit code.
                    std::cerr << "  [" << enc.name << "] EXCLUDED from the sweep: "
                                 "decode does not round-trip on " << ds.name << "\n";
                    validationFailures.push_back(enc.name + " on " + ds.name);
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

                            // A sequential codec has no decodeGatherInto override, so it
                            // pays a full-payload decode per range.  Past a few dozen
                            // ranges a single point runs for minutes, so emit the cell as
                            // not-viable rather than either measuring it or dropping it —
                            // that is a real property of the encoding under gather, and
                            // the plots should show it as such.
                            const bool skipped =
                                enc.isSequential && t.rangeCount > cfg.seqMaxK;

                            auto emitRow = [&](double timeNs, double p90Ns, double minNs,
                                               int64_t skipNs, int64_t matNs) {
                                csv << ds.name << ',' << enc.name << ','
                                    << (enc.isSequential ? 1 : 0) << ',' << (fastSkip ? 1 : 0) << ','
                                    << encodings::benchmark::gapModelName(cfg.gapModel) << ','
                                    << flushModeName(cfg.flush) << ',' << cfg.seed << ','
                                    << n << ',' << s0Frac << ',' << s0 << ',' << spanClamped << ','
                                    << sigma << ',' << t.selectivityAchieved << ','
                                    << runLength << ',' << t.runLengthActual << ','
                                    << t.rangeCountNominal << ',' << t.rangeCount << ','
                                    << t.selectedRows << ',' << spanClamped << ','
                                    << compressedBytes << ',' << ratio << ',';
                                if (skipped) {
                                    csv << ",,,,,,";  // timing and rate columns left empty
                                } else {
                                    const double selElemMeps  = timeNs > 0.0
                                        ? static_cast<double>(t.selectedRows) / timeNs * 1e3 : 0.0;
                                    const double spanElemMeps = timeNs > 0.0
                                        ? static_cast<double>(spanClamped) / timeNs * 1e3 : 0.0;
                                    const double usefulMBps   = timeNs > 0.0
                                        ? static_cast<double>(t.selectedRows * kElemSize) / timeNs * 1e3 : 0.0;
                                    // Compressed-input bandwidth, same model as
                                    // heatmap_benchmark.cpp: a sequential codec must read
                                    // the whole payload, a random-access one reads a
                                    // proportional slice of it.
                                    const double inputBytes = enc.isSequential
                                        ? static_cast<double>(compressedBytes)
                                        : static_cast<double>(t.selectedRows)
                                              * static_cast<double>(compressedBytes)
                                              / static_cast<double>(n);
                                    const double inputMBps = timeNs > 0.0 ? inputBytes / timeNs * 1e3 : 0.0;
                                    csv << timeNs << ',' << p90Ns << ',' << minNs << ','
                                        << selElemMeps << ',' << spanElemMeps << ','
                                        << usefulMBps << ',' << inputMBps << ',';
                                }
                                // -1 means the codec has no distinct skip phase; write it
                                // through as an empty field rather than a sentinel number.
                                if (skipNs >= 0) csv << skipNs;
                                csv << ',';
                                if (matNs >= 0) csv << matNs;
                                csv << ',';
                                csv << (t.clamped ? 1 : 0) << ',' << (skipped ? 1 : 0) << '\n';
                                ++emitted;
                            };

                            if (skipped) {
                                emitRow(0, 0, 0, -1, -1);
                                continue;
                            }

                            for (size_t w = 0; w < cfg.warmup; ++w) {
                                enc.codec->decodeGatherInto(encoded, t.ranges, sink.data(), t.selectedRows);
                                clobber(sink.data());
                            }

                            std::vector<int64_t> times;
                            times.reserve(cfg.iterations);
                            int64_t skipNs = -1, matNs = -1;
                            for (size_t m = 0; m < cfg.iterations; ++m) {
                                if (cfg.flush == FlushMode::Clflush) {
                                    flushSpan(encoded.data().data(), encoded.data().size());
                                    flushSpan(sink.data(), t.selectedRows * kElemSize);
                                }
                                enc.codec->resetGatherProfilingAccum();
                                const auto t0 = std::chrono::high_resolution_clock::now();
                                enc.codec->decodeGatherInto(encoded, t.ranges, sink.data(), t.selectedRows);
                                const auto t1 = std::chrono::high_resolution_clock::now();
                                clobber(sink.data());
                                times.push_back(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                                skipNs = enc.codec->gatherSkipTimeNs();
                                matNs  = enc.codec->gatherMaterializeTimeNs();
                            }

                            const auto ts = encodings::benchmark::summarize(times);
                            emitRow(static_cast<double>(ts.medianNs),
                                    static_cast<double>(ts.p90Ns),
                                    static_cast<double>(ts.minNs),
                                    skipNs, matNs);
                            ++measured;
                        }
                    }
                }
            }

            csv.flush();  // keep a long sweep inspectable while it runs
            std::cout << " " << emitted << " rows (" << measured << " measured, "
                      << (emitted - measured) << " not viable).\n";
        }
    }

    csv.close();
    std::cout << "\nResults written to: " << std::filesystem::absolute(cfg.output) << std::endl;

    if (!validationFailures.empty()) {
        std::cerr << "\n" << validationFailures.size()
                  << " encoder/dataset pair(s) failed validation and were excluded:\n";
        for (const auto& f : validationFailures) std::cerr << "  - " << f << "\n";
        return 2;
    }
    return 0;
}
