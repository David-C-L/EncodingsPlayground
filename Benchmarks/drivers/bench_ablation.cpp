// Random-access ablations: what SubIntSplit loses, and gains, as its allowed
// codec set changes.
//
// SubIntSplit's FastSkip is the MIN over its sections (SubIntSplitEncoder.hpp
// ~line 569: allSectionsRandomAccess()).  One section whose codec cannot answer a
// positional query costs the WHOLE encoding its random-access property, however
// few bits that section covers.  So admitting a single sequential codec to the
// DP's candidate set can buy compression and pay for it with the property the
// encoding was chosen for — and nothing in a compression table shows that
// happening.  This driver is that experiment: it varies the allowed set along the
// ladders of registry/CodecSetLadder.hpp and reports, per rung, both the bytes and
// the access-path consequences.
//
// It is NOT a matrix of finished encoders.  Every rung is a fresh DP run whose
// CHOSEN PLAN is reported alongside the aggregates — `segment_count`,
// `segment_plan`, `admitted_selected`, `est_bits` against `actual_bits` — because
// two rungs that compress identically are two different results: one where the
// extra codec did not help, and one where the DP declined it.  Only the plan
// columns separate those.
//
// The DP is run HERE rather than through SubIntSplitAutoEncoder, for two reasons.
// The auto encoder keeps its selection private (`lastSelection_` has no accessor),
// so the chosen plan would be unrecoverable except by parsing debug logging; and
// it would re-run the DP on every encode(), so one rung's plan could not be held
// fixed across the cells that measure it.  Running the selector explicitly and
// building a manual SubIntSplitEncoder from its segments is the same path AutoSIS
// takes internally (ensureEncoder → select → fromSegments) with the plan visible.
// This is the arrangement CONVENTIONS section 10 anticipates for the oracle
// drivers.
//
// Bit order is LSB_TO_MSB throughout.  MSB order requires mirroring the sample and
// remapping the segments afterwards (SubIntSplitAutoEncoder::mirrorBits and
// remapMirroredSegmentsToOriginal); duplicating that here would put untested
// index arithmetic between the DP and the codec, and bit order is not the
// independent variable of this experiment.
//
// The reordering sweep (SubStreamReordererType::{None, BWT512}) is the same
// experiment one layer down: BWT is a sequential transform offered to the DP per
// segment.  Note what the ladder machinery already tells us about it —
// BWTSectionEncoder declares RandomAccess (decodeAt works at O(W) cost per probe,
// BWTSectionEncoder.hpp ~line 153), so a BWT segment does NOT clear FastSkip.  The
// property survives and the point latency is what pays, which is exactly the case
// a bytes-and-flags table would report as free.
//
// The FPE index-type axis rides along as a fifth "ladder" with NoIndex as rung 0,
// carrying `index_bytes` from customMetadata.  It is here rather than in
// bench_index_oracle because the question is the same shape as a rung's — what a
// random-access property costs in bytes — while the oracle driver's question is
// which index a cost model should pick.
//
// Two deviations from the letter of CONVENTIONS, both deliberate:
//
//   * Rows are buffered per (dataset, ladder, reorderer) group and written when
//     the group finishes, rather than streamed.  `rel_time_vs_rung0` and
//     `rel_bytes_vs_rung0` are required to be precomputed so a plot is a
//     normalised bar chart with no post-processing, and rung 0's cell is not known
//     until it has been measured.  The buffer is one group (rungs x cells x cache
//     states), and the writer is still flushed per group, so a killed run keeps
//     every completed group.
//   * FPE_NoIndex fails a per-probe positional check on any dataset with tiers
//     (FINDINGS section 1: it materializes in the correct order while answering
//     positional queries in its internal one).  It is not dropped from the output,
//     because it is the byte-size baseline the whole FPE group is normalised
//     against; instead its point and gather cells are emitted with skipped=1,
//     positional_valid=0 and null timings, its bytes are kept, and the failure is
//     recorded so the process still exits 2.

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
#include "benchmark/TimingStats.hpp"
#include "benchmark/registry/CodecSetLadder.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "benchmark/targets/PlaygroundTarget.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace encodings;
using namespace encodings::benchmark;

namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);
constexpr size_t kClockCalibrationIterations = 4096;

/// The four selectivities the ablation reports, fixed rather than swept: they are
/// the cells the paper's normalised bar chart has room for, and a rung costs a DP
/// run, so a finer sigma axis multiplies the expensive dimension by nothing.
constexpr double kSigmas[] = {0.01, 0.1, 0.5, 1.0};

// ─── Configuration ───────────────────────────────────────────────────────────

enum class UniverseChoice { All, DpDefault };
enum class ReordererChoice { None, Bwt512, Both };

struct SweepConfig {
    size_t n{1'000'000};
    /// AutoSIS plan selection costs 13.6-14.5 s at its default 100k samples
    /// (FINDINGS) and every rung is a fresh selection, so the default here is an
    /// order of magnitude smaller.  It is the first flag to raise when a rung's
    /// plan looks unstable, and the first to lower when a sweep is too slow.
    size_t sampleSize{10'000};
    size_t sampleBlock{16};
    UniverseChoice universe{UniverseChoice::All};
    ReordererChoice reorderers{ReordererChoice::Both};
    size_t runLength{64 / kElemSize};
    size_t maxRanges{65536};
    size_t probes{1u << 14};
    double zipfTheta{1.4};
    size_t iterations{21};
    size_t warmup{2};
    size_t keepArtifacts{8};
    bool skipFpeIndex{false};
    bool skipBulk{false};
    /// The COLD arm's contract.  The cache-state axis of this driver is
    /// {hot} plus this, because a rung is only interesting against its own
    /// baseline in both states; passing `hot` here collapses the axis to hot only.
    CacheState cacheState{CacheState::ColdAll};
    EvictMethod evictMethod{EvictMethod::Auto};
    uint64_t seed{42};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    std::vector<std::string> ladderFilters;
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_ablation.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Sweep axes:")
        .opt("n", cfg.n, "stream length in elements")
        .enumOpt("universe", cfg.universe,
                 {{"all", UniverseChoice::All}, {"dp-default", UniverseChoice::DpDefault}},
                 "codec universe the ladders are built from")
        .enumOpt("reorderer", cfg.reorderers,
                 {{"none", ReordererChoice::None},
                  {"bwt512", ReordererChoice::Bwt512},
                  {"both", ReordererChoice::Both}},
                 "SubStreamReordererType offered to the DP")
        .repeated("ladder", cfg.ladderFilters, "only ladders whose name contains SUBSTR")
        .flag("skip-fpe-index", cfg.skipFpeIndex, "omit the FPE index-type group")
        .flag("skip-bulk", cfg.skipBulk, "omit the full-materialization cell");

    args.group("Plan selection:")
        .opt("sample-size", cfg.sampleSize, "samples per DP run (cost scales with this)")
        .opt("sample-block", cfg.sampleBlock, "contiguous block size within the sample");

    args.group("Access patterns:")
        .opt("run-length", cfg.runLength, "gather run length in elements")
        .opt("max-ranges", cfg.maxRanges, "cap on gather range count, 0 = unbounded")
        .opt("probes", cfg.probes, "point lookups per timed batch")
        .opt("zipf-theta", cfg.zipfTheta, "skew of the Zipf point trace");

    args.group("Measurement:")
        .opt("iterations", cfg.iterations, "timed repetitions per cell")
        .opt("warmup", cfg.warmup, "untimed repetitions per cell")
        .opt("keep-artifacts", cfg.keepArtifacts, "resident encoded payloads (a ladder at once)")
        .enumOpt("cache-state", cfg.cacheState,
                 {{"hot", CacheState::Hot},
                  {"cold-payload", CacheState::ColdPayload},
                  {"cold-all", CacheState::ColdAll}},
                 "contract for the cold arm; 'hot' collapses the axis to hot only")
        .enumOpt("evict-method", cfg.evictMethod,
                 {{"auto", EvictMethod::Auto},
                  {"clflush", EvictMethod::Clflush},
                  {"llc-thrash", EvictMethod::LlcThrash},
                  {"none", EvictMethod::None}},
                 "how a cold state is produced");

    args.group("Selection and output:")
        .opt("seed", cfg.seed, "all randomness derives from this")
        .repeated("dataset", cfg.datasetFilters, "only datasets whose name contains SUBSTR")
        .repeated("encoder", cfg.encoderFilters, "only variants whose name contains SUBSTR")
        .flag("validate", cfg.validate, "round-trip and positional checks before measuring")
        .flag("dry-run", cfg.dryRun, "print the sweep plan and exit")
        .opt("output", cfg.output, "result file")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

// ─── Cells ───────────────────────────────────────────────────────────────────

enum class Access { Bulk, Range, Gather, Point };

inline const char* accessName(Access a) {
    switch (a) {
        case Access::Bulk:   return "bulk";
        case Access::Range:  return "range";
        case Access::Gather: return "gather";
        case Access::Point:  return "point";
    }
    return "unknown";
}

/// One access pattern, built once per dataset size and shared by every variant —
/// a trace is a property of (N, seed), so rebuilding it per rung would spend the
/// build cost per cell and, for the Zipf pattern, risk two variants being compared
/// on different probe lists.
struct Cell {
    Access access{Access::Bulk};
    std::string pattern{"-"};
    double sigmaNominal{0.0};
    GatherTrace gather;
    PointTrace point;

    std::string label() const {
        switch (access) {
            case Access::Bulk:   return "bulk";
            case Access::Range:  return "range";
            case Access::Gather: return "gather@sigma=" + std::to_string(sigmaNominal);
            case Access::Point:  return "point@" + pattern;
        }
        return "?";
    }
};

std::vector<Cell> buildCells(const SweepConfig& cfg, size_t n) {
    std::vector<Cell> cells;

    if (!cfg.skipBulk) {
        Cell c;
        c.access = Access::Bulk;
        cells.push_back(std::move(c));
    }

    for (const double sigma : kSigmas) {
        GatherAccessParams p;
        p.start       = 0;
        p.span        = n;
        p.selectivity = sigma;
        p.runLength   = std::max<size_t>(1, cfg.runLength);
        p.gapModel    = GapModel::UniformDeterministic;
        p.seed        = cfg.seed;
        p.maxRanges   = cfg.maxRanges;
        Cell c;
        c.access       = Access::Gather;
        c.sigmaNominal = sigma;
        c.gather       = buildGatherTrace(n, p);
        if (c.gather.selectedRows == 0) continue;
        cells.push_back(std::move(c));
    }

    for (const PointPattern pattern : {PointPattern::Uniform, PointPattern::Zipf}) {
        PointTraceParams p;
        p.streamLength = n;
        p.probes       = cfg.probes;
        p.pattern      = pattern;
        p.zipfTheta    = cfg.zipfTheta;
        p.seed         = cfg.seed;
        Cell c;
        c.access  = Access::Point;
        c.pattern = pointPatternName(pattern);
        c.point   = buildPointTrace(p);
        cells.push_back(std::move(c));
    }

    return cells;
}

// ─── Variants ────────────────────────────────────────────────────────────────

/// One thing to measure, plus everything known about it before it is measured.
///
/// A ladder rung and an FPE index type are the same shape here on purpose: both
/// are "a codec, a group, and a position within that group whose element 0 is the
/// baseline", which is all the measurement and emission code needs to know.
struct Variant {
    std::string name;      ///< `encoding` column
    std::string family;
    std::string variantTag;///< `variant` column
    std::shared_ptr<Codec<Elem>> codec;

    std::string ladder;
    size_t rungIndex{0};
    std::string rungName;
    std::string admitted{"-"};
    std::string allowedCodecs{"-"};
    size_t allowedCount{0};
    size_t modelsBuilt{0};
    std::string modelsDropped{"-"};
    std::string reorderer{"None"};

    // Plan facts.  Nullopt for a variant that is not a SubIntSplit plan (the FPE
    // index group), where a segment count would be a fabricated 1.
    std::optional<size_t> segmentCount;
    std::optional<std::string> segmentPlan;
    std::optional<bool> planHasSequentialCodec;
    std::optional<bool> admittedSelected;
    std::optional<double> estBits;
    std::optional<int64_t> selectionNs;

    std::string indexType{"-"};
    /// Bulk-only validation for this variant: its positional answers are known
    /// not to be in stream order (FPE NoIndex), so a per-probe check would fail by
    /// construction and its point/gather cells are not measurable results.
    bool positionalUnordered{false};
};

/// Compact, stable plan rendering for the `segment_plan` column, e.g.
/// "0..12:BlockFrequencyPartitionEncoding|13..13:BitPacking".  Stable matters:
/// two runs of one rung must produce the same string or the column cannot be
/// grouped on.
std::string describePlan(const std::vector<encoders::selectors::SegmentPlan>& segments) {
    std::string out;
    for (const auto& s : segments) {
        if (!out.empty()) out += "|";
        out += std::to_string(s.bitStart) + ".." + std::to_string(s.bitEnd) + ":";
        if (s.reorderer != SubStreamReordererType::None) {
            out += encoders::selectors::subStreamReordererTypeToString(s.reorderer);
            out += "+";
        }
        out += encodingTypeToString(s.encoding);
    }
    return out;
}

/// Run the DP for one rung and build the encoder its plan describes.
///
/// Returns nullopt when the selector produced nothing, which is a rung result
/// ("no plan at this codec set"), not an error — the caller emits it as skipped.
struct PlannedRung {
    std::vector<encoders::selectors::SegmentPlan> segments;
    double totalCostBits{0.0};
    int64_t selectionNs{0};
    size_t modelsBuilt{0};
    std::vector<EncodingType> dropped;
    std::shared_ptr<Codec<Elem>> codec;
};

std::optional<PlannedRung> planRung(const LadderRung& rung, SubStreamReordererType reorderer,
                                    const std::vector<Elem>& sample, size_t fullCount,
                                    const SweepConfig& cfg, std::string& whyNot) {
    const std::vector<CostDim> dims{CostDim{}};  // compression only; see CodecSetLadder

    PlannedRung out;
    std::vector<std::unique_ptr<encoders::selectors::costs::EncodingCostModel>> models;
    try {
        models = buildRungCostModels(rung.allowed, dims, out.dropped);
    } catch (const std::exception& e) {
        whyNot = std::string("cost models: ") + e.what();
        return std::nullopt;
    }
    out.modelsBuilt = models.size();

    std::vector<std::unique_ptr<encoders::selectors::costs::ISubStreamReordererCostModel>> reorderers;
    if (reorderer == SubStreamReordererType::BWT512) {
        reorderers.push_back(std::make_unique<encoders::selectors::costs::BWTReordererCostModel>());
    }

    encoders::selectors::IDSubStreamEncodingSelector::Config sel;
    sel.minSegmentWidth     = 1;
    sel.splitPenalty        = recommendedSplitPenalty(dims, cfg.sampleSize);
    sel.enablePrune         = true;
    sel.enableMergePhase    = false;
    sel.useExhaustiveSearch = false;
    sel.verboseLevel        = 0;
    sel.orderHint           = encoders::BitSplitOrder::LSB_TO_MSB;
    // Left unset deliberately: the selector THROWS if it cannot open the grid CSV
    // for writing, so a debugging artifact would decide whether a rung exists.
    const encoders::selectors::IDSubStreamEncodingSelector selector(sel);

    const auto t0 = std::chrono::high_resolution_clock::now();
    encoders::selectors::IDSubStreamEncodingSelector::Result result;
    try {
        result = selector.select(sample, models, reorderers, fullCount);
    } catch (const std::exception& e) {
        whyNot = std::string("selector: ") + e.what();
        return std::nullopt;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    if (result.segments.empty()) {
        whyNot = "selector returned no segments";
        return std::nullopt;
    }

    out.segments      = result.segments;
    out.totalCostBits = result.total_cost;
    out.selectionNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    try {
        out.codec = encoders::makeSubIntSplitEncoderFromSegments<Elem>(
            out.segments, encoders::BitSplitOrder::LSB_TO_MSB);
    } catch (const std::exception& e) {
        whyNot = std::string("plan not constructible: ") + e.what();
        return std::nullopt;
    }
    return out;
}

std::string joinTypes(const std::vector<EncodingType>& types) {
    if (types.empty()) return "-";
    return describeCodecSet(types);
}

/// Every rung of one (ladder, reorderer) as a measurable group.
std::vector<Variant> buildLadderGroup(Ladder ladder, SubStreamReordererType reorderer,
                                      const std::vector<EncodingType>& universe,
                                      const std::vector<Elem>& sample, size_t fullCount,
                                      const SweepConfig& cfg,
                                      std::vector<std::string>& planFailures) {
    std::vector<Variant> out;
    const std::string reordererTag =
        encoders::selectors::subStreamReordererTypeToString(reorderer);

    for (const LadderRung& rung : buildLadder(ladder, universe)) {
        std::string whyNot;
        auto planned = planRung(rung, reorderer, sample, fullCount, cfg, whyNot);
        if (!planned) {
            planFailures.push_back(std::string(ladderName(ladder)) + "/" + rung.name + " [" +
                                   reordererTag + "]: " + whyNot);
            continue;
        }

        Variant v;
        v.name = std::string(ladderName(ladder)) + "/" + rung.name +
                 (reorderer == SubStreamReordererType::None ? "" : "+BWT512");
        v.family        = "sis-ablation";
        v.variantTag    = rung.name;
        v.codec         = planned->codec;
        v.ladder        = ladderName(ladder);
        v.rungIndex     = out.size();
        v.rungName      = rung.name;
        v.admitted      = rung.admitted.empty() ? "-" : rung.admitted;
        v.allowedCodecs = describeCodecSet(rung.allowed);
        v.allowedCount  = rung.allowed.size();
        v.modelsBuilt   = planned->modelsBuilt;
        v.modelsDropped = joinTypes(planned->dropped);
        v.reorderer     = reordererTag;
        v.segmentCount  = planned->segments.size();
        v.segmentPlan   = describePlan(planned->segments);
        v.estBits       = planned->totalCostBits;
        v.selectionNs   = planned->selectionNs;

        bool hasSequential = false;
        bool admittedSelected = false;
        for (const auto& s : planned->segments) {
            if (!sectionRandomAccess(s.encoding)) hasSequential = true;
            if (!rung.admitted.empty() && encodingTypeToString(s.encoding) == rung.admitted) {
                admittedSelected = true;
            }
        }
        v.planHasSequentialCodec = hasSequential;
        // Null rather than false on rung 0: there is no admitted codec there, and a
        // false would read as "the DP declined it".
        if (!rung.admitted.empty()) v.admittedSelected = admittedSelected;

        out.push_back(std::move(v));
    }
    return out;
}

/// The FPE positional-index axis, NoIndex first so it is the group's baseline.
std::vector<Variant> buildFpeGroup() {
    std::vector<Variant> out;
    for (auto& entry : fpeIndexFamily()) {
        Variant v;
        v.family     = "fpe-index";
        v.name       = entry.name;
        v.variantTag = entry.variant;
        v.codec      = entry.codec;
        v.ladder     = "fpe_index";
        v.rungName   = entry.variant;
        v.indexType  = entry.variant;
        v.reorderer  = "None";
        // NoIndex reorders rows by tier, so pointRead(i) does not answer for row i
        // of the stream (FINDINGS section 1).  Declared up front so its latency
        // cells are never emitted as measurements.
        v.positionalUnordered = entry.variant == "NoIndex";
        out.push_back(std::move(v));
    }
    // Baseline first, then the indexed variants in registry order.
    std::stable_partition(out.begin(), out.end(),
                          [](const Variant& v) { return v.indexType == "NoIndex"; });
    for (size_t i = 0; i < out.size(); ++i) out[i].rungIndex = i;
    return out;
}

// ─── Result schema ───────────────────────────────────────────────────────────

std::vector<ColumnSpec> ablationColumns() {
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
        // Ablation axes.
        stringCol("ladder"),
        intCol("rung_index"),
        stringCol("rung_name"),
        stringCol("admitted_codec"),
        intCol("admitted_selected"),
        stringCol("allowed_codecs"),
        intCol("allowed_count"),
        intCol("models_built"),
        stringCol("models_dropped"),
        stringCol("reorderer"),
        intCol("sample_size"),
        // The chosen plan.
        intCol("segment_count"),
        stringCol("segment_plan"),
        intCol("plan_has_sequential_codec"),
        intCol("sis_fast_skip"),
        doubleCol("est_bits"),
        doubleCol("actual_bits"),
        doubleCol("est_over_actual"),
        intCol("selection_time_ns"),
        // The FPE index axis.
        stringCol("index_type"),
        intCol("index_bytes"),
        intCol("positional_valid"),
        // The access cell.
        stringCol("access"),
        stringCol("pattern"),
        doubleCol("sigma_nominal"),
        doubleCol("sigma_achieved"),
        intCol("k_actual"),
        intCol("selected_elems"),
        intCol("probes"),
        doubleCol("distinct_fraction"),
        // Measurements.
        intCol("time_ns"),
        intCol("time_p90_ns"),
        intCol("time_min_ns"),
        doubleCol("ns_per_probe"),
        doubleCol("clock_overhead_ns"),
        intCol("gather_skip_ns"),
        intCol("gather_materialize_ns"),
        // Precomputed normalisation: a plot of these needs no post-processing.
        stringCol("baseline_variant"),
        doubleCol("rel_time_vs_rung0"),
        doubleCol("rel_bytes_vs_rung0"),
        intCol("truncated"),
        intCol("skipped"),
        stringCol("skip_reason"),
    };
}

// ─── Measurement records ─────────────────────────────────────────────────────

/// One measured (variant, cell, cache state).  Buffered because rel_* against
/// rung 0 cannot be written before rung 0's own cell has been measured.
struct CellResult {
    bool measured{false};
    std::string skipReason;
    int64_t timeNs{0}, p90Ns{0}, minNs{0}, evictNs{0};
    int64_t gatherSkipNs{-1}, gatherMatNs{-1};
    double nsPerProbe{0.0};
};

/// Substring filter over variants, mirroring registry::applyFilters.
///
/// Kept separate because a Variant is not an EncoderEntry: a ladder rung's
/// identity is (ladder, rung, codec), and filtering has to keep whole rungs
/// rather than individual codecs or the rel_*_vs_rung0 columns lose their
/// baseline.  An empty filter list means everything; a filter matching nothing
/// yields an empty group, which the caller reports rather than treating as an
/// error, since one ladder matching nothing is normal when another does.
std::vector<Variant> applyVariantFilters(std::vector<Variant> group,
                                         const std::vector<std::string>& filters) {
    if (filters.empty()) return group;
    std::vector<Variant> kept;
    kept.reserve(group.size());
    for (auto& v : group) {
        const bool hit = std::any_of(filters.begin(), filters.end(),
            [&](const std::string& f) {
                return v.name.find(f) != std::string::npos
                    || v.variantTag.find(f) != std::string::npos
                    || v.rungName.find(f) != std::string::npos;
            });
        if (hit) kept.push_back(std::move(v));
    }
    return kept;
}

struct VariantResult {
    Variant variant;
    bool encoded{false};
    std::string encodeFailure;
    size_t payloadBytes{0};
    double compressionRatio{0.0};
    bool fastSkip{false}, randomAccess{false}, isSequential{false};
    std::optional<double> indexBytes;
    int64_t selectionNs{-1};
    /// Indexed [cellIndex * stateCount + stateIndex].
    std::vector<CellResult> cells;
};

// ─── Preflight ───────────────────────────────────────────────────────────────

void printPreflight(const SweepConfig& cfg, const std::vector<EncodingType>& universe,
                    const std::vector<Ladder>& ladders,
                    const std::vector<SubStreamReordererType>& reorderers,
                    const std::vector<Cell>& cells,
                    const std::vector<DatasetEntry<Elem>>& datasets,
                    const std::vector<const char*>& states, const CacheController& cache,
                    const TimingSummary& clock) {
    size_t raCount = 0;
    for (const EncodingType et : universe) raCount += sectionRandomAccess(et) ? 1 : 0;

    std::cout << "\n── Codec universe ───────────────────────────────────────────\n"
              << "  types          " << universe.size() << " costable and constructible\n"
              << "  random access  " << raCount << "\n"
              << "  sequential     " << (universe.size() - raCount) << "  (";
    bool first = true;
    for (const EncodingType et : universe) {
        if (sectionRandomAccess(et)) continue;
        if (!first) std::cout << ", ";
        std::cout << encodingTypeToString(et);
        first = false;
    }
    std::cout << ")\n";

    std::cout << "\n── Ladders ──────────────────────────────────────────────────\n";
    size_t rungTotal = 0;
    for (const Ladder l : ladders) {
        const auto rungs = buildLadder(l, universe);
        rungTotal += rungs.size() * reorderers.size();
        std::cout << "  " << ladderName(l) << ": " << rungs.size() << " rungs\n";
        for (const LadderRung& r : rungs) {
            std::cout << "      " << r.name << "  (" << r.allowed.size() << " codecs";
            if (!r.admitted.empty()) std::cout << ", admits " << r.admitted;
            std::cout << ")\n";
        }
    }
    const size_t fpeVariants = cfg.skipFpeIndex ? 0 : buildFpeGroup().size();

    std::cout << "\n── Access cells (achieved structure) ────────────────────────\n";
    for (const Cell& c : cells) {
        std::cout << "  " << accessName(c.access);
        if (c.access == Access::Gather) {
            std::cout << "  sigma_nominal=" << c.sigmaNominal << " achieved="
                      << c.gather.selectivityAchieved << " k=" << c.gather.rangeCount
                      << " run=" << c.gather.runLengthActual
                      << " selected=" << c.gather.selectedRows
                      << (c.gather.clamped ? "  (clamped by --max-ranges)" : "");
        } else if (c.access == Access::Point) {
            std::cout << "  " << c.pattern << " probes=" << c.point.indices.size()
                      << " distinct_fraction=" << c.point.distinctFraction
                      << " footprint=" << c.point.footprintSpan;
        } else {
            std::cout << "  " << cfg.n << " elements";
        }
        std::cout << "\n";
    }

    const size_t variants = rungTotal + fpeVariants;
    std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
              << "  N              " << cfg.n << " elements ("
              << (cfg.n * kElemSize / (1024 * 1024)) << " MiB raw)\n"
              << "  sample size    " << cfg.sampleSize << " (one DP run per rung)\n"
              << "  DP runs        " << rungTotal << "\n"
              << "  variants       " << variants << " per dataset\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  cache states   " << states.size() << "\n"
              << "  cells          " << cells.size() << " per (variant, state)\n"
              << "  rows           " << variants * cells.size() * states.size() * datasets.size()
              << "\n"
              << "  cache          " << cache.describe() << "\n"
              << "  iterations     " << cfg.iterations << " timed, " << cfg.warmup << " warmup\n"
              << "  clock overhead " << clock.medianNs << " ns median per timing pair\n"
              << "  artifacts held " << cfg.keepArtifacts << "\n\n";
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Round-trip, the sigma=1 identity, and — for a variant that claims positional
/// order — that pointRead(i) answers for row i.  The last is the check that
/// catches a codec which materializes correctly while indexing internally.
bool validateVariant(DecodeHarness<PlaygroundTarget<Elem>>& harness,
                     PlaygroundTarget<Elem>& target, std::span<const Elem> reference,
                     const std::vector<Cell>& cells, bool positionalUnordered,
                     std::string& whyNot) {
    if (!harness.validate(reference, whyNot)) return false;

    for (const Cell& c : cells) {
        if (c.access != Access::Gather) continue;
        if (!harness.validateGather(c.gather.ranges, reference, whyNot)) return false;
    }

    if (positionalUnordered) return true;

    const size_t n = reference.size();
    for (const size_t index : {size_t{0}, n / 2, n - 1}) {
        const auto v = target.pointRead(index);
        if (!v || *v != reference[index]) {
            whyNot = "pointRead(" + std::to_string(index) + ") does not answer for that row";
            return false;
        }
    }
    // Capped per cell: at 16384 probes a full check is a second sweep.
    constexpr size_t kPerCell = 1024;
    for (const Cell& c : cells) {
        if (c.access != Access::Point) continue;
        const size_t step = std::max<size_t>(1, c.point.indices.size() / kPerCell);
        for (size_t i = 0; i < c.point.indices.size(); i += step) {
            const size_t index = c.point.indices[i];
            const auto v = target.pointRead(index);
            if (!v || *v != reference[index]) {
                whyNot = "pointRead(" + std::to_string(index) + ") under pattern " + c.pattern +
                         " does not answer for that row";
                return false;
            }
        }
    }
    return true;
}

// ─── Group measurement ───────────────────────────────────────────────────────

struct GroupContext {
    const SweepConfig* cfg;
    const std::vector<Cell>* cells;
    const std::vector<CacheState>* states;
    std::vector<std::unique_ptr<CacheController>>* controllers;
    std::span<const Elem> reference;
    std::span<Elem> sink;
    ArtifactCache<Elem>* artifacts;
    const typename DatasetCache<Elem>::Handle* handle;
    std::vector<std::string>* failures;
};

std::vector<VariantResult> measureGroup(const GroupContext& ctx, std::vector<Variant> variants) {
    const SweepConfig& cfg = *ctx.cfg;
    const std::vector<Cell>& cells = *ctx.cells;
    const size_t stateCount = ctx.states->size();

    std::vector<VariantResult> results;
    results.reserve(variants.size());

    for (Variant& v : variants) {
        VariantResult vr;
        vr.variant = v;
        vr.cells.assign(cells.size() * stateCount, CellResult{});

        EncoderEntry<Elem> entry;
        entry.name         = v.name;
        entry.codec        = v.codec;
        entry.family       = v.family;
        entry.variant      = v.variantTag;
        entry.isSequential = deriveIsSequential<Elem>(*v.codec);

        std::cout << "    [" << v.name << "] encoding..." << std::flush;
        const EncodedArtifact<Elem>* artifact = nullptr;
        try {
            artifact = &ctx.artifacts->get(entry, *ctx.handle, EncodeMeasurement::None);
        } catch (const std::exception& e) {
            std::cout << "\n";
            std::cerr << "    [" << v.name << "] encode FAILED: " << e.what() << "\n";
            vr.encodeFailure = e.what();
            ctx.failures->push_back(v.name + ": encode failed: " + e.what());
            results.push_back(std::move(vr));
            continue;
        }

        vr.encoded          = true;
        vr.payloadBytes     = artifact->payloadBytes;
        vr.compressionRatio = artifact->compressionRatio;
        vr.fastSkip         = artifact->fastSkip;
        vr.randomAccess     = artifact->randomAccess;
        vr.selectionNs      = artifact->selectionTimeNs;
        // Post-encode properties, not the pre-encode declaration: for a plan built
        // from DP segments the two agree, but the artifact is the one the sweep
        // actually decoded from.
        vr.isSequential = !vr.fastSkip && !vr.randomAccess;
        if (const auto it = artifact->encodeCustomMetrics.find("index_bytes");
            it != artifact->encodeCustomMetrics.end()) {
            vr.indexBytes = it->second;
        }
        std::cout << " " << vr.payloadBytes << " B (ratio " << vr.compressionRatio << "x"
                  << (vr.fastSkip ? ", FastSkip" : ", no FastSkip") << ")\n";

        PlaygroundTarget<Elem> target(*v.codec);
        target.adopt(artifact->encoded);

        bool positionalOk = !v.positionalUnordered;
        if (cfg.validate) {
            std::string whyNot;
            DecodeHarness<PlaygroundTarget<Elem>> probe(target, *(*ctx.controllers)[0]);
            if (!validateVariant(probe, target, ctx.reference, cells, v.positionalUnordered,
                                 whyNot)) {
                std::cerr << "    [" << v.name << "] validation FAILED: " << whyNot << "\n";
                ctx.failures->push_back(v.name + ": " + whyNot);
                // Bytes survive, timings do not: a variant that decodes wrongly has
                // no meaningful latency, but its size is still a measured fact and
                // is the baseline the rest of an FPE group divides by.
                for (CellResult& c : vr.cells) c.skipReason = "validation failed";
                results.push_back(std::move(vr));
                continue;
            }
        }
        if (v.positionalUnordered) {
            positionalOk = false;
            ctx.failures->push_back(v.name +
                                    ": positional queries answered in internal order; "
                                    "latency cells not measured");
        }

        for (size_t si = 0; si < stateCount; ++si) {
            DecodeHarness<PlaygroundTarget<Elem>> harness(target, *(*ctx.controllers)[si]);
            for (size_t ci = 0; ci < cells.size(); ++ci) {
                const Cell& cell = cells[ci];
                CellResult& out  = vr.cells[ci * stateCount + si];

                if (!positionalOk && cell.access != Access::Bulk) {
                    out.skipReason = "positional order not preserved";
                    continue;
                }

                MeasureSpec spec;
                spec.iterations = cfg.iterations;
                spec.warmup     = cfg.warmup;

                MeasureResult r;
                try {
                    switch (cell.access) {
                        case Access::Bulk:
                            r = harness.bulk(ctx.sink.subspan(0, ctx.reference.size()), spec);
                            break;
                        case Access::Range:
                            r = harness.range(0, ctx.reference.size(), ctx.sink, spec);
                            break;
                        case Access::Gather:
                            r = harness.gather(cell.gather.ranges, ctx.sink,
                                               cell.gather.selectedRows, spec);
                            break;
                        case Access::Point:
                            r = harness.points(cell.point, spec);
                            break;
                    }
                } catch (const std::exception& e) {
                    out.skipReason = e.what();
                    continue;
                }

                const TargetProfile prof = harness.profile();
                out.measured = true;
                out.timeNs   = r.time.medianNs;
                out.p90Ns    = r.time.p90Ns;
                out.minNs    = r.time.minNs;
                out.evictNs  = r.evict.medianNs;
                out.gatherSkipNs = cell.access == Access::Gather ? prof.gatherSkipNs : -1;
                out.gatherMatNs  = cell.access == Access::Gather ? prof.gatherMaterializeNs : -1;
                out.nsPerProbe   = cell.access == Access::Point
                                     ? nsPerProbe(r, cell.point.indices.size())
                                     : 0.0;
            }
        }

        // The payload and codec state go before the next variant is timed only if
        // the cache is at its bound; --keep-artifacts holds a whole ladder so the
        // rungs of one group stay comparable within a process.
        results.push_back(std::move(vr));
    }
    return results;
}

void emitGroup(ResultWriter& writer, const SweepConfig& cfg, const std::string& datasetName,
               const std::vector<Cell>& cells, const std::vector<CacheState>& states,
               const std::vector<std::unique_ptr<CacheController>>& controllers,
               const TimingSummary& clock, const std::vector<VariantResult>& group) {
    if (group.empty()) return;
    const VariantResult& base = group.front();
    const size_t stateCount = states.size();

    for (const VariantResult& vr : group) {
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            for (size_t si = 0; si < stateCount; ++si) {
                const Cell& cell      = cells[ci];
                const CellResult& res = vr.cells[ci * stateCount + si];
                const CellResult& ref = base.cells[ci * stateCount + si];
                const Variant& v      = vr.variant;

                auto row = writer.row();
                row.set("driver", "bench_ablation")
                    .set("dataset", datasetName)
                    .set("encoding", v.name)
                    .set("family", v.family)
                    .set("variant", v.variantTag)
                    .set("is_sequential", vr.isSequential)
                    .set("fast_skip", vr.fastSkip)
                    .set("random_access", vr.randomAccess)
                    .set("N", cfg.n)
                    .set("seed", cfg.seed)
                    .set("cache_state",
                         cacheStateName(controllers[si]->effectivePolicy().state))
                    .set("evict_method",
                         evictMethodName(controllers[si]->effectivePolicy().method))
                    .set("iterations", cfg.iterations)
                    .set("warmup", cfg.warmup)
                    .set("ladder", v.ladder)
                    .set("rung_index", v.rungIndex)
                    .set("rung_name", v.rungName)
                    .set("admitted_codec", v.admitted)
                    .set("allowed_codecs", v.allowedCodecs)
                    .set("allowed_count", v.allowedCount)
                    .set("models_built", v.modelsBuilt)
                    .set("models_dropped", v.modelsDropped)
                    .set("reorderer", v.reorderer)
                    .set("sample_size", cfg.sampleSize)
                    .set("index_type", v.indexType)
                    .set("access", accessName(cell.access))
                    .set("pattern", cell.pattern)
                    .set("baseline_variant", base.variant.rungName)
                    .set("truncated", cell.access == Access::Gather && cell.gather.clamped)
                    .set("skipped", !res.measured)
                    .set("clock_overhead_ns", static_cast<double>(clock.medianNs))
                    .setIf(res.measured || !vr.encoded ? false : true, "positional_valid", true);

                // positional_valid is a three-way fact: true when checked and
                // ordered, false when the codec answers in its own order, null when
                // --validate did not run.  setIf above only covers the true case, so
                // the other two are written explicitly.
                if (v.positionalUnordered) row.set("positional_valid", false);
                else if (!cfg.validate) row.setNull("positional_valid");
                else row.set("positional_valid", true);

                row.setIf(vr.encoded, "payload_bytes", vr.payloadBytes)
                    .setIf(vr.encoded, "compression_ratio", vr.compressionRatio)
                    .setIf(vr.encoded, "sis_fast_skip", vr.fastSkip)
                    .setIf(vr.selectionNs >= 0, "selection_time_ns", vr.selectionNs)
                    .setIf(vr.indexBytes.has_value(), "index_bytes",
                           static_cast<int64_t>(vr.indexBytes.value_or(0.0)))
                    .setIf(v.segmentCount.has_value(), "segment_count",
                           v.segmentCount.value_or(0))
                    .setIf(v.segmentPlan.has_value(), "segment_plan",
                           v.segmentPlan.value_or(std::string{}))
                    .setIf(v.planHasSequentialCodec.has_value(), "plan_has_sequential_codec",
                           v.planHasSequentialCodec.value_or(false))
                    .setIf(v.admittedSelected.has_value(), "admitted_selected",
                           v.admittedSelected.value_or(false))
                    .setIf(v.estBits.has_value(), "est_bits", v.estBits.value_or(0.0));

                const double actualBits = static_cast<double>(vr.payloadBytes) * 8.0;
                row.setIf(vr.encoded, "actual_bits", actualBits)
                    .setIf(v.estBits.has_value() && actualBits > 0.0, "est_over_actual",
                           v.estBits.value_or(0.0) / (actualBits > 0.0 ? actualBits : 1.0));

                // Cell structure, always reported as achieved.
                if (cell.access == Access::Gather) {
                    row.set("sigma_nominal", cell.sigmaNominal)
                        .set("sigma_achieved", cell.gather.selectivityAchieved)
                        .set("k_actual", cell.gather.rangeCount)
                        .set("selected_elems", cell.gather.selectedRows)
                        .setNull("probes")
                        .setNull("distinct_fraction");
                } else if (cell.access == Access::Point) {
                    row.setNull("sigma_nominal")
                        .setNull("sigma_achieved")
                        .setNull("k_actual")
                        .setNull("selected_elems")
                        .set("probes", cell.point.indices.size())
                        .set("distinct_fraction", cell.point.distinctFraction);
                } else {
                    row.setNull("sigma_nominal")
                        .setNull("sigma_achieved")
                        .setNull("k_actual")
                        .set("selected_elems", cfg.n)
                        .setNull("probes")
                        .setNull("distinct_fraction");
                }

                if (!res.measured) {
                    // Emitted, never dropped: a rung that could not be measured in
                    // one cell is a result about that rung, and a missing row is
                    // indistinguishable from a crash.
                    row.setNull("evict_ns")
                        .setNull("time_ns")
                        .setNull("time_p90_ns")
                        .setNull("time_min_ns")
                        .setNull("ns_per_probe")
                        .setNull("gather_skip_ns")
                        .setNull("gather_materialize_ns")
                        .setNull("rel_time_vs_rung0")
                        .set("skip_reason", res.skipReason.empty()
                                                ? (vr.encoded ? "not measured" : "encode failed")
                                                : res.skipReason);
                } else {
                    row.set("evict_ns", res.evictNs)
                        .set("time_ns", res.timeNs)
                        .set("time_p90_ns", res.p90Ns)
                        .set("time_min_ns", res.minNs)
                        .setIf(cell.access == Access::Point, "ns_per_probe", res.nsPerProbe)
                        .setIf(res.gatherSkipNs >= 0, "gather_skip_ns", res.gatherSkipNs)
                        .setIf(res.gatherMatNs >= 0, "gather_materialize_ns", res.gatherMatNs)
                        // Null, not 1.0, when the baseline cell itself was not
                        // measured: a ratio against an unmeasured denominator is not
                        // a normalisation, and 1.0 would plot as "no change".
                        .setIf(ref.measured && ref.timeNs > 0, "rel_time_vs_rung0",
                               static_cast<double>(res.timeNs) /
                                   static_cast<double>(ref.timeNs > 0 ? ref.timeNs : 1))
                        .setNull("skip_reason");
                }

                row.setIf(vr.encoded && base.payloadBytes > 0, "rel_bytes_vs_rung0",
                          static_cast<double>(vr.payloadBytes) /
                              static_cast<double>(base.payloadBytes > 0 ? base.payloadBytes : 1));
                writer.write(std::move(row));
            }
        }
    }
    writer.flush();
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_ablation",
                   "Codec-set and reordering ladders for SubIntSplit: what the "
                   "random-access property costs and what admitting a sequential "
                   "codec buys.");
    bindArgs(args, cfg);
    switch (args.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    cfg.iterations = std::max<size_t>(1, cfg.iterations);
    cfg.sampleSize = std::max<size_t>(1, cfg.sampleSize);

    const std::vector<EncodingType> universe =
        cfg.universe == UniverseChoice::All ? codecUniverse() : dpDefaultUniverse();

    std::vector<Ladder> ladders;
    for (const Ladder l : {Ladder::RaOnlyThenSequentialOneByOne, Ladder::RawUpwardThroughRa,
                           Ladder::SequentialOnly, Ladder::RaVsSequentialWhole}) {
        if (cfg.ladderFilters.empty()) {
            ladders.push_back(l);
            continue;
        }
        for (const std::string& needle : cfg.ladderFilters) {
            if (std::string(ladderName(l)).find(needle) != std::string::npos) {
                ladders.push_back(l);
                break;
            }
        }
    }
    if (ladders.empty() && !cfg.skipFpeIndex && !cfg.ladderFilters.empty()) {
        std::cerr << "ERROR: no ladders match --ladder filters\n";
        return 1;
    }

    auto datasets = applyFilters(int64Datasets(), cfg.datasetFilters);
    if (datasets.empty()) {
        std::cerr << "ERROR: no datasets match --dataset filters\n";
        return 1;
    }

    std::vector<SubStreamReordererType> reorderers;
    if (cfg.reorderers != ReordererChoice::Bwt512)
        reorderers.push_back(SubStreamReordererType::None);
    if (cfg.reorderers != ReordererChoice::None)
        reorderers.push_back(SubStreamReordererType::BWT512);

    // The cache-state axis: hot always, plus the requested cold contract unless it
    // is itself hot.  Two controllers rather than one re-resolved policy, so each
    // row is labelled by the controller that produced it.
    std::vector<CacheState> states{CacheState::Hot};
    if (cfg.cacheState != CacheState::Hot) states.push_back(cfg.cacheState);

    const CacheTopology topo = CacheTopology::detect();
    std::vector<std::unique_ptr<CacheController>> controllers;
    try {
        for (const CacheState s : states) {
            CachePolicy policy;
            policy.state  = s;
            policy.method = cfg.evictMethod;
            controllers.push_back(std::make_unique<CacheController>(policy, topo));
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    std::vector<const char*> stateNames;
    for (const auto& c : controllers) stateNames.push_back(cacheStateName(c->effectivePolicy().state));

    const std::vector<Cell> cells = buildCells(cfg, cfg.n);
    const TimingSummary clock = measureClockOverhead(kClockCalibrationIterations);
    printPreflight(cfg, universe, ladders, reorderers, cells, datasets, stateNames,
                   *controllers.front(), clock);
    if (cfg.dryRun) {
        std::cout << "Dry run: no measurements taken.\n";
        return 0;
    }

    RunManifest manifest = RunManifest::capture("bench_ablation", args.argvEcho());
    manifest.seed = cfg.seed;
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    for (const Ladder l : ladders) manifest.encoders.push_back(ladderName(l));
    manifest.extra["universe_size"]  = std::to_string(universe.size());
    manifest.extra["sample_size"]    = std::to_string(cfg.sampleSize);
    manifest.extra["cold_state"]     = cacheStateName(controllers.back()->effectivePolicy().state);
    manifest.extra["evict_method"]   = evictMethodName(controllers.back()->effectivePolicy().method);
    manifest.extra["clock_overhead_ns"] = std::to_string(clock.medianNs);
    manifest.writeSidecar(cfg.output);

    ResultWriter writer(cfg.output, ablationColumns(), cfg.format);

    // One sink for the whole run, sized for the widest cell (bulk / sigma = 1).
    std::vector<Elem> sink(cfg.n + 1);
    std::vector<std::string> failures;

    DatasetCache<Elem> dataCache;
    ArtifactCache<Elem> artifacts(
        typename ArtifactCache<Elem>::Options{std::max<size_t>(1, cfg.keepArtifacts)});

    for (auto& ds : datasets) {
        std::cout << "══ Dataset: " << ds.name << " ══\n  loading " << cfg.n << " elements..."
                  << std::flush;
        typename DatasetCache<Elem>::Handle handle;
        try {
            handle = dataCache.materialize(ds, cfg.n);
        } catch (const std::exception& e) {
            std::cout << "\n";
            std::cerr << "  WARNING: dataset '" << ds.name << "' skipped: " << e.what() << "\n";
            continue;
        }
        std::cout << " got " << handle.n << ".\n";

        // One sample per (dataset, N), shared by every rung: the DP's input must be
        // identical across rungs or a rung difference is partly a sample
        // difference.
        typename generators::samplers::StreamSampler<Elem>::Config samplerCfg;
        samplerCfg.maxSamples    = cfg.sampleSize;
        samplerCfg.blockSize     = cfg.sampleBlock;
        samplerCfg.stride        = 0;
        samplerCfg.maxPercentage = 0;
        if (samplerCfg.stride == 0) {
            samplerCfg.stride = std::max<size_t>(1, handle.n / std::max<size_t>(1, cfg.sampleSize));
        }
        std::vector<Elem> sample =
            generators::samplers::StreamSampler<Elem>::sample(handle.data, samplerCfg);
        if (sample.empty()) sample.push_back(handle.data.front());
        std::cout << "  sample: " << sample.size() << " values\n";

        GroupContext ctx;
        ctx.cfg         = &cfg;
        ctx.cells       = &cells;
        ctx.states      = &states;
        ctx.controllers = &controllers;
        ctx.reference   = handle.data;
        ctx.sink        = std::span<Elem>(sink);
        ctx.artifacts   = &artifacts;
        ctx.handle      = &handle;
        ctx.failures    = &failures;

        for (const Ladder ladder : ladders) {
            for (const SubStreamReordererType reorderer : reorderers) {
                std::cout << "  ── " << ladderName(ladder) << " ["
                          << encoders::selectors::subStreamReordererTypeToString(reorderer)
                          << "]\n";
                std::vector<Variant> group = buildLadderGroup(
                    ladder, reorderer, universe, sample, handle.n, cfg, failures);
                group = applyVariantFilters(std::move(group), cfg.encoderFilters);
                if (group.empty()) {
                    std::cout << "    (no rungs)\n";
                    continue;
                }
                emitGroup(writer, cfg, ds.name, cells, states, controllers, clock,
                          measureGroup(ctx, std::move(group)));
            }
        }

        if (!cfg.skipFpeIndex) {
            std::cout << "  ── fpe_index\n";
            std::vector<Variant> group =
                applyVariantFilters(buildFpeGroup(), cfg.encoderFilters);
            if (!group.empty()) {
                emitGroup(writer, cfg, ds.name, cells, states, controllers, clock,
                          measureGroup(ctx, std::move(group)));
            }
        }

        artifacts.clear();
        dataCache.releaseAll();
    }

    writer.close();
    std::cout << "\nResults written to: " << std::filesystem::absolute(cfg.output) << std::endl;

    int exitCode = 0;
    if (!failures.empty()) {
        std::cerr << "\n" << failures.size() << " variant(s) failed or were partially excluded:\n";
        for (const auto& f : failures) std::cerr << "  - " << f << "\n";
        exitCode = 2;
    }

    manifest.finishedAtIso = detail::isoNow();
    manifest.exitCode      = exitCode;
    manifest.writeSidecar(cfg.output);
    return exitCode;
}
