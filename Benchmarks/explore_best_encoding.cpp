#include "encoders/SubIntSplitEncoder.hpp"
#include "encoders/SubIntEncodingUtils.hpp"
#include "encoders/OpenZLEncoder.hpp"
#include "encoders/BlockFORFPEEncoder.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"
#include "encoders/selectors/MetricCollector.hpp"
#include "generators/ParquetColumnGenerator.hpp"
#include "generators/samplers/StreamSampling.hpp"
#include "generators/samplers/EncodingSamplingProfile.hpp"
#include "encodings/EncodingType.hpp"
#include "benchmark/OperatorGraphJson.hpp"
#include "benchmark/OracleGrid.hpp"
#include "benchmark/OpenZLGraphAnalysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Needed for ZSTD_c_literalCompressionMode/ZSTD_lcm_uncompressed (see
// zstdCompressLiteralsRaw below) -- these are "advanced" params gated behind
// this macro since their numeric ids aren't guaranteed stable across zstd
// versions. They're still dispatched through the fully stable, ABI-safe
// ZSTD_CCtx_setParameter(cctx, int, int) entry point, so this is safe to use
// against a dynamically-linked libzstd -- only the enum VALUES themselves
// come from this header, not any non-stable-ABI internals.
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

using namespace encodings;
using namespace encodings::encoders;
using namespace encodings::encoders::selectors;
using namespace encodings::encoders::selectors::costs;
using namespace encodings::generators::samplers;
namespace opgraph = encodings::benchmark::operatorgraph;

// The oracle grid and the OpenZL graph analysis used to be defined in this file.
// They now live in Source/benchmark/ so bench_costmodel_oracle and
// bench_openzl_graph can share them; this tool is a consumer like any other.
using namespace encodings::benchmark;

// ---------------------------------------------------------------------------
// Configuration — change these to explore different datasets or encoding sets.
// ---------------------------------------------------------------------------

struct ExploreConfig {
    // Dataset
    std::filesystem::path datasetPath =
        "/home/david/Documents/PhD/symbol-store/MetaNimbleProject/"
        "EncodingsPlayground/Datasets/TwitterSnowflake/tweet_ids.parquet";
    std::string columnName    = "tweet_id";
    std::string datasetLabel  = "Twitter Snowflake IDs";
    size_t datasetSize        = 10'000'000;
    // size_t datasetSize        = 100'000;

    // Encoding types used by BOTH AutoSIS cost models and the oracle search.
    // Defaults to the same set as CostModelSet-based AutoSIS_C in run_benchmarks.cpp.
    // std::vector<EncodingType> encodingTypes = CostModelSet::defaultEncodings();
    std::vector<EncodingType> encodingTypes = {
        EncodingType::RawEncoding,
        EncodingType::BitPacking,
        EncodingType::RunLengthEncoding,
        EncodingType::AdaptiveFrameOfReference,
        EncodingType::DictionaryEncoding,
        EncodingType::AdaptiveDictionaryEncoding,
        EncodingType::FrequencyPartitionEncoding,
        EncodingType::MainlyConstantEncoding,
        EncodingType::FrameOfReference,
        EncodingType::AdaptiveFramedBitPrefix,
        EncodingType::HuffmanEncoding,
        EncodingType::LZ4,
        EncodingType::FSEEncoding,
        EncodingType::OpenZL,
        EncodingType::BlockFrequencyPartitionEncoding,
        EncodingType::BlockFrequencyPartitionFOREncoding,
        EncodingType::BlockFSEEncoding,
        EncodingType::BlockFORFPEEncoding,
        EncodingType::RangePackFrequencyPartitionEncoding,
        EncodingType::RangePackBlockFrequencyPartitionEncoding,
        EncodingType::CascadingFrameOfReference,
        EncodingType::CascadingFORBlockFrequencyPartitionEncoding,
        EncodingType::RunLengthCascadingFOREncoding,
        EncodingType::CascadingFORFSEEncoding,
        EncodingType::CascadingFORBlockFSEEncoding,
        EncodingType::CascadingFORHuffmanEncoding,
        // PREV-policy compositions (bounded-frame consecutive-element delta --
        // see FOREncoder.hpp's FORReferencePolicy::PREV doc). Enabled by
        // default since these are the actual point of this registration: the
        // root-caused fix for bit-range [22..49]'s stubborn BitPacking floor.
        EncodingType::CascadingFORPrevFSEEncoding,
        EncodingType::CascadingFORPrevBlockFSEEncoding,
        EncodingType::CascadingFORPrevHuffmanEncoding,
        EncodingType::CascadingFORPrevFrequencyPartitionEncoding,
        EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding,
    };

    // Sampling — must match the CostModelSet overload of makeDefaultAutoSubIntSplitConfig
    // so the oracle and AutoSIS see the exact same sample.
    // size_t maxSamples      = 100'000;   // matches CostModelSet overload default
    size_t maxSamples      = 10'000;   // matches CostModelSet overload default
    size_t blockSize       = 16;       // matches CostModelSet overload default
    int    minSegmentWidth = 1;        // minimum bits per oracle segment

    // Reordering (BWT): off by default; flipping enables it for both AutoSIS and oracle.
    bool   allowReorderers = false;

    BitSplitOrder order = BitSplitOrder::LSB_TO_MSB;

    // Where to write the combined operator-graph JSON export for
    // plot_operator_graph.py (AutoSIS/OracleSIS bit-range plans + OpenZL DAG).
    std::string jsonOutputPath =
        "/home/david/Documents/PhD/symbol-store/MetaNimbleProject/"
        "EncodingsPlayground/Benchmarks/operator_graphs/twitter_snowflake.json";
};


static IDSubStreamEncodingSelector::Result runAutoSISSelection(
    const std::vector<int64_t>& sample,
    const ExploreConfig& cfg,
    size_t fullCount)
{
    CostModelSet cms;
    cms.add(CostModelDimension::Compression);
    cms.forEncodings(cfg.encodingTypes);

    auto fullCfg = makeDefaultAutoSubIntSplitConfig<int64_t>(
        cfg.order, std::move(cms), /*numSplits=*/-1, cfg.allowReorderers);
    fullCfg.selectorConfig.verboseLevel = 0;

    IDSubStreamEncodingSelector selector(fullCfg.selectorConfig);
    return selector.select(sample, fullCfg.costModels, fullCfg.reordererModels, fullCount);
}

// ---------------------------------------------------------------------------
// Cost model accuracy — estimated vs actual for each encoding on a segment.
// ---------------------------------------------------------------------------

struct CostModelEntry {
    EncodingType encoding;
    double       estimatedBits;  // cost model output (bits) for sampleSize values
    size_t       actualBytes;    // actual encoded bytes on the sample
};

static std::vector<CostModelEntry> computeCostModelAccuracy(
    const std::vector<uint64_t>& sectionValues,
    int bitWidth,
    const std::vector<EncodingType>& encodingTypes,
    const EncodingGrid& grid,
    int l, int r)
{
    // Filter to types that have analytical cost models (OpenZL has none).
    std::vector<EncodingType> costModelTypes;
    for (auto t : encodingTypes)
        if (t != EncodingType::OpenZL)
            costModelTypes.push_back(t);
    auto costModels = makeAutoSubIntSplitCostModelsFromTypes(costModelTypes);

    MetricCollector<uint64_t> collector;
    auto metrics = collector.compute(sectionValues, static_cast<MetricFlags>(MetricFlag::All));

    const size_t n = sectionValues.size();
    std::vector<CostModelEntry> entries;
    entries.reserve(costModels.size());

    for (const auto& model : costModels) {
        const double estBits   = model->computeCost(metrics, n, static_cast<size_t>(bitWidth));
        const size_t actBytes  = grid[l][r].bytesFor(model->encodingType());
        entries.push_back({model->encodingType(), estBits, actBytes});
    }

    // Sort by estimated bits ascending (cost model's ranking).
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.estimatedBits < b.estimatedBits; });
    return entries;
}

// ---------------------------------------------------------------------------
// Operator-graph JSON export helpers (bit-range plans)
// ---------------------------------------------------------------------------

// When `enc` is BlockFORFPEEncoding, re-encode just this segment's sample slice
// (BLOCKFORFPE_STATS is set for the whole process in main(), see below) to read
// back the aggregate tier stats BlockFORFPEEncoder writes into customMetadata.
// Returns nullopt for any other encoding, or if the codec throws on this slice.

// Builds one SegmentJson from a plan segment, pulling its ranked alternatives
// from the already-computed EncodingGrid cell for that exact bit-range.
// `isAutoSis` controls whether SegmentPlan::cost is exposed as estimatedCostBits
// (AutoSIS: a cost-model estimate in bits over the full dataset) — for oracle
// plans, seg.cost is already the same value as the grid's bestBytes, so it's
// represented only via sampleBytes, not estimatedCostBits.

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------



// ---------------------------------------------------------------------------
// Per-segment encoding ranking (for all oracle segments)
// ---------------------------------------------------------------------------

static void printSegmentEncodingRanking(
    const std::vector<SegmentPlan>& oraclePlan,
    const std::vector<SegmentPlan>& autoSisPlan,
    const EncodingGrid& grid)
{
    // Build quick lookup for AutoSIS encoding at each oracle segment boundary.
    // We use the cost model entries to find what AutoSIS would pick.
    std::map<std::pair<int,int>, EncodingType> autoSisByBounds;
    for (const auto& seg : autoSisPlan)
        autoSisByBounds[{seg.bitStart, seg.bitEnd}] = seg.encoding;

    std::cout << "\n=== Per-Segment Encoding Ranking (sample bytes) ===\n";

    for (const auto& oSeg : oraclePlan) {
        const int l = oSeg.bitStart, r = oSeg.bitEnd;
        const EncodingCell& cell = grid[l][r];

        if (cell.allResults.empty()) {
            std::cout << "  [" << l << ".." << r << "]  (no results)\n";
            continue;
        }

        const size_t best = cell.bestBytes;

        // AutoSIS's preferred encoding on this boundary (if same boundary exists).
        auto ait = autoSisByBounds.find({l, r});
        const bool sameRange = (ait != autoSisByBounds.end());
        const EncodingType autoSisEnc = sameRange ? ait->second : EncodingType::RawEncoding;

        std::cout << "\n  Segment [" << std::setw(2) << l << ".." << std::setw(2) << r
                  << "]  width=" << (r - l + 1) << "\n";

        for (size_t rank = 0; rank < cell.allResults.size(); ++rank) {
            const auto& [enc, bytes] = cell.allResults[rank];
            const bool isOracle   = (enc == oSeg.encoding);
            const bool isAutoSIS  = sameRange && (enc == autoSisEnc);

            std::string tag = "   ";
            if (isOracle && isAutoSIS) tag = "[OA]";
            else if (isOracle)          tag = "[O] ";
            else if (isAutoSIS)         tag = "[A] ";

            const double pctOver = (best > 0)
                ? (static_cast<double>(bytes) - static_cast<double>(best))
                  / static_cast<double>(best) * 100.0
                : 0.0;

            std::cout << "    #" << std::setw(2) << (rank + 1) << "  "
                      << tag << "  "
                      << padRight(encodingTypeToString(enc), 28)
                      << std::setw(9) << bytes << " bytes";
            if (rank == 0)
                std::cout << "  (best)";
            else
                std::cout << "  (+" << std::fixed << std::setprecision(1) << pctOver << "%)";
            if (isAutoSIS && !isOracle)
                std::cout << "  <- AutoSIS picked this";
            std::cout << "\n";
        }

        if (!sameRange)
            std::cout << "    [AutoSIS used different split boundaries here]\n";
    }
}

// ---------------------------------------------------------------------------
// Cost model accuracy table (per oracle segment)
// ---------------------------------------------------------------------------

static void printCostModelAccuracy(
    const std::vector<SegmentPlan>& oraclePlan,
    const std::vector<int64_t>& sample,
    const std::vector<EncodingType>& encodingTypes,
    const EncodingGrid& grid)
{
    const std::vector<uint64_t> uSample = [&] {
        std::vector<uint64_t> v(sample.size());
        for (size_t i = 0; i < sample.size(); ++i) v[i] = static_cast<uint64_t>(sample[i]);
        return v;
    }();

    std::cout << "\n=== Cost Model Accuracy (oracle segment boundaries) ===\n";
    std::cout << "  (est = cost model predicted bits/elem; act = actual bits/elem on sample; "
                 "error = (est-act)/act)\n";

    for (const auto& oSeg : oraclePlan) {
        const int l = oSeg.bitStart, r = oSeg.bitEnd;
        const int width = r - l + 1;
        const size_t n = sample.size();
        const uint64_t mask = (width == 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);

        std::vector<uint64_t> sectionValues(n);
        for (size_t i = 0; i < n; ++i)
            sectionValues[i] = (uSample[i] >> l) & mask;

        auto entries = computeCostModelAccuracy(sectionValues, width, encodingTypes, grid, l, r);

        // Encoding picked by cost model = first entry (lowest est bits)
        const EncodingType modelPick = entries.empty() ? oSeg.encoding : entries.front().encoding;

        std::cout << "\n  Segment [" << std::setw(2) << l << ".." << std::setw(2) << r
                  << "]  width=" << width
                  << "  oracle=" << encodingTypeToString(oSeg.encoding)
                  << "  model_pick=" << encodingTypeToString(modelPick)
                  << (modelPick == oSeg.encoding ? "  ✓" : "  ✗")
                  << "\n";
        std::cout << "    " << padRight("Encoding", 28)
                  << " " << std::setw(14) << "est(bits/elem)"
                  << " " << std::setw(14) << "act(bits/elem)"
                  << " " << std::setw(10) << "error"
                  << "\n";
        std::cout << "    " << std::string(66, '-') << "\n";

        for (const auto& e : entries) {
            const double actBPE = (e.actualBytes == std::numeric_limits<size_t>::max())
                ? std::numeric_limits<double>::quiet_NaN()
                : static_cast<double>(e.actualBytes) * 8.0 / static_cast<double>(n);
            const double estBPE = e.estimatedBits / static_cast<double>(n);
            const double err    = (actBPE > 0 && !std::isnan(actBPE))
                ? (estBPE - actBPE) / actBPE * 100.0
                : std::numeric_limits<double>::quiet_NaN();

            const bool isOracle = (e.encoding == oSeg.encoding);
            const bool isModel  = (e.encoding == modelPick);
            std::string marker = "   ";
            if (isOracle && isModel) marker = "[OM]";
            else if (isOracle)       marker = "[O] ";
            else if (isModel)        marker = "[M] ";

            std::cout << "    " << marker << " "
                      << padRight(encodingTypeToString(e.encoding), 28)
                      << std::fixed << std::setprecision(3)
                      << " " << std::setw(14) << estBPE;
            if (!std::isnan(actBPE))
                std::cout << " " << std::setw(14) << actBPE;
            else
                std::cout << " " << std::setw(14) << "N/A";
            if (!std::isnan(err))
                std::cout << " " << std::setw(9) << err << "%";
            else
                std::cout << " " << std::setw(10) << "N/A";
            std::cout << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Gap attribution — how much of AutoSIS→Oracle gap comes from each segment
// ---------------------------------------------------------------------------

static void printGapAttribution(
    const std::vector<SegmentPlan>& oraclePlan,
    const std::vector<SegmentPlan>& autoSisPlan,
    const std::vector<EncodingType>& encodingTypes,
    const std::vector<int64_t>& sample,
    const EncodingGrid& grid,
    size_t fullDatasetSize)
{
    const double scale = static_cast<double>(fullDatasetSize) / static_cast<double>(sample.size());

    // For each oracle segment, find what AutoSIS would pick (by cost model) on that boundary.
    // We use the cost model accuracy entries to find the model's preferred encoding.
    const std::vector<uint64_t> uSample = [&] {
        std::vector<uint64_t> v(sample.size());
        for (size_t i = 0; i < sample.size(); ++i) v[i] = static_cast<uint64_t>(sample[i]);
        return v;
    }();

    // Build AutoSIS plan boundary lookup.
    std::map<std::pair<int,int>, EncodingType> autoSisByBounds;
    for (const auto& seg : autoSisPlan)
        autoSisByBounds[{seg.bitStart, seg.bitEnd}] = seg.encoding;

    std::cout << "\n=== Gap Attribution (AutoSIS vs Oracle, full-dataset extrapolated) ===\n";
    std::cout << "  Scale factor: " << std::fixed << std::setprecision(1) << scale << "x "
              << "(sample=" << sample.size() << "  full=" << fullDatasetSize << ")\n\n";

    struct GapRow {
        int l, r;
        std::string autoSisEncName, oracleEncName;
        long long autoSisBytes, oracleBytes;  // on sample
        long long gapSample;
        double    gapFullEst;
        bool      boundaryMatch;
    };

    std::vector<GapRow> rows;
    long long totalGapSample = 0;

    for (const auto& oSeg : oraclePlan) {
        const int l = oSeg.bitStart, r = oSeg.bitEnd;
        const int width = r - l + 1;
        const size_t n = sample.size();
        const uint64_t mask = (width == 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);

        std::vector<uint64_t> sectionValues(n);
        for (size_t i = 0; i < n; ++i)
            sectionValues[i] = (uSample[i] >> l) & mask;

        auto ait = autoSisByBounds.find({l, r});
        const bool sameRange = (ait != autoSisByBounds.end());

        EncodingType autoSisEnc = oSeg.encoding; // fallback: oracle enc
        if (sameRange) {
            autoSisEnc = ait->second;
        } else {
            // AutoSIS doesn't use this boundary — find cost-model pick from accuracy table.
            auto entries = computeCostModelAccuracy(sectionValues, width, encodingTypes, grid, l, r);
            if (!entries.empty())
                autoSisEnc = entries.front().encoding;
        }

        const size_t autoSisBytesOnSeg = grid[l][r].bytesFor(autoSisEnc);
        const size_t oracleBytesOnSeg  = grid[l][r].bestBytes;

        const long long gapSample = (autoSisBytesOnSeg != std::numeric_limits<size_t>::max())
            ? static_cast<long long>(autoSisBytesOnSeg) - static_cast<long long>(oracleBytesOnSeg)
            : 0LL;
        totalGapSample += gapSample;

        rows.push_back({
            l, r,
            encodingTypeToString(autoSisEnc),
            encodingTypeToString(oSeg.encoding),
            static_cast<long long>(autoSisBytesOnSeg),
            static_cast<long long>(oracleBytesOnSeg),
            gapSample,
            static_cast<double>(gapSample) * scale,
            sameRange
        });
    }

    // Sort descending by gap magnitude.
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return std::abs(a.gapSample) > std::abs(b.gapSample); });

    std::cout << "  " << padRight("Segment", 10)
              << "  " << padRight("AutoSIS encoding", 26)
              << "  " << padRight("Oracle encoding", 26)
              << "  " << std::setw(11) << "sampleGap"
              << "  " << std::setw(14) << "fullGap(est)"
              << "  " << std::setw(7) << "share"
              << "\n";
    std::cout << "  " << std::string(100, '-') << "\n";

    for (const auto& row : rows) {
        const double share = (totalGapSample != 0)
            ? static_cast<double>(row.gapSample) / static_cast<double>(std::abs(totalGapSample)) * 100.0
            : 0.0;
        std::string segStr = "[" + std::to_string(row.l) + ".." + std::to_string(row.r) + "]";
        std::string boundNote = row.boundaryMatch ? "" : " *";

        std::cout << "  " << padRight(segStr, 10)
                  << "  " << padRight(row.autoSisEncName + boundNote, 26)
                  << "  " << padRight(row.oracleEncName, 26)
                  << "  " << std::setw(11) << row.gapSample
                  << "  " << std::setw(14) << std::fixed << std::setprecision(0) << row.gapFullEst
                  << "  " << std::setw(6)  << std::setprecision(1) << share << "%"
                  << "\n";
    }

    const double totalGapFullEst = static_cast<double>(totalGapSample) * scale;
    std::cout << "  " << std::string(100, '-') << "\n";
    std::cout << "  Total extrapolated gap: "
              << std::fixed << std::setprecision(0) << totalGapFullEst
              << " bytes  (sample gap=" << totalGapSample << " bytes)\n";
    std::cout << "  (* = AutoSIS used different split boundaries; cost-model pick used instead)\n";
}

// ---------------------------------------------------------------------------
// Summary quality metrics
// ---------------------------------------------------------------------------

static void printQualityMetrics(
    size_t autoSisBytes,
    size_t oracleBytes,
    std::optional<size_t> consecOracleBytes,
    std::optional<size_t> mergedOracleBytes,
    std::optional<size_t> openZlBytes,
    size_t datasetSize,
    const std::vector<SegmentPlan>& autoSisPlan,
    const std::vector<SegmentPlan>& oraclePlan,
    std::optional<size_t> autoSisThenOpenZlBytes = std::nullopt,
    std::optional<size_t> oracleThenOpenZlBytes = std::nullopt,
    std::optional<size_t> consecOracleThenOpenZlBytes = std::nullopt,
    std::optional<size_t> mergedOracleThenOpenZlBytes = std::nullopt,
    std::optional<size_t> openZlLowestBytes = std::nullopt,
    std::optional<size_t> openZlHighestBytes = std::nullopt)
{
    auto bpe = [&](size_t bytes) {
        return static_cast<double>(bytes) * 8.0 / static_cast<double>(datasetSize);
    };

    auto pctVsAutoSIS = [&](size_t flat) -> double {
        return (autoSisBytes > 0)
            ? (static_cast<double>(autoSisBytes) - static_cast<double>(flat))
              / static_cast<double>(autoSisBytes) * 100.0
            : 0.0;
    };
    auto pctVsOracle = [&](size_t flat) -> double {
        return (oracleBytes > 0)
            ? (static_cast<double>(oracleBytes) - static_cast<double>(flat))
              / static_cast<double>(oracleBytes) * 100.0
            : 0.0;
    };

    const double overheadVsOracle = (oracleBytes > 0)
        ? (static_cast<double>(autoSisBytes) - static_cast<double>(oracleBytes))
          / static_cast<double>(oracleBytes) * 100.0
        : 0.0;
    const double efficiency = (autoSisBytes > 0)
        ? static_cast<double>(oracleBytes) / static_cast<double>(autoSisBytes) * 100.0
        : 100.0;

    size_t matchCount = 0;
    std::map<std::pair<int,int>, EncodingType> oracleMap;
    for (const auto& seg : oraclePlan)
        oracleMap[{seg.bitStart, seg.bitEnd}] = seg.encoding;
    for (const auto& aSeg : autoSisPlan) {
        auto it = oracleMap.find({aSeg.bitStart, aSeg.bitEnd});
        if (it != oracleMap.end() && it->second == aSeg.encoding)
            ++matchCount;
    }

    std::cout << "\n=== Quality Metrics ===\n";
    std::cout << std::fixed;
    std::cout << "  AutoSIS    compressed:  " << std::setw(12) << autoSisBytes
              << " bytes  (" << std::setprecision(3) << bpe(autoSisBytes) << " bits/elem)\n";
    std::cout << "  Oracle (random)  :      " << std::setw(12) << oracleBytes
              << " bytes  (" << std::setprecision(3) << bpe(oracleBytes) << " bits/elem)\n";
    if (consecOracleBytes.has_value()) {
        const size_t co = *consecOracleBytes;
        const double pctVsRandom = (oracleBytes > 0)
            ? (static_cast<double>(oracleBytes) - static_cast<double>(co))
              / static_cast<double>(oracleBytes) * 100.0
            : 0.0;
        std::cout << "  Oracle (consec)  :      " << std::setw(12) << co
                  << " bytes  (" << std::setprecision(3) << bpe(co) << " bits/elem)"
                  << "  [" << std::setprecision(1) << (pctVsRandom >= 0 ? "+" : "")
                  << pctVsRandom << "% vs random oracle]\n";
    }
    if (mergedOracleBytes.has_value()) {
        const size_t mo = *mergedOracleBytes;
        const double pctVsRandom = (oracleBytes > 0)
            ? (static_cast<double>(oracleBytes) - static_cast<double>(mo))
              / static_cast<double>(oracleBytes) * 100.0
            : 0.0;
        std::cout << "  Oracle (merged)  :      " << std::setw(12) << mo
                  << " bytes  (" << std::setprecision(3) << bpe(mo) << " bits/elem)"
                  << "  [" << std::setprecision(1) << (pctVsRandom >= 0 ? "+" : "")
                  << pctVsRandom << "% vs random oracle]\n";
    }
    if (openZlBytes.has_value()) {
        const size_t oz = *openZlBytes;
        std::cout << "  OpenZL     flat:        " << std::setw(12) << oz
                  << " bytes  (" << std::setprecision(3) << bpe(oz) << " bits/elem)"
                  << "  [AutoSIS " << (pctVsAutoSIS(oz) >= 0 ? "+" : "")
                  << std::setprecision(2) << pctVsAutoSIS(oz)
                  << "% | Oracle " << (pctVsOracle(oz) >= 0 ? "+" : "")
                  << pctVsOracle(oz) << "% vs OpenZL]\n";
    } else {
        std::cout << "  OpenZL     flat:        (unavailable — HAVE_OPENZL not defined)\n";
    }
    auto printFlatOpenZlLevel = [&](const char* label, std::optional<size_t> bytes) {
        if (!bytes.has_value()) return;
        const size_t b = *bytes;
        std::cout << "  " << label << std::setw(12) << b
                  << " bytes  (" << std::setprecision(3) << bpe(b) << " bits/elem)";
        if (openZlBytes.has_value()) {
            const double pctVsDefault = (*openZlBytes > 0)
                ? (static_cast<double>(*openZlBytes) - static_cast<double>(b))
                  / static_cast<double>(*openZlBytes) * 100.0
                : 0.0;
            std::cout << "  [" << (pctVsDefault >= 0 ? "+" : "") << std::setprecision(2)
                      << pctVsDefault << "% vs OpenZL default level]";
        }
        std::cout << "\n";
    };
    printFlatOpenZlLevel("OpenZL flat (lowest) :  ", openZlLowestBytes);
    printFlatOpenZlLevel("OpenZL flat (highest):  ", openZlHighestBytes);

    // Double-compression: each plan's own already-encoded output, further
    // compressed by OpenZL -- shows how much (if anything) a generic
    // compressor can still squeeze out of what SubIntSplit already produced,
    // and how that compares against just running OpenZL directly on the raw
    // (undecomposed) data (the flat baseline above).
    std::cout << "\n";
    auto printDoubleCompression = [&](const char* label, size_t planBytes, std::optional<size_t> doubleBytes) {
        if (!doubleBytes.has_value()) {
            std::cout << "  " << label << " -> OpenZL: (unavailable)\n";
            return;
        }
        const size_t db = *doubleBytes;
        const double pctVsPlanAlone = (planBytes > 0)
            ? (static_cast<double>(planBytes) - static_cast<double>(db))
              / static_cast<double>(planBytes) * 100.0
            : 0.0;
        std::cout << "  " << label << " -> OpenZL: " << std::setw(12) << db
                  << " bytes  (" << std::setprecision(3) << bpe(db) << " bits/elem)"
                  << "  [" << (pctVsPlanAlone >= 0 ? "+" : "") << std::setprecision(2)
                  << pctVsPlanAlone << "% vs " << label << " alone";
        if (openZlBytes.has_value()) {
            const double pctVsFlatOpenZl = (*openZlBytes > 0)
                ? (static_cast<double>(db) - static_cast<double>(*openZlBytes))
                  / static_cast<double>(*openZlBytes) * 100.0
                : 0.0;
            std::cout << ", " << (pctVsFlatOpenZl >= 0 ? "+" : "") << pctVsFlatOpenZl
                      << "% vs flat OpenZL";
        }
        std::cout << "]\n";
    };
    printDoubleCompression("AutoSIS         ", autoSisBytes, autoSisThenOpenZlBytes);
    printDoubleCompression("Oracle (random) ", oracleBytes, oracleThenOpenZlBytes);
    if (consecOracleBytes.has_value())
        printDoubleCompression("Oracle (consec) ", *consecOracleBytes, consecOracleThenOpenZlBytes);
    if (mergedOracleBytes.has_value())
        printDoubleCompression("Oracle (merged) ", *mergedOracleBytes, mergedOracleThenOpenZlBytes);

    std::cout << "\n";
    std::cout << "  Plans identical:            " << (matchCount == autoSisPlan.size() &&
                                                       autoSisPlan.size() == oraclePlan.size() ? "Yes" : "No") << "\n";
    std::cout << "  Segments matching:          " << matchCount << " / "
              << std::max(autoSisPlan.size(), oraclePlan.size()) << "\n";
    std::cout << "  AutoSIS overhead vs oracle: "
              << std::setprecision(2) << (overheadVsOracle >= 0 ? "+" : "") << overheadVsOracle << "%\n";
    std::cout << "  Compression efficiency:     " << efficiency << "% of oracle\n";
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    const ExploreConfig cfg;

    // Always capture BlockFORFPE tier stats for the operator-graph JSON export
    // (see BLOCKFORFPE_STATS in BlockFORFPEEncoder.hpp). Must be set before any
    // BlockFORFPEEncoder::encode() call — the flag is cached in a static bool on
    // first use — so set it here, before the AutoSIS/grid/oracle sections below.
    setenv("BLOCKFORFPE_STATS", "1", 1);

    // ── 1. Header ─────────────────────────────────────────────────────────────
    std::cout << "=== Oracle vs AutoSubIntSplit vs OpenZL ===\n";
    std::cout << "Dataset:        " << cfg.datasetLabel
              << " (" << cfg.datasetSize << " elements)\n";
    std::cout << "Encoding types: ";
    for (size_t i = 0; i < cfg.encodingTypes.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << encodingTypeToString(cfg.encodingTypes[i]);
    }
    std::cout << "\nSample:         " << cfg.maxSamples
              << " elements (blockSize=" << cfg.blockSize << ")\n";
    std::cout << "Reorderers:     " << (cfg.allowReorderers ? "enabled (BWT)" : "disabled") << "\n";

    // ── 2. Load dataset ───────────────────────────────────────────────────────
    std::cout << "\nLoading dataset..." << std::flush;
    generators::ParquetColumnGenerator<int64_t> gen(cfg.datasetPath, cfg.columnName);
    std::vector<int64_t> fullData = gen.generate(cfg.datasetSize);
    std::cout << " " << fullData.size() << " elements loaded.\n";

    // ── 3. Sample ─────────────────────────────────────────────────────────────
    StreamSampler<int64_t>::Config samplerCfg;
    samplerCfg.maxSamples = cfg.maxSamples;
    samplerCfg.blockSize  = cfg.blockSize;
    samplerCfg.stride     = 0;
    auto sample = StreamSampler<int64_t>::sample(fullData, samplerCfg);
    std::cout << "Sample:         " << sample.size() << " elements drawn.\n";

    // ── 4. AutoSIS plan ───────────────────────────────────────────────────────
    std::cout << "\nRunning AutoSIS selection..." << std::flush;
    auto autoSisResult = runAutoSISSelection(sample, cfg, fullData.size());
    std::cout << " done.\n";
    printPlan(autoSisResult.segments, "AutoSIS Plan (cost-model based)");

    std::cout << "Encoding full dataset with AutoSIS plan..." << std::flush;
    auto autoSisEnc = makeSubIntSplitEncoderFromSegments<int64_t>(
        autoSisResult.segments, cfg.order);
    auto autoSisEncoded = autoSisEnc->encode(fullData);
    const size_t autoSisBytes = autoSisEncoded.data().size();
    std::cout << " " << autoSisBytes << " bytes.\n";

    std::cout << "Feeding AutoSIS output into OpenZL..." << std::flush;
    const auto autoSisThenOpenZlResult = tryOpenZLCompressFull(
        std::span<const uint8_t>(autoSisEncoded.data().data(), autoSisEncoded.data().size()));
    const std::optional<size_t> autoSisThenOpenZlBytes =
        autoSisThenOpenZlResult.has_value() ? std::optional<size_t>(autoSisThenOpenZlResult->buf.size()) : std::nullopt;
    if (autoSisThenOpenZlBytes.has_value()) std::cout << " " << *autoSisThenOpenZlBytes << " bytes.\n";
    else std::cout << " skipped (OpenZL unavailable or failed).\n";

    // ── 5. Full encoding grid ─────────────────────────────────────────────────
    std::cout << "\nComputing full encoding grid ("
              << 64 * 64 / 2 << " bit-ranges × "
              << cfg.encodingTypes.size() << " encodings)..." << std::flush;
    auto encGrid = computeEncodingGrid(sample, cfg.encodingTypes, cfg.minSegmentWidth);
    std::cout << " done.\n";

    // ── 5b. Consecutive-sample encoding grid (blockSize=512) ─────────────────
    // The random-sample oracle blind-spots locality-sensitive codecs (FORSection,
    // AdaptiveFOR) because random samples scatter elements across the full timeline.
    // A consecutive-block sample preserves within-block temporal locality: each
    // 512-element block spans ~5s of tweets, so FORSection<512> frames see only
    // ~13-bit residuals instead of the 39-bit residuals on random samples.
    constexpr size_t kConsecBlockSize = 512;
    StreamSampler<int64_t>::Config consecCfg;
    consecCfg.maxSamples = cfg.maxSamples;
    consecCfg.blockSize  = kConsecBlockSize;
    consecCfg.stride     = 0;
    auto consecSample = StreamSampler<int64_t>::sample(fullData, consecCfg);
    std::cout << "Consecutive sample: " << consecSample.size()
              << " elements (blockSize=" << kConsecBlockSize << ").\n";

    std::cout << "Computing consecutive-sample encoding grid..." << std::flush;
    auto consecGrid = computeEncodingGrid(consecSample, cfg.encodingTypes, cfg.minSegmentWidth);
    std::cout << " done.\n";

    // ── 5c. Wide-contiguous-sample encoding grid, merged grid ────────────────
    // RunLengthEncoding (and any future delta-style codec) needs to see real,
    // long runs across consecutive rows -- invisible to both the random sample
    // (16-element blocks) and the consecutive sample (512-element blocks,
    // tuned for FrameOfReference's frame size, not for run visibility). A
    // wider contiguous window gives it a fairer trial. Only computed for the
    // WideContiguous-classified subset of cfg.encodingTypes: this grid exists
    // solely to feed the merge below, not to stand alone as its own
    // diagnostic, so there's no reason to pay for encoding every type against it.
    constexpr size_t kWideBlockSize = 2048;
    StreamSampler<int64_t>::Config wideCfg;
    wideCfg.maxSamples = cfg.maxSamples;
    wideCfg.blockSize  = kWideBlockSize;
    wideCfg.stride     = 0;
    auto wideSample = StreamSampler<int64_t>::sample(fullData, wideCfg);
    std::cout << "Wide-contiguous sample: " << wideSample.size()
              << " elements (blockSize=" << kWideBlockSize << ").\n";

    std::vector<EncodingType> wideProfileTypes;
    for (auto e : cfg.encodingTypes)
        if (preferredSamplingProfile(e) == SamplingProfile::WideContiguous)
            wideProfileTypes.push_back(e);

    std::cout << "Computing wide-contiguous-sample encoding grid..." << std::flush;
    auto wideGrid = computeEncodingGrid(wideSample, wideProfileTypes, cfg.minSegmentWidth);
    std::cout << " done.\n";

    std::cout << "Merging per-profile encoding grids..." << std::flush;
    auto mergedGrid = mergeEncodingGridsByProfile(
        { {SamplingProfile::Random, &encGrid},
          {SamplingProfile::Consecutive, &consecGrid},
          {SamplingProfile::WideContiguous, &wideGrid} },
        { {SamplingProfile::Random, sample.size()},
          {SamplingProfile::Consecutive, consecSample.size()},
          {SamplingProfile::WideContiguous, wideSample.size()} },
        /*referenceSampleSize=*/sample.size(),
        cfg.encodingTypes, cfg.minSegmentWidth);
    std::cout << " done.\n";

    // ── 6. Oracle DP ──────────────────────────────────────────────────────────
    std::cout << "Running oracle DP (random sample)..." << std::flush;
    auto oraclePlan = runOracleDP(encGrid, cfg.minSegmentWidth);
    std::cout << " done.\n";
    printPlan(oraclePlan, "Oracle Plan (random sample)", &encGrid);

    std::cout << "Encoding full dataset with oracle plan..." << std::flush;
    auto oracleEnc = makeSubIntSplitEncoderFromSegments<int64_t>(oraclePlan, cfg.order);
    auto oracleEncoded = oracleEnc->encode(fullData);
    const size_t oracleBytes = oracleEncoded.data().size();
    std::cout << " " << oracleBytes << " bytes.\n";

    std::cout << "Feeding oracle (random) output into OpenZL..." << std::flush;
    const auto oracleThenOpenZlResult = tryOpenZLCompressFull(
        std::span<const uint8_t>(oracleEncoded.data().data(), oracleEncoded.data().size()));
    const std::optional<size_t> oracleThenOpenZlBytes =
        oracleThenOpenZlResult.has_value() ? std::optional<size_t>(oracleThenOpenZlResult->buf.size()) : std::nullopt;
    if (oracleThenOpenZlBytes.has_value()) std::cout << " " << *oracleThenOpenZlBytes << " bytes.\n";
    else std::cout << " skipped (OpenZL unavailable or failed).\n";

    // ── 6b. Consecutive-sample oracle ─────────────────────────────────────────
    std::cout << "Running oracle DP (consecutive sample)..." << std::flush;
    auto consecOraclePlan = runOracleDP(consecGrid, cfg.minSegmentWidth);
    std::cout << " done.\n";
    printPlan(consecOraclePlan, "Oracle Plan (consecutive sample)", &consecGrid);

    std::cout << "Encoding full dataset with consecutive-sample oracle plan..." << std::flush;
    auto consecOracleEnc = makeSubIntSplitEncoderFromSegments<int64_t>(consecOraclePlan, cfg.order);
    auto consecOracleEncoded = consecOracleEnc->encode(fullData);
    const size_t consecOracleBytes = consecOracleEncoded.data().size();
    std::cout << " " << consecOracleBytes << " bytes.\n";

    std::cout << "Feeding oracle (consecutive-sample) output into OpenZL..." << std::flush;
    const auto consecOracleThenOpenZlResult = tryOpenZLCompressFull(
        std::span<const uint8_t>(consecOracleEncoded.data().data(), consecOracleEncoded.data().size()));
    const std::optional<size_t> consecOracleThenOpenZlBytes =
        consecOracleThenOpenZlResult.has_value() ? std::optional<size_t>(consecOracleThenOpenZlResult->buf.size()) : std::nullopt;
    if (consecOracleThenOpenZlBytes.has_value()) std::cout << " " << *consecOracleThenOpenZlBytes << " bytes.\n";
    else std::cout << " skipped (OpenZL unavailable or failed).\n";

    // ── 6c. Merged-sampling oracle ────────────────────────────────────────────
    std::cout << "Running oracle DP (merged sampling)..." << std::flush;
    auto mergedPlan = runOracleDP(mergedGrid, cfg.minSegmentWidth);
    std::cout << " done.\n";
    printPlan(mergedPlan, "Oracle Plan (merged sampling)", &mergedGrid);

    std::cout << "Encoding full dataset with merged-sampling oracle plan..." << std::flush;
    auto mergedEnc = makeSubIntSplitEncoderFromSegments<int64_t>(mergedPlan, cfg.order);
    auto mergedEncoded = mergedEnc->encode(fullData);
    const size_t mergedBytes = mergedEncoded.data().size();
    std::cout << " " << mergedBytes << " bytes.\n";

    std::cout << "Feeding oracle (merged-sampling) output into OpenZL..." << std::flush;
    const auto mergedOracleThenOpenZlResult = tryOpenZLCompressFull(
        std::span<const uint8_t>(mergedEncoded.data().data(), mergedEncoded.data().size()));
    const std::optional<size_t> mergedOracleThenOpenZlBytes =
        mergedOracleThenOpenZlResult.has_value() ? std::optional<size_t>(mergedOracleThenOpenZlResult->buf.size()) : std::nullopt;
    if (mergedOracleThenOpenZlBytes.has_value()) std::cout << " " << *mergedOracleThenOpenZlBytes << " bytes.\n";
    else std::cout << " skipped (OpenZL unavailable or failed).\n";

    // ── 7. OpenZL flat baseline with introspection ────────────────────────────
    // Swept across compressionLevel's lowest/default/highest to see whether
    // OpenZL's own tuning knob moves the flat, whole-buffer number at all --
    // the default-level pass remains the one fed into every downstream
    // diagnostic/JSON export below (unchanged plumbing); lowest/highest are
    // reported alongside it purely for comparison.
    std::cout << "\nRunning OpenZL flat baseline with introspection (compressionLevel="
              << kOpenZLLevelDefault << ")..." << std::flush;
    std::optional<size_t> openZlBytes;
    OpenZLEncodeStats openZlStats;
    std::vector<uint8_t> openZlBuf;  // kept alive for the ZL_ReflectionCtx-based JSON export below
    try {
        openZlBuf = encodeOpenZLWithStats(
            std::span<const int64_t>(fullData), openZlStats, kOpenZLLevelDefault);
        openZlBytes = openZlBuf.size();
        std::cout << " " << *openZlBytes << " bytes.\n";
    } catch (const std::exception& ex) {
        std::cout << " skipped (" << ex.what() << ").\n";
    }

    std::optional<size_t> openZlLowestBytes, openZlHighestBytes;
    std::cout << "Running OpenZL flat baseline (compressionLevel=" << kOpenZLLevelLowest
              << ", lowest)..." << std::flush;
    try {
        OpenZLEncodeStats lowestStats;
        auto buf = encodeOpenZLWithStats(
            std::span<const int64_t>(fullData), lowestStats, kOpenZLLevelLowest);
        openZlLowestBytes = buf.size();
        std::cout << " " << *openZlLowestBytes << " bytes.\n";
    } catch (const std::exception& ex) {
        std::cout << " skipped (" << ex.what() << ").\n";
    }

    std::cout << "Running OpenZL flat baseline (compressionLevel=" << kOpenZLLevelHighest
              << ", highest)..." << std::flush;
    try {
        OpenZLEncodeStats highestStats;
        auto buf = encodeOpenZLWithStats(
            std::span<const int64_t>(fullData), highestStats, kOpenZLLevelHighest);
        openZlHighestBytes = buf.size();
        std::cout << " " << *openZlHighestBytes << " bytes.\n";
    } catch (const std::exception& ex) {
        std::cout << " skipped (" << ex.what() << ").\n";
    }

    // ── 8. Quality metrics ────────────────────────────────────────────────────
    printQualityMetrics(autoSisBytes, oracleBytes, consecOracleBytes, mergedBytes, openZlBytes,
                        fullData.size(), autoSisResult.segments, oraclePlan,
                        autoSisThenOpenZlBytes, oracleThenOpenZlBytes,
                        consecOracleThenOpenZlBytes, mergedOracleThenOpenZlBytes,
                        openZlLowestBytes, openZlHighestBytes);

    // ── 9. Per-segment encoding ranking ───────────────────────────────────────
    printSegmentEncodingRanking(oraclePlan, autoSisResult.segments, encGrid);

    // ── 9b. Per-segment encoding ranking (consecutive sample) ─────────────────
    std::cout << "\n=== Per-Segment Encoding Ranking (consecutive sample, blockSize="
              << kConsecBlockSize << ") ===\n";
    printSegmentEncodingRanking(consecOraclePlan, autoSisResult.segments, consecGrid);

    // ── 9c. Per-segment encoding ranking (merged sampling) ───────────────────
    std::cout << "\n=== Per-Segment Encoding Ranking (merged sampling) ===\n";
    printSegmentEncodingRanking(mergedPlan, autoSisResult.segments, mergedGrid);

    // ── 10. Cost model accuracy ───────────────────────────────────────────────
    printCostModelAccuracy(oraclePlan, sample, cfg.encodingTypes, encGrid);

    // ── 11. Gap attribution ───────────────────────────────────────────────────
    printGapAttribution(oraclePlan, autoSisResult.segments,
                        cfg.encodingTypes, sample, encGrid, fullData.size());

    // ── 12. OpenZL internal strategy analysis ─────────────────────────────────
    if (openZlBytes.has_value())
        printOpenZLAnalysis(openZlStats, *openZlBytes, fullData.size());

    // ── 13. Operator-graph JSON export (for plot_operator_graph.py) ──────────
    // Purely additive: failures here never affect the diagnostics printed above.
    try {
        auto toU64 = [](const std::vector<int64_t>& v) {
            std::vector<uint64_t> u(v.size());
            for (size_t i = 0; i < v.size(); ++i) u[i] = static_cast<uint64_t>(v[i]);
            return u;
        };
        const auto uSample       = toU64(sample);
        const auto uConsecSample = toU64(consecSample);
        const auto uFullData     = toU64(fullData);

        opgraph::OperatorGraphExport exportData;

        exportData.dataset.label           = cfg.datasetLabel;
        exportData.dataset.path            = cfg.datasetPath.string();
        exportData.dataset.column          = cfg.columnName;
        exportData.dataset.datasetSize     = fullData.size();
        exportData.dataset.sampleSize      = sample.size();
        exportData.dataset.blockSize       = cfg.blockSize;
        exportData.dataset.consecBlockSize = kConsecBlockSize;
        exportData.dataset.allowReorderers = cfg.allowReorderers;
        for (auto enc : cfg.encodingTypes)
            exportData.dataset.encodingTypes.push_back(encodingTypeToString(enc));

        auto bpe = [&](size_t bytes) {
            return static_cast<double>(bytes) * 8.0 / static_cast<double>(fullData.size());
        };
        const double overheadVsOraclePct = (oracleBytes > 0)
            ? (static_cast<double>(autoSisBytes) - static_cast<double>(oracleBytes))
              / static_cast<double>(oracleBytes) * 100.0
            : 0.0;
        const double efficiencyPct = (autoSisBytes > 0)
            ? static_cast<double>(oracleBytes) / static_cast<double>(autoSisBytes) * 100.0
            : 100.0;
        size_t matchCount = 0;
        {
            std::map<std::pair<int,int>, EncodingType> oracleMap;
            for (const auto& seg : oraclePlan) oracleMap[{seg.bitStart, seg.bitEnd}] = seg.encoding;
            for (const auto& aSeg : autoSisResult.segments) {
                auto it = oracleMap.find({aSeg.bitStart, aSeg.bitEnd});
                if (it != oracleMap.end() && it->second == aSeg.encoding) ++matchCount;
            }
        }

        exportData.summary.autoSisBytes      = autoSisBytes;
        exportData.summary.oracleRandomBytes = oracleBytes;
        exportData.summary.oracleConsecBytes = consecOracleBytes;
        exportData.summary.oracleMergedBytes = mergedBytes;
        if (openZlBytes.has_value()) exportData.summary.openZlBytes = *openZlBytes;
        exportData.summary.autoSisBpe        = bpe(autoSisBytes);
        exportData.summary.oracleRandomBpe   = bpe(oracleBytes);
        exportData.summary.oracleConsecBpe   = bpe(consecOracleBytes);
        exportData.summary.oracleMergedBpe   = bpe(mergedBytes);
        if (openZlBytes.has_value()) exportData.summary.openZlBpe = bpe(*openZlBytes);
        if (openZlLowestBytes.has_value()) {
            exportData.summary.openZlLowestBytes = *openZlLowestBytes;
            exportData.summary.openZlLowestBpe   = bpe(*openZlLowestBytes);
        }
        if (openZlHighestBytes.has_value()) {
            exportData.summary.openZlHighestBytes = *openZlHighestBytes;
            exportData.summary.openZlHighestBpe   = bpe(*openZlHighestBytes);
        }
        if (autoSisThenOpenZlBytes.has_value()) {
            exportData.summary.autoSisThenOpenZlBytes = *autoSisThenOpenZlBytes;
            exportData.summary.autoSisThenOpenZlBpe   = bpe(*autoSisThenOpenZlBytes);
        }
        if (oracleThenOpenZlBytes.has_value()) {
            exportData.summary.oracleRandomThenOpenZlBytes = *oracleThenOpenZlBytes;
            exportData.summary.oracleRandomThenOpenZlBpe   = bpe(*oracleThenOpenZlBytes);
        }
        if (consecOracleThenOpenZlBytes.has_value()) {
            exportData.summary.oracleConsecThenOpenZlBytes = *consecOracleThenOpenZlBytes;
            exportData.summary.oracleConsecThenOpenZlBpe   = bpe(*consecOracleThenOpenZlBytes);
        }
        if (mergedOracleThenOpenZlBytes.has_value()) {
            exportData.summary.oracleMergedThenOpenZlBytes = *mergedOracleThenOpenZlBytes;
            exportData.summary.oracleMergedThenOpenZlBpe   = bpe(*mergedOracleThenOpenZlBytes);
        }
        exportData.summary.overheadVsOraclePct = overheadVsOraclePct;
        exportData.summary.efficiencyPct       = efficiencyPct;
        exportData.summary.segmentsMatching    = matchCount;
        exportData.summary.segmentsTotal       =
            std::max(autoSisResult.segments.size(), oraclePlan.size());

        exportData.autoSis.totalCost = autoSisResult.total_cost;
        for (const auto& seg : autoSisResult.segments)
            exportData.autoSis.segments.push_back(
                buildSegmentJson(seg, encGrid, uSample, /*isAutoSis=*/true));

        for (const auto& seg : oraclePlan)
            exportData.oracleRandom.segments.push_back(
                buildSegmentJson(seg, encGrid, uSample, /*isAutoSis=*/false));

        for (const auto& seg : consecOraclePlan)
            exportData.oracleConsecutive.segments.push_back(
                buildSegmentJson(seg, consecGrid, uConsecSample, /*isAutoSis=*/false));

        for (const auto& seg : mergedPlan)
            exportData.oracleMerged.segments.push_back(
                buildSegmentJson(seg, mergedGrid, uSample, /*isAutoSis=*/false));

#ifdef HAVE_OPENZL
        if (openZlBytes.has_value()) {
            exportData.openZl = buildOpenZLGraphJson(openZlBuf, openZlStats);
        }

        // Double-compression graphs: the codec DAG OpenZL chose when
        // compressing each plan's OWN already-encoded output, built from the
        // SAME compression pass already run above (no redundant re-compress).
        if (autoSisThenOpenZlResult.has_value()) {
            exportData.autoSisThenOpenZl = buildOpenZLGraphJson(
                autoSisThenOpenZlResult->buf, autoSisThenOpenZlResult->stats);
        }
        if (oracleThenOpenZlResult.has_value()) {
            exportData.oracleRandomThenOpenZl = buildOpenZLGraphJson(
                oracleThenOpenZlResult->buf, oracleThenOpenZlResult->stats);
        }
        if (consecOracleThenOpenZlResult.has_value()) {
            exportData.oracleConsecThenOpenZl = buildOpenZLGraphJson(
                consecOracleThenOpenZlResult->buf, consecOracleThenOpenZlResult->stats);
        }
        if (mergedOracleThenOpenZlResult.has_value()) {
            exportData.oracleMergedThenOpenZl = buildOpenZLGraphJson(
                mergedOracleThenOpenZlResult->buf, mergedOracleThenOpenZlResult->stats);
        }

        // Apply OpenZL directly to each OracleSIS segment's own bit-range data
        // over the FULL dataset (both plans), so each split's chosen
        // SubIntSplit encoding can be compared against what OpenZL alone
        // achieves on that same bit range at full scale. This is the most
        // expensive part of the JSON export (one full-dataset OpenZL compress
        // per segment).
        std::cout << "Running per-segment OpenZL on OracleSIS splits (full dataset, this may take a while)..."
                   << std::flush;
        attachOpenZLToSegments(exportData.oracleRandom.segments, uFullData);
        attachOpenZLToSegments(exportData.oracleConsecutive.segments, uFullData);
        attachOpenZLToSegments(exportData.oracleMerged.segments, uFullData);
        std::cout << " done.\n";

        // Double-compression: OpenZL applied on top of each segment's already
        // SubIntSplit-encoded bytes, both over the full dataset.
        std::cout << "Running OpenZL on already-compressed OracleSIS bytes (full dataset)..." << std::flush;
        attachOpenZLOnOracleBytes(exportData.oracleRandom.segments, oraclePlan, uFullData);
        attachOpenZLOnOracleBytes(exportData.oracleConsecutive.segments, consecOraclePlan, uFullData);
        attachOpenZLOnOracleBytes(exportData.oracleMerged.segments, mergedPlan, uFullData);
        std::cout << " done.\n";
#endif

        exportData.save(cfg.jsonOutputPath);
    } catch (const std::exception& ex) {
        std::cerr << "Operator graph JSON export failed (non-fatal): " << ex.what() << "\n";
    }

    return 0;
}
