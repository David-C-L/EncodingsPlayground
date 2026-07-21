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

// ---------------------------------------------------------------------------
// Section codec factory
// ---------------------------------------------------------------------------

static std::shared_ptr<ISectionCodecIntegral<uint64_t>>
makeSectionCodec(EncodingType enc, uint8_t width) {
    using namespace detail_trisplit;
    switch (enc) {
        case EncodingType::RawEncoding:                     return makeRawSection<uint64_t>(width);
        case EncodingType::BitPacking:                      return makeRawBitPackedSection<uint64_t>(width);
        case EncodingType::RunLengthEncoding:               return makeRLESection<uint64_t>(width);
        case EncodingType::AdaptiveFrameOfReference:        return makeAdaptiveFORSection<uint64_t>(width);
        case EncodingType::DictionaryEncoding:              return makeDictionarySection<uint64_t>(width);
        case EncodingType::AdaptiveDictionaryEncoding:      return makeAdaptiveDictionarySection<uint64_t>(width);
        case EncodingType::BlockFrequencyPartitionEncoding: return makeBlockFrequencyPartitionSection<uint64_t>(width);
        case EncodingType::BlockFrequencyPartitionFOREncoding: return makeBlockFrequencyPartitionFORSection<uint64_t>(width);
        case EncodingType::FrequencyPartitionEncoding:      return makeFrequencyPartitionSection<uint64_t>(width);
        case EncodingType::MainlyConstantEncoding:          return makeMainlyConstantSection<uint64_t>(width);
        case EncodingType::FrameOfReference:                return makeFORSection<512, uint64_t>(width);
        case EncodingType::AdaptiveFramedBitPrefix:         return makeAdaptiveFramedBitPrefixSection<uint64_t>(width);
        case EncodingType::HuffmanEncoding:                 return makeHuffmanSection<uint64_t>(width);
        case EncodingType::LZ4:                             return makeLZ4Section<uint64_t>(width);
        case EncodingType::FSEEncoding:                     return makeFSESection<uint64_t>(width);
        case EncodingType::BlockFSEEncoding:                        return makeBlockFSESection<uint64_t>(width);
        case EncodingType::OpenZL:                          return makeOpenZLSection<0, uint64_t>(width);
        case EncodingType::BlockFORFPEEncoding:             return makeBlockFORFPESection<uint64_t>(width);
        case EncodingType::RangePackFrequencyPartitionEncoding:
            return makeRangePackFrequencyPartitionSection<uint64_t>(width);
        case EncodingType::RangePackBlockFrequencyPartitionEncoding:
            return makeRangePackBlockFrequencyPartitionSection<uint64_t>(width);
        case EncodingType::CascadingFrameOfReference:
            return makeCascadingFORSection<uint64_t>(width);
        case EncodingType::CascadingFORBlockFrequencyPartitionEncoding:
            return makeCascadingFORBlockFrequencyPartitionSection<uint64_t>(width);
        case EncodingType::RunLengthCascadingFOREncoding:
            return makeRLECascadingFORSection<uint64_t>(width);
        case EncodingType::CascadingFORFSEEncoding:
            return makeCascadingFORFSESection<uint64_t>(width);
        case EncodingType::CascadingFORBlockFSEEncoding:
            return makeCascadingFORBlockFSESection<uint64_t>(width);
        case EncodingType::CascadingFORHuffmanEncoding:
            return makeCascadingFORHuffmanSection<uint64_t>(width);
        case EncodingType::CascadingFORPrevFSEEncoding:
            return makeCascadingFORPrevFSESection<uint64_t>(width);
        case EncodingType::CascadingFORPrevBlockFSEEncoding:
            return makeCascadingFORPrevBlockFSESection<uint64_t>(width);
        case EncodingType::CascadingFORPrevHuffmanEncoding:
            return makeCascadingFORPrevHuffmanSection<uint64_t>(width);
        case EncodingType::CascadingFORPrevFrequencyPartitionEncoding:
            return makeCascadingFORPrevFrequencyPartitionSection<uint64_t>(width);
        case EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding:
            return makeCascadingFORPrevBlockFrequencyPartitionSection<uint64_t>(width);
        default:
            throw std::invalid_argument(std::string("makeSectionCodec: unsupported type: ") +
                                        encodingTypeToString(enc));
    }
}

// ---------------------------------------------------------------------------
// Full encoding grid — stores actual sample bytes for EVERY (range, encoding).
// ---------------------------------------------------------------------------

struct EncodingCell {
    // All (encoding, bytes) pairs tried, sorted ascending by bytes after compute.
    std::vector<std::pair<EncodingType, size_t>> allResults;
    size_t       bestBytes    = std::numeric_limits<size_t>::max();
    EncodingType bestEncoding = EncodingType::RawEncoding;

    size_t bytesFor(EncodingType enc) const {
        for (const auto& [e, b] : allResults)
            if (e == enc) return b;
        return std::numeric_limits<size_t>::max();
    }
};

// grid[bitStart][bitEnd]
using EncodingGrid = std::vector<std::vector<EncodingCell>>;

static EncodingGrid computeEncodingGrid(
    const std::vector<int64_t>& sample,
    const std::vector<EncodingType>& encodingTypes,
    int minSegmentWidth)
{
    constexpr int kBits = 64;
    EncodingGrid grid(kBits, std::vector<EncodingCell>(kBits));

    const size_t n = sample.size();
    std::vector<uint64_t> uSample(n);
    for (size_t i = 0; i < n; ++i)
        uSample[i] = static_cast<uint64_t>(sample[i]);

    std::vector<uint64_t> sectionValues(n);

    for (int l = 0; l < kBits; ++l) {
        for (int r = l + minSegmentWidth - 1; r < kBits; ++r) {
            const int width = r - l + 1;
            const uint64_t mask = (width == 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);

            for (size_t i = 0; i < n; ++i)
                sectionValues[i] = (uSample[i] >> l) & mask;

            std::span<const uint64_t> sectionSpan(sectionValues.data(), n);

            EncodingCell& cell = grid[l][r];
            for (EncodingType enc : encodingTypes) {
                try {
                    auto codec = makeSectionCodec(enc, static_cast<uint8_t>(width));
                    auto encoded = codec->encode(sectionSpan);
                    const size_t bytes = encoded.data().size();
                    cell.allResults.emplace_back(enc, bytes);
                    if (bytes < cell.bestBytes) {
                        cell.bestBytes    = bytes;
                        cell.bestEncoding = enc;
                    }
                } catch (...) {
                    // Some codecs legitimately fail on degenerate inputs; skip.
                }
            }

            // Sort ascending by bytes so ranking tables are cheap to produce.
            std::sort(cell.allResults.begin(), cell.allResults.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
        }
    }
    return grid;
}

// ---------------------------------------------------------------------------
// Oracle DP — partition bits [0..63] to minimise total sample bytes.
// ---------------------------------------------------------------------------

static std::vector<SegmentPlan> runOracleDP(
    const EncodingGrid& grid,
    int minSegmentWidth)
{
    constexpr int kBits = 64;
    constexpr size_t kInf = std::numeric_limits<size_t>::max() / 2;

    std::vector<size_t> dp(kBits + 1, kInf);
    std::vector<int>    parent(kBits + 1, -1);
    dp[0] = 0;

    for (int i = minSegmentWidth; i <= kBits; ++i) {
        for (int l = 0; l <= i - minSegmentWidth; ++l) {
            const size_t cost = grid[l][i - 1].bestBytes;
            if (dp[l] == kInf || cost == std::numeric_limits<size_t>::max()) continue;
            if (dp[l] + cost < dp[i]) {
                dp[i]     = dp[l] + cost;
                parent[i] = l;
            }
        }
    }

    std::vector<SegmentPlan> plan;
    int cur = kBits;
    while (cur > 0) {
        const int l = parent[cur];
        if (l < 0)
            throw std::runtime_error("runOracleDP: no valid partition found");
        const int r = cur - 1;
        SegmentPlan seg;
        seg.bitStart = l;
        seg.bitEnd   = r;
        seg.encoding = grid[l][r].bestEncoding;
        seg.cost     = static_cast<double>(grid[l][r].bestBytes);
        plan.push_back(seg);
        cur = l;
    }
    std::reverse(plan.begin(), plan.end());
    return plan;
}

// ---------------------------------------------------------------------------
// Merge per-sampling-profile grids into one grid — each EncodingType's cost
// always comes from its own designated profile's grid (see
// EncodingSamplingProfile.hpp), so a random-sample-only comparison never
// gets over/under-estimated by a naive "whichever sample looks cheaper"
// merge. runOracleDP is fully agnostic to how a grid was assembled (it only
// ever reads bestBytes/bestEncoding per cell), so it is reused unmodified on
// the result of this merge.
//
// Each source grid was built from a different-sized sample (StreamSampler's
// block sampling truncates to a multiple of blockSize, so e.g. a 10000-target
// sample becomes 9728 elements at blockSize=512, or 8192 at blockSize=2048).
// runOracleDP sums raw byte counts across segments assuming every cell is on
// the same scale -- true within a single grid (always the same N), but NOT
// true once cells from grids of different N are mixed into one merged grid.
// Left unscaled, an encoding costed against a smaller sample looks spuriously
// cheaper purely from having fewer elements to encode, independent of its
// true per-element compression quality -- so every profile's byte counts are
// rescaled here to `referenceSampleSize`'s scale before comparison.
// ---------------------------------------------------------------------------

static EncodingGrid mergeEncodingGridsByProfile(
    const std::map<SamplingProfile, const EncodingGrid*>& gridsByProfile,
    const std::map<SamplingProfile, size_t>& sampleSizeByProfile,
    size_t referenceSampleSize,
    const std::vector<EncodingType>& encodingTypes,
    int minSegmentWidth)
{
    constexpr int kBits = 64;
    EncodingGrid merged(kBits, std::vector<EncodingCell>(kBits));

    for (int l = 0; l < kBits; ++l) {
        for (int r = l + minSegmentWidth - 1; r < kBits; ++r) {
            EncodingCell& cell = merged[l][r];
            for (EncodingType enc : encodingTypes) {
                const auto profile = preferredSamplingProfile(enc);
                auto it = gridsByProfile.find(profile);
                if (it == gridsByProfile.end()) continue;
                const size_t rawBytes = (*it->second)[l][r].bytesFor(enc);
                if (rawBytes == std::numeric_limits<size_t>::max()) continue;

                auto sizeIt = sampleSizeByProfile.find(profile);
                const size_t thisN = (sizeIt != sampleSizeByProfile.end() && sizeIt->second > 0)
                    ? sizeIt->second : referenceSampleSize;
                const double scale = static_cast<double>(referenceSampleSize) / static_cast<double>(thisN);
                const size_t bytes = static_cast<size_t>(std::llround(static_cast<double>(rawBytes) * scale));

                cell.allResults.emplace_back(enc, bytes);
                if (bytes < cell.bestBytes) {
                    cell.bestBytes    = bytes;
                    cell.bestEncoding = enc;
                }
            }
            std::sort(cell.allResults.begin(), cell.allResults.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
        }
    }
    return merged;
}

// ---------------------------------------------------------------------------
// AutoSIS selection (mirrors makeDefaultAutoSubIntSplitConfig CostModelSet path)
// ---------------------------------------------------------------------------

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
static std::optional<opgraph::BlockFpeStatsJson> buildBlockFpeStats(
    EncodingType enc, int bitStart, int width, const std::vector<uint64_t>& uSample)
{
    if (enc != EncodingType::BlockFORFPEEncoding) return std::nullopt;

    const uint64_t mask = (width == 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    std::vector<uint64_t> section(uSample.size());
    for (size_t i = 0; i < uSample.size(); ++i)
        section[i] = (uSample[i] >> bitStart) & mask;

    try {
        auto codec = makeSectionCodec(enc, static_cast<uint8_t>(width));
        std::span<const uint64_t> sectionSpan(section.data(), section.size());
        auto encoded = codec->encode(sectionSpan);
        const auto& meta = encoded.metadata();
        if (!meta.customMetadata.count("avg_num_tiers")) return std::nullopt;

        opgraph::BlockFpeStatsJson stats;
        stats.avgNumTiers         = std::stod(meta.customMetadata.at("avg_num_tiers"));
        stats.avgTagBitWidth      = std::stod(meta.customMetadata.at("avg_tag_bit_width"));
        stats.avgFallbackFraction = std::stod(meta.customMetadata.at("avg_fallback_fraction"));
        stats.blockSize = meta.customMetadata.count("block_size")
            ? static_cast<uint32_t>(std::stoul(meta.customMetadata.at("block_size"))) : 0u;
        stats.numBlocks = meta.customMetadata.count("num_blocks")
            ? static_cast<uint32_t>(std::stoul(meta.customMetadata.at("num_blocks"))) : 0u;
        return stats;
    } catch (...) {
        return std::nullopt;
    }
}

// Builds one SegmentJson from a plan segment, pulling its ranked alternatives
// from the already-computed EncodingGrid cell for that exact bit-range.
// `isAutoSis` controls whether SegmentPlan::cost is exposed as estimatedCostBits
// (AutoSIS: a cost-model estimate in bits over the full dataset) — for oracle
// plans, seg.cost is already the same value as the grid's bestBytes, so it's
// represented only via sampleBytes, not estimatedCostBits.
static opgraph::SegmentJson buildSegmentJson(
    const SegmentPlan& seg,
    const EncodingGrid& grid,
    const std::vector<uint64_t>& uSample,
    bool isAutoSis)
{
    opgraph::SegmentJson sj;
    sj.bitStart  = seg.bitStart;
    sj.bitEnd    = seg.bitEnd;
    sj.width     = seg.bitEnd - seg.bitStart + 1;
    sj.encoding  = encodingTypeToString(seg.encoding);
    sj.reorderer = subStreamReordererTypeToString(seg.reorderer);
    if (isAutoSis) sj.estimatedCostBits = seg.cost;

    const EncodingCell& cell = grid[seg.bitStart][seg.bitEnd];
    sj.sampleBytes = cell.bytesFor(seg.encoding);
    if (sj.sampleBytes == std::numeric_limits<size_t>::max()) {
        // Chosen encoding wasn't found in this grid cell's results (shouldn't
        // normally happen since both use the same encodingTypes list) — fall
        // back to the grid's best-known bytes so the export stays finite.
        sj.sampleBytes = cell.bestBytes;
    }

    sj.alternatives.reserve(cell.allResults.size());
    for (size_t i = 0; i < cell.allResults.size(); ++i) {
        opgraph::AlternativeJson alt;
        alt.encoding    = encodingTypeToString(cell.allResults[i].first);
        alt.sampleBytes = cell.allResults[i].second;
        alt.rank        = static_cast<int>(i + 1);
        sj.alternatives.push_back(std::move(alt));
    }

    sj.blockFpeStats = buildBlockFpeStats(seg.encoding, seg.bitStart, sj.width, uSample);
    return sj;
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

static std::string padRight(const std::string& s, size_t w) {
    return s.size() < w ? s + std::string(w - s.size(), ' ') : s;
}

static void printPlan(
    const std::vector<SegmentPlan>& plan,
    const std::string& label,
    const EncodingGrid* grid = nullptr)
{
    std::cout << "\n--- " << label << " ---\n";
    for (const auto& seg : plan) {
        std::string enc = padRight(encodingTypeToString(seg.encoding), 28);
        std::cout << "  [" << std::setw(2) << seg.bitStart
                  << ".." << std::setw(2) << seg.bitEnd << "]  " << enc;
        if (grid) {
            size_t b = (*grid)[seg.bitStart][seg.bitEnd].bestBytes;
            std::cout << "  sample_bytes=" << b;
        } else {
            std::cout << "  cost_est=" << std::fixed << std::setprecision(1) << seg.cost;
        }
        std::cout << "\n";
    }
}

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
// OpenZL internal strategy capture — replicate compressBlock with hooks
// ---------------------------------------------------------------------------

struct OpenZLCodecStep {
    std::string name;
    size_t      headerBytes = 0;   // bytes written to frame for this codec's header
    size_t      inputBytes  = 0;   // total content size of all input streams
    size_t      outputBytes = 0;   // total content size of all output streams

    // zstd is a single opaque leaf as far as OpenZL's own codec/stream model is
    // concerned (one ZSTD_compress2 call -- see encode_zstd_binding.c), so
    // there's no real sub-DAG to report for what it does internally, and no
    // way to ask OpenZL itself to break it into an LZ-matching stage and a
    // literals-entropy-coding stage. This is an ESTIMATE, but obtained from
    // zstd's OWN real behavior rather than a substitute algorithm: we re-run
    // libzstd directly (see zstdCompressLiteralsRaw below) on the exact same
    // bytes this step received (captured via ZL_Input_ptr in
    // on_codecEncode_start), at the same compression level, with
    // ZSTD_c_literalCompressionMode forced to ZSTD_lcm_uncompressed -- i.e.
    // "what would this exact zstd call have produced if literals were stored
    // raw instead of Huffman-coded", isolating literal-entropy-coding as the
    // only difference from the real, unmodified outputBytes above. Only set
    // for steps whose name is "zstd".
    std::optional<size_t> lzOnlyBytes;
};

struct OpenZLEncodeStats {
    std::string                  selectedGraph;  // encoding strategy chosen by selector
    std::vector<OpenZLCodecStep> pipeline;       // codecs in execution order
};

// The introspection hooks (on_codecEncode_start/_end, used below) report a
// codec's own internal name -- e.g. "zl.private.zstd" -- which differs from
// the stripped "zstd" name ZL_ReflectionCtx/the JSON export use (see
// build_openzl_digraph's short_labels in plot_operator_graph.py, which strips
// the same "zl."/"zl.private." prefixes). Matching by suffix keeps both
// naming conventions working without hardcoding either one's exact prefix.
static bool isZstdCodecName(const std::string& name) {
    return name.ends_with("zstd");
}

// Compresses `data` with plain libzstd at the given level, with literals
// forced to be stored raw (ZSTD_lcm_uncompressed) instead of Huffman-coded --
// see OpenZLCodecStep::lzOnlyBytes for why. Returns std::nullopt on any zstd
// error (best-effort: a failed side-measurement should never affect the real
// OpenZL compression this is measured alongside).
static std::optional<size_t> zstdCompressLiteralsRaw(
    std::span<const uint8_t> data, int level)
{
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) return std::nullopt;

    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_literalCompressionMode, ZSTD_lcm_uncompressed);

    const size_t bound = ZSTD_compressBound(data.size());
    std::vector<uint8_t> out(bound);
    const size_t r = ZSTD_compress2(cctx, out.data(), bound, data.data(), data.size());
    ZSTD_freeCCtx(cctx);

    if (ZSTD_isError(r)) return std::nullopt;
    return r;
}

// OpenZL's own default level (ZL_COMPRESSIONLEVEL_DEFAULT in zl_compress.h)
// plus the lowest/highest regular zstd levels it forwards to when zstd ends
// up as the selected leaf codec -- swept against the flat, whole-buffer
// OpenZL baseline below so that comparison isn't pinned to a single,
// arbitrary (default) point on OpenZL's own tuning range.
constexpr int kOpenZLLevelLowest  = 1;
constexpr int kOpenZLLevelDefault = 6;
constexpr int kOpenZLLevelHighest = 22;

// Encodes `data` with the OpenZL ZL_GRAPH_NUMERIC compressor, attaches
// introspection hooks to capture the codec pipeline details, and returns the
// raw compressed bytes.  Throws on OpenZL unavailability or compression error.
// Templated on the element type so it can run on both the full int64_t
// dataset and uint64_t bit-range section values (see attachOpenZLToSegments).
// `compressionLevel`: 0 (default) leaves ZL_CParam_compressionLevel unset, so
// OpenZL uses its own internal default -- matches the "0 means not set"
// convention OpenZL itself uses for this parameter.
template <typename T>
static std::vector<uint8_t> encodeOpenZLWithStats(
    std::span<const T> data, OpenZLEncodeStats& stats, int compressionLevel = 0)
{
#ifndef HAVE_OPENZL
    (void)data; (void)stats; (void)compressionLevel;
    throw std::runtime_error("OpenZL not available (HAVE_OPENZL not defined)");
#else
    const size_t N        = data.size();
    const size_t srcBytes = N * sizeof(T);
    const void*  srcPtr   = static_cast<const void*>(data.data());

    ZL_CCtx* ctx = ZL_CCtx_create();
    if (!ctx) throw std::runtime_error("encodeOpenZLWithStats: ZL_CCtx_create failed");

    ZL_TypedRef* inRef = ZL_TypedRef_createNumeric(srcPtr, sizeof(T), N);
    if (!inRef) {
        ZL_CCtx_free(ctx);
        throw std::runtime_error("encodeOpenZLWithStats: ZL_TypedRef_createNumeric failed");
    }

    ZL_Compressor* compressor = ZL_Compressor_create();
    if (!compressor) {
        ZL_TypedRef_free(inRef);
        ZL_CCtx_free(ctx);
        throw std::runtime_error("encodeOpenZLWithStats: ZL_Compressor_create failed");
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
    const ZL_GraphID startGid = ZL_GRAPH_NUMERIC;
#pragma clang diagnostic pop

    if (ZL_isError(ZL_Compressor_selectStartingGraphID(compressor, startGid)) ||
        ZL_isError(ZL_CCtx_refCompressor(ctx, compressor)) ||
        ZL_isError(ZL_CCtx_setParameter(ctx, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)) ||
        (compressionLevel != 0 &&
         ZL_isError(ZL_CCtx_setParameter(ctx, ZL_CParam_compressionLevel, compressionLevel)))) {
        ZL_Compressor_free(compressor);
        ZL_TypedRef_free(inRef);
        ZL_CCtx_free(ctx);
        throw std::runtime_error("encodeOpenZLWithStats: compressor setup failed");
    }

    // Per-call capture state shared between hook callbacks via opaque pointer.
    struct CaptureCtx {
        OpenZLEncodeStats* stats;
        OpenZLCodecStep    pending;  // filled by on_codecEncode_start, pushed by _end
        // Raw bytes zstd receives this invocation (see OpenZLCodecStep::
        // lzOnlyBytes) -- only populated when isZstdCodecName(pending.name).
        std::vector<uint8_t> zstdInputCapture;
        // The compression level THIS pass actually resolved to -- mirrors
        // gcparams.c's own "0 means use ZL_COMPRESSIONLEVEL_DEFAULT" logic,
        // so zstdCompressLiteralsRaw's side-measurement runs at the same
        // level OpenZL's real zstd invocation used, isolating literal
        // compression as the only difference between the two runs.
        int effectiveZstdLevel;
    } capture{&stats, {}, {}, (compressionLevel != 0) ? compressionLevel : kOpenZLLevelDefault};

    ZL_CompressIntrospectionHooks hooks{};
    hooks.opaque = &capture;

    // Capture the encoding strategy name (fires once per block).
    hooks.on_migraphEncode_start = [](void* opaque, ZL_Graph*,
                                      const ZL_Compressor* cpr, ZL_GraphID g,
                                      ZL_Edge*[], size_t) noexcept {
        auto* c = static_cast<CaptureCtx*>(opaque);
        const char* name = ZL_Compressor_Graph_getName(cpr, g);
        if (name && name[0] != '\0' && c->stats->selectedGraph.empty())
            c->stats->selectedGraph = name;
    };

    // Start of each codec transform: record name and sum input stream sizes.
    hooks.on_codecEncode_start = [](void* opaque, ZL_Encoder*,
                                    const ZL_Compressor* cpr, ZL_NodeID nid,
                                    const ZL_Input* ins[], size_t nb) noexcept {
        auto* c = static_cast<CaptureCtx*>(opaque);
        c->pending = {};
        const char* name = ZL_Compressor_Node_getName(cpr, nid);
        c->pending.name = (name && name[0] != '\0') ? name : "(unknown)";
        for (size_t i = 0; i < nb; ++i)
            c->pending.inputBytes += ZL_Data_contentSize(ZL_codemodInputAsData(ins[i]));

        // zstd is an opaque leaf as far as OpenZL's own model goes (see
        // OpenZLCodecStep::lzOnlyBytes) -- grab the exact bytes it's about to
        // compress so a literals-uncompressed zstd side-measurement can run
        // on them once the real output size is known, in on_codecEncode_end.
        c->zstdInputCapture.clear();
        if (isZstdCodecName(c->pending.name)) {
            for (size_t i = 0; i < nb; ++i) {
                const auto* data = ZL_codemodInputAsData(ins[i]);
                const size_t sz = ZL_Data_contentSize(data);
                const auto* ptr = static_cast<const uint8_t*>(ZL_Input_ptr(ins[i]));
                if (ptr && sz > 0)
                    c->zstdInputCapture.insert(c->zstdInputCapture.end(), ptr, ptr + sz);
            }
        }
    };

    // Header bytes written to the compressed frame for this codec.
    hooks.on_ZL_Encoder_sendCodecHeader = [](void* opaque, ZL_Encoder*,
                                              const void*, size_t sz) noexcept {
        static_cast<CaptureCtx*>(opaque)->pending.headerBytes += sz;
    };

    // End of each codec transform: sum output stream sizes and push to pipeline.
    hooks.on_codecEncode_end = [](void* opaque, ZL_Encoder*,
                                  const ZL_Output* outs[], size_t nb,
                                  ZL_Report) noexcept {
        auto* c = static_cast<CaptureCtx*>(opaque);
        for (size_t i = 0; i < nb; ++i)
            c->pending.outputBytes += ZL_Data_contentSize(ZL_codemodConstOutputAsData(outs[i]));

        // Literals-uncompressed zstd side-measurement (see OpenZLCodecStep::
        // lzOnlyBytes) -- best-effort: leave unset if the side-measurement
        // itself fails so it never affects the real OpenZL compression pass.
        if (isZstdCodecName(c->pending.name) && !c->zstdInputCapture.empty()) {
            c->pending.lzOnlyBytes = zstdCompressLiteralsRaw(
                std::span<const uint8_t>(c->zstdInputCapture.data(), c->zstdInputCapture.size()),
                c->effectiveZstdLevel);
        }
        c->zstdInputCapture.clear();

        c->stats->pipeline.push_back(c->pending);
    };

    (void)ZL_CCtx_attachIntrospectionHooks(ctx, &hooks);

    const size_t bound = ZL_compressBound(srcBytes * 2);
    std::vector<uint8_t> buffer(bound);
    ZL_Report r = ZL_CCtx_compressTypedRef(ctx, buffer.data(), bound, inRef);

    (void)ZL_CCtx_detachAllIntrospectionHooks(ctx);
    ZL_Compressor_free(compressor);
    ZL_TypedRef_free(inRef);
    ZL_CCtx_free(ctx);

    if (ZL_isError(r))
        throw std::runtime_error(std::string("encodeOpenZLWithStats: compression failed"));

    buffer.resize(ZL_validResult(r));
    return buffer;
#endif
}

// Holds both the compressed bytes AND the codec-pipeline stats from a single
// OpenZL compression pass, so a caller can report the resulting size
// immediately AND (later, in the JSON export block) build the full operator
// graph from the SAME pass -- avoiding a second, redundant full-dataset
// OpenZL compression just to get the graph.
struct OpenZLCompressResult {
    std::vector<uint8_t> buf;
    OpenZLEncodeStats stats;
};

// Convenience wrapper around encodeOpenZLWithStats, tolerant of OpenZL being
// unavailable or failing on a given buffer (e.g. table-size limits on
// high-cardinality data) -- same try/catch shape as the flat OpenZL baseline
// above, reused here for double-compression (plan output -> OpenZL) so each
// call site doesn't need its own try/catch.
static std::optional<OpenZLCompressResult> tryOpenZLCompressFull(std::span<const uint8_t> bytes) {
    try {
        OpenZLCompressResult result;
        result.buf = encodeOpenZLWithStats<uint8_t>(bytes, result.stats);
        return result;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// ---------------------------------------------------------------------------
// OpenZL operator-graph JSON export — post-hoc reflection over the already-
// compressed frame via ZL_ReflectionCtx. Unlike the introspection hooks above
// (which only see a flat execution-order list), this walks the true stream/
// codec DAG. Mirrors openzl/tools/streamdump/stream_dump2.c's fill_csize() and
// stream-graph walk, translated to JSON instead of Graphviz DOT. No custom
// decoder registration is needed since ZL_GRAPH_NUMERIC only uses standard
// (built-in) codecs — see stream_dump2_noop_shim.c for the equivalent no-op
// case in OpenZL's own tool.
// ---------------------------------------------------------------------------

#ifdef HAVE_OPENZL
// Recursively computes stream `streamIdx`'s share of total compressed bytes by
// walking forward through consumer codecs. Direct port of stream_dump2.c's
// fill_csize().
static size_t fillStreamShare(
    const ZL_ReflectionCtx* rctx, std::vector<size_t>& shareBytes, size_t streamIdx)
{
    if (shareBytes[streamIdx] != std::numeric_limits<size_t>::max())
        return shareBytes[streamIdx];

    const ZL_DataInfo* info = ZL_ReflectionCtx_getStream_lastChunk(rctx, streamIdx);
    const ZL_CodecInfo* consumer = ZL_DataInfo_getConsumerCodec(info);
    if (!consumer) {
        shareBytes[streamIdx] = ZL_DataInfo_getContentSize(info);
        return shareBytes[streamIdx];
    }

    size_t total = ZL_CodecInfo_getHeaderSize(consumer);
    const size_t nbOutputs = ZL_CodecInfo_getNumOutputs(consumer);
    for (size_t i = 0; i < nbOutputs; ++i) {
        const ZL_DataInfo* out = ZL_CodecInfo_getOutput(consumer, i);
        total += fillStreamShare(rctx, shareBytes, ZL_DataInfo_getIndex(out));
    }
    shareBytes[streamIdx] = total;
    return total;
}

static const char* zlTypeName(ZL_Type t) {
    switch (t) {
        case ZL_Type_serial:  return "Serialized";
        case ZL_Type_struct:  return "Fixed_Width";
        case ZL_Type_numeric: return "Numeric";
        case ZL_Type_string:  return "Variable_Size";
        default:               return "Unknown";
    }
}

static opgraph::OpenZLGraphJson buildOpenZLGraphJson(
    const std::vector<uint8_t>& compressedBuffer, const OpenZLEncodeStats& stats)
{
    opgraph::OpenZLGraphJson g;
    g.compressedBytes = compressedBuffer.size();
    g.selectedGraph    = stats.selectedGraph;

    // Aggregate the zstd literals-uncompressed side-measurement across every
    // "zstd" invocation in this pass (see OpenZLCodecStep::lzOnlyBytes) into
    // one number -- Python re-aggregates all same-named codec invocations
    // into a single displayed node anyway (build_openzl_digraph), so there's
    // no need to attribute this per-invocation; only set if at least one
    // zstd step produced a measurement (zstdCompressLiteralsRaw can itself
    // fail/be skipped).
    size_t zstdLzOnlySum = 0;
    bool   haveZstdLzOnly = false;
    for (const auto& step : stats.pipeline) {
        if (isZstdCodecName(step.name) && step.lzOnlyBytes.has_value()) {
            zstdLzOnlySum += *step.lzOnlyBytes;
            haveZstdLzOnly = true;
        }
    }
    if (haveZstdLzOnly) g.zstdLzOnlyBytes = zstdLzOnlySum;

    ZL_ReflectionCtx* rctx = ZL_ReflectionCtx_create();
    if (!rctx) throw std::runtime_error("buildOpenZLGraphJson: ZL_ReflectionCtx_create failed");

    ZL_Report r = ZL_ReflectionCtx_setCompressedFrame(
        rctx, compressedBuffer.data(), compressedBuffer.size());
    if (ZL_isError(r)) {
        ZL_ReflectionCtx_free(rctx);
        throw std::runtime_error("buildOpenZLGraphJson: ZL_ReflectionCtx_setCompressedFrame failed");
    }

    g.frameFormatVersion       = ZL_ReflectionCtx_getFrameFormatVersion(rctx);
    g.frameHeaderSize          = ZL_ReflectionCtx_getFrameHeaderSize(rctx);
    g.frameFooterSize          = ZL_ReflectionCtx_getFrameFooterSize(rctx);
    g.totalTransformHeaderSize = ZL_ReflectionCtx_getTotalTransformHeaderSize_lastChunk(rctx);

    const size_t nbStreams = ZL_ReflectionCtx_getNumStreams_lastChunk(rctx);
    std::vector<size_t> shareBytes(nbStreams, std::numeric_limits<size_t>::max());
    g.streams.reserve(nbStreams);
    for (size_t i = 0; i < nbStreams; ++i) {
        const ZL_DataInfo* info = ZL_ReflectionCtx_getStream_lastChunk(rctx, i);
        const size_t share = fillStreamShare(rctx, shareBytes, i);

        opgraph::OpenZLStreamJson sj;
        sj.id                  = i;
        sj.type                = zlTypeName(ZL_DataInfo_getType(info));
        sj.eltWidth             = ZL_DataInfo_getEltWidth(info);
        sj.numElts              = ZL_DataInfo_getNumElts(info);
        sj.contentSize          = ZL_DataInfo_getContentSize(info);
        sj.compressedShareBytes = share;
        sj.compressedSharePct   = compressedBuffer.empty() ? 0.0
            : 100.0 * static_cast<double>(share) / static_cast<double>(compressedBuffer.size());

        if (const ZL_CodecInfo* producer = ZL_DataInfo_getProducerCodec(info))
            sj.producerCodecId = ZL_CodecInfo_getIndex(producer);
        if (const ZL_CodecInfo* consumer = ZL_DataInfo_getConsumerCodec(info))
            sj.consumerCodecId = ZL_CodecInfo_getIndex(consumer);

        g.streams.push_back(std::move(sj));
    }

    const size_t nbCodecs = ZL_ReflectionCtx_getNumCodecs_lastChunk(rctx);
    g.codecs.reserve(nbCodecs);
    for (size_t i = 0; i < nbCodecs; ++i) {
        const ZL_CodecInfo* info = ZL_ReflectionCtx_getCodec_lastChunk(rctx, i);

        opgraph::OpenZLCodecJson cj;
        cj.id  = i;
        const char* nm = ZL_CodecInfo_getName(info);
        cj.name        = nm ? nm : "(unknown)";
        cj.codecId     = ZL_CodecInfo_getCodecID(info);
        cj.isStandard  = ZL_CodecInfo_isStandardCodec(info);
        cj.headerSize  = ZL_CodecInfo_getHeaderSize(info);

        const size_t nbIn = ZL_CodecInfo_getNumInputs(info);
        cj.inputStreamIds.reserve(nbIn);
        for (size_t k = 0; k < nbIn; ++k)
            cj.inputStreamIds.push_back(ZL_DataInfo_getIndex(ZL_CodecInfo_getInput(info, k)));

        const size_t nbOut = ZL_CodecInfo_getNumOutputs(info);
        cj.outputStreamIds.reserve(nbOut);
        for (size_t k = 0; k < nbOut; ++k)
            cj.outputStreamIds.push_back(ZL_DataInfo_getIndex(ZL_CodecInfo_getOutput(info, k)));

        g.codecs.push_back(std::move(cj));
    }

    ZL_ReflectionCtx_free(rctx);
    return g;
}

// ---------------------------------------------------------------------------
// Per-segment OpenZL — applies OpenZL directly to each OracleSIS segment's own
// bit-range data, extracted from the FULL dataset (not the sample used for
// the grid/alternatives), so a segment's chosen SubIntSplit encoding can be
// compared against what OpenZL alone achieves on that exact same data at full
// scale. Mutates each SegmentJson in place, leaving openZlBytes/openZlGraph
// unset (null in the JSON) if OpenZL fails on a given segment. Note: running
// full-dataset OpenZL per segment is expensive (one ZL_CCtx compress over
// N=datasetSize elements per segment) — this dominates total runtime when
// there are many segments.
// ---------------------------------------------------------------------------

static void attachOpenZLToSegments(
    std::vector<opgraph::SegmentJson>& segments, const std::vector<uint64_t>& uFullData)
{
    for (auto& sj : segments) {
        const uint64_t mask = (sj.width == 64) ? ~uint64_t{0} : ((uint64_t{1} << sj.width) - 1);
        std::vector<uint64_t> section(uFullData.size());
        for (size_t i = 0; i < uFullData.size(); ++i)
            section[i] = (uFullData[i] >> sj.bitStart) & mask;

        try {
            OpenZLEncodeStats segStats;
            auto buf = encodeOpenZLWithStats<uint64_t>(
                std::span<const uint64_t>(section.data(), section.size()), segStats);
            sj.openZlBytes = buf.size();
            sj.openZlGraph = buildOpenZLGraphJson(buf, segStats);
        } catch (const std::exception&) {
            // OpenZL unavailable or failed on this segment — leave fields unset.
        }
    }
}

// ---------------------------------------------------------------------------
// Double-compression: takes each OracleSIS segment's ALREADY SubIntSplit-
// encoded byte buffer — re-derived from the oracle's chosen section codec
// over the FULL dataset's bit range — and runs OpenZL again on top of it, to
// see whether/how much further OpenZL can squeeze out of already-compressed
// data and what codec graph it picks on that (typically higher-entropy) byte
// stream. Also records the oracle codec's own full-dataset size as
// fullDatasetBytes, so the comparison is apples-to-apples with openZlBytes
// above rather than mixing it with the sample-scale sampleBytes field.
// `segmentsJson` and `segmentsPlan` must be the same plan in the same order
// (as produced together by buildSegmentJson over a SegmentPlan vector).
// ---------------------------------------------------------------------------

static void attachOpenZLOnOracleBytes(
    std::vector<opgraph::SegmentJson>& segmentsJson,
    const std::vector<SegmentPlan>& segmentsPlan,
    const std::vector<uint64_t>& uFullData)
{
    const size_t n = std::min(segmentsJson.size(), segmentsPlan.size());
    for (size_t i = 0; i < n; ++i) {
        auto& sj = segmentsJson[i];
        const auto& seg = segmentsPlan[i];
        const int width = seg.bitEnd - seg.bitStart + 1;
        const uint64_t mask = (width == 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
        std::vector<uint64_t> section(uFullData.size());
        for (size_t k = 0; k < uFullData.size(); ++k)
            section[k] = (uFullData[k] >> seg.bitStart) & mask;

        try {
            auto codec = makeSectionCodec(seg.encoding, static_cast<uint8_t>(width));
            std::span<const uint64_t> sectionSpan(section.data(), section.size());
            auto encoded = codec->encode(sectionSpan);
            const std::vector<uint8_t>& oracleBytes = encoded.data();
            sj.fullDatasetBytes = oracleBytes.size();

            OpenZLEncodeStats reStats;
            auto buf = encodeOpenZLWithStats<uint8_t>(
                std::span<const uint8_t>(oracleBytes.data(), oracleBytes.size()), reStats);
            sj.openZlOnOracleBytes = buf.size();
            sj.openZlOnOracleGraph = buildOpenZLGraphJson(buf, reStats);
        } catch (const std::exception&) {
            // OpenZL unavailable or failed on this segment — leave fields unset.
        }
    }
}
#endif // HAVE_OPENZL

// ---------------------------------------------------------------------------
// OpenZL strategy analysis printout
// ---------------------------------------------------------------------------

static void printOpenZLAnalysis(
    const OpenZLEncodeStats& stats,
    size_t totalCompressedBytes,
    size_t datasetSize)
{
    const double bpe = static_cast<double>(totalCompressedBytes) * 8.0
                       / static_cast<double>(datasetSize);
    const double ratio = static_cast<double>(totalCompressedBytes)
                         / static_cast<double>(datasetSize * sizeof(int64_t));

    std::cout << "\n=== OpenZL Internal Strategy (full dataset) ===\n";
    std::cout << "  Compressed:     " << totalCompressedBytes
              << " bytes  (" << std::fixed << std::setprecision(3) << bpe << " bits/elem"
              << "  ratio=" << ratio << "x)\n";
    std::cout << "  Encoding graph: \""
              << (stats.selectedGraph.empty() ? "(not captured)" : stats.selectedGraph)
              << "\"\n";

    if (stats.pipeline.empty()) {
        std::cout << "  (no codec pipeline data captured)\n";
        return;
    }

    // Aggregate ALL invocations of each unique codec name so that per-block
    // repeated calls (e.g. fse_v2 × thousands) collapse to a single row.
    // Preserve first-occurrence order for readability.
    struct AggStep {
        std::string name;
        size_t count       = 0;
        size_t totalIn     = 0;
        size_t totalOut    = 0;
        size_t totalHeader = 0;
    };
    std::vector<std::string>         nameOrder;
    std::map<std::string, AggStep>   byName;
    for (const auto& s : stats.pipeline) {
        if (!byName.count(s.name)) nameOrder.push_back(s.name);
        auto& a = byName[s.name];
        a.name = s.name;
        a.count++;
        a.totalIn     += s.inputBytes;
        a.totalOut    += s.outputBytes;
        a.totalHeader += s.headerBytes;
    }

    std::cout << "\n  Codec pipeline (" << nameOrder.size()
              << " unique stages, " << stats.pipeline.size() << " total calls):\n";
    size_t totalHeaderBytes = 0;
    std::vector<AggStep> agg;
    for (size_t i = 0; i < nameOrder.size(); ++i) {
        const auto& a = byName[nameOrder[i]];
        agg.push_back(a);
        totalHeaderBytes += a.totalHeader;
        const double r = (a.totalIn > 0)
            ? static_cast<double>(a.totalOut) / static_cast<double>(a.totalIn) : 0.0;

        std::cout << "    #" << std::setw(2) << (i + 1) << "  "
                  << padRight(a.name, 32);
        if (a.count > 1)
            std::cout << " ×" << std::setw(5) << a.count;
        else
            std::cout << "        ";
        std::cout << "  in:" << std::setw(13) << a.totalIn << " B"
                  << "  out:" << std::setw(13) << a.totalOut << " B"
                  << "  hdr:" << std::setw(6) << a.totalHeader << " B"
                  << "  ratio:" << std::setprecision(3) << r << "x"
                  << "\n";
    }

    // High-level summary
    std::cout << "\n  Summary:\n";
    std::cout << "    Unique codec stages:  " << agg.size()
              << "  (" << stats.pipeline.size() << " total invocations)\n";
    std::cout << "    Total codec headers:  " << totalHeaderBytes << " bytes\n";
    {
        const size_t firstIn = stats.pipeline.front().inputBytes;
        const double e2e = (firstIn > 0)
            ? static_cast<double>(totalCompressedBytes) / static_cast<double>(firstIn) : 0.0;
        std::cout << "    End-to-end ratio:     " << std::setprecision(3) << e2e << "x"
                  << "  (" << firstIn << " B  →  " << totalCompressedBytes << " B)\n";
    }

    // Detect strategy keywords from graph name and aggregated codec names.
    auto hasName = [&](const char* kw) {
        if (stats.selectedGraph.find(kw) != std::string::npos) return true;
        for (const auto& a : agg)
            if (a.name.find(kw) != std::string::npos) return true;
        return false;
    };

    const bool usesTranspose = hasName("transpose") || hasName("split");
    const bool usesHuffman   = hasName("huffman");
    const bool usesFieldLz   = hasName("field_lz");
    const bool usesDelta     = hasName("delta") || hasName("Delta");
    const bool usesRangePack = hasName("range_pack");

    std::cout << "\n  Strategy detected: ";
    std::vector<std::string> tags;
    if (usesRangePack) tags.push_back("range-packing");
    if (usesFieldLz)   tags.push_back("field-level LZ");
    if (usesTranspose) tags.push_back("byte-transposition");
    if (usesHuffman)   tags.push_back("Huffman entropy coding");
    if (usesDelta)     tags.push_back("delta coding");
    for (size_t i = 0; i < tags.size(); ++i) {
        if (i) std::cout << " + ";
        std::cout << tags[i];
    }
    if (tags.empty()) std::cout << "(see codec names above)";
    std::cout << "\n";

    std::cout << "\n  Interpretation (why OpenZL beats SubIntSplit):\n";

    if (usesTranspose && usesHuffman) {
        std::cout <<
            "    OpenZL transposes the int64 byte layout (struct-of-arrays): all byte-0\n"
            "    values together, all byte-1 values together, etc. — then Huffman-codes\n"
            "    each byte stream independently. High-order bytes (timestamp bits in\n"
            "    Snowflake IDs) change slowly across consecutive IDs, so those streams\n"
            "    achieve very low entropy and very high compression ratios (~0.27-0.31x\n"
            "    above). This is conceptually similar to SubIntSplit's bit-decomposition,\n"
            "    but at byte granularity with full Huffman entropy coding.\n"
            "    SubIntSplit currently uses FrequencyPartition/BitPacking for the high\n"
            "    bits — adding FSE/Huffman as section codecs would close most of this gap.\n";
    } else if (usesDelta) {
        std::cout <<
            "    OpenZL applied delta-coding before entropy compression. Snowflake IDs\n"
            "    are monotonically increasing, so deltas are small positive integers —\n"
            "    ideal for entropy coding. SubIntSplit (without BWT) cannot exploit\n"
            "    this sequential structure. Enabling allowReorderers=true (BWT) should\n"
            "    close much of this gap.\n";
    } else {
        std::cout <<
            "    Strategy: \"" << stats.selectedGraph << "\". See codec pipeline above.\n"
            "    The gap likely comes from cross-byte entropy coding that SubIntSplit's\n"
            "    independent per-bit-range encoding cannot replicate.\n";
    }

    // Identify which aggregated stage provides the most compression.
    size_t bestIdx = 0;
    double bestGain = 0.0;
    for (size_t i = 0; i < agg.size(); ++i) {
        const double gain = (agg[i].totalIn > agg[i].totalOut)
            ? static_cast<double>(agg[i].totalIn - agg[i].totalOut) : 0.0;
        if (gain > bestGain) { bestGain = gain; bestIdx = i; }
    }
    if (bestGain > 0) {
        const auto& best = agg[bestIdx];
        const double r = static_cast<double>(best.totalOut) / static_cast<double>(best.totalIn);
        std::cout << "\n  Biggest compression stage: \"" << best.name << "\""
                  << "  saved " << static_cast<size_t>(bestGain) << " bytes"
                  << "  (ratio=" << std::setprecision(3) << r << "x)\n";
    }
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
