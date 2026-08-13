#pragma once

// The oracle side of the SubIntSplit selection question: for every bit range
// [l..r] and every candidate section codec, how many bytes does that codec
// ACTUALLY produce on a sample?  A DP over that grid gives the best plan any
// selector could have chosen from the same candidate set, which is the reference
// AutoSIS's cost-model plan is judged against.
//
// This lived inside Benchmarks/explore_best_encoding.cpp, where it could only be
// reached by an interactive tool with hardcoded paths.  It is a header so that
// bench_costmodel_oracle (paper tables) and bench_openzl_graph (per-segment
// codec DAGs) measure the same grid the interactive tool prints.
//
// Cost: 64*64/2 bit ranges x |encodingTypes| encodes per grid.  Nothing here is
// memoized across calls, because a grid is a pure function of (sample,
// encodingTypes, minSegmentWidth) and every caller wants a different sample.

#include "benchmark/OperatorGraphJson.hpp"
#include "encoders/BlockFORFPEEncoder.hpp"
#include "encoders/OpenZLEncoder.hpp"
#include "encoders/SubIntEncodingUtils.hpp"
#include "encoders/SubIntSplitEncoder.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"
#include "encodings/EncodingType.hpp"
#include "generators/samplers/EncodingSamplingProfile.hpp"
#include "generators/samplers/StreamSampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace encodings::benchmark {

using encodings::EncodingType;
using encodings::encoders::selectors::SegmentPlan;
namespace opgraph = encodings::benchmark::operatorgraph;

// ─── Section codec factory ───────────────────────────────────────────────────

/// The oracle's candidate set, resolved to a concrete section codec.
///
/// Wider than `CostModelSet::defaultEncodings()` on purpose: a candidate with no
/// analytical cost model (OpenZL, the Cascading* compositions) can still be
/// oracle-encoded, and excluding it would understate the headroom the cost model
/// is missing.  Throwing on an unsupported type rather than returning null keeps
/// a typo in a driver's encoding list from silently shrinking the candidate set.
inline std::shared_ptr<encodings::encoders::ISectionCodecIntegral<uint64_t>>
makeSectionCodec(EncodingType enc, uint8_t width) {
    using namespace encodings::encoders::detail_trisplit;
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
        case EncodingType::BlockFSEEncoding:                return makeBlockFSESection<uint64_t>(width);
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

// ─── The grid ────────────────────────────────────────────────────────────────

constexpr int kGridBits = 64;

struct EncodingCell {
    /// Every (encoding, bytes) pair that encoded successfully, ascending by
    /// bytes — so rank is the index and no caller has to re-sort.
    std::vector<std::pair<EncodingType, size_t>> allResults;
    size_t       bestBytes    = std::numeric_limits<size_t>::max();
    EncodingType bestEncoding = EncodingType::RawEncoding;

    size_t bytesFor(EncodingType enc) const {
        for (const auto& [e, b] : allResults)
            if (e == enc) return b;
        return std::numeric_limits<size_t>::max();
    }

    /// 1-based position in the actual-bytes ranking; 0 when the encoding is not
    /// in this cell (it threw on this bit range).  0 rather than a large number
    /// so a reader cannot average it into a rank statistic by accident.
    int rankOf(EncodingType enc) const {
        for (size_t i = 0; i < allResults.size(); ++i)
            if (allResults[i].first == enc) return static_cast<int>(i + 1);
        return 0;
    }
};

/// grid[bitStart][bitEnd]; only cells with bitEnd >= bitStart + minSegmentWidth - 1
/// are populated.
using EncodingGrid = std::vector<std::vector<EncodingCell>>;

inline uint64_t sectionMask(int width) {
    return (width == 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
}

inline std::vector<uint64_t> toU64(std::span<const int64_t> v) {
    std::vector<uint64_t> u(v.size());
    for (size_t i = 0; i < v.size(); ++i) u[i] = static_cast<uint64_t>(v[i]);
    return u;
}

/// Encodes every (bit range, encoding) pair on `sample` and records the byte
/// count.  A codec that throws on a degenerate input is omitted from the cell
/// rather than recorded as infinitely large: "could not encode this range" and
/// "encoded it badly" are different facts and only the second belongs in a
/// ranking.
inline EncodingGrid computeEncodingGrid(
    const std::vector<int64_t>& sample,
    const std::vector<EncodingType>& encodingTypes,
    int minSegmentWidth)
{
    EncodingGrid grid(kGridBits, std::vector<EncodingCell>(kGridBits));

    const size_t n = sample.size();
    const std::vector<uint64_t> uSample = toU64(sample);
    std::vector<uint64_t> sectionValues(n);

    for (int l = 0; l < kGridBits; ++l) {
        for (int r = l + minSegmentWidth - 1; r < kGridBits; ++r) {
            const int width = r - l + 1;
            const uint64_t mask = sectionMask(width);

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
                }
            }

            std::sort(cell.allResults.begin(), cell.allResults.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
        }
    }
    return grid;
}

// ─── Oracle DP ───────────────────────────────────────────────────────────────

/// Partitions bits [0..63] to minimise total sample bytes.
///
/// Agnostic to how the grid was assembled — it only reads bestBytes and
/// bestEncoding per cell — which is what lets it run unmodified over a merged
/// multi-profile grid.  No split penalty: the oracle is the unconstrained lower
/// bound, and charging it AutoSIS's per-split header estimate would make it a
/// second selector rather than a reference.
inline std::vector<SegmentPlan> runOracleDP(const EncodingGrid& grid, int minSegmentWidth) {
    constexpr size_t kInf = std::numeric_limits<size_t>::max() / 2;

    std::vector<size_t> dp(kGridBits + 1, kInf);
    std::vector<int>    parent(kGridBits + 1, -1);
    dp[0] = 0;

    for (int i = minSegmentWidth; i <= kGridBits; ++i) {
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
    int cur = kGridBits;
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

// ─── Per-profile merge ───────────────────────────────────────────────────────

/// Merges per-sampling-profile grids so each EncodingType's cost always comes
/// from the grid of ITS OWN designated profile (see EncodingSamplingProfile.hpp),
/// never from "whichever sample happened to look cheaper".
///
/// The rescaling is load-bearing.  StreamSampler truncates to a multiple of
/// blockSize, so a 10000-target sample is 9728 elements at blockSize=512 and
/// 8192 at 2048; runOracleDP sums raw byte counts assuming one scale.  Left
/// unscaled, an encoding costed against the smaller sample looks cheaper purely
/// from having fewer elements to encode.
inline EncodingGrid mergeEncodingGridsByProfile(
    const std::map<encodings::generators::samplers::SamplingProfile, const EncodingGrid*>& gridsByProfile,
    const std::map<encodings::generators::samplers::SamplingProfile, size_t>& sampleSizeByProfile,
    size_t referenceSampleSize,
    const std::vector<EncodingType>& encodingTypes,
    int minSegmentWidth)
{
    using encodings::generators::samplers::preferredSamplingProfile;

    EncodingGrid merged(kGridBits, std::vector<EncodingCell>(kGridBits));

    for (int l = 0; l < kGridBits; ++l) {
        for (int r = l + minSegmentWidth - 1; r < kGridBits; ++r) {
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
                const double scale =
                    static_cast<double>(referenceSampleSize) / static_cast<double>(thisN);
                const size_t bytes =
                    static_cast<size_t>(std::llround(static_cast<double>(rawBytes) * scale));

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

// ─── The three samples ───────────────────────────────────────────────────────

/// Block sizes of the three samples the oracle is computed against.
///
/// These are three DIFFERENT samples of the same stream, not three tries at one
/// sample, and collapsing them would delete the finding they exist to expose:
///
///   16   — matches SubIntSplit's own default blockSize, so the random-sample
///          oracle sees exactly what AutoSIS sees.
///   512  — matches FrameOfReference<512>'s frame size.  A 16-element scatter
///          shows FOR ~39-bit residuals on Snowflake IDs where a real frame sees
///          ~13, so the random grid blind-spots every frame-local codec.
///   2048 — long enough for RunLength (and any delta-style codec) to see real
///          runs across consecutive rows, invisible at both sizes above.
constexpr size_t kRandomBlockSize = 16;
constexpr size_t kConsecBlockSize = 512;
constexpr size_t kWideBlockSize   = 2048;

struct ProfileSamples {
    std::vector<int64_t> random;
    std::vector<int64_t> consecutive;
    std::vector<int64_t> wide;
};

inline ProfileSamples drawProfileSamples(std::span<const int64_t> fullData, size_t maxSamples) {
    using Sampler = encodings::generators::samplers::StreamSampler<int64_t>;
    const auto draw = [&](size_t blockSize) {
        Sampler::Config c;
        c.maxSamples = maxSamples;
        c.blockSize  = blockSize;
        c.stride     = 0;
        return Sampler::sample(fullData, c);
    };
    return {draw(kRandomBlockSize), draw(kConsecBlockSize), draw(kWideBlockSize)};
}

/// The three grids plus their merge.  `wide` is computed only over the
/// WideContiguous-classified subset of `encodingTypes`: it exists solely to feed
/// the merge, never to stand alone as a diagnostic, so encoding every type
/// against it would be paid for and then discarded.
struct ProfileGrids {
    EncodingGrid random;
    EncodingGrid consecutive;
    EncodingGrid wide;
    EncodingGrid merged;
};

inline ProfileGrids computeProfileGrids(const ProfileSamples& samples,
                                        const std::vector<EncodingType>& encodingTypes,
                                        int minSegmentWidth) {
    using encodings::generators::samplers::SamplingProfile;
    using encodings::generators::samplers::preferredSamplingProfile;

    std::vector<EncodingType> wideTypes;
    for (auto e : encodingTypes)
        if (preferredSamplingProfile(e) == SamplingProfile::WideContiguous) wideTypes.push_back(e);

    ProfileGrids g;
    g.random      = computeEncodingGrid(samples.random, encodingTypes, minSegmentWidth);
    g.consecutive = computeEncodingGrid(samples.consecutive, encodingTypes, minSegmentWidth);
    g.wide        = computeEncodingGrid(samples.wide, wideTypes, minSegmentWidth);
    g.merged      = mergeEncodingGridsByProfile(
        {{SamplingProfile::Random, &g.random},
         {SamplingProfile::Consecutive, &g.consecutive},
         {SamplingProfile::WideContiguous, &g.wide}},
        {{SamplingProfile::Random, samples.random.size()},
         {SamplingProfile::Consecutive, samples.consecutive.size()},
         {SamplingProfile::WideContiguous, samples.wide.size()}},
        /*referenceSampleSize=*/samples.random.size(), encodingTypes, minSegmentWidth);
    return g;
}

// ─── SegmentPlan helpers ─────────────────────────────────────────────────────

inline std::map<std::pair<int, int>, EncodingType> planByBounds(
    const std::vector<SegmentPlan>& plan) {
    std::map<std::pair<int, int>, EncodingType> m;
    for (const auto& seg : plan) m[{seg.bitStart, seg.bitEnd}] = seg.encoding;
    return m;
}

/// Segments whose (encoding AND bounds) both appear in `other`.
inline size_t countMatchingSegments(const std::vector<SegmentPlan>& plan,
                                    const std::vector<SegmentPlan>& other) {
    const auto byBounds = planByBounds(other);
    size_t matches = 0;
    for (const auto& seg : plan) {
        auto it = byBounds.find({seg.bitStart, seg.bitEnd});
        if (it != byBounds.end() && it->second == seg.encoding) ++matches;
    }
    return matches;
}

/// Sums the plan's own chosen encodings on `grid`'s scale.  Not the same as the
/// DP objective for an AutoSIS plan, whose segments were chosen on a cost model
/// and may not be a grid cell's best.
inline size_t planSampleBytes(const std::vector<SegmentPlan>& plan, const EncodingGrid& grid) {
    size_t total = 0;
    for (const auto& seg : plan) {
        const size_t b = grid[seg.bitStart][seg.bitEnd].bytesFor(seg.encoding);
        total += (b == std::numeric_limits<size_t>::max())
                     ? grid[seg.bitStart][seg.bitEnd].bestBytes : b;
    }
    return total;
}

/// One segment's bit range extracted from a full stream, ready for a section
/// codec or a per-segment OpenZL pass.
inline std::vector<uint64_t> extractSection(const std::vector<uint64_t>& u, int bitStart, int width) {
    const uint64_t mask = sectionMask(width);
    std::vector<uint64_t> section(u.size());
    for (size_t i = 0; i < u.size(); ++i) section[i] = (u[i] >> bitStart) & mask;
    return section;
}

inline std::string padRight(const std::string& s, size_t w) {
    return s.size() < w ? s + std::string(w - s.size(), ' ') : s;
}

inline void printPlan(const std::vector<SegmentPlan>& plan, const std::string& label,
                      const EncodingGrid* grid = nullptr) {
    std::cout << "\n--- " << label << " ---\n";
    for (const auto& seg : plan) {
        std::cout << "  [" << std::setw(2) << seg.bitStart << ".." << std::setw(2) << seg.bitEnd
                  << "]  " << padRight(encodingTypeToString(seg.encoding), 28);
        if (grid) std::cout << "  sample_bytes=" << (*grid)[seg.bitStart][seg.bitEnd].bestBytes;
        else std::cout << "  cost_est=" << std::fixed << std::setprecision(1) << seg.cost;
        std::cout << "\n";
    }
}

// ─── Operator-graph JSON for a plan ──────────────────────────────────────────

/// BlockFORFPE's aggregate tier stats, which the encoder only writes into
/// customMetadata when BLOCKFORFPE_STATS is set — so this re-encodes the
/// segment's sample slice to read them back.  nullopt for every other encoding,
/// and for a slice the codec throws on.
inline std::optional<opgraph::BlockFpeStatsJson> buildBlockFpeStats(
    EncodingType enc, int bitStart, int width, const std::vector<uint64_t>& uSample) {
    if (enc != EncodingType::BlockFORFPEEncoding) return std::nullopt;

    const std::vector<uint64_t> section = extractSection(uSample, bitStart, width);
    try {
        auto codec = makeSectionCodec(enc, static_cast<uint8_t>(width));
        auto encoded = codec->encode(std::span<const uint64_t>(section.data(), section.size()));
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

/// `isAutoSis` decides whether SegmentPlan::cost is exposed as
/// estimatedCostBits: for an AutoSIS plan it is a cost-model estimate in bits
/// over the full dataset, but for an oracle plan it is already the same number
/// as the grid's bestBytes and would read as an estimate that happens to be
/// exactly right.
inline opgraph::SegmentJson buildSegmentJson(const SegmentPlan& seg, const EncodingGrid& grid,
                                             const std::vector<uint64_t>& uSample, bool isAutoSis) {
    opgraph::SegmentJson sj;
    sj.bitStart  = seg.bitStart;
    sj.bitEnd    = seg.bitEnd;
    sj.width     = seg.bitEnd - seg.bitStart + 1;
    sj.encoding  = encodingTypeToString(seg.encoding);
    sj.reorderer = encodings::encoders::selectors::subStreamReordererTypeToString(seg.reorderer);
    if (isAutoSis) sj.estimatedCostBits = seg.cost;

    const EncodingCell& cell = grid[seg.bitStart][seg.bitEnd];
    sj.sampleBytes = cell.bytesFor(seg.encoding);
    if (sj.sampleBytes == std::numeric_limits<size_t>::max()) sj.sampleBytes = cell.bestBytes;

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

}  // namespace encodings::benchmark
