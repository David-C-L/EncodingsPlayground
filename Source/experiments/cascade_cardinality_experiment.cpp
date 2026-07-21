// Sweeps CascadingFOREncoder's cardinality/bit-width statistics, and separately
// the compression rate BlockFrequencyPartitionEncoder achieves on the deepest
// cascade level's residuals, across two axes:
//   1. "plain_for_depth1"  - a single FOR level (no residual cascading) swept
//                            across many frame sizes, i.e. today's ordinary
//                            FOREncoder/AdaptiveFOREncoder shape.
//   2. "cascading_for"     - the residual stream cascaded to varying depth,
//                            starting from varying outermost frame sizes,
//                            shrinking geometrically at each level.
//
// Both sweeps share the same frame-size/depth grid (kFrameSizes / kStartFrameSizes
// / kDepths below) and datasets, so the two output CSVs are directly comparable
// row-for-row on (sweep, dataset, startFrameSize, depth).
//
// Output 1: cardinality/bit-width CSV (one row per level per config, both
// residual and reference streams cascaded) — see plot_cascade_cardinality.py.
// This never calls CascadingFOREncoder::encode()/decode(): analyzeCascade()
// replays the same partitioning logic without serializing.
//
// Output 2: BlockFPE compression CSV (one row per config per deepest-level
// policy) — runs the REAL, complete two-stage scheme end-to-end: a genuine
// CascadingFOREncoder<int64_t>(residualSchedule=<swept schedule>,
// referenceSchedule={}) whose residualLeafEncoder AND referenceLeafEncoder
// are both BlockFrequencyPartitionEncoder<int64_t>(bitPackFallback=true) —
// i.e. literally "FOR, then BlockFPE", applied to both the residual stream
// and (at every cascade level) that level's own reference array, with no
// further cascading on the reference side. This is deliberate: an earlier
// version of this experiment measured only the residual stream's compressed
// size via computeDeepestResiduals(), silently never accounting for the cost
// of storing the per-frame reference values needed to actually reconstruct
// the data — which both inflated the apparent compression ratio and biased
// every "best over sweep" comparison toward the smallest frame size (smaller
// frames shrink residuals but need more references, a cost that was never
// counted). Encoding the real dataset end-to-end via CascadingFOREncoder
// fixes this: reported compressedSize is the true, complete, decodable
// total, directly comparable to BlockFORFPEEncoder's equally-complete number
// in Output 3. bitPackFallback=true (see BlockFrequencyPartitionEncoder.hpp's
// constructor doc): the fallback tier is bit-packed to its minimal width
// rather than a fixed sizeof(T) bytes/element, closing the gap identified
// against BlockFORFPEEncoder. The deepest level's reference policy
// (MIN/FIRST/MID) is swept too: intermediate levels' policy is provably
// inert for residual values (see CascadingFOREncoder.hpp's
// telescoping-identity note), but the deepest level's policy can affect
// cross-window value alignment, which no per-window-local cardinality
// statistic captures — this is the one place that genuinely needs an
// empirical answer rather than a proof.
//
// Output 3: BlockFORFPEEncoder comparison CSV (one row per dataset) — runs
// the separate, pre-existing BlockFORFPEEncoder<int64_t> (its own hardcoded
// per-block FOR + FPE combo, single block size governs both stages) directly
// on each dataset's raw values, to compare its auto-selected block size and
// compression ratio against the two-stage CascadingFOREncoder +
// BlockFrequencyPartitionEncoder pipeline at the matching frame size.
//
// Datasets: three synthetic generators (random uniform, strictly increasing,
// synthetic Snowflake IDs) plus one real dataset — actual Twitter Snowflake
// tweet IDs from Datasets/TwitterSnowflake/tweet_ids.parquet (see
// create_twitter_snowflake_parquet.py in that directory), read via
// ParquetColumnGenerator, same path convention as Benchmarks/heatmap_benchmark.cpp.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "encoders/BlockFORFPEEncoder.hpp"
#include "encoders/BlockFrequencyPartitionEncoder.hpp"
#include "encoders/CascadingFOREncoder.hpp"
#include "encoders/DeltaPrepassEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/analysis/CascadeCardinalityAnalyzer.hpp"
#include "generators/CommonGenerators.hpp"
#include "generators/ParquetColumnGenerator.hpp"
#include "generators/RandomMinMaxGenerator.hpp"
#include "generators/SnowflakeIDGenerator.hpp"
#include "generators/StrictlyIncreasingMinMaxGenerator.hpp"
#include "reorderers/ReorderingCodec.hpp"
#include "reorderers/ReorderingType.hpp"
#include "reorderers/SortReorderer.hpp"

using namespace encodings::encoders;
using namespace encodings::encoders::analysis;
using namespace encodings::datagen;
using namespace encodings::reorderers;

namespace {

constexpr size_t kSampleCount = 200000;
// Fixed reference-cascade schedule reused across both sweeps, so every row
// also reports reference-stream cardinality without exploding the sweep's
// own parameter space (which is scoped to the residual stream's depth /
// starting frame size, per the experiment's purpose).
constexpr size_t kReferenceLeafFrame = 8;

// Shared sweep grid — both the cardinality sweep and the BlockFPE compression
// sweep iterate over exactly these configs, so their two CSVs line up row-for-row.
constexpr size_t kPlainFrameSizes[] = {
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
};
constexpr size_t kCascadeStartFrameSizes[] = {1024, 4096, 16384};
constexpr size_t kCascadeDepths[] = {1, 2, 3, 4};
constexpr size_t kCascadeShrinkFactor = 4; // outermost frame is largest; each level shrinks by this factor
constexpr size_t kCascadeMinFrameSize = 4;

// Deepest-level reference policies swept in the BlockFPE experiment. Only the
// deepest level's policy is varied — intermediate levels' policy is provably
// inert for residual values (telescoping identity), so sweeping them would be
// wasted work; see CascadingFOREncoder.hpp.
constexpr FORReferencePolicy kDeepestPolicies[] = {
    FORReferencePolicy::MIN, FORReferencePolicy::FIRST, FORReferencePolicy::MID
};

const char* policyName(FORReferencePolicy p) {
    switch (p) {
        case FORReferencePolicy::MIN:   return "MIN";
        case FORReferencePolicy::FIRST: return "FIRST";
        case FORReferencePolicy::MID:   return "MID";
        case FORReferencePolicy::PREV:  return "PREV"; // not in kDeepestPolicies (see its doc) -- case kept for -Wswitch completeness
    }
    return "?";
}

// Builds the residual frame-size schedule for a given (start, depth) pair,
// shared by both sweeps so "cascading_for" configs are identical in both CSVs.
// Returns CascadeLevelConfig directly (rather than a plain size_t list) so it
// can be assigned straight into CascadingFORConfig::residualSchedule.
std::vector<CascadeLevelConfig> buildCascadeSchedule(size_t start, size_t depth) {
    std::vector<CascadeLevelConfig> schedule;
    schedule.reserve(depth);
    size_t fs = start;
    for (size_t level = 0; level < depth; ++level) {
        schedule.push_back({std::max(fs, kCascadeMinFrameSize)});
        fs = std::max(fs / kCascadeShrinkFactor, kCascadeMinFrameSize);
    }
    return schedule;
}

struct Dataset {
    std::string name;
    std::vector<int64_t> values;
};

std::vector<Dataset> buildDatasets() {
    std::vector<Dataset> out;

    {
        RandomMinMaxGenerator<int64_t> gen(0, 1'000'000, /*seed=*/1);
        out.push_back({"RandomUniform_0_1e6", gen.generate(kSampleCount)});
    }
    {
        StrictlyIncreasingMinMaxGenerator<int64_t> gen(
            0, 500'000'000, /*minIncrement=*/1, /*maxIncrement=*/200, /*seed=*/2);
        out.push_back({"StrictlyIncreasing_step1_200", gen.generate(kSampleCount)});
    }
    {
        SnowflakeIDGenerator<int64_t> gen(INSTAGRAM_SNOWFLAKE_CONFIG, /*numMachines=*/64,
                                          /*seed=*/3, /*tsFillFraction=*/0.5);
        out.push_back({"SnowflakeID_Instagram", gen.generate(kSampleCount)});
    }
    {
        // Real Twitter Snowflake IDs (see Datasets/TwitterSnowflake/create_twitter_snowflake_parquet.py),
        // same path convention as Benchmarks/heatmap_benchmark.cpp etc.
        const std::filesystem::path parquetPath =
            "/home/david/Documents/PhD/symbol-store/MetaNimbleProject"
            "/EncodingsPlayground/Datasets/TwitterSnowflake/tweet_ids.parquet";
        encodings::generators::ParquetColumnGenerator<int64_t> gen(parquetPath, "tweet_id");
        out.push_back({"TwitterSnowflake_tweet_id", gen.generate(kSampleCount)});
    }

    return out;
}

CascadingFORConfig makeConfig(const std::vector<CascadeLevelConfig>& residualSchedule) {
    CascadingFORConfig cfg;
    cfg.residualSchedule = residualSchedule;
    cfg.referenceSchedule = { {kReferenceLeafFrame} };
    return cfg;
}

void writeRow(std::ofstream& csv, const std::string& sweep, const std::string& dataset,
              size_t startFrameSize, size_t depth, const CascadeLevelStats& s) {
    csv << sweep << ','
        << dataset << ','
        << startFrameSize << ','
        << depth << ','
        << (s.role == CascadeStreamRole::Residual ? "Residual" : "Reference") << ','
        << s.levelIndex << ','
        << (s.parentResidualLevel ? std::to_string(*s.parentResidualLevel) : "") << ','
        << s.frameSize << ','
        << s.numFrames << ','
        << s.numElements << ','
        << s.globalExactDistinct << ','
        << (s.globalDistinctCapped ? "1" : "0") << ','
        << s.globalHllEstimate << ','
        << s.intraFrameDistinctMin << ','
        << s.intraFrameDistinctMax << ','
        << s.intraFrameDistinctMean << ','
        << s.intraFrameDistinctMedian << ','
        << static_cast<int>(s.maxBitWidth) << ','
        << s.avgBitWidth << '\n';
}

void runPlainForDepth1Sweep(std::ofstream& csv, const std::vector<Dataset>& datasets) {
    for (const auto& ds : datasets) {
        for (size_t fs : kPlainFrameSizes) {
            const auto cfg = makeConfig({CascadeLevelConfig{fs}});
            const auto stats = analyzeCascade(std::span<const int64_t>(ds.values), cfg);
            for (const auto& s : stats) {
                writeRow(csv, "plain_for_depth1", ds.name, fs, 1, s);
            }
        }
        std::cout << "  plain_for_depth1: " << ds.name << " done\n";
    }
}

void runCascadingForSweep(std::ofstream& csv, const std::vector<Dataset>& datasets) {
    for (const auto& ds : datasets) {
        for (size_t start : kCascadeStartFrameSizes) {
            for (size_t depth : kCascadeDepths) {
                const auto schedule = buildCascadeSchedule(start, depth);
                const auto cfg = makeConfig(schedule);
                const auto stats = analyzeCascade(std::span<const int64_t>(ds.values), cfg);
                for (const auto& s : stats) {
                    writeRow(csv, "cascading_for", ds.name, start, depth, s);
                }
            }
        }
        std::cout << "  cascading_for: " << ds.name << " done\n";
    }
}

// ---------------------------------------------------------------------------
// BlockFPE-on-deepest-residuals compression experiment
// ---------------------------------------------------------------------------

void writeBlockFpeRow(std::ofstream& csv, const std::string& sweep, const std::string& dataset,
                      size_t startFrameSize, size_t depth, const std::vector<CascadeLevelConfig>& schedule,
                      const std::vector<int64_t>& rawData) {
    // referenceSchedule={}: reference arrays (at every residual-cascade level)
    // are leaf-encoded directly, not cascaded further — this is what makes it
    // "FOR followed by BlockFPE" rather than a deeper recursive scheme. Both
    // leaves are the same bitPackFallback=true BlockFrequencyPartitionEncoder
    // used for the residual stream, so every byte the real data needs —
    // residuals AND all reference arrays — is actually produced and counted.
    CascadingFORConfig cfg;
    cfg.residualSchedule = schedule;
    cfg.referenceSchedule = {};
    cfg.residualLeafEncoder  = std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true);
    cfg.referenceLeafEncoder = std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true);

    CascadingFOREncoder<int64_t> enc(cfg);
    const auto encoded = enc.encode(std::span<const int64_t>(rawData));
    const auto& meta = encoded.metadata();

    const size_t deepestFrameSize = schedule.empty() ? 0 : schedule.back().frameSize;
    const char* deepestPolicy = schedule.empty() ? "?" : policyName(schedule.back().policy);

    csv << sweep << ','
        << dataset << ','
        << startFrameSize << ','
        << depth << ','
        << deepestFrameSize << ','
        << deepestPolicy << ','
        << rawData.size() << ','
        << meta.uncompressedSize << ','
        << meta.compressedSize << ','
        << meta.compressionRatio() << '\n';
}

void runBlockFpeSweep(std::ofstream& csv, const std::vector<Dataset>& datasets) {
    for (const auto& ds : datasets) {
        for (size_t fs : kPlainFrameSizes) {
            for (FORReferencePolicy policy : kDeepestPolicies) {
                const std::vector<CascadeLevelConfig> schedule = {{fs, policy}};
                writeBlockFpeRow(csv, "plain_for_depth1", ds.name, fs, 1, schedule, ds.values);
            }
        }
        for (size_t start : kCascadeStartFrameSizes) {
            for (size_t depth : kCascadeDepths) {
                for (FORReferencePolicy policy : kDeepestPolicies) {
                    // Intermediate levels stay MIN (buildCascadeSchedule's default);
                    // only the deepest level's policy varies (see rationale above).
                    auto schedule = buildCascadeSchedule(start, depth);
                    schedule.back().policy = policy;
                    writeBlockFpeRow(csv, "cascading_for", ds.name, start, depth, schedule, ds.values);
                }
            }
        }
        std::cout << "  blockfpe_compression: " << ds.name << " done\n";
    }
}

// ---------------------------------------------------------------------------
// Reference-only cascade experiment — tests whether cascading ONLY the
// reference stream (keeping the residual side at a single, depth-1 FOR pass)
// matches or beats cascading the residual stream too. Motivated by the
// corrected Output 2 results: for SnowflakeID/StrictlyIncreasing, the best
// config came from a *residual* cascade even though the telescoping identity
// proves residual values are depth-invariant — the real benefit was the
// *reference* stream getting hierarchically narrowed (more, but narrower,
// reference elements can beat fewer-but-wider ones when the data has
// multi-scale structure). This isolates that effect: does cascading only the
// reference array capture the same benefit without paying for the residual
// cascade's redundant machinery (which cannot change the residual values,
// only adds more reference arrays to store)?
// ---------------------------------------------------------------------------

void writeReferenceOnlyCascadeRow(std::ofstream& csv, const std::string& dataset,
                                   size_t residualFrameSize, size_t refStart, size_t refDepth,
                                   const std::vector<CascadeLevelConfig>& refSchedule,
                                   const std::vector<int64_t>& rawData) {
    CascadingFORConfig cfg;
    cfg.residualSchedule = { {residualFrameSize} }; // depth=1 always -- see rationale above
    cfg.referenceSchedule = refSchedule;            // this is what's under test
    cfg.residualLeafEncoder  = std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true);
    cfg.referenceLeafEncoder = std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true);

    CascadingFOREncoder<int64_t> enc(cfg);
    const auto encoded = enc.encode(std::span<const int64_t>(rawData));
    const auto& meta = encoded.metadata();

    const size_t refDeepestFrameSize = refSchedule.empty() ? 0 : refSchedule.back().frameSize;

    csv << dataset << ','
        << residualFrameSize << ','
        << refStart << ','
        << refDepth << ','
        << refDeepestFrameSize << ','
        << rawData.size() << ','
        << meta.uncompressedSize << ','
        << meta.compressedSize << ','
        << meta.compressionRatio() << '\n';
}

void runReferenceOnlyCascadeSweep(std::ofstream& csv, const std::vector<Dataset>& datasets) {
    for (const auto& ds : datasets) {
        for (size_t residualFs : kPlainFrameSizes) {
            // Baseline: no reference cascading either (refSchedule empty) --
            // included so the reference cascade's own marginal effect is visible.
            writeReferenceOnlyCascadeRow(csv, ds.name, residualFs, 0, 0, {}, ds.values);

            for (size_t refStart : kCascadeStartFrameSizes) {
                for (size_t refDepth : kCascadeDepths) {
                    const auto refSchedule = buildCascadeSchedule(refStart, refDepth);
                    writeReferenceOnlyCascadeRow(csv, ds.name, residualFs, refStart, refDepth, refSchedule, ds.values);
                }
            }
        }
        std::cout << "  reference_only_cascade: " << ds.name << " done\n";
    }
}

// ---------------------------------------------------------------------------
// BlockFORFPEEncoder comparison — an integrated single-block-size FOR+FPE
// encoder, run directly on raw data (no CascadingFOREncoder involved).
// ---------------------------------------------------------------------------

void runBlockForFpeComparison(std::ofstream& csv, const std::vector<Dataset>& datasets) {
    for (const auto& ds : datasets) {
        BlockFORFPEEncoder<int64_t> enc;
        const auto encoded = enc.encode(std::span<const int64_t>(ds.values));
        const auto& meta = encoded.metadata();

        csv << ds.name << ','
            << ds.values.size() << ','
            << meta.customMetadata.at("block_size") << ','
            << meta.customMetadata.at("num_blocks") << ','
            << meta.customMetadata.at("avg_num_tiers") << ','
            << meta.customMetadata.at("avg_tag_bit_width") << ','
            << meta.customMetadata.at("avg_fallback_fraction") << ','
            << meta.uncompressedSize << ','
            << meta.compressedSize << ','
            << meta.compressionRatio() << '\n';
    }
    std::cout << "  blockforfpe_comparison: done\n";
}

// ---------------------------------------------------------------------------
// Delta vs. FOR prepass, with/without sorting — comparison experiment.
//
// Reuses the single already-validated best FOR config (residual depth=1,
// reference stream cascaded via buildCascadeSchedule(1024,4); see Output 4's
// header comment) rather than re-sweeping frame sizes, so this experiment
// stays focused on the genuinely new axes: {FOR, Delta} prepass x
// {BlockFPE tiered, plain bit-packing} leaf x {unsorted, sorted}. Widening to
// a frame-size x sort sweep is a natural follow-up if warranted.
//
// Sorting uses ReorderingCodec<int64_t,false> + SortReorderer<int64_t> (default
// FlatBitPacked permutation format). ReorderingCodec::encode() already reports
// meta.compressedSize as the complete total (permutation blob included), so no
// manual size arithmetic is needed here -- this is what keeps this comparison
// honest per the reference-cost lesson learned earlier in this experiment file.
// ---------------------------------------------------------------------------

constexpr size_t kDeltaForResidualFrameSize = 8;
constexpr size_t kDeltaForRefCascadeStart = 1024;
constexpr size_t kDeltaForRefCascadeDepth = 4;

void writeDeltaForSortingRow(std::ofstream& csv, const std::string& dataset,
                             const std::string& prepass, const std::string& leaf, bool sorted,
                             const encodings::EncodingMetadata& meta) {
    const double permBytes = sorted ? std::stod(meta.customMetadata.at("permutation_bytes")) : 0.0;
    const double permPct   = sorted ? std::stod(meta.customMetadata.at("permutation_pct_of_encoded")) : 0.0;

    csv << dataset << ','
        << prepass << ','
        << leaf << ','
        << (sorted ? "1" : "0") << ','
        << meta.elementCount << ','
        << meta.uncompressedSize << ','
        << meta.compressedSize << ','
        << meta.compressionRatio() << ','
        << permBytes << ','
        << permPct << '\n';
}

void runDeltaForSortingComparison(std::ofstream& csv, const std::vector<Dataset>& datasets) {
    for (const auto& ds : datasets) {
        struct NamedCodec {
            std::string prepass;
            std::string leaf;
            std::shared_ptr<encodings::Codec<int64_t, uint8_t>> codec;
        };

        CascadingFORConfig forSchedule;
        forSchedule.residualSchedule = { {kDeltaForResidualFrameSize} };
        forSchedule.referenceSchedule = buildCascadeSchedule(kDeltaForRefCascadeStart, kDeltaForRefCascadeDepth);

        std::vector<NamedCodec> baseCodecs;

        {
            CascadingFORConfig cfg = forSchedule;
            cfg.residualLeafEncoder  = std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true);
            cfg.referenceLeafEncoder = std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true);
            baseCodecs.push_back({"FOR", "BlockFPE", std::make_shared<CascadingFOREncoder<int64_t>>(cfg)});
        }
        {
            CascadingFORConfig cfg = forSchedule;
            cfg.residualLeafEncoder  = std::make_shared<RawBitPackedEncoder<int64_t>>();
            cfg.referenceLeafEncoder = std::make_shared<RawBitPackedEncoder<int64_t>>();
            baseCodecs.push_back({"FOR", "RawBitPacked", std::make_shared<CascadingFOREncoder<int64_t>>(cfg)});
        }
        {
            DeltaPrepassConfig cfg{std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true)};
            baseCodecs.push_back({"Delta", "BlockFPE", std::make_shared<DeltaPrepassEncoder<int64_t>>(cfg)});
        }
        {
            DeltaPrepassConfig cfg; // default leaf: RawBitPackedEncoder<int64_t>
            baseCodecs.push_back({"Delta", "RawBitPacked", std::make_shared<DeltaPrepassEncoder<int64_t>>(cfg)});
        }

        for (const auto& nc : baseCodecs) {
            {
                auto encoded = nc.codec->encode(std::span<const int64_t>(ds.values));
                writeDeltaForSortingRow(csv, ds.name, nc.prepass, nc.leaf, /*sorted=*/false, encoded.metadata());
            }
            {
                ReorderingCodec<int64_t, false> sortedCodec(
                    std::make_shared<SortReorderer<int64_t>>(), nc.codec, ReorderingType::Sort);
                auto encoded = sortedCodec.encode(std::span<const int64_t>(ds.values));
                writeDeltaForSortingRow(csv, ds.name, nc.prepass, nc.leaf, /*sorted=*/true, encoded.metadata());
            }
        }

        std::cout << "  delta_for_sorting_comparison: " << ds.name << " done\n";
    }
}

// ---------------------------------------------------------------------------
// Plain-FOR (no cascading) global cardinality decomposition: residual vs.
// reference vs. total, as a function of frame size. For a purely
// monotonically increasing sequence, this is provable: a frame of size `fs`
// always contains exactly the residuals {0,...,fs-1} (global residual
// cardinality = fs, independent of N), while every frame's minimum is
// distinct (global reference cardinality = numFrames = N/fs). So total
// global cardinality = fs + N/fs, minimized at fs = sqrt(N) by AM-GM -- a
// genuine U-shape in frame size. Computed directly here (not via
// CascadeCardinalityAnalyzer::analyzeCascade) because analyzeReferenceLevel
// emits no stats at all for an empty referenceSchedule (the raw per-frame
// reference array is "stored directly, nothing further to report" -- see
// CascadeCardinalityAnalyzer.hpp), so there's no existing path to the raw,
// unmodified reference array's own exact cardinality. Uses a plain
// std::unordered_set for genuinely exact (uncapped, non-estimated) counts,
// deliberately not MetricCollector/HLL.
// ---------------------------------------------------------------------------

void runForCardinalityDecompositionSweep(std::ofstream& csv, const std::vector<Dataset>& datasets) {
    for (const auto& ds : datasets) {
        const size_t N = ds.values.size();

        // Delta encoding has no frame-size parameter, so its cardinality is a
        // single constant per dataset -- repeated on every row so it plots as
        // a flat reference line alongside the frame-size-varying FOR curves.
        std::unordered_set<int64_t> deltaSet;
        for (size_t i = 1; i < N; ++i) deltaSet.insert(ds.values[i] - ds.values[i - 1]);
        const size_t deltaExactDistinct = deltaSet.size();

        for (size_t fs : kPlainFrameSizes) {
            const size_t numFrames = (N + fs - 1) / fs;
            std::unordered_set<int64_t> residualSet, referenceSet;
            for (size_t f = 0; f < numFrames; ++f) {
                const size_t lo = f * fs;
                const size_t hi = std::min(lo + fs, N);
                const int64_t ref = computeCascadeFrameReference(
                    std::span<const int64_t>(ds.values), lo, hi, FORReferencePolicy::MIN);
                referenceSet.insert(ref);
                for (size_t i = lo; i < hi; ++i) residualSet.insert(ds.values[i] - ref);
            }
            csv << ds.name << ',' << fs << ',' << numFrames << ',' << N << ','
                << residualSet.size() << ',' << referenceSet.size() << ','
                << (residualSet.size() + referenceSet.size()) << ','
                << deltaExactDistinct << '\n';
        }
        std::cout << "  for_cardinality_decomposition: " << ds.name << " done\n";
    }
}

// ---------------------------------------------------------------------------
// Bit-range cardinality heatmap: extends the cardinality decomposition above
// across every contiguous bit-range extractable from a 64-bit value --
// startBit in [0,64), width in [1, 64-startBit], ~2080 valid (startBit,width)
// combinations. For each bit-range, extract the corresponding sub-value from
// every element (uv >> startBit) & mask, then run the SAME plain-FOR
// residual/reference/total exact-cardinality sweep as above across
// kPlainFrameSizes, and record only the minimum total cardinality achieved
// and the frame size that achieves it -- a per-bit-range summary rather than
// the full per-frame-size detail, since the deliverable is a heatmap over
// (startBit, width), not ~2080 individual frame-size plots. Uses
// ankerl::unordered_dense::set (already a transitive dependency via
// CascadeCardinalityAnalyzer.hpp) rather than std::unordered_set: this sweep
// is ~80x larger than the one above (2080 bit-ranges x 13 frame sizes vs. 2
// datasets x 13), so insert throughput actually matters here.
//
// Also records rawExactDistinct: the bit-range's own distinct-value count
// with NO framing at all (a single global reference is a bijective shift, so
// it never changes distinct-value count -- this is just the plain distinct
// count of `extracted`). Comparing minTotalExactDistinct against this raw
// baseline answers "does FOR actually help this bit-range, or does the
// reference stream's overhead outweigh whatever residual reduction it gets" --
// for very narrow bit-ranges (already maximally compact) framing can make
// total cardinality WORSE by adding reference-stream overhead on top of an
// already-tiny residual range.
//
// Also records deltaExactDistinct: the distinct count of first-order
// consecutive deltas (extracted[i]-extracted[i-1], no frame parameter at all,
// unlike FOR) for the same bit-range -- a second, independent comparison
// point against rawExactDistinct, letting standard delta encoding's effect
// be plotted on equal footing against FOR's.
// ---------------------------------------------------------------------------

void runBitRangeCardinalityHeatmapSweep(std::ofstream& csv, const Dataset& ds) {
    const size_t N = ds.values.size();
    std::vector<int64_t> extracted(N);

    for (size_t startBit = 0; startBit < 64; ++startBit) {
        const size_t maxWidth = 64 - startBit;
        for (size_t width = 1; width <= maxWidth; ++width) {
            const uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1ULL);
            for (size_t i = 0; i < N; ++i) {
                const uint64_t uv = static_cast<uint64_t>(ds.values[i]);
                extracted[i] = static_cast<int64_t>((uv >> startBit) & mask);
            }

            ankerl::unordered_dense::set<int64_t> rawSet(extracted.begin(), extracted.end());
            const size_t rawExactDistinct = rawSet.size();

            ankerl::unordered_dense::set<int64_t> deltaSet;
            for (size_t i = 1; i < N; ++i) deltaSet.insert(extracted[i] - extracted[i - 1]);
            const size_t deltaExactDistinct = deltaSet.size();

            size_t bestFrameSize = 0;
            size_t minTotal = std::numeric_limits<size_t>::max();
            for (size_t fs : kPlainFrameSizes) {
                const size_t numFrames = (N + fs - 1) / fs;
                ankerl::unordered_dense::set<int64_t> residualSet, referenceSet;
                for (size_t f = 0; f < numFrames; ++f) {
                    const size_t lo = f * fs;
                    const size_t hi = std::min(lo + fs, N);
                    const int64_t ref = computeCascadeFrameReference(
                        std::span<const int64_t>(extracted), lo, hi, FORReferencePolicy::MIN);
                    referenceSet.insert(ref);
                    for (size_t i = lo; i < hi; ++i) residualSet.insert(extracted[i] - ref);
                }
                const size_t total = residualSet.size() + referenceSet.size();
                if (total < minTotal) {
                    minTotal = total;
                    bestFrameSize = fs;
                }
            }

            csv << ds.name << ',' << startBit << ',' << width << ',' << bestFrameSize << ',' << minTotal << ','
                << rawExactDistinct << ',' << deltaExactDistinct << '\n';
        }
        std::cout << "  bitrange_cardinality_heatmap: startBit=" << startBit << " done\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    // Makes BlockFORFPEEncoder populate avg_num_tiers/avg_tag_bit_width/
    // avg_fallback_fraction in its metadata (see BlockFORFPEEncoder.hpp's
    // statsEnabled()) without requiring the caller to set this externally.
    setenv("BLOCKFORFPE_STATS", "1", /*overwrite=*/1);

    const std::string outPath            = argc > 1 ? argv[1] : "cascade_cardinality_sweep.csv";
    const std::string blockFpeOutPath    = argc > 2 ? argv[2] : "cascade_blockfpe_compression.csv";
    const std::string blockForFpeOutPath = argc > 3 ? argv[3] : "cascade_blockforfpe_comparison.csv";
    const std::string referenceOnlyOutPath = argc > 4 ? argv[4] : "cascade_reference_only_compression.csv";
    const std::string deltaForSortingOutPath = argc > 5 ? argv[5] : "cascade_delta_for_sorting_comparison.csv";
    const std::string cardinalityDecompositionOutPath =
        argc > 6 ? argv[6] : "cascade_for_cardinality_decomposition.csv";
    const std::string bitRangeHeatmapOutPath =
        argc > 7 ? argv[7] : "cascade_bitrange_cardinality_heatmap.csv";

    std::ofstream csv(outPath);
    if (!csv) {
        std::cerr << "Failed to open output file: " << outPath << "\n";
        return 1;
    }
    csv << "sweep,dataset,startFrameSize,depth,role,levelIndex,parentResidualLevel,"
           "frameSize,numFrames,numElements,globalExactDistinct,globalDistinctCapped,"
           "globalHllEstimate,intraFrameDistinctMin,intraFrameDistinctMax,"
           "intraFrameDistinctMean,intraFrameDistinctMedian,maxBitWidth,avgBitWidth\n";

    std::ofstream blockFpeCsv(blockFpeOutPath);
    if (!blockFpeCsv) {
        std::cerr << "Failed to open output file: " << blockFpeOutPath << "\n";
        return 1;
    }
    // No blockSize/numBlocks column here: the real end-to-end scheme may pick
    // a different BlockFPE block size for the residual stream and for each
    // cascade level's own reference array, so there's no longer one
    // unambiguous "the block size" to report per row (unlike the integrated
    // BlockFORFPEEncoder in Output 3, which always has exactly one).
    blockFpeCsv << "sweep,dataset,startFrameSize,depth,deepestFrameSize,deepestPolicy,elements,"
                   "uncompressedBytes,compressedBytes,compressionRatio\n";

    std::ofstream blockForFpeCsv(blockForFpeOutPath);
    if (!blockForFpeCsv) {
        std::cerr << "Failed to open output file: " << blockForFpeOutPath << "\n";
        return 1;
    }
    blockForFpeCsv << "dataset,elements,blockSize,numBlocks,avgNumTiers,avgTagBitWidth,"
                      "avgFallbackFraction,uncompressedBytes,compressedBytes,compressionRatio\n";

    std::ofstream referenceOnlyCsv(referenceOnlyOutPath);
    if (!referenceOnlyCsv) {
        std::cerr << "Failed to open output file: " << referenceOnlyOutPath << "\n";
        return 1;
    }
    referenceOnlyCsv << "dataset,residualFrameSize,refStart,refDepth,refDeepestFrameSize,elements,"
                        "uncompressedBytes,compressedBytes,compressionRatio\n";

    std::ofstream deltaForSortingCsv(deltaForSortingOutPath);
    if (!deltaForSortingCsv) {
        std::cerr << "Failed to open output file: " << deltaForSortingOutPath << "\n";
        return 1;
    }
    deltaForSortingCsv << "dataset,prepass,leaf,sorted,elements,uncompressedBytes,compressedBytes,"
                          "compressionRatio,permutationBytes,permutationPctOfEncoded\n";

    std::ofstream cardinalityDecompositionCsv(cardinalityDecompositionOutPath);
    if (!cardinalityDecompositionCsv) {
        std::cerr << "Failed to open output file: " << cardinalityDecompositionOutPath << "\n";
        return 1;
    }
    cardinalityDecompositionCsv << "dataset,frameSize,numFrames,numElements,"
                                    "residualExactDistinct,referenceExactDistinct,totalExactDistinct,"
                                    "deltaExactDistinct\n";

    std::ofstream bitRangeHeatmapCsv(bitRangeHeatmapOutPath);
    if (!bitRangeHeatmapCsv) {
        std::cerr << "Failed to open output file: " << bitRangeHeatmapOutPath << "\n";
        return 1;
    }
    bitRangeHeatmapCsv << "dataset,startBit,width,bestFrameSize,minTotalExactDistinct,rawExactDistinct,"
                          "deltaExactDistinct\n";

    const auto datasets = buildDatasets();

    std::cout << "Running plain_for_depth1 sweep...\n";
    runPlainForDepth1Sweep(csv, datasets);

    std::cout << "Running cascading_for sweep...\n";
    runCascadingForSweep(csv, datasets);

    std::cout << "Running BlockFPE-on-deepest-residuals compression sweep...\n";
    runBlockFpeSweep(blockFpeCsv, datasets);

    std::cout << "Running BlockFORFPEEncoder comparison...\n";
    runBlockForFpeComparison(blockForFpeCsv, datasets);

    std::cout << "Running reference-only cascade sweep...\n";
    runReferenceOnlyCascadeSweep(referenceOnlyCsv, datasets);

    std::cout << "Running Delta vs FOR prepass, with/without sorting comparison...\n";
    runDeltaForSortingComparison(deltaForSortingCsv, datasets);

    std::cout << "Running plain-FOR cardinality decomposition sweep...\n";
    {
        std::vector<Dataset> decompositionDatasets;
        decompositionDatasets.push_back(
            {"Monotonic_step1", encodings::generators::SequentialGenerator<int64_t>(0, 1).generate(kSampleCount)});
        for (const auto& ds : datasets) {
            if (ds.name == "TwitterSnowflake_tweet_id") decompositionDatasets.push_back(ds);
        }
        runForCardinalityDecompositionSweep(cardinalityDecompositionCsv, decompositionDatasets);
    }

    std::cout << "Running bit-range cardinality heatmap sweep (TwitterSnowflake_tweet_id only)...\n";
    for (const auto& ds : datasets) {
        if (ds.name == "TwitterSnowflake_tweet_id") runBitRangeCardinalityHeatmapSweep(bitRangeHeatmapCsv, ds);
    }

    std::cout << "Wrote " << outPath << "\n";
    std::cout << "Wrote " << blockFpeOutPath << "\n";
    std::cout << "Wrote " << blockForFpeOutPath << "\n";
    std::cout << "Wrote " << referenceOnlyOutPath << "\n";
    std::cout << "Wrote " << deltaForSortingOutPath << "\n";
    std::cout << "Wrote " << cardinalityDecompositionOutPath << "\n";
    std::cout << "Wrote " << bitRangeHeatmapOutPath << "\n";
    return 0;
}
