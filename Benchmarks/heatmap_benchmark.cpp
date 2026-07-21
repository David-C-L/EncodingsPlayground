#include "generators/ParquetColumnGenerator.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/DictionaryEncoder.hpp"
#include "encoders/AdaptiveFramedBitPrefixEncoder.hpp"
#include "encoders/BlockFrequencyPartitionEncoder.hpp"
#include "encoders/BlockFSEEncoder.hpp"
#include "encoders/FrequencyPartitionEncoder.hpp"
#include "encoders/AdaptiveDictionaryEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/OpenZLEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"
#include "encoders/SubIntSplitEncoder.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using namespace encodings;
using namespace encodings::generators;
using namespace encodings::encoders;

namespace {

int64_t medianOf(std::vector<int64_t>& v) {
    auto mid = v.begin() + v.size() / 2;
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

struct EncoderEntry {
    std::string name;
    std::shared_ptr<Codec<int64_t>> codec;
    // true  → decodeRange must read the full compressed payload on every call (Zstd, OpenZL)
    // false → decodeRange reads O(B) bytes from the encoded payload (raw/bit-packed/dict/RLE/ABP)
    bool is_sequential;
};

} // namespace

int main() {
    // N: dataset size.  GRID: number of A and B fractions sampled on each axis.
    // The plotter interpolates between the GRID×GRID sample points to produce a
    // continuous heatmap.  GRID=64 gives ~2048 valid cells per encoder.
    constexpr size_t N       = 100'000;
    constexpr int    GRID    = 64;
    constexpr int    WARMUP  = 2;
    constexpr int    MEASURE = 3;

    const std::filesystem::path PARQUET =
        "/home/david/Documents/PhD/symbol-store/MetaNimbleProject"
        "/EncodingsPlayground/Datasets/TwitterSnowflake/tweet_ids.parquet";

    const std::filesystem::path OUT_CSV = "Benchmarks/results/heatmap_benchmark.csv";

    // ── Load data ────────────────────────────────────────────────────────────
    std::cout << "Loading " << N << " Twitter Snowflake IDs from parquet..." << std::endl;
    ParquetColumnGenerator<int64_t> gen(PARQUET, "tweet_id");
    auto data = gen.generate(N);
    std::cout << "Loaded " << data.size() << " elements.\n" << std::endl;

    // ── Register encoders ────────────────────────────────────────────────────
    std::vector<EncoderEntry> encoders;
    encoders.push_back({"Raw",               std::make_shared<RawEncoder<int64_t>>(),                       false});
    encoders.push_back({"RawBitPacked",      std::make_shared<RawBitPackedEncoder<int64_t>>(),              false});
    // encoders.push_back({"Dictionary",        std::make_shared<DictionaryEncoder<int64_t>>(),                false});
    // encoders.push_back({"AdaptiveDictionary", std::make_shared<AdaptiveDictionaryEncoder<int64_t>>(),                false});
    encoders.push_back({"BlockFPE", std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(),                false});
    // encoders.push_back({"BlockFPEGlobalFOR", std::make_shared<BlockFrequencyPartitionEncoder<int64_t, FORPrepass::GlobalFOR>>(),                false});
    // encoders.push_back({"FPE", std::make_shared<FrequencyPartitionEncoder<int64_t, FreqPartIndexType::PerTierBitmaps>>(),                false});
    encoders.push_back({"BlockFORFPE", std::make_shared<BlockFORFPEEncoder<int64_t>>(),                false});
    encoders.push_back({"AdaptiveBitPrefix", std::make_shared<AdaptiveFramedBitPrefixEncoder<int64_t>>(),   false});
    // encoders.push_back({"BlockFSE", std::make_shared<BlockFSEEncoder<int64_t>>(),   false});
    // encoders.push_back({"RLE",               std::make_shared<RunLengthEncoder<int64_t>>(),                 false});
#ifdef HAVE_OPENZL
    encoders.push_back({"OpenZL",            makeOpenZLCodec<int64_t>(),                                    true});
    // encoders.push_back({"OpenZL1024",        makeOpenZLCodec<int64_t, 1024>(),                              true});
#endif
    encoders.push_back({"Zstd",              std::make_shared<ZstdEncoder<int64_t>>(),                      true});
    // encoders.push_back({"Zstd_b1024",        makeZstdEncoder<int64_t, 1024>(),                              false});

    // using D = CostModelDimension;
    // auto cms = [](auto... dims) { CostModelSet s; (s.add(dims), ...); return s; };
    // encoders.push_back({"AutoSIS_C",
    //     makeDefaultAutoSubIntSplitEncoderProf(
    //         BitSplitOrder::LSB_TO_MSB, cms(D::Compression),
    //         false, false, -1, false, true, {.enabled = true}),
    //     false});

    // encoders.push_back({"ManSIS_C",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {2, 20, 32, 7, 3},
    //         {encodings::EncodingType::BitPacking,
    //         encodings::EncodingType::BlockFrequencyPartitionEncoding, 
    //         encodings::EncodingType::RawEncoding,
    //         encodings::EncodingType::RunLengthEncoding,
    //         encodings::EncodingType::BitPacking}),
    //     false});
    // encoders.push_back({"ManSIS_FPE",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {12,12,32,8},
    //         {encodings::EncodingType::FrequencyPartitionEncoding, 
    //         encodings::EncodingType::AdaptiveDictionaryEncoding,
    //         encodings::EncodingType::FrequencyPartitionEncoding,
    //         encodings::EncodingType::RunLengthEncoding}),
    //     false});
    // encoders.push_back({"ManSIS_BFPE",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {12,12,32,8},
    //         {encodings::EncodingType::BlockFrequencyPartitionEncoding, 
    //         encodings::EncodingType::AdaptiveDictionaryEncoding,
    //         encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::RunLengthEncoding}),
    //     false});
    // encoders.push_back({"ManSIS_FPE_Raw",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {12,12,32,8},
    //         {encodings::EncodingType::FrequencyPartitionEncoding, 
    //         encodings::EncodingType::AdaptiveDictionaryEncoding,
    //         encodings::EncodingType::RawEncoding,
    //         encodings::EncodingType::RunLengthEncoding}),
    //     false});
    // encoders.push_back({"ManSIS_BFPE_Raw",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {12,12,32,8},
    //         {encodings::EncodingType::BlockFrequencyPartitionEncoding, 
    //         encodings::EncodingType::AdaptiveDictionaryEncoding,
    //         encodings::EncodingType::RawEncoding,
    //         encodings::EncodingType::RunLengthEncoding}),
    //     false});
    encoders.push_back({"AutoSIS",
        makeSubIntSplitEncoderManual<int64_t>(
            {13,1,8,28,8,6},
            {encodings::EncodingType::BlockFrequencyPartitionEncoding, 
            encodings::EncodingType::BitPacking,
            encodings::EncodingType::BlockFrequencyPartitionEncoding,
            encodings::EncodingType::BitPacking,
            encodings::EncodingType::RunLengthCascadingFOREncoding,
            encodings::EncodingType::RunLengthEncoding}),
        false});
    encoders.push_back({"AutoSIS_Delta",
        makeSubIntSplitEncoderManual<int64_t>(
            {13,8,1,25,17},
            {encodings::EncodingType::BlockFrequencyPartitionEncoding,
            encodings::EncodingType::AdaptiveDictionaryEncoding,
            encodings::EncodingType::BitPacking,
            encodings::EncodingType::BitPacking,
            encodings::EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding}),
        false});
    encoders.push_back({"AutoSIS_DeltaBlockFSE",
        makeSubIntSplitEncoderManual<int64_t>(
            {3,4,5,2,3,4,1,23,4,4,8,3},
            {encodings::EncodingType::BlockFSEEncoding,
            encodings::EncodingType::BlockFSEEncoding,
            encodings::EncodingType::BlockFSEEncoding,
            encodings::EncodingType::BitPacking,
            encodings::EncodingType::BlockFSEEncoding,
            encodings::EncodingType::CascadingFORPrevBlockFSEEncoding,
            encodings::EncodingType::BlockFSEEncoding,
            encodings::EncodingType::BitPacking,
            encodings::EncodingType::CascadingFORPrevBlockFSEEncoding,
            encodings::EncodingType::CascadingFORPrevBlockFSEEncoding,
            encodings::EncodingType::RunLengthCascadingFOREncoding,
            encodings::EncodingType::BitPacking}),
        false});
    encoders.push_back({"AutoSIS_OpenZL",
        makeSubIntSplitEncoderManual<int64_t>(
            {3,9,3,7,32,10},
            {encodings::EncodingType::FSEEncoding,
            encodings::EncodingType::FSEEncoding,
            encodings::EncodingType::BitPacking,
            encodings::EncodingType::OpenZL,
            encodings::EncodingType::OpenZL,
            encodings::EncodingType::OpenZL}),
        false});
    // encoders.push_back({"ManSIS_BFORFPE",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {13,1,8,42},
    //         {encodings::EncodingType::BlockFORFPEEncoding, 
    //         encodings::EncodingType::BitPacking,
    //         encodings::EncodingType::BlockFORFPEEncoding,
    //         encodings::EncodingType::AdaptiveFramedBitPrefix}),
    //     false});
    // encoders.push_back({"ManSIS_GlobalFORBFPE",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {13,1,8,42},
    //         {encodings::EncodingType::BlockFrequencyPartitionFOREncoding,
    //         encodings::EncodingType::BitPacking,
    //         encodings::EncodingType::BlockFrequencyPartitionFOREncoding,
    //         encodings::EncodingType::AdaptiveFramedBitPrefix}),
    //     false});
    // // Segment-3 variants: order-sensitive codecs that beat AFBP on sorted tweet timestamps.
    // // FORSection<512>: per-block FOR with 512-element blocks, pure O(1) decode.
    // encoders.push_back({"ManSIS_FORBitPack",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {13,1,8,42},
    //         {encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::BitPacking,
    //         encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::FrameOfReference}),
    //     false});
    // // AdaptiveFOR: selects best frame size at encode time, O(1) decode.
    // encoders.push_back({"ManSIS_AdaptiveFOR",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {13,1,8,42},
    //         {encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::BitPacking,
    //         encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::AdaptiveFrameOfReference}),
    //     false});
    // // BlockFORFPE: per-block FOR + global FPE tiers, O(kRankSampleStride=64) decode.
    // encoders.push_back({"ManSIS_BlockFORFPE",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {13,1,8,42},
    //         {encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::BitPacking,
    //         encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::BlockFORFPEEncoding}),
    //     false});
    // // BitPack on segment 3 = GlobalFOR+BitPack (RawBitPackedEncoder already does global min subtraction).
    // encoders.push_back({"ManSIS_BitPack",
    //     makeSubIntSplitEncoderManual<int64_t>(
    //         {13,1,8,42},
    //         {encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::BitPacking,
    //         encodings::EncodingType::BlockFrequencyPartitionEncoding,
    //         encodings::EncodingType::BitPacking}),
    //     false});

    // ── Open CSV ─────────────────────────────────────────────────────────────
    std::filesystem::create_directories(OUT_CSV.parent_path());
    std::ofstream csv(OUT_CSV);
    if (!csv) {
        std::cerr << "ERROR: cannot open " << OUT_CSV << " for writing\n";
        return 1;
    }
    csv << "encoding,A_frac,B_frac,time_ns,elem_Meps,input_MBps,compression_ratio,is_sequential\n";
    csv << std::fixed;

    // ── Build fractional grid ─────────────────────────────────────────────────
    // A_frac ∈ { 0, 1/GRID, …, (GRID-1)/GRID }
    // B_frac ∈ { 1/GRID, 2/GRID, …, GRID/GRID }
    // Valid cells: A_frac + B_frac ≤ 1  (~GRID²/2 cells, all in the triangle)
    std::vector<double> a_fracs, b_fracs;
    a_fracs.reserve(GRID);
    b_fracs.reserve(GRID);
    for (int i = 0; i < GRID; ++i) a_fracs.push_back(static_cast<double>(i) / GRID);
    for (int j = 1; j <= GRID; ++j) b_fracs.push_back(static_cast<double>(j) / GRID);

    // ── Sweep ────────────────────────────────────────────────────────────────
    for (auto& entry : encoders) {
        std::cout << "[" << entry.name << "] encoding..." << std::flush;

        auto encoded = entry.codec->encode(std::span<const int64_t>(data.data(), data.size()));
        size_t cs    = encoded.metadata().compressedSize;
        double ratio = static_cast<double>(N * sizeof(int64_t)) / static_cast<double>(cs);

        std::cout << " → " << cs << " B  (ratio " << ratio << "×)\n";
        std::cout << "[" << entry.name << "] sweeping " << GRID << "×" << GRID
                  << " fractional grid..." << std::flush;

        size_t cells = 0;
        for (double a_frac : a_fracs) {
            for (double b_frac : b_fracs) {
                if (a_frac + b_frac > 1.0 + 1e-9) continue;

                size_t A = static_cast<size_t>(std::round(a_frac * static_cast<double>(N)));
                size_t B = std::max<size_t>(1,
                    static_cast<size_t>(std::round(b_frac * static_cast<double>(N))));
                if (A + B > N) B = N - A;
                if (B == 0) continue;

                // Warmup
                for (int w = 0; w < WARMUP; ++w)
                    (void)entry.codec->decodeRange(encoded, A, A + B);

                // Timed runs
                std::vector<int64_t> times;
                times.reserve(MEASURE);
                for (int m = 0; m < MEASURE; ++m) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    (void)entry.codec->decodeRange(encoded, A, A + B);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    times.push_back(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                }

                double t_ns = static_cast<double>(medianOf(times));

                // Throughput in M elements/s: B / t_ns * 1e9 / 1e6 = B / t_ns * 1e3
                double elem_Meps = (t_ns > 0.0) ? (static_cast<double>(B) / t_ns * 1e3) : 0.0;

                // Input-bandwidth model (MB/s):
                //   sequential encoders must read the entire compressed payload every call
                //   random-access encoders read O(B) bytes from the encoded payload
                double input_bytes = entry.is_sequential
                    ? static_cast<double>(cs)
                    : static_cast<double>(B) * static_cast<double>(cs) / static_cast<double>(N);
                double input_MBps = (t_ns > 0.0) ? (input_bytes / t_ns * 1e3) : 0.0;

                csv << entry.name << ","
                    << a_frac << "," << b_frac << ","
                    << t_ns << ","
                    << elem_Meps << ","
                    << input_MBps << ","
                    << ratio << ","
                    << (entry.is_sequential ? 1 : 0) << "\n";
                ++cells;
            }
        }
        std::cout << " " << cells << " cells done.\n";
    }

    csv.close();
    std::cout << "\nResults written to: " << std::filesystem::absolute(OUT_CSV) << std::endl;
    return 0;
}
