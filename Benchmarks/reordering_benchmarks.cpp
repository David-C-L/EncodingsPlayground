// Reordering benchmarks: compare (reorder + codec) vs plain codec on multiple
// datasets and inner codecs to quantify the compression vs. access-overhead
// trade-off of each reordering technique.

#include "benchmark/BenchmarkRunner.hpp"
#include "benchmark/BenchmarkOutput.hpp"
#include "generators/CommonGenerators.hpp"
#include "generators/ParquetColumnGenerator.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/DeltaCodec.hpp"
#include "encoders/ZstdEncoder.hpp"
#include "encoders/LZ4Encoder.hpp"
#include "encoders/SubIntSplitEncoder.hpp"
#include "encodings/ChainedCodec.hpp"

#include "reorderers/ReorderingCodec.hpp"
#include "reorderers/SortReorderer.hpp"
#include "reorderers/GrayCodeReorderer.hpp"
#include "reorderers/WindowedSortReorderer.hpp"
#include "reorderers/MTFReorderer.hpp"
#include "reorderers/BWTReorderer.hpp"
#include "reorderers/BitShuffleCodec.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace encodings;
using namespace encodings::benchmark;
using namespace encodings::generators;
using namespace encodings::encoders;
using namespace encodings::reorderers;

// ---------------------------------------------------------------------------
// Benchmark config
// ---------------------------------------------------------------------------

BenchmarkConfig makeConfig(const std::string& tag) {
    BenchmarkConfig cfg;
    cfg.dataSizes           = {10'000'000};  // moderate sizes (BWT is O(W^2 log W))
    cfg.iterations          = 2;
    cfg.warmupRuns          = 1;
    cfg.randomAccessSamples = 10;
    cfg.stridedAccessSamples = 10;
    cfg.stride              = 128;
    cfg.rangeQueryCount     = 10;
    cfg.rangeSizes          = {65536};
    cfg.validateCorrectness = true;
    cfg.validateRandomAccess = true;
    cfg.verboseOutput       = true;
    cfg.outputPath = "/home/david/Documents/PhD/symbol-store/MetaNimbleProject"
                     "/EncodingsPlayground/Benchmarks/results/reordering/" + tag;
    return cfg;
}

// ---------------------------------------------------------------------------
// Codec factories
// ---------------------------------------------------------------------------

using Codec64 = encodings::Codec<int64_t, uint8_t>;

std::shared_ptr<Codec64> makeAutoSubIntSplit() {
    return makeDefaultAutoSubIntSplitEncoder<int64_t>(encoders::BitSplitOrder::LSB_TO_MSB,
                                                       /*extended=*/false,
                                                       /*exclusive=*/false,
                                                       /*logging=*/false);
}

// Reordering codecs: Sort | innerCodec (profiling always on for research)
std::shared_ptr<Codec64> makeSort(std::shared_ptr<Codec64> inner,
                                   std::string name = "") {
    auto r = std::make_shared<SortReorderer<int64_t>>();
    return makeReorderingCodec<int64_t, /*EnableProfiling=*/true>(
        r, inner, ReorderingType::Sort,
        name.empty() ? "Sort|" + inner->name() : name);
}

// std::shared_ptr<Codec64> makeGray(std::shared_ptr<Codec64> inner,
//                                    std::string name = "") {
//     auto r = std::make_shared<GrayCodeReorderer<int64_t>>();
//     return makeReorderingCodec<int64_t>(r, inner, ReorderingType::GrayCode,
//                                         name.empty() ? "Gray|" + inner->name() : name);
// }

// EnableProfiling template parameter: set true in sweep variants so the
// reordering-layer timing hooks are active and appear in the JSON results.
template <size_t W, bool P = true>
std::shared_ptr<Codec64> makeWindowedSort(std::shared_ptr<Codec64> inner,
                                          std::string name = "") {
    auto r = std::make_shared<WindowedSortReorderer<int64_t, W>>();
    return makeReorderingCodec<int64_t, P>(r, inner, ReorderingType::WindowedSort,
                                           name.empty() ? "WSort|" + inner->name() : name);
}

template <size_t W, bool P = true>
std::shared_ptr<Codec64> makeMTF(std::shared_ptr<Codec64> inner,
                                  std::string name = "") {
    auto r = std::make_shared<MTFReorderer<int64_t, W>>();
    return makeReorderingCodec<int64_t, P>(r, inner, ReorderingType::MTF,
                                           name.empty() ? "MTF|" + inner->name() : name);
}

template <size_t W, bool P = true>
std::shared_ptr<Codec64> makeBWT(std::shared_ptr<Codec64> inner,
                                  std::string name = "") {
    auto r = std::make_shared<BWTReorderer<int64_t, W>>();
    return makeReorderingCodec<int64_t, P>(r, inner, ReorderingType::BWT,
                                           name.empty() ? "BWT|" + inner->name() : name);
}

// ---------------------------------------------------------------------------
// Compile-time window-parameter sweeps (profiling enabled)
// W is a template argument — runtime loops cannot instantiate template parameters.
// Use fold expressions over explicit compile-time size lists instead.
// ---------------------------------------------------------------------------

// Registers WSort<W>|AutoSubIntSplit for each W in the pack, with profiling on.
template <size_t... Ws, typename AddFn>
void sweepWindowedSort(AddFn&& add) {
    ((add("WSort" + std::to_string(Ws) + "|AutoSubIntSplit",
          makeWindowedSort<Ws, /*EnableProfiling=*/true>(makeAutoSubIntSplit()))), ...);
}

// Registers BWT<W>|AutoSubIntSplit for each W in the pack, with profiling on.
// Keep W small (≤512) — forward BWT is O(W² log W) per window.
template <size_t... Ws, typename AddFn>
void sweepBWT(AddFn&& add) {
    ((add("BWT" + std::to_string(Ws) + "|AutoSubIntSplit",
          makeBWT<Ws, /*EnableProfiling=*/true>(makeAutoSubIntSplit()))), ...);
}

} // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== Reordering Benchmarks ===\n\n";

    // ------------------------------------------------------------------
    // Datasets
    // ------------------------------------------------------------------

    std::vector<std::pair<std::string, std::shared_ptr<encodings::datagen::DataGenerator<int64_t>>>> datasets;
    std::string columnNameTweets = "tweet_id";
    std::filesystem::path dataDirTweets = "/home/david/Documents/PhD/symbol-store/EncodingsPlayground/Datasets/TwitterSnowflake/tweet_ids.parquet";
    datasets.emplace_back("Twitter Snowflake IDs",
        std::make_shared<ParquetColumnGenerator<int64_t>>(dataDirTweets, columnNameTweets));
    // datasets.emplace_back("Uniform",
    //     std::make_shared<UniformRandomGenerator<int64_t>>(0, 1'000'000));
    // datasets.emplace_back("Zipfian1k",
    //     std::make_shared<ZipfianGenerator<int64_t>>(1000, 1.5));
    // datasets.emplace_back("Repetitive",
    //     std::make_shared<RepetitiveGenerator<int64_t>>(50, 0, 1000));
    // datasets.emplace_back("NearlySorted",
    //     std::make_shared<NearlySortedGenerator<int64_t>>(0, 1, 0.05));

    // ------------------------------------------------------------------
    // Inner codecs (used both standalone and as the inner layer of reorderers)
    // ------------------------------------------------------------------

    auto raw = std::make_shared<RawEncoder<int64_t>>();
    auto rawBitPacked  = std::make_shared<RawBitPackedEncoder<int64_t>>();
    auto rle           = std::make_shared<RunLengthEncoder<int64_t>>();
    auto autoSubInt    = makeAutoSubIntSplit();
    // auto lz4           = std::make_shared<LZ4Encoder<int64_t>>();
    // auto zstd          = std::make_shared<ZstdEncoder<int64_t>>();
    auto bitShuffle    = std::make_shared<BitShuffleCodec<int64_t>>(nullptr);

    // ------------------------------------------------------------------
    // Register all encoder combinations
    // ------------------------------------------------------------------

    std::vector<std::pair<std::string, std::shared_ptr<Codec64>>> encoders;

    auto add = [&](std::string name, std::shared_ptr<Codec64> enc) {
        encoders.emplace_back(std::move(name), std::move(enc));
    };

    // Baselines
    add("Raw",                     raw);
    add("RawBitPacked",            rawBitPacked);
    // add("RLE",                     rle);
    add("AutoSubIntSplit",         autoSubInt);
    // add("LZ4",                     lz4);
    // add("Zstd",                    zstd);
    add("BitShuffle",              bitShuffle);

    // Sort + inner codec
    // add("Sort|RawBitPacked",       makeSort(rawBitPacked));
    // add("Sort|RLE",                makeSort(rle));
    add("Sort|AutoSubIntSplit",    makeSort(makeAutoSubIntSplit()));
    // add("Sort|LZ4",                makeSort(lz4));

    // GrayCode + inner codec
    // add("Gray|RawBitPacked",       makeGray(rawBitPacked));
    // add("Gray|AutoSubIntSplit",    makeGray(makeAutoSubIntSplit()));

    // WindowedSort sweep: W = 16, 32, 64, 128, 256, 512, 1024, 4096
    // Each entry is "WSort{W}|AutoSubIntSplit".
    sweepWindowedSort<32, 128, 512, 4096, 16384>(add);

    // BWT sweep: W = 16, 32, 64, 128, 256, 512
    // Larger W is impractical (O(W² log W) per window).
    sweepBWT<32, 128, 512, 4096, 16384>(add);

    // ------------------------------------------------------------------
    // Run benchmark
    // ------------------------------------------------------------------

    BenchmarkConfig cfg = makeConfig("all");
    std::filesystem::create_directories(cfg.outputPath);

    BenchmarkRunner<int64_t> runner(cfg);
    for (const auto& [name, enc] : encoders)
        runner.registerEncoder(name, enc);
    for (const auto& [name, gen] : datasets)
        runner.addDataset(name, gen);

    std::cout << "Running " << encoders.size() << " encoders × "
              << datasets.size() << " datasets × "
              << cfg.dataSizes.size() << " sizes = "
              << encoders.size() * datasets.size() * cfg.dataSizes.size()
              << " benchmark runs\n\n";

    auto results = runner.runAll();

    // Print table
    TableFormatter::printSummaryTable(results);

    // Save JSON
    const std::string jsonPath = cfg.outputPath + "/reordering_results.json";
    if (saveBenchmarkResults(results, jsonPath))
        std::cout << "\nResults saved: " << jsonPath << '\n';

    // ------------------------------------------------------------------
    // Permutation index overhead summary (reordering codecs only)
    // ------------------------------------------------------------------
    std::cout << "\n--- Permutation index overhead ---\n";
    std::cout << std::left
              << std::setw(35) << "Encoder"
              << std::setw(25) << "Dataset"
              << std::setw(14) << "perm bytes"
              << std::setw(16) << "% of encoded"
              << std::setw(16) << "% of raw"
              << std::setw(18) << "reorder_enc (ms)"
              << std::setw(20) << "unreorder_dec (ms)"
              << '\n'
              << std::string(144, '-') << '\n';
    for (const auto& r : results.results) {
        const auto& cm = r.metrics.customMetrics;
        if (cm.find("permutation_bytes") == cm.end()) continue;
        auto get = [&](const std::string& k) -> double {
            auto it = cm.find(k); return it != cm.end() ? it->second : 0.0;
        };
        std::cout << std::left
                  << std::setw(35) << r.encoderName.substr(0, 34)
                  << std::setw(25) << r.datasetName.substr(0, 24)
                  << std::setw(14) << static_cast<size_t>(get("permutation_bytes"))
                  << std::setw(16) << std::fixed << std::setprecision(1)
                      << get("permutation_pct_of_encoded") << "%"
                  << std::setw(15) << get("permutation_pct_of_uncompressed") << "%"
                  << std::setw(18) << std::setprecision(2)
                      << get("reorder_encode_time_ns") / 1e6
                  << std::setw(20) << get("unreorder_decode_all_time_ns") / 1e6
                  << '\n';
    }

    // Print highlight: best compression ratio per dataset
    std::cout << "\n--- Best compression ratio per dataset ---\n";
    std::map<std::string, std::pair<std::string, double>> best;
    for (const auto& r : results.results) {
        double ratio = r.metrics.memory.compressionRatio();
        auto& b = best[r.datasetName];
        if (b.second == 0.0 || ratio < b.second) { b.first = r.encoderName; b.second = ratio; }
    }
    for (const auto& [dataset, b] : best)
        std::cout << "  " << std::left << std::setw(20) << dataset
                  << " → " << std::setw(25) << b.first
                  << std::fixed << std::setprecision(4) << b.second << "x\n";

    return 0;
}
