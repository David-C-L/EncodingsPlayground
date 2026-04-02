#include "benchmark/BenchmarkRunner.hpp"
#include "benchmark/BenchmarkOutput.hpp"
#include "generators/CommonGenerators.hpp"
#include "generators/ParquetColumnGenerator.hpp"
#include "generators/GeneratorUtils.hpp"
#include "generators/SnowflakeIDGenerator.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/DeltaRunLengthEncoder.hpp"
#include "encoders/SubIntEncoder.hpp"
#include "encoders/FOREncoder.hpp"
#include "encoders/DeltaEncoder.hpp"
#include "encoders/DeltaCodec.hpp"
#include "encoders/DictionaryEncoder.hpp"
#include "encoders/OpenZLEncoder.hpp"
#include "encoders/TriSplitEncoder.hpp"
#include "encoders/SubIntSplitEncoder.hpp"
#include "encoders/VarIntEncoder.hpp"
#include "encoders/DeltaVarIntEncoder.hpp"
#include "encodings/ComposedEncoder.hpp"
#include <iostream>
#include <filesystem>

namespace {

using encodings::benchmark::BenchmarkConfig;

// Helper to build explicit access traces matching the discussed patterns.
// All end indices are exclusive.
struct AccessTrace {
    std::vector<size_t> randomIndices;
    std::vector<size_t> stridedIndices;
    std::vector<std::pair<size_t,size_t>> ranges;
    size_t randomSamples{0};
    size_t stridedSamples{0};
    size_t rangeQueries{0};
    size_t stride{0};
};

AccessTrace makeMostlyFilter(size_t n) {
    AccessTrace t;
    // Keep ~80% of rows with small holes.
    for (size_t i = 0; i < n; i += 1) {
        if ((i / 7) % 5 == 0) continue; // introduce periodic gaps
        t.randomIndices.push_back(i);
        if (t.randomIndices.size() >= 200) break;
    }
    // Strided indices (dense): every 4th up to cap
    for (size_t i = 0; i < n && t.stridedIndices.size() < 50; i += 4) {
        t.stridedIndices.push_back(i);
    }
    // Ranges: several large blocks with small gaps
    size_t block = 65536;
    for (size_t start = 0; start + block <= n && t.ranges.size() < 5; start += block + 8192) {
        t.ranges.emplace_back(start, start + block);
    }
    t.randomSamples = t.randomIndices.size();
    t.stridedSamples = t.stridedIndices.size();
    t.rangeQueries = t.ranges.size();
    t.stride = 4;
    return t;
}

AccessTrace makeMostlySkips(size_t n) {
    AccessTrace t;
    // Touch ~10% spread out: every 16th index, capped.
    for (size_t i = 0; i < n; i += 16) {
        t.randomIndices.push_back(i);
        if (t.randomIndices.size() >= 200) break;
    }
    // Strided: sparse
    for (size_t i = 0; i < n && t.stridedIndices.size() < 50; i += 256) {
        t.stridedIndices.push_back(i);
    }
    // Ranges: small/medium chunks spaced out
    size_t block = 2048;
    for (size_t start = 0; start + block <= n && t.ranges.size() < 5; start += 32768) {
        t.ranges.emplace_back(start, start + block);
    }
    t.randomSamples = t.randomIndices.size();
    t.stridedSamples = t.stridedIndices.size();
    t.rangeQueries = t.ranges.size();
    t.stride = 256;
    return t;
}

AccessTrace makeHalfAndHalf(size_t n) {
    AccessTrace t;
    // Keep about 50%: alternate keep/skip blocks of 8.
    size_t block = 8;
    bool keep = true;
    for (size_t i = 0; i < n; i += block) {
        if (keep) {
            for (size_t j = 0; j < block && i + j < n; ++j) {
                t.randomIndices.push_back(i + j);
                if (t.randomIndices.size() >= 200) break;
            }
        }
        keep = !keep;
        if (t.randomIndices.size() >= 200) break;
    }
    // Strided: moderate density
    for (size_t i = 0; i < n && t.stridedIndices.size() < 50; i += 16) {
        t.stridedIndices.push_back(i);
    }
    // Ranges: mid-sized blocks
    size_t rsize = 16384;
    for (size_t start = 0; start + rsize <= n && t.ranges.size() < 5; start += rsize * 2) {
        t.ranges.emplace_back(start, start + rsize);
    }
    t.randomSamples = t.randomIndices.size();
    t.stridedSamples = t.stridedIndices.size();
    t.rangeQueries = t.ranges.size();
    t.stride = 16;
    return t;
}

BenchmarkConfig baseConfig() {
    BenchmarkConfig config;
    config.dataSizes = {1000000};
    config.iterations = 1;
    config.warmupRuns = 0;
    // Keep very small samples to speed slow codecs like OpenZL.
    config.randomAccessSamples = 20;
    config.stridedAccessSamples = 20;
    config.stride = 128;
    config.rangeQueryCount = 20;
    config.rangeSizes = {65536};
    config.validateCorrectness = true;
    config.validateRandomAccess = true;
    config.verboseOutput = true;
    return config;
}

BenchmarkConfig applyTrace(BenchmarkConfig cfg, const AccessTrace& trace) {
    cfg.randomAccessIndices = trace.randomIndices;
    cfg.randomAccessSamples = trace.randomSamples ? trace.randomSamples : cfg.randomAccessSamples;
    cfg.stridedAccessIndices = trace.stridedIndices;
    cfg.stridedAccessSamples = trace.stridedSamples ? trace.stridedSamples : cfg.stridedAccessSamples;
    cfg.stride = trace.stride ? trace.stride : cfg.stride;
    cfg.rangeAccesses = trace.ranges;
    cfg.rangeQueryCount = trace.rangeQueries ? trace.rangeQueries : cfg.rangeQueryCount;
    return cfg;
}

} // namespace

using namespace encodings;
using namespace encodings::benchmark;
using namespace encodings::generators;
using namespace encodings::encoders;

int main() {
    std::cout << "=== Encoding Playground Benchmark Suite ===\n" << std::endl;

    // Scenarios with explicit access traces
    std::vector<std::pair<std::string, BenchmarkConfig>> scenarios;
    const size_t N = 1'000'000;
    scenarios.emplace_back("mostly_filter", applyTrace(baseConfig(), makeMostlyFilter(N)));
    scenarios.emplace_back("mostly_skips", applyTrace(baseConfig(), makeMostlySkips(N)));
    scenarios.emplace_back("half_and_half", applyTrace(baseConfig(), makeHalfAndHalf(N)));

    // SubInt Configs
    SubIntConfig<int32_t> subIntConfig13; subIntConfig13.splitMode = Split13();
    SubIntConfig<int32_t> subIntConfig22; subIntConfig22.splitMode = Split22();
    SubIntConfig<int32_t> subIntConfig31; subIntConfig31.splitMode = Split31();

    SubIntConfig<int64_t> subIntConfig17; subIntConfig17.splitMode = Split17();
    SubIntConfig<int64_t> subIntConfig26; subIntConfig26.splitMode = Split26();
    SubIntConfig<int64_t> subIntConfig35; subIntConfig35.splitMode = Split35();
    SubIntConfig<int64_t> subIntConfig44; subIntConfig44.splitMode = Split44();
    SubIntConfig<int64_t> subIntConfig53; subIntConfig53.splitMode = Split53();
    SubIntConfig<int64_t> subIntConfig62; subIntConfig62.splitMode = Split62();
    SubIntConfig<int64_t> subIntConfig71; subIntConfig71.splitMode = Split71();

    // FOR Config for int64_t subint
    FORConfig<int64_t, int64_t> forSubint17Config{FORReferencePolicy::MIN, std::make_shared<SubInt64Encoder>(subIntConfig17)};
    FORConfig<int64_t, int64_t> forSubint26Config{FORReferencePolicy::MIN, std::make_shared<SubInt64Encoder>(subIntConfig26)};
    FORConfig<int64_t, int64_t> forSubint35Config{FORReferencePolicy::MIN, std::make_shared<SubInt64Encoder>(subIntConfig35)};
    FORConfig<int64_t, int64_t> forSubint44Config{FORReferencePolicy::MIN, std::make_shared<SubInt64Encoder>(subIntConfig44)};
    FORConfig<int64_t, int64_t> forSubint53Config{FORReferencePolicy::MIN, std::make_shared<SubInt64Encoder>(subIntConfig53)};
    FORConfig<int64_t, int64_t> forSubint62Config{FORReferencePolicy::MIN, std::make_shared<SubInt64Encoder>(subIntConfig62)};
    FORConfig<int64_t, int64_t> forSubint71Config{FORReferencePolicy::MIN, std::make_shared<SubInt64Encoder>(subIntConfig71)};

    // Register encoders (once), then reuse across scenarios.
    std::vector<std::pair<std::string, std::shared_ptr<Codec<int64_t>>>> runnerEncoders;
    auto addEncoder = [&](const std::string& name, std::shared_ptr<Codec<int64_t>> enc) {
        runnerEncoders.emplace_back(name, enc);
    };

    addEncoder("Raw", std::make_shared<RawEncoder<int64_t>>());
    addEncoder("RawBitPacked", std::make_shared<RawBitPackedEncoder<int64_t>>());

    // addEncoder("SubInt17", std::make_shared<SubInt64Encoder>(subIntConfig17));
    // addEncoder("DeltaSubInt17", std::make_shared<DeltaSubIntEncoder<int64_t, 256>>(subIntConfig17));
    // addEncoder("FORSubInt17", std::make_shared<FOREncoder<int64_t, int64_t, 256>>(forSubint17Config));
    addEncoder("SubInt26", std::make_shared<SubInt64Encoder>(subIntConfig26));
    // addEncoder("DeltaSubInt26", std::make_shared<DeltaSubIntEncoder<int64_t, 256>>(subIntConfig26));
    // addEncoder("FORSubInt26", std::make_shared<FOREncoder<int64_t, int64_t, 256>>(forSubint26Config));
    // addEncoder("SubInt35", std::make_shared<SubInt64Encoder>(subIntConfig35));
    // addEncoder("DeltaSubInt35", std::make_shared<DeltaSubIntEncoder<int64_t, 256>>(subIntConfig35));
    // addEncoder("FORSubInt35", std::make_shared<FOREncoder<int64_t, int64_t, 256>>(forSubint35Config));
    // addEncoder("SubInt44", std::make_shared<SubInt64Encoder>(subIntConfig44));
    // addEncoder("DeltaSubInt44", std::make_shared<DeltaSubIntEncoder<int64_t, 256>>(subIntConfig44));
    // addEncoder("FORSubInt44", std::make_shared<FOREncoder<int64_t, int64_t, 256>>(forSubint44Config));
    // addEncoder("SubInt53", std::make_shared<SubInt64Encoder>(subIntConfig53));
    addEncoder("DeltaSubInt53", std::make_shared<DeltaSubIntEncoder<int64_t, 256>>(subIntConfig53));
    addEncoder("FORSubInt53", std::make_shared<FOREncoder<int64_t, int64_t, 256>>(forSubint53Config));
    // addEncoder("SubInt62", std::make_shared<SubInt64Encoder>(subIntConfig62));
    // addEncoder("DeltaSubInt62", std::make_shared<DeltaSubIntEncoder<int64_t, 256>>(subIntConfig62));
    // addEncoder("FORSubInt62", std::make_shared<FOREncoder<int64_t, int64_t, 256>>(forSubint62Config));
    // addEncoder("SubInt71", std::make_shared<SubInt64Encoder>(subIntConfig71));
    // addEncoder("DeltaSubInt71", std::make_shared<DeltaSubIntEncoder<int64_t, 256>>(subIntConfig71));
    // addEncoder("FORSubInt71", std::make_shared<FOREncoder<int64_t, int64_t, 256>>(forSubint71Config));

    addEncoder("AutoSubIntSplit", makeDefaultAutoSubIntSplitEncoder(BitSplitOrder::LSB_TO_MSB, false, true, true));
    // addEncoder("AutoSubIntSplitPrune", makeDefaultAutoSubIntSplitEncoder(BitSplitOrder::LSB_TO_MSB, false, true, true)); // with pruning enabled
    // addEncoder("AutoSubIntSplitMtoL", makeDefaultAutoSubIntSplitEncoder(BitSplitOrder::MSB_TO_LSB, false, true, true));
    // addEncoder("AutoSubIntSplitEx", makeDefaultAutoSubIntSplitEncoder(BitSplitOrder::LSB_TO_MSB, true, true, true)); // with exhaustive search
    // addEncoder("AutoSubIntSplitMtoLEx", makeDefaultAutoSubIntSplitEncoder(BitSplitOrder::MSB_TO_LSB, true, true, true));

    // addEncoder("TriSplitFORDictFOR", makeSnowflakeTriSplitFORDictFOR());
    // addEncoder("TriSplitFORRawFOR", makeSnowflakeTriSplitFORRawFOR());
    // addEncoder("TriSplitFORDictDict", makeSnowflakeTriSplitFORDictDict());
    // addEncoder("TriSplitFORDictRaw", makeSnowflakeTriSplitFORDictRaw());
    addEncoder("TriSplitFOROnly", makeSnowflakeTriSplitFOROnly());
    addEncoder("TriSplitBitPrefixOnly", makeSnowflakeTriSplitBitPrefixOnly());
    addEncoder("TriSplitDictOnly", makeSnowflakeTriSplitDictOnly());
    addEncoder("TriSplitAllRawBitPacked", makeSnowflakeTriSplitAllRawBitPacked());

    addEncoder("TriSplitOpenZLOnly", makeSnowflakeTriSplitOpenZLOnly());
    addEncoder("TriSplitOpenZLOnly64", makeSnowflakeTriSplitOpenZLOnly<64>());
    addEncoder("TriSplitOpenZLOnly256", makeSnowflakeTriSplitOpenZLOnly<256>());
    addEncoder("TriSplitOpenZLOnly1024", makeSnowflakeTriSplitOpenZLOnly<1024>());
    addEncoder("TriSplitOpenZLOnly4096", makeSnowflakeTriSplitOpenZLOnly<4096>());
    addEncoder("TriSplitOpenZLOnly16384", makeSnowflakeTriSplitOpenZLOnly<16384>());
    addEncoder("TriSplitOpenZLOnly65536", makeSnowflakeTriSplitOpenZLOnly<65536>());
    // addEncoder("TriSplitFOROnlyLtoM", makeSnowflakeTriSplitFOROnly(BitSplitOrder::LSB_TO_MSB));
    // addEncoder("TriSplitOpenZLOnlyLtoM", makeSnowflakeTriSplitOpenZLOnly(BitSplitOrder::LSB_TO_MSB));
    // addEncoder("TriSplitBitPrefixOnlyLtoM", makeSnowflakeTriSplitBitPrefixOnly(BitSplitOrder::LSB_TO_MSB));
    // addEncoder("TriSplitDictOnlyLtoM", makeSnowflakeTriSplitDictOnly(BitSplitOrder::LSB_TO_MSB));

    addEncoder("VarInt", std::make_shared<VarIntEncoder<int64_t>>());
    addEncoder("DeltaVarInt", std::make_shared<DeltaVarIntEncoder<int64_t>>());
    
    addEncoder("OpenZL", makeOpenZLCodec<int64_t>());
    addEncoder("OpenZL64", makeOpenZLCodec<int64_t, 64>());
    addEncoder("OpenZL256", makeOpenZLCodec<int64_t, 256>());
    addEncoder("OpenZL1024", makeOpenZLCodec<int64_t, 1024>());
    addEncoder("OpenZL4096", makeOpenZLCodec<int64_t, 4096>());
    addEncoder("OpenZL16384", makeOpenZLCodec<int64_t, 16384>());
    addEncoder("OpenZL65536", makeOpenZLCodec<int64_t, 65536>());

    // runner.registerEncoder("RLE", std::make_shared<RunLengthEncoder<int32_t>>());
    // runner.registerEncoder("DeltaRLE", std::make_shared<DeltaRunLengthEncoder<int32_t>>());
    // runner.registerEncoder("SubInt13", std::make_shared<SubInt32Encoder>(subIntConfig13));
    // runner.registerEncoder("SubInt22", std::make_shared<SubInt32Encoder>(subIntConfig22));
    // runner.registerEncoder("SubInt31", std::make_shared<SubInt32Encoder>(subIntConfig31));
    // runner.registerEncoder("Dictionary", std::make_shared<DictionaryEncoder<int32_t>>());
    // runner.registerEncoder("VarInt", std::make_shared<VarIntEncoder<int32_t>>());
    // runner.registerEncoder("DeltaVarInt", std::make_shared<DeltaVarIntEncoder<int32_t>>());
    
    // TODO: Add composed encoder example (requires encoders that work on bytes)
    // For now, we benchmark individual encoders

    std::cout << "Registered " << runnerEncoders.size() << " encoders\n" << std::endl;

    // Register datasets
    std::cout << "Registering data generators..." << std::endl;
    
    std::vector<std::pair<std::string, std::shared_ptr<DataGenerator<int64_t>>>> runnerDatasets;


    ///////////////// 32 BIT INT DATASETS
    // GBIF occurrence counts (real-world data with a mix of patterns)
    // std::filesystem::path dataDir = "/home/david/Documents/PhD/symbol-store/EncodingsPlayground/Datasets/iNaturalist_species_ids.parquet";
    // std::string columnName = "species_id";
    // runner.addDataset("GBIF Occurrences", 
    //     std::make_shared<ParquetColumnGenerator<int32_t>>(dataDir, columnName));
    
    // runner.addDataset("Instagram Snowflake IDs", 
    //     std::make_shared<SnowflakeIDGenerator<int32_t>>(INSTAGRAM_SNOWFLAKE_INT_CONFIG, 127, 42, 0.05)
    // );

    // auto parquetGen = ParquetColumnGenerator<int32_t>(dataDir, columnName);
    // auto histogram = computeHistogram<int32_t>(parquetGen, 10000000);
    // std::filesystem::path histogramPath = config.outputPath + "/gbif_species_id_histogram.csv";
    // writeHistogramCSV(histogram, histogramPath);

    // auto snowflakeGen = SnowflakeIDGenerator<int32_t>(INSTAGRAM_SNOWFLAKE_INT_CONFIG, 127, 42, 0.05);
    // auto histogramSnowflake = computeHistogram<int32_t>(snowflakeGen, 10000000);
    // std::filesystem::path histogramPathSnowflake = config.outputPath + "/instagram_snowflake_id_histogram.csv";
    // writeHistogramCSV(histogramSnowflake, histogramPathSnowflake);


    ///////////////// 64 BIT INT DATASETS

    auto snowflakeGenPtr = std::make_shared<SnowflakeIDGenerator<int64_t>>(INSTAGRAM_SNOWFLAKE_CONFIG, 4096, 42, 0.5);
    runnerDatasets.emplace_back("Instagram Snowflake IDs", snowflakeGenPtr);
    auto snowflakeGen = SnowflakeIDGenerator<int64_t>(INSTAGRAM_SNOWFLAKE_CONFIG, 4096, 42, 0.5);
    auto histogramSnowflake = computeHistogram<int64_t>(snowflakeGen, 10000000);
    // Histogram path will be written per scenario below based on outputPath
    
    // // Sequential data (great for Delta)
    // runner.addDataset("Sequential", 
    //     std::make_shared<SequentialGenerator<int32_t>>(0, 1));
    
    // // Repetitive data (great for RLE)
    // runner.addDataset("Repetitive", 
    //     std::make_shared<RepetitiveGenerator<int32_t>>(20, 0, 50));
    
    // // Low cardinality (great for Dictionary)
    // runner.addDataset("Zipfian", 
    //     std::make_shared<ZipfianGenerator<int32_t>>(100, 1.5));
    
    // // Random data (nothing helps much)
    // runner.addDataset("Random", 
    //     std::make_shared<UniformRandomGenerator<int32_t>>(
    //         std::numeric_limits<int32_t>::min() / 2,
    //         std::numeric_limits<int32_t>::max() / 2));
    
    // // Nearly sorted (good for Delta)
    // runner.addDataset("NearlySorted", 
    //     std::make_shared<NearlySortedGenerator<int32_t>>(0, 1, 0.05));
    
    // // Constant (best for RLE)
    // runner.addDataset("Constant", 
    //     std::make_shared<ConstantGenerator<int32_t>>(42));

    std::cout << "Registered " << runnerDatasets.size() << " datasets\n" << std::endl;

    // Run all scenarios
    for (const auto& [scenarioName, cfg] : scenarios) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Starting benchmark suite for scenario: " << scenarioName << std::endl;
        std::cout << "Total configurations: " 
                  << runnerEncoders.size() << " encoders × " 
                  << runnerDatasets.size() << " datasets × " 
                  << cfg.dataSizes.size() << " sizes = "
                  << runnerEncoders.size() * runnerDatasets.size() * cfg.dataSizes.size() << std::endl;
        std::cout << std::string(80, '=') << "\n" << std::endl;

        // Apply config and run
        BenchmarkRunner<int64_t> scenarioRunner(cfg);
        // Re-register encoders and datasets for this runner
        for (const auto& [name, enc] : runnerEncoders) {
            scenarioRunner.registerEncoder(name, enc);
        }
        for (const auto& [name, gen] : runnerDatasets) {
            scenarioRunner.addDataset(name, gen);
        }

        auto results = scenarioRunner.runAll();

        // Print summary table
        TableFormatter::printSummaryTable(results);

        // Save JSON results with scenario suffix
        std::filesystem::create_directories(cfg.outputPath);
        std::string jsonPath = cfg.outputPath + "/benchmark_results_" + scenarioName + ".json";
        if (saveBenchmarkResults(results, jsonPath)) {
            std::cout << "✓ JSON results saved successfully: " << jsonPath << std::endl;
        }

        // Print highlights for this scenario
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "HIGHLIGHTS (" << scenarioName << ")" << std::endl;
        std::cout << std::string(80, '=') << "\n" << std::endl;

        std::map<std::string, std::pair<std::string, double>> bestCompression;
        for (const auto& result : results.results) {
            double ratio = result.metrics.memory.compressionRatio();
            auto& best = bestCompression[result.datasetName];
            if (best.second == 0 || ratio < best.second) {
                best.first = result.encoderName;
                best.second = ratio;
            }
        }

        std::cout << "Best Compression Ratios:\n";
        for (const auto& [dataset, best] : bestCompression) {
            std::cout << "  " << std::left << std::setw(20) << dataset 
                      << " → " << std::setw(15) << best.first
                      << " " << std::fixed << std::setprecision(6) 
                      << best.second << "x\n";
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Scenario completed: " << scenarioName << std::endl;
        std::cout << std::string(80, '=') << "\n" << std::endl;
    }

    return 0;
}
