#include "benchmark/BenchmarkRunner.hpp"
#include "benchmark/BenchmarkOutput.hpp"
#include "generators/CommonGenerators.hpp"
#include "generators/ParquetColumnGenerator.hpp"
#include "generators/GeneratorUtils.hpp"
#include "generators/SnowflakeIDGenerator.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/DeltaRunLengthEncoder.hpp"
#include "encoders/SubIntEncoder.hpp"
#include "encoders/DeltaEncoder.hpp"
#include "encoders/DictionaryEncoder.hpp"
#include "encoders/VarIntEncoder.hpp"
#include "encoders/DeltaVarIntEncoder.hpp"
#include "encodings/ComposedEncoder.hpp"
#include <iostream>

using namespace encodings;
using namespace encodings::benchmark;
using namespace encodings::generators;
using namespace encodings::encoders;

int main() {
    std::cout << "=== Encoding Playground Benchmark Suite ===\n" << std::endl;
    
    // Configure benchmarks
    BenchmarkConfig config;
    config.dataSizes = {10000000};
    config.iterations = 1;
    config.warmupRuns = 0;
    config.randomAccessSamples = 10000;
    config.stridedAccessSamples = 1000;
    config.stride = 100;
    config.rangeQueryCount = 100;
    config.rangeSizes = {65536};
    config.validateCorrectness = true;
    config.validateRandomAccess = true;
    config.verboseOutput = true;
    
    // Create benchmark runner
    BenchmarkRunner<int32_t> runner(config);

    // SubInt Configs
    SubIntConfig subIntConfig13;
    subIntConfig13.splitMode = SplitMode::Split13;
    SubIntConfig subIntConfig22;
    subIntConfig22.splitMode = SplitMode::Split22;
    SubIntConfig subIntConfig31;
    subIntConfig31.splitMode = SplitMode::Split31;

    // Register encoders
    std::cout << "Registering encoders..." << std::endl;
    runner.registerEncoder("Raw", std::make_shared<RawEncoder<int32_t>>());
    // runner.registerEncoder("RLE", std::make_shared<RunLengthEncoder<int32_t>>());
    // runner.registerEncoder("DeltaRLE", std::make_shared<DeltaRunLengthEncoder<int32_t>>());
    runner.registerEncoder("SubInt13", std::make_shared<SubIntEncoder>(subIntConfig13));
    runner.registerEncoder("SubInt22", std::make_shared<SubIntEncoder>(subIntConfig22));
    runner.registerEncoder("SubInt31", std::make_shared<SubIntEncoder>(subIntConfig31));
    runner.registerEncoder("Dictionary", std::make_shared<DictionaryEncoder<int32_t>>());
    runner.registerEncoder("VarInt", std::make_shared<VarIntEncoder<int32_t>>());
    runner.registerEncoder("DeltaVarInt", std::make_shared<DeltaVarIntEncoder<int32_t>>());
    
    // TODO: Add composed encoder example (requires encoders that work on bytes)
    // For now, we benchmark individual encoders

    std::cout << "Registered " << runner.getNumEncoders() << " encoders\n" << std::endl;

    // Register datasets
    std::cout << "Registering data generators..." << std::endl;
    
    // GBIF occurrence counts (real-world data with a mix of patterns)
    std::filesystem::path dataDir = "/home/david/Documents/PhD/symbol-store/EncodingsPlayground/Datasets/iNaturalist_species_ids.parquet";
    std::string columnName = "species_id";
    runner.addDataset("GBIF Occurrences", 
        std::make_shared<ParquetColumnGenerator<int32_t>>(dataDir, columnName));
    
    runner.addDataset("Instagram Snowflake IDs", 
        std::make_shared<SnowflakeIDGenerator<int32_t>>(INSTAGRAM_SNOWFLAKE_INT_CONFIG, 127, 42, 0.05)
    );

    auto parquetGen = ParquetColumnGenerator<int32_t>(dataDir, columnName);
    auto histogram = computeHistogram<int32_t>(parquetGen, 10000000);
    std::filesystem::path histogramPath = config.outputPath + "/gbif_species_id_histogram.csv";
    writeHistogramCSV(histogram, histogramPath);

    auto snowflakeGen = SnowflakeIDGenerator<int32_t>(INSTAGRAM_SNOWFLAKE_INT_CONFIG, 127, 42, 0.05);
    auto histogramSnowflake = computeHistogram<int32_t>(snowflakeGen, 10000000);
    std::filesystem::path histogramPathSnowflake = config.outputPath + "/instagram_snowflake_id_histogram.csv";
    writeHistogramCSV(histogramSnowflake, histogramPathSnowflake);
    
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

    std::cout << "Registered " << runner.getNumDatasets() << " datasets\n" << std::endl;

    // Run all benchmarks
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Starting benchmark suite..." << std::endl;
    std::cout << "Total configurations: " 
              << runner.getNumEncoders() << " encoders × " 
              << runner.getNumDatasets() << " datasets × " 
              << runner.getNumSizes() << " sizes = "
              << runner.getNumConfigurations() << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;
    
    auto results = runner.runAll();
    
    // Print summary table
    TableFormatter::printSummaryTable(results);
    
    // Save JSON results
    std::string jsonPath = config.outputPath + "/benchmark_results.json";
    if (saveBenchmarkResults(results, jsonPath)) {
        std::cout << "✓ JSON results saved successfully" << std::endl;
    }
    
    // Print detailed results for specific interesting cases
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "HIGHLIGHTS" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;
    
    // Find best compression for each dataset
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
    
    // Find fastest encoders
    std::cout << "\nFastest Encoding (10m elements):\n";
    std::map<std::string, std::pair<std::string, double>> fastestEncode;
    
    for (const auto& result : results.results) {
        if (result.dataSize == 10000000) {
            double elemPerSec = result.metrics.timing.encodeElementsPerSecond;
            auto& fastest = fastestEncode[result.datasetName];
            
            if (elemPerSec > fastest.second) {
                fastest.first = result.encoderName;
                fastest.second = elemPerSec;
            }
        }
    }
    
    for (const auto& [dataset, fastest] : fastestEncode) {
        std::cout << "  " << std::left << std::setw(20) << dataset 
                  << " → " << std::setw(15) << fastest.first
                  << " " << std::scientific << std::setprecision(2)
                  << fastest.second << " elem/s\n";
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Benchmark suite completed successfully!" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) 
              << results.totalDuration().count() << " seconds" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;
    
    return 0;
}
