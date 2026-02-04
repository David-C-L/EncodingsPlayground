#include "benchmark/BenchmarkRunner.hpp"
#include "benchmark/BenchmarkOutput.hpp"
#include "generators/CommonGenerators.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/DeltaEncoder.hpp"
#include "encoders/DictionaryEncoder.hpp"
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
    config.dataSizes = {1000, 10000, 100000};
    config.iterations = 5;
    config.warmupRuns = 2;
    config.randomAccessSamples = 100;
    config.stridedAccessSamples = 100;
    config.stride = 10;
    config.rangeQueryCount = 10;
    config.rangeSizes = {10, 100, 1000};
    config.validateCorrectness = true;
    config.validateRandomAccess = true;
    config.verboseOutput = true;
    
    // Create benchmark runner
    BenchmarkRunner<int32_t> runner(config);
    
    // Register encoders
    std::cout << "Registering encoders..." << std::endl;
    runner.registerEncoder("Raw", std::make_shared<RawEncoder<int32_t>>());
    runner.registerEncoder("RLE", std::make_shared<RunLengthEncoder<int32_t>>());
    runner.registerEncoder("Delta", std::make_shared<DeltaEncoder<int32_t>>());
    runner.registerEncoder("Dictionary", std::make_shared<DictionaryEncoder<int32_t>>());
    
    // TODO: Add composed encoder example (requires encoders that work on bytes)
    // For now, we benchmark individual encoders

    std::cout << "Registered " << runner.getNumEncoders() << " encoders\n" << std::endl;

    // Register datasets
    std::cout << "Registering data generators..." << std::endl;
    
    // Sequential data (great for Delta)
    runner.addDataset("Sequential", 
        std::make_shared<SequentialGenerator<int32_t>>(0, 1));
    
    // Repetitive data (great for RLE)
    runner.addDataset("Repetitive", 
        std::make_shared<RepetitiveGenerator<int32_t>>(20, 0, 50));
    
    // Low cardinality (great for Dictionary)
    runner.addDataset("Zipfian", 
        std::make_shared<ZipfianGenerator<int32_t>>(100, 1.5));
    
    // Random data (nothing helps much)
    runner.addDataset("Random", 
        std::make_shared<UniformRandomGenerator<int32_t>>(
            std::numeric_limits<int32_t>::min() / 2,
            std::numeric_limits<int32_t>::max() / 2));
    
    // Nearly sorted (good for Delta)
    runner.addDataset("NearlySorted", 
        std::make_shared<NearlySortedGenerator<int32_t>>(0, 1, 0.05));
    
    // Constant (best for RLE)
    runner.addDataset("Constant", 
        std::make_shared<ConstantGenerator<int32_t>>(42));

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
    std::cout << "\nFastest Encoding (100k elements):\n";
    std::map<std::string, std::pair<std::string, double>> fastestEncode;
    
    for (const auto& result : results.results) {
        if (result.dataSize == 100000) {
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
