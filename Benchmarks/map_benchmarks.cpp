/**
 * @file map_benchmarks.cpp
 * @brief Comprehensive benchmarking suite for map encoders
 * 
 * Tests all 6 map encoding strategies:
 * 1. MapDictSeparate - Dictionary encode keys and values separately
 * 2. MapDictTogether - Dictionary encode (key,value) pairs together
 * 3. MapDeltaRLEDict - Delta+RLE for keys, Dictionary for values
 * 4. MapColumnarPairsDict - Columnar layout with pairs, dictionary encoded
 * 5. MapColumnarSeparateDict - Columnar separate with dictionary on both
 * 6. MapColumnarMixed - Columnar with Delta+RLE keys, Dictionary values
 */

#include "benchmark/BenchmarkRunner.hpp"
#include "benchmark/BenchmarkOutput.hpp"
#include "encoders/MapEncoders.hpp"
#include "encoders/MapEncodersRefactored.hpp"
#include "encoders/MapColumnarEncoders.hpp"
#include "encoders/DictionaryEncoder.hpp"
#include "encoders/DeltaEncoder.hpp"
#include "encoders/DeltaRunLengthEncoder.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include "encoders/BitParenthesesEncoder.hpp"
#include "encoders/RawEncoder.hpp"
#include "generators/MapGenerator.hpp"
#include "generators/MapGenerators.hpp"
#include "generators/StrictlyIncreasingMinMaxGenerator.hpp"
#include "generators/RandomMinMaxGenerator.hpp"
#include "generators/TPCHLineitemPartKeyGenerator.hpp"
#include <iostream>
#include <memory>
#include <fstream>

using namespace encodings::encoders;
using namespace encodings::datagen;
using namespace encodings::benchmark;

int main() {
    std::cout << "=== Map Encoding Benchmark Suite ===\n\n";
    
    // Map type: int32_t -> int32_t
    using K = int32_t;
    using V = int32_t;
    using MapType = std::map<K, V>;
    
    // Create benchmark config with verbose output
    BenchmarkConfig config;
    config.verboseOutput = true;
    config.iterations = 3;      // 3 iterations per test
    config.warmupRuns = 1;      // 1 warmup run
    config.dataSizes = {100000};  // Test with 1000000 maps
    
    // Create benchmark runner
    BenchmarkRunner<MapType> runner(config);
    
    // ===== Strategy 1: Dictionary encode keys and values separately =====
    {
        auto keyEncoder = std::make_shared<DictionaryEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto encoder = std::make_shared<MapSeparateEncoder<K, V>>(
            keyEncoder, valueEncoder);
        runner.registerEncoder("MapDictSeparate", encoder);
        std::cout << "Registered: MapDictSeparate\n";
    }
    
    // ===== Strategy 2: Dictionary encode (key,value) pairs together =====
    // Note: This would require a DictionaryEncoder<std::pair<K,V>>
    // For now, we'll skip this one as it requires additional pair encoder support
    
    // ===== Strategy 3: Delta+RLE for keys, Dictionary for values =====
    {
        // Use the combined DeltaRunLengthEncoder for keys
        auto keyEncoder = std::make_shared<DeltaRunLengthEncoder<K>>();
        
        // For values, use dictionary
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();

        auto encoder = std::make_shared<MapSeparateEncoder<K, V>>(
            keyEncoder, valueEncoder, "MapDeltaRLEDict");
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapDeltaRLEDict\n";
    }
    
    // // ===== Strategy 4: Columnar layout with pairs, dictionary encoded =====
    // {
    //     auto keyEncoder = std::make_shared<DictionaryEncoder<K>>();
    //     auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
    //     auto encoder = std::make_shared<MapColumnarPairsDictEncoder<K, V>>(
    //         keyEncoder, valueEncoder);
    //     runner.registerEncoder(encoder->name(), encoder);
    //     std::cout << "Registered: MapColumnarPairsDict\n";
    // }
    
    // // ===== Strategy 5: Columnar separate with dictionary on both =====
    // {
    //     auto keyEncoder = std::make_shared<DictionaryEncoder<K>>();
    //     auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
    //     auto encoder = std::make_shared<MapColumnarSeparateDictEncoder<K, V>>(
    //         keyEncoder, valueEncoder);
    //     runner.registerEncoder(encoder->name(), encoder);
    //     std::cout << "Registered: MapColumnarSeparateDict\n";
    // }
    
    // // ===== Strategy 6: Columnar with Delta+RLE keys, Dictionary values =====
    // {
    //     auto keyEncoder = std::make_shared<DeltaEncoder<K>>();
    //     auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
    //     auto encoder = std::make_shared<MapColumnarMixedEncoder<K, V>>(
    //         keyEncoder, valueEncoder);
    //     runner.registerEncoder(encoder->name(), encoder);
    //     std::cout << "Registered: MapColumnarMixed\n";
    // }

    // ===== Strategy 7: Columnar with group indices (new) =====
    {
        auto keyEncoder = std::make_shared<RunLengthEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto sizeEncoder = std::make_shared<RawEncoder<uint8_t>>();
        auto encoder = std::make_shared<MapGroupIndicesEncoder<K, V, uint8_t>>(
            keyEncoder, valueEncoder, sizeEncoder);
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapGroupIndices\n";
    }

    // ===== Strategy 7: Columnar with group indices (new) =====
    {
        auto keyEncoder = std::make_shared<RunLengthEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto sizeEncoder = std::make_shared<RunLengthEncoder<uint8_t>>();
        auto encoder = std::make_shared<MapGroupIndicesEncoder<K, V, uint8_t>>(
            keyEncoder, valueEncoder, sizeEncoder);
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapGroupIndices\n";
    }
    
    // ===== Strategy 7: Columnar with group indices (new) =====
    {
        auto keyEncoder = std::make_shared<RunLengthEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto sizeEncoder = std::make_shared<DeltaRunLengthEncoder<uint8_t>>();
        auto encoder = std::make_shared<MapGroupIndicesEncoder<K, V, uint8_t>>(
            keyEncoder, valueEncoder, sizeEncoder);
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapGroupIndices\n";
    }
    
    // ===== Strategy 7: Columnar with group indices (new) =====
    {
        auto keyEncoder = std::make_shared<RunLengthEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto sizeEncoder = std::make_shared<BitParenthesesEncoder<uint8_t>>();
        auto encoder = std::make_shared<MapGroupIndicesEncoder<K, V, uint8_t>>(
            keyEncoder, valueEncoder, sizeEncoder);
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapGroupIndices\n";
    }
    
    // ===== Strategy 8: Columnar with group indices (new) =====
    {
        auto keyEncoder = std::make_shared<DeltaRunLengthEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto sizeEncoder = std::make_shared<RawEncoder<uint8_t>>();
        auto encoder = std::make_shared<MapGroupIndicesEncoder<K, V, uint8_t>>(
            keyEncoder, valueEncoder, sizeEncoder);
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapGroupIndices Delta RLE\n";
    }

    // ===== Strategy 8: Columnar with group indices (new) =====
    {
        auto keyEncoder = std::make_shared<DeltaRunLengthEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto sizeEncoder = std::make_shared<RunLengthEncoder<uint8_t>>();
        auto encoder = std::make_shared<MapGroupIndicesEncoder<K, V, uint8_t>>(
            keyEncoder, valueEncoder, sizeEncoder);
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapGroupIndices Delta RLE\n";
    }

    // ===== Strategy 8: Columnar with group indices (new) =====
    {
        auto keyEncoder = std::make_shared<DeltaRunLengthEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto sizeEncoder = std::make_shared<DeltaRunLengthEncoder<uint8_t>>();
        auto encoder = std::make_shared<MapGroupIndicesEncoder<K, V, uint8_t>>(
            keyEncoder, valueEncoder, sizeEncoder);
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapGroupIndices Delta RLE\n";
    }

    // ===== Strategy 8: Columnar with group indices (new) =====
    {
        auto keyEncoder = std::make_shared<DeltaRunLengthEncoder<K>>();
        auto valueEncoder = std::make_shared<DictionaryEncoder<V>>();
        auto sizeEncoder = std::make_shared<BitParenthesesEncoder<uint8_t>>();
        auto encoder = std::make_shared<MapGroupIndicesEncoder<K, V, uint8_t>>(
            keyEncoder, valueEncoder, sizeEncoder);
        runner.registerEncoder(encoder->name(), encoder);
        std::cout << "Registered: MapGroupIndices Delta RLE\n";
    }
    
    std::cout << "\nTotal encoders registered: " << runner.getNumEncoders() << "\n\n";
    
    // ===== Register Generators =====
    
    // Generator 1: Sequential keys (optimal for Delta encoding)
    // auto seqKeyGen = std::make_shared<SequentialKeyMapGenerator<V>>(
    //     5, 20, 0);  // Maps with 5-20 entries, starting from key 0
    // runner.addDataset("SequentialKeyMap", seqKeyGen);
    // std::cout << "Registered generator: SequentialKeyMap\n";
    
    // Generator 2: Low cardinality keys (optimal for Dictionary encoding)
    // auto lowCardKeyGen = std::make_shared<LowCardinalityKeyMapGenerator<K, V>>(
    //     10, 5, 15);  // 10 unique keys, maps with 5-15 entries
    // runner.addDataset("LowCardinalityKeyMap", lowCardKeyGen);
    // std::cout << "Registered generator: LowCardinalityKeyMap\n";
    
    // Generator 3: Low cardinality values (optimal for Dictionary encoding)
    // auto lowCardValGen = std::make_shared<LowCardinalityValueMapGenerator<K, V>>(
    //     5, 5, 15);  // 5 unique values, maps with 5-15 entries
    // runner.addDataset("LowCardinalityValueMap", lowCardValGen);
    // std::cout << "Registered generator: LowCardinalityValueMap\n";
    
    // Generator 4: Constant size maps (optimal for RLE size encoding)
    // auto constSizeGen = std::make_shared<ConstantSizeMapGenerator<K, V>>(10);
    // runner.addDataset("ConstantSizeMap", constSizeGen);
    // std::cout << "Registered generator: ConstantSizeMap\n";
    
    // Generator 5: Varying sizes (tests size encoding robustness)
    // auto varySizeGen = std::make_shared<VaryingSizeMapGenerator<K, V>>(1, 50);
    // runner.addDataset("VaryingSizeMap", varySizeGen);
    // std::cout << "Registered generator: VaryingSizeMap\n";
    
    // // Generator 6: Columnar optimized (sequential keys + low cardinality values)
    // auto columnarGen = std::make_shared<ColumnarOptimizedMapGenerator<V>>(
    //     5, 10, 20);  // 5 unique values, maps with 10-20 entries
    // runner.addDataset("ColumnarOptimizedMap", columnarGen);
    // std::cout << "Registered generator: ColumnarOptimizedMap\n";

    // Generator 7: Post Clicks Map (strictly increasing keys, medium cardinality values)
    auto keyMinMaxGen = std::make_shared<StrictlyIncreasingMinMaxGenerator<K>>(
        10, 20, 1, 3, 42);  // minValue=10, maxValue=20, minIncrement=1, maxIncrement=3, seed=42
    auto valRandomGen = std::make_shared<TPCHLineitemPartKeyGenerator>();
    // auto valRandomGen = std::make_shared<RandomMinMaxGenerator<int32_t>>(1, 100, 42);
    auto sizesRandomGen = std::make_shared<RandomMinMaxGenerator<int64_t>>(10, 15, 42);
    auto postClicksGen = std::make_shared<MapGeneratorCompositional<K, V>>(
        keyMinMaxGen, valRandomGen, sizesRandomGen);
    runner.addDataset("PostClicksMap", postClicksGen);
    std::cout << "Registered generator: PostClicksMap\n";
    // TODO: Add more generators as needed
    
    std::cout << "\nTotal generators registered: " << runner.getNumDatasets() << "\n\n";
    
    // ===== Run Benchmarks =====
    
    std::cout << "Running benchmarks...\n";
    std::cout << "This will test " << (runner.getNumEncoders() * runner.getNumDatasets()) 
              << " encoder-generator combinations.\n\n";
    
    auto results = runner.runAll();  // 1000 maps per test
    
    std::cout << "Benchmarks complete!\n\n";
    
    // ===== Output Results =====
    
    // Console table
    std::cout << "=== Results Summary ===\n\n";
    encodings::benchmark::TableFormatter::printSummaryTable(results);
    
    // Save to JSON
    std::string jsonFile = "map_benchmark_results.json";
    encodings::benchmark::saveBenchmarkResults(results, jsonFile);
    
    // Analysis
    std::cout << "\n=== Performance Analysis ===\n\n";
    
    // Find best encoder for each metric
    if (!results.results.empty()) {
        // Best compression
        auto bestCompression = *std::min_element(results.results.begin(), results.results.end(),
            [](const auto& a, const auto& b) {
                return a.metrics.memory.compressionRatio() < b.metrics.memory.compressionRatio();
            });
        
        std::cout << "Best compression: " << bestCompression.encoderName 
                  << " on " << bestCompression.datasetName 
                  << " (" << bestCompression.metrics.memory.compressionRatio() << "x)\n";
        
        // Fastest encode
        auto fastestEncode = *std::min_element(results.results.begin(), results.results.end(),
            [](const auto& a, const auto& b) {
                return a.metrics.timing.encodeTime < b.metrics.timing.encodeTime;
            });
        
        std::cout << "Fastest encode: " << fastestEncode.encoderName 
                  << " on " << fastestEncode.datasetName 
                  << " (" << fastestEncode.metrics.timing.encodeTime.count() / 1e6 << " ms)\n";
        
        // Fastest decode
        auto fastestDecode = *std::min_element(results.results.begin(), results.results.end(),
            [](const auto& a, const auto& b) {
                return a.metrics.timing.decodeBulkTime < b.metrics.timing.decodeBulkTime;
            });
        
        std::cout << "Fastest decode: " << fastestDecode.encoderName 
                  << " on " << fastestDecode.datasetName 
                  << " (" << fastestDecode.metrics.timing.decodeBulkTime.count() / 1e6 << " ms)\n";
    }
    
    std::cout << "\n=== Recommendations ===\n\n";
    std::cout << "1. For sequential keys: Use MapDeltaRLEDict or MapColumnarMixed\n";
    std::cout << "2. For low-cardinality keys: Use MapDictSeparate or MapColumnarSeparateDict\n";
    std::cout << "3. For low-cardinality values: Use any Dict-based encoder\n";
    std::cout << "4. For constant-size maps: All encoders benefit from RLE size encoding\n";
    std::cout << "5. For columnar analytics: Use MapColumnar* variants\n";
    
    std::cout << "\nTo generate plots, run: python3 plot_map_results.py\n";
    
    return 0;
}
