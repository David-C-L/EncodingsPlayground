#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include "BenchmarkMetrics.hpp"
#include "encodings/Encoder.hpp"
#include "generators/DataGenerator.hpp"

namespace encodings::benchmark {

using namespace std::chrono;
using namespace encodings::datagen;  // For DataGenerator

/**
 * @brief Result of a single benchmark run
 */
template<typename T>
struct BenchmarkResult {
    std::string encoderName;
    std::string datasetName;
    size_t dataSize;
    BenchmarkMetrics metrics;
    
    // For composed encoders - track individual layer metrics
    std::vector<BenchmarkMetrics> layerMetrics;
    bool isComposed{false};
    
    // Original data for validation
    std::vector<T> originalData;
    encodings::EncodedData encodedData;
};

/**
 * @brief Collection of all benchmark results
 */
template<typename T>
struct BenchmarkResults {
    std::vector<BenchmarkResult<T>> results;
    BenchmarkConfig config;
    system_clock::time_point startTime;
    system_clock::time_point endTime;
    
    duration<double> totalDuration() const {
        return endTime - startTime;
    }
};

/**
 * @brief Main benchmarking runner
 */
template<typename T>
class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkConfig config = BenchmarkConfig{})
        : config_(config), rng_(std::random_device{}()) {}
    
    /**
     * @brief Register an encoder for benchmarking
     */
    void registerEncoder(const std::string& name, 
                        std::shared_ptr<encodings::Codec<T>> encoder) {
        encoders_.push_back({name, encoder});
    }
    
    /**
     * @brief Add a dataset (name + data generator)
     */
    void addDataset(const std::string& name,
                   std::shared_ptr<DataGenerator<T>> generator) {
        datasets_.push_back({name, generator});
    }

    int64_t getNumEncoders() const {
        return static_cast<int64_t>(encoders_.size());
    }

    int64_t getNumDatasets() const {
        return static_cast<int64_t>(datasets_.size());
    }

    int64_t getNumSizes() const {
        return static_cast<int64_t>(config_.dataSizes.size());
    }

    int64_t getNumConfigurations() const {
        return getNumEncoders() * getNumDatasets() * getNumSizes();
    }
    
    /**
     * @brief Run all benchmarks (all encoders × all datasets × all sizes)
     */
    BenchmarkResults<T> runAll() {
        BenchmarkResults<T> results;
        results.config = config_;
        results.startTime = system_clock::now();
        
        size_t totalBenchmarks = encoders_.size() * datasets_.size() * config_.dataSizes.size();
        size_t currentBenchmark = 0;
        
        std::cout << "Running " << totalBenchmarks << " benchmark configurations...\n" << std::endl;
        
        for (const auto& [encoderName, encoder] : encoders_) {
            for (const auto& [datasetName, generator] : datasets_) {
                for (size_t dataSize : config_.dataSizes) {
                    currentBenchmark++;
                    
                    if (config_.verboseOutput) {
                        std::cout << "[" << currentBenchmark << "/" << totalBenchmarks << "] "
                                  << "Benchmarking: " << encoderName 
                                  << " on " << datasetName 
                                  << " (n=" << dataSize << ")" << std::endl;
                    }
                    
                    auto result = runSingleBenchmark(encoderName, encoder, 
                                                    datasetName, generator, dataSize);
                    results.results.push_back(std::move(result));
                }
            }
        }
        
        results.endTime = system_clock::now();
        
        std::cout << "\nCompleted " << totalBenchmarks << " benchmarks in "
                  << results.totalDuration().count() << " seconds" << std::endl;
        
        return results;
    }
    
    /**
     * @brief Run benchmark for a specific configuration
     */
    BenchmarkResult<T> runSingleBenchmark(
        const std::string& encoderName,
        std::shared_ptr<encodings::Codec<T>> encoder,
        const std::string& datasetName,
        std::shared_ptr<DataGenerator<T>> generator,
        size_t dataSize) {
        
        BenchmarkResult<T> result;
        result.encoderName = encoderName;
        result.datasetName = datasetName;
        result.dataSize = dataSize;
        result.metrics.encoderName = encoderName;
        result.metrics.generatorName = datasetName;
        result.metrics.elementCount = dataSize;
        result.metrics.iterations = config_.iterations;
        result.metrics.warmupRuns = config_.warmupRuns;
        result.metrics.timestamp = system_clock::now();
        
        // Generate data
        generator->reset();
        result.originalData = generator->generate(dataSize);
        
        // Warmup runs
        for (size_t i = 0; i < config_.warmupRuns; ++i) {
            auto encoded = encoder->encode(result.originalData);
            auto decoded = encoder->decodeAll(encoded);
            (void)decoded; // Suppress unused warning
        }
        
        // Measured runs
        std::vector<nanoseconds> encodeTimes;
        std::vector<nanoseconds> decodeTimes;
        encodings::EncodedData encoded;
        
        for (size_t i = 0; i < config_.iterations; ++i) {
            // Measure encoding
            auto encodeStart = high_resolution_clock::now();
            encoded = encoder->encode(result.originalData);
            auto encodeEnd = high_resolution_clock::now();
            encodeTimes.push_back(duration_cast<nanoseconds>(encodeEnd - encodeStart));

            auto selectionIt = encoded.metadata().customMetadata.find("selectionTime_ns");
            if (selectionIt != encoded.metadata().customMetadata.end()) {
                try {
                    result.metrics.customMetrics["selectionTime_ns"] = std::stod(selectionIt->second);
                } catch (const std::exception&) {
                    // ignore parse errors
                }
            }
            
            // Measure decoding
            auto decodeStart = high_resolution_clock::now();
            auto decoded = encoder->decodeAll(encoded);
            auto decodeEnd = high_resolution_clock::now();
            decodeTimes.push_back(duration_cast<nanoseconds>(decodeEnd - decodeStart));
            
            // Validation on first iteration
            if (i == 0 && config_.validateCorrectness) {
                validateCorrectness(result.originalData, decoded, result.metrics.accuracy);
            }
        }
        
        result.encodedData = encoded;
        
        // Calculate timing statistics
        calculateTimingStats(encodeTimes, decodeTimes, result);
        
        // Memory metrics
        result.metrics.memory.originalSize = dataSize * sizeof(T);
        result.metrics.memory.encodedSize = encoded.size();
        
        // Random access benchmarks
        if (config_.testRandomAccess) {
            benchmarkRandomAccess(encoder, encoded, result.originalData, result.metrics);
        }
        
        if (config_.testStridedAccess) {
            benchmarkStridedAccess(encoder, encoded, result.originalData, result.metrics);
        }
        
        if (config_.testRangeAccess) {
            benchmarkRangeAccess(encoder, encoded, result.originalData, result.metrics);
        }
        
        return result;
    }
    
private:
    BenchmarkConfig config_;
    std::mt19937 rng_;
    
    std::vector<std::pair<std::string, std::shared_ptr<encodings::Codec<T>>>> encoders_;
    std::vector<std::pair<std::string, std::shared_ptr<DataGenerator<T>>>> datasets_;
    
    void calculateTimingStats(const std::vector<nanoseconds>& encodeTimes,
                              const std::vector<nanoseconds>& decodeTimes,
                              BenchmarkResult<T>& result) {
        // Mean encode time
        auto encodeSum = std::accumulate(encodeTimes.begin(), encodeTimes.end(), nanoseconds{0});
        result.metrics.timing.encodeTime = encodeSum / encodeTimes.size();
        
        // Mean decode time
        auto decodeSum = std::accumulate(decodeTimes.begin(), decodeTimes.end(), nanoseconds{0});
        result.metrics.timing.decodeBulkTime = decodeSum / decodeTimes.size();
        
        // Standard deviations
        if (encodeTimes.size() > 1) {
            double encodeVariance = 0.0;
            double decodeVariance = 0.0;
            
            for (const auto& time : encodeTimes) {
                double diff = (time - result.metrics.timing.encodeTime).count();
                encodeVariance += diff * diff;
            }
            
            for (const auto& time : decodeTimes) {
                double diff = (time - result.metrics.timing.decodeBulkTime).count();
                decodeVariance += diff * diff;
            }
            
            result.metrics.timing.encodeTimeStdDev = nanoseconds(
                static_cast<long long>(std::sqrt(encodeVariance / (encodeTimes.size() - 1)))
            );
            result.metrics.timing.decodeBulkTimeStdDev = nanoseconds(
                static_cast<long long>(std::sqrt(decodeVariance / (decodeTimes.size() - 1)))
            );
        }
        
        // Throughput calculations
        double encodeSeconds = result.metrics.timing.encodeTime.count() / 1e9;
        double decodeSeconds = result.metrics.timing.decodeBulkTime.count() / 1e9;
        
        if (encodeSeconds > 0) {
            result.metrics.timing.encodeElementsPerSecond = result.dataSize / encodeSeconds;
            result.metrics.timing.encodeThroughputMBps = 
                (result.metrics.memory.originalSize / (1024.0 * 1024.0)) / encodeSeconds;
        }
        
        if (decodeSeconds > 0) {
            result.metrics.timing.decodeBulkElementsPerSecond = result.dataSize / decodeSeconds;
            result.metrics.timing.decodeBulkThroughputMBps = 
                (result.metrics.memory.originalSize / (1024.0 * 1024.0)) / decodeSeconds;
        }
    }
    
    void validateCorrectness(const std::vector<T>& original,
                            const std::vector<T>& decoded,
                            AccuracyMetrics& accuracy) {
        if (original.size() != decoded.size()) {
            accuracy.isLossless = false;
            accuracy.mismatchCount = std::max(original.size(), decoded.size());
            return;
        }
        
        bool allMatch = true;
        for (size_t i = 0; i < original.size(); ++i) {
            if (original[i] != decoded[i]) {
                allMatch = false;
                accuracy.mismatchCount++;
            }
        }
        
        accuracy.isLossless = allMatch;
    }
    
    void benchmarkRandomAccess(std::shared_ptr<encodings::Codec<T>> encoder,
                               const encodings::EncodedData& encoded,
                               const std::vector<T>& original,
                               BenchmarkMetrics& metrics) {
        if (!encoder->properties().has(encodings::EncodingProperty::RandomAccess)) {
            return;
        }
        
        // Determine indices to probe: explicit trace overrides sampled indices.
        std::vector<size_t> indices;
        if (!config_.randomAccessIndices.empty()) {
            indices = config_.randomAccessIndices;
        } else {
            indices.resize(original.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng_);
        }

        size_t samplesToTest = std::min(config_.randomAccessSamples, indices.size());
        samplesToTest = std::min(samplesToTest, original.size());
        if (samplesToTest == 0) {
            return;
        }
        std::vector<nanoseconds> accessTimes;
        accessTimes.reserve(samplesToTest);
        
        bool accessCorrect = true;
        
        for (size_t i = 0; i < samplesToTest; ++i) {
            size_t idx = indices[i];
            
            auto start = high_resolution_clock::now();
            auto value = encoder->decodeAt(encoded, idx);
            auto end = high_resolution_clock::now();
            
            accessTimes.push_back(duration_cast<nanoseconds>(end - start));
            
            // Validate if enabled
            if (config_.validateRandomAccess && value && *value != original[idx]) {
                accessCorrect = false;
            }
        }
        
    // Calculate statistics
    auto totalTime = std::accumulate(accessTimes.begin(), accessTimes.end(), nanoseconds{0});
    metrics.randomAccess.averageRandomAccessTime = totalTime / accessTimes.size();
    metrics.randomAccess.minRandomAccessTime = *std::min_element(accessTimes.begin(), accessTimes.end());
    metrics.randomAccess.maxRandomAccessTime = *std::max_element(accessTimes.begin(), accessTimes.end());
    metrics.randomAccess.randomAccessCount = samplesToTest;
        
        if (!accessCorrect) {
            metrics.accuracy.isLossless = false;
            std::cerr << "Warning: Random access validation failed for " 
                      << metrics.encoderName << std::endl;
        }
    }
    
    void benchmarkStridedAccess(std::shared_ptr<encodings::Codec<T>> encoder,
                               const encodings::EncodedData& encoded,
                               const std::vector<T>& original,
                               BenchmarkMetrics& metrics) {
        if (!encoder->properties().has(encodings::EncodingProperty::RandomAccess)) {
            return;
        }
        
        std::vector<nanoseconds> accessTimes;
        size_t count = 0;
        
        if (!config_.stridedAccessIndices.empty()) {
            for (size_t idx : config_.stridedAccessIndices) {
                if (idx >= original.size()) continue;
                auto start = high_resolution_clock::now();
                auto value = encoder->decodeAt(encoded, idx);
                auto end = high_resolution_clock::now();
                accessTimes.push_back(duration_cast<nanoseconds>(end - start));
                ++count;
                (void)value;
                if (count >= config_.stridedAccessSamples) break;
            }
        } else {
            for (size_t idx = 0; idx < original.size(); idx += config_.stride) {
                auto start = high_resolution_clock::now();
                auto value = encoder->decodeAt(encoded, idx);
                auto end = high_resolution_clock::now();
                
                accessTimes.push_back(duration_cast<nanoseconds>(end - start));
                count++;
                
                (void)value;  // Suppress unused warning
                
                if (count >= config_.stridedAccessSamples) {
                    break;
                }
            }
        }

        if (!accessTimes.empty()) {
            auto totalTime = std::accumulate(accessTimes.begin(), accessTimes.end(), nanoseconds{0});
            metrics.randomAccess.averageStridedAccessTime = totalTime / accessTimes.size();
            metrics.randomAccess.stridedAccessCount = count;
            metrics.randomAccess.stride = config_.stridedAccessIndices.empty() ? config_.stride : 0;
        }
    }
    
    void benchmarkRangeAccess(std::shared_ptr<encodings::Codec<T>> encoder,
                             const encodings::EncodedData& encoded,
                             const std::vector<T>& original,
                             BenchmarkMetrics& metrics) {
        if (!encoder->properties().has(encodings::EncodingProperty::RandomAccess)) {
            return;
        }
        
        std::vector<nanoseconds> accessTimes;
        size_t totalRangeSize = 0;
        size_t queryCount = 0;
        
        if (!config_.rangeAccesses.empty()) {
            for (const auto& [startRaw, endRaw] : config_.rangeAccesses) {
                if (startRaw >= endRaw) continue;
                size_t start = std::min(startRaw, original.size());
                size_t end = std::min(endRaw, original.size());
                if (start >= end) continue;
                size_t rangeSize = end - start;

                auto accessStart = high_resolution_clock::now();
                auto range = encoder->decodeRange(encoded, start, end);
                auto accessEnd = high_resolution_clock::now();

                accessTimes.push_back(duration_cast<nanoseconds>(accessEnd - accessStart));
                totalRangeSize += rangeSize;
                queryCount++;
                if (queryCount >= config_.rangeQueryCount) break;
            }
        } else {
            for (size_t rangeSize : config_.rangeSizes) {
                if (rangeSize > original.size()) continue;
                
                for (size_t q = 0; q < config_.rangeQueryCount; ++q) {
                    // Random start position
                    std::uniform_int_distribution<size_t> dist(0, original.size() - rangeSize);
                    size_t start = dist(rng_);
                    size_t end = start + rangeSize;
                    
                    auto accessStart = high_resolution_clock::now();
                    auto range = encoder->decodeRange(encoded, start, end);
                    auto accessEnd = high_resolution_clock::now();
                    
                    accessTimes.push_back(duration_cast<nanoseconds>(accessEnd - accessStart));
                    totalRangeSize += rangeSize;
                    queryCount++;
                }
            }
        }
        
        if (!accessTimes.empty()) {
            auto totalTime = std::accumulate(accessTimes.begin(), accessTimes.end(), nanoseconds{0});
            metrics.randomAccess.averageRangeAccessTime = totalTime / accessTimes.size();
            metrics.randomAccess.rangeQueryCount = queryCount;
            metrics.randomAccess.averageRangeSize = totalRangeSize / queryCount;
        }
    }
};

} // namespace encodings::benchmark
