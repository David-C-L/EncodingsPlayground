#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <sstream>
#include "core/DataType.hpp"

namespace encodings::benchmark {

using namespace std::chrono;

/**
 * @brief Timing measurements for a benchmark run
 */
struct TimingMetrics {
    nanoseconds encodeTime{0};
    nanoseconds decodeBulkTime{0};
    nanoseconds decodeRandomAccessTime{0};
    nanoseconds decodeStridedAccessTime{0};
    nanoseconds decodeRangeAccessTime{0};
    
    // Statistical measures (if multiple runs)
    std::optional<nanoseconds> encodeTimeStdDev;
    std::optional<nanoseconds> decodeBulkTimeStdDev;
    std::optional<nanoseconds> decodeRandomAccessTimeStdDev;
    std::optional<nanoseconds> decodeStridedAccessTimeStdDev;
    std::optional<nanoseconds> decodeRangeAccessTimeStdDev;
    
    // Per-element throughput
    double encodeElementsPerSecond{0.0};
    double decodeBulkElementsPerSecond{0.0};
    
    // Throughput in MB/s
    double encodeThroughputMBps{0.0};
    double decodeBulkThroughputMBps{0.0};
};

/**
 * @brief Memory usage measurements
 */
struct MemoryMetrics {
    size_t originalSize{0};        // Size of unencoded data in bytes
    size_t encodedSize{0};         // Size of encoded data in bytes
    size_t peakMemoryUsage{0};     // Peak memory during encoding/decoding
    size_t encoderOverhead{0};     // Additional memory used by encoder state
    
    double compressionRatio() const {
        return originalSize > 0 
            ? static_cast<double>(encodedSize) / originalSize 
            : 0.0;
    }
    
    double spaceSavingsPercent() const {
        return originalSize > 0
            ? (1.0 - compressionRatio()) * 100.0
            : 0.0;
    }
    
    double bitsPerElement(size_t elementCount) const {
        return elementCount > 0
            ? (encodedSize * 8.0) / elementCount
            : 0.0;
    }
};

/**
 * @brief Accuracy/correctness metrics (for lossy encodings)
 */
struct AccuracyMetrics {
    bool isLossless{true};
    double maxAbsoluteError{0.0};
    double meanAbsoluteError{0.0};
    double meanSquaredError{0.0};
    size_t mismatchCount{0};
    
    double signalToNoiseRatio() const {
        // SNR calculation for lossy encodings
        return 0.0; // To be implemented based on data
    }
};

/**
 * @brief Random access performance metrics
 */
struct RandomAccessMetrics {
    // Random access (shuffled indices)
    nanoseconds averageRandomAccessTime{0};
    nanoseconds minRandomAccessTime{0};
    nanoseconds maxRandomAccessTime{0};
    size_t randomAccessCount{0};
    
    // Strided access pattern
    nanoseconds averageStridedAccessTime{0};
    size_t stridedAccessCount{0};
    size_t stride{0};
    
    // Range queries
    nanoseconds averageRangeAccessTime{0};
    size_t rangeQueryCount{0};
    size_t averageRangeSize{0};
};

/**
 * @brief Complete benchmark results for a single encoding run
 */
struct BenchmarkMetrics {
    // Identifiers
    std::string encoderName;
    std::string generatorName;
    core::DataType dataType;
    size_t elementCount{0};
    
    // Performance metrics
    TimingMetrics timing;
    MemoryMetrics memory;
    AccuracyMetrics accuracy;
    RandomAccessMetrics randomAccess;
    
    // Benchmark configuration
    size_t iterations{1};
    size_t warmupRuns{0};
    
    // Timestamp
    system_clock::time_point timestamp;
    
    // Custom metrics for specific encodings
    std::map<std::string, double> customMetrics;
    
    /**
     * @brief Calculate overall performance score (higher is better)
     * Combines compression ratio and speed
     */
    double performanceScore() const {
        // Simple score: MB/s divided by compression ratio
        // Better compression with good speed = higher score
        double encodeMBps = static_cast<double>(memory.originalSize) / 
                           (1024.0 * 1024.0) /
                           (timing.encodeTime.count() / 1e9);
        double compressionBonus = 1.0 / (memory.compressionRatio() + 0.1);
        return encodeMBps * compressionBonus;
    }
    
    /**
     * @brief Print a summary of the metrics
     */
    std::string summary() const {
        std::ostringstream oss;
        oss << "Encoder: " << encoderName << "\n"
            << "Data Type: " << core::dataTypeToString(dataType) << "\n"
            << "Elements: " << elementCount << "\n"
            << "Compression Ratio: " << memory.compressionRatio() << "\n"
            << "Space Savings: " << memory.spaceSavingsPercent() << "%\n"
            << "Encode Time: " << timing.encodeTime.count() / 1e6 << " ms\n"
            << "Decode Time: " << timing.decodeBulkTime.count() / 1e6 << " ms\n";
        return oss.str();
    }
};

/**
 * @brief Configuration for benchmark runs
 */
struct BenchmarkConfig {
    // Data generation
    std::vector<size_t> dataSizes{1000, 10000, 100000};  // Sizes to test
    
    // Benchmarking
    size_t iterations{10};               // Number of benchmark iterations (for statistics)
    size_t warmupRuns{3};                // Warmup iterations (not measured)
    
    // Random access testing
    bool testRandomAccess{true};
    size_t randomAccessSamples{100};     // Number of random reads to test
    
    // Strided access testing
    bool testStridedAccess{true};
    size_t stridedAccessSamples{100};
    size_t stride{10};                   // Access every Nth element
    
    // Range query testing
    bool testRangeAccess{true};
    size_t rangeQueryCount{10};
    std::vector<size_t> rangeSizes{10, 100, 1000};  // Different range sizes to test
    
    // Validation
    bool validateCorrectness{true};      // Verify decoded data matches original
    bool validateRandomAccess{true};     // Verify random access against sequential decode
    double maxAcceptableError{0.0};      // For lossy encodings
    
    // Output
    bool verboseOutput{false};
    bool saveResults{true};
    std::string outputPath{"/home/david/Documents/PhD/symbol-store/EncodingsPlayground/Benchmarks/results"};
};

} // namespace encodings::benchmark
