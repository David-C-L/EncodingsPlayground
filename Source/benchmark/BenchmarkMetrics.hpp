#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <sstream>
#include "core/DataType.hpp"
#include "encodings/RowRange.hpp"
#ifdef VTUNE_ENABLED
#include <ittnotify.h>
#endif

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
 *
 * Fields are split into two groups:
 *  1. Static sizes (always populated): originalSize, encodedSize.
 *  2. Dynamic heap measurements (populated only when
 *     BenchmarkConfig::measureMemory is true, in a pass separate from timing).
 *
 * "Peak" fields report the highest heap usage above the pre-operation baseline
 * as sampled via ScopedAllocationTrack around the operation.
 * "NetDelta" fields report the net change in live heap bytes between the start
 * and the end of the operation (i.e. allocations that were not freed before
 * the operation returned, such as the output buffer itself).
 */
struct MemoryMetrics {
    size_t originalSize{0};        ///< Size of unencoded data in bytes
    size_t encodedSize{0};         ///< Size of encoded data in bytes

    // ── Dynamic heap measurements (memory-measurement pass) ──────────────
    /// Peak heap above baseline during encode (includes all intermediate buffers).
    size_t encodePeakHeapBytes{0};
    /// Net heap change at end of encode (≈ encodedSize + any retained encoder state).
    size_t encodeNetHeapDeltaBytes{0};

    /// Peak heap above baseline during full (bulk) decode.
    size_t decodeBulkPeakHeapBytes{0};
    /// Net heap change at end of bulk decode (≈ decoded output size).
    size_t decodeBulkNetHeapDeltaBytes{0};

    /// Peak heap above baseline while executing all random-access decode calls.
    size_t decodeRandomPeakHeapBytes{0};
    /// Peak heap above baseline while executing all strided-access decode calls.
    size_t decodeStridedPeakHeapBytes{0};
    /// Peak heap above baseline while executing all range-access decode calls.
    size_t decodeRangePeakHeapBytes{0};
    /// Peak heap above baseline while executing the selective/gather decode call.
    size_t decodeSelectivePeakHeapBytes{0};

    // ── Legacy aliases kept for back-compat with existing JSON consumers ──
    /// Alias for encodePeakHeapBytes (populated alongside it).
    size_t peakMemoryUsage{0};
    /// Net encoder heap above the encoded output: encodeNetHeapDeltaBytes − encodedSize (≥ 0).
    size_t encoderOverhead{0};

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
 *
 * NOTE: these three patterns (shuffled point lookups, fixed-stride point
 * lookups, independent range queries) model worst-case uniform-random or
 * regularly-strided point access — e.g. an in-memory embedding-table /
 * DLRM-style sparse-selection-vector lookup. They do NOT model an on-disk
 * TableScan's filtered/selective read, where surviving rows form an ordered,
 * ascending list of contiguous ranges with gaps that are skipped rather than
 * scattered across the whole index space. For that pattern, see
 * SelectiveAccessMetrics / benchmarkSelectiveAccess below.
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
 * @brief Selective/gather-read performance metrics.
 *
 * Models a TableScan-style filtered read: an ordered, ascending, non-overlapping
 * list of surviving row ranges (RowRangeList), decoded in one decodeGatherInto()
 * call that skips the gaps between ranges rather than materializing them.
 * When the codec being measured overrides Codec::gatherSkipTimeNs() /
 * gatherMaterializeTimeNs() (currently only SubIntSplitEncoder), the skip vs.
 * materialize time split is also available — this is the quantity the
 * skip-latency argument cares about.
 */
struct SelectiveAccessMetrics {
    nanoseconds totalGatherTime{0};     ///< one decodeGatherInto() call over the whole trace
    size_t rangeCount{0};               ///< number of surviving ranges in the trace
    size_t totalSelectedRows{0};        ///< sum of range sizes (rows actually decoded)
    double selectivity{0.0};            ///< totalSelectedRows / (span from first range's begin to last range's end)
    double meanRunLength{0.0};          ///< totalSelectedRows / rangeCount

    /// True only if the codec overrode gatherSkipTimeNs()/gatherMaterializeTimeNs()
    /// (i.e. reported >= 0 for both) during the most recent run.
    bool skipMaterializeSplitAvailable{false};
    nanoseconds averageSkipTimeNs{0};
    nanoseconds averageMaterializeTimeNs{0};
};

/**
 * @brief Per-sub-stream breakdown metrics (non-empty only for SubIntSplitEncoder profiling variants)
 */
struct SubStreamMetrics {
    std::string name;                   ///< sub-codec name, e.g. "Dictionary(13bit)"
    uint8_t     bitWidth{0};            ///< bit width of this section
    size_t      encodedBytes{0};        ///< encoded bytes for this section
    double      encodeTime_ns{0.0};     ///< average encode time per iteration (ns)
    double      decodeBulkTime_ns{0.0}; ///< average bulk-decode time per iteration (ns)
    double      decodeAtTime_ns{0.0};   ///< average per-decodeAt call time (ns)
    double      decodeRangeTime_ns{0.0};///< average per-decodeRange call time (ns)
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
    SelectiveAccessMetrics selectiveAccess;
    
    // Benchmark configuration
    size_t iterations{1};
    size_t warmupRuns{0};
    
    // Timestamp
    system_clock::time_point timestamp;
    
    // Custom metrics for specific encodings
    std::map<std::string, double> customMetrics;

    // Per-sub-stream breakdown (non-empty only for SubIntSplitEncoder<T, true>)
    std::vector<SubStreamMetrics> subStreamMetrics;
    
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
 * @brief Selects which benchmark phases VTune pause/resume brackets when
 *        VTUNE_ENABLED is defined.  All phases default to true (instrument
 *        everything); set individual flags to false to exclude a phase.
 */
struct VTuneConfig {
    bool dataLoad{true};
    bool encode{true};
    bool decode{true};
    bool randomAccess{true};
    bool stridedAccess{true};
    bool rangeAccess{true};
    bool selectiveAccess{true};
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
    // Optional explicit random-access trace. If non-empty, this overrides sampling.
    std::vector<size_t> randomAccessIndices;
    
    // Strided access testing
    bool testStridedAccess{true};
    size_t stridedAccessSamples{100};
    size_t stride{10};                   // Access every Nth element
    // Optional explicit strided-access trace. If non-empty, this overrides stride sampling.
    std::vector<size_t> stridedAccessIndices;
    
    // Range query testing
    bool testRangeAccess{true};
    size_t rangeQueryCount{10};
    std::vector<size_t> rangeSizes{10, 100, 1000};  // Different range sizes to test
    // Optional explicit range-access trace (start, end) with end exclusive.
    std::vector<std::pair<size_t, size_t>> rangeAccesses;

    // Selective/gather access testing (models a TableScan filtered read: an
    // ascending, non-overlapping list of surviving row ranges with gaps to
    // skip between them). Opt-in per scenario — unlike random/strided/range,
    // there is no implicit default trace, since the point of this benchmark
    // is sensitivity analysis over an explicitly constructed (selectivity,
    // meanRunLength) trace; see SelectiveTraceGen.hpp::makeSelectiveTrace().
    bool testSelectiveAccess{true};
    encodings::RowRangeList selectiveAccessRanges;

    // Validation
    bool validateCorrectness{true};      // Verify decoded data matches original
    bool validateRandomAccess{true};     // Verify random access against sequential decode
    double maxAcceptableError{0.0};      // For lossy encodings
    
    // Output
    bool verboseOutput{false};
    bool saveResults{true};
    std::string outputPath{"/home/david/Documents/PhD/symbol-store/MetaNimbleProject/EncodingsPlayground/Benchmarks/results"};

    // Memory measurement
    /// When true, each benchmark is run a second time (after timing) to measure
    /// heap usage per phase. The extra pass does not affect timing results.
    bool measureMemory{true};
    /// Seed for every random choice this runner makes (random-access indices,
    /// range starts).  It used to seed from std::random_device while the memory
    /// pass used a hardcoded mt19937(42), so the timing pass and the memory pass
    /// measured different ranges and no run could be reproduced.
    uint64_t seed{42};

    // VTune instrumentation (only active when built with -DVTUNE_ENABLED=ON)
    VTuneConfig vtune{};
};

} // namespace encodings::benchmark
