#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#ifdef __linux__
#include <malloc.h>
#endif
#include "BenchmarkMetrics.hpp"
#include "MemoryTracker.hpp"
#include "AllocationTracker.hpp"
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
#ifdef VTUNE_ENABLED
        __itt_pause();  // suppress collection during setup; phases opt-in below
#endif
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
        encoder->reset();   // re-run encoder selection for this dataset
#ifdef VTUNE_ENABLED
        if (config_.vtune.dataLoad) __itt_resume();
#endif
        result.originalData = generator->generate(dataSize);
#ifdef VTUNE_ENABLED
        if (config_.vtune.dataLoad) __itt_pause();
#endif
        
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
        std::vector<SubStreamMetrics> substreamAccum; // filled on first iteration with subStream data

        for (size_t i = 0; i < config_.iterations; ++i) {
            if (config_.verboseOutput) {
                std::cout << "  [" << encoderName << "/" << datasetName << "] encode..." << std::flush;
            }
            // Measure encoding
#ifdef VTUNE_ENABLED
            if (config_.vtune.encode) __itt_resume();
#endif
            auto encodeStart = high_resolution_clock::now();
            encoded = encoder->encode(result.originalData);
            auto encodeEnd = high_resolution_clock::now();
#ifdef VTUNE_ENABLED
            if (config_.vtune.encode) __itt_pause();
#endif
            encodeTimes.push_back(duration_cast<nanoseconds>(encodeEnd - encodeStart));
            if (config_.verboseOutput) {
                std::cout << " done ("
                          << duration_cast<nanoseconds>(encodeEnd - encodeStart).count() / 1'000'000.0
                          << " ms)\n";
            }

            // Propagate all numeric customMetadata entries into customMetrics.
            for (const auto& [key, strVal] : encoded.metadata().customMetadata) {
                try {
                    result.metrics.customMetrics[key] = std::stod(strVal);
                } catch (...) {}
            }

            // Accumulate per-section encode metrics (only populated by profiling variants)
            const auto& ssEnc = encoded.metadata().subStreamEncodeMetrics;
            if (!ssEnc.empty()) {
                if (substreamAccum.empty()) substreamAccum.resize(ssEnc.size());
                for (size_t s = 0; s < ssEnc.size(); ++s) {
                    substreamAccum[s].name         = ssEnc[s].name;
                    substreamAccum[s].bitWidth     = ssEnc[s].bitWidth;
                    substreamAccum[s].encodedBytes = ssEnc[s].encodedBytes;
                    substreamAccum[s].encodeTime_ns += static_cast<double>(ssEnc[s].encodeTime_ns);
                }
            }

            if (config_.verboseOutput) {
                std::cout << "  [" << encoderName << "/" << datasetName << "] decode (bulk)..." << std::flush;
            }
            // Measure decoding
#ifdef VTUNE_ENABLED
            if (config_.vtune.decode) __itt_resume();
#endif
            auto decodeStart = high_resolution_clock::now();
            auto decoded = encoder->decodeAll(encoded);
            auto decodeEnd = high_resolution_clock::now();
#ifdef VTUNE_ENABLED
            if (config_.vtune.decode) __itt_pause();
#endif
            decodeTimes.push_back(duration_cast<nanoseconds>(decodeEnd - decodeStart));

            // Accumulate per-section bulk decode times
            const auto& bulkTimes = encoder->subStreamBulkDecodeTimeNs();
            for (size_t s = 0; s < substreamAccum.size() && s < bulkTimes.size(); ++s)
                substreamAccum[s].decodeBulkTime_ns += static_cast<double>(bulkTimes[s]);
            if (config_.verboseOutput) {
                std::cout << " done ("
                          << duration_cast<nanoseconds>(decodeEnd - decodeStart).count() / 1'000'000.0
                          << " ms)\n";
            }

            // Validation on first iteration
            if (i == 0 && config_.validateCorrectness) {
                validateCorrectness(result.originalData, decoded, result.metrics.accuracy);
            }
        }

        result.encodedData = encoded;

        // Calculate timing statistics
        calculateTimingStats(encodeTimes, decodeTimes, result);

        // Average per-section encode/decode times across iterations
        if (!substreamAccum.empty()) {
            const double iters = static_cast<double>(config_.iterations);
            for (auto& ss : substreamAccum) {
                ss.encodeTime_ns     /= iters;
                ss.decodeBulkTime_ns /= iters;
            }
        }

        // Memory metrics
        result.metrics.memory.originalSize = dataSize * sizeof(T);
        result.metrics.memory.encodedSize = encoded.size();

        // Random access benchmarks — reset per-section and reordering accumulators first
        // so we capture only the access-pattern phase (not the warmup/timing iterations).
        encoder->resetSubStreamDecodeAtAccum();
        encoder->resetReorderingProfilingAccum();
        if (config_.testRandomAccess) {
            if (config_.verboseOutput) {
                std::cout << "  [" << encoderName << "/" << datasetName << "] random access..." << std::flush;
            }
            benchmarkRandomAccess(encoder, encoded, result.originalData, result.metrics);
            if (config_.verboseOutput) { std::cout << " done\n"; }
        }
        // Read per-section random-access timing (divide by number of decodeAt calls made)
        {
            const auto& atTimes = encoder->subStreamDecodeAtAccumNs();
            const size_t atCount = std::max<size_t>(1, result.metrics.randomAccess.randomAccessCount);
            for (size_t s = 0; s < substreamAccum.size() && s < atTimes.size(); ++s)
                substreamAccum[s].decodeAtTime_ns = static_cast<double>(atTimes[s]) / atCount;
        }

        if (config_.testStridedAccess) {
            if (config_.verboseOutput) {
                std::cout << "  [" << encoderName << "/" << datasetName << "] strided access..." << std::flush;
            }
            benchmarkStridedAccess(encoder, encoded, result.originalData, result.metrics);
            if (config_.verboseOutput) { std::cout << " done\n"; }
        }

        // Range access — reset decodeRange accumulator before the loop.
        encoder->resetSubStreamDecodeRangeAccum();
        if (config_.testRangeAccess) {
            if (config_.verboseOutput) {
                std::cout << "  [" << encoderName << "/" << datasetName << "] range access..." << std::flush;
            }
            benchmarkRangeAccess(encoder, encoded, result.originalData, result.metrics);
            if (config_.verboseOutput) { std::cout << " done\n"; }
        }
        // Read per-section range-access timing (divide by number of decodeRange calls made)
        {
            const auto& rangeTimes = encoder->subStreamDecodeRangeAccumNs();
            const size_t rangeCount = std::max<size_t>(1, result.metrics.randomAccess.rangeQueryCount);
            for (size_t s = 0; s < substreamAccum.size() && s < rangeTimes.size(); ++s)
                substreamAccum[s].decodeRangeTime_ns = static_cast<double>(rangeTimes[s]) / rangeCount;
        }

        // Selective/gather access — reset gather profiling accumulators first.
        encoder->resetGatherProfilingAccum();
        if (config_.testSelectiveAccess) {
            if (config_.verboseOutput) {
                std::cout << "  [" << encoderName << "/" << datasetName << "] selective (gather) access..." << std::flush;
            }
            benchmarkSelectiveAccess(encoder, encoded, result.originalData, result.metrics);
            if (config_.verboseOutput) { std::cout << " done\n"; }
        }

        // Store final sub-stream breakdown
        if (!substreamAccum.empty())
            result.metrics.subStreamMetrics = std::move(substreamAccum);

        // Reordering-layer profiling hooks (non-negative when ReorderingCodec<T,true>)
        if (auto ns = encoder->reorderEncodeTimeNs(); ns >= 0)
            result.metrics.customMetrics["reorder_encode_time_ns"] = static_cast<double>(ns);
        if (auto ns = encoder->unreorderDecodeAllTimeNs(); ns >= 0)
            result.metrics.customMetrics["unreorder_decode_all_time_ns"] = static_cast<double>(ns);
        if (auto ns = encoder->permLookupDecodeAtAccumNs(); ns >= 0)
            result.metrics.customMetrics["perm_lookup_decode_at_ns"] = static_cast<double>(ns);
        if (auto ns = encoder->permLookupDecodeRangeAccumNs(); ns >= 0)
            result.metrics.customMetrics["perm_lookup_decode_range_ns"] = static_cast<double>(ns);

        // ── Memory measurement pass ──────────────────────────────────────
        // Run each workload a second time, measuring heap usage with a background
        // sampler. This is kept entirely separate from the timing pass above so
        // that memory-tracking overhead cannot inflate timing results.
        if (config_.measureMemory) {
            if (config_.verboseOutput) {
                std::cout << "  [memory pass] measuring heap usage..." << std::endl;
            }
            measureMemoryUsage(encoder, result.originalData, result);
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
    
    // Models worst-case uniform-random point lookups (e.g. embedding-table /
    // DLRM-style sparse-selection-vector access), NOT Nimble's on-disk
    // TableScan selective-read pattern — for the latter see
    // benchmarkSelectiveAccess() below.
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

#ifdef VTUNE_ENABLED
        if (config_.vtune.randomAccess) __itt_resume();
#endif
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
#ifdef VTUNE_ENABLED
        if (config_.vtune.randomAccess) __itt_pause();
#endif
        
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

#ifdef VTUNE_ENABLED
        if (config_.vtune.stridedAccess) __itt_resume();
#endif
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
#ifdef VTUNE_ENABLED
        if (config_.vtune.stridedAccess) __itt_pause();
#endif

        if (!accessTimes.empty()) {
            auto totalTime = std::accumulate(accessTimes.begin(), accessTimes.end(), nanoseconds{0});
            metrics.randomAccess.averageStridedAccessTime = totalTime / accessTimes.size();
            metrics.randomAccess.stridedAccessCount = count;
            metrics.randomAccess.stride = config_.stridedAccessIndices.empty() ? config_.stride : 0;
        }
    }
    
    // ────────────────────────────────────────────────────────────────────
    // Memory-measurement pass
    // ────────────────────────────────────────────────────────────────────

    /**
     * @brief Re-runs every workload (encode, bulk decode, random/strided/range
     *        access) in a dedicated pass, measuring heap usage with a background
     *        PeakHeapTracker.  All results are written into result.metrics.memory.
     *
     * This method must be called *after* all timing measurements so that the
     * PeakHeapTracker's background thread (and any malloc_trim calls) cannot
     * interfere with latency measurements.
     */
    void measureMemoryUsage(
            std::shared_ptr<encodings::Codec<T>> encoder,
            const std::vector<T>& originalData,
            BenchmarkResult<T>& result) {

        const size_t n = originalData.size();

        // Helper: net heap bytes above `before` (0 if the heap shrank).
        auto netDelta = [](size_t after, size_t before) -> size_t {
            return after > before ? after - before : 0;
        };

        // Helper: release free arena pages back to the OS so that the baseline
        // reading is as tight as possible.  No-op on non-Linux.
        auto trimHeap = [] {
#ifdef __linux__
            malloc_trim(0);
#endif
        };

        // ── Phase 1: Encode ───────────────────────────────────────────────
        // Peak: ScopedAllocationTrack intercepts every operator new/delete so
        //       transient intermediate buffers are captured even if freed before
        //       encode() returns.
        // Net delta: mallinfo2 before/after gives the retained heap after the
        //       call (≈ the encoded output buffer + any retained encoder state).
        encodings::EncodedData encoded;
        {
            trimHeap();
            size_t heapBefore = currentHeapBytes();
            {
                ScopedAllocationTrack track;
                encoded = encoder->encode(originalData);
                result.metrics.memory.encodePeakHeapBytes = track.stop();
            }
            result.metrics.memory.encodeNetHeapDeltaBytes =
                netDelta(currentHeapBytes(), heapBefore);

            // Populate legacy aliases.
            result.metrics.memory.peakMemoryUsage = result.metrics.memory.encodePeakHeapBytes;
            size_t encodedSz = result.metrics.memory.encodedSize;
            result.metrics.memory.encoderOverhead =
                result.metrics.memory.encodeNetHeapDeltaBytes > encodedSz
                    ? result.metrics.memory.encodeNetHeapDeltaBytes - encodedSz
                    : 0;
        }

        // ── Phase 2: Bulk decode ──────────────────────────────────────────
        {
            trimHeap();
            size_t heapBefore = currentHeapBytes();
            {
                ScopedAllocationTrack track;
                auto decoded = encoder->decodeAll(encoded);
                result.metrics.memory.decodeBulkPeakHeapBytes = track.stop();
                (void)decoded;
            }
            result.metrics.memory.decodeBulkNetHeapDeltaBytes =
                netDelta(currentHeapBytes(), heapBefore);
        }

        // ── Phase 3: Random access ────────────────────────────────────────
        // decodeAt returns std::optional<T> — usually a stack-allocated value
        // with 0 heap allocation for simple encoders.  However, some encoders
        // (e.g. SubIntSplitEncoder when a section codec lacks RandomAccess)
        // internally call decodeAll(), allocating a large std::vector<T> that
        // is freed before decodeAt returns.  mallinfo2() before/after sees net
        // delta = 0 in that case.  ScopedAllocationTrack intercepts the
        // operator new/delete calls and captures the true intra-call peak.
        if (config_.testRandomAccess &&
            encoder->properties().has(encodings::EncodingProperty::RandomAccess)) {

            std::vector<size_t> indices;
            if (!config_.randomAccessIndices.empty()) {
                indices = config_.randomAccessIndices;
            } else {
                indices.resize(n);
                std::iota(indices.begin(), indices.end(), 0);
                std::mt19937 fixedRng(42);
                std::shuffle(indices.begin(), indices.end(), fixedRng);
            }
            size_t samplesToTest = std::min({config_.randomAccessSamples,
                                             indices.size(), n});

            if (samplesToTest > 0) {
                // Wrap the entire batch in one tracker: the peak over all calls
                // is the worst-case working memory for a single decodeAt.
                // (Consecutive calls free their buffers before the next starts,
                //  so the peak is that of one call, not the sum.)
                ScopedAllocationTrack track;
                for (size_t i = 0; i < samplesToTest; ++i) {
                    auto value = encoder->decodeAt(encoded, indices[i]);
                    (void)value;
                }
                result.metrics.memory.decodeRandomPeakHeapBytes = track.stop();
            }
        }

        // ── Phase 4: Strided access ───────────────────────────────────────
        if (config_.testStridedAccess &&
            encoder->properties().has(encodings::EncodingProperty::RandomAccess)) {

            ScopedAllocationTrack track;
            if (!config_.stridedAccessIndices.empty()) {
                size_t count = 0;
                for (size_t idx : config_.stridedAccessIndices) {
                    if (idx >= n) continue;
                    auto value = encoder->decodeAt(encoded, idx);
                    (void)value;
                    if (++count >= config_.stridedAccessSamples) break;
                }
            } else {
                size_t count = 0;
                for (size_t idx = 0; idx < n; idx += config_.stride) {
                    auto value = encoder->decodeAt(encoded, idx);
                    (void)value;
                    if (++count >= config_.stridedAccessSamples) break;
                }
            }
            result.metrics.memory.decodeStridedPeakHeapBytes = track.stop();
        }

        // ── Phase 5: Range access ─────────────────────────────────────────
        // decodeRange returns a std::vector<T>.  We take one ScopedAllocationTrack
        // over all queries; the peak is the worst-case working memory for one
        // range query (output buffer + any internal temporary allocations).
        if (config_.testRangeAccess &&
            encoder->properties().has(encodings::EncodingProperty::RandomAccess)) {

            ScopedAllocationTrack track;
            if (!config_.rangeAccesses.empty()) {
                size_t queryCount = 0;
                for (const auto& [startRaw, endRaw] : config_.rangeAccesses) {
                    if (startRaw >= endRaw) continue;
                    size_t start = std::min(startRaw, n);
                    size_t end   = std::min(endRaw,   n);
                    if (start >= end) continue;
                    auto range = encoder->decodeRange(encoded, start, end);
                    (void)range;
                    if (++queryCount >= config_.rangeQueryCount) break;
                }
            } else {
                std::mt19937 fixedRng(42);
                for (size_t rangeSize : config_.rangeSizes) {
                    if (rangeSize > n) continue;
                    for (size_t q = 0; q < config_.rangeQueryCount; ++q) {
                        std::uniform_int_distribution<size_t> dist(0, n - rangeSize);
                        size_t start = dist(fixedRng);
                        auto range = encoder->decodeRange(encoded, start, start + rangeSize);
                        (void)range;
                    }
                }
            }
            result.metrics.memory.decodeRangePeakHeapBytes = track.stop();
        }

        // ── Phase 6: Selective/gather access ─────────────────────────────
        // decodeGatherInto writes into a caller-supplied buffer; the peak here
        // is the worst-case working memory for one full gather call (output
        // buffer + any internal temporaries used to skip/materialize ranges).
        if (config_.testSelectiveAccess &&
            !config_.selectiveAccessRanges.empty() &&
            encoder->properties().has(encodings::EncodingProperty::RandomAccess)) {

            encodings::RowRangeList ranges;
            size_t totalSelected = 0;
            for (const auto& r : config_.selectiveAccessRanges) {
                size_t b = std::min(r.begin, n);
                size_t e = std::min(r.end, n);
                if (b >= e) continue;
                ranges.push_back({b, e});
                totalSelected += (e - b);
            }
            if (!ranges.empty()) {
                std::vector<T> dst(totalSelected);
                ScopedAllocationTrack track;
                encoder->decodeGatherInto(encoded, ranges, dst.data(), totalSelected);
                result.metrics.memory.decodeSelectivePeakHeapBytes = track.stop();
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────

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

        // Pre-allocate a persistent output buffer large enough for the largest range
        // query.  decodeRangeInto writes directly into this buffer, avoiding the
        // per-call heap allocation and zero-initialisation of decodeRange's vector.
        const size_t maxRangeSize = [&]() -> size_t {
            if (!config_.rangeAccesses.empty()) {
                size_t mx = 0;
                for (const auto& [s, e] : config_.rangeAccesses)
                    mx = std::max(mx, e > s ? e - s : size_t{0});
                return std::min(mx, original.size());
            }
            if (!config_.rangeSizes.empty())
                return *std::max_element(config_.rangeSizes.begin(), config_.rangeSizes.end());
            return original.size();
        }();
        std::vector<T> rangeBuf(maxRangeSize);

#ifdef VTUNE_ENABLED
        if (config_.vtune.rangeAccess) __itt_resume();
#endif
        if (!config_.rangeAccesses.empty()) {
            for (const auto& [startRaw, endRaw] : config_.rangeAccesses) {
                if (startRaw >= endRaw) continue;
                size_t start = std::min(startRaw, original.size());
                size_t end = std::min(endRaw, original.size());
                if (start >= end) continue;
                size_t rangeSize = end - start;

                auto accessStart = high_resolution_clock::now();
                encoder->decodeRangeInto(encoded, start, end, rangeBuf.data(), rangeSize);
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
                    std::uniform_int_distribution<size_t> dist(0, original.size() - rangeSize);
                    size_t start = dist(rng_);
                    size_t end = start + rangeSize;

                    auto accessStart = high_resolution_clock::now();
                    encoder->decodeRangeInto(encoded, start, end, rangeBuf.data(), rangeSize);
                    auto accessEnd = high_resolution_clock::now();

                    accessTimes.push_back(duration_cast<nanoseconds>(accessEnd - accessStart));
                    totalRangeSize += rangeSize;
                    queryCount++;
                }
            }
        }
        
#ifdef VTUNE_ENABLED
        if (config_.vtune.rangeAccess) __itt_pause();
#endif
        if (!accessTimes.empty()) {
            auto totalTime = std::accumulate(accessTimes.begin(), accessTimes.end(), nanoseconds{0});
            metrics.randomAccess.averageRangeAccessTime = totalTime / accessTimes.size();
            metrics.randomAccess.rangeQueryCount = queryCount;
            metrics.randomAccess.averageRangeSize = totalRangeSize / queryCount;
        }
    }

    // ────────────────────────────────────────────────────────────────────

    // Models a TableScan-style selective read: an ascending, non-overlapping
    // list of surviving row ranges (config_.selectiveAccessRanges), decoded
    // via one decodeGatherInto() call that skips the gaps between ranges
    // rather than materializing them. See RandomAccessMetrics's doc-comment
    // for why this is a distinct pattern from benchmarkRandomAccess above.
    void benchmarkSelectiveAccess(std::shared_ptr<encodings::Codec<T>> encoder,
                                   const encodings::EncodedData& encoded,
                                   const std::vector<T>& original,
                                   BenchmarkMetrics& metrics) {
        if (!encoder->properties().has(encodings::EncodingProperty::RandomAccess)) {
            return;
        }
        if (config_.selectiveAccessRanges.empty()) {
            return;
        }

        // Clamp ranges to original.size(), same idiom as benchmarkRangeAccess's
        // rangeAccesses handling.
        encodings::RowRangeList ranges;
        size_t totalSelected = 0;
        for (const auto& r : config_.selectiveAccessRanges) {
            size_t b = std::min(r.begin, original.size());
            size_t e = std::min(r.end, original.size());
            if (b >= e) continue;
            ranges.push_back({b, e});
            totalSelected += (e - b);
        }
        if (ranges.empty()) {
            return;
        }
        const size_t totalSpanned = ranges.back().end - ranges.front().begin;

        std::vector<T> dst(totalSelected);

        encoder->resetGatherProfilingAccum();

#ifdef VTUNE_ENABLED
        if (config_.vtune.selectiveAccess) __itt_resume();
#endif
        auto start = high_resolution_clock::now();
        encoder->decodeGatherInto(encoded, ranges, dst.data(), totalSelected);
        auto end = high_resolution_clock::now();
#ifdef VTUNE_ENABLED
        if (config_.vtune.selectiveAccess) __itt_pause();
#endif

        metrics.selectiveAccess.totalGatherTime = duration_cast<nanoseconds>(end - start);
        metrics.selectiveAccess.rangeCount = ranges.size();
        metrics.selectiveAccess.totalSelectedRows = totalSelected;
        metrics.selectiveAccess.selectivity =
            totalSpanned ? static_cast<double>(totalSelected) / totalSpanned : 0.0;
        metrics.selectiveAccess.meanRunLength =
            static_cast<double>(totalSelected) / ranges.size();

        auto skipNs = encoder->gatherSkipTimeNs();
        auto matNs  = encoder->gatherMaterializeTimeNs();
        if (skipNs >= 0 && matNs >= 0) {
            metrics.selectiveAccess.skipMaterializeSplitAvailable = true;
            metrics.selectiveAccess.averageSkipTimeNs = nanoseconds(skipNs);
            metrics.selectiveAccess.averageMaterializeTimeNs = nanoseconds(matNs);
        }

        if (config_.validateRandomAccess) {
            bool ok = true;
            size_t off = 0;
            for (const auto& r : ranges) {
                for (size_t i = r.begin; i < r.end; ++i, ++off) {
                    if (dst[off] != original[i]) { ok = false; break; }
                }
                if (!ok) break;
            }
            if (!ok) {
                metrics.accuracy.isLossless = false;
                std::cerr << "Warning: Selective/gather access validation failed for "
                          << metrics.encoderName << std::endl;
            }
        }
    }
};

} // namespace encodings::benchmark
