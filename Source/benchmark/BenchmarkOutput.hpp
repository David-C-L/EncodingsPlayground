#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include "BenchmarkMetrics.hpp"
#include "BenchmarkRunner.hpp"

namespace encodings::benchmark {

/**
 * @brief Simple JS                  << std::setw(12) << std::fixed << std::setprecision(2)
                    << result.metrics.memory.compressionRatio()
                << std::setw(15) << std::fixed << std::setprecision(2)
                    << result.metrics.memory.bitsPerElement(result.metrics.elementCount)
                << "\n";rialization for benchmark results
 */
class JSONWriter {
public:
    static std::string escape(const std::string& str) {
        std::string result;
        for (char c : str) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c;
            }
        }
        return result;
    }
    
    static std::string toJSON(const TimingMetrics& timing) {
        std::ostringstream ss;
        ss << "{\n";
        ss << "      \"encodeTime_ns\": " << timing.encodeTime.count() << ",\n";
        ss << "      \"decodeBulkTime_ns\": " << timing.decodeBulkTime.count() << ",\n";
        ss << "      \"decodeRandomAccessTime_ns\": " << timing.decodeRandomAccessTime.count() << ",\n";
        ss << "      \"decodeStridedAccessTime_ns\": " << timing.decodeStridedAccessTime.count() << ",\n";
        ss << "      \"decodeRangeAccessTime_ns\": " << timing.decodeRangeAccessTime.count() << ",\n";
        
        if (timing.encodeTimeStdDev) {
            ss << "      \"encodeTimeStdDev_ns\": " << timing.encodeTimeStdDev->count() << ",\n";
        }
        if (timing.decodeBulkTimeStdDev) {
            ss << "      \"decodeBulkTimeStdDev_ns\": " << timing.decodeBulkTimeStdDev->count() << ",\n";
        }
        
        ss << "      \"encodeElementsPerSecond\": " << timing.encodeElementsPerSecond << ",\n";
        ss << "      \"decodeBulkElementsPerSecond\": " << timing.decodeBulkElementsPerSecond << ",\n";
        ss << "      \"encodeThroughputMBps\": " << timing.encodeThroughputMBps << ",\n";
        ss << "      \"decodeBulkThroughputMBps\": " << timing.decodeBulkThroughputMBps << "\n";
        ss << "    }";
        return ss.str();
    }
    
    static std::string toJSON(const MemoryMetrics& memory, size_t elementCount) {
        std::ostringstream ss;
        ss << "{\n";
        ss << "      \"originalSize\": " << memory.originalSize << ",\n";
        ss << "      \"encodedSize\": " << memory.encodedSize << ",\n";
        ss << "      \"compressionRatio\": " << memory.compressionRatio() << ",\n";
        ss << "      \"bitsPerElement\": " << memory.bitsPerElement(elementCount) << ",\n";
        // Dynamic heap measurements (zero when measureMemory was disabled)
        ss << "      \"encodePeakHeapBytes\": " << memory.encodePeakHeapBytes << ",\n";
        ss << "      \"encodeNetHeapDeltaBytes\": " << memory.encodeNetHeapDeltaBytes << ",\n";
        ss << "      \"encoderOverhead\": " << memory.encoderOverhead << ",\n";
        ss << "      \"decodeBulkPeakHeapBytes\": " << memory.decodeBulkPeakHeapBytes << ",\n";
        ss << "      \"decodeBulkNetHeapDeltaBytes\": " << memory.decodeBulkNetHeapDeltaBytes << ",\n";
        ss << "      \"decodeRandomPeakHeapBytes\": " << memory.decodeRandomPeakHeapBytes << ",\n";
        ss << "      \"decodeStridedPeakHeapBytes\": " << memory.decodeStridedPeakHeapBytes << ",\n";
        ss << "      \"decodeRangePeakHeapBytes\": " << memory.decodeRangePeakHeapBytes << ",\n";
        ss << "      \"decodeSelectivePeakHeapBytes\": " << memory.decodeSelectivePeakHeapBytes << "\n";
        ss << "    }";
        return ss.str();
    }
    
    static std::string toJSON(const AccuracyMetrics& accuracy) {
        std::ostringstream ss;
        ss << "{\n";
        ss << "      \"isLossless\": " << (accuracy.isLossless ? "true" : "false") << ",\n";
        ss << "      \"mismatchCount\": " << accuracy.mismatchCount << ",\n";
        ss << "      \"maxAbsoluteError\": " << accuracy.maxAbsoluteError << ",\n";
        ss << "      \"meanAbsoluteError\": " << accuracy.meanAbsoluteError << ",\n";
        ss << "      \"meanSquaredError\": " << accuracy.meanSquaredError << "\n";
        ss << "    }";
        return ss.str();
    }
    
    static std::string toJSON(const RandomAccessMetrics& randomAccess) {
        std::ostringstream ss;
        ss << "{\n";
        ss << "      \"averageRandomAccessTime_ns\": " << randomAccess.averageRandomAccessTime.count() << ",\n";
        ss << "      \"minRandomAccessTime_ns\": " << randomAccess.minRandomAccessTime.count() << ",\n";
        ss << "      \"maxRandomAccessTime_ns\": " << randomAccess.maxRandomAccessTime.count() << ",\n";
        ss << "      \"randomAccessCount\": " << randomAccess.randomAccessCount << ",\n";
        ss << "      \"averageStridedAccessTime_ns\": " << randomAccess.averageStridedAccessTime.count() << ",\n";
        ss << "      \"stridedAccessCount\": " << randomAccess.stridedAccessCount << ",\n";
        ss << "      \"stride\": " << randomAccess.stride << ",\n";
        ss << "      \"averageRangeAccessTime_ns\": " << randomAccess.averageRangeAccessTime.count() << ",\n";
        ss << "      \"rangeQueryCount\": " << randomAccess.rangeQueryCount << ",\n";
        ss << "      \"averageRangeSize\": " << randomAccess.averageRangeSize << "\n";
        ss << "    }";
        return ss.str();
    }

    static std::string toJSON(const SelectiveAccessMetrics& sel) {
        std::ostringstream ss;
        ss << "{\n";
        ss << "      \"totalGatherTime_ns\": " << sel.totalGatherTime.count() << ",\n";
        ss << "      \"rangeCount\": " << sel.rangeCount << ",\n";
        ss << "      \"totalSelectedRows\": " << sel.totalSelectedRows << ",\n";
        ss << "      \"selectivity\": " << sel.selectivity << ",\n";
        ss << "      \"meanRunLength\": " << sel.meanRunLength << ",\n";
        ss << "      \"skipMaterializeSplitAvailable\": " << (sel.skipMaterializeSplitAvailable ? "true" : "false");
        if (sel.skipMaterializeSplitAvailable) {
            ss << ",\n      \"averageSkipTimeNs\": " << sel.averageSkipTimeNs.count()
               << ",\n      \"averageMaterializeTimeNs\": " << sel.averageMaterializeTimeNs.count();
        }
        ss << "\n    }";
        return ss.str();
    }

    static std::string toJSON(const SubStreamMetrics& ss) {
        std::ostringstream o;
        o << "{\n";
        o << "        \"name\": \""            << escape(ss.name)           << "\",\n";
        o << "        \"bitWidth\": "          << static_cast<int>(ss.bitWidth) << ",\n";
        o << "        \"encodedBytes\": "      << ss.encodedBytes            << ",\n";
        o << "        \"encodeTime_ns\": "     << ss.encodeTime_ns           << ",\n";
        o << "        \"decodeBulkTime_ns\": " << ss.decodeBulkTime_ns       << ",\n";
        o << "        \"decodeAtTime_ns\": "   << ss.decodeAtTime_ns         << ",\n";
        o << "        \"decodeRangeTime_ns\": "<< ss.decodeRangeTime_ns      << "\n";
        o << "      }";
        return o.str();
    }

    static std::string toJSON(const BenchmarkMetrics& metrics) {
        std::ostringstream ss;
        ss << "  {\n";
        ss << "    \"encoderName\": \"" << escape(metrics.encoderName) << "\",\n";
        ss << "    \"generatorName\": \"" << escape(metrics.generatorName) << "\",\n";
        ss << "    \"elementCount\": " << metrics.elementCount << ",\n";
        ss << "    \"iterations\": " << metrics.iterations << ",\n";
        ss << "    \"warmupRuns\": " << metrics.warmupRuns << ",\n";
        
        auto timeT = system_clock::to_time_t(metrics.timestamp);
        ss << "    \"timestamp\": \"" << std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S") << "\",\n";
        
        ss << "    \"timing\": " << toJSON(metrics.timing) << ",\n";
        ss << "    \"memory\": " << toJSON(metrics.memory, metrics.elementCount) << ",\n";
        ss << "    \"accuracy\": " << toJSON(metrics.accuracy) << ",\n";
        ss << "    \"randomAccess\": " << toJSON(metrics.randomAccess) << ",\n";
        ss << "    \"selectiveAccess\": " << toJSON(metrics.selectiveAccess);
        if (!metrics.subStreamMetrics.empty()) {
            ss << ",\n    \"subStreamMetrics\": [\n";
            for (size_t i = 0; i < metrics.subStreamMetrics.size(); ++i) {
                ss << "      " << toJSON(metrics.subStreamMetrics[i]);
                if (i + 1 < metrics.subStreamMetrics.size()) ss << ",";
                ss << "\n";
            }
            ss << "    ]";
        }
        if (!metrics.customMetrics.empty()) {
            ss << ",\n    \"customMetrics\": {\n";
            size_t idx = 0;
            for (const auto& [key, value] : metrics.customMetrics) {
                ss << "      \"" << escape(key) << "\": " << value;
                if (idx + 1 < metrics.customMetrics.size()) {
                    ss << ",";
                }
                ss << "\n";
                ++idx;
            }
            ss << "    }\n";
        } else {
            ss << "\n";
        }
        ss << "  }";
        return ss.str();
    }
    
    template<typename T>
    static std::string toJSON(const BenchmarkResult<T>& result) {
        std::ostringstream ss;
        ss << "  {\n";
        ss << "    \"encoderName\": \"" << escape(result.encoderName) << "\",\n";
        ss << "    \"datasetName\": \"" << escape(result.datasetName) << "\",\n";
        ss << "    \"dataSize\": " << result.dataSize << ",\n";
        ss << "    \"isComposed\": " << (result.isComposed ? "true" : "false") << ",\n";
        ss << "    \"metrics\": " << toJSON(result.metrics);
        
        if (!result.layerMetrics.empty()) {
            ss << ",\n    \"layerMetrics\": [\n";
            for (size_t i = 0; i < result.layerMetrics.size(); ++i) {
                ss << toJSON(result.layerMetrics[i]);
                if (i < result.layerMetrics.size() - 1) {
                    ss << ",";
                }
                ss << "\n";
            }
            ss << "    ]";
        }
        
        ss << "\n  }";
        return ss.str();
    }
    
    template<typename T>
    static std::string toJSON(const BenchmarkResults<T>& results) {
        std::ostringstream ss;
        ss << "{\n";
        
        // Metadata
        auto startTimeT = system_clock::to_time_t(results.startTime);
        auto endTimeT = system_clock::to_time_t(results.endTime);
        
        ss << "  \"metadata\": {\n";
        ss << "    \"startTime\": \"" << std::put_time(std::localtime(&startTimeT), "%Y-%m-%d %H:%M:%S") << "\",\n";
        ss << "    \"endTime\": \"" << std::put_time(std::localtime(&endTimeT), "%Y-%m-%d %H:%M:%S") << "\",\n";
        ss << "    \"totalDuration_s\": " << results.totalDuration().count() << ",\n";
        ss << "    \"totalBenchmarks\": " << results.results.size() << "\n";
        ss << "  },\n";
        
        // Configuration
        ss << "  \"config\": {\n";
        ss << "    \"dataSizes\": [";
        for (size_t i = 0; i < results.config.dataSizes.size(); ++i) {
            ss << results.config.dataSizes[i];
            if (i < results.config.dataSizes.size() - 1) ss << ", ";
        }
        ss << "],\n";
        ss << "    \"iterations\": " << results.config.iterations << ",\n";
        ss << "    \"warmupRuns\": " << results.config.warmupRuns << ",\n";
        ss << "    \"randomAccessSamples\": " << results.config.randomAccessSamples << ",\n";
        ss << "    \"stridedAccessSamples\": " << results.config.stridedAccessSamples << ",\n";
        ss << "    \"stride\": " << results.config.stride << ",\n";
        ss << "    \"rangeQueryCount\": " << results.config.rangeQueryCount << "\n";
        ss << "  },\n";
        
        // Results
        ss << "  \"results\": [\n";
        for (size_t i = 0; i < results.results.size(); ++i) {
            ss << toJSON(results.results[i]);
            if (i < results.results.size() - 1) {
                ss << ",";
            }
            ss << "\n";
        }
        ss << "  ]\n";
        
        ss << "}\n";
        return ss.str();
    }
};

/**
 * @brief Table output formatter
 */
class TableFormatter {
public:
    template<typename T>
    static void printSummaryTable(const BenchmarkResults<T>& results, std::ostream& out = std::cout) {
        out << "\n" << std::string(120, '=') << "\n";
        out << "BENCHMARK SUMMARY\n";
        out << std::string(120, '=') << "\n\n";
        
        // Header
        out << std::left
            << std::setw(20) << "Encoder"
            << std::setw(20) << "Dataset"
            << std::setw(12) << "Size"
            << std::setw(15) << "Enc (ms)"
            << std::setw(15) << "Dec (ms)"
            << std::setw(12) << "Ratio"
            << std::setw(15) << "Bits/Elem"
            << "\n";
        out << std::string(120, '-') << "\n";
        
        // Rows
        for (const auto& result : results.results) {
            out << std::left
                << std::setw(20) << result.encoderName.substr(0, 19)
                << std::setw(20) << result.datasetName.substr(0, 19)
                << std::setw(12) << result.dataSize
                << std::setw(15) << std::fixed << std::setprecision(3) 
                    << (result.metrics.timing.encodeTime.count() / 1e6)
                << std::setw(15) << std::fixed << std::setprecision(3)
                    << (result.metrics.timing.decodeBulkTime.count() / 1e6)
                << std::setw(12) << std::fixed << std::setprecision(2)
                    << result.metrics.memory.compressionRatio()
                << std::setw(15) << std::fixed << std::setprecision(2)
                    << result.metrics.memory.bitsPerElement(result.dataSize)
                << "\n";
        }
        
        out << std::string(120, '=') << "\n\n";
    }
    
    template<typename T>
    static void printDetailedTable(const BenchmarkResults<T>& results, std::ostream& out = std::cout) {
        out << "\n" << std::string(140, '=') << "\n";
        out << "DETAILED BENCHMARK RESULTS\n";
        out << std::string(140, '=') << "\n\n";
        
        for (const auto& result : results.results) {
            out << "Encoder: " << result.encoderName << " | Dataset: " << result.datasetName 
                << " | Size: " << result.dataSize << "\n";
            out << std::string(140, '-') << "\n";
            
            const auto& m = result.metrics;
            
            // Timing
            out << "TIMING:\n";
            out << "  Encode:        " << std::setw(10) << (m.timing.encodeTime.count() / 1e6) << " ms"
                << "  (" << std::setprecision(2) << m.timing.encodeElementsPerSecond / 1e6 << " M elem/s)\n";
            out << "  Decode (bulk): " << std::setw(10) << (m.timing.decodeBulkTime.count() / 1e6) << " ms"
                << "  (" << std::setprecision(2) << m.timing.decodeBulkElementsPerSecond / 1e6 << " M elem/s)\n";
            
            if (m.randomAccess.randomAccessCount > 0) {
                out << "  Random access: " << std::setw(10) << m.randomAccess.averageRandomAccessTime.count() << " ns/read"
                    << "  (min: " << m.randomAccess.minRandomAccessTime.count()
                    << " ns, max: " << m.randomAccess.maxRandomAccessTime.count() << " ns)\n";
            }
            if (m.selectiveAccess.rangeCount > 0) {
                out << "  Selective (gather): " << std::setw(10) << m.selectiveAccess.totalGatherTime.count() << " ns total"
                    << "  (" << m.selectiveAccess.rangeCount << " ranges, "
                    << m.selectiveAccess.totalSelectedRows << " rows, selectivity="
                    << std::fixed << std::setprecision(2) << m.selectiveAccess.selectivity
                    << ", meanRunLength=" << m.selectiveAccess.meanRunLength << ")\n";
                if (m.selectiveAccess.skipMaterializeSplitAvailable) {
                    out << "    skip: " << m.selectiveAccess.averageSkipTimeNs.count()
                        << " ns, materialize: " << m.selectiveAccess.averageMaterializeTimeNs.count() << " ns\n";
                }
            }

            // Memory
            out << "\nMEMORY:\n";
            out << "  Original:               " << std::setw(12) << m.memory.originalSize << " bytes\n";
            out << "  Encoded:                " << std::setw(12) << m.memory.encodedSize << " bytes\n";
            out << "  Compression:            " << std::setw(12) << std::fixed << std::setprecision(2)
                << m.memory.compressionRatio() << "x\n";
            out << "  Bits/element:           " << std::setw(12) << std::fixed << std::setprecision(2)
                << m.memory.bitsPerElement(m.elementCount) << "\n";
            if (m.memory.encodePeakHeapBytes > 0 || m.memory.encodeNetHeapDeltaBytes > 0) {
                out << "  Encode peak heap:       " << std::setw(12) << m.memory.encodePeakHeapBytes
                    << " bytes  (net delta: " << m.memory.encodeNetHeapDeltaBytes << " bytes";
                if (m.memory.encoderOverhead > 0)
                    out << ", encoder overhead: " << m.memory.encoderOverhead << " bytes";
                out << ")\n";
                out << "  Decode bulk peak heap:  " << std::setw(12) << m.memory.decodeBulkPeakHeapBytes
                    << " bytes  (net delta: " << m.memory.decodeBulkNetHeapDeltaBytes << " bytes)\n";
                if (m.memory.decodeRandomPeakHeapBytes > 0)
                    out << "  Decode random peak:     " << std::setw(12)
                        << m.memory.decodeRandomPeakHeapBytes << " bytes\n";
                if (m.memory.decodeStridedPeakHeapBytes > 0)
                    out << "  Decode strided peak:    " << std::setw(12)
                        << m.memory.decodeStridedPeakHeapBytes << " bytes\n";
                if (m.memory.decodeRangePeakHeapBytes > 0)
                    out << "  Decode range peak:      " << std::setw(12)
                        << m.memory.decodeRangePeakHeapBytes << " bytes\n";
                if (m.memory.decodeSelectivePeakHeapBytes > 0)
                    out << "  Decode selective peak:  " << std::setw(12)
                        << m.memory.decodeSelectivePeakHeapBytes << " bytes\n";
            }
            
            // Accuracy
            out << "\nACCURACY:\n";
            out << "  Lossless:      " << (m.accuracy.isLossless ? "Yes" : "No") << "\n";
            if (m.accuracy.mismatchCount > 0) {
                out << "  Mismatches:    " << m.accuracy.mismatchCount << "\n";
            }
            
            out << "\n" << std::string(140, '=') << "\n\n";
        }
    }
};

/**
 * @brief Save benchmark results to file
 */
template<typename T>
inline bool saveBenchmarkResults(const BenchmarkResults<T>& results, 
                                const std::string& filepath) {
    try {
        // Create directory if needed
        std::filesystem::path path(filepath);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        
        std::ofstream file(filepath);
        if (!file) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return false;
        }
        
        file << JSONWriter::toJSON(results);
        
        std::cout << "Results saved to: " << filepath << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
        return false;
    }
}

} // namespace encodings::benchmark
