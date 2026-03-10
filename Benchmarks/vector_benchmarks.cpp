#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include "encoders/SphericalEncoder.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"
#include "generators/VectorGenerator.hpp"

using namespace encodings;
using namespace encodings::encoders;
using namespace encodings::datagen;  // VectorGenerator is in datagen namespace

// Statistics helper
struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    
    static Stats compute(const std::vector<double>& values) {
        Stats s;
        if (values.empty()) return s;
        
        s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        s.min = *std::min_element(values.begin(), values.end());
        s.max = *std::max_element(values.begin(), values.end());
        
        if (values.size() > 1) {
            double sq_sum = 0.0;
            for (double v : values) {
                sq_sum += (v - s.mean) * (v - s.mean);
            }
            s.stddev = std::sqrt(sq_sum / (values.size() - 1));
        }
        
        return s;
    }
};

// Error metrics for reconstruction quality
struct ErrorMetrics {
    double meanAbsoluteError = 0.0;
    double maxAbsoluteError = 0.0;
    double meanRelativeError = 0.0;
    double meanEuclideanDistance = 0.0;
    double maxEuclideanDistance = 0.0;
    double meanCosineSimilarity = 0.0;  // For unit vectors
    double meanAngleError = 0.0;        // In radians

    // Magnitude-specific errors (only set for SphericalEncoder with normalizeToUnit=true).
    // Stored as long double so float/double/long double MagT variants are distinguishable.
    long double meanMagnitudeAbsoluteError = 0.0L;
    long double maxMagnitudeAbsoluteError  = 0.0L;
    long double meanMagnitudeRelativeError = 0.0L;
    
    template<size_t D>
    static ErrorMetrics compute(
        const std::vector<std::vector<float>>& original,
        const std::vector<std::vector<float>>& reconstructed
    ) {
        ErrorMetrics metrics;
        if (original.size() != reconstructed.size() || original.empty()) {
            return metrics;
        }
        
        const size_t n = original.size();
        double sumAbsError = 0.0;
        double sumRelError = 0.0;
        double sumEuclidean = 0.0;
        double sumCosine = 0.0;
        double sumAngle = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            const auto& orig = original[i];
            const auto& recon = reconstructed[i];
            
            // Per-element errors
            double vecAbsError = 0.0;
            double vecSqError = 0.0;
            for (size_t j = 0; j < D && j < orig.size() && j < recon.size(); ++j) {
                double diff = std::abs(orig[j] - recon[j]);
                vecAbsError += diff;
                vecSqError += diff * diff;
                
                metrics.maxAbsoluteError = std::max(metrics.maxAbsoluteError, diff);
                
                // Relative error
                if (std::abs(orig[j]) > 1e-10) {
                    sumRelError += diff / std::abs(orig[j]);
                }
            }
            
            sumAbsError += vecAbsError / D;
            
            // Euclidean distance
            double euclidean = std::sqrt(vecSqError);
            sumEuclidean += euclidean;
            metrics.maxEuclideanDistance = std::max(metrics.maxEuclideanDistance, euclidean);
            
            // Cosine similarity and angle
            double dotProduct = 0.0;
            double origNorm = 0.0;
            double reconNorm = 0.0;
            
            for (size_t j = 0; j < D && j < orig.size() && j < recon.size(); ++j) {
                dotProduct += orig[j] * recon[j];
                origNorm += orig[j] * orig[j];
                reconNorm += recon[j] * recon[j];
            }
            
            origNorm = std::sqrt(origNorm);
            reconNorm = std::sqrt(reconNorm);
            
            if (origNorm > 1e-10 && reconNorm > 1e-10) {
                double cosine = dotProduct / (origNorm * reconNorm);
                // Clamp to [-1, 1] to avoid numerical issues with acos
                cosine = std::max(-1.0, std::min(1.0, cosine));
                sumCosine += cosine;
                
                double angle = std::acos(cosine);
                sumAngle += angle;
            }
        }
        
        metrics.meanAbsoluteError = sumAbsError / n;
        metrics.meanRelativeError = sumRelError / (n * D);
        metrics.meanEuclideanDistance = sumEuclidean / n;
        metrics.meanCosineSimilarity = sumCosine / n;
        metrics.meanAngleError = sumAngle / n;
        
        return metrics;
    }
};

// Benchmark result for a single configuration
struct BenchmarkResult {
    std::string encoderName;
    std::string generatorType;
    size_t dimension;
    size_t vectorCount;
    
    // Compression metrics
    Stats compressionRatio;
    Stats compressedSizeBytes;
    size_t uncompressedSizeBytes;
    
    // Performance metrics
    Stats encodeThroughputMBps;
    Stats decodeThroughputMBps;
    Stats encodeLatencyMs;
    Stats decodeLatencyMs;
    
    // Error metrics (mean over iterations)
    ErrorMetrics errors;
};

template<size_t D>
std::vector<std::vector<float>> generateVectors(
    const std::string& generatorType,
    size_t count,
    uint64_t seed
) {
    if (generatorType == "Unit") {
        UnitVectorGenerator<std::vector<float>, D> gen(seed, "gaussian");
        return gen.generate(count);
    } else if (generatorType == "NonUnit_Small") {
        // Magnitude range [0.1, 2.0]
        NonUnitVectorGenerator<std::vector<float>, D> gen(0.1f, 2.0f, seed);
        return gen.generate(count);
    } else if (generatorType == "NonUnit_Wide") {
        // Magnitude range [0.001, 1000.0]
        NonUnitVectorGenerator<std::vector<float>, D> gen(0.001f, 1000.0f, seed);
        return gen.generate(count);
    }
    
    return {};
}

template<size_t D>
BenchmarkResult runBenchmark(
    std::shared_ptr<Codec<std::vector<float>>> encoder,
    const std::string& encoderName,
    const std::string& generatorType,
    size_t vectorCount,
    size_t iterations = 3
) {
    BenchmarkResult result;
    result.encoderName = encoderName;
    result.generatorType = generatorType;
    result.dimension = D;
    result.vectorCount = vectorCount;
    result.uncompressedSizeBytes = vectorCount * D * sizeof(float);
    
    std::vector<double> compressionRatios;
    std::vector<double> compressedSizes;
    std::vector<double> encodeThroughputs;
    std::vector<double> decodeThroughputs;
    std::vector<double> encodeLatencies;
    std::vector<double> decodeLatencies;
    
    ErrorMetrics totalErrors;
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        // Generate data with different seed per iteration
        auto vectors = generateVectors<D>(generatorType, vectorCount, 42 + iter);
        
        // Encode
        auto encodeStart = std::chrono::high_resolution_clock::now();
        EncodedData encoded = encoder->encode(vectors);
        auto encodeEnd = std::chrono::high_resolution_clock::now();
        
        auto encodeDuration = std::chrono::duration_cast<std::chrono::microseconds>(
            encodeEnd - encodeStart
        ).count();
        
        // Decode
        auto decodeStart = std::chrono::high_resolution_clock::now();
        auto decoded = encoder->decodeAll(encoded);
        auto decodeEnd = std::chrono::high_resolution_clock::now();
        
        auto decodeDuration = std::chrono::duration_cast<std::chrono::microseconds>(
            decodeEnd - decodeStart
        ).count();
        
        // Metrics
        double compressedSize = static_cast<double>(encoded.size());
        double compressionRatio = result.uncompressedSizeBytes / compressedSize;
        
        double encodeThroughput = (result.uncompressedSizeBytes / (1024.0 * 1024.0)) / 
                                  (encodeDuration / 1e6);
        double decodeThroughput = (result.uncompressedSizeBytes / (1024.0 * 1024.0)) / 
                                  (decodeDuration / 1e6);
        
        compressionRatios.push_back(compressionRatio);
        compressedSizes.push_back(compressedSize);
        encodeThroughputs.push_back(encodeThroughput);
        decodeThroughputs.push_back(decodeThroughput);
        encodeLatencies.push_back(encodeDuration / 1000.0); // Convert to ms
        decodeLatencies.push_back(decodeDuration / 1000.0);
        
        // Compute errors
        auto errors = ErrorMetrics::compute<D>(vectors, decoded);
        totalErrors.meanAbsoluteError += errors.meanAbsoluteError;
        totalErrors.maxAbsoluteError = std::max(totalErrors.maxAbsoluteError, 
                                                errors.maxAbsoluteError);
        totalErrors.meanRelativeError += errors.meanRelativeError;
        totalErrors.meanEuclideanDistance += errors.meanEuclideanDistance;
        totalErrors.maxEuclideanDistance = std::max(totalErrors.maxEuclideanDistance,
                                                    errors.maxEuclideanDistance);
        totalErrors.meanCosineSimilarity += errors.meanCosineSimilarity;
        totalErrors.meanAngleError += errors.meanAngleError;

        // Magnitude-specific errors: query the IMagnitudeErrorProvider interface if available
        if (auto* magProvider = dynamic_cast<IMagnitudeErrorProvider<std::vector<float>>*>(encoder.get())) {
            auto magErrors = magProvider->computeMagnitudeErrors(vectors);
            totalErrors.meanMagnitudeAbsoluteError += magErrors.meanAbsoluteError;
            totalErrors.maxMagnitudeAbsoluteError   = std::max(
                totalErrors.maxMagnitudeAbsoluteError, magErrors.maxAbsoluteError);
            totalErrors.meanMagnitudeRelativeError  += magErrors.meanRelativeError;
        }
    }
    
    // Compute statistics
    result.compressionRatio = Stats::compute(compressionRatios);
    result.compressedSizeBytes = Stats::compute(compressedSizes);
    result.encodeThroughputMBps = Stats::compute(encodeThroughputs);
    result.decodeThroughputMBps = Stats::compute(decodeThroughputs);
    result.encodeLatencyMs = Stats::compute(encodeLatencies);
    result.decodeLatencyMs = Stats::compute(decodeLatencies);
    
    // Average errors
    result.errors.meanAbsoluteError = totalErrors.meanAbsoluteError / iterations;
    result.errors.maxAbsoluteError = totalErrors.maxAbsoluteError;
    result.errors.meanRelativeError = totalErrors.meanRelativeError / iterations;
    result.errors.meanEuclideanDistance = totalErrors.meanEuclideanDistance / iterations;
    result.errors.maxEuclideanDistance = totalErrors.maxEuclideanDistance;
    result.errors.meanCosineSimilarity = totalErrors.meanCosineSimilarity / iterations;
    result.errors.meanAngleError = totalErrors.meanAngleError / iterations;
    result.errors.meanMagnitudeAbsoluteError = totalErrors.meanMagnitudeAbsoluteError / iterations;
    result.errors.maxMagnitudeAbsoluteError  = totalErrors.maxMagnitudeAbsoluteError;
    result.errors.meanMagnitudeRelativeError = totalErrors.meanMagnitudeRelativeError / iterations;
    
    return result;
}

void writeResultsToJSON(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"benchmarks\": [\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"encoder\": \"" << r.encoderName << "\",\n";
        out << "      \"generator\": \"" << r.generatorType << "\",\n";
        out << "      \"dimension\": " << r.dimension << ",\n";
        out << "      \"vector_count\": " << r.vectorCount << ",\n";
        out << "      \"uncompressed_size_bytes\": " << r.uncompressedSizeBytes << ",\n";
        
        out << "      \"compression_ratio\": {\n";
        out << "        \"mean\": " << r.compressionRatio.mean << ",\n";
        out << "        \"stddev\": " << r.compressionRatio.stddev << ",\n";
        out << "        \"min\": " << r.compressionRatio.min << ",\n";
        out << "        \"max\": " << r.compressionRatio.max << "\n";
        out << "      },\n";
        
        out << "      \"compressed_size_bytes\": {\n";
        out << "        \"mean\": " << r.compressedSizeBytes.mean << ",\n";
        out << "        \"stddev\": " << r.compressedSizeBytes.stddev << ",\n";
        out << "        \"min\": " << r.compressedSizeBytes.min << ",\n";
        out << "        \"max\": " << r.compressedSizeBytes.max << "\n";
        out << "      },\n";
        
        out << "      \"encode_throughput_mbps\": {\n";
        out << "        \"mean\": " << r.encodeThroughputMBps.mean << ",\n";
        out << "        \"stddev\": " << r.encodeThroughputMBps.stddev << ",\n";
        out << "        \"min\": " << r.encodeThroughputMBps.min << ",\n";
        out << "        \"max\": " << r.encodeThroughputMBps.max << "\n";
        out << "      },\n";
        
        out << "      \"decode_throughput_mbps\": {\n";
        out << "        \"mean\": " << r.decodeThroughputMBps.mean << ",\n";
        out << "        \"stddev\": " << r.decodeThroughputMBps.stddev << ",\n";
        out << "        \"min\": " << r.decodeThroughputMBps.min << ",\n";
        out << "        \"max\": " << r.decodeThroughputMBps.max << "\n";
        out << "      },\n";
        
        out << "      \"encode_latency_ms\": {\n";
        out << "        \"mean\": " << r.encodeLatencyMs.mean << ",\n";
        out << "        \"stddev\": " << r.encodeLatencyMs.stddev << ",\n";
        out << "        \"min\": " << r.encodeLatencyMs.min << ",\n";
        out << "        \"max\": " << r.encodeLatencyMs.max << "\n";
        out << "      },\n";
        
        out << "      \"decode_latency_ms\": {\n";
        out << "        \"mean\": " << r.decodeLatencyMs.mean << ",\n";
        out << "        \"stddev\": " << r.decodeLatencyMs.stddev << ",\n";
        out << "        \"min\": " << r.decodeLatencyMs.min << ",\n";
        out << "        \"max\": " << r.decodeLatencyMs.max << "\n";
        out << "      },\n";
        
        out << "      \"errors\": {\n";
        out << "        \"mean_absolute_error\": " << r.errors.meanAbsoluteError << ",\n";
        out << "        \"max_absolute_error\": " << r.errors.maxAbsoluteError << ",\n";
        out << "        \"mean_relative_error\": " << r.errors.meanRelativeError << ",\n";
        out << "        \"mean_euclidean_distance\": " << r.errors.meanEuclideanDistance << ",\n";
        out << "        \"max_euclidean_distance\": " << r.errors.maxEuclideanDistance << ",\n";
        out << "        \"mean_cosine_similarity\": " << r.errors.meanCosineSimilarity << ",\n";
        out << "        \"mean_angle_error_rad\": " << r.errors.meanAngleError << ",\n";
        // Magnitude errors use 20 significant digits so float/double/long double are distinguishable
        out << std::setprecision(20);
        out << "        \"mean_magnitude_absolute_error\": " << r.errors.meanMagnitudeAbsoluteError << ",\n";
        out << "        \"max_magnitude_absolute_error\": "  << r.errors.maxMagnitudeAbsoluteError  << ",\n";
        out << "        \"mean_magnitude_relative_error\": " << r.errors.meanMagnitudeRelativeError << "\n";
        out << std::setprecision(6);
        out << "      }\n";
        
        out << "    }";
        if (i < results.size() - 1) {
            out << ",";
        }
        out << "\n";
    }
    
    out << "  ]\n";
    out << "}\n";
    out.close();
}

template<size_t D>
void runBenchmarksForDimension(std::vector<BenchmarkResult>& allResults) {
    std::cout << "\n=== Testing Dimension " << D << " ===\n" << std::endl;
    
    std::vector<size_t> vectorCounts = {1000, 100000};
    std::vector<std::string> generators = {"Unit", "NonUnit_Small", "NonUnit_Wide"};
    
    for (size_t count : vectorCounts) {
        std::cout << "Vector count: " << count << std::endl;
        
        for (const auto& genType : generators) {
            std::cout << "  Generator: " << genType << std::endl;
            
            // Spherical Encoder (no normalization)
            {
                auto encoder = std::make_shared<SphericalEncoder<std::vector<float>, D>>(
                    nullptr, nullptr, false
                );
                std::cout << "    Testing SphericalEncoder..." << std::flush;
                auto result = runBenchmark<D>(encoder, "SphericalEncoder", genType, count);
                allResults.push_back(result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) 
                          << result.compressionRatio.mean << "x" << std::endl;
            }
            
            // Spherical Encoder (with normalization, float mag)
            {
                auto encoder = std::make_shared<SphericalEncoder<std::vector<float>, D, float>>(
                    nullptr, nullptr, true
                );
                std::cout << "    Testing SphericalEncoder_Normalized..." << std::flush;
                auto result = runBenchmark<D>(encoder, "SphericalEncoder_Normalized_FloatMag", genType, count);
                allResults.push_back(result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) 
                          << result.compressionRatio.mean << "x" << std::endl;
            }

            // Spherical Encoder (with normalization, double mag)
            {
                auto encoder = std::make_shared<SphericalEncoder<std::vector<float>, D, double>>(
                    nullptr, nullptr, true
                );
                std::cout << "    Testing SphericalEncoder_Normalized..." << std::flush;
                auto result = runBenchmark<D>(encoder, "SphericalEncoder_Normalized_DoubleMag", genType, count);
                allResults.push_back(result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) 
                          << result.compressionRatio.mean << "x" << std::endl;
            }

            // Spherical Encoder (with normalization, long double mag)
            {
                auto encoder = std::make_shared<SphericalEncoder<std::vector<float>, D, long double>>(
                    nullptr, nullptr, true
                );
                std::cout << "    Testing SphericalEncoder_Normalized..." << std::flush;
                auto result = runBenchmark<D>(encoder, "SphericalEncoder_Normalized_LongDoubleMag", genType, count);
                allResults.push_back(result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) 
                          << result.compressionRatio.mean << "x" << std::endl;
            }
            
            // // Raw Encoder (baseline - no compression)
            // {
            //     auto encoder = std::make_shared<RawEncoder<std::vector<float>>>();
            //     std::cout << "    Testing RawEncoder..." << std::flush;
            //     auto result = runBenchmark<D>(encoder, "RawEncoder", genType, count);
            //     allResults.push_back(result);
            //     std::cout << " Ratio: " << std::fixed << std::setprecision(2) 
            //               << result.compressionRatio.mean << "x" << std::endl;
            // }
            
            // // Zstd Encoder (general purpose compression)
            // {
            //     auto encoder = std::make_shared<ZstdVectorEncoder<std::vector<float>>>();
            //     std::cout << "    Testing ZstdEncoder..." << std::flush;
            //     auto result = runBenchmark<D>(encoder, "ZstdEncoder", genType, count);
            //     allResults.push_back(result);
            //     std::cout << " Ratio: " << std::fixed << std::setprecision(2) 
            //               << result.compressionRatio.mean << "x" << std::endl;
            // }
        }
    }
}

int main() {
    std::cout << "Vector Encoding Benchmarks" << std::endl;
    std::cout << "==========================\n" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dimensions: 96, 384, 768" << std::endl;
    std::cout << "  Vector counts: 1000, 100000, 1000000" << std::endl;
    std::cout << "  Generators: Unit, NonUnit_Small [0.1, 2.0], NonUnit_Wide [0.001, 1000]" << std::endl;
    std::cout << "  Encoders: SphericalEncoder, SphericalEncoder_Normalized, RawEncoder, ZstdEncoder" << std::endl;
    std::cout << "  Iterations: 3" << std::endl;
    
    std::vector<BenchmarkResult> allResults;
    
    // Run benchmarks for each dimension
    runBenchmarksForDimension<96>(allResults);
    runBenchmarksForDimension<384>(allResults);
    runBenchmarksForDimension<768>(allResults);
    
    // Write results to JSON
    std::string outputFile = "vector_benchmark_results.json";
    writeResultsToJSON(allResults, outputFile);
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    std::cout << "Results written to: " << outputFile << std::endl;
    std::cout << "\nRun 'python3 plot_vector_results.py' to generate visualizations." << std::endl;
    
    return 0;
}
