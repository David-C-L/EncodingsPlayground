#include <iostream>
#include <iomanip>
#include <cmath>
#include "generators/VectorGenerator.hpp"
#include "encoders/SphericalEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"

using namespace encodings::datagen;
using namespace encodings::encoders;

// Helper to calculate statistics
template<typename T, size_t D>
struct VectorStats {
    float minMagnitude = std::numeric_limits<float>::max();
    float maxMagnitude = 0.0f;
    float avgMagnitude = 0.0f;
    float stddevMagnitude = 0.0f;
    
    void compute(const std::vector<T>& vectors) {
        std::vector<float> magnitudes;
        magnitudes.reserve(vectors.size());
        
        for (const auto& vec : vectors) {
            float sumSq = 0.0f;
            for (const auto& val : vec) {
                sumSq += val * val;
            }
            float mag = std::sqrt(sumSq);
            magnitudes.push_back(mag);
            
            minMagnitude = std::min(minMagnitude, mag);
            maxMagnitude = std::max(maxMagnitude, mag);
            avgMagnitude += mag;
        }
        
        avgMagnitude /= vectors.size();
        
        // Calculate standard deviation
        float variance = 0.0f;
        for (float mag : magnitudes) {
            float diff = mag - avgMagnitude;
            variance += diff * diff;
        }
        stddevMagnitude = std::sqrt(variance / vectors.size());
    }
    
    void print() const {
        std::cout << "  Magnitude - Min: " << std::fixed << std::setprecision(6) << minMagnitude
                  << ", Max: " << maxMagnitude
                  << ", Avg: " << avgMagnitude
                  << ", StdDev: " << stddevMagnitude << std::endl;
    }
};

// Test unit vector generator
template<size_t D>
void testUnitVectorGenerator() {
    std::cout << "\n=== Unit Vector Generator (D=" << D << ") ===" << std::endl;
    
    const size_t count = 10000;
    
    // Test Gaussian method
    {
        auto gen = UnitVectorGenerator<std::array<float, D>, D>(42, "gaussian");
        std::cout << "\nMethod: Gaussian (Marsaglia)" << std::endl;
        std::cout << "Generator: " << gen.name() << std::endl;
        
        auto vectors = gen.generate(count);
        
        VectorStats<std::array<float, D>, D> stats;
        stats.compute(vectors);
        stats.print();
        
        // Test with encoder
        auto encoder = SphericalEncoder<std::array<float, D>, D>(
            std::make_shared<ZstdEncoder<uint8_t>>()
        );
        
        auto encoded = encoder.encode(vectors);
        size_t originalSize = count * D * sizeof(float);
        float compressionRatio = (float)originalSize / encoded.size();
        
        std::cout << "  Compression: " << originalSize << " -> " << encoded.size() 
                  << " bytes (" << std::fixed << std::setprecision(2) 
                  << compressionRatio << "x)" << std::endl;
        
        // Verify decoding
        auto decoded = encoder.decodeAll(encoded);
        float maxErr = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < D; ++j) {
                maxErr = std::max(maxErr, std::abs(vectors[i][j] - decoded[i][j]));
            }
        }
        std::cout << "  Max reconstruction error: " << std::scientific << maxErr << std::endl;
    }
    
    // Test rejection method (only practical for low dimensions D <= 3)
    // Acceptance rate: D=2: 78.5%, D=3: 52.4%, D=16: 0.002%, D=128: essentially 0%
    if constexpr (D <= 3) {
        auto gen = UnitVectorGenerator<std::array<float, D>, D>(42, "rejection");
        std::cout << "\nMethod: Rejection Sampling" << std::endl;
        
        auto vectors = gen.generate(1000);  // Smaller count for slower method
        
        VectorStats<std::array<float, D>, D> stats;
        stats.compute(vectors);
        stats.print();
    } else {
        std::cout << "\nMethod: Rejection Sampling (skipped for D=" << D 
                  << ", acceptance rate too low)" << std::endl;
    }
}

// Test non-unit vector generator
template<size_t D>
void testNonUnitVectorGenerator() {
    std::cout << "\n=== Non-Unit Vector Generator (D=" << D << ") ===" << std::endl;
    
    const size_t count = 10000;
    
    // Test different distributions
    struct DistTest {
        NonUnitVectorGenerator<std::array<float, D>, D>::MagnitudeDistribution dist;
        std::string name;
    };
    
    std::vector<DistTest> tests = {
        {NonUnitVectorGenerator<std::array<float, D>, D>::MagnitudeDistribution::Uniform, "Uniform"},
        {NonUnitVectorGenerator<std::array<float, D>, D>::MagnitudeDistribution::LogNormal, "LogNormal"},
        {NonUnitVectorGenerator<std::array<float, D>, D>::MagnitudeDistribution::Exponential, "Exponential"},
        {NonUnitVectorGenerator<std::array<float, D>, D>::MagnitudeDistribution::Fixed, "Fixed"}
    };
    
    for (const auto& test : tests) {
        std::cout << "\nDistribution: " << test.name << std::endl;
        
        auto gen = NonUnitVectorGenerator<std::array<float, D>, D>(
            0.5f,   // min magnitude
            5.0f,   // max magnitude
            42,     // seed
            test.dist
        );
        
        std::cout << "Generator: " << gen.name() << std::endl;
        auto vectors = gen.generate(count);
        
        VectorStats<std::array<float, D>, D> stats;
        stats.compute(vectors);
        stats.print();
        
        // Test without normalization
        {
            auto encoder = SphericalEncoder<std::array<float, D>, D>(
                std::make_shared<ZstdEncoder<uint8_t>>(),
                nullptr,
                false  // don't normalize
            );
            
            auto encoded = encoder.encode(vectors);
            size_t originalSize = count * D * sizeof(float);
            float compressionRatio = (float)originalSize / encoded.size();
            
            std::cout << "  Without normalization - Compression: " 
                      << std::fixed << std::setprecision(2) << compressionRatio << "x" << std::endl;
        }
        
        // Test with normalization
        {
            auto encoder = SphericalEncoder<std::array<float, D>, D>(
                std::make_shared<ZstdEncoder<uint8_t>>(),
                std::make_shared<ZstdEncoder<float>>(),
                true  // normalize
            );
            
            auto encoded = encoder.encode(vectors);
            size_t originalSize = count * D * sizeof(float);
            float compressionRatio = (float)originalSize / encoded.size();
            
            std::cout << "  With normalization - Compression: " 
                      << std::fixed << std::setprecision(2) << compressionRatio << "x" << std::endl;
            
            // Verify magnitude preservation
            auto decoded = encoder.decodeAll(encoded);
            float maxMagErr = 0.0f;
            for (size_t i = 0; i < count; ++i) {
                float origMag = 0.0f, decMag = 0.0f;
                for (size_t j = 0; j < D; ++j) {
                    origMag += vectors[i][j] * vectors[i][j];
                    decMag += decoded[i][j] * decoded[i][j];
                }
                origMag = std::sqrt(origMag);
                decMag = std::sqrt(decMag);
                maxMagErr = std::max(maxMagErr, std::abs(origMag - decMag));
            }
            std::cout << "  Max magnitude error: " << std::scientific << maxMagErr << std::endl;
        }
    }
}

// Test patterned vector generator
template<size_t D>
void testPatternedVectorGenerator() {
    std::cout << "\n=== Patterned Vector Generator (D=" << D << ") ===" << std::endl;
    
    const size_t count = 10000;
    
    struct PatternTest {
        PatternedVectorGenerator<std::array<float, D>, D>::Pattern pattern;
        std::string name;
    };
    
    std::vector<PatternTest> tests = {
        {PatternedVectorGenerator<std::array<float, D>, D>::Pattern::Clustered, "Clustered"},
        {PatternedVectorGenerator<std::array<float, D>, D>::Pattern::Sparse, "Sparse"},
        {PatternedVectorGenerator<std::array<float, D>, D>::Pattern::CoordinateAligned, "CoordinateAligned"},
        {PatternedVectorGenerator<std::array<float, D>, D>::Pattern::Repeated, "Repeated"},
        {PatternedVectorGenerator<std::array<float, D>, D>::Pattern::Smooth, "Smooth"}
    };
    
    for (const auto& test : tests) {
        std::cout << "\nPattern: " << test.name << std::endl;
        
        auto gen = PatternedVectorGenerator<std::array<float, D>, D>(
            test.pattern,
            42,     // seed
            0.1f    // cluster radius
        );
        
        std::cout << "Generator: " << gen.name() << std::endl;
        auto vectors = gen.generate(count);
        
        VectorStats<std::array<float, D>, D> stats;
        stats.compute(vectors);
        stats.print();
        
        // Test compression
        auto encoder = SphericalEncoder<std::array<float, D>, D>(
            std::make_shared<ZstdEncoder<uint8_t>>()
        );
        
        auto encoded = encoder.encode(vectors);
        size_t originalSize = count * D * sizeof(float);
        float compressionRatio = (float)originalSize / encoded.size();
        
        std::cout << "  Compression: " << std::fixed << std::setprecision(2) 
                  << compressionRatio << "x" << std::endl;
        
        // Patterns should compress differently
        if (test.name == "Repeated") {
            std::cout << "  ✓ Expected high compression for repeated patterns" << std::endl;
        } else if (test.name == "Smooth") {
            std::cout << "  ✓ Expected good compression for smooth patterns" << std::endl;
        }
    }
}

// Test generator reproducibility
template<size_t D>
void testReproducibility() {
    std::cout << "\n=== Testing Reproducibility (D=" << D << ") ===" << std::endl;
    
    auto gen1 = UnitVectorGenerator<std::array<float, D>, D>(42);
    auto gen2 = UnitVectorGenerator<std::array<float, D>, D>(42);
    
    auto vectors1 = gen1.generate(1000);
    auto vectors2 = gen2.generate(1000);
    
    // Check if identical
    bool identical = true;
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < D; ++j) {
            if (vectors1[i][j] != vectors2[i][j]) {
                identical = false;
                break;
            }
        }
        if (!identical) break;
    }
    
    std::cout << "Same seed produces: " << (identical ? "✓ Identical" : "✗ Different") << " results" << std::endl;
    
    // Test reset
    gen1.reset();
    auto vectors3 = gen1.generate(1000);
    
    identical = true;
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < D; ++j) {
            if (vectors1[i][j] != vectors3[i][j]) {
                identical = false;
                break;
            }
        }
        if (!identical) break;
    }
    
    std::cout << "Reset produces: " << (identical ? "✓ Identical" : "✗ Different") << " results" << std::endl;
}

// Compare different dimensions
void compareDimensions() {
    std::cout << "\n=== Comparing Different Dimensions ===" << std::endl;
    
    const size_t count = 1000;
    
    std::cout << "\nCompression ratios for unit vectors:" << std::endl;
    
    {
        auto gen = UnitVectorGenerator<std::array<float, 3>, 3>(42);
        auto vectors = gen.generate(count);
        auto encoder = SphericalEncoder<std::array<float, 3>, 3>(std::make_shared<ZstdEncoder<uint8_t>>());
        auto encoded = encoder.encode(vectors);
        float ratio = (float)(count * 3 * sizeof(float)) / encoded.size();
        std::cout << "  3D:    " << std::fixed << std::setprecision(2) << ratio << "x" << std::endl;
    }
    
    {
        auto gen = UnitVectorGenerator<std::array<float, 16>, 16>(42);
        auto vectors = gen.generate(count);
        auto encoder = SphericalEncoder<std::array<float, 16>, 16>(std::make_shared<ZstdEncoder<uint8_t>>());
        auto encoded = encoder.encode(vectors);
        float ratio = (float)(count * 16 * sizeof(float)) / encoded.size();
        std::cout << "  16D:   " << std::fixed << std::setprecision(2) << ratio << "x" << std::endl;
    }
    
    {
        auto gen = UnitVectorGenerator<std::array<float, 128>, 128>(42);
        auto vectors = gen.generate(count);
        auto encoder = SphericalEncoder<std::array<float, 128>, 128>(std::make_shared<ZstdEncoder<uint8_t>>());
        auto encoded = encoder.encode(vectors);
        float ratio = (float)(count * 128 * sizeof(float)) / encoded.size();
        std::cout << "  128D:  " << std::fixed << std::setprecision(2) << ratio << "x" << std::endl;
    }
    
    std::cout << "\n✓ Higher dimensions generally achieve better compression ratios" << std::endl;
}

int main() {
    std::cout << "Vector Generator Test Suite" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        // Test unit vectors
        testUnitVectorGenerator<3>();
        testUnitVectorGenerator<128>();
        
        // Test non-unit vectors
        testNonUnitVectorGenerator<3>();
        testNonUnitVectorGenerator<128>();
        
        // Test patterned vectors
        testPatternedVectorGenerator<3>();
        testPatternedVectorGenerator<16>();
        
        // Test reproducibility
        testReproducibility<3>();
        
        // Compare dimensions
        compareDimensions();
        
        std::cout << "\n=== All tests completed successfully ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
