#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include "encoders/SphericalEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"

using namespace encodings::encoders;

// Helper to generate unit vectors
std::vector<std::array<float, 3>> generateUnitVectors(size_t count) {
    std::vector<std::array<float, 3>> vectors;
    vectors.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        float theta = 2.0f * M_PI * i / count;
        float phi = M_PI * (i % 10) / 10.0f;
        
        std::array<float, 3> vec = {
            std::sin(phi) * std::cos(theta),
            std::sin(phi) * std::sin(theta),
            std::cos(phi)
        };
        
        vectors.push_back(vec);
    }
    
    return vectors;
}

// Helper to generate non-unit vectors
std::vector<std::array<float, 3>> generateNonUnitVectors(size_t count) {
    std::vector<std::array<float, 3>> vectors;
    vectors.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        float scale = 1.0f + (i % 10) / 10.0f;
        float theta = 2.0f * M_PI * i / count;
        float phi = M_PI * (i % 10) / 10.0f;
        
        std::array<float, 3> vec = {
            scale * std::sin(phi) * std::cos(theta),
            scale * std::sin(phi) * std::sin(theta),
            scale * std::cos(phi)
        };
        
        vectors.push_back(vec);
    }
    
    return vectors;
}

// Helper to calculate magnitude
float magnitude(const std::array<float, 3>& vec) {
    return std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

// Helper to calculate error
float maxError(const std::vector<std::array<float, 3>>& original,
               const std::vector<std::array<float, 3>>& decoded) {
    float maxErr = 0.0f;
    for (size_t i = 0; i < original.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float err = std::abs(original[i][j] - decoded[i][j]);
            maxErr = std::max(maxErr, err);
        }
    }
    return maxErr;
}

void testUnitVectors() {
    std::cout << "\n=== Testing Unit Vectors ===" << std::endl;
    
    const size_t numVectors = 10000;
    auto vectors = generateUnitVectors(numVectors);
    
    // Test without normalization (already unit vectors)
    // Note: dimension is now a template parameter
    auto encoder = SphericalEncoder<std::array<float, 3>, 3>(
        std::make_shared<ZstdEncoder<uint8_t>>(),
        nullptr,
        false  // don't normalize (already unit)
    );
    
    auto encoded = encoder.encode(vectors);
    auto decoded = encoder.decodeAll(encoded);
    
    size_t originalSize = numVectors * 3 * sizeof(float);
    size_t compressedSize = encoded.size();
    float compressionRatio = static_cast<float>(originalSize) / compressedSize;
    float error = maxError(vectors, decoded);
    
    std::cout << "Original size: " << originalSize << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressedSize << " bytes" << std::endl;
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) 
              << compressionRatio << "x" << std::endl;
    std::cout << "Max reconstruction error: " << std::scientific 
              << error << std::endl;
}

void testNonUnitVectors() {
    std::cout << "\n=== Testing Non-Unit Vectors ===" << std::endl;
    
    const size_t numVectors = 10000;
    auto vectors = generateNonUnitVectors(numVectors);
    
    // Test 1: Without normalization
    std::cout << "\n--- Without Normalization ---" << std::endl;
    auto encoder1 = SphericalEncoder<std::array<float, 3>, 3>(
        std::make_shared<ZstdEncoder<uint8_t>>(),
        nullptr,
        false  // don't normalize
    );
    
    auto encoded1 = encoder1.encode(vectors);
    auto decoded1 = encoder1.decodeAll(encoded1);
    
    size_t originalSize = numVectors * 3 * sizeof(float);
    float compressionRatio1 = static_cast<float>(originalSize) / encoded1.size();
    float error1 = maxError(vectors, decoded1);
    
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) 
              << compressionRatio1 << "x" << std::endl;
    std::cout << "Max reconstruction error: " << std::scientific 
              << error1 << std::endl;
    
    // Test 2: With normalization
    std::cout << "\n--- With Normalization ---" << std::endl;
    auto encoder2 = SphericalEncoder<std::array<float, 3>, 3>(
        std::make_shared<ZstdEncoder<uint8_t>>(),
        std::make_shared<ZstdEncoder<float>>(),
        true  // normalize
    );
    
    auto encoded2 = encoder2.encode(vectors);
    auto decoded2 = encoder2.decodeAll(encoded2);
    
    float compressionRatio2 = static_cast<float>(originalSize) / encoded2.size();
    float error2 = maxError(vectors, decoded2);
    
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) 
              << compressionRatio2 << "x" << std::endl;
    std::cout << "Max reconstruction error: " << std::scientific 
              << error2 << std::endl;
    
    // Verify magnitude preservation with normalization
    float maxMagError = 0.0f;
    for (size_t i = 0; i < vectors.size(); ++i) {
        float origMag = magnitude(vectors[i]);
        float decMag = magnitude(decoded2[i]);
        maxMagError = std::max(maxMagError, std::abs(origMag - decMag));
    }
    std::cout << "Max magnitude error: " << std::scientific 
              << maxMagError << std::endl;
}

void testHighDimensional() {
    std::cout << "\n=== Testing High-Dimensional Vectors (128D) ===" << std::endl;
    
    const size_t numVectors = 1000;
    const size_t dim = 128;
    
    std::vector<std::vector<float>> vectors;
    vectors.reserve(numVectors);
    
    // Generate normalized vectors
    for (size_t i = 0; i < numVectors; ++i) {
        std::vector<float> vec(dim);
        float sumSq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            vec[j] = std::sin(2.0f * M_PI * i / numVectors + j * 0.1f);
            sumSq += vec[j] * vec[j];
        }
        float mag = std::sqrt(sumSq);
        for (size_t j = 0; j < dim; ++j) {
            vec[j] /= mag;
        }
        vectors.push_back(vec);
    }
    
    // Note: dimension is now a template parameter
    auto encoder = SphericalEncoder<std::vector<float>, 128>(
        std::make_shared<ZstdEncoder<uint8_t>>(),
        nullptr,
        false  // already unit vectors
    );
    
    auto encoded = encoder.encode(vectors);
    auto decoded = encoder.decodeAll(encoded);
    
    size_t originalSize = numVectors * dim * sizeof(float);
    size_t compressedSize = encoded.size();
    float compressionRatio = static_cast<float>(originalSize) / compressedSize;
    
    float maxErr = 0.0f;
    for (size_t i = 0; i < vectors.size(); ++i) {
        for (size_t j = 0; j < dim; ++j) {
            float err = std::abs(vectors[i][j] - decoded[i][j]);
            maxErr = std::max(maxErr, err);
        }
    }
    
    std::cout << "Original size: " << originalSize << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressedSize << " bytes" << std::endl;
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) 
              << compressionRatio << "x" << std::endl;
    std::cout << "Max reconstruction error: " << std::scientific 
              << maxErr << std::endl;
}

int main() {
    std::cout << "SphericalEncoder Test Suite" << std::endl;
    std::cout << "===========================" << std::endl;
    
    try {
        testUnitVectors();
        testNonUnitVectors();
        testHighDimensional();
        
        std::cout << "\n=== All tests completed successfully ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
