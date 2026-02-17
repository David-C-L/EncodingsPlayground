#pragma once

#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <numbers>
#include <memory>
#include <string>
#include <concepts>
#include "generators/DataGenerator.hpp"
#include "core/DataType.hpp"

namespace encodings::datagen {

using core::Vector32Type;
using core::Float32Type;

/**
 * @brief Generate random unit vectors (normalized to length 1)
 * 
 * Uses the Marsaglia method for uniform sampling on a hypersphere.
 * This ensures vectors are uniformly distributed on the unit sphere.
 * 
 * Perfect for testing:
 * - Embedding compression
 * - Spherical encoding
 * - Directional data
 * 
 * @tparam T The vector type (must satisfy Vector32Type)
 * @tparam Dimension The dimensionality of the vectors
 */
template<typename T, size_t Dimension>
    requires Vector32Type<T> && (Dimension >= 2)
class UnitVectorGenerator : public DataGenerator<T> {
public:
    /**
     * @brief Construct a unit vector generator
     * 
     * @param seed Random seed for reproducibility (default: 42)
     * @param method Sampling method: "gaussian" (default) or "rejection"
     */
    explicit UnitVectorGenerator(uint32_t seed = 42, const std::string& method = "gaussian")
        : rng_(seed), seed_(seed), method_(method), 
          normalDist_(0.0, 1.0), uniformDist_(-1.0, 1.0) {
    }
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> vectors;
        vectors.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            vectors.push_back(generateSingle());
        }
        
        return vectors;
    }
    
    std::string name() const override {
        return "UnitVectorGenerator<" + std::to_string(Dimension) + "D>_" + method_;
    }
    
    DataType dataType() const override {
        return core::DataType::Vector32;
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::map<std::string, std::string> getConfig() const override {
        return {
            {"type", "unit_vector"},
            {"dimension", std::to_string(Dimension)},
            {"seed", std::to_string(seed_)},
            {"method", method_},
            {"normalized", "true"}
        };
    }
    
private:
    std::mt19937 rng_;
    uint32_t seed_;
    std::string method_;
    std::normal_distribution<float> normalDist_;
    std::uniform_real_distribution<float> uniformDist_;
    
    /**
     * @brief Generate a single unit vector
     * 
     * Uses Marsaglia method: sample from normal distribution and normalize.
     * This gives uniform distribution on the unit sphere.
     */
    T generateSingle() {
        if (method_ == "gaussian" || method_ == "marsaglia") {
            return generateGaussian();
        } else if (method_ == "rejection") {
            return generateRejection();
        } else {
            return generateGaussian();  // Default
        }
    }
    
    /**
     * @brief Marsaglia method: sample from Gaussian and normalize
     * This is the most efficient and uniform method
     */
    T generateGaussian() {
        T vec;
        if constexpr (requires { vec.resize(Dimension); }) {
            vec.resize(Dimension);
        }
        
        double sumSq = 0.0;
        
        // Sample from normal distribution
        size_t idx = 0;
        for (auto it = vec.begin(); it != vec.end() && idx < Dimension; ++it, ++idx) {
            float val = normalDist_(rng_);
            *it = val;
            sumSq += static_cast<double>(val) * val;
        }
        
        // Normalize to unit length
        float invMag = static_cast<float>(1.0 / std::sqrt(sumSq));
        for (auto& val : vec) {
            val *= invMag;
        }
        
        return vec;
    }
    
    /**
     * @brief Rejection sampling method
     * 
     * WARNING: This method is IMPRACTICAL for dimensions > 3!
     * Acceptance probability decreases exponentially with dimension:
     * - D=2: ~78.5% acceptance
     * - D=3: ~52.4% acceptance  
     * - D=10: ~0.2% acceptance (500x slower than Gaussian)
     * - D=16: ~0.002% acceptance (50,000x slower!)
     * - D=128: essentially never terminates
     * 
     * Use gaussian/marsaglia method for D > 3.
     * This method is only provided for comparison/verification in low dimensions.
     */
    T generateRejection() {
        while (true) {
            T vec;
            if constexpr (requires { vec.resize(Dimension); }) {
                vec.resize(Dimension);
            }
            
            double sumSq = 0.0;
            
            // Sample uniformly from [-1, 1]^D
            size_t idx = 0;
            for (auto it = vec.begin(); it != vec.end() && idx < Dimension; ++it, ++idx) {
                float val = uniformDist_(rng_);
                *it = val;
                sumSq += static_cast<double>(val) * val;
            }
            
            // Reject if outside unit ball or at origin
            if (sumSq > 0.0 && sumSq <= 1.0) {
                // Normalize to unit sphere surface
                float invMag = static_cast<float>(1.0 / std::sqrt(sumSq));
                for (auto& val : vec) {
                    val *= invMag;
                }
                return vec;
            }
        }
    }
};

/**
 * @brief Generate random non-unit vectors
 * 
 * Generates vectors with random directions and magnitudes.
 * Useful for testing compression of general vector data.
 * 
 * Options:
 * - Uniform magnitude distribution
 * - Log-normal magnitude distribution
 * - Fixed magnitude range
 * 
 * @tparam T The vector type (must satisfy Vector32Type)
 * @tparam Dimension The dimensionality of the vectors
 */
template<typename T, size_t Dimension>
    requires Vector32Type<T> && (Dimension >= 2)
class NonUnitVectorGenerator : public DataGenerator<T> {
public:
    enum class MagnitudeDistribution {
        Uniform,      // Uniform in [minMag, maxMag]
        LogNormal,    // Log-normal distribution (common in real data)
        Exponential,  // Exponential distribution
        Fixed         // All vectors have same magnitude
    };
    
    /**
     * @brief Construct a non-unit vector generator
     * 
     * @param minMagnitude Minimum vector magnitude (default: 0.1)
     * @param maxMagnitude Maximum vector magnitude (default: 10.0)
     * @param seed Random seed for reproducibility (default: 42)
     * @param magnitudeDist Magnitude distribution type (default: Uniform)
     */
    explicit NonUnitVectorGenerator(
        float minMagnitude = 0.1f,
        float maxMagnitude = 10.0f,
        uint32_t seed = 42,
        MagnitudeDistribution magnitudeDist = MagnitudeDistribution::Uniform
    ) : rng_(seed), 
        seed_(seed),
        minMag_(minMagnitude),
        maxMag_(maxMagnitude),
        magnitudeDist_(magnitudeDist),
        normalDist_(0.0, 1.0),
        uniformMagDist_(minMagnitude, maxMagnitude),
        logNormalDist_(0.0, 1.0),  // Will be scaled
        exponentialDist_(1.0) {
        
        // Validate parameters
        if (minMagnitude <= 0.0f || maxMagnitude <= 0.0f || minMagnitude > maxMagnitude) {
            throw std::invalid_argument("Invalid magnitude range");
        }
    }
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> vectors;
        vectors.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            vectors.push_back(generateSingle());
        }
        
        return vectors;
    }
    
    std::string name() const override {
        std::string distName;
        switch (magnitudeDist_) {
            case MagnitudeDistribution::Uniform: distName = "Uniform"; break;
            case MagnitudeDistribution::LogNormal: distName = "LogNormal"; break;
            case MagnitudeDistribution::Exponential: distName = "Exponential"; break;
            case MagnitudeDistribution::Fixed: distName = "Fixed"; break;
        }
        return "NonUnitVectorGenerator<" + std::to_string(Dimension) + "D>_" + distName;
    }
    
    DataType dataType() const override {
        return core::DataType::Vector32;
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::map<std::string, std::string> getConfig() const override {
        std::string distName;
        switch (magnitudeDist_) {
            case MagnitudeDistribution::Uniform: distName = "uniform"; break;
            case MagnitudeDistribution::LogNormal: distName = "lognormal"; break;
            case MagnitudeDistribution::Exponential: distName = "exponential"; break;
            case MagnitudeDistribution::Fixed: distName = "fixed"; break;
        }
        
        return {
            {"type", "non_unit_vector"},
            {"dimension", std::to_string(Dimension)},
            {"seed", std::to_string(seed_)},
            {"min_magnitude", std::to_string(minMag_)},
            {"max_magnitude", std::to_string(maxMag_)},
            {"magnitude_distribution", distName},
            {"normalized", "false"}
        };
    }
    
private:
    std::mt19937 rng_;
    uint32_t seed_;
    float minMag_;
    float maxMag_;
    MagnitudeDistribution magnitudeDist_;
    
    std::normal_distribution<float> normalDist_;
    std::uniform_real_distribution<float> uniformMagDist_;
    std::lognormal_distribution<float> logNormalDist_;
    std::exponential_distribution<float> exponentialDist_;
    
    T generateSingle() {
        // Generate random direction (unit vector)
        T vec;
        if constexpr (requires { vec.resize(Dimension); }) {
            vec.resize(Dimension);
        }
        
        double sumSq = 0.0;
        
        // Sample direction from normal distribution
        size_t idx = 0;
        for (auto it = vec.begin(); it != vec.end() && idx < Dimension; ++it, ++idx) {
            float val = normalDist_(rng_);
            *it = val;
            sumSq += static_cast<double>(val) * val;
        }
        
        // Normalize to unit length
        float invMag = static_cast<float>(1.0 / std::sqrt(sumSq));
        for (auto& val : vec) {
            val *= invMag;
        }
        
        // Scale by random magnitude
        float magnitude = generateMagnitude();
        for (auto& val : vec) {
            val *= magnitude;
        }
        
        return vec;
    }
    
    float generateMagnitude() {
        switch (magnitudeDist_) {
            case MagnitudeDistribution::Uniform:
                return uniformMagDist_(rng_);
                
            case MagnitudeDistribution::LogNormal: {
                // Log-normal distribution scaled to [minMag, maxMag]
                float val = std::exp(logNormalDist_(rng_));
                // Scale to range
                float range = maxMag_ - minMag_;
                return minMag_ + (val / (1.0f + val)) * range;
            }
            
            case MagnitudeDistribution::Exponential: {
                // Exponential distribution scaled to [minMag, maxMag]
                float val = exponentialDist_(rng_);
                // Clamp and scale
                val = std::min(val, 5.0f);  // Clamp outliers
                return minMag_ + (val / 5.0f) * (maxMag_ - minMag_);
            }
            
            case MagnitudeDistribution::Fixed:
                // All vectors have the same magnitude (average)
                return (minMag_ + maxMag_) / 2.0f;
                
            default:
                return uniformMagDist_(rng_);
        }
    }
};

/**
 * @brief Generate vectors with specific patterns (for testing)
 * 
 * Useful for testing edge cases and specific compression scenarios:
 * - Clustered vectors (nearby in direction)
 * - Sparse vectors (many zeros)
 * - Coordinate-aligned vectors
 * - Repeated patterns
 */
template<typename T, size_t Dimension>
    requires Vector32Type<T> && (Dimension >= 2)
class PatternedVectorGenerator : public DataGenerator<T> {
public:
    enum class Pattern {
        Clustered,         // Vectors clustered in small regions
        Sparse,            // Many components are zero
        CoordinateAligned, // Aligned with coordinate axes
        Repeated,          // Repeating patterns
        Smooth             // Smoothly varying directions
    };
    
    explicit PatternedVectorGenerator(
        Pattern pattern,
        uint32_t seed = 42,
        float clusterRadius = 0.1f
    ) : rng_(seed), 
        seed_(seed),
        pattern_(pattern),
        clusterRadius_(clusterRadius),
        normalDist_(0.0, 1.0),
        uniformDist_(0.0, 1.0) {
    }
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> vectors;
        vectors.reserve(count);
        
        switch (pattern_) {
            case Pattern::Clustered:
                return generateClustered(count);
            case Pattern::Sparse:
                return generateSparse(count);
            case Pattern::CoordinateAligned:
                return generateCoordinateAligned(count);
            case Pattern::Repeated:
                return generateRepeated(count);
            case Pattern::Smooth:
                return generateSmooth(count);
            default:
                return generateClustered(count);
        }
    }
    
    std::string name() const override {
        std::string patternName;
        switch (pattern_) {
            case Pattern::Clustered: patternName = "Clustered"; break;
            case Pattern::Sparse: patternName = "Sparse"; break;
            case Pattern::CoordinateAligned: patternName = "CoordinateAligned"; break;
            case Pattern::Repeated: patternName = "Repeated"; break;
            case Pattern::Smooth: patternName = "Smooth"; break;
        }
        return "PatternedVectorGenerator<" + std::to_string(Dimension) + "D>_" + patternName;
    }
    
    DataType dataType() const override {
        return core::DataType::Vector32;
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::map<std::string, std::string> getConfig() const override {
        std::string patternName;
        switch (pattern_) {
            case Pattern::Clustered: patternName = "clustered"; break;
            case Pattern::Sparse: patternName = "sparse"; break;
            case Pattern::CoordinateAligned: patternName = "coordinate_aligned"; break;
            case Pattern::Repeated: patternName = "repeated"; break;
            case Pattern::Smooth: patternName = "smooth"; break;
        }
        
        return {
            {"type", "patterned_vector"},
            {"dimension", std::to_string(Dimension)},
            {"seed", std::to_string(seed_)},
            {"pattern", patternName},
            {"cluster_radius", std::to_string(clusterRadius_)}
        };
    }
    
private:
    std::mt19937 rng_;
    uint32_t seed_;
    Pattern pattern_;
    float clusterRadius_;
    
    std::normal_distribution<float> normalDist_;
    std::uniform_real_distribution<float> uniformDist_;
    
    std::vector<T> generateClustered(size_t count) {
        std::vector<T> vectors;
        vectors.reserve(count);
        
        // Generate a few cluster centers
        size_t numClusters = std::max(size_t(1), count / 100);
        std::vector<T> centers;
        UnitVectorGenerator<T, Dimension> centerGen(seed_);
        centers = centerGen.generate(numClusters);
        
        // Generate vectors near cluster centers
        for (size_t i = 0; i < count; ++i) {
            const T& center = centers[i % numClusters];
            T vec = perturbVector(center, clusterRadius_);
            vectors.push_back(vec);
        }
        
        return vectors;
    }
    
    std::vector<T> generateSparse(size_t count) {
        std::vector<T> vectors;
        vectors.reserve(count);
        
        // Each vector has only a few non-zero components
        size_t nonZeroCount = std::max(size_t(1), Dimension / 10);
        
        for (size_t i = 0; i < count; ++i) {
            T vec;
            if constexpr (requires { vec.resize(Dimension); }) {
                vec.resize(Dimension);
            }
            
            // Initialize to zero
            for (auto& val : vec) {
                val = 0.0f;
            }
            
            // Set a few random components
            for (size_t j = 0; j < nonZeroCount; ++j) {
                size_t idx = std::uniform_int_distribution<size_t>(0, Dimension - 1)(rng_);
                float val = normalDist_(rng_);
                size_t currentIdx = 0;
                for (auto it = vec.begin(); it != vec.end(); ++it, ++currentIdx) {
                    if (currentIdx == idx) {
                        *it = val;
                        break;
                    }
                }
            }
            
            // Normalize
            double sumSq = 0.0;
            for (const auto& val : vec) {
                sumSq += static_cast<double>(val) * val;
            }
            if (sumSq > 0.0) {
                float invMag = static_cast<float>(1.0 / std::sqrt(sumSq));
                for (auto& val : vec) {
                    val *= invMag;
                }
            }
            
            vectors.push_back(vec);
        }
        
        return vectors;
    }
    
    std::vector<T> generateCoordinateAligned(size_t count) {
        std::vector<T> vectors;
        vectors.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            T vec;
            if constexpr (requires { vec.resize(Dimension); }) {
                vec.resize(Dimension);
            }
            
            // Pick a random axis
            size_t axis = i % Dimension;
            
            // Set that axis to ±1, others to small noise
            size_t currentIdx = 0;
            for (auto it = vec.begin(); it != vec.end(); ++it, ++currentIdx) {
                if (currentIdx == axis) {
                    *it = (uniformDist_(rng_) > 0.5f) ? 1.0f : -1.0f;
                } else {
                    *it = normalDist_(rng_) * 0.1f;  // Small noise
                }
            }
            
            // Normalize
            double sumSq = 0.0;
            for (const auto& val : vec) {
                sumSq += static_cast<double>(val) * val;
            }
            float invMag = static_cast<float>(1.0 / std::sqrt(sumSq));
            for (auto& val : vec) {
                val *= invMag;
            }
            
            vectors.push_back(vec);
        }
        
        return vectors;
    }
    
    std::vector<T> generateRepeated(size_t count) {
        // Generate a small set of vectors and repeat them
        size_t uniqueCount = std::max(size_t(1), count / 10);
        UnitVectorGenerator<T, Dimension> gen(seed_);
        auto unique = gen.generate(uniqueCount);
        
        std::vector<T> vectors;
        vectors.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            vectors.push_back(unique[i % uniqueCount]);
        }
        
        return vectors;
    }
    
    std::vector<T> generateSmooth(size_t count) {
        std::vector<T> vectors;
        vectors.reserve(count);
        
        // Generate a smooth trajectory on the sphere
        UnitVectorGenerator<T, Dimension> gen(seed_);
        T current = gen.generate(1)[0];
        
        float stepSize = 0.1f;  // Angular step size
        
        for (size_t i = 0; i < count; ++i) {
            vectors.push_back(current);
            current = perturbVector(current, stepSize);
        }
        
        return vectors;
    }
    
    T perturbVector(const T& vec, float radius) {
        T result;
        if constexpr (requires { result.resize(Dimension); }) {
            result.resize(Dimension);
        }
        
        // Add Gaussian noise and renormalize
        double sumSq = 0.0;
        
        size_t idx = 0;
        auto vecIt = vec.begin();
        for (auto it = result.begin(); it != result.end() && idx < Dimension; ++it, ++vecIt, ++idx) {
            float noise = normalDist_(rng_) * radius;
            float val = *vecIt + noise;
            *it = val;
            sumSq += static_cast<double>(val) * val;
        }
        
        // Normalize
        float invMag = static_cast<float>(1.0 / std::sqrt(sumSq));
        for (auto& val : result) {
            val *= invMag;
        }
        
        return result;
    }
};

} // namespace encodings::datagen
