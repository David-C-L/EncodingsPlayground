#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include "DataGenerator.hpp"
#include "../core/DataType.hpp"

namespace encodings::datagen {

/**
 * @brief TPC-H Lineitem L_PARTKEY column generator
 * 
 * Generates part keys for the TPC-H lineitem table following the specification:
 * - Part keys range from 1 to (scale_factor * 200,000)
 * - Distribution is primarily uniform with optional popularity skew
 * - Some parts are more popular than others (Zipfian-like distribution)
 * - Cardinality: SF * 200,000 unique values
 * 
 * TPC-H Specification:
 * - L_PARTKEY: Foreign key to P_PARTKEY
 * - Type: Identifier (int32_t in this implementation)
 * - Selectivity: Varies based on query, typically 1/200,000 for point queries
 * 
 * Usage:
 *   TPCHLineitemPartKeyGenerator gen(1.0, 42);  // SF=1.0, seed=42
 *   auto partKeys = gen.generate(1000);         // Generate 1000 part keys
 */
class TPCHLineitemPartKeyGenerator : public DataGenerator<int32_t> {
public:
    /**
     * @brief Construct a TPC-H lineitem part key generator
     * 
     * @param scaleFactor TPC-H scale factor (0.01 = 10MB, 1.0 = 1GB, 10.0 = 10GB, etc.)
     * @param seed Random seed for reproducibility
     * @param zipfExponent Skew parameter (0.0 = uniform, 1.0 = Zipfian, default 0.3 for mild skew)
     */
    TPCHLineitemPartKeyGenerator(
        double scaleFactor = 1.0,
        int64_t seed = 42,
        double zipfExponent = 0.3)
        : scaleFactor_(scaleFactor),
          seed_(seed),
          zipfExponent_(zipfExponent),
          rng_(seed) {
        
        if (scaleFactor <= 0.0) {
            throw std::invalid_argument("TPCHLineitemPartKeyGenerator: scaleFactor must be positive");
        }
        
        // TPC-H spec: 200,000 parts per scale factor
        maxPartKey_ = static_cast<int32_t>(scaleFactor * 200000);
        minPartKey_ = 1;
        
        uniformDist_ = std::uniform_real_distribution<double>(0.0, 1.0);
        
        // For Zipfian: Use rejection sampling with fast approximation
        // instead of slow inverse transform sampling
        if (zipfExponent_ > 0.0) {
            // Pre-compute H(N) = sum(1/i^s) for normalization
            // Use approximation for large N: H(N,s) ≈ N^(1-s)/(1-s) for s != 1
            if (std::abs(zipfExponent_ - 1.0) < 1e-6) {
                // s ≈ 1: H(N,1) ≈ ln(N)
                harmonicNumber_ = std::log(static_cast<double>(maxPartKey_));
            } else {
                // s != 1: Use Riemann zeta approximation
                harmonicNumber_ = (std::pow(maxPartKey_, 1.0 - zipfExponent_) - 1.0) / (1.0 - zipfExponent_);
            }
        }
    }
    
    std::vector<int32_t> generate(size_t count) override {
        std::vector<int32_t> result;
        result.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            if (zipfExponent_ > 0.0) {
                result.push_back(generateZipfian());
            } else {
                result.push_back(generateUniform());
            }
        }
        
        return result;
    }
    
    std::string name() const override {
        return "TPCHLineitemPartKey(SF=" + std::to_string(scaleFactor_) + 
               ",zipf=" + std::to_string(zipfExponent_) + ")";
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::map<std::string, std::string> getConfig() const override {
        return {
            {"type", "TPCH_LINEITEM_PARTKEY"},
            {"scaleFactor", std::to_string(scaleFactor_)},
            {"minPartKey", std::to_string(minPartKey_)},
            {"maxPartKey", std::to_string(maxPartKey_)},
            {"cardinality", std::to_string(maxPartKey_)},
            {"zipfExponent", std::to_string(zipfExponent_)},
            {"seed", std::to_string(seed_)}
        };
    }
    
    /**
     * @brief Get the cardinality (number of unique part keys)
     */
    int32_t getCardinality() const {
        return maxPartKey_;
    }
    
    /**
     * @brief Get the scale factor
     */
    double getScaleFactor() const {
        return scaleFactor_;
    }

private:
    /**
     * @brief Generate a uniform random part key
     */
    int32_t generateUniform() {
        std::uniform_int_distribution<int32_t> dist(minPartKey_, maxPartKey_);
        return dist(rng_);
    }
    
    /**
     * @brief Generate a Zipfian-distributed part key (popular items more frequent)
     * 
     * Uses rejection sampling for fast generation.
     * Based on "Quickly Generating Billion-Record Synthetic Databases" (Gray et al.)
     */
    int32_t generateZipfian() {
        // Use rejection sampling with a simpler approximation
        // For Zipf distribution, P(X=k) ∝ 1/k^s
        
        while (true) {
            double u = uniformDist_(rng_);
            double v = uniformDist_(rng_);
            
            // Sample from power law distribution
            double x = std::pow(u, -1.0 / zipfExponent_);
            
            // Scale to our range
            int32_t candidate = static_cast<int32_t>(x);
            
            if (candidate < minPartKey_ || candidate > maxPartKey_) {
                continue; // Reject out of range
            }
            
            // Acceptance test (rejection sampling)
            double prob = std::pow(candidate, -zipfExponent_);
            double acceptance = prob * std::pow(x, zipfExponent_ - 1.0);
            
            if (v <= acceptance) {
                return candidate;
            }
        }
    }
    
    double scaleFactor_;
    int64_t seed_;
    double zipfExponent_;
    int32_t minPartKey_;
    int32_t maxPartKey_;
    double harmonicNumber_;  // For Zipfian normalization
    
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniformDist_;
};

/**
 * @brief TPC-H Lineitem L_PARTKEY generator with clustering
 * 
 * Generates part keys with temporal clustering to simulate real-world patterns
 * where certain parts are popular during specific time periods.
 * This is more realistic than pure Zipfian distribution.
 */
class TPCHLineitemPartKeyClusteredGenerator : public DataGenerator<int32_t> {
public:
    TPCHLineitemPartKeyClusteredGenerator(
        double scaleFactor = 1.0,
        int64_t seed = 42,
        size_t clusterSize = 100,
        size_t numHotParts = 1000)
        : scaleFactor_(scaleFactor),
          seed_(seed),
          clusterSize_(clusterSize),
          numHotParts_(numHotParts),
          rng_(seed),
          currentClusterRemaining_(0) {
        
        maxPartKey_ = static_cast<int32_t>(scaleFactor * 200000);
        minPartKey_ = 1;
        
        uniformDist_ = std::uniform_real_distribution<double>(0.0, 1.0);
        partKeyDist_ = std::uniform_int_distribution<int32_t>(minPartKey_, maxPartKey_);
        hotPartDist_ = std::uniform_int_distribution<int32_t>(minPartKey_, 
            std::min(maxPartKey_, static_cast<int32_t>(numHotParts)));
    }
    
    std::vector<int32_t> generate(size_t count) override {
        std::vector<int32_t> result;
        result.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            // 70% from current cluster, 30% random
            if (currentClusterRemaining_ == 0 || uniformDist_(rng_) > 0.7) {
                // Start new cluster or random access
                currentClusterPart_ = (uniformDist_(rng_) < 0.6) 
                    ? hotPartDist_(rng_)      // 60% hot parts
                    : partKeyDist_(rng_);      // 40% any part
                currentClusterRemaining_ = clusterSize_;
            }
            
            result.push_back(currentClusterPart_);
            currentClusterRemaining_--;
        }
        
        return result;
    }
    
    std::string name() const override {
        return "TPCHLineitemPartKeyClustered(SF=" + std::to_string(scaleFactor_) + ")";
    }
    
    void reset() override {
        rng_.seed(seed_);
        currentClusterRemaining_ = 0;
    }
    
    std::map<std::string, std::string> getConfig() const override {
        return {
            {"type", "TPCH_LINEITEM_PARTKEY_CLUSTERED"},
            {"scaleFactor", std::to_string(scaleFactor_)},
            {"clusterSize", std::to_string(clusterSize_)},
            {"numHotParts", std::to_string(numHotParts_)},
            {"seed", std::to_string(seed_)}
        };
    }

private:
    double scaleFactor_;
    int64_t seed_;
    size_t clusterSize_;
    size_t numHotParts_;
    int32_t minPartKey_;
    int32_t maxPartKey_;
    
    std::mt19937_64 rng_;
    int32_t currentClusterPart_;
    size_t currentClusterRemaining_;
    std::uniform_real_distribution<double> uniformDist_;
    std::uniform_int_distribution<int32_t> partKeyDist_;
    std::uniform_int_distribution<int32_t> hotPartDist_;
};

} // namespace encodings::datagen
