#pragma once

#include "generators/DataGenerator.hpp"
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

namespace encodings::generators {

using namespace encodings::datagen;  // For DataGenerator
using core::PrimitiveType;
using core::IntegralType;
using core::FloatingPointType;

/// Default seed for every generated dataset.
///
/// These generators used to seed from std::random_device in BOTH the constructor
/// and reset(), which broke reproducibility twice over: two processes produced
/// different streams, so a repeated sweep could not be compared against itself,
/// and reset() *changed the stream mid-process*, so a harness that reset between
/// encoders handed each encoder different data and then compared their sizes.
/// A benchmark generator must be a pure function of its seed, and reset() must
/// rewind rather than reroll.  Pass an explicit seed to vary the stream
/// deliberately; nothing varies it by accident.
inline constexpr uint64_t kDefaultGeneratorSeed = 42;

/**
 * @brief Uniform random distribution
 */
template<PrimitiveType T>
class UniformRandomGenerator : public DataGenerator<T> {
public:
    UniformRandomGenerator(T min = T{}, T max = T{}, uint64_t seed = kDefaultGeneratorSeed)
        : min_(min), max_(max), seed_(seed), rng_(seed) {}
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> data;
        data.reserve(count);
        
        if constexpr (IntegralType<T>) {
            std::uniform_int_distribution<T> dist(min_, max_);
            for (size_t i = 0; i < count; ++i) {
                data.push_back(dist(rng_));
            }
        } else if constexpr (FloatingPointType<T>) {
            std::uniform_real_distribution<T> dist(min_, max_);
            for (size_t i = 0; i < count; ++i) {
                data.push_back(dist(rng_));
            }
        }
        
        return data;
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::string name() const override {
        return "UniformRandom";
    }
    
private:
    T min_;
    T max_;
    uint64_t seed_;
    std::mt19937_64 rng_;
};

/**
 * @brief Sequential monotonically increasing values
 */
template<PrimitiveType T>
class SequentialGenerator : public DataGenerator<T> {
public:
    SequentialGenerator(T start = T{0}, T step = T{1})
        : start_(start), step_(step) {}
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> data;
        data.reserve(count);
        
        T value = start_;
        for (size_t i = 0; i < count; ++i) {
            data.push_back(value);
            value += step_;
        }
        
        return data;
    }
    
    void reset() override {}
    
    std::string name() const override {
        return "Sequential";
    }
    
private:
    T start_;
    T step_;
};

/**
 * @brief Repetitive values (perfect for RLE)
 */
template<PrimitiveType T>
class RepetitiveGenerator : public DataGenerator<T> {
public:
    RepetitiveGenerator(size_t runLength = 10, T minValue = T{0}, T maxValue = T{100},
                        uint64_t seed = kDefaultGeneratorSeed)
        : runLength_(runLength), minValue_(minValue), maxValue_(maxValue),
          seed_(seed), rng_(seed) {}
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> data;
        data.reserve(count);
        
        size_t remaining = count;
        
        while (remaining > 0) {
            // Random value
            T value;
            if constexpr (IntegralType<T>) {
                std::uniform_int_distribution<T> dist(minValue_, maxValue_);
                value = dist(rng_);
            } else {
                std::uniform_real_distribution<T> dist(minValue_, maxValue_);
                value = dist(rng_);
            }
            
            // Repeat it
            size_t count_this_run = std::min(runLength_, remaining);
            for (size_t i = 0; i < count_this_run; ++i) {
                data.push_back(value);
            }
            
            remaining -= count_this_run;
        }
        
        return data;
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::string name() const override {
        return "Repetitive";
    }
    
private:
    size_t runLength_;
    T minValue_;
    T maxValue_;
    uint64_t seed_;
    std::mt19937_64 rng_;
};

/**
 * @brief Zipfian distribution (low-cardinality, skewed)
 */
template<IntegralType T>
class ZipfianGenerator : public DataGenerator<T> {
public:
    ZipfianGenerator(size_t cardinality = 100, double exponent = 1.0,
                     uint64_t seed = kDefaultGeneratorSeed)
        : cardinality_(cardinality), exponent_(exponent), seed_(seed), rng_(seed) {
        
        // Precompute cumulative probabilities
        cumulativeProbs_.reserve(cardinality);
        double sum = 0.0;
        for (size_t i = 1; i <= cardinality; ++i) {
            sum += 1.0 / std::pow(static_cast<double>(i), exponent);
        }
        
        double cumulative = 0.0;
        for (size_t i = 1; i <= cardinality; ++i) {
            cumulative += (1.0 / std::pow(static_cast<double>(i), exponent)) / sum;
            cumulativeProbs_.push_back(cumulative);
        }
    }
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> data;
        data.reserve(count);
        
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < count; ++i) {
            double r = dist(rng_);
            
            // Binary search to find the value
            auto it = std::lower_bound(cumulativeProbs_.begin(), cumulativeProbs_.end(), r);
            size_t idx = std::distance(cumulativeProbs_.begin(), it);
            
            data.push_back(static_cast<T>(idx));
        }
        
        return data;
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::string name() const override {
        return "Zipfian";
    }
    
private:
    size_t cardinality_;
    double exponent_;
    uint64_t seed_;
    std::mt19937_64 rng_;
    std::vector<double> cumulativeProbs_;
};

/**
 * @brief Constant value (extreme RLE case)
 */
template<PrimitiveType T>
class ConstantGenerator : public DataGenerator<T> {
public:
    explicit ConstantGenerator(T value = T{42})
        : value_(value) {}
    
    std::vector<T> generate(size_t count) override {
        return std::vector<T>(count, value_);
    }
    
    void reset() override {}
    
    std::string name() const override {
        return "Constant";
    }
    
private:
    T value_;
};

/**
 * @brief Nearly sorted with some noise
 */
template<PrimitiveType T>
class NearlySortedGenerator : public DataGenerator<T> {
public:
    NearlySortedGenerator(T start = T{0}, T step = T{1}, double noiseFraction = 0.1,
                          uint64_t seed = kDefaultGeneratorSeed)
        : start_(start), step_(step), noiseFraction_(noiseFraction),
          seed_(seed), rng_(seed) {}
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> data;
        data.reserve(count);
        
        // Generate sequential
        T value = start_;
        for (size_t i = 0; i < count; ++i) {
            data.push_back(value);
            value += step_;
        }
        
        // Shuffle a fraction
        size_t numToShuffle = static_cast<size_t>(count * noiseFraction_);
        std::uniform_int_distribution<size_t> dist(0, count - 1);
        
        for (size_t i = 0; i < numToShuffle; ++i) {
            size_t idx1 = dist(rng_);
            size_t idx2 = dist(rng_);
            std::swap(data[idx1], data[idx2]);
        }
        
        return data;
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::string name() const override {
        return "NearlySorted";
    }
    
private:
    T start_;
    T step_;
    double noiseFraction_;
    uint64_t seed_;
    std::mt19937_64 rng_;
};

/**
 * @brief Normal distribution
 */
template<FloatingPointType T>
class NormalGenerator : public DataGenerator<T> {
public:
    NormalGenerator(T mean = T{0}, T stddev = T{1}, uint64_t seed = kDefaultGeneratorSeed)
        : mean_(mean), stddev_(stddev), seed_(seed), rng_(seed) {}
    
    std::vector<T> generate(size_t count) override {
        std::vector<T> data;
        data.reserve(count);
        
        std::normal_distribution<T> dist(mean_, stddev_);
        for (size_t i = 0; i < count; ++i) {
            data.push_back(dist(rng_));
        }
        
        return data;
    }
    
    void reset() override {
        rng_.seed(seed_);
    }
    
    std::string name() const override {
        return "Normal";
    }
    
private:
    T mean_;
    T stddev_;
    uint64_t seed_;
    std::mt19937_64 rng_;
};

} // namespace encodings::generators
