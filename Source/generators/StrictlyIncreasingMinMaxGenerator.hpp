#pragma once

#include <random>
#include "DataGenerator.hpp"
#include "../core/DataType.hpp"


namespace encodings::datagen {

    using core::IntegralType;

    template<typename T>
    class StrictlyIncreasingMinMaxGenerator : public DataGenerator<T> {
    public:
        StrictlyIncreasingMinMaxGenerator(T minValue, T maxValue, int64_t minIncrement, int64_t maxIncrement, int64_t seed)
            : minValue_(std::move(minValue)), maxValue_(std::move(maxValue)), 
            minIncrement_(minIncrement), maxIncrement_(maxIncrement), seed_(seed), rng_(seed) {
                if (minValue_ >= maxValue_) {
                    throw std::invalid_argument("StrictlyIncreasingMinMaxGenerator: minValue must be less than maxValue");
                }
                if (minIncrement_ <= 0 || maxIncrement_ <= 0 || minIncrement_ > maxIncrement_) {
                    throw std::invalid_argument("StrictlyIncreasingMinMaxGenerator: Invalid increment range");
                }
                std::vector<int64_t> incrementProbabilities;
                int64_t totalIncrements = maxIncrement_ - minIncrement_ + 1;
                incrementProbabilities.reserve(totalIncrements);
                for (int64_t i = totalIncrements - 1, currWeight = 1; i >= 0; --i, currWeight *= 2) {
                    incrementProbabilities[i] = currWeight;
                }
                incrementDist_ = std::discrete_distribution<int64_t>(incrementProbabilities.begin(), incrementProbabilities.end());
            }

        std::vector<T> generate(size_t count) override {
            std::vector<T> result;
            result.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                int64_t increment = minIncrement_ + incrementDist_(rng_);
                T nextValue = static_cast<T>(static_cast<int64_t>(i == 0 ? minValue_ : result.back()) + increment);
                if (nextValue > maxValue_) {
                    nextValue = maxValue_;
                }
                result.push_back(nextValue);
            }
            return result;
        }

        std::string name() const override {
            return "StrictlyIncreasingMinMax<" + std::to_string(minValue_) + "," + std::to_string(maxValue_) + ">";
        }

        void reset() override {}

        std::map<std::string, std::string> getConfig() const override {
            return {
                {"minValue", std::to_string(minValue_)},
                {"maxValue", std::to_string(maxValue_)},
                {"minIncrement", std::to_string(minIncrement_)},
                {"maxIncrement", std::to_string(maxIncrement_)},
                {"seed", std::to_string(seed_)}
            };
        }

    private:
        T minValue_;
        T maxValue_;
        int64_t minIncrement_;
        int64_t maxIncrement_;
        int64_t seed_;
        std::mt19937 rng_;
        std::discrete_distribution<int64_t> incrementDist_;
    };

} // namespace encodings::datagen