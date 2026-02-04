#pragma once

#include <vector>
#include <random>
#include "DataGenerator.hpp"
#include "../core/DataType.hpp"

namespace encodings::datagen {

    using core::NumericType;
    using core::IntegralType;
    using core::FloatingPointType;

    template<NumericType T>
    class RandomMinMaxGenerator : public DataGenerator<T> {
    public:
        RandomMinMaxGenerator(T minValue, T maxValue, int64_t seed)
            : minValue_(std::move(minValue)), maxValue_(std::move(maxValue)), seed_(seed), rng_(seed) {

            if (minValue_ > maxValue_) {
                throw std::invalid_argument("RandomMinMaxGenerator: minValue cannot be greater than maxValue");
            }

            if constexpr (IntegralType<T>) {
                intDist_ = std::uniform_int_distribution<int64_t>(minValue_, maxValue_);
            } else {
                realDist_ = std::uniform_real_distribution<double>(minValue_, maxValue_);
            }
        }

        std::vector<T> generate(size_t count) override {
            std::vector<T> result;
            result.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                if constexpr (IntegralType<T>) {
                    result.push_back(static_cast<T>(intDist_(rng_)));
                } else {
                    result.push_back(static_cast<T>(realDist_(rng_)));
                }
            }
            return result;
        }

        std::string name() const override {
            return "RandomMinMax<" + std::to_string(minValue_) + "," + std::to_string(maxValue_) + ">";
        }

        void reset() override {}

        std::map<std::string, std::string> getConfig() const override {
            return {
                {"minValue", std::to_string(minValue_)},
                {"maxValue", std::to_string(maxValue_)},
                {"seed", std::to_string(seed_)}
            };
        }

    private:
        T minValue_;
        T maxValue_;
        int64_t seed_;
        std::mt19937 rng_;
        std::uniform_int_distribution<int64_t> intDist_;
        std::uniform_real_distribution<double> realDist_;
    };

} // namespace encodings::datagen