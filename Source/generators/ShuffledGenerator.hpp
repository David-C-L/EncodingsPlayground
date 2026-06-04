#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <string>
#include "generators/DataGenerator.hpp"

namespace encodings::datagen {

template<typename T>
class ShuffledGenerator : public DataGenerator<T> {
public:
    explicit ShuffledGenerator(std::shared_ptr<DataGenerator<T>> inner,
                               size_t seed = 42)
        : inner_(std::move(inner)), seed_(seed), rng_(seed) {}

    std::vector<T> generate(size_t count) override {
        auto data = inner_->generate(count);
        std::shuffle(data.begin(), data.end(), rng_);
        return data;
    }

    std::string name() const override {
        return "Shuffled(" + inner_->name() + ")";
    }

    void reset() override {
        inner_->reset();
        rng_.seed(seed_);
    }

    std::map<std::string, std::string> getConfig() const override {
        auto cfg = inner_->getConfig();
        cfg["shuffle_seed"] = std::to_string(seed_);
        return cfg;
    }

private:
    std::shared_ptr<DataGenerator<T>> inner_;
    size_t seed_;
    std::mt19937_64 rng_;
};

} // namespace encodings::datagen
