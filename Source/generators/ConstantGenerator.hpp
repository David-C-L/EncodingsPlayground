#pragma once

#include "DataGenerator.hpp"

namespace encodings::datagen {

    template<typename T>
    class ConstantGenerator : public DataGenerator<T> {
    public:
        explicit ConstantGenerator(T value) : value_(std::move(value)) {}

        std::vector<T> generate(size_t count) override {
            return std::vector<T>(count, value_);
        }

        std::string name() const override {
            return "Constant<" + std::to_string(value_) + ">";
        }

        void reset() override {}

        std::map<std::string, std::string> getConfig() const override {
            return {{"value", std::to_string(value_)}};
        }

    private:
        T value_;
    };

} // namespace encodings::datagen