#pragma once

#include "DataGenerator.hpp"


namespace encodings::datagen {

    template<typename K, typename V>
    class MapGeneratorCompositional : public DataGenerator<std::map<K, V>> {
    public:
        MapGeneratorCompositional(std::shared_ptr<DataGenerator<K>> keyGen,
                    std::shared_ptr<DataGenerator<V>> valueGen,
                    std::shared_ptr<DataGenerator<int64_t>> sizeGen)

            : keyGen_(std::move(keyGen)), valueGen_(std::move(valueGen)), sizeGen_(std::move(sizeGen)) {}


        std::vector<std::map<K, V>> generate(size_t count) override {
            std::vector<std::map<K, V>> result;
            result.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                std::map<K, V> mapEntry;
                int64_t mapSize = sizeGen_->generate(1)[0];
                auto keys = keyGen_->generate(mapSize);
                auto values = valueGen_->generate(mapSize);
                for (size_t j = 0; j < static_cast<size_t>(mapSize); ++j) {
                    mapEntry.emplace(std::move(keys[j]), std::move(values[j]));
                }
                result.push_back(std::move(mapEntry));
            }
            return result;
        }

        std::string name() const override {
            return "Map<" + keyGen_->name() + ", " + valueGen_->name() + ">";
        }

        void reset() override {
            keyGen_->reset();
            valueGen_->reset();
            sizeGen_->reset();
        }

        std::map<std::string, std::string> getConfig() const override {
            auto config = keyGen_->getConfig();
            auto valueConfig = valueGen_->getConfig();
            auto sizeConfig = sizeGen_->getConfig();

            // Prefix keys to avoid collisions
            for (const auto& [k, v] : valueConfig) {
                config["value_" + k] = v;
            }
            for (const auto& [k, v] : sizeConfig) {
                config["size_" + k] = v;
            }
            return config;
        }

    private:
        std::shared_ptr<DataGenerator<K>> keyGen_;
        std::shared_ptr<DataGenerator<V>> valueGen_;
        std::shared_ptr<DataGenerator<int64_t>> sizeGen_;

};

} // namespace encodings::datagen