#pragma once

#include "generators/DataGenerator.hpp"
#include <map>
#include <set>
#include <random>
#include <algorithm>

namespace encodings::datagen {

/**
 * @brief Base class for map generators
 */
template<typename K, typename V>
class MapGenerator : public DataGenerator<std::map<K, V>> {
public:
    virtual ~MapGenerator() = default;
};

/**
 * @brief Generate maps with sequential keys
 * 
 * Keys are sequential integers, useful for Delta+RLE encoding.
 */
template<typename V>
class SequentialKeyMapGenerator : public MapGenerator<int32_t, V> {
public:
    using MapType = std::map<int32_t, V>;
    
    SequentialKeyMapGenerator(
        size_t minMapSize = 5,
        size_t maxMapSize = 20,
        int32_t startKey = 0)
        : minMapSize_(minMapSize),
          maxMapSize_(maxMapSize),
          startKey_(startKey) {
    }
    
    std::vector<MapType> generate(size_t count) override {
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> sizeDist(minMapSize_, maxMapSize_);
        std::normal_distribution<double> valueDist(0.0, 100.0);
        
        std::vector<MapType> result;
        result.reserve(count);
        
        int32_t currentKey = startKey_;
        
        for (size_t i = 0; i < count; ++i) {
            MapType map;
            size_t mapSize = sizeDist(rng);
            
            for (size_t j = 0; j < mapSize; ++j) {
                V value;
                if constexpr (std::is_integral_v<V>) {
                    value = static_cast<V>(valueDist(rng));
                } else if constexpr (std::is_floating_point_v<V>) {
                    value = static_cast<V>(valueDist(rng));
                } else {
                    value = V{}; // Default construct
                }
                
                map[currentKey++] = value;
            }
            
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "SequentialKeyMap";
    }
    
    void reset() override {
        // Reset can be a no-op since we use a fixed seed
    }
    
private:
    size_t minMapSize_;
    size_t maxMapSize_;
    int32_t startKey_;
};

/**
 * @brief Generate maps with low-cardinality keys
 * 
 * Keys are drawn from a small set, useful for dictionary encoding.
 */
template<typename K, typename V>
class LowCardinalityKeyMapGenerator : public MapGenerator<K, V> {
public:
    using MapType = std::map<K, V>;
    
    LowCardinalityKeyMapGenerator(
        size_t numUniqueKeys = 10,
        size_t minMapSize = 5,
        size_t maxMapSize = 20)
        : numUniqueKeys_(numUniqueKeys),
          minMapSize_(minMapSize),
          maxMapSize_(maxMapSize) {
    }
    
    std::vector<MapType> generate(size_t count) override {
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> sizeDist(minMapSize_, maxMapSize_);
        std::uniform_int_distribution<size_t> keyDist(0, numUniqueKeys_ - 1);
        std::normal_distribution<double> valueDist(0.0, 100.0);
        
        // Generate key pool
        std::vector<K> keyPool;
        if constexpr (std::is_integral_v<K>) {
            for (size_t i = 0; i < numUniqueKeys_; ++i) {
                keyPool.push_back(static_cast<K>(i * 100));
            }
        }
        
        std::vector<MapType> result;
        result.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            MapType map;
            size_t mapSize = std::min(sizeDist(rng), numUniqueKeys_);
            
            // Use a set to track which keys we've used
            std::set<K> usedKeys;
            
            for (size_t j = 0; j < mapSize; ++j) {
                K key;
                do {
                    key = keyPool[keyDist(rng)];
                } while (usedKeys.count(key) > 0);
                
                usedKeys.insert(key);
                
                V value;
                if constexpr (std::is_integral_v<V>) {
                    value = static_cast<V>(valueDist(rng));
                } else if constexpr (std::is_floating_point_v<V>) {
                    value = static_cast<V>(valueDist(rng));
                } else {
                    value = V{};
                }
                
                map[key] = value;
            }
            
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "LowCardinalityKeyMap";
    }
    
    void reset() override {}
    
private:
    size_t numUniqueKeys_;
    size_t minMapSize_;
    size_t maxMapSize_;
};

/**
 * @brief Generate maps with low-cardinality values
 * 
 * Values are drawn from a small set, useful for dictionary encoding.
 */
template<typename K, typename V>
class LowCardinalityValueMapGenerator : public MapGenerator<K, V> {
public:
    using MapType = std::map<K, V>;
    
    LowCardinalityValueMapGenerator(
        size_t numUniqueValues = 5,
        size_t minMapSize = 5,
        size_t maxMapSize = 20)
        : numUniqueValues_(numUniqueValues),
          minMapSize_(minMapSize),
          maxMapSize_(maxMapSize) {
    }
    
    std::vector<MapType> generate(size_t count) override {
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> sizeDist(minMapSize_, maxMapSize_);
        std::uniform_int_distribution<size_t> valueDist(0, numUniqueValues_ - 1);
        
        // Generate value pool
        std::vector<V> valuePool;
        if constexpr (std::is_integral_v<V>) {
            for (size_t i = 0; i < numUniqueValues_; ++i) {
                valuePool.push_back(static_cast<V>(i * 10));
            }
        } else if constexpr (std::is_floating_point_v<V>) {
            for (size_t i = 0; i < numUniqueValues_; ++i) {
                valuePool.push_back(static_cast<V>(i * 10.5));
            }
        }
        
        std::vector<MapType> result;
        result.reserve(count);
        
        int32_t currentKey = 0;
        
        for (size_t i = 0; i < count; ++i) {
            MapType map;
            size_t mapSize = sizeDist(rng);
            
            for (size_t j = 0; j < mapSize; ++j) {
                K key;
                if constexpr (std::is_integral_v<K>) {
                    key = static_cast<K>(currentKey++);
                }
                
                V value = valuePool[valueDist(rng)];
                map[key] = value;
            }
            
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "LowCardinalityValueMap";
    }
    
    void reset() override {}
    
private:
    size_t numUniqueValues_;
    size_t minMapSize_;
    size_t maxMapSize_;
};

/**
 * @brief Generate maps with constant size
 * 
 * All maps have the same size, useful for RLE size encoding.
 */
template<typename K, typename V>
class ConstantSizeMapGenerator : public MapGenerator<K, V> {
public:
    using MapType = std::map<K, V>;
    
    ConstantSizeMapGenerator(size_t mapSize = 10)
        : mapSize_(mapSize) {
    }
    
    std::vector<MapType> generate(size_t count) override {
        std::mt19937 rng(42);
        std::normal_distribution<double> valueDist(0.0, 100.0);
        
        std::vector<MapType> result;
        result.reserve(count);
        
        int32_t currentKey = 0;
        
        for (size_t i = 0; i < count; ++i) {
            MapType map;
            
            for (size_t j = 0; j < mapSize_; ++j) {
                K key;
                if constexpr (std::is_integral_v<K>) {
                    key = static_cast<K>(currentKey++);
                }
                
                V value;
                if constexpr (std::is_integral_v<V>) {
                    value = static_cast<V>(valueDist(rng));
                } else if constexpr (std::is_floating_point_v<V>) {
                    value = static_cast<V>(valueDist(rng));
                } else {
                    value = V{};
                }
                
                map[key] = value;
            }
            
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "ConstantSizeMap";
    }
    
    void reset() override {}
    
private:
    size_t mapSize_;
};

/**
 * @brief Generate maps with varying sizes
 * 
 * Map sizes follow a distribution, useful for testing size encoding strategies.
 */
template<typename K, typename V>
class VaryingSizeMapGenerator : public MapGenerator<K, V> {
public:
    using MapType = std::map<K, V>;
    
    VaryingSizeMapGenerator(
        size_t minMapSize = 1,
        size_t maxMapSize = 50)
        : minMapSize_(minMapSize),
          maxMapSize_(maxMapSize) {
    }
    
    std::vector<MapType> generate(size_t count) override {
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> sizeDist(minMapSize_, maxMapSize_);
        std::normal_distribution<double> valueDist(0.0, 100.0);
        
        std::vector<MapType> result;
        result.reserve(count);
        
        int32_t currentKey = 0;
        
        for (size_t i = 0; i < count; ++i) {
            MapType map;
            size_t mapSize = sizeDist(rng);
            
            for (size_t j = 0; j < mapSize; ++j) {
                K key;
                if constexpr (std::is_integral_v<K>) {
                    key = static_cast<K>(currentKey++);
                }
                
                V value;
                if constexpr (std::is_integral_v<V>) {
                    value = static_cast<V>(valueDist(rng));
                } else if constexpr (std::is_floating_point_v<V>) {
                    value = static_cast<V>(valueDist(rng));
                } else {
                    value = V{};
                }
                
                map[key] = value;
            }
            
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "VaryingSizeMap";
    }
    
    void reset() override {}
    
private:
    size_t minMapSize_;
    size_t maxMapSize_;
};

/**
 * @brief Generate maps optimized for columnar encoding
 * 
 * Keys have patterns that benefit from Delta encoding,
 * values have low cardinality that benefits from Dictionary encoding.
 */
template<typename V>
class ColumnarOptimizedMapGenerator : public MapGenerator<int32_t, V> {
public:
    using MapType = std::map<int32_t, V>;
    
    ColumnarOptimizedMapGenerator(
        size_t numUniqueValues = 5,
        size_t minMapSize = 10,
        size_t maxMapSize = 20)
        : numUniqueValues_(numUniqueValues),
          minMapSize_(minMapSize),
          maxMapSize_(maxMapSize) {
    }
    
    std::vector<MapType> generate(size_t count) override {
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> sizeDist(minMapSize_, maxMapSize_);
        std::uniform_int_distribution<size_t> valueDist(0, numUniqueValues_ - 1);
        
        // Generate value pool
        std::vector<V> valuePool;
        if constexpr (std::is_integral_v<V>) {
            for (size_t i = 0; i < numUniqueValues_; ++i) {
                valuePool.push_back(static_cast<V>(i * 10));
            }
        } else if constexpr (std::is_floating_point_v<V>) {
            for (size_t i = 0; i < numUniqueValues_; ++i) {
                valuePool.push_back(static_cast<V>(i * 10.5));
            }
        }
        
        std::vector<MapType> result;
        result.reserve(count);
        
        int32_t currentKey = 0;
        
        for (size_t i = 0; i < count; ++i) {
            MapType map;
            size_t mapSize = sizeDist(rng);
            
            for (size_t j = 0; j < mapSize; ++j) {
                // Keys are sequential (good for Delta)
                int32_t key = currentKey++;
                
                // Values are low cardinality (good for Dictionary)
                V value = valuePool[valueDist(rng)];
                
                map[key] = value;
            }
            
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "ColumnarOptimizedMap";
    }
    
    void reset() override {}
    
private:
    size_t numUniqueValues_;
    size_t minMapSize_;
    size_t maxMapSize_;
};

} // namespace datagen
