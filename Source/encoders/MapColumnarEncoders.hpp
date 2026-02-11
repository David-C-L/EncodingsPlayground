#pragma once

#include "encoders/MapEncoders.hpp"
#include "encoders/RunLengthEncoder.hpp"
#include <algorithm>

namespace encodings::encoders {

/**
 * @brief Strategy 4: Columnar layout with pairs, then dictionary encode
 * 
 * Transform to columnar layout (all keys together, all values together),
 * then encode as pairs.
 * 
 * Format: [size_encoding][num_keys][keys...][values...]
 */
template<typename K, typename V>
class MapColumnarPairsDictEncoder : public MapEncoder<K, V> {
public:
    using MapType = typename MapEncoder<K, V>::MapType;
    
    MapColumnarPairsDictEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::shared_ptr<Codec<V>> valueEncoder)
        : keyEncoder_(keyEncoder), valueEncoder_(valueEncoder) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
        this->properties_.add(EncodingProperty::Vectorizable);
    }
    
    EncodedData encode(std::span<const MapType> data) override {
        if (data.empty()) {
            return this->createEncodedData(std::vector<uint8_t>{}, 0, 0, "MapColumnarPairsDict");
        }
        
        // Encode sizes
        auto sizeData = this->encodeMapSizes(data);
        
        // Extract all keys and values in columnar layout
        std::vector<K> allKeys;
        std::vector<V> allValues;
        
        for (const auto& map : data) {
            for (const auto& [key, value] : map) {
                allKeys.push_back(key);
                allValues.push_back(value);
            }
        }
        
        // Encode in columnar format
        auto encodedKeys = keyEncoder_->encode(allKeys);
        auto encodedValues = valueEncoder_->encode(allValues);
        
        // Combine: [size_len][key_len][value_len][num_keys][sizes][keys][values]
        std::vector<uint8_t> result;
        uint32_t sizeLen = sizeData.size();
        uint32_t keyLen = encodedKeys.size();
        uint32_t valueLen = encodedValues.size();
        uint32_t numKeys = allKeys.size();
        
        result.resize(4 * sizeof(uint32_t) + sizeLen + keyLen + valueLen);
        
        size_t offset = 0;
        std::memcpy(result.data() + offset, &sizeLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &keyLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &valueLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &numKeys, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        std::memcpy(result.data() + offset, sizeData.data(), sizeLen);
        offset += sizeLen;
        std::memcpy(result.data() + offset, encodedKeys.data().data(), keyLen);
        offset += keyLen;
        std::memcpy(result.data() + offset, encodedValues.data().data(), valueLen);
        
        size_t uncompressedSize = data.size() * sizeof(MapType);
        return this->createEncodedData(std::move(result), data.size(), uncompressedSize, "MapColumnarPairsDict");
    }
    
    std::vector<MapType> decodeAll(const EncodedData& encoded) override {
        if (encoded.empty()) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        uint32_t sizeLen, keyLen, valueLen, numKeys;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&keyLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&valueLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&numKeys, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        auto sizes = this->decodeMapSizes(data, offset);
        
        // Calculate total pairs
        size_t totalPairs = 0;
        for (size_t size : sizes) {
            totalPairs += size;
        }
        
        std::vector<uint8_t> keyBytes(data + offset, data + offset + keyLen);
        EncodingMetadata keyMetadata;
        keyMetadata.encodingName = "SubDecoder";
        keyMetadata.dataType = core::DataType::Int32;
        keyMetadata.elementCount = totalPairs;
        EncodedData keyData(std::move(keyBytes), std::move(keyMetadata));
        offset += keyLen;
        auto allKeys = keyEncoder_->decodeAll(keyData);
        
        std::vector<uint8_t> valueBytes(data + offset, data + offset + valueLen);
        EncodingMetadata valueMetadata;
        valueMetadata.encodingName = "SubDecoder";
        valueMetadata.dataType = core::DataType::Int32;
        valueMetadata.elementCount = totalPairs;
        EncodedData valueData(std::move(valueBytes), std::move(valueMetadata));
        auto allValues = valueEncoder_->decodeAll(valueData);
        
        // Reconstruct maps from columnar layout
        std::vector<MapType> result;
        result.reserve(sizes.size());
        
        size_t keyValueIdx = 0;
        for (size_t mapSize : sizes) {
            MapType map;
            for (size_t i = 0; i < mapSize; ++i) {
                map[allKeys[keyValueIdx]] = allValues[keyValueIdx];
                keyValueIdx++;
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "MapColumnarPairsDict";
    }
    
private:
    std::shared_ptr<Codec<K>> keyEncoder_;
    std::shared_ptr<Codec<V>> valueEncoder_;
};

/**
 * @brief Strategy 5: Columnar separate with dictionary encoding on both
 * 
 * Store all keys together, all values together in separate vectors.
 * Both are dictionary encoded.
 * 
 * Format: [size_encoding][key_encoding][value_encoding]
 */
template<typename K, typename V>
class MapColumnarSeparateDictEncoder : public MapEncoder<K, V> {
public:
    using MapType = typename MapEncoder<K, V>::MapType;
    
    MapColumnarSeparateDictEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::shared_ptr<Codec<V>> valueEncoder)
        : keyEncoder_(keyEncoder), valueEncoder_(valueEncoder) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
        this->properties_.add(EncodingProperty::Vectorizable);
    }
    
    EncodedData encode(std::span<const MapType> data) override {
        if (data.empty()) {
            return this->createEncodedData(std::vector<uint8_t>{}, 0, 0, "MapColumnarSeparateDict");
        }
        
        // Encode sizes
        auto sizeData = this->encodeMapSizes(data);
        
        // Separate keys and values into columnar vectors
        std::vector<K> keyColumn;
        std::vector<V> valueColumn;
        
        for (const auto& map : data) {
            for (const auto& [key, value] : map) {
                keyColumn.push_back(key);
                valueColumn.push_back(value);
            }
        }
        
        // Dictionary encode each column separately
        auto encodedKeys = keyEncoder_->encode(keyColumn);
        auto encodedValues = valueEncoder_->encode(valueColumn);
        
        // Combine
        std::vector<uint8_t> result;
        uint32_t sizeLen = sizeData.size();
        uint32_t keyLen = encodedKeys.size();
        uint32_t valueLen = encodedValues.size();
        
        result.resize(3 * sizeof(uint32_t) + sizeLen + keyLen + valueLen);
        
        size_t offset = 0;
        std::memcpy(result.data() + offset, &sizeLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &keyLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &valueLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        std::memcpy(result.data() + offset, sizeData.data(), sizeLen);
        offset += sizeLen;
        std::memcpy(result.data() + offset, encodedKeys.data().data(), keyLen);
        offset += keyLen;
        std::memcpy(result.data() + offset, encodedValues.data().data(), valueLen);
        
        size_t uncompressedSize = data.size() * sizeof(MapType);
        return this->createEncodedData(std::move(result), data.size(), uncompressedSize, "MapColumnarSeparateDict");
    }
    
    std::vector<MapType> decodeAll(const EncodedData& encoded) override {
        if (encoded.empty()) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        uint32_t sizeLen, keyLen, valueLen;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&keyLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&valueLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        auto sizes = this->decodeMapSizes(data, offset);
        
        // Calculate total pairs
        size_t totalPairs = 0;
        for (size_t size : sizes) {
            totalPairs += size;
        }
        
        std::vector<uint8_t> keyBytes(data + offset, data + offset + keyLen);
        EncodingMetadata keyMetadata;
        keyMetadata.encodingName = "SubDecoder";
        keyMetadata.dataType = core::DataType::Int32;
        keyMetadata.elementCount = totalPairs;
        EncodedData keyData(std::move(keyBytes), std::move(keyMetadata));
        offset += keyLen;
        auto keyColumn = keyEncoder_->decodeAll(keyData);
        
        std::vector<uint8_t> valueBytes(data + offset, data + offset + valueLen);
        EncodingMetadata valueMetadata;
        valueMetadata.encodingName = "SubDecoder";
        valueMetadata.dataType = core::DataType::Int32;
        valueMetadata.elementCount = totalPairs;
        EncodedData valueData(std::move(valueBytes), std::move(valueMetadata));
        auto valueColumn = valueEncoder_->decodeAll(valueData);
        
        // Reconstruct maps from columnar data
        std::vector<MapType> result;
        result.reserve(sizes.size());
        
        size_t idx = 0;
        for (size_t mapSize : sizes) {
            MapType map;
            for (size_t i = 0; i < mapSize; ++i) {
                map[keyColumn[idx]] = valueColumn[idx];
                idx++;
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "MapColumnarSeparateDict";
    }
    
private:
    std::shared_ptr<Codec<K>> keyEncoder_;
    std::shared_ptr<Codec<V>> valueEncoder_;
};

/**
 * @brief Strategy 6: Columnar with Delta+RLE keys, Dictionary values
 * 
 * Best for maps with sequential/monotonic keys and low-cardinality values.
 * 
 * Format: [size_encoding][key_encoding][value_encoding]
 */
template<typename K, typename V>
class MapColumnarMixedEncoder : public MapEncoder<K, V> {
public:
    using MapType = typename MapEncoder<K, V>::MapType;
    
    MapColumnarMixedEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,      // Delta+RLE encoder
        std::shared_ptr<Codec<V>> valueEncoder)    // Dictionary encoder
        : keyEncoder_(keyEncoder), valueEncoder_(valueEncoder) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
        this->properties_.add(EncodingProperty::Vectorizable);
    }
    
    EncodedData encode(std::span<const MapType> data) override {
        if (data.empty()) {
            return this->createEncodedData(std::vector<uint8_t>{}, 0, 0, "MapColumnarMixed");
        }
        
        // Encode sizes
        auto sizeData = this->encodeMapSizes(data);
        
        // Extract columnar data
        std::vector<K> keyColumn;
        std::vector<V> valueColumn;
        
        for (const auto& map : data) {
            for (const auto& [key, value] : map) {
                keyColumn.push_back(key);
                valueColumn.push_back(value);
            }
        }
        
        // Encode keys with Delta+RLE, values with Dictionary
        auto encodedKeys = keyEncoder_->encode(keyColumn);
        auto encodedValues = valueEncoder_->encode(valueColumn);
        
        // Combine
        std::vector<uint8_t> result;
        uint32_t sizeLen = sizeData.size();
        uint32_t keyLen = encodedKeys.size();
        uint32_t valueLen = encodedValues.size();
        
        result.resize(3 * sizeof(uint32_t) + sizeLen + keyLen + valueLen);
        
        size_t offset = 0;
        std::memcpy(result.data() + offset, &sizeLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &keyLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &valueLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        std::memcpy(result.data() + offset, sizeData.data(), sizeLen);
        offset += sizeLen;
        std::memcpy(result.data() + offset, encodedKeys.data().data(), keyLen);
        offset += keyLen;
        std::memcpy(result.data() + offset, encodedValues.data().data(), valueLen);
        
        size_t uncompressedSize = data.size() * sizeof(MapType);
        return this->createEncodedData(std::move(result), data.size(), uncompressedSize, "MapColumnarMixed");
    }
    
    std::vector<MapType> decodeAll(const EncodedData& encoded) override {
        if (encoded.empty()) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        uint32_t sizeLen, keyLen, valueLen;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&keyLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&valueLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        auto sizes = this->decodeMapSizes(data, offset);
        
        // Calculate total pairs
        size_t totalPairs = 0;
        for (size_t size : sizes) {
            totalPairs += size;
        }
        
        std::vector<uint8_t> keyBytes(data + offset, data + offset + keyLen);
        EncodingMetadata keyMetadata;
        keyMetadata.encodingName = "SubDecoder";
        keyMetadata.dataType = core::DataType::Int32;
        keyMetadata.elementCount = totalPairs;
        EncodedData keyData(std::move(keyBytes), std::move(keyMetadata));
        offset += keyLen;
        auto keyColumn = keyEncoder_->decodeAll(keyData);
        
        std::vector<uint8_t> valueBytes(data + offset, data + offset + valueLen);
        EncodingMetadata valueMetadata;
        valueMetadata.encodingName = "SubDecoder";
        valueMetadata.dataType = core::DataType::Int32;
        valueMetadata.elementCount = totalPairs;
        EncodedData valueData(std::move(valueBytes), std::move(valueMetadata));
        auto valueColumn = valueEncoder_->decodeAll(valueData);
        
        // Reconstruct maps
        std::vector<MapType> result;
        result.reserve(sizes.size());
        
        size_t idx = 0;
        for (size_t mapSize : sizes) {
            MapType map;
            for (size_t i = 0; i < mapSize; ++i) {
                map[keyColumn[idx]] = valueColumn[idx];
                idx++;
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "MapColumnarMixed";
    }
    
private:
    std::shared_ptr<Codec<K>> keyEncoder_;
    std::shared_ptr<Codec<V>> valueEncoder_;
};


/**
 * @brief Strategy 4: Columnar layout with pairs, then dictionary encode
 * 
 * Transform to columnar layout (all keys together, all values together),
 * then encode as pairs.
 * 
 * Format: [size_encoding][num_keys][keys...][values...]
 */
template<typename K, typename V, typename S = uint32_t>
class MapGroupIndicesEncoder : public MapEncoder<K, V> {
public:
    using MapType = typename MapEncoder<K, V>::MapType;

    MapGroupIndicesEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::shared_ptr<Codec<V>> valueEncoder,
        std::shared_ptr<Codec<S>> sizeCodec)
        : keyEncoder_(keyEncoder), valueEncoder_(valueEncoder), sizeCodec_(sizeCodec) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
        this->properties_.add(EncodingProperty::Vectorizable);
    }

    MapGroupIndicesEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::shared_ptr<Codec<V>> valueEncoder)
        : MapGroupIndicesEncoder(keyEncoder, valueEncoder, std::make_shared<RunLengthEncoder<S>>()) {}
    
    EncodedData encode(std::span<const MapType> data) override {
        if (data.empty()) {
            return this->createEncodedData(std::vector<uint8_t>{}, 0, 0, "MapGroupIndices");
        }
        
        // Encode sizes using the provided size codec
        auto encodedSizes = this->encodeMapSizesWithCodec(data, sizeCodec_);

        // Find the largest map size to determine number of groups
        // Also build frequency map for efficient reservation
        size_t largestMapSize = 0;
        std::map<size_t, size_t> sizeFrequency;
        for (const auto& map : data) {
            if (map.size() > largestMapSize) {
                largestMapSize = map.size();
            }
            sizeFrequency[map.size()]++;
        }

        // Group keys and values by their position within each map
        // Each group i contains all k-v pairs at position i across all maps
        std::vector<std::vector<K>> keyGroups(largestMapSize);
        std::vector<std::vector<V>> valueGroups(largestMapSize);
        
        // Reserve space: group i will have elements from all maps with size > i
        for (size_t i = 0; i < largestMapSize; ++i) {
            size_t groupSize = 0;
            for (const auto& [size, freq] : sizeFrequency) {
                if (size > i) {
                    groupSize += freq;
                }
            }
            keyGroups[i].reserve(groupSize);
            valueGroups[i].reserve(groupSize);
        }

        for (const auto& map : data) {
            size_t currI = 0;
            for (const auto& [key, value] : map) {
                keyGroups[currI].push_back(key);
                valueGroups[currI].push_back(value);
                currI++;
            }
        }

        // Encode each group separately
        std::vector<EncodedData> encodedKeyGroups(largestMapSize);
        std::vector<EncodedData> encodedValueGroups(largestMapSize);
        std::vector<uint32_t> keyGroupSizes(largestMapSize);
        std::vector<uint32_t> valueGroupSizes(largestMapSize);
        
        for (size_t i = 0; i < largestMapSize; ++i) {
            encodedKeyGroups[i] = keyEncoder_->encode(keyGroups[i]);
            encodedValueGroups[i] = valueEncoder_->encode(valueGroups[i]);
            keyGroupSizes[i] = encodedKeyGroups[i].size();
            valueGroupSizes[i] = encodedValueGroups[i].size();
        }

        // Calculate total encoded lengths
        uint32_t sizeLen = encodedSizes.size();
        uint32_t keyLen = std::accumulate(keyGroupSizes.begin(), keyGroupSizes.end(), 0u);
        uint32_t valueLen = std::accumulate(valueGroupSizes.begin(), valueGroupSizes.end(), 0u);
        uint32_t numGroups = static_cast<uint32_t>(largestMapSize);
        
        // Format: [size_len][key_len][value_len][num_groups]
        //         [sizes][key_group_sizes...][value_group_sizes...]
        //         [key_groups...][value_groups...]
        std::vector<uint8_t> result;
        size_t headerSize = 4 * sizeof(uint32_t) + sizeLen + 
                            2 * numGroups * sizeof(uint32_t);
        result.resize(headerSize + keyLen + valueLen);
        
        size_t offset = 0;
        std::memcpy(result.data() + offset, &sizeLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &keyLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &valueLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &numGroups, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Write sizes
        std::memcpy(result.data() + offset, encodedSizes.data().data(), sizeLen);
        offset += sizeLen;
        
        // Write encoded group sizes (needed to find boundaries in byte stream)
        std::memcpy(result.data() + offset, keyGroupSizes.data(), numGroups * sizeof(uint32_t));
        offset += numGroups * sizeof(uint32_t);
        std::memcpy(result.data() + offset, valueGroupSizes.data(), numGroups * sizeof(uint32_t));
        offset += numGroups * sizeof(uint32_t);
        
        // Write key groups
        for (const auto& encoded : encodedKeyGroups) {
            std::memcpy(result.data() + offset, encoded.data().data(), encoded.size());
            offset += encoded.size();
        }
        
        // Write value groups
        for (const auto& encoded : encodedValueGroups) {
            std::memcpy(result.data() + offset, encoded.data().data(), encoded.size());
            offset += encoded.size();
        }

        size_t uncompressedSize = 0;
        for (const auto& map : data) {
            uncompressedSize += map.size() * (sizeof(K) + sizeof(V));
        }
        return this->createEncodedData(std::move(result), data.size(), uncompressedSize, "MapGroupIndices");
    }
    
    std::vector<MapType> decodeAll(const EncodedData& encoded) override {
        if (encoded.empty()) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        // Read header
        uint32_t sizeLen, keyLen, valueLen, numGroups;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&keyLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&valueLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&numGroups, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Decode sizes using the size codec
        std::vector<uint8_t> sizeBytes(data + offset, data + offset + sizeLen);
        EncodingMetadata sizeMetadata;
        sizeMetadata.encodingName = "SizeDecoder";
        sizeMetadata.elementCount = encoded.metadata().elementCount;
        EncodedData sizeData(std::move(sizeBytes), std::move(sizeMetadata));
        auto sizes = this->decodeMapSizesWithCodec(sizeData, sizeCodec_);
        offset += sizeLen;
        
        // Read encoded group sizes
        std::vector<uint32_t> keyGroupSizes(numGroups);
        std::vector<uint32_t> valueGroupSizes(numGroups);
        std::memcpy(keyGroupSizes.data(), data + offset, numGroups * sizeof(uint32_t));
        offset += numGroups * sizeof(uint32_t);
        std::memcpy(valueGroupSizes.data(), data + offset, numGroups * sizeof(uint32_t));
        offset += numGroups * sizeof(uint32_t);
        
        // Calculate how many elements are in each group
        std::vector<size_t> groupElementCounts(numGroups, 0);
        for (const auto& size : sizes) {
            for (size_t i = 0; i < size && i < numGroups; ++i) {
                groupElementCounts[i]++;
            }
        }
        
        // Decode each key group
        std::vector<std::vector<K>> keyGroups(numGroups);
        for (size_t i = 0; i < numGroups; ++i) {
            std::vector<uint8_t> groupBytes(data + offset, data + offset + keyGroupSizes[i]);
            EncodingMetadata groupMetadata;
            groupMetadata.encodingName = "GroupDecoder";
            groupMetadata.elementCount = groupElementCounts[i];
            EncodedData groupData(std::move(groupBytes), std::move(groupMetadata));
            keyGroups[i] = keyEncoder_->decodeAll(groupData);
            offset += keyGroupSizes[i];
        }
        
        // Decode each value group
        std::vector<std::vector<V>> valueGroups(numGroups);
        for (size_t i = 0; i < numGroups; ++i) {
            std::vector<uint8_t> groupBytes(data + offset, data + offset + valueGroupSizes[i]);
            EncodingMetadata groupMetadata;
            groupMetadata.encodingName = "GroupDecoder";
            groupMetadata.elementCount = groupElementCounts[i];
            EncodedData groupData(std::move(groupBytes), std::move(groupMetadata));
            valueGroups[i] = valueEncoder_->decodeAll(groupData);
            offset += valueGroupSizes[i];
        }
        
        // Reconstruct maps from groups
        std::vector<MapType> result;
        result.reserve(sizes.size());
        
        for (size_t mapIdx = 0; mapIdx < sizes.size(); ++mapIdx) {
            MapType map;
            for (size_t posIdx = 0; posIdx < sizes[mapIdx]; ++posIdx) {
                // Find which element in group posIdx corresponds to this map
                size_t groupElementIdx = 0;
                for (size_t m = 0; m < mapIdx; ++m) {
                    if (sizes[m] > posIdx) {
                        groupElementIdx++;
                    }
                }
                
                map[keyGroups[posIdx][groupElementIdx]] = valueGroups[posIdx][groupElementIdx];
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "MapGroupIndices" + keyEncoder_->name() + valueEncoder_->name();
    }
    
private:
    std::shared_ptr<Codec<K>> keyEncoder_;
    std::shared_ptr<Codec<V>> valueEncoder_;
    std::shared_ptr<Codec<S>> sizeCodec_;
};


} // namespace encodings::encoders
