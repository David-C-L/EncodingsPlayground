#pragma once

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "core/DataType.hpp"
#include <map>
#include <vector>
#include <memory>
#include <cstring>

namespace encodings::encoders {

/**
 * @brief Base class for map
 * Different strategies exist for encoding the structure.
 */
template<typename K, typename V>
class MapEncoder : public Codec<std::map<K, V>> {
public:
    using MapType = std::map<K, V>;
    
    virtual ~MapEncoder() = default;
    
    EncodingProperties properties() const override {
        return properties_;
    }
    
    EncodingType encodingType() const override {
        return EncodingType::Structural;
    }
    
    /**
     * @brief Decode a single map at the given index
     * 
     * This requires decoding all maps up to the requested index
     * since maps are stored sequentially.
     */
    std::optional<MapType> decodeAt(const EncodedData& encoded, size_t index) override {
        auto allMaps = this->decodeAll(encoded);
        if (index >= allMaps.size()) {
            return std::nullopt;
        }
        return allMaps[index];
    }
    
    /**
     * @brief Decode a range of maps
     * 
     * Decodes maps in the range [start, end)
     */
    std::vector<MapType> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        auto allMaps = this->decodeAll(encoded);
        if (start >= allMaps.size()) {
            return {};
        }
        
        size_t actualEnd = std::min(end, allMaps.size());
        std::vector<MapType> result;
        result.reserve(actualEnd - start);
        
        for (size_t i = start; i < actualEnd; ++i) {
            result.push_back(std::move(allMaps[i]));
        }
        
        return result;
    }
    
protected:
    EncodingProperties properties_;
    
    /**
     * @brief Create EncodedData with proper metadata
     */
    EncodedData createEncodedData(std::vector<uint8_t> data, 
                                   size_t elementCount, 
                                   size_t uncompressedSize,
                                   const std::string& encoderName) const {
        EncodedData result;
        result.data() = std::move(data);
        result.metadata().encodingName = encoderName;
        result.metadata().dataType = core::DataType::Map;
        result.metadata().elementCount = elementCount;
        result.metadata().compressedSize = result.data().size();
        result.metadata().uncompressedSize = uncompressedSize;
        result.metadata().supportsRandomAccess = false; // Maps don't support random access by default
        return result;
    }

    /** 
     * @brief Encode map sizes using a provided codec for the size type
     */
    template<typename CodecType>
    requires core::IntegralType<CodecType>
    EncodedData encodeMapSizesWithCodec(std::span<const MapType> maps, std::shared_ptr<Codec<CodecType>> sizeCodec) const {
        std::vector<CodecType> sizes;
        sizes.reserve(maps.size());
        for (const auto& map : maps) {
            sizes.push_back(static_cast<CodecType>(map.size()));
        }
        return sizeCodec->encode(sizes);
    }

    /** 
     * @brief Decode map sizes using a provided codec for the size type
     */
    template <typename CodecType>
    requires core::IntegralType<CodecType>
    std::vector<CodecType> decodeMapSizesWithCodec(const EncodedData& encoded, std::shared_ptr<Codec<CodecType>> sizeCodec) const {
        return sizeCodec->decodeAll(encoded);
    }

    /**
     * @brief Encode map sizes using RLE
     * 
     * Since many maps may have the same size, RLE is effective.
     */
    std::vector<uint8_t> encodeMapSizes(std::span<const MapType> maps) const {
        std::vector<uint8_t> result;
        
        if (maps.empty()) {
            return result;
        }
        
        // Simple RLE encoding of sizes
        std::vector<std::pair<size_t, size_t>> runs; // (size, count)
        size_t currentSize = maps[0].size();
        size_t runLength = 1;
        
        for (size_t i = 1; i < maps.size(); ++i) {
            if (maps[i].size() == currentSize) {
                runLength++;
            } else {
                runs.push_back({currentSize, runLength});
                currentSize = maps[i].size();
                runLength = 1;
            }
        }
        runs.push_back({currentSize, runLength});
        
        // Encode: [num_runs, (size, count)...]
        uint32_t numRuns = runs.size();
        result.resize(sizeof(uint32_t) + runs.size() * 2 * sizeof(uint32_t));
        
        size_t offset = 0;
        std::memcpy(result.data() + offset, &numRuns, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        for (const auto& [size, count] : runs) {
            uint32_t sz = static_cast<uint32_t>(size);
            uint32_t cnt = static_cast<uint32_t>(count);
            std::memcpy(result.data() + offset, &sz, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            std::memcpy(result.data() + offset, &cnt, sizeof(uint32_t));
            offset += sizeof(uint32_t);
        }
        
        return result;
    }

    /**
     * @brief Decode map sizes from RLE encoding
     */
    std::vector<size_t> decodeMapSizes(const uint8_t* data, size_t& offset) const {
        std::vector<size_t> sizes;
        
        uint32_t numRuns;
        std::memcpy(&numRuns, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        for (uint32_t i = 0; i < numRuns; ++i) {
            uint32_t size, count;
            std::memcpy(&size, data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            std::memcpy(&count, data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            
            for (uint32_t j = 0; j < count; ++j) {
                sizes.push_back(size);
            }
        }
        
        return sizes;
    }
};

/**
 * @brief Strategy 1: Dictionary encode keys and values separately
 * 
 * Format: [size_encoding][key_encoding][value_encoding]
 */
template<typename K, typename V>
class MapDictSeparateEncoder : public MapEncoder<K, V> {
public:
    using MapType = typename MapEncoder<K, V>::MapType;
    
    MapDictSeparateEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::shared_ptr<Codec<V>> valueEncoder)
        : keyEncoder_(keyEncoder), valueEncoder_(valueEncoder) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
    }
    
    EncodedData encode(std::span<const MapType> data) override {
        if (data.empty()) {
            return this->createEncodedData(std::vector<uint8_t>{}, 0, 0, "MapDictSeparate");
        }
        
        // Encode sizes
        auto sizeData = this->encodeMapSizes(data);
        
        // Flatten all keys and values
        std::vector<K> allKeys;
        std::vector<V> allValues;
        
        for (const auto& map : data) {
            for (const auto& [key, value] : map) {
                allKeys.push_back(key);
                allValues.push_back(value);
            }
        }
        
        // Encode keys and values separately
        auto encodedKeys = keyEncoder_->encode(allKeys);
        auto encodedValues = valueEncoder_->encode(allValues);
        
        // Combine: [size_len][key_len][value_len][sizes][keys][values]
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
        
        // Calculate uncompressed size (approximate as map structures)
        size_t uncompressedSize = data.size() * sizeof(MapType);
        
        return this->createEncodedData(std::move(result), data.size(), uncompressedSize, "MapDictSeparate");
    }
    
    std::vector<MapType> decodeAll(const EncodedData& encoded) override {
        if (encoded.empty()) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        // Read lengths
        uint32_t sizeLen, keyLen, valueLen;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&keyLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&valueLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Decode sizes
        auto sizes = this->decodeMapSizes(data, offset);
        
        // Calculate total number of key-value pairs
        size_t totalPairs = 0;
        for (size_t size : sizes) {
            totalPairs += size;
        }
        
        // Decode keys
        std::vector<uint8_t> keyBytes(data + offset, data + offset + keyLen);
        EncodingMetadata keyMetadata;
        keyMetadata.encodingName = "SubDecoder";
        keyMetadata.dataType = core::DataType::Int32;
        keyMetadata.elementCount = totalPairs;  // Critical for DictionaryEncoder!
        EncodedData keyData(std::move(keyBytes), std::move(keyMetadata));
        offset += keyLen;
        auto allKeys = keyEncoder_->decodeAll(keyData);
        
        // Decode values
        std::vector<uint8_t> valueBytes(data + offset, data + offset + valueLen);
        EncodingMetadata valueMetadata;
        valueMetadata.encodingName = "SubDecoder";
        valueMetadata.dataType = core::DataType::Int32;
        valueMetadata.elementCount = totalPairs;  // Critical for DictionaryEncoder!
        EncodedData valueData(std::move(valueBytes), std::move(valueMetadata));
        auto allValues = valueEncoder_->decodeAll(valueData);
        
        // Reconstruct maps
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
    
    /**
     * @brief Optimized decode of a single map at index
     * 
     * Decodes only the keys/values needed for the target map
     */
    std::optional<MapType> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.empty()) {
            return std::nullopt;
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        // Read lengths
        uint32_t sizeLen, keyLen, valueLen;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&keyLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&valueLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Decode sizes to find our target map
        auto sizes = this->decodeMapSizes(data, offset);
        
        if (index >= sizes.size()) {
            return std::nullopt;
        }
        
        // Calculate total number of key-value pairs
        size_t totalPairs = 0;
        for (size_t size : sizes) {
            totalPairs += size;
        }
        
        // Calculate offset to the keys/values for our target map
        size_t keyValueOffset = 0;
        for (size_t i = 0; i < index; ++i) {
            keyValueOffset += sizes[i];
        }
        
        // Decode all keys and values (we need them all since they're encoded together)
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
        
        // Reconstruct only the target map
        MapType map;
        for (size_t i = 0; i < sizes[index]; ++i) {
            map[allKeys[keyValueOffset + i]] = allValues[keyValueOffset + i];
        }
        
        return map;
    }
    
    /**
     * @brief Optimized decode of a range of maps
     */
    std::vector<MapType> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.empty()) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        // Read lengths
        uint32_t sizeLen, keyLen, valueLen;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&keyLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&valueLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Decode sizes
        auto sizes = this->decodeMapSizes(data, offset);
        
        if (start >= sizes.size()) {
            return {};
        }
        
        size_t actualEnd = std::min(end, sizes.size());
        
        // Calculate total number of key-value pairs
        size_t totalPairs = 0;
        for (size_t size : sizes) {
            totalPairs += size;
        }
        
        // Calculate the key/value range we need
        size_t keyValueStart = 0;
        for (size_t i = 0; i < start; ++i) {
            keyValueStart += sizes[i];
        }
        
        size_t keyValueEnd = keyValueStart;
        for (size_t i = start; i < actualEnd; ++i) {
            keyValueEnd += sizes[i];
        }
        
        // Decode all keys and values
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
        
        // Reconstruct maps in the range
        std::vector<MapType> result;
        result.reserve(actualEnd - start);
        
        size_t keyValueIdx = keyValueStart;
        for (size_t mapIdx = start; mapIdx < actualEnd; ++mapIdx) {
            MapType map;
            for (size_t i = 0; i < sizes[mapIdx]; ++i) {
                map[allKeys[keyValueIdx]] = allValues[keyValueIdx];
                keyValueIdx++;
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "MapDictSeparate";
    }
    
private:
    std::shared_ptr<Codec<K>> keyEncoder_;
    std::shared_ptr<Codec<V>> valueEncoder_;
};

/**
 * @brief Strategy 2: Dictionary encode keys and values together as pairs
 * 
 * Format: [size_encoding][pair_encoding]
 */
template<typename K, typename V>
class MapDictTogetherEncoder : public MapEncoder<K, V> {
public:
    using MapType = typename MapEncoder<K, V>::MapType;
    using PairType = std::pair<K, V>;
    
    MapDictTogetherEncoder(std::shared_ptr<Codec<PairType>> pairEncoder)
        : pairEncoder_(pairEncoder) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
    }
    
    EncodedData encode(std::span<const MapType> data) override {
        if (data.empty()) {
            return this->createEncodedData(std::vector<uint8_t>{}, 0, 0, "MapDictTogether");
        }
        
        // Encode sizes
        auto sizeData = this->encodeMapSizes(data);
        
        // Flatten all pairs
        std::vector<PairType> allPairs;
        for (const auto& map : data) {
            for (const auto& pair : map) {
                allPairs.push_back(pair);
            }
        }
        
        // Encode pairs
        auto encodedPairs = pairEncoder_->encode(allPairs);
        
        // Combine
        std::vector<uint8_t> result;
        uint32_t sizeLen = sizeData.size();
        uint32_t pairLen = encodedPairs.size();
        
        result.resize(2 * sizeof(uint32_t) + sizeLen + pairLen);
        
        size_t offset = 0;
        std::memcpy(result.data() + offset, &sizeLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &pairLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, sizeData.data(), sizeLen);
        offset += sizeLen;
        std::memcpy(result.data() + offset, encodedPairs.data().data(), pairLen);
        
        size_t uncompressedSize = data.size() * sizeof(MapType);
        return this->createEncodedData(std::move(result), data.size(), uncompressedSize, "MapDictTogether");
    }
    
    std::vector<MapType> decodeAll(const EncodedData& encoded) override {
        if (encoded.empty()) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        uint32_t sizeLen, pairLen;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&pairLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        auto sizes = this->decodeMapSizes(data, offset);
        
        // Calculate total pairs
        size_t totalPairs = 0;
        for (size_t size : sizes) {
            totalPairs += size;
        }
        
        std::vector<uint8_t> pairBytes(data + offset, data + offset + pairLen);
        EncodingMetadata pairMetadata;
        pairMetadata.encodingName = "SubDecoder";
        pairMetadata.dataType = core::DataType::Int32;
        pairMetadata.elementCount = totalPairs;
        EncodedData pairData(std::move(pairBytes), std::move(pairMetadata));
        auto allPairs = pairEncoder_->decodeAll(pairData);
        
        std::vector<MapType> result;
        result.reserve(sizes.size());
        
        size_t pairIdx = 0;
        for (size_t mapSize : sizes) {
            MapType map;
            for (size_t i = 0; i < mapSize; ++i) {
                map[allPairs[pairIdx].first] = allPairs[pairIdx].second;
                pairIdx++;
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        return "MapDictTogether";
    }
    
private:
    std::shared_ptr<Codec<PairType>> pairEncoder_;
};

/**
 * @brief Strategy 3: Delta+RLE for keys, Dictionary for values
 * 
 * Useful when keys are sequential/monotonic
 */
template<typename K, typename V>
class MapDeltaRLEDictEncoder : public MapEncoder<K, V> {
public:
    using MapType = typename MapEncoder<K, V>::MapType;
    
    MapDeltaRLEDictEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::shared_ptr<Codec<V>> valueEncoder)
        : keyEncoder_(keyEncoder), valueEncoder_(valueEncoder) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
    }
    
    EncodedData encode(std::span<const MapType> data) override {
        // Same implementation as MapDictSeparateEncoder
        // The difference is in which encoders are passed in (Delta+RLE vs Dict)
        if (data.empty()) {
            return this->createEncodedData(std::vector<uint8_t>{}, 0, 0, "MapDeltaRLEDict");
        }
        
        auto sizeData = this->encodeMapSizes(data);
        
        std::vector<K> allKeys;
        std::vector<V> allValues;
        
        for (const auto& map : data) {
            for (const auto& [key, value] : map) {
                allKeys.push_back(key);
                allValues.push_back(value);
            }
        }
        
        auto encodedKeys = keyEncoder_->encode(allKeys);
        auto encodedValues = valueEncoder_->encode(allValues);
        
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
        return this->createEncodedData(std::move(result), data.size(), uncompressedSize, "MapDeltaRLEDict");
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
        auto allKeys = keyEncoder_->decodeAll(keyData);
        
        std::vector<uint8_t> valueBytes(data + offset, data + offset + valueLen);
        EncodingMetadata valueMetadata;
        valueMetadata.encodingName = "SubDecoder";
        valueMetadata.dataType = core::DataType::Int32;
        valueMetadata.elementCount = totalPairs;
        EncodedData valueData(std::move(valueBytes), std::move(valueMetadata));
        auto allValues = valueEncoder_->decodeAll(valueData);
        
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
        return "MapDeltaRLEDict";
    }
    
private:
    std::shared_ptr<Codec<K>> keyEncoder_;
    std::shared_ptr<Codec<V>> valueEncoder_;
};

} // namespace encodings::encoders
