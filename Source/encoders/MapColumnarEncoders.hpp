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
        this->properties_.add(EncodingProperty::RandomAccess);
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
 * @brief Strategy 7: Columnar with group keys, then codec per group
 * 
 * Transform to flatmap where values are grouped by key across all maps,
 * then encode each group of values separately with its own codec.
 * This is optimal when:
 * - Maps share common keys across instances
 * - Values for each key have good compression patterns
 * 
 * Supports two modes:
 * 1. Single value encoder: Same codec applied to all value groups
 * 2. Per-group encoders: Different codec for each key's values (set at encode time)
 * 
 * Example: [{a:1,b:2}, {a:3,c:4}, {a:5,b:6}]
 *   -> key a: [1,3,5], key b: [2,6], key c: [4]
 *   -> encode each stream with its own optimal codec
 * 
 * Format: [size_encoding][num_groups][group_keys][presence_bitmap]
 *         [value_group_sizes...][value_groups...]
 */
// TODO: make value types be std::variant of all value types possible for encoding
template<typename K, typename V, typename S = uint32_t>
class MapGroupKeysEncoder : public MapEncoder<K, V> {
public:
    using MapType = typename MapEncoder<K, V>::MapType;
    
    /**
     * @brief Construct with single value encoder for all groups
     */
    MapGroupKeysEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::shared_ptr<Codec<V>> valueEncoder,
        std::shared_ptr<Codec<S>> sizeCodec)
        : keyEncoder_(keyEncoder),
          sizeCodec_(sizeCodec),
          singleValueEncoder_(valueEncoder),
          usePerGroupEncoders_(false) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
        this->properties_.add(EncodingProperty::Vectorizable);
    }

    MapGroupKeysEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::shared_ptr<Codec<V>> valueEncoder)
        : MapGroupKeysEncoder(keyEncoder, valueEncoder, std::make_shared<RunLengthEncoder<S>>()) {}
    
    /**
     * @brief Construct with per-group value encoders
     * 
     * @param keyEncoder Encoder for the unique keys
     * @param valueEncoderMap Map from key to its specific value encoder
     * @param defaultValueEncoder Fallback encoder for keys not in the map
     * @param sizeCodec Encoder for map sizes
     */
    MapGroupKeysEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::map<K, std::shared_ptr<Codec<V>>> valueEncoderMap,
        std::shared_ptr<Codec<V>> defaultValueEncoder,
        std::shared_ptr<Codec<S>> sizeCodec)
        : keyEncoder_(keyEncoder),
          valueEncoderMap_(std::move(valueEncoderMap)),
          defaultValueEncoder_(defaultValueEncoder),
          sizeCodec_(sizeCodec),
          usePerGroupEncoders_(true) {
        
        this->properties_.add(EncodingProperty::Lossless);
        this->properties_.add(EncodingProperty::DictionaryBased);
        this->properties_.add(EncodingProperty::Vectorizable);
    }
    
    MapGroupKeysEncoder(
        std::shared_ptr<Codec<K>> keyEncoder,
        std::map<K, std::shared_ptr<Codec<V>>> valueEncoderMap,
        std::shared_ptr<Codec<V>> defaultValueEncoder)
        : MapGroupKeysEncoder(keyEncoder, std::move(valueEncoderMap), defaultValueEncoder, 
                             std::make_shared<RunLengthEncoder<S>>()) {}
    
    EncodedData encode(std::span<const MapType> data) override {
        if (data.empty()) {
            return this->createEncodedData(std::vector<uint8_t>{}, 0, 0, "MapGroupKeys");
        }
        
        // Encode map sizes
        auto encodedSizes = this->encodeMapSizesWithCodec(data, sizeCodec_);

        // Find all unique keys across all maps (preserving insertion order)
        std::vector<K> uniqueKeys;
        std::map<K, size_t> keyToGroupIndex;
        
        for (const auto& map : data) {
            for (const auto& [key, value] : map) {
                if (keyToGroupIndex.find(key) == keyToGroupIndex.end()) {
                    keyToGroupIndex[key] = uniqueKeys.size();
                    uniqueKeys.push_back(key);
                }
            }
        }
        
        size_t numGroups = uniqueKeys.size();
        
        // Build value groups: for each key, collect all values across all maps
        // Also build presence bitmap: tracks which maps contain which keys
        std::vector<std::vector<V>> valueGroups(numGroups);
        std::vector<std::vector<bool>> presenceBitmap(data.size(), std::vector<bool>(numGroups, false));
        
        // First pass: count occurrences to reserve space
        std::vector<size_t> groupCounts(numGroups, 0);
        for (size_t mapIdx = 0; mapIdx < data.size(); ++mapIdx) {
            for (const auto& [key, value] : data[mapIdx]) {
                groupCounts[keyToGroupIndex[key]]++;
            }
        }
        
        for (size_t i = 0; i < numGroups; ++i) {
            valueGroups[i].reserve(groupCounts[i]);
        }
        
        // Second pass: populate value groups and presence bitmap
        for (size_t mapIdx = 0; mapIdx < data.size(); ++mapIdx) {
            for (const auto& [key, value] : data[mapIdx]) {
                size_t groupIdx = keyToGroupIndex[key];
                valueGroups[groupIdx].push_back(value);
                presenceBitmap[mapIdx][groupIdx] = true;
            }
        }
        
        // Encode the unique keys
        auto encodedKeys = keyEncoder_->encode(uniqueKeys);
        
        // Encode presence bitmap as packed bits
        // Each map has numGroups bits indicating presence of each key
        size_t bitmapBytesPerMap = (numGroups + 7) / 8;  // Round up to nearest byte
        std::vector<uint8_t> packedBitmap(data.size() * bitmapBytesPerMap, 0);
        
        for (size_t mapIdx = 0; mapIdx < data.size(); ++mapIdx) {
            for (size_t groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
                if (presenceBitmap[mapIdx][groupIdx]) {
                    size_t bitPosition = mapIdx * numGroups + groupIdx;
                    size_t byteIdx = bitPosition / 8;
                    size_t bitIdx = bitPosition % 8;
                    packedBitmap[byteIdx] |= (1 << bitIdx);
                }
            }
        }
        
        // Encode each value group separately with appropriate encoder
        std::vector<EncodedData> encodedValueGroups(numGroups);
        std::vector<uint32_t> valueGroupSizes(numGroups);
        uint32_t totalValueLen = 0;
        
        for (size_t i = 0; i < numGroups; ++i) {
            // Select encoder for this group
            std::shared_ptr<Codec<V>> encoder;
            if (usePerGroupEncoders_) {
                const K& key = uniqueKeys[i];
                auto it = valueEncoderMap_.find(key);
                encoder = (it != valueEncoderMap_.end()) ? it->second : defaultValueEncoder_;
            } else {
                encoder = singleValueEncoder_;
            }
            
            encodedValueGroups[i] = encoder->encode(valueGroups[i]);
            valueGroupSizes[i] = encodedValueGroups[i].size();
            totalValueLen += valueGroupSizes[i];
        }
        
        uint32_t sizeLen = encodedSizes.size();
        uint32_t keyLen = encodedKeys.size();
        uint32_t numGroupsU32 = static_cast<uint32_t>(numGroups);
        uint32_t bitmapLen = static_cast<uint32_t>(packedBitmap.size());
        
        // Format: [size_len][key_len][bitmap_len][total_value_len][num_groups]
        //         [sizes][keys][bitmap][value_group_sizes...][value_groups...]
        std::vector<uint8_t> result;
        size_t headerSize = 5 * sizeof(uint32_t) + sizeLen + keyLen + bitmapLen +
                            numGroups * sizeof(uint32_t);
        result.resize(headerSize + totalValueLen);
        
        size_t offset = 0;
        std::memcpy(result.data() + offset, &sizeLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &keyLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &bitmapLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &totalValueLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &numGroupsU32, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Write sizes
        std::memcpy(result.data() + offset, encodedSizes.data().data(), sizeLen);
        offset += sizeLen;
        
        // Write keys
        std::memcpy(result.data() + offset, encodedKeys.data().data(), keyLen);
        offset += keyLen;
        
        // Write presence bitmap
        std::memcpy(result.data() + offset, packedBitmap.data(), bitmapLen);
        offset += bitmapLen;
        
        // Write value group sizes
        std::memcpy(result.data() + offset, valueGroupSizes.data(), numGroups * sizeof(uint32_t));
        offset += numGroups * sizeof(uint32_t);
        
        // Write value groups
        for (const auto& encoded : encodedValueGroups) {
            std::memcpy(result.data() + offset, encoded.data().data(), encoded.size());
            offset += encoded.size();
        }
        
        size_t uncompressedSize = 0;
        for (const auto& map : data) {
            uncompressedSize += map.size() * (sizeof(K) + sizeof(V));
        }
        
        return this->createEncodedData(std::move(result), data.size(), uncompressedSize, "MapGroupKeys");
    }
    
    std::vector<MapType> decodeAll(const EncodedData& encoded) override {
        if (encoded.empty()) {
            return {};
        }
        
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        // Read header
        uint32_t sizeLen, keyLen, bitmapLen, totalValueLen, numGroups;
        std::memcpy(&sizeLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&keyLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&bitmapLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&totalValueLen, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&numGroups, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Decode sizes
        std::vector<uint8_t> sizeBytes(data + offset, data + offset + sizeLen);
        EncodingMetadata sizeMetadata;
        sizeMetadata.encodingName = "SizeDecoder";
        sizeMetadata.elementCount = encoded.metadata().elementCount;
        EncodedData sizeData(std::move(sizeBytes), std::move(sizeMetadata));
        auto sizes = this->decodeMapSizesWithCodec(sizeData, sizeCodec_);
        offset += sizeLen;
        
        // Decode keys
        std::vector<uint8_t> keyBytes(data + offset, data + offset + keyLen);
        EncodingMetadata keyMetadata;
        keyMetadata.encodingName = "KeyDecoder";
        keyMetadata.elementCount = numGroups;
        EncodedData keyData(std::move(keyBytes), std::move(keyMetadata));
        auto uniqueKeys = keyEncoder_->decodeAll(keyData);
        offset += keyLen;
        
        // Decode presence bitmap
        std::vector<uint8_t> packedBitmap(data + offset, data + offset + bitmapLen);
        offset += bitmapLen;
        
        // Read value group sizes
        std::vector<uint32_t> valueGroupSizes(numGroups);
        std::memcpy(valueGroupSizes.data(), data + offset, numGroups * sizeof(uint32_t));
        offset += numGroups * sizeof(uint32_t);
        
        // Calculate group element counts from bitmap
        std::vector<size_t> groupElementCounts(numGroups, 0);
        for (size_t mapIdx = 0; mapIdx < sizes.size(); ++mapIdx) {
            for (size_t groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
                size_t bitPosition = mapIdx * numGroups + groupIdx;
                size_t byteIdx = bitPosition / 8;
                size_t bitIdx = bitPosition % 8;
                if (packedBitmap[byteIdx] & (1 << bitIdx)) {
                    groupElementCounts[groupIdx]++;
                }
            }
        }
        
        // Decode value groups with appropriate decoder
        std::vector<std::vector<V>> valueGroups(numGroups);
        for (size_t i = 0; i < numGroups; ++i) {
            std::vector<uint8_t> groupBytes(data + offset, data + offset + valueGroupSizes[i]);
            EncodingMetadata groupMetadata;
            groupMetadata.encodingName = "ValueGroupDecoder";
            groupMetadata.elementCount = groupElementCounts[i];
            EncodedData groupData(std::move(groupBytes), std::move(groupMetadata));
            
            // Select decoder for this group
            std::shared_ptr<Codec<V>> decoder;
            if (usePerGroupEncoders_) {
                const K& key = uniqueKeys[i];
                auto it = valueEncoderMap_.find(key);
                decoder = (it != valueEncoderMap_.end()) ? it->second : defaultValueEncoder_;
            } else {
                decoder = singleValueEncoder_;
            }
            
            valueGroups[i] = decoder->decodeAll(groupData);
            offset += valueGroupSizes[i];
        }
        
        // Reconstruct maps
        std::vector<MapType> result;
        result.reserve(sizes.size());
        
        // Track current index in each value group
        std::vector<size_t> groupIndices(numGroups, 0);
        
        for (size_t mapIdx = 0; mapIdx < sizes.size(); ++mapIdx) {
            MapType map;
            for (size_t groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
                size_t bitPosition = mapIdx * numGroups + groupIdx;
                size_t byteIdx = bitPosition / 8;
                size_t bitIdx = bitPosition % 8;
                if (packedBitmap[byteIdx] & (1 << bitIdx)) {
                    map[uniqueKeys[groupIdx]] = valueGroups[groupIdx][groupIndices[groupIdx]++];
                }
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::string name() const override {
        if (usePerGroupEncoders_) {
            return "MapGroupKeys_" + keyEncoder_->name() + "_PerGroup";
        }
        return "MapGroupKeys_" + keyEncoder_->name() + "_" + singleValueEncoder_->name();
    }
    
private:
    std::shared_ptr<Codec<K>> keyEncoder_;
    std::shared_ptr<Codec<S>> sizeCodec_;
    
    // Single encoder mode
    std::shared_ptr<Codec<V>> singleValueEncoder_;
    
    // Per-group encoder mode
    std::map<K, std::shared_ptr<Codec<V>>> valueEncoderMap_;
    std::shared_ptr<Codec<V>> defaultValueEncoder_;
    bool usePerGroupEncoders_;
};

/**
 * @brief Strategy 8: Columnar with group indices
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
        this->properties_.add(EncodingProperty::RandomAccess);
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
        // Build cumulative frequency from largest to smallest for O(n) reservation
        std::vector<size_t> cumulativeFreq(largestMapSize + 1, 0);
        for (const auto& [size, freq] : sizeFrequency) {
            for (size_t i = 0; i < size && i < largestMapSize; ++i) {
                cumulativeFreq[i] += freq;
            }
        }
        
        for (size_t i = 0; i < largestMapSize; ++i) {
            keyGroups[i].reserve(cumulativeFreq[i]);
            valueGroups[i].reserve(cumulativeFreq[i]);
        }

        for (const auto& map : data) {
            size_t currI = 0;
            for (const auto& [key, value] : map) {
                keyGroups[currI].push_back(key);
                valueGroups[currI].push_back(value);
                currI++;
            }
        }

        // Encode each group separately and calculate lengths
        std::vector<EncodedData> encodedKeyGroups(largestMapSize);
        std::vector<EncodedData> encodedValueGroups(largestMapSize);
        std::vector<uint32_t> keyGroupSizes(largestMapSize);
        std::vector<uint32_t> valueGroupSizes(largestMapSize);
        
        uint32_t keyLen = 0;
        uint32_t valueLen = 0;
        
        for (size_t i = 0; i < largestMapSize; ++i) {
            encodedKeyGroups[i] = keyEncoder_->encode(keyGroups[i]);
            encodedValueGroups[i] = valueEncoder_->encode(valueGroups[i]);
            keyGroupSizes[i] = encodedKeyGroups[i].size();
            valueGroupSizes[i] = encodedValueGroups[i].size();
            keyLen += keyGroupSizes[i];
            valueLen += valueGroupSizes[i];
        }

        uint32_t sizeLen = encodedSizes.size();
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

        // Calculate uncompressed size based on actual data
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
        
        return decodeRangeInternal(data, offset, numGroups, sizes, 0, sizes.size());
    }
    
    std::optional<MapType> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.empty() || index >= encoded.metadata().elementCount) {
            return std::nullopt;
        }
        
        auto result = decodeRange(encoded, index, index + 1);
        if (result.empty()) {
            return std::nullopt;
        }
        return std::move(result[0]);
    }
    
    std::vector<MapType> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.empty() || start >= encoded.metadata().elementCount) {
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
        
        // Decode only the sizes we need for the range
        std::vector<uint8_t> sizeBytes(data + offset, data + offset + sizeLen);
        EncodingMetadata sizeMetadata;
        sizeMetadata.encodingName = "SizeDecoder";
        sizeMetadata.elementCount = encoded.metadata().elementCount;
        EncodedData sizeData(std::move(sizeBytes), std::move(sizeMetadata));
        
        // Decode the specific range of sizes using the codec's decodeRange if available
        auto allSizes = this->decodeMapSizesWithCodec(sizeData, sizeCodec_);
        
        size_t actualEnd = std::min(end, allSizes.size());
        if (start >= actualEnd) {
            return {};
        }
        
        // Extract just the sizes we need
        std::vector<S> sizes(allSizes.begin() + start, allSizes.begin() + actualEnd);
        
        offset += sizeLen;
        
        return decodeRangeInternal(data, offset, numGroups, allSizes, start, actualEnd);
    }
    
    std::string name() const override {
        return "MapGroupIndices" + keyEncoder_->name() + valueEncoder_->name() + sizeCodec_->name();
    }
    
private:
    std::shared_ptr<Codec<K>> keyEncoder_;
    std::shared_ptr<Codec<V>> valueEncoder_;
    std::shared_ptr<Codec<S>> sizeCodec_;
    
    /**
     * @brief Internal method to decode a range of maps efficiently
     * 
     * @param data Pointer to the encoded data (after header)
     * @param offset Offset to the group sizes section
     * @param numGroups Total number of groups
     * @param allSizes All map sizes (needed to calculate group element counts)
     * @param start Start index of maps to decode
     * @param end End index of maps to decode (exclusive)
     */
    std::vector<MapType> decodeRangeInternal(
        const uint8_t* data, 
        size_t offset,
        uint32_t numGroups,
        const std::vector<S>& allSizes,
        size_t start,
        size_t end) {
        
        // Read encoded group sizes
        std::vector<uint32_t> keyGroupSizes(numGroups);
        std::vector<uint32_t> valueGroupSizes(numGroups);
        std::memcpy(keyGroupSizes.data(), data + offset, numGroups * sizeof(uint32_t));
        offset += numGroups * sizeof(uint32_t);
        std::memcpy(valueGroupSizes.data(), data + offset, numGroups * sizeof(uint32_t));
        offset += numGroups * sizeof(uint32_t);
        
        // Calculate how many elements are in each group
        std::vector<size_t> groupElementCounts(numGroups, 0);
        for (const auto& size : allSizes) {
            for (size_t i = 0; i < size && i < numGroups; ++i) {
                groupElementCounts[i]++;
            }
        }
        
        // Determine which groups we actually need to decode
        // We need groups 0 to max(sizes in our range) - 1
        size_t maxSizeInRange = 0;
        for (size_t i = start; i < end; ++i) {
            if (allSizes[i] > maxSizeInRange) {
                maxSizeInRange = allSizes[i];
            }
        }
        
        // Decode only the key groups we need
        std::vector<std::vector<K>> keyGroups(maxSizeInRange);
        size_t keyOffset = offset;
        for (size_t i = 0; i < numGroups; ++i) {
            if (i < maxSizeInRange && groupElementCounts[i] > 0) {
                // Decode this group
                std::vector<uint8_t> groupBytes(data + keyOffset, data + keyOffset + keyGroupSizes[i]);
                EncodingMetadata groupMetadata;
                groupMetadata.encodingName = "GroupDecoder";
                groupMetadata.elementCount = groupElementCounts[i];
                EncodedData groupData(std::move(groupBytes), std::move(groupMetadata));
                keyGroups[i] = keyEncoder_->decodeAll(groupData);
            }
            // Skip this group in the data stream
            keyOffset += keyGroupSizes[i];
        }
        
        // Decode only the value groups we need
        std::vector<std::vector<V>> valueGroups(maxSizeInRange);
        size_t valueOffset = keyOffset;
        for (size_t i = 0; i < numGroups; ++i) {
            if (i < maxSizeInRange && groupElementCounts[i] > 0) {
                // Decode this group
                std::vector<uint8_t> groupBytes(data + valueOffset, data + valueOffset + valueGroupSizes[i]);
                EncodingMetadata groupMetadata;
                groupMetadata.encodingName = "GroupDecoder";
                groupMetadata.elementCount = groupElementCounts[i];
                EncodedData groupData(std::move(groupBytes), std::move(groupMetadata));
                valueGroups[i] = valueEncoder_->decodeAll(groupData);
            }
            // Skip this group in the data stream
            valueOffset += valueGroupSizes[i];
        }
        
        // Precompute group element starting indices for each map
        // groupStartIndices[mapIdx][posIdx] = index in group posIdx where mapIdx's element is
        std::vector<std::vector<size_t>> groupStartIndices(end - start);
        for (size_t mapIdx = start; mapIdx < end; ++mapIdx) {
            groupStartIndices[mapIdx - start].resize(allSizes[mapIdx]);
            for (size_t posIdx = 0; posIdx < allSizes[mapIdx]; ++posIdx) {
                // Count how many maps before this one have an element at posIdx
                size_t count = 0;
                for (size_t m = 0; m < mapIdx; ++m) {
                    if (allSizes[m] > posIdx) {
                        count++;
                    }
                }
                groupStartIndices[mapIdx - start][posIdx] = count;
            }
        }
        
        // Reconstruct only the maps in our range
        std::vector<MapType> result;
        result.reserve(end - start);
        
        for (size_t mapIdx = start; mapIdx < end; ++mapIdx) {
            MapType map;
            size_t localIdx = mapIdx - start;
            for (size_t posIdx = 0; posIdx < allSizes[mapIdx]; ++posIdx) {
                size_t groupElementIdx = groupStartIndices[localIdx][posIdx];
                map[keyGroups[posIdx][groupElementIdx]] = valueGroups[posIdx][groupElementIdx];
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
};


} // namespace encodings::encoders
