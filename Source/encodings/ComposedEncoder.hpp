#pragma once

#include <memory>
#include <string>
#include <vector>
#include <span>
#include <concepts>
#include "Encoder.hpp"
#include "EncodedData.hpp"
#include "EncodingProperty.hpp"

namespace encodings {

/**
 * @brief Sequential composition of two encoders: First -> Second
 * 
 * This enables codec chaining like: Delta -> RLE -> BitPacking
 * 
 * The output of FirstEncoder becomes the input to SecondEncoder.
 * Note: SecondEncoder must work on byte arrays (uint8_t).
 * 
 * @tparam T The original data type
 * @tparam FirstEncoder The first encoder to apply
 * @tparam SecondEncoder The second encoder to apply (operates on bytes)
 */
template<typename TIn, typename TMid, typename TOut,
 typename FirstEncoder, typename SecondEncoder>
    requires std::is_base_of_v<Encoder<TIn, TMid>, FirstEncoder> &&
             std::is_base_of_v<Encoder<TMid, TOut>, SecondEncoder>
class ComposedEncoder : public Codec<TIn, TOut> {
public:
    ComposedEncoder(FirstEncoder first, SecondEncoder second)
        : first_(std::move(first)), second_(std::move(second)) {}

    EncodedBuffer<TOut> encode(std::span<const TIn> data) override {
        // Apply first encoding
        auto intermediateEncoded = first_.encode(data);
        
        // Apply second encoding to the bytes from first encoding
        auto finalEncoded = second_.encode(intermediateEncoded.data());
        
        // Update metadata to reflect composition
        auto& metadata = finalEncoded.metadata();
        metadata.encodingName = name();
        metadata.dataType = this->dataType();
        metadata.encodedType = this->encodedType();
        metadata.elementCount = data.size();
        metadata.uncompressedSize = intermediateEncoded.size();
        metadata.compressedSize = finalEncoded.size();
        metadata.customMetadata["first_encoding"] = first_.name();
        metadata.customMetadata["second_encoding"] = second_.name();
        metadata.customMetadata["intermediate_size"] = 
            std::to_string(intermediateEncoded.size());
        
        return finalEncoded;
    }
    
    std::vector<TIn> decodeAll(const EncodedBuffer<TOut>& encoded) override {
        // Reverse order: decode second, then first
        auto intermediateDecoded = second_.decodeAll(encoded);
        
        // Reconstruct intermediate EncodedData
        EncodedData intermediate(std::move(intermediateDecoded), {
            .encodingName = first_.name(),
            .dataType = first_.encodedType(),
            .encodedType = first_.dataType(),
            .elementCount = 0, // Unknown at this stage
            .compressedSize = intermediate.size(),
            .uncompressedSize = 0, // Unknown at this stage
        });
        
        return first_.decodeAll(intermediate);
    }

    std::optional<TIn> decodeAt(const EncodedBuffer<TOut>& encoded, size_t index) override {
        // Random access only works if both encoders support it
        if (!properties().has(EncodingProperty::RandomAccess)) {
            return std::nullopt;
        }
        
        // This is complex - need to decode through layers
        // For now, fall back to decoding all
        auto all = decodeAll(encoded);
        if (index >= all.size()) return std::nullopt;
        return all[index];
    }

    std::vector<TIn> decodeRange(const EncodedBuffer<TOut>& encoded, size_t start, size_t end) override {
        // Similar challenge - decode all and slice
        auto all = decodeAll(encoded);
        if (start >= all.size()) return {};
        end = std::min(end, all.size());
        return std::vector<TIn>(all.begin() + start, all.begin() + end);
    }
    
    EncodingType encodingType() const override {
        return EncodingType::Composed;
    }
    
    std::string name() const override {
        return first_.name() + " | " + second_.name();
    }
    
    EncodingProperties properties() const override {
        auto firstProps = first_.properties();
        auto secondProps = second_.properties();
        
        // Composition constraints:
        // - Random access only if BOTH support it
        // - Lossless only if BOTH are lossless
        // - Order preserved only if BOTH preserve order
        // - Sequential if EITHER requires sequential
        
        EncodingProperties result;
        
        // Random access requires both
        if (firstProps.has(EncodingProperty::RandomAccess) && 
            secondProps.has(EncodingProperty::RandomAccess)) {
            result |= EncodingProperty::RandomAccess;
        } else {
            result |= EncodingProperty::SequentialOnly;
        }
        
        // Lossless requires both
        if (firstProps.has(EncodingProperty::Lossless) && 
            secondProps.has(EncodingProperty::Lossless)) {
            result |= EncodingProperty::Lossless;
        } else {
            result |= EncodingProperty::Lossy;
        }
        
        // Order preservation requires both
        if (firstProps.has(EncodingProperty::PreservesOrder) && 
            secondProps.has(EncodingProperty::PreservesOrder)) {
            result |= EncodingProperty::PreservesOrder;
        } else if (firstProps.has(EncodingProperty::ReordersData) || 
                   secondProps.has(EncodingProperty::ReordersData)) {
            result |= EncodingProperty::ReordersData;
        }
        
        // Always composable
        result |= EncodingProperty::Composable;
        
        // Memory overhead is additive
        if (firstProps.has(EncodingProperty::HighMemoryOverhead) || 
            secondProps.has(EncodingProperty::HighMemoryOverhead)) {
            result |= EncodingProperty::HighMemoryOverhead;
        }
        
        // Requires full data if either does
        if (firstProps.has(EncodingProperty::RequiresFullData) || 
            secondProps.has(EncodingProperty::RequiresFullData)) {
            result |= EncodingProperty::RequiresFullData;
        }
        
        return result;
    }
    
private:
    FirstEncoder first_;
    SecondEncoder second_;
};

/**
 * @brief Helper function to create composed encoders
 * 
 * Usage: auto composed = composeEncoders(deltaEncoder, rleEncoder);
 */
template<typename TIn, typename TMid, typename TOut,
 typename First, typename Second>
auto composeEncoders(First&& first, Second&& second) {
    return ComposedEncoder<TIn, TMid, TOut, std::decay_t<First>, std::decay_t<Second>>(
        std::forward<First>(first),
        std::forward<Second>(second)
    );
}

/**
 * @brief Three-way composition helper
 */
template<typename TIn, typename TMid1, typename TMid2, typename TOut,
 typename First, typename Second, typename Third>
auto composeEncoders(First&& first, Second&& second, Third&& third) {
    auto firstTwo = composeEncoders<TIn, TMid1, TMid2>(
        std::forward<First>(first), 
        std::forward<Second>(second)
    );
    return composeEncoders<TIn, TMid2, TOut>(
        std::move(firstTwo), 
        std::forward<Third>(third)
    );
}

/**
 * @brief Encoder for composite types that applies different encoders to components
 * 
 * For example: encoding a vector of pairs by applying different encoders to
 * first and second elements.
 * 
 * @tparam Container The container type (e.g., std::vector<std::pair<K,V>>)
 * @tparam ComponentEncoders Encoders for each component
 */
template<typename Container>
class StructuralEncoder {
    // This is a placeholder for structural composition
    // Implementation depends on specific container types
    // See specialized versions below
};

/**
 * @brief Specialized encoder for std::vector<std::pair<K, V>>
 * 
 * Applies separate encoders to keys and values.
 */
template<typename K, typename V, typename KeyEncoder, typename ValueEncoder>
    requires std::is_base_of_v<Encoder<K>, KeyEncoder> &&
             std::is_base_of_v<Encoder<V>, ValueEncoder>
class PairVectorEncoder : public Codec<std::pair<K, V>> {
public:
    PairVectorEncoder(KeyEncoder keyEncoder, ValueEncoder valueEncoder)
        : keyEncoder_(std::move(keyEncoder)), 
          valueEncoder_(std::move(valueEncoder)) {}
    
    EncodedData encode(std::span<const std::pair<K, V>> data) override {
        // Separate keys and values
        std::vector<K> keys;
        std::vector<V> values;
        keys.reserve(data.size());
        values.reserve(data.size());
        
        for (const auto& [key, value] : data) {
            keys.push_back(key);
            values.push_back(value);
        }
        
        // Encode separately
        auto encodedKeys = keyEncoder_.encode(keys);
        auto encodedValues = valueEncoder_.encode(values);
        
        // Combine into single EncodedData
        EncodedData result;
        
        // Store sizes first
        size_t keySize = encodedKeys.size();
        size_t valueSize = encodedValues.size();
        
        result.data().resize(sizeof(size_t) * 2 + keySize + valueSize);
        
        // Write header
        std::memcpy(result.data().data(), &keySize, sizeof(size_t));
        std::memcpy(result.data().data() + sizeof(size_t), &valueSize, sizeof(size_t));
        
        // Write encoded data
        std::memcpy(result.data().data() + sizeof(size_t) * 2, 
                   encodedKeys.data().data(), keySize);
        std::memcpy(result.data().data() + sizeof(size_t) * 2 + keySize, 
                   encodedValues.data().data(), valueSize);
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().elementCount = data.size();
        result.metadata().compressedSize = result.size();
        result.metadata().uncompressedSize = data.size() * (sizeof(K) + sizeof(V));
        
        return result;
    }
    
    std::vector<std::pair<K, V>> decodeAll(const EncodedData& encoded) override {
        // Read header
        size_t keySize, valueSize;
        std::memcpy(&keySize, encoded.data().data(), sizeof(size_t));
        std::memcpy(&valueSize, encoded.data().data() + sizeof(size_t), sizeof(size_t));
        
        // Extract encoded keys and values
        std::vector<uint8_t> keyBytes(keySize);
        std::vector<uint8_t> valueBytes(valueSize);
        
        std::memcpy(keyBytes.data(), 
                   encoded.data().data() + sizeof(size_t) * 2, keySize);
        std::memcpy(valueBytes.data(), 
                   encoded.data().data() + sizeof(size_t) * 2 + keySize, valueSize);
        
        // Decode
        EncodedData keyEncoded(std::move(keyBytes), EncodingMetadata{});
        EncodedData valueEncoded(std::move(valueBytes), EncodingMetadata{});
        
        auto keys = keyEncoder_.decodeAll(keyEncoded);
        auto values = valueEncoder_.decodeAll(valueEncoded);
        
        // Combine
        std::vector<std::pair<K, V>> result;
        result.reserve(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            result.emplace_back(keys[i], values[i]);
        }
        
        return result;
    }
    
    std::optional<std::pair<K, V>> decodeAt(const EncodedData& encoded, size_t index) override {
        // Complex - would need indexed access to both components
        auto all = decodeAll(encoded);
        if (index >= all.size()) return std::nullopt;
        return all[index];
    }
    
    std::vector<std::pair<K, V>> decodeRange(const EncodedData& encoded, 
                                              size_t start, size_t end) override {
        auto all = decodeAll(encoded);
        if (start >= all.size()) return {};
        end = std::min(end, all.size());
        return std::vector<std::pair<K, V>>(all.begin() + start, all.begin() + end);
    }
    
    EncodingType encodingType() const override {
        return EncodingType::Structural;
    }
    
    std::string name() const override {
        return "PairVector<" + keyEncoder_.name() + ", " + valueEncoder_.name() + ">";
    }
    
    EncodingProperties properties() const override {
        auto keyProps = keyEncoder_.properties();
        auto valueProps = valueEncoder_.properties();
        
        // Similar logic to ComposedEncoder
        EncodingProperties result;
        
        // Properties require both components to support them
        if (keyProps.has(EncodingProperty::Lossless) && 
            valueProps.has(EncodingProperty::Lossless)) {
            result |= EncodingProperty::Lossless;
        }
        
        result |= EncodingProperty::Composable;
        
        return result;
    }
    
private:
    KeyEncoder keyEncoder_;
    ValueEncoder valueEncoder_;
};

/**
 * @brief Helper to create pair vector encoders
 */
template<typename K, typename V, typename KeyEnc, typename ValEnc>
auto encodePairs(KeyEnc&& keyEncoder, ValEnc&& valueEncoder) {
    return PairVectorEncoder<K, V, std::decay_t<KeyEnc>, std::decay_t<ValEnc>>(
        std::forward<KeyEnc>(keyEncoder),
        std::forward<ValEnc>(valueEncoder)
    );
}

/**
 * @brief Map encoder that separately encodes keys and values
 */
template<typename K, typename V, typename KeyEncoder, typename ValueEncoder>
    requires std::is_base_of_v<Encoder<K>, KeyEncoder> &&
             std::is_base_of_v<Encoder<V>, ValueEncoder>
class MapEncoder : public Codec<std::map<K, V>> {
public:
    MapEncoder(KeyEncoder keyEncoder, ValueEncoder valueEncoder)
        : keyEncoder_(std::move(keyEncoder)), 
          valueEncoder_(std::move(valueEncoder)) {}
    
    EncodedData encode(std::span<const std::map<K, V>> data) override {
        // For each map, encode as vector of pairs
        // This is simplified - real implementation would be more sophisticated
        
        // Flatten all maps into a single vector of pairs
        std::vector<std::pair<K, V>> allPairs;
        std::vector<size_t> mapSizes;
        
        for (const auto& map : data) {
            mapSizes.push_back(map.size());
            for (const auto& [key, value] : map) {
                allPairs.emplace_back(key, value);
            }
        }
        
        // Use PairVectorEncoder for the pairs
        PairVectorEncoder<K, V, KeyEncoder, ValueEncoder> pairEncoder(
            keyEncoder_, valueEncoder_
        );
        
        auto encodedPairs = pairEncoder.encode(allPairs);
        
        // Prepend map sizes
        EncodedData result;
        size_t headerSize = sizeof(size_t) * (1 + mapSizes.size());
        result.data().resize(headerSize + encodedPairs.size());
        
        // Write number of maps
        size_t numMaps = mapSizes.size();
        std::memcpy(result.data().data(), &numMaps, sizeof(size_t));
        
        // Write map sizes
        std::memcpy(result.data().data() + sizeof(size_t), 
                   mapSizes.data(), mapSizes.size() * sizeof(size_t));
        
        // Write encoded pairs
        std::memcpy(result.data().data() + headerSize,
                   encodedPairs.data().data(), encodedPairs.size());
        
        result.metadata().encodingName = name();
        result.metadata().elementCount = data.size();
        
        return result;
    }
    
    std::vector<std::map<K, V>> decodeAll(const EncodedData& encoded) override {
        // Read number of maps
        size_t numMaps;
        std::memcpy(&numMaps, encoded.data().data(), sizeof(size_t));
        
        // Read map sizes
        std::vector<size_t> mapSizes(numMaps);
        std::memcpy(mapSizes.data(), 
                   encoded.data().data() + sizeof(size_t),
                   numMaps * sizeof(size_t));
        
        // Decode all pairs
        size_t headerSize = sizeof(size_t) * (1 + numMaps);
        std::vector<uint8_t> pairBytes(
            encoded.data().begin() + headerSize, 
            encoded.data().end()
        );
        
        EncodedData pairEncoded(std::move(pairBytes), EncodingMetadata{});
        PairVectorEncoder<K, V, KeyEncoder, ValueEncoder> pairEncoder(
            keyEncoder_, valueEncoder_
        );
        
        auto allPairs = pairEncoder.decodeAll(pairEncoded);
        
        // Reconstruct maps
        std::vector<std::map<K, V>> result;
        result.reserve(numMaps);
        
        size_t pairIndex = 0;
        for (size_t mapSize : mapSizes) {
            std::map<K, V> map;
            for (size_t i = 0; i < mapSize; ++i) {
                map.insert(allPairs[pairIndex++]);
            }
            result.push_back(std::move(map));
        }
        
        return result;
    }
    
    std::optional<std::map<K, V>> decodeAt(const EncodedData& encoded, size_t index) override {
        auto all = decodeAll(encoded);
        if (index >= all.size()) return std::nullopt;
        return all[index];
    }
    
    std::vector<std::map<K, V>> decodeRange(const EncodedData& encoded, 
                                            size_t start, size_t end) override {
        auto all = decodeAll(encoded);
        if (start >= all.size()) return {};
        end = std::min(end, all.size());
        return std::vector<std::map<K, V>>(all.begin() + start, all.begin() + end);
    }
    
    EncodingType encodingType() const override {
        return EncodingType::Structural;
    }
    
    std::string name() const override {
        return "Map<" + keyEncoder_.name() + ", " + valueEncoder_.name() + ">";
    }
    
    EncodingProperties properties() const override {
        auto keyProps = keyEncoder_.properties();
        auto valueProps = valueEncoder_.properties();
        
        EncodingProperties result;
        
        if (keyProps.has(EncodingProperty::Lossless) && 
            valueProps.has(EncodingProperty::Lossless)) {
            result |= EncodingProperty::Lossless;
        }
        
        result |= EncodingProperty::Composable;
        
        return result;
    }
    
private:
    KeyEncoder keyEncoder_;
    ValueEncoder valueEncoder_;
};

} // namespace encodings
