#pragma once

#include <span>
#include <vector>
#include <unordered_map>

#include <ankerl/unordered_dense.h>
#include <cstring>
#include <algorithm>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

/**
 * @brief Dictionary encoding for all types
 * 
 * Replaces repeated values with integer keys into a dictionary.
 * Highly effective for data with low cardinality (many repeated values).
 * Supports random access by direct lookup.
 * 
 * Format: [size_of_dict (8 bytes),
 *          size_of_dict_keys_in_bytes (8 bytes),
 *          dict_entries (size_of_dict * sizeof(T)),
 *          keys (num_elements * sizeof(DictKeyType))]
 * 
 * DictKeyType is chosen based on dictionary size:
 * - uint8_t for dict size <= 256
 * - uint16_t for dict size <= 65536
 * - uint32_t otherwise
 * 
 * @tparam T The type to encode
 */
template<typename T>
class DictionaryEncoder : public Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
        if (data.empty()) {
            return createEmptyEncoding();
        }
        
        // Build dictionary and encode keys
        ankerl::unordered_dense::map<T, size_t> valueToKey;
        std::vector<T> dictionary;
        std::vector<size_t> keys;
        keys.reserve(data.size());
        
        for (const auto& value : data) {
            auto it = valueToKey.find(value);
            if (it == valueToKey.end()) {
                size_t newKey = dictionary.size();
                valueToKey[value] = newKey;
                dictionary.push_back(value);
                keys.push_back(newKey);
            } else {
                keys.push_back(it->second);
            }
        }
        
        const size_t dictSize = dictionary.size();
        
        // Choose appropriate key type based on dictionary size
        if (dictSize <= 256) {
            return encodeWithKeyType<uint8_t>(dictionary, keys, data.size());
        } else if (dictSize <= 65536) {
            return encodeWithKeyType<uint16_t>(dictionary, keys, data.size());
        } else {
            return encodeWithKeyType<uint32_t>(dictionary, keys, data.size());
        }
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < 2 * sizeof(size_t)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t dictSize, dictKeysSize;
        std::memcpy(&dictSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&dictKeysSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        if (dictSize == 0) {
            return {};
        }
        
        // Read dictionary
        std::vector<T> dictionary = readDictionary(readPtr, dictSize);
        readPtr += dictSize * getElementSize();
        
        // Determine key type and decode
        const size_t numElements = encoded.metadata().elementCount;
        
        if (dictKeysSize == numElements * sizeof(uint8_t)) {
            return decodeWithKeyType<uint8_t>(dictionary, readPtr, numElements);
        } else if (dictKeysSize == numElements * sizeof(uint16_t)) {
            return decodeWithKeyType<uint16_t>(dictionary, readPtr, numElements);
        } else {
            return decodeWithKeyType<uint32_t>(dictionary, readPtr, numElements);
        }
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.size() < 2 * sizeof(size_t)) {
            return std::nullopt;
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t dictSize, dictKeysSize;
        std::memcpy(&dictSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&dictKeysSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        const size_t numElements = encoded.metadata().elementCount;
        
        if (index >= numElements || dictSize == 0) {
            return std::nullopt;
        }
        
        // Read dictionary
        std::vector<T> dictionary = readDictionary(readPtr, dictSize);
        readPtr += dictSize * getElementSize();
        
        // Read the key at the specified index
        size_t key;
        if (dictKeysSize == numElements * sizeof(uint8_t)) {
            uint8_t k;
            std::memcpy(&k, readPtr + index * sizeof(uint8_t), sizeof(uint8_t));
            key = k;
        } else if (dictKeysSize == numElements * sizeof(uint16_t)) {
            uint16_t k;
            std::memcpy(&k, readPtr + index * sizeof(uint16_t), sizeof(uint16_t));
            key = k;
        } else {
            uint32_t k;
            std::memcpy(&k, readPtr + index * sizeof(uint32_t), sizeof(uint32_t));
            key = k;
        }
        
        if (key >= dictionary.size()) {
            return std::nullopt;
        }
        
        return dictionary[key];
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.size() < 2 * sizeof(size_t)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t dictSize, dictKeysSize;
        std::memcpy(&dictSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&dictKeysSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        const size_t numElements = encoded.metadata().elementCount;
        
        if (start >= numElements || dictSize == 0) {
            return {};
        }
        
        end = std::min(end, numElements);
        
        // Read dictionary
        std::vector<T> dictionary = readDictionary(readPtr, dictSize);
        readPtr += dictSize * getElementSize();
        
        // Decode range based on key type
        if (dictKeysSize == numElements * sizeof(uint8_t)) {
            return decodeRangeWithKeyType<uint8_t>(dictionary, readPtr, start, end);
        } else if (dictKeysSize == numElements * sizeof(uint16_t)) {
            return decodeRangeWithKeyType<uint16_t>(dictionary, readPtr, start, end);
        } else {
            return decodeRangeWithKeyType<uint32_t>(dictionary, readPtr, start, end);
        }
    }
    
    EncodingType encodingType() const override {
        return EncodingType::DictionaryEncoding;
    }
    
    std::string name() const override {
        return "Dictionary";
    }
    
    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::RandomAccess)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::DictionaryBased
            | EncodingProperty::RequiresFullData
            | EncodingProperty::VariableSize
            | EncodingProperty::HighMemoryOverhead
            | EncodingProperty::Composable;
    }
    
    size_t estimateEncodedSize(size_t elementCount) const override {
        // Pessimistic: assume unique values, uint32_t keys
        return 2 * sizeof(size_t) + elementCount * (sizeof(T) + sizeof(uint32_t));
    }
    
private:
    template<typename KeyType>
    EncodedData encodeWithKeyType(const std::vector<T>& dictionary, 
                                   const std::vector<size_t>& keys,
                                   size_t numElements) {
        const size_t dictSize = dictionary.size();
        const size_t dictBytesSize = dictSize * getElementSize();
        const size_t keysBytesSize = numElements * sizeof(KeyType);
        const size_t headerSize = 2 * sizeof(size_t);
        const size_t totalSize = headerSize + dictBytesSize + keysBytesSize;
        
        EncodedData result;
        result.data().resize(totalSize);
        
        uint8_t* writePtr = result.data().data();
        
        // Write header
        std::memcpy(writePtr, &dictSize, sizeof(size_t));
        writePtr += sizeof(size_t);
        
        std::memcpy(writePtr, &keysBytesSize, sizeof(size_t));
        writePtr += sizeof(size_t);
        
        // Write dictionary
        writeDictionary(dictionary, writePtr);
        writePtr += dictBytesSize;
        
        // Write keys
        for (size_t key : keys) {
            KeyType compactKey = static_cast<KeyType>(key);
            std::memcpy(writePtr, &compactKey, sizeof(KeyType));
            writePtr += sizeof(KeyType);
        }
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = numElements;
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = numElements * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["dict_size"] = std::to_string(dictSize);
        result.metadata().customMetadata["key_type_bytes"] = std::to_string(sizeof(KeyType));
        result.metadata().customMetadata["cardinality_ratio"] = 
            std::to_string(static_cast<double>(dictSize) / numElements);
        
        return result;
    }
    
    template<typename KeyType>
    std::vector<T> decodeWithKeyType(const std::vector<T>& dictionary,
                                      const uint8_t* keysPtr,
                                      size_t numElements) {
        std::vector<T> result;
        result.reserve(numElements);
        
        for (size_t i = 0; i < numElements; ++i) {
            KeyType key;
            std::memcpy(&key, keysPtr + i * sizeof(KeyType), sizeof(KeyType));
            result.push_back(dictionary[key]);
        }
        
        return result;
    }
    
    template<typename KeyType>
    std::vector<T> decodeRangeWithKeyType(const std::vector<T>& dictionary,
                                           const uint8_t* keysPtr,
                                           size_t start,
                                           size_t end) {
        std::vector<T> result;
        result.reserve(end - start);
        
        const uint8_t* rangePtr = keysPtr + start * sizeof(KeyType);
        
        for (size_t i = 0; i < (end - start); ++i) {
            KeyType key;
            std::memcpy(&key, rangePtr + i * sizeof(KeyType), sizeof(KeyType));
            result.push_back(dictionary[key]);
        }
        
        return result;
    }
    
    size_t getElementSize() const {
        if constexpr (std::is_trivially_copyable_v<T>) {
            return sizeof(T);
        } else if constexpr (std::is_same_v<T, std::string>) {
            // Variable size, use average estimate
            return 32; // Placeholder
        } else {
            return sizeof(T);
        }
    }
    
    void writeDictionary(const std::vector<T>& dictionary, uint8_t* dest) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(dest, dictionary.data(), dictionary.size() * sizeof(T));
        } else if constexpr (std::is_same_v<T, std::string>) {
            for (const auto& str : dictionary) {
                size_t len = str.size();
                std::memcpy(dest, &len, sizeof(size_t));
                dest += sizeof(size_t);
                std::memcpy(dest, str.data(), len);
                dest += len;
            }
        } else {
            // Fallback
            std::memcpy(dest, dictionary.data(), dictionary.size() * sizeof(T));
        }
    }
    
    std::vector<T> readDictionary(const uint8_t* src, size_t dictSize) {
        std::vector<T> dictionary;
        dictionary.reserve(dictSize);
        
        if constexpr (std::is_trivially_copyable_v<T>) {
            dictionary.resize(dictSize);
            std::memcpy(dictionary.data(), src, dictSize * sizeof(T));
        } else if constexpr (std::is_same_v<T, std::string>) {
            for (size_t i = 0; i < dictSize; ++i) {
                size_t len;
                std::memcpy(&len, src, sizeof(size_t));
                src += sizeof(size_t);
                dictionary.emplace_back(reinterpret_cast<const char*>(src), len);
                src += len;
            }
        } else {
            // Fallback
            dictionary.resize(dictSize);
            std::memcpy(dictionary.data(), src, dictSize * sizeof(T));
        }
        
        return dictionary;
    }
    
    EncodedData createEmptyEncoding() {
        EncodedData result;
        result.data().resize(2 * sizeof(size_t));
        
        size_t zero = 0;
        std::memcpy(result.data().data(), &zero, sizeof(size_t));
        std::memcpy(result.data().data() + sizeof(size_t), &zero, sizeof(size_t));
        
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = 2 * sizeof(size_t);
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
};

} // namespace encodings::encoders
