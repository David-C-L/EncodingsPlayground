#pragma once

#include <span>
#include <vector>
#include <unordered_map>
#include <optional>

#include <ankerl/unordered_dense.h>
#include <cstring>
#include <algorithm>
#include <bit>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/BitPacker.hpp"

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
 *          key_bit_width (1 byte),
 *          dict_entries (size_of_dict * sizeof(T)),
 *          bit-packed keys (num_elements * key_bit_width bits, LSB order)]
 * 
 * Key bit width is chosen based on dictionary size (ceil(log2(dict_size))
 * snapped to {1, 2, 4, 8, 16}; larger dictionaries fall back to 32-bit
 * uncompressed keys for simplicity.
 * 
 * @tparam T The type to encode
 */
template<typename T>
class DictionaryEncoder : public Codec<T> {
public:
    explicit DictionaryEncoder(bool allowNonPowerOfTwoKeyWidths = true)
        : allowNonPowerOfTwoKeyWidths_(allowNonPowerOfTwoKeyWidths) {}

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
        const uint32_t keyBitWidth = chooseKeyBitWidth(dictSize);

        if (keyBitWidth == 32) {
            return encodeRawKeys<uint32_t>(dictionary, keys, data.size(), keyBitWidth);
        }

        return encodeBitPacked(dictionary, keys, data.size(), keyBitWidth);
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < 2 * sizeof(size_t) + sizeof(uint8_t)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t dictSize, dictKeysSize;
        std::memcpy(&dictSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&dictKeysSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        const uint32_t keyBitWidth = *readPtr;
        readPtr += sizeof(uint8_t);
        
        if (dictSize == 0) {
            return {};
        }
        
        // Read dictionary
        std::vector<T> dictionary = readDictionary(readPtr, dictSize);
        readPtr += dictSize * getElementSize();
        
        const size_t numElements = encoded.metadata().elementCount;

        if (keyBitWidth == 32) {
            return decodeRawKeys<uint32_t>(dictionary, readPtr, numElements);
        }

        return decodeBitPacked(dictionary, readPtr, numElements, dictKeysSize, keyBitWidth);
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.size() < 2 * sizeof(size_t) + sizeof(uint8_t)) {
            return std::nullopt;
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t dictSize, dictKeysSize;
        std::memcpy(&dictSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&dictKeysSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        const uint32_t keyBitWidth = *readPtr;
        readPtr += sizeof(uint8_t);
        
        const size_t numElements = encoded.metadata().elementCount;
        
        if (index >= numElements || dictSize == 0) {
            return std::nullopt;
        }
        
        // Read dictionary
        std::vector<T> dictionary = readDictionary(readPtr, dictSize);
        readPtr += dictSize * getElementSize();
        
        if (keyBitWidth == 32) {
            uint32_t k;
            std::memcpy(&k, readPtr + index * sizeof(uint32_t), sizeof(uint32_t));
            return k < dictionary.size() ? std::optional<T>(dictionary[k]) : std::nullopt;
        }

        const size_t bitOffset = index * keyBitWidth;
        encodings::core::BitReader reader(readPtr, dictKeysSize, encodings::core::BitOrder::LSB);
        reader.seekToBit(bitOffset);
        const uint32_t k = reader.read(keyBitWidth);
        if (k >= dictionary.size()) {
            return std::nullopt;
        }
        return dictionary[k];
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.size() < 2 * sizeof(size_t) + sizeof(uint8_t)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t dictSize, dictKeysSize;
        std::memcpy(&dictSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&dictKeysSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        const uint32_t keyBitWidth = *readPtr;
        readPtr += sizeof(uint8_t);
        
        const size_t numElements = encoded.metadata().elementCount;
        
        if (start >= numElements || dictSize == 0) {
            return {};
        }
        
        end = std::min(end, numElements);
        
        // Read dictionary
        std::vector<T> dictionary = readDictionary(readPtr, dictSize);
        readPtr += dictSize * getElementSize();

        if (keyBitWidth == 32) {
            return decodeRangeRawKeys<uint32_t>(dictionary, readPtr, start, end);
        }

        return decodeRangeBitPacked(dictionary, readPtr, dictKeysSize, start, end, keyBitWidth);
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
        // Pessimistic: assume unique values, 32-bit keys (no compression)
        return 2 * sizeof(size_t) + sizeof(uint8_t) + elementCount * (sizeof(T) + sizeof(uint32_t));
    }
    
private:
    bool allowNonPowerOfTwoKeyWidths_{true};

    uint32_t chooseKeyBitWidth(size_t dictSize) const {
        if (dictSize <= 1) return 1; // still write one bit per entry
        const uint32_t bits = static_cast<uint32_t>(std::max<size_t>(1, std::bit_width(dictSize - 1)));

        if (!allowNonPowerOfTwoKeyWidths_) {
            if (bits <= 1) return 1;
            if (bits <= 2) return 2;
            if (bits <= 4) return 4;
            if (bits <= 8) return 8;
            if (bits <= 16) return 16;
            return 32;
        }

        return bits <= 32 ? bits : 32;
    }

    EncodedData encodeBitPacked(const std::vector<T>& dictionary,
                                 const std::vector<size_t>& keys,
                                 size_t numElements,
                                 uint32_t keyBitWidth) {
        const size_t dictSize = dictionary.size();
        const size_t dictBytesSize = dictSize * getElementSize();
        const size_t keysBitsSize = numElements * static_cast<size_t>(keyBitWidth);
        const size_t keysBytesSize = (keysBitsSize + 7) / 8;
        const size_t headerSize = 2 * sizeof(size_t) + sizeof(uint8_t);
        const size_t totalSize = headerSize + dictBytesSize + keysBytesSize;

        EncodedData result;
        result.data().resize(totalSize);

        uint8_t* writePtr = result.data().data();

        // Write header
        std::memcpy(writePtr, &dictSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        std::memcpy(writePtr, &keysBytesSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        *writePtr = static_cast<uint8_t>(keyBitWidth);
        writePtr += sizeof(uint8_t);

        // Write dictionary
        writeDictionary(dictionary, writePtr);
        writePtr += dictBytesSize;

        // Bit-pack keys
        std::vector<uint8_t> keyBuffer;
        keyBuffer.reserve(keysBytesSize);
        encodings::core::BitWriter writer(keyBuffer, encodings::core::BitOrder::LSB);
        for (size_t key : keys) {
            writer.write(static_cast<uint32_t>(key), keyBitWidth);
        }
        writer.flush();

        // Copy packed keys
        std::memcpy(writePtr, keyBuffer.data(), keyBuffer.size());

        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = numElements;
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = numElements * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["dict_size"] = std::to_string(dictSize);
        result.metadata().customMetadata["key_bits"] = std::to_string(keyBitWidth);
        result.metadata().customMetadata["cardinality_ratio"] = 
            std::to_string(static_cast<double>(dictSize) / numElements);

        return result;
    }

    template<typename KeyType>
    EncodedData encodeRawKeys(const std::vector<T>& dictionary,
                               const std::vector<size_t>& keys,
                               size_t numElements,
                               uint32_t keyBitWidth) {
        const size_t dictSize = dictionary.size();
        const size_t dictBytesSize = dictSize * getElementSize();
        const size_t keysBytesSize = numElements * sizeof(KeyType);
        const size_t headerSize = 2 * sizeof(size_t) + sizeof(uint8_t);
        const size_t totalSize = headerSize + dictBytesSize + keysBytesSize;

        EncodedData result;
        result.data().resize(totalSize);

        uint8_t* writePtr = result.data().data();

        std::memcpy(writePtr, &dictSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        std::memcpy(writePtr, &keysBytesSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        *writePtr = static_cast<uint8_t>(keyBitWidth);
        writePtr += sizeof(uint8_t);

        writeDictionary(dictionary, writePtr);
        writePtr += dictBytesSize;

        for (size_t key : keys) {
            KeyType compactKey = static_cast<KeyType>(key);
            std::memcpy(writePtr, &compactKey, sizeof(KeyType));
            writePtr += sizeof(KeyType);
        }

        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = numElements;
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = numElements * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["dict_size"] = std::to_string(dictSize);
        result.metadata().customMetadata["key_bits"] = std::to_string(keyBitWidth);
        result.metadata().customMetadata["cardinality_ratio"] = 
            std::to_string(static_cast<double>(dictSize) / numElements);

        return result;
    }
    
    std::vector<T> decodeBitPacked(const std::vector<T>& dictionary,
                                    const uint8_t* keysPtr,
                                    size_t numElements,
                                    size_t keysBytesSize,
                                    uint32_t keyBitWidth) {
        std::vector<T> result;
        result.reserve(numElements);

        encodings::core::BitReader reader(keysPtr, keysBytesSize, encodings::core::BitOrder::LSB);
        for (size_t i = 0; i < numElements; ++i) {
            const uint32_t key = reader.read(keyBitWidth);
            result.push_back(dictionary[key]);
        }

        return result;
    }

    template<typename KeyType>
    std::vector<T> decodeRawKeys(const std::vector<T>& dictionary,
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
    
    std::vector<T> decodeRangeBitPacked(const std::vector<T>& dictionary,
                                         const uint8_t* keysPtr,
                                         size_t keysBytesSize,
                                         size_t start,
                                         size_t end,
                                         uint32_t keyBitWidth) {
        std::vector<T> result;
        result.reserve(end - start);

        encodings::core::BitReader reader(keysPtr, keysBytesSize, encodings::core::BitOrder::LSB);
        reader.seekToBit(start * keyBitWidth);

        for (size_t i = start; i < end; ++i) {
            const uint32_t key = reader.read(keyBitWidth);
            result.push_back(dictionary[key]);
        }

        return result;
    }

    template<typename KeyType>
    std::vector<T> decodeRangeRawKeys(const std::vector<T>& dictionary,
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
        const size_t headerSize = 2 * sizeof(size_t) + sizeof(uint8_t);
        result.data().resize(headerSize);

        size_t zero = 0;
        uint8_t keyBits = 0;
        uint8_t* ptr = result.data().data();
        std::memcpy(ptr, &zero, sizeof(size_t));
        ptr += sizeof(size_t);
        std::memcpy(ptr, &zero, sizeof(size_t));
        ptr += sizeof(size_t);
        std::memcpy(ptr, &keyBits, sizeof(uint8_t));

        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = headerSize;
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;

        return result;
    }
};

} // namespace encodings::encoders
