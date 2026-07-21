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
#include "encoders/detail/DictionaryHelpers.hpp"

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
 * Random access is O(1): the dictionary and key array are accessed via zero-copy
 * pointer views into the encoded buffer — no allocation on the hot path.
 * Header metadata is cached across repeated calls on the same encoded buffer.
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
        const uint32_t keyBitWidth = detail::chooseKeyBitWidth(dictSize, allowNonPowerOfTwoKeyWidths_);

        // Invalidate cache: new encoded data produced.
        cache_.base = nullptr;

        if (keyBitWidth == 32) {
            return encodeRawKeys<uint32_t>(dictionary, keys, data.size(), keyBitWidth);
        }

        return encodeBitPacked(dictionary, keys, data.size(), keyBitWidth);
    }

    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < kHeaderSize) {
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
        std::vector<T> dictionary = detail::readDictionary<T>(readPtr, dictSize);
        readPtr += dictSize * getElementSize();

        const size_t numElements = encoded.metadata().elementCount;

        if (keyBitWidth == 32) {
            return decodeRawKeys<uint32_t>(dictionary, readPtr, numElements);
        }

        return decodeBitPacked(readPtr, numElements, dictKeysSize, keyBitWidth,
                       reinterpret_cast<const uint8_t*>(dictionary.data()));
    }

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const View v = getView(encoded);
        if (v.dictBytes == nullptr || index >= v.numElements) [[unlikely]] {
            return std::nullopt;
        }

        if (v.keyBitWidth == 32) {
            uint32_t k;
            std::memcpy(&k, v.keysData + index * sizeof(uint32_t), sizeof(uint32_t));
            return k < v.dictSize ? std::optional<T>(detail::loadDictValue<T>(v.dictBytes, k)) : std::nullopt;
        }

        const size_t bitOffset = index * v.keyBitWidth;
        const size_t wordIdx   = bitOffset >> 6;
        const uint32_t offset  = static_cast<uint32_t>(bitOffset & 63u);
        uint64_t word;
        std::memcpy(&word, v.keysData + wordIdx * sizeof(uint64_t), sizeof(uint64_t));
        uint64_t key = word >> offset;
        if (offset + v.keyBitWidth > 64u) {
            uint64_t next;
            std::memcpy(&next, v.keysData + (wordIdx + 1) * sizeof(uint64_t), sizeof(uint64_t));
            key |= next << (64u - offset);
        }
        const uint64_t mask = (v.keyBitWidth >= 64u) ? ~uint64_t{0}
                                                      : ((uint64_t{1} << v.keyBitWidth) - 1u);
        key &= mask;
        if (key >= v.dictSize) [[unlikely]] return std::nullopt;
        return detail::loadDictValue<T>(v.dictBytes, static_cast<size_t>(key));
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        const View v = getView(encoded);
        if (v.dictBytes == nullptr || start >= v.numElements) return {};
        end = std::min(end, v.numElements);

        const size_t count = end - start;

        if (v.keyBitWidth == 32) {
            std::vector<T> result(count);
            T* dst = result.data();
            const uint8_t* rangePtr = v.keysData + start * sizeof(uint32_t);
            for (size_t i = 0; i < count; ++i) {
                uint32_t key;
                std::memcpy(&key, rangePtr + i * sizeof(uint32_t), sizeof(uint32_t));
                dst[i] = detail::loadDictValue<T>(v.dictBytes, key);
            }
            return result;
        }

        std::vector<T> result(count);
        detail::decodeGeneral<T>(v.keysData, v.dictBytes, result.data(), count,
                                 v.keyBitWidth, start * v.keyBitWidth);
        return result;
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

    static constexpr size_t kHeaderSize = 2 * sizeof(size_t) + sizeof(uint8_t);

    // ---------------------------------------------------------------------------
    // Zero-copy view into the encoded buffer.
    //
    // For trivially-copyable T (all integral types used in SubIntSplit), the
    // dictionary bytes point directly into EncodedData::data() — no allocation,
    // no copy. Callers must load entries with memcpy because the buffer is only
    // byte-aligned.
    // The hot path for decodeAt is:
    //   1. getView()         → pointer comparison, cache hit → no work
    //   2. index key array   → single memcpy or BitReader.seekToBit + read
    //   3. dict entry load    → single memcpy from byte-aligned storage
    // ---------------------------------------------------------------------------
    struct View {
        size_t         numElements{0};
        size_t         dictSize{0};
        uint32_t       keyBitWidth{0};
        size_t         keysSize{0};      // dictKeysSize from header (bytes)
        const uint8_t* dictBytes{nullptr}; // direct pointer into encoded buffer
        const uint8_t* keysData{nullptr};   // direct pointer into encoded buffer
    };

    // ---------------------------------------------------------------------------
    // Per-encoder metadata cache.
    //
    // Parsing the header and computing two data pointers is trivial, but for
    // tight random-access loops (e.g. benchmark sweep over 1M indices) even a
    // few memcpy calls add up.  We cache the last-seen buffer's base pointer
    // and derived View so the hot path is just a pointer comparison.
    //
    // Thread-safety: intentionally NOT thread-safe — each encoder instance is
    // owned by exactly one section codec and called from a single thread.
    // ---------------------------------------------------------------------------
    struct Cache {
        const uint8_t* base{nullptr};
        View           view{};
    };
    mutable Cache cache_;

    View getView(const EncodedData& encoded) const {
        const uint8_t* base = encoded.data().data();
        if (base == cache_.base) [[likely]] {
            return cache_.view;
        }

        if (encoded.size() < kHeaderSize) return {};

        // Parse header — all within the first cache line of the buffer.
        size_t dictSize, keysSize;
        std::memcpy(&dictSize, base,                  sizeof(size_t));
        std::memcpy(&keysSize, base + sizeof(size_t), sizeof(size_t));
        const uint32_t keyBitWidth = base[2 * sizeof(size_t)];

        if (dictSize == 0) {
            cache_.base = base;
            cache_.view = {};
            return {};
        }

        // For non-trivially-copyable T (e.g. std::string) we cannot provide a
        // zero-copy dict pointer.  Fall back to returning a null view; callers
        // will detect this and use the slow path.
        if constexpr (!std::is_trivially_copyable_v<T>) {
            return {};
        }

        View v;
        v.numElements  = encoded.metadata().elementCount;
        v.dictSize     = dictSize;
        v.keyBitWidth  = keyBitWidth;
        v.keysSize     = keysSize;
        v.dictBytes    = base + kHeaderSize;
        v.keysData     = base + kHeaderSize + dictSize * sizeof(T);

        cache_.base = base;
        cache_.view = v;
        return v;
    }

    EncodedData encodeBitPacked(const std::vector<T>& dictionary,
                                 const std::vector<size_t>& keys,
                                 size_t numElements,
                                 uint32_t keyBitWidth) {
        const size_t dictSize = dictionary.size();
        const size_t dictBytesSize = dictSize * getElementSize();
        const size_t keysBitsSize = numElements * static_cast<size_t>(keyBitWidth);
        const size_t keysBytesSize = (keysBitsSize + 7) / 8;
        // Pad to 8-byte boundary so decodeGeneral/decodeBatch can safely load
        // full uint64_t words at any position without reading past the buffer.
        const size_t keysPaddedSize = keysBytesSize + 8;
        const size_t totalSize = kHeaderSize + dictBytesSize + keysPaddedSize;

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
        detail::writeDictionary(dictionary, writePtr);
        writePtr += dictBytesSize;

        // Bit-pack keys
        std::vector<uint8_t> keyBuffer;
        keyBuffer.reserve(keysBytesSize);
        encodings::core::BitWriter writer(keyBuffer, encodings::core::BitOrder::LSB);
        for (size_t key : keys) {
            writer.write(static_cast<uint32_t>(key), keyBitWidth);
        }
        writer.flush();

        // Copy packed keys; the 8-byte pad after them is already zero (resize fills 0).
        std::memcpy(writePtr, keyBuffer.data(), keyBuffer.size());

        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = numElements;
        result.metadata().compressedSize = kHeaderSize + dictBytesSize + keysBytesSize;
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
        const size_t totalSize = kHeaderSize + dictBytesSize + keysBytesSize;

        EncodedData result;
        result.data().resize(totalSize);

        uint8_t* writePtr = result.data().data();

        std::memcpy(writePtr, &dictSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        std::memcpy(writePtr, &keysBytesSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        *writePtr = static_cast<uint8_t>(keyBitWidth);
        writePtr += sizeof(uint8_t);

        detail::writeDictionary(dictionary, writePtr);
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

    std::vector<T> decodeBitPacked(const uint8_t* keysPtr,
                                    size_t numElements,
                                    size_t /*keysBytesSize*/,
                                    uint32_t keyBitWidth,
                                    const uint8_t* dictBytes) {
        std::vector<T> result(numElements);
        detail::dispatchDecode<T>(keysPtr, dictBytes, result.data(), numElements, keyBitWidth, 0);
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

    EncodedData createEmptyEncoding() {
        EncodedData result;
        result.data().resize(kHeaderSize);

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
        result.metadata().compressedSize = kHeaderSize;
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;

        return result;
    }
};

} // namespace encodings::encoders
