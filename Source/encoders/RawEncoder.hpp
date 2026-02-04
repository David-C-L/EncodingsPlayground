#pragma once

#include <span>
#include <vector>
#include <cstring>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

/**
 * @brief Raw encoder - no compression, just stores data as-is
 * 
 * This encoder provides a baseline for comparison and supports all types.
 * Data is stored in native binary format with full random access support.
 * 
 * Format: [element_count (8 bytes), raw_bytes...]
 * 
 * @tparam T The type of data to encode
 */
template<typename T>
class RawEncoder : public Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
        EncodedData result;
        
        const size_t elementCount = data.size();
        const size_t elementSize = sizeof(T);
        const size_t headerSize = sizeof(size_t);
        const size_t totalSize = headerSize + (elementCount * elementSize);
        
        result.data().resize(totalSize);
        
        // Write element count
        std::memcpy(result.data().data(), &elementCount, sizeof(size_t));
        
        // Write raw data
        if constexpr (std::is_trivially_copyable_v<T>) {
            // For trivially copyable types, direct memcpy
            std::memcpy(result.data().data() + headerSize, 
                       data.data(), 
                       elementCount * elementSize);
        } else {
            // For non-trivial types (e.g., std::string), serialize each element
            uint8_t* writePtr = result.data().data() + headerSize;
            for (const auto& element : data) {
                serializeElement(element, writePtr);
                writePtr += elementSize;
            }
        }
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = elementCount;
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = elementCount * elementSize;
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < sizeof(size_t)) {
            return {};
        }
        
        // Read element count
        size_t elementCount;
        std::memcpy(&elementCount, encoded.data().data(), sizeof(size_t));
        
        std::vector<T> result;
        result.reserve(elementCount);
        
        const size_t headerSize = sizeof(size_t);
        const uint8_t* readPtr = encoded.data().data() + headerSize;
        
        if constexpr (std::is_trivially_copyable_v<T>) {
            // For trivially copyable types, direct memcpy
            result.resize(elementCount);
            std::memcpy(result.data(), readPtr, elementCount * sizeof(T));
        } else {
            // For non-trivial types, deserialize each element
            for (size_t i = 0; i < elementCount; ++i) {
                result.push_back(deserializeElement(readPtr));
                readPtr += sizeof(T);
            }
        }
        
        return result;
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.size() < sizeof(size_t)) {
            return std::nullopt;
        }
        
        // Read element count
        size_t elementCount;
        std::memcpy(&elementCount, encoded.data().data(), sizeof(size_t));
        
        if (index >= elementCount) {
            return std::nullopt;
        }
        
        const size_t headerSize = sizeof(size_t);
        const size_t offset = headerSize + (index * sizeof(T));
        const uint8_t* readPtr = encoded.data().data() + offset;
        
        if constexpr (std::is_trivially_copyable_v<T>) {
            T value;
            std::memcpy(&value, readPtr, sizeof(T));
            return value;
        } else {
            return deserializeElement(readPtr);
        }
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.size() < sizeof(size_t)) {
            return {};
        }
        
        // Read element count
        size_t elementCount;
        std::memcpy(&elementCount, encoded.data().data(), sizeof(size_t));
        
        if (start >= elementCount) {
            return {};
        }
        
        end = std::min(end, elementCount);
        const size_t count = end - start;
        
        std::vector<T> result;
        result.reserve(count);
        
        const size_t headerSize = sizeof(size_t);
        const size_t offset = headerSize + (start * sizeof(T));
        const uint8_t* readPtr = encoded.data().data() + offset;
        
        if constexpr (std::is_trivially_copyable_v<T>) {
            result.resize(count);
            std::memcpy(result.data(), readPtr, count * sizeof(T));
        } else {
            for (size_t i = 0; i < count; ++i) {
                result.push_back(deserializeElement(readPtr));
                readPtr += sizeof(T);
            }
        }
        
        return result;
    }
    
    EncodingType encodingType() const override {
        return EncodingType::RawEncoding;
    }
    
    std::string name() const override {
        return "Raw";
    }
    
    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::RandomAccess)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::FixedSize
            | EncodingProperty::SupportsUpdates
            | EncodingProperty::StreamingFriendly
            | EncodingProperty::Vectorizable
            | EncodingProperty::LowMemoryOverhead
            | EncodingProperty::Composable;
    }
    
    size_t estimateEncodedSize(size_t elementCount) const override {
        return sizeof(size_t) + (elementCount * sizeof(T));
    }
    
private:
    // Helper for non-trivial types (specializations needed for actual use)
    void serializeElement(const T& element, uint8_t* dest) {
        if constexpr (std::is_same_v<T, std::string>) {
            // For strings: store length + data
            size_t len = element.size();
            std::memcpy(dest, &len, sizeof(size_t));
            std::memcpy(dest + sizeof(size_t), element.data(), len);
        } else {
            // Fallback: treat as trivially copyable
            std::memcpy(dest, &element, sizeof(T));
        }
    }
    
    T deserializeElement(const uint8_t* src) {
        if constexpr (std::is_same_v<T, std::string>) {
            size_t len;
            std::memcpy(&len, src, sizeof(size_t));
            return std::string(reinterpret_cast<const char*>(src + sizeof(size_t)), len);
        } else {
            T value;
            std::memcpy(&value, src, sizeof(T));
            return value;
        }
    }
};

} // namespace encodings::encoders
