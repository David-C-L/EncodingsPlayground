#pragma once

#include <span>
#include <vector>
#include <cstring>
#include <concepts>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

    using core::DataType;
    using core::PrimitiveType;
    using core::typeToDataType;
    using core::IntegralType;
/**
 * @brief Delta encoding for integral types
 * 
 * Stores the first value, then differences between consecutive elements.
 * Highly effective for monotonic or slowly-varying sequences.
 * Supports random access by accumulating deltas from the start.
 * 
 * Format: [size_of_deltas_in_bytes (8 bytes),
 *          start_value (sizeof(T) bytes),
 *          deltas ((n-1) * sizeof(T) bytes)]
 * 
 * @tparam T The integral type to encode
 */
template<typename T>
    requires IntegralType<T>
class DeltaEncoder : public Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
        if (data.empty()) {
            return createEmptyEncoding();
        }
        
        if (data.size() == 1) {
            return createSingleElementEncoding(data[0]);
        }
        
        // Calculate deltas
        std::vector<T> deltas;
        deltas.reserve(data.size() - 1);
        
        T startValue = data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            deltas.push_back(data[i] - data[i - 1]);
        }
        
        const size_t deltasSize = deltas.size() * sizeof(T);
        const size_t headerSize = sizeof(size_t) + sizeof(T);
        const size_t totalSize = headerSize + deltasSize;
        
        EncodedData result;
        result.data().resize(totalSize);
        
        uint8_t* writePtr = result.data().data();
        
        // Write header
        std::memcpy(writePtr, &deltasSize, sizeof(size_t));
        writePtr += sizeof(size_t);
        
        std::memcpy(writePtr, &startValue, sizeof(T));
        writePtr += sizeof(T);
        
        // Write deltas
        std::memcpy(writePtr, deltas.data(), deltasSize);
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = data.size();
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = data.size() * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        
        // Calculate statistics for metadata
        T minDelta = deltas.empty() ? 0 : deltas[0];
        T maxDelta = deltas.empty() ? 0 : deltas[0];
        for (const auto& delta : deltas) {
            minDelta = std::min(minDelta, delta);
            maxDelta = std::max(maxDelta, delta);
        }
        
        result.metadata().customMetadata["min_delta"] = std::to_string(minDelta);
        result.metadata().customMetadata["max_delta"] = std::to_string(maxDelta);
        result.metadata().customMetadata["num_deltas"] = std::to_string(deltas.size());
        
        return result;
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < sizeof(size_t) + sizeof(T)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t deltasSize;
        std::memcpy(&deltasSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        T startValue;
        std::memcpy(&startValue, readPtr, sizeof(T));
        readPtr += sizeof(T);
        
        const size_t numDeltas = deltasSize / sizeof(T);
        const size_t totalElements = numDeltas + 1;
        
        // Read deltas
        std::vector<T> deltas(numDeltas);
        if (numDeltas > 0) {
            std::memcpy(deltas.data(), readPtr, deltasSize);
        }
        
        // Reconstruct original values
        std::vector<T> result;
        result.reserve(totalElements);
        result.push_back(startValue);
        
        T currentValue = startValue;
        for (const auto& delta : deltas) {
            currentValue += delta;
            result.push_back(currentValue);
        }
        
        return result;
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.size() < sizeof(size_t) + sizeof(T)) {
            return std::nullopt;
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t deltasSize;
        std::memcpy(&deltasSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        T startValue;
        std::memcpy(&startValue, readPtr, sizeof(T));
        readPtr += sizeof(T);
        
        const size_t numDeltas = deltasSize / sizeof(T);
        const size_t totalElements = numDeltas + 1;
        
        if (index >= totalElements) {
            return std::nullopt;
        }
        
        if (index == 0) {
            return startValue;
        }
        
        // Need to accumulate deltas from start to index
        T currentValue = startValue;
        const T* deltasPtr = reinterpret_cast<const T*>(readPtr);
        
        for (size_t i = 0; i < index; ++i) {
            currentValue += deltasPtr[i];
        }
        
        return currentValue;
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.size() < sizeof(size_t) + sizeof(T)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t deltasSize;
        std::memcpy(&deltasSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        T startValue;
        std::memcpy(&startValue, readPtr, sizeof(T));
        readPtr += sizeof(T);
        
        const size_t numDeltas = deltasSize / sizeof(T);
        const size_t totalElements = numDeltas + 1;
        
        if (start >= totalElements) {
            return {};
        }
        
        end = std::min(end, totalElements);
        
        // Accumulate deltas up to start position
        T currentValue = startValue;
        const T* deltasPtr = reinterpret_cast<const T*>(readPtr);
        
        for (size_t i = 0; i < start; ++i) {
            currentValue += deltasPtr[i];
        }
        
        // Build result range
        std::vector<T> result;
        result.reserve(end - start);
        
        if (start == 0) {
            result.push_back(startValue);
            start = 1;
        }
        
        for (size_t i = start; i < end; ++i) {
            if (i > 0) {
                currentValue += deltasPtr[i - 1];
            }
            result.push_back(currentValue);
        }
        
        return result;
    }
    
    EncodingType encodingType() const override {
        return EncodingType::DeltaEncoding;
    }
    
    std::string name() const override {
        return "Delta";
    }
    
    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::RandomAccess)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::DeltaBased
            | EncodingProperty::FixedSize
            | EncodingProperty::StreamingFriendly
            | EncodingProperty::OptimizedForSorted
            | EncodingProperty::LowMemoryOverhead
            | EncodingProperty::Composable;
    }
    
    size_t estimateEncodedSize(size_t elementCount) const override {
        if (elementCount == 0) {
            return sizeof(size_t);
        }
        return sizeof(size_t) + sizeof(T) + (elementCount - 1) * sizeof(T);
    }
    
private:
    EncodedData createEmptyEncoding() {
        EncodedData result;
        result.data().resize(sizeof(size_t));
        
        size_t zero = 0;
        std::memcpy(result.data().data(), &zero, sizeof(size_t));
        
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = sizeof(size_t);
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
    
    EncodedData createSingleElementEncoding(T value) {
        EncodedData result;
        const size_t totalSize = sizeof(size_t) + sizeof(T);
        result.data().resize(totalSize);
        
        size_t deltasSize = 0;
        std::memcpy(result.data().data(), &deltasSize, sizeof(size_t));
        std::memcpy(result.data().data() + sizeof(size_t), &value, sizeof(T));
        
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 1;
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = sizeof(T);
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
};

} // namespace encodings::encoders
