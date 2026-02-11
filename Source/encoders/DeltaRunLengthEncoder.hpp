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
 * @brief Delta + Run-Length encoding for integral types
 * 
 * First computes deltas between consecutive elements, then applies run-length
 * encoding to the deltas. Highly effective for sequences with repetitive
 * changes (e.g., constant increments, plateaus).
 * 
 * Format: [num_runs (8 bytes),
 *          size_of_run_starts_in_bytes (8 bytes),
 *          size_of_run_values_in_bytes (8 bytes),
 *          start_value (sizeof(T) bytes),
 *          run_starts (num_runs * sizeof(size_t) bytes),
 *          run_values (num_runs * sizeof(T) bytes)]
 * 
 * @tparam T The integral type to encode
 */
template<typename T>
    requires IntegralType<T>
class DeltaRunLengthEncoder : public Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
        if (data.empty()) {
            return createEmptyEncoding();
        }
        
        if (data.size() == 1) {
            return createSingleElementEncoding(data[0]);
        }
        
        // Step 1: Calculate deltas
        std::vector<T> deltas;
        deltas.reserve(data.size() - 1);
        
        T startValue = data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            deltas.push_back(data[i] - data[i - 1]);
        }
        
        // Step 2: Run-length encode the deltas
        std::vector<size_t> runStarts;
        std::vector<T> runValues;
        
        if (!deltas.empty()) {
            runStarts.push_back(0);
            runValues.push_back(deltas[0]);
            
            for (size_t i = 1; i < deltas.size(); ++i) {
                if (deltas[i] != deltas[i - 1]) {
                    runStarts.push_back(i);
                    runValues.push_back(deltas[i]);
                }
            }
        }
        
        const size_t numRuns = runStarts.size();
        const size_t runStartsSize = numRuns * sizeof(size_t);
        const size_t runValuesSize = numRuns * sizeof(T);
        const size_t headerSize = 3 * sizeof(size_t) + sizeof(T);
        const size_t totalSize = headerSize + runStartsSize + runValuesSize;
        
        EncodedData result;
        result.data().resize(totalSize);
        
        uint8_t* writePtr = result.data().data();
        
        // Write header
        std::memcpy(writePtr, &numRuns, sizeof(size_t));
        writePtr += sizeof(size_t);
        
        std::memcpy(writePtr, &runStartsSize, sizeof(size_t));
        writePtr += sizeof(size_t);
        
        std::memcpy(writePtr, &runValuesSize, sizeof(size_t));
        writePtr += sizeof(size_t);
        
        std::memcpy(writePtr, &startValue, sizeof(T));
        writePtr += sizeof(T);
        
        // Write run starts
        if (numRuns > 0) {
            std::memcpy(writePtr, runStarts.data(), runStartsSize);
            writePtr += runStartsSize;
            
            // Write run values
            std::memcpy(writePtr, runValues.data(), runValuesSize);
        }
        
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
        result.metadata().customMetadata["num_runs"] = std::to_string(numRuns);
        result.metadata().customMetadata["compression_ratio"] = std::to_string(
            deltas.empty() ? 0.0 : static_cast<double>(numRuns) / deltas.size()
        );
        
        return result;
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < 3 * sizeof(size_t) + sizeof(T)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t numRuns;
        std::memcpy(&numRuns, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        size_t runStartsSize;
        std::memcpy(&runStartsSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        size_t runValuesSize;
        std::memcpy(&runValuesSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        T startValue;
        std::memcpy(&startValue, readPtr, sizeof(T));
        readPtr += sizeof(T);
        
        // Read run starts
        std::vector<size_t> runStarts(numRuns);
        if (numRuns > 0) {
            std::memcpy(runStarts.data(), readPtr, runStartsSize);
            readPtr += runStartsSize;
        }
        
        // Read run values
        std::vector<T> runValues(numRuns);
        if (numRuns > 0) {
            std::memcpy(runValues.data(), readPtr, runValuesSize);
        }
        
        // Reconstruct deltas from runs
        std::vector<T> deltas;
        if (numRuns > 0) {
            for (size_t i = 0; i < numRuns; ++i) {
                size_t runStart = runStarts[i];
                size_t runEnd = (i + 1 < numRuns) ? runStarts[i + 1] : 
                                (encoded.metadata().elementCount - 1);
                
                for (size_t j = runStart; j < runEnd; ++j) {
                    deltas.push_back(runValues[i]);
                }
            }
        }
        
        // Reconstruct original values from deltas
        std::vector<T> result;
        result.reserve(deltas.size() + 1);
        result.push_back(startValue);
        
        T currentValue = startValue;
        for (const auto& delta : deltas) {
            currentValue += delta;
            result.push_back(currentValue);
        }
        
        return result;
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.size() < 3 * sizeof(size_t) + sizeof(T)) {
            return std::nullopt;
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t numRuns;
        std::memcpy(&numRuns, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        size_t runStartsSize;
        std::memcpy(&runStartsSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        size_t runValuesSize;
        std::memcpy(&runValuesSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        T startValue;
        std::memcpy(&startValue, readPtr, sizeof(T));
        readPtr += sizeof(T);
        
        const size_t totalElements = encoded.metadata().elementCount;
        
        if (index >= totalElements) {
            return std::nullopt;
        }
        
        if (index == 0) {
            return startValue;
        }
        
        // Read run starts and values
        const size_t* runStartsPtr = reinterpret_cast<const size_t*>(readPtr);
        const T* runValuesPtr = reinterpret_cast<const T*>(readPtr + runStartsSize);
        
        // Accumulate deltas from start to index
        T currentValue = startValue;
        size_t currentRun = 0;
        
        for (size_t i = 0; i < index; ++i) {
            // Find which run contains delta at position i
            while (currentRun + 1 < numRuns && i >= runStartsPtr[currentRun + 1]) {
                currentRun++;
            }
            
            currentValue += runValuesPtr[currentRun];
        }
        
        return currentValue;
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.size() < 3 * sizeof(size_t) + sizeof(T)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t numRuns;
        std::memcpy(&numRuns, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        size_t runStartsSize;
        std::memcpy(&runStartsSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        size_t runValuesSize;
        std::memcpy(&runValuesSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        T startValue;
        std::memcpy(&startValue, readPtr, sizeof(T));
        readPtr += sizeof(T);
        
        const size_t totalElements = encoded.metadata().elementCount;
        
        if (start >= totalElements) {
            return {};
        }
        
        end = std::min(end, totalElements);
        
        // Read run starts and values
        const size_t* runStartsPtr = reinterpret_cast<const size_t*>(readPtr);
        const T* runValuesPtr = reinterpret_cast<const T*>(readPtr + runStartsSize);
        
        // Accumulate deltas up to start position
        T currentValue = startValue;
        size_t currentRun = 0;
        
        for (size_t i = 0; i < start; ++i) {
            // Find which run contains delta at position i
            while (currentRun + 1 < numRuns && i >= runStartsPtr[currentRun + 1]) {
                currentRun++;
            }
            
            currentValue += runValuesPtr[currentRun];
        }
        
        // Build result range
        std::vector<T> result;
        result.reserve(end - start);
        
        if (start == 0) {
            result.push_back(startValue);
        } else {
            result.push_back(currentValue);
        }
        
        for (size_t i = start + 1; i < end; ++i) {
            // Find which run contains delta at position i-1
            while (currentRun + 1 < numRuns && (i - 1) >= runStartsPtr[currentRun + 1]) {
                currentRun++;
            }
            
            currentValue += runValuesPtr[currentRun];
            result.push_back(currentValue);
        }
        
        return result;
    }
    
    EncodingType encodingType() const override {
        return EncodingType::DeltaEncoding;
    }
    
    std::string name() const override {
        return "DeltaRLE";
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
            return 3 * sizeof(size_t) + sizeof(T);
        }
        // Worst case: every delta is different (no compression)
        size_t numDeltas = elementCount - 1;
        size_t worstCaseRuns = numDeltas;
        return 3 * sizeof(size_t) + sizeof(T) + 
               worstCaseRuns * sizeof(size_t) + 
               worstCaseRuns * sizeof(T);
    }
    
private:
    EncodedData createEmptyEncoding() {
        EncodedData result;
        const size_t headerSize = 3 * sizeof(size_t) + sizeof(T);
        result.data().resize(headerSize);
        
        uint8_t* writePtr = result.data().data();
        size_t zero = 0;
        
        std::memcpy(writePtr, &zero, sizeof(size_t)); // numRuns = 0
        writePtr += sizeof(size_t);
        std::memcpy(writePtr, &zero, sizeof(size_t)); // runStartsSize = 0
        writePtr += sizeof(size_t);
        std::memcpy(writePtr, &zero, sizeof(size_t)); // runValuesSize = 0
        writePtr += sizeof(size_t);
        
        T zeroValue = 0;
        std::memcpy(writePtr, &zeroValue, sizeof(T)); // startValue = 0
        
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = headerSize;
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
    
    EncodedData createSingleElementEncoding(T value) {
        EncodedData result;
        const size_t headerSize = 3 * sizeof(size_t) + sizeof(T);
        result.data().resize(headerSize);
        
        uint8_t* writePtr = result.data().data();
        size_t zero = 0;
        
        std::memcpy(writePtr, &zero, sizeof(size_t)); // numRuns = 0
        writePtr += sizeof(size_t);
        std::memcpy(writePtr, &zero, sizeof(size_t)); // runStartsSize = 0
        writePtr += sizeof(size_t);
        std::memcpy(writePtr, &zero, sizeof(size_t)); // runValuesSize = 0
        writePtr += sizeof(size_t);
        
        std::memcpy(writePtr, &value, sizeof(T)); // startValue
        
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 1;
        result.metadata().compressedSize = headerSize;
        result.metadata().uncompressedSize = sizeof(T);
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
};

} // namespace encodings::encoders
