#pragma once

#include <span>
#include <vector>
#include <cstring>
#include <concepts>
#include <algorithm>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

/**
 * @brief Run-Length Encoding for integral types
 * 
 * Compresses sequences of repeated values into (start_position, value) pairs.
 * Supports random access by binary searching run starts.
 * 
 * Format: [num_runs (8 bytes), 
 *          size_of_run_starts_in_bytes (8 bytes), 
 *          size_of_run_values_in_bytes (8 bytes),
 *          run_starts (num_runs * sizeof(size_t)),
 *          run_values (num_runs * sizeof(T))]
 * 
 * @tparam T The integral type to encode
 */
template<typename T>
    requires core::IntegralType<T>
class RunLengthEncoder : public Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
        if (data.empty()) {
            return createEmptyEncoding();
        }
        
        // Identify runs
        std::vector<size_t> runStarts;
        std::vector<T> runValues;
        
        runStarts.push_back(0);
        runValues.push_back(data[0]);
        
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] != data[i - 1]) {
                runStarts.push_back(i);
                runValues.push_back(data[i]);
            }
        }
        
        const size_t numRuns = runStarts.size();
        const size_t runStartsSize = numRuns * sizeof(size_t);
        const size_t runValuesSize = numRuns * sizeof(T);
        const size_t headerSize = 3 * sizeof(size_t);
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
        
        // Write run starts
        std::memcpy(writePtr, runStarts.data(), runStartsSize);
        writePtr += runStartsSize;
        
        // Write run values
        std::memcpy(writePtr, runValues.data(), runValuesSize);
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = data.size();
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = data.size() * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["num_runs"] = std::to_string(numRuns);
        result.metadata().customMetadata["compression_ratio"] = 
            std::to_string(static_cast<double>(totalSize) / (data.size() * sizeof(T)));
        
        return result;
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < 3 * sizeof(size_t)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t numRuns, runStartsSize, runValuesSize;
        std::memcpy(&numRuns, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&runStartsSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&runValuesSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        if (numRuns == 0) {
            return {};
        }
        
        // Read run starts
        std::vector<size_t> runStarts(numRuns);
        std::memcpy(runStarts.data(), readPtr, runStartsSize);
        readPtr += runStartsSize;
        
        // Read run values
        std::vector<T> runValues(numRuns);
        std::memcpy(runValues.data(), readPtr, runValuesSize);
        
        // Reconstruct original data
        // The last run extends to the end, which we get from metadata
        size_t totalElements = encoded.metadata().elementCount;
        std::vector<T> result;
        result.reserve(totalElements);
        
        for (size_t runIdx = 0; runIdx < numRuns; ++runIdx) {
            size_t runStart = runStarts[runIdx];
            size_t runEnd = (runIdx + 1 < numRuns) ? runStarts[runIdx + 1] : totalElements;
            T value = runValues[runIdx];
            
            for (size_t i = runStart; i < runEnd; ++i) {
                result.push_back(value);
            }
        }
        
        return result;
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.size() < 3 * sizeof(size_t)) {
            return std::nullopt;
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t numRuns, runStartsSize, runValuesSize;
        std::memcpy(&numRuns, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&runStartsSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&runValuesSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        if (numRuns == 0 || index >= encoded.metadata().elementCount) {
            return std::nullopt;
        }
        
        // Read run starts
        std::vector<size_t> runStarts(numRuns);
        std::memcpy(runStarts.data(), readPtr, runStartsSize);
        readPtr += runStartsSize;
        
        // Binary search to find which run contains this index
        size_t runIdx = std::upper_bound(runStarts.begin(), runStarts.end(), index) - runStarts.begin() - 1;
        
        // Read the value for this run
        T value;
        std::memcpy(&value, readPtr + (runIdx * sizeof(T)), sizeof(T));
        
        return value;
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.size() < 3 * sizeof(size_t)) {
            return {};
        }
        
        size_t totalElements = encoded.metadata().elementCount;
        if (start >= totalElements) {
            return {};
        }
        
        end = std::min(end, totalElements);
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        size_t numRuns, runStartsSize, runValuesSize;
        std::memcpy(&numRuns, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&runStartsSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        std::memcpy(&runValuesSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);
        
        if (numRuns == 0) {
            return {};
        }
        
        // Read run starts and values
        std::vector<size_t> runStarts(numRuns);
        std::memcpy(runStarts.data(), readPtr, runStartsSize);
        readPtr += runStartsSize;
        
        std::vector<T> runValues(numRuns);
        std::memcpy(runValues.data(), readPtr, runValuesSize);
        
        // Find first run that overlaps with [start, end)
        size_t firstRun = std::upper_bound(runStarts.begin(), runStarts.end(), start) - runStarts.begin() - 1;
        
        std::vector<T> result;
        result.reserve(end - start);
        
        for (size_t runIdx = firstRun; runIdx < numRuns; ++runIdx) {
            size_t runStart = runStarts[runIdx];
            size_t runEnd = (runIdx + 1 < numRuns) ? runStarts[runIdx + 1] : totalElements;
            
            if (runStart >= end) {
                break;
            }
            
            size_t effectiveStart = std::max(runStart, start);
            size_t effectiveEnd = std::min(runEnd, end);
            
            for (size_t i = effectiveStart; i < effectiveEnd; ++i) {
                result.push_back(runValues[runIdx]);
            }
        }
        
        return result;
    }
    
    EncodingType encodingType() const override {
        return EncodingType::RunLengthEncoding;
    }
    
    std::string name() const override {
        return "RunLength";
    }
    
    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::RandomAccess)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::RunLengthBased
            | EncodingProperty::VariableSize
            | EncodingProperty::StreamingFriendly
            | EncodingProperty::LowMemoryOverhead
            | EncodingProperty::Composable;
    }
    
    size_t estimateEncodedSize(size_t elementCount) const override {
        // Worst case: no compression (every element is a new run)
        return 3 * sizeof(size_t) + elementCount * (sizeof(size_t) + sizeof(T));
    }
    
private:
    EncodedData createEmptyEncoding() {
        EncodedData result;
        result.data().resize(3 * sizeof(size_t));
        
        size_t zero = 0;
        std::memcpy(result.data().data(), &zero, sizeof(size_t));
        std::memcpy(result.data().data() + sizeof(size_t), &zero, sizeof(size_t));
        std::memcpy(result.data().data() + 2 * sizeof(size_t), &zero, sizeof(size_t));
        
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = 3 * sizeof(size_t);
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
};

} // namespace encodings::encoders
