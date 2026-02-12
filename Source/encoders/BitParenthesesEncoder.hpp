#pragma once

#include <span>
#include <vector>
#include <cstring>
#include <concepts>
#include <cstdint>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

/**
 * @brief Bit-Parentheses Encoding for integral types (sizes/counts)
 * 
 * Encodes a sequence of non-negative integers as balanced parentheses (bit sequence).
 * Each value n is encoded as n ones followed by a zero.
 * The bit sequence is RLE compressed for efficiency.
 * 
 * Example: [3, 5, 4] -> bits: [1,1,1,0, 1,1,1,1,1,0, 1,1,1,1,0] 
 *                     -> RLE: [(0,1,3), (3,0,1), (4,1,5), (9,0,1), (10,1,4), (14,0,1)]
 *                     -> Stored as run lengths and values
 * 
 * This is efficient for:
 * - Small positive integers (map sizes, array lengths)
 * - Large values (long runs of 1s compress well with RLE)
 * - Any pattern with repetition in the bit sequence
 * 
 * Format:
 * [num_elements (4 bytes),
 *  total_bits (4 bytes),
 *  num_runs (4 bytes),
 *  run_lengths (num_runs * 4 bytes),
 *  run_values_packed ((num_runs + 7) / 8 bytes),
 *  element_boundaries (num_elements * 4 bytes)]
 * 
 * @tparam T The integral type to encode (must be non-negative or will be treated as unsigned)
 */
template<typename T>
    requires core::IntegralType<T>
class BitParenthesesEncoder : public Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
        if (data.empty()) {
            return createEmptyEncoding();
        }

        // Step 1 & 2 combined: Directly build RLE runs without intermediate bit vector
        // Each element contributes: count ones + 1 zero
        // Optimization: We know the pattern is always [1...1,0] so we can construct runs directly
        std::vector<uint32_t> runLengths;
        std::vector<bool> runValues;
        std::vector<uint32_t> elementBoundaries;
        elementBoundaries.reserve(data.size());
        
        uint32_t totalBits = 0;
        
        for (const auto& value : data) {
            uint32_t count = static_cast<uint32_t>(value);
            
            // Add run of 'count' ones
            if (count > 0) {
                // Merge with previous run if it's also ones
                if (!runValues.empty() && runValues.back()) {
                    runLengths.back() += count;
                } else {
                    runLengths.push_back(count);
                    runValues.push_back(true);
                }
            }
            
            // Add run of 1 zero (separator)
            // Merge with previous run if it's also zeros
            if (!runValues.empty() && !runValues.back()) {
                runLengths.back() += 1;
            } else {
                runLengths.push_back(1);
                runValues.push_back(false);
            }
            
            totalBits += count + 1;
            elementBoundaries.push_back(totalBits);
        }
        
        // Step 3: Pack run values into bits
        const uint32_t numRuns = static_cast<uint32_t>(runLengths.size());
        const uint32_t runValueBytes = (numRuns + 7) / 8;
        std::vector<uint8_t> runValuesPacked(runValueBytes, 0);
        
        for (uint32_t i = 0; i < numRuns; ++i) {
            if (runValues[i]) {
                runValuesPacked[i / 8] |= (1 << (i % 8));
            }
        }
        
        // Step 4: Serialize
        const uint32_t numElements = static_cast<uint32_t>(data.size());
        const size_t headerSize = 3 * sizeof(uint32_t); // numElements, totalBits, numRuns
        const size_t runLengthsSize = numRuns * sizeof(uint32_t);
        const size_t boundariesSize = numElements * sizeof(uint32_t);
        const size_t totalSize = headerSize + runLengthsSize + runValueBytes + boundariesSize;
        
        EncodedData result;
        result.data().resize(totalSize);
        
        uint8_t* writePtr = result.data().data();
        
        // Write header
        std::memcpy(writePtr, &numElements, sizeof(uint32_t));
        writePtr += sizeof(uint32_t);
        std::memcpy(writePtr, &totalBits, sizeof(uint32_t));
        writePtr += sizeof(uint32_t);
        std::memcpy(writePtr, &numRuns, sizeof(uint32_t));
        writePtr += sizeof(uint32_t);
        
        // Write run lengths
        std::memcpy(writePtr, runLengths.data(), runLengthsSize);
        writePtr += runLengthsSize;
        
        // Write packed run values
        std::memcpy(writePtr, runValuesPacked.data(), runValueBytes);
        writePtr += runValueBytes;
        
        // Write boundaries
        std::memcpy(writePtr, elementBoundaries.data(), boundariesSize);
        
        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = numElements;
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = numElements * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
    
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < 3 * sizeof(uint32_t)) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        uint32_t numElements, totalBits, numRuns;
        std::memcpy(&numElements, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        std::memcpy(&totalBits, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        std::memcpy(&numRuns, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        
        if (numElements == 0) {
            return {};
        }
        
        // Read run lengths
        std::vector<uint32_t> runLengths(numRuns);
        std::memcpy(runLengths.data(), readPtr, numRuns * sizeof(uint32_t));
        readPtr += numRuns * sizeof(uint32_t);
        
        // Read packed run values
        const uint32_t runValueBytes = (numRuns + 7) / 8;
        const uint8_t* runValuesPacked = readPtr;
        readPtr += runValueBytes;
        
        // Read boundaries
        std::vector<uint32_t> elementBoundaries(numElements);
        std::memcpy(elementBoundaries.data(), readPtr, numElements * sizeof(uint32_t));
        
        // OPTIMIZATION: Decode sequentially with stateful iteration
        std::vector<T> result;
        result.reserve(numElements);
        
        uint32_t prevBoundary = 0;
        uint32_t runIdx = 0;
        uint32_t currentRunPos = 0;  // Current position in the bit stream
        
        for (uint32_t i = 0; i < numElements; ++i) {
            uint32_t boundary = elementBoundaries[i];
            
            // Count ones between prevBoundary and boundary-1
            uint32_t onesCount = countOnesInRLERangeStateful(
                runLengths, runValuesPacked, 
                prevBoundary, boundary - 1,
                runIdx, currentRunPos);
            
            result.push_back(static_cast<T>(onesCount));
            prevBoundary = boundary;
        }
        
        return result;
    }
    
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.size() < 3 * sizeof(uint32_t)) {
            return std::nullopt;
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        uint32_t numElements, totalBits, numRuns;
        std::memcpy(&numElements, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        std::memcpy(&totalBits, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        std::memcpy(&numRuns, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        
        if (index >= numElements) {
            return std::nullopt;
        }
        
        // OPTIMIZATION: Only read the 2 boundaries we need, not all of them
        const size_t boundariesOffset = 3 * sizeof(uint32_t) + 
                                        numRuns * sizeof(uint32_t) + 
                                        ((numRuns + 7) / 8);
        
        uint32_t prevBoundary = 0;
        if (index > 0) {
            std::memcpy(&prevBoundary, 
                       encoded.data().data() + boundariesOffset + (index - 1) * sizeof(uint32_t),
                       sizeof(uint32_t));
        }
        
        uint32_t boundary;
        std::memcpy(&boundary,
                   encoded.data().data() + boundariesOffset + index * sizeof(uint32_t),
                   sizeof(uint32_t));
        
        // Still need to read run data for counting
        std::vector<uint32_t> runLengths(numRuns);
        std::memcpy(runLengths.data(), readPtr, numRuns * sizeof(uint32_t));
        readPtr += numRuns * sizeof(uint32_t);
        
        const uint8_t* runValuesPacked = readPtr;
        
        uint32_t onesCount = countOnesInRLERange(runLengths, runValuesPacked, prevBoundary, boundary - 1);
        
        return static_cast<T>(onesCount);
    }
    
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.size() < 3 * sizeof(uint32_t) || start >= end) {
            return {};
        }
        
        const uint8_t* readPtr = encoded.data().data();
        
        // Read header
        uint32_t numElements, totalBits, numRuns;
        std::memcpy(&numElements, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        std::memcpy(&totalBits, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        std::memcpy(&numRuns, readPtr, sizeof(uint32_t));
        readPtr += sizeof(uint32_t);
        
        if (start >= numElements) {
            return {};
        }
        
        end = std::min(end, static_cast<size_t>(numElements));
        
        // Read run lengths
        std::vector<uint32_t> runLengths(numRuns);
        std::memcpy(runLengths.data(), readPtr, numRuns * sizeof(uint32_t));
        readPtr += numRuns * sizeof(uint32_t);
        
        // Read packed run values
        const uint32_t runValueBytes = (numRuns + 7) / 8;
        const uint8_t* runValuesPacked = readPtr;
        readPtr += runValueBytes;
        
        // OPTIMIZATION: Only read the boundaries we need
        const size_t boundariesOffset = 3 * sizeof(uint32_t) + 
                                        numRuns * sizeof(uint32_t) + 
                                        runValueBytes;
        const size_t boundariesNeeded = end - start + (start > 0 ? 1 : 0);
        const size_t boundariesStartIdx = (start > 0) ? start - 1 : 0;
        
        std::vector<uint32_t> elementBoundaries(boundariesNeeded);
        std::memcpy(elementBoundaries.data(),
                   encoded.data().data() + boundariesOffset + boundariesStartIdx * sizeof(uint32_t),
                   boundariesNeeded * sizeof(uint32_t));
        
        // Decode range with stateful iteration
        std::vector<T> result;
        result.reserve(end - start);
        
        uint32_t prevBoundary = (start == 0) ? 0 : elementBoundaries[0];
        uint32_t runIdx = 0;
        uint32_t currentRunPos = 0;
        
        // If we're not starting from 0, we need to find the correct run position
        if (start > 0 && prevBoundary > 0) {
            // Fast-forward to the correct run
            while (runIdx < numRuns && currentRunPos < prevBoundary) {
                currentRunPos += runLengths[runIdx];
                runIdx++;
            }
            // Back up one run since we overshot
            if (runIdx > 0) {
                runIdx--;
                currentRunPos -= runLengths[runIdx];
            }
        }
        
        for (size_t i = start; i < end; ++i) {
            size_t boundaryIdx = (start == 0) ? i : i - start + 1;
            uint32_t boundary = elementBoundaries[boundaryIdx];
            
            uint32_t onesCount = countOnesInRLERangeStateful(
                runLengths, runValuesPacked,
                prevBoundary, boundary - 1,
                runIdx, currentRunPos);
            
            result.push_back(static_cast<T>(onesCount));
            prevBoundary = boundary;
        }
        
        return result;
    }
    
    std::string name() const override {
        return "RLEBitParentheses";
    }
    
    EncodingType encodingType() const override {
        return EncodingType::BitPacking;
    }
    
    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::Lossless)
            | EncodingProperty::RandomAccess
            | EncodingProperty::VariableSize
            | EncodingProperty::BitPackingBased
            | EncodingProperty::Composable;
    }
    
    size_t estimateEncodedSize(size_t elementCount) const override {
        // Conservative estimate assuming average value of 10
        size_t avgBitsPerElement = 11; // 10 ones + 1 zero
        size_t totalBits = elementCount * avgBitsPerElement;
        size_t bitVectorBytes = (totalBits + 7) / 8;
        size_t boundariesBytes = elementCount * sizeof(uint32_t);
        return 2 * sizeof(uint32_t) + bitVectorBytes + boundariesBytes;
    }
    
private:
    /**
     * @brief Get a run value at the given index from packed bits
     */
    bool getRunValue(const uint8_t* runValuesPacked, uint32_t runIdx) const {
        return (runValuesPacked[runIdx / 8] & (1 << (runIdx % 8))) != 0;
    }
    
    /**
     * @brief Count ones with stateful iteration (reuses run position between calls)
     * @param runIdx In/out parameter - current run index (updated for next call)
     * @param currentRunPos In/out parameter - current position in bit stream (updated for next call)
     */
    uint32_t countOnesInRLERangeStateful(
        const std::vector<uint32_t>& runLengths,
        const uint8_t* runValuesPacked,
        uint32_t startPos,
        uint32_t endPos,
        uint32_t& runIdx,
        uint32_t& currentRunPos) const {
        
        if (startPos > endPos || runLengths.empty()) {
            return 0;
        }
        
        uint32_t onesCount = 0;
        
        // Continue from where we left off
        while (runIdx < runLengths.size()) {
            uint32_t runLength = runLengths[runIdx];
            uint32_t runEnd = currentRunPos + runLength - 1;
            
            // Check if this run overlaps with [startPos, endPos]
            if (runEnd >= startPos && currentRunPos <= endPos) {
                // Calculate overlap
                uint32_t overlapStart = std::max(currentRunPos, startPos);
                uint32_t overlapEnd = std::min(runEnd, endPos);
                uint32_t overlapLength = overlapEnd - overlapStart + 1;
                
                // If this is a run of 1s, add to count
                if (getRunValue(runValuesPacked, runIdx)) {
                    onesCount += overlapLength;
                }
            }
            
            currentRunPos += runLength;
            runIdx++;
            
            // Early exit if we've passed the range
            if (currentRunPos > endPos) {
                break;
            }
        }
        
        return onesCount;
    }
    
    /**
     * @brief Count the number of ones in the RLE bit range [startPos, endPos] inclusive
     * (Stateless version for random access)
     */
    uint32_t countOnesInRLERange(
        const std::vector<uint32_t>& runLengths,
        const uint8_t* runValuesPacked,
        uint32_t startPos,
        uint32_t endPos) const {
        
        uint32_t runIdx = 0;
        uint32_t currentRunPos = 0;
        return countOnesInRLERangeStateful(runLengths, runValuesPacked, startPos, endPos, runIdx, currentRunPos);
    }
    
    EncodedData createEmptyEncoding() const {
        EncodedData result;
        result.data().resize(3 * sizeof(uint32_t));
        
        uint32_t zero = 0;
        std::memcpy(result.data().data(), &zero, sizeof(uint32_t));
        std::memcpy(result.data().data() + sizeof(uint32_t), &zero, sizeof(uint32_t));
        std::memcpy(result.data().data() + 2 * sizeof(uint32_t), &zero, sizeof(uint32_t));
        
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = 3 * sizeof(uint32_t);
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;
        
        return result;
    }
};

} // namespace encodings::encoders
