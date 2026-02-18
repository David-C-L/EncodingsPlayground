#pragma once

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"

namespace encodings::core {
    constexpr uint32_t log2_exact(size_t val)
    {
        assert(val != 0 && (val & (val - 1)) == 0);
        uint32_t result = 0;
        while (val >>= 1)
            ++result;
        return result;
    }

    /**
     * @brief Fast bitset with optional value grouping and compression support
     * 
     * Primary use cases:
     * - Null/existence checks for columnar data
     * - Bloom filters and range queries
     * - Presence tracking in sparse map encodings
     * 
     * Features:
     * - Powers-of-2 value grouping (default: 1 value per bit)
     * - Compression via pluggable codecs
     * - Optimized for random access with mixed sparsity
     * 
     * Future optimizations:
     * - Specialized implementations for mostly-empty bitsets (e.g., sparse bitmap indexes)
     * - Specialized implementations for mostly-full bitsets (e.g., inverted representation)
     * - Run-length encoding for sequential patterns
     * 
     * TODO: Consider storing/serializing codec with bitset for self-describing format
     */
    struct FastBitset
    {
      std::vector<uint64_t> bits_;
      size_t valuesPerBit_;
      size_t numBits_;
      uint32_t shift_; // log2(valuesPerBit_)
      
      // Transient codec for compression (not serialized)
      // TODO: Consider making this persistent for self-describing bitsets
      std::shared_ptr<encodings::Codec<uint64_t>> codec_;

      /**
       * @brief Construct a bitset with optional value grouping and compression
       * 
       * @param numValues Total number of values to track
       * @param valuesPerBit Number of values mapped to each bit (default: 1, must be power of 2)
       * @param codec Compression codec for encode/decode (default: RawEncoder)
       */
      explicit FastBitset(
          size_t numValues, 
          size_t valuesPerBit = 1,
          std::shared_ptr<encodings::Codec<uint64_t>> codec = nullptr)
          : valuesPerBit_(valuesPerBit),
            shift_(log2_exact(valuesPerBit)),
            codec_(codec)
      {

        if ((valuesPerBit & (valuesPerBit - 1)) != 0)
        {
          throw std::invalid_argument("valuesPerBit in FastBitset must be a power of two");
        }

        numBits_ = (numValues + valuesPerBit - 1) >> shift_;
        bits_.assign((numBits_ + 63) / 64, 0);
        
        // Default to RawEncoder if no codec provided
        if (!codec_) {
          codec_ = createDefaultCodec();
        }
      }

      inline __attribute__((always_inline)) size_t valueToBitIndex(size_t value) const
      {
        return value >> shift_;
      }

      inline __attribute__((always_inline)) bool test(size_t value) const
      {
        size_t bit = valueToBitIndex(value);
        return bits_[bit >> 6] & (1ULL << (bit & 63));
      }

      inline __attribute__((always_inline)) void set(size_t value)
      {
        size_t bit = valueToBitIndex(value);
        bits_[bit >> 6] |= (1ULL << (bit & 63));
      }

      /**
       * @brief Atomically set a bit and return its previous value
       * 
       * @param value The value index to set
       * @return true if bit was already set, false if it was clear
       */
      inline __attribute__((always_inline)) bool test_and_set(size_t value)
      {
        size_t bit = valueToBitIndex(value);
        size_t wordIdx = bit >> 6;
        uint64_t mask = UINT64_C(1) << (bit & 63);

        uint64_t oldWord = bits_[wordIdx];
        bits_[wordIdx] = oldWord | mask;
        return (oldWord & mask) != 0; // Return true if bit was already set
      }

      inline __attribute__((always_inline)) void reset(size_t value)
      {
        size_t bit = valueToBitIndex(value);
        bits_[bit >> 6] &= ~(1ULL << (bit & 63));
      }

      inline __attribute__((always_inline)) void clear()
      {
        std::fill(bits_.begin(), bits_.end(), 0);
      }

      inline __attribute__((always_inline)) size_t sizeInBits() const
      {
        return numBits_;
      }

      inline __attribute__((always_inline)) size_t size() const
      {
        return numBits_ * valuesPerBit_;
      }

      inline __attribute__((always_inline)) void restructure(size_t numValues, size_t valuesPerBit)
      {
        if ((valuesPerBit & (valuesPerBit - 1)) != 0)
        {
          throw std::invalid_argument("valuesPerBit in FastBitset must be a power of two");
        }

        valuesPerBit_ = valuesPerBit;
        shift_ = log2_exact(valuesPerBit);
        numBits_ = (numValues + valuesPerBit - 1) >> shift_;
        bits_.assign((numBits_ + 63) / 64, 0);
      }

      // Set contiguous range [valueStart, valueStart + count)
      inline __attribute__((always_inline)) void setRange(size_t valueStart, size_t count)
      {
        if (count == 0)
          return;

        size_t firstBit = valueToBitIndex(valueStart);
        size_t lastBit = valueToBitIndex(valueStart + count - 1);
        size_t startWord = firstBit >> 6;
        size_t endWord = lastBit >> 6;

        if (startWord >= bits_.size() || endWord >= bits_.size())
        {
          throw std::runtime_error("Requested range is outside of supported ranges.");
        }

        if (startWord == endWord)
        {
          uint64_t mask = ((1ULL << (lastBit - firstBit + 1)) - 1) << (firstBit & 63);
          bits_[startWord] |= mask;
        }
        else
        {
          uint64_t firstMask = ~0ULL << (firstBit & 63);
          bits_[startWord++] |= firstMask;

          for (size_t w = startWord; w < endWord; ++w)
          {
            bits_[w] = ~0ULL;
          }

          uint64_t lastMask = (lastBit & 63) == 63 ? ~0ULL : ((1ULL << ((lastBit & 63) + 1)) - 1);
          bits_[endWord] |= lastMask;
        }
      }

      // Test contiguous range [valueStart, valueStart + count)
      inline __attribute__((always_inline)) bool testRange(size_t valueStart, size_t count) const
      {
        if (count == 0)
          return true;

        size_t firstBit = valueToBitIndex(valueStart);
        size_t lastBit = valueToBitIndex(valueStart + count - 1);
        size_t startWord = firstBit >> 6;
        size_t endWord = lastBit >> 6;

        if (startWord == endWord)
        {
          uint64_t mask = ((1ULL << (lastBit - firstBit + 1)) - 1) << (firstBit & 63);
          return (bits_[startWord] & mask) == mask;
        }
        else
        {
          // First partial word
          uint64_t firstMask = ~0ULL << (firstBit & 63);
          if ((bits_[startWord++] & firstMask) != firstMask)
            return false;

          // Middle whole words
          for (size_t w = startWord; w < endWord; w++)
          {
            if (bits_[w] != ~0ULL)
              return false;
          }

          // Last partial word
          uint64_t lastMask = ((lastBit & 63) == 63) ? ~0ULL : ((1ULL << ((lastBit & 63) + 1)) - 1);
          return (bits_[endWord] & lastMask) == lastMask;
        }
      }

      inline __attribute__((always_inline)) std::pair<size_t, size_t> getValueRange(size_t value, size_t n) const
      {
        if (n == 0)
          return {value, value}; // No values to cover

        size_t coveredStart = (value >> shift_) << shift_;
        size_t coveredEnd = (((value + n - 1) >> shift_) + 1) << shift_;
        return {coveredStart, coveredEnd};
      }

      inline __attribute__((always_inline)) std::pair<size_t, size_t> getValueRange(size_t value) const
      {
        size_t blockStart = (value >> shift_) << shift_;
        return {blockStart, blockStart + valuesPerBit_};
      }

      void printSetBits() const
      {
        for (size_t i = 0; i < numBits_; ++i)
        {
          size_t wordIndex = i >> 6;
          size_t bitOffset = i & 63;
          if (bits_[wordIndex] & (1ULL << bitOffset))
          {
            // Bit i is set
            // size_t startValue = i << shift_;
          }
        }
      }

      /**
       * @brief Count number of set bits using hardware POPCNT instruction
       * 
       * Performance: O(words) instead of O(bits) using __builtin_popcountll
       * 
       * @return Number of set bits (not values!)
       */
      inline __attribute__((always_inline)) size_t numSetBits() const
      {
        size_t count = 0;
        for (const uint64_t word : bits_)
        {
          count += __builtin_popcountll(word);
        }
        return count;
      }

      /**
       * @brief Count number of set values
       * 
       * Note: With valuesPerBit > 1, multiple values map to one bit,
       * so this returns setBits * valuesPerBit (conservative estimate)
       * 
       * @return Number of values that would test as "present"
       */
      inline __attribute__((always_inline)) size_t numSetValues() const
      {
        return numSetBits() * valuesPerBit_;
      }

      /**
       * @brief Encode the bitset using the configured codec
       * 
       * Format: [numValues: 8 bytes][valuesPerBit: 4 bytes][numBits: 8 bytes]
       *         [numWords: 8 bytes][compressed bits...]
       * 
       * Tradeoff: In-place decode (lower memory) vs new allocation (higher memory, faster)
       * Choice: New allocation for better performance, accepting higher transient memory usage
       * 
       * @return EncodedData containing metadata and compressed bits
       */
      encodings::EncodedData encode() const
      {
        using namespace encodings;
        
        // Encode the bit vector using the configured codec
        EncodedData compressedBits = codec_->encode(bits_);
        
        // Build header with metadata
        std::vector<uint8_t> result;
        size_t headerSize = 3 * sizeof(uint64_t) + sizeof(uint32_t);
        result.reserve(headerSize + compressedBits.size());
        
        // Write metadata
        uint64_t numValues = size();
        uint32_t valuesPerBitU32 = static_cast<uint32_t>(valuesPerBit_);
        uint64_t numBitsU64 = numBits_;
        uint64_t numWords = bits_.size();
        
        result.resize(headerSize);
        size_t offset = 0;
        
        std::memcpy(result.data() + offset, &numValues, sizeof(uint64_t));
        offset += sizeof(uint64_t);
        std::memcpy(result.data() + offset, &valuesPerBitU32, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(result.data() + offset, &numBitsU64, sizeof(uint64_t));
        offset += sizeof(uint64_t);
        std::memcpy(result.data() + offset, &numWords, sizeof(uint64_t));
        offset += sizeof(uint64_t);
        
        // Append compressed bits
        result.insert(result.end(), 
                     compressedBits.data().begin(), 
                     compressedBits.data().end());
        
        // Create EncodedData with metadata
        EncodingMetadata metadata;
        metadata.encodingName = "FastBitset_" + codec_->name();
        metadata.dataType = DataType::UInt64;
        metadata.elementCount = bits_.size();
        metadata.compressedSize = result.size();
        metadata.uncompressedSize = headerSize + bits_.size() * sizeof(uint64_t);
        metadata.supportsRandomAccess = false; // Must decompress entire bitset
        
        return EncodedData(std::move(result), std::move(metadata));
      }

      /**
       * @brief Decode a bitset from encoded data (static factory method)
       * 
       * Tradeoff analysis:
       * - Static factory: Clean, creates new object, clear ownership
       * - Instance method: Reuses memory, but mutates state (confusing semantics)
       * - Separate decode(): Requires codec parameter duplication
       * 
       * Choice: Static factory for simplicity and clear semantics
       * 
       * @param encoded The encoded bitset data
       * @param codec The codec to use for decompression (must match encoding codec)
       * @return New FastBitset instance
       */
      static FastBitset decode(
          const encodings::EncodedData& encoded,
          std::shared_ptr<encodings::Codec<uint64_t>> codec = nullptr)
      {
        using namespace encodings;
        
        if (encoded.size() < 3 * sizeof(uint64_t) + sizeof(uint32_t))
        {
          throw std::runtime_error("FastBitset::decode: Insufficient data");
        }
        
        // Read metadata
        const uint8_t* data = encoded.data().data();
        size_t offset = 0;
        
        uint64_t numValues;
        uint32_t valuesPerBitU32;
        uint64_t numBitsU64;
        uint64_t numWords;
        
        std::memcpy(&numValues, data + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);
        std::memcpy(&valuesPerBitU32, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&numBitsU64, data + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);
        std::memcpy(&numWords, data + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);
        
        // Use provided codec or create default
        if (!codec) {
          codec = createDefaultCodec();
        }
        
        // Decompress bits
        std::vector<uint8_t> compressedData(data + offset, data + encoded.size());
        EncodingMetadata bitsMetadata;
        bitsMetadata.encodingName = "CompressedBits";
        bitsMetadata.elementCount = numWords;
        bitsMetadata.dataType = DataType::UInt64;
        bitsMetadata.uncompressedSize = numWords * sizeof(uint64_t);
        EncodedData compressedBits(std::move(compressedData), std::move(bitsMetadata));
        
        std::vector<uint64_t> decompressedBits = codec->decodeAll(compressedBits);
        
        if (decompressedBits.size() != numWords)
        {
          throw std::runtime_error("FastBitset::decode: Decompressed size mismatch");
        }
        
        // Construct new bitset
        FastBitset result(numValues, valuesPerBitU32, codec);
        result.bits_ = std::move(decompressedBits);
        result.numBits_ = numBitsU64;
        
        return result;
      }

      FastBitset() = default;
      FastBitset(const FastBitset &other) = default;
      FastBitset &operator=(const FastBitset &other) = default;
      FastBitset(FastBitset &&other) noexcept = default;
      FastBitset &operator=(FastBitset &&other) noexcept = default;
      ~FastBitset() = default;
      
    private:
      /**
       * @brief Create default RawEncoder codec
       * 
       * Note: Using forward declaration to avoid circular dependency
       * The actual RawEncoder is defined in the encoders module
       */
      static std::shared_ptr<encodings::Codec<uint64_t>> createDefaultCodec();
    };
} // namespace encodings::core