#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/BitPacker.hpp"
#include "encoders/detail/DictionaryHelpers.hpp"

namespace encodings::encoders {

/**
 * @brief Block-partitioned dictionary encoding with per-block key widths.
 *
 * Divides data into fixed-size blocks (element count chosen by greedy sweep).
 * Each block has its own dictionary and bit-packed keys at the minimum width
 * needed for that block's cardinality.  Different blocks may have different
 * key widths, so block byte sizes vary; a flat block-descriptor index stores
 * explicit byte offsets enabling O(1) decodeAt.
 *
 * Format:
 *   Header (16 bytes):
 *     [0..7]   N          uint64_t  element count
 *     [8..11]  blockSize  uint32_t  elements per block
 *     [12..15] numBlocks  uint32_t  number of blocks
 *
 *   Block Index (numBlocks × 13 bytes):
 *     Per block:
 *       [0..7]  payloadByteOffset  uint64_t  byte offset from payload section start
 *       [8..11] dictSize           uint32_t  distinct values in this block
 *       [12]    keyBitWidth        uint8_t   bit-width of keys for this block
 *
 *   Payload (variable):
 *     Per block (in index order):
 *       dict entries: dictSize × sizeof(T) bytes
 *       keys:         ceil(blockElements × keyBitWidth / 8) bytes  (LSB-first)
 *       padding:      8 zero bytes  (enables safe uint64_t loads in decodeAt)
 *
 * @tparam T The value type to encode.  Must be trivially copyable.
 */
template<typename T>
class AdaptiveDictionaryEncoder : public Codec<T> {
    static_assert(std::is_trivially_copyable_v<T>,
        "AdaptiveDictionaryEncoder requires a trivially copyable value type");

public:
    explicit AdaptiveDictionaryEncoder(bool allowNonPowerOfTwoKeyWidths = true)
        : allowNonPowerOfTwoKeyWidths_(allowNonPowerOfTwoKeyWidths) {}

    // -----------------------------------------------------------------------
    // Binary format constants
    // -----------------------------------------------------------------------
    static constexpr size_t   kFileHeaderSize   = 16; // N(8)+blockSize(4)+numBlocks(4)
    static constexpr size_t   kBlockDescSize    = 13; // payloadOffset(8)+dictSize(4)+keyWidth(1)
    static constexpr size_t   kKeysPaddingBytes =  8; // zero-pad after keys for safe 64-bit loads

    // Block size candidates (power-of-two element counts).
    static constexpr uint32_t kCandidates[] = {
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
    };

    // -----------------------------------------------------------------------
    // Encode
    // -----------------------------------------------------------------------
    EncodedData encode(std::span<const T> data) override {
        cache_.base = nullptr;

        if (data.empty()) return createEmptyEncoding();

        const uint64_t N = static_cast<uint64_t>(data.size());

        // Phase 1: choose block size by greedy cost sweep.
        const uint32_t bestBs    = planBlockSize(data);
        const uint32_t numBlocks = static_cast<uint32_t>((N + bestBs - 1) / bestBs);

        // Phase 2: encode with the chosen block size.
        EncodedData result;
        auto& out = result.data();

        // Upper bound: all-unique dict, 32-bit-aligned keys worst case.
        const size_t upperBound = kFileHeaderSize
            + static_cast<size_t>(numBlocks) * kBlockDescSize
            + static_cast<size_t>(N) * (sizeof(T) + sizeof(uint32_t))
            + static_cast<size_t>(numBlocks) * kKeysPaddingBytes;
        out.reserve(upperBound);

        // Write file header.
        out.resize(kFileHeaderSize);
        uint8_t* hdr = out.data();
        std::memcpy(hdr,      &N,         sizeof(uint64_t));
        std::memcpy(hdr + 8,  &bestBs,    sizeof(uint32_t));
        std::memcpy(hdr + 12, &numBlocks, sizeof(uint32_t));

        // Pre-allocate block index (zero-filled; descriptors written in-loop).
        const size_t reservedIndexOffset = out.size(); // == kFileHeaderSize
        out.resize(reservedIndexOffset + static_cast<size_t>(numBlocks) * kBlockDescSize, 0u);
        const size_t payloadStart = out.size();

        // Reuse map/dict/keys across blocks to avoid repeated heap allocations.
        ankerl::unordered_dense::map<T, size_t> valueToKey;
        std::vector<T>      dictionary;
        std::vector<size_t> keys;
        std::vector<uint8_t> keyBuf;

        for (uint32_t b = 0; b < numBlocks; ++b) {
            const size_t blockStart    = static_cast<size_t>(b) * bestBs;
            const size_t blockEnd      = std::min(blockStart + static_cast<size_t>(bestBs),
                                                   static_cast<size_t>(N));
            const size_t blockElements = blockEnd - blockStart;

            // Build per-block dictionary and key stream.
            valueToKey.clear();
            dictionary.clear();
            keys.clear();
            keys.reserve(blockElements);

            for (size_t i = blockStart; i < blockEnd; ++i) {
                const T& val = data[i];
                auto it = valueToKey.find(val);
                if (it == valueToKey.end()) {
                    const size_t newKey = dictionary.size();
                    valueToKey.emplace(val, newKey);
                    dictionary.push_back(val);
                    keys.push_back(newKey);
                } else {
                    keys.push_back(it->second);
                }
            }

            const uint32_t localDictSize = static_cast<uint32_t>(dictionary.size());
            const uint8_t  localKeyWidth = static_cast<uint8_t>(
                detail::chooseKeyBitWidth(localDictSize, allowNonPowerOfTwoKeyWidths_));

            // Descriptor: offset from payload section start to this block's data.
            const uint64_t payloadByteOffset =
                static_cast<uint64_t>(out.size() - payloadStart);

            // Write block descriptor at pre-reserved slot.
            // out.reserve() above guarantees no reallocation; fresh data() after each resize.
            uint8_t* descPtr = out.data() + reservedIndexOffset
                             + static_cast<size_t>(b) * kBlockDescSize;
            std::memcpy(descPtr,     &payloadByteOffset, sizeof(uint64_t));
            std::memcpy(descPtr + 8, &localDictSize,     sizeof(uint32_t));
            descPtr[12] = localKeyWidth;

            // Append dictionary entries.
            const size_t dictByteCount = localDictSize * sizeof(T);
            const size_t prevSize = out.size();
            out.resize(prevSize + dictByteCount);
            std::memcpy(out.data() + prevSize, dictionary.data(), dictByteCount);

            // Bit-pack keys (LSB-first to match decodeGeneral / decodeBatch).
            const size_t keyBytesNeeded = (blockElements * localKeyWidth + 7) / 8;
            keyBuf.clear();
            keyBuf.reserve(keyBytesNeeded + 1);
            encodings::core::BitWriter wr(keyBuf, encodings::core::BitOrder::LSB);
            for (const size_t key : keys) {
                wr.write(static_cast<uint32_t>(key), localKeyWidth);
            }
            wr.flush();
            out.insert(out.end(), keyBuf.begin(), keyBuf.end());

            // 8-byte zero padding: lets decodeAt load a full uint64_t at the last
            // bit position without reading past this block's key data.
            out.resize(out.size() + kKeysPaddingBytes, 0u);
        }

        result.metadata().encodingName        = name();
        result.metadata().dataType            = this->dataType();
        result.metadata().elementCount        = static_cast<size_t>(N);
        result.metadata().compressedSize      = out.size();
        result.metadata().uncompressedSize    = static_cast<size_t>(N) * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["block_size"] = std::to_string(bestBs);
        result.metadata().customMetadata["num_blocks"] = std::to_string(numBlocks);

        return result;
    }

    // -----------------------------------------------------------------------
    // Decode — all elements
    // -----------------------------------------------------------------------
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        const BlockView v = getView(encoded);
        if (v.N == 0) return {};

        std::vector<T> result(v.N);
        for (uint32_t b = 0; b < v.numBlocks; ++b) {
            const BlockDesc   d            = readDesc(v.indexBase, b);
            const size_t      blockStart   = static_cast<size_t>(b) * v.blockSize;
            const size_t      blockElements = std::min(
                static_cast<size_t>(v.blockSize), v.N - blockStart);
            const uint8_t*    blockPayload = v.payloadBase + d.payloadByteOffset;
            const uint8_t*    dictBytes    = blockPayload;
            const uint8_t*    keysPtr      = blockPayload + d.dictSize * sizeof(T);

            detail::dispatchDecode<T>(keysPtr, dictBytes,
                                      result.data() + blockStart,
                                      blockElements, d.keyBitWidth, 0);
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Decode — single element, O(1)
    // -----------------------------------------------------------------------
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || index >= v.N) [[unlikely]] return std::nullopt;

        const uint32_t blockIdx   = static_cast<uint32_t>(index / v.blockSize);
        const size_t   inBlockIdx = index % v.blockSize;
        const BlockDesc d         = readDesc(v.indexBase, blockIdx);

        const uint8_t* blockPayload = v.payloadBase + d.payloadByteOffset;
        const uint8_t* dictBytes    = blockPayload;
        const uint8_t* keysPtr      = blockPayload + d.dictSize * sizeof(T);

        // Raw uint32 path (can only arise if keyBitWidth == 32, which requires
        // dictSize > 2^31 — impossible given max blockSize 65536, but kept for safety).
        if (d.keyBitWidth == 32) [[unlikely]] {
            uint32_t k;
            std::memcpy(&k, keysPtr + inBlockIdx * sizeof(uint32_t), sizeof(uint32_t));
            if (k >= d.dictSize) [[unlikely]] return std::nullopt;
            return detail::loadDictValue<T>(dictBytes, k);
        }

        // Bit-packed path: same word-load trick as DictionaryEncoder::decodeAt.
        // 8-byte block padding makes the cross-word load always safe.
        const size_t   bitOffset = inBlockIdx * d.keyBitWidth;
        const size_t   wordIdx   = bitOffset >> 6;
        const uint32_t off       = static_cast<uint32_t>(bitOffset & 63u);
        uint64_t word;
        std::memcpy(&word, keysPtr + wordIdx * sizeof(uint64_t), sizeof(uint64_t));
        uint64_t key = word >> off;
        if (off + d.keyBitWidth > 64u) {
            uint64_t next;
            std::memcpy(&next, keysPtr + (wordIdx + 1) * sizeof(uint64_t), sizeof(uint64_t));
            key |= next << (64u - off);
        }
        const uint64_t mask = (d.keyBitWidth >= 64u)
                              ? ~uint64_t{0}
                              : ((uint64_t{1} << d.keyBitWidth) - 1u);
        key &= mask;
        if (key >= d.dictSize) [[unlikely]] return std::nullopt;
        return detail::loadDictValue<T>(dictBytes, static_cast<size_t>(key));
    }

    // -----------------------------------------------------------------------
    // Decode — range [start, end)
    // -----------------------------------------------------------------------
    std::vector<T> decodeRange(const EncodedData& encoded,
                                size_t start, size_t end) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || start >= v.N) return {};
        end = std::min(end, v.N);
        if (start >= end) return {};

        std::vector<T> result;
        result.reserve(end - start);
        result.resize(end - start);
        T* dst = result.data();
        size_t written = 0;

        const uint32_t firstBlock = static_cast<uint32_t>(start / v.blockSize);
        const uint32_t lastBlock  = static_cast<uint32_t>((end - 1) / v.blockSize);

        for (uint32_t b = firstBlock; b <= lastBlock; ++b) {
            const BlockDesc d           = readDesc(v.indexBase, b);
            const size_t    bStart      = static_cast<size_t>(b) * v.blockSize;
            const size_t    bEnd        = std::min(bStart + static_cast<size_t>(v.blockSize), v.N);
            const size_t    localStart  = (b == firstBlock) ? (start - bStart) : 0u;
            const size_t    localEnd    = (b == lastBlock)  ? (end   - bStart) : (bEnd - bStart);
            const size_t    localCount  = localEnd - localStart;

            const uint8_t*  blockPayload = v.payloadBase + d.payloadByteOffset;
            const uint8_t*  dictBytes    = blockPayload;
            const uint8_t*  keysPtr      = blockPayload + d.dictSize * sizeof(T);

            if (localStart == 0) {
                detail::dispatchDecode<T>(keysPtr, dictBytes,
                                          dst + written, localCount,
                                          d.keyBitWidth, 0);
            } else {
                detail::decodeGeneral<T>(keysPtr, dictBytes,
                                         dst + written, localCount,
                                         d.keyBitWidth,
                                         localStart * d.keyBitWidth);
            }
            written += localCount;
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Decode into caller-supplied buffer (avoids allocation on hot paths)
    // -----------------------------------------------------------------------
    void decodeAllInto(const EncodedData& encoded, T* dst, size_t n) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || n == 0) return;
        const size_t count = std::min(n, v.N);

        for (uint32_t b = 0; b < v.numBlocks; ++b) {
            const BlockDesc d            = readDesc(v.indexBase, b);
            const size_t    blockStart   = static_cast<size_t>(b) * v.blockSize;
            if (blockStart >= count) break;
            const size_t    blockElements = std::min(
                static_cast<size_t>(v.blockSize), count - blockStart);
            const uint8_t*  blockPayload = v.payloadBase + d.payloadByteOffset;
            const uint8_t*  dictBytes    = blockPayload;
            const uint8_t*  keysPtr      = blockPayload + d.dictSize * sizeof(T);

            detail::dispatchDecode<T>(keysPtr, dictBytes,
                                      dst + blockStart, blockElements,
                                      d.keyBitWidth, 0);
        }
    }

    void decodeRangeInto(const EncodedData& encoded,
                          size_t start, size_t end,
                          T* dst, size_t n) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || start >= v.N || n == 0) return;
        end = std::min(end, v.N);
        if (start >= end) return;

        const uint32_t firstBlock = static_cast<uint32_t>(start / v.blockSize);
        const uint32_t lastBlock  = static_cast<uint32_t>((end - 1) / v.blockSize);
        size_t written = 0;

        for (uint32_t b = firstBlock; b <= lastBlock && written < n; ++b) {
            const BlockDesc d           = readDesc(v.indexBase, b);
            const size_t    bStart      = static_cast<size_t>(b) * v.blockSize;
            const size_t    bEnd        = std::min(bStart + static_cast<size_t>(v.blockSize), v.N);
            const size_t    localStart  = (b == firstBlock) ? (start - bStart) : 0u;
            const size_t    localEnd    = (b == lastBlock)  ? (end   - bStart) : (bEnd - bStart);
            const size_t    localCount  = std::min(localEnd - localStart, n - written);

            const uint8_t*  blockPayload = v.payloadBase + d.payloadByteOffset;
            const uint8_t*  dictBytes    = blockPayload;
            const uint8_t*  keysPtr      = blockPayload + d.dictSize * sizeof(T);

            if (localStart == 0) {
                detail::dispatchDecode<T>(keysPtr, dictBytes,
                                          dst + written, localCount,
                                          d.keyBitWidth, 0);
            } else {
                detail::decodeGeneral<T>(keysPtr, dictBytes,
                                         dst + written, localCount,
                                         d.keyBitWidth,
                                         localStart * d.keyBitWidth);
            }
            written += localCount;
        }
    }

    // -----------------------------------------------------------------------
    // Interface
    // -----------------------------------------------------------------------
    EncodingType encodingType() const override {
        return EncodingType::AdaptiveDictionaryEncoding;
    }

    std::string name() const override { return "AdaptiveDictionary"; }

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
        // Pessimistic: assume all-unique, 32-bit keys.
        return kFileHeaderSize
             + 32 * kBlockDescSize
             + elementCount * (sizeof(T) + sizeof(uint32_t));
    }

private:
    bool allowNonPowerOfTwoKeyWidths_{true};

    // -----------------------------------------------------------------------
    // Per-block descriptor (not stored persistently; read on demand from index)
    // -----------------------------------------------------------------------
    struct BlockDesc {
        uint64_t payloadByteOffset;
        uint32_t dictSize;
        uint8_t  keyBitWidth;
    };

    static BlockDesc readDesc(const uint8_t* indexBase, uint32_t blockIdx) noexcept {
        const uint8_t* p = indexBase + static_cast<size_t>(blockIdx) * kBlockDescSize;
        BlockDesc d;
        std::memcpy(&d.payloadByteOffset, p,     sizeof(uint64_t));
        std::memcpy(&d.dictSize,          p + 8, sizeof(uint32_t));
        d.keyBitWidth = p[12];
        return d;
    }

    // -----------------------------------------------------------------------
    // Zero-copy view into the encoded buffer (pointer-comparison cache)
    // -----------------------------------------------------------------------
    struct BlockView {
        uint64_t        N{0};
        uint32_t        blockSize{0};
        uint32_t        numBlocks{0};
        const uint8_t*  indexBase{nullptr};    // → first BlockDesc in buffer
        const uint8_t*  payloadBase{nullptr};  // → start of Payload section
    };

    struct Cache {
        const uint8_t* base{nullptr};
        BlockView      view{};
    };
    mutable Cache cache_;

    BlockView getView(const EncodedData& encoded) const {
        const uint8_t* base = encoded.data().data();
        if (base == cache_.base) [[likely]] return cache_.view;

        if (encoded.size() < kFileHeaderSize) {
            cache_.base = base;
            cache_.view = {};
            return {};
        }

        uint64_t N;         std::memcpy(&N,         base,      sizeof(uint64_t));
        uint32_t blockSize; std::memcpy(&blockSize, base + 8,  sizeof(uint32_t));
        uint32_t numBlocks; std::memcpy(&numBlocks, base + 12, sizeof(uint32_t));

        if (N == 0 || blockSize == 0 || numBlocks == 0) {
            cache_.base = base;
            cache_.view = {};
            return {};
        }

        BlockView v;
        v.N           = static_cast<size_t>(N);
        v.blockSize   = blockSize;
        v.numBlocks   = numBlocks;
        v.indexBase   = base + kFileHeaderSize;
        v.payloadBase = v.indexBase + static_cast<size_t>(numBlocks) * kBlockDescSize;

        cache_.base = base;
        cache_.view = v;
        return v;
    }

    // -----------------------------------------------------------------------
    // planBlockSize — greedy sweep over candidates; returns best element count
    // -----------------------------------------------------------------------
    uint32_t planBlockSize(std::span<const T> data) const {
        const size_t N = data.size();
        uint32_t bestBs    = kCandidates[0];
        size_t   bestBytes = std::numeric_limits<size_t>::max();

        ankerl::unordered_dense::set<T> seen;

        for (const uint32_t bs : kCandidates) {
            const size_t numBlocks =
                (N + static_cast<size_t>(bs) - 1) / static_cast<size_t>(bs);
            size_t blockPayloadCost = 0;

            for (size_t b = 0; b < numBlocks; ++b) {
                const size_t blockStart    = b * bs;
                const size_t blockEnd      = std::min(blockStart + static_cast<size_t>(bs), N);
                const size_t blockElements = blockEnd - blockStart;

                seen.clear();
                for (size_t i = blockStart; i < blockEnd; ++i) {
                    seen.insert(data[i]);
                }

                const size_t localDictSize = seen.size();
                const uint32_t localKeyWidth =
                    detail::chooseKeyBitWidth(localDictSize, allowNonPowerOfTwoKeyWidths_);

                blockPayloadCost += localDictSize * sizeof(T)
                                  + (blockElements * localKeyWidth + 7) / 8
                                  + kKeysPaddingBytes;
            }

            const size_t totalBytes = kFileHeaderSize
                                    + numBlocks * kBlockDescSize
                                    + blockPayloadCost;
            if (totalBytes < bestBytes) {
                bestBytes = totalBytes;
                bestBs    = bs;
            }
        }

        return bestBs;
    }

    // -----------------------------------------------------------------------
    // createEmptyEncoding
    // -----------------------------------------------------------------------
    EncodedData createEmptyEncoding() const {
        EncodedData result;
        result.data().resize(kFileHeaderSize, 0u);
        result.metadata().encodingName         = name();
        result.metadata().dataType             = this->dataType();
        result.metadata().elementCount         = 0;
        result.metadata().compressedSize       = kFileHeaderSize;
        result.metadata().uncompressedSize     = 0;
        result.metadata().supportsRandomAccess = true;
        return result;
    }
};

} // namespace encodings::encoders
