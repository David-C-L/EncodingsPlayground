#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encoders/FSEEncoder.hpp"

namespace encodings::encoders {

/**
 * BlockFSEEncoder — block-indexed Finite State Entropy encoding.
 *
 * Wraps FSEEncoder<T> (sequential-only tANS) with a flat per-block byte-offset
 * index.  Each block is independently FSE-encoded with its own frequency table,
 * making it fully self-contained.  The index enables direct block seek, yielding
 * O(blockSize) decodeAt and O(range_size + blockSize) decodeRange.
 *
 * When to prefer over BlockFrequencyPartitionEncoder:
 *   BlockFSE is advantageous when a section has more than ~22 distinct values
 *   and a skewed distribution — conditions where BlockFP's tier-key capacity
 *   (2+4+16 slots) saturates and pushes many elements to the full-width fallback.
 *   BlockFSE codes every value near its Shannon entropy limit regardless of
 *   cardinality.
 *
 * Binary format:
 *   File header (16 bytes):
 *     [0..7]   N          uint64_t   total element count
 *     [8..11]  blockSize  uint32_t   elements per block
 *     [12..15] numBlocks  uint32_t   number of blocks
 *
 *   Block index (numBlocks x 8 bytes):
 *     per block: payloadByteOffset uint64_t (byte offset from payload start)
 *
 *   Payload: concatenated self-contained FSE blocks.
 *     Each carries its own FSEEncoder wire format header + symbol table + bitstream.
 *     Block byte lengths are derived from consecutive index offsets; the last block
 *     extends to the end of the encoded buffer.
 */
template<typename T>
    requires std::is_integral_v<T>
class BlockFSEEncoder : public Codec<T> {
public:
    static constexpr size_t   kFileHeaderSize = 16; // N(8)+blockSize(4)+numBlocks(4)
    static constexpr size_t   kBlockDescSize  =  8; // payloadByteOffset(8) only
    static constexpr size_t   kFSEHeaderFixed = FSEEncoder<T>::kHeaderFixed; // 17

    static constexpr uint32_t kCandidates[] = { 256, 512, 1024, 2048, 4096, 8192 };

    // ===================================================================
    // Encode
    // ===================================================================
    EncodedData encode(std::span<const T> data) override {
        cache_.base = nullptr;
        fseCache_   = {};
        if (data.empty()) return createEmptyEncoding();

        const uint64_t N         = static_cast<uint64_t>(data.size());
        const uint32_t bestBs    = planBlockSize(data);
        const uint32_t numBlocks = static_cast<uint32_t>((N + bestBs - 1) / bestBs);

        EncodedData result;
        auto& out = result.data();
        out.reserve(kFileHeaderSize
            + static_cast<size_t>(numBlocks) * kBlockDescSize
            + static_cast<size_t>(N) * sizeof(T));

        // File header
        out.resize(kFileHeaderSize);
        std::memcpy(out.data(),      &N,         sizeof(uint64_t));
        std::memcpy(out.data() + 8,  &bestBs,    sizeof(uint32_t));
        std::memcpy(out.data() + 12, &numBlocks, sizeof(uint32_t));

        // Pre-allocate block index (offsets filled per block below)
        const size_t indexOffset = out.size();
        out.resize(indexOffset + static_cast<size_t>(numBlocks) * kBlockDescSize, 0u);
        const size_t payloadStart = out.size();

        FSEEncoder<T> fseEnc;
        for (uint32_t b = 0; b < numBlocks; ++b) {
            const size_t bStart = static_cast<size_t>(b) * bestBs;
            const size_t bEnd   = std::min(bStart + static_cast<size_t>(bestBs),
                                            static_cast<size_t>(N));
            const auto blockEncoded = fseEnc.encode(
                std::span<const T>(data.data() + bStart, bEnd - bStart));
            const auto& blockBytes = blockEncoded.data();

            // Record byte offset into block index
            const uint64_t payloadByteOffset = static_cast<uint64_t>(out.size() - payloadStart);
            uint8_t* desc = out.data() + indexOffset + static_cast<size_t>(b) * kBlockDescSize;
            std::memcpy(desc, &payloadByteOffset, sizeof(uint64_t));

            out.insert(out.end(), blockBytes.begin(), blockBytes.end());
        }

        result.metadata().encodingName         = name();
        result.metadata().dataType             = this->dataType();
        result.metadata().elementCount         = static_cast<size_t>(N);
        result.metadata().compressedSize       = out.size();
        result.metadata().uncompressedSize     = static_cast<size_t>(N) * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["block_size"] = std::to_string(bestBs);
        result.metadata().customMetadata["num_blocks"] = std::to_string(numBlocks);
        return result;
    }

    // ===================================================================
    // Decode all elements
    // ===================================================================
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        const BlockView v = getView(encoded);
        if (v.N == 0) return {};
        std::vector<T> result(v.N);
        decodeAllInto(encoded, result.data(), result.size());
        return result;
    }

    // ===================================================================
    // Decode single element
    // ===================================================================
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || index >= v.N) [[unlikely]] return std::nullopt;
        const uint32_t b         = static_cast<uint32_t>(index / v.blockSize);
        const size_t   localIdx  = index % v.blockSize;
        const uint64_t offset    = readOffset(v.indexBase, b);
        const size_t   byteCount = blockByteSize(v, b);
        const size_t   bElems    = std::min(static_cast<size_t>(v.blockSize),
                                            v.N - static_cast<size_t>(b) * v.blockSize);
        scratch_.resize(bElems);
        decodeBlockDirect(v.payloadBase + offset, byteCount, scratch_.data(), bElems);
        if (localIdx >= scratch_.size()) [[unlikely]] return std::nullopt;
        return scratch_[localIdx];
    }

    // ===================================================================
    // Decode range [start, end)
    // ===================================================================
    std::vector<T> decodeRange(const EncodedData& encoded,
                                size_t start, size_t end) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || start >= v.N) return {};
        end = std::min(end, v.N);
        if (start >= end) return {};
        std::vector<T> result(end - start);
        decodeRangeInto(encoded, start, end, result.data(), result.size());
        return result;
    }

    // ===================================================================
    // Decode into caller-supplied buffer (zero allocation on hot path)
    // ===================================================================
    void decodeAllInto(const EncodedData& encoded, T* dst, size_t n) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || n == 0) return;
        const size_t count   = std::min(n, v.N);
        size_t       written = 0;

        for (uint32_t b = 0; b < v.numBlocks && written < count; ++b) {
            const uint64_t offset    = readOffset(v.indexBase, b);
            const size_t   byteCount = blockByteSize(v, b);
            // Decode at most (count - written) elements: FSEEncoder stops after
            // min(numElements_from_header, maxDst) — no scratch buffer needed.
            const size_t toCopy = decodeBlockDirect(
                v.payloadBase + offset, byteCount, dst + written, count - written);
            written += toCopy;
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
            const uint64_t offset     = readOffset(v.indexBase, b);
            const size_t   byteCount  = blockByteSize(v, b);
            const size_t   bElemStart = static_cast<size_t>(b) * v.blockSize;
            const size_t   bElemEnd   = std::min(bElemStart + static_cast<size_t>(v.blockSize), v.N);
            const size_t   bElems     = bElemEnd - bElemStart;
            const size_t   localStart = (b == firstBlock) ? (start - bElemStart) : 0u;
            const size_t   localEnd   = (b == lastBlock)  ? (end   - bElemStart) : bElems;
            const size_t   localCount = std::min(localEnd - localStart, n - written);

            if (localStart == 0) {
                // No skipping needed: decode localCount elements directly into dst.
                // (If localCount == bElems this is a full block; if less, FSE stops early.)
                const size_t got = decodeBlockDirect(
                    v.payloadBase + offset, byteCount, dst + written, localCount);
                written += got;
            } else {
                // Non-zero skip: must decode the full block into scratch to advance
                // past localStart elements, then copy [localStart, localEnd) to dst.
                scratch_.resize(bElems);
                decodeBlockDirect(v.payloadBase + offset, byteCount, scratch_.data(), bElems);
                const size_t toCopy = std::min(localCount, bElems > localStart ? bElems - localStart : 0u);
                std::memcpy(dst + written, scratch_.data() + localStart, toCopy * sizeof(T));
                written += toCopy;
            }
        }
    }

    // ===================================================================
    // Interface
    // ===================================================================
    EncodingType encodingType() const override { return EncodingType::BlockFSEEncoding; }
    std::string  name()         const override { return "BlockFSE"; }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::RandomAccess)
             | EncodingProperty::Lossless
             | EncodingProperty::PreservesOrder
             | EncodingProperty::RequiresFullData
             | EncodingProperty::VariableSize
             | EncodingProperty::Composable;
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        return kFileHeaderSize + 32 * kBlockDescSize + elementCount * sizeof(T);
    }

private:
    // ===================================================================
    // Zero-copy view into the encoded buffer (pointer-comparison cache)
    // ===================================================================
    struct BlockView {
        size_t         N{0};
        uint32_t       blockSize{0}, numBlocks{0};
        size_t         totalPayloadBytes{0};
        const uint8_t* indexBase{nullptr};
        const uint8_t* payloadBase{nullptr};
    };
    struct Cache { const uint8_t* base{nullptr}; BlockView view{}; };
    mutable Cache                                      cache_;
    mutable typename FSEEncoder<T>::BlockDecodeContext fseCache_; // decode-table LRU-1 cache
    mutable std::vector<T>                             scratch_;  // reused for partial-block decodes

    BlockView getView(const EncodedData& encoded) const {
        const uint8_t* base = encoded.data().data();
        if (base == cache_.base) [[likely]] return cache_.view;
        if (encoded.size() < kFileHeaderSize) { cache_ = {base, {}}; return {}; }
        uint64_t N;         std::memcpy(&N,         base,      sizeof(uint64_t));
        uint32_t blockSize; std::memcpy(&blockSize, base + 8,  sizeof(uint32_t));
        uint32_t numBlocks; std::memcpy(&numBlocks, base + 12, sizeof(uint32_t));
        if (N == 0 || blockSize == 0 || numBlocks == 0) { cache_ = {base, {}}; return {}; }
        BlockView v;
        v.N         = static_cast<size_t>(N);
        v.blockSize = blockSize;
        v.numBlocks = numBlocks;
        v.indexBase   = base + kFileHeaderSize;
        v.payloadBase = v.indexBase + static_cast<size_t>(numBlocks) * kBlockDescSize;
        const size_t payloadStart = kFileHeaderSize + static_cast<size_t>(numBlocks) * kBlockDescSize;
        v.totalPayloadBytes = (encoded.size() > payloadStart)
                            ? (encoded.size() - payloadStart) : 0u;
        cache_ = {base, v};
        return v;
    }

    // ===================================================================
    // Helpers
    // ===================================================================
    static uint64_t readOffset(const uint8_t* indexBase, uint32_t b) noexcept {
        uint64_t offset;
        std::memcpy(&offset, indexBase + static_cast<size_t>(b) * kBlockDescSize, sizeof(uint64_t));
        return offset;
    }

    // Exact byte count for block b, derived from consecutive index entries.
    static size_t blockByteSize(const BlockView& v, uint32_t b) noexcept {
        const uint64_t thisOffset = readOffset(v.indexBase, b);
        if (b + 1 < v.numBlocks) {
            const uint64_t nextOffset = readOffset(v.indexBase, b + 1);
            return static_cast<size_t>(nextOffset - thisOffset);
        }
        return (v.totalPayloadBytes > static_cast<size_t>(thisOffset))
             ? (v.totalPayloadBytes - static_cast<size_t>(thisOffset)) : 0u;
    }

    // ===================================================================
    // decodeBlockDirect — decode one self-contained FSE block from raw bytes
    // directly into dst[0..maxDst), using the per-instance decode-table cache.
    //
    // Eliminates: (a) EncodedData byte copy, (b) result vector allocation,
    // (c) decode-table rebuild when adjacent blocks share the same distribution.
    //
    // Returns the number of elements written (= min(numElements_in_block, maxDst)).
    // ===================================================================
    size_t decodeBlockDirect(const uint8_t* blockStart, size_t byteCount,
                              T* dst, size_t maxDst) const {
        return FSEEncoder<T>{}.decodeBlockInto(blockStart, byteCount, dst, maxDst, &fseCache_);
    }

    // ===================================================================
    // planBlockSize — greedy cost sweep; returns best element count per block.
    // ===================================================================
    uint32_t planBlockSize(std::span<const T> data) const {
        const size_t N     = data.size();
        uint32_t bestBs    = kCandidates[0];
        size_t   bestBytes = std::numeric_limits<size_t>::max();
        constexpr size_t kSymTableEntryBytes = sizeof(T) + 2;

        ankerl::unordered_dense::map<T, uint64_t> freq;
        freq.reserve(256);

        for (const uint32_t bs : kCandidates) {
            const size_t numBlocks = (N + bs - 1) / bs;
            size_t payloadCost = 0;

            for (size_t b = 0; b < numBlocks; ++b) {
                const size_t bStart = b * bs;
                const size_t bEnd   = std::min(bStart + static_cast<size_t>(bs), N);
                const size_t bCount = bEnd - bStart;

                freq.clear();
                for (size_t i = bStart; i < bEnd; ++i) ++freq[data[i]];

                // FSE header overhead
                payloadCost += kFSEHeaderFixed + freq.size() * kSymTableEntryBytes;

                // Shannon entropy estimate -> payload bytes
                double entropy = 0.0;
                for (const auto& [sym, cnt] : freq) {
                    const double p = static_cast<double>(cnt) / static_cast<double>(bCount);
                    if (p > 0.0) entropy -= p * std::log2(p);
                }
                payloadCost += static_cast<size_t>(
                    std::ceil(entropy * static_cast<double>(bCount) / 8.0));
            }

            const size_t total = kFileHeaderSize + numBlocks * kBlockDescSize + payloadCost;
            if (total < bestBytes) { bestBytes = total; bestBs = bs; }
        }
        return bestBs;
    }

    // ===================================================================
    // createEmptyEncoding
    // ===================================================================
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
