#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
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
#include "core/BitPacker.hpp"
#include "encoders/detail/DictionaryHelpers.hpp"

namespace encodings::encoders {

/**
 * FORPrepass — selects the global pre-pass applied before block-wise FPE encoding.
 *   None      : no pre-pass; values encoded as-is (original behaviour).
 *   GlobalFOR : Frame-of-Reference over the entire array — subtract the global
 *               minimum from every element to produce small non-negative residuals,
 *               then encode residuals block-wise with FPE.  Analogous to OpenZL's
 *               range_pack followed by huffman encoding.  Requires an integral T.
 */
enum class FORPrepass { None, GlobalFOR };

/**
 * BlockFrequencyPartitionEncoder — block-local frequency-partitioned encoding.
 *
 * Values within each block are sorted by frequency and assigned to up to three
 * compact tiers:
 *   Tier 0: top-2 most frequent values, 1-bit keys (capacity 2)
 *   Tier 1: next-4 most frequent values, 2-bit keys (capacity 4)
 *   Tier 2: next-16 most frequent values, 4-bit keys (capacity 16)
 *   Fallback: remaining values, stored raw at sizeof(T)/element by default, or
 *             bit-packed as (fallbackMin, fallbackBits)-deltas when the
 *             constructor's bitPackFallback flag is set (see constructor doc).
 *             Encode-time-only choice — decode reads the per-block
 *             fallbackBits field, so it never needs to match the encoding
 *             instance's own setting.
 *
 * Unlike global FrequencyPartitionEncoding, which needs a Rank9-backed bitmap
 * spanning all N elements, each block stores a compact tier-tag bitfield
 * (1–2 bits/element) that fits in L1 cache.  This gives O(block_size) decodeAt
 * and O(range_size) decodeRange with no large auxiliary data structures.
 *
 * Tag code layout (compact, based on which tiers are active):
 *   code 0 = tier-0 (always)
 *   code 1 = tier-1 if numTier1>0, else fallback
 *   code 2 = tier-2 if (numTier1>0 && numTier2>0), else fallback
 *   code 3 = fallback (only when all three tiers are active)
 *
 * Binary format (Prepass == None):
 *   Header (16 bytes):
 *     [0..7]   N          uint64_t  element count
 *     [8..11]  blockSize  uint32_t  elements per block
 *     [12..15] numBlocks  uint32_t  number of blocks
 *
 * Binary format (Prepass == GlobalFOR) — header extended by sizeof(T):
 *   Header (16 + sizeof(T) bytes):
 *     [0..7]   N          uint64_t  element count
 *     [8..11]  blockSize  uint32_t  elements per block
 *     [12..15] numBlocks  uint32_t  number of blocks
 *     [16..16+sizeof(T)-1]  globalMin  T  global minimum subtracted from all values
 *
 *   Block index (numBlocks × (24 + sizeof(T) + 1) bytes):
 *     Per block:
 *       [0..7]   payloadByteOffset  uint64_t  byte offset into payload section
 *       [8]      tagBitWidth        uint8_t   bits per tier-tag (0, 1, or 2)
 *       [9]      numTier0           uint8_t   distinct values in tier 0 (0–2)
 *       [10]     numTier1           uint8_t   distinct values in tier 1 (0–4)
 *       [11]     numTier2           uint8_t   distinct values in tier 2 (0–16)
 *       [12..15] tier0Count         uint32_t  elements assigned to tier 0
 *       [16..19] tier1Count         uint32_t  elements assigned to tier 1
 *       [20..23] tier2Count         uint32_t  elements assigned to tier 2
 *       [24..24+sizeof(T)-1] fallbackMin  T   subtracted from fallback deltas
 *       [24+sizeof(T)]       fallbackBits uint8_t  bits/element, or
 *                                                   kRawFallbackSentinel (0xFF)
 *                                                   for raw sizeof(T) storage
 *     numFallback = blockElements - tier0Count - tier1Count - tier2Count
 *
 *   Payload (variable, contiguous):
 *     Per block:
 *       tier-tag bitfield:  ceil(blockElements × tagBitWidth / 8) bytes
 *       tier-0 dict:        numTier0 × sizeof(T)
 *       tier-0 keys:        ceil(tier0Count × keyBits(numTier0) / 8) bytes + 8B padding
 *       tier-1 dict:        numTier1 × sizeof(T)         [omitted if numTier1==0]
 *       tier-1 keys:        ceil(tier1Count × keyBits(numTier1) / 8) + 8B padding  [omitted if numTier1==0]
 *       tier-2 dict:        numTier2 × sizeof(T)         [omitted if numTier2==0]
 *       tier-2 keys:        ceil(tier2Count × keyBits(numTier2) / 8) + 8B padding  [omitted if numTier2==0]
 *       fallback values:    numFallback × sizeof(T), OR (if fallbackBits !=
 *                           kRawFallbackSentinel) ceil(numFallback × fallbackBits
 *                           / 64) × 8 bytes of (value - fallbackMin) deltas,
 *                           word-packed LSB-first (see packFallbackBits)
 *
 *   keyBits(n) = bit_width(n - 1): the exact (possibly non-power-of-two) number
 *   of bits needed to index n dictionary entries — e.g. 0 for a singleton tier,
 *   3 for a 5-8 entry tier2. Computed identically at encode and decode time from
 *   numTierK, which is already stored above, so no extra descriptor bytes are
 *   needed and the format stays self-describing (see detail::exactKeyBitWidth).
 */
template<typename T, FORPrepass Prepass = FORPrepass::None>
class BlockFrequencyPartitionEncoder : public Codec<T> {
    static_assert(std::is_trivially_copyable_v<T>,
        "BlockFrequencyPartitionEncoder requires a trivially copyable value type");
    static_assert(Prepass != FORPrepass::GlobalFOR || std::is_integral_v<T>,
        "BlockFrequencyPartitionEncoder: GlobalFOR prepass requires an integral value type");

    // Capacity ceilings only — NOT the wire bit-width. The actual per-block key
    // width is the exact bits needed for however many values are actually used
    // (numTierK <= cap), computed via detail::exactKeyBitWidth(numTierK); e.g. a
    // tier2 with only 5 distinct values costs 3 bits/key, not the 4-bit ceiling.
    static constexpr uint8_t kTier0Cap = 2;   // <=1-bit key → 2 values
    static constexpr uint8_t kTier1Cap = 4;   // <=2-bit key → 4 values
    static constexpr uint8_t kTier2Cap = 16;  // <=4-bit key → 16 values

public:
    // Header is extended by sizeof(T) for the GlobalFOR global minimum.
    static constexpr size_t kFileHeaderSize =
        16 + (Prepass == FORPrepass::GlobalFOR ? sizeof(T) : 0);

    // Base 24 bytes: payloadOff(8)+tagBW(1)+nT0-2(3)+t0-2Count(12). Extended by
    // sizeof(T)+1 for the fallback bit-packing fields (fallbackMin, fallbackBits)
    // — always present regardless of whether bit-packing is enabled for this
    // instance, so the wire format is self-describing per block (see
    // kRawFallbackSentinel) and decode never needs to know the encoder's config.
    static constexpr size_t   kBlockDescSize   = 24 + sizeof(T) + 1;
    static constexpr size_t   kKeyPaddingBytes =  8; // zero-pad after keys for safe 64-bit loads

    // fallbackBits sentinel meaning "fallback stored raw at sizeof(T) bytes/element"
    // (the original behaviour) rather than bit-packed at fallbackBits bits/element
    // offset from fallbackMin. 0xFF is unambiguous: valid bit-packed widths are 0-64.
    static constexpr uint8_t kRawFallbackSentinel = 0xFFu;

    // Floor lowered from 64 to 8: cascading-FOR experiments showed real, large
    // compressibility gains at block granularities as small as 16 for at least
    // some datasets, which the old {64,...} floor made this planner structurally
    // blind to (planBlockSize is an exact cost sweep over these candidates, so
    // it can only find what's in the list).
    static constexpr uint32_t kCandidates[] = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };

    // Per-element classification used during encoding
    struct ElemCode { uint8_t tagCode; uint8_t tierKey; };

    // ---------------------------------------------------------------------
    // Config
    // ---------------------------------------------------------------------
    // bitPackFallback: when true, each block's fallback tier (values that don't
    // fit in tier0/1/2) is stored as (fallbackMin, fallbackBits)-packed deltas
    // instead of raw sizeof(T)-byte values — the same idea RawBitPackedEncoder
    // uses, applied specifically to the fallback tier. Purely an encode-time
    // choice: decode always reads the per-block fallbackBits field to know how
    // to interpret that block's fallback bytes, so a decoding instance's own
    // bitPackFallback_ value never matters (self-describing wire format).
    explicit BlockFrequencyPartitionEncoder(bool bitPackFallback = false)
        : bitPackFallback_(bitPackFallback) {}

    // ===================================================================
    // Encode
    // ===================================================================
    EncodedData encode(std::span<const T> data) override {
        cache_.base = nullptr;
        if (data.empty()) return createEmptyEncoding();

        const uint64_t N        = static_cast<uint64_t>(data.size());
        const uint32_t bestBs   = planBlockSize(data);
        const uint32_t numBlocks = static_cast<uint32_t>((N + bestBs - 1) / bestBs);

        EncodedData result;
        auto& out = result.data();
        out.reserve(kFileHeaderSize
            + static_cast<size_t>(numBlocks) * kBlockDescSize
            + static_cast<size_t>(N) * (sizeof(T) + 1)
            + static_cast<size_t>(numBlocks) * (kTier0Cap + kTier1Cap + kTier2Cap) * sizeof(T)
            + static_cast<size_t>(numBlocks) * 3 * kKeyPaddingBytes);

        // File header (extended by sizeof(T) for GlobalFOR global minimum)
        out.resize(kFileHeaderSize, 0u);
        std::memcpy(out.data(),      &N,         sizeof(uint64_t));
        std::memcpy(out.data() + 8,  &bestBs,    sizeof(uint32_t));
        std::memcpy(out.data() + 12, &numBlocks, sizeof(uint32_t));

        // GlobalFOR pre-pass: compute global minimum, write it into the header,
        // and produce residuals = data[i] - globalMin for all i.
        // Unsigned arithmetic avoids signed-overflow UB for signed integral T.
        std::vector<T> residualStorage;
        std::span<const T> workData = data;
        if constexpr (Prepass == FORPrepass::GlobalFOR) {
            using U = std::make_unsigned_t<T>;
            const T globalMin = *std::min_element(data.begin(), data.end());
            std::memcpy(out.data() + 16, &globalMin, sizeof(T));
            residualStorage.resize(data.size());
            for (size_t i = 0; i < data.size(); ++i)
                residualStorage[i] = static_cast<T>(
                    static_cast<U>(data[i]) - static_cast<U>(globalMin));
            workData = std::span<const T>(residualStorage);
        }

        // Pre-allocate block index (filled per block below)
        const size_t indexOffset = out.size();
        out.resize(indexOffset + static_cast<size_t>(numBlocks) * kBlockDescSize, 0u);
        const size_t payloadStart = out.size();

        // Temporaries reused across blocks
        ankerl::unordered_dense::map<T, uint32_t> freq;
        std::vector<std::pair<uint32_t, T>> sortedFreq;
        ankerl::unordered_dense::map<T, std::pair<uint8_t, uint8_t>> valToCode;
        std::vector<ElemCode> elemCodes;
        std::vector<uint8_t> tagBuf, keyBuf;
        std::vector<T> fbVals;

        for (uint32_t b = 0; b < numBlocks; ++b) {
            const size_t bStart = static_cast<size_t>(b) * bestBs;
            const size_t bEnd   = std::min(bStart + static_cast<size_t>(bestBs), static_cast<size_t>(N));
            const size_t bCount = bEnd - bStart;

            // --- Frequency sort ---
            freq.clear();
            for (size_t i = bStart; i < bEnd; ++i) freq[workData[i]]++;
            sortedFreq.clear();
            sortedFreq.reserve(freq.size());
            for (const auto& [v, c] : freq) sortedFreq.push_back({c, v});
            std::sort(sortedFreq.rbegin(), sortedFreq.rend());

            // --- Assign values to tiers in frequency order ---
            T tier0Dict[kTier0Cap], tier1Dict[kTier1Cap], tier2Dict[kTier2Cap];
            uint8_t numTier0 = 0, numTier1 = 0, numTier2 = 0;
            size_t sfIdx = 0;
            while (sfIdx < sortedFreq.size() && numTier0 < kTier0Cap)
                tier0Dict[numTier0++] = sortedFreq[sfIdx++].second;
            while (sfIdx < sortedFreq.size() && numTier1 < kTier1Cap)
                tier1Dict[numTier1++] = sortedFreq[sfIdx++].second;
            while (sfIdx < sortedFreq.size() && numTier2 < kTier2Cap)
                tier2Dict[numTier2++] = sortedFreq[sfIdx++].second;
            const bool hasFallback = (sfIdx < sortedFreq.size());

            // --- Determine compact tag code layout ---
            // Codes are contiguous starting at 0; tier codes precede fallback.
            const uint8_t fbCode =
                static_cast<uint8_t>((numTier0 > 0 ? 1u : 0u)
                                   + (numTier1 > 0 ? 1u : 0u)
                                   + (numTier2 > 0 ? 1u : 0u));
            const uint8_t numCodes = fbCode + (hasFallback ? 1u : 0u);
            const uint8_t tagBitWidth = (numCodes <= 1u) ? 0u : (numCodes == 2u) ? 1u : 2u;
            // tier-2's code is one below fallback (only reachable when numTier1>0)
            const uint8_t t2Code = (numTier1 > 0u) ? 2u : 1u;

            // --- Build val → (tagCode, tierKey) lookup ---
            valToCode.clear();
            valToCode.reserve(numTier0 + numTier1 + numTier2);
            for (uint8_t k = 0; k < numTier0; ++k) valToCode[tier0Dict[k]] = {0u, k};
            for (uint8_t k = 0; k < numTier1; ++k) valToCode[tier1Dict[k]] = {1u, k};
            for (uint8_t k = 0; k < numTier2; ++k) valToCode[tier2Dict[k]] = {t2Code, k};

            // --- Classify each element; count per-tier element totals ---
            elemCodes.resize(bCount);
            uint32_t tier0Count = 0, tier1Count = 0, tier2Count = 0;
            for (size_t i = 0; i < bCount; ++i) {
                const auto it = valToCode.find(workData[bStart + i]);
                if (it != valToCode.end()) {
                    elemCodes[i] = {it->second.first, it->second.second};
                    if      (it->second.first == 0u)     tier0Count++;
                    else if (it->second.first == 1u)     tier1Count++;
                    else                                  tier2Count++;
                } else {
                    elemCodes[i] = {fbCode, 0u};
                }
            }

            // --- Fallback values for this block (gathered once; reused below for
            //     both the fallbackMin/fallbackBits descriptor fields and the
            //     actual write, whichever mode is active) ---
            fbVals.clear();
            if (hasFallback) {
                for (size_t i = 0; i < bCount; ++i) {
                    if (elemCodes[i].tagCode == fbCode) fbVals.push_back(workData[bStart + i]);
                }
            }
            T fallbackMin{};
            uint8_t fallbackBits = kRawFallbackSentinel;
            if (bitPackFallback_ && !fbVals.empty()) {
                using U = std::make_unsigned_t<T>;
                T fbMax = fbVals[0];
                fallbackMin = fbVals[0];
                for (T v : fbVals) { fallbackMin = std::min(fallbackMin, v); fbMax = std::max(fbMax, v); }
                const uint64_t span = static_cast<uint64_t>(static_cast<U>(fbMax) - static_cast<U>(fallbackMin));
                fallbackBits = span == 0 ? 0u : static_cast<uint8_t>(64 - std::countl_zero(span));
            }

            // --- Write block descriptor ---
            const uint64_t payloadByteOffset = static_cast<uint64_t>(out.size() - payloadStart);
            uint8_t* desc = out.data() + indexOffset + static_cast<size_t>(b) * kBlockDescSize;
            std::memcpy(desc,      &payloadByteOffset, sizeof(uint64_t));
            desc[8]  = tagBitWidth;
            desc[9]  = numTier0;
            desc[10] = numTier1;
            desc[11] = numTier2;
            std::memcpy(desc + 12, &tier0Count, sizeof(uint32_t));
            std::memcpy(desc + 16, &tier1Count, sizeof(uint32_t));
            std::memcpy(desc + 20, &tier2Count, sizeof(uint32_t));
            std::memcpy(desc + 24, &fallbackMin, sizeof(T));
            desc[24 + sizeof(T)] = fallbackBits;

            // --- Tier-tag bitfield ---
            if (tagBitWidth > 0) {
                const size_t tagBytes = (bCount * tagBitWidth + 7) / 8;
                tagBuf.assign(tagBytes, 0u);
                for (size_t i = 0; i < bCount; ++i) {
                    const size_t bitPos = i * tagBitWidth;
                    tagBuf[bitPos >> 3] |= static_cast<uint8_t>(elemCodes[i].tagCode << (bitPos & 7u));
                }
                out.insert(out.end(), tagBuf.begin(), tagBuf.end());
            }

            // --- Tier key widths: exact bits needed for the values actually
            //     used in each tier (<= the tier's capacity ceiling), not a
            //     fixed power-of-two. Recomputed identically at decode time
            //     from numTier0/1/2, which are already stored above. ---
            const uint8_t keyBits0 = static_cast<uint8_t>(detail::exactKeyBitWidth(numTier0));
            const uint8_t keyBits1 = (numTier1 > 0)
                ? static_cast<uint8_t>(detail::exactKeyBitWidth(numTier1)) : 0u;
            const uint8_t keyBits2 = (numTier2 > 0)
                ? static_cast<uint8_t>(detail::exactKeyBitWidth(numTier2)) : 0u;

            // --- Tier-0: dict + keys ---
            appendBytes(out, tier0Dict, numTier0 * sizeof(T));
            writeTierKeys(out, elemCodes, bCount, 0u, keyBits0, keyBuf);

            // --- Tier-1: dict + keys (only if active) ---
            if (numTier1 > 0) {
                appendBytes(out, tier1Dict, numTier1 * sizeof(T));
                writeTierKeys(out, elemCodes, bCount, 1u, keyBits1, keyBuf);
            }

            // --- Tier-2: dict + keys (only if active) ---
            if (numTier2 > 0) {
                appendBytes(out, tier2Dict, numTier2 * sizeof(T));
                writeTierKeys(out, elemCodes, bCount, t2Code, keyBits2, keyBuf);
            }

            // --- Fallback: bit-packed deltas from fallbackMin, or raw sizeof(T)
            //     values, per fallbackBits (element order, matching the rank
            //     order decodeBlockRange reconstructs via its forward scan) ---
            if (fallbackBits != kRawFallbackSentinel) {
                if (fallbackBits > 0) {
                    using U = std::make_unsigned_t<T>;
                    std::vector<uint64_t> deltas(fbVals.size());
                    for (size_t i = 0; i < fbVals.size(); ++i) {
                        deltas[i] = static_cast<uint64_t>(static_cast<U>(fbVals[i]) - static_cast<U>(fallbackMin));
                    }
                    packFallbackBits(out, deltas, fallbackBits);
                }
                // fallbackBits == 0: every fallback element equals fallbackMin;
                // nothing to store beyond the descriptor's fallbackMin field.
            } else {
                for (const T& v : fbVals) {
                    appendBytes(out, &v, sizeof(T));
                }
            }
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
    // Decode — all elements
    // ===================================================================
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        const BlockView v = getView(encoded);
        if (v.N == 0) return {};
        std::vector<T> result(v.N);
        for (uint32_t b = 0; b < v.numBlocks; ++b) {
            const BlockDesc d = readDesc(v.indexBase, b);
            const size_t bStart = static_cast<size_t>(b) * v.blockSize;
            const size_t bElems = std::min(static_cast<size_t>(v.blockSize), v.N - bStart);
            decodeBlockRange(d, v.payloadBase + d.payloadByteOffset,
                             bElems, 0, bElems, result.data() + bStart, v.globalMin);
        }
        return result;
    }

    // ===================================================================
    // Decode — single element
    // ===================================================================
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || index >= v.N) [[unlikely]] return std::nullopt;
        const uint32_t b        = static_cast<uint32_t>(index / v.blockSize);
        const size_t   localIdx = index % v.blockSize;
        const BlockDesc d       = readDesc(v.indexBase, b);
        const size_t bStart  = static_cast<size_t>(b) * v.blockSize;
        const size_t bElems  = std::min(static_cast<size_t>(v.blockSize), v.N - bStart);
        T val{};
        decodeBlockRange(d, v.payloadBase + d.payloadByteOffset,
                         bElems, localIdx, localIdx + 1, &val, v.globalMin);
        return val;
    }

    // ===================================================================
    // Decode — range [start, end)
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
    // Decode into caller-supplied buffers (no allocation on hot paths)
    // ===================================================================
    void decodeAllInto(const EncodedData& encoded, T* dst, size_t n) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || n == 0) return;
        const size_t count = std::min(n, v.N);
        for (uint32_t b = 0; b < v.numBlocks; ++b) {
            const BlockDesc d      = readDesc(v.indexBase, b);
            const size_t bStart    = static_cast<size_t>(b) * v.blockSize;
            if (bStart >= count) break;
            const size_t bElems    = std::min(static_cast<size_t>(v.blockSize), count - bStart);
            decodeBlockRange(d, v.payloadBase + d.payloadByteOffset,
                             bElems, 0, bElems, dst + bStart, v.globalMin);
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
            const BlockDesc d          = readDesc(v.indexBase, b);
            const size_t    bElemStart = static_cast<size_t>(b) * v.blockSize;
            const size_t    bElemEnd   = std::min(bElemStart + static_cast<size_t>(v.blockSize), v.N);
            const size_t    bElems     = bElemEnd - bElemStart;
            const size_t    localStart = (b == firstBlock) ? (start - bElemStart) : 0u;
            const size_t    localEnd   = (b == lastBlock)  ? (end   - bElemStart) : bElems;
            const size_t    localCount = std::min(localEnd - localStart, n - written);
            decodeBlockRange(d, v.payloadBase + d.payloadByteOffset,
                             bElems, localStart, localStart + localCount, dst + written,
                             v.globalMin);
            written += localCount;
        }
    }

    // ===================================================================
    // Interface
    // ===================================================================
    EncodingType encodingType() const override {
        if constexpr (Prepass == FORPrepass::GlobalFOR)
            return EncodingType::BlockFrequencyPartitionFOREncoding;
        else
            return EncodingType::BlockFrequencyPartitionEncoding;
    }
    std::string name() const override {
        if constexpr (Prepass == FORPrepass::GlobalFOR)
            return "BlockFrequencyPartitionGlobalFOR";
        else
            return "BlockFrequencyPartition";
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
        return kFileHeaderSize + 32 * kBlockDescSize + elementCount * (sizeof(T) + 1u);
    }

private:
    // ===================================================================
    // Block descriptor — read on demand from the index; never stored persistently
    // ===================================================================
    struct BlockDesc {
        uint64_t payloadByteOffset;
        uint8_t  tagBitWidth;
        uint8_t  numTier0, numTier1, numTier2;
        uint32_t tier0Count, tier1Count, tier2Count;
        T        fallbackMin;
        uint8_t  fallbackBits; // kRawFallbackSentinel => raw sizeof(T) storage
    };

    static BlockDesc readDesc(const uint8_t* indexBase, uint32_t b) noexcept {
        const uint8_t* p = indexBase + static_cast<size_t>(b) * kBlockDescSize;
        BlockDesc d{};
        std::memcpy(&d.payloadByteOffset, p,      sizeof(uint64_t));
        d.tagBitWidth = p[8]; d.numTier0 = p[9]; d.numTier1 = p[10]; d.numTier2 = p[11];
        std::memcpy(&d.tier0Count, p + 12, sizeof(uint32_t));
        std::memcpy(&d.tier1Count, p + 16, sizeof(uint32_t));
        std::memcpy(&d.tier2Count, p + 20, sizeof(uint32_t));
        std::memcpy(&d.fallbackMin, p + 24, sizeof(T));
        d.fallbackBits = p[24 + sizeof(T)];
        return d;
    }

    // ===================================================================
    // Zero-copy view into the encoded buffer (pointer-comparison cache)
    // ===================================================================
    struct BlockView {
        size_t         N{0};
        uint32_t       blockSize{0}, numBlocks{0};
        const uint8_t* indexBase{nullptr};
        const uint8_t* payloadBase{nullptr};
        T              globalMin{};  // only populated for GlobalFOR prepass
    };
    struct Cache { const uint8_t* base{nullptr}; BlockView view{}; };
    mutable Cache cache_;

    // Encode-time-only setting (see constructor doc) — irrelevant to decode.
    bool bitPackFallback_ = false;

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
        v.blockSize = blockSize; v.numBlocks = numBlocks;
        if constexpr (Prepass == FORPrepass::GlobalFOR) {
            std::memcpy(&v.globalMin, base + 16, sizeof(T));
        }
        v.indexBase   = base + kFileHeaderSize;
        v.payloadBase = v.indexBase + static_cast<size_t>(numBlocks) * kBlockDescSize;
        cache_ = {base, v};
        return v;
    }

    // ===================================================================
    // Core decode — emit elements in [localStart, localEnd) from one block.
    //
    // Scans the compact tier-tag bitfield (1–2 bits/element, fits in L1 cache)
    // from element 0 to localEnd, writing output only for i ∈ [localStart, localEnd).
    // Running per-tier ranks are maintained so only a single forward pass is needed.
    //
    // globalMin is added back to each decoded value when Prepass == GlobalFOR;
    // for the None variant the parameter is T{} and the branch is eliminated.
    // ===================================================================
    static void decodeBlockRange(
        const BlockDesc& d,
        const uint8_t*   payload,
        size_t           blockElements,
        size_t           localStart,
        size_t           localEnd,
        T*               dst,
        T                globalMin) noexcept
    {
        const uint8_t* p = payload;
        const size_t   tagBytes  = (d.tagBitWidth > 0)
                                   ? (blockElements * d.tagBitWidth + 7) / 8 : 0;
        const uint8_t* tagBits   = p;  p += tagBytes;

        // Exact per-tier key widths (bits), recomputed identically to encode()
        // from the already-stored numTier0/1/2 — see detail::exactKeyBitWidth.
        const uint32_t kb0 = detail::exactKeyBitWidth(d.numTier0);
        const uint32_t kb1 = (d.numTier1 > 0) ? detail::exactKeyBitWidth(d.numTier1) : 0u;
        const uint32_t kb2 = (d.numTier2 > 0) ? detail::exactKeyBitWidth(d.numTier2) : 0u;

        // Fast path: tagBitWidth==0 → entire block is tier-0 (≤2 distinct values).
        // Disabled for GlobalFOR because the optimised helpers write values directly
        // without the globalMin restoration step.
        if (d.tagBitWidth == 0) {
            const uint8_t* t0Dict = p;
            const uint8_t* t0Keys = p + d.numTier0 * sizeof(T);
            if constexpr (Prepass == FORPrepass::None) {
                if (localStart == 0) {
                    detail::dispatchDecode<T>(t0Keys, t0Dict, dst, localEnd, kb0, 0);
                } else {
                    detail::decodeGeneral<T>(t0Keys, t0Dict, dst,
                                              localEnd - localStart, kb0, localStart);
                }
                return;
            }
            // GlobalFOR: fall through to the general path below so globalMin is added.
        }

        // Resolve tier pointers using stored element counts (no bitfield scan needed)
        const uint8_t* t0Dict = p;  p += d.numTier0 * sizeof(T);
        const uint8_t* t0Keys = p;  p += (d.tier0Count * kb0 + 7) / 8 + kKeyPaddingBytes;

        const uint8_t* t1Dict = nullptr, *t1Keys = nullptr;
        if (d.numTier1 > 0) {
            t1Dict = p;  p += d.numTier1 * sizeof(T);
            t1Keys = p;  p += (d.tier1Count * kb1 + 7) / 8 + kKeyPaddingBytes;
        }

        const uint8_t* t2Dict = nullptr, *t2Keys = nullptr;
        if (d.numTier2 > 0) {
            t2Dict = p;  p += d.numTier2 * sizeof(T);
            t2Keys = p;  p += (d.tier2Count * kb2 + 7) / 8 + kKeyPaddingBytes;
        }

        const uint8_t* fb = p;  // fallback raw values
        const uint8_t t2Code = (d.numTier1 > 0) ? 2u : 1u;
        const uint8_t tagMask = static_cast<uint8_t>((1u << d.tagBitWidth) - 1u);

        uint32_t ranks[4] = {};
        T* dstPtr = dst;

        for (size_t i = 0; i < localEnd; ++i) {
            const size_t  bitPos = i * d.tagBitWidth;
            const uint8_t code   = (d.tagBitWidth == 0)
                ? 0u
                : (tagBits[bitPos >> 3] >> (bitPos & 7u)) & tagMask;
            uint32_t& rank = ranks[code];

            T val;
            if (code == 0u) {
                const uint32_t key = extractTierKey(t0Keys, rank, kb0);
                val = detail::loadDictValue<T>(t0Dict, key);
            } else if (d.numTier1 > 0 && code == 1u) {
                const uint32_t key = extractTierKey(t1Keys, rank, kb1);
                val = detail::loadDictValue<T>(t1Dict, key);
            } else if (d.numTier2 > 0 && code == t2Code) {
                const uint32_t key = extractTierKey(t2Keys, rank, kb2);
                val = detail::loadDictValue<T>(t2Dict, key);
            } else if (d.fallbackBits == kRawFallbackSentinel) {
                val = detail::loadDictValue<T>(fb, rank);
            } else if (d.fallbackBits == 0) {
                val = d.fallbackMin; // every fallback element in this block equals fallbackMin
            } else {
                using U = std::make_unsigned_t<T>;
                const uint64_t delta = extractPackedBits(fb, rank, d.fallbackBits);
                val = static_cast<T>(static_cast<U>(d.fallbackMin) + static_cast<U>(delta));
            }

            rank++;
            if (i >= localStart) {
                if constexpr (Prepass == FORPrepass::GlobalFOR) {
                    using U = std::make_unsigned_t<T>;
                    val = static_cast<T>(static_cast<U>(val) + static_cast<U>(globalMin));
                }
                *dstPtr++ = val;
            }
        }
    }

    // ===================================================================
    // writeTierKeys — bit-pack the keys for one tier into the output buffer.
    // Filters elements by tagCode and packs keys at keyBitWidth bits each.
    // Appends kKeyPaddingBytes of zeros so callers can do safe 64-bit loads.
    // ===================================================================
    static void writeTierKeys(
        std::vector<uint8_t>&         out,
        const std::vector<ElemCode>&  elemCodes,
        size_t                        bCount,
        uint8_t                       filterCode,
        uint8_t                       keyBitWidth,
        std::vector<uint8_t>&         keyBuf)
    {
        keyBuf.clear();
        encodings::core::BitWriter wr(keyBuf, encodings::core::BitOrder::LSB);
        for (size_t i = 0; i < bCount; ++i) {
            if (elemCodes[i].tagCode == filterCode) {
                wr.write(static_cast<uint32_t>(elemCodes[i].tierKey), keyBitWidth);
            }
        }
        wr.flush();
        out.insert(out.end(), keyBuf.begin(), keyBuf.end());
        out.resize(out.size() + kKeyPaddingBytes, 0u);
    }

    // ===================================================================
    // appendBytes — append raw bytes into the output vector
    // ===================================================================
    template<typename U>
    static void appendBytes(std::vector<uint8_t>& out, const U* src, size_t bytes) {
        const size_t prevSize = out.size();
        out.resize(prevSize + bytes);
        std::memcpy(out.data() + prevSize, src, bytes);
    }

    // ===================================================================
    // Fallback-tier bit-packing — word-based (uint64_t), safe for 1-64 bit
    // widths (unlike core/BitPacker.hpp's BitWriter/BitReader, which cap at
    // 32 bits). Mirrors RawBitPackedEncoder.hpp's approach, scoped to just
    // the fallback tier's deltas-from-fallbackMin.
    // ===================================================================
    static void packFallbackBits(std::vector<uint8_t>& out, const std::vector<uint64_t>& deltas, uint8_t bits) {
        if (bits == 0 || deltas.empty()) return;
        const size_t n = deltas.size();
        const size_t bitsTotal = static_cast<size_t>(bits) * n;
        const size_t words = (bitsTotal + 63) / 64;
        std::vector<uint64_t> buf(words, 0u);
        const uint64_t mask = (bits == 64) ? ~0ull : ((1ull << bits) - 1ull);
        size_t bitPos = 0;
        for (uint64_t delta : deltas) {
            const size_t wordIdx = bitPos >> 6;
            const uint32_t off = static_cast<uint32_t>(bitPos & 63u);
            const uint64_t v = delta & mask;
            buf[wordIdx] |= v << off;
            if (off + bits > 64) buf[wordIdx + 1] |= v >> (64 - off);
            bitPos += bits;
        }
        const size_t prevSize = out.size();
        out.resize(prevSize + words * sizeof(uint64_t));
        std::memcpy(out.data() + prevSize, buf.data(), words * sizeof(uint64_t));
    }

    // Extract the `index`-th `bits`-wide value (unaligned 8-byte loads via
    // memcpy). General-purpose packed-field reader: used for the fallback
    // tier's deltas and, below, for tier keys whose width doesn't divide 8.
    static uint64_t extractPackedBits(const uint8_t* base, size_t index, uint8_t bits) {
        if (bits == 0) return 0;
        const size_t bitPos = index * bits;
        const size_t wordIdx = bitPos >> 6;
        const uint32_t off = static_cast<uint32_t>(bitPos & 63u);
        uint64_t w0;
        std::memcpy(&w0, base + wordIdx * sizeof(uint64_t), sizeof(uint64_t));
        uint64_t value = w0 >> off;
        if (off + bits > 64) {
            uint64_t w1;
            std::memcpy(&w1, base + (wordIdx + 1) * sizeof(uint64_t), sizeof(uint64_t));
            value |= w1 << (64 - off);
        }
        const uint64_t mask = (bits == 64) ? ~0ull : ((1ull << bits) - 1ull);
        return value & mask;
    }

    // Extract the `rank`-th tier key, `bits` wide. Widths that divide 8 (the
    // common case for fully-populated tiers: 0, 1, 2, 4) use the cheap
    // single-byte shift+mask that today's fixed-width code already used, so
    // that case decodes exactly as fast as before. Widths that don't divide 8
    // (3, 5, 6, 7 — only reachable via a partially-populated tier1/tier2) fall
    // back to the word-safe general extractor, since such a key can straddle
    // a byte boundary.
    static uint32_t extractTierKey(const uint8_t* keys, uint32_t rank, uint32_t bits) noexcept {
        switch (bits) {
            case 0: return 0u;
            case 1: return (keys[rank >> 3] >> (rank & 7u)) & 1u;
            case 2: { const size_t bp = static_cast<size_t>(rank) * 2;
                      return (keys[bp >> 3] >> (bp & 7u)) & 3u; }
            case 4: { const size_t bp = static_cast<size_t>(rank) * 4;
                      return (keys[bp >> 3] >> (bp & 7u)) & 15u; }
            default: return static_cast<uint32_t>(
                         extractPackedBits(keys, rank, static_cast<uint8_t>(bits)));
        }
    }

    // ===================================================================
    // planBlockSize — greedy sweep over candidates; picks minimum encoded cost
    // ===================================================================
    uint32_t planBlockSize(std::span<const T> data) const {
        const size_t N = data.size();
        uint32_t bestBs    = kCandidates[0];
        size_t   bestBytes = std::numeric_limits<size_t>::max();

        ankerl::unordered_dense::map<T, uint32_t> freq;
        std::vector<std::pair<uint32_t, T>> sortedFreq;

        for (const uint32_t bs : kCandidates) {
            const size_t numBlocks = (N + bs - 1) / bs;
            size_t payloadCost = 0;

            for (size_t b = 0; b < numBlocks; ++b) {
                const size_t bStart = b * bs;
                const size_t bEnd   = std::min(bStart + static_cast<size_t>(bs), N);
                const size_t bCount = bEnd - bStart;

                freq.clear();
                for (size_t i = bStart; i < bEnd; ++i) freq[data[i]]++;
                sortedFreq.clear();
                for (const auto& [v, c] : freq) sortedFreq.push_back({c, v});
                std::sort(sortedFreq.rbegin(), sortedFreq.rend());

                uint8_t  numTier0 = 0, numTier1 = 0, numTier2 = 0;
                uint32_t tier0Count = 0, tier1Count = 0, tier2Count = 0;
                size_t sfIdx = 0;
                while (sfIdx < sortedFreq.size() && numTier0 < kTier0Cap) {
                    tier0Count += sortedFreq[sfIdx].first;
                    numTier0++;  sfIdx++;
                }
                while (sfIdx < sortedFreq.size() && numTier1 < kTier1Cap) {
                    tier1Count += sortedFreq[sfIdx].first;
                    numTier1++;  sfIdx++;
                }
                while (sfIdx < sortedFreq.size() && numTier2 < kTier2Cap) {
                    tier2Count += sortedFreq[sfIdx].first;
                    numTier2++;  sfIdx++;
                }
                const bool hasFallback = (sfIdx < sortedFreq.size());
                const uint32_t fbCount = static_cast<uint32_t>(bCount)
                                       - tier0Count - tier1Count - tier2Count;

                const uint8_t numCodes =
                    static_cast<uint8_t>((numTier0 > 0 ? 1u : 0u)
                                       + (numTier1 > 0 ? 1u : 0u)
                                       + (numTier2 > 0 ? 1u : 0u)
                                       + (hasFallback  ? 1u : 0u));
                const uint8_t tagBW = (numCodes <= 1u) ? 0u : (numCodes == 2u) ? 1u : 2u;

                const uint32_t kb0 = detail::exactKeyBitWidth(numTier0);
                const uint32_t kb1 = numTier1 > 0 ? detail::exactKeyBitWidth(numTier1) : 0u;
                const uint32_t kb2 = numTier2 > 0 ? detail::exactKeyBitWidth(numTier2) : 0u;

                const size_t tagCost  = tagBW > 0 ? (bCount * tagBW + 7) / 8 : 0;
                const size_t t0Cost   = numTier0 * sizeof(T) + (tier0Count * kb0 + 7) / 8 + kKeyPaddingBytes;
                const size_t t1Cost   = numTier1 > 0
                    ? (numTier1 * sizeof(T) + (tier1Count * kb1 + 7) / 8 + kKeyPaddingBytes) : 0;
                const size_t t2Cost   = numTier2 > 0
                    ? (numTier2 * sizeof(T) + (tier2Count * kb2 + 7) / 8 + kKeyPaddingBytes) : 0;

                // Fallback cost: bit-packed-from-min (word-rounded, matching
                // packFallbackBits' actual output) when enabled, else raw sizeof(T).
                size_t fbCost = fbCount * sizeof(T);
                if (bitPackFallback_ && fbCount > 0) {
                    using U = std::make_unsigned_t<T>;
                    T fbMin = sortedFreq[sfIdx].second;
                    T fbMax = fbMin;
                    for (size_t k = sfIdx; k < sortedFreq.size(); ++k) {
                        fbMin = std::min(fbMin, sortedFreq[k].second);
                        fbMax = std::max(fbMax, sortedFreq[k].second);
                    }
                    const uint64_t span = static_cast<uint64_t>(static_cast<U>(fbMax) - static_cast<U>(fbMin));
                    const uint8_t fbBits = span == 0 ? 0u : static_cast<uint8_t>(64 - std::countl_zero(span));
                    fbCost = (static_cast<size_t>(fbBits) * fbCount + 63) / 64 * sizeof(uint64_t);
                }
                payloadCost += tagCost + t0Cost + t1Cost + t2Cost + fbCost;
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
