#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "core/BitPacker.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

/**
 * BlockFORFPEEncoder — block-local Frame-of-Reference + Frequency-Partition encoding.
 *
 * Each block:
 *   1. FOR stage: subtract block minimum (frameRef) to produce narrow unsigned residuals.
 *   2. FPE stage: sort unique residuals by descending frequency; greedily assign them to
 *      up to kMaxTiers tiers whose key widths are ceil(log2(tierDictSize)) — non-power-of-two.
 *      A tier-tag bitfield records each element's tier at its original position, preserving
 *      input order. A sampled rank table (one sample every kRankSampleStride elements)
 *      enables O(kRankSampleStride) random access per decodeAt() call instead of O(blockSize).
 *
 * Binary format:
 *   File header (16 bytes):
 *     [0..7]   N           uint64_t  element count
 *     [8..11]  blockSize   uint32_t  elements per block
 *     [12..15] numBlocks   uint32_t  ceil(N / blockSize)
 *
 *   Block index (numBlocks × kBlockDescBytes):
 *     [0..7]   payloadByteOffset  uint64_t
 *     [8..8+sizeof(TIn)-1]  frameRef  TIn  (block minimum)
 *     residualBits : uint8_t  ceil(log2(max_residual+1)), at least 1
 *     numTiers     : uint8_t  0..kMaxTiers
 *     tagBitWidth  : uint8_t  0, 1, or 2
 *     tierDictSizes[kMaxTiers] : uint8_t[]
 *     tierKeyBits[kMaxTiers]   : uint8_t[]  ceil(log2(tierDictSizes[k]))
 *     _pad to next 4-byte boundary
 *     tierElemCounts[kMaxTiers] : uint32_t[]
 *     numFallback               : uint32_t
 *
 *   Payload (per block):
 *     Rank sample table:  ceil(blockElems/kRankSampleStride) × numTiers × uint16_t
 *     Tag bitfield:       ceil(blockElems × tagBitWidth / 8) bytes + kPadBytes
 *     For each active tier k:
 *       Dict:  tierDictSizes[k] × residualBits bits  (bit-packed LSB, +kPadBytes boundary)
 *       Keys:  tierElemCounts[k] × tierKeyBits[k] bits (bit-packed LSB, +kPadBytes)
 *     Fallback: numFallback × residualBits bits (bit-packed LSB, +kPadBytes)
 */
template <typename TIn>
requires std::is_integral_v<TIn>
class BlockFORFPEEncoder : public Codec<TIn, uint8_t> {
public:
    static constexpr uint8_t  kMaxTiers          = 3;
    static constexpr uint32_t kRankSampleStride  = 64;   // rank table sample interval
    static constexpr size_t   kPadBytes          = 0;    // no alignment padding (readBits handles unaligned reads)
    static constexpr size_t   kFileHeaderBytes   = 16;
    // Floor lowered from 64 to 8: cascading-FOR experiments showed real, large
    // compressibility gains at block granularities as small as 16 for at least
    // some datasets, which the old {64,...} floor made this exact-cost planner
    // structurally blind to (it can only find what's in the candidate list).
    static constexpr uint32_t kCandidates[]      = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    // Fixed-size part of the block descriptor (before variable-size TIn frameRef field).
    // Layout after payloadByteOffset(8) + frameRef(sizeof(TIn)):
    //   residualBits(1) numTiers(1) tagBitWidth(1) tierDictSizes[3](3) tierKeyBits[3](3) pad(1)
    //   tierElemCounts[3](12) numFallback(4)
    static constexpr size_t kBlockDescBytes =
        8                   // payloadByteOffset
        + sizeof(TIn)       // frameRef
        + 1 + 1 + 1         // residualBits, numTiers, tagBitWidth
        + kMaxTiers         // tierDictSizes[kMaxTiers]
        + kMaxTiers         // tierKeyBits[kMaxTiers]
        + (((8 + sizeof(TIn) + 3 + kMaxTiers * 2) % 4 == 0) ? 0
            : 4 - ((8 + sizeof(TIn) + 3 + kMaxTiers * 2) % 4))  // pad to 4-byte
        + kMaxTiers * 4     // tierElemCounts[kMaxTiers]
        + 4;                // numFallback

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    EncodedBuffer<uint8_t> encode(std::span<const TIn> data) override {
        cache_.base = nullptr;
        if (data.empty()) return makeEmpty();

        const uint64_t N         = static_cast<uint64_t>(data.size());
        const uint32_t bestBs    = planBlockSize(data);
        const uint32_t numBlocks = static_cast<uint32_t>((N + bestBs - 1) / bestBs);

        EncodedBuffer<uint8_t> result;
        auto& out = result.data();
        out.reserve(kFileHeaderBytes
            + static_cast<size_t>(numBlocks) * kBlockDescBytes
            + static_cast<size_t>(N) * (sizeof(TIn) + 2));

        // File header
        out.resize(kFileHeaderBytes);
        std::memcpy(out.data(),      &N,         8);
        std::memcpy(out.data() + 8,  &bestBs,    4);
        std::memcpy(out.data() + 12, &numBlocks, 4);

        // Pre-allocate block index
        const size_t indexOffset = out.size();
        out.resize(indexOffset + static_cast<size_t>(numBlocks) * kBlockDescBytes, 0u);
        const size_t payloadStart = out.size();

        // Per-block temporaries
        ankerl::unordered_dense::map<uint64_t, uint32_t> freq;
        std::vector<std::pair<uint32_t, uint64_t>> sortedFreq;

        // Aggregate tier stats across blocks, only accumulated when statsEnabled()
        // so this loop's hot path is unaffected when disabled.
        const bool collectStats = statsEnabled();
        uint64_t statsTotalTiers = 0;
        uint64_t statsTotalTagBitWidth = 0;
        double   statsTotalFallbackFraction = 0.0;

        for (uint32_t b = 0; b < numBlocks; ++b) {
            const size_t bStart  = static_cast<size_t>(b) * bestBs;
            const size_t bEnd    = std::min(bStart + static_cast<size_t>(bestBs),
                                            static_cast<size_t>(N));
            const size_t bCount  = bEnd - bStart;

            // --- FOR: compute frameRef and residuals ---
            TIn frameRef = data[bStart];
            for (size_t i = bStart + 1; i < bEnd; ++i)
                if (data[i] < frameRef) frameRef = data[i];

            const uint64_t maxResidual = static_cast<uint64_t>(
                *std::max_element(data.begin() + bStart, data.begin() + bEnd) - frameRef);
            const uint8_t residualBits = static_cast<uint8_t>(
                maxResidual == 0 ? 1u : std::bit_width(maxResidual));

            // --- Build frequency map of residuals ---
            freq.clear();
            for (size_t i = bStart; i < bEnd; ++i)
                freq[static_cast<uint64_t>(data[i] - frameRef)]++;

            sortedFreq.clear();
            sortedFreq.reserve(freq.size());
            for (auto& [v, c] : freq) sortedFreq.push_back({c, v});
            std::sort(sortedFreq.rbegin(), sortedFreq.rend());

            // --- Plan tiers ---
            TierPlan plan = planTiers(sortedFreq, bCount, residualBits);

            // --- Build element codes: (tagCode, key) for each element ---
            // valToCode maps residual → (tagCode, keyIndex)
            ankerl::unordered_dense::map<uint64_t, std::pair<uint8_t,uint8_t>> valToCode;
            valToCode.reserve(plan.totalDictSize());
            {
                size_t sfIdx = 0;
                for (uint8_t t = 0; t < plan.numTiers; ++t) {
                    for (uint8_t k = 0; k < plan.tierDictSizes[t]; ++k, ++sfIdx)
                        valToCode[sortedFreq[sfIdx].second] = {t, k};
                }
            }

            struct ElemCode { uint8_t tagCode; uint8_t key; };
            std::vector<ElemCode> elemCodes(bCount);
            uint32_t tierCounts[kMaxTiers] = {};
            uint32_t fallbackCount = 0;
            for (size_t i = 0; i < bCount; ++i) {
                uint64_t r = static_cast<uint64_t>(data[bStart + i] - frameRef);
                auto it = valToCode.find(r);
                if (it != valToCode.end()) {
                    elemCodes[i] = {it->second.first, it->second.second};
                    tierCounts[it->second.first]++;
                } else {
                    elemCodes[i] = {plan.numTiers, 0}; // fallback tag code
                    fallbackCount++;
                }
            }

            if (collectStats) {
                statsTotalTiers += plan.numTiers;
                statsTotalTagBitWidth += plan.tagBitWidth;
                statsTotalFallbackFraction +=
                    bCount ? (static_cast<double>(fallbackCount) / static_cast<double>(bCount)) : 0.0;
            }

            // --- Build rank sample table ---
            // rankSamples[i][t] = cumulative count of tier-t elements before position i*S
            const uint32_t numSamples = static_cast<uint32_t>(
                (bCount + kRankSampleStride - 1) / kRankSampleStride);
            // Store as numTiers × uint16_t per sample, interleaved: [t0,t1,t2, t0,t1,t2, ...]
            std::vector<uint16_t> rankSamples;
            if (plan.numTiers > 0) {
                rankSamples.resize(static_cast<size_t>(numSamples) * plan.numTiers, 0u);
                uint32_t running[kMaxTiers] = {};
                for (uint32_t s = 0; s < numSamples; ++s) {
                    for (uint8_t t = 0; t < plan.numTiers; ++t)
                        rankSamples[static_cast<size_t>(s) * plan.numTiers + t] =
                            static_cast<uint16_t>(running[t]);
                    const size_t end = std::min(
                        static_cast<size_t>((s + 1) * kRankSampleStride), bCount);
                    for (size_t i = static_cast<size_t>(s) * kRankSampleStride; i < end; ++i)
                        if (elemCodes[i].tagCode < plan.numTiers)
                            running[elemCodes[i].tagCode]++;
                }
            }

            // --- Write block descriptor ---
            const uint64_t payloadByteOffset =
                static_cast<uint64_t>(out.size() - payloadStart);
            writeDesc(out.data() + indexOffset
                      + static_cast<size_t>(b) * kBlockDescBytes,
                      payloadByteOffset, frameRef, residualBits, plan,
                      tierCounts, fallbackCount);

            // --- Write rank sample table ---
            const size_t rankBytes = rankSamples.size() * sizeof(uint16_t);
            const size_t rankBase = out.size();
            out.resize(rankBase + rankBytes);
            if (rankBytes > 0)
                std::memcpy(out.data() + rankBase, rankSamples.data(), rankBytes);

            // --- Write tag bitfield ---
            if (plan.tagBitWidth > 0) {
                const size_t tagBytes = (bCount * plan.tagBitWidth + 7) / 8;
                const size_t tagBase  = out.size();
                out.resize(tagBase + tagBytes, 0u);
                uint8_t* tagBuf = out.data() + tagBase;
                for (size_t i = 0; i < bCount; ++i) {
                    const size_t bitPos = i * plan.tagBitWidth;
                    tagBuf[bitPos >> 3] |=
                        static_cast<uint8_t>(elemCodes[i].tagCode << (bitPos & 7u));
                }
                out.resize(out.size() + kPadBytes, 0u);
            }

            // --- Write tier dicts + keys ---
            {
                size_t sfIdx = 0;
                std::vector<uint8_t> tmpBits;
                for (uint8_t t = 0; t < plan.numTiers; ++t) {
                    const uint8_t dSize = plan.tierDictSizes[t];
                    const uint8_t kBits = plan.tierKeyBits[t];

                    // Dict: dSize entries × residualBits each
                    tmpBits.clear();
                    {
                        core::BitWriter dw(tmpBits, core::BitOrder::LSB);
                        for (uint8_t k = 0; k < dSize; ++k, ++sfIdx)
                            writeResidual(dw, sortedFreq[sfIdx].second, residualBits);
                        dw.flush();
                    }
                    out.insert(out.end(), tmpBits.begin(), tmpBits.end());
                    out.resize(out.size() + kPadBytes, 0u);

                    // Keys: tierCounts[t] entries × kBits each
                    tmpBits.clear();
                    if (kBits > 0) {
                        core::BitWriter kw(tmpBits, core::BitOrder::LSB);
                        for (size_t i = 0; i < bCount; ++i)
                            if (elemCodes[i].tagCode == t)
                                kw.write(elemCodes[i].key, kBits);
                        kw.flush();
                    }
                    out.insert(out.end(), tmpBits.begin(), tmpBits.end());
                    out.resize(out.size() + kPadBytes, 0u);
                }
            }

            // --- Write fallback: residuals at residualBits each ---
            if (fallbackCount > 0) {
                std::vector<uint8_t> fbBits;
                core::BitWriter fw(fbBits, core::BitOrder::LSB);
                for (size_t i = 0; i < bCount; ++i) {
                    if (elemCodes[i].tagCode == plan.numTiers) {
                        uint64_t r = static_cast<uint64_t>(data[bStart + i] - frameRef);
                        writeResidual(fw, r, residualBits);
                    }
                }
                fw.flush();
                out.insert(out.end(), fbBits.begin(), fbBits.end());
                out.resize(out.size() + kPadBytes, 0u);
            }
        }

        result.metadata().encodingName         = name();
        result.metadata().dataType             = this->dataType();
        result.metadata().elementCount         = static_cast<size_t>(N);
        result.metadata().compressedSize       = out.size();
        result.metadata().uncompressedSize     = static_cast<size_t>(N) * sizeof(TIn);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["block_size"] = std::to_string(bestBs);
        result.metadata().customMetadata["num_blocks"] = std::to_string(numBlocks);
        if (collectStats && numBlocks > 0) {
            const double avgNumTiers = static_cast<double>(statsTotalTiers) / static_cast<double>(numBlocks);
            const double avgTagBitWidth = static_cast<double>(statsTotalTagBitWidth) / static_cast<double>(numBlocks);
            const double avgFallbackFraction = statsTotalFallbackFraction / static_cast<double>(numBlocks);
            result.metadata().customMetadata["avg_num_tiers"] = std::to_string(avgNumTiers);
            result.metadata().customMetadata["avg_tag_bit_width"] = std::to_string(avgTagBitWidth);
            result.metadata().customMetadata["avg_fallback_fraction"] = std::to_string(avgFallbackFraction);
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // decodeAll
    // -----------------------------------------------------------------------
    std::vector<TIn> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        const BlockView v = getView(encoded);
        if (v.N == 0) return {};
        std::vector<TIn> result(v.N);
        for (uint32_t b = 0; b < v.numBlocks; ++b) {
            const BlockDesc d    = readDesc(v.indexBase, b);
            const size_t bStart  = static_cast<size_t>(b) * v.blockSize;
            const size_t bElems  = std::min(static_cast<size_t>(v.blockSize), v.N - bStart);
            decodeBlockRange(d, v.payloadBase + d.payloadByteOffset,
                             bElems, 0, bElems, result.data() + bStart);
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // decodeAt — O(kRankSampleStride) via sampled rank table
    // -----------------------------------------------------------------------
    std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || index >= v.N) return std::nullopt;
        const uint32_t b       = static_cast<uint32_t>(index / v.blockSize);
        const size_t localIdx  = index % v.blockSize;
        const BlockDesc d      = readDesc(v.indexBase, b);
        const size_t bStart    = static_cast<size_t>(b) * v.blockSize;
        const size_t bElems    = std::min(static_cast<size_t>(v.blockSize), v.N - bStart);
        TIn val{};
        decodeBlockRange(d, v.payloadBase + d.payloadByteOffset,
                         bElems, localIdx, localIdx + 1, &val);
        return val;
    }

    // -----------------------------------------------------------------------
    // decodeRange
    // -----------------------------------------------------------------------
    std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& encoded,
                                  size_t start, size_t end) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || start >= v.N) return {};
        end = std::min(end, v.N);
        if (start >= end) return {};
        std::vector<TIn> result(end - start);
        decodeRangeInto(encoded, start, end, result.data(), result.size());
        return result;
    }

    void decodeAllInto(const EncodedBuffer<uint8_t>& encoded,
                        TIn* dst, size_t n) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || n == 0) return;
        const size_t count = std::min(n, v.N);
        for (uint32_t b = 0; b < v.numBlocks; ++b) {
            const BlockDesc d   = readDesc(v.indexBase, b);
            const size_t bStart = static_cast<size_t>(b) * v.blockSize;
            if (bStart >= count) break;
            const size_t bElems = std::min(static_cast<size_t>(v.blockSize), count - bStart);
            decodeBlockRange(d, v.payloadBase + d.payloadByteOffset,
                             bElems, 0, bElems, dst + bStart);
        }
    }

    void decodeRangeInto(const EncodedBuffer<uint8_t>& encoded,
                          size_t start, size_t end,
                          TIn* dst, size_t n) override {
        const BlockView v = getView(encoded);
        if (v.N == 0 || start >= v.N || n == 0) return;
        end = std::min(end, v.N);
        if (start >= end) return;

        const uint32_t firstBlock = static_cast<uint32_t>(start / v.blockSize);
        const uint32_t lastBlock  = static_cast<uint32_t>((end - 1) / v.blockSize);
        size_t written = 0;

        for (uint32_t b = firstBlock; b <= lastBlock && written < n; ++b) {
            const BlockDesc d       = readDesc(v.indexBase, b);
            const size_t bElemStart = static_cast<size_t>(b) * v.blockSize;
            const size_t bElemEnd   = std::min(bElemStart + static_cast<size_t>(v.blockSize), v.N);
            const size_t bElems     = bElemEnd - bElemStart;
            const size_t localStart = (b == firstBlock) ? (start - bElemStart) : 0u;
            const size_t localEnd   = (b == lastBlock)  ? (end   - bElemStart) : bElems;
            const size_t localCount = std::min(localEnd - localStart, n - written);
            decodeBlockRange(d, v.payloadBase + d.payloadByteOffset,
                             bElems, localStart, localStart + localCount, dst + written);
            written += localCount;
        }
    }

    // -----------------------------------------------------------------------
    // Interface
    // -----------------------------------------------------------------------
    EncodingType encodingType() const override {
        return EncodingType::BlockFORFPEEncoding;
    }
    std::string name() const override { return "BlockFORFPE"; }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::RandomAccess)
             | EncodingProperty::Lossless
             | EncodingProperty::PreservesOrder
             | EncodingProperty::DictionaryBased
             | EncodingProperty::BitPackingBased
             | EncodingProperty::RequiresFullData
             | EncodingProperty::VariableSize
             | EncodingProperty::HighMemoryOverhead
             | EncodingProperty::Composable;
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        return kFileHeaderBytes
             + 32 * kBlockDescBytes
             + elementCount * (sizeof(TIn) + 2u);
    }

    void reset() override { cache_.base = nullptr; }

private:
    // When set (BLOCKFORFPE_STATS env var), encode() accumulates per-block tier
    // stats (avg tier count, avg tag bit width, avg fallback fraction) and
    // exposes them via EncodedBuffer::metadata().customMetadata. Zero-cost when
    // unset; never affects the wire format (out/.data()). Mirrors the
    // getenv-gated static-bool pattern in FrequencyPartitionEncoder.hpp's
    // verboseEnabled().
    static bool statsEnabled() {
        static bool v = (std::getenv("BLOCKFORFPE_STATS") != nullptr);
        return v;
    }

    // -----------------------------------------------------------------------
    // TierPlan — result of planTiers()
    // -----------------------------------------------------------------------
    struct TierPlan {
        uint8_t numTiers                   = 0;
        uint8_t tagBitWidth                = 0;
        uint8_t tierDictSizes[kMaxTiers]   = {};
        uint8_t tierKeyBits[kMaxTiers]     = {};

        size_t totalDictSize() const {
            size_t s = 0;
            for (uint8_t t = 0; t < numTiers; ++t) s += tierDictSizes[t];
            return s;
        }
    };

    // -----------------------------------------------------------------------
    // planTiers — greedy sweep over sorted frequency list
    //
    // For each candidate dictionary size K (1..uniqueCount), estimate the cost
    // of using K values in the current tier vs. treating them as fallback.
    // Greedily add tiers while cost decreases.
    // -----------------------------------------------------------------------
    static TierPlan planTiers(
        const std::vector<std::pair<uint32_t, uint64_t>>& sortedFreq,
        size_t N,
        uint8_t residualBits)
    {
        if (sortedFreq.empty() || N == 0) return {};

        const size_t uniqueCount = sortedFreq.size();

        // Precompute cumulative coverage (number of elements covered by top-K values)
        std::vector<size_t> cumCoverage(uniqueCount + 1, 0);
        for (size_t k = 0; k < uniqueCount; ++k)
            cumCoverage[k + 1] = cumCoverage[k] + sortedFreq[k].first;

        // Cost of encoding all remaining elements as fallback at full residualBits
        auto fallbackCostBits = [&](size_t covered) -> double {
            return static_cast<double>(N - covered) * residualBits;
        };

        // Cost of a tier: elements in [startIdx, startIdx+dictSize) now pay kBits instead
        // of residualBits, plus dict overhead of dictSize × residualBits.
        auto tierCost2 = [&](size_t startIdx, size_t dictSize) -> double {
            if (dictSize == 0) return 0.0;
            const uint8_t kBits = (dictSize <= 1) ? 0u
                : static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(dictSize) - 1u));
            const size_t elems = cumCoverage[startIdx + dictSize] - cumCoverage[startIdx];
            return static_cast<double>(elems) * kBits
                 + static_cast<double>(dictSize) * residualBits;
        };

        // Tag overhead for T tiers + fallback (T+1 codes → ceil(log2(T+1)) bits)
        auto tagBitsForTiers = [](uint8_t numT, bool hasFallback) -> uint8_t {
            const uint32_t codes = static_cast<uint32_t>(numT) + (hasFallback ? 1u : 0u);
            if (codes <= 1u) return 0u;
            if (codes == 2u) return 1u;
            return 2u;
        };

        TierPlan best{};
        // baseline: all fallback, no tiers
        double bestCost = fallbackCostBits(0)
                        + static_cast<double>(N) * tagBitsForTiers(0, true);

        // Try adding up to kMaxTiers tiers
        size_t startIdx = 0; // index into sortedFreq where next tier begins
        TierPlan current{};
        for (uint8_t t = 0; t < kMaxTiers && startIdx < uniqueCount; ++t) {
            // Find best dictSize for this tier
            double bestTierGain = 0.0;
            size_t bestK = 0;
            const size_t maxK = std::min(uniqueCount - startIdx, size_t(255));

            // Count current tag bits
            const uint8_t curTagBits   = tagBitsForTiers(current.numTiers, true);
            const uint8_t newTagBits   = tagBitsForTiers(static_cast<uint8_t>(current.numTiers + 1), true);
            const double tagDeltaPerElem = static_cast<double>(newTagBits) - curTagBits;
            const double tagOverhead   = static_cast<double>(N) * tagDeltaPerElem;

            for (size_t k = 1; k <= maxK; ++k) {
                // Gain: elements in this tier now pay kBits instead of residualBits
                const uint8_t kBits = (k <= 1u) ? 0u
                    : static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(k) - 1u));
                const size_t  elems = cumCoverage[startIdx + k] - cumCoverage[startIdx];
                const double  dictSz = static_cast<double>(k) * residualBits;
                const double  gain = static_cast<double>(elems) * (residualBits - kBits) - dictSz;

                if (gain - tagOverhead > bestTierGain) {
                    bestTierGain = gain - tagOverhead;
                    bestK = k;
                }
            }

            if (bestK == 0) break; // no benefit

            const uint8_t kBits = (bestK <= 1u) ? 0u
                : static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(bestK) - 1u));
            current.tierDictSizes[t] = static_cast<uint8_t>(bestK);
            current.tierKeyBits[t]   = kBits;
            current.numTiers         = static_cast<uint8_t>(t + 1);
            startIdx += bestK;

            const bool hasFallback = (startIdx < uniqueCount);
            current.tagBitWidth = tagBitsForTiers(current.numTiers, hasFallback);

            // Compute total cost of current plan
            double cost = static_cast<double>(N) * current.tagBitWidth; // tag
            size_t si = 0;
            for (uint8_t tt = 0; tt < current.numTiers; ++tt) {
                cost += tierCost2(si, current.tierDictSizes[tt]);
                si += current.tierDictSizes[tt];
            }
            cost += fallbackCostBits(cumCoverage[si]);

            if (cost < bestCost) {
                bestCost = cost;
                best = current;
                // Update tagBitWidth in best for the case hasFallback=false
                best.tagBitWidth = tagBitsForTiers(best.numTiers, startIdx < uniqueCount);
            }
        }

        return best;
    }

    // -----------------------------------------------------------------------
    // planBlockSize — sweep candidates, pick minimum estimated bytes
    // -----------------------------------------------------------------------
    uint32_t planBlockSize(std::span<const TIn> data) const {
        const size_t N = data.size();
        if (N == 0) return kCandidates[0];

        uint32_t bestBs   = kCandidates[0];
        double   bestCost = std::numeric_limits<double>::infinity();

        ankerl::unordered_dense::map<uint64_t, uint32_t> freq;
        std::vector<std::pair<uint32_t, uint64_t>> sortedFreq;

        for (uint32_t bs : kCandidates) {
            const uint32_t numBlocks =
                static_cast<uint32_t>((N + bs - 1) / bs);
            double totalCost =
                static_cast<double>(kFileHeaderBytes + numBlocks * kBlockDescBytes) * 8.0;

            for (uint32_t b = 0; b < numBlocks; ++b) {
                const size_t bStart = static_cast<size_t>(b) * bs;
                const size_t bEnd   = std::min(bStart + static_cast<size_t>(bs), N);
                const size_t bCount = bEnd - bStart;

                TIn frameRef = data[bStart];
                uint64_t maxRes = 0;
                for (size_t i = bStart; i < bEnd; ++i) {
                    if (data[i] < frameRef) frameRef = data[i];
                }
                for (size_t i = bStart; i < bEnd; ++i) {
                    uint64_t r = static_cast<uint64_t>(data[i] - frameRef);
                    if (r > maxRes) maxRes = r;
                }

                const uint8_t rBits = maxRes == 0 ? 1u
                    : static_cast<uint8_t>(std::bit_width(maxRes));

                freq.clear();
                for (size_t i = bStart; i < bEnd; ++i)
                    freq[static_cast<uint64_t>(data[i] - frameRef)]++;

                sortedFreq.clear();
                for (auto& [v,c] : freq) sortedFreq.push_back({c, v});
                std::sort(sortedFreq.rbegin(), sortedFreq.rend());

                const TierPlan plan = planTiers(sortedFreq, bCount, rBits);

                // Tag bitfield + pad
                const double tagBytes = (plan.tagBitWidth > 0)
                    ? static_cast<double>((bCount * plan.tagBitWidth + 7) / 8 + kPadBytes)
                    : 0.0;

                // Rank sample table
                const double rankBytes = static_cast<double>(
                    ((bCount + kRankSampleStride - 1) / kRankSampleStride)
                    * plan.numTiers * sizeof(uint16_t));

                // Tier dicts + keys
                double tierBytes = 0.0;
                {
                    size_t si = 0;
                    for (uint8_t t = 0; t < plan.numTiers; ++t) {
                        const size_t dSz = plan.tierDictSizes[t];
                        const uint8_t kb = plan.tierKeyBits[t];
                        // dict
                        tierBytes += static_cast<double>(
                            (dSz * rBits + 7) / 8 + kPadBytes);
                        // keys
                        size_t elems = 0;
                        for (size_t k = si; k < si + dSz; ++k)
                            elems += sortedFreq[k].first;
                        tierBytes += static_cast<double>(
                            (elems * kb + 7) / 8 + kPadBytes);
                        si += dSz;
                    }
                }

                // Fallback
                size_t covered = 0;
                for (uint8_t t = 0; t < plan.numTiers; ++t) covered += plan.tierDictSizes[t];
                size_t fbElems = bCount;
                for (size_t k = 0; k < covered && k < sortedFreq.size(); ++k)
                    fbElems -= sortedFreq[k].first;
                const double fbBytes = static_cast<double>(
                    (fbElems * rBits + 7) / 8 + (fbElems > 0 ? kPadBytes : 0));

                // FOR ref (in descriptor, already counted above) + block cost
                totalCost += (tagBytes + rankBytes + tierBytes + fbBytes) * 8.0;
            }

            if (totalCost < bestCost) {
                bestCost = totalCost;
                bestBs   = bs;
            }
        }

        return bestBs;
    }

    // -----------------------------------------------------------------------
    // BlockDesc — runtime view of one block's descriptor
    // -----------------------------------------------------------------------
    struct BlockDesc {
        uint64_t payloadByteOffset;
        TIn      frameRef;
        uint8_t  residualBits;
        uint8_t  numTiers;
        uint8_t  tagBitWidth;
        uint8_t  tierDictSizes[kMaxTiers];
        uint8_t  tierKeyBits[kMaxTiers];
        uint32_t tierElemCounts[kMaxTiers];
        uint32_t numFallback;
    };

    // Write a descriptor into a 96-byte-max block at `p`
    static void writeDesc(uint8_t* p,
                          uint64_t payloadByteOffset,
                          TIn frameRef,
                          uint8_t residualBits,
                          const TierPlan& plan,
                          const uint32_t tierCounts[kMaxTiers],
                          uint32_t numFallback)
    {
        std::memcpy(p, &payloadByteOffset, 8); p += 8;
        std::memcpy(p, &frameRef, sizeof(TIn)); p += sizeof(TIn);
        *p++ = residualBits;
        *p++ = plan.numTiers;
        *p++ = plan.tagBitWidth;
        for (uint8_t t = 0; t < kMaxTiers; ++t) *p++ = plan.tierDictSizes[t];
        for (uint8_t t = 0; t < kMaxTiers; ++t) *p++ = plan.tierKeyBits[t];
        // Compute padding to 4-byte boundary
        const size_t written = 8 + sizeof(TIn) + 3 + kMaxTiers * 2;
        const size_t pad = (4 - (written % 4)) % 4;
        for (size_t i = 0; i < pad; ++i) *p++ = 0;
        for (uint8_t t = 0; t < kMaxTiers; ++t) {
            std::memcpy(p, &tierCounts[t], 4); p += 4;
        }
        std::memcpy(p, &numFallback, 4);
    }

    static BlockDesc readDesc(const uint8_t* indexBase, uint32_t b) noexcept {
        const uint8_t* p = indexBase + static_cast<size_t>(b) * kBlockDescBytes;
        BlockDesc d{};
        std::memcpy(&d.payloadByteOffset, p, 8); p += 8;
        std::memcpy(&d.frameRef, p, sizeof(TIn)); p += sizeof(TIn);
        d.residualBits = *p++;
        d.numTiers     = *p++;
        d.tagBitWidth  = *p++;
        for (uint8_t t = 0; t < kMaxTiers; ++t) d.tierDictSizes[t] = *p++;
        for (uint8_t t = 0; t < kMaxTiers; ++t) d.tierKeyBits[t]   = *p++;
        const size_t written = 8 + sizeof(TIn) + 3 + kMaxTiers * 2;
        const size_t pad = (4 - (written % 4)) % 4;
        p += pad;
        for (uint8_t t = 0; t < kMaxTiers; ++t) {
            std::memcpy(&d.tierElemCounts[t], p, 4); p += 4;
        }
        std::memcpy(&d.numFallback, p, 4);
        return d;
    }

    // -----------------------------------------------------------------------
    // Zero-copy view into encoded buffer (pointer cache)
    // -----------------------------------------------------------------------
    struct BlockView {
        size_t         N{0};
        uint32_t       blockSize{0}, numBlocks{0};
        const uint8_t* indexBase{nullptr};
        const uint8_t* payloadBase{nullptr};
    };
    struct Cache { const uint8_t* base{nullptr}; BlockView view{}; };
    mutable Cache cache_;

    BlockView getView(const EncodedBuffer<uint8_t>& encoded) const {
        const uint8_t* base = encoded.data().data();
        if (base == cache_.base) [[likely]] return cache_.view;
        if (encoded.size() < kFileHeaderBytes) { cache_ = {base, {}}; return {}; }
        uint64_t N;        std::memcpy(&N,        base,     8);
        uint32_t blockSize; std::memcpy(&blockSize, base + 8, 4);
        uint32_t numBlocks; std::memcpy(&numBlocks, base + 12, 4);
        if (N == 0 || blockSize == 0 || numBlocks == 0) { cache_ = {base, {}}; return {}; }
        BlockView v;
        v.N         = static_cast<size_t>(N);
        v.blockSize = blockSize;
        v.numBlocks = numBlocks;
        v.indexBase   = base + kFileHeaderBytes;
        v.payloadBase = v.indexBase + static_cast<size_t>(numBlocks) * kBlockDescBytes;
        cache_ = {base, v};
        return v;
    }

    // -----------------------------------------------------------------------
    // decodeBlockRange — core decoder
    //
    // Decodes elements [localStart, localEnd) within a single block.
    // Uses the sampled rank table to start scanning near the target position,
    // giving O(kRankSampleStride) random access.
    // -----------------------------------------------------------------------
    static void decodeBlockRange(
        const BlockDesc& d,
        const uint8_t*   payload,
        size_t           blockElements,
        size_t           localStart,
        size_t           localEnd,
        TIn*             dst) noexcept
    {
        const uint8_t* p = payload;

        // --- Rank sample table ---
        const uint32_t numSamples = static_cast<uint32_t>(
            (blockElements + kRankSampleStride - 1) / kRankSampleStride);
        const size_t rankTableBytes =
            static_cast<size_t>(numSamples) * d.numTiers * sizeof(uint16_t);
        const uint16_t* rankTable = reinterpret_cast<const uint16_t*>(p);
        p += rankTableBytes;

        // --- Tag bitfield ---
        const size_t tagBytes = (d.tagBitWidth > 0)
            ? (blockElements * d.tagBitWidth + 7) / 8 : 0;
        const uint8_t* tagBuf = p;
        p += tagBytes + (d.tagBitWidth > 0 ? kPadBytes : 0);

        // --- Tier section pointers ---
        // Layout: for each tier: dict (residualBits per entry, +pad), keys (+pad)
        struct TierPtrs { const uint8_t* dict; size_t dictBitLen; const uint8_t* keys; };
        TierPtrs tp[kMaxTiers]{};
        for (uint8_t t = 0; t < d.numTiers; ++t) {
            const size_t dictBits = static_cast<size_t>(d.tierDictSizes[t]) * d.residualBits;
            const size_t dictBytes = (dictBits + 7) / 8;
            tp[t].dict        = p;
            tp[t].dictBitLen  = dictBits;
            p += dictBytes + kPadBytes;
            tp[t].keys = p;
            const size_t keyBits = static_cast<size_t>(d.tierElemCounts[t]) * d.tierKeyBits[t];
            p += (keyBits + 7) / 8 + kPadBytes;
        }
        const uint8_t* fbPtr = p; // fallback section

        // Fast path: no tiers (all fallback) or no elements to decode
        if (localStart >= localEnd) return;

        // --- Determine scan start from rank table ---
        const size_t scanStart = (localStart > 0 && d.numTiers > 0)
            ? (localStart / kRankSampleStride) * kRankSampleStride
            : 0u;
        const uint32_t sampleIdx = static_cast<uint32_t>(scanStart / kRankSampleStride);

        // Load initial ranks from sample
        uint32_t ranks[kMaxTiers] = {};
        uint32_t fbRank = 0;
        if (d.numTiers > 0 && sampleIdx > 0 && sampleIdx <= numSamples) {
            const uint16_t* sample = rankTable + static_cast<size_t>(sampleIdx) * d.numTiers;
            for (uint8_t t = 0; t < d.numTiers; ++t)
                ranks[t] = sample[t];
            // Compute fallback rank from total elements before scanStart minus tier elements
            size_t totalTiersBefore = 0;
            for (uint8_t t = 0; t < d.numTiers; ++t) totalTiersBefore += ranks[t];
            fbRank = static_cast<uint32_t>(scanStart) - static_cast<uint32_t>(totalTiersBefore);
        }

        const uint8_t tagMask = static_cast<uint8_t>((1u << d.tagBitWidth) - 1u);
        TIn* dstPtr = dst;

        for (size_t i = scanStart; i < localEnd; ++i) {
            // Extract tag code
            uint8_t code;
            if (d.tagBitWidth == 0) {
                code = 0;
            } else {
                const size_t bitPos = i * d.tagBitWidth;
                code = (tagBuf[bitPos >> 3] >> (bitPos & 7u)) & tagMask;
            }

            if (i >= localStart) {
                // Decode element
                uint64_t residual = 0;
                if (code < d.numTiers) {
                    const uint8_t t  = code;
                    const uint8_t kb = d.tierKeyBits[t];
                    uint32_t key = 0;
                    if (kb > 0) {
                        const size_t bitOff = static_cast<size_t>(ranks[t]) * kb;
                        key = static_cast<uint32_t>(readBits(tp[t].keys, bitOff, kb));
                    }
                    // Read dict entry at key position
                    const size_t dBitOff = static_cast<size_t>(key) * d.residualBits;
                    residual = readBits(tp[t].dict, dBitOff, d.residualBits);
                } else {
                    // Fallback: read residual at fbRank * residualBits
                    const size_t fBitOff = static_cast<size_t>(fbRank) * d.residualBits;
                    residual = readBits(fbPtr, fBitOff, d.residualBits);
                }
                *dstPtr++ = d.frameRef + static_cast<TIn>(residual);
            }

            // Advance rank
            if (code < d.numTiers) {
                ranks[code]++;
            } else {
                fbRank++;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Read `numBits` bits from a byte array starting at absolute `bitOffset`.
    // Works for numBits 1..64 regardless of alignment.  The byte array must
    // have at least ceil((bitOffset + numBits) / 8) bytes accessible.
    static uint64_t readBits(const uint8_t* data, size_t bitOffset, uint32_t numBits) noexcept {
        data += bitOffset >> 3;
        const uint32_t shift     = static_cast<uint32_t>(bitOffset & 7u);
        const uint32_t byteCount = (numBits + shift + 7u) / 8u;
        uint64_t acc = 0;
        for (uint32_t b = 0; b < byteCount; ++b)
            acc |= static_cast<uint64_t>(data[b]) << (b * 8u);
        return (numBits >= 64u)
            ? (acc >> shift)
            : ((acc >> shift) & ((1ull << numBits) - 1ull));
    }

    // Write a residual value of up to 64 bits via two 32-bit writes if needed.
    static void writeResidual(core::BitWriter& w, uint64_t val, uint8_t bits) {
        if (bits <= 32u) {
            w.write(static_cast<uint32_t>(val), bits);
        } else {
            w.write(static_cast<uint32_t>(val & 0xFFFFFFFFu), 32u);
            w.write(static_cast<uint32_t>(val >> 32), static_cast<uint32_t>(bits) - 32u);
        }
    }

    static EncodedBuffer<uint8_t> makeEmpty() {
        EncodedBuffer<uint8_t> r;
        r.metadata().encodingName         = "BlockFORFPE";
        r.metadata().elementCount         = 0;
        r.metadata().compressedSize       = 0;
        r.metadata().uncompressedSize     = 0;
        r.metadata().supportsRandomAccess = true;
        return r;
    }

};

} // namespace encodings::encoders
