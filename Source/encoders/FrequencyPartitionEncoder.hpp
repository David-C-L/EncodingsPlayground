#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

// ---------------------------------------------------------------------------
// Index type selection
// ---------------------------------------------------------------------------

/// Selects the per-position index representation stored in the wire format.
enum class FreqPartIndexType : uint8_t {
    PerTierBitmaps = 0, ///< one N-bit bitmap per tier (default)
    TierTagArray   = 1, ///< single ceil(log2(numTiers+1))-bit tag per position
    EliasFano      = 2, ///< per-tier Elias-Fano positions + per-tier keys
    NoIndex        = 3, ///< no position index; rows are reordered by tier then fallback
};

/// Frequency-partition encoding with per-tier bitmaps for O(numTiers) random access.
///
/// The most-frequent values are assigned short fixed-width keys in ascending key-width
/// tiers. Active tiers are those whose key width is strictly less than half the storage
/// type width. Each tier stores a dictionary, a position index, and bit-packed keys.
/// Remaining values are stored raw in a fallback section.
///
/// Wire format (PerTierBitmaps, indexType=0):
///   [8]  numElements (uint64_t)
///   [1]  numTiers   (uint8_t)
///   [1]  indexType  (uint8_t, FreqPartIndexType)
///   Per tier:
///     [1]  keyBits    (uint8_t)
///     [4]  dictCount  (uint32_t)
///     [dictCount*sizeof(T)]  dictValues (most-frequent first, key 0 = top)
///     [ceil(numElements/64)*8]  bitmap (uint64_t words, bit i set iff pos i in tier)
///     [ceil(tierCount*keyBits/8)]  bit-packed keys (LSB-first)
///   [4]  fallbackCount (uint32_t)
///   [fallbackCount*sizeof(T)]  raw fallback values in original index order
///
/// Wire format (TierTagArray, indexType=1):
///   [8]  numElements (uint64_t)
///   [1]  numTiers   (uint8_t)
///   [1]  indexType  (uint8_t)
///   [ceil(numElements * tagBits / 8)]  packed tag array
///     (tag = tier index 0..numTiers-1, or numTiers for fallback;
///      tagBits = ceil(log2(numTiers+1)), LSB-first packing)
///   Per tier:
///     [1]  keyBits    (uint8_t)
///     [4]  dictCount  (uint32_t)
///     [dictCount*sizeof(T)]  dictValues
///     [ceil(tierCount*keyBits/8)]  bit-packed keys (in position order)
///   [4]  fallbackCount (uint32_t)
///   [fallbackCount*sizeof(T)]  raw fallback values in original index order
///
template <typename T, FreqPartIndexType IndexType = FreqPartIndexType::PerTierBitmaps>
    requires (std::is_same_v<T, uint8_t>  || std::is_same_v<T, uint16_t> ||
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
              std::is_same_v<T, int8_t>   || std::is_same_v<T, int16_t>  ||
              std::is_same_v<T, int32_t>  || std::is_same_v<T, int64_t>)
class FrequencyPartitionEncoder : public Codec<T> {
public:
    using EncodedData = EncodedBuffer<uint8_t>;

    static constexpr size_t kMaxKeyBits     = sizeof(T) * 4; // half storage width in bits
    static constexpr size_t kMaxTierDefs    = 32;            // upper bound on distinct key widths
    static constexpr size_t kTagChunkSize       = 256; // tags unpacked per bulk call
    static constexpr size_t kRankSampleStride   = 256; // TierTagArray sampled rank index granularity

    // Number of active tiers for this T: key widths 1..kMaxKeyBits (inclusive).
    // With non-power-of-2 widths the tier count equals kMaxKeyBits since we use
    // widths 1,2,...,kMaxKeyBits. Each tier's actual capacity is 2^keyBits.
    static constexpr size_t kNumActiveTiers = kMaxKeyBits;

    explicit FrequencyPartitionEncoder() = default;

    // ---------------------------------------------------------------------------
    // Encode
    // ---------------------------------------------------------------------------

    EncodedBuffer<uint8_t> encode(std::span<const T> input) override {
        const size_t N = input.size();

        // --- Phase 1: frequency count ---
        ankerl::unordered_dense::map<T, uint64_t> freq;
        freq.reserve(N < (size_t{1} << 20) ? N : (size_t{1} << 20));
        for (const T v : input) ++freq[v];

        // --- Phase 2: sort by frequency descending (ties broken by value) ---
        std::vector<std::pair<T, uint64_t>> sortedFreq(freq.begin(), freq.end());
        std::sort(sortedFreq.begin(), sortedFreq.end(),
                  [](const auto& a, const auto& b) {
                      return a.second != b.second ? a.second > b.second : a.first < b.first;
                  });
        const size_t numUnique = sortedFreq.size();

        // --- Phase 3: assign to tiers; accumulate element counts from freq table ---
        // valueInfo maps value → (tierIdx, key).
        ankerl::unordered_dense::map<T, std::pair<uint8_t, uint32_t>> valueInfo;
        valueInfo.reserve(numUnique);

        struct TierMeta {
            uint8_t keyBits{0};
            std::vector<T> dict; // dict[key] = value
        };
        TierMeta tiers[kNumActiveTiers];
        size_t tierElemCount[kNumActiveTiers] = {}; // total input elements per tier (from freq)
        size_t numTiersUsed = 0;
        {
            size_t assigned = 0;
            for (; numTiersUsed < kNumActiveTiers && assigned < numUnique; ++numTiersUsed) {
                const size_t t        = numTiersUsed;
                const size_t cap      = size_t{1} << (t + 1);
                const size_t toAssign = std::min(cap, numUnique - assigned);
                tiers[t].dict.reserve(toAssign);
                for (size_t i = 0; i < toAssign; ++i) {
                    const auto& sf = sortedFreq[assigned + i];
                    tiers[t].dict.push_back(sf.first);
                    valueInfo[sf.first] = {static_cast<uint8_t>(t), static_cast<uint32_t>(i)};
                    tierElemCount[t] += sf.second;
                }
                tiers[t].keyBits = static_cast<uint8_t>(
                    toAssign <= 1 ? 1
                                  : static_cast<uint8_t>(sizeof(size_t) * 8
                                                         - static_cast<size_t>(__builtin_clzll(toAssign - 1))));
                assigned += toAssign;
            }
        }

        // --- Phase 4: prune tiers using freq-derived counts (no input scan needed) ---
        const size_t numWords = (N + 63) / 64;
        bool includeTier[kNumActiveTiers] = {};
        for (size_t t = 0; t < numTiersUsed; ++t) {
            const size_t dSize = tiers[t].dict.size();
            if (dSize == 0) continue;
            const size_t count   = tierElemCount[t];
            const uint8_t kb     = tiers[t].keyBits;
            const size_t cost    = numWords * 64 + dSize * sizeof(T) * 8 + (count * kb + 7) / 8 * 8;
            const size_t savings = count * sizeof(T) * 8;
            includeTier[t] = (savings > cost);
        }

        uint8_t numTiersWithData  = 0;
        size_t  includedElemCount = 0;
        for (size_t t = 0; t < numTiersUsed; ++t) {
            if (includeTier[t] && !tiers[t].dict.empty()) {
                ++numTiersWithData;
                includedElemCount += tierElemCount[t];
            }
        }

        // --- Phase 5: pre-reserve per-tier key vectors and fallback ---
        std::vector<std::vector<uint32_t>> tierKeys(kNumActiveTiers);
        for (size_t t = 0; t < numTiersUsed; ++t)
            if (includeTier[t]) tierKeys[t].reserve(tierElemCount[t]);

        std::vector<T> fallback;
        fallback.reserve(N - includedElemCount);

        // --- Phase 6: accurate output size estimate, then single input scan + serialize ---
        std::vector<uint8_t> out;
        {
            size_t estSize = 10; // numElements(8) + numTiers(1) + indexType(1)
            for (size_t t = 0; t < numTiersUsed; ++t) {
                if (!includeTier[t] || tiers[t].dict.empty()) continue;
                const uint8_t kb = tiers[t].keyBits;
                estSize += 1 + 4 + tiers[t].dict.size() * sizeof(T); // keyBits + dictCount + dict
                estSize += packedKeyBytes(tierElemCount[t], kb);
                if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps)
                    estSize += numWords * sizeof(uint64_t);
                else if constexpr (IndexType == FreqPartIndexType::NoIndex)
                    estSize += 4; // tierCount uint32
                else if constexpr (IndexType == FreqPartIndexType::EliasFano)
                    estSize += 4 + 1 + 4 + 4 + numWords * sizeof(uint64_t); // generous EF overhead
            }
            if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
                const uint8_t tb = ceilLog2WithMinOne(static_cast<uint32_t>(numTiersWithData) + 1u);
                estSize += (N * tb + 7) / 8;
            }
            estSize += 4 + (N - includedElemCount) * sizeof(T);
            out.reserve(estSize);
        }

        const auto appendBytes = [&](const void* src, size_t n) {
            const auto* p = static_cast<const uint8_t*>(src);
            out.insert(out.end(), p, p + n);
        };
        const auto appendT = [&](auto v) { appendBytes(&v, sizeof(v)); };

        struct TierLogInfo {
            uint8_t logicalTier{0};
            uint8_t keyBits{0};
            size_t dictCount{0};
            size_t elemCount{0};
            size_t dictBytes{0};
            size_t keyBytes{0};
            size_t indexBytes{0};
        };
        std::vector<TierLogInfo> tierLogs;
        tierLogs.reserve(numTiersWithData);

        const size_t tagBytes =
            (IndexType == FreqPartIndexType::TierTagArray)
                ? ((N * ceilLog2WithMinOne(static_cast<uint32_t>(numTiersWithData) + 1u) + 7) / 8)
                : 0;

        appendT(static_cast<uint64_t>(N));
        appendT(numTiersWithData);
        appendT(static_cast<uint8_t>(IndexType));

        if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps) {
            // Allocate bitmaps only for included tiers.
            std::vector<std::vector<uint64_t>> bitmaps(numTiersUsed);
            for (size_t t = 0; t < numTiersUsed; ++t)
                if (includeTier[t] && !tiers[t].dict.empty())
                    bitmaps[t].assign(numWords, 0);

            // Single input scan: fill bitmaps + tier keys + fallback.
            for (size_t pos = 0; pos < N; ++pos) {
                const T val = input[pos];
                auto it = valueInfo.find(val);
                if (it != valueInfo.end()) {
                    const auto [t, key] = it->second;
                    if (includeTier[t]) {
                        bitmaps[t][pos / 64] |= uint64_t{1} << (pos % 64);
                        tierKeys[t].push_back(key);
                    } else {
                        fallback.push_back(val);
                    }
                } else {
                    fallback.push_back(val);
                }
            }

            for (size_t t = 0; t < numTiersUsed; ++t) {
                if (!includeTier[t] || tiers[t].dict.empty()) continue;
                const uint8_t kb     = tiers[t].keyBits;
                const auto&   dict   = tiers[t].dict;
                const auto&   keys   = tierKeys[t];
                const size_t dictBytes   = dict.size() * sizeof(T);
                const size_t keyBytes    = packedKeyBytes(keys.size(), kb);
                const size_t bitmapBytes = numWords * sizeof(uint64_t);
                appendT(kb);
                appendT(static_cast<uint32_t>(dict.size()));
                for (const T v : dict) appendT(v);
                for (size_t w = 0; w < numWords; ++w) appendT(bitmaps[t][w]);
                packKeys(out, keys, kb);
                tierLogs.push_back({static_cast<uint8_t>(t + 1), kb,
                                    dict.size(), keys.size(), dictBytes, keyBytes, bitmapBytes});
            }

        } else if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
            // tagBits = ceil(log2(numTiersWithData + 1)), minimum 1.
            const uint8_t tagBits = ceilLog2WithMinOne(static_cast<uint32_t>(numTiersWithData) + 1u);

            // Build tier remap: logical tier index → 0-based output index.
            uint8_t tierRemap[kNumActiveTiers];
            std::fill(tierRemap, tierRemap + kNumActiveTiers, uint8_t{0xFF});
            {
                uint8_t remapIdx = 0;
                for (size_t t = 0; t < numTiersUsed; ++t)
                    if (includeTier[t] && !tiers[t].dict.empty())
                        tierRemap[t] = remapIdx++;
            }

            // Pre-allocate zeroed tag array in output; record base offset.
            const size_t tagBytesLocal = (N * tagBits + 7) / 8;
            const size_t tagBase       = out.size();
            out.resize(out.size() + tagBytesLocal, 0);

            // Single input scan: collect tier keys + write tags into the pre-allocated region.
            // Tags are accumulated in a uint64_t and flushed one byte at a time, eliminating
            // the per-element read-modify-write of packSingleTag.
            {
                uint64_t tagAcc     = 0;
                size_t   tagAccBits = 0;
                size_t   tagByteOut = 0;

                for (size_t pos = 0; pos < N; ++pos) {
                    const T val = input[pos];
                    auto it = valueInfo.find(val);
                    uint8_t tag = numTiersWithData;
                    if (it != valueInfo.end()) {
                        const auto [t, key] = it->second;
                        if (includeTier[t]) {
                            tag = tierRemap[t];
                            tierKeys[t].push_back(key);
                        } else {
                            fallback.push_back(val);
                        }
                    } else {
                        fallback.push_back(val);
                    }
                    tagAcc |= static_cast<uint64_t>(tag) << tagAccBits;
                    tagAccBits += tagBits;
                    if (tagAccBits >= 8) {
                        out[tagBase + tagByteOut++] = static_cast<uint8_t>(tagAcc & 0xFF);
                        tagAcc     >>= 8;
                        tagAccBits  -= 8;
                    }
                }
                if (tagAccBits > 0)
                    out[tagBase + tagByteOut] = static_cast<uint8_t>(tagAcc & 0xFF);
            }

            for (size_t t = 0; t < numTiersUsed; ++t) {
                if (!includeTier[t] || tiers[t].dict.empty()) continue;
                const uint8_t kb   = tiers[t].keyBits;
                const auto&   dict = tiers[t].dict;
                const auto&   keys = tierKeys[t];
                const size_t dictBytes = dict.size() * sizeof(T);
                const size_t keyBytes  = packedKeyBytes(keys.size(), kb);
                appendT(kb);
                appendT(static_cast<uint32_t>(dict.size()));
                for (const T v : dict) appendT(v);
                packKeys(out, keys, kb);
                tierLogs.push_back({static_cast<uint8_t>(t + 1), kb,
                                    dict.size(), keys.size(), dictBytes, keyBytes, 0});
            }

        } else if constexpr (IndexType == FreqPartIndexType::EliasFano) {
            // Allocate positions only for included tiers.
            std::vector<std::vector<uint32_t>> tierPositions(numTiersUsed);
            for (size_t t = 0; t < numTiersUsed; ++t)
                if (includeTier[t] && !tiers[t].dict.empty())
                    tierPositions[t].reserve(tierElemCount[t]);

            // Single input scan.
            for (size_t pos = 0; pos < N; ++pos) {
                const T val = input[pos];
                auto it = valueInfo.find(val);
                if (it != valueInfo.end()) {
                    const auto [t, key] = it->second;
                    if (includeTier[t]) {
                        tierPositions[t].push_back(static_cast<uint32_t>(pos));
                        tierKeys[t].push_back(key);
                    } else {
                        fallback.push_back(val);
                    }
                } else {
                    fallback.push_back(val);
                }
            }

            for (size_t t = 0; t < numTiersUsed; ++t) {
                if (!includeTier[t] || tiers[t].dict.empty()) continue;
                const uint8_t kb   = tiers[t].keyBits;
                const auto&   dict = tiers[t].dict;
                const auto&   keys = tierKeys[t];
                const auto&   posv = tierPositions[t];
                const size_t dictBytes = dict.size() * sizeof(T);
                const size_t keyBytes  = packedKeyBytes(keys.size(), kb);

                const uint8_t lowBits = chooseEliasFanoLowBits(N, posv.size());
                std::vector<uint32_t> lows;
                lows.reserve(posv.size());
                const uint32_t lowMask = (lowBits == 32) ? 0xFFFFFFFFu
                                       : ((lowBits == 0) ? 0u : ((1u << lowBits) - 1u));
                for (uint32_t ppos : posv) lows.push_back(ppos & lowMask);
                const size_t lowBytes = packedKeyBytes(lows.size(), lowBits);

                const size_t highBitsLen   = (N >> lowBits) + posv.size() + 1;
                const size_t highWordCount = (highBitsLen + 63) / 64;
                std::vector<uint64_t> highBits(highWordCount, 0);
                for (size_t i = 0; i < posv.size(); ++i) {
                    const size_t hi     = static_cast<size_t>(posv[i] >> lowBits);
                    const size_t bitPos = hi + i;
                    highBits[bitPos / 64] |= uint64_t{1} << (bitPos % 64);
                }

                appendT(kb);
                appendT(static_cast<uint32_t>(dict.size()));
                for (const T v : dict) appendT(v);
                appendT(static_cast<uint32_t>(posv.size()));
                appendT(lowBits);
                appendT(static_cast<uint32_t>(lowBytes));
                appendT(static_cast<uint32_t>(highWordCount));
                packKeys(out, lows, lowBits);
                for (size_t w = 0; w < highWordCount; ++w) appendT(highBits[w]);
                packKeys(out, keys, kb);
                tierLogs.push_back({static_cast<uint8_t>(t + 1), kb, dict.size(), keys.size(),
                                    dictBytes, keyBytes,
                                    static_cast<size_t>(4 + 1 + 4 + 4) + lowBytes + highWordCount * sizeof(uint64_t)});
            }

        } else {
            // NoIndex: no positional structures needed.
            // Single input scan.
            for (size_t pos = 0; pos < N; ++pos) {
                const T val = input[pos];
                auto it = valueInfo.find(val);
                if (it != valueInfo.end()) {
                    const auto [t, key] = it->second;
                    if (includeTier[t]) {
                        tierKeys[t].push_back(key);
                    } else {
                        fallback.push_back(val);
                    }
                } else {
                    fallback.push_back(val);
                }
            }

            for (size_t t = 0; t < numTiersUsed; ++t) {
                if (!includeTier[t] || tiers[t].dict.empty()) continue;
                const uint8_t kb   = tiers[t].keyBits;
                const auto&   dict = tiers[t].dict;
                const auto&   keys = tierKeys[t];
                const size_t dictBytes = dict.size() * sizeof(T);
                const size_t keyBytes  = packedKeyBytes(keys.size(), kb);
                appendT(kb);
                appendT(static_cast<uint32_t>(dict.size()));
                for (const T v : dict) appendT(v);
                appendT(static_cast<uint32_t>(keys.size()));
                packKeys(out, keys, kb);
                tierLogs.push_back({static_cast<uint8_t>(t + 1), kb,
                                    dict.size(), keys.size(), dictBytes, keyBytes, 0});
            }
        }

        if (fallback.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
            throw std::runtime_error("FrequencyPartitionEncoder: fallback count exceeds uint32 limit");
        appendT(static_cast<uint32_t>(fallback.size()));
        for (const T v : fallback) appendT(v);

        // 8-byte zero padding so unpackKey can always do a single 8-byte memcpy
        // without reading past the buffer end.  Not counted in compressedSize.
        const size_t logicalSize = out.size();
        out.resize(out.size() + 8, 0);

        EncodedBuffer<uint8_t> result;
        result.metadata().elementCount         = N;
        result.metadata().encodingName         = name();
        result.metadata().supportsRandomAccess = true;
        result.metadata().compressedSize       = logicalSize;
        result.metadata().uncompressedSize     = N * sizeof(T);
        result.data() = std::move(out);

        if (verboseEnabled()) {
            const size_t rawBytes = N * sizeof(T);
            const size_t headerBytes = 8 + 1 + 1;
            const size_t fallbackBytes = 4 + fallback.size() * sizeof(T);
            size_t totalIndexBytes = 0;
            if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps) {
                totalIndexBytes = numTiersWithData * numWords * sizeof(uint64_t);
            } else if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
                totalIndexBytes = tagBytes;
            } else if constexpr (IndexType == FreqPartIndexType::EliasFano) {
                for (const auto& tl : tierLogs) totalIndexBytes += tl.indexBytes;
            }

            const double totalBpe = N ? (static_cast<double>(result.metadata().compressedSize) * 8.0 / static_cast<double>(N)) : 0.0;
            const double indexBpe = N ? (static_cast<double>(totalIndexBytes) * 8.0 / static_cast<double>(N)) : 0.0;
            const double indexPctCompressed = result.metadata().compressedSize
                ? (100.0 * static_cast<double>(totalIndexBytes) / static_cast<double>(result.metadata().compressedSize))
                : 0.0;
            const double indexPctRaw = rawBytes
                ? (100.0 * static_cast<double>(totalIndexBytes) / static_cast<double>(rawBytes))
                : 0.0;

            std::cerr << "[FreqPart] N=" << N
                      << " unique=" << numUnique
                      << " mode="
                      << (IndexType == FreqPartIndexType::PerTierBitmaps ? "PerTierBitmaps"
                          : IndexType == FreqPartIndexType::TierTagArray ? "TierTagArray"
                          : IndexType == FreqPartIndexType::EliasFano ? "EliasFano"
                          : "NoIndex")
                      << " tiers=" << static_cast<int>(numTiersWithData)
                      << " total=" << result.metadata().compressedSize << "B (~" << totalBpe << " b/elem)"
                      << " ratio=" << (rawBytes ? (static_cast<double>(result.metadata().compressedSize) / static_cast<double>(rawBytes)) : 0.0)
                      << "\n";

            std::cerr << "  index_overhead=" << totalIndexBytes << "B (~" << indexBpe << " b/elem, "
                      << indexPctCompressed << "% of compressed, " << indexPctRaw << "% of raw)"
                      << " header=" << headerBytes << "B"
                      << " fallback=" << fallbackBytes << "B (count=" << fallback.size() << ")"
                      << "\n";

            for (const auto& tl : tierLogs) {
                const size_t tierBytes = 1 + 4 + tl.dictBytes + tl.keyBytes + tl.indexBytes;
                const double tierBpe = N ? (static_cast<double>(tierBytes) * 8.0 / static_cast<double>(N)) : 0.0;
                const double tierDensity = N ? (100.0 * static_cast<double>(tl.elemCount) / static_cast<double>(N)) : 0.0;
                std::cerr << "  tier" << static_cast<int>(tl.logicalTier)
                          << " keyBits=" << static_cast<int>(tl.keyBits)
                          << " elems=" << tl.elemCount << " (" << tierDensity << "%)"
                          << " dict=" << tl.dictCount
                          << " bytes={meta:" << (1 + 4)
                          << ",dict:" << tl.dictBytes
                          << ",keys:" << tl.keyBytes
                          << ",index:" << tl.indexBytes
                          << ",total:" << tierBytes << "}"
                          << " (~" << tierBpe << " b/elem)"
                          << "\n";
            }
        }

        return result;
    }

    // ---------------------------------------------------------------------------
    // Decode helpers
    // ---------------------------------------------------------------------------

private:
    static bool verboseEnabled() {
        static bool v = (std::getenv("FREQPART_VERBOSE") != nullptr);
        return v;
    }

    struct ParsedHeader {
        uint64_t numElements{0};
        uint8_t  numTiers{0};
        FreqPartIndexType indexType{FreqPartIndexType::PerTierBitmaps};
        size_t   numWords{0};

        struct TierData {
            uint8_t  keyBits{0};
            std::vector<T>        dict;
            std::vector<uint64_t> bitmap;     // only for PerTierBitmaps
            std::vector<uint32_t> rankIndex;  // Rank9 superblock: rankIndex[b] = popcount(bitmap[0..8b-1])
            std::vector<uint32_t> positions;  // for EliasFano
            size_t   keysOffset{0};
            size_t   tierCount{0};
        };
        std::vector<TierData> tiers;

        // TierTagArray fields.
        size_t   tagArrayOffset{0}; // byte offset of packed tag array
        uint8_t  tagBits{0};        // bits per tag

        size_t   fallbackOffset{0};
        uint32_t fallbackCount{0};

        // Precomputed from PerTierBitmaps: OR of all tier bitmaps.
        std::vector<uint64_t> coveredBitmap;
        // fallbackPrefixPop[w] = number of uncovered (fallback) elements in words 0..w-1.
        std::vector<size_t>   fallbackPrefixPop;
        // Prefix of decoded element counts per tier (for NoIndex mode).
        std::vector<size_t>   tierPrefix;

        // TierTagArray sampled rank index.
        // tierRankSamples[t][i] = count of elements with tag==t in positions [0, i*kRankSampleStride).
        // tierRankSamples[numTiers][i] = fallback count (tag >= numTiers).
        // Allows O(kRankSampleStride) random access instead of O(N).
        std::vector<std::vector<uint32_t>> tierRankSamples;
    };

    struct HeaderCache {
        const uint8_t* dataPtr{nullptr};
        size_t         dataSize{0};
        ParsedHeader   header{};
        bool           valid{false};
    };

    HeaderCache headerCache_{};

    static ParsedHeader parseHeader(const EncodedBuffer<uint8_t>& enc) {
        const uint8_t* p   = enc.data().data();
        const uint8_t* end = p + enc.data().size();
        (void)end;

        const auto readU64 = [&]() -> uint64_t { uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; };
        const auto readU32 = [&]() -> uint32_t { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
        const auto readU8  = [&]() -> uint8_t  { return *p++; };
        const auto readT   = [&]() -> T        { T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; };

        ParsedHeader h;
        h.numElements = readU64();
        h.numTiers    = readU8();
        h.indexType   = static_cast<FreqPartIndexType>(readU8());
        if (h.indexType != IndexType) {
            throw std::runtime_error("FrequencyPartitionEncoder::parseHeader: wire indexType does not match template IndexType");
        }
        h.numWords    = (h.numElements + 63) / 64;

        h.tiers.resize(h.numTiers);

    if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps) {
            for (auto& td : h.tiers) {
                td.keyBits = readU8();
                const uint32_t dictCount = readU32();
                td.dict.resize(dictCount);
                for (auto& v : td.dict) v = readT();

                td.bitmap.resize(h.numWords);
                for (auto& w : td.bitmap) { std::memcpy(&w, p, 8); p += 8; }

                td.tierCount = 0;
                for (uint64_t w : td.bitmap) td.tierCount += static_cast<size_t>(__builtin_popcountll(w));

                // Build Rank9 superblock index: one cumulative count per 8-word (512-bit) block.
                const size_t numBlocks = (h.numWords + 7) / 8;
                td.rankIndex.resize(numBlocks + 1, 0);
                for (size_t blk = 0; blk < numBlocks; ++blk) {
                    td.rankIndex[blk + 1] = td.rankIndex[blk];
                    const size_t wEnd = std::min((blk + 1) * 8, h.numWords);
                    for (size_t w = blk * 8; w < wEnd; ++w)
                        td.rankIndex[blk + 1] += static_cast<uint32_t>(__builtin_popcountll(td.bitmap[w]));
                }

                td.keysOffset = static_cast<size_t>(p - enc.data().data());
                p += packedKeyBytes(td.tierCount, td.keyBits);
            }

            // Build covered bitmap and fallback prefix-popcount index.
            h.coveredBitmap.assign(h.numWords, 0);
            for (const auto& td : h.tiers)
                for (size_t w = 0; w < h.numWords; ++w)
                    h.coveredBitmap[w] |= td.bitmap[w];

            h.fallbackPrefixPop.resize(h.numWords + 1, 0);
            for (size_t w = 0; w < h.numWords; ++w) {
                uint64_t uncov = ~h.coveredBitmap[w];
                // Mask out bits beyond numElements in the last word.
                if (w == h.numWords - 1 && (h.numElements % 64) != 0)
                    uncov &= (uint64_t{1} << (h.numElements % 64)) - 1;
                h.fallbackPrefixPop[w + 1] = h.fallbackPrefixPop[w]
                                           + static_cast<size_t>(__builtin_popcountll(uncov));
            }
    } else if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
            // TierTagArray: tag array comes first.
            h.tagBits = ceilLog2WithMinOne(static_cast<uint32_t>(h.numTiers) + 1u);
            h.tagArrayOffset = static_cast<size_t>(p - enc.data().data());
            const size_t tagBytes = (h.numElements * h.tagBits + 7) / 8;
            p += tagBytes;

            // Single pass over the tag array: count elements per tier and build sampled rank index.
            // tierRankSamples has numTiers+1 rows (one per tier + one for fallback).
            // Row t, sample i = count of tag==t in [0, i*kRankSampleStride).
            {
                const uint8_t* tagPtr    = enc.data().data() + h.tagArrayOffset;
                const uint8_t  tb        = h.tagBits;
                const uint8_t  tmask     = static_cast<uint8_t>((1u << tb) - 1u);
                const size_t   numBuckets = static_cast<size_t>(h.numTiers) + 1; // +1 for fallback
                const size_t   numSamples = (h.numElements + kRankSampleStride - 1) / kRankSampleStride + 1;

                h.tierRankSamples.assign(numBuckets, std::vector<uint32_t>(numSamples, 0));

                // counts[t] = running count for tier t; counts[numTiers] = fallback count.
                uint32_t counts[kNumActiveTiers + 1] = {};
                size_t bitCursor = 0;

                for (size_t i = 0; i < h.numElements; ++i) {
                    if (i % kRankSampleStride == 0) {
                        const size_t si = i / kRankSampleStride;
                        for (size_t t = 0; t < numBuckets; ++t)
                            h.tierRankSamples[t][si] = counts[t];
                    }
                    const size_t by = bitCursor >> 3;
                    const size_t bo = bitCursor & 7;
                    uint16_t buf = tagPtr[by];
                    if (bo + tb > 8) buf |= static_cast<uint16_t>(tagPtr[by + 1]) << 8;
                    const uint8_t tag = static_cast<uint8_t>((buf >> bo) & tmask);
                    const uint8_t bucket = (tag < h.numTiers) ? tag : h.numTiers;
                    ++counts[bucket];
                    bitCursor += tb;
                }
                // Write sentinel sample past the last full stride.
                {
                    const size_t si = (h.numElements + kRankSampleStride - 1) / kRankSampleStride;
                    for (size_t t = 0; t < numBuckets; ++t)
                        h.tierRankSamples[t][si] = counts[t];
                }
                for (size_t t = 0; t < h.numTiers; ++t) h.tiers[t].tierCount = counts[t];
            }

            // Parse per-tier metadata sequentially; tierCount is now known so p advances correctly.
            for (auto& td : h.tiers) {
                td.keyBits = readU8();
                const uint32_t dictCount = readU32();
                td.dict.resize(dictCount);
                for (auto& v : td.dict) v = readT();
                td.keysOffset = static_cast<size_t>(p - enc.data().data());
                p += packedKeyBytes(td.tierCount, td.keyBits);
            }
    } else if constexpr (IndexType == FreqPartIndexType::EliasFano) {
            for (auto& td : h.tiers) {
                td.keyBits = readU8();
                const uint32_t dictCount = readU32();
                td.dict.resize(dictCount);
                for (auto& v : td.dict) v = readT();

                td.tierCount = readU32();
                const uint8_t lowBits = readU8();
                const uint32_t lowBytes = readU32();
                const uint32_t highWords = readU32();

                const uint8_t* lowBase = p;
                p += lowBytes;
                const uint8_t* highBase = p;
                p += static_cast<size_t>(highWords) * sizeof(uint64_t);

                td.positions = decodeEliasFanoPositions(lowBase, lowBits, highBase, highWords, td.tierCount);

                td.keysOffset = static_cast<size_t>(p - enc.data().data());
                p += packedKeyBytes(td.tierCount, td.keyBits);
            }

            // Only build the covered bitmap and fallback prefix-pop table when there
            // is actually a fallback section — peek at fallbackCount without advancing p.
            {
                uint32_t peekedFallbackCount = 0;
                std::memcpy(&peekedFallbackCount, p, 4);
                if (peekedFallbackCount > 0) {
                    h.coveredBitmap.assign(h.numWords, 0);
                    for (const auto& td : h.tiers)
                        for (uint32_t pos : td.positions)
                            h.coveredBitmap[pos / 64] |= uint64_t{1} << (pos % 64);

                    h.fallbackPrefixPop.resize(h.numWords + 1, 0);
                    for (size_t w = 0; w < h.numWords; ++w) {
                        uint64_t uncov = ~h.coveredBitmap[w];
                        if (w == h.numWords - 1 && (h.numElements % 64) != 0)
                            uncov &= (uint64_t{1} << (h.numElements % 64)) - 1;
                        h.fallbackPrefixPop[w + 1] = h.fallbackPrefixPop[w]
                                                   + static_cast<size_t>(__builtin_popcountll(uncov));
                    }
                }
            }
        } else {
            // NoIndex: tiers contain only dict + tierCount + packed keys.
            for (auto& td : h.tiers) {
                td.keyBits = readU8();
                const uint32_t dictCount = readU32();
                td.dict.resize(dictCount);
                for (auto& v : td.dict) v = readT();
                td.tierCount = readU32();
                td.keysOffset = static_cast<size_t>(p - enc.data().data());
                p += packedKeyBytes(td.tierCount, td.keyBits);
            }
        }

        h.fallbackOffset = static_cast<size_t>(p - enc.data().data());
        h.fallbackCount  = readU32();

        h.tierPrefix.assign(h.tiers.size() + 1, 0);
        for (size_t t = 0; t < h.tiers.size(); ++t) {
            h.tierPrefix[t + 1] = h.tierPrefix[t] + h.tiers[t].tierCount;
        }
        return h;
    }

    const ParsedHeader& getParsedHeader(const EncodedBuffer<uint8_t>& enc) {
        const uint8_t* ptr = enc.data().data();
        const size_t   sz  = enc.data().size();
        if (!headerCache_.valid || headerCache_.dataPtr != ptr || headerCache_.dataSize != sz) {
            headerCache_.header   = parseHeader(enc);
            headerCache_.dataPtr  = ptr;
            headerCache_.dataSize = sz;
            headerCache_.valid    = true;
        }
        return headerCache_.header;
    }

    // ---------------------------------------------------------------------------
    // Bit-packing helpers
    // ---------------------------------------------------------------------------

    static constexpr size_t packedKeyBytes(size_t count, uint8_t keyBits) {
        return (count * keyBits + 7) / 8;
    }

    static constexpr uint8_t ceilLog2WithMinOne(uint32_t x) {
        if (x <= 1u) return 1u;
        return static_cast<uint8_t>(std::bit_width(x - 1u));
    }

    static uint8_t chooseEliasFanoLowBits(size_t universe, size_t n) {
        if (n == 0 || universe <= n) return 0;
        const size_t ratio = universe / n;
        if (ratio <= 1) return 0;
        return static_cast<uint8_t>(std::min<size_t>(31, std::bit_width(ratio) - 1));
    }

    static std::vector<uint32_t> decodeEliasFanoPositions(
        const uint8_t* lowBase,
        uint8_t lowBits,
        const uint8_t* highBase,
        uint32_t highWords,
        size_t count)
    {
        std::vector<uint32_t> out;
        out.reserve(count);
        size_t rank = 0;
        for (uint32_t w = 0; w < highWords && rank < count; ++w) {
            uint64_t bits = 0;
            std::memcpy(&bits, highBase + static_cast<size_t>(w) * sizeof(uint64_t), sizeof(uint64_t));
            while (bits && rank < count) {
                const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(bits));
                const uint32_t high = static_cast<uint32_t>(w * 64 + bit - rank);
                const uint32_t low = unpackKey(lowBase, rank, lowBits);
                out.push_back((high << lowBits) | low);
                ++rank;
                bits &= (bits - 1);
            }
        }
        return out;
    }

    // Pack keys into LSB-first bit-packed format, appending to `out`.
    static void packKeys(std::vector<uint8_t>& out, const std::vector<uint32_t>& keys, uint8_t keyBits) {
        const size_t byteCount  = packedKeyBytes(keys.size(), keyBits);
        const size_t baseOffset = out.size();
        out.resize(out.size() + byteCount, 0);
        for (size_t r = 0; r < keys.size(); ++r) {
            const uint32_t k   = keys[r];
            size_t bitPos      = r * keyBits;
            uint32_t remaining = k;
            uint8_t  bitsLeft  = keyBits;
            while (bitsLeft > 0) {
                const size_t byteIdx = bitPos / 8;
                const size_t bitOff  = bitPos % 8;
                const uint8_t chunk  = static_cast<uint8_t>(bitsLeft > (8 - bitOff)
                                         ? (8 - bitOff) : bitsLeft);
                out[baseOffset + byteIdx] |= static_cast<uint8_t>(
                    (remaining & ((1u << chunk) - 1u)) << bitOff);
                remaining >>= chunk;
                bitPos    += chunk;
                bitsLeft  -= chunk;
            }
        }
    }

    // Unpack key at rank `r` from packed keys at `base`, with `keyBits` bits per key.
    // Requires 8 bytes of zero-padding past the end of the packed array (guaranteed by encode).
    static uint32_t unpackKey(const uint8_t* base, size_t r, uint8_t keyBits) {
        const size_t  bitPos  = r * keyBits;
        const size_t  byteIdx = bitPos >> 3;
        const size_t  bitOff  = bitPos & 7;
        const uint32_t mask   = (keyBits == 32) ? ~0u : ((1u << keyBits) - 1u);
        uint64_t buf = 0;
        std::memcpy(&buf, base + byteIdx, sizeof(uint64_t));
        return static_cast<uint32_t>((buf >> bitOff) & mask);
    }

    // Pack a single tag value into the tag array in `out` starting at `baseOffset`.
    static void packSingleTag(std::vector<uint8_t>& out, size_t baseOffset,
                               size_t pos, uint8_t tagBits, uint8_t tag) {
        size_t bitPos    = pos * tagBits;
        uint8_t remaining = tag;
        uint8_t bitsLeft  = tagBits;
        while (bitsLeft > 0) {
            const size_t byteIdx = bitPos / 8;
            const size_t bitOff  = bitPos % 8;
            const uint8_t chunk  = static_cast<uint8_t>(
                bitsLeft > (8 - bitOff) ? (8 - bitOff) : bitsLeft);
            out[baseOffset + byteIdx] |= static_cast<uint8_t>(
                (remaining & ((1u << chunk) - 1u)) << bitOff);
            remaining >>= chunk;
            bitPos    += chunk;
            bitsLeft  -= chunk;
        }
    }

    // Read a single tag value from the packed tag array.
    static uint8_t unpackTagAt(const uint8_t* base, size_t pos, uint8_t tagBits) {
        const size_t bitPos  = pos * tagBits;
        const size_t byteIdx = bitPos / 8;
        const size_t bitOff  = bitPos % 8;
        const uint8_t mask   = static_cast<uint8_t>((1u << tagBits) - 1u);
        uint16_t buf = static_cast<uint16_t>(base[byteIdx]);
        if (bitOff + tagBits > 8) buf |= static_cast<uint16_t>(base[byteIdx + 1]) << 8;
        return static_cast<uint8_t>((buf >> bitOff) & mask);
    }

    // Unpack `count` consecutive tags starting at bit offset `startBit` into dst[].
    // dst must be caller-allocated (at least `count` bytes); no heap allocation occurs.
    // The loop is branch-free on the tag value, enabling auto-vectorisation of the
    // unpack itself independently of the dispatch logic in the callers.
    static void unpackTagsInto(const uint8_t* __restrict__ base,
                                size_t startBit, uint8_t tagBits,
                                size_t count,
                                uint8_t* __restrict__ dst) noexcept {
        const uint8_t mask = static_cast<uint8_t>((1u << tagBits) - 1u);
        size_t bitCursor = startBit;
        for (size_t i = 0; i < count; ++i) {
            const size_t by = bitCursor >> 3;
            const size_t bo = bitCursor & 7;
            uint16_t buf = base[by];
            if (bo + tagBits > 8) buf |= static_cast<uint16_t>(base[by + 1]) << 8;
            dst[i] = static_cast<uint8_t>((buf >> bo) & mask);
            bitCursor += tagBits;
        }
    }

    // Popcount of all bits in `bitmap` strictly before position `i` — O(N/64), kept for coveredBitmap.
    static size_t popcountPrefix(const std::vector<uint64_t>& bitmap, size_t i) {
        size_t rank = 0;
        const size_t fullWords = i / 64;
        for (size_t w = 0; w < fullWords; ++w)
            rank += static_cast<size_t>(__builtin_popcountll(bitmap[w]));
        const size_t rem = i % 64;
        if (rem > 0)
            rank += static_cast<size_t>(__builtin_popcountll(bitmap[fullWords] & ((uint64_t{1} << rem) - 1)));
        return rank;
    }

    // O(1) rank query using the Rank9 superblock index stored in TierData.
    // Returns the number of set bits in td.bitmap strictly before position `i`.
    static size_t popcountPrefixFast(const typename ParsedHeader::TierData& td, size_t i) {
        const size_t blk     = i / 512;
        const size_t wordOff = (i % 512) / 64;
        const size_t bitOff  = i % 64;
        size_t rank = td.rankIndex[blk];
        for (size_t w = blk * 8; w < blk * 8 + wordOff; ++w)
            rank += static_cast<size_t>(__builtin_popcountll(td.bitmap[w]));
        if (bitOff)
            rank += static_cast<size_t>(__builtin_popcountll(
                td.bitmap[blk * 8 + wordOff] & ((uint64_t{1} << bitOff) - 1)));
        return rank;
    }

    // Compute fallback rank before position `i` using the precomputed prefix table.
    static size_t fallbackRankAt(const ParsedHeader& h, size_t i) {
        size_t rank      = h.fallbackPrefixPop[i / 64];
        uint64_t uncov   = ~h.coveredBitmap[i / 64];
        rank += static_cast<size_t>(__builtin_popcountll(uncov & ((uint64_t{1} << (i % 64)) - 1)));
        return rank;
    }

    // ---------------------------------------------------------------------------
    // Decode helpers per index mode (split for profiler visibility)
    // ---------------------------------------------------------------------------

    void decodeAllPerTierBitmapsInto(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, T* out) const {
        const size_t N = static_cast<size_t>(h.numElements);

        for (const auto& td : h.tiers) {
            const uint8_t* keysBase = enc.data().data() + td.keysOffset;
            size_t rank = 0;
            for (size_t w = 0; w < h.numWords; ++w) {
                uint64_t word = td.bitmap[w];
                while (word) {
                    const size_t bit = static_cast<size_t>(__builtin_ctzll(word));
                    const size_t pos = w * 64 + bit;
                    if (pos < N) out[pos] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
                    ++rank;
                    word &= word - 1;
                }
            }
        }

        if (h.fallbackCount > 0) {
            const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
            size_t fi = 0;
            for (size_t w = 0; w < h.numWords; ++w) {
                uint64_t uncov = ~h.coveredBitmap[w];
                if (w == h.numWords - 1 && (N % 64) != 0) uncov &= (uint64_t{1} << (N % 64)) - 1;
                while (uncov) {
                    const size_t bit = static_cast<size_t>(__builtin_ctzll(uncov));
                    const size_t pos = w * 64 + bit;
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    out[pos] = v;
                    ++fi;
                    uncov &= uncov - 1;
                }
            }
        }
    }

    std::vector<T> decodeAllPerTierBitmaps(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h) const {
        std::vector<T> result(h.numElements);
        decodeAllPerTierBitmapsInto(enc, h, result.data());
        return result;
    }

    void decodeAllTierTagArrayInto(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, T* out) const {
        const size_t N = static_cast<size_t>(h.numElements);
        const uint8_t* tagBase = enc.data().data() + h.tagArrayOffset;
        const uint8_t  tagBits = h.tagBits;

        std::vector<const uint8_t*> tierKeysBase(h.numTiers, nullptr);
        std::vector<size_t>         tierBitPos(h.numTiers, 0);
        for (size_t t = 0; t < h.numTiers; ++t)
            tierKeysBase[t] = enc.data().data() + h.tiers[t].keysOffset;

        const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
        size_t fi = 0;
        size_t tagBitCursor = 0;

        // Unpack tags in fixed-size chunks into a stack buffer.
        // The unpack loop (branch-free) and the dispatch loop are separate, letting
        // the compiler vectorise the unpack independently of the key/dict logic.
        alignas(64) uint8_t tagScratch[kTagChunkSize];

        for (size_t pos = 0; pos < N; ) {
            const size_t chunk = std::min<size_t>(kTagChunkSize, N - pos);
            unpackTagsInto(tagBase, tagBitCursor, tagBits, chunk, tagScratch);
            tagBitCursor += chunk * tagBits;

            for (size_t i = 0; i < chunk; ++i, ++pos) {
                const uint8_t tag = tagScratch[i];
                if (tag < h.numTiers) {
                    const auto& td = h.tiers[tag];
                    const uint8_t  kb    = td.keyBits;
                    const size_t   kBp   = tierBitPos[tag];
                    const size_t   kBy   = kBp >> 3;
                    const size_t   kBo   = kBp & 7;
                    const uint32_t kMask = (kb == 32) ? ~0u : ((1u << kb) - 1u);
                    uint64_t kBuf = 0;
                    std::memcpy(&kBuf, tierKeysBase[tag] + kBy, (kBo + kb + 7) >> 3);
                    out[pos] = td.dict[static_cast<uint32_t>((kBuf >> kBo) & kMask)];
                    tierBitPos[tag] += kb;
                } else {
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    out[pos] = v;
                    ++fi;
                }
            }
        }
    }

    std::vector<T> decodeAllTierTagArray(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h) const {
        std::vector<T> result(h.numElements);
        decodeAllTierTagArrayInto(enc, h, result.data());
        return result;
    }

    void decodeAllEliasFanoInto(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, T* out) const {
        const size_t N = static_cast<size_t>(h.numElements);

        for (const auto& td : h.tiers) {
            const uint8_t* keysBase = enc.data().data() + td.keysOffset;
            for (size_t rank = 0; rank < td.positions.size(); ++rank) {
                const size_t pos = td.positions[rank];
                if (pos < N) out[pos] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
            }
        }

        if (h.fallbackCount > 0) {
            const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
            size_t fi = 0;
            for (size_t w = 0; w < h.numWords; ++w) {
                uint64_t uncov = ~h.coveredBitmap[w];
                if (w == h.numWords - 1 && (N % 64) != 0) uncov &= (uint64_t{1} << (N % 64)) - 1;
                while (uncov) {
                    const size_t bit = static_cast<size_t>(__builtin_ctzll(uncov));
                    const size_t pos = w * 64 + bit;
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    out[pos] = v;
                    ++fi;
                    uncov &= uncov - 1;
                }
            }
        }
    }

    std::vector<T> decodeAllEliasFano(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h) const {
        std::vector<T> result(h.numElements);
        decodeAllEliasFanoInto(enc, h, result.data());
        return result;
    }

    void decodeAllNoIndexInto(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, T* out) const {
        size_t pos = 0;
        for (const auto& td : h.tiers) {
            const uint8_t* keysBase = enc.data().data() + td.keysOffset;
            for (size_t rank = 0; rank < td.tierCount; ++rank) {
                out[pos++] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
            }
        }
        const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
        for (size_t i = 0; i < h.fallbackCount; ++i) {
            T v; std::memcpy(&v, fp + i, sizeof(T));
            out[pos++] = v;
        }
    }

    std::vector<T> decodeAllNoIndex(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h) const {
        std::vector<T> result;
        result.reserve(static_cast<size_t>(h.numElements));
        // Use Into variant writing into result's storage after sizing it.
        result.resize(h.numElements);
        decodeAllNoIndexInto(enc, h, result.data());
        return result;
    }

    std::optional<T> decodeAtPerTierBitmaps(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, size_t i) const {
        for (const auto& td : h.tiers) {
            if (td.bitmap[i / 64] & (uint64_t{1} << (i % 64))) {
                const size_t rank = popcountPrefixFast(td, i);
                return td.dict[unpackKey(enc.data().data() + td.keysOffset, rank, td.keyBits)];
            }
        }
        const size_t fallbackRank = fallbackRankAt(h, i);
        const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
        T v; std::memcpy(&v, fp + fallbackRank, sizeof(T));
        return v;
    }

    std::optional<T> decodeAtTierTagArray(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, size_t i) const {
        const uint8_t* tagBase = enc.data().data() + h.tagArrayOffset;
        const uint8_t  tagBits = h.tagBits;
        const uint8_t  tag     = unpackTagAt(tagBase, i, tagBits);
        const uint8_t  bucket  = (tag < h.numTiers) ? tag : h.numTiers;

        // Jump to nearest sample, then scan at most kRankSampleStride positions.
        const size_t sampleIdx = i / kRankSampleStride;
        size_t rank = h.tierRankSamples[bucket][sampleIdx];
        const size_t scanStart = sampleIdx * kRankSampleStride;
        for (size_t j = scanStart; j < i; ++j) {
            const uint8_t t = unpackTagAt(tagBase, j, tagBits);
            const uint8_t b = (t < h.numTiers) ? t : h.numTiers;
            if (b == bucket) ++rank;
        }

        if (tag < h.numTiers) {
            const auto& td = h.tiers[tag];
            return td.dict[unpackKey(enc.data().data() + td.keysOffset, rank, td.keyBits)];
        }
        const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
        T v; std::memcpy(&v, fp + rank, sizeof(T));
        return v;
    }

    std::optional<T> decodeAtEliasFano(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, size_t i) const {
        for (const auto& td : h.tiers) {
            auto it = std::lower_bound(td.positions.begin(), td.positions.end(), static_cast<uint32_t>(i));
            if (it != td.positions.end() && *it == i) {
                const size_t rank = static_cast<size_t>(it - td.positions.begin());
                return td.dict[unpackKey(enc.data().data() + td.keysOffset, rank, td.keyBits)];
            }
        }
        const size_t fallbackRank = fallbackRankAt(h, i);
        const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
        T v; std::memcpy(&v, fp + fallbackRank, sizeof(T));
        return v;
    }

    std::optional<T> decodeAtNoIndex(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, size_t i) const {
        if (i < h.tierPrefix.back()) {
            const auto it = std::upper_bound(h.tierPrefix.begin(), h.tierPrefix.end(), i);
            const size_t t = static_cast<size_t>(it - h.tierPrefix.begin() - 1);
            const size_t rank = i - h.tierPrefix[t];
            const auto& td = h.tiers[t];
            return td.dict[unpackKey(enc.data().data() + td.keysOffset, rank, td.keyBits)];
        }
        const size_t fi = i - h.tierPrefix.back();
        const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
        T v; std::memcpy(&v, fp + fi, sizeof(T));
        return v;
    }

    void decodeRangePerTierBitmapsInto(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h,
                                       size_t start, size_t end, T* dst) const {
        const size_t N      = static_cast<size_t>(h.numElements);
        const size_t wStart = start / 64;
        const size_t wEnd   = (end + 63) / 64;  // exclusive; shared by tier + fallback loops

        // Word-based scan: iterate only over 64-bit words that have set bits,
        // using __builtin_ctzll to find each set position.  This eliminates the
        // per-element unpredictable branch that per-position testing would cause
        // (particularly severe for uint16_t where tier coverage is ~50/50).
        for (const auto& td : h.tiers) {
            size_t rank = popcountPrefixFast(td, start);
            const uint8_t* keysBase = enc.data().data() + td.keysOffset;
            for (size_t w = wStart; w < wEnd; ++w) {
                uint64_t word = td.bitmap[w];
                if (w == wStart && (start & 63))  word &= ~((uint64_t{1} << (start & 63)) - 1);
                if (w + 1 == wEnd && (end & 63))   word &=  (uint64_t{1} << (end   & 63)) - 1;
                while (word) {
                    const size_t bit = static_cast<size_t>(__builtin_ctzll(word));
                    dst[w * 64 + bit - start] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
                    ++rank;
                    word &= word - 1;
                }
            }
        }
        if (h.fallbackCount > 0) {
            size_t fi = fallbackRankAt(h, start);
            const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
            for (size_t w = wStart; w < wEnd; ++w) {
                uint64_t uncov = ~h.coveredBitmap[w];
                if (w == wStart && (start & 63))  uncov &= ~((uint64_t{1} << (start & 63)) - 1);
                // Combine end-of-range mask and N-boundary mask for the last word:
                const size_t endBit = (w + 1 == wEnd       && (end & 63)) ? (end & 63) : 64;
                const size_t nBit   = (w + 1 == h.numWords && (N   & 63)) ? (N   & 63) : 64;
                if (const size_t clamp = std::min(endBit, nBit); clamp < 64)
                    uncov &= (uint64_t{1} << clamp) - 1;
                while (uncov) {
                    const size_t bit = static_cast<size_t>(__builtin_ctzll(uncov));
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    dst[w * 64 + bit - start] = v;
                    ++fi;
                    uncov &= uncov - 1;
                }
            }
        }
    }

    std::vector<T> decodeRangePerTierBitmaps(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, size_t start, size_t end) const {
        std::vector<T> result(end - start);
        decodeRangePerTierBitmapsInto(enc, h, start, end, result.data());
        return result;
    }

    void decodeRangeTierTagArrayInto(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h,
                                     size_t start, size_t end, T* dst) const {
        const uint8_t* tagBase = enc.data().data() + h.tagArrayOffset;
        const uint8_t  tagBits = h.tagBits;

        std::vector<size_t> tierBitPos(h.numTiers, 0);
        size_t fallbackRankAtStart = 0;

        alignas(64) uint8_t scratch[kTagChunkSize];

        {
            size_t bitCursor = 0;
            for (size_t j = 0; j < start; ) {
                const size_t chunk = std::min<size_t>(kTagChunkSize, start - j);
                unpackTagsInto(tagBase, bitCursor, tagBits, chunk, scratch);
                bitCursor += chunk * tagBits;
                for (size_t k = 0; k < chunk; ++k) {
                    const uint8_t t = scratch[k];
                    if (t < h.numTiers) tierBitPos[t] += h.tiers[t].keyBits;
                    else                ++fallbackRankAtStart;
                }
                j += chunk;
            }
        }

        std::vector<const uint8_t*> tierKeysBase(h.numTiers, nullptr);
        for (size_t t = 0; t < h.numTiers; ++t)
            tierKeysBase[t] = enc.data().data() + h.tiers[t].keysOffset;
        const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
        size_t fi = fallbackRankAtStart;

        size_t tagBitCursor = start * tagBits;
        size_t outIdx = 0;
        for (size_t pos = start; pos < end; ) {
            const size_t chunk = std::min<size_t>(kTagChunkSize, end - pos);
            unpackTagsInto(tagBase, tagBitCursor, tagBits, chunk, scratch);
            tagBitCursor += chunk * tagBits;

            for (size_t i = 0; i < chunk; ++i, ++pos, ++outIdx) {
                const uint8_t tag = scratch[i];
                if (tag < h.numTiers) {
                    const auto& td = h.tiers[tag];
                    const uint8_t  kb    = td.keyBits;
                    const size_t   kBp   = tierBitPos[tag];
                    const size_t   kBy   = kBp >> 3;
                    const size_t   kBo   = kBp & 7;
                    const uint32_t kMask = (kb == 32) ? ~0u : ((1u << kb) - 1u);
                    uint64_t kBuf = 0;
                    std::memcpy(&kBuf, tierKeysBase[tag] + kBy, (kBo + kb + 7) >> 3);
                    dst[outIdx] = td.dict[static_cast<uint32_t>((kBuf >> kBo) & kMask)];
                    tierBitPos[tag] += kb;
                } else {
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    dst[outIdx] = v;
                    ++fi;
                }
            }
        }
    }

    std::vector<T> decodeRangeTierTagArray(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, size_t start, size_t end) const {
        std::vector<T> result(end - start);
        decodeRangeTierTagArrayInto(enc, h, start, end, result.data());
        return result;
    }

    void decodeRangeEliasFanoInto(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h,
                                  size_t start, size_t end, T* dst) const {
        for (const auto& td : h.tiers) {
            auto it = std::lower_bound(td.positions.begin(), td.positions.end(), static_cast<uint32_t>(start));
            size_t rank = static_cast<size_t>(it - td.positions.begin());
            const uint8_t* keysBase = enc.data().data() + td.keysOffset;
            for (; it != td.positions.end() && *it < end; ++it, ++rank) {
                dst[*it - start] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
            }
        }
        if (h.fallbackCount > 0) {
            const size_t N      = static_cast<size_t>(h.numElements);
            const size_t wStart = start / 64;
            const size_t wEnd   = (end + 63) / 64;
            size_t fi = fallbackRankAt(h, start);
            const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
            for (size_t w = wStart; w < wEnd; ++w) {
                uint64_t uncov = ~h.coveredBitmap[w];
                if (w == wStart && (start & 63))  uncov &= ~((uint64_t{1} << (start & 63)) - 1);
                const size_t endBit = (w + 1 == wEnd       && (end & 63)) ? (end & 63) : 64;
                const size_t nBit   = (w + 1 == h.numWords && (N   & 63)) ? (N   & 63) : 64;
                if (const size_t clamp = std::min(endBit, nBit); clamp < 64)
                    uncov &= (uint64_t{1} << clamp) - 1;
                while (uncov) {
                    const size_t bit = static_cast<size_t>(__builtin_ctzll(uncov));
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    dst[w * 64 + bit - start] = v;
                    ++fi;
                    uncov &= uncov - 1;
                }
            }
        }
    }

    std::vector<T> decodeRangeEliasFano(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, size_t start, size_t end) const {
        std::vector<T> result(end - start);
        decodeRangeEliasFanoInto(enc, h, start, end, result.data());
        return result;
    }

    void decodeRangeNoIndexInto(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h,
                                size_t start, size_t end, T* dst) const {
        const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);

        auto it = std::upper_bound(h.tierPrefix.begin(), h.tierPrefix.end(), start);
        size_t t = static_cast<size_t>(it - h.tierPrefix.begin() - 1);

        for (size_t idx = start, outIdx = 0; idx < end; ++idx, ++outIdx) {
            while (t + 1 < h.tiers.size() && idx >= h.tierPrefix[t + 1]) ++t;

            if (idx < h.tierPrefix.back()) {
                const auto& td = h.tiers[t];
                const size_t rank = idx - h.tierPrefix[t];
                dst[outIdx] = td.dict[unpackKey(enc.data().data() + td.keysOffset, rank, td.keyBits)];
            } else {
                T v; std::memcpy(&v, fp + (idx - h.tierPrefix.back()), sizeof(T));
                dst[outIdx] = v;
            }
        }
    }

    std::vector<T> decodeRangeNoIndex(const EncodedBuffer<uint8_t>& enc, const ParsedHeader& h, size_t start, size_t end) const {
        std::vector<T> result(end - start);
        decodeRangeNoIndexInto(enc, h, start, end, result.data());
        return result;
    }

public:
    // ---------------------------------------------------------------------------
    // Decode all
    // ---------------------------------------------------------------------------

    void decodeAllInto(const EncodedBuffer<uint8_t>& enc, T* dst, size_t n) override {
        const ParsedHeader& h = getParsedHeader(enc);
        if (h.numElements != n) [[unlikely]]
            throw std::runtime_error("FrequencyPartitionEncoder::decodeAllInto: size mismatch");
        if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps) {
            decodeAllPerTierBitmapsInto(enc, h, dst);
        } else if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
            decodeAllTierTagArrayInto(enc, h, dst);
        } else if constexpr (IndexType == FreqPartIndexType::EliasFano) {
            decodeAllEliasFanoInto(enc, h, dst);
        } else {
            decodeAllNoIndexInto(enc, h, dst);
        }
    }

    std::vector<T> decodeAll(const EncodedBuffer<uint8_t>& enc) override {
        const ParsedHeader& h = getParsedHeader(enc);
        if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps) {
            return decodeAllPerTierBitmaps(enc, h);
        } else if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
            return decodeAllTierTagArray(enc, h);
        } else if constexpr (IndexType == FreqPartIndexType::EliasFano) {
            return decodeAllEliasFano(enc, h);
        } else {
            return decodeAllNoIndex(enc, h);
        }
    }

    // ---------------------------------------------------------------------------
    // Decode at index i
    // ---------------------------------------------------------------------------

    std::optional<T> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t i) override {
        const ParsedHeader& h = getParsedHeader(enc);
        if (i >= h.numElements) return std::nullopt;
        if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps) {
            return decodeAtPerTierBitmaps(enc, h, i);
        } else if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
            return decodeAtTierTagArray(enc, h, i);
        } else if constexpr (IndexType == FreqPartIndexType::EliasFano) {
            return decodeAtEliasFano(enc, h, i);
        } else {
            return decodeAtNoIndex(enc, h, i);
        }
    }

    // ---------------------------------------------------------------------------
    // Decode range [start, end)
    // ---------------------------------------------------------------------------

    void decodeRangeInto(const EncodedBuffer<uint8_t>& enc,
                         size_t start, size_t end,
                         T* dst, size_t n) override {
        const ParsedHeader& h = getParsedHeader(enc);
        const size_t N = static_cast<size_t>(h.numElements);
        end = std::min(end, N);
        if (start >= end || (end - start) != n) [[unlikely]]
            throw std::runtime_error("FrequencyPartitionEncoder::decodeRangeInto: size mismatch");
        if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps) {
            decodeRangePerTierBitmapsInto(enc, h, start, end, dst);
        } else if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
            decodeRangeTierTagArrayInto(enc, h, start, end, dst);
        } else if constexpr (IndexType == FreqPartIndexType::EliasFano) {
            decodeRangeEliasFanoInto(enc, h, start, end, dst);
        } else {
            decodeRangeNoIndexInto(enc, h, start, end, dst);
        }
    }

    std::vector<T> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) override {
        const ParsedHeader& h = getParsedHeader(enc);
        const size_t N = static_cast<size_t>(h.numElements);
        if (start >= end || start >= N) return {};
        end = std::min(end, N);

        if constexpr (IndexType == FreqPartIndexType::PerTierBitmaps) {
            return decodeRangePerTierBitmaps(enc, h, start, end);
        } else if constexpr (IndexType == FreqPartIndexType::TierTagArray) {
            return decodeRangeTierTagArray(enc, h, start, end);
        } else if constexpr (IndexType == FreqPartIndexType::EliasFano) {
            return decodeRangeEliasFano(enc, h, start, end);
        } else {
            return decodeRangeNoIndex(enc, h, start, end);
        }
    }

    // ---------------------------------------------------------------------------
    // Metadata
    // ---------------------------------------------------------------------------

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::FrequencyPartitionEncoding;
    }

    std::string name() const override { return "FrequencyPartitionEncoder"; }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::Lossless)
             | EncodingProperty::RequiresFullData
             | EncodingProperty::VariableSize
             | EncodingProperty::RandomAccess;
    }
};

} // namespace encodings::encoders
