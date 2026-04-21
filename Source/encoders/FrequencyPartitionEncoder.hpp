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
template <typename T>
    requires (std::is_same_v<T, uint8_t>  || std::is_same_v<T, uint16_t> ||
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
              std::is_same_v<T, int8_t>   || std::is_same_v<T, int16_t>  ||
              std::is_same_v<T, int32_t>  || std::is_same_v<T, int64_t>)
class FrequencyPartitionEncoder : public Codec<T> {
public:
    using EncodedData = EncodedBuffer<uint8_t>;

    static constexpr size_t kMaxKeyBits     = sizeof(T) * 4; // half storage width in bits
    static constexpr size_t kMaxTierDefs    = 32;            // upper bound on distinct key widths

    // Number of active tiers for this T: key widths 1..kMaxKeyBits (inclusive).
    // With non-power-of-2 widths the tier count equals kMaxKeyBits since we use
    // widths 1,2,...,kMaxKeyBits. Each tier's actual capacity is 2^keyBits.
    static constexpr size_t kNumActiveTiers = kMaxKeyBits;

    explicit FrequencyPartitionEncoder(
        FreqPartIndexType indexType = FreqPartIndexType::PerTierBitmaps)
        : indexType_(indexType) {}

    // ---------------------------------------------------------------------------
    // Encode
    // ---------------------------------------------------------------------------

    EncodedBuffer<uint8_t> encode(std::span<const T> input) override {
        const size_t N = input.size();

        // Count symbol frequencies.
        ankerl::unordered_dense::map<T, uint64_t> freq;
        freq.reserve(N < 1024 ? N : 1024);
        for (const T v : input) ++freq[v];

        // Sort by frequency descending (ties broken by value for determinism).
        std::vector<std::pair<T, uint64_t>> sortedFreq(freq.begin(), freq.end());
        std::sort(sortedFreq.begin(), sortedFreq.end(),
                  [](const auto& a, const auto& b) {
                      return a.second != b.second ? a.second > b.second : a.first < b.first;
                  });

        const size_t numUnique = sortedFreq.size();

        // Assign values to tiers with non-power-of-2 key widths.
        // Tier t uses exactly ceil(log2(t_dictSize)) bits for its keys,
        // where t_dictSize is however many values are assigned to it.
        // We fill tiers greedily: tier 0 gets up to 2^1=2 values, tier 1 up to
        // 2^2=4 values, ..., tier (kb-1) up to 2^kb values (kb = 1..kMaxKeyBits).
        // valueInfo maps value → (tierIdx, key).
        ankerl::unordered_dense::map<T, std::pair<uint8_t, uint32_t>> valueInfo;
        valueInfo.reserve(numUnique);

        struct TierMeta {
            uint8_t keyBits{0};
            std::vector<T> dict; // dict[key] = value
        };
        TierMeta tiers[kNumActiveTiers];
        size_t assigned = 0;

        for (size_t t = 0; t < kNumActiveTiers && assigned < numUnique; ++t) {
            // Tier t's capacity (using standard 1-indexed key width t+1 bits).
            const size_t cap = static_cast<size_t>(1) << (t + 1);
            const size_t toAssign = std::min(cap, numUnique - assigned);
            tiers[t].dict.reserve(toAssign);
            for (size_t i = 0; i < toAssign; ++i) {
                const T val = sortedFreq[assigned + i].first;
                tiers[t].dict.push_back(val);
                valueInfo[val] = {static_cast<uint8_t>(t), static_cast<uint32_t>(i)};
            }
            // Set key width to ceil(log2(toAssign)), minimum 1 bit.
            tiers[t].keyBits = static_cast<uint8_t>(
                toAssign <= 1 ? 1
                              : static_cast<uint8_t>(sizeof(size_t) * 8
                                                     - static_cast<size_t>(__builtin_clzll(toAssign - 1))));
            assigned += toAssign;
        }

        // Bitmap and key lists per tier.
        const size_t numWords = (N + 63) / 64;
        std::vector<std::vector<uint64_t>> bitmaps(kNumActiveTiers,
                                                    std::vector<uint64_t>(numWords, 0));
        std::vector<std::vector<uint32_t>> tierKeys(kNumActiveTiers);
        std::vector<std::vector<uint32_t>> tierPositions(kNumActiveTiers);
        std::vector<T> fallback;

        for (size_t pos = 0; pos < N; ++pos) {
            const T val = input[pos];
            auto it = valueInfo.find(val);
            if (it != valueInfo.end()) {
                const auto [t, key] = it->second;
                bitmaps[t][pos / 64] |= (uint64_t{1} << (pos % 64));
                tierKeys[t].push_back(key);
                tierPositions[t].push_back(static_cast<uint32_t>(pos));
            } else {
                fallback.push_back(val);
            }
        }

        // Optimal tier selection: include a tier only when its savings exceed its cost.
        // Cost of including tier t: bitmap (numWords*64 bits) + dict + keys.
        // Savings: tier elements no longer stored as raw fallback values.
        // Values from excluded tiers are demoted to fallback.
        std::vector<bool> includeTier(kNumActiveTiers, false);
        for (size_t t = 0; t < kNumActiveTiers; ++t) {
            const size_t dSize = tiers[t].dict.size();
            if (dSize == 0) continue;
            const size_t count = tierKeys[t].size();
            const uint8_t kb   = tiers[t].keyBits;
            const size_t cost    = numWords * 64
                                 + dSize * sizeof(T) * 8
                                 + (count * kb + 7) / 8 * 8;
            const size_t savings = count * sizeof(T) * 8;
            if (savings > cost) includeTier[t] = true;
        }

        // Rebuild fallback to include elements from excluded tiers (preserve position order).
        // We need to re-scan input to get original order correct.
        {
            // Check if any tier was excluded that had elements.
            bool anyExcluded = false;
            for (size_t t = 0; t < kNumActiveTiers; ++t)
                if (!includeTier[t] && !tiers[t].dict.empty()) { anyExcluded = true; break; }

            if (anyExcluded) {
                // Rebuild fallback in position order.
                fallback.clear();
                for (size_t pos = 0; pos < N; ++pos) {
                    const T val = input[pos];
                    auto it = valueInfo.find(val);
                    if (it == valueInfo.end()) {
                        fallback.push_back(val);
                    } else {
                        const uint8_t t = it->second.first;
                        if (!includeTier[t]) fallback.push_back(val);
                    }
                }
                // Clear excluded tier key lists and bitmaps.
                for (size_t t = 0; t < kNumActiveTiers; ++t) {
                    if (!includeTier[t]) {
                        tierKeys[t].clear();
                        tierPositions[t].clear();
                        std::fill(bitmaps[t].begin(), bitmaps[t].end(), 0);
                    }
                }
            }
        }

        // Count tiers with data to emit.
        uint8_t numTiersWithData = 0;
        for (size_t t = 0; t < kNumActiveTiers; ++t)
            if (includeTier[t] && !tiers[t].dict.empty()) ++numTiersWithData;

        // Serialise.
        std::vector<uint8_t> out;
        out.reserve(N * sizeof(T));

        const auto appendBytes = [&](const void* src, size_t n) {
            const auto* p = static_cast<const uint8_t*>(src);
            out.insert(out.end(), p, p + n);
        };
        const auto appendT = [&](auto v) { appendBytes(&v, sizeof(v)); };

        appendT(static_cast<uint64_t>(N));
        appendT(numTiersWithData);
        appendT(static_cast<uint8_t>(indexType_));

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
            (indexType_ == FreqPartIndexType::TierTagArray)
                ? ((N * ceilLog2WithMinOne(static_cast<uint32_t>(numTiersWithData) + 1u) + 7) / 8)
                : 0;

        if (indexType_ == FreqPartIndexType::PerTierBitmaps) {
            for (size_t t = 0; t < kNumActiveTiers; ++t) {
                if (!includeTier[t] || tiers[t].dict.empty()) continue;
                const uint8_t kb    = tiers[t].keyBits;
                const auto&   dict  = tiers[t].dict;
                const auto&   keys  = tierKeys[t];
                const size_t dictBytes = dict.size() * sizeof(T);
                const size_t keyBytes  = packedKeyBytes(keys.size(), kb);
                const size_t bitmapBytes = numWords * sizeof(uint64_t);
                appendT(kb);
                appendT(static_cast<uint32_t>(dict.size()));
                for (const T v : dict) appendT(v);
                for (size_t w = 0; w < numWords; ++w) appendT(bitmaps[t][w]);
                packKeys(out, keys, kb);

                tierLogs.push_back(TierLogInfo{
                    static_cast<uint8_t>(t + 1),
                    kb,
                    dict.size(),
                    keys.size(),
                    dictBytes,
                    keyBytes,
                    bitmapBytes,
                });
            }
        } else if (indexType_ == FreqPartIndexType::TierTagArray) {
            // TierTagArray: emit the tag array first, then per-tier dict+keys.
            // tagBits = ceil(log2(numTiersWithData + 1)), minimum 1.
            const uint8_t tagBits = ceilLog2WithMinOne(static_cast<uint32_t>(numTiersWithData) + 1u);

            // Build a tier-index remap: active tier index → 0-based output index.
            std::vector<uint8_t> tierRemap(kNumActiveTiers, 0xFF);
            uint8_t remapIdx = 0;
            for (size_t t = 0; t < kNumActiveTiers; ++t)
                if (includeTier[t] && !tiers[t].dict.empty())
                    tierRemap[t] = remapIdx++;

            // Pack tag array (tag = remapped tier idx, or numTiersWithData for fallback).
            const size_t tagBytes = (N * tagBits + 7) / 8;
            const size_t tagBase  = out.size();
            out.resize(out.size() + tagBytes, 0);
            for (size_t pos = 0; pos < N; ++pos) {
                const T val = input[pos];
                auto it = valueInfo.find(val);
                uint8_t tag = numTiersWithData; // fallback
                if (it != valueInfo.end()) {
                    const uint8_t t = it->second.first;
                    if (includeTier[t]) tag = tierRemap[t];
                }
                packSingleTag(out, tagBase, pos, tagBits, tag);
            }

            // Per-tier dict and keys.
            for (size_t t = 0; t < kNumActiveTiers; ++t) {
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

                tierLogs.push_back(TierLogInfo{
                    static_cast<uint8_t>(t + 1),
                    kb,
                    dict.size(),
                    keys.size(),
                    dictBytes,
                    keyBytes,
                    0,
                });
            }
        } else if (indexType_ == FreqPartIndexType::EliasFano) {
            for (size_t t = 0; t < kNumActiveTiers; ++t) {
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
                const uint32_t lowMask = (lowBits == 32) ? 0xFFFFFFFFu : ((lowBits == 0) ? 0u : ((1u << lowBits) - 1u));
                for (uint32_t ppos : posv) lows.push_back(ppos & lowMask);
                const size_t lowBytes = packedKeyBytes(lows.size(), lowBits);

                const size_t highBitsLen = ((N >> lowBits) + posv.size() + 1);
                const size_t highWordCount = (highBitsLen + 63) / 64;
                std::vector<uint64_t> highBits(highWordCount, 0);
                for (size_t i = 0; i < posv.size(); ++i) {
                    const size_t hi = static_cast<size_t>(posv[i] >> lowBits);
                    const size_t bitPos = hi + i;
                    highBits[bitPos / 64] |= (uint64_t{1} << (bitPos % 64));
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

                tierLogs.push_back(TierLogInfo{
                    static_cast<uint8_t>(t + 1),
                    kb,
                    dict.size(),
                    keys.size(),
                    dictBytes,
                    keyBytes,
                    static_cast<size_t>(4 + 1 + 4 + 4) + lowBytes + highWordCount * sizeof(uint64_t),
                });
            }
        } else {
            // NoIndex: no positional index emitted; rows decode in reordered
            // stream order (all values for tier0, then tier1, ..., fallback).
            for (size_t t = 0; t < kNumActiveTiers; ++t) {
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

                tierLogs.push_back(TierLogInfo{
                    static_cast<uint8_t>(t + 1),
                    kb,
                    dict.size(),
                    keys.size(),
                    dictBytes,
                    keyBytes,
                    0,
                });
            }
        }

        if (fallback.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
            throw std::runtime_error("FrequencyPartitionEncoder: fallback count exceeds uint32 limit");
        appendT(static_cast<uint32_t>(fallback.size()));
        for (const T v : fallback) appendT(v);

        EncodedBuffer<uint8_t> result;
        result.metadata().elementCount         = N;
        result.metadata().encodingName         = name();
        result.metadata().supportsRandomAccess = true;
        result.metadata().compressedSize       = out.size();
        result.metadata().uncompressedSize     = N * sizeof(T);
        result.data() = std::move(out);

        if (verboseEnabled()) {
            const size_t rawBytes = N * sizeof(T);
            const size_t headerBytes = 8 + 1 + 1;
            const size_t fallbackBytes = 4 + fallback.size() * sizeof(T);
            size_t totalIndexBytes = 0;
            if (indexType_ == FreqPartIndexType::PerTierBitmaps) {
                totalIndexBytes = numTiersWithData * numWords * sizeof(uint64_t);
            } else if (indexType_ == FreqPartIndexType::TierTagArray) {
                totalIndexBytes = tagBytes;
            } else if (indexType_ == FreqPartIndexType::EliasFano) {
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
                      << (indexType_ == FreqPartIndexType::PerTierBitmaps ? "PerTierBitmaps"
                          : indexType_ == FreqPartIndexType::TierTagArray ? "TierTagArray"
                          : indexType_ == FreqPartIndexType::EliasFano ? "EliasFano"
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
        return v || true;
    }

    FreqPartIndexType indexType_{FreqPartIndexType::PerTierBitmaps};

    struct ParsedHeader {
        uint64_t numElements{0};
        uint8_t  numTiers{0};
        FreqPartIndexType indexType{FreqPartIndexType::PerTierBitmaps};
        size_t   numWords{0};

        struct TierData {
            uint8_t  keyBits{0};
            std::vector<T>        dict;
            std::vector<uint64_t> bitmap;   // only for PerTierBitmaps
            std::vector<uint32_t> positions; // for EliasFano
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
        h.numWords    = (h.numElements + 63) / 64;

        h.tiers.resize(h.numTiers);

        if (h.indexType == FreqPartIndexType::PerTierBitmaps) {
            for (auto& td : h.tiers) {
                td.keyBits = readU8();
                const uint32_t dictCount = readU32();
                td.dict.resize(dictCount);
                for (auto& v : td.dict) v = readT();

                td.bitmap.resize(h.numWords);
                for (auto& w : td.bitmap) { std::memcpy(&w, p, 8); p += 8; }

                td.tierCount = 0;
                for (uint64_t w : td.bitmap) td.tierCount += static_cast<size_t>(__builtin_popcountll(w));

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
        } else if (h.indexType == FreqPartIndexType::TierTagArray) {
            // TierTagArray: tag array comes first.
            h.tagBits = ceilLog2WithMinOne(static_cast<uint32_t>(h.numTiers) + 1u);
            h.tagArrayOffset = static_cast<size_t>(p - enc.data().data());
            const size_t tagBytes = (h.numElements * h.tagBits + 7) / 8;
            p += tagBytes;

            // Per-tier dict and keys (no bitmaps stored).
            for (auto& td : h.tiers) {
                td.keyBits = readU8();
                const uint32_t dictCount = readU32();
                td.dict.resize(dictCount);
                for (auto& v : td.dict) v = readT();
                // tierCount determined by scanning tag array — defer; we use keysOffset.
                td.keysOffset = static_cast<size_t>(p - enc.data().data());
                // We don't know tierCount yet without scanning tags; we'll compute it lazily.
                // Instead, store keysOffset and use a two-pass approach in decode methods.
                // For now, scan tags to count.
                const uint8_t thisTierIdx = static_cast<uint8_t>(&td - h.tiers.data());
                td.tierCount = 0;
                const uint8_t* tagBase = enc.data().data() + h.tagArrayOffset;
                for (size_t i = 0; i < h.numElements; ++i) {
                    if (unpackTagAt(tagBase, i, h.tagBits) == thisTierIdx) ++td.tierCount;
                }
                p += packedKeyBytes(td.tierCount, td.keyBits);
            }
        } else if (h.indexType == FreqPartIndexType::EliasFano) {
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

            h.coveredBitmap.assign(h.numWords, 0);
            for (const auto& td : h.tiers) {
                for (uint32_t pos : td.positions) {
                    h.coveredBitmap[pos / 64] |= (uint64_t{1} << (pos % 64));
                }
            }

            h.fallbackPrefixPop.resize(h.numWords + 1, 0);
            for (size_t w = 0; w < h.numWords; ++w) {
                uint64_t uncov = ~h.coveredBitmap[w];
                if (w == h.numWords - 1 && (h.numElements % 64) != 0)
                    uncov &= (uint64_t{1} << (h.numElements % 64)) - 1;
                h.fallbackPrefixPop[w + 1] = h.fallbackPrefixPop[w]
                                           + static_cast<size_t>(__builtin_popcountll(uncov));
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
    static uint32_t unpackKey(const uint8_t* base, size_t r, uint8_t keyBits) {
        const size_t  bitPos  = r * keyBits;
        const size_t  byteIdx = bitPos / 8;
        const size_t  bitOff  = bitPos % 8;
        const uint32_t mask   = (keyBits == 32) ? ~0u : ((1u << keyBits) - 1u);

        // Read up to 5 bytes to cover any key width ≤ 32 bits crossing byte boundaries.
        uint64_t buf = 0;
        const size_t bytesNeeded = (bitOff + keyBits + 7) / 8;
        for (size_t b = 0; b < bytesNeeded; ++b)
            buf |= static_cast<uint64_t>(base[byteIdx + b]) << (b * 8);
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

    // Popcount of all bits in `bitmap` strictly before position `i`.
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

    // Compute fallback rank before position `i` using the precomputed prefix table.
    static size_t fallbackRankAt(const ParsedHeader& h, size_t i) {
        size_t rank      = h.fallbackPrefixPop[i / 64];
        uint64_t uncov   = ~h.coveredBitmap[i / 64];
        rank += static_cast<size_t>(__builtin_popcountll(uncov & ((uint64_t{1} << (i % 64)) - 1)));
        return rank;
    }

    // ---------------------------------------------------------------------------
    // Decode all
    // ---------------------------------------------------------------------------

public:
    std::vector<T> decodeAll(const EncodedBuffer<uint8_t>& enc) override {
        const ParsedHeader& h = getParsedHeader(enc);
        const size_t N = static_cast<size_t>(h.numElements);
        std::vector<T> out(N);

        if (h.indexType == FreqPartIndexType::PerTierBitmaps) {
            // Tier passes: iterate set bits in each tier's bitmap.
            for (const auto& td : h.tiers) {
                const uint8_t* keysBase = enc.data().data() + td.keysOffset;
                size_t rank = 0;
                for (size_t w = 0; w < h.numWords; ++w) {
                    uint64_t word = td.bitmap[w];
                    while (word) {
                        const size_t bit = static_cast<size_t>(__builtin_ctzll(word));
                        const size_t pos = w * 64 + bit;
                        if (pos < N)
                            out[pos] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
                        ++rank;
                        word &= word - 1;
                    }
                }
            }

            // Fallback pass: iterate uncovered positions via ~coveredBitmap.
            if (h.fallbackCount > 0) {
                const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
                size_t fi = 0;
                for (size_t w = 0; w < h.numWords; ++w) {
                    uint64_t uncov = ~h.coveredBitmap[w];
                    // Mask bits beyond numElements in the last word.
                    if (w == h.numWords - 1 && (N % 64) != 0)
                        uncov &= (uint64_t{1} << (N % 64)) - 1;
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
        } else if (h.indexType == FreqPartIndexType::TierTagArray) {
            // TierTagArray: single linear scan of tag array.
            const uint8_t* tagBase = enc.data().data() + h.tagArrayOffset;
            // Per-tier key readers: track current rank per tier.
            std::vector<size_t> tierRank(h.numTiers, 0);
            std::vector<const uint8_t*> tierKeysBase(h.numTiers, nullptr);
            for (size_t t = 0; t < h.numTiers; ++t)
                tierKeysBase[t] = enc.data().data() + h.tiers[t].keysOffset;

            const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
            size_t fi = 0;
            for (size_t pos = 0; pos < N; ++pos) {
                const uint8_t tag = unpackTagAt(tagBase, pos, h.tagBits);
                if (tag < h.numTiers) {
                    const auto& td = h.tiers[tag];
                    out[pos] = td.dict[unpackKey(tierKeysBase[tag], tierRank[tag], td.keyBits)];
                    ++tierRank[tag];
                } else {
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    out[pos] = v;
                    ++fi;
                }
            }
        } else if (h.indexType == FreqPartIndexType::EliasFano) {
            for (const auto& td : h.tiers) {
                const uint8_t* keysBase = enc.data().data() + td.keysOffset;
                for (size_t rank = 0; rank < td.positions.size(); ++rank) {
                    const size_t pos = td.positions[rank];
                    if (pos < N) {
                        out[pos] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
                    }
                }
            }

            if (h.fallbackCount > 0) {
                const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
                size_t fi = 0;
                for (size_t w = 0; w < h.numWords; ++w) {
                    uint64_t uncov = ~h.coveredBitmap[w];
                    if (w == h.numWords - 1 && (N % 64) != 0)
                        uncov &= (uint64_t{1} << (N % 64)) - 1;
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
        } else {
            // NoIndex: decoded order is tier0..tierK then fallback.
            out.clear();
            out.reserve(N);
            for (const auto& td : h.tiers) {
                const uint8_t* keysBase = enc.data().data() + td.keysOffset;
                for (size_t rank = 0; rank < td.tierCount; ++rank) {
                    out.push_back(td.dict[unpackKey(keysBase, rank, td.keyBits)]);
                }
            }
            const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
            for (size_t i = 0; i < h.fallbackCount; ++i) {
                T v; std::memcpy(&v, fp + i, sizeof(T));
                out.push_back(v);
            }
        }

        return out;
    }

    // ---------------------------------------------------------------------------
    // Decode at index i
    // ---------------------------------------------------------------------------

    std::optional<T> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t i) override {
        const ParsedHeader& h = getParsedHeader(enc);
        if (i >= h.numElements) return std::nullopt;

        if (h.indexType == FreqPartIndexType::PerTierBitmaps) {
            for (const auto& td : h.tiers) {
                if (td.bitmap[i / 64] & (uint64_t{1} << (i % 64))) {
                    const size_t rank = popcountPrefix(td.bitmap, i);
                    return td.dict[unpackKey(enc.data().data() + td.keysOffset, rank, td.keyBits)];
                }
            }
            // Fallback: use precomputed prefix table — O(numWords) instead of O(i×numTiers).
            const size_t fallbackRank = fallbackRankAt(h, i);
            const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
            T v; std::memcpy(&v, fp + fallbackRank, sizeof(T));
            return v;
        } else if (h.indexType == FreqPartIndexType::TierTagArray) {
            // TierTagArray: read tag at position i, then scan to compute rank.
            const uint8_t* tagBase = enc.data().data() + h.tagArrayOffset;
            const uint8_t  tag     = unpackTagAt(tagBase, i, h.tagBits);
            if (tag < h.numTiers) {
                const auto& td = h.tiers[tag];
                // Rank = number of positions < i with the same tag.
                size_t rank = 0;
                for (size_t j = 0; j < i; ++j)
                    if (unpackTagAt(tagBase, j, h.tagBits) == tag) ++rank;
                return td.dict[unpackKey(enc.data().data() + td.keysOffset, rank, td.keyBits)];
            } else {
                size_t fi = 0;
                for (size_t j = 0; j < i; ++j)
                    if (unpackTagAt(tagBase, j, h.tagBits) >= h.numTiers) ++fi;
                const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
                T v; std::memcpy(&v, fp + fi, sizeof(T));
                return v;
            }
        } else if (h.indexType == FreqPartIndexType::EliasFano) {
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
        } else {
            // NoIndex: index is in reordered output space.
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
    }

    // ---------------------------------------------------------------------------
    // Decode range [start, end)
    // ---------------------------------------------------------------------------

    std::vector<T> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) override {
        const ParsedHeader& h = getParsedHeader(enc);
        const size_t N = static_cast<size_t>(h.numElements);
        if (start >= end || start >= N) return {};
        end = std::min(end, N);
        const size_t rangeLen = end - start;
        std::vector<T> out(rangeLen);

        if (h.indexType == FreqPartIndexType::PerTierBitmaps) {
            // Tier passes.
            for (const auto& td : h.tiers) {
                size_t rank = popcountPrefix(td.bitmap, start);
                const uint8_t* keysBase = enc.data().data() + td.keysOffset;
                for (size_t pos = start; pos < end; ++pos) {
                    if (td.bitmap[pos / 64] & (uint64_t{1} << (pos % 64))) {
                        out[pos - start] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
                        ++rank;
                    }
                }
            }

            // Fallback pass using precomputed prefix table.
            if (h.fallbackCount > 0) {
                const size_t fallbackRankStart = fallbackRankAt(h, start);
                const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
                size_t fi = fallbackRankStart;
                for (size_t pos = start; pos < end; ++pos) {
                    if (!(h.coveredBitmap[pos / 64] & (uint64_t{1} << (pos % 64)))) {
                        T v; std::memcpy(&v, fp + fi, sizeof(T));
                        out[pos - start] = v;
                        ++fi;
                    }
                }
            }
        } else if (h.indexType == FreqPartIndexType::TierTagArray) {
            // TierTagArray: scan tag array, track per-tier and fallback ranks.
            const uint8_t* tagBase = enc.data().data() + h.tagArrayOffset;

            // Compute ranks at `start` by scanning prefix.
            std::vector<size_t> tierRankAtStart(h.numTiers, 0);
            size_t fallbackRankAtStart = 0;
            for (size_t j = 0; j < start; ++j) {
                const uint8_t tag = unpackTagAt(tagBase, j, h.tagBits);
                if (tag < h.numTiers) ++tierRankAtStart[tag];
                else ++fallbackRankAtStart;
            }

            std::vector<const uint8_t*> tierKeysBase(h.numTiers, nullptr);
            for (size_t t = 0; t < h.numTiers; ++t)
                tierKeysBase[t] = enc.data().data() + h.tiers[t].keysOffset;
            const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);

            std::vector<size_t> tierRank = tierRankAtStart;
            size_t fi = fallbackRankAtStart;
            for (size_t pos = start; pos < end; ++pos) {
                const uint8_t tag = unpackTagAt(tagBase, pos, h.tagBits);
                if (tag < h.numTiers) {
                    const auto& td = h.tiers[tag];
                    out[pos - start] = td.dict[unpackKey(tierKeysBase[tag], tierRank[tag], td.keyBits)];
                    ++tierRank[tag];
                } else {
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    out[pos - start] = v;
                    ++fi;
                }
            }
        } else if (h.indexType == FreqPartIndexType::EliasFano) {
            for (const auto& td : h.tiers) {
                auto it = std::lower_bound(td.positions.begin(), td.positions.end(), static_cast<uint32_t>(start));
                size_t rank = static_cast<size_t>(it - td.positions.begin());
                const uint8_t* keysBase = enc.data().data() + td.keysOffset;
                for (; it != td.positions.end() && *it < end; ++it, ++rank) {
                    out[*it - start] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
                }
            }

            if (h.fallbackCount > 0) {
                const size_t fallbackRankStart = fallbackRankAt(h, start);
                const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
                size_t fi = fallbackRankStart;
                for (size_t pos = start; pos < end; ++pos) {
                    if (!(h.coveredBitmap[pos / 64] & (uint64_t{1} << (pos % 64)))) {
                        T v; std::memcpy(&v, fp + fi, sizeof(T));
                        out[pos - start] = v;
                        ++fi;
                    }
                }
            }
        } else {
            // NoIndex range in reordered output space.
            for (size_t idx = start; idx < end; ++idx) {
                if (idx < h.tierPrefix.back()) {
                    const auto it = std::upper_bound(h.tierPrefix.begin(), h.tierPrefix.end(), idx);
                    const size_t t = static_cast<size_t>(it - h.tierPrefix.begin() - 1);
                    const size_t rank = idx - h.tierPrefix[t];
                    const auto& td = h.tiers[t];
                    out[idx - start] = td.dict[unpackKey(enc.data().data() + td.keysOffset, rank, td.keyBits)];
                } else {
                    const size_t fi = idx - h.tierPrefix.back();
                    const T* fp = reinterpret_cast<const T*>(enc.data().data() + h.fallbackOffset + 4);
                    T v; std::memcpy(&v, fp + fi, sizeof(T));
                    out[idx - start] = v;
                }
            }
        }

        return out;
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
