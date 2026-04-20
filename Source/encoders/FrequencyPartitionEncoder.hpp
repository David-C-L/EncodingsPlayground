#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

/// Frequency-partition encoding with per-tier bitmaps for O(numTiers) random access.
///
/// The most-frequent values are assigned short fixed-width keys in ascending key-width
/// tiers (1, 2, 4, 8, 16, 32-bit). Active tiers are those whose key width is strictly
/// less than half the storage type width (matching reference getMaxKeyBits = sizeof(T)*4).
/// Each tier stores a dictionary, a 1-bit-per-element bitmap, and bit-packed keys.
/// Remaining values are stored raw in a fallback section.
///
/// Wire format:
///   [8]  numElements (uint64_t)
///   [1]  numTiers   (uint8_t)
///   Per tier:
///     [1]  keyBits    (uint8_t)
///     [4]  dictCount  (uint32_t)
///     [dictCount*sizeof(T)]  dictValues (most-frequent first, key 0 = top)
///     [ceil(numElements/64)*8]  bitmap (uint64_t words, bit i set iff pos i in this tier)
///     [ceil(tierCount*keyBits/8)]  bit-packed keys (LSB-first for ≤8-bit; aligned for 16/32)
///   [4]  fallbackCount (uint32_t)
///   [fallbackCount*sizeof(T)]  raw fallback values in original index order
///
template <typename T>
    requires (std::is_same_v<T, uint8_t>  || std::is_same_v<T, uint16_t> ||
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
class FrequencyPartitionEncoder : public Codec<T> {
public:
    using EncodedData = EncodedBuffer<uint8_t>;

    // All possible tier key widths (bits). Active tiers for a given T are those
    // where kTierKeyBits[t] < sizeof(T)*4 (i.e. key width < half storage width).
    static constexpr uint8_t kTierKeyBits[] = {1, 2, 4, 8, 16, 32};
    static constexpr size_t  kNumTierDefs   = 6;

    static constexpr size_t kMaxKeyBits = sizeof(T) * 4; // half storage type width in bits

    // Number of active tiers for this T (computed at compile time).
    static constexpr size_t kNumActiveTiers = []() constexpr {
        size_t n = 0;
        for (uint8_t kb : kTierKeyBits) if (kb < kMaxKeyBits) ++n;
        return n;
    }();

    // ---------------------------------------------------------------------------
    // Encode
    // ---------------------------------------------------------------------------

    EncodedBuffer<uint8_t> encode(std::span<const T> input) override {
        const size_t N = input.size();

        // Count symbol frequencies.
        ankerl::unordered_dense::map<T, uint32_t> freq;
        freq.reserve(N < 1024 ? N : 1024);
        for (const T v : input) ++freq[v];

        // Sort by frequency descending (ties broken by value for determinism).
        std::vector<std::pair<T, uint32_t>> sortedFreq(freq.begin(), freq.end());
        std::sort(sortedFreq.begin(), sortedFreq.end(),
                  [](const auto& a, const auto& b) {
                      return a.second != b.second ? a.second > b.second : a.first < b.first;
                  });

        const size_t numUnique = sortedFreq.size();

        // Assign values to tiers: fill each tier to its capacity (2^keyBits) in order.
        // valueInfo maps value → (tierIdx, key).
        ankerl::unordered_dense::map<T, std::pair<uint8_t, uint32_t>> valueInfo;
        valueInfo.reserve(numUnique);

        struct TierMeta {
            uint8_t keyBits;
            std::vector<T> dict; // dict[key] = value
        };
        TierMeta tiers[kNumActiveTiers];
        size_t assigned = 0;

        for (size_t t = 0; t < kNumActiveTiers && assigned < numUnique; ++t) {
            const uint8_t kb = kTierKeyBits[t];
            const size_t cap = static_cast<size_t>(1) << kb;
            tiers[t].keyBits = kb;
            const size_t toAssign = std::min(cap, numUnique - assigned);
            tiers[t].dict.reserve(toAssign);
            for (size_t i = 0; i < toAssign; ++i) {
                const T val = sortedFreq[assigned + i].first;
                tiers[t].dict.push_back(val);
                valueInfo[val] = {static_cast<uint8_t>(t), static_cast<uint32_t>(i)};
            }
            assigned += toAssign;
        }

        const size_t numActiveTiers = kNumActiveTiers; // tiers with keyBits < kMaxKeyBits
        // (some tiers may have empty dict if numUnique < total capacity — that is fine)

        // Bitmap: ceil(N/64) uint64_t words per tier.
        const size_t numWords = (N + 63) / 64;
        std::vector<std::vector<uint64_t>> bitmaps(numActiveTiers,
                                                    std::vector<uint64_t>(numWords, 0));

        // Per-tier key lists (unpacked, will be bit-packed later).
        std::vector<std::vector<uint32_t>> tierKeys(numActiveTiers);
        std::vector<T> fallback;

        for (size_t pos = 0; pos < N; ++pos) {
            const T val = input[pos];
            auto it = valueInfo.find(val);
            if (it != valueInfo.end()) {
                const auto [t, key] = it->second;
                bitmaps[t][pos / 64] |= (uint64_t{1} << (pos % 64));
                tierKeys[t].push_back(key);
            } else {
                fallback.push_back(val);
            }
        }

        // Determine how many tiers have at least one entry to write.
        uint8_t numTiersWithData = 0;
        for (size_t t = 0; t < numActiveTiers; ++t) {
            if (!tiers[t].dict.empty()) ++numTiersWithData;
        }

        // Serialise.
        std::vector<uint8_t> out;
        out.reserve(N * sizeof(T)); // rough upper bound

        const auto appendBytes = [&](const void* src, size_t n) {
            const auto* p = static_cast<const uint8_t*>(src);
            out.insert(out.end(), p, p + n);
        };
        const auto appendT = [&](auto v) {
            appendBytes(&v, sizeof(v));
        };

        appendT(static_cast<uint64_t>(N));
        appendT(numTiersWithData);

        for (size_t t = 0; t < numActiveTiers; ++t) {
            if (tiers[t].dict.empty()) continue;
            const uint8_t kb = tiers[t].keyBits;
            const auto& dict = tiers[t].dict;
            const auto& keys = tierKeys[t];

            appendT(kb);
            appendT(static_cast<uint32_t>(dict.size()));
            for (const T v : dict) appendT(v);
            for (size_t w = 0; w < numWords; ++w) appendT(bitmaps[t][w]);
            packKeys(out, keys, kb);
        }

        appendT(static_cast<uint32_t>(fallback.size()));
        for (const T v : fallback) appendT(v);

        EncodedBuffer<uint8_t> result;
        result.metadata().elementCount         = N;
        result.metadata().encodingName         = name();
        result.metadata().supportsRandomAccess = true;
        result.metadata().compressedSize       = out.size();
        result.metadata().uncompressedSize     = N * sizeof(T);
        result.data() = std::move(out);
        return result;
    }

    // ---------------------------------------------------------------------------
    // Decode helpers
    // ---------------------------------------------------------------------------

private:
    struct ParsedHeader {
        uint64_t numElements{0};
        uint8_t  numTiers{0};
        size_t   numWords{0};

        struct TierData {
            uint8_t  keyBits{0};
            std::vector<T>        dict;
            std::vector<uint64_t> bitmap;
            size_t   keysOffset{0}; // byte offset into raw data for packed keys
            size_t   tierCount{0};  // elements in this tier
        };
        std::vector<TierData> tiers;

        size_t   fallbackOffset{0};
        uint32_t fallbackCount{0};
    };

    static ParsedHeader parseHeader(const EncodedBuffer<uint8_t>& enc) {
        const uint8_t* p   = enc.data().data();
        const uint8_t* end = p + enc.data().size();

        const auto readU64 = [&]() -> uint64_t {
            uint64_t v; std::memcpy(&v, p, 8); p += 8; return v;
        };
        const auto readU32 = [&]() -> uint32_t {
            uint32_t v; std::memcpy(&v, p, 4); p += 4; return v;
        };
        const auto readU8  = [&]() -> uint8_t { return *p++; };
        const auto readT   = [&]() -> T { T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; };

        ParsedHeader h;
        h.numElements = readU64();
        h.numTiers    = readU8();
        h.numWords    = (h.numElements + 63) / 64;

        h.tiers.resize(h.numTiers);
        for (auto& td : h.tiers) {
            td.keyBits = readU8();
            const uint32_t dictCount = readU32();
            td.dict.resize(dictCount);
            for (auto& v : td.dict) v = readT();

            td.bitmap.resize(h.numWords);
            for (auto& w : td.bitmap) { std::memcpy(&w, p, 8); p += 8; }

            // Count tier elements (popcount of bitmap).
            td.tierCount = 0;
            for (uint64_t w : td.bitmap) td.tierCount += static_cast<size_t>(__builtin_popcountll(w));

            td.keysOffset = static_cast<size_t>(p - enc.data().data());
            const size_t keyBytes = packedKeyBytes(td.tierCount, td.keyBits);
            p += keyBytes;
        }

        h.fallbackOffset = static_cast<size_t>(p - enc.data().data());
        h.fallbackCount  = readU32();
        (void)end;
        return h;
    }

    // Return number of bytes needed to store `count` keys of `keyBits` bits.
    static constexpr size_t packedKeyBytes(size_t count, uint8_t keyBits) {
        if (keyBits <= 8) return (count * keyBits + 7) / 8;
        if (keyBits == 16) return count * 2;
        return count * 4; // 32-bit
    }

    // Pack keys into bit-packed format and append to `out`.
    static void packKeys(std::vector<uint8_t>& out, const std::vector<uint32_t>& keys, uint8_t keyBits) {
        if (keyBits == 32) {
            for (uint32_t k : keys) {
                const uint8_t* b = reinterpret_cast<const uint8_t*>(&k);
                out.insert(out.end(), b, b + 4);
            }
            return;
        }
        if (keyBits == 16) {
            for (uint32_t k : keys) {
                const uint16_t k16 = static_cast<uint16_t>(k);
                const uint8_t* b = reinterpret_cast<const uint8_t*>(&k16);
                out.insert(out.end(), b, b + 2);
            }
            return;
        }
        // Bit-pack for 1/2/4/8-bit keys, LSB-first within each byte.
        const size_t byteCount = packedKeyBytes(keys.size(), keyBits);
        const size_t baseOffset = out.size();
        out.resize(out.size() + byteCount, 0);
        for (size_t r = 0; r < keys.size(); ++r) {
            const size_t bitPos = r * keyBits;
            const size_t byteIdx = bitPos / 8;
            const size_t bitOff  = bitPos % 8;
            out[baseOffset + byteIdx] |= static_cast<uint8_t>(keys[r] << bitOff);
            if (keyBits > 8 - bitOff && byteIdx + 1 < byteCount) {
                out[baseOffset + byteIdx + 1] |= static_cast<uint8_t>(keys[r] >> (8 - bitOff));
            }
        }
    }

    // Unpack key at rank `r` from packed keys starting at `base`, using `keyBits` width.
    static uint32_t unpackKey(const uint8_t* base, size_t r, uint8_t keyBits) {
        if (keyBits == 32) {
            uint32_t v; std::memcpy(&v, base + r * 4, 4); return v;
        }
        if (keyBits == 16) {
            uint16_t v; std::memcpy(&v, base + r * 2, 2); return v;
        }
        const size_t bitPos = r * keyBits;
        const size_t byteIdx = bitPos / 8;
        const size_t bitOff  = bitPos % 8;
        const uint32_t mask  = (1u << keyBits) - 1u;
        uint32_t val = static_cast<uint32_t>(base[byteIdx]) >> bitOff;
        if (keyBits > 8 - bitOff) val |= static_cast<uint32_t>(base[byteIdx + 1]) << (8 - bitOff);
        return val & mask;
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

    // ---------------------------------------------------------------------------
    // Decode all
    // ---------------------------------------------------------------------------

public:
    std::vector<T> decodeAll(const EncodedBuffer<uint8_t>& enc) override {
        const ParsedHeader h = parseHeader(enc);
        const size_t N = static_cast<size_t>(h.numElements);
        std::vector<T> out(N);

        for (const auto& td : h.tiers) {
            const uint8_t* keysBase = enc.data().data() + td.keysOffset;
            size_t rank = 0;
            for (size_t w = 0; w < h.numWords; ++w) {
                uint64_t word = td.bitmap[w];
                while (word) {
                    const size_t bit = static_cast<size_t>(__builtin_ctzll(word));
                    const size_t pos = w * 64 + bit;
                    if (pos < N) {
                        out[pos] = td.dict[unpackKey(keysBase, rank, td.keyBits)];
                    }
                    ++rank;
                    word &= word - 1;
                }
            }
        }

        // Fallback pass: iterate positions not covered by any tier.
        if (h.fallbackCount > 0) {
            const uint8_t* fp = enc.data().data() + h.fallbackOffset + 4; // skip uint32_t fallbackCount
            size_t fi = 0;
            for (size_t pos = 0; pos < N; ++pos) {
                bool covered = false;
                for (const auto& td : h.tiers) {
                    if (td.bitmap[pos / 64] & (uint64_t{1} << (pos % 64))) {
                        covered = true;
                        break;
                    }
                }
                if (!covered) {
                    T v; std::memcpy(&v, fp + fi * sizeof(T), sizeof(T));
                    out[pos] = v;
                    ++fi;
                }
            }
        }

        return out;
    }

    // ---------------------------------------------------------------------------
    // Decode at index i
    // ---------------------------------------------------------------------------

    std::optional<T> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t i) override {
        const ParsedHeader h = parseHeader(enc);
        if (i >= h.numElements) return std::nullopt;

        for (const auto& td : h.tiers) {
            if (td.bitmap[i / 64] & (uint64_t{1} << (i % 64))) {
                const size_t rank = popcountPrefix(td.bitmap, i);
                const uint8_t* keysBase = enc.data().data() + td.keysOffset;
                return td.dict[unpackKey(keysBase, rank, td.keyBits)];
            }
        }

        // Fallback: count how many fallback elements precede position i.
        size_t fallbackRank = 0;
        for (size_t pos = 0; pos < i; ++pos) {
            bool covered = false;
            for (const auto& td : h.tiers) {
                if (td.bitmap[pos / 64] & (uint64_t{1} << (pos % 64))) {
                    covered = true;
                    break;
                }
            }
            if (!covered) ++fallbackRank;
        }

        const uint8_t* fp = enc.data().data() + h.fallbackOffset + 4;
        T v; std::memcpy(&v, fp + fallbackRank * sizeof(T), sizeof(T));
        return v;
    }

    // ---------------------------------------------------------------------------
    // Decode range [start, end)
    // ---------------------------------------------------------------------------

    std::vector<T> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) override {
        const ParsedHeader h = parseHeader(enc);
        const size_t N = static_cast<size_t>(h.numElements);
        if (start >= end || start >= N) return {};
        end = std::min(end, N);

        const size_t rangeLen = end - start;
        std::vector<T> out(rangeLen);

        // Per-tier: find rank at `start`, then iterate bitmap bits [start, end).
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

        // Fallback: count rank at start, then fill remaining positions.
        if (h.fallbackCount > 0) {
            // Count fallback elements before `start`.
            size_t fallbackRankAtStart = 0;
            for (size_t pos = 0; pos < start; ++pos) {
                bool covered = false;
                for (const auto& td : h.tiers) {
                    if (td.bitmap[pos / 64] & (uint64_t{1} << (pos % 64))) { covered = true; break; }
                }
                if (!covered) ++fallbackRankAtStart;
            }

            const uint8_t* fp = enc.data().data() + h.fallbackOffset + 4;
            size_t fi = fallbackRankAtStart;
            for (size_t pos = start; pos < end; ++pos) {
                bool covered = false;
                for (const auto& td : h.tiers) {
                    if (td.bitmap[pos / 64] & (uint64_t{1} << (pos % 64))) { covered = true; break; }
                }
                if (!covered) {
                    T v; std::memcpy(&v, fp + fi * sizeof(T), sizeof(T));
                    out[pos - start] = v;
                    ++fi;
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
