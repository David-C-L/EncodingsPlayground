#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <ankerl/unordered_dense.h>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/BitPacker.hpp"

namespace encodings::encoders {

using core::BitOrder;
using core::BitWriter;
using core::BitReader;

// Faster drop-in replacements for std::unordered_map / std::unordered_set
template<typename K, typename V>
using HashMap = ankerl::unordered_dense::map<K, V>;
template<typename K>
using HashSet = ankerl::unordered_dense::set<K>;

// ============================================================================
// Public types
// ============================================================================

/**
 * @brief Which byte split to apply to each int32_t.
 *
 *  Split13  ->  [byte 0]        | [bytes 1-3]      (1-byte left,  3-byte right)
 *  Split22  ->  [bytes 0-1]     | [bytes 2-3]      (2-byte left,  2-byte right)
 *  Split31  ->  [bytes 0-2]     | [byte 3]         (3-byte left,  1-byte right)
 *
 * "left" always refers to the lower-address (little-endian) bytes.
 */
enum class SplitMode : uint8_t {
    Split13 = 0,   // 1-byte left  | 3-byte right
    Split22 = 1,   // 2-byte left  | 2-byte right
    Split31 = 2,   // 3-byte left  | 1-byte right
};

/**
 * @brief Maximum code widths (bits) by group size in bytes.
 *
 *  1-byte group  -> max 4 bits  (can address up to 16 distinct values)
 *  2-byte group  -> max 8 bits  (can address up to 256 distinct values)
 *  3-byte group  -> max 16 bits (can address up to 65536 distinct values)
 */
static constexpr uint32_t kMaxCodeWidth1 =  4;
static constexpr uint32_t kMaxCodeWidth2 =  8;
static constexpr uint32_t kMaxCodeWidth3 = 16;

/**
 * @brief User-supplied fixed code widths (overrides adaptive selection).
 *
 * Set a field to 0 to request adaptive selection for that group.
 * Width must be a power-of-two number of bits in {1, 2, 4, 8, 16}.
 */
struct SubIntConfig {
    SplitMode   splitMode       = SplitMode::Split22;
    uint32_t    leftCodeWidth   = 0;   ///< 0 = adaptive
    uint32_t    rightCodeWidth  = 0;   ///< 0 = adaptive
    BitOrder    bitOrder        = BitOrder::LSB;
};

// ============================================================================
// Internal helpers
// ============================================================================
namespace detail {

/// Number of bytes in the left group for each SplitMode.
inline constexpr uint32_t leftBytes(SplitMode m) noexcept {
    switch (m) {
        case SplitMode::Split13: return 1;
        case SplitMode::Split22: return 2;
        case SplitMode::Split31: return 3;
    }
    return 0;
}

/// Number of bytes in the right group for each SplitMode.
inline constexpr uint32_t rightBytes(SplitMode m) noexcept {
    return 4u - leftBytes(m);
}

/// Maximum code width for a group of `bytes` bytes.
inline constexpr uint32_t maxCodeWidth(uint32_t bytes) noexcept {
    switch (bytes) {
        case 1: return kMaxCodeWidth1;
        case 2: return kMaxCodeWidth2;
        case 3: return kMaxCodeWidth3;
    }
    return 0;
}

/// Smallest power-of-two number of bits that can index `n` distinct values.
/// Returns 0 if n == 0 or n == 1 (need 0 bits, but we use 1 as minimum).
inline uint32_t adaptiveWidth(uint32_t n, uint32_t maxW) noexcept {
    if (n <= 1)  return 1;          // degenerate; 1 bit minimum
    // ceil(log2(n))
    uint32_t bits = static_cast<uint32_t>(std::bit_width(n - 1u));
    // Round up to next power-of-two number of bits: 1,2,4,8,16
    uint32_t w = 1;
    while (w < bits) w <<= 1;
    return std::min(w, maxW);
}

/// Extract the left group value from a raw int32 (little-endian bytes).
inline uint32_t extractLeft(uint32_t raw, SplitMode m) noexcept {
    const uint32_t lb = leftBytes(m);
    const uint32_t mask = (lb == 4) ? ~0u : ((1u << (lb * 8u)) - 1u);
    return raw & mask;
}

/// Extract the right group value from a raw int32.
inline uint32_t extractRight(uint32_t raw, SplitMode m) noexcept {
    const uint32_t lb = leftBytes(m);
    return raw >> (lb * 8u);
}

/// Reconstruct a raw int32 from left and right group values.
inline uint32_t reconstruct(uint32_t left, uint32_t right, SplitMode m) noexcept {
    return left | (right << (leftBytes(m) * 8u));
}

/// Validate a user-supplied code width is a non-zero power of two ≤ maxW.
inline bool validWidth(uint32_t w, uint32_t maxW) noexcept {
    return w != 0 && w <= maxW && (w & (w - 1)) == 0;
}

// ---------------------------------------------------------------------------
// Dictionary builder
// ---------------------------------------------------------------------------

/**
 * Build a value→code map for a set of observed values.
 * Returns false if the number of distinct values exceeds 2^maxW (overflow).
 * On overflow, `dict` is left empty — caller must handle raw fallback.
 */
inline bool buildDict(
    const std::vector<uint32_t>& values,
    uint32_t                     maxW,
    uint32_t                     fixedW,       ///< 0 = adaptive
    HashMap<uint32_t, uint32_t>& dict,
    uint32_t&                    codeWidth)
{
    // Single pass: assign insertion-order codes via dict.size() before insert
    for (uint32_t v : values) {
        dict.emplace(v, static_cast<uint32_t>(dict.size()));
    }

    const uint32_t n = static_cast<uint32_t>(dict.size());
    const uint32_t capacity = (fixedW == 0)
        ? (1u << adaptiveWidth(n, maxW))
        : (1u << fixedW);

    if (n > capacity) {
        dict.clear();
        codeWidth = 0;
        return false;
    }

    codeWidth = (fixedW != 0) ? fixedW : adaptiveWidth(n, maxW);
    return true;
}

// ---------------------------------------------------------------------------
// Wire-format helpers
// ---------------------------------------------------------------------------

inline void writeU8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
}
inline void writeU16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v));
    buf.push_back(static_cast<uint8_t>(v >> 8));
}
inline void writeU32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v));
    buf.push_back(static_cast<uint8_t>(v >>  8));
    buf.push_back(static_cast<uint8_t>(v >> 16));
    buf.push_back(static_cast<uint8_t>(v >> 24));
}
inline void writeU64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        buf.push_back(static_cast<uint8_t>(v >> (i * 8)));
}

inline uint8_t  readU8 (const uint8_t* p)             { return *p; }
inline uint16_t readU16(const uint8_t* p)             {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
inline uint32_t readU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])        |
           (static_cast<uint32_t>(p[1]) <<  8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t readU64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(p[i]) << (i * 8);
    return v;
}

// ---------------------------------------------------------------------------
// Dictionary serialisation helpers
// ---------------------------------------------------------------------------

/// Write one dictionary to `buf`. `groupBytes` is the byte width of each key.
/// Format:
///   [code_width : 1 byte]
///   [num_entries: 2 bytes]
///   for each entry (in code order):
///       [key: groupBytes bytes (little-endian)]
void writeDict(
    std::vector<uint8_t>&                  buf,
    const HashMap<uint32_t, uint32_t>&     dict,
    uint32_t                               codeWidth,
    uint32_t                               groupBytes)
{
    // Sort by code so we can recover dict from just the keys in order.
    std::vector<std::pair<uint32_t,uint32_t>> entries(dict.begin(), dict.end());
    std::sort(entries.begin(), entries.end(),
              [](auto& a, auto& b){ return a.second < b.second; });

    writeU8(buf, static_cast<uint8_t>(codeWidth));
    writeU16(buf, static_cast<uint16_t>(entries.size()));
    for (auto& [key, _] : entries) {
        // Write `groupBytes` bytes of key, little-endian
        for (uint32_t b = 0; b < groupBytes; ++b)
            buf.push_back(static_cast<uint8_t>(key >> (b * 8)));
    }
}

/// Read one dictionary back. Returns {dict, codeWidth, bytes_consumed}.
struct ReadDictResult {
    HashMap<uint32_t, uint32_t> codeToValue; ///< code -> group value
    uint32_t                    codeWidth;
    size_t                      bytesRead;
};

ReadDictResult readDict(const uint8_t* p, uint32_t groupBytes) {
    ReadDictResult r;
    r.codeWidth = readU8(p);      p += 1;
    uint16_t n  = readU16(p);     p += 2;
    r.bytesRead = 3 + static_cast<size_t>(n) * groupBytes;
    for (uint32_t code = 0; code < n; ++code) {
        uint32_t key = 0;
        for (uint32_t b = 0; b < groupBytes; ++b)
            key |= static_cast<uint32_t>(*p++) << (b * 8);
        r.codeToValue[code] = key;
    }
    return r;
}

} // namespace detail

// ============================================================================
// SubIntEncoder
// ============================================================================

/**
 * @brief Sub-integer code encoder for int32_t values.
 *
 * Splits each 32-bit integer into two byte groups according to a chosen
 * SplitMode, builds a dictionary for each group, and packs the resulting
 * codes into a compact bit stream.
 *
 * Wire format:
 *
 *   [num_ints          : 8 bytes]
 *   [split_mode        : 1 byte ]   (0=Split13, 1=Split22, 2=Split31)
 *   [bit_order         : 1 byte ]   (0=LSB, 1=MSB)
 *   [fallback_flags    : 1 byte ]   bit0 = left_raw, bit1 = right_raw
 *   [shared_dict_flag  : 1 byte ]   1 if Split22 uses a single shared dict
 *   -- if left NOT raw:
 *   [left dict                  ]   see writeDict()
 *   -- if right NOT raw AND NOT shared:
 *   [right dict                 ]
 *   [encoded_ints_length: 8 bytes]
 *   [encoded_ints               ]   packed bit stream
 *
 * Random access:
 *   bits_per_int = left_width + right_width  (constant when no raw fallback)
 *   When one or both sides are raw the per-int bit width is still constant
 *   (raw side contributes exactly groupBytes*8 bits per int).
 *
 * Special case: if BOTH sides fall back to raw, the data is stored as plain
 * int32_t values (identical to RawEncoder<int32_t> output), and the header
 * records fallback_flags = 0b11.
 */
class SubIntEncoder : public encodings::Codec<int32_t> {
public:
    explicit SubIntEncoder(SubIntConfig cfg = {}) : cfg_(cfg) {}

    // -----------------------------------------------------------------------
    // Codec<int32_t>
    // -----------------------------------------------------------------------

    encodings::EncodedData encode(std::span<const int32_t> data) override {
        using namespace detail;

        // computeHistograms(data, true); // For debugging/logging – not needed for encoding

        const size_t N = data.size();

        const uint32_t lb    = leftBytes(cfg_.splitMode);
        const uint32_t rb    = rightBytes(cfg_.splitMode);
        const uint32_t lMaxW = maxCodeWidth(lb);
        const uint32_t rMaxW = maxCodeWidth(rb);

        // ---- 1. Build dictionaries in a single scan ----------------------
        //
        // We split and accumulate into the dicts in one pass, avoiding the
        // allocation of separate leftVals / rightVals vectors.  We do still
        // need the raw group values for the bit-writing pass, so we keep them
        // but interleaved in one vector to improve locality.
        //
        // Layout: [left0, right0, left1, right1, ...]
        std::vector<uint32_t> groups(N * 2);
        HashMap<uint32_t, uint32_t> leftDict, rightDict;
        leftDict.reserve(1u << lMaxW);
        rightDict.reserve(1u << rMaxW);

        for (size_t i = 0; i < N; ++i) {
            const uint32_t raw = static_cast<uint32_t>(data[i]);
            const uint32_t lv  = extractLeft (raw, cfg_.splitMode);
            const uint32_t rv  = extractRight(raw, cfg_.splitMode);
            groups[2*i]   = lv;
            groups[2*i+1] = rv;
            leftDict.emplace (lv, static_cast<uint32_t>(leftDict.size()));
            rightDict.emplace(rv, static_cast<uint32_t>(rightDict.size()));
        }

        uint32_t leftCW  = 0;
        uint32_t rightCW = 0;
        bool     leftRaw  = false;
        bool     rightRaw = false;
        bool     shared   = false;

        // Check overflow
        {
            const uint32_t lCap = 1u << lMaxW;
            const uint32_t rCap = 1u << rMaxW;
            if (leftDict.size() > lCap) {
                std::cerr << "[SubIntEncoder] WARNING: left group overflow – raw fallback.\n";
                leftDict.clear();
                leftRaw = true;
            }
            if (rightDict.size() > rCap) {
                std::cerr << "[SubIntEncoder] WARNING: right group overflow – raw fallback.\n";
                rightDict.clear();
                rightRaw = true;
            }
        }

        if (!leftRaw)  leftCW  = adaptiveWidth(static_cast<uint32_t>(leftDict.size()),  lMaxW);
        if (!rightRaw) rightCW = adaptiveWidth(static_cast<uint32_t>(rightDict.size()), rMaxW);

        // Split22 shared-dict check
        if (cfg_.splitMode == SplitMode::Split22 && !leftRaw && !rightRaw) {
            shared = checkSharedBenefit(leftDict, rightDict, leftCW, rightCW, lMaxW, N);
            if (shared) {
                buildMergedDict(leftDict, rightDict, lMaxW, leftCW);
                rightDict.clear();
                rightCW = leftCW;
            }
        }

        const bool bothRaw = leftRaw && rightRaw;

        // ---- 2. Build flat code-lookup arrays (value → code) --------------
        //
        // For the non-raw sides we replace HashMap::at() in the hot loop with
        // a direct array lookup.  Max capacity is bounded by the max code width
        // (16 values for 1-byte, 256 for 2-byte, 65536 for 3-byte group).
        //
        // For 1-byte groups  (lb==1): values ∈ [0,255]   → array[256]
        // For 2-byte groups  (lb==2): values ∈ [0,65535] → array[65536]
        // For 3-byte groups  (lb==3): values ∈ [0,16M-1] → too large for flat;
        //                              fall back to the hash map (rare).
        //
        // The array is sized to the maximum possible value + 1, filled via the
        // dict, so lookup is: code = leftCodeArr[value].
        const bool useFlatLeft  = !leftRaw  && lb  <= 2u;
        const bool useFlatRight = !rightRaw && rb  <= 2u;
        const uint32_t lArrSize = useFlatLeft  ? (1u << (lb  * 8u)) : 0u;
        const uint32_t rArrSize = useFlatRight ? (1u << (rb  * 8u)) : 0u;
        std::vector<uint32_t> leftCodeArr(lArrSize, 0);
        std::vector<uint32_t> rightCodeArr(rArrSize, 0);
        if (useFlatLeft)
            for (auto& [val, code] : leftDict)  leftCodeArr[val]  = code;
        if (useFlatRight && !shared)
            for (auto& [val, code] : rightDict) rightCodeArr[val] = code;
        else if (useFlatRight && shared)
            for (auto& [val, code] : leftDict)  rightCodeArr[val] = code;

        // ---- 3. Serialise header ------------------------------------------
        std::vector<uint8_t> header;
        header.reserve(32);
        writeU64(header, static_cast<uint64_t>(N));
        writeU8(header, static_cast<uint8_t>(cfg_.splitMode));
        writeU8(header, (cfg_.bitOrder == BitOrder::MSB) ? 1u : 0u);

        const uint8_t fallbackFlags =
            (leftRaw  ? 0x01u : 0x00u) |
            (rightRaw ? 0x02u : 0x00u);
        writeU8(header, fallbackFlags);
        writeU8(header, shared ? 1u : 0u);

        if (!leftRaw)
            writeDict(header, leftDict, leftCW, lb);
        if (!rightRaw && !shared)
            writeDict(header, rightDict, rightCW, rb);

        // ---- 4. Encode the integer stream --------------------------------
        std::vector<uint8_t> stream;

        if (bothRaw) {
            stream.resize(N * sizeof(int32_t));
            std::memcpy(stream.data(), data.data(), stream.size());
        } else {
            const uint32_t lBits = leftRaw  ? (lb * 8u) : leftCW;
            const uint32_t rBits = rightRaw ? (rb * 8u) : rightCW;
            const size_t totalBits = N * (lBits + rBits);
            stream.reserve((totalBits + 7) / 8);

            BitWriter bw(stream, cfg_.bitOrder);
            for (size_t i = 0; i < N; ++i) {
                const uint32_t lv = groups[2*i];
                const uint32_t rv = groups[2*i+1];

                // Left side
                if (leftRaw) {
                    bw.write(lv, lb * 8u);
                } else if (useFlatLeft) {
                    bw.write(leftCodeArr[lv], leftCW);
                } else {
                    bw.write(leftDict.at(lv), leftCW);
                }

                // Right side
                if (rightRaw) {
                    bw.write(rv, rb * 8u);
                } else if (useFlatRight) {
                    bw.write(rightCodeArr[rv], rightCW);
                } else {
                    const auto& rMap = shared ? leftDict : rightDict;
                    bw.write(rMap.at(rv), rightCW);
                }
            }
            bw.flush();
        }

        // ---- 5. Assemble final buffer (single allocation) ----------------
        std::vector<uint8_t> out;
        out.reserve(header.size() + 8 + stream.size());
        out.insert(out.end(), header.begin(), header.end());
        writeU64(out, static_cast<uint64_t>(stream.size()));
        out.insert(out.end(), stream.begin(), stream.end());

        // ---- 6. Metadata -------------------------------------------------
        encodings::EncodingMetadata meta;
        meta.encodingName     = name();
        meta.dataType         = encodings::DataType::Int32;
        meta.elementCount     = N;
        meta.compressedSize   = out.size();
        meta.uncompressedSize = N * sizeof(int32_t);
        meta.supportsRandomAccess = !bothRaw;

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // -----------------------------------------------------------------------

    std::vector<int32_t> decodeAll(const encodings::EncodedData& encoded) override {
        DecodeCtx ctx = parseHeader(encoded);
        return decodeSequential(ctx, 0, ctx.N);
    }

    std::optional<int32_t> decodeAt(const encodings::EncodedData& encoded,
                                     size_t index) override {
        DecodeCtx ctx = parseHeader(encoded);
        if (index >= ctx.N) return std::nullopt;
        return decodeOne(ctx, index);
    }

    std::vector<int32_t> decodeRange(const encodings::EncodedData& encoded,
                                      size_t start, size_t end) override {
        DecodeCtx ctx = parseHeader(encoded);
        if (start >= ctx.N) return {};
        end = std::min(end, ctx.N);
        return decodeSequential(ctx, start, end);
    }

    // -----------------------------------------------------------------------

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::SubIntEncoding;
    }

    std::string name() const override {
        switch (cfg_.splitMode) {
            case SplitMode::Split13: return "SubInt(1|3)";
            case SplitMode::Split22: return "SubInt(2|2)";
            case SplitMode::Split31: return "SubInt(3|1)";
        }
        return "SubInt";
    }

    encodings::EncodingProperties properties() const override {
        return encodings::EncodingProperties(encodings::EncodingProperty::RandomAccess)
             | encodings::EncodingProperty::Lossless
             | encodings::EncodingProperty::PreservesOrder
             | encodings::EncodingProperty::DictionaryBased
             | encodings::EncodingProperty::BitPackingBased
             | encodings::EncodingProperty::RequiresFullData
             | encodings::EncodingProperty::ImmutableOnly;
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        // Conservative upper bound: header ≈ 512 bytes + full bit stream
        const uint32_t lb = detail::leftBytes (cfg_.splitMode);
        const uint32_t rb = detail::rightBytes(cfg_.splitMode);
        const uint32_t lBits = (cfg_.leftCodeWidth  != 0) ? cfg_.leftCodeWidth  : detail::maxCodeWidth(lb);
        const uint32_t rBits = (cfg_.rightCodeWidth != 0) ? cfg_.rightCodeWidth : detail::maxCodeWidth(rb);
        const size_t streamBytes = (elementCount * (lBits + rBits) + 7) / 8;
        return 512 + 8 + streamBytes;
    }

    // -----------------------------------------------------------------------
    // Histogram support
    // -----------------------------------------------------------------------

    /**
     * @brief Frequency histogram for one group side (left or right).
     *
     * `counts` maps each distinct group value to the number of times it
     * appears in the input passed to computeHistograms().
     * `totalValues` is the total number of values seen (== input size).
     */
    struct GroupHistogram {
        HashMap<uint32_t, size_t> counts;
        size_t                    totalValues{0};
        uint32_t                  groupBytes{0};  ///< 1, 2, or 3
    };

    struct Histograms {
        GroupHistogram left;
        GroupHistogram right;
        SplitMode      splitMode;
    };

    /**
     * @brief Compute value-frequency histograms for the left and right groups.
     *
     * @param data    Input integers to analyse.
     * @param verbose If true, also prints the histograms to stdout.
     * @return        Pair of histograms (left, right).
     */
    Histograms computeHistograms(std::span<const int32_t> data,
                                 bool verbose = false) const
    {
        using namespace detail;

        Histograms h;
        h.splitMode        = cfg_.splitMode;
        h.left.groupBytes  = leftBytes (cfg_.splitMode);
        h.right.groupBytes = rightBytes(cfg_.splitMode);
        h.left.totalValues = h.right.totalValues = data.size();

        for (int32_t v : data) {
            const uint32_t raw = static_cast<uint32_t>(v);
            ++h.left.counts [extractLeft (raw, cfg_.splitMode)];
            ++h.right.counts[extractRight(raw, cfg_.splitMode)];
        }

        if (verbose) {
            printHistograms(h, std::cout);
        }

        return h;
    }

    /**
     * @brief Print pre-computed histograms to an output stream.
     *
     * Entries are sorted by descending frequency, then ascending value.
     * A simple bar chart is rendered alongside the counts.
     *
     * @param h    Histograms produced by computeHistograms().
     * @param out  Destination stream (default: std::cout).
     */
    static void printHistograms(const Histograms& h,
                                std::ostream& out = std::cout)
    {
        auto printOne = [&](const GroupHistogram& gh, const std::string& label) {
            out << "=== " << label << " group ("
                << gh.groupBytes << " byte"
                << (gh.groupBytes != 1 ? "s" : "") << ", "
                << gh.counts.size() << " distinct values, "
                << gh.totalValues   << " total) ===\n";

            if (gh.counts.empty()) { out << "  (empty)\n\n"; return; }

            // Sort by descending count, then ascending value for ties
            std::vector<std::pair<uint32_t, size_t>> entries(
                gh.counts.begin(), gh.counts.end());
            std::sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) {
                    return a.second != b.second
                        ? a.second > b.second
                        : a.first  < b.first;
                });

            const size_t maxCount = entries.front().second;
            constexpr int kBarWidth = 40;

            // Column widths depend on group byte size
            const int valWidth = gh.groupBytes <= 1 ? 3
                               : gh.groupBytes == 2 ? 5 : 8;

            size_t num_less_than = 0;
            size_t num_to_be_less_than = 1000;
            for (const auto& [val, cnt] : entries) {
                if (cnt < num_to_be_less_than) {
                    ++num_less_than;
                    continue; // skip some for now
                }
                const double pct  = gh.totalValues > 0
                    ? 100.0 * cnt / gh.totalValues : 0.0;
                const int barLen  = maxCount > 0
                    ? static_cast<int>(kBarWidth * cnt / maxCount) : 0;

                out << "  0x"
                    << std::hex << std::setw(valWidth) << std::setfill('0') << val
                    << std::dec << std::setfill(' ')
                    << "  " << std::setw(10) << cnt
                    << "  (" << std::fixed << std::setprecision(2)
                    << std::setw(6) << pct << "%)  |"
                    << std::string(barLen, '#')
                    << "\n";
            }
            if (num_less_than > 0) {
                const double pct = gh.totalValues > 0
                    ? 100.0 * num_less_than / gh.totalValues : 0.0;
                out << "  (and " << num_less_than << " less than " << num_to_be_less_than << " occurrences, "
                    << std::fixed << std::setprecision(2) << pct << "% of total)\n";
            }
            out << "\n";
        };

        printOne(h.left,  "Left");
        printOne(h.right, "Right");
    }

private:
    SubIntConfig cfg_;

    // -----------------------------------------------------------------------
    // Shared-dict benefit check (Split22 only)
    // -----------------------------------------------------------------------

    /**
     * Returns true if using a single merged dictionary for both left and right
     * groups saves bits in the encoded stream compared to two separate dicts.
     *
     * Savings analysis:
     *   - Separate dicts: each int costs leftCW + rightCW bits
     *   - Merged dict  : each int costs 2 * mergedCW bits
     *   - Plus header overhead: a merged dict eliminates one dict block
     */
    static bool checkSharedBenefit(
        const HashMap<uint32_t,uint32_t>& leftDict,
        const HashMap<uint32_t,uint32_t>& rightDict,
        uint32_t leftCW, uint32_t rightCW,
        uint32_t maxW, size_t N)
    {
        // Build union of left and right distinct values
        HashSet<uint32_t> unionSet;
        for (auto& [v,_] : leftDict)  unionSet.insert(v);
        for (auto& [v,_] : rightDict) unionSet.insert(v);
        const uint32_t unionSize = static_cast<uint32_t>(unionSet.size());

        // Would the union overflow the max code width?
        const uint32_t capacity = 1u << maxW;
        if (unionSize > capacity) return false;  // can't share anyway

        const uint32_t mergedCW = detail::adaptiveWidth(unionSize, maxW);

        // Bits saved in the stream per int: (leftCW + rightCW) - 2*mergedCW
        // (positive = sharing is cheaper in the stream)
        const int64_t streamSavingBits =
            static_cast<int64_t>(leftCW + rightCW) -
            static_cast<int64_t>(2 * mergedCW);
        const int64_t streamSavingBytes =
            (static_cast<int64_t>(N) * streamSavingBits) / 8;

        // Header cost of the eliminated right dict:
        //   3 bytes fixed + num_entries * 2 bytes (2-byte groups)
        const int64_t eliminatedDictBytes =
            3 + static_cast<int64_t>(rightDict.size()) * 2;

        // Extra entries in the left dict due to merging:
        const int64_t extraLeftEntries =
            static_cast<int64_t>(unionSize) -
            static_cast<int64_t>(leftDict.size());
        const int64_t extraLeftDictBytes = extraLeftEntries * 2;

        const int64_t netSaving = streamSavingBytes
                                + eliminatedDictBytes
                                - extraLeftDictBytes;
        return netSaving > 0;
    }

    /// Rebuild leftDict as the merged dict for both sides (Split22).
    /// Takes the existing left and right dicts directly — no extra allocation.
    static void buildMergedDict(
        HashMap<uint32_t,uint32_t>&       mergedDict,
        const HashMap<uint32_t,uint32_t>& rightDict,
        uint32_t                          maxW,
        uint32_t&                         codeWidth)
    {
        // Add any right-only values to the left dict with fresh codes.
        // emplace() is a no-op when the key already exists, so left-side
        // values keep their original insertion-order codes.
        uint32_t nextCode = static_cast<uint32_t>(mergedDict.size());
        for (auto& [val, _] : rightDict) {
            if (mergedDict.emplace(val, nextCode).second)
                ++nextCode;
        }
        const uint32_t n = static_cast<uint32_t>(mergedDict.size());
        codeWidth = detail::adaptiveWidth(n, maxW);
        // Re-number all codes 0..n-1 preserving insertion order
        uint32_t c = 0;
        for (auto& [_, code] : mergedDict) code = c++;
    }

    // -----------------------------------------------------------------------
    // Decode support
    // -----------------------------------------------------------------------

    struct DecodeCtx {
        size_t   N{0};
        SplitMode mode{SplitMode::Split22};
        BitOrder  order{BitOrder::LSB};
        bool     leftRaw{false};
        bool     rightRaw{false};
        bool     bothRaw{false};
        bool     shared{false};

        uint32_t leftCW{0};   ///< code width in bits (0 if raw)
        uint32_t rightCW{0};
        uint32_t lb{0};       ///< left group bytes
        uint32_t rb{0};       ///< right group bytes
        uint32_t bitsPerInt{0};

        // Decode tables: code → group value
        std::vector<uint32_t> leftTable;   ///< indexed by code
        std::vector<uint32_t> rightTable;  ///< indexed by code (empty if shared)

        // Pointer into the encoded buffer where the packed stream begins
        const uint8_t* streamPtr{nullptr};
        size_t         streamBytes{0};
    };

    static DecodeCtx parseHeader(const encodings::EncodedData& encoded) {
        using namespace detail;
        const uint8_t* p   = encoded.data().data();
        const uint8_t* end = p + encoded.size();

        auto require = [&](size_t n, const char* ctx) {
            if (p + n > end)
                throw std::runtime_error(
                    std::string("SubIntEncoder::decode: truncated at ") + ctx);
        };

        // num_ints
        require(8, "num_ints");
        DecodeCtx ctx;
        ctx.N    = static_cast<size_t>(readU64(p)); p += 8;

        // split_mode
        require(1, "split_mode");
        ctx.mode = static_cast<SplitMode>(readU8(p)); p += 1;

        // bit_order
        require(1, "bit_order");
        ctx.order = readU8(p) ? BitOrder::MSB : BitOrder::LSB; p += 1;

        // fallback_flags
        require(1, "fallback_flags");
        const uint8_t ff = readU8(p); p += 1;
        ctx.leftRaw  = (ff & 0x01) != 0;
        ctx.rightRaw = (ff & 0x02) != 0;
        ctx.bothRaw  = ctx.leftRaw && ctx.rightRaw;

        // shared_dict_flag
        require(1, "shared_dict_flag");
        ctx.shared = readU8(p) != 0; p += 1;

        ctx.lb = leftBytes (ctx.mode);
        ctx.rb = rightBytes(ctx.mode);

        // Left dict
        if (!ctx.leftRaw) {
            require(3, "left_dict_header");
            auto dr = readDict(p, ctx.lb);
            ctx.leftCW = dr.codeWidth;
            ctx.leftTable.resize(static_cast<size_t>(1) << ctx.leftCW, 0);
            for (auto& [code, val] : dr.codeToValue)
                ctx.leftTable[code] = val;
            p += dr.bytesRead;
        } else {
            ctx.leftCW = ctx.lb * 8u;
        }

        // Right dict
        if (!ctx.rightRaw && !ctx.shared) {
            require(3, "right_dict_header");
            auto dr = readDict(p, ctx.rb);
            ctx.rightCW = dr.codeWidth;
            ctx.rightTable.resize(static_cast<size_t>(1) << ctx.rightCW, 0);
            for (auto& [code, val] : dr.codeToValue)
                ctx.rightTable[code] = val;
            p += dr.bytesRead;
        } else if (ctx.shared) {
            ctx.rightCW = ctx.leftCW;
            // rightTable not needed; we'll use leftTable for both
        } else {
            ctx.rightCW = ctx.rb * 8u;
        }

        ctx.bitsPerInt = ctx.leftCW + ctx.rightCW;

        // encoded_ints_length
        require(8, "stream_length");
        ctx.streamBytes = static_cast<size_t>(readU64(p)); p += 8;

        require(ctx.streamBytes, "stream_data");
        ctx.streamPtr = p;

        return ctx;
    }

    static int32_t decodeOne(const DecodeCtx& ctx, size_t index) {
        using namespace detail;

        if (ctx.bothRaw) {
            int32_t v;
            std::memcpy(&v,
                ctx.streamPtr + index * sizeof(int32_t),
                sizeof(int32_t));
            return v;
        }

        const size_t bitPos = index * ctx.bitsPerInt;
        BitReader br(ctx.streamPtr, ctx.streamBytes, ctx.order);
        br.seekToBit(bitPos);
        return decodeNextFromReader(ctx, br);
    }

    /**
     * @brief Decode a contiguous half-open range [start, end) in a single
     *        sequential pass over the bit stream — no per-element seek.
     *
     * The decode variant (leftRaw / rightRaw / shared) is dispatched once
     * outside the hot loop via a templated lambda, so the per-element body
     * contains no branches on those flags.
     */
    static std::vector<int32_t> decodeSequential(
        const DecodeCtx& ctx, size_t start, size_t end)
    {
        using namespace detail;

        const size_t count = end - start;
        std::vector<int32_t> result;
        result.reserve(count);

        if (ctx.bothRaw) {
            result.resize(count);
            std::memcpy(result.data(),
                        ctx.streamPtr + start * sizeof(int32_t),
                        count * sizeof(int32_t));
            return result;
        }

        // Precompute the shift used by reconstruct() so it's not recomputed
        // inside the loop via a switch statement each iteration.
        const uint32_t leftShift = ctx.lb * 8u;

        BitReader br(ctx.streamPtr, ctx.streamBytes, ctx.order);
        br.seekToBit(start * ctx.bitsPerInt);

        // Dispatch once on the decode variant; each lambda captures only
        // what it needs and has a branchless inner body.
        auto runLoop = [&](auto decodeOne) {
            for (size_t i = 0; i < count; ++i)
                result.push_back(decodeOne(br));
        };

        if (!ctx.leftRaw && !ctx.rightRaw && !ctx.shared) {
            // Common path: both sides dictionary-coded, separate dicts
            const auto& lt = ctx.leftTable;
            const auto& rt = ctx.rightTable;
            const uint32_t lcw = ctx.leftCW;
            const uint32_t rcw = ctx.rightCW;
            runLoop([&](BitReader& r) -> int32_t {
                const uint32_t lv = lt[r.read(lcw)];
                const uint32_t rv = rt[r.read(rcw)];
                return static_cast<int32_t>(lv | (rv << leftShift));
            });
        } else if (!ctx.leftRaw && !ctx.rightRaw && ctx.shared) {
            // Split22 shared dict
            const auto& lt = ctx.leftTable;
            const uint32_t lcw = ctx.leftCW;
            const uint32_t rcw = ctx.rightCW;
            runLoop([&](BitReader& r) -> int32_t {
                const uint32_t lv = lt[r.read(lcw)];
                const uint32_t rv = lt[r.read(rcw)];
                return static_cast<int32_t>(lv | (rv << leftShift));
            });
        } else if (ctx.leftRaw && !ctx.rightRaw) {
            const auto& rt = ctx.rightTable;
            const uint32_t lbits = ctx.lb * 8u;
            const uint32_t rcw   = ctx.rightCW;
            runLoop([&](BitReader& r) -> int32_t {
                const uint32_t lv = r.read(lbits);
                const uint32_t rv = rt[r.read(rcw)];
                return static_cast<int32_t>(lv | (rv << leftShift));
            });
        } else if (!ctx.leftRaw && ctx.rightRaw) {
            const auto& lt = ctx.leftTable;
            const uint32_t lcw   = ctx.leftCW;
            const uint32_t rbits = ctx.rb * 8u;
            runLoop([&](BitReader& r) -> int32_t {
                const uint32_t lv = lt[r.read(lcw)];
                const uint32_t rv = r.read(rbits);
                return static_cast<int32_t>(lv | (rv << leftShift));
            });
        } else {
            // Both raw (but bitsPerInt != 0, so it's not the plain-memcpy case)
            const uint32_t lbits = ctx.lb * 8u;
            const uint32_t rbits = ctx.rb * 8u;
            runLoop([&](BitReader& r) -> int32_t {
                const uint32_t lv = r.read(lbits);
                const uint32_t rv = r.read(rbits);
                return static_cast<int32_t>(lv | (rv << leftShift));
            });
        }

        return result;
    }

    /**
     * @brief Decode a single integer from a BitReader that is already
     *        positioned at the start of that integer's bits.
     * Advances the reader by exactly ctx.bitsPerInt bits.
     */
    static inline int32_t decodeNextFromReader(const DecodeCtx& ctx, BitReader& br) {
        using namespace detail;

        // Left group
        uint32_t leftVal;
        if (ctx.leftRaw) {
            leftVal = br.read(ctx.lb * 8u);
        } else {
            const uint32_t code = br.read(ctx.leftCW);
            if (code >= ctx.leftTable.size())
                throw std::runtime_error("SubIntEncoder::decode: left code out of range");
            leftVal = ctx.leftTable[code];
        }

        // Right group
        uint32_t rightVal;
        if (ctx.rightRaw) {
            rightVal = br.read(ctx.rb * 8u);
        } else {
            const uint32_t code = br.read(ctx.rightCW);
            const auto& tbl = ctx.shared ? ctx.leftTable : ctx.rightTable;
            if (code >= tbl.size())
                throw std::runtime_error("SubIntEncoder::decode: right code out of range");
            rightVal = tbl[code];
        }

        return static_cast<int32_t>(reconstruct(leftVal, rightVal, ctx.mode));
    }
};

} // namespace encodings::encoders
