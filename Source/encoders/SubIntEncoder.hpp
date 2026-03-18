#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Optional debug output.
// Define SUBINT_DEBUG=1 before including this header (or via -DSUBINT_DEBUG=1)
// to enable per-element encode/decode trace prints.
// ---------------------------------------------------------------------------
#ifndef SUBINT_DEBUG
#  define SUBINT_DEBUG 0
#endif

#if SUBINT_DEBUG
#  define SUBINT_DBG(...)  do { std::cerr << "[SubInt] " << __VA_ARGS__ << "\n"; } while(0)
#else
#  define SUBINT_DBG(...)  do {} while(0)
#endif

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
 * @brief Describes a two-group byte split of an integer.
 *
 * For a 4-byte (int32_t) integer, valid splits are:
 *   {1,3}, {2,2}, {3,1}
 *
 * For an 8-byte (int64_t) integer, valid splits are:
 *   {1,7}, {2,6}, {3,5}, {4,4}, {5,3}, {6,2}, {7,1}
 *
 * "left" always refers to the lower-address (little-endian) bytes.
 * The encoder validates at construction that leftBytes + rightBytes == sizeof(T).
 */
struct SplitMode {
    uint8_t leftBytes;   ///< bytes in the left  (low)  group
    uint8_t rightBytes;  ///< bytes in the right (high) group

    constexpr bool operator==(const SplitMode&) const noexcept = default;
};

// ---- Named helpers for int32_t splits ------------------------------------
inline constexpr SplitMode Split13() noexcept { return {1, 3}; }
inline constexpr SplitMode Split22() noexcept { return {2, 2}; }
inline constexpr SplitMode Split31() noexcept { return {3, 1}; }

// ---- Named helpers for int64_t splits ------------------------------------
inline constexpr SplitMode Split17() noexcept { return {1, 7}; }
inline constexpr SplitMode Split26() noexcept { return {2, 6}; }
inline constexpr SplitMode Split35() noexcept { return {3, 5}; }
inline constexpr SplitMode Split44() noexcept { return {4, 4}; }
inline constexpr SplitMode Split53() noexcept { return {5, 3}; }
inline constexpr SplitMode Split62() noexcept { return {6, 2}; }
inline constexpr SplitMode Split71() noexcept { return {7, 1}; }

/**
 * @brief Maximum code widths (bits) by group size in bytes.
 *
 *  1-byte group  ->  4 bits  (up to     16 distinct values)
 *  2-byte group  ->  8 bits  (up to    256 distinct values)
 *  3-byte group  -> 16 bits  (up to 65 536 distinct values)
 *  4-byte group  -> 16 bits  (up to 65 536 distinct values)
 *  5-byte group  -> 32 bits  (up to ~4 billion — uses HashMap for lookup)
 *  6-byte group  -> 32 bits
 *  7-byte group  -> 32 bits
 */
static constexpr uint32_t kMaxCodeWidth1 =  4;
static constexpr uint32_t kMaxCodeWidth2 =  8;
static constexpr uint32_t kMaxCodeWidth3 = 16;
static constexpr uint32_t kMaxCodeWidth4 = 16;
static constexpr uint32_t kMaxCodeWidth5 = 32;
static constexpr uint32_t kMaxCodeWidth6 = 32;
static constexpr uint32_t kMaxCodeWidth7 = 32;

/**
 * @brief Configuration for SubIntEncoder<T>.
 *
 * @tparam T  int32_t or int64_t — controls which SplitMode values are legal
 *            and what the default split is.
 *
 * The encoder throws std::invalid_argument at construction time if
 * splitMode.leftBytes + splitMode.rightBytes != sizeof(T).
 */
template <typename T>
    requires (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>)
struct SubIntConfig {
    /// Default split: {2,2} for int32_t, {4,4} for int64_t.
    SplitMode splitMode = []() -> SplitMode {
        if constexpr (std::is_same_v<T, int32_t>) return Split22();
        else                                        return Split44();
    }();
    uint32_t  leftCodeWidth  = 0;   ///< 0 = adaptive
    uint32_t  rightCodeWidth = 0;   ///< 0 = adaptive
    BitOrder  bitOrder       = BitOrder::LSB;
};

// ============================================================================
// Internal helpers
// ============================================================================
namespace detail {

/// Maximum code width for a group of `bytes` bytes.
inline constexpr uint32_t maxCodeWidth(uint32_t bytes) noexcept {
    switch (bytes) {
        case 1: return kMaxCodeWidth1;
        case 2: return kMaxCodeWidth2;
        case 3: return kMaxCodeWidth3;
        case 4: return kMaxCodeWidth4;
        case 5: return kMaxCodeWidth5;
        case 6: return kMaxCodeWidth6;
        case 7: return kMaxCodeWidth7;
    }
    return 0;
}

/**
 * @brief Whether a group of `bytes` bytes should use a flat lookup array
 *        (vs. HashMap) for encode/decode.
 *
 * Groups of 1–4 bytes have a max code width of ≤ 16 bits, so the flat decode
 * table is at most 2^16 = 65 536 entries — perfectly acceptable.
 * Groups of 5–7 bytes have a max code width of 32 bits; a flat array indexed
 * by code would need up to 2^32 entries, so we use HashMaps instead.
 *
 * Note: the flat *encode* array is indexed by value, not code.  For 3-byte
 * groups the value space is 2^24 = 16 M entries (64 MB) — acceptable.
 * For 4-byte groups 2^32 entries would be 256 MB+, so encode also falls back
 * to HashMap for groups ≥ 4 bytes.  The flat *decode* table for 4-byte groups
 * is still fine (max 2^16 entries for codes), and is handled separately in
 * DecodeCtx via useFlatArray(bytes).
 */
inline constexpr bool useFlatArray(uint32_t bytes) noexcept {
    return bytes <= 4u;
}

/// Smallest power-of-two bit count that can index `n` distinct values.
/// Returns 1 as the minimum (even for n ≤ 1).
inline uint32_t adaptiveWidth(uint32_t n, uint32_t maxW) noexcept {
    if (n <= 1) return 1;
    uint32_t bits = static_cast<uint32_t>(std::bit_width(n - 1u));
    // Round up to the next power-of-two bit count: 1,2,4,8,16,32
    uint32_t w = 1;
    while (w < bits) w <<= 1;
    return std::min(w, maxW);
}

/// Extract the left group value from a raw unsigned integer (little-endian).
inline uint64_t extractLeft(uint64_t raw, uint32_t lb) noexcept {
    const uint64_t mask = (lb >= 8u) ? ~uint64_t{0}
                                     : ((uint64_t{1} << (lb * 8u)) - 1u);
    return raw & mask;
}

/// Extract the right group value from a raw unsigned integer.
inline uint64_t extractRight(uint64_t raw, uint32_t lb) noexcept {
    return raw >> (lb * 8u);
}

/// Reconstruct a raw unsigned integer from left and right group values.
inline uint64_t reconstruct(uint64_t left, uint64_t right, uint32_t lb) noexcept {
    return left | (right << (lb * 8u));
}

/// Validate a user-supplied code width is a non-zero power of two ≤ maxW.
inline bool validWidth(uint32_t w, uint32_t maxW) noexcept {
    return w != 0 && w <= maxW && (w & (w - 1)) == 0;
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

inline uint8_t  readU8 (const uint8_t* p) { return *p; }
inline uint16_t readU16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
inline uint32_t readU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])         |
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
// Dictionary size-benefit check
// ---------------------------------------------------------------------------

/**
 * @brief Returns true if a dictionary encoding is cheaper than raw storage.
 *
 * A dictionary is only worth building when the space it saves in the bit
 * stream exceeds the space it costs in the header:
 *
 *   dict_header_bytes = 5 + dictSize * groupBytes
 *   stream_with_dict  = N * codeWidth  bits  →  ceil / 8 bytes
 *   stream_raw        = N * groupBytes * 8 bits  →  N * groupBytes bytes
 *
 *   saving = stream_raw - stream_with_dict - dict_header_bytes  >  0
 *
 * @param N          Number of elements.
 * @param dictSize   Number of distinct group values.
 * @param codeWidth  Bit width assigned by adaptiveWidth().
 * @param groupBytes Bytes per group value.
 */
inline bool dictWorthIt(size_t   N,
                        size_t   dictSize,
                        uint32_t codeWidth,
                        uint32_t groupBytes) noexcept
{
    const size_t rawBytes    = N * groupBytes;
    const size_t streamBytes = (N * codeWidth + 7u) / 8u;
    const size_t headerBytes = 5u + dictSize * groupBytes;
    // Dict is beneficial only when it actually saves space vs raw.
    return (streamBytes + headerBytes) < rawBytes;
}

// ---------------------------------------------------------------------------
// Dictionary serialisation helpers
// ---------------------------------------------------------------------------

/**
 * Write one dictionary to `buf`.
 *
 * Format:
 *   [code_width  : 1 byte ]
 *   [num_entries : 4 bytes]   (uint32_t — supports up to 2^32 entries)
 *   for each entry (in code order):
 *       [key : groupBytes bytes, little-endian]
 */
void writeDict(
    std::vector<uint8_t>&              buf,
    const HashMap<uint64_t, uint32_t>& dict,
    uint32_t                           codeWidth,
    uint32_t                           groupBytes)
{
    // Sort by code so the decoder can recover the dict from keys in order.
    std::vector<std::pair<uint64_t, uint32_t>> entries(dict.begin(), dict.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b){ return a.second < b.second; });

    writeU8 (buf, static_cast<uint8_t> (codeWidth));
    writeU32(buf, static_cast<uint32_t>(entries.size()));
    for (auto& [key, _] : entries) {
        for (uint32_t b = 0; b < groupBytes; ++b)
            buf.push_back(static_cast<uint8_t>(key >> (b * 8)));
    }
}

/// Read one dictionary back.
struct ReadDictResult {
    HashMap<uint32_t, uint64_t> codeToValue; ///< code (uint32_t) → group value (uint64_t)
    uint32_t                    codeWidth;
    size_t                      bytesRead;
};

ReadDictResult readDict(const uint8_t* p, uint32_t groupBytes) {
    ReadDictResult r;
    r.codeWidth       = readU8 (p);   p += 1;
    const uint32_t n  = readU32(p);   p += 4;
    r.bytesRead = 5 + static_cast<size_t>(n) * groupBytes;
    r.codeToValue.reserve(n);
    for (uint32_t code = 0; code < n; ++code) {
        uint64_t key = 0;
        for (uint32_t b = 0; b < groupBytes; ++b)
            key |= static_cast<uint64_t>(*p++) << (b * 8);
        r.codeToValue[code] = key;
    }
    return r;
}

} // namespace detail

// ============================================================================
// SubIntEncoder<T>
// ============================================================================

/**
 * @brief Sub-integer code encoder for int32_t or int64_t values.
 *
 * Splits each integer into two byte groups according to a chosen SplitMode,
 * builds a dictionary for each group, and packs the resulting codes into a
 * compact bit stream.
 *
 * Template parameter T must be int32_t or int64_t.
 * The SplitMode must satisfy leftBytes + rightBytes == sizeof(T); otherwise
 * the constructor throws std::invalid_argument.
 *
 * ── Wire format ──────────────────────────────────────────────────────────────
 *
 *   [num_ints            : 8 bytes]
 *   [left_group_bytes    : 1 byte ]
 *   [right_group_bytes   : 1 byte ]
 *   [bit_order           : 1 byte ]   (0 = LSB-first, 1 = MSB-first)
 *   [fallback_flags      : 1 byte ]   bit0 = left_raw, bit1 = right_raw
 *   [shared_dict_flag    : 1 byte ]   1 if equal-size groups share one dict
 *   -- if left NOT raw:
 *   [left dict                    ]   see writeDict() / readDict()
 *   -- if right NOT raw AND NOT shared:
 *   [right dict                   ]
 *   [encoded_ints_length : 8 bytes]
 *   [encoded_ints                 ]   packed bit stream
 *
 * ── Dictionary format ────────────────────────────────────────────────────────
 *
 *   [code_width  : 1 byte ]
 *   [num_entries : 4 bytes]
 *   [key_0 … key_n : groupBytes bytes each, little-endian]
 *
 * ── Flat vs. HashMap lookup ──────────────────────────────────────────────────
 *
 *   Encode (value → code):
 *     Groups 1–3 bytes: flat array indexed by value (max 16 M entries).
 *     Groups 4–7 bytes: HashMap<uint64_t, uint32_t>.
 *
 *   Decode (code → value):
 *     Groups 1–4 bytes (maxCodeWidth ≤ 16): flat vector indexed by code
 *       (max 65 536 entries).
 *     Groups 5–7 bytes (maxCodeWidth = 32): HashMap<uint32_t, uint64_t>.
 *
 * ── Shared-dict optimisation ─────────────────────────────────────────────────
 *
 *   When leftBytes == rightBytes (symmetric splits: {2,2}, {4,4}) the encoder
 *   checks whether a single merged dictionary saves bytes overall.  If so,
 *   only the left dict is written and shared_dict_flag is set to 1.
 *
 * ── Raw fallback ─────────────────────────────────────────────────────────────
 *
 *   If a group exceeds its max code capacity, values are stored verbatim
 *   (groupBytes * 8 bits each).  If both groups fall back the stream is plain
 *   sizeof(T)-byte integers (memcpy path).
 *
 * ── Random access ────────────────────────────────────────────────────────────
 *
 *   bitsPerInt = leftCW + rightCW  (constant; raw sides count as groupBytes*8).
 *   Element i starts at bit offset i * bitsPerInt.
 */
template <typename T>
    requires (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>)
class SubIntEncoder : public encodings::Codec<T> {
public:
    using UnsignedT = std::make_unsigned_t<T>;

    explicit SubIntEncoder(SubIntConfig<T> cfg = {}) : cfg_(cfg) {
        const uint32_t total = cfg_.splitMode.leftBytes + cfg_.splitMode.rightBytes;
        if (total != sizeof(T)) {
            throw std::invalid_argument(
                "SubIntEncoder: splitMode bytes (" +
                std::to_string(cfg_.splitMode.leftBytes) + "+" +
                std::to_string(cfg_.splitMode.rightBytes) +
                ") do not sum to sizeof(T) = " + std::to_string(sizeof(T)));
        }
    }

    // -----------------------------------------------------------------------
    // Codec<T>
    // -----------------------------------------------------------------------

    encodings::EncodedData encode(std::span<const T> data) override {
        using namespace detail;

        const size_t   N     = data.size();
        const uint32_t lb    = cfg_.splitMode.leftBytes;
        const uint32_t rb    = cfg_.splitMode.rightBytes;
        const uint32_t lMaxW = maxCodeWidth(lb);
        const uint32_t rMaxW = maxCodeWidth(rb);

        // ---- 1. Build dictionaries in a single scan ----------------------
        //
        // Groups are stored interleaved: [left0, right0, left1, right1, ...]
        // for cache locality during the encoding pass.
        std::vector<uint64_t> groups(N * 2);
        HashMap<uint64_t, uint32_t> leftDict, rightDict;
        // For groups with maxCodeWidth ≤ 16 we know the max capacity; reserve
        // it.  For 32-bit maxCodeWidth, reserving 2^32 is impractical — let
        // the map grow dynamically.
        if (lMaxW <= 16) leftDict.reserve(size_t{1} << lMaxW);
        if (rMaxW <= 16) rightDict.reserve(size_t{1} << rMaxW);

        for (size_t i = 0; i < N; ++i) {
            const UnsignedT raw = static_cast<UnsignedT>(data[i]);
            const uint64_t  lv  = extractLeft (raw, lb);
            const uint64_t  rv  = extractRight(raw, lb);
            groups[2*i]   = lv;
            groups[2*i+1] = rv;
            leftDict.emplace (lv, static_cast<uint32_t>(leftDict.size()));
            rightDict.emplace(rv, static_cast<uint32_t>(rightDict.size()));
        }

        uint32_t leftCW   = 0;
        uint32_t rightCW  = 0;
        bool     leftRaw  = false;
        bool     rightRaw = false;
        bool     shared   = false;

        // Check overflow.  Use uint64_t shift to avoid UB when maxW == 32.
        {
            const uint64_t lCap = uint64_t{1} << lMaxW;
            const uint64_t rCap = uint64_t{1} << rMaxW;
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

        // Size-benefit check: if the dictionary costs more than it saves,
        // fall back to raw storage for that side.
        if (!leftRaw && !dictWorthIt(N, leftDict.size(), leftCW, lb)) {
            SUBINT_DBG("left dict not worth it (" << leftDict.size()
                       << " entries, " << lb << "B groups, CW=" << leftCW
                       << ") – raw fallback");
            leftDict.clear();
            leftRaw = true;
            leftCW  = lb * 8u;
        }
        if (!rightRaw && !dictWorthIt(N, rightDict.size(), rightCW, rb)) {
            SUBINT_DBG("right dict not worth it (" << rightDict.size()
                       << " entries, " << rb << "B groups, CW=" << rightCW
                       << ") – raw fallback");
            rightDict.clear();
            rightRaw = true;
            rightCW  = rb * 8u;
        }

        // Shared-dict check: only for symmetric splits (leftBytes == rightBytes).
        if (lb == rb && !leftRaw && !rightRaw) {
            shared = checkSharedBenefit(leftDict, rightDict, leftCW, rightCW, lMaxW, lb, N);
            if (shared) {
                buildMergedDict(leftDict, rightDict, lMaxW, leftCW);
                rightDict.clear();
                rightCW = leftCW;
            }
        }

        const bool bothRaw = leftRaw && rightRaw;

        // ---- 2. Build encode lookup tables (value → code) ----------------
        //
        // Flat arrays (indexed by value) are used for groups of 1–3 bytes
        // where the value space is at most 2^24 = 16 M entries (64 MB).
        // For groups of 4+ bytes the HashMap is used directly for encoding.
        const bool useFlatEncLeft  = !leftRaw  && lb <= 3u;
        const bool useFlatEncRight = !rightRaw && rb <= 3u;
        const uint64_t lArrSize = useFlatEncLeft  ? (uint64_t{1} << (lb * 8u)) : 0u;
        const uint64_t rArrSize = useFlatEncRight ? (uint64_t{1} << (rb * 8u)) : 0u;
        std::vector<uint32_t> leftCodeArr (static_cast<size_t>(lArrSize), 0u);
        std::vector<uint32_t> rightCodeArr(static_cast<size_t>(rArrSize), 0u);
        if (useFlatEncLeft)
            for (auto& [val, code] : leftDict)  leftCodeArr[static_cast<size_t>(val)]  = code;
        if (useFlatEncRight && !shared)
            for (auto& [val, code] : rightDict) rightCodeArr[static_cast<size_t>(val)] = code;
        else if (useFlatEncRight && shared)
            for (auto& [val, code] : leftDict)  rightCodeArr[static_cast<size_t>(val)] = code;

        // ---- 3. Serialise header ------------------------------------------
        std::vector<uint8_t> header;
        header.reserve(64);
        writeU64(header, static_cast<uint64_t>(N));
        writeU8 (header, static_cast<uint8_t>(lb));
        writeU8 (header, static_cast<uint8_t>(rb));
        writeU8 (header, (cfg_.bitOrder == BitOrder::MSB) ? 1u : 0u);

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
            stream.resize(N * sizeof(T));
            std::memcpy(stream.data(), data.data(), stream.size());
        } else {
            const uint32_t lBits = leftRaw  ? (lb * 8u) : leftCW;
            const uint32_t rBits = rightRaw ? (rb * 8u) : rightCW;
            const size_t totalBits = N * (lBits + rBits);
            stream.reserve((totalBits + 7) / 8);

            SUBINT_DBG("encode: lb=" << lb << " rb=" << rb
                       << " leftCW=" << lBits << " rightCW=" << rBits
                       << " leftRaw=" << leftRaw << " rightRaw=" << rightRaw
                       << " shared=" << shared);

            BitWriter bw(stream, cfg_.bitOrder);
            for (size_t i = 0; i < N; ++i) {
                const uint64_t lv = groups[2*i];
                const uint64_t rv = groups[2*i+1];

                // Left side
                uint32_t lcode = 0;
                if (leftRaw) {
                    writeBits64(bw, lv, lb * 8u);
                } else if (useFlatEncLeft) {
                    lcode = leftCodeArr[static_cast<size_t>(lv)];
                    bw.write(lcode, leftCW);
                } else {
                    lcode = leftDict.at(lv);
                    bw.write(lcode, leftCW);
                }

                // Right side
                uint32_t rcode = 0;
                if (rightRaw) {
                    writeBits64(bw, rv, rb * 8u);
                } else if (useFlatEncRight) {
                    rcode = rightCodeArr[static_cast<size_t>(rv)];
                    bw.write(rcode, rightCW);
                } else {
                    const auto& rMap = shared ? leftDict : rightDict;
                    rcode = rMap.at(rv);
                    bw.write(rcode, rightCW);
                }

                SUBINT_DBG("  enc[" << i << "]  lv=0x" << std::hex << lv
                           << (leftRaw  ? " lraw" : (" lcode=" + std::to_string(lcode)))
                           << "  rv=0x" << std::hex << rv
                           << (rightRaw ? " rraw" : (" rcode=" + std::to_string(rcode))));
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
        meta.dataType         = encodings::core::typeToDataType<T>;
        meta.elementCount     = N;
        meta.compressedSize   = out.size();
        meta.uncompressedSize = N * sizeof(T);
        meta.supportsRandomAccess = !bothRaw;

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // -----------------------------------------------------------------------

    std::vector<T> decodeAll(const encodings::EncodedData& encoded) override {
        const DecodeCtx& ctx = this->getCachedCtx(encoded);
        return decodeSequential(ctx, 0, ctx.N);
    }

    std::optional<T> decodeAt(const encodings::EncodedData& encoded,
                               size_t index) override {
        const DecodeCtx& ctx = this->getCachedCtx(encoded);
        if (index >= ctx.N) return std::nullopt;
        return decodeOne(ctx, index);
    }

    std::vector<T> decodeRange(const encodings::EncodedData& encoded,
                                size_t start, size_t end) override {
        const DecodeCtx& ctx = this->getCachedCtx(encoded);
        if (start >= ctx.N) return {};
        end = std::min(end, ctx.N);
        return decodeSequential(ctx, start, end);
    }

    // -----------------------------------------------------------------------

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::SubIntEncoding;
    }

    std::string name() const override {
        return "SubInt(" +
               std::to_string(cfg_.splitMode.leftBytes) + "|" +
               std::to_string(cfg_.splitMode.rightBytes) + ")";
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
        const uint32_t lb    = cfg_.splitMode.leftBytes;
        const uint32_t rb    = cfg_.splitMode.rightBytes;
        const uint32_t lBits = (cfg_.leftCodeWidth  != 0) ? cfg_.leftCodeWidth
                                                           : detail::maxCodeWidth(lb);
        const uint32_t rBits = (cfg_.rightCodeWidth != 0) ? cfg_.rightCodeWidth
                                                           : detail::maxCodeWidth(rb);
        const size_t streamBytes = (elementCount * (lBits + rBits) + 7) / 8;
        return 512 + 8 + streamBytes;
    }

    // -----------------------------------------------------------------------
    // Histogram support
    // -----------------------------------------------------------------------

    /**
     * @brief Frequency histogram for one group side (left or right).
     */
    struct GroupHistogram {
        HashMap<uint64_t, size_t> counts;
        size_t                    totalValues{0};
        uint32_t                  groupBytes{0};
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
     * @return        Histograms struct with left and right GroupHistograms.
     */
    Histograms computeHistograms(std::span<const T> data,
                                 bool verbose = false) const
    {
        using namespace detail;
        const uint32_t lb = cfg_.splitMode.leftBytes;

        Histograms h;
        h.splitMode        = cfg_.splitMode;
        h.left.groupBytes  = lb;
        h.right.groupBytes = cfg_.splitMode.rightBytes;
        h.left.totalValues = h.right.totalValues = data.size();

        for (T v : data) {
            const UnsignedT raw = static_cast<UnsignedT>(v);
            ++h.left.counts [extractLeft (raw, lb)];
            ++h.right.counts[extractRight(raw, lb)];
        }

        if (verbose) printHistograms(h, std::cout);
        return h;
    }

    /**
     * @brief Print pre-computed histograms to an output stream.
     *
     * Entries are sorted by descending frequency, then ascending value.
     * A simple bar chart is rendered alongside the counts.
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

            std::vector<std::pair<uint64_t, size_t>> entries(
                gh.counts.begin(), gh.counts.end());
            std::sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) {
                    return a.second != b.second
                        ? a.second > b.second
                        : a.first  < b.first;
                });

            const size_t maxCount = entries.front().second;
            constexpr int kBarWidth = 40;
            const int valWidth = gh.groupBytes <= 1 ? 3
                               : gh.groupBytes == 2 ? 5
                               : gh.groupBytes <= 4 ? 8 : 14;

            size_t num_less_than = 0;
            constexpr size_t num_to_be_less_than = 1000;
            for (const auto& [val, cnt] : entries) {
                if (cnt < num_to_be_less_than) { ++num_less_than; continue; }
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
                    << std::string(barLen, '#') << "\n";
            }
            if (num_less_than > 0) {
                const double pct = gh.totalValues > 0
                    ? 100.0 * num_less_than / gh.totalValues : 0.0;
                out << "  (and " << num_less_than << " less than "
                    << num_to_be_less_than << " occurrences, "
                    << std::fixed << std::setprecision(2) << pct << "% of total)\n";
            }
            out << "\n";
        };

        printOne(h.left,  "Left");
        printOne(h.right, "Right");
    }

private:
    SubIntConfig<T> cfg_;

    // -----------------------------------------------------------------------
    // Decode context type (must be declared before the cache members below)
    // -----------------------------------------------------------------------

    struct DecodeCtx {
        size_t   N{0};
        uint32_t lb{0};        ///< left group bytes
        uint32_t rb{0};        ///< right group bytes
        BitOrder order{BitOrder::LSB};
        bool     leftRaw{false};
        bool     rightRaw{false};
        bool     bothRaw{false};
        bool     shared{false};

        uint32_t leftCW{0};    ///< code width in bits (== lb*8 if raw)
        uint32_t rightCW{0};
        uint32_t bitsPerInt{0};

        // ── Decode tables: code → group value ──────────────────────────────
        //
        // For groups 1–4 bytes (maxCodeWidth ≤ 16): flat vector<uint64_t>
        //   indexed by code; size = 2^codeWidth (max 65 536 entries).
        // For groups 5–7 bytes (maxCodeWidth = 32): HashMap<uint32_t,uint64_t>.
        //
        // Exactly one of {flatTable, hashTable} is populated per side.
        std::vector<uint64_t>       leftFlatTable;
        std::vector<uint64_t>       rightFlatTable;
        HashMap<uint32_t, uint64_t> leftHashTable;
        HashMap<uint32_t, uint64_t> rightHashTable;

        bool useFlatLeft{false};   ///< true iff leftFlatTable is in use
        bool useFlatRight{false};  ///< true iff rightFlatTable is in use

        const uint8_t* streamPtr{nullptr};
        size_t         streamBytes{0};
    };

    // -----------------------------------------------------------------------
    // Cached decode context — avoids re-parsing the header on every decodeAt /
    // decodeRange call when the same EncodedData object is reused (e.g. during
    // random-access benchmarking).  Keyed by the raw data pointer + size so it
    // is automatically invalidated if a new EncodedData is passed.
    // mutable so that the const-correct decode path can update it.
    // -----------------------------------------------------------------------
    mutable const uint8_t* cachedPtr_  {nullptr};
    mutable size_t         cachedSize_ {0};
    mutable DecodeCtx      cachedCtx_;

    const DecodeCtx& getCachedCtx(const encodings::EncodedData& encoded) const {
        const uint8_t* ptr  = encoded.data().data();
        const size_t   sz   = encoded.data().size();
        if (ptr != cachedPtr_ || sz != cachedSize_) {
            cachedCtx_  = parseHeader(encoded);
            cachedPtr_  = ptr;
            cachedSize_ = sz;
        }
        return cachedCtx_;
    }

    // -----------------------------------------------------------------------
    // Shared-dict benefit check (symmetric splits only)
    // -----------------------------------------------------------------------

    /**
     * Returns true if a single merged dictionary for both groups saves bytes
     * compared to two separate dictionaries.
     *
     * Only called for symmetric splits (leftBytes == rightBytes).
     *
     * @param groupBytes  Number of bytes in each group (lb == rb).
     */
    static bool checkSharedBenefit(
        const HashMap<uint64_t, uint32_t>& leftDict,
        const HashMap<uint64_t, uint32_t>& rightDict,
        uint32_t leftCW, uint32_t rightCW,
        uint32_t maxW, uint32_t groupBytes, size_t N)
    {
        HashSet<uint64_t> unionSet;
        for (auto& [v, _] : leftDict)  unionSet.insert(v);
        for (auto& [v, _] : rightDict) unionSet.insert(v);
        const uint32_t unionSize = static_cast<uint32_t>(unionSet.size());

        // Use uint64_t shift to avoid UB when maxW == 32.
        const uint64_t capacity = uint64_t{1} << maxW;
        if (static_cast<uint64_t>(unionSize) > capacity) return false;

        const uint32_t mergedCW = detail::adaptiveWidth(unionSize, maxW);

        const int64_t streamSavingBits =
            static_cast<int64_t>(leftCW + rightCW) -
            static_cast<int64_t>(2 * mergedCW);
        const int64_t streamSavingBytes =
            (static_cast<int64_t>(N) * streamSavingBits) / 8;

        // Header saving from eliminating the right dict
        // (5 bytes fixed overhead + entries × groupBytes).
        const int64_t eliminatedDictBytes =
            5 + static_cast<int64_t>(rightDict.size()) * static_cast<int64_t>(groupBytes);

        const int64_t extraLeftEntries =
            static_cast<int64_t>(unionSize) - static_cast<int64_t>(leftDict.size());
        const int64_t extraLeftDictBytes =
            extraLeftEntries * static_cast<int64_t>(groupBytes);

        return (streamSavingBytes + eliminatedDictBytes - extraLeftDictBytes) > 0;
    }

    /// Rebuild leftDict as the merged dict for both sides.
    static void buildMergedDict(
        HashMap<uint64_t, uint32_t>&       mergedDict,
        const HashMap<uint64_t, uint32_t>& rightDict,
        uint32_t                           maxW,
        uint32_t&                          codeWidth)
    {
        uint32_t nextCode = static_cast<uint32_t>(mergedDict.size());
        for (auto& [val, _] : rightDict)
            if (mergedDict.emplace(val, nextCode).second) ++nextCode;
        const uint32_t n = static_cast<uint32_t>(mergedDict.size());
        codeWidth = detail::adaptiveWidth(n, maxW);
        uint32_t c = 0;
        for (auto& [_, code] : mergedDict) code = c++;
    }

    // -----------------------------------------------------------------------
    // Helpers for reading/writing more than 32 bits via BitWriter/BitReader
    // -----------------------------------------------------------------------

    /**
     * @brief Write the lowest @p bits bits of @p value into @p bw.
     *
     * BitWriter::write() accepts at most 32-bit values.  For raw groups with
     * lb or rb > 4 (i.e. 40, 48, or 56 bits) we split into two writes: the
     * lower 32 bits first, then the remaining upper bits.
     */
    static void writeBits64(BitWriter& bw, uint64_t value, uint32_t bits) {
        if (bits <= 32u) {
            bw.write(static_cast<uint32_t>(value), bits);
        } else {
            // Write the lower 32 bits, then the upper (bits-32) bits.
            bw.write(static_cast<uint32_t>(value & 0xFFFF'FFFFu), 32u);
            bw.write(static_cast<uint32_t>(value >> 32u), bits - 32u);
        }
    }

    /**
     * @brief Read @p bits bits from @p br and return as uint64_t.
     *
     * BitReader::read() returns uint32_t (max 32 bits).  For raw groups with
     * lb or rb > 4 we read in two calls and reassemble.
     */
    static uint64_t readBits64(BitReader& br, uint32_t bits) {
        if (bits <= 32u) {
            return static_cast<uint64_t>(br.read(bits));
        } else {
            // Read the lower 32 bits first, then the upper (bits-32) bits.
            const uint64_t lo = static_cast<uint64_t>(br.read(32u));
            const uint64_t hi = static_cast<uint64_t>(br.read(bits - 32u));
            return lo | (hi << 32u);
        }
    }

    // -----------------------------------------------------------------------
    // Decode support
    // -----------------------------------------------------------------------

    static DecodeCtx parseHeader(const encodings::EncodedData& encoded) {
        using namespace detail;
        const uint8_t* p   = encoded.data().data();
        const uint8_t* end = p + encoded.size();

        auto require = [&](size_t n, const char* ctx) {
            if (p + n > end)
                throw std::runtime_error(
                    std::string("SubIntEncoder::decode: truncated at ") + ctx);
        };

        require(8, "num_ints");
        DecodeCtx ctx;
        ctx.N = static_cast<size_t>(readU64(p)); p += 8;

        require(2, "group_bytes");
        ctx.lb = readU8(p); p += 1;
        ctx.rb = readU8(p); p += 1;

        require(1, "bit_order");
        ctx.order = readU8(p) ? BitOrder::MSB : BitOrder::LSB; p += 1;

        require(1, "fallback_flags");
        const uint8_t ff = readU8(p); p += 1;
        ctx.leftRaw  = (ff & 0x01) != 0;
        ctx.rightRaw = (ff & 0x02) != 0;
        ctx.bothRaw  = ctx.leftRaw && ctx.rightRaw;

        require(1, "shared_dict_flag");
        ctx.shared = readU8(p) != 0; p += 1;

        ctx.useFlatLeft  = useFlatArray(ctx.lb);
        ctx.useFlatRight = useFlatArray(ctx.rb);

        // ---- Left dict / raw ----
        if (!ctx.leftRaw) {
            require(5, "left_dict_header");
            auto dr = readDict(p, ctx.lb);
            ctx.leftCW = dr.codeWidth;
            if (ctx.useFlatLeft) {
                ctx.leftFlatTable.assign(size_t{1} << ctx.leftCW, 0);
                for (auto& [code, val] : dr.codeToValue)
                    ctx.leftFlatTable[code] = val;
            } else {
                ctx.leftHashTable = std::move(dr.codeToValue);
            }
            p += dr.bytesRead;
        } else {
            ctx.leftCW = ctx.lb * 8u;
        }

        // ---- Right dict / raw / shared ----
        if (!ctx.rightRaw && !ctx.shared) {
            require(5, "right_dict_header");
            auto dr = readDict(p, ctx.rb);
            ctx.rightCW = dr.codeWidth;
            if (ctx.useFlatRight) {
                ctx.rightFlatTable.assign(size_t{1} << ctx.rightCW, 0);
                for (auto& [code, val] : dr.codeToValue)
                    ctx.rightFlatTable[code] = val;
            } else {
                ctx.rightHashTable = std::move(dr.codeToValue);
            }
            p += dr.bytesRead;
        } else if (ctx.shared) {
            ctx.rightCW = ctx.leftCW;
            // Right side reuses the left table — no separate allocation needed.
        } else {
            ctx.rightCW = ctx.rb * 8u;
        }

        ctx.bitsPerInt = ctx.leftCW + ctx.rightCW;

        require(8, "stream_length");
        ctx.streamBytes = static_cast<size_t>(readU64(p)); p += 8;

        require(ctx.streamBytes, "stream_data");
        ctx.streamPtr = p;

        return ctx;
    }

    // ---- Single-element decode (random access) ----------------------------

    static T decodeOne(const DecodeCtx& ctx, size_t index) {
        if (ctx.bothRaw) {
            T v;
            std::memcpy(&v, ctx.streamPtr + index * sizeof(T), sizeof(T));
            return v;
        }
        const size_t bitPos = index * ctx.bitsPerInt;
        BitReader br(ctx.streamPtr, ctx.streamBytes, ctx.order);
        br.seekToBit(bitPos);
        return decodeNextFromReader(ctx, br);
    }

    // ---- Sequential decode -----------------------------------------------

    /**
     * @brief Decode [start, end) in a single pass.
     *        The decode variant is dispatched once outside the hot loop.
     */
    static std::vector<T> decodeSequential(
        const DecodeCtx& ctx, size_t start, size_t end)
    {
        const size_t count = end - start;
        std::vector<T> result;
        result.reserve(count);

        if (ctx.bothRaw) {
            result.resize(count);
            std::memcpy(result.data(),
                        ctx.streamPtr + start * sizeof(T),
                        count * sizeof(T));
            return result;
        }

        BitReader br(ctx.streamPtr, ctx.streamBytes, ctx.order);
        br.seekToBit(start * ctx.bitsPerInt);

        // Resolve a left code to its group value.
        auto resolveLeft = [&](uint32_t code) -> uint64_t {
            if (ctx.useFlatLeft) return ctx.leftFlatTable[code];
            return ctx.leftHashTable.at(code);
        };
        // Resolve a right code to its group value (shared uses left table).
        auto resolveRight = [&](uint32_t code) -> uint64_t {
            if (ctx.shared) {
                if (ctx.useFlatLeft) return ctx.leftFlatTable[code];
                return ctx.leftHashTable.at(code);
            }
            if (ctx.useFlatRight) return ctx.rightFlatTable[code];
            return ctx.rightHashTable.at(code);
        };

        auto runLoop = [&](auto decodeOneFn) {
            for (size_t i = 0; i < count; ++i)
                result.push_back(decodeOneFn(br, start + i));
        };

        const uint32_t lb = ctx.lb;

        SUBINT_DBG("decode: lb=" << lb << " rb=" << ctx.rb
                   << " leftCW=" << ctx.leftCW << " rightCW=" << ctx.rightCW
                   << " bitsPerInt=" << ctx.bitsPerInt
                   << " leftRaw=" << ctx.leftRaw << " rightRaw=" << ctx.rightRaw
                   << " shared=" << ctx.shared
                   << " useFlatLeft=" << ctx.useFlatLeft
                   << " useFlatRight=" << ctx.useFlatRight);

        if (!ctx.leftRaw && !ctx.rightRaw) {
            const uint32_t lcw = ctx.leftCW, rcw = ctx.rightCW;
            runLoop([&](BitReader& r, [[maybe_unused]] size_t idx) -> T {
                uint32_t lcode = r.read(lcw);
                uint32_t rcode = r.read(rcw);
                uint64_t lval  = resolveLeft(lcode);
                uint64_t rval  = resolveRight(rcode);
                T result = static_cast<T>(detail::reconstruct(lval, rval, lb));
                SUBINT_DBG("  dec[" << idx << "]  lcode=" << lcode << " lval=0x" << std::hex << lval
                           << "  rcode=" << std::dec << rcode << " rval=0x" << std::hex << rval
                           << "  result=0x" << (typename std::make_unsigned<T>::type)result << std::dec);
                return result;
            });
        } else if (ctx.leftRaw && !ctx.rightRaw) {
            const uint32_t lbits = lb * 8u, rcw = ctx.rightCW;
            runLoop([&](BitReader& r, [[maybe_unused]] size_t idx) -> T {
                uint64_t lval  = r.read(lbits);
                uint32_t rcode = r.read(rcw);
                uint64_t rval  = resolveRight(rcode);
                T result = static_cast<T>(detail::reconstruct(lval, rval, lb));
                SUBINT_DBG("  dec[" << idx << "]  lval(raw)=0x" << std::hex << lval
                           << "  rcode=" << std::dec << rcode << " rval=0x" << std::hex << rval
                           << "  result=0x" << (typename std::make_unsigned<T>::type)result << std::dec);
                return result;
            });
        } else if (!ctx.leftRaw && ctx.rightRaw) {
            const uint32_t lcw = ctx.leftCW, rbits = ctx.rb * 8u;
            runLoop([&](BitReader& r, [[maybe_unused]] size_t idx) -> T {
                uint32_t lcode = r.read(lcw);
                uint64_t lval  = resolveLeft(lcode);
                uint64_t rval  = r.read(rbits);
                T result = static_cast<T>(detail::reconstruct(lval, rval, lb));
                SUBINT_DBG("  dec[" << idx << "]  lcode=" << lcode << " lval=0x" << std::hex << lval
                           << "  rval(raw)=0x" << rval
                           << "  result=0x" << (typename std::make_unsigned<T>::type)result << std::dec);
                return result;
            });
        } else {
            // Both raw via bit stream (not the plain-memcpy bothRaw path)
            const uint32_t lbits = lb * 8u, rbits = ctx.rb * 8u;
            runLoop([&](BitReader& r, [[maybe_unused]] size_t idx) -> T {
                uint64_t lval = r.read(lbits);
                uint64_t rval = r.read(rbits);
                T result = static_cast<T>(detail::reconstruct(lval, rval, lb));
                SUBINT_DBG("  dec[" << idx << "]  lval(raw)=0x" << std::hex << lval
                           << "  rval(raw)=0x" << rval
                           << "  result=0x" << (typename std::make_unsigned<T>::type)result << std::dec);
                return result;
            });
        }

        return result;
    }

    /**
     * @brief Decode a single integer from a BitReader already positioned at
     *        the start of that integer's bits.  Advances by exactly bitsPerInt.
     */
    static T decodeNextFromReader(const DecodeCtx& ctx, BitReader& br) {
        // Left group
        uint64_t leftVal;
        if (ctx.leftRaw) {
            leftVal = readBits64(br, ctx.lb * 8u);
        } else {
            const uint32_t code = br.read(ctx.leftCW);
            if (ctx.useFlatLeft) {
                if (code >= ctx.leftFlatTable.size())
                    throw std::runtime_error("SubIntEncoder::decode: left code out of range");
                leftVal = ctx.leftFlatTable[code];
            } else {
                auto it = ctx.leftHashTable.find(code);
                if (it == ctx.leftHashTable.end())
                    throw std::runtime_error("SubIntEncoder::decode: left code not found");
                leftVal = it->second;
            }
        }

        // Right group
        uint64_t rightVal;
        if (ctx.rightRaw) {
            rightVal = readBits64(br, ctx.rb * 8u);
        } else {
            const uint32_t code = br.read(ctx.rightCW);
            if (ctx.shared) {
                if (ctx.useFlatLeft) {
                    if (code >= ctx.leftFlatTable.size())
                        throw std::runtime_error("SubIntEncoder::decode: shared right code out of range");
                    rightVal = ctx.leftFlatTable[code];
                } else {
                    auto it = ctx.leftHashTable.find(code);
                    if (it == ctx.leftHashTable.end())
                        throw std::runtime_error("SubIntEncoder::decode: shared right code not found");
                    rightVal = it->second;
                }
            } else if (ctx.useFlatRight) {
                if (code >= ctx.rightFlatTable.size())
                    throw std::runtime_error("SubIntEncoder::decode: right code out of range");
                rightVal = ctx.rightFlatTable[code];
            } else {
                auto it = ctx.rightHashTable.find(code);
                if (it == ctx.rightHashTable.end())
                    throw std::runtime_error("SubIntEncoder::decode: right code not found");
                rightVal = it->second;
            }
        }

        return static_cast<T>(detail::reconstruct(leftVal, rightVal, ctx.lb));
    }
};

// ============================================================================
// Convenience type aliases
// ============================================================================

/// SubIntEncoder specialised for int32_t (splits: {1,3}, {2,2}, {3,1})
using SubInt32Encoder = SubIntEncoder<int32_t>;

/// SubIntEncoder specialised for int64_t (splits: {1,7}–{7,1})
using SubInt64Encoder = SubIntEncoder<int64_t>;

} // namespace encodings::encoders
