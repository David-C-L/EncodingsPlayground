#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/BitPacker.hpp"

// Optional compression back-ends (same guards used by LZ4Encoder / ZstdEncoder)
#if __has_include(<lz4.h>)
#include <lz4.h>
#define PERM_STORE_LZ4 1
#endif
#if __has_include(<zstd.h>)
#include <zstd.h>
#define PERM_STORE_ZSTD 1
#endif

namespace encodings::reorderers {

using encodings::core::BitOrder;
using encodings::core::BitReader;
using encodings::core::BitWriter;

// ---------------------------------------------------------------------------
// Format tags
// ---------------------------------------------------------------------------

enum class PermFormat : uint8_t {
    // Position-permutation formats (for SortReorderer)
    None           = 0,  // No permutation stored; positions unchanged (GrayCode)
    FlatBitPacked  = 1,  // N × bpe bits; O(1) random access
    DeltaBitPacked = 2,  // Delta + zigzag bit-packed; sequential access only
    ChunkRelative  = 3,  // Intra-chunk ranks with implicit base; O(1) random access
                         //   (requires fwd[i] ∈ [chunk(i)*C, (chunk(i)+1)*C) — WindowedSort)

    // Compressed delta variants (better for structured permutations; bulk-decode only)
    DeltaZstd      = 4,  // Delta + zigzag → Zstd compress; sequential access only
    DeltaLZ4       = 5,  // Delta + zigzag → LZ4 compress; faster, slightly larger

    // Group-structured inverse permutation (sort permutations with repeated values)
    ValueGrouped      = 6,  // Per-value-group delta-encoded positions; O(N) unpack
    InverseEliasFano  = 7,  // Per-group Elias-Fano; theoretically optimal; O(N) unpack

    // Compressed intra-chunk rank array (WindowedSort compressed variants)
    ChunkRelativeZstd = 8,  // ChunkRelative ranks → Zstd; sequential access
    ChunkRelativeLZ4  = 9,  // ChunkRelative ranks → LZ4; faster decode
};

// Compile-time random access capability.
template <PermFormat F> inline constexpr bool permFormatSupportsRandomAccess = false;
template <> inline constexpr bool permFormatSupportsRandomAccess<PermFormat::None>          = true;
template <> inline constexpr bool permFormatSupportsRandomAccess<PermFormat::FlatBitPacked> = true;
template <> inline constexpr bool permFormatSupportsRandomAccess<PermFormat::ChunkRelative> = true;
// All others default to false (sequential / bulk decode required)

// ---------------------------------------------------------------------------
// Wire layouts
//
//  None:              [tag:1]
//  FlatBitPacked:     [tag:1][bpe:1][N:8][packed values: ⌈N*bpe/8⌉ bytes]
//  DeltaBitPacked:    [tag:1][bpe:1][firstBpe:1][N:8][first at firstBpe bits]
//                     [N-1 zigzag deltas at bpe bits]
//  ChunkRelative:     [tag:1][chunkSize:4][rankBits:1][N:8]
//                     [intra-chunk ranks: ⌈N*rankBits/8⌉ bytes]
//                     base for chunk k = k * chunkSize (implicit)
//
//  DeltaZstd/LZ4:     [tag:1][bpe:1][firstBpe:1][N:8][compressedSize:4]
//                     [Zstd/LZ4 compressed delta stream]
//
//  ValueGrouped:      [tag:1][N:8][K:8][groupBlobTotalBytes:4]
//                     For each group k:
//                       [count:4][firstBpe:1][first:firstBpe bits]
//                       [deltaBpe:1][count-1 zigzag deltas at deltaBpe bits]
//
//  InverseEliasFano:  [tag:1][N:8][K:8]
//                     [K group sizes as uint32_t — for prefix-sum skipping]
//                     For each group k:
//                       [l:1][m:4][m*l low bits][high unary bitmap ⌈(N>>l)+m⌉ bits]
//
//  ChunkRelativeZstd/LZ4: [tag:1][chunkSize:4][rankBits:1][N:8]
//                     [decompressedSize:4][compressedSize:4][compressed rank bytes]
// ---------------------------------------------------------------------------

namespace detail_ps {

// Write uint64_t N as 8 little-endian bytes.
inline void writeU64(std::vector<uint8_t>& v, uint64_t n) {
    for (int b = 0; b < 8; ++b) v.push_back(static_cast<uint8_t>(n >> (8 * b)));
}
inline uint64_t readU64(const uint8_t* p) {
    uint64_t n = 0;
    for (int b = 0; b < 8; ++b) n |= static_cast<uint64_t>(p[b]) << (8 * b);
    return n;
}
inline void writeU32(std::vector<uint8_t>& v, uint32_t n) {
    for (int b = 0; b < 4; ++b) v.push_back(static_cast<uint8_t>(n >> (8 * b)));
}
inline uint32_t readU32(const uint8_t* p) {
    uint32_t n = 0;
    for (int b = 0; b < 4; ++b) n |= static_cast<uint32_t>(p[b]) << (8 * b);
    return n;
}

// Random O(1) bit read from a byte array (LSB bit order).
inline size_t readBitsAt(const uint8_t* data, size_t bitPos, uint32_t numBits) noexcept {
    if (numBits == 0) return 0;
    const size_t byteStart = bitPos >> 3;
    const uint32_t bitOff  = static_cast<uint32_t>(bitPos & 7);
    uint64_t window = 0;
    const uint32_t bytesNeeded = (bitOff + numBits + 7) >> 3;
    for (uint32_t b = 0; b < bytesNeeded; ++b)
        window |= static_cast<uint64_t>(data[byteStart + b]) << (8u * b);
    window >>= bitOff;
    if (numBits < 64) window &= (uint64_t{1} << numBits) - 1u;
    return static_cast<size_t>(window);
}

inline uint8_t bitsNeeded(size_t maxVal) noexcept {
    return maxVal == 0 ? 1u : static_cast<uint8_t>(std::bit_width(maxVal));
}

// Pack zigzag-delta sequence into a raw byte vector (reused by DeltaBitPacked / DeltaCompressed).
inline std::vector<uint8_t> packDeltaStream(std::span<const size_t> fwdPerm,
                                             uint8_t& outBpe, uint8_t& outFirstBpe) {
    const size_t N = fwdPerm.size();
    outFirstBpe = detail_ps::bitsNeeded(N > 1 ? N - 1 : 1);

    size_t maxZigzag = 0;
    for (size_t i = 1; i < N; ++i) {
        const int64_t d = static_cast<int64_t>(fwdPerm[i]) - static_cast<int64_t>(fwdPerm[i - 1]);
        const uint64_t zz = static_cast<uint64_t>((d << 1) ^ (d >> 63));
        if (zz > maxZigzag) maxZigzag = zz;
    }
    outBpe = detail_ps::bitsNeeded(maxZigzag > 0 ? maxZigzag : 1);

    std::vector<uint8_t> raw;
    raw.reserve(4 + (N * outBpe + 7) / 8);
    {
        BitWriter w(raw, BitOrder::LSB);
        w.write(static_cast<uint32_t>(fwdPerm[0]), outFirstBpe);
        for (size_t i = 1; i < N; ++i) {
            const int64_t d = static_cast<int64_t>(fwdPerm[i]) - static_cast<int64_t>(fwdPerm[i - 1]);
            const uint32_t zz = static_cast<uint32_t>((d << 1) ^ (d >> 63));
            w.write(zz, outBpe);
        }
        w.flush();
    }
    return raw;
}

// Decode zigzag-delta stream into a vector<size_t>.
inline std::vector<size_t> unpackDeltaStream(const uint8_t* ptr, size_t N,
                                              uint8_t bpe, uint8_t firstBpe) {
    std::vector<size_t> result;
    result.reserve(N);
    BitReader r(ptr, /* large enough */ 0x7fffffff, BitOrder::LSB);
    size_t prev = r.read(firstBpe);
    result.push_back(prev);
    for (size_t i = 1; i < N; ++i) {
        const uint32_t zz = r.read(bpe);
        const int64_t d = static_cast<int64_t>(zz >> 1) ^ -static_cast<int64_t>(zz & 1);
        prev = static_cast<size_t>(static_cast<int64_t>(prev) + d);
        result.push_back(prev);
    }
    return result;
}

#ifdef PERM_STORE_ZSTD
inline std::vector<uint8_t> zstdCompress(std::span<const uint8_t> src) {
    const size_t bound = ZSTD_compressBound(src.size());
    std::vector<uint8_t> dst(bound);
    const size_t sz = ZSTD_compress(dst.data(), bound, src.data(), src.size(), /*level=*/3);
    if (ZSTD_isError(sz)) throw std::runtime_error("PermutationStore: Zstd compress failed");
    dst.resize(sz);
    return dst;
}
inline std::vector<uint8_t> zstdDecompress(const uint8_t* src, size_t srcSize,
                                             size_t decompressedSize) {
    std::vector<uint8_t> dst(decompressedSize);
    const size_t sz = ZSTD_decompress(dst.data(), decompressedSize, src, srcSize);
    if (ZSTD_isError(sz)) throw std::runtime_error("PermutationStore: Zstd decompress failed");
    return dst;
}
#endif

#ifdef PERM_STORE_LZ4
inline std::vector<uint8_t> lz4Compress(std::span<const uint8_t> src) {
    const int bound = LZ4_compressBound(static_cast<int>(src.size()));
    if (bound <= 0) throw std::runtime_error("PermutationStore: LZ4 input too large");
    std::vector<uint8_t> dst(static_cast<size_t>(bound));
    const int sz = LZ4_compress_default(
        reinterpret_cast<const char*>(src.data()), reinterpret_cast<char*>(dst.data()),
        static_cast<int>(src.size()), bound);
    if (sz <= 0) throw std::runtime_error("PermutationStore: LZ4 compress failed");
    dst.resize(static_cast<size_t>(sz));
    return dst;
}
inline std::vector<uint8_t> lz4Decompress(const uint8_t* src, size_t srcSize,
                                            size_t decompressedSize) {
    std::vector<uint8_t> dst(decompressedSize);
    const int sz = LZ4_decompress_safe(
        reinterpret_cast<const char*>(src), reinterpret_cast<char*>(dst.data()),
        static_cast<int>(srcSize), static_cast<int>(decompressedSize));
    if (sz < 0) throw std::runtime_error("PermutationStore: LZ4 decompress failed");
    return dst;
}
#endif

} // namespace detail_ps

// ---------------------------------------------------------------------------
// PermutationStore
// ---------------------------------------------------------------------------

class PermutationStore {
public:
    // -----------------------------------------------------------------------
    // Runtime query
    // -----------------------------------------------------------------------

    static bool supportsRandomAccess(std::span<const uint8_t> blob) noexcept {
        if (blob.empty()) return true;
        switch (static_cast<PermFormat>(blob[0])) {
            case PermFormat::None:          return true;
            case PermFormat::FlatBitPacked: return true;
            case PermFormat::ChunkRelative: return true;
            default:                        return false;
        }
    }

    // -----------------------------------------------------------------------
    // Pack — general dispatchers
    // -----------------------------------------------------------------------

    // Standard pack (formats that don't need group information).
    static std::vector<uint8_t> pack(std::span<const size_t> fwdPerm,
                                     PermFormat fmt       = PermFormat::FlatBitPacked,
                                     size_t     chunkSize = 256) {
        if (fwdPerm.empty()) return {static_cast<uint8_t>(PermFormat::None)};
        switch (fmt) {
            case PermFormat::None:              return {static_cast<uint8_t>(PermFormat::None)};
            case PermFormat::FlatBitPacked:     return packFlat(fwdPerm);
            case PermFormat::DeltaBitPacked:    return packDelta(fwdPerm);
            case PermFormat::ChunkRelative:     return packChunkRelative(fwdPerm, chunkSize);
            case PermFormat::DeltaZstd:         return packDeltaCompressed(fwdPerm, PermFormat::DeltaZstd);
            case PermFormat::DeltaLZ4:          return packDeltaCompressed(fwdPerm, PermFormat::DeltaLZ4);
            case PermFormat::ChunkRelativeZstd: return packChunkCompressed(fwdPerm, chunkSize, PermFormat::ChunkRelativeZstd);
            case PermFormat::ChunkRelativeLZ4:  return packChunkCompressed(fwdPerm, chunkSize, PermFormat::ChunkRelativeLZ4);
            // ValueGrouped / InverseEliasFano require group sizes — use packWithGroups
            default: return packFlat(fwdPerm);
        }
    }

    // Pack with group sizes (for ValueGrouped and InverseEliasFano).
    // groupSizes[k] = number of elements in sorted value-group k.
    // The sum of groupSizes must equal fwdPerm.size().
    static std::vector<uint8_t> packWithGroups(std::span<const size_t> fwdPerm,
                                               std::span<const size_t> groupSizes,
                                               PermFormat fmt) {
        if (fwdPerm.empty()) return {static_cast<uint8_t>(PermFormat::None)};
        if (groupSizes.empty()) return pack(fwdPerm, PermFormat::FlatBitPacked);
        switch (fmt) {
            case PermFormat::ValueGrouped:     return packValueGrouped(fwdPerm, groupSizes);
            case PermFormat::InverseEliasFano: return packInverseEliasFano(fwdPerm, groupSizes);
            default:                           return pack(fwdPerm, fmt);
        }
    }

    // -----------------------------------------------------------------------
    // Individual pack implementations
    // -----------------------------------------------------------------------

    static std::vector<uint8_t> packFlat(std::span<const size_t> fwdPerm) {
        const size_t N = fwdPerm.size();
        if (N == 0) return {static_cast<uint8_t>(PermFormat::None)};
        const uint8_t bpe = detail_ps::bitsNeeded(N > 1 ? N - 1 : 1);
        std::vector<uint8_t> blob;
        blob.reserve(10 + (N * bpe + 7) / 8);
        blob.push_back(static_cast<uint8_t>(PermFormat::FlatBitPacked));
        blob.push_back(bpe);
        detail_ps::writeU64(blob, static_cast<uint64_t>(N));
        { BitWriter w(blob, BitOrder::LSB); for (size_t v : fwdPerm) w.write(static_cast<uint32_t>(v), bpe); w.flush(); }
        return blob;
    }

    static std::vector<uint8_t> packDelta(std::span<const size_t> fwdPerm) {
        const size_t N = fwdPerm.size();
        if (N == 0) return {static_cast<uint8_t>(PermFormat::None)};
        uint8_t bpe = 0, firstBpe = 0;
        auto raw = detail_ps::packDeltaStream(fwdPerm, bpe, firstBpe);
        std::vector<uint8_t> blob;
        blob.reserve(11 + raw.size());
        blob.push_back(static_cast<uint8_t>(PermFormat::DeltaBitPacked));
        blob.push_back(bpe); blob.push_back(firstBpe);
        detail_ps::writeU64(blob, static_cast<uint64_t>(N));
        blob.insert(blob.end(), raw.begin(), raw.end());
        return blob;
    }

    static std::vector<uint8_t> packChunkRelative(std::span<const size_t> fwdPerm, size_t chunkSize) {
        const size_t N = fwdPerm.size();
        if (N == 0) return {static_cast<uint8_t>(PermFormat::None)};
        chunkSize = std::clamp(chunkSize, size_t{2}, N);
        const uint8_t rankBits = detail_ps::bitsNeeded(chunkSize - 1);
        std::vector<uint8_t> blob;
        blob.reserve(14 + (N * rankBits + 7) / 8);
        blob.push_back(static_cast<uint8_t>(PermFormat::ChunkRelative));
        detail_ps::writeU32(blob, static_cast<uint32_t>(chunkSize));
        blob.push_back(rankBits);
        detail_ps::writeU64(blob, static_cast<uint64_t>(N));
        { BitWriter w(blob, BitOrder::LSB);
          for (size_t i = 0; i < N; ++i) w.write(static_cast<uint32_t>(fwdPerm[i] - (i/chunkSize)*chunkSize), rankBits);
          w.flush(); }
        return blob;
    }

    static std::vector<uint8_t> packDeltaCompressed(std::span<const size_t> fwdPerm, PermFormat fmt) {
        const size_t N = fwdPerm.size();
        if (N == 0) return {static_cast<uint8_t>(PermFormat::None)};
        uint8_t bpe = 0, firstBpe = 0;
        auto raw = detail_ps::packDeltaStream(fwdPerm, bpe, firstBpe);

        std::vector<uint8_t> compressed;
        if (fmt == PermFormat::DeltaZstd) {
#ifdef PERM_STORE_ZSTD
            compressed = detail_ps::zstdCompress(raw);
#else
            throw std::runtime_error("PermutationStore: Zstd not available (PERM_STORE_ZSTD undefined)");
#endif
        } else {
#ifdef PERM_STORE_LZ4
            compressed = detail_ps::lz4Compress(raw);
#else
            throw std::runtime_error("PermutationStore: LZ4 not available (PERM_STORE_LZ4 undefined)");
#endif
        }

        std::vector<uint8_t> blob;
        blob.reserve(15 + compressed.size());
        blob.push_back(static_cast<uint8_t>(fmt));
        blob.push_back(bpe); blob.push_back(firstBpe);
        detail_ps::writeU64(blob, static_cast<uint64_t>(N));
        detail_ps::writeU32(blob, static_cast<uint32_t>(raw.size()));          // decompressed size
        detail_ps::writeU32(blob, static_cast<uint32_t>(compressed.size()));   // compressed size
        blob.insert(blob.end(), compressed.begin(), compressed.end());
        return blob;
    }

    static std::vector<uint8_t> packChunkCompressed(std::span<const size_t> fwdPerm,
                                                     size_t chunkSize, PermFormat fmt) {
        const size_t N = fwdPerm.size();
        if (N == 0) return {static_cast<uint8_t>(PermFormat::None)};
        chunkSize = std::clamp(chunkSize, size_t{2}, N);
        const uint8_t rankBits = detail_ps::bitsNeeded(chunkSize - 1);

        // Build raw bit-packed rank bytes (identical to ChunkRelative payload)
        std::vector<uint8_t> raw;
        raw.reserve((N * rankBits + 7) / 8);
        { BitWriter w(raw, BitOrder::LSB);
          for (size_t i = 0; i < N; ++i) w.write(static_cast<uint32_t>(fwdPerm[i] - (i/chunkSize)*chunkSize), rankBits);
          w.flush(); }

        std::vector<uint8_t> compressed;
        if (fmt == PermFormat::ChunkRelativeZstd) {
#ifdef PERM_STORE_ZSTD
            compressed = detail_ps::zstdCompress(raw);
#else
            throw std::runtime_error("PermutationStore: Zstd not available");
#endif
        } else {
#ifdef PERM_STORE_LZ4
            compressed = detail_ps::lz4Compress(raw);
#else
            throw std::runtime_error("PermutationStore: LZ4 not available");
#endif
        }

        std::vector<uint8_t> blob;
        blob.reserve(18 + compressed.size());
        blob.push_back(static_cast<uint8_t>(fmt));
        detail_ps::writeU32(blob, static_cast<uint32_t>(chunkSize));
        blob.push_back(rankBits);
        detail_ps::writeU64(blob, static_cast<uint64_t>(N));
        detail_ps::writeU32(blob, static_cast<uint32_t>(raw.size()));          // decompressed bytes
        detail_ps::writeU32(blob, static_cast<uint32_t>(compressed.size()));   // compressed bytes
        blob.insert(blob.end(), compressed.begin(), compressed.end());
        return blob;
    }

    // packValueGrouped: groups contain the original positions of equal-valued sorted elements.
    // groupSizes[k] = number of elements in value group k (in sorted order).
    static std::vector<uint8_t> packValueGrouped(std::span<const size_t> fwdPerm,
                                                  std::span<const size_t> groupSizes) {
        const size_t N = fwdPerm.size();
        const size_t K = groupSizes.size();

        // Build inverse permutation: inv[sortedPos] = originalPos
        std::vector<size_t> inv(N);
        for (size_t i = 0; i < N; ++i) inv[fwdPerm[i]] = i;

        const uint8_t bpeFirst = detail_ps::bitsNeeded(N > 1 ? N - 1 : 1);

        std::vector<uint8_t> blob;
        blob.reserve(21 + N * 4);
        blob.push_back(static_cast<uint8_t>(PermFormat::ValueGrouped));
        detail_ps::writeU64(blob, static_cast<uint64_t>(N));
        detail_ps::writeU64(blob, static_cast<uint64_t>(K));

        // Placeholder for total group-data byte count (filled in at end)
        const size_t groupDataSizeOffset = blob.size();
        detail_ps::writeU32(blob, 0);

        size_t sortedPos = 0;
        for (size_t k = 0; k < K; ++k) {
            const size_t m = groupSizes[k];
            detail_ps::writeU32(blob, static_cast<uint32_t>(m));
            if (m == 0) continue;

            // First original position
            const size_t first = inv[sortedPos];
            // Compute max delta for this group
            size_t maxDelta = 0;
            for (size_t r = 1; r < m; ++r) {
                const size_t delta = inv[sortedPos + r] - inv[sortedPos + r - 1];
                if (delta > maxDelta) maxDelta = delta;
            }
            const uint8_t deltaBpe = detail_ps::bitsNeeded(maxDelta > 0 ? maxDelta : 1);

            blob.push_back(bpeFirst);
            blob.push_back(deltaBpe);
            {
                BitWriter w(blob, BitOrder::LSB);
                w.write(static_cast<uint32_t>(first), bpeFirst);
                for (size_t r = 1; r < m; ++r) {
                    const size_t delta = inv[sortedPos + r] - inv[sortedPos + r - 1];
                    w.write(static_cast<uint32_t>(delta), deltaBpe);
                }
                w.flush();
            }
            sortedPos += m;
        }

        // Fill in group-data byte count
        const uint32_t groupDataBytes = static_cast<uint32_t>(blob.size() - groupDataSizeOffset - 4);
        for (int b = 0; b < 4; ++b)
            blob[groupDataSizeOffset + b] = static_cast<uint8_t>(groupDataBytes >> (8 * b));

        return blob;
    }

    // packInverseEliasFano: Elias-Fano encoding per value group.
    static std::vector<uint8_t> packInverseEliasFano(std::span<const size_t> fwdPerm,
                                                      std::span<const size_t> groupSizes) {
        const size_t N = fwdPerm.size();
        const size_t K = groupSizes.size();

        // Build inverse permutation
        std::vector<size_t> inv(N);
        for (size_t i = 0; i < N; ++i) inv[fwdPerm[i]] = i;

        std::vector<uint8_t> blob;
        blob.reserve(17 + N * 3);
        blob.push_back(static_cast<uint8_t>(PermFormat::InverseEliasFano));
        detail_ps::writeU64(blob, static_cast<uint64_t>(N));
        detail_ps::writeU64(blob, static_cast<uint64_t>(K));

        // Write K group sizes for prefix-sum skipping
        for (size_t k = 0; k < K; ++k)
            detail_ps::writeU32(blob, static_cast<uint32_t>(groupSizes[k]));

        // Encode each group with Elias-Fano
        size_t sortedPos = 0;
        for (size_t k = 0; k < K; ++k) {
            const size_t m = groupSizes[k];
            if (m == 0) { sortedPos += m; continue; }

            // l = floor(log2(N/m)); ensure l >= 0
            const uint32_t l = (m < N) ? static_cast<uint32_t>(std::bit_width(N / m)) - 1u : 0u;
            const size_t highLen = (N >> l) + m + 1; // upper bound on high bitmap length

            blob.push_back(static_cast<uint8_t>(l));

            // Low bits: m values at l bits each
            if (l > 0) {
                BitWriter wl(blob, BitOrder::LSB);
                for (size_t r = 0; r < m; ++r)
                    wl.write(static_cast<uint32_t>(inv[sortedPos + r] & ((size_t{1} << l) - 1)), l);
                wl.flush();
            }

            // High bits: unary-coded quotients as differences
            // highBitmap[q[i] + i] = 1, all others = 0
            // Encode as a sequence of "run of zeros then a one" per element
            std::vector<uint8_t> highBuf;
            highBuf.reserve((highLen + 7) / 8);
            {
                BitWriter wh(highBuf, BitOrder::LSB);
                size_t prevQ = 0;
                for (size_t r = 0; r < m; ++r) {
                    const size_t q = inv[sortedPos + r] >> l;
                    // Write (q - prevQ) zeros then one one
                    size_t numZeros = q - prevQ;
                    while (numZeros > 16) { wh.write(0, 16); numZeros -= 16; }
                    if (numZeros > 0) wh.write(0, static_cast<uint32_t>(numZeros));
                    wh.write(1, 1);
                    prevQ = q;
                }
                wh.flush();
            }
            detail_ps::writeU32(blob, static_cast<uint32_t>(highBuf.size()));
            blob.insert(blob.end(), highBuf.begin(), highBuf.end());

            sortedPos += m;
        }

        return blob;
    }

    // -----------------------------------------------------------------------
    // Unpack all values → forward permutation
    // -----------------------------------------------------------------------

    static std::vector<size_t> unpackForward(std::span<const uint8_t> blob) {
        if (blob.empty()) return {};
        switch (static_cast<PermFormat>(blob[0])) {

        case PermFormat::None:
            return {};

        case PermFormat::FlatBitPacked: {
            const uint8_t bpe = blob[1];
            const size_t N = static_cast<size_t>(detail_ps::readU64(blob.data() + 2));
            std::vector<size_t> r(N);
            BitReader rd(blob.data() + 10, blob.size() - 10, BitOrder::LSB);
            for (size_t i = 0; i < N; ++i) r[i] = rd.read(bpe);
            return r;
        }

        case PermFormat::DeltaBitPacked: {
            const uint8_t bpe = blob[1], firstBpe = blob[2];
            const size_t N = static_cast<size_t>(detail_ps::readU64(blob.data() + 3));
            return detail_ps::unpackDeltaStream(blob.data() + 11, N, bpe, firstBpe);
        }

        case PermFormat::ChunkRelative: {
            const uint32_t cs = detail_ps::readU32(blob.data() + 1);
            const uint8_t rankBits = blob[5];
            const size_t N = static_cast<size_t>(detail_ps::readU64(blob.data() + 6));
            std::vector<size_t> r(N);
            BitReader rd(blob.data() + 14, blob.size() - 14, BitOrder::LSB);
            for (size_t i = 0; i < N; ++i)
                r[i] = (i / cs) * cs + rd.read(rankBits);
            return r;
        }

        case PermFormat::DeltaZstd:
        case PermFormat::DeltaLZ4: {
            const uint8_t bpe = blob[1], firstBpe = blob[2];
            const size_t N = static_cast<size_t>(detail_ps::readU64(blob.data() + 3));
            const uint32_t decompSize = detail_ps::readU32(blob.data() + 11);
            const uint32_t compSize   = detail_ps::readU32(blob.data() + 15);
            const uint8_t* compData   = blob.data() + 19;
            std::vector<uint8_t> raw;
            if (static_cast<PermFormat>(blob[0]) == PermFormat::DeltaZstd) {
#ifdef PERM_STORE_ZSTD
                raw = detail_ps::zstdDecompress(compData, compSize, decompSize);
#else
                throw std::runtime_error("PermutationStore: Zstd not available");
#endif
            } else {
#ifdef PERM_STORE_LZ4
                raw = detail_ps::lz4Decompress(compData, compSize, decompSize);
#else
                throw std::runtime_error("PermutationStore: LZ4 not available");
#endif
            }
            return detail_ps::unpackDeltaStream(raw.data(), N, bpe, firstBpe);
        }

        case PermFormat::ChunkRelativeZstd:
        case PermFormat::ChunkRelativeLZ4: {
            const uint32_t cs = detail_ps::readU32(blob.data() + 1);
            const uint8_t rankBits = blob[5];
            const size_t N = static_cast<size_t>(detail_ps::readU64(blob.data() + 6));
            const uint32_t decompSize = detail_ps::readU32(blob.data() + 14);
            const uint32_t compSize   = detail_ps::readU32(blob.data() + 18);
            const uint8_t* compData   = blob.data() + 22;
            std::vector<uint8_t> raw;
            if (static_cast<PermFormat>(blob[0]) == PermFormat::ChunkRelativeZstd) {
#ifdef PERM_STORE_ZSTD
                raw = detail_ps::zstdDecompress(compData, compSize, decompSize);
#else
                throw std::runtime_error("PermutationStore: Zstd not available");
#endif
            } else {
#ifdef PERM_STORE_LZ4
                raw = detail_ps::lz4Decompress(compData, compSize, decompSize);
#else
                throw std::runtime_error("PermutationStore: LZ4 not available");
#endif
            }
            std::vector<size_t> r(N);
            BitReader rd(raw.data(), raw.size(), BitOrder::LSB);
            for (size_t i = 0; i < N; ++i)
                r[i] = (i / cs) * cs + rd.read(rankBits);
            return r;
        }

        case PermFormat::ValueGrouped: {
            const size_t N = static_cast<size_t>(detail_ps::readU64(blob.data() + 1));
            const size_t K = static_cast<size_t>(detail_ps::readU64(blob.data() + 9));
            // Skip groupDataTotalBytes (4 bytes)
            const uint8_t* p = blob.data() + 21;  // points to first group's data

            std::vector<size_t> fwd(N);
            size_t sortedPos = 0;
            for (size_t k = 0; k < K; ++k) {
                const uint32_t m = detail_ps::readU32(p); p += 4;
                if (m == 0) continue;
                const uint8_t bpeFirst = *p++;
                const uint8_t deltaBpe = *p++;
                BitReader rd(p, blob.size(), BitOrder::LSB);
                size_t cur = rd.read(bpeFirst);
                fwd[cur] = sortedPos;
                for (uint32_t r = 1; r < m; ++r) {
                    const size_t delta = rd.read(deltaBpe);
                    cur += delta;
                    fwd[cur] = sortedPos + r;
                }
                // Advance p past the bits we consumed
                const size_t bitsConsumed = bpeFirst + (m - 1) * static_cast<size_t>(deltaBpe);
                p += (bitsConsumed + 7) / 8;
                sortedPos += m;
            }
            return fwd;
        }

        case PermFormat::InverseEliasFano: {
            const size_t N = static_cast<size_t>(detail_ps::readU64(blob.data() + 1));
            const size_t K = static_cast<size_t>(detail_ps::readU64(blob.data() + 9));
            const uint8_t* p = blob.data() + 17 + K * 4;  // skip past group-size table

            std::vector<size_t> fwd(N);
            size_t sortedPos = 0;

            // Reconstruct group sizes from the stored table
            const uint8_t* gsTable = blob.data() + 17;
            for (size_t k = 0; k < K; ++k) {
                const uint32_t m = detail_ps::readU32(gsTable + k * 4);
                if (m == 0) continue;
                const uint32_t l = *p++;

                // Decode low bits (m * l bits)
                std::vector<size_t> lows(m, 0);
                if (l > 0) {
                    BitReader rdl(p, blob.size(), BitOrder::LSB);
                    for (uint32_t r = 0; r < m; ++r) lows[r] = rdl.read(l);
                    p += (static_cast<size_t>(m) * l + 7) / 8;
                }

                // Decode high bits (unary bitmap)
                const uint32_t highBufBytes = detail_ps::readU32(p); p += 4;
                // Scan bit-by-bit; each '1' corresponds to one element
                std::vector<size_t> highs;
                highs.reserve(m);
                {
                    size_t bitPos = 0;
                    size_t q = 0;  // current quotient (number of leading zeros before this 1)
                    while (highs.size() < m) {
                        const size_t byteIdx = bitPos / 8;
                        const uint32_t bitOff = static_cast<uint32_t>(bitPos % 8);
                        const uint8_t bit = (p[byteIdx] >> bitOff) & 1u;
                        if (bit == 1) {
                            highs.push_back(q);
                        } else {
                            ++q;
                        }
                        ++bitPos;
                    }
                }
                p += highBufBytes;

                // Reconstruct positions and forward permutation
                for (uint32_t r = 0; r < m; ++r) {
                    const size_t origPos = (highs[r] << l) | lows[r];
                    fwd[origPos] = sortedPos + r;
                }
                sortedPos += m;
            }
            return fwd;
        }

        }
        return {};
    }

    // -----------------------------------------------------------------------
    // Single-element O(1) lookup (only for formats that support it)
    // -----------------------------------------------------------------------

    static size_t forwardAt(std::span<const uint8_t> blob, size_t origIdx) {
        if (blob.empty()) return origIdx;
        switch (static_cast<PermFormat>(blob[0])) {
        case PermFormat::None:
            return origIdx;
        case PermFormat::FlatBitPacked: {
            const uint8_t bpe = blob[1];
            return detail_ps::readBitsAt(blob.data() + 10, origIdx * bpe, bpe);
        }
        case PermFormat::ChunkRelative: {
            const uint32_t cs = detail_ps::readU32(blob.data() + 1);
            const uint8_t rankBits = blob[5];
            const size_t base = (origIdx / cs) * cs;
            return base + detail_ps::readBitsAt(blob.data() + 14, origIdx * rankBits, rankBits);
        }
        default:
            // Sequential-access formats: fall back to full decode
            return unpackForward(blob)[origIdx];
        }
    }

    // -----------------------------------------------------------------------
    // Bulk lookup
    // -----------------------------------------------------------------------

    static std::vector<size_t> forwardBulk(std::span<const uint8_t> blob,
                                           std::span<const size_t>   origIndices) {
        if (origIndices.empty()) return {};
        const size_t M = origIndices.size();
        std::vector<size_t> result(M);

        if (supportsRandomAccess(blob)) {
            for (size_t i = 0; i < M; ++i) result[i] = forwardAt(blob, origIndices[i]);
            return result;
        }

        // All sequential/group formats: full decode then index
        auto allFwd = unpackForward(blob);
        for (size_t i = 0; i < M; ++i) result[i] = allFwd[origIndices[i]];
        return result;
    }

    // -----------------------------------------------------------------------
    // Size estimation (bytes)
    // -----------------------------------------------------------------------

    static size_t estimatePackedSize(size_t N, PermFormat fmt, size_t chunkSize = 256,
                                     size_t K = 0) noexcept {
        switch (fmt) {
        case PermFormat::None:              return 1;
        case PermFormat::FlatBitPacked: {
            const uint8_t bpe = detail_ps::bitsNeeded(N > 1 ? N - 1 : 1);
            return 10 + (N * bpe + 7) / 8;
        }
        case PermFormat::DeltaBitPacked: {
            const size_t avgDelta = 2 * static_cast<size_t>(std::sqrt(static_cast<double>(N)));
            const uint8_t bpe = detail_ps::bitsNeeded(avgDelta > 0 ? avgDelta : 1);
            return 11 + (N * bpe + 7) / 8;
        }
        case PermFormat::ChunkRelative: {
            chunkSize = std::max(chunkSize, size_t{2});
            const uint8_t rankBits = detail_ps::bitsNeeded(chunkSize - 1);
            return 14 + (N * rankBits + 7) / 8;
        }
        case PermFormat::DeltaZstd:
        case PermFormat::DeltaLZ4:
            // Rough: 30–70% of DeltaBitPacked for structured data
            return estimatePackedSize(N, PermFormat::DeltaBitPacked) / 2;
        case PermFormat::ValueGrouped:
        case PermFormat::InverseEliasFano: {
            // ≈ N × log2(N/K) bits for uniform K-cardinality
            if (K == 0) K = std::max(size_t{1}, N / 100);  // rough default
            const auto bitsPerElem = static_cast<size_t>(std::log2(static_cast<double>(N) / K) + 1);
            return (N * bitsPerElem + 7) / 8;
        }
        case PermFormat::ChunkRelativeZstd:
        case PermFormat::ChunkRelativeLZ4:
            // Rough: 20–80% of ChunkRelative for structured data
            return estimatePackedSize(N, PermFormat::ChunkRelative, chunkSize) / 3;
        }
        return N * 8;
    }

    // Convenience: human-readable format name
    static const char* formatName(PermFormat fmt) noexcept {
        switch (fmt) {
        case PermFormat::None:              return "None";
        case PermFormat::FlatBitPacked:     return "FlatBitPacked";
        case PermFormat::DeltaBitPacked:    return "DeltaBitPacked";
        case PermFormat::ChunkRelative:     return "ChunkRelative";
        case PermFormat::DeltaZstd:         return "DeltaZstd";
        case PermFormat::DeltaLZ4:          return "DeltaLZ4";
        case PermFormat::ValueGrouped:      return "ValueGrouped";
        case PermFormat::InverseEliasFano:  return "InverseEliasFano";
        case PermFormat::ChunkRelativeZstd: return "ChunkRelativeZstd";
        case PermFormat::ChunkRelativeLZ4:  return "ChunkRelativeLZ4";
        }
        return "Unknown";
    }
};

} // namespace encodings::reorderers
