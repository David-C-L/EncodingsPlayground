#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/BitPacker.hpp"

namespace encodings::reorderers {

using encodings::core::BitOrder;
using encodings::core::BitReader;
using encodings::core::BitWriter;

// ---------------------------------------------------------------------------
// Format tags
// ---------------------------------------------------------------------------

enum class PermFormat : uint8_t {
    None           = 0,  // No permutation stored; positions unchanged (e.g. GrayCode)
    FlatBitPacked  = 1,  // N × bpe bits; O(1) random access via bit-seek
    DeltaBitPacked = 2,  // Delta + zigzag encoded then bit-packed; sequential access
    ChunkRelative  = 3,  // Intra-chunk ranks with implicit base; O(1) random access
                         // Requires fwd[i] ∈ [chunk(i)*C, (chunk(i)+1)*C) — use for WindowedSort
};

// Compile-time random access capability per format.
template <PermFormat F> inline constexpr bool permFormatSupportsRandomAccess = false;
template <> inline constexpr bool permFormatSupportsRandomAccess<PermFormat::None>          = true;
template <> inline constexpr bool permFormatSupportsRandomAccess<PermFormat::FlatBitPacked> = true;
template <> inline constexpr bool permFormatSupportsRandomAccess<PermFormat::ChunkRelative> = true;
// DeltaBitPacked stays false

// ---------------------------------------------------------------------------
// Wire layouts
//
//  None:           [tag:1]
//  FlatBitPacked:  [tag:1][bpe:1][N:8][packed values: ceil(N*bpe/8) bytes]
//  DeltaBitPacked: [tag:1][bpe:1][firstBpe:1][N:8][first:firstBpe bits][N-1 zigzag deltas: bpe bits each]
//  ChunkRelative:  [tag:1][chunkSize:4][rankBits:1][N:8][intra-chunk ranks: ceil(N*rankBits/8) bytes]
//                  base for chunk k is implicit: k * chunkSize
// ---------------------------------------------------------------------------

namespace detail {

// Write uint64_t N as 8 little-endian bytes into vec.
inline void writeN(std::vector<uint8_t>& vec, uint64_t n) {
    for (int b = 0; b < 8; ++b)
        vec.push_back(static_cast<uint8_t>(n >> (8 * b)));
}

inline uint64_t readN(const uint8_t* p) {
    uint64_t n = 0;
    for (int b = 0; b < 8; ++b)
        n |= static_cast<uint64_t>(p[b]) << (8 * b);
    return n;
}

// Read numBits bits starting at bit position bitPos from data (LSB bit order).
// Works for numBits up to 32.
inline size_t readBitsAt(const uint8_t* data, size_t bitPos, uint32_t numBits) noexcept {
    if (numBits == 0) return 0;
    const size_t byteStart = bitPos >> 3;
    const uint32_t bitOff  = static_cast<uint32_t>(bitPos & 7);
    // Collect up to 5 bytes (covers 40 bits, enough for 32-bit values with any bitOff).
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

} // namespace detail

// ---------------------------------------------------------------------------
// PermutationStore — static methods only
// ---------------------------------------------------------------------------

class PermutationStore {
public:
    // -----------------------------------------------------------------------
    // Runtime query
    // -----------------------------------------------------------------------

    static bool supportsRandomAccess(std::span<const uint8_t> blob) noexcept {
        if (blob.empty()) return true;
        switch (static_cast<PermFormat>(blob[0])) {
            case PermFormat::None:           return true;
            case PermFormat::FlatBitPacked:  return true;
            case PermFormat::ChunkRelative:  return true;
            case PermFormat::DeltaBitPacked: return false;
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // Pack — general dispatcher
    // -----------------------------------------------------------------------

    static std::vector<uint8_t> pack(std::span<const size_t> fwdPerm,
                                     PermFormat fmt       = PermFormat::FlatBitPacked,
                                     size_t     chunkSize = 256) {
        if (fwdPerm.empty()) return {static_cast<uint8_t>(PermFormat::None)};
        switch (fmt) {
            case PermFormat::None:           return {static_cast<uint8_t>(PermFormat::None)};
            case PermFormat::FlatBitPacked:  return packFlat(fwdPerm);
            case PermFormat::DeltaBitPacked: return packDelta(fwdPerm);
            case PermFormat::ChunkRelative:  return packChunkRelative(fwdPerm, chunkSize);
        }
        return packFlat(fwdPerm);
    }

    // -----------------------------------------------------------------------
    // FlatBitPacked
    // -----------------------------------------------------------------------

    static std::vector<uint8_t> packFlat(std::span<const size_t> fwdPerm) {
        const size_t N = fwdPerm.size();
        if (N == 0) return {static_cast<uint8_t>(PermFormat::None)};
        const uint8_t bpe = detail::bitsNeeded(N > 1 ? N - 1 : 1);

        std::vector<uint8_t> blob;
        blob.reserve(10 + (N * bpe + 7) / 8);
        blob.push_back(static_cast<uint8_t>(PermFormat::FlatBitPacked));
        blob.push_back(bpe);
        detail::writeN(blob, static_cast<uint64_t>(N));
        {
            BitWriter w(blob, BitOrder::LSB);
            for (size_t v : fwdPerm)
                w.write(static_cast<uint32_t>(v), bpe);
            w.flush();
        }
        return blob;
    }

    // -----------------------------------------------------------------------
    // DeltaBitPacked (zigzag delta encoding for compressible sort permutations)
    // -----------------------------------------------------------------------

    static std::vector<uint8_t> packDelta(std::span<const size_t> fwdPerm) {
        const size_t N = fwdPerm.size();
        if (N == 0) return {static_cast<uint8_t>(PermFormat::None)};

        const uint8_t firstBpe = detail::bitsNeeded(N > 1 ? N - 1 : 1);

        // Find max zigzag delta value to determine bpe for deltas
        size_t maxZigzag = 0;
        for (size_t i = 1; i < N; ++i) {
            const int64_t d = static_cast<int64_t>(fwdPerm[i]) - static_cast<int64_t>(fwdPerm[i - 1]);
            const uint64_t zz = static_cast<uint64_t>((d << 1) ^ (d >> 63));
            if (zz > maxZigzag) maxZigzag = zz;
        }
        const uint8_t bpe = detail::bitsNeeded(maxZigzag > 0 ? maxZigzag : 1);

        std::vector<uint8_t> blob;
        blob.reserve(11 + (N * bpe + 7) / 8);
        blob.push_back(static_cast<uint8_t>(PermFormat::DeltaBitPacked));
        blob.push_back(bpe);
        blob.push_back(firstBpe);
        detail::writeN(blob, static_cast<uint64_t>(N));
        {
            BitWriter w(blob, BitOrder::LSB);
            w.write(static_cast<uint32_t>(fwdPerm[0]), firstBpe);
            for (size_t i = 1; i < N; ++i) {
                const int64_t d = static_cast<int64_t>(fwdPerm[i]) - static_cast<int64_t>(fwdPerm[i - 1]);
                const uint32_t zz = static_cast<uint32_t>((d << 1) ^ (d >> 63));
                w.write(zz, bpe);
            }
            w.flush();
        }
        return blob;
    }

    // -----------------------------------------------------------------------
    // ChunkRelative
    // Requires: for each element i in chunk k, fwdPerm[i] ∈ [k*C, (k+1)*C).
    // (This is guaranteed by WindowedSortReorderer but NOT by SortReorderer.)
    // -----------------------------------------------------------------------

    static std::vector<uint8_t> packChunkRelative(std::span<const size_t> fwdPerm, size_t chunkSize) {
        const size_t N = fwdPerm.size();
        if (N == 0) return {static_cast<uint8_t>(PermFormat::None)};
        chunkSize = std::clamp(chunkSize, size_t{2}, N);

        const uint8_t rankBits = detail::bitsNeeded(chunkSize - 1);

        std::vector<uint8_t> blob;
        blob.reserve(14 + (N * rankBits + 7) / 8);
        blob.push_back(static_cast<uint8_t>(PermFormat::ChunkRelative));
        // chunkSize as 4-byte LE
        const uint32_t cs32 = static_cast<uint32_t>(chunkSize);
        blob.push_back(cs32 & 0xFF);
        blob.push_back((cs32 >> 8) & 0xFF);
        blob.push_back((cs32 >> 16) & 0xFF);
        blob.push_back((cs32 >> 24) & 0xFF);
        blob.push_back(rankBits);
        detail::writeN(blob, static_cast<uint64_t>(N));
        {
            BitWriter w(blob, BitOrder::LSB);
            for (size_t i = 0; i < N; ++i) {
                const size_t base = (i / chunkSize) * chunkSize;
                const size_t rank = fwdPerm[i] - base;
                w.write(static_cast<uint32_t>(rank), rankBits);
            }
            w.flush();
        }
        return blob;
    }

    // -----------------------------------------------------------------------
    // Unpack all values
    // -----------------------------------------------------------------------

    static std::vector<size_t> unpackForward(std::span<const uint8_t> blob) {
        if (blob.empty()) return {};
        switch (static_cast<PermFormat>(blob[0])) {
            case PermFormat::None:
                return {};

            case PermFormat::FlatBitPacked: {
                const uint8_t bpe = blob[1];
                const size_t N    = static_cast<size_t>(detail::readN(blob.data() + 2));
                std::vector<size_t> result(N);
                BitReader r(blob.data() + 10, blob.size() - 10, BitOrder::LSB);
                for (size_t i = 0; i < N; ++i)
                    result[i] = r.read(bpe);
                return result;
            }

            case PermFormat::DeltaBitPacked: {
                const uint8_t bpe      = blob[1];
                const uint8_t firstBpe = blob[2];
                const size_t N         = static_cast<size_t>(detail::readN(blob.data() + 3));
                std::vector<size_t> result;
                result.reserve(N);
                BitReader r(blob.data() + 11, blob.size() - 11, BitOrder::LSB);
                size_t prev = r.read(firstBpe);
                result.push_back(prev);
                for (size_t i = 1; i < N; ++i) {
                    const uint32_t zz  = r.read(bpe);
                    const int64_t  d   = static_cast<int64_t>(zz >> 1) ^ -static_cast<int64_t>(zz & 1);
                    prev = static_cast<size_t>(static_cast<int64_t>(prev) + d);
                    result.push_back(prev);
                }
                return result;
            }

            case PermFormat::ChunkRelative: {
                const uint32_t cs32 = static_cast<uint32_t>(blob[1])
                                    | (static_cast<uint32_t>(blob[2]) << 8)
                                    | (static_cast<uint32_t>(blob[3]) << 16)
                                    | (static_cast<uint32_t>(blob[4]) << 24);
                const size_t  chunkSize = cs32;
                const uint8_t rankBits  = blob[5];
                const size_t  N         = static_cast<size_t>(detail::readN(blob.data() + 6));
                std::vector<size_t> result(N);
                BitReader r(blob.data() + 14, blob.size() - 14, BitOrder::LSB);
                for (size_t i = 0; i < N; ++i)
                    result[i] = (i / chunkSize) * chunkSize + r.read(rankBits);
                return result;
            }
        }
        return {};
    }

    // -----------------------------------------------------------------------
    // Single-element O(1) random access
    // -----------------------------------------------------------------------

    static size_t forwardAt(std::span<const uint8_t> blob, size_t origIdx) {
        if (blob.empty()) return origIdx;
        switch (static_cast<PermFormat>(blob[0])) {
            case PermFormat::None:
                return origIdx;

            case PermFormat::FlatBitPacked: {
                const uint8_t bpe = blob[1];
                // skip tag(1) + bpe(1) + N(8) = 10 bytes header
                return detail::readBitsAt(blob.data() + 10, origIdx * bpe, bpe);
            }

            case PermFormat::ChunkRelative: {
                const uint32_t cs32 = static_cast<uint32_t>(blob[1])
                                    | (static_cast<uint32_t>(blob[2]) << 8)
                                    | (static_cast<uint32_t>(blob[3]) << 16)
                                    | (static_cast<uint32_t>(blob[4]) << 24);
                const size_t  chunkSize = cs32;
                const uint8_t rankBits  = blob[5];
                // skip tag(1) + cs(4) + rankBits(1) + N(8) = 14 bytes header
                const size_t base = (origIdx / chunkSize) * chunkSize;
                return base + detail::readBitsAt(blob.data() + 14, origIdx * rankBits, rankBits);
            }

            case PermFormat::DeltaBitPacked:
                // No O(1) path — decode sequentially (override in subclass for better perf)
                return unpackForward(blob)[origIdx];
        }
        return origIdx;
    }

    // -----------------------------------------------------------------------
    // Bulk lookup
    // For random-access formats: O(1) per index.
    // For DeltaBitPacked: sort indices, do one sequential scan.
    // -----------------------------------------------------------------------

    static std::vector<size_t> forwardBulk(std::span<const uint8_t> blob,
                                           std::span<const size_t>   origIndices) {
        if (origIndices.empty()) return {};
        const size_t M = origIndices.size();
        std::vector<size_t> result(M);

        if (supportsRandomAccess(blob)) {
            for (size_t i = 0; i < M; ++i)
                result[i] = forwardAt(blob, origIndices[i]);
            return result;
        }

        // DeltaBitPacked: one sequential scan over the full permutation,
        // answering sorted query indices as we go.
        auto allFwd = unpackForward(blob);
        for (size_t i = 0; i < M; ++i)
            result[i] = allFwd[origIndices[i]];
        return result;
    }

    // -----------------------------------------------------------------------
    // Size estimation (bytes)
    // -----------------------------------------------------------------------

    static size_t estimatePackedSize(size_t N, PermFormat fmt, size_t chunkSize = 256) noexcept {
        switch (fmt) {
            case PermFormat::None:
                return 1;
            case PermFormat::FlatBitPacked: {
                const uint8_t bpe = detail::bitsNeeded(N > 1 ? N - 1 : 1);
                return 10 + (N * bpe + 7) / 8;
            }
            case PermFormat::DeltaBitPacked: {
                // Rough: assume average delta ≈ sqrt(N) → log2(2*sqrt(N)) bits each
                const auto avgDelta = static_cast<size_t>(2 * std::sqrt(static_cast<double>(N)));
                const uint8_t bpe  = detail::bitsNeeded(avgDelta > 0 ? avgDelta : 1);
                return 11 + (N * bpe + 7) / 8;
            }
            case PermFormat::ChunkRelative: {
                chunkSize = std::max(chunkSize, size_t{2});
                const uint8_t rankBits = detail::bitsNeeded(chunkSize - 1);
                return 14 + (N * rankBits + 7) / 8;
            }
        }
        return N * 8;
    }
};

} // namespace encodings::reorderers
