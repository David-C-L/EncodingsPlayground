#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __AVX2__
#  include <immintrin.h>
#endif

namespace encodings::encoders::detail {

// ---------------------------------------------------------------------------
// loadDictValue — load one T from a byte-aligned dictionary buffer.
// ---------------------------------------------------------------------------
template<typename T>
    requires std::is_trivially_copyable_v<T>
inline T loadDictValue(const uint8_t* dictBytes, size_t index) noexcept {
    T value;
    std::memcpy(&value, dictBytes + index * sizeof(T), sizeof(T));
    return value;
}

// ---------------------------------------------------------------------------
// chooseKeyBitWidth — minimum bit-width needed to index dictSize entries.
// dictSize 0 or 1 returns 1 (still write one bit per entry).
// ---------------------------------------------------------------------------
inline uint32_t chooseKeyBitWidth(size_t dictSize, bool allowNonPowerOfTwo) noexcept {
    if (dictSize <= 1) return 1;
    const uint32_t bits = static_cast<uint32_t>(
        std::max<size_t>(1, std::bit_width(dictSize - 1)));
    if (!allowNonPowerOfTwo) {
        if (bits <= 1)  return 1;
        if (bits <= 2)  return 2;
        if (bits <= 4)  return 4;
        if (bits <= 8)  return 8;
        if (bits <= 16) return 16;
        return 32;
    }
    return bits <= 32 ? bits : 32;
}

// ---------------------------------------------------------------------------
// exactKeyBitWidth — minimum bits to index dictSize entries, with no snapping
// to a power-of-two bucket. dictSize==1 correctly yields 0 (nothing to encode
// — std::bit_width(0) is 0 per the standard). dictSize==0 is not a valid
// input; callers must only invoke this for active (non-empty) tiers.
// ---------------------------------------------------------------------------
inline uint32_t exactKeyBitWidth(size_t dictSize) noexcept {
    return static_cast<uint32_t>(std::bit_width(dictSize - 1));
}

// ---------------------------------------------------------------------------
// decodeGeneral — arbitrary bit-width, arbitrary startBit offset.
// Buffer must have >= 8 bytes padding beyond the logical key data.
// ---------------------------------------------------------------------------
template<typename T>
    requires std::is_trivially_copyable_v<T>
inline void decodeGeneral(
    const uint8_t* __restrict__ keys,
    const uint8_t* __restrict__ dictBytes,
    T*             __restrict__ dst,
    size_t n,
    uint32_t keyBitWidth,
    size_t startBit) noexcept
{
    const uint64_t mask = (keyBitWidth >= 64u)
                          ? ~uint64_t{0}
                          : ((uint64_t{1} << keyBitWidth) - 1u);
    size_t bitPos = startBit;
    for (size_t i = 0; i < n; ++i, bitPos += keyBitWidth) {
        const size_t   wordIdx = bitPos >> 6;
        const uint32_t offset  = static_cast<uint32_t>(bitPos & 63u);
        uint64_t word;
        std::memcpy(&word, keys + wordIdx * sizeof(uint64_t), sizeof(uint64_t));
        uint64_t key = word >> offset;
        if (offset + keyBitWidth > 64u) {
            uint64_t next;
            std::memcpy(&next, keys + (wordIdx + 1) * sizeof(uint64_t), sizeof(uint64_t));
            key |= next << (64u - offset);
        }
        dst[i] = loadDictValue<T>(dictBytes, static_cast<size_t>(key & mask));
    }
}

// ---------------------------------------------------------------------------
// decodeBatch<W> — optimised decoder for power-of-two key widths.
// W must divide 64.  AVX2 path active for T=uint8_t, W=4.
// ---------------------------------------------------------------------------
template<uint32_t W, typename T>
    requires std::is_trivially_copyable_v<T>
inline void decodeBatch(
    const uint8_t* __restrict__ keys,
    const uint8_t* __restrict__ dictBytes,
    T*             __restrict__ dst,
    size_t n) noexcept
{
    static_assert(64u % W == 0, "W must divide 64");
    constexpr uint32_t kPerWord = 64u / W;
    constexpr uint64_t kMask    = (W == 64u) ? ~uint64_t{0}
                                              : ((uint64_t{1} << W) - 1u);

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, uint8_t> && W == 4) {
        size_t i = 0;
        if (n >= 32) {
            const __m128i dict_xmm = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(dictBytes));
            const __m256i dict_ymm = _mm256_broadcastsi128_si256(dict_xmm);
            for (; i + 32 <= n; i += 32) {
                const __m128i raw = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(keys + i / 2));
                const __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
                const __m128i hi = _mm_and_si128(
                    _mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
                const __m256i idx = _mm256_set_m128i(
                    _mm_unpackhi_epi8(lo, hi),
                    _mm_unpacklo_epi8(lo, hi));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(dst + i),
                    _mm256_shuffle_epi8(dict_ymm, idx));
            }
        }
        size_t bitPos = i * W;
        for (; i < n; ++i, bitPos += W) {
            const size_t wordIdx = bitPos >> 6;
            uint64_t word;
            std::memcpy(&word, keys + wordIdx * sizeof(uint64_t), sizeof(uint64_t));
            dst[i] = loadDictValue<T>(dictBytes,
                static_cast<size_t>((word >> (bitPos & 63u)) & kMask));
        }
        return;
    }
#endif

    const size_t fullWords = n / kPerWord;
    for (size_t w = 0; w < fullWords; ++w) {
        uint64_t word;
        std::memcpy(&word, keys + w * sizeof(uint64_t), sizeof(uint64_t));
        T* dw = dst + w * kPerWord;
        for (uint32_t j = 0; j < kPerWord; ++j, word >>= W) {
            dw[j] = loadDictValue<T>(dictBytes, static_cast<size_t>(word & kMask));
        }
    }
    const size_t rem = n % kPerWord;
    if (rem > 0) {
        uint64_t word;
        std::memcpy(&word, keys + fullWords * sizeof(uint64_t), sizeof(uint64_t));
        T* dw = dst + fullWords * kPerWord;
        for (size_t j = 0; j < rem; ++j, word >>= W) {
            dw[j] = loadDictValue<T>(dictBytes, static_cast<size_t>(word & kMask));
        }
    }
}

// ---------------------------------------------------------------------------
// dispatchDecode — route to decodeBatch<W> or decodeGeneral.
// Fast batch paths require startBit == 0.
// ---------------------------------------------------------------------------
template<typename T>
    requires std::is_trivially_copyable_v<T>
inline void dispatchDecode(
    const uint8_t* __restrict__ keys,
    const uint8_t* __restrict__ dictBytes,
    T*             __restrict__ dst,
    size_t n,
    uint32_t keyBitWidth,
    size_t startBit = 0) noexcept
{
    if (startBit == 0) {
        switch (keyBitWidth) {
            case  1: decodeBatch< 1, T>(keys, dictBytes, dst, n); return;
            case  2: decodeBatch< 2, T>(keys, dictBytes, dst, n); return;
            case  4: decodeBatch< 4, T>(keys, dictBytes, dst, n); return;
            case  8: decodeBatch< 8, T>(keys, dictBytes, dst, n); return;
            case 16: decodeBatch<16, T>(keys, dictBytes, dst, n); return;
            default: break;
        }
    }
    decodeGeneral<T>(keys, dictBytes, dst, n, keyBitWidth, startBit);
}

// ---------------------------------------------------------------------------
// writeDictionary — serialize dictionary to pre-allocated dest.
// ---------------------------------------------------------------------------
template<typename T>
inline void writeDictionary(const std::vector<T>& dictionary, uint8_t* dest) {
    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(dest, dictionary.data(), dictionary.size() * sizeof(T));
    } else if constexpr (std::is_same_v<T, std::string>) {
        for (const auto& str : dictionary) {
            const size_t len = str.size();
            std::memcpy(dest, &len, sizeof(size_t));
            dest += sizeof(size_t);
            std::memcpy(dest, str.data(), len);
            dest += len;
        }
    } else {
        std::memcpy(dest, dictionary.data(), dictionary.size() * sizeof(T));
    }
}

// ---------------------------------------------------------------------------
// readDictionary — deserialize dictionary from src into an owned vector.
// ---------------------------------------------------------------------------
template<typename T>
inline std::vector<T> readDictionary(const uint8_t* src, size_t dictSize) {
    if constexpr (std::is_same_v<T, std::string>) {
        std::vector<T> dictionary;
        dictionary.reserve(dictSize);
        for (size_t i = 0; i < dictSize; ++i) {
            size_t len;
            std::memcpy(&len, src, sizeof(size_t));
            src += sizeof(size_t);
            dictionary.emplace_back(reinterpret_cast<const char*>(src), len);
            src += len;
        }
        return dictionary;
    } else {
        std::vector<T> dictionary(dictSize);
        std::memcpy(dictionary.data(), src, dictSize * sizeof(T));
        return dictionary;
    }
}

} // namespace encodings::encoders::detail
