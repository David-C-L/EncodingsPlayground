#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

namespace encodings::exceptions {

// ---------------------------------------------------------------------------
// BitmapRank
//
// Shared mechanical core for "dense bitmap + O(word-count) prefix/range rank
// of set bits" — extracted from MainlyConstantEncoder's original
// prefixNonCommonCount/rangeNonCommonCount so that MainlyConstantEncoder and
// the new ExceptionEncoder (exceptions/ExceptionEncoder.hpp) both depend on
// one implementation instead of duplicating this bit-twiddling.
//
// Convention: bit i (LSB-first within word i/64) is 1 iff position i is
// "set". Callers decide what "set" means (MainlyConstantEncoder used to
// treat 1 = common and count non-common positions via ~word; here the
// utility always counts literal set bits — callers wanting the complement
// count simply subtract from the range width, e.g.
// nonSetCount = (limit) - BitmapRank::prefixSetCount(bitmap, limit)).
//
// This is Dense-mode only (Phase 1 of the exception-handling design, see
// ~/.claude/plans/could-you-analyse-the-misty-widget.md). An RLE-compressed
// mode is planned for Phase 2 and will be added as an alternative
// representation alongside this one, not a replacement.
// ---------------------------------------------------------------------------
class BitmapRank {
public:
    // Builds a packed dense bitmap (ceil(n/64) uint64 words, zero-initialised)
    // from a predicate evaluated at each index in [0, n).
    template <typename Pred>
    static std::vector<uint64_t> build(size_t n, Pred&& isSet) {
        const size_t numWords = (n + 63) / 64;
        std::vector<uint64_t> bitmap(numWords, 0);
        for (size_t i = 0; i < n; ++i) {
            if (isSet(i)) {
                bitmap[i / 64] |= (uint64_t{1} << (i % 64));
            }
        }
        return bitmap;
    }

    // Unaligned 8-byte load — compiles to a single movq on x86. Bitmap byte
    // offsets inside a larger serialized buffer are not guaranteed to be
    // 8-byte aligned, so this must use memcpy rather than a reinterpret_cast.
    static uint64_t loadWord(const uint8_t* p) noexcept {
        uint64_t w;
        std::memcpy(&w, p, 8);
        return w;
    }

    // Count set bits in [0, limit). 4-way independent accumulator breaks the
    // serial dependency chain on `count` across word iterations.
    static uint32_t prefixSetCount(const uint8_t* bitmapStart, size_t limit) noexcept {
        if (limit == 0) return 0;
        const uint32_t fullWords = static_cast<uint32_t>(limit / 64);
        uint32_t w = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (; w + 4 <= fullWords; w += 4) {
            c0 += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + (w + 0) * 8)));
            c1 += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + (w + 1) * 8)));
            c2 += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + (w + 2) * 8)));
            c3 += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + (w + 3) * 8)));
        }
        uint32_t count = c0 + c1 + c2 + c3;
        for (; w < fullWords; ++w) {
            count += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + w * 8)));
        }
        const uint32_t rem = static_cast<uint32_t>(limit % 64);
        if (rem > 0) {
            const uint64_t mask = (uint64_t{1} << rem) - 1;
            count += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + fullWords * 8) & mask));
        }
        return count;
    }

    // Count set bits in [limitA, limitB). Starts at word limitA/64, so callers
    // that already hold the rank at limitA (e.g. carrying a cursor forward
    // across an ascending RowRangeList) avoid recomputing the prefix from bit 0.
    static uint32_t rangeSetCount(const uint8_t* bitmapStart, size_t limitA, size_t limitB) noexcept {
        if (limitA >= limitB) return 0;
        const uint32_t wA = static_cast<uint32_t>(limitA / 64), remA = static_cast<uint32_t>(limitA % 64);
        const uint32_t wB = static_cast<uint32_t>(limitB / 64), remB = static_cast<uint32_t>(limitB % 64);
        if (wA == wB) {
            uint64_t word = loadWord(bitmapStart + wA * 8);
            if (remA) word &= ~((uint64_t{1} << remA) - 1);
            if (remB) word &= (uint64_t{1} << remB) - 1;
            return static_cast<uint32_t>(__builtin_popcountll(word));
        }
        uint32_t count = 0;
        {
            uint64_t word = loadWord(bitmapStart + wA * 8);
            if (remA) word &= ~((uint64_t{1} << remA) - 1);
            count += static_cast<uint32_t>(__builtin_popcountll(word));
        }
        uint32_t w = wA + 1;
        uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (; w + 4 <= wB; w += 4) {
            c0 += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + (w + 0) * 8)));
            c1 += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + (w + 1) * 8)));
            c2 += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + (w + 2) * 8)));
            c3 += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + (w + 3) * 8)));
        }
        count += c0 + c1 + c2 + c3;
        for (; w < wB; ++w) {
            count += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + w * 8)));
        }
        if (remB) {
            const uint64_t mask = (uint64_t{1} << remB) - 1;
            count += static_cast<uint32_t>(__builtin_popcountll(loadWord(bitmapStart + wB * 8) & mask));
        }
        return count;
    }
};

} // namespace encodings::exceptions
