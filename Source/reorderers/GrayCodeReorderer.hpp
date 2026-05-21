#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "Reorderer.hpp"

namespace encodings::reorderers {

// ---------------------------------------------------------------------------
// GrayCodeReorderer<T>
//
// Value bijection: gray(v) = v ^ (v >> 1).
// Adjacent integers in Gray-code representation differ by exactly 1 bit.
// This benefits bit-sections in SubIntSplit that straddle value transitions
// in slowly-varying integer sequences.
//
// No permutation is stored (positions unchanged; only values transform).
// originalToReorderedIndex simply returns origIdx.
// ---------------------------------------------------------------------------

namespace detail_gray {

// Compile-time Gray decode (binary from Gray code).
// Works for T = int32_t or int64_t via unsigned reinterpretation.
template <typename T>
[[nodiscard]] inline T grayDecode(T gray) noexcept {
    using U = std::make_unsigned_t<T>;
    U g = static_cast<U>(gray);
    U b = g;
    g >>= 1; while (g) { b ^= g; g >>= 1; }
    return static_cast<T>(b);
}

// For 64-bit: unrolled log2(64)=6 steps, branchless.
[[nodiscard]] inline uint64_t grayDecode64(uint64_t g) noexcept {
    g ^= (g >> 32);
    g ^= (g >> 16);
    g ^= (g >> 8);
    g ^= (g >> 4);
    g ^= (g >> 2);
    g ^= (g >> 1);
    return g;
}

[[nodiscard]] inline uint32_t grayDecode32(uint32_t g) noexcept {
    g ^= (g >> 16);
    g ^= (g >> 8);
    g ^= (g >> 4);
    g ^= (g >> 2);
    g ^= (g >> 1);
    return g;
}

} // namespace detail_gray

template <ReorderableType T>
class GrayCodeReorderer : public Reorderer<T> {
public:
    // -----------------------------------------------------------------------
    // Forward: v → v ^ (v >> 1)  (element-wise, trivially SIMD-able)
    // -----------------------------------------------------------------------

    ReorderResult<T> reorder(std::span<const T> data) override {
        std::vector<T> out(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            using U = std::make_unsigned_t<T>;
            const U u = static_cast<U>(data[i]);
            out[i] = static_cast<T>(u ^ (u >> 1));
        }
        // No permutation data — positions unchanged
        return {std::move(out), {}};
    }

    // -----------------------------------------------------------------------
    // Inverse: Gray → binary decode (O(log bits) per element, unrolled)
    // -----------------------------------------------------------------------

    std::vector<T> unreorder(std::span<const T>       reordered,
                             std::span<const uint8_t> /* permData */) override {
        std::vector<T> out(reordered.size());
        for (size_t i = 0; i < reordered.size(); ++i) {
            if constexpr (sizeof(T) == 8) {
                out[i] = static_cast<T>(detail_gray::grayDecode64(
                    static_cast<uint64_t>(reordered[i])));
            } else {
                out[i] = static_cast<T>(detail_gray::grayDecode32(
                    static_cast<uint32_t>(reordered[i])));
            }
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Random access: positions are unchanged — O(1), no permutation needed
    // -----------------------------------------------------------------------

    std::optional<size_t> originalToReorderedIndex(
        size_t origIdx, std::span<const uint8_t> /* permData */) const override {
        return origIdx;
    }

    std::optional<std::vector<size_t>> originalToReorderedIndices(
        std::span<const size_t>  origIndices,
        std::span<const uint8_t> /* permData */) const override {
        // Positions are unchanged; just echo the input indices
        return std::vector<size_t>(origIndices.begin(), origIndices.end());
    }

    size_t estimatePermutationSize(size_t /* N */) const override { return 1; } // None tag

    std::string name() const override { return "GrayCode"; }
};

} // namespace encodings::reorderers
