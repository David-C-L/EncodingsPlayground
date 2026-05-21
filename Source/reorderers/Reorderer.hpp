#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "PermutationStore.hpp"

namespace encodings::reorderers {

// Constraint: reorderers are defined for 32- and 64-bit integer sequences.
template <typename T>
concept ReorderableType = std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

// Result of a reorder() call: the reordered value sequence plus an opaque
// permutation blob (may be empty for value-only transforms like GrayCode).
template <ReorderableType T>
struct ReorderResult {
    std::vector<T>       reorderedValues;
    std::vector<uint8_t> permutationData;
};

// ---------------------------------------------------------------------------
// Abstract Reorderer<T>
// ---------------------------------------------------------------------------

template <ReorderableType T>
class Reorderer {
public:
    virtual ~Reorderer() = default;

    // Reorder the data for better compression.
    // Returns the reordered values and an opaque permutation blob.
    virtual ReorderResult<T> reorder(std::span<const T> data) = 0;

    // Invert: recover the original sequence from the reordered form.
    virtual std::vector<T> unreorder(std::span<const T>       reordered,
                                     std::span<const uint8_t> permData) = 0;

    // Map original index i to reordered index j.
    // Returns nullopt when O(1) lookup is not available (e.g. BWT W=0, MTF W=0).
    virtual std::optional<size_t> originalToReorderedIndex(
        size_t origIdx, std::span<const uint8_t> permData) const = 0;

    // Bulk map: avoid repeated single-element calls.
    // Default: delegates to originalToReorderedIndex per element.
    // Override when a single sequential pass over the blob is cheaper (e.g. DeltaBitPacked).
    // Returns nullopt if single-element lookup is unavailable.
    virtual std::optional<std::vector<size_t>> originalToReorderedIndices(
        std::span<const size_t>  origIndices,
        std::span<const uint8_t> permData) const {
        if (origIndices.empty()) return std::vector<size_t>{};
        std::vector<size_t> result;
        result.reserve(origIndices.size());
        for (size_t idx : origIndices) {
            auto r = originalToReorderedIndex(idx, permData);
            if (!r) return std::nullopt;
            result.push_back(*r);
        }
        return result;
    }

    // True when the forward transform equals the inverse (e.g. BitShuffle).
    virtual bool isSelfInverse() const { return false; }

    // Estimated permutation storage in bytes for N elements.
    virtual size_t estimatePermutationSize(size_t N) const = 0;

    virtual std::string name() const = 0;
};

} // namespace encodings::reorderers
