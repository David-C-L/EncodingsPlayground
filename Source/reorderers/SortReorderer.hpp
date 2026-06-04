#pragma once

#include <algorithm>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "Reorderer.hpp"
#include "PermutationStore.hpp"

namespace encodings::reorderers {

// ---------------------------------------------------------------------------
// SortReorderer<T>
//
// Stable-sorts the value sequence (ascending).  Identical values stay adjacent
// → optimal for RLE, tightens the value range for FOR/BitPacking, and reduces
// per-section entropy for SubIntSplit.
//
// permFmt controls how the permutation is stored.  Formats that exploit
// value-group structure (ValueGrouped, InverseEliasFano) require the sorted
// values to identify group boundaries; reorder() handles this automatically.
// ---------------------------------------------------------------------------

template <ReorderableType T>
class SortReorderer : public Reorderer<T> {
public:
    explicit SortReorderer(PermFormat permFmt = PermFormat::FlatBitPacked)
        : permFmt_(permFmt) {}

    // -----------------------------------------------------------------------
    // Forward: stable argsort + sort
    // -----------------------------------------------------------------------

    ReorderResult<T> reorder(std::span<const T> data) override {
        const size_t N = data.size();
        if (N == 0) return {};

        // sortedIndices[j] = original index that ends up at sorted position j
        std::vector<size_t> sortedIndices(N);
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::stable_sort(sortedIndices.begin(), sortedIndices.end(),
                         [&](size_t a, size_t b) { return data[a] < data[b]; });

        // fwd[i] = sorted position of original element i
        std::vector<size_t> fwdPerm(N);
        for (size_t j = 0; j < N; ++j)
            fwdPerm[sortedIndices[j]] = j;

        // Build sorted value sequence
        std::vector<T> reordered(N);
        for (size_t j = 0; j < N; ++j)
            reordered[j] = data[sortedIndices[j]];

        // Pack permutation — group-based formats need value group sizes
        std::vector<uint8_t> permData;
        if (needsGroupSizes(permFmt_)) {
            auto groupSizes = computeGroupSizes(std::span<const T>(reordered));
            permData = PermutationStore::packWithGroups(fwdPerm, groupSizes, permFmt_);
        } else {
            permData = PermutationStore::pack(fwdPerm, permFmt_);
        }

        return {std::move(reordered), std::move(permData)};
    }

    // -----------------------------------------------------------------------
    // Inverse: original[i] = reordered[fwd[i]]
    // -----------------------------------------------------------------------

    std::vector<T> unreorder(std::span<const T>       reordered,
                             std::span<const uint8_t> permData) override {
        const size_t N = reordered.size();
        auto fwdPerm = PermutationStore::unpackForward(permData);
        std::vector<T> result(N);
        for (size_t i = 0; i < N; ++i)
            result[i] = reordered[fwdPerm[i]];
        return result;
    }

    // -----------------------------------------------------------------------
    // Random access
    // -----------------------------------------------------------------------

    std::optional<size_t> originalToReorderedIndex(
        size_t origIdx, std::span<const uint8_t> permData) const override {
        if (!PermutationStore::supportsRandomAccess(permData)) return std::nullopt;
        return PermutationStore::forwardAt(permData, origIdx);
    }

    std::optional<std::vector<size_t>> originalToReorderedIndices(
        std::span<const size_t>  origIndices,
        std::span<const uint8_t> permData) const override {
        return PermutationStore::forwardBulk(permData, origIndices);
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    size_t estimatePermutationSize(size_t N) const override {
        return PermutationStore::estimatePackedSize(N, permFmt_);
    }

    std::string name() const override {
        return std::string("Sort[") + PermutationStore::formatName(permFmt_) + "]";
    }

private:
    // Returns true when packWithGroups (rather than pack) is needed.
    static bool needsGroupSizes(PermFormat fmt) noexcept {
        return fmt == PermFormat::ValueGrouped || fmt == PermFormat::InverseEliasFano;
    }

    // Count consecutive equal values in the sorted sequence.
    static std::vector<size_t> computeGroupSizes(std::span<const T> sorted) {
        std::vector<size_t> sizes;
        if (sorted.empty()) return sizes;
        size_t count = 1;
        for (size_t i = 1; i < sorted.size(); ++i) {
            if (sorted[i] == sorted[i - 1]) {
                ++count;
            } else {
                sizes.push_back(count);
                count = 1;
            }
        }
        sizes.push_back(count);
        return sizes;
    }

    PermFormat permFmt_;
};

} // namespace encodings::reorderers
