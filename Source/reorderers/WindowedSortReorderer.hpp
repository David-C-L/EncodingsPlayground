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
// WindowedSortReorderer<T, W>
//
// Stable-sorts within non-overlapping windows of compile-time size W.
// Balances compression improvement against random access overhead.
//
// Key property: elements in input window k map to output positions [k*W, (k+1)*W).
// This makes the permutation ChunkRelative-representable with only
// ceil(log2(W)) bits per element (e.g. 8 bits for W=256).
//
// Random access at original index i:
//   chunk k = i / W
//   intra-chunk rank = PermutationStore::forwardAt(permBlob, i)  ← O(1)
//   reordered index j = k*W + rank
// ---------------------------------------------------------------------------

template <ReorderableType T, size_t W = 256>
class WindowedSortReorderer : public Reorderer<T> {
    static_assert(W >= 2, "Window size must be at least 2");
public:
    // permFmt must be ChunkRelative (O(1) RA), ChunkRelativeZstd, or ChunkRelativeLZ4.
    explicit WindowedSortReorderer(PermFormat permFmt = PermFormat::ChunkRelative)
        : permFmt_(permFmt) {}

    // -----------------------------------------------------------------------
    // Forward: sort within each window of size W
    // -----------------------------------------------------------------------

    ReorderResult<T> reorder(std::span<const T> data) override {
        const size_t N = data.size();
        if (N == 0) return {};

        std::vector<T>      reordered(N);
        std::vector<size_t> fwdPerm(N);

        for (size_t k = 0; k * W < N; ++k) {
            const size_t wStart = k * W;
            const size_t wEnd   = std::min(wStart + W, N);
            const size_t wLen   = wEnd - wStart;

            // Argsort within window
            std::vector<size_t> idx(wLen);
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(),
                             [&](size_t a, size_t b) {
                                 return data[wStart + a] < data[wStart + b];
                             });

            // Fill reordered values and forward permutation
            for (size_t r = 0; r < wLen; ++r) {
                const size_t origPos    = wStart + idx[r];
                const size_t sortedPos  = wStart + r;
                reordered[sortedPos]    = data[origPos];
                fwdPerm[origPos]        = sortedPos;
            }
        }

        std::vector<uint8_t> permData;
        if (permFmt_ == PermFormat::ChunkRelativeZstd || permFmt_ == PermFormat::ChunkRelativeLZ4) {
            permData = PermutationStore::packChunkCompressed(fwdPerm, W, permFmt_);
        } else {
            permData = PermutationStore::packChunkRelative(fwdPerm, W);
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
    // Random access: O(1) via ChunkRelative lookup
    // -----------------------------------------------------------------------

    std::optional<size_t> originalToReorderedIndex(
        size_t origIdx, std::span<const uint8_t> permData) const override {
        return PermutationStore::forwardAt(permData, origIdx);
    }

    // Override bulk: group queries by window, process each window's entries once.
    std::optional<std::vector<size_t>> originalToReorderedIndices(
        std::span<const size_t>  origIndices,
        std::span<const uint8_t> permData) const override {
        return PermutationStore::forwardBulk(permData, origIndices);
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    size_t estimatePermutationSize(size_t N) const override {
        return PermutationStore::estimatePackedSize(N, permFmt_, W);
    }

    std::string name() const override {
        return std::string("WindowedSort<") + std::to_string(W) + ">[" +
               PermutationStore::formatName(permFmt_) + "]";
    }

private:
    PermFormat permFmt_;
};

} // namespace encodings::reorderers
