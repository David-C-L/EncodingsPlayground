#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "Reorderer.hpp"

namespace encodings::reorderers {

// ---------------------------------------------------------------------------
// BWTReorderer<T, W>
//
// Integer Burrows-Wheeler Transform via cyclic suffix array.
// Transforms a sequence so that values with the same preceding context are
// adjacent, creating long runs of repeated values → excellent for RLE and
// entropy coders.
//
// W = 0 : global BWT (full sequence; sequential decode only, no random access)
// W > 0 : windowed BWT (independent windows of size W; O(W) per decodeAt)
//
// Permutation data layout:
//   [N:8][W:8][numWindows:8][primaryIndex_0:8]...[primaryIndex_numWindows-1:8]
//
// Complexity:
//   Forward: O(N * W * log(N)) using std::sort with O(W) comparator
//             (upgrade to SA-IS for O(N log N) later if needed)
//   Inverse:  O(N * alphabet_size) for LF-mapping reconstruction
//
// WARNING: For W=0 and large N, the O(N^2 log N) suffix sort is expensive.
// Use W > 0 (e.g. W=4096) for practical performance on large datasets.
// ---------------------------------------------------------------------------

template <ReorderableType T, size_t W = 4096>
class BWTReorderer : public Reorderer<T> {
public:
    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    ReorderResult<T> reorder(std::span<const T> data) override {
        const size_t N = data.size();
        if (N == 0) return {};

        if constexpr (W == 0) {
            // Global BWT
            auto [bwtOut, primaryIdx] = bwtForward(data);
            auto permData = encodePermData(N, 0, {primaryIdx});
            return {std::move(bwtOut), std::move(permData)};
        } else {
            // Windowed BWT
            std::vector<T>      out(N);
            std::vector<size_t> primaryIndices;
            const size_t        numWindows = (N + W - 1) / W;
            primaryIndices.reserve(numWindows);

            for (size_t k = 0; k < numWindows; ++k) {
                const size_t wStart = k * W;
                const size_t wEnd   = std::min(wStart + W, N);
                auto [bwtW, pidx]   = bwtForward(data.subspan(wStart, wEnd - wStart));
                std::copy(bwtW.begin(), bwtW.end(), out.begin() + static_cast<ptrdiff_t>(wStart));
                primaryIndices.push_back(pidx);
            }

            auto permData = encodePermData(N, W, primaryIndices);
            return {std::move(out), std::move(permData)};
        }
    }

    // -----------------------------------------------------------------------
    // Inverse
    // -----------------------------------------------------------------------

    std::vector<T> unreorder(std::span<const T>       reordered,
                             std::span<const uint8_t> permData) override {
        const size_t N = reordered.size();
        auto [Nstored, windowSize, primaryIndices] = decodePermData(permData);
        (void)Nstored;

        if (windowSize == 0) {
            // Global inverse
            return bwtInverse(reordered, primaryIndices[0]);
        }

        // Windowed inverse
        std::vector<T> out(N);
        const size_t   numWindows = primaryIndices.size();
        for (size_t k = 0; k < numWindows; ++k) {
            const size_t wStart = k * windowSize;
            const size_t wEnd   = std::min(wStart + windowSize, N);
            auto wOut = bwtInverse(reordered.subspan(wStart, wEnd - wStart), primaryIndices[k]);
            std::copy(wOut.begin(), wOut.end(), out.begin() + static_cast<ptrdiff_t>(wStart));
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Random access
    // BWT inversion is inherently sequential; no O(1) per-element path.
    // decodeAt for windowed mode decodes only the relevant window (O(W)).
    // -----------------------------------------------------------------------

    std::optional<size_t> originalToReorderedIndex(
        size_t /* origIdx */,
        std::span<const uint8_t> /* permData */) const override {
        return std::nullopt; // BWT has no forward position mapping
    }

    std::optional<std::vector<size_t>> originalToReorderedIndices(
        std::span<const size_t>  /* origIndices */,
        std::span<const uint8_t> /* permData */) const override {
        return std::nullopt;
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    size_t estimatePermutationSize(size_t N) const override {
        const size_t numWindows = (W == 0) ? 1 : (N + W - 1) / W;
        return 24 + numWindows * 8;
    }

    std::string name() const override {
        if constexpr (W == 0)
            return "BWT";
        else
            return "BWT<" + std::to_string(W) + ">";
    }

private:
    // -----------------------------------------------------------------------
    // Core cyclic BWT forward (O(L^2 log L) suffix sort, L = window length)
    // -----------------------------------------------------------------------

    static std::pair<std::vector<T>, size_t> bwtForward(std::span<const T> seq) {
        const size_t L = seq.size();
        if (L == 0) return {{}, 0};
        if (L == 1) return {std::vector<T>{seq[0]}, 0};

        // Build cyclic suffix array by sorting rotation indices
        std::vector<size_t> sa(L);
        std::iota(sa.begin(), sa.end(), 0);
        std::stable_sort(sa.begin(), sa.end(), [&](size_t a, size_t b) {
            for (size_t k = 0; k < L; ++k) {
                const T va = seq[(a + k) % L];
                const T vb = seq[(b + k) % L];
                if (va != vb) return va < vb;
            }
            return false;
        });

        // BWT last column + primary index
        std::vector<T> bwtOut(L);
        size_t primaryIdx = 0;
        for (size_t j = 0; j < L; ++j) {
            bwtOut[j] = seq[(sa[j] + L - 1) % L];
            if (sa[j] == 0) primaryIdx = j;
        }
        return {std::move(bwtOut), primaryIdx};
    }

    // -----------------------------------------------------------------------
    // Core cyclic BWT inverse via LF-mapping (O(L * log(alphabet)))
    // -----------------------------------------------------------------------

    static std::vector<T> bwtInverse(std::span<const T> bwt, size_t primaryIdx) {
        const size_t L = bwt.size();
        if (L == 0) return {};
        if (L == 1) return {bwt[0]};

        // Build sorted first column and LF-mapping
        // LF[i] = position in sorted order of (FC[i], bwt[i])
        std::vector<size_t> order(L);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return bwt[a] < bwt[b]; });

        // LF mapping: lf[order[i]] = i  (order maps sorted pos → bwt pos)
        // i.e., lf[j] = where in the sorted order does bwt index j land?
        std::vector<size_t> lf(L);
        for (size_t i = 0; i < L; ++i)
            lf[order[i]] = i;

        // Reconstruct: follow lf from primaryIdx
        std::vector<T> out(L);
        size_t cur = primaryIdx;
        for (size_t i = L; i-- > 0;) {
            out[i] = bwt[cur];
            cur     = lf[cur];
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Permutation data serialisation
    // -----------------------------------------------------------------------

    static std::vector<uint8_t> encodePermData(size_t N, size_t Wval,
                                               const std::vector<size_t>& pidxs) {
        std::vector<uint8_t> blob;
        blob.reserve(24 + pidxs.size() * 8);
        auto appendU64 = [&](uint64_t v) {
            for (int b = 0; b < 8; ++b) blob.push_back(static_cast<uint8_t>(v >> (8 * b)));
        };
        appendU64(static_cast<uint64_t>(N));
        appendU64(static_cast<uint64_t>(Wval));
        appendU64(static_cast<uint64_t>(pidxs.size()));
        for (size_t p : pidxs) appendU64(static_cast<uint64_t>(p));
        return blob;
    }

    struct PermDataDecoded { size_t N; size_t windowSize; std::vector<size_t> primaryIndices; };

    static PermDataDecoded decodePermData(std::span<const uint8_t> blob) {
        auto readU64 = [&](size_t off) -> uint64_t {
            uint64_t v = 0;
            for (int b = 0; b < 8; ++b)
                v |= static_cast<uint64_t>(blob[off + b]) << (8 * b);
            return v;
        };
        const size_t N           = static_cast<size_t>(readU64(0));
        const size_t windowSize  = static_cast<size_t>(readU64(8));
        const size_t numWindows  = static_cast<size_t>(readU64(16));
        std::vector<size_t> pidxs(numWindows);
        for (size_t k = 0; k < numWindows; ++k)
            pidxs[k] = static_cast<size_t>(readU64(24 + k * 8));
        return {N, windowSize, std::move(pidxs)};
    }
};

} // namespace encodings::reorderers
