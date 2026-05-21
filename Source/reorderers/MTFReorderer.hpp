#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "Reorderer.hpp"

namespace encodings::reorderers {

// ---------------------------------------------------------------------------
// MTFReorderer<T, W>
//
// Move-To-Front transform over the value alphabet.  Each value is replaced by
// its current rank in the MTF list, then moved to the front.  Output values
// are small integers (ranks) when temporal locality is high → compresses well
// with entropy coders or RLE.
//
// W = 0 : global MTF (whole sequence; sequential decode only, no random access)
// W > 0 : windowed MTF (independent windows of size W; O(W) per decodeAt)
//
// Permutation data layout:
//   [N as uint64_t: 8 bytes]
//   [W as uint64_t: 8 bytes]  (0 for global)
//   [alphabet size as uint64_t: 8 bytes]
//   [alphabet values: alphabet_size * sizeof(T) bytes]
//
// The alphabet is the sorted set of distinct values; stored so that the
// decoder can reconstruct the initial MTF list without the original data.
// ---------------------------------------------------------------------------

template <ReorderableType T, size_t W = 0>
class MTFReorderer : public Reorderer<T> {
public:
    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    ReorderResult<T> reorder(std::span<const T> data) override {
        const size_t N = data.size();
        if (N == 0) return {};

        // Build sorted alphabet
        std::vector<T> alphabet(data.begin(), data.end());
        std::sort(alphabet.begin(), alphabet.end());
        alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());

        // Encode permutation data (alphabet for reconstruction)
        std::vector<uint8_t> permData = encodePermData(N, alphabet);

        // Apply MTF
        std::vector<T> out(N);
        if constexpr (W == 0) {
            applyMTF(data, std::span<T>(alphabet), std::span<T>(out));
        } else {
            for (size_t k = 0; k * W < N; ++k) {
                const size_t wStart = k * W;
                const size_t wEnd   = std::min(wStart + W, N);
                std::vector<T> localAlpha = alphabet; // fresh copy per window
                applyMTF(data.subspan(wStart, wEnd - wStart),
                         std::span<T>(localAlpha),
                         std::span<T>(out).subspan(wStart, wEnd - wStart));
            }
        }

        return {std::move(out), std::move(permData)};
    }

    // -----------------------------------------------------------------------
    // Inverse
    // -----------------------------------------------------------------------

    std::vector<T> unreorder(std::span<const T>       reordered,
                             std::span<const uint8_t> permData) override {
        auto [N, Wstored, alphabet] = decodePermData(permData);
        (void)N; (void)Wstored;

        std::vector<T> out(reordered.size());
        if constexpr (W == 0) {
            applyIMTF(reordered, std::span<T>(alphabet), std::span<T>(out));
        } else {
            const size_t sz = reordered.size();
            for (size_t k = 0; k * W < sz; ++k) {
                const size_t wStart = k * W;
                const size_t wEnd   = std::min(wStart + W, sz);
                std::vector<T> localAlpha = alphabet;
                applyIMTF(reordered.subspan(wStart, wEnd - wStart),
                          std::span<T>(localAlpha),
                          std::span<T>(out).subspan(wStart, wEnd - wStart));
            }
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Random access
    // Global MTF: no O(1) path (sequential only) → nullopt.
    // Windowed MTF: replay window containing origIdx (O(W)).
    // -----------------------------------------------------------------------

    std::optional<size_t> originalToReorderedIndex(
        size_t /* origIdx */, std::span<const uint8_t> /* permData */) const override {
        // MTF does not provide a forward index mapping (it's a value transform,
        // and the value at position i in the MTF output is the rank in the
        // current alphabet — not a position in a sorted sequence).
        return std::nullopt;
    }

    // For windowed mode: group by window, replay each window once.
    std::optional<std::vector<size_t>> originalToReorderedIndices(
        std::span<const size_t>  /* origIndices */,
        std::span<const uint8_t> /* permData */) const override {
        return std::nullopt; // MTF is a value transform; decodeAt uses unreorder path
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    size_t estimatePermutationSize(size_t /* N */) const override {
        // Alphabet dominates for large cardinality; small for low-cardinality data
        return 24; // header only; actual depends on cardinality (known at encode time)
    }

    std::string name() const override {
        if constexpr (W == 0)
            return "MTF";
        else
            return "MTF<" + std::to_string(W) + ">";
    }

private:
    // -----------------------------------------------------------------------
    // Core MTF forward pass (operates on caller-supplied alphabet copy)
    // -----------------------------------------------------------------------

    static void applyMTF(std::span<const T> in, std::span<T> alphabet, std::span<T> out) {
        // For large alphabets a hash map gives O(1) rank lookup.
        std::unordered_map<T, size_t> rankOf;
        rankOf.reserve(alphabet.size() * 2);
        for (size_t i = 0; i < alphabet.size(); ++i)
            rankOf[alphabet[i]] = i;

        for (size_t i = 0; i < in.size(); ++i) {
            const T v = in[i];
            const size_t rank = rankOf.at(v);
            out[i] = static_cast<T>(rank);
            // Move v to front: shift [0, rank) right by one
            for (size_t r = rank; r > 0; --r) {
                alphabet[r] = alphabet[r - 1];
                rankOf[alphabet[r]] = r;
            }
            alphabet[0] = v;
            rankOf[v]   = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Core MTF inverse pass
    // -----------------------------------------------------------------------

    static void applyIMTF(std::span<const T> ranks, std::span<T> alphabet, std::span<T> out) {
        for (size_t i = 0; i < ranks.size(); ++i) {
            const size_t rank = static_cast<size_t>(ranks[i]);
            const T v = alphabet[rank];
            out[i] = v;
            for (size_t r = rank; r > 0; --r)
                alphabet[r] = alphabet[r - 1];
            alphabet[0] = v;
        }
    }

    // -----------------------------------------------------------------------
    // Permutation data serialisation
    // -----------------------------------------------------------------------

    static std::vector<uint8_t> encodePermData(size_t N, const std::vector<T>& alphabet) {
        std::vector<uint8_t> blob;
        const size_t alphabetSize = alphabet.size();
        blob.reserve(24 + alphabetSize * sizeof(T));
        auto appendU64 = [&](uint64_t v) {
            for (int b = 0; b < 8; ++b) blob.push_back(static_cast<uint8_t>(v >> (8 * b)));
        };
        appendU64(static_cast<uint64_t>(N));
        appendU64(static_cast<uint64_t>(W));
        appendU64(static_cast<uint64_t>(alphabetSize));
        for (T v : alphabet) {
            using U = std::make_unsigned_t<T>;
            U uv = static_cast<U>(v);
            for (size_t b = 0; b < sizeof(T); ++b)
                blob.push_back(static_cast<uint8_t>(uv >> (8 * b)));
        }
        return blob;
    }

    struct PermDataDecoded { size_t N; size_t Wval; std::vector<T> alphabet; };

    static PermDataDecoded decodePermData(std::span<const uint8_t> blob) {
        auto readU64 = [&](size_t off) -> uint64_t {
            uint64_t v = 0;
            for (int b = 0; b < 8; ++b)
                v |= static_cast<uint64_t>(blob[off + b]) << (8 * b);
            return v;
        };
        const size_t N            = static_cast<size_t>(readU64(0));
        const size_t Wval         = static_cast<size_t>(readU64(8));
        const size_t alphabetSize = static_cast<size_t>(readU64(16));
        std::vector<T> alphabet(alphabetSize);
        for (size_t i = 0; i < alphabetSize; ++i) {
            using U = std::make_unsigned_t<T>;
            U uv = 0;
            for (size_t b = 0; b < sizeof(T); ++b)
                uv |= static_cast<U>(blob[24 + i * sizeof(T) + b]) << (8 * b);
            alphabet[i] = static_cast<T>(uv);
        }
        return {N, Wval, std::move(alphabet)};
    }
};

} // namespace encodings::reorderers
