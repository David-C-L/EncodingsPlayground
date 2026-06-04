#pragma once

// Self-contained windowed Burrows-Wheeler Transform section encoder.
// Does NOT include anything from Source/reorderers/ to avoid circular deps.
// (encodings_encoders ← encodings_reorderers ← encodings_encoders would cycle.)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

// ---------------------------------------------------------------------------
// BWTSectionEncoder<T, W>
//
// Applies windowed cyclic BWT (window size W) to the input, then encodes the
// BWT-transformed sequence with an inner Codec<T, uint8_t> (e.g. Dictionary,
// RLE, BitPacking).
//
// BWT creates long runs of repeated symbols from data with autocorrelation,
// making RLE and Dictionary encoding dramatically more effective.
//
// Wire format:
//   [N:8][numWindows:8][primaryIndex_0:8][primaryIndex_1:8]...[innerEncoded]
//
// T must be an unsigned integer type (uint8_t, uint16_t, uint32_t, uint64_t)
// since section values extracted by SubIntSplitEncoder are always non-negative.
// ---------------------------------------------------------------------------

template <typename T, size_t W = 512>
    requires (std::is_unsigned_v<T> && W >= 2)
class BWTSectionEncoder : public encodings::Codec<T, uint8_t> {
public:
    explicit BWTSectionEncoder(std::shared_ptr<encodings::Codec<T, uint8_t>> innerCodec)
        : inner_(std::move(innerCodec)) {}

    // -----------------------------------------------------------------------
    // Encode: BWT per window → inner codec
    // -----------------------------------------------------------------------

    encodings::EncodedData encode(std::span<const T> data) override {
        const size_t N = data.size();
        if (N == 0) return makeEmpty();

        const size_t numWindows = (N + W - 1) / W;
        std::vector<uint64_t> primaryIndices(numWindows);
        std::vector<T> bwtSeq(N);

        for (size_t k = 0; k < numWindows; ++k) {
            const size_t wStart = k * W;
            const size_t wEnd   = std::min(wStart + W, N);
            auto [wBwt, pidx]   = bwtForward(data.subspan(wStart, wEnd - wStart));
            std::copy(wBwt.begin(), wBwt.end(), bwtSeq.begin() + static_cast<ptrdiff_t>(wStart));
            primaryIndices[k] = static_cast<uint64_t>(pidx);
        }

        // Encode BWT-transformed sequence with inner codec
        encodings::EncodedData innerEncoded = inner_->encode(bwtSeq);

        // Assemble: [N:8][numWindows:8][primaryIndices][innerEncoded bytes]
        std::vector<uint8_t> out;
        const size_t headerSize = 16 + numWindows * 8;
        out.reserve(headerSize + innerEncoded.data().size());
        appendU64(out, static_cast<uint64_t>(N));
        appendU64(out, static_cast<uint64_t>(numWindows));
        for (uint64_t pi : primaryIndices) appendU64(out, pi);
        out.insert(out.end(), innerEncoded.data().begin(), innerEncoded.data().end());

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::typeToDataType<T>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(T);
        meta.supportsRandomAccess = inner_->properties().has(encodings::EncodingProperty::RandomAccess);
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // -----------------------------------------------------------------------
    // Decode all
    // -----------------------------------------------------------------------

    std::vector<T> decodeAll(const encodings::EncodedData& encoded) override {
        const auto [N, numWindows, pidxs, innerEnc] = parseHeader(encoded.data());
        auto bwtSeq = inner_->decodeAll(makeInnerEncoded(innerEnc, N));
        return applyAllInverse(bwtSeq, N, numWindows, pidxs);
    }

    // -----------------------------------------------------------------------
    // Random access: decode only the window containing the requested index,
    // then apply inverse BWT of that window. Falls back to full decodeAll if
    // the inner codec doesn't support decodeRange.
    // -----------------------------------------------------------------------

    std::optional<T> decodeAt(const encodings::EncodedData& encoded, size_t index) override {
        const auto [N, numWindows, pidxs, innerEnc] = parseHeader(encoded.data());
        if (index >= N) return std::nullopt;

        const size_t k      = index / W;
        const size_t wStart = k * W;
        const size_t wEnd   = std::min(wStart + W, N);
        const size_t posInWindow = index - wStart;

        // Try range decode of just this window
        auto innerEncoded = makeInnerEncoded(innerEnc, N);
        auto wBwt = inner_->decodeRange(innerEncoded, wStart, wEnd);
        auto wOrig = bwtInverse(wBwt, static_cast<size_t>(pidxs[k]));
        if (posInWindow < wOrig.size()) return wOrig[posInWindow];
        return std::nullopt;
    }

    std::vector<T> decodeRange(const encodings::EncodedData& encoded,
                               size_t start, size_t end) override {
        const auto [N, numWindows, pidxs, innerEnc] = parseHeader(encoded.data());
        end = std::min(end, N);
        if (start >= end) return {};

        // Decode full sequence and slice — correct for all inner codecs
        auto bwtSeq = inner_->decodeAll(makeInnerEncoded(innerEnc, N));
        auto orig   = applyAllInverse(bwtSeq, N, numWindows, pidxs);
        return {orig.begin() + static_cast<ptrdiff_t>(start),
                orig.begin() + static_cast<ptrdiff_t>(end)};
    }

    // -----------------------------------------------------------------------
    // Codec interface
    // -----------------------------------------------------------------------

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::ReorderingEncoding;
    }

    std::string name() const override {
        return "BWT<" + std::to_string(W) + ">|" + inner_->name();
    }

    encodings::EncodingProperties properties() const override {
        return encodings::EncodingProperties(encodings::EncodingProperty::Lossless)
             | encodings::EncodingProperty::RequiresFullData
             | encodings::EncodingProperty::RandomAccess  // decodeAt works at O(W) cost
             | encodings::EncodingProperty::Composable;
    }

private:
    // -----------------------------------------------------------------------
    // Cyclic BWT forward: O(L² log L) suffix sort — acceptable for L = W = 512
    // -----------------------------------------------------------------------

    static std::pair<std::vector<T>, size_t> bwtForward(std::span<const T> seq) {
        const size_t L = seq.size();
        if (L == 0) return {{}, 0};
        if (L == 1) return {std::vector<T>{seq[0]}, 0};

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

        std::vector<T> bwtOut(L);
        size_t primaryIdx = 0;
        for (size_t j = 0; j < L; ++j) {
            bwtOut[j] = seq[(sa[j] + L - 1) % L];
            if (sa[j] == 0) primaryIdx = j;
        }
        return {std::move(bwtOut), primaryIdx};
    }

    // -----------------------------------------------------------------------
    // Cyclic BWT inverse: LF-mapping reconstruction, O(L log L)
    // -----------------------------------------------------------------------

    static std::vector<T> bwtInverse(std::span<const T> bwt, size_t primaryIdx) {
        const size_t L = bwt.size();
        if (L == 0) return {};
        if (L == 1) return {bwt[0]};

        std::vector<size_t> order(L);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return bwt[a] < bwt[b]; });

        std::vector<size_t> lf(L);
        for (size_t i = 0; i < L; ++i) lf[order[i]] = i;

        std::vector<T> out(L);
        size_t cur = primaryIdx;
        for (size_t i = L; i-- > 0;) {
            out[i] = bwt[cur];
            cur    = lf[cur];
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    std::vector<T> applyAllInverse(const std::vector<T>& bwtSeq, size_t N,
                                   size_t numWindows,
                                   const std::vector<uint64_t>& pidxs) {
        std::vector<T> result(N);
        for (size_t k = 0; k < numWindows; ++k) {
            const size_t wStart = k * W;
            const size_t wEnd   = std::min(wStart + W, N);
            std::span<const T> wBwt(bwtSeq.data() + wStart, wEnd - wStart);
            auto wOrig = bwtInverse(wBwt, static_cast<size_t>(pidxs[k]));
            std::copy(wOrig.begin(), wOrig.end(), result.begin() + static_cast<ptrdiff_t>(wStart));
        }
        return result;
    }

    struct ParsedHeader {
        size_t N;
        size_t numWindows;
        std::vector<uint64_t> primaryIndices;
        std::span<const uint8_t> innerBytes;
    };

    static ParsedHeader parseHeader(const std::vector<uint8_t>& raw) {
        const size_t N          = static_cast<size_t>(readU64(raw.data()));
        const size_t numWindows = static_cast<size_t>(readU64(raw.data() + 8));
        std::vector<uint64_t> pidxs(numWindows);
        for (size_t k = 0; k < numWindows; ++k)
            pidxs[k] = readU64(raw.data() + 16 + k * 8);
        const size_t innerOff = 16 + numWindows * 8;
        return {N, numWindows, std::move(pidxs),
                std::span<const uint8_t>(raw.data() + innerOff, raw.size() - innerOff)};
    }

    encodings::EncodedData makeInnerEncoded(std::span<const uint8_t> bytes, size_t N) const {
        encodings::EncodingMetadata meta;
        meta.encodingName     = inner_->name();
        meta.dataType         = encodings::typeToDataType<T>;
        meta.elementCount     = N;
        meta.compressedSize   = bytes.size();
        meta.uncompressedSize = N * sizeof(T);
        meta.supportsRandomAccess = inner_->properties().has(encodings::EncodingProperty::RandomAccess);
        return encodings::EncodedData(std::vector<uint8_t>(bytes.begin(), bytes.end()),
                                      std::move(meta));
    }

    encodings::EncodedData makeEmpty() const {
        encodings::EncodingMetadata meta;
        meta.encodingName = name();
        meta.elementCount = 0;
        return encodings::EncodedData({}, std::move(meta));
    }

    static void appendU64(std::vector<uint8_t>& v, uint64_t n) {
        for (int b = 0; b < 8; ++b) v.push_back(static_cast<uint8_t>(n >> (8 * b)));
    }
    static uint64_t readU64(const uint8_t* p) {
        uint64_t n = 0;
        for (int b = 0; b < 8; ++b) n |= static_cast<uint64_t>(p[b]) << (8 * b);
        return n;
    }

    std::shared_ptr<encodings::Codec<T, uint8_t>> inner_;
};

} // namespace encodings::encoders
