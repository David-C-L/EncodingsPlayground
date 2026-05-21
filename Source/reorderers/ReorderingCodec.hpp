#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "Reorderer.hpp"
#include "ReorderingType.hpp"
#include "PermutationStore.hpp"

namespace encodings::reorderers {

// ---------------------------------------------------------------------------
// ReorderingCodec<T, EnableProfiling>
//
// A Codec<T, uint8_t> that wraps any Reorderer<T> and any inner
// Codec<T, uint8_t>.  Encode path:
//   1. reorder data → reorderedValues + permBlob
//   2. innerCodec->encode(reorderedValues) → innerEncoded
//   3. assemble: [N:8][permSize:8][permBlob][innerEncoded]
//
// Decode path mirrors this exactly.  Random access (decodeAt) uses the
// stored permutation to map original index i → reordered index j, then
// delegates to innerCodec->decodeAt(j) if the inner codec supports it.
//
// EnableProfiling = true: times the reordering/unreordering sub-steps and
// accumulates permutation-lookup latency across repeated decodeAt/decodeRange
// calls.  Zero overhead when false (via [[no_unique_address]]).
// ---------------------------------------------------------------------------

namespace detail_rc {
using clock = std::chrono::steady_clock;
inline int64_t elapsed_ns(clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count();
}
} // namespace detail_rc

template <ReorderableType T, bool EnableProfiling = false>
class ReorderingCodec : public encodings::Codec<T, uint8_t> {
public:
    using InnerCodec = encodings::Codec<T, uint8_t>;

    ReorderingCodec(std::shared_ptr<Reorderer<T>>  reorderer,
                    std::shared_ptr<InnerCodec>     innerCodec,
                    ReorderingType                  rtype,
                    std::string                     codecName = "")
        : reorderer_(std::move(reorderer))
        , inner_(std::move(innerCodec))
        , rtype_(rtype)
        , name_(codecName.empty()
                    ? reorderer_->name() + " | " + inner_->name()
                    : std::move(codecName))
    {}

    // -----------------------------------------------------------------------
    // Encode
    // -----------------------------------------------------------------------

    encodings::EncodedData encode(std::span<const T> data) override {
        const size_t N = data.size();

        // Step 1: reorder (timed when profiling)
        ReorderResult<T> reordered;
        if constexpr (EnableProfiling) {
            auto t0 = detail_rc::clock::now();
            reordered = reorderer_->reorder(data);
            profiling_.reorderEncodeTime_ns = detail_rc::elapsed_ns(t0);
        } else {
            reordered = reorderer_->reorder(data);
        }
        auto& permBlob       = reordered.permutationData;
        auto& reorderedVals  = reordered.reorderedValues;

        // Step 2: encode reordered values with inner codec
        encodings::EncodedData innerEncoded = inner_->encode(reorderedVals);

        // Step 3: assemble [N:8][permSize:8][permBlob][innerBytes]
        const uint64_t permSize  = permBlob.size();
        const size_t   innerSize = innerEncoded.data().size();

        std::vector<uint8_t> out;
        out.reserve(16 + permSize + innerSize);
        auto appendU64 = [&](uint64_t v) {
            for (int b = 0; b < 8; ++b) out.push_back(static_cast<uint8_t>(v >> (8 * b)));
        };
        appendU64(static_cast<uint64_t>(N));
        appendU64(permSize);
        out.insert(out.end(), permBlob.begin(), permBlob.end());
        out.insert(out.end(), innerEncoded.data().begin(), innerEncoded.data().end());

        // Build metadata
        encodings::EncodingMetadata meta;
        meta.encodingName         = name_;
        meta.dataType             = encodings::typeToDataType<T>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(T);
        meta.supportsRandomAccess = inner_->properties().has(encodings::EncodingProperty::RandomAccess)
                                 && PermutationStore::supportsRandomAccess(permBlob);
        meta.customMetadata["reordering_type"]              = reorderingTypeToString(rtype_);
        meta.customMetadata["inner_codec"]                  = inner_->name();
        meta.customMetadata["permutation_bytes"]            = std::to_string(permBlob.size());
        meta.customMetadata["permutation_pct_of_encoded"]   =
            std::to_string(out.size() > 0 ? permBlob.size() * 100.0 / out.size() : 0.0);
        meta.customMetadata["permutation_pct_of_uncompressed"] =
            std::to_string(N > 0 ? permBlob.size() * 100.0 / (N * sizeof(T)) : 0.0);
        if constexpr (EnableProfiling) {
            meta.customMetadata["reorder_encode_time_ns"] =
                std::to_string(profiling_.reorderEncodeTime_ns);
        }

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // -----------------------------------------------------------------------
    // Decode all
    // -----------------------------------------------------------------------

    std::vector<T> decodeAll(const encodings::EncodedData& encoded) override {
        const auto [N, permBlob, innerBytes] = parseHeader(encoded.data());
        auto reordered = inner_->decodeAll(makeInnerEncoded(innerBytes, N));

        if constexpr (EnableProfiling) {
            auto t0 = detail_rc::clock::now();
            auto result = reorderer_->unreorder(reordered, permBlob);
            profiling_.unreorderDecodeAllTime_ns = detail_rc::elapsed_ns(t0);
            return result;
        } else {
            return reorderer_->unreorder(reordered, permBlob);
        }
    }

    // -----------------------------------------------------------------------
    // Random access at original index i
    // -----------------------------------------------------------------------

    std::optional<T> decodeAt(const encodings::EncodedData& encoded, size_t index) override {
        const auto [N, permBlob, innerBytes] = parseHeader(encoded.data());
        if (index >= N) return std::nullopt;

        // Try O(1) path via permutation lookup + inner random access
        if (PermutationStore::supportsRandomAccess(permBlob) &&
            inner_->properties().has(encodings::EncodingProperty::RandomAccess)) {
            std::optional<T> result;
            if constexpr (EnableProfiling) {
                auto t0 = detail_rc::clock::now();
                const size_t j = PermutationStore::forwardAt(permBlob, index);
                profiling_.permLookupDecodeAtAccum_ns += detail_rc::elapsed_ns(t0);
                result = inner_->decodeAt(makeInnerEncoded(innerBytes, N), j);
            } else {
                const size_t j = PermutationStore::forwardAt(permBlob, index);
                result = inner_->decodeAt(makeInnerEncoded(innerBytes, N), j);
            }
            return result;
        }

        // Fallback: full decode
        auto all = decodeAll(encoded);
        return index < all.size() ? std::optional<T>{all[index]} : std::nullopt;
    }

    // -----------------------------------------------------------------------
    // Range decode
    // -----------------------------------------------------------------------

    std::vector<T> decodeRange(const encodings::EncodedData& encoded,
                               size_t start, size_t end) override {
        const auto [N, permBlob, innerBytes] = parseHeader(encoded.data());
        end = std::min(end, N);
        if (start >= end) return {};

        const bool innerRA = inner_->properties().has(encodings::EncodingProperty::RandomAccess);
        if (innerRA) {
            std::vector<size_t> origIndices;
            origIndices.reserve(end - start);
            for (size_t i = start; i < end; ++i) origIndices.push_back(i);

            std::optional<std::vector<size_t>> reorderedIndices;
            if constexpr (EnableProfiling) {
                auto t0 = detail_rc::clock::now();
                reorderedIndices = reorderer_->originalToReorderedIndices(origIndices, permBlob);
                profiling_.permLookupDecodeRangeAccum_ns += detail_rc::elapsed_ns(t0);
            } else {
                reorderedIndices = reorderer_->originalToReorderedIndices(origIndices, permBlob);
            }

            if (reorderedIndices) {
                auto innerEnc = makeInnerEncoded(innerBytes, N);
                std::vector<T> result(end - start);
                for (size_t i = 0; i < result.size(); ++i) {
                    auto v = inner_->decodeAt(innerEnc, (*reorderedIndices)[i]);
                    if (v) result[i] = *v;
                }
                return result;
            }
        }

        // Fallback: full decode then slice
        auto all = decodeAll(encoded);
        return {all.begin() + static_cast<ptrdiff_t>(start),
                all.begin() + static_cast<ptrdiff_t>(end)};
    }

    // -----------------------------------------------------------------------
    // Profiling virtual hook overrides
    // -----------------------------------------------------------------------

    int64_t reorderEncodeTimeNs() const override {
        if constexpr (EnableProfiling) return profiling_.reorderEncodeTime_ns;
        return -1;
    }
    int64_t unreorderDecodeAllTimeNs() const override {
        if constexpr (EnableProfiling) return profiling_.unreorderDecodeAllTime_ns;
        return -1;
    }
    int64_t permLookupDecodeAtAccumNs() const override {
        if constexpr (EnableProfiling) return profiling_.permLookupDecodeAtAccum_ns;
        return -1;
    }
    int64_t permLookupDecodeRangeAccumNs() const override {
        if constexpr (EnableProfiling) return profiling_.permLookupDecodeRangeAccum_ns;
        return -1;
    }
    void resetReorderingProfilingAccum() override {
        if constexpr (EnableProfiling) {
            profiling_.permLookupDecodeAtAccum_ns    = 0;
            profiling_.permLookupDecodeRangeAccum_ns = 0;
        }
    }

    // -----------------------------------------------------------------------
    // ReorderingCodec-specific accessor
    // -----------------------------------------------------------------------

    ReorderingType reorderingType() const noexcept { return rtype_; }

    // -----------------------------------------------------------------------
    // Codec<T> interface
    // -----------------------------------------------------------------------

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::ReorderingEncoding;
    }

    std::string name() const override { return name_; }

    encodings::EncodingProperties properties() const override {
        const auto ip = inner_->properties();
        encodings::EncodingProperties p;
        p |= encodings::EncodingProperty::Lossless;
        p |= encodings::EncodingProperty::ReordersData;
        p |= encodings::EncodingProperty::RequiresFullData;
        p |= encodings::EncodingProperty::Composable;
        if (ip.has(encodings::EncodingProperty::RandomAccess))
            p |= encodings::EncodingProperty::RandomAccess;
        if (ip.has(encodings::EncodingProperty::Vectorizable))
            p |= encodings::EncodingProperty::Vectorizable;
        return p;
    }

    size_t estimateEncodedSize(size_t n) const override {
        return 16 + reorderer_->estimatePermutationSize(n) + inner_->estimateEncodedSize(n);
    }

private:
    // -----------------------------------------------------------------------
    // Header parsing: returns (N, permBlob span, innerBytes span)
    // -----------------------------------------------------------------------

    struct ParsedHeader {
        size_t N;
        std::span<const uint8_t> permBlob;
        std::span<const uint8_t> innerBytes;
    };

    static ParsedHeader parseHeader(const std::vector<uint8_t>& raw) {
        uint64_t N = 0, permSize = 0;
        for (int b = 0; b < 8; ++b) N        |= static_cast<uint64_t>(raw[b])     << (8 * b);
        for (int b = 0; b < 8; ++b) permSize  |= static_cast<uint64_t>(raw[8 + b]) << (8 * b);
        const size_t permOff  = 16;
        const size_t innerOff = permOff + static_cast<size_t>(permSize);
        return {
            static_cast<size_t>(N),
            std::span<const uint8_t>(raw.data() + permOff,  static_cast<size_t>(permSize)),
            std::span<const uint8_t>(raw.data() + innerOff, raw.size() - innerOff),
        };
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

    // -----------------------------------------------------------------------
    // Profiling state (zero-size when EnableProfiling = false)
    // -----------------------------------------------------------------------

    struct NoProfiling {};
    struct YesProfiling {
        mutable int64_t reorderEncodeTime_ns          = 0;
        mutable int64_t unreorderDecodeAllTime_ns     = 0;
        mutable int64_t permLookupDecodeAtAccum_ns    = 0;
        mutable int64_t permLookupDecodeRangeAccum_ns = 0;
    };
    [[no_unique_address]] std::conditional_t<EnableProfiling, YesProfiling, NoProfiling> profiling_;

    std::shared_ptr<Reorderer<T>> reorderer_;
    std::shared_ptr<InnerCodec>   inner_;
    ReorderingType                rtype_;
    std::string                   name_;
};

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

template <ReorderableType T, bool EnableProfiling = false>
std::shared_ptr<ReorderingCodec<T, EnableProfiling>>
makeReorderingCodec(std::shared_ptr<Reorderer<T>>                    reorderer,
                    std::shared_ptr<encodings::Codec<T, uint8_t>>    innerCodec,
                    ReorderingType                                    rtype,
                    std::string                                       name = "") {
    return std::make_shared<ReorderingCodec<T, EnableProfiling>>(
        std::move(reorderer), std::move(innerCodec), rtype, std::move(name));
}

} // namespace encodings::reorderers
