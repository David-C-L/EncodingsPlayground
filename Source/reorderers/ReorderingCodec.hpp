#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encodings/RowRange.hpp"
#include "Reorderer.hpp"
#include "ReorderingType.hpp"
#include "PermutationStore.hpp"

namespace encodings::reorderers {

// Whether an unpacked forward permutation may stay resident between decode calls.
//
// This is a constructor argument rather than an internal optimisation because it
// changes what a measurement means.  The sequential-access permutation formats
// (DeltaBitPacked, DeltaZstd, DeltaLZ4, ValueGrouped, InverseEliasFano) can only
// answer "where did original row i go?" by unpacking the whole permutation, and
// that O(N) unpack is the cost the format trade-off is about.  Keeping the result
// resident turns it into a one-off amortised over every later call, which is a
// legitimate implementation choice and a misleading benchmark unless it is
// labelled — so a caller must ask for it explicitly, and a driver reports the two
// residencies as separate rows.
enum class PermResidency : uint8_t {
    PerCall,   ///< unpack per call; the honest per-access cost of the format
    Resident,  ///< unpack once and keep; exposed via internalBuffers() so cold-all can cool it
};

constexpr const char* permResidencyToString(PermResidency r) noexcept {
    return r == PermResidency::Resident ? "resident" : "per-call";
}

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
//
// THE *Into OVERRIDES ARE THE MEASUREMENT SURFACE.  decodeAllInto,
// decodeRangeInto and decodeGatherInto were previously not overridden at all, so
// every benchmark of the reordering dimension ran Decoder's allocate-a-vector-
// and-copy fallback and reported that fallback's cost as the cost of reordering.
// They are implemented here, and `bypassIntoOverrides()` keeps the old path
// reachable on the same object so the two can be compared without swapping
// codecs, artifacts or datasets underneath the comparison.
//
// Only Sort and WindowedSort carry a *position* permutation, i.e. one where
// original[i] == reordered[fwd[i]].  GrayCode, MTF, BWT and BitShuffle transform
// values, and their permutation blobs hold alphabets and primary indices rather
// than a PermutationStore blob — so for those the reorderer's own inverse is the
// only correct route and a range or gather is a full-stream inversion.  That is a
// property of the transform, not a gap in this class.
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
    /// Named so bypassIntoOverrides() can reach the fallbacks it replaces.
    using Base = encodings::Decoder<T, uint8_t>;

    ReorderingCodec(std::shared_ptr<Reorderer<T>>  reorderer,
                    std::shared_ptr<InnerCodec>     innerCodec,
                    ReorderingType                  rtype,
                    std::string                     codecName = "",
                    PermResidency                   residency = PermResidency::PerCall)
        : reorderer_(std::move(reorderer))
        , inner_(std::move(innerCodec))
        , rtype_(rtype)
        , name_(codecName.empty()
                    ? reorderer_->name() + " | " + inner_->name()
                    : std::move(codecName))
        , residency_(residency)
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
        auto reordered = inner_->decodeAll(innerEncodedFor(innerBytes, N));

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
                result = inner_->decodeAt(innerEncodedFor(innerBytes, N), j);
            } else {
                const size_t j = PermutationStore::forwardAt(permBlob, index);
                result = inner_->decodeAt(innerEncodedFor(innerBytes, N), j);
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
                const auto& innerEnc = innerEncodedFor(innerBytes, N);
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
    // Caller-owned-buffer decode paths
    //
    // These are what every benchmark driver calls.  Each one writes its result
    // straight into dst: no per-call output vector, and no second copy out of
    // one.  The permutation work each of them does is timed into the same
    // accumulators the allocating entry points use, so a driver can separate
    // "inverting the permutation" from "decoding the reordered values" without
    // instrumenting the driver side.
    // -----------------------------------------------------------------------

    void decodeAllInto(const encodings::EncodedData& encoded, T* dst, size_t n) override {
        if (bypassInto_) return Base::decodeAllInto(encoded, dst, n);

        const auto [N, permBlob, innerBytes] = parseHeader(encoded.data());
        if (n != N)
            throw std::runtime_error("ReorderingCodec::decodeAllInto: expected " +
                                     std::to_string(N) + " elements, got " + std::to_string(n));
        if (N == 0) return;

        innerScratch_.resize(N);
        inner_->decodeAllInto(innerEncodedFor(innerBytes, N), innerScratch_.data(), N);

        if constexpr (EnableProfiling) {
            auto t0 = detail_rc::clock::now();
            unreorderInto(permBlob, N, dst);
            profiling_.unreorderDecodeAllTime_ns = detail_rc::elapsed_ns(t0);
        } else {
            unreorderInto(permBlob, N, dst);
        }
    }

    void decodeRangeInto(const encodings::EncodedData& encoded,
                         size_t start, size_t end,
                         T* dst, size_t n) override {
        if (bypassInto_) return Base::decodeRangeInto(encoded, start, end, dst, n);

        const auto [N, permBlob, innerBytes] = parseHeader(encoded.data());
        end = std::min(end, N);
        const size_t count = start < end ? end - start : 0;
        if (n != count)
            throw std::runtime_error("ReorderingCodec::decodeRangeInto: expected " +
                                     std::to_string(count) + " elements, got " +
                                     std::to_string(n));
        if (count == 0) return;

        // A contiguous run of ORIGINAL rows is a scattered set of inner
        // positions, so the inner codec must be able to seek; without that the
        // only route is to invert the whole stream and slice it.
        if (innerRandomAccess() &&
            mapForward(permBlob, N, count, [start, end](auto&& emit) {
                for (size_t i = start; i < end; ++i) emit(i);
            })) {
            decodeMappedInto(innerEncodedFor(innerBytes, N), dst);
            return;
        }

        const auto all = originalOrder(encoded, N);
        std::copy(all.begin() + static_cast<ptrdiff_t>(start),
                  all.begin() + static_cast<ptrdiff_t>(end), dst);
    }

    void decodeGatherInto(const encodings::EncodedData& encoded,
                          const encodings::RowRangeList& ranges,
                          T* dst, size_t n) override {
        if (bypassInto_) return Base::decodeGatherInto(encoded, ranges, dst, n);

        const auto [N, permBlob, innerBytes] = parseHeader(encoded.data());

        size_t total = 0;
        for (const auto& r : ranges) {
            if (r.end > N)
                throw std::runtime_error("ReorderingCodec::decodeGatherInto: range end " +
                                         std::to_string(r.end) + " past the stream length " +
                                         std::to_string(N));
            total += r.size();
        }
        if (total != n)
            throw std::runtime_error("ReorderingCodec::decodeGatherInto: expected " +
                                     std::to_string(total) + " elements, got " +
                                     std::to_string(n));
        if (total == 0) return;

        if (innerRandomAccess() &&
            mapForward(permBlob, N, total, [&ranges](auto&& emit) {
                for (const auto& r : ranges)
                    for (size_t i = r.begin; i < r.end; ++i) emit(i);
            })) {
            decodeMappedInto(innerEncodedFor(innerBytes, N), dst);
            return;
        }

        // No position mapping (MTF, BWT, GrayCode) or a sequential inner codec:
        // one full inversion serves every range, which is strictly better than
        // the base class's one full inversion PER range.
        const auto all = originalOrder(encoded, N);
        size_t off = 0;
        for (const auto& r : ranges) {
            std::copy(all.begin() + static_cast<ptrdiff_t>(r.begin),
                      all.begin() + static_cast<ptrdiff_t>(r.end), dst + off);
            off += r.size();
        }
    }

    // -----------------------------------------------------------------------
    // Cold-state support
    // -----------------------------------------------------------------------

    /// Decoder-owned memory holding already-decoded state, for --cache-state
    /// cold-all.
    ///
    /// Three kinds of thing live here, and flushing the encoded payload cools
    /// none of them: a resident unpacked permutation (the object of study for the
    /// sequential formats), the reordered-value and original-order scratch
    /// buffers, and the copy of the inner section bytes that this class has to
    /// keep because Codec's decode entry points take an owning EncodedBuffer.
    /// That last one is the same trap FINDINGS section 6 records for
    /// SubIntSplit's slice(): a payload-only flush left everything actually read
    /// resident.  The inner codec's own buffers are appended so a cold-all row
    /// covers the whole chain rather than only its outermost layer.
    std::vector<std::span<const std::byte>> internalBuffers() const override {
        std::vector<std::span<const std::byte>> out;
        const auto add = [&out](const auto& container) {
            if (container.empty()) return;
            out.push_back(std::as_bytes(
                std::span<const typename std::decay_t<decltype(container)>::value_type>(
                    container.data(), container.size())));
        };
        add(perm_);
        add(innerScratch_);
        add(origScratch_);
        add(mapped_);
        add(innerEncoded_.data());
        const auto innerBufs = inner_->internalBuffers();
        out.insert(out.end(), innerBufs.begin(), innerBufs.end());
        return out;
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
    /// Also drops every cached decode artifact: the cached inner EncodedData is
    /// keyed on the address and length of the section bytes it was built from, and
    /// a freed payload can be replaced by a different one of the same length at
    /// the same address.  reset() is called before each encode, which is the
    /// boundary at which that can happen.
    void reset() override {
        inner_->reset();
        permValid_ = false;
        innerKeyPtr_ = nullptr;
        innerKeySize_ = 0;
        innerKeyN_ = 0;
    }

    void resetReorderingProfilingAccum() override {
        if constexpr (EnableProfiling) {
            profiling_.permLookupDecodeAtAccum_ns    = 0;
            profiling_.permLookupDecodeRangeAccum_ns = 0;
        }
    }

    // -----------------------------------------------------------------------
    // ReorderingCodec-specific accessors
    // -----------------------------------------------------------------------

    ReorderingType reorderingType() const noexcept { return rtype_; }

    PermResidency permResidency() const noexcept { return residency_; }

    /// Bytes of unpacked permutation currently resident, i.e. what cold-all has
    /// to cool for this codec.  Zero under PermResidency::PerCall between calls
    /// only in the sense that the values are recomputed anyway — the buffer is
    /// still reported by internalBuffers(), because it is still hot.
    size_t residentPermutationBytes() const noexcept { return perm_.size() * sizeof(size_t); }

    /// Route the *Into entry points back through Decoder's allocate-and-copy
    /// fallback.
    ///
    /// Exists so the overrides above can be measured against the path they
    /// replaced on ONE object, with one artifact and one dataset: a separate
    /// wrapper codec would have re-encoded and moved the payload, which is the
    /// mistake FINDINGS section 3 records — an apparent 2x that turned out to be
    /// a difference between two harnesses rather than two code paths.
    void bypassIntoOverrides(bool bypass) noexcept { bypassInto_ = bypass; }
    bool intoOverridesBypassed() const noexcept { return bypassInto_; }

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
        if (ip.has(encodings::EncodingProperty::RandomAccess)) {
            p |= encodings::EncodingProperty::RandomAccess;
            // FastSkip means "this codec has a real decodeGatherInto override".
            // It does, but only where a position permutation plus a seekable inner
            // codec let a gather touch the selected rows alone; for a value
            // transform the override still exists and still helps (one inversion
            // for the whole range list instead of one per range), yet it reads the
            // entire stream, so claiming FastSkip there would overstate it.
            if (positionPermuted()) p |= encodings::EncodingProperty::FastSkip;
        }
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

    /// The inner section bytes as an EncodedData, built at most once per artifact.
    ///
    /// Codec's decode entry points take an owning EncodedBuffer, so handing the
    /// inner codec its slice means copying it.  Doing that per call put a copy of
    /// the entire payload inside every decodeAt — a point lookup then cost O(N)
    /// and reported the copy, not the lookup.  The copy is therefore kept, keyed
    /// on the address, length and element count of the bytes it was made from,
    /// and dropped by reset(); it is reported by internalBuffers() so a cold-all
    /// row cools it rather than silently measuring it warm.
    const encodings::EncodedData& innerEncodedFor(std::span<const uint8_t> bytes, size_t N) {
        if (innerKeyPtr_ != bytes.data() || innerKeySize_ != bytes.size() || innerKeyN_ != N) {
            innerEncoded_ = makeInnerEncoded(bytes, N);
            innerKeyPtr_  = bytes.data();
            innerKeySize_ = bytes.size();
            innerKeyN_    = N;
        }
        return innerEncoded_;
    }

    bool innerRandomAccess() const {
        return inner_->properties().has(encodings::EncodingProperty::RandomAccess);
    }

    /// True when the permutation blob is a PermutationStore position permutation,
    /// i.e. when original[i] == reordered[fwd[i]] holds.
    ///
    /// Decided from the reordering type, not from the blob: MTF stores an
    /// alphabet and BWT stores per-window primary indices, both starting with a
    /// little-endian length whose first byte would be read as a PermFormat tag by
    /// any blob-sniffing test.  GrayCode is excluded even though its blob is a
    /// valid (empty) PermutationStore blob, because it rewrites values in place
    /// and an identity mapping would return the transformed value.
    bool positionPermuted() const noexcept {
        return rtype_ == ReorderingType::Sort || rtype_ == ReorderingType::WindowedSort;
    }

    /// O(1) per-element mapping, without unpacking anything.
    ///
    /// Both conditions are needed: the reordering type decides whether byte 0 may
    /// be read as a format tag at all, and the format decides whether forwardAt
    /// is a bit extraction or a full unpack in disguise — PermutationStore's
    /// forwardAt falls back to unpackForward()[i] for the sequential formats, so
    /// using it per element there would unpack the whole permutation once per
    /// element.
    bool pointMappable(std::span<const uint8_t> permBlob) const noexcept {
        return positionPermuted() && PermutationStore::supportsRandomAccess(permBlob);
    }

    /// The full forward permutation, or an empty span when this reorderer has none.
    ///
    /// Under PermResidency::PerCall this unpacks on every call, which for the
    /// sequential formats is an O(N) cost per access and is exactly the quantity
    /// the format comparison is about.  A size that disagrees with N means the
    /// blob is not a full position permutation after all (GrayCode's None blob is
    /// the case that reaches here), and the caller falls back to the reorderer's
    /// own inverse rather than trusting a short mapping.
    std::span<const size_t> forwardPermutation(std::span<const uint8_t> permBlob, size_t N) {
        if (!positionPermuted()) return {};
        if (residency_ == PermResidency::Resident && permValid_ && perm_.size() == N) return perm_;

        perm_ = PermutationStore::unpackForward(permBlob);
        if (perm_.size() != N) {
            permValid_ = false;
            return {};
        }
        permValid_ = (residency_ == PermResidency::Resident);
        return perm_;
    }

    /// Inverts innerScratch_ (reordered order) into dst (original order).
    void unreorderInto(std::span<const uint8_t> permBlob, size_t N, T* dst) {
        const auto fwd = forwardPermutation(permBlob, N);
        if (!fwd.empty()) {
            for (size_t i = 0; i < N; ++i) dst[i] = innerScratch_[fwd[i]];
            return;
        }
        // Value transforms own their inverse and return it by value; there is no
        // buffer-filling entry point on Reorderer to write into dst directly.
        auto out = reorderer_->unreorder(std::span<const T>(innerScratch_), permBlob);
        if (out.size() != N)
            throw std::runtime_error("ReorderingCodec: unreorder produced " +
                                     std::to_string(out.size()) + " of " + std::to_string(N) +
                                     " elements");
        std::copy(out.begin(), out.end(), dst);
    }

    /// Fills mapped_ with the inner positions of the original rows `each` emits,
    /// in output order.  False means this reorderer exposes no position mapping.
    ///
    /// The mapping is a separate pass from the decode rather than fused with it so
    /// that its cost is separable: perm_lookup_decode_range_ns is the column that
    /// says how much of a selective read went on inverting the permutation, and a
    /// fused loop could only be timed as a whole.
    template <typename EachIndex>
    bool mapForward(std::span<const uint8_t> permBlob, size_t N, size_t total, EachIndex&& each) {
        int64_t elapsed = 0;
        const auto t0 = detail_rc::clock::now();

        mapped_.clear();
        mapped_.reserve(total);
        bool ok = true;
        if (pointMappable(permBlob)) {
            each([&](size_t i) { mapped_.push_back(PermutationStore::forwardAt(permBlob, i)); });
        } else {
            const auto fwd = forwardPermutation(permBlob, N);
            if (fwd.empty()) {
                ok = false;
            } else {
                each([&](size_t i) { mapped_.push_back(fwd[i]); });
            }
        }

        if constexpr (EnableProfiling) {
            elapsed = detail_rc::elapsed_ns(t0);
            profiling_.permLookupDecodeRangeAccum_ns += elapsed;
        }
        (void)elapsed;
        return ok;
    }

    /// One inner random access per mapped position.
    void decodeMappedInto(const encodings::EncodedData& innerEnc, T* dst) {
        for (size_t k = 0; k < mapped_.size(); ++k) {
            const auto v = inner_->decodeAt(innerEnc, mapped_[k]);
            // A mapped position out of range means the permutation and the
            // payload disagree; silently leaving dst[k] at whatever it held would
            // surface as a plausible-looking wrong value in a result table.
            if (!v)
                throw std::runtime_error("ReorderingCodec: inner codec has no element at "
                                         "reordered position " + std::to_string(mapped_[k]));
            dst[k] = *v;
        }
    }

    /// The whole stream in original order, in a reused buffer.
    std::span<const T> originalOrder(const encodings::EncodedData& encoded, size_t N) {
        origScratch_.resize(N);
        decodeAllInto(encoded, origScratch_.data(), N);
        return origScratch_;
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
    PermResidency                 residency_{PermResidency::PerCall};
    bool                          bypassInto_{false};

    // Decode-time scratch, reused across calls so a per-call allocation is not
    // charged to the codec, and enumerated by internalBuffers() so a cold-all row
    // does not measure them warm.
    std::vector<size_t>    perm_;          ///< unpacked forward permutation
    bool                   permValid_{false};  ///< perm_ may be reused (Resident only)
    std::vector<T>         innerScratch_;  ///< values in reordered order
    std::vector<T>         origScratch_;   ///< values in original order
    std::vector<size_t>    mapped_;        ///< inner positions of the requested rows
    encodings::EncodedData innerEncoded_;  ///< the inner section bytes, copied once
    const uint8_t*         innerKeyPtr_{nullptr};
    size_t                 innerKeySize_{0};
    size_t                 innerKeyN_{0};
};

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

template <ReorderableType T, bool EnableProfiling = false>
std::shared_ptr<ReorderingCodec<T, EnableProfiling>>
makeReorderingCodec(std::shared_ptr<Reorderer<T>>                    reorderer,
                    std::shared_ptr<encodings::Codec<T, uint8_t>>    innerCodec,
                    ReorderingType                                    rtype,
                    std::string                                       name = "",
                    PermResidency                                     residency =
                        PermResidency::PerCall) {
    return std::make_shared<ReorderingCodec<T, EnableProfiling>>(
        std::move(reorderer), std::move(innerCodec), rtype, std::move(name), residency);
}

} // namespace encodings::reorderers
