#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "benchmark/targets/BenchTarget.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/RowRange.hpp"

namespace encodings::benchmark {

namespace detail {

/// Whether this repo's Codec<T> has grown an internalBuffers() method yet.  It is
/// being added on a separate branch, and this header must compile on both sides of
/// that merge — so the capability is detected rather than assumed.
template <typename C>
concept HasInternalBuffers = requires(const C& c) { c.internalBuffers(); };

}  // namespace detail

/**
 * @brief BenchTargetC over this repo's Codec<T>: a pass-through, nothing more.
 *
 * Every method is a single inline forwarding call with no branch, no virtual of
 * its own and no allocation, because the adapter sits inside the timed region of
 * every decode driver.  The one indirection present is the codec's own virtual
 * dispatch, which the driver would pay anyway when calling the codec directly —
 * so a measurement taken through this target and one taken through the codec must
 * agree within noise, and the verification TU checks exactly that.
 *
 * The target owns the encoded artifact and borrows the codec.  That split follows
 * from CONVENTIONS section 2: the artifact is produced once and then measured
 * many times, while the codec is a registry-owned object shared across the sweep.
 */
template <typename T>
class PlaygroundTarget {
public:
    using Elem = T;

    explicit PlaygroundTarget(Codec<T>& codec) : codec_(&codec) {}

    std::string name() const { return codec_->name(); }

    /// Produces and retains the artifact all subsequent calls decode from.
    /// Returns a reference rather than a copy so a driver can read metadata
    /// (per-section byte counts, selected plan) without re-encoding.
    const EncodedBuffer<uint8_t>& encode(std::span<const T> src) {
        encoded_ = codec_->encode(src);
        return encoded_;
    }

    /// Adopt an artifact produced elsewhere — ArtifactCache encodes once per
    /// (encoder, dataset, N) and hands the result to every decode driver.
    void adopt(EncodedBuffer<uint8_t> encoded) { encoded_ = std::move(encoded); }

    std::span<const std::byte> payloadBytes() const {
        return std::as_bytes(std::span<const uint8_t>(encoded_.data()));
    }

    const EncodedBuffer<uint8_t>& artifact() const { return encoded_; }

    inline void materializeAll(T* dst, size_t n) { codec_->decodeAllInto(encoded_, dst, n); }

    inline void materializeRange(size_t begin, size_t end, T* dst, size_t n) {
        codec_->decodeRangeInto(encoded_, begin, end, dst, n);
    }

    inline void skipThenMaterialize(const RowRangeList& ranges, T* dst, size_t n) {
        codec_->decodeGatherInto(encoded_, ranges, dst, n);
    }

    /// Native here: decodeAt is a real random-access entry point, so rows from
    /// this target are not labelled emulated.  Returns the optional unchanged —
    /// a driver must consume it via clobber() rather than discard it, or the
    /// lookup is dead code the optimizer is entitled to delete.
    inline std::optional<T> pointRead(size_t index) { return codec_->decodeAt(encoded_, index); }

    TargetCaps capabilities() const {
        const EncodingProperties props = codec_->properties();
        TargetCaps c;
        c.randomAccess   = props.has(EncodingProperty::RandomAccess);
        c.fastSkip       = props.has(EncodingProperty::FastSkip);
        c.nativePointRead = true;
        // FastSkip is defined as "this codec provides a genuine decodeGatherInto
        // override", so it is the codec's own declaration and can be read
        // directly.  decodeRangeInto has no corresponding declared property and
        // the base class supplies a working fallback, so overridesRangeInto
        // cannot be inferred from the codec at all: it defaults to false meaning
        // NOT DECLARED, and only the registry may assert otherwise via
        // declareRangeIntoOverride().  A driver must therefore not read `false`
        // here as proof that the fallback is in use.
        c.overridesGather     = c.fastSkip;
        c.overridesRangeInto  = rangeIntoOverride_;
        return c;
    }

    void declareRangeIntoOverride(bool declared) { rangeIntoOverride_ = declared; }

    /// Zeroes every accumulator the codec exposes, not only the one the calling
    /// driver reads.  A stale gather counter surviving into a point-lookup cell
    /// would be reported as that cell's skip time, which is worse than reporting
    /// nothing.
    inline void resetProfiling() {
        codec_->resetGatherProfilingAccum();
        codec_->resetSubStreamDecodeAtAccum();
        codec_->resetSubStreamDecodeRangeAccum();
        codec_->resetReorderingProfilingAccum();
    }

    /// The -1 and empty-vector defaults come straight from Codec's own defaults,
    /// which already mean "not reported" — they are passed through rather than
    /// normalized so the null survives to the result row.
    TargetProfile profile() const {
        TargetProfile p;
        p.gatherSkipNs        = codec_->gatherSkipTimeNs();
        p.gatherMaterializeNs = codec_->gatherMaterializeTimeNs();
        p.subStreamBulkNs     = codec_->subStreamBulkDecodeTimeNs();
        p.subStreamPointNs    = codec_->subStreamDecodeAtAccumNs();
        p.subStreamRangeNs    = codec_->subStreamDecodeRangeAccumNs();
        return p;
    }

    /// Empty when the codec cannot enumerate its internal structures, which
    /// includes the case where Codec<T> does not yet have the method at all.
    /// Either way the driver's cold-all path must fall back to an LLC thrash and
    /// label the row accordingly rather than claim a clean cold measurement.
    auto internalBuffers() const {
        if constexpr (detail::HasInternalBuffers<Codec<T>>) {
            return codec_->internalBuffers();
        } else {
            return std::vector<std::span<const std::byte>>{};
        }
    }

    /// The escape hatch. Cost-model estimates, index-type selection and
    /// sub-stream layout are all reached through here, deliberately outside the
    /// concept — see BenchTarget.hpp (b).
    Codec<T>& native() { return *codec_; }
    const Codec<T>& native() const { return *codec_; }

private:
    Codec<T>*              codec_;
    EncodedBuffer<uint8_t> encoded_{};
    bool                   rangeIntoOverride_{false};
};

static_assert(BenchTargetC<PlaygroundTarget<uint64_t>>);
static_assert(BenchTargetC<PlaygroundTarget<int64_t>>);
static_assert(BenchTargetC<PlaygroundTarget<uint32_t>>);

}  // namespace encodings::benchmark
