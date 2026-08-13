#pragma once

// A codec wrapper that turns off one fast path and forwards everything else.
//
// The ladders in registry/CodecSetLadder.hpp hold the decode paths fixed and vary
// the plan.  This is the complement: it holds the plan fixed — the same artifact,
// the same sections, the same bytes — and removes one decode path, so the
// difference between the wrapped and unwrapped run is that path's contribution
// and nothing else.  Two paths are separable this way:
//
//   Disable::GatherFastPath  decodeGatherInto() runs the base-class
//                            per-range decodeRangeInto() loop instead of the
//                            codec's skip-then-materialize override, and
//                            properties() drops FastSkip so a driver labels the
//                            row as the fallback it is.
//   Disable::RangeInto       decodeRangeInto() runs decodeRange() plus a copy,
//                            i.e. exactly the Decoder base-class fallback, which
//                            measures the cost of the caller-owned-buffer
//                            contract for a codec that overrides it.
//
// Doing this by wrapping rather than by editing the codec is what makes the A/B
// valid: an #ifdef or a runtime flag inside the codec changes its inlining and
// its register pressure whether the flag is set or not, so the "fast path on"
// arm would no longer be the codec that ships.  Here the fast-path arm is the
// unwrapped codec, untouched.
//
// The wrapper adds one virtual call per decode to the disabled path only.  For a
// bulk or point read it forwards through a single inline hop, so a measurement of
// a path this wrapper does not disable is comparable with an unwrapped one.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encodings/RowRange.hpp"

namespace encodings::benchmark {

template <typename T>
class ForceFallbackCodec final : public encodings::Codec<T, uint8_t> {
public:
    /// Bit flags rather than an enum of modes: the two paths are independent and
    /// a driver that disables both is measuring a codec reduced entirely to its
    /// allocating decodeRange(), which is a meaningful third arm.
    enum class Disable : uint8_t { GatherFastPath = 1, RangeInto = 2 };

    ForceFallbackCodec(std::shared_ptr<encodings::Codec<T, uint8_t>> inner, uint8_t disabled)
        : inner_(std::move(inner)), disabled_(disabled) {
        if (!inner_) throw std::invalid_argument("ForceFallbackCodec: inner must not be null");
    }

    ForceFallbackCodec(std::shared_ptr<encodings::Codec<T, uint8_t>> inner, Disable one)
        : ForceFallbackCodec(std::move(inner), static_cast<uint8_t>(one)) {}

    bool disabled(Disable d) const { return (disabled_ & static_cast<uint8_t>(d)) != 0; }

    /// Human-readable disable set, for the `variant` column.  Empty string when
    /// nothing is disabled, so a pass-through wrapper is not mistaken for an arm.
    static std::string describe(uint8_t disabled) {
        std::string out;
        if ((disabled & static_cast<uint8_t>(Disable::GatherFastPath)) != 0) out += "no_gather";
        if ((disabled & static_cast<uint8_t>(Disable::RangeInto)) != 0) {
            if (!out.empty()) out += "+";
            out += "no_range_into";
        }
        return out;
    }

    EncodedBuffer<uint8_t> encode(std::span<const T> data) override {
        return inner_->encode(data);
    }

    std::vector<T> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        return inner_->decodeAll(encoded);
    }

    std::optional<T> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        return inner_->decodeAt(encoded, index);
    }

    std::vector<T> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start,
                               size_t end) override {
        return inner_->decodeRange(encoded, start, end);
    }

    void decodeAllInto(const EncodedBuffer<uint8_t>& encoded, T* dst, size_t n) override {
        inner_->decodeAllInto(encoded, dst, n);
    }

    /// With RangeInto disabled this is verbatim Decoder<T>::decodeRangeInto — the
    /// allocating decodeRange() and a copy — reached without inheriting it, since
    /// the inner codec's override would otherwise win through the forward.
    void decodeRangeInto(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end, T* dst,
                         size_t n) override {
        if (!disabled(Disable::RangeInto)) {
            inner_->decodeRangeInto(encoded, start, end, dst, n);
            return;
        }
        std::vector<T> vals = inner_->decodeRange(encoded, start, end);
        if (vals.size() != n)
            throw std::runtime_error("ForceFallbackCodec::decodeRangeInto: decoded size mismatch");
        std::copy(vals.begin(), vals.end(), dst);
    }

    /// With GatherFastPath disabled this is verbatim
    /// Decoder<T>::decodeGatherInto's per-range loop, but dispatched through
    /// THIS object's decodeRangeInto — so combining both flags composes, and a
    /// gather measured with only GatherFastPath off still uses the codec's own
    /// range path.
    void decodeGatherInto(const EncodedBuffer<uint8_t>& encoded, const RowRangeList& ranges,
                          T* dst, size_t n) override {
        if (!disabled(Disable::GatherFastPath)) {
            inner_->decodeGatherInto(encoded, ranges, dst, n);
            return;
        }
        size_t off = 0;
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            decodeRangeInto(encoded, r.begin, r.end, dst + off, count);
            off += count;
        }
        if (off != n)
            throw std::runtime_error("ForceFallbackCodec::decodeGatherInto: decoded size mismatch");
    }

    EncodingType encodingType() const override { return inner_->encodingType(); }

    std::string name() const override {
        const std::string suffix = describe(disabled_);
        return suffix.empty() ? inner_->name() : inner_->name() + "[" + suffix + "]";
    }

    /// FastSkip means "this codec provides a genuine decodeGatherInto override"
    /// (EncodingProperty.hpp).  Once that override is bypassed the claim is false,
    /// and leaving it set would let a driver's `fast_skip` column contradict the
    /// path the row actually measured.  RandomAccess is untouched: the fallback
    /// loop still seeks per range.
    EncodingProperties properties() const override {
        EncodingProperties props = inner_->properties();
        if (disabled(Disable::GatherFastPath)) props.remove(EncodingProperty::FastSkip);
        return props;
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        return inner_->estimateEncodedSize(elementCount);
    }

    void reset() override { inner_->reset(); }

    /// Forwarded, not defaulted, for the same reason SubIntSplitAutoEncoder
    /// forwards it: the wrapper owns no buffers, and an empty vector here would
    /// silently downgrade every cold-all row of the wrapped codec to an LLC
    /// thrash while the inner codec could enumerate its sections perfectly well.
    std::vector<std::span<const std::byte>> internalBuffers() const override {
        return inner_->internalBuffers();
    }

    // Profiling forwards.  A wrapped codec must report the same counters as an
    // unwrapped one or the two arms of the A/B are not comparable; where the
    // disabled path has no counter the inner codec's -1 / empty default is passed
    // through unchanged rather than normalised here.
    std::vector<int64_t> subStreamBulkDecodeTimeNs() const override {
        return inner_->subStreamBulkDecodeTimeNs();
    }
    std::vector<int64_t> subStreamDecodeAtAccumNs() const override {
        return inner_->subStreamDecodeAtAccumNs();
    }
    std::vector<int64_t> subStreamDecodeRangeAccumNs() const override {
        return inner_->subStreamDecodeRangeAccumNs();
    }
    void resetSubStreamDecodeAtAccum() override { inner_->resetSubStreamDecodeAtAccum(); }
    void resetSubStreamDecodeRangeAccum() override { inner_->resetSubStreamDecodeRangeAccum(); }

    int64_t reorderEncodeTimeNs() const override { return inner_->reorderEncodeTimeNs(); }
    int64_t unreorderDecodeAllTimeNs() const override {
        return inner_->unreorderDecodeAllTimeNs();
    }
    int64_t permLookupDecodeAtAccumNs() const override {
        return inner_->permLookupDecodeAtAccumNs();
    }
    int64_t permLookupDecodeRangeAccumNs() const override {
        return inner_->permLookupDecodeRangeAccumNs();
    }
    void resetReorderingProfilingAccum() override { inner_->resetReorderingProfilingAccum(); }

    /// -1 when the fast path is disabled: the skip/materialize split is a property
    /// of the override that was just bypassed, and forwarding the inner codec's
    /// stale accumulator would report the previous cell's skip time as this one's.
    int64_t gatherSkipTimeNs() const override {
        return disabled(Disable::GatherFastPath) ? -1 : inner_->gatherSkipTimeNs();
    }
    int64_t gatherMaterializeTimeNs() const override {
        return disabled(Disable::GatherFastPath) ? -1 : inner_->gatherMaterializeTimeNs();
    }
    void resetGatherProfilingAccum() override { inner_->resetGatherProfilingAccum(); }

    const encodings::Codec<T, uint8_t>& inner() const { return *inner_; }

private:
    std::shared_ptr<encodings::Codec<T, uint8_t>> inner_;
    uint8_t disabled_{0};
};

/// Disable-set literal.  A free function rather than an `operator|` on the nested
/// enum, which could not deduce T from its arguments.
template <typename T>
inline uint8_t disableSet(std::initializer_list<typename ForceFallbackCodec<T>::Disable> flags) {
    uint8_t out = 0;
    for (const auto f : flags) out = static_cast<uint8_t>(out | static_cast<uint8_t>(f));
    return out;
}

}  // namespace encodings::benchmark
