#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "core/DataType.hpp"
#include "encoders/FOREncoder.hpp"          // FORReferencePolicy, detail_for::writeLE/readLE
#include "encoders/RawBitPackedEncoder.hpp" // default leaf codec (true minimal-bit storage)
#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

// =============================================================================
//  CascadingFOREncoder<TIn>
//
//  A recursive Frame-of-Reference cascade built for experimentation: the
//  residual stream is recursively re-FOR'd through a runtime-configurable
//  schedule of (typically shrinking) frame sizes, and — independently — every
//  level's per-frame reference array is ALSO recursively FOR'd through its
//  own schedule. This lets both streams be observed (cardinality, bit width,
//  size) across arbitrary depth/frame-size combinations without recompiling,
//  since frame size and depth are ordinary runtime vectors rather than
//  compile-time template parameters (contrast with FOREncoder<TIn,TOut,N>).
//
//  Internal arithmetic is always done in int64_t regardless of TIn, so no
//  C++ type needs to change across cascade depth (avoids a combinatorial
//  explosion of FOREncoder<T1,T2,N> instantiations for a runtime schedule).
//  Genuine minimal-width WIRE storage is still achieved because every
//  recursion stopping point serializes through a pluggable leaf Codec — by
//  default RawBitPackedEncoder<int64_t>, which already picks the true
//  minimal bit width needed (see RawBitPackedEncoder.hpp) — rather than
//  padding to 8 bytes.
//
//  Breaking the regress
//  --------------------
//  If a reference array's own reference array were cascaded too, the
//  recursion would never terminate. Resolution: the *residual* cascade
//  recurses through `residualSchedule`, and at every one of its levels hands
//  its reference array to a fully independent recursion over
//  `referenceSchedule`. That reference cascade recurses through its own
//  frame-size levels too — but its OWN internal reference arrays are stored
//  directly via `referenceLeafEncoder` (never cascaded again). So "cascade
//  both streams" means: residual stream cascades to arbitrary depth, and at
//  every one of its levels the reference array also cascades to its own
//  (independently configured) depth — but that is where it stops.
//
//  Telescoping identity: intermediate residual levels don't change the
//  deepest residual values
//  ------------------------------------------------------------------------
//  computeCascadeFrameReference() (MIN/FIRST/MID) is shift-equivariant: for
//  any constant c, policy(x - c, window) == policy(x, window) - c. Because
//  each cascade level's frame size evenly divides the level above it (so
//  every finer window is nested inside exactly one coarser window), this
//  means an outer level's per-window reference constant is EXACTLY cancelled
//  the moment the next level recomputes its own (possibly different) policy
//  on that same window: residual_k = data - ref_k = (data - ref_{k-1}) -
//  (policy_k(data - ref_{k-1}, window) which itself equals policy_k(data,
//  window) - ref_{k-1}). Induction over the whole schedule collapses to:
//  the residual at any depth equals the ORIGINAL data minus policy_deepest
//  applied directly to the original data at the deepest frame size — i.e.
//  cascading through intermediate residual levels, with any mix of
//  MIN/FIRST/MID policies, is a mathematical no-op for the final residual
//  values. Only the deepest level's frame size and policy matter for
//  residual-stream compressibility; a `cascading_for` schedule and a
//  `plain_for_depth1` schedule sharing the same deepest frame size produce
//  byte-identical `computeDeepestResiduals()` output (see
//  test_cascading_for_encoder.cpp's telescoping-identity test).
//
//  This does NOT extend to the reference stream: refs[f] at a given level is
//  the policy's actual per-window VALUE (not a residual), and those values
//  differ across policies/depths in ways that are not shift-cancelled by
//  anything downstream — cascading genuinely matters there. It's also why
//  intermediate-level policy choice is provably inert for residual
//  cardinality/bit-width stats (CascadeCardinalityAnalyzer.hpp) but the
//  DEEPEST level's policy can still affect downstream compressibility
//  (e.g. BlockFrequencyPartitionEncoder) via cross-window value alignment —
//  something no per-window-local statistic captures, since it's a property
//  of whether different windows' locally-referenced residuals happen to
//  coincide numerically, not of any single window's own content.
//
//  NOTE on FORReferencePolicy::PREV and the telescoping identity above: PREV
//  is NOT one of the policies covered by computeCascadeFrameReference()
//  (MIN/FIRST/MID) and is NOT shift-equivariant in the same per-window sense
//  used by the proof — its "residual" isn't a single scalar subtracted per
//  window at all (see FOREncoder.hpp's FORReferencePolicy::PREV doc).
//  Consequently the telescoping-identity proof above does NOT extend to a
//  schedule that uses PREV at an intermediate residual level — intermediate
//  PREV levels are NOT provably inert the way intermediate MIN/FIRST/MID
//  levels are. This is why PREV should be used only at the deepest (or only)
//  residual level in any given schedule, not a bug — see
//  SubIntEncodingUtils.hpp's makeCascadingFORPrev*Section factories.
//
//  Wire format
//  -----------
//  Top-level:
//    [0 .. 7]   N                (uint64_t)
//    [8 .. 15]  residualLevels   (uint64_t) — cfg_.residualSchedule.size()
//    [16 .. 23] referenceLevels  (uint64_t) — cfg_.referenceSchedule.size()
//    [24 .. end] body            — output of encodeResidualLevel(data, 0)
//
//  Each non-leaf recursive level (both residual and reference roles) writes:
//    [0 .. 7]   n          (uint64_t) — element count at this level
//    [8 .. 15]  frameSize  (uint64_t)
//    [16 .. 23] numFrames  (uint64_t)
//    [24 .. 31] refBytes   (uint64_t) — length of the following ref payload
//    [32 .. 39] resBytes   (uint64_t) — length of the following residual payload
//    [40 .. 40+refBytes)   reference payload
//    [40+refBytes .. end)  residual/continuation payload
//
//  A leaf level (recursion index == schedule.size()) writes nothing beyond
//  whatever `leafEncoder->encode()` itself produces.
//
//  Random access
//  -------------
//  decodeAt/decodeRange perform a genuine recursive frame-walk: every level's
//  wire format already stores refBytes/resBytes lengths explicitly, so nested
//  byte sub-spans are sliced via pure pointer arithmetic with no decoding at
//  all at intermediate levels. Total cost is O(residualSchedule.size() *
//  referenceSchedule.size()) recursive calls, each O(1) when the configured
//  leaf encoders (residualLeafEncoder/referenceLeafEncoder) report
//  EncodingProperty::RandomAccess (mirrors FOREncoder::decodeAt's existing
//  "delegate if RandomAccess, else full-decode" idiom, applied at each leaf
//  boundary) -- falling back to a one-time cached decodeAll() of just that
//  leaf's own (small) buffer when a configured leaf lacks RandomAccess.
//  decodeRange is frame-aware at the outermost (level-0) residual frame only:
//  each touched frame's reference is computed once, then the residual
//  continuation is fetched per-index via the same recursive decodeAt-style
//  descent -- a deliberate, non-maximal scoping choice that still avoids
//  materialising the full N-element array. This cache is NOT thread-safe,
//  matching FOREncoder's existing caveat.
//
//  FORReferencePolicy::PREV changes this at whichever level(s) use it: that
//  level's residual is a running delta within its frame rather than a single
//  scalar per frame (see FOREncoder.hpp's PREV doc), so reconstructing an
//  index at that level costs O(idx - frameStart + 1) <= O(that level's
//  FrameSize) instead of O(1) -- still a config-time-bounded constant
//  independent of N, so RandomAccess is still reported honestly, just with a
//  larger (bounded) constant factor. Using PREV at multiple nested levels
//  multiplies this cost across those levels' frame sizes -- recommended
//  usage is PREV at the deepest/only residual level only (see the
//  telescoping-identity note above). decodeRange's level-0 loop special-cases
//  PREV to walk once per touched frame (batching through the residual leaf's
//  own decodeRange when the schedule has exactly one level) rather than
//  degrading to O(range x FrameSize) by re-walking from the frame start once
//  per queried index.
// =============================================================================

enum class CascadeStreamRole : uint8_t { Residual = 0, Reference = 1 };

// One level of a cascade schedule.
struct CascadeLevelConfig {
    size_t frameSize;
    FORReferencePolicy policy = FORReferencePolicy::MIN;
};

// Configuration for CascadingFOREncoder. residualSchedule is applied
// recursively to the main data stream (level 0 = outermost/largest frame).
// referenceSchedule is applied independently — and fresh — to every level's
// per-frame reference array; see the class-level doc for why its own
// internal reference arrays are not cascaded again.
struct CascadingFORConfig {
    std::vector<CascadeLevelConfig> residualSchedule;
    std::vector<CascadeLevelConfig> referenceSchedule;
    std::shared_ptr<Codec<int64_t, uint8_t>> residualLeafEncoder =
        std::make_shared<RawBitPackedEncoder<int64_t>>();
    std::shared_ptr<Codec<int64_t, uint8_t>> referenceLeafEncoder =
        std::make_shared<RawBitPackedEncoder<int64_t>>();
};

// Shared by CascadingFOREncoder and CascadeCardinalityAnalyzer so the analyzer
// replays the exact same partitioning logic the encoder uses, rather than a
// second hand-written copy that could drift out of sync.
inline int64_t computeCascadeFrameReference(std::span<const int64_t> data, size_t lo, size_t hi,
                                            FORReferencePolicy policy) {
    switch (policy) {
        case FORReferencePolicy::FIRST:
            return data[lo];
        case FORReferencePolicy::MIN:
            return *std::min_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                      data.begin() + static_cast<ptrdiff_t>(hi));
        case FORReferencePolicy::MID: {
            const int64_t mn = *std::min_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                                  data.begin() + static_cast<ptrdiff_t>(hi));
            const int64_t mx = *std::max_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                                  data.begin() + static_cast<ptrdiff_t>(hi));
            return mn + (mx - mn) / 2;
        }
        case FORReferencePolicy::PREV:
            // PREV has no single-scalar-per-window reference value (see
            // FOREncoder.hpp's PREV doc) -- it cannot be expressed through
            // this function's interface. Callers needing PREV must go
            // through detail_for::computeFrameReferenceAndResiduals instead
            // (used directly by CascadingFOREncoder::encodeResidualLevel/
            // encodeReferenceLevel); CascadeCardinalityAnalyzer, the other
            // caller of this function, only ever uses MIN/FIRST/MID.
            throw std::invalid_argument(
                "computeCascadeFrameReference: PREV is not representable as a single "
                "per-window scalar reference -- use detail_for::computeFrameReferenceAndResiduals");
    }
    return data[lo]; // unreachable
}

// Replays only the residual-role descent (no reference-cascade recursion, no
// serialization) and returns the residual array at the bottom of the deepest
// level — i.e. exactly what residualLeafEncoder would receive. Useful for
// experiments that want to feed the deepest-level residuals to a different
// encoder entirely (see cascade_blockfpe_experiment.cpp), without paying for
// a full encode() or for reference-stream statistics they don't need.
inline std::vector<int64_t> computeDeepestResiduals(std::span<const int64_t> data,
                                                     const std::vector<CascadeLevelConfig>& residualSchedule) {
    std::vector<int64_t> current(data.begin(), data.end());
    for (const auto& level : residualSchedule) {
        const size_t frameSize = level.frameSize;
        const size_t N = current.size();
        const size_t numFrames = (N + frameSize - 1) / frameSize;
        std::vector<int64_t> next(N);
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f * frameSize;
            const size_t hi = std::min(lo + frameSize, N);
            // Uses the shared computeFrameReferenceAndResiduals (not
            // computeCascadeFrameReference) so this replays the encoder's
            // exact per-frame logic for EVERY policy including PREV, not
            // just MIN/FIRST/MID.
            detail_for::computeFrameReferenceAndResiduals<int64_t>(
                std::span<const int64_t>(current), lo, hi, level.policy, std::span<int64_t>(next));
        }
        current = std::move(next);
    }
    return current;
}

template <typename TIn>
class CascadingFOREncoder : public Codec<TIn, uint8_t> {
    static_assert(std::is_integral_v<TIn> && std::is_signed_v<TIn>,
                  "CascadingFOREncoder: TIn must be a signed integral type");

    static constexpr size_t kTopHeaderSize   = 24; // 3 * uint64_t
    static constexpr size_t kLevelHeaderSize = 40; // 5 * uint64_t

public:
    explicit CascadingFOREncoder(CascadingFORConfig cfg) : cfg_(std::move(cfg)) {
        if (!cfg_.residualLeafEncoder)
            throw std::invalid_argument("CascadingFOREncoder: residualLeafEncoder must not be null");
        if (!cfg_.referenceLeafEncoder)
            throw std::invalid_argument("CascadingFOREncoder: referenceLeafEncoder must not be null");
        for (const auto& lvl : cfg_.residualSchedule)
            if (lvl.frameSize == 0)
                throw std::invalid_argument("CascadingFOREncoder: residualSchedule frameSize must be >= 1");
        for (const auto& lvl : cfg_.referenceSchedule)
            if (lvl.frameSize == 0)
                throw std::invalid_argument("CascadingFOREncoder: referenceSchedule frameSize must be >= 1");
    }

    // ---- Encoder::encode ---------------------------------------------------

    EncodedBuffer<uint8_t> encode(std::span<const TIn> data) override {
        const size_t N = data.size();
        if (N == 0) return makeEmpty();

        std::vector<int64_t> wide(N);
        for (size_t i = 0; i < N; ++i) wide[i] = static_cast<int64_t>(data[i]);

        std::vector<uint8_t> body = encodeResidualLevel(std::span<const int64_t>(wide), 0);

        std::vector<uint8_t> out;
        out.reserve(kTopHeaderSize + body.size());
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(N));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(cfg_.residualSchedule.size()));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(cfg_.referenceSchedule.size()));
        out.insert(out.end(), body.begin(), body.end());

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::core::typeToDataType<TIn>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(TIn);
        meta.supportsRandomAccess = properties().has(EncodingProperty::RandomAccess);

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // ---- Decoder::decodeAll -------------------------------------------------

    std::vector<TIn> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        if (encoded.data().size() < kTopHeaderSize) return {};
        const uint8_t* p = encoded.data().data();
        const uint64_t N               = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t residualLevels  = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t referenceLevels = detail_for::readLE<uint64_t>(p); p += 8;
        if (N == 0) return {};

        if (residualLevels != cfg_.residualSchedule.size() ||
            referenceLevels != cfg_.referenceSchedule.size()) {
            throw std::runtime_error(
                "CascadingFOREncoder::decodeAll: schedule mismatch between the config used to "
                "encode and the config of the decoding instance");
        }

        std::span<const uint8_t> body(p, encoded.data().size() - kTopHeaderSize);
        std::vector<int64_t> wide = decodeResidualLevel(body, 0, static_cast<size_t>(N));

        std::vector<TIn> out(wide.size());
        for (size_t i = 0; i < wide.size(); ++i) out[i] = static_cast<TIn>(wide[i]);
        return out;
    }

    // ---- Decoder::decodeAt / decodeRange ------------------------------------
    // See class-level doc: genuine O(depth) recursive frame-walk, no full
    // materialisation (except within a single small leaf buffer when that
    // leaf itself lacks RandomAccess).

    std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        if (encoded.data().size() < kTopHeaderSize) return std::nullopt;
        const uint8_t* p = encoded.data().data();
        const uint64_t N               = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t residualLevels  = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t referenceLevels = detail_for::readLE<uint64_t>(p); p += 8;
        if (N == 0 || index >= N) return std::nullopt;

        if (residualLevels != cfg_.residualSchedule.size() ||
            referenceLevels != cfg_.referenceSchedule.size()) {
            throw std::runtime_error(
                "CascadingFOREncoder::decodeAt: schedule mismatch between the config used to "
                "encode and the config of the decoding instance");
        }

        std::span<const uint8_t> body(p, encoded.data().size() - kTopHeaderSize);
        const int64_t wide = decodeResidualLevelAt(body, 0, index);
        return static_cast<TIn>(wide);
    }

    std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        if (encoded.data().size() < kTopHeaderSize) return {};
        const uint8_t* p = encoded.data().data();
        const uint64_t N               = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t residualLevels  = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t referenceLevels = detail_for::readLE<uint64_t>(p); p += 8;
        if (N == 0) return {};
        end = std::min(end, static_cast<size_t>(N));
        if (start >= end) return {};

        if (residualLevels != cfg_.residualSchedule.size() ||
            referenceLevels != cfg_.referenceSchedule.size()) {
            throw std::runtime_error(
                "CascadingFOREncoder::decodeRange: schedule mismatch between the config used to "
                "encode and the config of the decoding instance");
        }

        std::span<const uint8_t> body(p, encoded.data().size() - kTopHeaderSize);

        std::vector<TIn> out;
        out.reserve(end - start);

        if (cfg_.residualSchedule.empty()) {
            // Depth-0: the residual leaf IS the data -- no frame structure to amortise.
            for (size_t i = start; i < end; ++i)
                out.push_back(static_cast<TIn>(decodeResidualLevelAt(body, 0, i)));
            return out;
        }

        // Frame-aware at level 0 only (see class-level doc): parse level-0's
        // header once, group [start,end) by its frameSize, computing each
        // touched frame's reference exactly once via a fresh reference descent.
        if (body.size() < kLevelHeaderSize)
            throw std::runtime_error("CascadingFOREncoder::decodeRange: truncated level-0 header");
        const uint8_t* lp = body.data();
        lp += 8; // n -- not needed (idx range is already known from start/end)
        const uint64_t frameSize0 = detail_for::readLE<uint64_t>(lp); lp += 8;
        lp += 8; // numFrames -- not needed for pointer-arithmetic-only access
        const uint64_t refLen0 = detail_for::readLE<uint64_t>(lp); lp += 8;
        const uint64_t resLen0 = detail_for::readLE<uint64_t>(lp); lp += 8;
        std::span<const uint8_t> refSpan0(lp, static_cast<size_t>(refLen0));
        std::span<const uint8_t> resSpan0(lp + refLen0, static_cast<size_t>(resLen0));

        size_t lastFrameIdx = std::numeric_limits<size_t>::max();
        int64_t cachedRefVal = 0;

        if (cfg_.residualSchedule[0].policy != FORReferencePolicy::PREV) {
            for (size_t i = start; i < end; ++i) {
                const size_t frameIdx = i / static_cast<size_t>(frameSize0);
                if (frameIdx != lastFrameIdx) {
                    cachedRefVal = decodeReferenceLevelAt(refSpan0, 0, frameIdx);
                    lastFrameIdx = frameIdx;
                }
                const int64_t residualVal = decodeResidualLevelAt(resSpan0, 1, i);
                out.push_back(static_cast<TIn>(cachedRefVal + residualVal));
            }
            return out;
        }

        // PREV at level 0: walk once per touched frame rather than once per
        // index, accumulating residuals via decodeResidualLevelAt(resSpan0, 1,
        // k) (correct at any residualSchedule depth -- see that function's own
        // per-level PREV branch). When residualSchedule.size() == 1 (the
        // recommended/registered usage), resSpan0 IS cfg_.residualLeafEncoder's
        // own encoded bytes, so we batch through its own decodeRange for real
        // per-frame efficiency (at most 2 leaf decodeRange calls per touched
        // frame) instead of one decodeResidualLevelAt call per index.
        const bool leafDirect = (cfg_.residualSchedule.size() == 1) &&
            cfg_.residualLeafEncoder->properties().has(EncodingProperty::RandomAccess);
        size_t i = start;
        while (i < end) {
            const size_t frameIdx = i / static_cast<size_t>(frameSize0);
            const size_t frameLo  = frameIdx * static_cast<size_t>(frameSize0);
            const size_t frameHi  = std::min(frameLo + static_cast<size_t>(frameSize0), static_cast<size_t>(N));
            const size_t segEnd   = std::min(frameHi, end);
            if (frameIdx != lastFrameIdx) {
                cachedRefVal = decodeReferenceLevelAt(refSpan0, 0, frameIdx);
                lastFrameIdx = frameIdx;
            }

            int64_t acc = cachedRefVal;
            if (i > frameLo) {
                // Seed the accumulator up to i (window doesn't start on this frame's boundary).
                if (leafDirect) {
                    auto seed = cfg_.residualLeafEncoder->decodeRange(spanToBuffer(resSpan0), frameLo + 1, i + 1);
                    for (int64_t v : seed) acc += v;
                } else {
                    for (size_t k = frameLo + 1; k <= i; ++k) acc += decodeResidualLevelAt(resSpan0, 1, k);
                }
            }
            out.push_back(static_cast<TIn>(acc)); // value at index i

            if (i + 1 < segEnd) {
                if (leafDirect) {
                    auto seg = cfg_.residualLeafEncoder->decodeRange(spanToBuffer(resSpan0), i + 1, segEnd);
                    for (int64_t v : seg) { acc += v; out.push_back(static_cast<TIn>(acc)); }
                } else {
                    for (size_t k = i + 1; k < segEnd; ++k) {
                        acc += decodeResidualLevelAt(resSpan0, 1, k);
                        out.push_back(static_cast<TIn>(acc));
                    }
                }
            }
            i = segEnd;
        }
        return out;
    }

    // ---- Identity ------------------------------------------------------------

    EncodingType encodingType() const override { return EncodingType::CascadingFrameOfReference; }

    std::string name() const override {
        return "CascadingFOR<" + std::string(typeid(TIn).name()) +
               ",resLevels=" + std::to_string(cfg_.residualSchedule.size()) +
               ",refLevels=" + std::to_string(cfg_.referenceSchedule.size()) + ">";
    }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        auto props = EncodingProperties{}
            .add(EP::Lossless)
            .add(EP::PreservesOrder)
            .add(EP::DeltaBased)
            .add(EP::Composable);
        // Every recursion level uses the SAME configured leaf type, so checking
        // the leaf once is representative of the whole cascade (mirrors
        // RangePackSectionCodec::properties()'s innerFactory_(8) probe pattern).
        // Unlike RunLengthEncoder's runStarts (bounded by R << N), an empty
        // residualSchedule means the residual leaf IS the full N-element data,
        // so a non-RandomAccess leaf there genuinely means no random access --
        // hence this is a real (not just approximate) check, not unconditional.
        if (cfg_.residualLeafEncoder->properties().has(EP::RandomAccess) &&
            cfg_.referenceLeafEncoder->properties().has(EP::RandomAccess)) {
            props.add(EP::RandomAccess);
        }
        return props;
    }

private:
    // ---- Encode-side recursion -----------------------------------------------

    // Residual-role recursion: descends cfg_.residualSchedule. At each level,
    // hands its own reference array to a fresh reference-role recursion, then
    // continues on residuals at level+1.
    std::vector<uint8_t> encodeResidualLevel(std::span<const int64_t> data, size_t level) const {
        if (level == cfg_.residualSchedule.size()) {
            return cfg_.residualLeafEncoder->encode(data).data();
        }

        const size_t frameSize = cfg_.residualSchedule[level].frameSize;
        const FORReferencePolicy policy = cfg_.residualSchedule[level].policy;
        const size_t N = data.size();
        const size_t numFrames = (N + frameSize - 1) / frameSize;

        std::vector<int64_t> refs(numFrames);
        std::vector<int64_t> residuals(N);
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f * frameSize;
            const size_t hi = std::min(lo + frameSize, N);
            refs[f] = detail_for::computeFrameReferenceAndResiduals<int64_t>(
                data, lo, hi, policy, std::span<int64_t>(residuals));
        }

        std::vector<uint8_t> refBytes = encodeReferenceLevel(std::span<const int64_t>(refs), 0);
        std::vector<uint8_t> resBytes = encodeResidualLevel(std::span<const int64_t>(residuals), level + 1);

        return packLevel(N, frameSize, numFrames, refBytes, resBytes);
    }

    // Reference-role recursion: descends cfg_.referenceSchedule. Its OWN
    // reference arrays are stored directly via referenceLeafEncoder (not
    // cascaded again) — this is what keeps the recursion finite.
    std::vector<uint8_t> encodeReferenceLevel(std::span<const int64_t> data, size_t level) const {
        if (level == cfg_.referenceSchedule.size()) {
            return cfg_.referenceLeafEncoder->encode(data).data();
        }

        const size_t frameSize = cfg_.referenceSchedule[level].frameSize;
        const FORReferencePolicy policy = cfg_.referenceSchedule[level].policy;
        const size_t N = data.size();
        const size_t numFrames = (N + frameSize - 1) / frameSize;

        std::vector<int64_t> refs(numFrames);
        std::vector<int64_t> residuals(N);
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f * frameSize;
            const size_t hi = std::min(lo + frameSize, N);
            refs[f] = detail_for::computeFrameReferenceAndResiduals<int64_t>(
                data, lo, hi, policy, std::span<int64_t>(residuals));
        }

        std::vector<uint8_t> refBytes = cfg_.referenceLeafEncoder->encode(std::span<const int64_t>(refs)).data();
        std::vector<uint8_t> resBytes = encodeReferenceLevel(std::span<const int64_t>(residuals), level + 1);

        return packLevel(N, frameSize, numFrames, refBytes, resBytes);
    }

    static std::vector<uint8_t> packLevel(size_t n, size_t frameSize, size_t numFrames,
                                          const std::vector<uint8_t>& refBytes,
                                          const std::vector<uint8_t>& resBytes) {
        std::vector<uint8_t> out;
        out.reserve(kLevelHeaderSize + refBytes.size() + resBytes.size());
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(n));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(frameSize));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(numFrames));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(refBytes.size()));
        detail_for::writeLE<uint64_t>(out, static_cast<uint64_t>(resBytes.size()));
        out.insert(out.end(), refBytes.begin(), refBytes.end());
        out.insert(out.end(), resBytes.begin(), resBytes.end());
        return out;
    }

    // ---- Decode-side recursion -------------------------------------------------

    static EncodedBuffer<uint8_t> spanToBuffer(std::span<const uint8_t> bytes) {
        std::vector<uint8_t> copy(bytes.begin(), bytes.end());
        return EncodedBuffer<uint8_t>(std::move(copy), encodings::EncodingMetadata{});
    }

    std::vector<int64_t> decodeResidualLevel(std::span<const uint8_t> bytes, size_t level, size_t n) const {
        if (level == cfg_.residualSchedule.size()) {
            EncodedBuffer<uint8_t> leafBuf = spanToBuffer(bytes);
            auto values = cfg_.residualLeafEncoder->decodeAll(leafBuf);
            if (values.size() != n)
                throw std::runtime_error("CascadingFOREncoder: residual leaf decode size mismatch");
            return values;
        }

        if (bytes.size() < kLevelHeaderSize)
            throw std::runtime_error("CascadingFOREncoder: truncated residual level header");
        const uint8_t* p = bytes.data();
        p += 8; // n (already known from caller; re-derivable but not needed)
        const uint64_t frameSize = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t numFrames = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t refLen    = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t resLen    = detail_for::readLE<uint64_t>(p); p += 8;

        std::span<const uint8_t> refSpan(p, static_cast<size_t>(refLen));
        std::span<const uint8_t> resSpan(p + refLen, static_cast<size_t>(resLen));

        std::vector<int64_t> refs      = decodeReferenceLevel(refSpan, 0, static_cast<size_t>(numFrames));
        std::vector<int64_t> residuals = decodeResidualLevel(resSpan, level + 1, n);

        std::vector<int64_t> out(n);
        if (cfg_.residualSchedule[level].policy == FORReferencePolicy::PREV) {
            for (size_t i = 0; i < n; ++i) {
                const size_t f  = i / static_cast<size_t>(frameSize);
                const size_t lo = f * static_cast<size_t>(frameSize);
                out[i] = (i == lo) ? refs[f] : (out[i - 1] + residuals[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                out[i] = refs[i / static_cast<size_t>(frameSize)] + residuals[i];
            }
        }
        return out;
    }

    std::vector<int64_t> decodeReferenceLevel(std::span<const uint8_t> bytes, size_t level, size_t n) const {
        if (level == cfg_.referenceSchedule.size()) {
            EncodedBuffer<uint8_t> leafBuf = spanToBuffer(bytes);
            auto values = cfg_.referenceLeafEncoder->decodeAll(leafBuf);
            if (values.size() != n)
                throw std::runtime_error("CascadingFOREncoder: reference leaf decode size mismatch");
            return values;
        }

        if (bytes.size() < kLevelHeaderSize)
            throw std::runtime_error("CascadingFOREncoder: truncated reference level header");
        const uint8_t* p = bytes.data();
        p += 8; // n
        const uint64_t frameSize = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t numFrames = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t refLen    = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t resLen    = detail_for::readLE<uint64_t>(p); p += 8;

        std::span<const uint8_t> refSpan(p, static_cast<size_t>(refLen));
        std::span<const uint8_t> resSpan(p + refLen, static_cast<size_t>(resLen));

        // This level's own reference array was stored directly (see class-level doc).
        EncodedBuffer<uint8_t> refBuf = spanToBuffer(refSpan);
        std::vector<int64_t> refs = cfg_.referenceLeafEncoder->decodeAll(refBuf);
        if (refs.size() != numFrames)
            throw std::runtime_error("CascadingFOREncoder: reference-of-reference leaf decode size mismatch");

        std::vector<int64_t> residuals = decodeReferenceLevel(resSpan, level + 1, n);

        std::vector<int64_t> out(n);
        if (cfg_.referenceSchedule[level].policy == FORReferencePolicy::PREV) {
            for (size_t i = 0; i < n; ++i) {
                const size_t f  = i / static_cast<size_t>(frameSize);
                const size_t lo = f * static_cast<size_t>(frameSize);
                out[i] = (i == lo) ? refs[f] : (out[i - 1] + residuals[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                out[i] = refs[i / static_cast<size_t>(frameSize)] + residuals[i];
            }
        }
        return out;
    }

    // ---- Decode-side recursion, index-based (true random access) --------------
    //
    // These replicate decodeResidualLevel/decodeReferenceLevel's exact
    // recurrence (out[i] = refs[i / frameSize] + residuals[i]) index-by-index
    // instead of array-by-array, so nested byte sub-spans are sliced via pure
    // pointer arithmetic with no decoding at intermediate levels. See the
    // class-level doc's "Random access" section for the complexity argument.

    // idx = global data index, CONSTANT across all residual levels.
    int64_t decodeResidualLevelAt(std::span<const uint8_t> bytes, size_t level, size_t idx) const {
        if (level == cfg_.residualSchedule.size()) {
            return decodeLeafAt(cfg_.residualLeafEncoder, bytes, idx, residualLeafCache_);
        }

        if (bytes.size() < kLevelHeaderSize)
            throw std::runtime_error("CascadingFOREncoder::decodeResidualLevelAt: truncated header");
        const uint8_t* p = bytes.data();
        p += 8; // n -- not needed (idx is already known from the caller)
        const uint64_t frameSize = detail_for::readLE<uint64_t>(p); p += 8;
        p += 8; // numFrames -- not needed for pointer-arithmetic-only access
        const uint64_t refLen = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t resLen = detail_for::readLE<uint64_t>(p); p += 8;

        std::span<const uint8_t> refSpan(p, static_cast<size_t>(refLen));
        std::span<const uint8_t> resSpan(p + refLen, static_cast<size_t>(resLen));

        const size_t frameIdx = idx / static_cast<size_t>(frameSize); // REBASE global idx -> this level's frame space
        const int64_t refVal = decodeReferenceLevelAt(refSpan, /*level=*/0, frameIdx); // fresh reference descent

        if (cfg_.residualSchedule[level].policy != FORReferencePolicy::PREV) {
            const int64_t residualVal = decodeResidualLevelAt(resSpan, level + 1, idx); // same idx, next residual level
            return refVal + residualVal;
        }

        // PREV: walk from this frame's start to idx, accumulating residuals.
        // Bounded by O(idx - frameStart + 1) <= O(this level's FrameSize) -- a
        // config-time constant, independent of N (see class-level "Random
        // access" doc). Recommended usage is PREV at the deepest/only
        // residual level only (see the telescoping-identity note above) --
        // this still works correctly if used elsewhere, just compounds cost
        // with any further nested levels' own frame sizes.
        const size_t frameLo = frameIdx * static_cast<size_t>(frameSize);
        if (idx == frameLo) return refVal;
        int64_t acc = refVal;
        for (size_t k = frameLo + 1; k <= idx; ++k) {
            acc += decodeResidualLevelAt(resSpan, level + 1, k);
        }
        return acc;
    }

    // idx = index into the array being reconstructed at THIS reference cascade
    // (this residual level's own per-frame ref array at level 0; narrows by
    // /frameSize deeper in) -- NOT the original global data index.
    int64_t decodeReferenceLevelAt(std::span<const uint8_t> bytes, size_t level, size_t idx) const {
        if (level == cfg_.referenceSchedule.size()) {
            return decodeLeafAt(cfg_.referenceLeafEncoder, bytes, idx, referenceLeafCache_);
        }

        if (bytes.size() < kLevelHeaderSize)
            throw std::runtime_error("CascadingFOREncoder::decodeReferenceLevelAt: truncated header");
        const uint8_t* p = bytes.data();
        p += 8; // n
        const uint64_t frameSize = detail_for::readLE<uint64_t>(p); p += 8;
        p += 8; // numFrames
        const uint64_t refLen = detail_for::readLE<uint64_t>(p); p += 8;
        const uint64_t resLen = detail_for::readLE<uint64_t>(p); p += 8;

        std::span<const uint8_t> refSpan(p, static_cast<size_t>(refLen));
        std::span<const uint8_t> resSpan(p + refLen, static_cast<size_t>(resLen));

        // This level's OWN reference array is a LEAF lookup (never cascaded
        // again -- matches the encode-side design, see class-level doc).
        const size_t frameIdx = idx / static_cast<size_t>(frameSize);
        const int64_t refVal = decodeLeafAt(cfg_.referenceLeafEncoder, refSpan, frameIdx, referenceOfReferenceLeafCache_);

        if (cfg_.referenceSchedule[level].policy != FORReferencePolicy::PREV) {
            const int64_t residualVal = decodeReferenceLevelAt(resSpan, level + 1, idx);
            return refVal + residualVal;
        }

        // PREV: same bounded per-frame walk as decodeResidualLevelAt's PREV branch.
        const size_t frameLo = frameIdx * static_cast<size_t>(frameSize);
        if (idx == frameLo) return refVal;
        int64_t acc = refVal;
        for (size_t k = frameLo + 1; k <= idx; ++k) {
            acc += decodeReferenceLevelAt(resSpan, level + 1, k);
        }
        return acc;
    }

    // Small helper struct: caches a single leaf's fully-decoded array, keyed by
    // the identity (pointer+size) of the byte span it was decoded from -- only
    // used when that leaf lacks RandomAccess. Three independent instances are
    // needed (residual leaf / reference leaf / reference-of-reference leaf)
    // since the latter two both use referenceLeafEncoder but on structurally
    // different byte spans, and must not collide in a single cache slot.
    struct LeafDecodeCache {
        const uint8_t* ptr{nullptr};
        size_t size{0};
        std::vector<int64_t> decoded;
    };
    mutable LeafDecodeCache residualLeafCache_;
    mutable LeafDecodeCache referenceLeafCache_;
    mutable LeafDecodeCache referenceOfReferenceLeafCache_;

    // Decodes a single index from a leaf-encoded buffer. O(1) when the leaf
    // reports RandomAccess (delegates directly to its own decodeAt); otherwise
    // falls back to a one-time cached decodeAll() of just this leaf's own
    // (small) buffer -- mirrors FOREncoder::decodeAt's existing
    // "delegate if RandomAccess, else full decode" idiom.
    int64_t decodeLeafAt(const std::shared_ptr<Codec<int64_t,uint8_t>>& leaf,
                         std::span<const uint8_t> bytes,
                         size_t idx,
                         LeafDecodeCache& cacheSlot) const {
        if (leaf->properties().has(EncodingProperty::RandomAccess)) {
            EncodedBuffer<uint8_t> buf = spanToBuffer(bytes);
            auto v = leaf->decodeAt(buf, idx);
            if (!v) throw std::runtime_error("CascadingFOREncoder: leaf decodeAt returned nullopt for valid index");
            return *v;
        }
        if (cacheSlot.ptr != bytes.data() || cacheSlot.size != bytes.size()) {
            EncodedBuffer<uint8_t> buf = spanToBuffer(bytes);
            cacheSlot.decoded = leaf->decodeAll(buf);
            cacheSlot.ptr = bytes.data();
            cacheSlot.size = bytes.size();
        }
        if (idx >= cacheSlot.decoded.size())
            throw std::runtime_error("CascadingFOREncoder: leaf cache index out of range");
        return cacheSlot.decoded[idx];
    }

    // ---- Empty result ----------------------------------------------------------

    static EncodedBuffer<uint8_t> makeEmpty() {
        std::vector<uint8_t> out(kTopHeaderSize, 0);
        encodings::EncodingMetadata meta;
        meta.encodingName         = "CascadingFOR(empty)";
        meta.elementCount         = 0;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = 0;
        meta.supportsRandomAccess = true; // trivially -- no elements to access either way
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    CascadingFORConfig cfg_;
};

} // namespace encodings::encoders
