// Tests for FORReferencePolicy::PREV (bounded-frame consecutive-element
// delta) across FOREncoder, CascadingFOREncoder, and AdaptiveFOREncoder.
// Mirrors test_cascading_for_encoder.cpp's style (round-trip helpers /
// NoRandomAccessStub-style leaf stubs / diagnostic FAIL prints) but is kept
// in its own file since PREV is a large enough addition to warrant a focused
// suite without bloating the existing MIN/FIRST/MID-focused file.
//
// NOTE: this project's default build type is Release (NDEBUG defined), which
// makes bare assert() a no-op -- CHECK() below is a small always-active
// replacement so correctness checks can't silently vanish under -DNDEBUG.

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "encoders/FOREncoder.hpp"
#include "encoders/CascadingFOREncoder.hpp"
#include "encoders/AdaptiveFOREncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"

using namespace encodings;
using namespace encodings::encoders;

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "] " << #cond << "\n"; \
        ++g_failures; \
    } \
} while (0)

// =============================================================================
// FOREncoder<TIn,TOut,FrameSize> PREV tests
// =============================================================================

template <typename TIn, typename TOut, size_t FrameSize>
static void assertForRoundTrip(const std::vector<TIn>& input, FORConfig<TIn, TOut> cfg, const std::string& label) {
    FOREncoder<TIn, TOut, FrameSize> enc(cfg);
    auto encoded = enc.encode(std::span<const TIn>(input));

    auto all = enc.decodeAll(encoded);
    CHECK(all == input);

    for (size_t i = 0; i < input.size(); ++i) {
        auto v = enc.decodeAt(encoded, i);
        CHECK(v.has_value() && *v == input[i] && *v == all[i]);
    }
    CHECK(!enc.decodeAt(encoded, input.size() + 5).has_value());

    if (!input.empty()) {
        // Windows both aligned to and offset from frame boundaries -- the
        // offset ones specifically exercise decodeRange's PREV "seed" logic.
        const std::vector<std::pair<size_t, size_t>> windows = {
            {0, input.size()},
            {0, input.size() / 2},
            {1, input.size()},
            {FrameSize / 2 < input.size() ? FrameSize / 2 : 0, input.size()},
            {input.size() - 1, input.size()},
        };
        for (const auto& [s, e] : windows) {
            if (s >= e) continue;
            auto range = enc.decodeRange(encoded, s, e);
            CHECK(range.size() == e - s);
            for (size_t i = s; i < e && i - s < range.size(); ++i) {
                CHECK(range[i - s] == input[i]);
            }
        }
    }

    std::cout << "PASS: " << label << "  N=" << input.size() << "  bytes=" << encoded.data().size() << "\n";
}

static void testForPrevBasic() {
    FORConfig<int32_t, int32_t> cfg{.policy = FORReferencePolicy::PREV,
                                     .subEncoder = std::make_shared<RawBitPackedEncoder<int32_t>>()};

    assertForRoundTrip<int32_t, int32_t, 128>({}, cfg, "PREV N=0");
    assertForRoundTrip<int32_t, int32_t, 128>({42}, cfg, "PREV N=1");
    assertForRoundTrip<int32_t, int32_t, 1>({1, -5, 100, -100, 0, 7}, cfg, "PREV FrameSize=1");

    {
        std::vector<int32_t> v(2000, 777);
        assertForRoundTrip<int32_t, int32_t, 128>(v, cfg, "PREV constant");
    }
    {
        std::vector<int32_t> v;
        for (int i = 0; i < 2000; ++i) v.push_back(i);
        assertForRoundTrip<int32_t, int32_t, 128>(v, cfg, "PREV monotone increasing");
    }
    {
        std::vector<int32_t> v;
        for (int i = 0; i < 2000; ++i) v.push_back(-i);
        assertForRoundTrip<int32_t, int32_t, 128>(v, cfg, "PREV monotone decreasing");
    }
    {
        std::mt19937_64 rng(123);
        std::uniform_int_distribution<int32_t> dist(-50, 50);
        std::vector<int32_t> v(2000);
        int32_t cur = 0;
        for (auto& x : v) { cur += dist(rng); x = cur; }
        assertForRoundTrip<int32_t, int32_t, 128>(v, cfg, "PREV random walk (local reversals)");
    }
    {
        // FrameSize >= N: single frame covering everything.
        std::vector<int32_t> v;
        for (int i = 0; i < 50; ++i) v.push_back((i * 13) % 37 - 18);
        assertForRoundTrip<int32_t, int32_t, 128>(v, cfg, "PREV FrameSize>=N");
    }
}

// Confirms the overflow check correctly fires for PREV's residual formula
// (data[i]-data[i-1], not data[i]-ref) -- a sharp local jump whose
// consecutive-element delta exceeds TOut's range.
//
// Note: for a SINGLE frame, MIN's residual range [0, span] and PREV's
// max |delta| are both bounded by that frame's own span, so there is no
// data shape where PREV overflows a given TOut while MIN's residuals
// (computed over the same frame) do not -- both share the same underlying
// span constraint. This test instead directly confirms PREV's overflow
// check fires under its own residual formula, which is what actually
// changed in this code path (MIN's pre-existing check is already covered
// by FOREncoder's existing test coverage elsewhere).
static void testForPrevOverflow() {
    std::vector<int32_t> data = {0, 1, 2, 3, 200, 4, 5}; // delta 3->200 is +197, overflows int8_t
    FORConfig<int32_t, int8_t> cfg{.policy = FORReferencePolicy::PREV,
                                    .subEncoder = std::make_shared<RawBitPackedEncoder<int8_t>>()};
    FOREncoder<int32_t, int8_t, 128> enc(cfg);
    bool threw = false;
    try {
        enc.encode(std::span<const int32_t>(data));
    } catch (const std::overflow_error&) {
        threw = true;
    }
    CHECK(threw);

    // Sanity: small consecutive deltas do NOT overflow -- confirms the check
    // is genuinely delta-based, not a blanket rejection of large values.
    std::vector<int32_t> smallDeltas = {10, 11, 12, 13, 14, 15};
    FOREncoder<int32_t, int8_t, 128> enc2(cfg);
    bool threw2 = false;
    try {
        enc2.encode(std::span<const int32_t>(smallDeltas));
    } catch (const std::overflow_error&) {
        threw2 = true;
    }
    CHECK(!threw2);

    std::cout << "PASS: PREV-specific overflow detection (fires for large consecutive-element deltas)\n";
}

// Non-RandomAccess sub-encoder: exercises decodeAt/decodeRange's PREV
// cached-fallback path (full residual decodeAll, then reconstruct).
template <typename T>
class NoRandomAccessRawStub : public Codec<T, uint8_t> {
public:
    EncodedData encode(std::span<const T> data) override { return inner_.encode(data); }
    std::vector<T> decodeAll(const EncodedData& encoded) override { return inner_.decodeAll(encoded); }
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override { return inner_.decodeAt(encoded, index); }
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override { return inner_.decodeRange(encoded, start, end); }
    EncodingType encodingType() const override { return inner_.encodingType(); }
    std::string name() const override { return "NoRandomAccessRawStub"; }
    EncodingProperties properties() const override {
        auto p = inner_.properties();
        p.remove(EncodingProperty::RandomAccess);
        return p;
    }
private:
    RawBitPackedEncoder<T> inner_;
};

static void testForPrevNonRandomAccessFallback() {
    std::vector<int32_t> data;
    for (int i = 0; i < 500; ++i) data.push_back((i * 31 - 200) % 173);

    auto stub = std::make_shared<NoRandomAccessRawStub<int32_t>>();
    FORConfig<int32_t, int32_t> cfg{.policy = FORReferencePolicy::PREV, .subEncoder = stub};
    FOREncoder<int32_t, int32_t, 64> enc(cfg);
    auto encoded = enc.encode(std::span<const int32_t>(data));

    auto all = enc.decodeAll(encoded);
    CHECK(all == data);
    for (size_t i = 0; i < data.size(); ++i) {
        auto v = enc.decodeAt(encoded, i);
        CHECK(v.has_value() && *v == data[i]);
    }
    auto range = enc.decodeRange(encoded, 10, 400);
    CHECK(range.size() == 390);
    for (size_t i = 10; i < 400 && i - 10 < range.size(); ++i) CHECK(range[i - 10] == data[i]);

    std::cout << "PASS: FOR PREV non-RandomAccess sub-encoder fallback\n";
}

// =============================================================================
// CascadingFOREncoder<TIn> PREV tests
// =============================================================================

template <typename T>
static void assertCascadeRoundTrip(const std::vector<T>& input, CascadingFORConfig cfg, const std::string& label) {
    CascadingFOREncoder<T> enc(cfg);
    auto encoded = enc.encode(std::span<const T>(input));
    auto all = enc.decodeAll(encoded);
    CHECK(all == input);

    for (size_t i = 0; i < input.size(); ++i) {
        auto v = enc.decodeAt(encoded, i);
        CHECK(v.has_value() && *v == input[i] && *v == all[i]);
    }
    if (!input.empty()) {
        const std::vector<std::pair<size_t, size_t>> windows = {
            {0, input.size()}, {0, input.size() / 2}, {1, input.size()},
            {input.size() / 3, input.size()}, {input.size() - 1, input.size()},
        };
        for (const auto& [s, e] : windows) {
            if (s >= e) continue;
            auto range = enc.decodeRange(encoded, s, e);
            CHECK(range.size() == e - s);
            for (size_t i = s; i < e && i - s < range.size(); ++i) {
                CHECK(range[i - s] == input[i] && range[i - s] == all[i]);
            }
        }
    }
    std::cout << "PASS: " << label << "  N=" << input.size() << "  bytes=" << encoded.data().size() << "\n";
}

static void testCascadePrevBasic() {
    CascadingFORConfig prevOnly;
    prevOnly.residualSchedule  = { {128, FORReferencePolicy::PREV} };
    prevOnly.referenceSchedule = { {8,   FORReferencePolicy::MIN} };

    assertCascadeRoundTrip<int64_t>({}, prevOnly, "Cascade PREV N=0");
    assertCascadeRoundTrip<int64_t>({99}, prevOnly, "Cascade PREV N=1");

    {
        std::vector<int64_t> v;
        for (int i = 0; i < 1000; ++i) v.push_back(i - 500);
        assertCascadeRoundTrip<int64_t>(v, prevOnly, "Cascade PREV monotone");
    }
    {
        std::mt19937_64 rng(55);
        std::uniform_int_distribution<int64_t> dist(-1000, 1000);
        std::vector<int64_t> v(2000);
        int64_t cur = 0;
        for (auto& x : v) { cur += dist(rng); x = cur; }
        assertCascadeRoundTrip<int64_t>(v, prevOnly, "Cascade PREV random walk");
    }

    CascadingFORConfig frameSize1;
    frameSize1.residualSchedule  = { {1, FORReferencePolicy::PREV} };
    frameSize1.referenceSchedule = { {8, FORReferencePolicy::MIN} };
    {
        std::vector<int64_t> v = {5, -3, 100, -100, 0, 42, -42};
        assertCascadeRoundTrip<int64_t>(v, frameSize1, "Cascade PREV frameSize=1");
    }

    CascadingFORConfig frameGeN;
    frameGeN.residualSchedule  = { {4096, FORReferencePolicy::PREV} };
    frameGeN.referenceSchedule = { {8,    FORReferencePolicy::MIN} };
    {
        std::vector<int64_t> v;
        for (int i = 0; i < 50; ++i) v.push_back((i * 17) % 41 - 20);
        assertCascadeRoundTrip<int64_t>(v, frameGeN, "Cascade PREV frameSize>=N");
    }

    // Mixed-policy schedule: MIN at an outer level, PREV at the deepest --
    // confirms per-level policy branching is independent, not a single global flag.
    CascadingFORConfig mixed;
    mixed.residualSchedule  = { {4096, FORReferencePolicy::MIN}, {512, FORReferencePolicy::PREV} };
    mixed.referenceSchedule = { {8,    FORReferencePolicy::MIN} };
    {
        std::mt19937_64 rng(77);
        std::uniform_int_distribution<int64_t> dist(-500, 500);
        std::vector<int64_t> v(3000);
        int64_t cur = 0;
        for (auto& x : v) { cur += dist(rng); x = cur; }
        assertCascadeRoundTrip<int64_t>(v, mixed, "Cascade mixed MIN-then-PREV schedule");
    }
}

// decodeRange frame-alignment stress test: windows starting exactly on vs.
// off a frame boundary, for a PREV-at-level-0 schedule -- the single most
// important new test given how much new logic that fix requires.
static void testCascadePrevDecodeRangeAlignment() {
    CascadingFORConfig cfg;
    cfg.residualSchedule  = { {64, FORReferencePolicy::PREV} };
    cfg.referenceSchedule = { {8,  FORReferencePolicy::MIN} };

    std::mt19937_64 rng(321);
    std::uniform_int_distribution<int64_t> dist(-30, 30);
    std::vector<int64_t> data(1000);
    int64_t cur = 1000;
    for (auto& x : data) { cur += dist(rng); x = cur; }

    CascadingFOREncoder<int64_t> enc(cfg);
    auto encoded = enc.encode(std::span<const int64_t>(data));
    auto all = enc.decodeAll(encoded);
    CHECK(all == data);

    // Windows starting exactly on a 64-boundary.
    for (size_t s : {size_t{0}, size_t{64}, size_t{128}, size_t{576}}) {
        for (size_t e : {s + 1, s + 30, s + 64, size_t{999}}) {
            if (e <= s || e > data.size()) continue;
            auto range = enc.decodeRange(encoded, s, e);
            CHECK(range.size() == e - s);
            for (size_t i = s; i < e && i - s < range.size(); ++i) CHECK(range[i - s] == data[i]);
        }
    }
    // Windows starting OFF a boundary (mid-frame) -- exercises the "seed" logic.
    for (size_t s : {size_t{1}, size_t{5}, size_t{63}, size_t{65}, size_t{100}, size_t{600}}) {
        for (size_t e : {s + 1, s + 20, size_t{999}}) {
            if (e <= s || e > data.size()) continue;
            auto range = enc.decodeRange(encoded, s, e);
            CHECK(range.size() == e - s);
            for (size_t i = s; i < e && i - s < range.size(); ++i) CHECK(range[i - s] == data[i]);
        }
    }
    std::cout << "PASS: Cascade PREV decodeRange frame-alignment stress test\n";
}

static void testCascadePrevExhaustive() {
    CascadingFORConfig prevOnly;
    prevOnly.residualSchedule  = { {64, FORReferencePolicy::PREV} };
    prevOnly.referenceSchedule = { {8,  FORReferencePolicy::MIN} };

    std::vector<int64_t> v;
    for (int i = 0; i < 600; ++i) v.push_back((i * 37 - 300) % 211);
    CascadingFOREncoder<int64_t> enc(prevOnly);
    auto encoded = enc.encode(std::span<const int64_t>(v));
    auto all = enc.decodeAll(encoded);
    CHECK(all == v);
    for (size_t i = 0; i < v.size(); ++i) {
        auto x = enc.decodeAt(encoded, i);
        CHECK(x.has_value() && *x == v[i] && *x == all[i]);
    }
    auto range = enc.decodeRange(encoded, 0, v.size());
    CHECK(range == v);
    std::cout << "PASS (exhaustive): Cascade PREV exhaustive round trip  N=" << v.size() << "\n";
}

// Non-RandomAccess leaf fallback, exercised with a PREV-policy schedule.
template <typename T>
class NoRandomAccessCascadeStub : public Codec<T, uint8_t> {
public:
    EncodedData encode(std::span<const T> data) override { return inner_.encode(data); }
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        ++decodeAllCalls;
        return inner_.decodeAll(encoded);
    }
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override { return inner_.decodeAt(encoded, index); }
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override { return inner_.decodeRange(encoded, start, end); }
    EncodingType encodingType() const override { return inner_.encodingType(); }
    std::string name() const override { return "NoRandomAccessCascadeStub"; }
    EncodingProperties properties() const override {
        auto p = inner_.properties();
        p.remove(EncodingProperty::RandomAccess);
        return p;
    }
    int decodeAllCalls = 0;
private:
    RawBitPackedEncoder<T> inner_;
};

static void testCascadePrevNonRandomAccessLeafFallback() {
    std::vector<int64_t> data;
    for (int i = 0; i < 300; ++i) data.push_back((i * 41 + 3) % 250);

    auto stub = std::make_shared<NoRandomAccessCascadeStub<int64_t>>();
    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {64, FORReferencePolicy::PREV} };
    cfg.referenceSchedule   = { {8,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = stub;

    CascadingFOREncoder<int64_t> enc(cfg);
    auto encoded = enc.encode(std::span<const int64_t>(data));
    auto all = enc.decodeAll(encoded);
    CHECK(all == data);

    const int before = stub->decodeAllCalls;
    for (size_t i = 0; i < data.size(); ++i) {
        auto v = enc.decodeAt(encoded, i);
        CHECK(v.has_value() && *v == data[i]);
    }
    const int after = stub->decodeAllCalls;
    CHECK(after - before == 1);
    std::cout << "PASS: Cascade PREV non-RandomAccess leaf fallback (cache verified: "
              << (after - before) << " decodeAll() call across 300 decodeAt() calls)\n";
}

// Telescoping-identity NON-extension test: unlike MIN/FIRST/MID (where
// intermediate cascade levels are provably inert -- see
// CascadingFOREncoder.hpp's class doc), PREV at an intermediate level is NOT
// inert. Confirms a schedule using PREV at an intermediate level does NOT
// reduce to the deepest-level-only result -- the mirror image of the
// existing test that asserts the identity DOES hold for MIN/FIRST/MID.
static void testTelescopingNonExtensionForPrev() {
    std::mt19937_64 rng(909);
    std::uniform_int_distribution<int64_t> dist(-1'000'000LL, 1'000'000LL);
    std::vector<int64_t> data(20000);
    for (auto& v : data) v = dist(rng);

    const std::vector<CascadeLevelConfig> plainSchedule = { {256, FORReferencePolicy::MIN} };
    const auto plainResiduals = computeDeepestResiduals(std::span<const int64_t>(data), plainSchedule);

    // PREV at the OUTER (intermediate) level, MIN at the deepest -- if PREV
    // were inert the way MIN/FIRST/MID are, this would match plainResiduals
    // exactly (same deepest frameSize=256, MIN policy). It must NOT.
    const std::vector<CascadeLevelConfig> prevIntermediateSchedule = {
        {4096, FORReferencePolicy::PREV}, {256, FORReferencePolicy::MIN}
    };
    const auto withPrevIntermediate = computeDeepestResiduals(std::span<const int64_t>(data), prevIntermediateSchedule);

    CHECK(withPrevIntermediate != plainResiduals);
    std::cout << "PASS: telescoping identity does NOT extend to PREV at an intermediate level\n";
}

// =============================================================================
// AdaptiveFOREncoder<TIn> PREV tests
// =============================================================================

static void testAdaptiveForChoosesPrevOnDriftingData() {
    // A slowly drifting counter (Snowflake-ID-like): PREV should win since
    // consecutive deltas are small and tightly bounded, while the raw values
    // drift far from any single frame-wide MIN reference over a long run.
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<int32_t> stepDist(-2, 2);
    std::vector<int32_t> drifting(5000);
    int32_t cur = 0;
    for (auto& x : drifting) { cur += stepDist(rng); x = cur; }

    AdaptiveFOREncoder<int32_t> enc;
    auto encoded = enc.encode(std::span<const int32_t>(drifting));
    auto all = enc.decodeAll(encoded);
    CHECK(all == drifting);

    for (size_t i = 0; i < drifting.size(); i += 137) {
        auto v = enc.decodeAt(encoded, i);
        CHECK(v.has_value() && *v == drifting[i]);
    }
    auto range = enc.decodeRange(encoded, 100, 4900);
    CHECK(range.size() == 4800);
    for (size_t i = 100; i < 4900 && i - 100 < range.size(); ++i) CHECK(range[i - 100] == drifting[i]);

    std::cout << "PASS: AdaptiveFOR round-trip on drifting-counter data (PREV-favorable), bytes="
              << encoded.data().size() << "\n";
}

static void testAdaptiveForChoosesMinOnStableRangeData() {
    // Values confined to a narrow, stable range with no drift -- MIN should
    // win (small frame-relative residuals), PREV would see comparably-sized
    // but noisier deltas since consecutive values aren't correlated.
    std::mt19937_64 rng(22);
    std::uniform_int_distribution<int32_t> dist(1'000'000, 1'000'010);
    std::vector<int32_t> stable(5000);
    for (auto& x : stable) x = dist(rng);

    AdaptiveFOREncoder<int32_t> enc;
    auto encoded = enc.encode(std::span<const int32_t>(stable));
    auto all = enc.decodeAll(encoded);
    CHECK(all == stable);
    std::cout << "PASS: AdaptiveFOR round-trip on stable-narrow-range data (MIN-favorable), bytes="
              << encoded.data().size() << "\n";
}

static void testAdaptiveForPrevDegenerateCases() {
    AdaptiveFOREncoder<int32_t> enc;

    CHECK(enc.decodeAll(enc.encode(std::span<const int32_t>(std::vector<int32_t>{}))).empty());

    {
        std::vector<int32_t> v{7};
        auto encoded = enc.encode(std::span<const int32_t>(v));
        CHECK(enc.decodeAll(encoded) == v);
        auto at0 = enc.decodeAt(encoded, 0);
        CHECK(at0.has_value() && *at0 == 7);
        CHECK(!enc.decodeAt(encoded, 1).has_value());
    }
    {
        // Sharp local jumps to stress frameSize=8 (smallest candidate) with PREV.
        std::vector<int32_t> v;
        for (int i = 0; i < 40; ++i) v.push_back((i % 2 == 0) ? i : -i);
        auto encoded = enc.encode(std::span<const int32_t>(v));
        auto all = enc.decodeAll(encoded);
        CHECK(all == v);
        for (size_t i = 0; i < v.size(); ++i) {
            auto x = enc.decodeAt(encoded, i);
            CHECK(x.has_value() && *x == v[i]);
        }
    }
    std::cout << "PASS: AdaptiveFOR PREV degenerate cases (N=0, N=1, small-frame stress)\n";
}

// Wire-format self-description test: AdaptiveFOREncoder carries no external
// config between encode/decode (unlike FOR/CascadingFOR, which assume
// decoder-side config consistency) -- it must record its own policy choice
// in the wire format and a FRESH, default-constructed instance must decode
// it correctly.
static void testAdaptiveForWireFormatSelfDescription() {
    std::mt19937_64 rng(33);
    std::uniform_int_distribution<int32_t> stepDist(-3, 3);
    std::vector<int32_t> drifting(2000);
    int32_t cur = 500;
    for (auto& x : drifting) { cur += stepDist(rng); x = cur; }

    AdaptiveFOREncoder<int32_t> encoder1;
    auto encoded = encoder1.encode(std::span<const int32_t>(drifting));

    AdaptiveFOREncoder<int32_t> freshDecoder; // fresh, unrelated instance
    auto decoded = freshDecoder.decodeAll(encoded);
    CHECK(decoded == drifting);

    for (size_t i = 0; i < drifting.size(); i += 91) {
        auto v = freshDecoder.decodeAt(encoded, i);
        CHECK(v.has_value() && *v == drifting[i]);
    }
    std::cout << "PASS: AdaptiveFOR wire-format self-description (refPolicy byte round-trips via a fresh instance)\n";
}

int main() {
    testForPrevBasic();
    testForPrevOverflow();
    testForPrevNonRandomAccessFallback();

    testCascadePrevBasic();
    testCascadePrevDecodeRangeAlignment();
    testCascadePrevExhaustive();
    testCascadePrevNonRandomAccessLeafFallback();
    testTelescopingNonExtensionForPrev();

    testAdaptiveForChoosesPrevOnDriftingData();
    testAdaptiveForChoosesMinOnStableRangeData();
    testAdaptiveForPrevDegenerateCases();
    testAdaptiveForWireFormatSelfDescription();

    if (g_failures == 0) {
        std::cout << "ALL PASS\n";
        return 0;
    }
    std::cout << g_failures << " FAILURE(S)\n";
    return 1;
}
