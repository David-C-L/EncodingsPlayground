#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "encoders/CascadingFOREncoder.hpp"
#include "encoders/analysis/CascadeCardinalityAnalyzer.hpp"

using namespace encodings;
using namespace encodings::encoders;
using namespace encodings::encoders::analysis;

template <typename T>
static void assertRoundTrip(const std::vector<T>& input, CascadingFORConfig cfg, const std::string& label) {
    CascadingFOREncoder<T> enc(cfg);
    auto encoded = enc.encode(std::span<const T>(input));

    const auto all = enc.decodeAll(encoded);
    if (all != input) {
        std::cerr << "FAIL decodeAll: " << label << "\n";
        assert(false);
    }

    if (!input.empty()) {
        const std::vector<size_t> idxs = {
            0, input.size() / 5, input.size() / 2,
            (input.size() * 4) / 5, input.size() - 1,
        };
        for (size_t i : idxs) {
            auto v = enc.decodeAt(encoded, i);
            if (!v.has_value() || *v != input[i]) {
                std::cerr << "FAIL decodeAt[" << i << "]: " << label << "\n";
                assert(false);
            }
        }

        const size_t rStart = input.size() / 4;
        const size_t rEnd   = (input.size() * 3) / 4;
        if (rStart < rEnd) {
            auto range = enc.decodeRange(encoded, rStart, rEnd);
            for (size_t i = rStart; i < rEnd; ++i) {
                if (range[i - rStart] != input[i]) {
                    std::cerr << "FAIL decodeRange[" << i << "]: " << label << "\n";
                    assert(false);
                }
            }
        }
    }

    std::cout << "PASS: " << label
              << "  N=" << input.size()
              << "  bytes=" << encoded.data().size();
    if (!input.empty()) {
        std::cout << "  ratio=" << static_cast<double>(encoded.data().size()) / (input.size() * sizeof(T));
    }
    std::cout << "\n";
}

// Exhaustive index/range coverage: every decodeAt(i) checked against BOTH
// ground truth (input) and decodeAll()'s own output, plus several decodeRange
// windows checked index-by-index -- stronger than assertRoundTrip's 5-sample
// spot check, exercising the new recursive decodeResidualLevelAt/
// decodeReferenceLevelAt implementation across every index, not just a
// handful.
template <typename T>
static void assertExhaustiveRoundTrip(const std::vector<T>& input, CascadingFORConfig cfg, const std::string& label) {
    CascadingFOREncoder<T> enc(cfg);
    auto encoded = enc.encode(std::span<const T>(input));
    const auto all = enc.decodeAll(encoded);
    if (all != input) {
        std::cerr << "FAIL exhaustive decodeAll: " << label << "\n";
        assert(false);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        auto v = enc.decodeAt(encoded, i);
        if (!v.has_value() || *v != input[i] || *v != all[i]) {
            std::cerr << "FAIL exhaustive decodeAt[" << i << "]: " << label << "\n";
            assert(false);
        }
    }
    if (!input.empty()) {
        const std::vector<std::pair<size_t, size_t>> windows = {
            {0, input.size()},
            {0, input.size() / 2},
            {input.size() / 4, (input.size() * 3) / 4},
            {input.size() - 1, input.size()},
        };
        for (const auto& [s, e] : windows) {
            if (s >= e) continue;
            auto range = enc.decodeRange(encoded, s, e);
            if (range.size() != e - s) {
                std::cerr << "FAIL exhaustive decodeRange size [" << s << "," << e << "): " << label << "\n";
                assert(false);
            }
            for (size_t i = s; i < e; ++i) {
                if (range[i - s] != input[i] || range[i - s] != all[i]) {
                    std::cerr << "FAIL exhaustive decodeRange[" << i << "]: " << label << "\n";
                    assert(false);
                }
            }
        }
    }
    std::cout << "PASS (exhaustive): " << label << "  N=" << input.size() << "\n";
}

// Test-local stub: wraps RawBitPackedEncoder<T> but reports NO RandomAccess,
// to exercise CascadingFOREncoder's leaf-level cached-fallback path
// (decodeLeafAt's "else" branch). Counts decodeAll() invocations so the test
// can confirm the cache actually caches: a single decodeAll() call should
// serve many subsequent decodeAt() calls on the same buffer, not one
// decodeAll() per decodeAt().
template <typename T>
class NoRandomAccessStub : public Codec<T, uint8_t> {
public:
    EncodedData encode(std::span<const T> data) override { return inner_.encode(data); }
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        ++decodeAllCalls;
        return inner_.decodeAll(encoded);
    }
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        return inner_.decodeAt(encoded, index);
    }
    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        return inner_.decodeRange(encoded, start, end);
    }
    EncodingType encodingType() const override { return inner_.encodingType(); }
    std::string name() const override { return "NoRandomAccessStub"; }
    EncodingProperties properties() const override {
        auto p = inner_.properties();
        p.remove(EncodingProperty::RandomAccess);
        return p;
    }

    int decodeAllCalls = 0;

private:
    RawBitPackedEncoder<T> inner_;
};

// Test: a non-RandomAccess residual leaf must still round-trip correctly via
// the cached-decodeAll fallback, and that fallback must actually cache (one
// decodeAll() call serving many decodeAt() calls on the same encoded buffer).
static void testNonRandomAccessLeafFallback() {
    std::vector<int64_t> data;
    for (int i = 0; i < 300; ++i) data.push_back((i * 37 + 5) % 400);

    auto stub = std::make_shared<NoRandomAccessStub<int64_t>>();

    CascadingFORConfig cfg;
    cfg.residualSchedule    = { {64, FORReferencePolicy::MIN} };
    cfg.referenceSchedule   = { {8,  FORReferencePolicy::MIN} };
    cfg.residualLeafEncoder = stub;
    // referenceLeafEncoder stays default (RawBitPackedEncoder, RandomAccess).

    CascadingFOREncoder<int64_t> enc(cfg);
    auto encoded = enc.encode(std::span<const int64_t>(data));

    const auto all = enc.decodeAll(encoded);
    if (all != data) {
        std::cerr << "FAIL non-RandomAccess leaf fallback: decodeAll mismatch\n";
        assert(false);
    }

    const int callsBeforeSweep = stub->decodeAllCalls;
    for (size_t i = 0; i < data.size(); ++i) {
        auto v = enc.decodeAt(encoded, i);
        if (!v.has_value() || *v != data[i]) {
            std::cerr << "FAIL non-RandomAccess leaf fallback: decodeAt[" << i << "] mismatch\n";
            assert(false);
        }
    }
    const int callsAfterSweep = stub->decodeAllCalls;
    // decodeAll() was already invoked once via enc.decodeAll(encoded) above,
    // and CascadingFOREncoder::decodeAll() doesn't share the decodeAt cache
    // (it recurses fresh each time) -- so the sweep of 300 decodeAt() calls
    // should add exactly one more decodeAllCalls increment (the first
    // decodeAt() populates the cache; the remaining 299 hit it), not 300.
    if (callsAfterSweep - callsBeforeSweep != 1) {
        std::cerr << "FAIL non-RandomAccess leaf fallback: expected exactly 1 decodeAll() "
                  << "call across the decodeAt sweep, got " << (callsAfterSweep - callsBeforeSweep) << "\n";
        assert(false);
    }

    std::cout << "PASS: non-RandomAccess leaf fallback (cache verified: "
              << (callsAfterSweep - callsBeforeSweep) << " decodeAll() call across 300 decodeAt() calls)\n";
}

// Test: properties() must report RandomAccess for the default (RawBitPackedEncoder
// leaves) config, and must NOT report it when either leaf role is a
// non-RandomAccess stub.
static void testPropertiesReporting() {
    CascadingFORConfig defaultCfg;
    defaultCfg.residualSchedule  = { {64} };
    defaultCfg.referenceSchedule = { {8} };
    CascadingFOREncoder<int64_t> defaultEnc(defaultCfg);
    assert(defaultEnc.properties().has(EncodingProperty::RandomAccess));

    CascadingFORConfig noResidualRA;
    noResidualRA.residualSchedule    = { {64} };
    noResidualRA.referenceSchedule   = { {8} };
    noResidualRA.residualLeafEncoder = std::make_shared<NoRandomAccessStub<int64_t>>();
    CascadingFOREncoder<int64_t> noResidualRAEnc(noResidualRA);
    assert(!noResidualRAEnc.properties().has(EncodingProperty::RandomAccess));

    CascadingFORConfig noReferenceRA;
    noReferenceRA.residualSchedule     = { {64} };
    noReferenceRA.referenceSchedule    = { {8} };
    noReferenceRA.referenceLeafEncoder = std::make_shared<NoRandomAccessStub<int64_t>>();
    CascadingFOREncoder<int64_t> noReferenceRAEnc(noReferenceRA);
    assert(!noReferenceRAEnc.properties().has(EncodingProperty::RandomAccess));

    std::cout << "PASS: properties() RandomAccess reporting\n";
}

// Test 2: analyzeCascade's per-level frame/element counts must match a
// hand-computed partitioning, independent of the encoder machinery.
static void testManualNestingEquivalence() {
    std::vector<int64_t> data;
    for (int i = 0; i < 1000; ++i) data.push_back((i * 131 + 7) % 5000);

    CascadingFORConfig cfg;
    cfg.residualSchedule = { {64}, {16} };
    cfg.referenceSchedule = { {4} };

    auto stats = analyzeCascade(std::span<const int64_t>(data), cfg);

    // Expect: residual level 0 (N=1000, frame=64, numFrames=16),
    //         reference level for residual-level-0's refs (N=16, frame=4, numFrames=4),
    //         residual level 1 (N=1000, frame=16, numFrames=63),
    //         reference level for residual-level-1's refs (N=63, frame=4, numFrames=16).
    assert(stats.size() == 4);

    assert(stats[0].role == CascadeStreamRole::Residual);
    assert(stats[0].levelIndex == 0);
    assert(stats[0].numElements == 1000);
    assert(stats[0].frameSize == 64);
    assert(stats[0].numFrames == (1000 + 63) / 64);

    assert(stats[1].role == CascadeStreamRole::Reference);
    assert(stats[1].parentResidualLevel.has_value() && *stats[1].parentResidualLevel == 0);
    assert(stats[1].numElements == (1000 + 63) / 64); // = numFrames of residual level 0
    assert(stats[1].frameSize == 4);

    assert(stats[2].role == CascadeStreamRole::Residual);
    assert(stats[2].levelIndex == 1);
    assert(stats[2].numElements == 1000);
    assert(stats[2].frameSize == 16);
    assert(stats[2].numFrames == (1000 + 15) / 16);

    assert(stats[3].role == CascadeStreamRole::Reference);
    assert(stats[3].parentResidualLevel.has_value() && *stats[3].parentResidualLevel == 1);
    assert(stats[3].numElements == (1000 + 15) / 16); // = numFrames of residual level 1

    std::cout << "PASS: manual nesting equivalence\n";
}

// Test 3: a stream with exactly K distinct values planted in every frame
// should report intraFrameDistinctMean ~= K.
static void testCardinalitySanity() {
    constexpr size_t kFrame = 100;
    constexpr size_t kPlantedDistinct = 7;
    constexpr size_t kFrames = 20;

    std::vector<int64_t> data;
    data.reserve(kFrame * kFrames);
    for (size_t f = 0; f < kFrames; ++f) {
        for (size_t i = 0; i < kFrame; ++i) {
            data.push_back(static_cast<int64_t>(i % kPlantedDistinct));
        }
    }

    CascadingFORConfig cfg;
    cfg.residualSchedule = { {kFrame} };
    cfg.referenceSchedule = {};

    auto stats = analyzeCascade(std::span<const int64_t>(data), cfg);
    assert(stats.size() == 1);
    const auto& s = stats[0];

    if (std::abs(s.intraFrameDistinctMean - static_cast<double>(kPlantedDistinct)) > 0.01) {
        std::cerr << "FAIL cardinality sanity: mean=" << s.intraFrameDistinctMean
                  << " expected=" << kPlantedDistinct << "\n";
        assert(false);
    }
    assert(s.intraFrameDistinctMin == kPlantedDistinct);
    assert(s.intraFrameDistinctMax == kPlantedDistinct);
    // Every frame repeats the same K values, so global distinct == K too.
    assert(s.globalExactDistinct == kPlantedDistinct);

    std::cout << "PASS: cardinality sanity (planted K=" << kPlantedDistinct << ")\n";
}

// Test 4: for cardinality well under MetricCollector's exact-count cap,
// the HLL estimate should be within its documented ~3.25% standard error
// (checked generously at 4 standard errors to avoid test flakiness).
static void testHllExactAgreement() {
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<int64_t> dist(0, 49999); // 50000 possible distinct values
    std::vector<int64_t> data(200000);
    for (auto& v : data) v = dist(rng);

    CascadingFORConfig cfg;
    cfg.residualSchedule = { {4096} };
    cfg.referenceSchedule = {};

    auto stats = analyzeCascade(std::span<const int64_t>(data), cfg);
    assert(stats.size() == 1);
    const auto& s = stats[0];
    assert(!s.globalDistinctCapped);

    const double relError = std::abs(s.globalHllEstimate - static_cast<double>(s.globalExactDistinct))
                             / static_cast<double>(s.globalExactDistinct);
    if (relError > 0.13) { // 4 * 3.25%
        std::cerr << "FAIL HLL/exact agreement: exact=" << s.globalExactDistinct
                  << " hll=" << s.globalHllEstimate << " relError=" << relError << "\n";
        assert(false);
    }
    std::cout << "PASS: HLL/exact agreement (exact=" << s.globalExactDistinct
              << " hll=" << s.globalHllEstimate << ")\n";
}

// Test 5: telescoping identity — under MIN/FIRST/MID policy with frame sizes
// that evenly divide each other, cascading through intermediate residual
// levels is a mathematical no-op: only the deepest level's frame size and
// policy determine the final residual values (see CascadingFOREncoder.hpp's
// "Telescoping identity" note). A depth-3 cascade [4096,1024,256] must
// produce byte-identical deepest residuals to a plain depth-1 [256] schedule,
// and this must hold regardless of which policy the intermediate levels use.
static void testTelescopingIdentity() {
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<int64_t> dist(-1'000'000LL, 1'000'000LL);
    std::vector<int64_t> data(50000);
    for (auto& v : data) v = dist(rng);

    const std::vector<CascadeLevelConfig> plainSchedule = { {256} };
    const auto plainResiduals = computeDeepestResiduals(std::span<const int64_t>(data), plainSchedule);

    const std::vector<std::vector<CascadeLevelConfig>> cascadedSchedules = {
        { {4096}, {1024}, {256} },                                              // all MIN (default)
        { {4096, FORReferencePolicy::FIRST}, {1024, FORReferencePolicy::MID}, {256} },  // mixed intermediate policies
        { {4096, FORReferencePolicy::MID}, {256} },                             // depth 2, still deepest=256
    };

    for (const auto& schedule : cascadedSchedules) {
        const auto cascadedResiduals = computeDeepestResiduals(std::span<const int64_t>(data), schedule);
        if (cascadedResiduals != plainResiduals) {
            std::cerr << "FAIL telescoping identity: cascaded schedule (depth=" << schedule.size()
                      << ") diverged from plain depth-1 schedule at the same deepest frame size\n";
            assert(false);
        }
    }

    std::cout << "PASS: telescoping identity (intermediate levels/policies are inert for residual values)\n";
}

int main() {
    CascadingFORConfig deep;
    deep.residualSchedule  = { {4096}, {1024}, {256}, {64} };
    deep.referenceSchedule = { {8} };

    CascadingFORConfig shallow; // depth 0: straight to leaf encoders on both streams

    CascadingFORConfig oddFrames; // non-multiple-of-frame, non-power-of-two frame sizes
    oddFrames.residualSchedule  = { {7}, {3} };
    oddFrames.referenceSchedule = { {2} };

    assertRoundTrip<int32_t>({}, deep, "N=0 deep");
    assertRoundTrip<int32_t>({42}, deep, "N=1 deep");
    assertRoundTrip<int32_t>({1, 2, 3, 4, 5}, deep, "N<frame deep");
    assertRoundTrip<int32_t>(std::vector<int32_t>(10, 5), shallow, "depth0 constant");

    {
        std::vector<int32_t> v;
        for (int i = 0; i < 100; ++i) v.push_back(i - 50);
        assertRoundTrip<int32_t>(v, oddFrames, "non-pow2 frames monotone");
    }

    assertRoundTrip<int64_t>(std::vector<int64_t>(5000, -123456789LL), deep, "all-equal int64");

    {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                                      std::numeric_limits<int64_t>::max());
        std::vector<int64_t> v(3000);
        for (auto& x : v) x = dist(rng);
        assertRoundTrip<int64_t>(v, deep, "full-range random int64");
    }

    {
        std::vector<int32_t> v;
        for (int i = 0; i < 2000; ++i) v.push_back(-1000000 + i * 3);
        assertRoundTrip<int32_t>(v, deep, "negative monotone int32");
    }

    {
        std::mt19937_64 rng(7);
        std::uniform_int_distribution<int64_t> dist(1'000'000'000LL, 1'000'000'100LL);
        std::vector<int64_t> v(10000);
        for (auto& x : v) x = dist(rng);
        assertRoundTrip<int64_t>(v, deep, "small-range large-base int64");
    }

    testManualNestingEquivalence();
    testCardinalitySanity();
    testHllExactAgreement();
    testTelescopingIdentity();

    // Exhaustive decodeAt/decodeRange coverage (every index, not just 5 sample
    // points) for the true recursive random-access implementation, across
    // deep/shallow/oddFrames at sizes small enough to keep the sweep fast.
    {
        std::vector<int32_t> v;
        for (int i = 0; i < 500; ++i) v.push_back((i * 37 - 250) % 613);
        assertExhaustiveRoundTrip<int32_t>(v, deep, "exhaustive deep");
    }
    {
        std::vector<int32_t> v;
        for (int i = 0; i < 500; ++i) v.push_back((i * 17 + 3) % 97);
        assertExhaustiveRoundTrip<int32_t>(v, shallow, "exhaustive depth0 (shallow)");
    }
    {
        std::vector<int32_t> v;
        for (int i = 0; i < 500; ++i) v.push_back(i - 250);
        assertExhaustiveRoundTrip<int32_t>(v, oddFrames, "exhaustive non-pow2 frames");
    }
    assertExhaustiveRoundTrip<int32_t>({}, deep, "exhaustive N=0");
    assertExhaustiveRoundTrip<int32_t>({7}, deep, "exhaustive N=1");

    testNonRandomAccessLeafFallback();
    testPropertiesReporting();

    std::cout << "ALL PASS\n";
    return 0;
}
