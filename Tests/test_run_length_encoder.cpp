#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "encoders/RunLengthEncoder.hpp"
#include "encoders/SubIntEncodingUtils.hpp"
#include "GatherTestHelpers.hpp"

using namespace encodings;
using namespace encodings::encoders;

// ---------------------------------------------------------------------------
// Test-local stub: wraps an existing ISectionCodecIntegral<uint64_t> but
// reports NO RandomAccess, to exercise RunLengthEncoder's findRunViaCodec
// cached-fallback path. Counts decodeAll() invocations so tests can confirm
// the cache actually caches (one decodeAll() serving many probes/decodeAt
// calls on the same buffer, not one decodeAll() per call).
// ---------------------------------------------------------------------------
class NoRandomAccessCodecWrapper final : public ISectionCodecIntegral<uint64_t> {
public:
    explicit NoRandomAccessCodecWrapper(std::shared_ptr<ISectionCodecIntegral<uint64_t>> inner)
        : inner_(std::move(inner)) {}

    EncodedBuffer<uint8_t> encode(std::span<const uint64_t> data) override {
        return inner_->encode(data);
    }
    std::vector<uint64_t> decodeAll(const EncodedBuffer<uint8_t>& enc) override {
        ++decodeAllCalls;
        return inner_->decodeAll(enc);
    }
    std::optional<uint64_t> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t idx) override {
        return inner_->decodeAt(enc, idx);
    }
    std::vector<uint64_t> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) override {
        return inner_->decodeRange(enc, start, end);
    }
    void decodeAllInto(const EncodedBuffer<uint8_t>& enc, uint64_t* dst, size_t n) override {
        inner_->decodeAllInto(enc, dst, n);
    }
    void decodeRangeInto(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end, uint64_t* dst, size_t n) override {
        inner_->decodeRangeInto(enc, start, end, dst, n);
    }
    EncodingProperties properties() const override {
        auto p = inner_->properties();
        p.remove(EncodingProperty::RandomAccess);
        return p;
    }
    std::string name() const override { return "NoRandomAccess(" + inner_->name() + ")"; }

    int decodeAllCalls = 0;

private:
    std::shared_ptr<ISectionCodecIntegral<uint64_t>> inner_;
};

// Exhaustive round-trip: decodeAll, every decodeAt(i), and several decodeRange
// windows, all checked against ground truth (input) directly.
template <typename T>
static void assertRoundTrip(RunLengthConfig<T> cfg, const std::vector<T>& input, const std::string& label) {
    RunLengthEncoder<T> enc(std::move(cfg));
    auto encoded = enc.encode(std::span<const T>(input));

    const auto all = enc.decodeAll(encoded);
    if (all != input) {
        std::cerr << "FAIL decodeAll: " << label << "\n";
        assert(false);
    }

    for (size_t i = 0; i < input.size(); ++i) {
        auto v = enc.decodeAt(encoded, i);
        if (!v.has_value() || *v != input[i]) {
            std::cerr << "FAIL decodeAt[" << i << "]: " << label << "\n";
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
                std::cerr << "FAIL decodeRange size [" << s << "," << e << "): " << label << "\n";
                assert(false);
            }
            for (size_t i = s; i < e; ++i) {
                if (range[i - s] != input[i]) {
                    std::cerr << "FAIL decodeRange[" << i << "]: " << label << "\n";
                    assert(false);
                }
            }
        }

        // decodeAllInto / decodeRangeInto
        std::vector<T> intoAll(input.size());
        enc.decodeAllInto(encoded, intoAll.data(), intoAll.size());
        if (intoAll != input) {
            std::cerr << "FAIL decodeAllInto: " << label << "\n";
            assert(false);
        }
        const size_t rs = input.size() / 4, re = (input.size() * 3) / 4;
        if (rs < re) {
            std::vector<T> intoRange(re - rs);
            enc.decodeRangeInto(encoded, rs, re, intoRange.data(), intoRange.size());
            for (size_t i = rs; i < re; ++i) {
                if (intoRange[i - rs] != input[i]) {
                    std::cerr << "FAIL decodeRangeInto[" << i << "]: " << label << "\n";
                    assert(false);
                }
            }
        }

        // decodeGatherInto -- exercises both the composed and non-composed
        // paths via whichever cfg this call site configured, across a
        // representative set of RowRangeList trace shapes.
        if (!encodings::testutil::checkGatherAllTraceShapes<T>(enc, encoded, input)) {
            std::cerr << "FAIL decodeGatherInto: " << label << "\n";
            assert(false);
        }
    }

    std::cout << "PASS: " << label << "  N=" << input.size()
              << "  bytes=" << encoded.data().size() << "\n";
}

// Test: with both factories unset, the wire format must be byte-identical to
// the pre-composition raw layout: [numRuns:8][runStartsSize:8][runValuesSize:8]
// [runStarts: numRuns*sizeof(size_t)][runValues: numRuns*sizeof(T)].
static void testRawWireFormatUnchanged() {
    std::vector<uint32_t> input = {5, 5, 5, 7, 7, 9};
    RunLengthEncoder<uint32_t> enc;
    auto encoded = enc.encode(std::span<const uint32_t>(input));

    // Runs: (start=0,val=5), (start=3,val=7), (start=5,val=9) -> numRuns=3
    const size_t numRuns = 3;
    const size_t runStartsSize = numRuns * sizeof(size_t);
    const size_t runValuesSize = numRuns * sizeof(uint32_t);
    const size_t expectedTotal = 3 * sizeof(size_t) + runStartsSize + runValuesSize;

    if (encoded.data().size() != expectedTotal) {
        std::cerr << "FAIL raw wire format: size mismatch, got " << encoded.data().size()
                  << " expected " << expectedTotal << "\n";
        assert(false);
    }

    const uint8_t* p = encoded.data().data();
    size_t gotNumRuns, gotStartsSize, gotValuesSize;
    std::memcpy(&gotNumRuns, p, sizeof(size_t)); p += sizeof(size_t);
    std::memcpy(&gotStartsSize, p, sizeof(size_t)); p += sizeof(size_t);
    std::memcpy(&gotValuesSize, p, sizeof(size_t));
    assert(gotNumRuns == numRuns);
    assert(gotStartsSize == runStartsSize);
    assert(gotValuesSize == runValuesSize);

    std::cout << "PASS: raw wire format unchanged (byte-identical to pre-composition layout)\n";
}

// Test: findRunViaCodec's binary search correctness in isolation, with
// hand-picked non-uniform run lengths, every index checked against a
// manually computed expected run index.
static void testNonUniformRunLengths() {
    // Runs of length 1, 100, 1, 100, 1 -> starts = {0, 1, 101, 102, 202}, N=203.
    std::vector<uint64_t> input;
    input.push_back(1);
    for (int i = 0; i < 100; ++i) input.push_back(2);
    input.push_back(3);
    for (int i = 0; i < 100; ++i) input.push_back(4);
    input.push_back(5);

    RunLengthConfig<uint64_t> cfg;
    cfg.runStartsFactory = [](uint8_t bits) {
        return detail_trisplit::makeCascadingFORSection<uint64_t>(bits);
    };
    RunLengthEncoder<uint64_t> enc(std::move(cfg));
    auto encoded = enc.encode(std::span<const uint64_t>(input));

    const std::vector<size_t> starts = {0, 1, 101, 102, 202};
    const std::vector<uint64_t> values = {1, 2, 3, 4, 5};
    for (size_t runIdx = 0; runIdx < starts.size(); ++runIdx) {
        const size_t runStart = starts[runIdx];
        const size_t runEnd = (runIdx + 1 < starts.size()) ? starts[runIdx + 1] : input.size();
        for (size_t i = runStart; i < runEnd; ++i) {
            auto v = enc.decodeAt(encoded, i);
            if (!v.has_value() || *v != values[runIdx]) {
                std::cerr << "FAIL non-uniform run lengths: decodeAt[" << i << "] expected "
                          << values[runIdx] << "\n";
                assert(false);
            }
        }
    }
    std::cout << "PASS: non-uniform run lengths (hand-verified run boundaries)\n";
}

// Test: a deliberately non-RandomAccess runStartsCodec must still round-trip
// correctly via the cached-decodeAll fallback, and that fallback must
// actually cache.
static void testNonRandomAccessRunStartsFallback() {
    std::vector<uint32_t> input;
    for (int i = 0; i < 200; ++i) input.push_back(static_cast<uint32_t>((i / 7) % 11));

    auto wrapper = std::make_shared<NoRandomAccessCodecWrapper>(
        detail_trisplit::makeRawBitPackedSection<uint64_t>(64));

    RunLengthConfig<uint32_t> cfg;
    cfg.runStartsFactory = [wrapper](uint8_t /*bits*/) -> std::shared_ptr<ISectionCodecIntegral<uint64_t>> {
        return wrapper;
    };
    RunLengthEncoder<uint32_t> enc(std::move(cfg));
    auto encoded = enc.encode(std::span<const uint32_t>(input));

    const auto all = enc.decodeAll(encoded);
    if (all != input) {
        std::cerr << "FAIL non-RandomAccess runStarts fallback: decodeAll mismatch\n";
        assert(false);
    }

    const int callsBefore = wrapper->decodeAllCalls;
    for (size_t i = 0; i < input.size(); ++i) {
        auto v = enc.decodeAt(encoded, i);
        if (!v.has_value() || *v != input[i]) {
            std::cerr << "FAIL non-RandomAccess runStarts fallback: decodeAt[" << i << "] mismatch\n";
            assert(false);
        }
    }
    const int callsAfter = wrapper->decodeAllCalls;
    if (callsAfter - callsBefore != 1) {
        std::cerr << "FAIL non-RandomAccess runStarts fallback: expected exactly 1 decodeAll() "
                  << "call across the decodeAt sweep, got " << (callsAfter - callsBefore) << "\n";
        assert(false);
    }

    std::cout << "PASS: non-RandomAccess runStarts fallback (cache verified: "
              << (callsAfter - callsBefore) << " decodeAll() call across "
              << input.size() << " decodeAt() calls)\n";
}

int main() {
    // --- Default (raw) path: all-equal, single element, empty ---
    assertRoundTrip<int64_t>({}, std::vector<int64_t>(256, 42LL), "default raw: all-equal int64");
    assertRoundTrip<int64_t>({}, {999LL}, "default raw: single element");
    assertRoundTrip<int64_t>({}, {}, "default raw: empty");

    // --- Default (raw) path: every-element-a-new-run (worst case, numRuns=N) ---
    {
        std::vector<int32_t> v;
        for (int i = 0; i < 200; ++i) v.push_back(i);
        assertRoundTrip<int32_t>({}, v, "default raw: every-element-new-run");
    }

    // --- Default (raw) path: skewed distribution ---
    {
        std::mt19937 rng(17);
        std::vector<int64_t> v(4000);
        std::uniform_int_distribution<int> which(0, 99);
        std::uniform_int_distribution<int64_t> rnd(0, 1000);
        for (auto& x : v) {
            int w = which(rng);
            if (w < 90) x = 42;
            else if (w < 99) x = 43;
            else x = rnd(rng);
        }
        assertRoundTrip<int64_t>({}, v, "default raw: skewed 90/9/1 split");
    }

    testRawWireFormatUnchanged();

    // --- runStartsFactory = CascadingFOR: same coverage as the raw path ---
    auto cascadingFORStarts = [](uint8_t bits) {
        return detail_trisplit::makeCascadingFORSection<uint64_t>(bits);
    };

    {
        RunLengthConfig<int64_t> cfg;
        cfg.runStartsFactory = cascadingFORStarts;
        assertRoundTrip<int64_t>(std::move(cfg), std::vector<int64_t>(256, 42LL),
                                  "CascadingFOR runStarts: all-equal int64");
    }
    {
        RunLengthConfig<int64_t> cfg;
        cfg.runStartsFactory = cascadingFORStarts;
        assertRoundTrip<int64_t>(std::move(cfg), {999LL}, "CascadingFOR runStarts: single element");
    }
    {
        RunLengthConfig<int64_t> cfg;
        cfg.runStartsFactory = cascadingFORStarts;
        assertRoundTrip<int64_t>(std::move(cfg), {}, "CascadingFOR runStarts: empty");
    }
    {
        RunLengthConfig<int32_t> cfg;
        cfg.runStartsFactory = cascadingFORStarts;
        std::vector<int32_t> v;
        for (int i = 0; i < 200; ++i) v.push_back(i);
        assertRoundTrip<int32_t>(std::move(cfg), v, "CascadingFOR runStarts: every-element-new-run");
    }
    {
        RunLengthConfig<int64_t> cfg;
        cfg.runStartsFactory = cascadingFORStarts;
        std::mt19937 rng(17);
        std::vector<int64_t> v(4000);
        std::uniform_int_distribution<int> which(0, 99);
        std::uniform_int_distribution<int64_t> rnd(0, 1000);
        for (auto& x : v) {
            int w = which(rng);
            if (w < 90) x = 42;
            else if (w < 99) x = 43;
            else x = rnd(rng);
        }
        assertRoundTrip<int64_t>(std::move(cfg), v, "CascadingFOR runStarts: skewed 90/9/1 split");
    }

    testNonUniformRunLengths();
    testNonRandomAccessRunStartsFallback();

    std::cout << "All RunLengthEncoder tests passed.\n";
    return 0;
}
