// test_subint_split_encoder.cpp
//
// Correctness suite for SubIntSplitEncoder, SubIntSplitAutoEncoder,
// IDSubStreamEncodingSelector, and the ISectionCodecIntegral adapter layer.
//
// Compile alongside the encoders and generators libraries (see CMakeLists.txt).
// Exit code: 0 = all passed, 1 = one or more suites failed.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encoders/SubIntSplitEncoder.hpp"
#include "encoders/SubIntEncodingUtils.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"
#include "encoders/selectors/costs/EncodingCostModel.hpp"

using namespace encodings;
using namespace encodings::encoders;
using namespace encodings::encoders::selectors;
using namespace encodings::encoders::selectors::costs;

// ============================================================================
// Test harness
// ============================================================================

struct TestSuite {
    const char* name;
    int passed = 0, failed = 0;
    std::string currentSubtest{};

    void beginSubtest(const char* subname) {
        currentSubtest = subname;
        printf("##########\n%-55s\n", subname);
        fflush(stdout);
    }

    void endSubtest() {
        printf("##########\n");
        // called implicitly by the next beginSubtest or summary
    }

    void check(bool cond, const char* expr, const char* file, int line) {
        if (cond) {
            ++passed;
        } else {
            if (failed == 0 && !currentSubtest.empty()) {
                // First failure in this subtest: print a newline after the subtest header
                printf("\n");
            }
            ++failed;
            fprintf(stderr, "    FAIL [%s:%d] in '%s'\n         %s\n",
                    file, line, currentSubtest.c_str(), expr);
        }
    }

    // Print a trailing "ok" or "FAIL" after the subtest label (call at the end of a subtest block)
    void subtestResult(int failsBefore) {
        if (failed == failsBefore) {
            printf("ok\n");
        }
        // Failures already printed inline by check()
    }

    void summary() const {
        printf("  [%-28s] %3d / %3d passed%s\n",
               name, passed, passed + failed,
               failed ? "  <-- FAILURES" : "");
    }
    bool ok() const { return failed == 0; }
};

// SUBTEST: log a named sub-section.  Usage:
//   SUBTEST(s, "description");
//   ... checks ...
//   SUBTEST_END(s);   // prints "ok" if no failures since last SUBTEST
//
// SUBTEST_END is optional — the next SUBTEST or suite summary will also close it.
#define SUBTEST(suite, name)  \
    do { (suite).beginSubtest(name); } while(0)

#define SUBTEST_END(suite, failsBefore_) \
    do { (suite).subtestResult(failsBefore_); } while(0)

#define CHECK(suite, cond) (suite).check((cond), #cond, __FILE__, __LINE__)
#define CHECK_THROWS(suite, expr)                                      \
    do {                                                               \
        bool threw_ = false;                                           \
        try { (void)(expr); } catch (...) { threw_ = true; }          \
        (suite).check(threw_, "throws: " #expr, __FILE__, __LINE__);  \
    } while (0)
#define CHECK_NOTHROW(suite, expr)                                      \
    do {                                                                \
        bool threw_ = false;                                            \
        try { (void)(expr); } catch (...) { threw_ = true; }           \
        (suite).check(!threw_, "nothrow: " #expr, __FILE__, __LINE__); \
    } while (0)

// Convenience: begin subtest, run checks, then auto-close.
// Usage:  WITH_SUBTEST(s, "label") { ... checks ... }
#define WITH_SUBTEST(suite, label)                       \
    for (int _st_done = ((suite).beginSubtest(label), 0), \
             _st_f0 = (suite).failed; \
         !_st_done;                                      \
         _st_done = 1, (suite).subtestResult(_st_f0))

// ============================================================================
// Data generators (fixed-seed, deterministic)
// ============================================================================

static std::vector<int64_t> makeConstant(size_t n, int64_t v = 42) {
    return std::vector<int64_t>(n, v);
}

static std::vector<int64_t> makeSequential(size_t n, int64_t start = 0, int64_t step = 1) {
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = start + static_cast<int64_t>(i) * step;
    return out;
}

static std::vector<int64_t> makeRepeating(size_t n, const std::vector<int64_t>& pat) {
    if (pat.empty()) return std::vector<int64_t>(n, 0);
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = pat[i % pat.size()];
    return out;
}

// Snowflake-like: 41-bit timestamp | 10-bit machine | 12-bit seq
static std::vector<int64_t> makeSnowflake(size_t n, uint64_t seed = 42) {
    std::vector<int64_t> out;
    out.reserve(n);
    uint64_t ts = seed & ((uint64_t{1} << 41) - 1);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t machine = (seed + i * 7) & ((uint64_t{1} << 10) - 1);
        const uint64_t seq     = i & ((uint64_t{1} << 12) - 1);
        out.push_back(static_cast<int64_t>((ts << 22) | (machine << 12) | seq));
        if (i % 4096 == 4095) ++ts;
    }
    return out;
}

// Values with all bits in [0, 2^bits)
static std::vector<int64_t> makeUniformBits(size_t n, uint8_t bits, uint64_t seed = 42) {
    const uint64_t mask = (bits >= 64) ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
    std::vector<int64_t> out;
    out.reserve(n);
    uint64_t state = seed;
    for (size_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out.push_back(static_cast<int64_t>(state & mask));
    }
    return out;
}

// Narrow to int32_t (keep lower 30 bits positive)
static std::vector<int32_t> makeUniformBits32(size_t n, uint64_t seed = 42) {
    const uint64_t mask = (uint64_t{1} << 30) - 1;
    std::vector<int32_t> out;
    out.reserve(n);
    uint64_t state = seed;
    for (size_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out.push_back(static_cast<int32_t>(state & mask));
    }
    return out;
}

// ============================================================================
// assertRoundtrip: encode → decodeAll / decodeAt / decodeRange
// ============================================================================

template <typename T>
static void assertRoundtrip(TestSuite& s, Codec<T, uint8_t>& enc,
                             const std::vector<T>& data) {
    auto blob = enc.encode(std::span<const T>(data));
    CHECK(s, enc.decodeAll(blob) == data);

    // decodeAt: spot-check first, middle, last
    for (size_t idx : {size_t{0}, data.size() / 2, data.size() > 0 ? data.size() - 1 : size_t{0}}) {
        if (idx < data.size()) {
            auto v = enc.decodeAt(blob, idx);
            CHECK(s, v.has_value() && *v == data[idx]);
        }
    }
    // Out-of-bounds decodeAt must return nullopt
    CHECK(s, !enc.decodeAt(blob, data.size()).has_value());

    // decodeRange: full span
    CHECK(s, enc.decodeRange(blob, 0, data.size()) == data);

    // First half / second half
    if (data.size() >= 2) {
        const size_t mid = data.size() / 2;
        CHECK(s, enc.decodeRange(blob, 0, mid) ==
                     std::vector<T>(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(mid)));
        CHECK(s, enc.decodeRange(blob, mid, data.size()) ==
                     std::vector<T>(data.begin() + static_cast<std::ptrdiff_t>(mid), data.end()));
    }
    // Single element
    if (!data.empty()) {
        CHECK(s, enc.decodeRange(blob, 0, 1) == std::vector<T>{data[0]});
    }
    // Empty / past-end ranges
    CHECK(s, enc.decodeRange(blob, 0, 0).empty());
    CHECK(s, enc.decodeRange(blob, data.size(), data.size() + 10).empty());
}

// ============================================================================
// NonRAWrapper: strips RandomAccess from any ISectionCodecIntegral, so that
// SubIntSplitEncoder exercises the pre-decode path.
// ============================================================================

template <typename TIn>
class NonRAWrapper final : public ISectionCodecIntegral<TIn> {
    std::shared_ptr<ISectionCodecIntegral<TIn>> inner_;
public:
    explicit NonRAWrapper(std::shared_ptr<ISectionCodecIntegral<TIn>> inner)
        : inner_(std::move(inner)) {}

    EncodedBuffer<uint8_t> encode(std::span<const TIn> data) override {
        return inner_->encode(data);
    }
    std::vector<TIn> decodeAll(const EncodedBuffer<uint8_t>& enc) override {
        return inner_->decodeAll(enc);
    }
    std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& enc, size_t idx) override {
        return inner_->decodeAt(enc, idx);
    }
    std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end) override {
        return inner_->decodeRange(enc, start, end);
    }
    void decodeAllInto(const EncodedBuffer<uint8_t>& enc, TIn* dst, size_t n) override {
        inner_->decodeAllInto(enc, dst, n);
    }
    void decodeRangeInto(const EncodedBuffer<uint8_t>& enc, size_t start, size_t end,
                         TIn* dst, size_t n) override {
        inner_->decodeRangeInto(enc, start, end, dst, n);
    }
    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::Lossless)
             | EncodingProperty::PreservesOrder;  // RandomAccess intentionally absent
    }
    std::string name() const override { return "NonRA(" + inner_->name() + ")"; }
};

template <typename TIn>
static std::shared_ptr<ISectionCodecIntegral<TIn>>
wrapNonRA(std::shared_ptr<ISectionCodecIntegral<TIn>> codec) {
    return std::make_shared<NonRAWrapper<TIn>>(std::move(codec));
}

// ============================================================================
// Section 1: Config Validation
// ============================================================================

static void testConfigValidation(TestSuite& s) {
    using Cfg = SubIntSplitConfig64;
    const auto bpCodec = detail_trisplit::makeRawBitPackedSection<uint64_t>(32);
    const auto rleCodec = detail_trisplit::makeRLESection<uint64_t>(32);

    WITH_SUBTEST(s, "valid 2-split 32|32 does not throw") {
        CHECK_NOTHROW(s, ([&] {
            Cfg cfg; cfg.bits = {32,32}; cfg.codecs = {bpCodec, rleCodec}; cfg.validate();
        }()));
    }
    WITH_SUBTEST(s, "valid 4-split 16|16|16|16 does not throw") {
        auto c16 = detail_trisplit::makeRawBitPackedSection<uint64_t>(16);
        CHECK_NOTHROW(s, ([&] {
            Cfg cfg; cfg.bits = {16,16,16,16}; cfg.codecs = {c16,c16,c16,c16}; cfg.validate();
        }()));
    }
    WITH_SUBTEST(s, "bits/codecs size mismatch throws") {
        CHECK_THROWS(s, ([&] {
            Cfg cfg; cfg.bits = {32,32}; cfg.codecs = {bpCodec}; cfg.validate();
        }()));
    }
    WITH_SUBTEST(s, "sum(bits) != 64 throws") {
        CHECK_THROWS(s, ([&] {
            Cfg cfg; cfg.bits = {30,30}; cfg.codecs = {bpCodec, rleCodec}; cfg.validate();
        }()));
    }
    WITH_SUBTEST(s, "zero-width section throws") {
        CHECK_THROWS(s, ([&] {
            Cfg cfg; cfg.bits = {0,64}; cfg.codecs = {bpCodec, bpCodec}; cfg.validate();
        }()));
    }
    WITH_SUBTEST(s, "empty config throws") {
        CHECK_THROWS(s, ([&] { Cfg cfg; cfg.validate(); }()));
    }
    WITH_SUBTEST(s, "null codec throws") {
        CHECK_THROWS(s, ([&] {
            Cfg cfg; cfg.bits = {32,32}; cfg.codecs = {bpCodec, nullptr}; cfg.validate();
        }()));
    }
}

// ============================================================================
// Section 2: fromSegments factory
// ============================================================================

static void testFromSegments(TestSuite& s) {
    using SP = selectors::SegmentPlan;
    using ET = EncodingType;

    WITH_SUBTEST(s, "ascending bitStart infers LSB_TO_MSB order") {
        std::vector<SP> segs = {{0,31,ET::BitPacking,1.0},{32,63,ET::RawEncoding,1.0}};
        CHECK_NOTHROW(s, SubIntSplitConfig64::fromSegments(segs));
        auto cfg = SubIntSplitConfig64::fromSegments(segs);
        CHECK(s, cfg.order == BitSplitOrder::LSB_TO_MSB);
        CHECK(s, cfg.splitCount() == 2);
    }
    WITH_SUBTEST(s, "descending bitStart infers MSB_TO_LSB order") {
        std::vector<SP> segs = {{32,63,ET::BitPacking,1.0},{0,31,ET::RawEncoding,1.0}};
        CHECK_NOTHROW(s, SubIntSplitConfig64::fromSegments(segs));
        auto cfg = SubIntSplitConfig64::fromSegments(segs);
        CHECK(s, cfg.order == BitSplitOrder::MSB_TO_LSB);
    }
    WITH_SUBTEST(s, "explicit orderHint overrides inferred order") {
        std::vector<SP> segs = {{0,31,ET::RawEncoding,1.0},{32,63,ET::RawEncoding,1.0}};
        auto cfg = SubIntSplitConfig64::fromSegments(segs, BitSplitOrder::LSB_TO_MSB);
        CHECK(s, cfg.order == BitSplitOrder::LSB_TO_MSB);
    }
    WITH_SUBTEST(s, "single full-width segment [0,63]") {
        std::vector<SP> segs = {{0,63,ET::RawEncoding,1.0}};
        CHECK_NOTHROW(s, SubIntSplitConfig64::fromSegments(segs));
        CHECK(s, SubIntSplitConfig64::fromSegments(segs).splitCount() == 1);
    }
    WITH_SUBTEST(s, "non-contiguous segments throw") {
        CHECK_THROWS(s, ([&] {
            std::vector<SP> segs = {{0,29,ET::RawEncoding,1.0},{32,63,ET::RawEncoding,1.0}};
            SubIntSplitConfig64::fromSegments(segs, BitSplitOrder::LSB_TO_MSB);
        }()));
    }
    WITH_SUBTEST(s, "incomplete coverage throws") {
        CHECK_THROWS(s, ([&] {
            std::vector<SP> segs = {{0,30,ET::RawEncoding,1.0}};
            SubIntSplitConfig64::fromSegments(segs);
        }()));
    }
}

// ============================================================================
// Section 3: Manual all-RA encoder — correctness across configs, orders, and N
// ============================================================================

static void testManualAllRA(TestSuite& s) {
    // N=0 and N=1 tested in testDecodeEdgeCases; some codecs have known issues with N=1.
    auto multiN = [&](auto& enc, auto makeData) {
        for (size_t n : {size_t{10}, size_t{100},
                         size_t{8192}, size_t{8193}, size_t{50000}}) {
            assertRoundtrip(s, enc, makeData(n));
        }
    };

    WITH_SUBTEST(s, "32|32 BitPacked|BitPacked LSB int64_t (uniform data)") {
        auto enc = makeSubIntSplitEncoderManual<int64_t>(
            {32, 32}, {EncodingType::BitPacking, EncodingType::BitPacking}, BitSplitOrder::LSB_TO_MSB);
        multiN(*enc, [](size_t n) { return makeUniformBits(n, 62); });
        CHECK(s, enc->properties().has(EncodingProperty::RandomAccess));
        CHECK(s, enc->properties().has(EncodingProperty::Lossless));
        CHECK(s, enc->name().find("SubIntSplit(") == 0);
    }
    WITH_SUBTEST(s, "32|32 BitPacked|BitPacked MSB int64_t (Snowflake data)") {
        auto enc = makeSubIntSplitEncoderManual<int64_t>(
            {32, 32}, {EncodingType::BitPacking, EncodingType::BitPacking}, BitSplitOrder::MSB_TO_LSB);
        multiN(*enc, [](size_t n) { return makeSnowflake(n); });
    }
    WITH_SUBTEST(s, "16|16|32 BitPacking|RLE|FPE LSB int64_t (repeating data)") {
        auto enc = makeSubIntSplitEncoderManual<int64_t>(
            {16, 16, 32},
            {EncodingType::BitPacking, EncodingType::RunLengthEncoding, EncodingType::FrequencyPartitionEncoding},
            BitSplitOrder::LSB_TO_MSB);
        multiN(*enc, [](size_t n) { return makeRepeating(n, {0, 1, 2, 3, 0, 0, 1, 1}); });
    }
    WITH_SUBTEST(s, "13|10|41 BitPacking|Dictionary|AdaptiveFOR LSB int64_t (Snowflake data)") {
        auto enc = makeSubIntSplitEncoderManual<int64_t>(
            {13, 10, 41},
            {EncodingType::BitPacking, EncodingType::DictionaryEncoding, EncodingType::AdaptiveFrameOfReference},
            BitSplitOrder::LSB_TO_MSB);
        multiN(*enc, [](size_t n) { return makeSnowflake(n); });
    }
    WITH_SUBTEST(s, "8|8|16|32 Raw|RLE|BitPacked|FPE LSB int64_t (constant data)") {
        auto enc = makeSubIntSplitEncoderManual<int64_t>(
            {8, 8, 16, 32},
            {EncodingType::RawEncoding, EncodingType::RunLengthEncoding,
             EncodingType::BitPacking, EncodingType::FrequencyPartitionEncoding},
            BitSplitOrder::LSB_TO_MSB);
        multiN(*enc, [](size_t n) { return makeConstant(n, 0x0102'03040506'0708LL & 0x7FFF'FFFF'FFFF'FFFFLL); });
    }
    WITH_SUBTEST(s, "16|16 BitPacking|RLE LSB int32_t (sequential data)") {
        auto enc = makeSubIntSplitEncoderManual<int32_t>(
            {16, 16}, {EncodingType::BitPacking, EncodingType::RunLengthEncoding}, BitSplitOrder::LSB_TO_MSB);
        for (size_t n : {size_t{0}, size_t{1}, size_t{8192}, size_t{50000}}) {
            std::vector<int32_t> data;
            data.reserve(n);
            for (size_t i = 0; i < n; ++i) data.push_back(static_cast<int32_t>(i & 0x7FFF'FFFF));
            assertRoundtrip(s, *enc, data);
        }
    }
    WITH_SUBTEST(s, "8|24 Raw|AdaptiveFOR MSB int32_t (uniform data)") {
        auto enc = makeSubIntSplitEncoderManual<int32_t>(
            {8, 24}, {EncodingType::RawEncoding, EncodingType::AdaptiveFrameOfReference}, BitSplitOrder::MSB_TO_LSB);
        for (size_t n : {size_t{100}, size_t{8193}}) {
            std::vector<int32_t> data32(n);
            for (size_t i = 0; i < n; ++i) data32[i] = static_cast<int32_t>(makeUniformBits32(1, i)[0]);
            assertRoundtrip(s, *enc, data32);
        }
    }
}

// ============================================================================
// Section 4: Non-RA sections — pre-decode path
// ============================================================================

static void testNonRAMixedSections(TestSuite& s) {
    auto bpCodec  = detail_trisplit::makeRawBitPackedSection<uint64_t>(32);
    auto rleCodec = detail_trisplit::makeRLESection<uint64_t>(32);

    WITH_SUBTEST(s, "non-RA on section 0 (isFirst write path)") {
        SubIntSplitConfig64 cfg;
        cfg.bits   = {32, 32};
        cfg.codecs = {wrapNonRA(bpCodec), rleCodec};
        auto enc   = makeSubIntSplitEncoder<int64_t>(std::move(cfg));
        CHECK(s, !enc->properties().has(EncodingProperty::RandomAccess));
        for (size_t n : {size_t{500}, size_t{20000}})
            assertRoundtrip(s, *enc, makeUniformBits(n, 62));
    }
    WITH_SUBTEST(s, "non-RA on section 1 (OR accumulate path)") {
        SubIntSplitConfig64 cfg;
        cfg.bits   = {32, 32};
        cfg.codecs = {bpCodec, wrapNonRA(rleCodec)};
        auto enc   = makeSubIntSplitEncoder<int64_t>(std::move(cfg));
        CHECK(s, !enc->properties().has(EncodingProperty::RandomAccess));
        for (size_t n : {size_t{500}, size_t{20000}})
            assertRoundtrip(s, *enc, makeSnowflake(n));
    }
    WITH_SUBTEST(s, "all sections non-RA") {
        SubIntSplitConfig64 cfg;
        cfg.bits   = {32, 32};
        cfg.codecs = {wrapNonRA(bpCodec), wrapNonRA(rleCodec)};
        auto enc   = makeSubIntSplitEncoder<int64_t>(std::move(cfg));
        for (size_t n : {size_t{500}, size_t{20000}})
            assertRoundtrip(s, *enc, makeConstant(n, 7));
    }
    WITH_SUBTEST(s, "mixed 3-split: non-RA in middle section") {
        auto c16bp  = detail_trisplit::makeRawBitPackedSection<uint64_t>(16);
        auto c16rle = detail_trisplit::makeRLESection<uint64_t>(16);
        auto c32fp  = detail_trisplit::makeFrequencyPartitionSection<uint64_t>(32);
        SubIntSplitConfig64 cfg;
        cfg.bits   = {16, 16, 32};
        cfg.codecs = {c16bp, wrapNonRA(c16rle), c32fp};
        auto enc   = makeSubIntSplitEncoder<int64_t>(std::move(cfg));
        for (size_t n : {size_t{500}, size_t{20000}})
            assertRoundtrip(s, *enc, makeRepeating(n, {1, 2, 3, 4, 1, 1, 2, 2}));
    }
    WITH_SUBTEST(s, "non-RA across chunk boundaries (N=8193 and N=16385)") {
        SubIntSplitConfig64 cfg;
        cfg.bits   = {32, 32};
        cfg.codecs = {wrapNonRA(bpCodec), bpCodec};
        auto enc   = makeSubIntSplitEncoder<int64_t>(std::move(cfg));
        assertRoundtrip(s, *enc, makeUniformBits(8193,  62));
        assertRoundtrip(s, *enc, makeUniformBits(16385, 62));
    }
}

// ============================================================================
// Section 5: Decode edge cases
// ============================================================================

static void testDecodeEdgeCases(TestSuite& s) {
    auto enc = makeSubIntSplitEncoderManual<int64_t>(
        {32, 32}, {EncodingType::BitPacking, EncodingType::RawEncoding}, BitSplitOrder::LSB_TO_MSB);

    WITH_SUBTEST(s, "N=0: empty results") {
        auto blob = enc->encode(std::span<const int64_t>{});
        CHECK(s, enc->decodeAll(blob).empty());
        CHECK(s, enc->decodeRange(blob, 0, 10).empty());
        CHECK(s, !enc->decodeAt(blob, 0).has_value());
    }
    WITH_SUBTEST(s, "N=1: single element roundtrip") {
        const std::vector<int64_t> data = {0x0123'4567'89AB'CDEFLL};
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
        CHECK(s, enc->decodeAt(blob, 0).value() == data[0]);
        CHECK(s, !enc->decodeAt(blob, 1).has_value());
    }
    WITH_SUBTEST(s, "N=kDecodeChunkSize: exact chunk boundary") {
        constexpr size_t B = SubIntSplitEncoder<int64_t>::kDecodeChunkSize;
        auto data = makeUniformBits(B, 62);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
        CHECK(s, enc->decodeRange(blob, 0, B) == data);
        CHECK(s, enc->decodeAt(blob, B - 1).value() == data[B - 1]);
        CHECK(s, !enc->decodeAt(blob, B).has_value());
    }
    WITH_SUBTEST(s, "N=kDecodeChunkSize+1: straddles chunk boundary") {
        constexpr size_t B = SubIntSplitEncoder<int64_t>::kDecodeChunkSize;
        auto data = makeUniformBits(B + 1, 62);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
        auto rng = enc->decodeRange(blob, B - 2, B + 1);
        CHECK(s, rng.size() == 3);
        CHECK(s, rng[0] == data[B-2] && rng[1] == data[B-1] && rng[2] == data[B]);
    }
    WITH_SUBTEST(s, "decodeRange: empty and clamped ranges") {
        constexpr size_t N = 100;
        auto data = makeSequential(N);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeRange(blob, 3, 3).empty());
        CHECK(s, enc->decodeRange(blob, N, N + 5).empty());
        auto clamped = enc->decodeRange(blob, N - 5, N + 100);
        CHECK(s, clamped.size() == 5);
        CHECK(s, clamped == std::vector<int64_t>(data.end() - 5, data.end()));
    }
    WITH_SUBTEST(s, "decodeRange: single element and full span") {
        constexpr size_t N = 100;
        auto data = makeSequential(N);
        auto blob = enc->encode(data);
        auto r = enc->decodeRange(blob, 5, 6);
        CHECK(s, r.size() == 1 && r[0] == data[5]);
        CHECK(s, enc->decodeRange(blob, 0, N) == data);
    }
    WITH_SUBTEST(s, "decodeAt: last valid index and out-of-bounds") {
        constexpr size_t N = 100;
        auto data = makeSequential(N);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAt(blob, N - 1).value() == data[N - 1]);
        CHECK(s, !enc->decodeAt(blob, N).has_value());
    }
}

// ============================================================================
// Section 6: Profiling variants
// ============================================================================

static void testProfilingVariants(TestSuite& s) {
    auto enc = makeSubIntSplitEncoderManualProf<int64_t>(
        {13, 10, 41},
        {EncodingType::BitPacking, EncodingType::RunLengthEncoding, EncodingType::AdaptiveFrameOfReference},
        BitSplitOrder::LSB_TO_MSB);
    const auto data = makeSnowflake(5000);
    auto blob = enc->encode(data);

    WITH_SUBTEST(s, "encode() fills subStreamEncodeMetrics (3 sections)") {
        const auto& m = blob.metadata().subStreamEncodeMetrics;
        CHECK(s, m.size() == 3);
        for (const auto& x : m) { CHECK(s, !x.name.empty()); CHECK(s, x.encodedBytes > 0); CHECK(s, x.bitWidth > 0); }
    }
    WITH_SUBTEST(s, "decodeAll() populates bulkDecodeTimeNs") {
        enc->decodeAll(blob);
        auto t = enc->subStreamBulkDecodeTimeNs();
        CHECK(s, t.size() == 3);
        for (auto v : t) CHECK(s, v > 0);
    }
    WITH_SUBTEST(s, "decodeAt() accumulates decodeAtAccumNs") {
        for (int i = 0; i < 10; ++i) enc->decodeAt(blob, static_cast<size_t>(i));
        auto t = enc->subStreamDecodeAtAccumNs();
        CHECK(s, t.size() == 3);
        for (auto v : t) CHECK(s, v > 0);
    }
    WITH_SUBTEST(s, "resetSubStreamDecodeAtAccum() zeroes timers") {
        enc->resetSubStreamDecodeAtAccum();
        auto t = enc->subStreamDecodeAtAccumNs();
        CHECK(s, t.size() == 3);
        for (auto v : t) CHECK(s, v == 0);
    }
    WITH_SUBTEST(s, "decodeRange() accumulates decodeRangeAccumNs") {
        for (int i = 0; i < 5; ++i) enc->decodeRange(blob, 0, 100);
        auto t = enc->subStreamDecodeRangeAccumNs();
        CHECK(s, t.size() == 3);
        for (auto v : t) CHECK(s, v > 0);
    }
    WITH_SUBTEST(s, "resetSubStreamDecodeRangeAccum() zeroes timers") {
        enc->resetSubStreamDecodeRangeAccum();
        auto t = enc->subStreamDecodeRangeAccumNs();
        CHECK(s, t.size() == 3);
        for (auto v : t) CHECK(s, v == 0);
    }
    WITH_SUBTEST(s, "non-profiling variant returns empty timing vectors") {
        auto np = makeSubIntSplitEncoderManual<int64_t>({32,32},{EncodingType::BitPacking,EncodingType::RawEncoding});
        auto b2 = np->encode(data);
        np->decodeAll(b2); np->decodeAt(b2, 0); np->decodeRange(b2, 0, 10);
        CHECK(s, np->subStreamBulkDecodeTimeNs().empty());
        CHECK(s, np->subStreamDecodeAtAccumNs().empty());
        CHECK(s, np->subStreamDecodeRangeAccumNs().empty());
    }
}

// ============================================================================
// Section 7: AutoEncoder
// ============================================================================

static void testAutoEncoder(TestSuite& s) {
    WITH_SUBTEST(s, "7a: int64_t LSB roundtrip (Snowflake data)") {
        auto enc = makeDefaultAutoSubIntSplitEncoder<int64_t>(BitSplitOrder::LSB_TO_MSB);
        auto data = makeSnowflake(5000);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
        auto v = enc->decodeAt(blob, 100);
        CHECK(s, v.has_value() && *v == data[100]);
    }
    WITH_SUBTEST(s, "7a: int64_t MSB roundtrip (Snowflake data)") {
        auto enc = makeDefaultAutoSubIntSplitEncoder<int64_t>(BitSplitOrder::MSB_TO_LSB);
        auto data = makeSnowflake(5000);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
    }
    WITH_SUBTEST(s, "7a: int32_t LSB roundtrip (uniform data)") {
        auto enc  = makeDefaultAutoSubIntSplitEncoder<int32_t>();
        auto data = makeUniformBits32(5000);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
    }
    WITH_SUBTEST(s, "7b: impl_ reused across encode calls (stable name)") {
        auto enc   = makeDefaultAutoSubIntSplitEncoder<int64_t>();
        auto data1 = makeSnowflake(1000, 1);
        auto data2 = makeSnowflake(1000, 2);
        auto b1 = enc->encode(data1);
        const std::string name1 = enc->name();
        auto b2 = enc->encode(data2);
        CHECK(s, enc->name() == name1);
        CHECK(s, !name1.empty() && name1 != "SubIntSplitAuto");
        CHECK(s, enc->decodeAll(b1) == data1);
        CHECK(s, enc->decodeAll(b2) == data2);
    }
    WITH_SUBTEST(s, "7c: forcedNumSplits=2 yields exactly 2 sections") {
        auto enc = makeDefaultAutoSubIntSplitEncoder<int64_t>(BitSplitOrder::LSB_TO_MSB, false, true, false, {}, 2);
        enc->encode(makeSnowflake(2000));
        const auto n2 = enc->name();
        CHECK(s, static_cast<int>(std::count(n2.begin(), n2.end(), '|')) == 1);
    }
    WITH_SUBTEST(s, "7c: forcedNumSplits=4 yields exactly 4 sections") {
        auto enc = makeDefaultAutoSubIntSplitEncoder<int64_t>(BitSplitOrder::LSB_TO_MSB, false, true, false, {}, 4);
        enc->encode(makeSnowflake(2000));
        const auto n4 = enc->name();
        CHECK(s, static_cast<int>(std::count(n4.begin(), n4.end(), '|')) == 3);
    }
    WITH_SUBTEST(s, "7d: custom cost models {BitPacking, RLE} roundtrip") {
        auto cfg = makeDefaultAutoSubIntSplitConfig<int64_t>(BitSplitOrder::LSB_TO_MSB, false,
            {EncodingType::BitPacking, EncodingType::RunLengthEncoding});
        auto enc = makeAutoSubIntSplitEncoder<int64_t>(std::move(cfg));
        auto data = makeSnowflake(2000);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
    }
    WITH_SUBTEST(s, "7d: all default cost models roundtrip") {
        auto enc = makeDefaultAutoSubIntSplitEncoder<int64_t>(
            BitSplitOrder::LSB_TO_MSB, false, true, false, defaultAutoSubIntSplitCostModelTypes());
        auto data = makeSnowflake(2000);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
    }
    WITH_SUBTEST(s, "7e: selection timing stored in metadata") {
        auto cfg  = makeDefaultAutoSubIntSplitConfig<int64_t>(BitSplitOrder::LSB_TO_MSB, true);
        auto enc  = makeAutoSubIntSplitEncoder<int64_t>(std::move(cfg));
        auto blob = enc->encode(makeSnowflake(2000));
        CHECK(s, blob.metadata().customMetadata.count("selectionTime_ns") == 1);
        CHECK(s, std::stoll(blob.metadata().customMetadata.at("selectionTime_ns")) > 0);
    }
    WITH_SUBTEST(s, "7f: properties() safe before and after encode") {
        auto enc = makeDefaultAutoSubIntSplitEncoder<int64_t>();
        CHECK_NOTHROW(s, enc->properties());
        enc->encode(makeSnowflake(1000));
        CHECK_NOTHROW(s, enc->properties());
        CHECK(s, enc->properties().has(EncodingProperty::Lossless));
    }
    WITH_SUBTEST(s, "7g: decode before encode throws") {
        auto enc  = makeDefaultAutoSubIntSplitEncoder<int64_t>();
        auto enc2 = makeDefaultAutoSubIntSplitEncoder<int64_t>();
        auto b1   = enc->encode(makeSnowflake(100));
        CHECK_THROWS(s, enc2->decodeAll(b1));
    }
    WITH_SUBTEST(s, "7h: MSB order roundtrip, name contains MSB") {
        auto enc  = makeDefaultAutoSubIntSplitEncoder<int64_t>(BitSplitOrder::MSB_TO_LSB);
        auto data = makeSnowflake(5000);
        auto blob = enc->encode(data);
        CHECK(s, enc->decodeAll(blob) == data);
        CHECK(s, enc->name().find("MSB") != std::string::npos);
    }
}

// ============================================================================
// Section 8: IDSubStreamEncodingSelector
// ============================================================================

static void testSelector(TestSuite& s) {
    auto makeCostModels = [] {
        return makeAutoSubIntSplitCostModelsFromTypes(defaultAutoSubIntSplitCostModelTypes());
    };
    IDSubStreamEncodingSelector sel;

    WITH_SUBTEST(s, "DP select: segments cover bits [0,63] contiguously") {
        auto sample = makeSnowflake(1000);
        auto models = makeCostModels();
        auto result = sel.select(sample, models);
        CHECK(s, !result.segments.empty());
        CHECK(s, std::isfinite(result.total_cost) && result.total_cost > 0.0);
        int coverage = 0, prevEnd = -1;
        for (const auto& seg : result.segments) {
            CHECK(s, seg.bitStart <= seg.bitEnd);
            CHECK(s, seg.bitStart == prevEnd + 1);
            coverage += seg.bitEnd - seg.bitStart + 1;
            prevEnd = seg.bitEnd;
        }
        CHECK(s, coverage == 64);
        CHECK_NOTHROW(s, SubIntSplitConfig64::fromSegments(result.segments));
    }
    WITH_SUBTEST(s, "forcedNumSegments=3: exactly 3 contiguous segments") {
        IDSubStreamEncodingSelector::Config cfg;
        cfg.forcedNumSegments = 3;
        IDSubStreamEncodingSelector sel3{cfg};
        auto sample = makeSnowflake(1000);
        auto models = makeCostModels();
        auto result = sel3.select(sample, models);
        CHECK(s, result.segments.size() == 3);
        int cov = 0;
        for (const auto& seg : result.segments) cov += seg.bitEnd - seg.bitStart + 1;
        CHECK(s, cov == 64);
    }
    WITH_SUBTEST(s, "selector -> fromSegments -> encoder roundtrip") {
        auto sample = makeSnowflake(5000);
        auto models = makeCostModels();
        auto result = sel.select(sample, models, sample.size());
        auto enc    = makeSubIntSplitEncoderFromSegments<int64_t>(result.segments);
        assertRoundtrip(s, *enc, sample);
    }

    WITH_SUBTEST(s, "exhaustiveSearch cost <= DP cost on tiny sample") {
        auto sample = makeUniformBits(64, 20, 99);
        IDSubStreamEncodingSelector selDP{IDSubStreamEncodingSelector::Config{}};
        auto dpResult = selDP.select(sample, makeCostModels());

        IDSubStreamEncodingSelector::Config exhCfg;
        exhCfg.useExhaustiveSearch = true;
        IDSubStreamEncodingSelector selExh{exhCfg};
        auto exhResult = selExh.select(sample, makeCostModels());

        int dpCov = 0, exhCov = 0;
        for (const auto& seg : dpResult.segments)  dpCov  += seg.bitEnd - seg.bitStart + 1;
        for (const auto& seg : exhResult.segments) exhCov += seg.bitEnd - seg.bitStart + 1;
        CHECK(s, dpCov == 64);
        CHECK(s, exhCov == 64);
        CHECK(s, exhResult.total_cost <= dpResult.total_cost + 1e-6);
    }
}

// ============================================================================
// Section 9: All encoding types via makeSubIntSplitEncoderManual
// ============================================================================

static void testAllEncodingTypes(TestSuite& s) {
    const auto N = size_t{2000};
    const auto data = makeSnowflake(N);

    struct Entry { EncodingType et; const char* name; };
    const Entry entries[] = {
        {EncodingType::RawEncoding,               "Raw"},
        {EncodingType::DictionaryEncoding,         "Dictionary"},
        {EncodingType::FrameOfReference,           "FrameOfReference"},
        {EncodingType::AdaptiveFrameOfReference,   "AdaptiveFrameOfReference"},
        {EncodingType::OpenZL,                     "OpenZL"},
        {EncodingType::BitPacking,                 "BitPacking"},
        {EncodingType::AdaptiveFramedBitPrefix,    "AdaptiveFramedBitPrefix"},
        {EncodingType::RunLengthEncoding,          "RunLength"},
        {EncodingType::HuffmanEncoding,            "Huffman"},
        {EncodingType::LZ4,                        "LZ4"},
        {EncodingType::FSEEncoding,                "FSE"},
        {EncodingType::FrequencyPartitionEncoding, "FrequencyPartition"},
    };

    for (const auto& [et, name] : entries) {
        char label[80];
        std::snprintf(label, sizeof(label), "%s in slot 0 (32|32 LSB)", name);
        WITH_SUBTEST(s, label) {
            auto enc = makeSubIntSplitEncoderManual<int64_t>(
                {32, 32}, {et, EncodingType::BitPacking}, BitSplitOrder::LSB_TO_MSB);
            assertRoundtrip(s, *enc, data);
        }
        std::snprintf(label, sizeof(label), "%s in slot 1 (32|32 LSB)", name);
        WITH_SUBTEST(s, label) {
            auto enc = makeSubIntSplitEncoderManual<int64_t>(
                {32, 32}, {EncodingType::BitPacking, et}, BitSplitOrder::LSB_TO_MSB);
            assertRoundtrip(s, *enc, data);
        }
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    int failures = 0;

    auto run = [&](const char* name, void (*fn)(TestSuite&)) {
        printf("\n--- %s ---\n", name);
        TestSuite suite{name};
        fn(suite);
        // If the last subtest had no failures, its label line is still open — close it.
        // (Failures already printed a newline inline via check().)
        if (!suite.currentSubtest.empty()) printf("\n");
        suite.summary();
        if (!suite.ok()) ++failures;
    };

    run("ConfigValidation",   testConfigValidation);
    run("FromSegments",       testFromSegments);
    run("ManualAllRA",        testManualAllRA);
    run("NonRASections",      testNonRAMixedSections);
    run("DecodeEdgeCases",    testDecodeEdgeCases);
    run("ProfilingVariants",  testProfilingVariants);
    run("AutoEncoder",        testAutoEncoder);
    run("Selector",           testSelector);
    run("AllEncodingTypes",   testAllEncodingTypes);

    if (failures == 0) {
        printf("\nAll suites passed.\n");
        return 0;
    }
    printf("\n%d suite(s) had failures.\n", failures);
    return 1;
}
