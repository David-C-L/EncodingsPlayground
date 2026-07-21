#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "encoders/BlockFrequencyPartitionEncoder.hpp"
#include "encoders/CascadingFOREncoder.hpp"
#include "encoders/DeltaPrepassEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "reorderers/ReorderingCodec.hpp"
#include "reorderers/ReorderingType.hpp"
#include "reorderers/SortReorderer.hpp"

using namespace encodings::encoders;
using namespace encodings::reorderers;

template <typename T>
static void assertRoundTrip(const std::vector<T>& input, DeltaPrepassConfig cfg, const std::string& label) {
    DeltaPrepassEncoder<T> enc(cfg);
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

// Confirms DeltaPrepassEncoder composes correctly under ReorderingCodec+SortReorderer,
// and separately that CascadingFOREncoder does too (not re-testing ReorderingCodec
// itself, just confirming the composition round-trips).
static void testSortedComposition() {
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<int64_t> dist(-500'000LL, 500'000LL);
    std::vector<int64_t> data(5000);
    for (auto& v : data) v = dist(rng);

    {
        auto inner = std::make_shared<DeltaPrepassEncoder<int64_t>>(
            DeltaPrepassConfig{std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true)});
        ReorderingCodec<int64_t, false> codec(std::make_shared<SortReorderer<int64_t>>(), inner, ReorderingType::Sort);
        auto encoded = codec.encode(std::span<const int64_t>(data));
        auto decoded = codec.decodeAll(encoded);
        if (decoded != data) {
            std::cerr << "FAIL: sorted DeltaPrepassEncoder composition\n";
            assert(false);
        }
        std::cout << "PASS: sorted DeltaPrepassEncoder composition  bytes=" << encoded.data().size() << "\n";
    }

    {
        CascadingFORConfig forCfg;
        forCfg.residualSchedule = { {8} };
        forCfg.referenceSchedule = { {1024}, {256}, {64}, {16} };
        auto inner = std::make_shared<CascadingFOREncoder<int64_t>>(forCfg);
        ReorderingCodec<int64_t, false> codec(std::make_shared<SortReorderer<int64_t>>(), inner, ReorderingType::Sort);
        auto encoded = codec.encode(std::span<const int64_t>(data));
        auto decoded = codec.decodeAll(encoded);
        if (decoded != data) {
            std::cerr << "FAIL: sorted CascadingFOREncoder composition\n";
            assert(false);
        }
        std::cout << "PASS: sorted CascadingFOREncoder composition  bytes=" << encoded.data().size() << "\n";
    }
}

int main() {
    DeltaPrepassConfig rawLeaf; // default RawBitPackedEncoder<int64_t>
    DeltaPrepassConfig blockFpeLeaf{std::make_shared<BlockFrequencyPartitionEncoder<int64_t>>(/*bitPackFallback=*/true)};

    for (auto* cfgPair : {&rawLeaf, &blockFpeLeaf}) {
        DeltaPrepassConfig& cfg = *cfgPair;
        const std::string leafName = (&cfg == &rawLeaf) ? "raw" : "blockfpe";

        assertRoundTrip<int32_t>({}, cfg, "N=0 " + leafName);
        assertRoundTrip<int32_t>({42}, cfg, "N=1 " + leafName);
        assertRoundTrip<int32_t>(std::vector<int32_t>(10, 7), cfg, "all-equal " + leafName);

        {
            std::vector<int64_t> v;
            for (int i = 0; i < 2000; ++i) v.push_back(-1'000'000LL + i * 3);
            assertRoundTrip<int64_t>(v, cfg, "monotone increasing " + leafName);
        }
        {
            std::vector<int64_t> v;
            for (int i = 0; i < 2000; ++i) v.push_back(1'000'000LL - i * 3);
            assertRoundTrip<int64_t>(v, cfg, "monotone decreasing " + leafName);
        }
        {
            std::mt19937_64 rng(42);
            std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min() / 2,
                                                          std::numeric_limits<int64_t>::max() / 2);
            std::vector<int64_t> v(3000);
            for (auto& x : v) x = dist(rng);
            assertRoundTrip<int64_t>(v, cfg, "full-range random " + leafName);
        }
        {
            // Near-overflow deltas: adjacent extreme values, widened to int64_t
            // arithmetic so this must not overflow during delta computation.
            std::vector<int64_t> v = {
                std::numeric_limits<int64_t>::min() / 2,
                std::numeric_limits<int64_t>::max() / 2,
                std::numeric_limits<int64_t>::min() / 2,
                0,
                std::numeric_limits<int64_t>::max() / 2,
            };
            assertRoundTrip<int64_t>(v, cfg, "near-overflow deltas " + leafName);
        }
    }

    testSortedComposition();

    std::cout << "ALL PASS\n";
    return 0;
}
