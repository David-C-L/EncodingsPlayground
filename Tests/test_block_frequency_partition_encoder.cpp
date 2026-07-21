#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "encoders/BlockFrequencyPartitionEncoder.hpp"

using namespace encodings::encoders;

template <typename T>
static void assertRoundTrip(const std::vector<T>& input, bool bitPackFallback, const std::string& label) {
    BlockFrequencyPartitionEncoder<T> enc(bitPackFallback);
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

        std::vector<T> into(input.size());
        enc.decodeAllInto(encoded, into.data(), into.size());
        if (into != input) {
            std::cerr << "FAIL decodeAllInto: " << label << "\n";
            assert(false);
        }
    }

    std::cout << "PASS: " << label
              << "  N=" << input.size()
              << "  bitPackFallback=" << bitPackFallback
              << "  bytes=" << encoded.data().size() << "\n";
}

// Runs both bitPackFallback=false and =true for the same input, following the
// same round-trip checks each time. The fallback bit-packing feature is purely
// an encode-time choice (decode reads the per-block fallbackBits field), so
// correctness must hold identically in both modes.
template <typename T>
static void assertRoundTripBothModes(const std::vector<T>& input, const std::string& label) {
    assertRoundTrip(input, false, label + " (raw fallback)");
    assertRoundTrip(input, true,  label + " (bit-packed fallback)");
}

// Same as assertRoundTripBothModes, but also prints the block size the
// planner actually chose -- useful when the test data is constructed to
// target a specific per-block tier population (e.g. exercising a
// non-power-of-two tier key width) and we want visibility into whether the
// planner's choice of block size lines up with that intent.
template <typename T>
static void assertRoundTripBothModesVerbose(const std::vector<T>& input, const std::string& label) {
    BlockFrequencyPartitionEncoder<T> enc(false);
    auto encoded = enc.encode(std::span<const T>(input));
    std::cout << "  (chosen block_size=" << encoded.metadata().customMetadata["block_size"]
               << " num_blocks=" << encoded.metadata().customMetadata["num_blocks"] << ") "
               << label << "\n";
    assertRoundTripBothModes(input, label);
}

// Builds a periodic "staircase" array: 2 distinct values repeated `hotCount`
// times each (tier0), 4 distinct values repeated `midCount` times each
// (tier1), and `rareDistinct` distinct values repeated `rareCount` times each
// (targets tier2 with exactly `rareDistinct` entries -- a non-power-of-two
// key width whenever rareDistinct isn't 1, 2, 4, 8, or 16). No filler values
// are added, so a block that sees a whole number of periods has no fallback.
static std::vector<int32_t> buildStaircase(int rareDistinct, int hotCount, int midCount,
                                            int rareCount, int repeats) {
    std::vector<int32_t> period;
    int32_t nextValue = 0;
    for (int i = 0; i < 2; ++i) { for (int c = 0; c < hotCount; ++c) period.push_back(nextValue); ++nextValue; }
    for (int i = 0; i < 4; ++i) { for (int c = 0; c < midCount; ++c) period.push_back(nextValue); ++nextValue; }
    for (int i = 0; i < rareDistinct; ++i) { for (int c = 0; c < rareCount; ++c) period.push_back(nextValue); ++nextValue; }

    std::vector<int32_t> data;
    data.reserve(period.size() * repeats);
    for (int r = 0; r < repeats; ++r) data.insert(data.end(), period.begin(), period.end());
    return data;
}

int main() {
    assertRoundTripBothModes<int64_t>({}, "empty");
    assertRoundTripBothModes<int64_t>({42}, "single element");
    assertRoundTripBothModes<int64_t>(std::vector<int64_t>(1000, -123456789LL), "all-equal");

    // Skewed: a few hot values dominate (tiered), remainder falls back with a
    // narrow but large-magnitude range -- the case bit-packed fallback targets.
    {
        std::mt19937_64 rng(7);
        std::uniform_int_distribution<int64_t> hot(0, 1);
        std::uniform_int_distribution<int64_t> fb(1'000'000'000LL, 1'000'000'100LL);
        std::vector<int64_t> d;
        d.reserve(20000);
        for (int i = 0; i < 20000; ++i) {
            d.push_back(i % 10 < 8 ? hot(rng) : fb(rng));
        }
        assertRoundTripBothModes<int64_t>(d, "skewed with narrow large-magnitude fallback");
    }

    // High-cardinality random data across the full range -- everything falls
    // back; bit-packing should still narrow to the actual observed span.
    {
        std::mt19937_64 rng(11);
        std::uniform_int_distribution<int64_t> dist(-1'000'000'000LL, 1'000'000'000LL);
        std::vector<int64_t> d(5000);
        for (auto& v : d) v = dist(rng);
        assertRoundTripBothModes<int64_t>(d, "high-cardinality wide-range");
    }

    // Negative values, small magnitude
    {
        std::vector<int32_t> d;
        for (int i = 0; i < 3000; ++i) d.push_back(static_cast<int32_t>(-500 + (i % 900)));
        assertRoundTripBothModes<int32_t>(d, "negative small-magnitude int32");
    }

    // Unsigned type near the top of its range (exercises unsigned delta arithmetic)
    {
        std::mt19937_64 rng(13);
        std::uniform_int_distribution<uint32_t> dist(0, 4000000000u);
        std::vector<uint32_t> d(4000);
        for (auto& v : d) v = dist(rng);
        assertRoundTripBothModes<uint32_t>(d, "uint32 near-max-range");
    }

    // Short last block (not a multiple of the chosen block size)
    {
        std::vector<int64_t> d;
        for (int i = 0; i < 513; ++i) d.push_back(i % 5);
        assertRoundTripBothModes<int64_t>(d, "short last block");
    }

    // --- Non-power-of-two tier key width coverage ---
    // These target exactly-N-distinct tier2 populations (N=5,6,7) via a
    // periodic staircase: 2 hot values (tier0), 4 mid values (tier1), then N
    // rare-but-not-fallback values (tier2). Whenever a block sees N tier2
    // entries with N not in {1,2,4,8,16}, its tier2 keys need a non-power-of-
    // two width (bit_width(N-1) bits: 3 bits for N=5..8).
    assertRoundTripBothModesVerbose<int32_t>(
        buildStaircase(/*rareDistinct=*/5, /*hotCount=*/10, /*midCount=*/4, /*rareCount=*/2, /*repeats=*/200),
        "staircase tier2=5 (3-bit keys)");
    assertRoundTripBothModesVerbose<int32_t>(
        buildStaircase(/*rareDistinct=*/6, /*hotCount=*/10, /*midCount=*/4, /*rareCount=*/2, /*repeats=*/200),
        "staircase tier2=6 (3-bit keys)");
    assertRoundTripBothModesVerbose<int32_t>(
        buildStaircase(/*rareDistinct=*/7, /*hotCount=*/10, /*midCount=*/4, /*rareCount=*/2, /*repeats=*/200),
        "staircase tier2=7 (3-bit keys)");

    // Singleton tier1: tier0 full (2 hot values), exactly 1 additional distinct
    // value populates tier1 alone (rareDistinct=1 here lands in tier2's slot in
    // buildStaircase's layout only once tier1 has 4 -- so instead build this
    // case directly: 2 hot + 1 rare, no mid values, so the rare value is the
    // 3rd distinct value overall and lands in tier1 at position 0 -> numTier1=1,
    // a 0-bit tier1 key width).
    {
        std::vector<int32_t> period;
        for (int c = 0; c < 20; ++c) period.push_back(0);
        for (int c = 0; c < 20; ++c) period.push_back(1);
        for (int c = 0; c < 3;  ++c) period.push_back(2);
        std::vector<int32_t> d;
        for (int r = 0; r < 200; ++r) d.insert(d.end(), period.begin(), period.end());
        assertRoundTripBothModesVerbose<int32_t>(d, "singleton tier1 (0-bit keys)");
    }

    // Singleton tier0 across changing blocks: long constant runs (longer than
    // any block size the planner is likely to choose) with the constant value
    // changing between runs, so most/all blocks see exactly 1 distinct value
    // (numTier0==1, 0-bit tier0 keys) but the *value* differs block to block --
    // stresses the 0-bit path across many blocks and many dictionaries, beyond
    // the single "all-equal" test which only ever has one dictionary value.
    {
        std::vector<int64_t> d;
        for (int run = 0; run < 30; ++run)
            for (int i = 0; i < 1024; ++i) d.push_back(run % 3);
        assertRoundTripBothModesVerbose<int64_t>(d, "alternating constant runs (0-bit tier0 keys)");
    }

    std::cout << "All BlockFrequencyPartitionEncoder tests passed.\n";
    return 0;
}
