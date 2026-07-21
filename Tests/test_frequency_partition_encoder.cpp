#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "encoders/FrequencyPartitionEncoder.hpp"
#include "GatherTestHelpers.hpp"

using namespace encodings::encoders;

namespace {

template <typename T, FreqPartIndexType Mode>
void assertRoundTrip(const std::vector<T>& input,
                     const std::string& label) {
    FrequencyPartitionEncoder<T, Mode> enc;
    auto encoded = enc.encode(input);

    const auto all = enc.decodeAll(encoded);
    const bool expectsReordered = (Mode == FreqPartIndexType::NoIndex);
    if (!expectsReordered) {
        assert(all == input);
    }

    // Spot-check random access at representative positions.
    if (!input.empty()) {
        const std::vector<size_t> idxs = {
            0,
            input.size() / 5,
            input.size() / 2,
            (input.size() * 4) / 5,
            input.size() - 1,
        };
        for (size_t i : idxs) {
            const std::optional<T> v = enc.decodeAt(encoded, i);
            if (!v.has_value()) {
                std::cerr << "Error: decodeAt returned no value for index " << i << "\n";
                return;
            }
            assert(v.has_value());
            if (!expectsReordered) {
                assert(*v == input[i]);
            } else {
                assert(*v == all[i]);
            }
        }
    }

    // Range checks (including clamped and empty cases).
    const auto fullRange = enc.decodeRange(encoded, 0, input.size());
    if (!expectsReordered) {
        assert(fullRange == input);
    } else {
        assert(fullRange == all);
    }

    const auto emptyRange = enc.decodeRange(encoded, input.size(), input.size());
    assert(emptyRange.empty());

    if (input.size() >= 7) {
        const size_t s = 3;
        const size_t e = input.size() - 2;
        const auto r = enc.decodeRange(encoded, s, e);
        const std::vector<T> expected(all.begin() + static_cast<std::ptrdiff_t>(s),
                                      all.begin() + static_cast<std::ptrdiff_t>(e));
        assert(r == expected);
    }

    // decodeAt out-of-range should return nullopt.
    {
        const auto oob = enc.decodeAt(encoded, input.size());
        assert(!oob.has_value());
        if (oob.has_value()) {
            std::cerr << "Error: decodeAt returned a value for out-of-range index " << input.size() << "\n";
             return;
        }
    }

    // decodeGatherInto against decodeAll() (== `all`, the ground truth even
    // for NoIndex's reordered output) across representative trace shapes.
    // Exercises the real fast-path override for TierTagArray/PerTierBitmaps
    // and the inherited default fallback for EliasFano/NoIndex.
    if (!encodings::testutil::checkGatherAllTraceShapes<T>(enc, encoded, all)) {
        std::cerr << "Error: decodeGatherInto mismatch for " << label << "\n";
        assert(false);
    }

    std::cout << "  ✓ " << label << " (n=" << input.size()
              << ", mode="
              << (Mode == FreqPartIndexType::PerTierBitmaps ? "PerTierBitmaps"
                  : Mode == FreqPartIndexType::TierTagArray ? "TierTagArray"
                  : Mode == FreqPartIndexType::EliasFano ? "EliasFano"
                  : "NoIndex")
              << ")\n";
}

std::vector<uint32_t> makeSkewedData() {
    std::vector<uint32_t> v;
    v.reserve(20000);
    for (uint32_t i = 0; i < 12000; ++i) v.push_back(7);
    for (uint32_t i = 0; i < 5000; ++i) v.push_back(42);
    for (uint32_t i = 0; i < 2000; ++i) v.push_back(1000 + (i % 13));
    for (uint32_t i = 0; i < 1000; ++i) v.push_back(100000 + i);
    return v;
}

std::vector<uint32_t> makeNonPowerTierData() {
    // 13 unique symbols forces a partially filled tier where keyBits=3 for 7 values.
    std::vector<uint32_t> v;
    v.reserve(13000);
    for (uint32_t rep = 0; rep < 1000; ++rep) {
        for (uint32_t x = 0; x < 13; ++x) {
            v.push_back(100 + x);
        }
    }
    return v;
}

std::vector<uint64_t> makeMostlyUniqueData() {
    std::vector<uint64_t> v;
    v.reserve(25000);
    for (uint64_t i = 0; i < 25000; ++i) {
        v.push_back((i << 20) ^ (0x9e3779b97f4a7c15ULL + i * 17ULL));
    }
    return v;
}

std::vector<uint64_t> makeRandomMixedData() {
    std::mt19937_64 rng(123456789ULL);
    std::vector<uint64_t> v;
    v.reserve(30000);
    for (size_t i = 0; i < 30000; ++i) {
        const uint64_t r = rng();
        if ((r & 0xF) < 10) {
            v.push_back(r % 64); // high-frequency small domain
        } else {
            v.push_back(r ^ (static_cast<uint64_t>(i) << 32)); // long tail
        }
    }
    return v;
}

std::vector<int32_t> makeSignedSkewedData() {
    std::vector<int32_t> v;
    v.reserve(20000);
    for (int32_t i = 0; i < 9000; ++i) v.push_back(-7);
    for (int32_t i = 0; i < 4000; ++i) v.push_back(42);
    for (int32_t i = 0; i < 4000; ++i) v.push_back((i % 17) - 8);
    for (int32_t i = 0; i < 3000; ++i) v.push_back(-100000 + i);
    return v;
}

std::vector<int64_t> makeSignedMixed64Data() {
    std::mt19937_64 rng(987654321ULL);
    std::vector<int64_t> v;
    v.reserve(30000);
    for (size_t i = 0; i < 30000; ++i) {
        const uint64_t r = rng();
        if ((r & 0xF) < 10) {
            v.push_back(static_cast<int64_t>((r % 257ULL) - 128));
        } else {
            const int64_t signedTail = static_cast<int64_t>(r ^ (static_cast<uint64_t>(i) << 33));
            v.push_back(signedTail - static_cast<int64_t>(1ULL << 62));
        }
    }
    return v;
}

template <typename T, FreqPartIndexType Mode>
void runForMode() {
    if constexpr (std::is_same_v<T, uint32_t>) {
        assertRoundTrip<T, Mode>(makeSkewedData(), "uint32 skewed");
        assertRoundTrip<T, Mode>(makeNonPowerTierData(), "uint32 non-power tier width");
    } else if constexpr (std::is_same_v<T, int32_t>) {
        assertRoundTrip<T, Mode>(makeSignedSkewedData(), "int32 signed skewed");
    } else {
        if constexpr (std::is_same_v<T, uint64_t>) {
            assertRoundTrip<T, Mode>(makeMostlyUniqueData(), "uint64 mostly unique");
            assertRoundTrip<T, Mode>(makeRandomMixedData(), "uint64 random mixed");
        } else {
            assertRoundTrip<T, Mode>(makeSignedMixed64Data(), "int64 signed mixed");
        }
    }
}

} // namespace

int main() {
    std::cout << "=== FrequencyPartitionEncoder correctness tests ===\n";

    runForMode<uint32_t, FreqPartIndexType::PerTierBitmaps>();
    runForMode<int32_t,  FreqPartIndexType::PerTierBitmaps>();
    runForMode<uint64_t, FreqPartIndexType::PerTierBitmaps>();
    runForMode<int64_t,  FreqPartIndexType::PerTierBitmaps>();

    runForMode<uint32_t, FreqPartIndexType::TierTagArray>();
    runForMode<int32_t,  FreqPartIndexType::TierTagArray>();
    runForMode<uint64_t, FreqPartIndexType::TierTagArray>();
    runForMode<int64_t,  FreqPartIndexType::TierTagArray>();

    runForMode<uint32_t, FreqPartIndexType::EliasFano>();
    runForMode<int32_t,  FreqPartIndexType::EliasFano>();
    runForMode<uint64_t, FreqPartIndexType::EliasFano>();
    runForMode<int64_t,  FreqPartIndexType::EliasFano>();

    runForMode<uint32_t, FreqPartIndexType::NoIndex>();
    runForMode<int32_t,  FreqPartIndexType::NoIndex>();
    runForMode<uint64_t, FreqPartIndexType::NoIndex>();
    runForMode<int64_t,  FreqPartIndexType::NoIndex>();

    std::cout << "=== All FrequencyPartitionEncoder tests passed ===\n";
    return 0;
}
