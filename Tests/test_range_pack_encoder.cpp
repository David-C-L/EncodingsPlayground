#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "encoders/SubIntEncodingUtils.hpp"

using namespace encodings::encoders;
using encodings::EncodingType;

template <typename T>
static void assertRoundTrip(std::shared_ptr<ISectionCodecIntegral<T>> codec,
                             const std::vector<T>& input, const std::string& label) {
    auto encoded = codec->encode(std::span<const T>(input));

    const auto all = codec->decodeAll(encoded);
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
            auto v = codec->decodeAt(encoded, i);
            if (!v.has_value() || *v != input[i]) {
                std::cerr << "FAIL decodeAt[" << i << "]: " << label << "\n";
                assert(false);
            }
        }

        const size_t rStart = input.size() / 4;
        const size_t rEnd   = (input.size() * 3) / 4;
        if (rStart < rEnd) {
            auto range = codec->decodeRange(encoded, rStart, rEnd);
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
              << "  bytes=" << encoded.data().size() << "\n";
}

int main() {
    using T = uint64_t;

    // --- Round trip correctness: large non-zero minimum, narrow range ---
    {
        std::vector<T> v;
        for (int i = 0; i <= 500; ++i) v.push_back(1'000'000ULL + static_cast<T>(i));

        auto rangePackFpe = detail_trisplit::makeRangePackFrequencyPartitionSection<T>(64);
        assertRoundTrip<T>(rangePackFpe, v, "RangePack+FPE, large-min narrow-range");

        auto rangePackBlockFpe = detail_trisplit::makeRangePackBlockFrequencyPartitionSection<T>(64);
        assertRoundTrip<T>(rangePackBlockFpe, v, "RangePack+BlockFPE, large-min narrow-range");
    }

    // --- Single element ---
    {
        std::vector<T> v{999'999ULL};
        assertRoundTrip<T>(detail_trisplit::makeRangePackFrequencyPartitionSection<T>(64), v, "single element");
    }

    // --- All-equal ---
    {
        std::vector<T> v(300, 42ULL);
        assertRoundTrip<T>(detail_trisplit::makeRangePackBlockFrequencyPartitionSection<T>(64), v, "all-equal");
    }

    // --- Skewed distribution with large offset ---
    {
        std::mt19937 rng(17);
        std::vector<T> v(4000);
        std::uniform_int_distribution<int> which(0, 99);
        std::uniform_int_distribution<int> rnd(0, 1000);
        for (auto& x : v) {
            int w = which(rng);
            if (w < 90) x = 5'000'000ULL;
            else if (w < 99) x = 5'000'001ULL;
            else x = 5'000'000ULL + static_cast<T>(rnd(rng));
        }
        assertRoundTrip<T>(detail_trisplit::makeRangePackFrequencyPartitionSection<T>(64), v, "skewed large-offset (FPE)");
        assertRoundTrip<T>(detail_trisplit::makeRangePackBlockFrequencyPartitionSection<T>(64), v, "skewed large-offset (BlockFPE)");
    }

    // --- Confirm genuine type-narrowing byte-size win ---
    {
        std::vector<T> v;
        for (int i = 0; i <= 500; ++i) v.push_back(1'000'000ULL + static_cast<T>(i % 100));

        auto plainFpe      = detail_trisplit::makeFrequencyPartitionSection<T>(64);
        auto rangePackFpe  = detail_trisplit::makeRangePackFrequencyPartitionSection<T>(64);
        auto plainBlockFpe = detail_trisplit::makeBlockFrequencyPartitionSection<T>(64);
        auto blockFpeFor   = detail_trisplit::makeBlockFrequencyPartitionFORSection<T>(64);
        auto rangeBlockFpe = detail_trisplit::makeRangePackBlockFrequencyPartitionSection<T>(64);

        const size_t plainFpeBytes      = plainFpe->encode(std::span<const T>(v)).data().size();
        const size_t rangePackFpeBytes  = rangePackFpe->encode(std::span<const T>(v)).data().size();
        const size_t plainBlockFpeBytes = plainBlockFpe->encode(std::span<const T>(v)).data().size();
        const size_t blockFpeForBytes   = blockFpeFor->encode(std::span<const T>(v)).data().size();
        const size_t rangeBlockFpeBytes = rangeBlockFpe->encode(std::span<const T>(v)).data().size();

        std::cout << "\n--- Type-narrowing byte-size comparison ---\n";
        std::cout << "plain FPE:                    " << plainFpeBytes << " bytes\n";
        std::cout << "RangePack+FPE:                 " << rangePackFpeBytes << " bytes\n";
        std::cout << "plain BlockFPE:                " << plainBlockFpeBytes << " bytes\n";
        std::cout << "BlockFPE+GlobalFOR (same-width): " << blockFpeForBytes << " bytes\n";
        std::cout << "RangePack+BlockFPE (narrowed):  " << rangeBlockFpeBytes << " bytes\n";

        if (rangePackFpeBytes >= plainFpeBytes) {
            std::cerr << "FAIL: RangePack+FPE (" << rangePackFpeBytes
                       << ") did not beat plain FPE (" << plainFpeBytes << ")\n";
            assert(false);
        }
        if (rangeBlockFpeBytes >= plainBlockFpeBytes) {
            std::cerr << "FAIL: RangePack+BlockFPE (" << rangeBlockFpeBytes
                       << ") did not beat plain BlockFPE (" << plainBlockFpeBytes << ")\n";
            assert(false);
        }
        if (rangeBlockFpeBytes >= blockFpeForBytes) {
            std::cerr << "FAIL: RangePack+BlockFPE (" << rangeBlockFpeBytes
                       << ") did not beat same-width BlockFPE+GlobalFOR (" << blockFpeForBytes << ")\n";
            assert(false);
        }
    }

    std::cout << "\nAll RangePack tests passed.\n";
    return 0;
}
