#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "encoders/BlockFORFPEEncoder.hpp"

using namespace encodings::encoders;

template <typename T>
static void assertRoundTrip(const std::vector<T>& input, const std::string& label) {
    BlockFORFPEEncoder<T> enc;
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

        // Range test: decode [quarter, three-quarters)
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
              << "  ratio=" << static_cast<double>(encoded.size())
                               / (input.size() * sizeof(T)) << "\n";
}

int main() {
    // --- All-equal ---
    assertRoundTrip<int64_t>(std::vector<int64_t>(256, 42LL), "all-equal int64");

    // --- Single element ---
    assertRoundTrip<int64_t>({999LL}, "single element int64");

    // --- Small range (fits in 1 byte after FOR) ---
    {
        std::vector<int64_t> v;
        for (int i = 0; i < 1000; ++i) v.push_back(1'000'000'000LL + (i % 100));
        assertRoundTrip<int64_t>(v, "small-range int64 (1e9 base, 100 distinct)");
    }

    // --- Monotone increasing (Snowflake-like) ---
    {
        std::vector<int64_t> v;
        for (int i = 0; i < 10'000; ++i) v.push_back(static_cast<int64_t>(i) * 1024 + 1'000'000'000LL);
        assertRoundTrip<int64_t>(v, "monotone-increasing int64");
    }

    // --- Random int64 (large range) ---
    {
        std::mt19937_64 rng(42);
        std::vector<int64_t> v(5000);
        std::generate(v.begin(), v.end(), [&]{ return static_cast<int64_t>(rng()); });
        assertRoundTrip<int64_t>(v, "random int64");
    }

    // --- int32 small range ---
    {
        std::vector<int32_t> v;
        for (int i = 0; i < 2000; ++i) v.push_back(1000 + (i % 50));
        assertRoundTrip<int32_t>(v, "small-range int32");
    }

    // --- Highly skewed distribution ---
    {
        std::mt19937 rng(17);
        std::vector<int64_t> v(4000);
        // 90% value 42, 9% value 43, 1% random in [0, 1000]
        std::uniform_int_distribution<int> which(0, 99);
        std::uniform_int_distribution<int64_t> rnd(0, 1000);
        for (auto& x : v) {
            int w = which(rng);
            if (w < 90) x = 42;
            else if (w < 99) x = 43;
            else x = rnd(rng);
        }
        assertRoundTrip<int64_t>(v, "skewed int64 (90/9/1 split)");
    }

    // --- Last block shorter than blockSize ---
    {
        std::vector<int64_t> v(513); // 513 = 512+1, forces tiny last block
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int64_t>(i % 7) * 1000LL;
        assertRoundTrip<int64_t>(v, "short last block");
    }

    std::cout << "All tests passed.\n";
    return 0;
}
