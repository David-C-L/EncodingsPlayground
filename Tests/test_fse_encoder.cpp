#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "encoders/FSEEncoder.hpp"
#include "encoders/SubIntSplitEncoder.hpp"

using namespace encodings;
using namespace encodings::encoders;

static void test_fse_roundtrip_basic() {
    FSEEncoder<uint64_t> enc;
    std::vector<uint64_t> data;
    data.reserve(10000);
    for (uint64_t i = 0; i < 10000; ++i) {
        data.push_back((i % 11) * 3 + (i % 2));
    }

    auto blob = enc.encode(data);
    auto out = enc.decodeAll(blob);
    assert(out == data);

    auto at = enc.decodeAt(blob, 42);
    assert(at.has_value() && *at == data[42]);
    if (at == 0) {
        std::cerr << "";
    }

    auto rg = enc.decodeRange(blob, 100, 150);
    assert(rg.size() == 50);
    for (size_t i = 0; i < rg.size(); ++i) {
        assert(rg[i] == data[100 + i]);
    }
}

static void test_fse_roundtrip_single_symbol() {
    FSEEncoder<uint32_t> enc;
    std::vector<uint32_t> data(4096, 17u);
    auto blob = enc.encode(data);
    auto out = enc.decodeAll(blob);
    assert(out == data);
}

static void test_subint_manual_fse_section() {
    // 64-bit total = 13 + 10 + 41
    auto split = makeSubIntSplitEncoderManual<int64_t>(
        {13, 10, 41},
        {EncodingType::BitPacking, EncodingType::FSEEncoding, EncodingType::AdaptiveFrameOfReference},
        BitSplitOrder::LSB_TO_MSB
    );

    std::vector<int64_t> data;
    data.reserve(4096);
    for (int64_t i = 0; i < 4096; ++i) {
        // Keeps each section in-range and introduces moderate repetition in middle section.
        const uint64_t lo13 = static_cast<uint64_t>(i) & ((1ULL << 13) - 1ULL);
        const uint64_t mid10 = (static_cast<uint64_t>(i) / 7ULL) & ((1ULL << 10) - 1ULL);
        const uint64_t hi41 = (static_cast<uint64_t>(i) * 13ULL) & ((1ULL << 41) - 1ULL);
        const uint64_t packed = lo13 | (mid10 << 13) | (hi41 << 23);
        data.push_back(static_cast<int64_t>(packed));
    }

    auto blob = split->encode(data);
    auto out = split->decodeAll(blob);
    assert(out == data);
}

int main() {
    test_fse_roundtrip_basic();
    test_fse_roundtrip_single_symbol();
    test_subint_manual_fse_section();
    std::cout << "FSE tests passed\n";
    return 0;
}
