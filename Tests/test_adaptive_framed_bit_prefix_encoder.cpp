/**
 * @file test_adaptive_framed_bit_prefix_encoder.cpp
 * @brief Correctness tests for AdaptiveFramedBitPrefixEncoder, including decodeGatherInto.
 */

#include "encoders/AdaptiveFramedBitPrefixEncoder.hpp"
#include "GatherTestHelpers.hpp"
#include <iostream>
#include <random>
#include <vector>

using namespace encodings::encoders;

namespace {

bool testRoundTrip() {
    std::cout << "Test 1: decodeAll roundtrip\n";
    std::vector<int64_t> data;
    for (int i = 0; i < 500; ++i) data.push_back((i / 37) * 1000 + (i % 37));
    AdaptiveFramedBitPrefixEncoder<int64_t> encoder;
    auto encoded = encoder.encode(data);
    auto decoded = encoder.decodeAll(encoded);
    bool ok = (decoded == data);
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testGatherAllTraceShapes() {
    std::cout << "Test 2: decodeGatherInto against decodeAll() across trace shapes\n";
    bool allOk = true;
    for (auto& data : std::vector<std::vector<int64_t>>{
             []() {
                 std::vector<int64_t> v;
                 for (int i = 0; i < 500; ++i) v.push_back((i / 37) * 1000 + (i % 37));
                 return v;
             }(),
             std::vector<int64_t>(200, 42),  // single common prefix covering all bits
         }) {
        AdaptiveFramedBitPrefixEncoder<int64_t> encoder;
        auto encoded = encoder.encode(data);
        allOk &= encodings::testutil::checkGatherAllTraceShapes<int64_t>(encoder, encoded, data);
    }
    std::cout << "  " << (allOk ? "PASS" : "FAIL") << "\n";
    return allOk;
}

bool testGatherLargerScattered() {
    std::cout << "Test 3: decodeGatherInto on a larger scattered trace (multi-frame)\n";
    std::vector<int64_t> data(20000);
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<int64_t> dist(0, 1000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (static_cast<int64_t>(i / 200) * 1000) + dist(rng) % 64;
    AdaptiveFramedBitPrefixEncoder<int64_t> encoder;
    auto encoded = encoder.encode(data);
    auto ranges = encodings::benchmark::makeSelectiveTrace(data.size(), {.selectivity = 0.15, .meanRunLength = 5.0});
    bool ok = encodings::testutil::checkGatherMatchesDecodeAll<int64_t>(encoder, encoded, data, ranges, "large_scattered");
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

}  // namespace

int main() {
    std::cout << "=== AdaptiveFramedBitPrefixEncoder Tests ===\n\n";
    bool ok = true;
    ok &= testRoundTrip();
    ok &= testGatherAllTraceShapes();
    ok &= testGatherLargerScattered();
    std::cout << "\n" << (ok ? "All tests PASSED" : "SOME TESTS FAILED") << "\n";
    return ok ? 0 : 1;
}
