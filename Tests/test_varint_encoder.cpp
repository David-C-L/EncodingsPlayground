/**
 * @file test_varint_encoder.cpp
 * @brief Correctness tests for VarIntEncoder, including decodeGatherInto.
 */

#include "encoders/VarIntEncoder.hpp"
#include "GatherTestHelpers.hpp"
#include <iostream>
#include <vector>

using namespace encodings::encoders;

namespace {

bool testRoundTrip() {
    std::cout << "Test 1: decodeAll roundtrip\n";
    std::vector<int64_t> data = {0, 1, -1, 127, 128, -128, 1000000, -1000000, 42, 0};
    VarIntEncoder<int64_t> encoder;
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
             {0, 1, -1, 127, 128, -128, 1000000, -1000000, 42, 0},
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         }) {
        VarIntEncoder<int64_t> encoder;
        auto encoded = encoder.encode(data);
        allOk &= encodings::testutil::checkGatherAllTraceShapes<int64_t>(encoder, encoded, data);
    }
    std::cout << "  " << (allOk ? "PASS" : "FAIL") << "\n";
    return allOk;
}

bool testGatherLargerScattered() {
    std::cout << "Test 3: decodeGatherInto on a larger scattered trace\n";
    std::vector<int64_t> data(5000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<int64_t>(i) * 3 - 7000;
    VarIntEncoder<int64_t> encoder;
    auto encoded = encoder.encode(data);
    auto ranges = encodings::benchmark::makeSelectiveTrace(data.size(), {.selectivity = 0.15, .meanRunLength = 4.0});
    bool ok = encodings::testutil::checkGatherMatchesDecodeAll<int64_t>(encoder, encoded, data, ranges, "large_scattered");
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

}  // namespace

int main() {
    std::cout << "=== VarIntEncoder Tests ===\n\n";
    bool ok = true;
    ok &= testRoundTrip();
    ok &= testGatherAllTraceShapes();
    ok &= testGatherLargerScattered();
    std::cout << "\n" << (ok ? "All tests PASSED" : "SOME TESTS FAILED") << "\n";
    return ok ? 0 : 1;
}
