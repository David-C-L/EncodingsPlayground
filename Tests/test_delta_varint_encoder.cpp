/**
 * @file test_delta_varint_encoder.cpp
 * @brief Correctness tests for DeltaVarIntEncoder, including decodeGatherInto.
 */

#include "encoders/DeltaVarIntEncoder.hpp"
#include "GatherTestHelpers.hpp"
#include <iostream>
#include <vector>

using namespace encodings::encoders;

namespace {

bool testRoundTrip() {
    std::cout << "Test 1: decodeAll roundtrip\n";
    std::vector<int64_t> data = {1000, 1001, 1003, 1002, 1050, 1049, 1049, 2000, 1999, 1998};
    DeltaVarIntEncoder<int64_t> encoder;
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
             {1000, 1001, 1003, 1002, 1050, 1049, 1049, 2000, 1999, 1998},
             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         }) {
        DeltaVarIntEncoder<int64_t> encoder;
        auto encoded = encoder.encode(data);
        allOk &= encodings::testutil::checkGatherAllTraceShapes<int64_t>(encoder, encoded, data);
    }
    std::cout << "  " << (allOk ? "PASS" : "FAIL") << "\n";
    return allOk;
}

bool testGatherLargerScattered() {
    std::cout << "Test 3: decodeGatherInto on a larger scattered trace\n";
    std::vector<int64_t> data(5000);
    int64_t v = 10000;
    for (size_t i = 0; i < data.size(); ++i) {
        v += static_cast<int64_t>((i * 7) % 13) - 6;
        data[i] = v;
    }
    DeltaVarIntEncoder<int64_t> encoder;
    auto encoded = encoder.encode(data);
    auto ranges = encodings::benchmark::makeSelectiveTrace(data.size(), {.selectivity = 0.15, .meanRunLength = 4.0});
    bool ok = encodings::testutil::checkGatherMatchesDecodeAll<int64_t>(encoder, encoded, data, ranges, "large_scattered");
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

}  // namespace

int main() {
    std::cout << "=== DeltaVarIntEncoder Tests ===\n\n";
    bool ok = true;
    ok &= testRoundTrip();
    ok &= testGatherAllTraceShapes();
    ok &= testGatherLargerScattered();
    std::cout << "\n" << (ok ? "All tests PASSED" : "SOME TESTS FAILED") << "\n";
    return ok ? 0 : 1;
}
