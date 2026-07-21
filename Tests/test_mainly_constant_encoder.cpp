/**
 * @file test_mainly_constant_encoder.cpp
 * @brief Correctness tests for MainlyConstantEncoder, including decodeGatherInto
 *        for both the raw/leaf (maxDepth_==0) and recursive (innerCodec_) cases.
 */

#include "encoders/MainlyConstantEncoder.hpp"
#include "GatherTestHelpers.hpp"
#include <iostream>
#include <random>
#include <vector>

using namespace encodings::encoders;

namespace {

std::vector<int64_t> makeMostlyConstantData(size_t n, int64_t commonValue, unsigned seed) {
    std::vector<int64_t> data(n, commonValue);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> posDist(0, n - 1);
    std::uniform_int_distribution<int64_t> valDist(1, 100000);
    // ~10% uncommon values, scattered.
    for (size_t i = 0; i < n / 10; ++i) {
        data[posDist(rng)] = valDist(rng);
    }
    return data;
}

bool testRoundTripRaw() {
    std::cout << "Test 1: decodeAll roundtrip (raw/leaf)\n";
    auto data = makeMostlyConstantData(2000, 42, 1);
    MainlyConstantEncoder<int64_t> encoder;  // maxDepth=0
    auto encoded = encoder.encode(data);
    auto decoded = encoder.decodeAll(encoded);
    bool ok = (decoded == data);
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testGatherRaw() {
    std::cout << "Test 2: decodeGatherInto against decodeAll() (raw/leaf) across trace shapes\n";
    bool allOk = true;
    for (unsigned seed : {1u, 2u, 3u}) {
        auto data = makeMostlyConstantData(3000, 42, seed);
        MainlyConstantEncoder<int64_t> encoder;
        auto encoded = encoder.encode(data);
        allOk &= encodings::testutil::checkGatherAllTraceShapes<int64_t>(encoder, encoded, data);
    }
    std::cout << "  " << (allOk ? "PASS" : "FAIL") << "\n";
    return allOk;
}

bool testGatherRecursive() {
    std::cout << "Test 3: decodeGatherInto against decodeAll() (recursive, depth=2) across trace shapes\n";
    bool allOk = true;
    for (unsigned seed : {1u, 2u}) {
        auto data = makeMostlyConstantData(5000, 42, seed);
        auto encoder = makeRecursiveMainlyConstantEncoder<int64_t>(2);
        auto encoded = encoder->encode(data);
        auto decoded = encoder->decodeAll(encoded);
        if (decoded != data) {
            std::cerr << "FAIL decodeAll roundtrip (recursive), seed=" << seed << "\n";
            allOk = false;
            continue;
        }
        allOk &= encodings::testutil::checkGatherAllTraceShapes<int64_t>(*encoder, encoded, data);
    }
    std::cout << "  " << (allOk ? "PASS" : "FAIL") << "\n";
    return allOk;
}

bool testGatherLargerScattered() {
    std::cout << "Test 4: decodeGatherInto on a larger scattered trace (raw + recursive)\n";
    auto data = makeMostlyConstantData(50000, 7, 42);
    auto ranges = encodings::benchmark::makeSelectiveTrace(data.size(), {.selectivity = 0.2, .meanRunLength = 6.0});

    MainlyConstantEncoder<int64_t> rawEncoder;
    auto rawEncoded = rawEncoder.encode(data);
    bool ok1 = encodings::testutil::checkGatherMatchesDecodeAll<int64_t>(rawEncoder, rawEncoded, data, ranges, "raw_large_scattered");

    auto recEncoder = makeRecursiveMainlyConstantEncoder<int64_t>(2);
    auto recEncoded = recEncoder->encode(data);
    bool ok2 = encodings::testutil::checkGatherMatchesDecodeAll<int64_t>(*recEncoder, recEncoded, data, ranges, "recursive_large_scattered");

    bool ok = ok1 && ok2;
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

}  // namespace

int main() {
    std::cout << "=== MainlyConstantEncoder Tests ===\n\n";
    bool ok = true;
    ok &= testRoundTripRaw();
    ok &= testGatherRaw();
    ok &= testGatherRecursive();
    ok &= testGatherLargerScattered();
    std::cout << "\n" << (ok ? "All tests PASSED" : "SOME TESTS FAILED") << "\n";
    return ok ? 0 : 1;
}
