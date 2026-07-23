/**
 * @file test_exception_encoder.cpp
 * @brief Correctness tests for ExceptionEncoder<T> wrapping RawBitPackedEncoder<T>
 *        with FORExceptionDetector<T> -- Phase 1 of the exception-handling design
 *        (see ~/.claude/plans/could-you-analyse-the-misty-widget.md).
 */

#include "exceptions/ExceptionEncoder.hpp"
#include "exceptions/FORExceptionDetector.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "GatherTestHelpers.hpp"

#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace encodings::exceptions;
using encodings::encoders::RawBitPackedEncoder;

namespace {

using Codec64 = encodings::Codec<int64_t, uint8_t>;

std::shared_ptr<ExceptionEncoder<int64_t>> makeEncoder() {
    return std::make_shared<ExceptionEncoder<int64_t>>(
        std::make_shared<FORExceptionDetector<int64_t>>(),
        std::make_shared<RawBitPackedEncoder<int64_t>>());
}

// Mostly small values clustered near zero, with a small fraction of huge
// outliers scattered through the stream -- the scenario the exception
// mechanism is meant to help with.
std::vector<int64_t> makeDataWithOutliers(size_t n, double outlierFraction, unsigned seed) {
    std::vector<int64_t> data(n);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> smallDist(1000, 1100); // fits in ~7 bits of residual
    std::uniform_int_distribution<int64_t> outlierDist(1'000'000, 2'000'000'000);
    std::bernoulli_distribution isOutlier(outlierFraction);
    for (size_t i = 0; i < n; ++i) {
        data[i] = isOutlier(rng) ? outlierDist(rng) : smallDist(rng);
    }
    return data;
}

bool testEmptyData() {
    std::cout << "Test 1: empty data roundtrip\n";
    auto encoder = makeEncoder();
    std::vector<int64_t> data;
    auto encoded = encoder->encode(data);
    auto decoded = encoder->decodeAll(encoded);
    bool ok = decoded.empty();
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testSingleElement() {
    std::cout << "Test 2: single-element roundtrip\n";
    auto encoder = makeEncoder();
    std::vector<int64_t> data{42};
    auto encoded = encoder->encode(data);
    auto decoded = encoder->decodeAll(encoded);
    bool ok = (decoded == data);
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testRoundTripNoOutliers() {
    std::cout << "Test 3: decodeAll roundtrip, no outliers\n";
    auto encoder = makeEncoder();
    auto data = makeDataWithOutliers(5000, 0.0, 1);
    auto encoded = encoder->encode(data);
    auto decoded = encoder->decodeAll(encoded);
    bool ok = (decoded == data);
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testRoundTripWithOutliers() {
    std::cout << "Test 4: decodeAll roundtrip, 2% outliers\n";
    auto encoder = makeEncoder();
    auto data = makeDataWithOutliers(20000, 0.02, 2);
    auto encoded = encoder->encode(data);
    auto decoded = encoder->decodeAll(encoded);
    bool ok = (decoded == data);
    const auto exceptionCount = encoded.metadata().customMetadata.at("exception_count");
    std::cout << "  exception_count=" << exceptionCount
              << " compressedSize=" << encoded.metadata().compressedSize
              << " uncompressedSize=" << encoded.metadata().uncompressedSize << "\n";
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testDecodeAtMatchesDecodeAll() {
    std::cout << "Test 5: decodeAt agrees with decodeAll across every index\n";
    auto encoder = makeEncoder();
    auto data = makeDataWithOutliers(3000, 0.05, 3);
    auto encoded = encoder->encode(data);
    auto decoded = encoder->decodeAll(encoded);
    bool ok = true;
    for (size_t i = 0; i < data.size(); ++i) {
        auto v = encoder->decodeAt(encoded, i);
        if (!v || *v != decoded[i]) {
            std::cerr << "  FAIL at index " << i << "\n";
            ok = false;
            break;
        }
    }
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testDecodeAtOutOfBounds() {
    std::cout << "Test 6: decodeAt out-of-bounds returns nullopt\n";
    auto encoder = makeEncoder();
    auto data = makeDataWithOutliers(100, 0.1, 4);
    auto encoded = encoder->encode(data);
    bool ok = !encoder->decodeAt(encoded, data.size()).has_value()
            && !encoder->decodeAt(encoded, data.size() + 1000).has_value();
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testDecodeRangeMatchesDecodeAll() {
    std::cout << "Test 7: decodeRange matches decodeAll slice\n";
    auto encoder = makeEncoder();
    auto data = makeDataWithOutliers(2000, 0.03, 5);
    auto encoded = encoder->encode(data);
    auto full = encoder->decodeAll(encoded);
    auto slice = encoder->decodeRange(encoded, 500, 1500);
    std::vector<int64_t> expected(full.begin() + 500, full.begin() + 1500);
    bool ok = (slice == expected);
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testGatherMatchesDecodeAll() {
    std::cout << "Test 8: decodeGatherInto matches decodeAll across trace shapes\n";
    auto encoder = makeEncoder();
    auto data = makeDataWithOutliers(4000, 0.04, 6);
    auto encoded = encoder->encode(data);
    bool ok = encodings::testutil::checkGatherAllTraceShapes<int64_t>(*encoder, encoded, data);
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool testCompressesBetterThanPlainRawBitPacked() {
    std::cout << "Test 9: exception-wrapped beats plain RawBitPackedEncoder when outliers present\n";
    auto data = makeDataWithOutliers(50000, 0.01, 7);

    auto exceptionEncoder = makeEncoder();
    auto exceptionEncoded = exceptionEncoder->encode(data);

    RawBitPackedEncoder<int64_t> plainEncoder;
    auto plainEncoded = plainEncoder.encode(std::span<const int64_t>(data));

    std::cout << "  exception-wrapped: " << exceptionEncoded.metadata().compressedSize << " bytes"
              << " (ratio=" << exceptionEncoded.metadata().compressionRatio() << ")\n";
    std::cout << "  plain RawBitPacked: " << plainEncoded.metadata().compressedSize << " bytes"
              << " (ratio=" << plainEncoded.metadata().compressionRatio() << ")\n";

    bool ok = exceptionEncoded.metadata().compressedSize < plainEncoded.metadata().compressedSize;
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    std::cout << "=== ExceptionEncoder Tests ===\n\n";
    bool allPassed = true;
    allPassed &= testEmptyData();
    allPassed &= testSingleElement();
    allPassed &= testRoundTripNoOutliers();
    allPassed &= testRoundTripWithOutliers();
    allPassed &= testDecodeAtMatchesDecodeAll();
    allPassed &= testDecodeAtOutOfBounds();
    allPassed &= testDecodeRangeMatchesDecodeAll();
    allPassed &= testGatherMatchesDecodeAll();
    allPassed &= testCompressesBetterThanPlainRawBitPacked();

    std::cout << "\n" << (allPassed ? "All tests PASSED" : "SOME TESTS FAILED") << "\n";
    return allPassed ? 0 : 1;
}
