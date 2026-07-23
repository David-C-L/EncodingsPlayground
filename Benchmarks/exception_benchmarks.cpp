// Exception-handling benchmarks (Phase 1 of the exception-handling design,
// see ~/.claude/plans/could-you-analyse-the-misty-widget.md): compares
// ExceptionEncoder<FORExceptionDetector, RawBitPackedEncoder> against plain
// RawBitPackedEncoder across synthetic datasets parameterized by outlier
// fraction and clustering (scattered vs. bursty), the two axes the design
// doc calls out as relevant to which BitmapRank representation wins
// (Dense here; Phase 2 adds RLE).
//
// Deliberately self-contained rather than built on BenchmarkRunner/
// BenchmarkConfig: this needs its own outlier-fraction-parameterized
// dataset generation loop, not another row in the general encoder/dataset
// matrix -- exactly the same reasoning that keeps reordering_benchmarks.cpp
// a separate file from run_benchmarks.cpp.

#include "encoders/RawBitPackedEncoder.hpp"
#include "exceptions/ExceptionEncoder.hpp"
#include "exceptions/FORExceptionDetector.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using encodings::encoders::RawBitPackedEncoder;
using namespace encodings::exceptions;

namespace {

// ---------------------------------------------------------------------------
// Synthetic outlier dataset generator
// ---------------------------------------------------------------------------

enum class OutlierPattern { Scattered, Clustered };

std::vector<int64_t> makeOutlierData(size_t n, double outlierFraction,
                                       OutlierPattern pattern, unsigned seed) {
    std::vector<int64_t> data(n);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> smallDist(1000, 1100);
    std::uniform_int_distribution<int64_t> outlierDist(1'000'000, 2'000'000'000);

    for (auto& v : data) v = smallDist(rng);

    const size_t outlierCount = static_cast<size_t>(static_cast<double>(n) * outlierFraction);
    if (outlierCount == 0) return data;

    if (pattern == OutlierPattern::Scattered) {
        std::uniform_int_distribution<size_t> posDist(0, n - 1);
        for (size_t i = 0; i < outlierCount; ++i) {
            data[posDist(rng)] = outlierDist(rng);
        }
    } else {
        // A handful of contiguous bursts summing to ~outlierCount positions.
        const size_t numBursts = std::max<size_t>(1, outlierCount / 200);
        const size_t burstLen = std::max<size_t>(1, outlierCount / numBursts);
        std::uniform_int_distribution<size_t> startDist(0, n > burstLen ? n - burstLen : 0);
        for (size_t b = 0; b < numBursts; ++b) {
            const size_t start = startDist(rng);
            for (size_t i = start; i < std::min(n, start + burstLen); ++i) {
                data[i] = outlierDist(rng);
            }
        }
    }
    return data;
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

template <typename F>
int64_t timedNs(F&& f) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    f();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now() - t0)
        .count();
}

struct Result {
    std::string datasetLabel;
    std::string encoderLabel;
    size_t compressedBytes{};
    size_t uncompressedBytes{};
    int64_t encodeNs{};
    int64_t decodeNs{};
    bool roundTripOk{};
};

Result benchmarkRawBitPacked(const std::string& label, const std::vector<int64_t>& data) {
    RawBitPackedEncoder<int64_t> encoder;
    Result r;
    r.datasetLabel = label;
    r.encoderLabel = "RawBitPacked (plain)";
    encodings::EncodedBuffer<uint8_t> encoded;
    r.encodeNs = timedNs([&] { encoded = encoder.encode(std::span<const int64_t>(data)); });
    std::vector<int64_t> decoded;
    r.decodeNs = timedNs([&] { decoded = encoder.decodeAll(encoded); });
    r.compressedBytes = encoded.metadata().compressedSize;
    r.uncompressedBytes = encoded.metadata().uncompressedSize;
    r.roundTripOk = (decoded == data);
    return r;
}

Result benchmarkExceptionWrapped(const std::string& label, const std::vector<int64_t>& data) {
    auto encoder = std::make_shared<ExceptionEncoder<int64_t>>(
        std::make_shared<FORExceptionDetector<int64_t>>(),
        std::make_shared<RawBitPackedEncoder<int64_t>>());
    Result r;
    r.datasetLabel = label;
    r.encoderLabel = "ExceptionWrapped(FOR|RawBitPacked)";
    encodings::EncodedBuffer<uint8_t> encoded;
    r.encodeNs = timedNs([&] { encoded = encoder->encode(std::span<const int64_t>(data)); });
    std::vector<int64_t> decoded;
    r.decodeNs = timedNs([&] { decoded = encoder->decodeAll(encoded); });
    r.compressedBytes = encoded.metadata().compressedSize;
    r.uncompressedBytes = encoded.metadata().uncompressedSize;
    r.roundTripOk = (decoded == data);
    return r;
}

void printResults(const std::vector<Result>& results) {
    std::cout << std::left
              << std::setw(28) << "Dataset"
              << std::setw(36) << "Encoder"
              << std::setw(14) << "Bytes"
              << std::setw(10) << "Ratio"
              << std::setw(14) << "Encode(us)"
              << std::setw(14) << "Decode(us)"
              << std::setw(8) << "OK"
              << '\n'
              << std::string(124, '-') << '\n';
    for (const auto& r : results) {
        const double ratio = r.uncompressedBytes > 0
            ? static_cast<double>(r.compressedBytes) / static_cast<double>(r.uncompressedBytes)
            : 0.0;
        std::cout << std::left
                  << std::setw(28) << r.datasetLabel
                  << std::setw(36) << r.encoderLabel
                  << std::setw(14) << r.compressedBytes
                  << std::setw(10) << std::fixed << std::setprecision(4) << ratio
                  << std::setw(14) << (r.encodeNs / 1000)
                  << std::setw(14) << (r.decodeNs / 1000)
                  << std::setw(8) << (r.roundTripOk ? "yes" : "NO")
                  << '\n';
    }
}

} // namespace

int main() {
    std::cout << "=== Exception-Handling Benchmarks (Phase 1) ===\n\n";

    constexpr size_t kN = 500'000;
    struct Scenario {
        std::string label;
        double outlierFraction;
        OutlierPattern pattern;
    };
    const std::vector<Scenario> scenarios = {
        {"0% outliers", 0.0, OutlierPattern::Scattered},
        {"0.1% scattered", 0.001, OutlierPattern::Scattered},
        {"1% scattered", 0.01, OutlierPattern::Scattered},
        {"5% scattered", 0.05, OutlierPattern::Scattered},
        {"1% clustered", 0.01, OutlierPattern::Clustered},
        {"5% clustered", 0.05, OutlierPattern::Clustered},
    };

    std::vector<Result> results;
    bool anyRoundTripFailed = false;
    unsigned seed = 1000;
    for (const auto& sc : scenarios) {
        auto data = makeOutlierData(kN, sc.outlierFraction, sc.pattern, seed++);
        auto plain = benchmarkRawBitPacked(sc.label, data);
        auto wrapped = benchmarkExceptionWrapped(sc.label, data);
        anyRoundTripFailed |= !plain.roundTripOk || !wrapped.roundTripOk;
        results.push_back(plain);
        results.push_back(wrapped);
    }

    printResults(results);

    if (anyRoundTripFailed) {
        std::cerr << "\nFAIL: at least one round-trip mismatch detected.\n";
        return 1;
    }
    std::cout << "\nAll round-trips verified correct.\n";
    return 0;
}
