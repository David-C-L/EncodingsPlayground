#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "generators/CommonGenerators.hpp"
#include "generators/SnowflakeIDGenerator.hpp"
#include "encoders/SubIntSplitEncoder.hpp"

using namespace encodings;
using namespace encodings::generators;
using namespace encodings::encoders;

namespace {

std::vector<size_t> parseSamples(const std::string& input) {
    std::vector<size_t> out;
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        out.push_back(static_cast<size_t>(std::stoull(token)));
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    size_t dataSize = 1'000'000;
    std::vector<size_t> sampleSizes{
        10, 20, 30, 40, 50, 60, 70, 80, 90,
        100, 200, 300, 400, 500, 600, 700, 800, 900,
        1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
        10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
        100000
    };
    std::filesystem::path outputPath = "subint_sample_sweep.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            dataSize = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--samples" && i + 1 < argc) {
            sampleSizes = parseSamples(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: sweep_subint_samples [--size N] [--samples a,b,c] [--output path]\n";
            return 0;
        }
    }

    if (sampleSizes.empty()) {
        std::cerr << "No sample sizes provided.\n";
        return 1;
    }

    std::filesystem::create_directories(outputPath.parent_path().empty() ? "." : outputPath.parent_path());

    SnowflakeIDGenerator<int64_t> generator(INSTAGRAM_SNOWFLAKE_CONFIG, 4096, 42, 0.5);
    auto data = generator.generate(dataSize);

    std::ofstream csv(outputPath);
    if (!csv) {
        std::cerr << "Failed to open output file: " << outputPath << "\n";
        return 1;
    }

    csv << "sampleSize,encodeTime_ns,selectionTime_ns,compressionRatio,bitsPerElement,compressedSize,uncompressedSize\n";

    for (size_t samples : sampleSizes) {
        auto cfg = makeDefaultAutoSubIntSplitConfig(BitSplitOrder::LSB_TO_MSB, true);
        cfg.samplerConfig.maxSamples = samples;
        cfg.samplerConfig.maxPercentage = 0.0;
        cfg.samplerConfig.stride = 0;
        cfg.debugLogging = false;

        auto encoder = makeAutoSubIntSplitEncoder(std::move(cfg));

        auto start = std::chrono::high_resolution_clock::now();
        auto encoded = encoder->encode(data);
        auto end = std::chrono::high_resolution_clock::now();
        auto encodeTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        double selectionTimeNs = 0.0;
        const auto& meta = encoded.metadata();
        auto it = meta.customMetadata.find("selectionTime_ns");
        if (it != meta.customMetadata.end()) {
            try {
                selectionTimeNs = std::stod(it->second);
            } catch (const std::exception&) {
                selectionTimeNs = 0.0;
            }
        }

        double compressionRatio = meta.uncompressedSize > 0
            ? static_cast<double>(meta.compressedSize) / static_cast<double>(meta.uncompressedSize)
            : 0.0;
        double bitsPerElement = dataSize > 0
            ? (static_cast<double>(meta.compressedSize) * 8.0) / static_cast<double>(dataSize)
            : 0.0;

        csv << samples << ','
            << encodeTimeNs << ','
            << selectionTimeNs << ','
            << compressionRatio << ','
            << bitsPerElement << ','
            << meta.compressedSize << ','
            << meta.uncompressedSize << '\n';

        std::cout << "Samples=" << samples
                  << " encode_ms=" << (encodeTimeNs / 1e6)
                  << " selection_ms=" << (selectionTimeNs / 1e6)
                  << " ratio=" << compressionRatio
                  << " bits/elem=" << bitsPerElement
                  << "\n";
    }

    std::cout << "\nSaved sweep CSV to: " << outputPath << "\n";
    return 0;
}
