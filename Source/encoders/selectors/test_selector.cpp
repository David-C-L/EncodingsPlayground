#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"
#include "encoders/selectors/costs/EncodingCostModel.hpp"
#include "generators/SnowflakeIDGenerator.hpp"

using encodings::encoders::selectors::IDSubStreamEncodingSelector;
using encodings::encoders::selectors::SegmentPlan;
using encodings::encoders::selectors::costs::EncodingCostModel;
using encodings::encoders::selectors::costs::RawCostModel;
using encodings::encoders::selectors::costs::FORCostModel;
using encodings::encoders::selectors::costs::DictionaryCostModel;
using encodings::encoders::selectors::costs::RLECostModel;
using encodings::encoders::selectors::costs::AdaptiveFORCostModel;

static bool validatePlanCoversBits(const std::vector<SegmentPlan>& segments, std::string& error) {
    if (segments.empty()) {
        error = "segments list is empty";
        return false;
    }
    if (segments.front().bitStart != 0) {
        error = "first segment does not start at bit 0";
        return false;
    }
    if (segments.back().bitEnd != 63) {
        error = "last segment does not end at bit 63";
        return false;
    }

    int expectedStart = 0;
    for (const auto& seg : segments) {
        if (seg.bitStart != expectedStart) {
            error = "segment start mismatch";
            return false;
        }
        if (seg.bitEnd < seg.bitStart) {
            error = "segment end precedes start";
            return false;
        }
        expectedStart = seg.bitEnd + 1;
    }
    if (expectedStart != 64) {
        error = "segments do not cover all 64 bits";
        return false;
    }
    return true;
}

static std::vector<std::unique_ptr<EncodingCostModel>> createDefaultEncodings() {
    std::vector<std::unique_ptr<EncodingCostModel>> encodings;
    encodings.emplace_back(std::make_unique<RawCostModel>());
    encodings.emplace_back(std::make_unique<FORCostModel>());
    encodings.emplace_back(std::make_unique<AdaptiveFORCostModel>());
    encodings.emplace_back(std::make_unique<DictionaryCostModel>());
    encodings.emplace_back(std::make_unique<RLECostModel>());
    return encodings;
}

static IDSubStreamEncodingSelector createSelectorWithDefaultVerboseConfig() {
    return IDSubStreamEncodingSelector(IDSubStreamEncodingSelector::Config{
        .minSegmentWidth = 1,
        .splitPenalty = 0.0,
        .enablePrune = false,
        .entropyPruneThreshold = 1.0,
        .verboseLevel = 2
    });
}

static void testSimpleSample() {

    const auto encodings = createDefaultEncodings();
    IDSubStreamEncodingSelector selector = createSelectorWithDefaultVerboseConfig();

    std::vector<uint64_t> sample;
    sample.reserve(1024);
    for (uint64_t i = 0; i < 1024; ++i) {
        sample.push_back((i % 4 == 0) ? 0ULL : (i * 17ULL));
    }
    const auto result = selector.select(sample, encodings);

    if (!std::isfinite(result.total_cost)) {
        std::cerr << "Selector test failed: total_cost is not finite.\n";
        return;
    }
    {
        std::string error;
        if (!validatePlanCoversBits(result.segments, error)) {
            std::cerr << "Selector test failed: " << error << ".\n";
            return;
        }
    }

    std::cout << "Simple sample test passed with " << result.segments.size() << " segments.\n";
    std::cout << result.toString() << "\n";
}

static void testSnowflakeSample() {
    const auto encodings = createDefaultEncodings();
    IDSubStreamEncodingSelector selector = createSelectorWithDefaultVerboseConfig();
    encodings::datagen::SnowflakeIDGenerator<uint64_t> snowflake(
    encodings::datagen::INSTAGRAM_SNOWFLAKE_CONFIG,
        4096,
        42,
        0.5);
    const auto snowflakeSample = snowflake.generate(10000);
    const auto snowflakeResult = selector.select(snowflakeSample, encodings);
    if (!std::isfinite(snowflakeResult.total_cost)) {
        std::cerr << "Snowflake selector test failed: total_cost is not finite.\n";
        return;
    }
    {
        std::string error;
        if (!validatePlanCoversBits(snowflakeResult.segments, error)) {
            std::cerr << "Snowflake selector test failed: " << error << ".\n";
            return;
        }
    }

    std::cout << "Selector test passed with " << snowflakeResult.segments.size() << " segments.\n";
    std::cout << snowflakeResult.toString() << "\n";
}

int main() {
    try {
        testSimpleSample();
        testSnowflakeSample();
    } catch (const std::exception& ex) {
        std::cerr << "Test failed with exception: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "All selector tests passed successfully.\n";
    return 0;
}
