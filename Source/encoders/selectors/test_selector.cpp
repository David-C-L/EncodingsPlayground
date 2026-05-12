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

static bool validatePlanCoversBits(const std::vector<SegmentPlan>& segments,
                                  int maxBit,
                                  std::string& error) {
    if (segments.empty()) {
        error = "segments list is empty";
        return false;
    }
    if (segments.front().bitStart != 0) {
        error = "first segment does not start at bit 0";
        return false;
    }
    if (segments.back().bitEnd != maxBit) {
        error = "last segment does not end at expected max bit";
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
    if (expectedStart != maxBit + 1) {
        error = "segments do not cover all bits";
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
        if (!validatePlanCoversBits(result.segments, 63, error)) {
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
        if (!validatePlanCoversBits(snowflakeResult.segments, 63, error)) {
            std::cerr << "Snowflake selector test failed: " << error << ".\n";
            return;
        }
    }

    std::cout << "Selector test passed with " << snowflakeResult.segments.size() << " segments.\n";
    std::cout << snowflakeResult.toString() << "\n";
}

static void testForcedNumSegmentsDP() {
    const auto encodings = createDefaultEncodings();

    std::vector<uint64_t> sample;
    sample.reserve(1024);
    for (uint64_t i = 0; i < 1024; ++i)
        sample.push_back((i % 4 == 0) ? 0ULL : (i * 17ULL));

    for (int K : {1, 2, 3, 4}) {
        IDSubStreamEncodingSelector selector(IDSubStreamEncodingSelector::Config{
            .minSegmentWidth = 1,
            .splitPenalty = 0.0,
            .forcedNumSegments = K
        });
        const auto result = selector.select(sample, encodings);

        if (!std::isfinite(result.total_cost)) {
            std::cerr << "forcedNumSegments DP test failed (K=" << K << "): total_cost is not finite.\n";
            return;
        }
        if (static_cast<int>(result.segments.size()) != K) {
            std::cerr << "forcedNumSegments DP test failed (K=" << K << "): expected " << K
                      << " segments, got " << result.segments.size() << ".\n";
            return;
        }
        std::string error;
        if (!validatePlanCoversBits(result.segments, 63, error)) {
            std::cerr << "forcedNumSegments DP test failed (K=" << K << "): " << error << ".\n";
            return;
        }
        std::cout << "  K=" << K << ": " << result.segments.size()
                  << " segments, cost=" << result.total_cost << "\n";
    }
    std::cout << "forcedNumSegments DP test passed.\n";
}

static void testForcedNumSegmentsDPvsExhaustive() {
    // DP and exhaustive search must find the same optimal cost for every forced K
    const auto encodings = createDefaultEncodings();

    std::vector<uint64_t> sample;
    for (uint64_t i = 0; i < 200; ++i)
        sample.push_back(i * 31ULL);

    for (int K : {1, 2, 3}) {
        IDSubStreamEncodingSelector::Config baseCfg{
            .minSegmentWidth = 1,
            .splitPenalty = 0.0,
            .forcedNumSegments = K
        };

        IDSubStreamEncodingSelector::Config exCfg = baseCfg;
        exCfg.useExhaustiveSearch = true;

        const auto dpResult = IDSubStreamEncodingSelector(baseCfg).select(sample, encodings);
        const auto exResult = IDSubStreamEncodingSelector(exCfg).select(sample, encodings);

        if (!std::isfinite(dpResult.total_cost) || !std::isfinite(exResult.total_cost)) {
            std::cerr << "forcedNumSegments DP vs exhaustive test failed (K=" << K << "): non-finite cost.\n";
            return;
        }
        if (static_cast<int>(dpResult.segments.size()) != K
            || static_cast<int>(exResult.segments.size()) != K) {
            std::cerr << "forcedNumSegments DP vs exhaustive test failed (K=" << K
                      << "): wrong segment count (dp=" << dpResult.segments.size()
                      << " ex=" << exResult.segments.size() << ").\n";
            return;
        }
        if (std::abs(dpResult.total_cost - exResult.total_cost) > 1e-9) {
            std::cerr << "forcedNumSegments DP vs exhaustive cost mismatch (K=" << K
                      << "): dp=" << dpResult.total_cost
                      << " ex=" << exResult.total_cost << ".\n";
            return;
        }
        std::cout << "  K=" << K << ": dp_cost=" << dpResult.total_cost
                  << " ex_cost=" << exResult.total_cost << " (match)\n";
    }
    std::cout << "forcedNumSegments DP vs exhaustive agreement test passed.\n";
}

static void testForcedNumSegmentsEdgeCases() {
    const auto encodings = createDefaultEncodings();

    std::vector<uint64_t> sample;
    for (uint64_t i = 0; i < 512; ++i)
        sample.push_back(i);

    // K=1: must produce exactly one segment spanning all 64 bits
    {
        IDSubStreamEncodingSelector selector(IDSubStreamEncodingSelector::Config{
            .minSegmentWidth = 1,
            .splitPenalty = 5.0,
            .forcedNumSegments = 1
        });
        const auto result = selector.select(sample, encodings);
        if (static_cast<int>(result.segments.size()) != 1) {
            std::cerr << "forcedNumSegments K=1 failed: got " << result.segments.size() << " segments.\n";
            return;
        }
        if (result.segments[0].bitStart != 0 || result.segments[0].bitEnd != 63) {
            std::cerr << "forcedNumSegments K=1 failed: segment covers ["
                      << result.segments[0].bitStart << ".."
                      << result.segments[0].bitEnd << "], expected [0..63].\n";
            return;
        }
        std::cout << "  K=1 single-segment edge case passed.\n";
    }

    // forcedNumSegments exceeds max feasible (64 bits / minSegmentWidth=32 → max 2 segments)
    // should clamp to 2 and return a valid plan
    {
        IDSubStreamEncodingSelector selector(IDSubStreamEncodingSelector::Config{
            .minSegmentWidth = 32,
            .splitPenalty = 0.0,
            .forcedNumSegments = 100
        });
        const auto result = selector.select(sample, encodings);
        if (result.segments.size() != 2) {
            std::cerr << "forcedNumSegments clamp test failed: expected 2 segments after clamp, got "
                      << result.segments.size() << ".\n";
            return;
        }
        std::string error;
        if (!validatePlanCoversBits(result.segments, 63, error)) {
            std::cerr << "forcedNumSegments clamp test failed: " << error << ".\n";
            return;
        }
        std::cout << "  Over-large K clamp edge case passed.\n";
    }

    std::cout << "forcedNumSegments edge case tests passed.\n";
}

static void testForcedNumSegmentsSnowflake() {
    // Realistic data: verify forced K produces valid plans and K=1 cost >= unconstrained cost
    const auto encodings = createDefaultEncodings();
    encodings::datagen::SnowflakeIDGenerator<uint64_t> snowflake(
        encodings::datagen::INSTAGRAM_SNOWFLAKE_CONFIG,
        4096,
        42,
        0.5);
    const auto sample = snowflake.generate(2000);

    double unconstrainedCost = 0.0;
    {
        IDSubStreamEncodingSelector selector(IDSubStreamEncodingSelector::Config{
            .minSegmentWidth = 1,
            .splitPenalty = 0.0
        });
        const auto result = selector.select(sample, encodings);
        unconstrainedCost = result.total_cost;
    }

    for (int K : {1, 2, 3, 4}) {
        IDSubStreamEncodingSelector selector(IDSubStreamEncodingSelector::Config{
            .minSegmentWidth = 1,
            .splitPenalty = 0.0,
            .forcedNumSegments = K
        });
        const auto result = selector.select(sample, encodings);

        if (!std::isfinite(result.total_cost)) {
            std::cerr << "forcedNumSegments snowflake test failed (K=" << K << "): non-finite cost.\n";
            return;
        }
        if (static_cast<int>(result.segments.size()) != K) {
            std::cerr << "forcedNumSegments snowflake test failed (K=" << K << "): got "
                      << result.segments.size() << " segments.\n";
            return;
        }
        std::string error;
        if (!validatePlanCoversBits(result.segments, 63, error)) {
            std::cerr << "forcedNumSegments snowflake test failed (K=" << K << "): " << error << ".\n";
            return;
        }
        // Constrained cost must be >= unconstrained cost (unconstrained is globally optimal)
        if (result.total_cost < unconstrainedCost - 1e-9) {
            std::cerr << "forcedNumSegments snowflake test failed (K=" << K
                      << "): constrained cost " << result.total_cost
                      << " < unconstrained cost " << unconstrainedCost << ".\n";
            return;
        }
        std::cout << "  K=" << K << ": cost=" << result.total_cost
                  << " (unconstrained=" << unconstrainedCost << ")\n";
    }
    std::cout << "forcedNumSegments snowflake test passed.\n";
}

static void testForcedNumSegments32() {
    const auto encodings = createDefaultEncodings();

    std::vector<uint32_t> sample;
    sample.reserve(1024);
    for (uint32_t i = 0; i < 1024; ++i)
        sample.push_back(i * 7U);

    for (int K : {1, 2, 3}) {
        IDSubStreamEncodingSelector selector(IDSubStreamEncodingSelector::Config{
            .minSegmentWidth = 1,
            .splitPenalty = 0.0,
            .forcedNumSegments = K
        });
        const auto result = selector.select(sample, encodings);

        if (!std::isfinite(result.total_cost)) {
            std::cerr << "forcedNumSegments 32-bit test failed (K=" << K << "): non-finite cost.\n";
            return;
        }
        if (static_cast<int>(result.segments.size()) != K) {
            std::cerr << "forcedNumSegments 32-bit test failed (K=" << K << "): got "
                      << result.segments.size() << " segments.\n";
            return;
        }
        std::string error;
        if (!validatePlanCoversBits(result.segments, 31, error)) {
            std::cerr << "forcedNumSegments 32-bit test failed (K=" << K << "): " << error << ".\n";
            return;
        }
        std::cout << "  K=" << K << ": cost=" << result.total_cost << "\n";
    }
    std::cout << "forcedNumSegments 32-bit test passed.\n";
}

static void testSimpleSample32() {
    const auto encodings = createDefaultEncodings();
    IDSubStreamEncodingSelector selector = createSelectorWithDefaultVerboseConfig();

    std::vector<uint32_t> sample;
    sample.reserve(2048);
    for (uint32_t i = 0; i < 2048; ++i) {
        sample.push_back((i % 5 == 0) ? 0U : (i * 13U));
    }

    const auto result = selector.select(sample, encodings);
    if (!std::isfinite(result.total_cost)) {
        std::cerr << "Selector 32-bit test failed: total_cost is not finite.\n";
        return;
    }
    {
        std::string error;
        if (!validatePlanCoversBits(result.segments, 31, error)) {
            std::cerr << "Selector 32-bit test failed: " << error << ".\n";
            return;
        }
    }

    std::cout << "Simple 32-bit sample test passed with " << result.segments.size() << " segments.\n";
    std::cout << result.toString() << "\n";
}

int main() {
    try {
        testSimpleSample();
        testSnowflakeSample();
        testSimpleSample32();
        testForcedNumSegmentsDP();
        testForcedNumSegmentsDPvsExhaustive();
        testForcedNumSegmentsEdgeCases();
        testForcedNumSegmentsSnowflake();
        testForcedNumSegments32();
    } catch (const std::exception& ex) {
        std::cerr << "Test failed with exception: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "All selector tests passed successfully.\n";
    return 0;
}
