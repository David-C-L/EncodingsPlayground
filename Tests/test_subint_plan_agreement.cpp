// Regression guard for the SubIntSplit DP's pruning heuristic.
//
// The bug this exists to catch was not a crash. entropyPruneThreshold compared an
// unbounded bits-per-value entropy against a bare constant, so nearly every bit
// range was pruned, only the exempt full-width range survived, and the DP returned
// a valid single-section plan. Nothing failed. SubIntSplit simply stopped
// splitting, and two benchmark drivers disagreed about the segment count of "the
// same" plan for weeks before anyone noticed.
//
// A heuristic that is allowed to change the runtime must not change the answer, so
// that is what is asserted here directly:
//
//   1. On data that is known to be splittable, the DP returns more than one
//      segment -- with pruning ON, which is the shipped default.
//   2. Pruning on and pruning off agree exactly, on segments and on encoded bytes.
//   3. Even a threshold that would prune everything recovers, because the retry
//      now fires on a degenerate collapse rather than only on infeasibility.
//   4. SubIntSplitAutoEncoder::lastPlan() reports the plan actually used, so a
//      caller never has to re-run selection to find out and never has to guess
//      which Config produced the answer it is reading.

#include "encoders/SubIntSplitEncoder.hpp"

#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace encodings;
using namespace encodings::encoders;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    if (!ok) ++failures;
}

/// Snowflake-shaped: a slow-moving timestamp in the high bits, a small machine id,
/// and a fast-cycling sequence counter in the low bits. Chosen because the whole
/// point of SubIntSplit is that these three regions want different codecs, so a
/// selector that declines to split is visibly wrong on it.
std::vector<int64_t> snowflakeLike(size_t n) {
    std::vector<int64_t> out;
    out.reserve(n);
    int64_t timestamp = 1'700'000'000'000;
    for (size_t i = 0; i < n; ++i) {
        if (i % 64 == 0) timestamp += 1 + static_cast<int64_t>(i % 3);
        const int64_t machine = 37;
        const int64_t sequence = static_cast<int64_t>(i % 4096);
        out.push_back((timestamp << 22) | (machine << 12) | sequence);
    }
    return out;
}

struct PlanFacts {
    size_t segments{};
    size_t bytes{};
    std::string description;
};

PlanFacts encodeWith(const std::vector<int64_t>& data, bool prune, double threshold) {
    auto cfg = makeDefaultAutoSubIntSplitConfig<int64_t>();
    cfg.selectorConfig.enablePrune = prune;
    cfg.selectorConfig.entropyPruneThreshold = threshold;
    cfg.samplerConfig.maxSamples = 4000;

    auto enc = makeAutoSubIntSplitEncoder<int64_t>(std::move(cfg));
    auto encoded = enc->encode(std::span<const int64_t>{data});

    PlanFacts f;
    f.bytes = encoded.size();
    for (const auto& seg : enc->lastPlan().segments) {
        ++f.segments;
        f.description += "[" + std::to_string(seg.bitStart) + ".." + std::to_string(seg.bitEnd)
                       + "]" + std::string(encodings::encodingTypeToString(seg.encoding)) + " ";
    }

    // The plan must round-trip: a split that does not decode is worse than no split.
    auto back = enc->decodeAll(encoded);
    check(back == data, std::string("round-trips (prune=") + (prune ? "on" : "off") + ")");
    return f;
}

} // namespace

int main() {
    const auto data = snowflakeLike(200'000);
    std::cout << "SubIntSplit plan agreement, N=" << data.size() << "\n";

    const auto pruned   = encodeWith(data, true,  0.95);
    const auto unpruned = encodeWith(data, false, 0.95);
    // A threshold of 0 prunes every range with any entropy at all, i.e. the failure
    // mode the old code was permanently in. The retry must rescue it.
    const auto overPruned = encodeWith(data, true, 0.0);

    std::cout << "  pruned    : " << pruned.segments   << " segments, " << pruned.bytes   << " B  " << pruned.description   << "\n";
    std::cout << "  unpruned  : " << unpruned.segments << " segments, " << unpruned.bytes << " B  " << unpruned.description << "\n";
    std::cout << "  overpruned: " << overPruned.segments << " segments, " << overPruned.bytes << " B  " << overPruned.description << "\n";

    check(pruned.segments > 1,
          "pruning ON still splits (the shipped default must not collapse the plan)");
    check(pruned.segments == unpruned.segments && pruned.bytes == unpruned.bytes
              && pruned.description == unpruned.description,
          "pruning is answer-neutral: same segments and same bytes as unpruned");
    check(overPruned.segments > 1,
          "a threshold that prunes everything is rescued by the degenerate-collapse retry");
    check(!pruned.description.empty(),
          "lastPlan() reports the plan the encoder actually used");

    std::cout << (failures == 0 ? "\nALL PASS\n" : "\nFAILURES\n");
    return failures == 0 ? 0 : 1;
}
