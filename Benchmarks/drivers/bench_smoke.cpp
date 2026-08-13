// Correctness smoke test, not a measurement driver.
//
// This is what remains of run_benchmarks.cpp once every measurement concern was
// split into its own driver. It exists to fail loudly in CI when a codec stops
// round-tripping, so it runs one small dataset through every registered encoder
// and checks the four decode shapes against the source data. It deliberately
// reports no timings: nothing here is pinned, warmed or repeated, and a number
// produced under those conditions would invite exactly the misreading that
// CONVENTIONS.md section 3a exists to prevent.
//
// Codecs known not to round-trip are reported as failures rather than skipped --
// BlockFORFPE on TwitterSnowflake is the standing example. A driver that hides a
// known-broken codec is how it stays broken.

#include "benchmark/ArtifactCache.hpp"
#include "benchmark/CachePolicy.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/DecodeHarness.hpp"
#include "benchmark/GatherTraceGen.hpp"
#include "benchmark/PointTraceGen.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "benchmark/targets/PlaygroundTarget.hpp"

#include <iostream>
#include <string>
#include <vector>

using Elem = int64_t;

int main() {
    using namespace encodings::benchmark;

    constexpr size_t kN = 20000;

    auto datasets = int64Datasets();
    if (datasets.empty()) {
        std::cerr << "bench_smoke: no datasets available\n";
        return 1;
    }
    // A generated dataset keeps the test independent of the Datasets/ tree, which
    // may be absent in a fresh checkout.
    const DatasetEntry<Elem>* pick = nullptr;
    for (const auto& d : datasets)
        if (!d.fileBacked) { pick = &d; break; }
    if (!pick) pick = &datasets.front();

    DatasetCache<Elem> data;
    const auto handle = data.materialize(*pick, kN);
    const std::vector<Elem> reference(handle.data.begin(), handle.data.end());

    ArtifactCache<Elem> artifacts;
    CacheTopology topo = CacheTopology::detect();
    CachePolicy pol;                       // hot, passive
    CacheController cache{pol, topo};

    auto trace = buildGatherTrace(kN, {.start = 0,
                                   .span = kN,
                                   .selectivity = 0.3,
                                   .runLength = 8,
                                   .gapModel = GapModel::UniformDeterministic,
                                   .seed = 42});
    auto points = buildPointTrace({.streamLength = kN, .probes = 512,
                                   .pattern = PointPattern::Uniform, .seed = 42});

    size_t passed = 0;
    std::vector<std::string> failures;

    for (auto& enc : allEncoders()) {
        const std::string& name = enc.name;
        try {
            const auto& art = artifacts.get(enc, handle, EncodeMeasurement::None);
            PlaygroundTarget<Elem> target{*enc.codec};
            target.adopt(art.encoded);
            DecodeHarness<PlaygroundTarget<Elem>> harness{target, cache};

            std::string why;
            if (!harness.validate(std::span<const Elem>{reference}, why)) {
                failures.push_back(name + ": " + why);
                continue;
            }
            if (!harness.validateGather(trace.ranges, std::span<const Elem>{reference}, why)) {
                failures.push_back(name + ": gather: " + why);
                continue;
            }
            if (target.capabilities().randomAccess) {
                bool ok = true;
                for (size_t i = 0; i < points.indices.size() && ok; ++i) {
                    const size_t idx = points.indices[i];
                    const auto got = target.pointRead(idx);
                    ok = got.has_value() && *got == reference[idx];
                    if (!ok) failures.push_back(name + ": pointRead mismatch at row "
                                                + std::to_string(idx));
                }
                if (!ok) continue;
            }
            ++passed;
            std::cout << "  ok   " << name << "\n";
        } catch (const std::exception& e) {
            failures.push_back(name + ": threw: " + e.what());
        }
    }

    std::cout << "\nbench_smoke: " << passed << " passed, " << failures.size()
              << " failed (dataset " << pick->name << ", N=" << kN << ")\n";
    for (const auto& f : failures) std::cout << "  FAIL " << f << "\n";
    return failures.empty() ? 0 : 1;
}
