// Oracle over the FrequencyPartition positional-index dimension.
//
// The SubIntSplit dimension has a cost model, so "cost-model accuracy" there
// means estimate-versus-actual and bench_costmodel_oracle measures it.  This
// dimension has no cost model at all: FreqPartIndexType is a template parameter
// with no DP, no candidate enumeration and nothing that predicts it, so there is
// nothing an estimate could be accurate against.
//
// The oracle here is therefore the argmin of a MEASURED objective, not of an
// estimate, and this driver exists to be the ground truth a future cost model is
// validated against.  It is deliberately shaped so that adding that model needs
// no change here: `predictor_pick` is a slot, currently filled by a cheap
// heuristic over statistics the encoders already publish, and `regret_bytes` /
// `regret_ns` score whatever occupies it exactly as top-1 and regret score the
// SubIntSplit cost model.  Swap the heuristic for a real model and the table
// shape is unchanged.
//
// Two objectives are reported because they answer different questions.  The
// Pareto frontier over (bytes, latency) says which index types are viable at
// all; a type off the frontier is beaten on both axes at once and no weighting
// can rescue it.  The scalarised oracle J = bytes + lambda*time then says which
// of the survivors wins as a function of how much space is worth relative to
// time, which is the choice a cost model would actually have to make.

#include "benchmark/ArtifactCache.hpp"
#include "benchmark/CachePolicy.hpp"
#include "benchmark/Cli.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/DecodeHarness.hpp"
#include "benchmark/GatherTraceGen.hpp"
#include "benchmark/MeasureLoop.hpp"
#include "benchmark/PointTraceGen.hpp"
#include "benchmark/ResultWriter.hpp"
#include "benchmark/RunManifest.hpp"
#include "benchmark/TimingStats.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "benchmark/targets/PlaygroundTarget.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace encodings;
using namespace encodings::benchmark;

namespace {

using Elem = int64_t;

struct Config {
    size_t n{1'000'000};
    size_t probes{1u << 14};
    size_t iterations{21};
    size_t warmup{2};
    uint64_t seed{42};
    double sigma{0.1};
    size_t runLength{8};
    std::vector<std::string> datasetFilters;
    std::vector<double> lambdas;   // ns per byte; see sweepLambdas()
    std::string cacheState{"hot"};
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_index_oracle.csv"};
};

/// Per-index-type measurement for one (dataset, N) cell.
struct IndexPoint {
    std::string name;          ///< index type
    size_t payloadBytes{};
    std::optional<int64_t> indexBytes;
    double bulkNs{}, pointNsUniform{}, pointNsZipf{}, gatherNs{};
    bool randomAccess{};
    bool viable{true};         ///< false => excluded from the frontier, not silently dropped
    // Predictor inputs, from metadata the encoders already publish.
    std::optional<double> avgNumTiers, avgFallbackFraction;
};

/// Latency term of the objective.  Point latency is what a positional index is
/// FOR, so it is the term the oracle weighs against bytes; bulk and gather are
/// recorded alongside because an index that wrecks them is not free either.
double latencyTerm(const IndexPoint& p) { return p.pointNsUniform; }

/// A point is Pareto-optimal when nothing else is at least as good on both axes
/// and strictly better on one.
bool isPareto(const IndexPoint& p, const std::vector<IndexPoint>& all) {
    for (const auto& q : all) {
        if (&q == &p || !q.viable) continue;
        const bool noWorse = q.payloadBytes <= p.payloadBytes && latencyTerm(q) <= latencyTerm(p);
        const bool better  = q.payloadBytes <  p.payloadBytes || latencyTerm(q) <  latencyTerm(p);
        if (noWorse && better) return false;
    }
    return true;
}

/// Default lambda ladder, in nanoseconds per byte.
///
/// Log-spaced across the range where the answer can actually change: at the low
/// end bytes dominate and the smallest payload always wins, at the high end
/// latency dominates and the fastest always does.  A sweep that does not cross
/// the crossover reports one regime twice, which is the same trap the working-set
/// axis has in the cache-state drivers.
std::vector<double> sweepLambdas() {
    std::vector<double> out;
    for (int e = -6; e <= 2; ++e) out.push_back(std::pow(10.0, e));
    return out;
}

/// The predictor slot.
///
/// Deliberately crude: it is a placeholder occupying the interface a real cost
/// model will use, not a contribution. It picks on the same statistics a model
/// would have cheaply available at selection time -- tier count and fallback
/// fraction, both already published as encoder metadata -- rather than on
/// anything measured, because a predictor that consults measurements is not a
/// predictor.
std::string predictorPick(const std::vector<IndexPoint>& points, double lambda) {
    // A stream that mostly falls back has no tiers worth indexing, so the index
    // is pure overhead; otherwise prefer the compact index unless time is being
    // valued heavily, in which case take the fastest structure.
    const IndexPoint* noIndex = nullptr;
    for (const auto& p : points) if (p.name.find("NoIndex") != std::string::npos) noIndex = &p;
    if (noIndex && noIndex->avgFallbackFraction && *noIndex->avgFallbackFraction > 0.9)
        return noIndex->name;
    for (const auto& p : points)
        if (lambda >= 1.0 && p.name.find("PerTierBitmaps") != std::string::npos) return p.name;
    for (const auto& p : points)
        if (p.name.find("EliasFano") != std::string::npos) return p.name;
    return points.empty() ? std::string{"-"} : points.front().name;
}

std::optional<double> metaNumber(const EncodedArtifact<Elem>& art, const std::string& key) {
    auto it = art.encodeCustomMetrics.find(key);
    if (it == art.encodeCustomMetrics.end()) return std::nullopt;
    return it->second;
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    ArgParser parser{"bench_index_oracle",
                     "Measured-argmin oracle over the FPE positional-index types: which "
                     "index a perfect chooser would pick, as a function of how much space "
                     "is worth relative to time."};
    parser.group("Sweep")
          .opt("--n", cfg.n, "stream length in elements")
          .opt("--probes", cfg.probes, "point probes per cell")
          .opt("--sigma", cfg.sigma, "gather selectivity")
          .opt("--run-length", cfg.runLength, "gather run length")
          .repeated("--dataset", cfg.datasetFilters, "only datasets containing SUBSTR")
          .group("Measurement")
          .opt("--iterations", cfg.iterations, "timed iterations per cell")
          .opt("--warmup", cfg.warmup, "untimed iterations per cell")
          .opt("--seed", cfg.seed, "seed for every random choice")
          .opt("--cache-state", cfg.cacheState, "hot | cold-payload | cold-all")
          .group("Output")
          .opt("--output", cfg.output, "result file")
          .flag("--validate", cfg.validate, "round-trip before measuring")
          .flag("--dry-run", cfg.dryRun, "print the plan and exit");

    switch (parser.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    cfg.lambdas = sweepLambdas();

    auto datasets = applyFilters(int64Datasets(), cfg.datasetFilters);
    if (datasets.empty()) {
        std::cerr << "ERROR: no datasets match --dataset filters\n";
        return 1;
    }
    auto indexTypes = fpeIndexFamily();

    const auto topo = CacheTopology::detect();
    std::cout << "\n── Index oracle ─────────────────────────────────────────────\n"
              << "  N              " << cfg.n << "\n"
              << "  index types    " << indexTypes.size() << "\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  lambdas        " << cfg.lambdas.size() << " (1e-6 .. 1e2 ns/byte)\n"
              << "  cache          " << cfg.cacheState << "\n"
              << "  topology       " << topo.describe() << "\n";
    if (cfg.dryRun) { std::cout << "\nDry run: no measurements taken.\n"; return 0; }

    RunManifest manifest = RunManifest::capture("bench_index_oracle", parser.argvEcho());
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    manifest.seed = cfg.seed;
    manifest.writeSidecar(cfg.output);

    ResultWriter writer{cfg.output,
        {stringCol("driver"), stringCol("dataset"), stringCol("index_type"),
         intCol("N"), intCol("seed"), stringCol("cache_state"),
         intCol("payload_bytes"), intCol("index_bytes"), doubleCol("index_pct_of_payload"),
         doubleCol("bulk_ns"), doubleCol("point_ns_uniform"), doubleCol("point_ns_zipf"),
         doubleCol("gather_ns"), boolCol("random_access"), boolCol("viable"),
         boolCol("on_pareto_frontier"),
         doubleCol("lambda_ns_per_byte"), doubleCol("objective_J"),
         stringCol("oracle_pick"), stringCol("predictor_pick"),
         doubleCol("regret_bytes"), doubleCol("regret_ns")},
        ResultFormat::Csv};

    CachePolicy pol;
    if (cfg.cacheState == "cold-payload") pol.state = CacheState::ColdPayload;
    else if (cfg.cacheState == "cold-all") pol.state = CacheState::ColdAll;
    CacheController cache{pol, topo};

    DatasetCache<Elem> data;
    ArtifactCache<Elem> artifacts{{.maxResident = 4}};
    std::vector<std::string> failures;
    MeasureSpec spec{.iterations = cfg.iterations, .warmup = cfg.warmup};

    for (auto& ds : datasets) {
        std::cout << "\n══ " << ds.name << " ══\n";
        typename DatasetCache<Elem>::Handle handle;
        try {
            handle = data.materialize(ds, cfg.n);
        } catch (const std::exception& e) {
            std::cerr << "  skipped: " << e.what() << "\n";
            continue;
        }
        const std::vector<Elem> reference(handle.data.begin(), handle.data.end());
        const size_t n = handle.n;

        auto trace = buildGatherTrace(n, {.start = 0, .span = n, .selectivity = cfg.sigma,
                                          .runLength = cfg.runLength,
                                          .gapModel = GapModel::UniformDeterministic,
                                          .seed = cfg.seed});
        auto uniform = buildPointTrace({.streamLength = n, .probes = cfg.probes,
                                        .pattern = PointPattern::Uniform, .seed = cfg.seed});
        auto zipf = buildPointTrace({.streamLength = n, .probes = cfg.probes,
                                     .pattern = PointPattern::Zipf, .zipfTheta = 1.4,
                                     .seed = cfg.seed});

        std::vector<Elem> sink(n);
        std::vector<IndexPoint> points;

        for (auto& enc : indexTypes) {
            IndexPoint pt;
            pt.name = enc.variant.empty() ? enc.name : enc.variant;
            try {
                const auto& art = artifacts.get(enc, handle, EncodeMeasurement::None);
                PlaygroundTarget<Elem> target{*enc.codec};
                target.adopt(art.encoded);
                DecodeHarness<PlaygroundTarget<Elem>> harness{target, cache};

                pt.payloadBytes = art.payloadBytes;
                pt.randomAccess = target.capabilities().randomAccess;
                if (auto ib = metaNumber(art, "index_bytes")) pt.indexBytes = static_cast<int64_t>(*ib);
                pt.avgNumTiers = metaNumber(art, "avg_num_tiers");
                pt.avgFallbackFraction = metaNumber(art, "avg_fallback_fraction");

                std::string why;
                if (cfg.validate && !harness.validate(std::span<const Elem>{reference}, why)) {
                    failures.push_back(ds.name + "/" + pt.name + ": " + why);
                    pt.viable = false;
                    points.push_back(pt);
                    continue;
                }

                pt.bulkNs   = static_cast<double>(harness.bulk(std::span<Elem>{sink}, spec).time.medianNs);
                pt.gatherNs = static_cast<double>(
                    harness.gather(trace.ranges, std::span<Elem>{sink}, trace.selectedRows, spec)
                        .time.medianNs);
                if (pt.randomAccess) {
                    const auto u = harness.points(uniform, spec);
                    const auto z = harness.points(zipf, spec);
                    pt.pointNsUniform = static_cast<double>(u.time.medianNs)
                                      / static_cast<double>(cfg.probes);
                    pt.pointNsZipf    = static_cast<double>(z.time.medianNs)
                                      / static_cast<double>(cfg.probes);
                } else {
                    // Not viable for the objective: a positional oracle cannot rank a
                    // codec that cannot answer a positional query. Recorded, not dropped.
                    pt.viable = false;
                }
                std::cout << "  " << pt.name << ": payload=" << pt.payloadBytes
                          << " index=" << (pt.indexBytes ? std::to_string(*pt.indexBytes) : "-")
                          << " point=" << pt.pointNsUniform << "ns\n";
            } catch (const std::exception& e) {
                failures.push_back(ds.name + "/" + pt.name + ": threw: " + e.what());
                pt.viable = false;
            }
            points.push_back(pt);
        }

        // Frontier, then the scalarised oracle per lambda.
        for (const auto& lambda : cfg.lambdas) {
            const IndexPoint* best = nullptr;
            double bestJ = std::numeric_limits<double>::infinity();
            for (const auto& p : points) {
                if (!p.viable) continue;
                const double J = static_cast<double>(p.payloadBytes) + lambda * latencyTerm(p);
                if (J < bestJ) { bestJ = J; best = &p; }
            }
            const std::string oracle = best ? best->name : "-";
            const std::string predicted = predictorPick(points, lambda);
            const IndexPoint* pred = nullptr;
            for (const auto& p : points) if (p.name == predicted) pred = &p;

            for (const auto& p : points) {
                const double J = p.viable
                    ? static_cast<double>(p.payloadBytes) + lambda * latencyTerm(p)
                    : std::numeric_limits<double>::quiet_NaN();
                auto row = writer.row();
                row.set("driver", "bench_index_oracle")
                   .set("dataset", ds.name)
                   .set("index_type", p.name)
                   .set("N", static_cast<int64_t>(n))
                   .set("seed", static_cast<int64_t>(cfg.seed))
                   .set("cache_state", cfg.cacheState)
                   .set("payload_bytes", static_cast<int64_t>(p.payloadBytes))
                   .setIf(p.indexBytes.has_value(), "index_bytes", p.indexBytes.value_or(0))
                   .setIf(p.indexBytes.has_value() && p.payloadBytes > 0, "index_pct_of_payload",
                          100.0 * static_cast<double>(p.indexBytes.value_or(0))
                                / static_cast<double>(p.payloadBytes ? p.payloadBytes : 1))
                   .setIf(p.viable, "bulk_ns", p.bulkNs)
                   .setIf(p.viable, "point_ns_uniform", p.pointNsUniform)
                   .setIf(p.viable, "point_ns_zipf", p.pointNsZipf)
                   .setIf(p.viable, "gather_ns", p.gatherNs)
                   .set("random_access", p.randomAccess)
                   .set("viable", p.viable)
                   .set("on_pareto_frontier", p.viable && isPareto(p, points))
                   .set("lambda_ns_per_byte", lambda)
                   .setIf(p.viable, "objective_J", J)
                   .set("oracle_pick", oracle)
                   .set("predictor_pick", predicted);
                if (best && pred && pred->viable) {
                    row.set("regret_bytes",
                            static_cast<double>(pred->payloadBytes)
                          - static_cast<double>(best->payloadBytes))
                       .set("regret_ns", latencyTerm(*pred) - latencyTerm(*best));
                }
                writer.write(std::move(row));
            }
        }
        writer.flush();

        std::cout << "  Pareto frontier: ";
        for (const auto& p : points) if (p.viable && isPareto(p, points)) std::cout << p.name << " ";
        std::cout << "\n";
    }

    writer.close();
    manifest.finishedAtIso = detail::isoNow();
    manifest.exitCode = failures.empty() ? 0 : 2;
    manifest.writeSidecar(cfg.output);

    std::cout << "\nResults written to: " << std::filesystem::absolute(cfg.output) << "\n";
    if (!failures.empty()) {
        std::cerr << "\n" << failures.size() << " cell(s) failed validation and were "
                  << "marked non-viable:\n";
        for (const auto& f : failures) std::cerr << "  - " << f << "\n";
        return 2;
    }
    return 0;
}
