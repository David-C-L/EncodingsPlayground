// Cost-model estimate vs oracle actual.
//
// AutoSIS picks a bit-range plan from analytical cost models; the oracle picks
// one from measured byte counts over the same candidate set on the same sample.
// This driver measures the gap between the two, on three axes the interactive
// explorer could not sweep: dataset, sampling profile, and sample size.
//
// Three tables are written, as separate files next to --output:
//
//   .accuracy  per (cell, encoding) — est vs act bits/elem, both rankings, and
//              whether the encoding has an analytical cost model at all.
//   .plans     per plan segment, for autosis / oracle_random / oracle_consec /
//              oracle_merged.
//   .summary   per (dataset, profile, sample size, encoding set) — top-1
//              accuracy, Spearman rho, mean |rel err|, regret in bytes.
//
// has_cost_model is a COLUMN, never a filter.  OpenZL and the Cascading*
// compositions are in the oracle's candidate set but not in
// CostModelSet::defaultEncodings(), so dropping their rows would flatter the
// model by hiding the candidates it cannot rank at all.
//
// The accuracy table covers the random and consecutive profiles only.  Each is
// self-consistent: estimates and actuals come from the same sample.  The merged
// grid deliberately mixes cells costed against three different samples (see
// mergeEncodingGridsByProfile), so "the estimate on this sample" is not defined
// for it — it appears in the plans table, where only its chosen segments matter.
//
// RUNTIME: plan selection, not encoding, is the expensive part.  FINDINGS.md
// records AutoSIS selection at 13.6-14.5 s for 100k samples against a ~10 ms
// encode, scaling with sample count rather than search strategy, and this driver
// additionally pays 2080 grid cells x |candidates| sample encodes per (sample
// size, encoding set).  Defaults are therefore modest; raise --sample-sizes
// deliberately.

#include "benchmark/Cli.hpp"
#include "benchmark/CostModelGrid.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/OracleGrid.hpp"
#include "benchmark/ResultWriter.hpp"
#include "benchmark/RunManifest.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
// For applyFilters only — the codec registry itself is not used here, because the
// unit under test is a bit-range candidate set, not a whole-column encoder.
#include "benchmark/registry/EncoderRegistry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace encodings;
using namespace encodings::benchmark;

namespace {

using Elem = int64_t;

// ─── Candidate sets ──────────────────────────────────────────────────────────

/// The seven types AutoSIS_C runs with by default: every one of them has an
/// analytical cost model, so this set answers "how good is the model on its own
/// candidates".
std::vector<EncodingType> defaultEncodingSet() {
    return encodings::encoders::selectors::costs::CostModelSet::defaultEncodings();
}

/// The explorer's wide set.  Adds candidates the model cannot rank (OpenZL) and
/// compositions it ranks with a derived model, so this set answers the different
/// question of how much the model is leaving on the table.
std::vector<EncodingType> extendedEncodingSet() {
    return {
        EncodingType::RawEncoding,
        EncodingType::BitPacking,
        EncodingType::RunLengthEncoding,
        EncodingType::AdaptiveFrameOfReference,
        EncodingType::DictionaryEncoding,
        EncodingType::AdaptiveDictionaryEncoding,
        EncodingType::FrequencyPartitionEncoding,
        EncodingType::MainlyConstantEncoding,
        EncodingType::FrameOfReference,
        EncodingType::AdaptiveFramedBitPrefix,
        EncodingType::HuffmanEncoding,
        EncodingType::LZ4,
        EncodingType::FSEEncoding,
        EncodingType::OpenZL,
        EncodingType::BlockFrequencyPartitionEncoding,
        EncodingType::BlockFrequencyPartitionFOREncoding,
        EncodingType::BlockFSEEncoding,
        EncodingType::RangePackFrequencyPartitionEncoding,
        EncodingType::RangePackBlockFrequencyPartitionEncoding,
        EncodingType::CascadingFrameOfReference,
        EncodingType::CascadingFORBlockFrequencyPartitionEncoding,
        EncodingType::RunLengthCascadingFOREncoding,
        EncodingType::CascadingFORFSEEncoding,
        EncodingType::CascadingFORBlockFSEEncoding,
        EncodingType::CascadingFORHuffmanEncoding,
        EncodingType::CascadingFORPrevFSEEncoding,
        EncodingType::CascadingFORPrevBlockFSEEncoding,
        EncodingType::CascadingFORPrevHuffmanEncoding,
        EncodingType::CascadingFORPrevFrequencyPartitionEncoding,
        EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding,
    };
    // BlockFORFPEEncoding is deliberately absent: it does not round-trip
    // TwitterSnowflake (FINDINGS.md), and a size it cannot decode back from is
    // not a candidate an oracle may pick.
}

struct EncodingSet {
    std::string name;
    std::vector<EncodingType> types;
};

// ─── Configuration ───────────────────────────────────────────────────────────

struct SweepConfig {
    size_t n{1'000'000};
    std::vector<size_t> sampleSizes{2'000};
    std::vector<std::string> setNames{"default"};
    std::vector<std::string> profileNames{"random", "consec"};
    size_t minSegmentWidth{1};
    bool   allowReorderers{false};
    bool   encodeFull{false};
    uint64_t seed{42};
    std::vector<std::string> datasetFilters;
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_costmodel_oracle.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Sweep axes:")
        .opt("n", cfg.n, "full-dataset elements (extrapolation base)")
        .list("sample-sizes", cfg.sampleSizes, "sample element targets")
        .repeated("encoding-set", cfg.setNames, "candidate set: default | extended")
        .repeated("profile", cfg.profileNames, "accuracy profile: random | consec")
        .opt("min-segment-width", cfg.minSegmentWidth, "narrowest oracle segment in bits");

    args.group("Selection:")
        .flag("allow-reorderers", cfg.allowReorderers, "let AutoSIS consider BWT")
        .flag("encode-full", cfg.encodeFull,
              "also encode each plan segment over the full dataset (full_bytes)")
        .opt("seed", cfg.seed, "recorded in the manifest; sampling itself is deterministic")
        .repeated("dataset", cfg.datasetFilters, "only datasets whose name contains SUBSTR");

    args.group("Output:")
        .flag("dry-run", cfg.dryRun, "print the sweep plan and exit")
        .opt("output", cfg.output, "base result path; .accuracy/.plans/.summary are derived")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

/// `<stem>.<tag><ext>` next to the requested output, so the three tables share a
/// directory and a manifest and cannot be mistaken for three unrelated runs.
std::filesystem::path withTag(const std::filesystem::path& base, const std::string& tag) {
    std::filesystem::path out = base;
    out.replace_filename(base.stem().string() + "." + tag + base.extension().string());
    return out;
}

// ─── Schemas ─────────────────────────────────────────────────────────────────

/// Shared identity columns.  Appended to each table's own list rather than
/// hand-repeated, so the three tables can be joined on them.
void appendIdentityColumns(std::vector<ColumnSpec>& cols) {
    cols.push_back(stringCol("driver"));
    cols.push_back(stringCol("dataset"));
    cols.push_back(intCol("N"));
    cols.push_back(intCol("seed"));
    cols.push_back(stringCol("encoding_set"));
    cols.push_back(intCol("sample_size_nominal"));
    cols.push_back(intCol("sample_size_actual"));
    cols.push_back(intCol("min_segment_width"));
}

std::vector<ColumnSpec> accuracyColumns() {
    std::vector<ColumnSpec> c;
    appendIdentityColumns(c);
    c.push_back(stringCol("profile"));
    c.push_back(intCol("profile_block_size"));
    c.push_back(intCol("l"));
    c.push_back(intCol("r"));
    c.push_back(intCol("width"));
    c.push_back(stringCol("encoding"));
    c.push_back(doubleCol("est_bits_per_elem"));
    c.push_back(doubleCol("act_bits_per_elem"));
    c.push_back(doubleCol("rel_err"));
    c.push_back(intCol("model_rank"));
    c.push_back(intCol("actual_rank"));
    c.push_back(intCol("is_oracle_pick"));
    c.push_back(intCol("is_model_pick"));
    c.push_back(intCol("has_cost_model"));
    return c;
}

std::vector<ColumnSpec> planColumns() {
    std::vector<ColumnSpec> c;
    appendIdentityColumns(c);
    c.push_back(stringCol("plan"));
    c.push_back(intCol("bit_start"));
    c.push_back(intCol("bit_end"));
    c.push_back(intCol("width"));
    c.push_back(stringCol("encoding"));
    c.push_back(stringCol("reorderer"));
    c.push_back(intCol("sample_bytes"));
    c.push_back(doubleCol("est_bits"));
    c.push_back(intCol("full_bytes"));
    c.push_back(intCol("has_cost_model"));
    return c;
}

std::vector<ColumnSpec> summaryColumns() {
    std::vector<ColumnSpec> c;
    appendIdentityColumns(c);
    c.push_back(stringCol("profile"));
    c.push_back(intCol("cells"));
    c.push_back(intCol("candidates"));
    c.push_back(intCol("candidates_with_cost_model"));
    c.push_back(doubleCol("top1_accuracy"));
    c.push_back(doubleCol("spearman_rho"));
    c.push_back(doubleCol("mean_abs_rel_err"));
    c.push_back(intCol("regret_bytes_sample"));
    c.push_back(doubleCol("regret_bytes_extrapolated"));
    c.push_back(intCol("metric_compute_calls"));
    c.push_back(intCol("grid_build_ms"));
    c.push_back(intCol("selection_ms"));
    return c;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Bit ranges any of the four plans uses.  The accuracy table is emitted over
/// this union rather than over one plan's segments, because the interesting cells
/// include the ones AutoSIS chose and the oracle rejected as well as the reverse.
std::set<std::pair<int, int>> planCells(const std::vector<const std::vector<SegmentPlan>*>& plans) {
    std::set<std::pair<int, int>> cells;
    for (const auto* p : plans)
        for (const auto& seg : *p) cells.emplace(seg.bitStart, seg.bitEnd);
    return cells;
}

size_t fullSegmentBytes(const std::vector<uint64_t>& uFull, const SegmentPlan& seg) {
    const int width = seg.bitEnd - seg.bitStart + 1;
    try {
        const std::vector<uint64_t> section = extractSection(uFull, seg.bitStart, width);
        auto codec = makeSectionCodec(seg.encoding, static_cast<uint8_t>(width));
        return codec->encode(std::span<const uint64_t>(section.data(), section.size()))
            .data().size();
    } catch (...) {
        return std::numeric_limits<size_t>::max();
    }
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_costmodel_oracle",
                   "Compare AutoSIS cost-model estimates against the measured oracle over "
                   "datasets x sampling profiles x sample sizes x candidate sets.");
    bindArgs(args, cfg);
    switch (args.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    if (cfg.minSegmentWidth == 0) cfg.minSegmentWidth = 1;
    if (cfg.sampleSizes.empty()) { std::cerr << "ERROR: --sample-sizes is empty\n"; return 1; }

    std::vector<EncodingSet> sets;
    for (const auto& name : cfg.setNames) {
        if (name == "default")       sets.push_back({name, defaultEncodingSet()});
        else if (name == "extended") sets.push_back({name, extendedEncodingSet()});
        else {
            std::cerr << "ERROR: unknown --encoding-set '" << name
                      << "' (expected default|extended)\n";
            return 1;
        }
    }
    for (const auto& p : cfg.profileNames) {
        if (p != "random" && p != "consec") {
            std::cerr << "ERROR: unknown --profile '" << p << "' (expected random|consec)\n";
            return 1;
        }
    }
    const bool wantRandom = std::find(cfg.profileNames.begin(), cfg.profileNames.end(),
                                      "random") != cfg.profileNames.end();
    const bool wantConsec = std::find(cfg.profileNames.begin(), cfg.profileNames.end(),
                                      "consec") != cfg.profileNames.end();

    auto datasets = applyFilters(int64Datasets(), cfg.datasetFilters);
    // A filter matching nothing is an error, not an empty result file.
    if (datasets.empty()) { std::cerr << "ERROR: no datasets match --dataset filters\n"; return 1; }

    std::cout << "\n── Sweep configuration ──────────────────────────────────────\n"
              << "  N                  " << cfg.n << " elements\n"
              << "  sample sizes       ";
    for (size_t i = 0; i < cfg.sampleSizes.size(); ++i)
        std::cout << (i ? ", " : "") << cfg.sampleSizes[i];
    std::cout << "\n  candidate sets     ";
    for (size_t i = 0; i < sets.size(); ++i)
        std::cout << (i ? ", " : "") << sets[i].name << " (" << sets[i].types.size() << " types)";
    std::cout << "\n  accuracy profiles  ";
    for (size_t i = 0; i < cfg.profileNames.size(); ++i)
        std::cout << (i ? ", " : "") << cfg.profileNames[i];
    std::cout << "\n  sample block sizes random=" << kRandomBlockSize
              << " consec=" << kConsecBlockSize << " wide=" << kWideBlockSize << "\n"
              << "  min segment width  " << cfg.minSegmentWidth << " bit(s)\n"
              << "  reorderers         " << (cfg.allowReorderers ? "BWT allowed" : "disabled") << "\n"
              << "  full-dataset encode " << (cfg.encodeFull ? "yes" : "no (full_bytes null)") << "\n"
              << "  datasets           " << datasets.size() << "\n";

    {
        // Grid cells, not rows: this is the number that decides how long a run
        // takes, and it is quadratic in nothing the user typed.
        size_t gridCells = 0;
        for (int l = 0; l < kGridBits; ++l)
            for (int r = l + static_cast<int>(cfg.minSegmentWidth) - 1; r < kGridBits; ++r)
                ++gridCells;
        size_t sampleEncodes = 0;
        for (const auto& s : sets) sampleEncodes += gridCells * s.types.size();
        sampleEncodes *= cfg.sampleSizes.size() * datasets.size();
        std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
                  << "  grid cells         " << gridCells << " per grid\n"
                  << "  grids              "
                  << (3 * sets.size() * cfg.sampleSizes.size() * datasets.size())
                  << " (random, consecutive, wide per point)\n"
                  << "  sample encodes     ~" << sampleEncodes
                  << " (x3 profiles, wide subset excepted)\n"
                  << "  autosis selections "
                  << (sets.size() * cfg.sampleSizes.size() * datasets.size()) << "\n\n";
    }

    if (cfg.dryRun) {
        std::cout << "Dry run: nothing computed.\n";
        return 0;
    }

    RunManifest manifest = RunManifest::capture("bench_costmodel_oracle", args.argvEcho());
    manifest.seed = cfg.seed;
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    for (const auto& s : sets) manifest.encoders.push_back("set:" + s.name);
    manifest.extra["accuracy_table"] = withTag(cfg.output, "accuracy").string();
    manifest.extra["plans_table"]    = withTag(cfg.output, "plans").string();
    manifest.extra["summary_table"]  = withTag(cfg.output, "summary").string();
    // Before the sweep, so a killed run still has provenance.
    manifest.writeSidecar(cfg.output);

    ResultWriter accuracyOut(withTag(cfg.output, "accuracy"), accuracyColumns(), cfg.format);
    ResultWriter plansOut(withTag(cfg.output, "plans"), planColumns(), cfg.format);
    ResultWriter summaryOut(withTag(cfg.output, "summary"), summaryColumns(), cfg.format);

    DatasetCache<Elem> dataCache;
    int exitCode = 0;

    for (auto& ds : datasets) {
        std::cout << "══ Dataset: " << ds.name << " ══\n  loading " << cfg.n
                  << " elements..." << std::flush;
        typename DatasetCache<Elem>::Handle handle;
        try {
            handle = dataCache.materialize(ds, cfg.n);
        } catch (const std::exception& e) {
            std::cout << "\n";
            std::cerr << "  WARNING: dataset '" << ds.name << "' skipped: " << e.what() << "\n";
            continue;
        }
        std::cout << " ok.\n";

        // Only materialised when --encode-full asks for it: at N=10M this is
        // another 80 MB next to the dataset itself.
        std::vector<uint64_t> uFull;
        if (cfg.encodeFull) uFull = toU64(handle.data);

        for (size_t sampleSize : cfg.sampleSizes) {
            const ProfileSamples samples = drawProfileSamples(handle.data, sampleSize);
            if (samples.random.empty()) {
                std::cerr << "  WARNING: sample size " << sampleSize
                          << " yielded no elements; skipped.\n";
                continue;
            }
            const std::vector<uint64_t> uRandom = toU64(samples.random);
            const std::vector<uint64_t> uConsec = toU64(samples.consecutive);

            for (const auto& set : sets) {
                std::cout << "  [" << set.name << " | sample " << samples.random.size()
                          << "] grids..." << std::flush;
                const auto t0 = std::chrono::steady_clock::now();
                const ProfileGrids grids = computeProfileGrids(
                    samples, set.types, static_cast<int>(cfg.minSegmentWidth));
                const auto t1 = std::chrono::steady_clock::now();
                std::cout << " done. selection..." << std::flush;

                AutoPlanOptions opts;
                opts.encodingTypes   = set.types;
                opts.allowReorderers = cfg.allowReorderers;
                opts.fullCount       = handle.n;
                const PlanWithEstimates autoSis = autoPlan(samples.random, opts);
                const auto t2 = std::chrono::steady_clock::now();
                std::cout << " done.\n";

                const auto gridMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                const auto selMs  = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

                const int msw = static_cast<int>(cfg.minSegmentWidth);
                const std::vector<SegmentPlan> oracleRandom = runOracleDP(grids.random, msw);
                const std::vector<SegmentPlan> oracleConsec = runOracleDP(grids.consecutive, msw);
                const std::vector<SegmentPlan> oracleMerged = runOracleDP(grids.merged, msw);

                const auto cells = planCells(
                    {&autoSis.segments, &oracleRandom, &oracleConsec, &oracleMerged});

                CostModelGrid cmRandom(uRandom, set.types, grids.random);
                CostModelGrid cmConsec(uConsec, set.types, grids.consecutive);

                const auto identity = [&](ResultWriter::Row& row) -> ResultWriter::Row& {
                    return row.set("driver", "bench_costmodel_oracle")
                        .set("dataset", ds.name)
                        .set("N", handle.n)
                        .set("seed", cfg.seed)
                        .set("encoding_set", set.name)
                        .set("sample_size_nominal", sampleSize)
                        .set("sample_size_actual", samples.random.size())
                        .set("min_segment_width", cfg.minSegmentWidth);
                };

                // ── accuracy ──────────────────────────────────────────────
                struct ProfileView {
                    const char*   name;
                    size_t        blockSize;
                    CostModelGrid* cm;
                    const EncodingGrid* grid;
                    const std::vector<SegmentPlan>* oraclePlan;
                    size_t        sampleElems;
                    bool          wanted;
                };
                const std::vector<ProfileView> views{
                    {"random", kRandomBlockSize, &cmRandom, &grids.random, &oracleRandom,
                     samples.random.size(), wantRandom},
                    {"consec", kConsecBlockSize, &cmConsec, &grids.consecutive, &oracleConsec,
                     samples.consecutive.size(), wantConsec},
                };

                for (const auto& view : views) {
                    if (!view.wanted) continue;
                    size_t top1Hits = 0, top1Cells = 0;
                    double rhoSum = 0.0; size_t rhoCells = 0;
                    double relSum = 0.0; size_t relCount = 0;

                    for (const auto& [l, r] : cells) {
                        const auto& entries = view.cm->at(l, r);
                        const EncodingType modelPick  = view.cm->modelPick(l, r);
                        const EncodingType oraclePick = view.cm->oraclePickAmongModelled(l, r);
                        const int width = r - l + 1;

                        ++top1Cells;
                        if (modelPick == oraclePick) ++top1Hits;
                        const double rho = spearmanRho(entries);
                        if (std::isfinite(rho)) { rhoSum += rho; ++rhoCells; }

                        for (const auto& e : entries) {
                            const double actBpe = bitsPerElem(e.actualBytes, view.sampleElems);
                            const double estBpe = e.hasCostModel
                                ? e.estBits / static_cast<double>(view.sampleElems)
                                : std::numeric_limits<double>::quiet_NaN();
                            const double relErr = (std::isfinite(actBpe) && actBpe > 0.0 &&
                                                   std::isfinite(estBpe))
                                ? (estBpe - actBpe) / actBpe
                                : std::numeric_limits<double>::quiet_NaN();
                            if (std::isfinite(relErr)) { relSum += std::abs(relErr); ++relCount; }

                            auto row = accuracyOut.row();
                            identity(row)
                                .set("profile", view.name)
                                .set("profile_block_size", view.blockSize)
                                .set("l", l)
                                .set("r", r)
                                .set("width", width)
                                .set("encoding", encodingTypeToString(e.enc))
                                // Typed nulls, not sentinels: a candidate with no
                                // cost model has no estimate, and a candidate that
                                // threw on this range has no actual.
                                .setIf(std::isfinite(estBpe), "est_bits_per_elem", estBpe)
                                .setIf(std::isfinite(actBpe), "act_bits_per_elem", actBpe)
                                .setIf(std::isfinite(relErr), "rel_err", relErr)
                                .setIf(e.hasCostModel, "model_rank", e.modelRank)
                                .setIf(e.actualRank > 0, "actual_rank", e.actualRank)
                                .set("is_oracle_pick", e.enc == oraclePick)
                                .set("is_model_pick", e.enc == modelPick)
                                .set("has_cost_model", e.hasCostModel);
                            accuracyOut.write(std::move(row));
                        }
                    }

                    // Regret: what the model's pick costs above the best
                    // cost-modelled candidate, over the oracle plan of THIS
                    // profile — the bytes a perfect model would have saved
                    // without changing the split boundaries.
                    long long regret = 0;
                    for (const auto& seg : *view.oraclePlan) {
                        const EncodingType mp = view.cm->modelPick(seg.bitStart, seg.bitEnd);
                        const EncodingType op =
                            view.cm->oraclePickAmongModelled(seg.bitStart, seg.bitEnd);
                        const size_t mb = (*view.grid)[seg.bitStart][seg.bitEnd].bytesFor(mp);
                        const size_t ob = (*view.grid)[seg.bitStart][seg.bitEnd].bytesFor(op);
                        if (mb == std::numeric_limits<size_t>::max() ||
                            ob == std::numeric_limits<size_t>::max()) continue;
                        regret += static_cast<long long>(mb) - static_cast<long long>(ob);
                    }
                    const double scale = view.sampleElems > 0
                        ? static_cast<double>(handle.n) / static_cast<double>(view.sampleElems)
                        : 0.0;

                    auto srow = summaryOut.row();
                    identity(srow)
                        .set("profile", view.name)
                        .set("cells", top1Cells)
                        .set("candidates", set.types.size())
                        .set("candidates_with_cost_model", view.cm->modelledCount())
                        .setIf(top1Cells > 0, "top1_accuracy",
                               top1Cells ? static_cast<double>(top1Hits)
                                               / static_cast<double>(top1Cells) : 0.0)
                        .setIf(rhoCells > 0, "spearman_rho",
                               rhoCells ? rhoSum / static_cast<double>(rhoCells) : 0.0)
                        .setIf(relCount > 0, "mean_abs_rel_err",
                               relCount ? relSum / static_cast<double>(relCount) : 0.0)
                        .set("regret_bytes_sample", regret)
                        .set("regret_bytes_extrapolated", static_cast<double>(regret) * scale)
                        .set("metric_compute_calls", view.cm->metricComputeCalls())
                        .set("grid_build_ms", gridMs)
                        .set("selection_ms", selMs);
                    summaryOut.write(std::move(srow));
                }

                // ── plans ─────────────────────────────────────────────────
                struct PlanView {
                    const char* name;
                    const std::vector<SegmentPlan>* plan;
                    const EncodingGrid* grid;
                    CostModelGrid* cm;   ///< null when the plan's grid mixes samples
                };
                const std::vector<PlanView> planViews{
                    {"autosis",       &autoSis.segments, &grids.random,      &cmRandom},
                    {"oracle_random", &oracleRandom,     &grids.random,      &cmRandom},
                    {"oracle_consec", &oracleConsec,     &grids.consecutive, &cmConsec},
                    // oracle_merged has no cost-model column: its cells were
                    // costed against three different samples, so no single
                    // sample's estimate belongs next to them.
                    {"oracle_merged", &oracleMerged,     &grids.merged,      nullptr},
                };

                for (const auto& pv : planViews) {
                    for (size_t i = 0; i < pv.plan->size(); ++i) {
                        const SegmentPlan& seg = (*pv.plan)[i];
                        const int width = seg.bitEnd - seg.bitStart + 1;
                        const size_t sampleBytes =
                            (*pv.grid)[seg.bitStart][seg.bitEnd].bytesFor(seg.encoding);

                        // AutoSIS's own estimate is the DP cost that chose the
                        // segment; for an oracle plan the estimate is whatever the
                        // model would have said about the encoding the oracle
                        // picked, which is only defined when it has a model.
                        double estBits = std::numeric_limits<double>::quiet_NaN();
                        bool hasModel = false;
                        if (std::string(pv.name) == "autosis") {
                            estBits  = autoSis.estBitsPerSegment[i];
                            hasModel = true;
                        } else if (pv.cm != nullptr) {
                            for (const auto& e : pv.cm->at(seg.bitStart, seg.bitEnd)) {
                                if (e.enc == seg.encoding && e.hasCostModel) {
                                    estBits  = e.estBits;
                                    hasModel = true;
                                    break;
                                }
                            }
                        }

                        const size_t fullBytes = cfg.encodeFull
                            ? fullSegmentBytes(uFull, seg) : std::numeric_limits<size_t>::max();

                        auto row = plansOut.row();
                        identity(row)
                            .set("plan", pv.name)
                            .set("bit_start", seg.bitStart)
                            .set("bit_end", seg.bitEnd)
                            .set("width", width)
                            .set("encoding", encodingTypeToString(seg.encoding))
                            .set("reorderer",
                                 encodings::encoders::selectors::subStreamReordererTypeToString(
                                     seg.reorderer))
                            .setIf(sampleBytes != std::numeric_limits<size_t>::max(),
                                   "sample_bytes", sampleBytes)
                            .setIf(std::isfinite(estBits), "est_bits", estBits)
                            .setIf(fullBytes != std::numeric_limits<size_t>::max(),
                                   "full_bytes", fullBytes)
                            .set("has_cost_model", hasModel);
                        plansOut.write(std::move(row));
                    }
                }

                accuracyOut.flush();
                plansOut.flush();
                summaryOut.flush();

                std::cout << "    grid " << gridMs << " ms, selection " << selMs << " ms, "
                          << cells.size() << " cells, "
                          << cmRandom.metricComputeCalls() << "+"
                          << cmConsec.metricComputeCalls()
                          << " MetricCollector::compute calls\n";
            }
        }
        // One dataset at a time: an int64 stream at N=10M is 80 MB and the cache
        // never evicts on its own.
        dataCache.releaseAll();
    }

    accuracyOut.close();
    plansOut.close();
    summaryOut.close();
    std::cout << "\nResults written to:\n"
              << "  " << std::filesystem::absolute(withTag(cfg.output, "accuracy")) << "\n"
              << "  " << std::filesystem::absolute(withTag(cfg.output, "plans")) << "\n"
              << "  " << std::filesystem::absolute(withTag(cfg.output, "summary")) << std::endl;

    manifest.finishedAtIso = detail::isoNow();
    manifest.exitCode      = exitCode;
    manifest.writeSidecar(cfg.output);
    return exitCode;
}
