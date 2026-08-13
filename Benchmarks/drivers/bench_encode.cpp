// Encode cost, in isolation.
//
// This is the one driver that deliberately re-encodes (CONVENTIONS section 2):
// every other driver takes EncodeMeasurement::None so that an AutoSIS codec runs
// its cost-model DP exactly once and the plan being measured is the plan being
// reported.  Here the encode *is* the subject, so ArtifactCache is asked for
// EncodeMeasurement::TimedAndPeakHeap: warmup + iterations timed encodes, plus
// one extra instrumented encode for the heap figures, which is kept out of the
// timed loop because ScopedAllocationTrack puts a CAS on every operator new.
//
// What `encode_ns` therefore means for an AutoSIS entry needs saying out loud.
// SubIntSplitAutoEncoder::reset() drops the resolved inner encoder, and
// ensureEncoder() re-runs sampling and DP selection only when there is none.
// ArtifactCache resets once and then encodes warmup+iterations times, so
// selection is paid by the FIRST encode after the reset and by no other.  With
// warmup >= 1 that is a warmup encode, so:
//
//   selection_ns   one-off cost of choosing the plan, reported per cell
//   encode_ns      cost of applying an already-chosen plan
//
// which is the split a reader wants, and is why --warmup is clamped to at least 1
// rather than allowed to reach 0 and fold a one-off multi-second DP into the
// first timing sample.
//
// The fourth axis is --sample-sizes, absorbed from Benchmarks/sweep_subint_samples.cpp:
// how many sampled points the AutoSIS selector gets to choose a plan from.  It is
// applied by rebuilding the codec with samplerConfig.maxSamples set, not merely
// recorded — see autoSisWithSampleSize().
//
// Two tables come out of one sweep:
//
//   <output>              one row per (encoder, dataset, N, sample_size)
//   <output>_sections.*   per-substream encoded bytes and encode times, for the
//                         `*_Prof` registry entries that populate them

#include "benchmark/ArtifactCache.hpp"
#include "benchmark/CachePolicy.hpp"
#include "benchmark/Cli.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/ResultWriter.hpp"
#include "benchmark/RunManifest.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "benchmark/targets/PlaygroundTarget.hpp"
#include "encoders/SubIntSplitEncoder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace encodings;
using namespace encodings::benchmark;

namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);

// ─── Configuration ───────────────────────────────────────────────────────────

/// Every field is bound to a flag by ArgParser, so --help prints these values as
/// its defaults and cannot drift from them.
struct SweepConfig {
    // Smaller by default than the decode drivers' N: a cell here costs
    // (warmup + iterations + 1) encodes, and for AutoSIS one of those also runs
    // the DP.  A default that takes an hour is a default nobody runs.
    std::vector<size_t> sizes{100'000};
    // The registry's AutoSIS default, so a default run of this driver measures
    // the same sampler configuration every other driver encodes with.
    std::vector<size_t> sampleSizes{100'000};
    bool exhaustive{false};
    size_t iterations{5};
    size_t warmup{1};
    uint64_t seed{42};
    CacheState cacheState{CacheState::Hot};
    EvictMethod evictMethod{EvictMethod::Auto};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    bool validate{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_encode.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Sweep axes:")
        .list("sizes", cfg.sizes, "stream lengths in elements")
        .list("sample-sizes", cfg.sampleSizes,
              "AutoSIS samplerConfig.maxSamples values; ignored by other families")
        .flag("exhaustive", cfg.exhaustive,
              "AutoSIS: enumerate all split combinations instead of running the DP");

    args.group("Measurement:")
        .opt("iterations", cfg.iterations, "timed encodes per cell")
        .opt("warmup", cfg.warmup, "untimed encodes per cell (clamped to >= 1; see file header)")
        .opt("seed", cfg.seed, "recorded for provenance");

    // Accepted and recorded, never acted on: an encode reads its input and writes
    // a fresh payload, so there is no payload to cool.  Rejecting these flags
    // would make this driver the one exception in a sweep script that passes the
    // same arguments to all of them.
    args.group("Cache state (recorded only; an encode has no payload to cool):")
        .enumOpt("cache-state", cfg.cacheState,
                 {{"hot", CacheState::Hot},
                  {"cold-payload", CacheState::ColdPayload},
                  {"cold-all", CacheState::ColdAll}},
                 "recorded in the output")
        .enumOpt("evict-method", cfg.evictMethod,
                 {{"auto", EvictMethod::Auto},
                  {"clflush", EvictMethod::Clflush},
                  {"llc-thrash", EvictMethod::LlcThrash},
                  {"none", EvictMethod::None}},
                 "recorded in the output");

    args.group("Selection and output:")
        .repeated("dataset", cfg.datasetFilters, "only datasets whose name contains SUBSTR")
        .repeated("encoder", cfg.encoderFilters, "only encoders whose name contains SUBSTR")
        .flag("validate", cfg.validate, "round-trip each cell before recording its timings")
        .flag("dry-run", cfg.dryRun, "print the preflight summary and exit")
        .opt("output", cfg.output, "result file")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

bool resolveConfig(SweepConfig& cfg, std::string& whyNot) {
    for (auto* axis : {&cfg.sizes, &cfg.sampleSizes}) {
        std::sort(axis->begin(), axis->end());
        axis->erase(std::unique(axis->begin(), axis->end()), axis->end());
        std::erase(*axis, size_t{0});
    }
    if (cfg.sizes.empty()) { whyNot = "--sizes resolved to an empty list"; return false; }
    if (cfg.sampleSizes.empty()) {
        whyNot = "--sample-sizes resolved to an empty list";
        return false;
    }
    cfg.iterations = std::max<size_t>(1, cfg.iterations);
    // Never 0: with no warmup, an AutoSIS cell's first timing sample would carry
    // the one-off DP selection cost and its p90 would be that instead of an
    // encode.  See the file header.
    cfg.warmup = std::max<size_t>(1, cfg.warmup);
    return true;
}

std::filesystem::path sectionsPath(const std::filesystem::path& output) {
    std::filesystem::path p = output;
    p.replace_filename(output.stem().string() + "_sections" + output.extension().string());
    return p;
}

// ─── The sample-size axis ────────────────────────────────────────────────────

/// One (encoder, sample size) cell of the sweep.
///
/// `baseName` is what reaches the `encoding` column, so the sample-size axis
/// appears as a column and not as a family of differently-named encoders.
/// `entry.name` carries the sample size because it is part of ArtifactCache's
/// key: two cells that differ only in sampler configuration would otherwise
/// collide there and the second would silently report the first's timings.
struct EncoderCell {
    EncoderEntry<Elem> entry;
    std::string baseName;
    std::optional<size_t> sampleSize;
};

/// Rebuilds an AutoSIS entry with a given samplerConfig.maxSamples.
///
/// The config is reassembled here rather than mutated on the registry's codec
/// because the sampler configuration is captured by SubIntSplitAutoEncoder's
/// constructor and there is no setter — recording a --sample-sizes value without
/// rebuilding would produce a column that varies while the measured encoder does
/// not, which is exactly the failure sweep_subint_samples.cpp avoided by
/// constructing a fresh encoder per sample size.
///
/// Everything except maxSamples mirrors registry/EncoderRegistry.hpp's
/// sisAutoEncoders(): selection timing on (selection_ns is unrecoverable
/// afterwards), prune on, exhaustive search off unless --exhaustive, verbose
/// silenced and no cost-grid CSV — that path throws if it cannot open its file.
EncoderEntry<Elem> autoSisWithSampleSize(const EncoderEntry<Elem>& base, size_t maxSamples,
                                         bool exhaustive) {
    using encodings::encoders::BitSplitOrder;

    // The registry writes the order into `variant` and marks the profiling
    // instantiation with a `_Prof` suffix, so both are recoverable from the entry
    // without this driver carrying a second copy of the AutoSIS list.
    const BitSplitOrder order = base.variant == "MSB_TO_LSB" ? BitSplitOrder::MSB_TO_LSB
                                                             : BitSplitOrder::LSB_TO_MSB;
    const bool profiling = base.name.find("_Prof") != std::string::npos;

    auto build = [&] {
        auto c = encodings::encoders::makeDefaultAutoSubIntSplitConfig<Elem>(
            order, /*enableSelectionTiming=*/true);
        c.selectorConfig.orderHint = order;
        c.selectorConfig.useExhaustiveSearch = exhaustive;
        c.selectorConfig.enablePrune = true;
        c.selectorConfig.verboseLevel = 0;
        c.selectorConfig.costGridCsvPath.reset();
        c.samplerConfig.maxSamples = maxSamples;
        // maxPercentage overrides maxSamples when set, and stride short-circuits
        // the derivation of one from the other, so both are pinned for the swept
        // value to be the binding cap.
        c.samplerConfig.maxPercentage = 0.0;
        c.samplerConfig.stride = 0;
        return c;
    };

    EncoderEntry<Elem> e = base;
    e.codec = profiling
        ? std::static_pointer_cast<encodings::Codec<Elem>>(
              encodings::encoders::makeAutoSubIntSplitEncoderProf<Elem>(build()))
        : std::static_pointer_cast<encodings::Codec<Elem>>(
              encodings::encoders::makeAutoSubIntSplitEncoder<Elem>(build()));
    e.name = base.name + "@samples=" + std::to_string(maxSamples);
    return e;
}

/// Expands the encoder axis against the sample-size axis.
///
/// Only the `sis-auto` family has a sampler, so every other family is measured
/// once with a null `sample_size` rather than once per sample size — repeating an
/// identical measurement would multiply the sweep's cost and invite a reader to
/// plot a flat line as if it meant something.
std::vector<EncoderCell> buildCells(const std::vector<EncoderEntry<Elem>>& encoders,
                                    const SweepConfig& cfg) {
    std::vector<EncoderCell> cells;
    for (const auto& base : encoders) {
        if (base.family != "sis-auto") {
            cells.push_back(EncoderCell{base, base.name, std::nullopt});
            continue;
        }
        for (size_t s : cfg.sampleSizes) {
            cells.push_back(EncoderCell{autoSisWithSampleSize(base, s, cfg.exhaustive),
                                        base.name, s});
        }
    }
    return cells;
}

// ─── Result schema ───────────────────────────────────────────────────────────

/// The main table, in file order.
///
/// `time_ns`/`time_p90_ns`/`time_min_ns` are the CONVENTIONS section 6 names and
/// carry the same values as `encode_*`: a reader concatenating driver outputs gets
/// the driver's one measured quantity under the common name, and a reader who
/// wants to be explicit gets `encode_ns`.
std::vector<ColumnSpec> mainColumns() {
    return {
        stringCol("driver"),
        stringCol("dataset"),
        stringCol("encoding"),
        stringCol("family"),
        stringCol("variant"),
        intCol("is_sequential"),
        intCol("fast_skip"),
        intCol("random_access"),
        intCol("N"),
        intCol("N_requested"),
        intCol("seed"),
        stringCol("cache_state"),
        stringCol("evict_method"),
        intCol("evict_ns"),
        intCol("payload_bytes"),
        intCol("original_bytes"),
        doubleCol("bits_per_element"),
        doubleCol("compression_ratio"),
        intCol("iterations"),
        intCol("warmup"),
        intCol("sample_size"),
        intCol("exhaustive_search"),
        intCol("encode_ns"),
        intCol("encode_p90_ns"),
        intCol("encode_min_ns"),
        doubleCol("encode_Meps"),
        doubleCol("encode_MBps"),
        intCol("selection_ns"),
        doubleCol("selection_over_encode"),
        intCol("encode_peak_heap_bytes"),
        intCol("encode_net_heap_bytes"),
        intCol("substream_count"),
        intCol("substream_encode_ns_sum"),
        intCol("time_ns"),
        intCol("time_p90_ns"),
        intCol("time_min_ns"),
        intCol("truncated"),
        intCol("skipped"),
    };
}

/// The per-substream companion table.
///
/// `bit_lo`/`bit_hi` are offsets *in section order*, accumulated from the section
/// bit widths — not value bit positions, which depend on the plan's BitSplitOrder
/// and are not carried on subStreamEncodeMetrics.
///
/// `section_encode_ns` comes from the same encode that produced
/// `section_encoded_bytes`, i.e. the last timed encode of the cell, not from a
/// separate run: re-encoding to recover it would report a different encode's
/// numbers (CONVENTIONS section 2).
std::vector<ColumnSpec> sectionColumns() {
    return {
        stringCol("driver"),
        stringCol("dataset"),
        stringCol("encoding"),
        stringCol("family"),
        stringCol("variant"),
        intCol("N"),
        intCol("sample_size"),
        intCol("section_index"),
        intCol("section_count"),
        intCol("bit_width"),
        intCol("bit_lo"),
        intCol("bit_hi"),
        stringCol("section_encoding"),
        intCol("section_encoded_bytes"),
        intCol("section_encode_ns"),
        intCol("payload_bytes"),
    };
}

// ─── Preflight ───────────────────────────────────────────────────────────────

void printAxis(std::string_view label, const std::vector<size_t>& values) {
    std::cout << "  " << std::setw(15) << std::left << label << std::right;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << values[i];
    }
    std::cout << "  (" << values.size() << " step" << (values.size() == 1 ? "" : "s") << ")\n";
}

void printPreflight(const SweepConfig& cfg,
                    const std::vector<EncoderCell>& cells,
                    const std::vector<DatasetEntry<Elem>>& datasets) {
    std::cout << "\n── Sweep configuration ──────────────────────────────────────\n";
    printAxis("sizes", cfg.sizes);
    printAxis("sample-sizes", cfg.sampleSizes);

    size_t autoCells = 0, sectionCells = 0;
    for (const auto& c : cells) {
        if (c.sampleSize) ++autoCells;
        if (c.baseName.find("_Prof") != std::string::npos) ++sectionCells;
    }

    std::cout << "  " << std::setw(15) << std::left << "encoder cells" << std::right
              << cells.size() << "  (" << autoCells << " AutoSIS x sample size, "
              << (cells.size() - autoCells) << " single-config)\n"
              << "  " << std::setw(15) << std::left << "datasets" << std::right
              << datasets.size() << "\n"
              << "  " << std::setw(15) << std::left << "exhaustive" << std::right
              << (cfg.exhaustive ? "on" : "off")
              << "  (AutoSIS selector: enumerate all splits vs DP)\n"
              << "  " << std::setw(15) << std::left << "iterations" << std::right
              << cfg.iterations << " timed, " << cfg.warmup << " warmup\n"
              << "  " << std::setw(15) << std::left << "cache state" << std::right
              << cacheStateName(cfg.cacheState) << "/" << evictMethodName(cfg.evictMethod)
              << "  (recorded only)\n";

    const size_t rows = cfg.sizes.size() * cells.size() * datasets.size();
    // The +1 is the instrumented heap encode, which is a real encode() call and a
    // real part of the run's wall-clock cost even though it is not a timing sample.
    const size_t encodes = rows * (cfg.iterations + cfg.warmup + 1);
    std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
              << "  main rows      " << rows << "\n"
              << "  encode calls   " << encodes << "\n"
              << "  DP selections  " << (cfg.sizes.size() * autoCells * datasets.size())
              << "  (one per AutoSIS cell: reset() drops the plan, the first encode "
                 "after it re-selects)\n"
              << "  section rows   from " << sectionCells << " profiling cell(s)\n\n";
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Full materialization compared against the source stream.
///
/// Deliberately not DecodeHarness::validate(): that additionally asserts the
/// gather-equals-range identity, which needs a CacheController and belongs to the
/// drivers that measure selective reads.  What this driver has to establish is
/// only that the payload it timed the production of actually decodes.
bool roundTrips(PlaygroundTarget<Elem>& target, std::span<const Elem> reference,
                std::vector<Elem>& scratch, std::string& whyNot) {
    const size_t n = reference.size();
    scratch.assign(n, Elem{});
    target.materializeAll(scratch.data(), n);
    for (size_t i = 0; i < n; ++i) {
        if (scratch[i] != reference[i]) {
            whyNot = "materializeAll mismatch at row " + std::to_string(i) + ": got " +
                     std::to_string(scratch[i]) + ", expected " + std::to_string(reference[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_encode",
                   "Encode time, AutoSIS selection time and encode peak heap over "
                   "(encoder, dataset, N, sample_size).  The only driver that "
                   "re-encodes.");
    bindArgs(args, cfg);
    switch (args.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }
    std::string whyNot;
    if (!resolveConfig(cfg, whyNot)) {
        std::cerr << "ERROR: " << whyNot << "\n";
        return 1;
    }

    auto encoders = applyFilters(allEncoders(), cfg.encoderFilters);
    auto datasets = applyFilters(int64Datasets(), cfg.datasetFilters);
    // A filter that matches nothing is an error, not an empty result file.
    if (encoders.empty()) { std::cerr << "ERROR: no encoders match --encoder filters\n"; return 1; }
    if (datasets.empty()) { std::cerr << "ERROR: no datasets match --dataset filters\n"; return 1; }

    auto cells = buildCells(encoders, cfg);

    printPreflight(cfg, cells, datasets);
    if (cfg.dryRun) {
        std::cout << "Dry run: nothing encoded.\n";
        return 0;
    }

    RunManifest manifest = RunManifest::capture("bench_encode", args.argvEcho());
    manifest.seed = cfg.seed;
    const CacheTopology topo = CacheTopology::detect();
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    for (const auto& c : cells) manifest.encoders.push_back(c.entry.name);
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    manifest.extra["cache_state"] = cacheStateName(cfg.cacheState);
    manifest.extra["evict_method"] = evictMethodName(cfg.evictMethod);
    manifest.extra["exhaustive_search"] = cfg.exhaustive ? "1" : "0";
    manifest.extra["sections_output"] = sectionsPath(cfg.output).string();
    // Before the sweep, so a killed run still has provenance.
    manifest.writeSidecar(cfg.output);

    ResultWriter writer(cfg.output, mainColumns(), cfg.format);
    ResultWriter sections(sectionsPath(cfg.output), sectionColumns(), cfg.format);

    std::vector<std::string> validationFailures;
    std::vector<Elem> scratch;
    size_t mainRows = 0, sectionRows = 0;

    DatasetCache<Elem> dataCache;
    ArtifactCache<Elem> artifacts;

    for (size_t requestedN : cfg.sizes) {
        std::cout << "══ N = " << requestedN << " ══\n";
        for (auto& ds : datasets) {
            std::cout << "  [" << ds.name << "] loading..." << std::flush;
            typename DatasetCache<Elem>::Handle handle;
            try {
                handle = dataCache.materialize(ds, requestedN);
            } catch (const std::exception& e) {
                // A source that cannot yield N elements is a filtered-out cell with
                // a warning, not the end of the sweep (CONVENTIONS section 9).
                std::cout << "\n";
                std::cerr << "  WARNING: dataset '" << ds.name << "' skipped at N="
                          << requestedN << ": " << e.what() << "\n";
                continue;
            }
            const size_t n = handle.n;
            std::cout << " " << n << " elements.\n";

            for (auto& cell : cells) {
                auto& enc = cell.entry;
                const auto& artifact = artifacts.get(enc, handle,
                                                     EncodeMeasurement::TimedAndPeakHeap,
                                                     cfg.iterations, cfg.warmup);
                const auto& meta = artifact.encoded.metadata();
                // TimedAndPeakHeap guarantees both, but reading them out of the
                // optionals rather than assuming keeps the null path honest if the
                // cache ever hands back a cheaper artifact.
                const bool timed = artifact.encodeTimeNs.has_value();
                const TimingSummary t = artifact.encodeTimeNs.value_or(TimingSummary{});

                PlaygroundTarget<Elem> target(*enc.codec);
                // A copy of the payload, not a further encode: the artifact already
                // carries the bytes the timed encodes produced.
                target.adopt(artifact.encoded);

                if (cfg.validate) {
                    std::string why;
                    if (!roundTrips(target, handle.data, scratch, why)) {
                        // Excluded from the output entirely (CONVENTIONS section 5):
                        // an encode time for a payload that does not decode is
                        // worthless rather than merely imprecise.
                        std::cerr << "  [" << enc.name << "] EXCLUDED at N=" << n << ": " << why
                                  << "\n";
                        validationFailures.push_back(enc.name + " on " + ds.name + " at N=" +
                                                     std::to_string(n) + ": " + why);
                        artifacts.evict(enc.name);
                        continue;
                    }
                }

                const double encodeNs = static_cast<double>(t.medianNs);
                const double meps = encodeNs > 0.0
                    ? static_cast<double>(n) / encodeNs * 1e3 : 0.0;
                const double mbps = encodeNs > 0.0
                    ? static_cast<double>(n * kElemSize) / encodeNs * 1e3 : 0.0;
                const double bitsPerElement = n != 0
                    ? static_cast<double>(artifact.payloadBytes) * 8.0 / static_cast<double>(n)
                    : 0.0;

                const bool haveSelection = artifact.selectionTimeNs >= 0;
                const auto& secs = meta.subStreamEncodeMetrics;
                int64_t sectionNsSum = 0;
                for (const auto& m : secs) sectionNsSum += m.encodeTime_ns;

                auto row = writer.row();
                row.set("driver", "bench_encode")
                    .set("dataset", ds.name)
                    .set("encoding", cell.baseName)
                    .set("family", enc.family)
                    .set("variant", enc.variant)
                    .set("is_sequential", enc.isSequential)
                    .set("fast_skip", artifact.fastSkip)
                    .set("random_access", artifact.randomAccess)
                    .set("N", n)
                    .set("N_requested", requestedN)
                    .set("seed", cfg.seed)
                    .set("cache_state", cacheStateName(cfg.cacheState))
                    .set("evict_method", evictMethodName(cfg.evictMethod))
                    // No payload is cooled for an encode, so this is not 0 ns of
                    // eviction, it is no eviction at all.
                    .setNull("evict_ns")
                    .set("payload_bytes", artifact.payloadBytes)
                    .set("original_bytes", artifact.originalBytes)
                    .set("bits_per_element", bitsPerElement)
                    .set("compression_ratio", artifact.compressionRatio)
                    .set("iterations", cfg.iterations)
                    .set("warmup", cfg.warmup)
                    // Null for every family without a sampler: the axis does not
                    // apply, which is not the same as a sample size of 0.
                    .setIf(cell.sampleSize.has_value(), "sample_size",
                           cell.sampleSize.value_or(0))
                    .setIf(cell.sampleSize.has_value(), "exhaustive_search", cfg.exhaustive)
                    .setIf(timed, "encode_ns", t.medianNs)
                    .setIf(timed, "encode_p90_ns", t.p90Ns)
                    .setIf(timed, "encode_min_ns", t.minNs)
                    .setIf(timed, "encode_Meps", meps)
                    .setIf(timed, "encode_MBps", mbps)
                    // -1 means the codec reported no selectionTime_ns — every codec
                    // except an AutoSIS built with enableSelectionTiming.  That
                    // reaches the file as a null, never as a sentinel a reader could
                    // average.
                    .setIf(haveSelection, "selection_ns", artifact.selectionTimeNs)
                    .setIf(haveSelection && encodeNs > 0.0, "selection_over_encode",
                           static_cast<double>(artifact.selectionTimeNs) / encodeNs)
                    .setIf(artifact.encodePeakHeapBytes.has_value(), "encode_peak_heap_bytes",
                           artifact.encodePeakHeapBytes.value_or(0))
                    .setIf(artifact.encodeNetHeapDeltaBytes.has_value(), "encode_net_heap_bytes",
                           artifact.encodeNetHeapDeltaBytes.value_or(0))
                    .setIf(!secs.empty(), "substream_count", secs.size())
                    .setIf(!secs.empty(), "substream_encode_ns_sum", sectionNsSum)
                    .setIf(timed, "time_ns", t.medianNs)
                    .setIf(timed, "time_p90_ns", t.p90Ns)
                    .setIf(timed, "time_min_ns", t.minNs)
                    .set("truncated", n < requestedN)
                    .set("skipped", false);
                writer.write(std::move(row));
                ++mainRows;

                size_t bitLo = 0;
                for (size_t s = 0; s < secs.size(); ++s) {
                    const auto& m = secs[s];
                    auto srow = sections.row();
                    srow.set("driver", "bench_encode")
                        .set("dataset", ds.name)
                        .set("encoding", cell.baseName)
                        .set("family", enc.family)
                        .set("variant", enc.variant)
                        .set("N", n)
                        .setIf(cell.sampleSize.has_value(), "sample_size",
                               cell.sampleSize.value_or(0))
                        .set("section_index", s)
                        .set("section_count", secs.size())
                        .set("bit_width", static_cast<int64_t>(m.bitWidth))
                        .set("bit_lo", bitLo)
                        .set("bit_hi", bitLo + m.bitWidth)
                        .set("section_encoding", m.name)
                        .set("section_encoded_bytes", m.encodedBytes)
                        .set("section_encode_ns", m.encodeTime_ns)
                        .set("payload_bytes", artifact.payloadBytes);
                    sections.write(std::move(srow));
                    ++sectionRows;
                    bitLo += m.bitWidth;
                }

                std::cout << "    " << std::setw(34) << std::left << enc.name << std::right
                          << std::setw(12) << t.medianNs << " ns encode  "
                          << std::fixed << std::setprecision(1) << std::setw(8) << mbps
                          << " MB/s  peak "
                          << (artifact.encodePeakHeapBytes.value_or(0) / 1024) << " KiB";
                if (haveSelection) {
                    // Printed per cell, not only written to the file: for AutoSIS
                    // this is the term that dominates the cell and it is the number
                    // a reader of the console is looking for.
                    std::cout << "  selection " << std::setprecision(3)
                              << (static_cast<double>(artifact.selectionTimeNs) / 1e6) << " ms ("
                              << std::setprecision(1)
                              << (encodeNs > 0.0
                                      ? static_cast<double>(artifact.selectionTimeNs) / encodeNs
                                      : 0.0)
                              << "x encode)";
                }
                std::cout << std::defaultfloat << "\n";

                // The payload, the codec's internal state and the resolved AutoSIS
                // plan go before the next cell is timed, so its encode is measured
                // against the same fresh state this one was.
                artifacts.evict(enc.name);
            }
            writer.flush();  // keep a long sweep inspectable while it runs
            sections.flush();
        }
        // One N resident at a time: the cache never evicts on its own, and an
        // int64 stream at N = 10M is 80 MB per dataset.
        dataCache.releaseAll();
    }

    writer.close();
    sections.close();
    std::cout << "\nResults written to: " << std::filesystem::absolute(cfg.output)
              << "\nSections written to: " << std::filesystem::absolute(sectionsPath(cfg.output))
              << "\n  " << mainRows << " main rows, " << sectionRows << " section rows"
              << std::endl;

    int exitCode = 0;
    if (!validationFailures.empty()) {
        std::cerr << "\n" << validationFailures.size()
                  << " encoder/dataset pair(s) failed validation and were excluded:\n";
        for (const auto& f : validationFailures) std::cerr << "  - " << f << "\n";
        exitCode = 2;
    }

    manifest.finishedAtIso = detail::isoNow();
    manifest.exitCode = exitCode;
    manifest.writeSidecar(cfg.output);
    return exitCode;
}
