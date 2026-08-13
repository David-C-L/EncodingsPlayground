// Encoded size, and nothing else.
//
// This is the cheapest driver and the table that gets regenerated most often, so
// it takes no timings at all: every cell is one encode() through ArtifactCache
// with EncodeMeasurement::None, and the whole sweep is bounded by the cost of
// encoding each (encoder, dataset, N) once.  Anything that would make a run cost
// minutes-per-cell — repeated encodes, warmups, cache preparation, heap
// instrumentation — belongs in bench_encode instead.
//
// The swept space is (encoder, dataset, N), with N a *list* (--sizes) rather than
// the single --n of the timing drivers: a size axis is what turns bits/element
// into a curve, and re-running the driver per size would re-pay the parquet load
// and lose the shared manifest.
//
// Two tables come out of one sweep:
//
//   <output>              one row per (encoder, dataset, N)
//   <output>_sections.*   one row per (encoder, dataset, N, section) for the
//                         SubIntSplit plans, which is where a plan's byte budget
//                         is actually decided
//
// The per-section table is populated from EncodingMetadata::subStreamEncodeMetrics,
// which only SubIntSplitEncoder<T, true> fills — the `*_Prof` registry entries.
// A non-Prof plan contributes no section rows rather than fabricated ones, and
// its total in the main table is still directly comparable, because profiling
// perturbs only timings and not the bytes written.
//
// --cache-state is accepted here purely for CLI uniformity: compression cannot
// depend on it.  --self-test is what makes that a checked claim rather than an
// assumption — see selfTest() for why it is the cheapest guard against the
// madvise-zeroing failure mode CONVENTIONS section 8 forbids.

#include "benchmark/ArtifactCache.hpp"
#include "benchmark/CachePolicy.hpp"
#include "benchmark/Cli.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/ResultWriter.hpp"
#include "benchmark/RunManifest.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "benchmark/targets/PlaygroundTarget.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
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
    // One million by default: large enough that block-structured codecs reach
    // their steady-state block count and small enough that a full sweep of every
    // registered encoder, AutoSIS included, finishes in minutes.
    std::vector<size_t> sizes{1'000'000};
    uint64_t seed{42};
    CacheState cacheState{CacheState::Hot};
    EvictMethod evictMethod{EvictMethod::Auto};
    std::vector<std::string> datasetFilters;
    std::vector<std::string> encoderFilters;
    bool validate{false};
    bool selfTest{false};
    bool dryRun{false};
    std::filesystem::path output{"Benchmarks/results/bench_compression.csv"};
    ResultFormat format{ResultFormat::Csv};
};

void bindArgs(ArgParser& args, SweepConfig& cfg) {
    args.group("Sweep axes:")
        .list("sizes", cfg.sizes, "stream lengths in elements")
        .opt("seed", cfg.seed, "recorded for provenance; no randomness in this driver");

    // Accepted, not used: compression is cache-independent, and --self-test is
    // what proves it.  Rejecting these flags would make this driver the one
    // exception in a sweep script that passes the same arguments to all of them.
    args.group("Cache state (accepted for CLI uniformity; see --self-test):")
        .enumOpt("cache-state", cfg.cacheState,
                 {{"hot", CacheState::Hot},
                  {"cold-payload", CacheState::ColdPayload},
                  {"cold-all", CacheState::ColdAll}},
                 "recorded in the output; must not change any payload byte")
        .enumOpt("evict-method", cfg.evictMethod,
                 {{"auto", EvictMethod::Auto},
                  {"clflush", EvictMethod::Clflush},
                  {"llc-thrash", EvictMethod::LlcThrash},
                  {"none", EvictMethod::None}},
                 "how a cold state would be produced");

    args.group("Selection and output:")
        .repeated("dataset", cfg.datasetFilters, "only datasets whose name contains SUBSTR")
        .repeated("encoder", cfg.encoderFilters, "only encoders whose name contains SUBSTR")
        .flag("validate", cfg.validate, "round-trip each cell before recording its size")
        .flag("self-test", cfg.selfTest,
              "encode under hot and cold-all and require identical payload bytes, then exit")
        .flag("dry-run", cfg.dryRun, "print the preflight summary and exit")
        .opt("output", cfg.output, "result file")
        .enumOpt("format", cfg.format,
                 {{"csv", ResultFormat::Csv}, {"parquet", ResultFormat::Parquet}},
                 "result file format");
}

bool resolveConfig(SweepConfig& cfg, std::string& whyNot) {
    std::sort(cfg.sizes.begin(), cfg.sizes.end());
    cfg.sizes.erase(std::unique(cfg.sizes.begin(), cfg.sizes.end()), cfg.sizes.end());
    // A zero would reach DatasetCache::materialize as an exception per dataset
    // rather than as one diagnosis of the flag that caused it.
    std::erase(cfg.sizes, size_t{0});
    if (cfg.sizes.empty()) {
        whyNot = "--sizes resolved to an empty list";
        return false;
    }
    return true;
}

/// `foo.csv` -> `foo_sections.csv`.  The two tables share a stem so a plotting
/// script can find one from the other, and share the single manifest sidecar
/// written next to the main file.
std::filesystem::path sectionsPath(const std::filesystem::path& output) {
    std::filesystem::path p = output;
    p.replace_filename(output.stem().string() + "_sections" + output.extension().string());
    return p;
}

// ─── Result schema ───────────────────────────────────────────────────────────

/// The main table, in file order.
///
/// The timing columns CONVENTIONS section 6 requires of every row are present and
/// always null: this driver measures no time, and a 0 in `time_ns` would be read
/// as an instantaneous codec by anything that concatenates driver outputs.
/// `index_bytes` and `permutation_bytes` are likewise null for codecs that do not
/// publish them in `customMetadata` — absent is not zero.
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
        intCol("index_bytes"),
        intCol("permutation_bytes"),
        intCol("section_count"),
        intCol("iterations"),
        intCol("warmup"),
        intCol("time_ns"),
        intCol("time_p90_ns"),
        intCol("time_min_ns"),
        intCol("truncated"),
        intCol("skipped"),
    };
}

/// The per-section companion table.
///
/// `bit_lo`/`bit_hi` are offsets *in section order*, accumulated from the section
/// bit widths.  They are not value bit positions: the mapping depends on the
/// plan's BitSplitOrder, which subStreamEncodeMetrics does not carry, and
/// inventing it here would silently mirror every MSB plan.
std::vector<ColumnSpec> sectionColumns() {
    return {
        stringCol("driver"),
        stringCol("dataset"),
        stringCol("encoding"),
        stringCol("family"),
        stringCol("variant"),
        intCol("N"),
        intCol("seed"),
        intCol("section_index"),
        intCol("section_count"),
        intCol("bit_width"),
        intCol("bit_lo"),
        intCol("bit_hi"),
        stringCol("section_encoding"),
        intCol("section_encoded_bytes"),
        doubleCol("section_bits_per_element"),
        intCol("payload_bytes"),
    };
}

// ─── Preflight ───────────────────────────────────────────────────────────────

void printPreflight(const SweepConfig& cfg,
                    const std::vector<EncoderEntry<Elem>>& encoders,
                    const std::vector<DatasetEntry<Elem>>& datasets) {
    std::cout << "\n── Sweep configuration ──────────────────────────────────────\n"
              << "  sizes          ";
    for (size_t i = 0; i < cfg.sizes.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << cfg.sizes[i];
    }
    std::cout << "  (" << cfg.sizes.size() << " step" << (cfg.sizes.size() == 1 ? "" : "s")
              << ", largest = " << (cfg.sizes.back() * kElemSize / (1024 * 1024)) << " MiB raw)\n"
              << "  element size   " << kElemSize << " bytes (int64_t)\n"
              << "  encoders       " << encoders.size() << "\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  cache state    " << cacheStateName(cfg.cacheState) << "/"
              << evictMethodName(cfg.evictMethod) << "  (recorded only)\n"
              << "  validate       " << (cfg.validate ? "yes" : "no") << "\n";

    size_t sectionEncoders = 0;
    for (const auto& e : encoders) {
        if (e.name.find("_Prof") != std::string::npos) ++sectionEncoders;
    }

    const size_t cells = cfg.sizes.size() * encoders.size() * datasets.size();
    std::cout << "\n── Workload ─────────────────────────────────────────────────\n"
              << "  cells          " << cells << "\n"
              << "  encode calls   " << cells << "  (one per cell; no timing)\n"
              << "  main rows      " << cells << "\n"
              << "  section rows   from " << sectionEncoders
              << " profiling encoder(s) x " << (cfg.sizes.size() * datasets.size())
              << " (dataset, N) pair(s)\n\n";
}

// ─── Shared per-cell work ────────────────────────────────────────────────────

/// FNV-1a over the payload.  Used only to compare two encodes of the same input
/// byte for byte; the point is detecting a payload that changed, so a fast
/// non-cryptographic digest with a full-buffer pass is exactly the right tool.
uint64_t payloadDigest(std::span<const std::byte> bytes) {
    uint64_t h = 1469598103934665603ULL;
    for (std::byte b : bytes) {
        h ^= static_cast<uint64_t>(std::to_integer<uint8_t>(b));
        h *= 1099511628211ULL;
    }
    return h;
}

/// Full materialization compared against the source stream.
///
/// Deliberately not DecodeHarness::validate(): that additionally asserts the
/// gather-equals-range identity, which needs a CacheController and belongs to the
/// drivers that measure selective reads.  What this driver has to establish is
/// only that the bytes it is about to report the size of actually decode.
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

// ─── Self-test ───────────────────────────────────────────────────────────────

/// What one cell contributes to the hot-vs-cold comparison.
struct CellFingerprint {
    size_t payloadBytes{};
    uint64_t digest{};
    /// Whether the payload decoded back to the source stream.  Recorded rather
    /// than asserted: FPE_NoIndex reorders rows by tier and BlockFORFPE has a
    /// known decode bug, so a codec that cannot round-trip is a codec whose
    /// post-eviction decode proves nothing, not a self-test failure.
    bool decodes{};
};

using Fingerprints = std::map<std::string, CellFingerprint>;

/// Encodes every selected cell once and fingerprints the payload.
///
/// When `state` is not Hot a CacheController is driven over the payload, the
/// sink and the codec's internal buffers between the encode and the fingerprint,
/// and the payload is then round-tripped.  That ordering is the whole point: the
/// failure mode CONVENTIONS section 8 forbids — MADV_DONTNEED on private
/// anonymous heap memory faulting pages back in zeroed — corrupts the payload
/// *silently*, and shows up in this driver as impossibly good compression.  A
/// fingerprint taken before eviction, or without a decode after it, would not
/// see it.
///
/// `dataCache` is borrowed, not owned, and this is load-bearing: the generators
/// reseed from std::random_device on reset(), so two passes with two caches would
/// compress two *different* streams and every random dataset would report a
/// spurious mismatch.  Sharing the cache is what makes the two passes a test of
/// the cache plumbing rather than of the generators.
Fingerprints collectFingerprints(const SweepConfig& cfg,
                                 std::vector<EncoderEntry<Elem>>& encoders,
                                 std::vector<DatasetEntry<Elem>>& datasets,
                                 DatasetCache<Elem>& dataCache,
                                 CacheState state,
                                 const Fingerprints* baseline,
                                 std::vector<std::string>& problems,
                                 std::vector<std::string>& notCheckable) {
    Fingerprints out;

    std::unique_ptr<CacheController> cache;
    if (state != CacheState::Hot) {
        CachePolicy policy;
        policy.state = state;
        policy.method = cfg.evictMethod;
        cache = std::make_unique<CacheController>(policy, CacheTopology::detect());
    }

    // The self-test runs at the smallest requested size: it is checking that the
    // cache plumbing has no side effect on the payload, which is a property of
    // the plumbing and not of N, so paying the largest size for it would only
    // make the guard expensive enough to skip.
    const size_t n = cfg.sizes.front();

    ArtifactCache<Elem> artifacts;
    std::vector<Elem> scratch;

    for (auto& ds : datasets) {
        typename DatasetCache<Elem>::Handle handle;
        try {
            handle = dataCache.materialize(ds, n);
        } catch (const std::exception& e) {
            std::cerr << "  WARNING: dataset '" << ds.name << "' skipped: " << e.what() << "\n";
            continue;
        }

        for (auto& enc : encoders) {
            const std::string key = std::to_string(n) + "|" + ds.name + "|" + enc.name;
            const auto& artifact = artifacts.get(enc, handle, EncodeMeasurement::None);

            PlaygroundTarget<Elem> target(*enc.codec);
            target.adopt(artifact.encoded);

            if (cache) {
                std::vector<Elem> sink(handle.n);
                EvictionTargets targets;
                targets.payload = target.payloadBytes();
                targets.sink = std::as_writable_bytes(std::span<Elem>(sink));
                targets.codecInternal = target.internalBuffers();
                cache->prepare(targets);
            }

            std::string whyNot;
            const bool decodes = roundTrips(target, handle.data, scratch, whyNot);
            out[key] = CellFingerprint{artifact.payloadBytes,
                                       payloadDigest(target.payloadBytes()), decodes};

            if (!decodes) {
                bool decodedBefore = false;
                if (baseline != nullptr) {
                    const auto prior = baseline->find(key);
                    decodedBefore = prior != baseline->end() && prior->second.decodes;
                }
                if (decodedBefore) {
                    // Decoded before eviction and not after: exactly the corruption
                    // this test exists to catch.
                    problems.push_back(key + " (" + cacheStateName(state) +
                                       "): payload stopped decoding after eviction: " + whyNot);
                } else if (baseline == nullptr) {
                    // Recorded once, on the pass that has no baseline, so a codec
                    // that never round-trips is listed as one uncovered cell rather
                    // than once per pass.
                    notCheckable.push_back(key);
                }
            }
            artifacts.evict(enc.name);
        }
    }
    return out;
}

/// Runs the sweep twice and requires the payloads to be identical.
int selfTest(const SweepConfig& cfg,
             std::vector<EncoderEntry<Elem>>& encoders,
             std::vector<DatasetEntry<Elem>>& datasets) {
    std::cout << "\n── Self-test: compression must be cache-independent ─────────\n"
              << "  N              " << cfg.sizes.front() << "\n"
              << "  comparing      hot vs cold-all payload bytes and payload digest\n\n";

    std::vector<std::string> problems, notCheckable;
    // One cache, both passes: see collectFingerprints.
    DatasetCache<Elem> dataCache;

    std::cout << "  pass 1: hot ..." << std::flush;
    const Fingerprints hot =
        collectFingerprints(cfg, encoders, datasets, dataCache, CacheState::Hot, nullptr, problems,
                            notCheckable);
    std::cout << " " << hot.size() << " cells\n";

    std::cout << "  pass 2: cold-all ..." << std::flush;
    Fingerprints cold;
    try {
        cold = collectFingerprints(cfg, encoders, datasets, dataCache, CacheState::ColdAll, &hot,
                                   problems, notCheckable);
    } catch (const std::exception& e) {
        // A cold state this machine cannot deliver is an error, never a silent
        // downgrade to hot — the comparison would then be hot against hot and
        // would pass while checking nothing.
        std::cout << "\n";
        std::cerr << "ERROR: cold-all is not available here: " << e.what() << "\n";
        return 1;
    }
    std::cout << " " << cold.size() << " cells\n\n";

    size_t compared = 0, mismatches = 0;
    for (const auto& [key, h] : hot) {
        const auto found = cold.find(key);
        if (found == cold.end()) {
            problems.push_back(key + ": present under hot, absent under cold-all");
            continue;
        }
        ++compared;
        const CellFingerprint& c = found->second;
        if (h.payloadBytes != c.payloadBytes) {
            ++mismatches;
            problems.push_back(key + ": payload_bytes " + std::to_string(h.payloadBytes) +
                               " (hot) vs " + std::to_string(c.payloadBytes) + " (cold-all)");
        } else if (h.digest != c.digest) {
            ++mismatches;
            problems.push_back(key + ": payload_bytes agree at " +
                               std::to_string(h.payloadBytes) +
                               " but the bytes differ (digest mismatch)");
        }
    }

    for (const auto& p : problems) std::cerr << "  FAIL: " << p << "\n";
    // Reported, not counted as failures: a codec that does not round-trip before
    // eviction cannot demonstrate anything about eviction, and hiding it would
    // let the test claim coverage it does not have.
    if (!notCheckable.empty()) {
        std::cout << "  " << notCheckable.size()
                  << " cell(s) do not round-trip and so cannot check post-eviction decode:\n";
        for (const auto& c : notCheckable) std::cout << "    - " << c << "\n";
    }
    std::cout << "  " << compared << " cells compared, " << mismatches
              << " payload mismatch(es), " << problems.size() << " problem(s) total\n";
    if (!problems.empty()) {
        std::cout << "  SELF-TEST FAILED\n";
        return 2;
    }
    std::cout << "  SELF-TEST PASSED: hot and cold-all produce identical payload bytes\n";
    return 0;
}

}  // namespace

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SweepConfig cfg;
    ArgParser args("bench_compression",
                   "Encoded size per (encoder, dataset, N).  No timing: see "
                   "bench_encode for encode cost and the bench_decode_* drivers "
                   "for decode cost.");
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

    printPreflight(cfg, encoders, datasets);
    if (cfg.dryRun) {
        std::cout << "Dry run: nothing encoded.\n";
        return 0;
    }
    if (cfg.selfTest) return selfTest(cfg, encoders, datasets);

    RunManifest manifest = RunManifest::capture("bench_compression", args.argvEcho());
    manifest.seed = cfg.seed;
    const CacheTopology topo = CacheTopology::detect();
    manifest.setCacheSizes(topo.l1dBytes, topo.l2Bytes, topo.llcBytes, topo.lineBytes);
    for (const auto& e : encoders) manifest.encoders.push_back(e.name);
    for (const auto& d : datasets) {
        manifest.datasets.push_back(d.name);
        if (d.fileBacked) manifest.fingerprintDataset(d.path);
    }
    manifest.extra["cache_state"] = cacheStateName(cfg.cacheState);
    manifest.extra["evict_method"] = evictMethodName(cfg.evictMethod);
    manifest.extra["timed"] = "false";
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

            for (auto& enc : encoders) {
                const auto& artifact = artifacts.get(enc, handle, EncodeMeasurement::None);
                const auto& meta = artifact.encoded.metadata();

                PlaygroundTarget<Elem> target(*enc.codec);
                // A copy of the payload, not a second encode: the target owns the
                // bytes it decodes from, and re-encoding to validate would report a
                // size measured from a different encode than the one validated.
                target.adopt(artifact.encoded);

                if (cfg.validate) {
                    std::string why;
                    if (!roundTrips(target, handle.data, scratch, why)) {
                        // Excluded from the output entirely (CONVENTIONS section 5):
                        // a size for a payload that does not decode is worthless
                        // rather than merely imprecise.
                        std::cerr << "  [" << enc.name << "] EXCLUDED at N=" << n << ": " << why
                                  << "\n";
                        validationFailures.push_back(enc.name + " on " + ds.name + " at N=" +
                                                     std::to_string(n) + ": " + why);
                        artifacts.evict(enc.name);
                        continue;
                    }
                }

                const double bitsPerElement =
                    n != 0 ? static_cast<double>(artifact.payloadBytes) * 8.0 /
                                 static_cast<double>(n)
                           : 0.0;

                const auto indexBytes = artifact.encodeCustomMetrics.find("index_bytes");
                const auto permBytes = artifact.encodeCustomMetrics.find("permutation_bytes");
                const auto& secs = meta.subStreamEncodeMetrics;

                auto row = writer.row();
                row.set("driver", "bench_compression")
                    .set("dataset", ds.name)
                    .set("encoding", enc.name)
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
                    .set("payload_bytes", artifact.payloadBytes)
                    .set("original_bytes", artifact.originalBytes)
                    .set("bits_per_element", bitsPerElement)
                    .set("compression_ratio", artifact.compressionRatio)
                    // Null, not 0, when the codec does not publish the figure: FPE
                    // computes its index bytes only under verbose logging, so an
                    // absent key means "not reported", not "no index".
                    .setIf(indexBytes != artifact.encodeCustomMetrics.end(), "index_bytes",
                           static_cast<int64_t>(
                               indexBytes != artifact.encodeCustomMetrics.end()
                                   ? indexBytes->second : 0.0))
                    .setIf(permBytes != artifact.encodeCustomMetrics.end(), "permutation_bytes",
                           static_cast<int64_t>(
                               permBytes != artifact.encodeCustomMetrics.end()
                                   ? permBytes->second : 0.0))
                    .setIf(!secs.empty(), "section_count", secs.size())
                    // This driver takes no timings at all; a 0 here would read as a
                    // free codec to anything that concatenates driver outputs.
                    .setNull("evict_ns")
                    .setNull("iterations")
                    .setNull("warmup")
                    .setNull("time_ns")
                    .setNull("time_p90_ns")
                    .setNull("time_min_ns")
                    .set("truncated", n < requestedN)
                    .set("skipped", false);
                writer.write(std::move(row));
                ++mainRows;

                size_t bitLo = 0;
                for (size_t s = 0; s < secs.size(); ++s) {
                    const auto& m = secs[s];
                    auto srow = sections.row();
                    srow.set("driver", "bench_compression")
                        .set("dataset", ds.name)
                        .set("encoding", enc.name)
                        .set("family", enc.family)
                        .set("variant", enc.variant)
                        .set("N", n)
                        .set("seed", cfg.seed)
                        .set("section_index", s)
                        .set("section_count", secs.size())
                        .set("bit_width", static_cast<int64_t>(m.bitWidth))
                        .set("bit_lo", bitLo)
                        .set("bit_hi", bitLo + m.bitWidth)
                        .set("section_encoding", m.name)
                        .set("section_encoded_bytes", m.encodedBytes)
                        .set("section_bits_per_element",
                             n != 0 ? static_cast<double>(m.encodedBytes) * 8.0 /
                                          static_cast<double>(n)
                                    : 0.0)
                        .set("payload_bytes", artifact.payloadBytes);
                    sections.write(std::move(srow));
                    ++sectionRows;
                    bitLo += m.bitWidth;
                }

                std::cout << "    " << std::setw(24) << std::left << enc.name << std::right
                          << std::setw(12) << artifact.payloadBytes << " B  "
                          << std::fixed << std::setprecision(3) << std::setw(8)
                          << bitsPerElement << " b/elem  ratio "
                          << std::setprecision(2) << artifact.compressionRatio << "x";
                if (!secs.empty()) std::cout << "  (" << secs.size() << " sections)";
                std::cout << std::defaultfloat << "\n";

                // The payload and the codec's internal state go before the next
                // encoder, so a long sweep's peak footprint is one artifact.
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
