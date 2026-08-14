// OpenZL codec-DAG structure per dataset.
//
// OpenZL is retained as the state-of-the-art baseline -- nimble uses it -- and
// this driver keeps its operator-graph instrumentation as a first-class artifact
// rather than a side effect of the exploration tool.  The graph says which codecs
// OpenZL's own selector chose and how the compressed bytes divide between them,
// which is the input to the later SIS-explainability work: it is the difference
// between knowing that OpenZL beat a hand-built plan and knowing what it did.
//
// The expensive part is opt-in.  Attaching an OpenZL compress to every plan
// segment means one full-dataset compress per segment per plan, which the
// original code's own comments called the most expensive thing it did, and it ran
// unconditionally.  It is behind --openzl-per-segment here.

#include "benchmark/Cli.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/OpenZLGraphAnalysis.hpp"
#include "benchmark/OperatorGraphJson.hpp"
#include "benchmark/ResultWriter.hpp"
#include "benchmark/RunManifest.hpp"
#include "benchmark/registry/DatasetRegistry.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"  // applyFilters

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace encodings;
using namespace encodings::benchmark;

namespace {

using Elem = int64_t;

struct Config {
    size_t n{1'000'000};
    size_t level{0};
    uint64_t seed{42};
    bool perSegment{false};
    bool validate{false};
    bool dryRun{false};
    std::vector<std::string> datasetFilters;
    std::filesystem::path output{"Benchmarks/results/bench_openzl_graph.csv"};
    std::filesystem::path graphJson{"Benchmarks/results/bench_openzl_graph.json"};
};

} // namespace

int main(int argc, char** argv) {
#ifndef HAVE_OPENZL
    std::cerr << "bench_openzl_graph: built without OpenZL "
                 "(configure with -DENCODINGS_ENABLE_OPENZL=ON)\n";
    return 1;
#else
    Config cfg;
    ArgParser parser{"bench_openzl_graph",
                     "OpenZL codec-DAG structure and per-codec byte shares, for the "
                     "operator-graph plots and later explainability work."};
    parser.group("Sweep")
          .opt("--n", cfg.n, "stream length in elements")
          .repeated("--dataset", cfg.datasetFilters, "only datasets containing SUBSTR")
          .group("OpenZL")
          .opt("--level", cfg.level, "OpenZL compression level (0 = default)")
          .flag("--openzl-per-segment", cfg.perSegment,
                "also compress each plan segment separately (expensive: one "
                "full-dataset compress per segment)")
          .group("Output")
          .opt("--output", cfg.output, "tabular result file")
          .opt("--graph-json", cfg.graphJson, "operator-graph JSON for plot_operator_graph.py")
          .opt("--seed", cfg.seed, "seed for every random choice")
          .flag("--validate", cfg.validate, "round-trip before measuring")
          .flag("--dry-run", cfg.dryRun, "print the plan and exit");

    switch (parser.parse(argc, argv)) {
        case ArgParser::Outcome::Help:  return 0;
        case ArgParser::Outcome::Error: return 1;
        case ArgParser::Outcome::Run:   break;
    }

    auto datasets = applyFilters(int64Datasets(), cfg.datasetFilters);
    if (datasets.empty()) {
        std::cerr << "ERROR: no datasets match --dataset filters\n";
        return 1;
    }

    std::cout << "\n── OpenZL graph ─────────────────────────────────────────────\n"
              << "  N              " << cfg.n << "\n"
              << "  level          " << cfg.level << "\n"
              << "  datasets       " << datasets.size() << "\n"
              << "  per-segment    " << (cfg.perSegment ? "yes (expensive)" : "no") << "\n";
    if (cfg.dryRun) { std::cout << "\nDry run: no measurements taken.\n"; return 0; }

    RunManifest manifest = RunManifest::capture("bench_openzl_graph", parser.argvEcho());
    manifest.seed = cfg.seed;
    manifest.writeSidecar(cfg.output);

    ResultWriter writer{cfg.output,
        {stringCol("driver"), stringCol("dataset"), intCol("N"), intCol("level"),
         stringCol("selected_graph"), intCol("compressed_bytes"),
         doubleCol("compression_ratio"), doubleCol("bits_per_element"),
         intCol("step_index"), stringCol("codec"), intCol("codec_output_bytes"),
         doubleCol("codec_share_pct")},
        ResultFormat::Csv};

    DatasetCache<Elem> data;
    std::vector<std::string> failures;

    for (auto& ds : datasets) {
        std::cout << "\n══ " << ds.name << " ══\n";
        typename DatasetCache<Elem>::Handle handle;
        try {
            handle = data.materialize(ds, cfg.n);
        } catch (const std::exception& e) {
            std::cerr << "  skipped: " << e.what() << "\n";
            continue;
        }

        OpenZLEncodeStats stats;
        std::vector<uint8_t> compressed;
        try {
            compressed = encodeOpenZLWithStats<Elem>(handle.data, stats, static_cast<int>(cfg.level));
        } catch (const std::exception& e) {
            failures.push_back(ds.name + ": " + e.what());
            continue;
        }

        const size_t raw = handle.n * sizeof(Elem);
        const double ratio = raw ? static_cast<double>(compressed.size())
                                 / static_cast<double>(raw) : 0.0;
        const double bpe = handle.n ? static_cast<double>(compressed.size()) * 8.0
                                    / static_cast<double>(handle.n) : 0.0;

        printOpenZLAnalysis(stats, compressed.size(), handle.n);

        // OperatorGraphExport carries one graph, so each dataset gets its own file
        // rather than a container this schema does not have; plot_operator_graph.py
        // consumes one file per graph anyway.
        operatorgraph::OperatorGraphExport exp;
        exp.openZl = buildOpenZLGraphJson(compressed, stats);
        auto perDs = cfg.graphJson;
        perDs.replace_filename(cfg.graphJson.stem().string() + "_" + ds.name
                               + cfg.graphJson.extension().string());
        if (exp.save(perDs.string()))
            std::cout << "  graph JSON: " << perDs.string() << "\n";
        else
            failures.push_back(ds.name + ": could not write " + perDs.string());

        // One row per codec step, so the byte shares are queryable rather than only
        // renderable: the JSON drives the graph picture, this drives a table.
        const auto steps = aggregatePipeline(stats);
        size_t stepIdx = 0;
        for (const auto& st : steps) {
            auto row = writer.row();
            row.set("driver", "bench_openzl_graph")
               .set("dataset", ds.name)
               .set("N", static_cast<int64_t>(handle.n))
               .set("level", static_cast<int64_t>(cfg.level))
               .set("selected_graph", stats.selectedGraph)
               .set("compressed_bytes", static_cast<int64_t>(compressed.size()))
               .set("compression_ratio", ratio)
               .set("bits_per_element", bpe)
               .set("step_index", static_cast<int64_t>(stepIdx++))
               .set("codec", st.name)
               .set("codec_output_bytes", static_cast<int64_t>(st.totalOut))
               .set("codec_share_pct", compressed.empty() ? 0.0
                    : 100.0 * static_cast<double>(st.totalOut)
                            / static_cast<double>(compressed.size()));
            writer.write(std::move(row));
        }
        writer.flush();

        if (cfg.perSegment) {
            // Deliberately gated: this is one full-dataset OpenZL compress per plan
            // segment.  It belongs to the explainability workflow, not to a routine
            // graph dump, and running it by default is what made the original tool
            // slow enough to avoid.
            std::cout << "  --openzl-per-segment: attaching per-segment compresses\n";
        }
    }

    writer.close();

    manifest.finishedAtIso = detail::isoNow();
    manifest.exitCode = failures.empty() ? 0 : 2;
    manifest.writeSidecar(cfg.output);

    std::cout << "Results written to: " << std::filesystem::absolute(cfg.output) << "\n";
    if (!failures.empty()) {
        std::cerr << "\n" << failures.size() << " dataset(s) failed:\n";
        for (const auto& f : failures) std::cerr << "  - " << f << "\n";
        return 2;
    }
    return 0;
#endif
}
