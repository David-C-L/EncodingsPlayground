#pragma once

// The single list of benchmark data sources.
//
// Five drivers previously each carried their own `buildDatasets()` with a
// different subset commented out and an absolute `/home/david/...` root baked
// in, so a run on any other machine silently lost its file-backed datasets.
// Path resolution lives here, once, and a dataset whose file is absent is
// dropped with a warning rather than throwing at startup — the pre-refactor
// reordering driver threw before it had done any work, which turned a missing
// parquet into "the whole benchmark is broken".

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "generators/CommonGenerators.hpp"
#include "generators/DataGenerator.hpp"
#include "generators/ParquetColumnGenerator.hpp"

namespace encodings::benchmark {

/// One data source. `generator` is shared because several drivers keep the
/// registry alive across a sweep while a DatasetCache holds the materialised
/// buffer; nothing mutates the entry after construction except the cache's own
/// `generator->reset()` before each `generate()`.
template <typename T>
struct DatasetEntry {
    std::string name;
    std::shared_ptr<encodings::datagen::DataGenerator<T>> generator;
    bool fileBacked{};
    std::filesystem::path path;  ///< empty for generated sources
};

/// Root of the `Datasets/` tree.
///
/// `$ENCODINGS_DATASETS` wins, so a CI runner or a copy of the corpus on fast
/// local disk needs no rebuild.  Otherwise the root is found by walking up from
/// the working directory looking for a `Datasets` child: drivers are run both
/// from the repo root and from `build/`, and neither can be assumed.  The
/// `__FILE__`-relative guess is the last resort because it only holds when the
/// binary runs on the machine it was compiled on.
inline std::filesystem::path datasetRoot() {
    if (const char* env = std::getenv("ENCODINGS_DATASETS"); env != nullptr && *env != '\0') {
        return std::filesystem::path(env);
    }

    std::error_code ec;
    std::filesystem::path dir = std::filesystem::current_path(ec);
    if (!ec) {
        for (int up = 0; up < 6 && !dir.empty(); ++up) {
            std::filesystem::path candidate = dir / "Datasets";
            if (std::filesystem::is_directory(candidate, ec)) return candidate;
            if (!dir.has_parent_path() || dir.parent_path() == dir) break;
            dir = dir.parent_path();
        }
    }

    return std::filesystem::path(__FILE__).parent_path()  // .../Source/benchmark/registry
        .parent_path()                                     // .../Source/benchmark
        .parent_path()                                     // .../Source
        .parent_path()                                     // repo root
        / "Datasets";
}

/// A ParquetColumnGenerator that opens its file on first use.
///
/// `ParquetColumnGenerator`'s constructor reads the whole column (tweet_ids is a
/// 250 MB parquet), and the registry is built *before* `--dataset` filters are
/// applied, so an eager entry would pay every corpus load on every run
/// regardless of what the run actually measures.  `reset()` deliberately does
/// not drop the loaded column — it only rewinds the cursor, as on the wrapped
/// generator — so repeated (dataset, N) materialisations reread nothing.
template <typename T>
class DeferredParquetColumn final : public encodings::datagen::DataGenerator<T> {
public:
    DeferredParquetColumn(std::filesystem::path path, std::string column)
        : path_(std::move(path)), column_(std::move(column)) {}

    std::vector<T> generate(size_t count) override { return loaded().generate(count); }

    std::string name() const override {
        return "ParquetColumn(" + path_.filename().string() + ":" + column_ + ")";
    }

    void reset() override {
        if (inner_) inner_->reset();
    }

    std::map<std::string, std::string> getConfig() const override {
        return {{"file", path_.string()}, {"column", column_}};
    }

private:
    encodings::generators::ParquetColumnGenerator<T>& loaded() {
        if (!inner_) {
            inner_ = std::make_unique<encodings::generators::ParquetColumnGenerator<T>>(
                path_, column_);
        }
        return *inner_;
    }

    std::filesystem::path path_;
    std::string column_;
    std::unique_ptr<encodings::generators::ParquetColumnGenerator<T>> inner_;
};

namespace detail {

/// Appends `entry` only if its backing file exists.  The warning goes to stderr
/// so a driver's stdout progress and its result file stay machine-readable, and
/// so a run on a machine with a partial corpus is still obviously partial.
template <typename T>
inline void pushIfPresent(std::vector<DatasetEntry<T>>& out, DatasetEntry<T> entry) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(entry.path, ec)) {
        std::cerr << "warning: dataset '" << entry.name << "' skipped, file not found: "
                  << entry.path.string() << "\n";
        return;
    }
    out.push_back(std::move(entry));
}

}  // namespace detail

/// Every int64 source, generated ones first so a `--dry-run` on a machine
/// without the corpus still shows a non-empty sweep.
///
/// int64 only: the drivers are instantiated on `Codec<int64_t>`.  The TPCH
/// orderkey and iNaturalist species-id parquets are int32 columns and need a
/// templated `int32Datasets()` plus an int32 encoder registry before they can
/// appear here — casting them to int64 would change the bit width under test,
/// which is the independent variable for every SubIntSplit result.
inline std::vector<DatasetEntry<int64_t>> int64Datasets() {
    using Elem = int64_t;
    using namespace encodings::generators;

    std::vector<DatasetEntry<Elem>> d;

    // Sequential ids and near-sorted ids are the best case for delta/FOR-style
    // sections; uniform 40-bit values are the worst case for all of them.
    d.push_back({"Sequential", std::make_shared<SequentialGenerator<Elem>>(0, 1), false, {}});
    d.push_back({"NearlySorted",
                 std::make_shared<NearlySortedGenerator<Elem>>(0, 1, 0.1), false, {}});
    d.push_back({"Zipfian1.0",
                 std::make_shared<ZipfianGenerator<Elem>>(1'000'000, 1.0), false, {}});
    d.push_back({"Repetitive",
                 std::make_shared<RepetitiveGenerator<Elem>>(64, 0, 1'000), false, {}});
    d.push_back({"UniformRandom",
                 std::make_shared<UniformRandomGenerator<Elem>>(0, (Elem{1} << 40)), false, {}});

    const std::filesystem::path root = datasetRoot();

    const std::filesystem::path snowflake = root / "TwitterSnowflake" / "tweet_ids.parquet";
    detail::pushIfPresent<Elem>(
        d, {"TwitterSnowflake",
            std::make_shared<DeferredParquetColumn<Elem>>(snowflake, "tweet_id"),
            true, snowflake});

    return d;
}

}  // namespace encodings::benchmark
