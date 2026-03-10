#pragma once

#include "generators/DataGenerator.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace encodings::generators {

// ============================================================================
// Histogram types
// ============================================================================

/**
 * @brief A value-frequency histogram for a single numeric type.
 *
 * `counts` is an ordered map (value → count) so that iteration is always
 * in ascending value order, which keeps CSV output deterministic.
 */
template<typename T>
struct Histogram {
    std::map<T, size_t> counts;      ///< value → frequency
    size_t              total{0};    ///< total number of values observed
    std::string         generatorName;

    /// Number of distinct values.
    size_t numDistinct() const { return counts.size(); }

    /// Value with the highest frequency (first if tied).
    T mode() const {
        return std::max_element(counts.begin(), counts.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; }
        )->first;
    }

    /// Minimum observed value.
    T minValue() const { return counts.begin()->first; }

    /// Maximum observed value.
    T maxValue() const { return counts.rbegin()->first; }
};

// ============================================================================
// computeHistogram
// ============================================================================

/**
 * @brief Draw `sampleCount` values from `generator` and build a histogram.
 *
 * The generator's state is left advanced by `sampleCount` positions after
 * the call (i.e. the cursor is NOT reset). Call generator.reset() before
 * this function if you want to sample from the beginning.
 *
 * @param generator    Source of values.
 * @param sampleCount  How many values to draw.
 * @return             Histogram of the drawn values.
 */
template<typename T>
Histogram<T> computeHistogram(DataGenerator<T>& generator, size_t sampleCount) {
    const std::vector<T> values = generator.generate(sampleCount);

    Histogram<T> hist;
    hist.total         = values.size();
    hist.generatorName = generator.name();

    for (const T& v : values) {
        ++hist.counts[v];
    }

    return hist;
}

// ============================================================================
// writeHistogramCSV
// ============================================================================

/**
 * @brief Write a histogram to a CSV file.
 *
 * The output file has three columns:
 *   value, count, frequency
 *
 * where `frequency` is `count / total` (a fraction in [0, 1]).
 *
 * @param hist      Histogram to serialise.
 * @param filePath  Destination file path. Parent directories must exist.
 * @param append    If true, append to an existing file instead of overwriting.
 *                  The header row is written only when creating a new file.
 */
template<typename T>
void writeHistogramCSV(const Histogram<T>& hist,
                       const std::filesystem::path& filePath,
                       bool append = false)
{
    const bool fileExists = std::filesystem::exists(filePath);
    std::ofstream out(filePath,
        append ? (std::ios::out | std::ios::app) : std::ios::out);

    if (!out.is_open()) {
        throw std::runtime_error(
            "GeneratorUtils::writeHistogramCSV: cannot open '" +
            filePath.string() + "' for writing");
    }

    // Header — only when starting a fresh file
    if (!append || !fileExists) {
        out << "value,count,frequency\n";
    }

    const double total = static_cast<double>(hist.total);
    out << std::fixed;
    out.precision(10);

    for (const auto& [val, cnt] : hist.counts) {
        out << val << ','
            << cnt << ','
            << (total > 0.0 ? cnt / total : 0.0)
            << '\n';
    }
}

// ============================================================================
// writeHistogramsCSV  (convenience: multiple histograms → one file each)
// ============================================================================

/**
 * @brief Write a collection of named histograms to separate CSV files inside
 *        a given directory.
 *
 * Each file is named `<generatorName>.csv` (spaces replaced with underscores).
 *
 * @param histograms  Map of label → histogram.
 * @param outputDir   Directory to write files into (created if absent).
 */
template<typename T>
void writeHistogramsCSV(
    const std::vector<std::pair<std::string, Histogram<T>>>& histograms,
    const std::filesystem::path& outputDir)
{
    std::filesystem::create_directories(outputDir);

    for (const auto& [label, hist] : histograms) {
        std::string filename = label;
        std::replace(filename.begin(), filename.end(), ' ', '_');
        std::replace(filename.begin(), filename.end(), '/', '_');
        filename += ".csv";

        writeHistogramCSV(hist, outputDir / filename);
    }
}

} // namespace encodings::generators
