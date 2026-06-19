#pragma once

#include "generators/DataGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__has_include)
#  if __has_include(<zstd.h>)
#    include <zstd.h>
#    define ENCODINGS_GENERATORS_HAVE_ZSTD 1
#  endif
#endif

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

// ============================================================================
// Sortedness metrics
// ============================================================================
//
// These metrics quantify how "sorted"/temporally-coherent a generator's
// output is, which predicts whether a reordering pass (Sort, WindowedSort,
// BWT, MTF, ...) is likely to improve compression.

/**
 * @brief Tunable parameters for the more expensive sortedness metrics.
 */
struct SortednessOptions {
    size_t compressionSampleSize = 1'000'000; ///< rows used for the zstd CR delta (4.3)
    int    zstdLevel             = 1;
    size_t apEnSampleSize        = 5'000;     ///< ApEn is O(N^2 * m); cap N (4.5)
    size_t apEnM                 = 2;
    double apEnRFactor           = 0.2;       ///< r = apEnRFactor * stddev(values)
    size_t miNumBins             = 32;        ///< histogram bins for mutual information (4.7)
    size_t inversionsSampleCap   = 2'000'000; ///< cap for merge-sort inversion count (4.4)
};

/**
 * @brief Sortedness/compressibility metrics for a single generated column.
 */
struct SortednessMetrics {
    std::string generatorName;
    size_t      sampleSize{0};

    double lag1Autocorrelation{0.0};        ///< 4.1  rho_1 (1 = smooth/sorted, 0 = random)
    double runLengthEntropy{0.0};           ///< 4.2  H_RLE, in bits
    double runLengthEntropyNormalized{0.0}; ///< 4.2  S_RLE = 1 - H_RLE / log2(N)

    bool   compressionAvailable{false};     ///< false if zstd.h was not found at build time
    double compressionRatioOriginal{0.0};   ///< 4.3  uncompressed / compressed (original order)
    double compressionRatioSorted{0.0};     ///< 4.3  uncompressed / compressed (sorted order)
    double compressionRatioDelta{0.0};      ///< 4.3  CR_sorted / CR_original - 1

    double normalizedInversions{0.0};       ///< 4.4  tau_dist in [0, 1] (0 = sorted, 0.5 = random)
    double approximateEntropy{0.0};         ///< 4.5  ApEn(m, r, N)

    size_t cardinality{0};                  ///< 4.6  number of distinct values
    double skewness{0.0};                   ///< 4.6  third standardized moment

    double mutualInformationAdjacent{0.0};  ///< 4.7  MI(X_t; X_{t+1}), in bits
};

/**
 * @brief 4.1 Lag-1 Pearson autocorrelation.
 *
 * rho_1 ~= 1 -> smooth/sorted (delta encoding works well).
 * rho_1 ~= 0 -> effectively random (reordering can help significantly).
 */
template<typename T>
double computeLag1Autocorrelation(const std::vector<T>& values) {
    const size_t n = values.size();
    if (n < 2) return 0.0;

    double mean = 0.0;
    for (const T& v : values) mean += static_cast<double>(v);
    mean /= static_cast<double>(n);

    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(values[i]) - mean;
        denominator += d * d;
        if (i + 1 < n) {
            const double d1 = static_cast<double>(values[i + 1]) - mean;
            numerator += d * d1;
        }
    }

    return denominator > 0.0 ? numerator / denominator : 0.0;
}

/**
 * @brief 4.2 Run-length entropy (compression proxy).
 *
 * Returns {H_RLE, S_RLE} where H_RLE is the Shannon entropy (bits) of the
 * run-length distribution, and S_RLE = 1 - H_RLE / log2(N) is a normalized
 * sortedness score (S_RLE ~= 0 -> random order, reordering will help greatly).
 */
template<typename T>
std::pair<double, double> computeRunLengthEntropy(const std::vector<T>& values) {
    const size_t n = values.size();
    if (n == 0) return {0.0, 0.0};

    std::unordered_map<size_t, size_t> runLengthCounts;
    size_t currentRun = 1;
    size_t numRuns = 0;
    for (size_t i = 1; i < n; ++i) {
        if (values[i] == values[i - 1]) {
            ++currentRun;
        } else {
            ++runLengthCounts[currentRun];
            ++numRuns;
            currentRun = 1;
        }
    }
    ++runLengthCounts[currentRun];
    ++numRuns;

    double entropy = 0.0;
    const double total = static_cast<double>(numRuns);
    for (const auto& [runLen, count] : runLengthCounts) {
        (void)runLen;
        const double p = static_cast<double>(count) / total;
        if (p > 0.0) entropy -= p * std::log2(p);
    }

    if (n <= 1) return {entropy, 1.0};

    const double hMax = std::log2(static_cast<double>(n));
    const double normalized = hMax > 0.0 ? (1.0 - entropy / hMax) : 1.0;
    return {entropy, normalized};
}

/**
 * @brief 4.3 Empirical compression ratio delta (zstd, original vs. sorted order).
 *
 * Compresses the first min(N, opts.compressionSampleSize) elements at
 * opts.zstdLevel, both as-is and sorted, and fills out.compressionRatio*.
 * Leaves out.compressionAvailable == false (no-op) if zstd.h was not found
 * at build time.
 */
template<typename T>
void computeCompressionRatioDelta(const std::vector<T>& values,
                                   SortednessMetrics& out,
                                   const SortednessOptions& opts) {
#ifdef ENCODINGS_GENERATORS_HAVE_ZSTD
    if (values.empty()) return;

    const size_t sampleCount = std::min(values.size(), opts.compressionSampleSize);
    const size_t bytesSize = sampleCount * sizeof(T);

    std::vector<T> sortedSample(values.begin(), values.begin() + sampleCount);
    std::sort(sortedSample.begin(), sortedSample.end());

    auto compressedSize = [&](const T* data) -> size_t {
        const size_t bound = ZSTD_compressBound(bytesSize);
        std::vector<uint8_t> output(bound);
        const size_t csize = ZSTD_compress(output.data(), bound, data, bytesSize, opts.zstdLevel);
        return ZSTD_isError(csize) ? bytesSize : csize;
    };

    const size_t origCompressed   = compressedSize(values.data());
    const size_t sortedCompressed = compressedSize(sortedSample.data());

    out.compressionAvailable = true;
    out.compressionRatioOriginal = origCompressed > 0
        ? static_cast<double>(bytesSize) / static_cast<double>(origCompressed) : 0.0;
    out.compressionRatioSorted = sortedCompressed > 0
        ? static_cast<double>(bytesSize) / static_cast<double>(sortedCompressed) : 0.0;
    out.compressionRatioDelta = out.compressionRatioOriginal > 0.0
        ? out.compressionRatioSorted / out.compressionRatioOriginal - 1.0 : 0.0;
#else
    (void)values; (void)out; (void)opts;
#endif
}

/**
 * @brief 4.4 Normalized number of inversions (Kendall's Tau distance from sorted).
 *
 * Counts inversions in the first min(N, sampleCap) elements via a
 * merge-sort-based O(n log n) algorithm, normalized by C(n, 2).
 *
 * 0    -> already sorted
 * 0.5  -> random order
 * 1    -> reverse sorted
 */
template<typename T>
double computeNormalizedInversions(const std::vector<T>& values, size_t sampleCap) {
    const size_t n = std::min(values.size(), sampleCap);
    if (n < 2) return 0.0;

    std::vector<T> arr(values.begin(), values.begin() + n);
    std::vector<T> buffer(n);

    std::function<uint64_t(size_t, size_t)> mergeCount = [&](size_t lo, size_t hi) -> uint64_t {
        if (hi - lo <= 1) return 0;
        const size_t mid = lo + (hi - lo) / 2;
        uint64_t count = mergeCount(lo, mid) + mergeCount(mid, hi);

        size_t i = lo, j = mid, k = lo;
        while (i < mid && j < hi) {
            if (arr[i] <= arr[j]) {
                buffer[k++] = arr[i++];
            } else {
                count += (mid - i);
                buffer[k++] = arr[j++];
            }
        }
        while (i < mid) buffer[k++] = arr[i++];
        while (j < hi)  buffer[k++] = arr[j++];
        for (size_t x = lo; x < hi; ++x) arr[x] = buffer[x];

        return count;
    };

    const uint64_t inversions = mergeCount(0, n);
    const double totalPairs = static_cast<double>(n) * static_cast<double>(n - 1) / 2.0;
    return totalPairs > 0.0 ? static_cast<double>(inversions) / totalPairs : 0.0;
}

/**
 * @brief 4.5 Approximate Entropy (ApEn).
 *
 * Computed on the first min(N, sampleSize) elements (ApEn is O(N^2 * m), so
 * sampleSize should stay small). Tolerance r = rFactor * stddev(values).
 * Low ApEn -> regular/predictable -> compresses well.
 * High ApEn -> irregular -> reordering helps.
 */
template<typename T>
double computeApproximateEntropy(const std::vector<T>& values, size_t m, double rFactor, size_t sampleSize) {
    const size_t n = std::min(values.size(), sampleSize);
    if (n <= m + 1) return 0.0;

    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += static_cast<double>(values[i]);
    mean /= static_cast<double>(n);

    double variance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(values[i]) - mean;
        variance += d * d;
    }
    variance /= static_cast<double>(n);

    const double r = rFactor * std::sqrt(variance);
    if (r <= 0.0) return 0.0;

    auto phi = [&](size_t mm) -> double {
        const size_t numVectors = n - mm + 1;
        double sumLog = 0.0;
        for (size_t i = 0; i < numVectors; ++i) {
            size_t matches = 0;
            for (size_t j = 0; j < numVectors; ++j) {
                double maxDiff = 0.0;
                for (size_t k = 0; k < mm; ++k) {
                    const double diff = std::fabs(
                        static_cast<double>(values[i + k]) - static_cast<double>(values[j + k]));
                    maxDiff = std::max(maxDiff, diff);
                }
                if (maxDiff <= r) ++matches;
            }
            sumLog += std::log(static_cast<double>(matches) / static_cast<double>(numVectors));
        }
        return sumLog / static_cast<double>(numVectors);
    };

    return phi(m) - phi(m + 1);
}

/**
 * @brief 4.6 Cardinality and value-distribution skew.
 *
 * High skew + low cardinality -> sorting by this column first will create
 * long homogeneous runs -> large compression gains.
 */
template<typename T>
std::pair<size_t, double> computeCardinalitySkew(const std::vector<T>& values) {
    if (values.empty()) return {0, 0.0};

    const std::unordered_set<T> uniqueVals(values.begin(), values.end());
    const size_t n = values.size();

    double mean = 0.0;
    for (const T& v : values) mean += static_cast<double>(v);
    mean /= static_cast<double>(n);

    double m2 = 0.0, m3 = 0.0;
    for (const T& v : values) {
        const double d = static_cast<double>(v) - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= static_cast<double>(n);
    m3 /= static_cast<double>(n);

    const double sigma = std::sqrt(m2);
    const double skew = sigma > 0.0 ? m3 / (sigma * sigma * sigma) : 0.0;

    return {uniqueVals.size(), skew};
}

/**
 * @brief 4.7 Mutual information between adjacent rows, MI(X_t; X_{t+1}), in bits.
 *
 * Values are bucketed into numBins equal-width bins over [min, max]. High MI
 * -> adjacent rows are similar -> already well-ordered for compression.
 */
template<typename T>
double computeMutualInformationAdjacent(const std::vector<T>& values, size_t numBins) {
    const size_t n = values.size();
    if (n < 2 || numBins == 0) return 0.0;

    double minV = static_cast<double>(values[0]);
    double maxV = minV;
    for (const T& v : values) {
        const double d = static_cast<double>(v);
        minV = std::min(minV, d);
        maxV = std::max(maxV, d);
    }

    const double range = maxV - minV;
    if (range <= 0.0) return 0.0; // constant column -> no information

    auto binIndex = [&](double v) -> size_t {
        const size_t b = static_cast<size_t>((v - minV) / range * static_cast<double>(numBins));
        return std::min(b, numBins - 1);
    };

    std::vector<std::vector<size_t>> joint(numBins, std::vector<size_t>(numBins, 0));
    std::vector<size_t> marginalX(numBins, 0);
    std::vector<size_t> marginalY(numBins, 0);

    const size_t numPairs = n - 1;
    for (size_t i = 0; i < numPairs; ++i) {
        const size_t bx = binIndex(static_cast<double>(values[i]));
        const size_t by = binIndex(static_cast<double>(values[i + 1]));
        ++joint[bx][by];
        ++marginalX[bx];
        ++marginalY[by];
    }

    double mi = 0.0;
    const double total = static_cast<double>(numPairs);
    for (size_t bx = 0; bx < numBins; ++bx) {
        if (marginalX[bx] == 0) continue;
        const double px = static_cast<double>(marginalX[bx]) / total;
        for (size_t by = 0; by < numBins; ++by) {
            if (joint[bx][by] == 0) continue;
            const double pxy = static_cast<double>(joint[bx][by]) / total;
            const double py = static_cast<double>(marginalY[by]) / total;
            if (py > 0.0) mi += pxy * std::log2(pxy / (px * py));
        }
    }
    return mi;
}

/**
 * @brief Compute all sortedness metrics (4.1-4.7) for a single column.
 */
template<typename T>
SortednessMetrics computeSortednessMetrics(
    const std::vector<T>& values,
    const std::string& generatorName,
    const SortednessOptions& opts = {})
{
    SortednessMetrics out;
    out.generatorName = generatorName;
    out.sampleSize    = values.size();

    out.lag1Autocorrelation = computeLag1Autocorrelation(values);

    const auto [hRle, sRle] = computeRunLengthEntropy(values);
    out.runLengthEntropy           = hRle;
    out.runLengthEntropyNormalized = sRle;

    computeCompressionRatioDelta(values, out, opts);

    out.normalizedInversions = computeNormalizedInversions(values, opts.inversionsSampleCap);
    out.approximateEntropy   = computeApproximateEntropy(values, opts.apEnM, opts.apEnRFactor, opts.apEnSampleSize);

    const auto [cardinality, skew] = computeCardinalitySkew(values);
    out.cardinality = cardinality;
    out.skewness     = skew;

    out.mutualInformationAdjacent = computeMutualInformationAdjacent(values, opts.miNumBins);

    return out;
}

// ============================================================================
// writeSortednessMetricsJSON
// ============================================================================

/**
 * @brief Write a collection of named SortednessMetrics to a single JSON file.
 *
 * Output shape: { "datasets": [ { "name": ..., "lag1Autocorrelation": ..., ... }, ... ] }
 *
 * @param metrics   List of (label, metrics) pairs, in output order.
 * @param filePath  Destination file path. Parent directories are created if absent.
 */
inline void writeSortednessMetricsJSON(
    const std::vector<std::pair<std::string, SortednessMetrics>>& metrics,
    const std::filesystem::path& filePath)
{
    if (filePath.has_parent_path()) {
        std::filesystem::create_directories(filePath.parent_path());
    }

    std::ofstream out(filePath);
    if (!out.is_open()) {
        throw std::runtime_error(
            "GeneratorUtils::writeSortednessMetricsJSON: cannot open '" +
            filePath.string() + "' for writing");
    }

    auto escape = [](const std::string& s) {
        std::string r;
        for (char c : s) {
            switch (c) {
                case '"':  r += "\\\""; break;
                case '\\': r += "\\\\"; break;
                default:   r += c;
            }
        }
        return r;
    };

    out.precision(10);

    out << "{\n  \"datasets\": [\n";
    for (size_t i = 0; i < metrics.size(); ++i) {
        const auto& [name, m] = metrics[i];
        out << "    {\n";
        out << "      \"name\": \"" << escape(name) << "\",\n";
        out << "      \"sampleSize\": " << m.sampleSize << ",\n";
        out << "      \"lag1Autocorrelation\": " << m.lag1Autocorrelation << ",\n";
        out << "      \"runLengthEntropy\": " << m.runLengthEntropy << ",\n";
        out << "      \"runLengthEntropyNormalized\": " << m.runLengthEntropyNormalized << ",\n";
        out << "      \"compressionAvailable\": " << (m.compressionAvailable ? "true" : "false") << ",\n";
        out << "      \"compressionRatioOriginal\": " << m.compressionRatioOriginal << ",\n";
        out << "      \"compressionRatioSorted\": " << m.compressionRatioSorted << ",\n";
        out << "      \"compressionRatioDelta\": " << m.compressionRatioDelta << ",\n";
        out << "      \"normalizedInversions\": " << m.normalizedInversions << ",\n";
        out << "      \"approximateEntropy\": " << m.approximateEntropy << ",\n";
        out << "      \"cardinality\": " << m.cardinality << ",\n";
        out << "      \"skewness\": " << m.skewness << ",\n";
        out << "      \"mutualInformationAdjacent\": " << m.mutualInformationAdjacent << "\n";
        out << "    }";
        if (i + 1 < metrics.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

} // namespace encodings::generators
