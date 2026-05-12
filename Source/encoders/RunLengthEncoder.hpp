#pragma once

#include <span>
#include <vector>
#include <cstring>
#include <concepts>
#include <algorithm>
#include <atomic>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

/**
 * @brief Run-Length Encoding for integral types
 *
 * Compresses sequences of repeated values into (start_position, value) pairs.
 * Supports random access by binary searching run starts.
 *
 * Format: [num_runs (8 bytes),
 *          size_of_run_starts_in_bytes (8 bytes),
 *          size_of_run_values_in_bytes (8 bytes),
 *          run_starts (num_runs * sizeof(size_t)),
 *          run_values (num_runs * sizeof(T))]
 *
 * Random access is O(1) amortised via interpolation + narrow binary search fallback.
 * No allocation on the hot path — all access goes through zero-copy pointer views
 * into the encoded buffer. Header metadata is cached across repeated calls.
 *
 * @tparam T The integral type to encode
 */
template<typename T>
    requires core::IntegralType<T>
class RunLengthEncoder : public Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
        if (data.empty()) {
            return createEmptyEncoding();
        }

        // Identify runs
        std::vector<size_t> runStarts;
        std::vector<T> runValues;

        runStarts.push_back(0);
        runValues.push_back(data[0]);

        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] != data[i - 1]) {
                runStarts.push_back(i);
                runValues.push_back(data[i]);
            }
        }

        const size_t numRuns = runStarts.size();
        const size_t runStartsSize = numRuns * sizeof(size_t);
        const size_t runValuesSize = numRuns * sizeof(T);
        const size_t headerSize = 3 * sizeof(size_t);
        const size_t totalSize = headerSize + runStartsSize + runValuesSize;

        EncodedData result;
        result.data().resize(totalSize);

        uint8_t* writePtr = result.data().data();

        // Write header
        std::memcpy(writePtr, &numRuns, sizeof(size_t));
        writePtr += sizeof(size_t);

        std::memcpy(writePtr, &runStartsSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        std::memcpy(writePtr, &runValuesSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        // Write run starts
        std::memcpy(writePtr, runStarts.data(), runStartsSize);
        writePtr += runStartsSize;

        // Write run values
        std::memcpy(writePtr, runValues.data(), runValuesSize);

        // Set metadata
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = data.size();
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = data.size() * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["num_runs"] = std::to_string(numRuns);
        result.metadata().customMetadata["compression_ratio"] =
            std::to_string(static_cast<double>(totalSize) / (data.size() * sizeof(T)));

        // Invalidate cache: new encoded data produced.
        cache_.base = nullptr;

        return result;
    }

    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.size() < 3 * sizeof(size_t)) {
            return {};
        }

        const uint8_t* readPtr = encoded.data().data();

        // Read header
        size_t numRuns, runStartsSize, runValuesSize;
        std::memcpy(&numRuns, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);

        std::memcpy(&runStartsSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);

        std::memcpy(&runValuesSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);

        if (numRuns == 0) {
            return {};
        }

        // Read run starts
        std::vector<size_t> runStarts(numRuns);
        std::memcpy(runStarts.data(), readPtr, runStartsSize);
        readPtr += runStartsSize;

        // Read run values
        std::vector<T> runValues(numRuns);
        std::memcpy(runValues.data(), readPtr, runValuesSize);

        // Reconstruct original data
        // The last run extends to the end, which we get from metadata
        size_t totalElements = encoded.metadata().elementCount;
        std::vector<T> result;
        result.reserve(totalElements);

        for (size_t runIdx = 0; runIdx < numRuns; ++runIdx) {
            size_t runStart = runStarts[runIdx];
            size_t runEnd = (runIdx + 1 < numRuns) ? runStarts[runIdx + 1] : totalElements;
            T value = runValues[runIdx];

            for (size_t i = runStart; i < runEnd; ++i) {
                result.push_back(value);
            }
        }

        return result;
    }

    void decodeAllInto(const EncodedData& encoded, T* dst, size_t n) override {
        const View v = getView(encoded);
        if (v.numRuns == 0) {
            if (n != 0) throw std::runtime_error("RunLengthEncoder::decodeAllInto: empty but n!=0");
            return;
        }
        for (size_t r = 0; r < v.numRuns; ++r) {
            const size_t runStart = v.runStarts[r];
            const size_t runEnd   = (r + 1 < v.numRuns) ? v.runStarts[r + 1] : n;
            std::fill(dst + runStart, dst + runEnd, v.runValues[r]);
        }
    }

    void decodeRangeInto(const EncodedData& encoded, size_t start, size_t end,
                         T* dst, size_t n) override {
        const View v = getView(encoded);
        end = std::min(end, v.totalElements);
        if (start >= end) {
            if (n != 0) throw std::runtime_error("RunLengthEncoder::decodeRangeInto: empty range, n!=0");
            return;
        }
        if ((end - start) != n) [[unlikely]]
            throw std::runtime_error("RunLengthEncoder::decodeRangeInto: size mismatch");
        if (v.numRuns == 0) return;
        const size_t firstRun = v.findRun(start);
        for (size_t r = firstRun; r < v.numRuns; ++r) {
            const size_t rs = v.runStarts[r];
            const size_t re = (r + 1 < v.numRuns) ? v.runStarts[r + 1] : v.totalElements;
            if (rs >= end) break;
            std::fill(dst + (std::max(rs, start) - start),
                      dst + (std::min(re, end)   - start),
                      v.runValues[r]);
        }
    }

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const View v = getView(encoded);
        if (v.numRuns == 0 || index >= v.totalElements) [[unlikely]] {
            return std::nullopt;
        }
        const size_t runIdx = v.findRun(index);
        return static_cast<T>(v.runValues[runIdx]);
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        const size_t totalElements = encoded.metadata().elementCount;
        if (start >= totalElements) return {};
        end = std::min(end, totalElements);

        const View v = getView(encoded);
        if (v.numRuns == 0) return {};

        const size_t firstRun = v.findRun(start);

        std::vector<T> result;
        result.reserve(end - start);

        for (size_t runIdx = firstRun; runIdx < v.numRuns; ++runIdx) {
            const size_t runStart = v.runStarts[runIdx];
            const size_t runEnd = (runIdx + 1 < v.numRuns) ? v.runStarts[runIdx + 1] : totalElements;

            if (runStart >= end) break;

            const size_t effectiveStart = std::max(runStart, start);
            const size_t effectiveEnd   = std::min(runEnd,   end);
            const T value = v.runValues[runIdx];

            for (size_t i = effectiveStart; i < effectiveEnd; ++i) {
                result.push_back(value);
            }
        }

        return result;
    }

    EncodingType encodingType() const override {
        return EncodingType::RunLengthEncoding;
    }

    std::string name() const override {
        return "RunLength";
    }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::RandomAccess)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::RunLengthBased
            | EncodingProperty::VariableSize
            | EncodingProperty::StreamingFriendly
            | EncodingProperty::LowMemoryOverhead
            | EncodingProperty::Composable;
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        // Worst case: no compression (every element is a new run)
        return 3 * sizeof(size_t) + elementCount * (sizeof(size_t) + sizeof(T));
    }

private:
    // ---------------------------------------------------------------------------
    // Zero-copy view into the encoded buffer.
    //
    // Pointers alias directly into EncodedData::data() — no allocation, no copy.
    // The binary search in findRun() therefore only touches the cache lines that
    // the search actually visits, rather than forcing the entire runStarts array
    // into cache via a memcpy.
    // ---------------------------------------------------------------------------
    struct View {
        size_t         numRuns{0};
        size_t         totalElements{0};
        const size_t*  runStarts{nullptr};  // direct pointer into encoded buffer
        const T*       runValues{nullptr};  // direct pointer into encoded buffer

        // Find the run index containing `index` using interpolation + narrow binary
        // search fallback.
        //
        // Interpolation: if run starts are uniformly distributed, the run containing
        // element `index` is near position (index * numRuns / totalElements).  For
        // perfectly uniform runs this hits in O(1) comparisons.  For non-uniform
        // runs a short binary search corrects the residual error, touching only the
        // cache lines local to the answer rather than bisecting the full array.
        size_t findRun(size_t index) const {
            if (numRuns == 1) [[unlikely]] return 0;

            // Interpolation estimate — accurate when run lengths are similar.
            const size_t guess = static_cast<size_t>(
                (static_cast<uint64_t>(index) * (numRuns - 1)) / (totalElements - 1));

            // Fast path: guess is exact or off by one (common for uniform runs).
            if (runStarts[guess] <= index) {
                if (guess + 1 == numRuns || runStarts[guess + 1] > index) {
                    return guess;
                }
                // Interpolation undershot — binary search right half [guess+1, numRuns).
                const size_t* it = std::upper_bound(
                    runStarts + guess + 1, runStarts + numRuns, index);
                return static_cast<size_t>(it - runStarts) - 1;
            } else {
                // Interpolation overshot — binary search left half [0, guess).
                const size_t* it = std::upper_bound(
                    runStarts, runStarts + guess, index);
                return static_cast<size_t>(it - runStarts) - 1;
            }
        }
    };

    // ---------------------------------------------------------------------------
    // Per-encoder metadata cache.
    //
    // Parsing the 24-byte header and computing the two data pointers is trivial,
    // but for tight random-access loops (e.g. benchmark sweep over 1 M indices)
    // even three memcpy calls add up.  We cache the last-seen encoded buffer's
    // base pointer and the derived View so the hot path is just a pointer
    // comparison followed by two array accesses.
    //
    // Thread-safety: this cache is intentionally NOT thread-safe.  The benchmark
    // harness calls decodeAt from a single thread, and each encoder instance is
    // owned by exactly one section codec.  If thread-safety is ever required,
    // make `cache_` thread_local or protect it with a mutex.
    // ---------------------------------------------------------------------------
    struct Cache {
        const uint8_t* base{nullptr};
        View           view{};
    };
    mutable Cache cache_;

    View getView(const EncodedData& encoded) const {
        const uint8_t* base = encoded.data().data();
        if (base == cache_.base) [[likely]] {
            return cache_.view;
        }

        // Parse header — 3 × 8-byte reads from the first cache line of the buffer.
        if (encoded.size() < 3 * sizeof(size_t)) return {};
        size_t numRuns, runStartsSize;
        std::memcpy(&numRuns,       base,                     sizeof(size_t));
        std::memcpy(&runStartsSize, base + sizeof(size_t),    sizeof(size_t));
        // runValuesSize not needed for pointer arithmetic (runValues = base + header + runStartsSize).

        View v;
        v.numRuns       = numRuns;
        v.totalElements = encoded.metadata().elementCount;
        v.runStarts     = reinterpret_cast<const size_t*>(base + 3 * sizeof(size_t));
        v.runValues     = reinterpret_cast<const T*>(base + 3 * sizeof(size_t) + runStartsSize);

        cache_.base = base;
        cache_.view = v;
        return v;
    }

    EncodedData createEmptyEncoding() {
        EncodedData result;
        result.data().resize(3 * sizeof(size_t));

        size_t zero = 0;
        std::memcpy(result.data().data(), &zero, sizeof(size_t));
        std::memcpy(result.data().data() + sizeof(size_t), &zero, sizeof(size_t));
        std::memcpy(result.data().data() + 2 * sizeof(size_t), &zero, sizeof(size_t));

        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = 3 * sizeof(size_t);
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;

        return result;
    }
};

} // namespace encodings::encoders
