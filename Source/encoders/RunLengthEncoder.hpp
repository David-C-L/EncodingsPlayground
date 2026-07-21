#pragma once

#include <span>
#include <vector>
#include <cstring>
#include <concepts>
#include <algorithm>
#include <atomic>
#include <bit>
#include <functional>
#include <memory>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encoders/ISectionCodecIntegral.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

// =============================================================================
//  RunLengthConfig<T>
//
//  Optional, independently-pluggable codecs for RunLengthEncoder's two internal
//  arrays: runStarts (absolute run-start positions, monotonically increasing,
//  R entries where R = number of runs, R << N) and runValues (the R distinct
//  values, one per run). Both factories share the exact InnerFactory shape
//  RangePackSectionCodec already uses ((uint8_t bits) -> shared_ptr<ISectionCodecIntegral<uint64_t>>),
//  so any existing detail_trisplit::makeXSection-shaped factory (CascadingFOR,
//  RangePack, BlockFrequencyPartition, ...) can be passed directly with zero new
//  adapter code. The uint64_t element type here is purely the interface/logical
//  type at the ISectionCodecIntegral boundary -- it does not constrain wire-level
//  storage width; every factory still narrows internally via chooseTypeBits(bits)
//  exactly as it already does everywhere else in this codebase.
//
//  nullptr (the default) means "use today's raw, zero-copy, fixed-width array" --
//  byte-identical to the pre-composition wire format and decode path, so leaving
//  both factories unset is a strict no-op change.
// =============================================================================
template <typename T>
struct RunLengthConfig {
    using RunStartsFactory = std::function<std::shared_ptr<ISectionCodecIntegral<uint64_t>>(uint8_t bits)>;
    using RunValuesFactory = std::function<std::shared_ptr<ISectionCodecIntegral<uint64_t>>(uint8_t bits)>;

    RunStartsFactory runStartsFactory = nullptr;
    RunValuesFactory runValuesFactory = nullptr;
};

/**
 * @brief Run-Length Encoding for integral types
 *
 * Compresses sequences of repeated values into (start_position, value) pairs.
 * Supports random access by binary searching run starts.
 *
 * Format: [num_runs (8 bytes),
 *          size_of_run_starts_payload_in_bytes (8 bytes),
 *          size_of_run_values_payload_in_bytes (8 bytes),
 *          run_starts_payload,
 *          run_values_payload]
 *
 * By default (no RunLengthConfig, or a default-constructed one) each payload is
 * the raw fixed-width array exactly as before: run_starts is
 * num_runs * sizeof(size_t) absolute positions, run_values is num_runs * sizeof(T)
 * values -- random access is O(1) amortised via interpolation + narrow binary
 * search fallback directly over zero-copy pointer views into the encoded buffer.
 *
 * When RunLengthConfig::runStartsFactory / runValuesFactory are set, the
 * corresponding payload is instead whatever that inner ISectionCodecIntegral
 * codec's own encode() produces (an opaque, self-contained blob) -- e.g. a
 * CascadingFOR-compressed runStarts array, since run-start positions are
 * monotonically increasing and the run-starts stream is sparse (R << N),
 * exactly the class of "aggregate-level stream" a sequential-style transform is
 * safe to apply to without sacrificing random access. Random access in this
 * case is preserved via a manual binary search that calls the inner codec's own
 * decodeAt() at each probed position (never materialising the full array) when
 * the inner codec reports RandomAccess, falling back to a one-time cached
 * decodeAll() of just that (still small, R-sized) array otherwise -- mirrors
 * FOREncoder::decodeAt's existing "delegate if RandomAccess, else full decode"
 * idiom.
 *
 * @tparam T The integral type to encode
 */
template<typename T>
    requires core::IntegralType<T>
class RunLengthEncoder : public Codec<T> {
public:
    RunLengthEncoder() = default;
    explicit RunLengthEncoder(RunLengthConfig<T> cfg) : cfg_(std::move(cfg)) {}

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

        std::vector<uint8_t> runStartsPayload;
        if (cfg_.runStartsFactory) {
            std::vector<uint64_t> u64Starts(runStarts.begin(), runStarts.end());
            auto codec = cfg_.runStartsFactory(bitsForPositions(data.size()));
            auto encoded = codec->encode(std::span<const uint64_t>(u64Starts));
            runStartsPayload = std::move(encoded.data());
        } else {
            runStartsPayload.resize(numRuns * sizeof(size_t));
            std::memcpy(runStartsPayload.data(), runStarts.data(), runStartsPayload.size());
        }

        std::vector<uint8_t> runValuesPayload;
        if (cfg_.runValuesFactory) {
            std::vector<uint64_t> u64Values(runValues.begin(), runValues.end());
            auto codec = cfg_.runValuesFactory(static_cast<uint8_t>(sizeof(T) * 8));
            auto encoded = codec->encode(std::span<const uint64_t>(u64Values));
            runValuesPayload = std::move(encoded.data());
        } else {
            runValuesPayload.resize(numRuns * sizeof(T));
            std::memcpy(runValuesPayload.data(), runValues.data(), runValuesPayload.size());
        }

        const size_t runStartsSize = runStartsPayload.size();
        const size_t runValuesSize = runValuesPayload.size();
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

        // Write run starts payload
        std::memcpy(writePtr, runStartsPayload.data(), runStartsSize);
        writePtr += runStartsSize;

        // Write run values payload
        std::memcpy(writePtr, runValuesPayload.data(), runValuesSize);

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

        // Invalidate both caches: new encoded data produced.
        cache_.base = nullptr;
        composedCache_.base = nullptr;
        composedCache_.view.reset();

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

        const size_t totalElements = encoded.metadata().elementCount;

        std::vector<size_t> runStarts(numRuns);
        if (cfg_.runStartsFactory) {
            auto codec = cfg_.runStartsFactory(bitsForPositions(totalElements));
            auto buf = spanToBuffer(std::span<const uint8_t>(readPtr, runStartsSize));
            auto vals = codec->decodeAll(buf);
            if (vals.size() != numRuns)
                throw std::runtime_error("RunLengthEncoder::decodeAll: runStarts codec size mismatch");
            for (size_t i = 0; i < numRuns; ++i) runStarts[i] = static_cast<size_t>(vals[i]);
        } else {
            std::memcpy(runStarts.data(), readPtr, runStartsSize);
        }
        readPtr += runStartsSize;

        std::vector<T> runValues(numRuns);
        if (cfg_.runValuesFactory) {
            auto codec = cfg_.runValuesFactory(static_cast<uint8_t>(sizeof(T) * 8));
            auto buf = spanToBuffer(std::span<const uint8_t>(readPtr, runValuesSize));
            auto vals = codec->decodeAll(buf);
            if (vals.size() != numRuns)
                throw std::runtime_error("RunLengthEncoder::decodeAll: runValues codec size mismatch");
            for (size_t i = 0; i < numRuns; ++i) runValues[i] = static_cast<T>(vals[i]);
        } else {
            std::memcpy(runValues.data(), readPtr, runValuesSize);
        }

        // Reconstruct original data
        // The last run extends to the end, which we get from metadata
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
        if (!isComposed()) {
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
            return;
        }

        const ComposedView& v = getComposedView(encoded);
        if (v.numRuns == 0) {
            if (n != 0) throw std::runtime_error("RunLengthEncoder::decodeAllInto: empty but n!=0");
            return;
        }
        for (size_t r = 0; r < v.numRuns; ++r) {
            const size_t runStart = v.getRunStart(r);
            const size_t runEnd   = (r + 1 < v.numRuns) ? v.getRunStart(r + 1) : n;
            std::fill(dst + runStart, dst + runEnd, v.getRunValue(r));
        }
    }

    void decodeRangeInto(const EncodedData& encoded, size_t start, size_t end,
                         T* dst, size_t n) override {
        if (!isComposed()) {
            const View v = getView(encoded);
            end = std::min(end, v.totalElements);
            if (start >= end) {
                if (n != 0) throw std::runtime_error("RunLengthEncoder::decodeRangeInto: empty range, n!=0");
                return;
            }
            if ((end - start) != n) [[unlikely]]
                throw std::runtime_error("RunLengthEncoder::decodeRangeInto: size mismatch");
            if (v.numRuns == 0) return;
            size_t runIdx = v.findRun(start);
            fillRunsRaw(v, start, end, dst, 0, runIdx);
            return;
        }

        const ComposedView& v = getComposedView(encoded);
        end = std::min(end, v.totalElements);
        if (start >= end) {
            if (n != 0) throw std::runtime_error("RunLengthEncoder::decodeRangeInto: empty range, n!=0");
            return;
        }
        if ((end - start) != n) [[unlikely]]
            throw std::runtime_error("RunLengthEncoder::decodeRangeInto: size mismatch");
        if (v.numRuns == 0) return;
        size_t runIdx = v.findRun(start);
        fillRunsComposed(v, start, end, dst, 0, runIdx);
    }

    // Gather (selective row-range) fast path: findRun() is called at most once
    // (to seed runIdx for the first range) instead of once per range -- every
    // subsequent range advances runIdx via fillRuns*'s bounded forward scan,
    // since ranges arrive ascending and non-overlapping.
    void decodeGatherInto(const EncodedData& encoded,
                          const RowRangeList& ranges,
                          T* dst, size_t n) override {
        if (ranges.empty()) {
            if (n != 0) throw std::runtime_error("RunLengthEncoder::decodeGatherInto: decoded size mismatch");
            return;
        }
        if (!isComposed()) {
            decodeGatherRaw(encoded, ranges, dst, n);
        } else {
            decodeGatherComposed(encoded, ranges, dst, n);
        }
    }

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (!isComposed()) {
            const View v = getView(encoded);
            if (v.numRuns == 0 || index >= v.totalElements) [[unlikely]] {
                return std::nullopt;
            }
            const size_t runIdx = v.findRun(index);
            return static_cast<T>(v.runValues[runIdx]);
        }

        const ComposedView& v = getComposedView(encoded);
        if (v.numRuns == 0 || index >= v.totalElements) [[unlikely]] {
            return std::nullopt;
        }
        const size_t runIdx = v.findRun(index);
        return v.getRunValue(runIdx);
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        const size_t totalElements = encoded.metadata().elementCount;
        if (start >= totalElements) return {};
        end = std::min(end, totalElements);

        if (!isComposed()) {
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

        const ComposedView& v = getComposedView(encoded);
        if (v.numRuns == 0) return {};

        const size_t firstRun = v.findRun(start);

        std::vector<T> result;
        result.reserve(end - start);

        for (size_t runIdx = firstRun; runIdx < v.numRuns; ++runIdx) {
            const size_t runStart = v.getRunStart(runIdx);
            const size_t runEnd = (runIdx + 1 < v.numRuns) ? v.getRunStart(runIdx + 1) : totalElements;

            if (runStart >= end) break;

            const size_t effectiveStart = std::max(runStart, start);
            const size_t effectiveEnd   = std::min(runEnd,   end);
            const T value = v.getRunValue(runIdx);

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
        // RandomAccess is preserved unconditionally, even when a configured
        // runStartsCodec itself lacks it: the fallback (a one-time cached decode
        // of the runStarts array) is O(R), not O(N), because runStarts is bounded
        // by the number of runs -- exactly the "sparse aggregate stream" a
        // sequential-style inner transform is safe to apply to.
        return EncodingProperties(EncodingProperty::RandomAccess)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::RunLengthBased
            | EncodingProperty::VariableSize
            | EncodingProperty::StreamingFriendly
            | EncodingProperty::LowMemoryOverhead
            | EncodingProperty::Composable
            | EncodingProperty::FastSkip;
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        // Worst case: no compression (every element is a new run)
        return 3 * sizeof(size_t) + elementCount * (sizeof(size_t) + sizeof(T));
    }

private:
    bool isComposed() const {
        return static_cast<bool>(cfg_.runStartsFactory) || static_cast<bool>(cfg_.runValuesFactory);
    }

    // Bits needed to represent any run-start VALUE, which ranges over [0, N)
    // (N = total element count), not [0, numRuns) -- only the run-start INDEX is
    // bounded by numRuns.
    static uint8_t bitsForPositions(size_t totalElements) {
        if (totalElements <= 1) return 1;
        return static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(totalElements - 1)));
    }

    static EncodedBuffer<uint8_t> spanToBuffer(std::span<const uint8_t> bytes) {
        std::vector<uint8_t> copy(bytes.begin(), bytes.end());
        return EncodedBuffer<uint8_t>(std::move(copy), encodings::EncodingMetadata{});
    }

    // ---------------------------------------------------------------------------
    // Zero-copy view into the encoded buffer -- used only when neither
    // runStartsFactory nor runValuesFactory is configured (the legacy, raw,
    // byte-identical-to-before path). Completely unchanged from before this
    // file gained composability.
    //
    // Pointers alias directly into EncodedData::data() -- no allocation, no copy.
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
    // Composed view -- used when either runStartsFactory or runValuesFactory is
    // configured. Whichever stream is NOT composed still accesses its array via
    // a raw pointer directly into the encoded buffer (same zero-copy idea as
    // View above); whichever stream IS composed goes through its
    // ISectionCodecIntegral instance, with a plain binary-search-via-decodeAt
    // (never materialising the full array) when that codec reports RandomAccess,
    // or a one-time cached decodeAll() of just that (small, R-sized) array
    // otherwise.
    // ---------------------------------------------------------------------------
    struct ComposedView {
        size_t numRuns{0};
        size_t totalElements{0};

        // runStarts access
        const size_t* rawRunStarts{nullptr};
        std::shared_ptr<ISectionCodecIntegral<uint64_t>> runStartsCodec;
        EncodedBuffer<uint8_t> runStartsBuf;
        mutable std::vector<uint64_t> cachedRunStartsAll;
        mutable bool cachedRunStartsAllReady{false};

        // runValues access
        const T* rawRunValues{nullptr};
        std::shared_ptr<ISectionCodecIntegral<uint64_t>> runValuesCodec;
        EncodedBuffer<uint8_t> runValuesBuf;

        size_t getRunStart(size_t runIdx) const {
            if (!runStartsCodec) return rawRunStarts[runIdx];
            if (!runStartsCodec->properties().has(EncodingProperty::RandomAccess)) {
                ensureCachedRunStarts();
                return static_cast<size_t>(cachedRunStartsAll[runIdx]);
            }
            auto v = runStartsCodec->decodeAt(runStartsBuf, runIdx);
            if (!v) throw std::runtime_error("RunLengthEncoder: runStartsCodec decodeAt returned nullopt for valid index");
            return static_cast<size_t>(*v);
        }

        T getRunValue(size_t runIdx) const {
            if (!runValuesCodec) return rawRunValues[runIdx];
            auto v = runValuesCodec->decodeAt(runValuesBuf, runIdx);
            if (!v) throw std::runtime_error("RunLengthEncoder: runValuesCodec decodeAt returned nullopt for valid index");
            return static_cast<T>(*v);
        }

        void ensureCachedRunStarts() const {
            if (cachedRunStartsAllReady) return;
            cachedRunStartsAll = runStartsCodec->decodeAll(runStartsBuf);
            cachedRunStartsAllReady = true;
        }

        // Finds the largest run index k such that getRunStart(k) <= targetIdx.
        // A plain binary search calling getRunStart() at each probe: O(1) per
        // probe whether backed by a raw pointer or by a RandomAccess-capable
        // inner codec's decodeAt(), so O(log numRuns) total -- never
        // materialises the full runStarts array unless the inner codec itself
        // lacks RandomAccess (handled by getRunStart's own cached fallback).
        size_t findRun(size_t targetIdx) const {
            if (numRuns == 1) return 0;
            size_t lo = 0, hi = numRuns;  // invariant: answer in [lo, hi)
            while (lo + 1 < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (getRunStart(mid) <= targetIdx) lo = mid; else hi = mid;
            }
            return lo;
        }
    };

    struct ComposedCache {
        const uint8_t* base{nullptr};
        std::unique_ptr<ComposedView> view;
    };
    mutable ComposedCache composedCache_;

    const ComposedView& getComposedView(const EncodedData& encoded) const {
        const uint8_t* base = encoded.data().data();
        if (composedCache_.base == base && composedCache_.view) {
            return *composedCache_.view;
        }

        auto view = std::make_unique<ComposedView>();

        if (encoded.size() >= 3 * sizeof(size_t)) {
            const uint8_t* p = base;
            size_t numRuns, runStartsSize, runValuesSize;
            std::memcpy(&numRuns, p, sizeof(size_t)); p += sizeof(size_t);
            std::memcpy(&runStartsSize, p, sizeof(size_t)); p += sizeof(size_t);
            std::memcpy(&runValuesSize, p, sizeof(size_t)); p += sizeof(size_t);

            view->numRuns = numRuns;
            view->totalElements = encoded.metadata().elementCount;

            const uint8_t* runStartsPtr = p;
            const uint8_t* runValuesPtr = p + runStartsSize;

            if (numRuns > 0) {
                if (cfg_.runStartsFactory) {
                    view->runStartsCodec = cfg_.runStartsFactory(bitsForPositions(view->totalElements));
                    view->runStartsBuf = spanToBuffer(std::span<const uint8_t>(runStartsPtr, runStartsSize));
                } else {
                    view->rawRunStarts = reinterpret_cast<const size_t*>(runStartsPtr);
                }

                if (cfg_.runValuesFactory) {
                    view->runValuesCodec = cfg_.runValuesFactory(static_cast<uint8_t>(sizeof(T) * 8));
                    view->runValuesBuf = spanToBuffer(std::span<const uint8_t>(runValuesPtr, runValuesSize));
                } else {
                    view->rawRunValues = reinterpret_cast<const T*>(runValuesPtr);
                }
            }
        }

        composedCache_.view = std::move(view);
        composedCache_.base = base;
        return *composedCache_.view;
    }

    // ---------------------------------------------------------------------------
    // Per-encoder metadata cache (legacy/raw path only).
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

    // Fills dst[dstOff .. dstOff+(end-start)) with the run-length-decoded values
    // for element indices [start, end), resuming the run search from *runIdx
    // (updated in place) instead of restarting from a fresh findRun() call.
    // Shared by decodeRangeInto and decodeGatherInto's per-range fill step.
    static void fillRunsRaw(const View& v, size_t start, size_t end,
                            T* dst, size_t dstOff, size_t& runIdx) {
        while (runIdx + 1 < v.numRuns && v.runStarts[runIdx + 1] <= start) ++runIdx;
        for (size_t r = runIdx; r < v.numRuns; ++r) {
            const size_t rs = v.runStarts[r];
            const size_t re = (r + 1 < v.numRuns) ? v.runStarts[r + 1] : v.totalElements;
            if (rs >= end) break;
            std::fill(dst + dstOff + (std::max(rs, start) - start),
                      dst + dstOff + (std::min(re, end)   - start),
                      v.runValues[r]);
            runIdx = r;
        }
    }

    static void fillRunsComposed(const ComposedView& v, size_t start, size_t end,
                                 T* dst, size_t dstOff, size_t& runIdx) {
        while (runIdx + 1 < v.numRuns && v.getRunStart(runIdx + 1) <= start) ++runIdx;
        for (size_t r = runIdx; r < v.numRuns; ++r) {
            const size_t rs = v.getRunStart(r);
            const size_t re = (r + 1 < v.numRuns) ? v.getRunStart(r + 1) : v.totalElements;
            if (rs >= end) break;
            std::fill(dst + dstOff + (std::max(rs, start) - start),
                      dst + dstOff + (std::min(re, end)   - start),
                      v.getRunValue(r));
            runIdx = r;
        }
    }

    void decodeGatherRaw(const EncodedData& encoded, const RowRangeList& ranges,
                        T* dst, size_t n) {
        const View v = getView(encoded);
        size_t off = 0;
        size_t runIdx = 0;
        if (v.numRuns > 0) {
            const size_t seedIdx = std::min(ranges.front().begin,
                v.totalElements > 0 ? v.totalElements - 1 : size_t{0});
            runIdx = v.findRun(seedIdx);
        }
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            const size_t end = std::min(r.end, v.totalElements);
            if (r.begin >= end) continue;
            if (v.numRuns > 0) fillRunsRaw(v, r.begin, end, dst, off, runIdx);
            off += (end - r.begin);
        }
        if (off != n) throw std::runtime_error("RunLengthEncoder::decodeGatherInto: decoded size mismatch");
    }

    void decodeGatherComposed(const EncodedData& encoded, const RowRangeList& ranges,
                             T* dst, size_t n) {
        const ComposedView& v = getComposedView(encoded);
        size_t off = 0;
        size_t runIdx = 0;
        if (v.numRuns > 0) {
            const size_t seedIdx = std::min(ranges.front().begin,
                v.totalElements > 0 ? v.totalElements - 1 : size_t{0});
            runIdx = v.findRun(seedIdx);
        }
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            const size_t end = std::min(r.end, v.totalElements);
            if (r.begin >= end) continue;
            if (v.numRuns > 0) fillRunsComposed(v, r.begin, end, dst, off, runIdx);
            off += (end - r.begin);
        }
        if (off != n) throw std::runtime_error("RunLengthEncoder::decodeGatherInto: decoded size mismatch");
    }

    RunLengthConfig<T> cfg_{};

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
