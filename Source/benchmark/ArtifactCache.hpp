#pragma once

// Encode once per (encoder, dataset, N), and carry everything the encode
// produced on the artifact.
//
// This is the core of the driver refactor (CONVENTIONS section 2).  Two things
// make it a correctness feature rather than an optimisation:
//
//   * For an AutoSIS codec, every encode() re-runs cost-model DP selection.  A
//     driver that re-encodes per iteration is not measuring one plan repeatedly,
//     it is measuring whatever plan the DP picked that time.
//   * Per-section byte counts, selection time and custom metrics are produced by
//     the encode that made the payload.  Re-encoding to "recover" them yields
//     numbers from a different encode than the one being reported.
//
// So a decode driver asks for EncodeMeasurement::None and gets exactly one
// encode() call; encodeCallCount() lets a test assert that.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef __linux__
#include <malloc.h>
#endif

#include "benchmark/AllocationTracker.hpp"
#include "benchmark/DatasetCache.hpp"
#include "benchmark/MemoryTracker.hpp"
#include "benchmark/TimingStats.hpp"
#include "benchmark/registry/EncoderRegistry.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"

namespace encodings::benchmark {

/// How much the cache is allowed to spend producing the artifact.
///
/// Timing and heap tracking are opt-in per call because they cost extra
/// encode() calls, which only `bench_encode` wants: it is the one driver whose
/// subject is the encode itself, run in its own process with nothing else going
/// on.  Every other driver takes None.
enum class EncodeMeasurement {
    None,              ///< exactly one encode()
    Timed,             ///< warmup + iterations encodes, summarised
    TimedAndPeakHeap,  ///< Timed, plus one instrumented encode for heap figures
};

/// One encoded payload plus everything observed while producing it.
template <typename T>
struct EncodedArtifact {
    std::string encoderName, datasetName;
    size_t elementCount{}, payloadBytes{}, originalBytes{};
    double compressionRatio{};  ///< originalBytes / payloadBytes, i.e. "5.0x"
    encodings::EncodedData encoded;
    std::shared_ptr<encodings::Codec<T>> codec;
    bool fastSkip{}, randomAccess{};

    /// Populated only for Timed / TimedAndPeakHeap; nullopt means "not measured",
    /// which is distinct from "measured as zero" and must reach the result file
    /// as a null rather than a 0.
    std::optional<TimingSummary> encodeTimeNs;
    std::optional<size_t> encodePeakHeapBytes, encodeNetHeapDeltaBytes;

    /// -1 when the codec reported no selectionTime_ns (every codec except an
    /// AutoSIS built with enableSelectionTiming).
    int64_t selectionTimeNs{-1};

    /// Numeric entries of metadata().customMetadata, parsed once here so drivers
    /// do not each re-parse the string map.  Non-numeric entries are dropped.
    std::map<std::string, double> encodeCustomMetrics;
};

/// Bounded cache of encoded payloads.
template <typename T>
class ArtifactCache {
public:
    struct Options {
        /// Payloads are large (a weakly-compressing codec at N = 10M holds tens
        /// of MB, and the codec keeps its own internal structures alive), so the
        /// default keeps exactly one.  Raise it only for a driver that
        /// interleaves encoders within one dataset.
        size_t maxResident{1};
    };

    explicit ArtifactCache(Options options = {}) : options_(options) {
        if (options_.maxResident == 0) options_.maxResident = 1;
    }

    /// Returns the artifact for (encoder, dataset, N), encoding it if needed.
    ///
    /// The reference is valid until this entry is evicted — which the very next
    /// get() will do at the default maxResident of 1.  Consume it (or copy what
    /// you need) before asking for another artifact; do not hold two.
    ///
    /// `enc` is taken by non-const reference because encoding mutates the codec
    /// (reset(), internal selection state) and the same entry is reused across
    /// datasets.
    const EncodedArtifact<T>& get(EncoderEntry<T>& enc,
                                  const typename DatasetCache<T>::Handle& dataset,
                                  EncodeMeasurement measurement = EncodeMeasurement::None,
                                  size_t iterations = 5,
                                  size_t warmup = 2) {
        if (!enc.codec) {
            throw std::invalid_argument("ArtifactCache::get: encoder '" + enc.name +
                                        "' has no codec");
        }

        const Key key{enc.name, dataset.name, dataset.n};
        auto found = index_.find(key);
        if (found != index_.end()) {
            // A cached artifact produced under None carries no timings, so a
            // later Timed request has to encode again rather than silently
            // returning an artifact that is missing the very thing being asked
            // for.  Requests that need no more than the cache holds are hits.
            if (satisfies(found->second->artifact, measurement)) {
                entries_.splice(entries_.begin(), entries_, found->second);
                return found->second->artifact;
            }
            entries_.erase(found->second);
            index_.erase(found);
        }

        EncodedArtifact<T> artifact = encode(enc, dataset, measurement, iterations, warmup);

        entries_.push_front(Entry{key, std::move(artifact)});
        index_[key] = entries_.begin();
        trim();
        return entries_.front().artifact;
    }

    /// Drops every artifact for one encoder — used when a driver finishes an
    /// encoder and wants its payload and codec state gone before timing the
    /// next one, so cache-state measurements are not polluted.
    void evict(const std::string& encoderName) {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->key.encoderName == encoderName) {
                index_.erase(it->key);
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void clear() {
        index_.clear();
        entries_.clear();
    }

    /// Total encode() calls issued by this cache.  Exists so a test can prove
    /// encode-once: two None gets for the same key must leave this at 1.
    size_t encodeCallCount() const { return encodeCalls_; }

    size_t residentArtifacts() const { return entries_.size(); }

private:
    struct Key {
        std::string encoderName, datasetName;
        size_t n{};
        bool operator<(const Key& o) const {
            return std::tie(encoderName, datasetName, n) <
                   std::tie(o.encoderName, o.datasetName, o.n);
        }
    };

    struct Entry {
        Key key;
        EncodedArtifact<T> artifact;
    };

    static bool satisfies(const EncodedArtifact<T>& a, EncodeMeasurement m) {
        switch (m) {
            case EncodeMeasurement::None:
                return true;
            case EncodeMeasurement::Timed:
                return a.encodeTimeNs.has_value();
            case EncodeMeasurement::TimedAndPeakHeap:
                return a.encodeTimeNs.has_value() && a.encodePeakHeapBytes.has_value();
        }
        return false;
    }

    encodings::EncodedData runEncode(EncoderEntry<T>& enc,
                                     const typename DatasetCache<T>::Handle& dataset) {
        ++encodeCalls_;
        return enc.codec->encode(dataset.data);
    }

    EncodedArtifact<T> encode(EncoderEntry<T>& enc,
                              const typename DatasetCache<T>::Handle& dataset,
                              EncodeMeasurement measurement,
                              size_t iterations,
                              size_t warmup) {
        EncodedArtifact<T> a;
        a.encoderName = enc.name;
        a.datasetName = dataset.name;
        a.elementCount = dataset.n;
        a.originalBytes = dataset.n * sizeof(T);
        a.codec = enc.codec;

        // reset() clears cached selection state so this encode sees the dataset
        // fresh, rather than reusing a plan chosen for the previous one.
        enc.codec->reset();

        if (measurement == EncodeMeasurement::None) {
            a.encoded = runEncode(enc, dataset);
        } else {
            for (size_t i = 0; i < warmup; ++i) {
                encodings::EncodedData scratch = runEncode(enc, dataset);
                consumePayload(scratch);
            }

            const size_t timed = iterations == 0 ? 1 : iterations;
            std::vector<int64_t> samples;
            samples.reserve(timed);
            for (size_t i = 0; i < timed; ++i) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                encodings::EncodedData out = runEncode(enc, dataset);
                const auto t1 = std::chrono::high_resolution_clock::now();
                samples.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                consumePayload(out);
                a.encoded = std::move(out);
            }
            a.encodeTimeNs = summarize(samples);

            if (measurement == EncodeMeasurement::TimedAndPeakHeap) {
                // Heap figures come from their own encode, outside the timed
                // loop: ScopedAllocationTrack puts a CAS on every operator new,
                // and mallinfo2 takes the allocator lock, so an instrumented
                // iteration is not a valid timing sample.
                //
                // malloc_trim first so the baseline is the allocator's floor
                // rather than free blocks the timed loop happened to leave
                // behind, which would otherwise absorb this encode's peak.
#ifdef __linux__
                malloc_trim(0);
#endif
                const size_t heapBefore = currentHeapBytes();
                size_t peak = 0;
                {
                    ScopedAllocationTrack track;
                    encodings::EncodedData out = runEncode(enc, dataset);
                    peak = track.stop();
                    consumePayload(out);
                    a.encoded = std::move(out);
                }
                const size_t heapAfter = currentHeapBytes();
                a.encodePeakHeapBytes = peak;
                a.encodeNetHeapDeltaBytes = heapAfter > heapBefore ? heapAfter - heapBefore : 0;
            }
        }

        const auto& meta = a.encoded.metadata();
        // compressedSize is what the codec reports it wrote; encoded.size() is
        // the buffer it handed back.  They differ for codecs that over-reserve,
        // and the reported figure is the one that would go to disk.
        a.payloadBytes = meta.compressedSize != 0 ? meta.compressedSize : a.encoded.size();
        a.compressionRatio = a.payloadBytes != 0
            ? static_cast<double>(a.originalBytes) / static_cast<double>(a.payloadBytes)
            : 0.0;

        const auto props = enc.codec->properties();
        a.fastSkip = props.has(encodings::EncodingProperty::FastSkip);
        a.randomAccess = props.has(encodings::EncodingProperty::RandomAccess);

        for (const auto& [k, v] : meta.customMetadata) {
            if (k == "selectionTime_ns") {
                if (const auto parsed = parseNumber(v)) {
                    a.selectionTimeNs = static_cast<int64_t>(*parsed);
                }
                continue;
            }
            if (const auto parsed = parseNumber(v)) a.encodeCustomMetrics[k] = *parsed;
        }
        return a;
    }

    /// Whole-string numeric parse: a partially numeric value ("64 blocks") is
    /// not a metric and must be dropped rather than silently recorded as 64.
    static std::optional<double> parseNumber(const std::string& text) {
        if (text.empty()) return std::nullopt;
        char* end = nullptr;
        const double value = std::strtod(text.c_str(), &end);
        if (end != text.c_str() + text.size()) return std::nullopt;
        return value;
    }

    /// Keeps the optimizer from eliding an encode whose result is discarded.
    ///
    /// Same empty-asm trick as MeasureLoop.hpp's clobber(), kept local rather
    /// than included: MeasureLoop pulls in CachePolicy.hpp (clflush, LLC
    /// topology), which an encode-only driver has no use for.
    static void consumePayload(const encodings::EncodedData& data) {
        const void* p = data.data().data();
        asm volatile("" : : "r"(p) : "memory");
    }

    void trim() {
        while (entries_.size() > options_.maxResident) {
            index_.erase(entries_.back().key);
            entries_.pop_back();
        }
    }

    Options options_;
    std::list<Entry> entries_;  ///< front = most recently used
    std::map<Key, typename std::list<Entry>::iterator> index_;
    size_t encodeCalls_{0};
};

}  // namespace encodings::benchmark
