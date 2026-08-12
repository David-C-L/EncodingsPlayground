#pragma once

// Materialises each (dataset, N) at most once per process.
//
// Drivers sweep encoders inside datasets, so a naive loop regenerates the same
// stream once per encoder.  For a parquet-backed source that is a re-read; for a
// seeded generator it is worse than wasteful, because the generators seed from
// std::random_device on reset() and a regenerated stream is NOT the same data —
// two encoders would then be compared on different inputs.

#include <cstddef>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/registry/DatasetRegistry.hpp"

namespace encodings::benchmark {

/// Owns the generated buffers; hands out non-owning views.
///
/// SIZE: an int64 stream at N = 10M is 80 MB, and the cache never evicts on its
/// own.  A driver that sweeps N should hold one N at a time and call
/// releaseAll() between them rather than accumulating every point of the axis.
template <typename T>
class DatasetCache {
public:
    struct Handle {
        std::string name;
        size_t n{};
        std::span<const T> data;
    };

    /// The returned span stays valid until releaseAll() — buffers live in a
    /// node-based map, so materialising further datasets never reallocates or
    /// invalidates an earlier one.
    Handle materialize(const DatasetEntry<T>& entry, size_t n) {
        if (n == 0) throw std::invalid_argument("DatasetCache::materialize: n must be > 0");

        const Key key{entry.name, n};
        auto it = buffers_.find(key);
        if (it == buffers_.end()) {
            if (!entry.generator) {
                throw std::invalid_argument("DatasetCache::materialize: dataset '" + entry.name +
                                            "' has no generator");
            }
            // reset() before generate() so a file-backed source rewinds to row 0
            // and every (dataset, N) starts from the same place in the column.
            entry.generator->reset();
            std::vector<T> data = entry.generator->generate(n);
            if (data.size() != n) {
                throw std::runtime_error("DatasetCache::materialize: dataset '" + entry.name +
                                         "' produced " + std::to_string(data.size()) +
                                         " of " + std::to_string(n) + " requested elements");
            }
            it = buffers_.emplace(key, std::move(data)).first;
        }

        return Handle{entry.name, n, std::span<const T>(it->second.data(), it->second.size())};
    }

    /// Frees every buffer.  All outstanding Handles dangle afterwards; call it
    /// only between sweep phases, never while a Handle is in use.
    void releaseAll() { buffers_.clear(); }

    size_t residentBuffers() const { return buffers_.size(); }

    size_t residentBytes() const {
        size_t total = 0;
        for (const auto& [key, buf] : buffers_) total += buf.size() * sizeof(T);
        return total;
    }

private:
    using Key = std::pair<std::string, size_t>;
    std::map<Key, std::vector<T>> buffers_;
};

}  // namespace encodings::benchmark
