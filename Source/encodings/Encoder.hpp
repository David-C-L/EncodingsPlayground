#pragma once

#include <cstddef>
#include <span>
#include <vector>
#include <memory>
#include <string>
#include <concepts>
#include "core/DataType.hpp"
#include "EncodedData.hpp"
#include "EncodingProperty.hpp"
#include "EncodingType.hpp"
#include "RowRange.hpp"

namespace encodings {

    using core::DataType;
    using core::PrimitiveType;
    using core::MapType;
    using core::typeToDataType;

/**
 * @brief Abstract base class for encoding strategies
 * 
 * @tparam TIn The type of data to encode (e.g., int32_t, std::string)
 * @tparam TOut The type of data produced by the encoder (default is uint8_t for byte-oriented encodings)
 */
template<typename TIn, typename TOut = uint8_t>
class Encoder {
public:
    virtual ~Encoder() = default;
    
    /**
     * @brief Encode a span of data elements
     * 
     * @param data Input data to encode (span is a lightweight view, no copy of data occurs)
     * @return EncodedData containing compressed bytes and metadata
     * 
     * @note std::span is passed by value following C++ standard library conventions.
     *       The underlying data is not copied; span is merely a view (pointer + size).
     */
    virtual EncodedBuffer<TOut> encode(std::span<const TIn> data) = 0;
    
    /**
     * @brief Get the encoding type of this encoding scheme
     */
    virtual EncodingType encodingType() const = 0;

    /**
     * @brief Get the name of this encoding scheme
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get the properties of this encoding scheme
     * 
     * This provides a rich set of metadata about the encoding's capabilities
     * and characteristics. Prefer using this over individual boolean methods.
     */
    virtual EncodingProperties properties() const = 0;
    
    /**
     * @brief Get the data type this encoder handles
     */
    virtual DataType dataType() const {
        if constexpr (PrimitiveType<TIn>) {
            return typeToDataType<TIn>;
        } else if constexpr (core::MapType<TIn>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }

    /**
     * @brief Get the data type this encoder produces
     */
    virtual DataType encodedType() const {
        if constexpr (PrimitiveType<TOut>) {
            return typeToDataType<TOut>;
        } else if constexpr (core::MapType<TOut>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }
    
    /**
     * @brief Estimate the encoded size without actually encoding
     * Useful for pre-allocation
     * 
     * @param elementCount Number of elements to encode
     * @return Estimated size in bytes (0 if cannot estimate)
     */
    virtual size_t estimateEncodedSize(size_t /* elementCount */) const {
        return 0; // Default: cannot estimate
    }
};

/**
 * @brief Abstract base class for decoding strategies
 * 
 * @tparam T The type of data to decode
 */
template<typename TIn, typename TOut = uint8_t>
class Decoder {
public:
    virtual ~Decoder() = default;
    
    /**
     * @brief Decode all data at once (bulk read)
     * 
     * @param encoded The encoded data to decode
     * @return Vector containing all decoded elements
     */
    virtual std::vector<TIn> decodeAll(const EncodedBuffer<TOut>& encoded) = 0;
    
    /**
     * @brief Decode a single element at a specific index (random access)
     * 
     * @param encoded The encoded data
     * @param index Index of the element to decode
     * @return The decoded element, or std::nullopt if index is out of bounds
     */
    virtual std::optional<TIn> decodeAt(const EncodedBuffer<TOut>& encoded, size_t index) = 0;
    
    /**
     * @brief Decode a range of elements (batch random access)
     * 
     * @param encoded The encoded data
     * @param start Starting index (inclusive)
     * @param end Ending index (exclusive)
     * @return Vector containing decoded elements in the range
     */
    virtual std::vector<TIn> decodeRange(const EncodedBuffer<TOut>& encoded, size_t start, size_t end) = 0;

    /**
     * @brief Decode all elements directly into a caller-supplied buffer.
     *
     * Avoids the heap allocation and zero-initialisation that decodeAll() would
     * perform for its return vector.  The default implementation calls decodeAll()
     * and copies; override in hot-path codecs to write directly into dst.
     *
     * @param encoded  The encoded data to decode
     * @param dst      Pre-allocated output buffer of at least n elements
     * @param n        Expected element count (must equal encoded element count)
     */
    virtual void decodeAllInto(const EncodedBuffer<TOut>& encoded, TIn* dst, size_t n) {
        auto vals = decodeAll(encoded);
        if (vals.size() != n) [[unlikely]]
            throw std::runtime_error("Codec::decodeAllInto: decoded size mismatch");
        std::copy(vals.begin(), vals.end(), dst);
    }

    /**
     * @brief Decode a range of elements directly into a caller-supplied buffer.
     *
     * Default: calls decodeRange and copies.  Override in hot-path codecs to
     * write directly into dst without any heap allocation.
     *
     * @param encoded  The encoded data
     * @param start    Starting index (inclusive)
     * @param end      Ending index (exclusive)
     * @param dst      Pre-allocated output buffer of at least n elements
     * @param n        Expected element count (must equal end-start after clamping)
     */
    virtual void decodeRangeInto(const EncodedBuffer<TOut>& encoded,
                                  size_t start, size_t end,
                                  TIn* dst, size_t n) {
        auto vals = decodeRange(encoded, start, end);
        if (vals.size() != n) [[unlikely]]
            throw std::runtime_error("Codec::decodeRangeInto: decoded size mismatch");
        std::copy(vals.begin(), vals.end(), dst);
    }

    /**
     * @brief Decode a gather-style ordered list of row ranges ("selective read").
     *
     * Models a TableScan-style selective read: an ascending, non-overlapping
     * list of surviving contiguous row ranges, with gaps between them that
     * are skipped rather than materialized. Writes all selected values
     * contiguously into dst, in range order.
     *
     * @param encoded  The encoded data
     * @param ranges   Ascending, non-overlapping row ranges to decode
     * @param dst      Pre-allocated output buffer of at least n elements
     * @param n        Expected total element count (sum of range sizes)
     *
     * Default: falls back to one decodeRangeInto() call per range, which is
     * exactly today's benchmarkRangeAccess behaviour applied range-by-range —
     * every existing codec supports this immediately with zero changes.
     * Override in stateful/sequential codecs to implement a genuine
     * skip-then-materialize fast path.
     */
    virtual void decodeGatherInto(const EncodedBuffer<TOut>& encoded,
                                   const RowRangeList& ranges,
                                   TIn* dst, size_t n) {
        size_t off = 0;
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            decodeRangeInto(encoded, r.begin, r.end, dst + off, count);
            off += count;
        }
        if (off != n) [[unlikely]]
            throw std::runtime_error("Decoder::decodeGatherInto: decoded size mismatch");
    }

    /**
     * @brief Get the encoding type of this decoding scheme (should match encoder)
     */
    virtual EncodingType encodingType() const = 0;

    /**
     * @brief Get the name of this decoding scheme (should match encoder)
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Get the properties of this decoding scheme
     * Should match the encoder's properties
     */
    virtual EncodingProperties properties() const = 0;
    
    /**
     * @brief Get the data type this decoder handles
     */
    virtual DataType dataType() const {
        if constexpr (PrimitiveType<TIn>) {
            return typeToDataType<TIn>;
        } else if constexpr (core::MapType<TIn>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }

    /**
     * @brief Decoder-owned memory that is not part of the encoded buffer.
     *
     * Exists for the benchmark harness's cold-all cache state.  Evicting the
     * encoded payload cools only what the driver can address; a decoder that
     * keeps a *decoded* auxiliary structure alive across calls — a positional
     * index (FPE per-tier bitmaps, tag array, Elias-Fano positions), a
     * dictionary, a rank sample table, a parsed header — leaves that structure
     * hot no matter how thoroughly the payload was flushed.  For the
     * reordering-index work that index *is* the object of study, so a cold-all
     * row that silently kept it in L2 measures the wrong thing.
     *
     * Returning {} is correct and is the default: the harness then falls back to
     * an LLC thrash, which is coarser (it also disturbs the TLB and the branch
     * predictors) but never overstates coldness.  Buffers returned must stay
     * valid and unowned-by-the-caller for the duration of the decode they
     * precede — typically members of the decoder that a prior decode populated,
     * so a caller should invoke this after at least one decode.
     */
    virtual std::vector<std::span<const std::byte>> internalBuffers() const { return {}; }

    /**
     * @brief Get the data type this decoder expects as input (encoded type)
     */
    virtual DataType encodedType() const {
        if constexpr (PrimitiveType<TOut>) {
            return typeToDataType<TOut>;
        } else if constexpr (core::MapType<TOut>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }
};

/**
 * @brief Combined encoder/decoder interface for convenience
 * 
 * Many encoding schemes naturally implement both encoding and decoding,
 * so this class provides a unified interface.
 * 
 * @tparam T The type of data to encode/decode
 */
template<typename TIn, typename TOut = uint8_t>
class Codec : public Encoder<TIn, TOut>, public Decoder<TIn, TOut> {
public:
    virtual ~Codec() = default;
    
    // Inherit from both Encoder and Decoder
    using Encoder<TIn, TOut>::encode;
    using Decoder<TIn, TOut>::decodeAll;
    using Decoder<TIn, TOut>::decodeAt;
    using Decoder<TIn, TOut>::decodeRange;

    // Per-sub-stream decode profiling — overridden by SubIntSplitEncoder<T, true> only.
    // Default implementations return empty / no-op so all other codecs are unaffected.

    /// Per-section decode times (ns) from the most recent decodeAll() call.
    /// Index matches subStreamEncodeMetrics order in the encoded buffer's metadata.
    virtual std::vector<int64_t> subStreamBulkDecodeTimeNs()   const { return {}; }

    /// Per-section accumulated ns across decodeAt() calls since the last reset.
    virtual std::vector<int64_t> subStreamDecodeAtAccumNs()    const { return {}; }

    /// Per-section accumulated ns across decodeRange() calls since the last reset.
    virtual std::vector<int64_t> subStreamDecodeRangeAccumNs() const { return {}; }

    /// Zero the decodeAt accumulator — call before each random-access benchmark loop.
    virtual void resetSubStreamDecodeAtAccum()    {}

    /// Zero the decodeRange accumulator — call before each range-access benchmark loop.
    virtual void resetSubStreamDecodeRangeAccum() {}

    /// Reset any cached encoder-selection state so the next encode() re-runs
    /// selection on fresh data.  Called by BenchmarkRunner between datasets.
    virtual void reset() {}

    // Reordering-layer profiling — overridden by ReorderingCodec<T, true> only.
    // Default: return -1 (unavailable) / no-op, so all non-reordering codecs are unaffected.

    /// Nanoseconds spent in Reorderer::reorder() during the most recent encode() call.
    virtual int64_t reorderEncodeTimeNs()            const { return -1; }

    /// Nanoseconds spent in Reorderer::unreorder() during the most recent decodeAll() call.
    virtual int64_t unreorderDecodeAllTimeNs()        const { return -1; }

    /// Accumulated ns for permutation-lookup steps across all decodeAt() calls since last reset.
    virtual int64_t permLookupDecodeAtAccumNs()       const { return -1; }

    /// Accumulated ns for permutation-lookup steps across all decodeRange() calls since last reset.
    virtual int64_t permLookupDecodeRangeAccumNs()    const { return -1; }

    /// Zero the reordering-layer accumulators before each benchmark loop.
    virtual void    resetReorderingProfilingAccum()   {}

    // Gather (selective row-range) profiling — overridden by codecs that
    // implement a genuine skip-then-materialize fast path in decodeGatherInto().
    // Default: -1 = "not measured / codec has no distinct skip phase" (e.g. it
    // falls back to independent decodeRangeInto calls with no skip concept).

    /// Nanoseconds spent skipping (advancing past gap rows without materializing)
    /// during the most recent decodeGatherInto() call, accumulated over all gaps.
    virtual int64_t gatherSkipTimeNs() const { return -1; }

    /// Nanoseconds spent materializing the surviving contiguous runs during the
    /// most recent decodeGatherInto() call.
    virtual int64_t gatherMaterializeTimeNs() const { return -1; }

    /// Zero the gather-phase accumulators — call before each selective-access
    /// benchmark loop.
    virtual void resetGatherProfilingAccum() {}

    // Single encodingType() implementation for both interfaces
    EncodingType encodingType() const override = 0;

    // Single name() implementation for both interfaces
    std::string name() const override = 0;
    
    // Single properties() implementation for both interfaces
    EncodingProperties properties() const override = 0;
    
    // Single dataType() implementation for both interfaces
    DataType dataType() const override {
        if constexpr (PrimitiveType<TIn>) {
            return typeToDataType<TIn>;
        } else if constexpr (core::MapType<TIn>) {
            return DataType::Map;
        } else if constexpr (core::Vector32Type<TIn>) {
            return DataType::Vector32;
        } else {
            return DataType::Array; // Default for composite types
        }
    }

    // Single encodedType() implementation for both interfaces
    DataType encodedType() const override {
        if constexpr (PrimitiveType<TOut>) {
            return typeToDataType<TOut>;
        } else if constexpr (core::MapType<TOut>) {
            return DataType::Map;
        } else if constexpr (core::Vector32Type<TOut>) {
            return DataType::Vector32;
        } else {
            return DataType::Array; // Default for composite types
        }
    }
};

} // namespace encodings
