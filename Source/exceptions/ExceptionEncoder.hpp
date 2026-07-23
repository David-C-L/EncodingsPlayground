#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "core/DataType.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "exceptions/BitmapRank.hpp"
#include "exceptions/ExceptionDetector.hpp"
#include "exceptions/ExceptionMetricUtils.hpp"

namespace encodings::exceptions {

// ---------------------------------------------------------------------------
// ExceptionEncoder<T>
//
// Generic Codec<T,uint8_t> decorator: wraps an ExceptionDetector<T> (the
// per-encoding-family "what counts as non-conforming" metric) and any inner
// Codec<T,uint8_t>. Structurally parallel to BWTSectionEncoder<T,W> /
// ReorderingCodec<T> -- see Axis 2 of the exception-handling design doc
// (~/.claude/plans/could-you-analyse-the-misty-widget.md).
//
// Encode path:
//   1. metric = detector_->computeMetric(data)          -- real full data
//   2. bound  = ExceptionMetricUtils::sweepMinCostBound(metric, ...)
//   3. partition data into conformingValues (metric[i] <= bound, in original
//      relative order) and exceptionValues (metric[i] > bound, ditto)
//   4. innerEncoded = innerCodec_->encode(conformingValues)  -- the inner
//      codec computes its OWN parameters (e.g. RawBitPackedEncoder's own
//      min/bitWidth) fresh from the now-outlier-free conforming subset; no
//      coordination with the detector's chosen bound is required beyond
//      "which indices are exceptions"
//   5. exceptionValues stored raw (Phase 1 decision -- see design doc's
//      "Decisions locked in during this review")
//
// Wire format:
//   [8]  N                    (uint64_t) element count
//   [4]  bitmapByteCount      (uint32_t) = ((N+63)/64)*8
//   [B]  exception bitmap     (packed uint64_t words; 1 = exception)
//   [4]  exceptionCount       (uint32_t)
//   [E]  exceptionValues      (raw T[exceptionCount])
//   [4]  innerEncodedSize     (uint32_t)
//   [I]  inner-encoded conforming values
//
// Decode uses BitmapRank (exceptions/BitmapRank.hpp, shared with
// MainlyConstantEncoder) for O(word-count) prefix/range rank of the
// exception bitmap.
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_integral_v<T>
class ExceptionEncoder : public encodings::Codec<T, uint8_t> {
public:
    using EncodedData = encodings::EncodedBuffer<uint8_t>;
    static constexpr uint32_t kMaxBound = static_cast<uint32_t>(sizeof(T) * 8);

    ExceptionEncoder(std::shared_ptr<ExceptionDetector<T>> detector,
                      std::shared_ptr<encodings::Codec<T, uint8_t>> innerCodec)
        : detector_(std::move(detector)), inner_(std::move(innerCodec)) {}

    // -------------------------------------------------------------------------
    // Encode
    // -------------------------------------------------------------------------
    EncodedData encode(std::span<const T> data) override {
        const size_t N = data.size();
        if (N == 0) {
            return makeEmpty();
        }

        const std::vector<uint32_t> metric = detector_->computeMetric(data);
        const double bitmapOverheadBits = static_cast<double>(N); // 1 bit/row, dense mode
        const auto choice = ExceptionMetricUtils::sweepMinCostBound(
            metric, kMaxBound, /*exceptionBitsPerValue=*/static_cast<double>(sizeof(T) * 8),
            bitmapOverheadBits);
        const uint32_t bound = choice.bound;

        std::vector<T> conformingValues;
        std::vector<T> exceptionValues;
        conformingValues.reserve(N - choice.exceptionCount);
        exceptionValues.reserve(choice.exceptionCount);
        for (size_t i = 0; i < N; ++i) {
            if (metric[i] > bound) {
                exceptionValues.push_back(data[i]);
            } else {
                conformingValues.push_back(data[i]);
            }
        }

        const auto bitmap = BitmapRank::build(N, [&](size_t i) { return metric[i] > bound; });
        const uint32_t bitmapByteCount = static_cast<uint32_t>(bitmap.size() * sizeof(uint64_t));

        EncodedData innerEncoded = inner_->encode(std::span<const T>(conformingValues));

        const uint32_t exceptionCount = static_cast<uint32_t>(exceptionValues.size());
        const uint32_t exceptionBytes = exceptionCount * static_cast<uint32_t>(sizeof(T));
        const uint32_t innerEncodedSize = static_cast<uint32_t>(innerEncoded.data().size());

        const size_t totalSize = sizeof(uint64_t) + sizeof(uint32_t) + bitmapByteCount
                                + sizeof(uint32_t) + exceptionBytes
                                + sizeof(uint32_t) + innerEncodedSize;

        EncodedData result;
        result.data().resize(totalSize);
        uint8_t* pos = result.data().data();
        writeU64(pos, static_cast<uint64_t>(N));
        write(pos, bitmapByteCount);
        std::memcpy(pos, bitmap.data(), bitmapByteCount);
        pos += bitmapByteCount;
        write(pos, exceptionCount);
        if (exceptionBytes > 0) {
            std::memcpy(pos, exceptionValues.data(), exceptionBytes);
            pos += exceptionBytes;
        }
        write(pos, innerEncodedSize);
        if (innerEncodedSize > 0) {
            std::memcpy(pos, innerEncoded.data().data(), innerEncodedSize);
        }

        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = N;
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = N * sizeof(T);
        result.metadata().supportsRandomAccess =
            inner_->properties().has(EncodingProperty::RandomAccess);
        result.metadata().customMetadata["exception_count"] = std::to_string(exceptionCount);
        result.metadata().customMetadata["bound_bits"] = std::to_string(bound);
        return result;
    }

    // -------------------------------------------------------------------------
    // Decode
    // -------------------------------------------------------------------------
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.data().empty()) return {};
        const ParsedHeader h = parseHeader(encoded.data().data());
        if (h.N == 0) return {};

        const std::vector<T> conforming = inner_->decodeAll(makeInnerEncoded(h));

        std::vector<T> out(h.N);
        size_t ci = 0, ei = 0;
        for (size_t i = 0; i < h.N; ++i) {
            const uint64_t word = BitmapRank::loadWord(h.bitmapStart + (i / 64) * 8);
            const bool isException = (word >> (i % 64)) & uint64_t{1};
            if (isException) {
                T val;
                std::memcpy(&val, h.exceptionValuesStart + ei * sizeof(T), sizeof(T));
                out[i] = val;
                ++ei;
            } else {
                out[i] = conforming[ci++];
            }
        }
        return out;
    }

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.data().empty()) return std::nullopt;
        const ParsedHeader h = parseHeader(encoded.data().data());
        if (index >= h.N) return std::nullopt;

        const uint64_t word = BitmapRank::loadWord(h.bitmapStart + (index / 64) * 8);
        const bool isException = (word >> (index % 64)) & uint64_t{1};
        const uint32_t rankBefore = BitmapRank::prefixSetCount(h.bitmapStart, index);

        if (isException) {
            T val;
            std::memcpy(&val, h.exceptionValuesStart + static_cast<size_t>(rankBefore) * sizeof(T), sizeof(T));
            return val;
        }

        const size_t conformingIndex = index - rankBefore;
        if (inner_->properties().has(EncodingProperty::RandomAccess)) {
            return inner_->decodeAt(makeInnerEncoded(h), conformingIndex);
        }
        // Fallback: full decode of the conforming stream, then index.
        auto conforming = inner_->decodeAll(makeInnerEncoded(h));
        if (conformingIndex >= conforming.size()) return std::nullopt;
        return conforming[conformingIndex];
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (encoded.data().empty() || start >= end) return {};
        const auto all = decodeAll(encoded);
        if (start >= all.size()) return {};
        end = std::min(end, all.size());
        return std::vector<T>(all.begin() + static_cast<ptrdiff_t>(start),
                              all.begin() + static_cast<ptrdiff_t>(end));
    }

    EncodingType encodingType() const override { return EncodingType::ExceptionWrappedEncoding; }

    std::string name() const override {
        return "ExceptionWrapped(" + detector_->name() + "|" + inner_->name() + ")";
    }

    EncodingProperties properties() const override {
        EncodingProperties props;
        props.add(EncodingProperty::Lossless)
             .add(EncodingProperty::PreservesOrder)
             .add(EncodingProperty::Composable)
             .add(EncodingProperty::RequiresFullData);
        if (inner_->properties().has(EncodingProperty::RandomAccess)) {
            props.add(EncodingProperty::RandomAccess);
        }
        return props;
    }

private:
    struct ParsedHeader {
        size_t N{};
        const uint8_t* bitmapStart{};
        uint32_t exceptionCount{};
        const uint8_t* exceptionValuesStart{};
        uint32_t innerEncodedSize{};
        const uint8_t* innerEncodedStart{};
    };

    static void writeU64(uint8_t*& pos, uint64_t v) {
        std::memcpy(pos, &v, sizeof(uint64_t));
        pos += sizeof(uint64_t);
    }
    template <typename V>
    static void write(uint8_t*& pos, const V& v) {
        std::memcpy(pos, &v, sizeof(V));
        pos += sizeof(V);
    }
    static uint64_t readU64(const uint8_t* p) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(uint64_t));
        return v;
    }
    template <typename V>
    static V read(const uint8_t*& p) {
        V v;
        std::memcpy(&v, p, sizeof(V));
        p += sizeof(V);
        return v;
    }

    static ParsedHeader parseHeader(const uint8_t* base) {
        ParsedHeader h;
        const uint8_t* p = base;
        h.N = static_cast<size_t>(readU64(p));
        p += sizeof(uint64_t);
        const uint32_t bitmapByteCount = read<uint32_t>(p);
        h.bitmapStart = p;
        p += bitmapByteCount;
        h.exceptionCount = read<uint32_t>(p);
        h.exceptionValuesStart = p;
        p += static_cast<size_t>(h.exceptionCount) * sizeof(T);
        h.innerEncodedSize = read<uint32_t>(p);
        h.innerEncodedStart = p;
        return h;
    }

    EncodedData makeInnerEncoded(const ParsedHeader& h) const {
        encodings::EncodingMetadata meta;
        meta.encodingName = inner_->name();
        meta.dataType = encodings::core::typeToDataType<T>;
        meta.elementCount = h.N - h.exceptionCount;
        meta.compressedSize = h.innerEncodedSize;
        meta.uncompressedSize = meta.elementCount * sizeof(T);
        meta.supportsRandomAccess = inner_->properties().has(EncodingProperty::RandomAccess);
        return encodings::EncodedData(
            std::vector<uint8_t>(h.innerEncodedStart, h.innerEncodedStart + h.innerEncodedSize),
            std::move(meta));
    }

    EncodedData makeEmpty() const {
        EncodedData result;
        result.data().resize(sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t), 0);
        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = 0;
        result.metadata().compressedSize = result.data().size();
        result.metadata().uncompressedSize = 0;
        result.metadata().supportsRandomAccess = true;
        return result;
    }

    std::shared_ptr<ExceptionDetector<T>> detector_;
    std::shared_ptr<encodings::Codec<T, uint8_t>> inner_;
};

} // namespace encodings::exceptions
