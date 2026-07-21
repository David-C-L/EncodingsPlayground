#pragma once

#include "encoders/selectors/MetricCollector.hpp"
#include "encoders/selectors/costs/EncodingCostModel.hpp"
#include "encodings/EncodingType.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace encodings::encoders::selectors::costs {

// ---------------------------------------------------------------------------
// Speed estimation constants
// ---------------------------------------------------------------------------
//
// Speed cost = I/O component + CPU component:
//
//   speed_cost(E) = encoded_bits(E) × kBandwidthNsPerBit
//                 + numValues       × kCpuNsPerElem[E]
//
// encoded_bits(E) comes from the matching compression cost model held inside
// each speed model class.  Because compressed encodings produce fewer bits,
// their I/O component is smaller than Raw's, which can make the speed ratio
// drop below 1.0 — enabling the DP to prefer compressed encodings even in
// speed-only or speed-heavy modes.
//
// kBandwidthNsPerBit calibrates the I/O component to your memory hierarchy
// (default ≈ 33 Gbit/s effective bandwidth).  Tune if profiling reveals
// systematic mis-rankings.

namespace detail {

// Memory bandwidth: ns to transfer one encoded bit.
// Default ≈ 20 Gbit/s effective (conservative DRAM + cache-miss overhead for
// columnar analytics).  At this value the bandwidth crossover for a 1-bit
// sub-stream vs its 8-bit raw form is positive (~0.35 ns I/O saving vs
// ~0.20 ns CPU overhead) so pure speed-only modes reward compression.
// For cache-resident workloads lower this toward 0.01; for DRAM-heavy toward 0.1.
inline constexpr double kBandwidthNsPerBit = 0.05;

// CPU-only encode throughput (ns per input element, steady-state).
inline constexpr double kCpuEncodeRaw               = 0.50;
inline constexpr double kCpuEncodeBitPacking        = 1.00;
inline constexpr double kCpuEncodeFOR               = 1.00;
inline constexpr double kCpuEncodeAdaptiveFOR       = 1.20;
inline constexpr double kCpuEncodeAdaptiveBitPrefix = 1.50;
inline constexpr double kCpuEncodeRLE               = 1.00;
inline constexpr double kCpuEncodeDictBase          = 2.00; // + logCard / 10
inline constexpr double kCpuEncodeAdaptiveDict      = 2.20; // Dict + multi-pass block sweep
inline constexpr double kCpuEncodeHuffmanBase       = 3.00; // + logCard / 8
inline constexpr double kCpuEncodeFSEBase           = 2.00; // + logCard / 8
inline constexpr double kCpuEncodeLZ4               = 1.20;
inline constexpr double kCpuEncodeFreqPartBase      = 2.50; // + logCard / 10

// CPU-only decode-all throughput (ns per output element, sequential scan).
inline constexpr double kCpuDecodeAllRaw               = 0.50;
inline constexpr double kCpuDecodeAllBitPacking        = 0.70;
inline constexpr double kCpuDecodeAllFOR               = 0.70;
inline constexpr double kCpuDecodeAllAdaptiveFOR       = 0.70;
inline constexpr double kCpuDecodeAllAdaptiveBitPrefix = 1.00;
inline constexpr double kCpuDecodeAllRLE               = 1.20;
inline constexpr double kCpuDecodeAllDict              = 1.00;
inline constexpr double kCpuDecodeAllAdaptiveDict      = 1.05; // Dict + per-block descriptor reads
inline constexpr double kCpuDecodeAllHuffman           = 5.00;
inline constexpr double kCpuDecodeAllFSE               = 3.00;
inline constexpr double kCpuDecodeAllLZ4               = 1.00;
inline constexpr double kCpuDecodeAllFreqPartition     = 1.50;

// CPU-only random-access cost for O(1) decoders (ns per single access).
// Sequential encodings (Huffman, FSE, LZ4) must scan from pos 0 and are
// handled separately in DecodeAtSpeedCostModel::computeCost.
inline constexpr double kCpuDecodeAtRaw               = 1.0;
inline constexpr double kCpuDecodeAtBitPacking        = 1.5;
inline constexpr double kCpuDecodeAtFOR               = 2.0;
inline constexpr double kCpuDecodeAtAdaptiveFOR       = 2.0;
inline constexpr double kCpuDecodeAtAdaptiveBitPrefix = 3.0;
// RLE: interpolated binary search over run-starts array; converges in ~3
// probes regardless of runCount because run starts are approximately uniform.
inline constexpr double kCpuDecodeAtRLE               = 3.0;
inline constexpr double kCpuDecodeAtDict              = 2.0;
inline constexpr double kCpuDecodeAtAdaptiveDict      = 2.2;  // Dict + block index lookup
inline constexpr double kCpuDecodeAtFreqPartition     = 3.0;

// ---------------------------------------------------------------------------
// Per-encoding CPU estimators (return CPU-only ns/elem, not total cost)
// ---------------------------------------------------------------------------

inline double estimateCpuEncodeNs(
    encodings::EncodingType type,
    const SegmentMetrics& metrics)
{
    const double logCard = std::log2(std::max(2.0, metrics.hllEstimatedCardinality));
    switch (type) {
        case encodings::EncodingType::RawEncoding:               return kCpuEncodeRaw;
        case encodings::EncodingType::BitPacking:                return kCpuEncodeBitPacking;
        case encodings::EncodingType::FrameOfReference:          return kCpuEncodeFOR;
        case encodings::EncodingType::AdaptiveFrameOfReference:  return kCpuEncodeAdaptiveFOR;
        case encodings::EncodingType::AdaptiveFramedBitPrefix:   return kCpuEncodeAdaptiveBitPrefix;
        case encodings::EncodingType::RunLengthEncoding:         return kCpuEncodeRLE;
        case encodings::EncodingType::DictionaryEncoding:        return kCpuEncodeDictBase     + logCard / 10.0;
        case encodings::EncodingType::AdaptiveDictionaryEncoding: return kCpuEncodeAdaptiveDict + logCard / 10.0;
        case encodings::EncodingType::HuffmanEncoding:           return kCpuEncodeHuffmanBase  + logCard /  8.0;
        case encodings::EncodingType::FSEEncoding:               return kCpuEncodeFSEBase      + logCard /  8.0;
        case encodings::EncodingType::LZ4:                       return kCpuEncodeLZ4;
        case encodings::EncodingType::FrequencyPartitionEncoding:return kCpuEncodeFreqPartBase + logCard / 10.0;
        default:                                                  return kCpuEncodeRaw;
    }
}

inline double estimateCpuDecodeAllNs(
    encodings::EncodingType type,
    const SegmentMetrics& /*metrics*/)
{
    switch (type) {
        case encodings::EncodingType::RawEncoding:               return kCpuDecodeAllRaw;
        case encodings::EncodingType::BitPacking:                return kCpuDecodeAllBitPacking;
        case encodings::EncodingType::FrameOfReference:          return kCpuDecodeAllFOR;
        case encodings::EncodingType::AdaptiveFrameOfReference:  return kCpuDecodeAllAdaptiveFOR;
        case encodings::EncodingType::AdaptiveFramedBitPrefix:   return kCpuDecodeAllAdaptiveBitPrefix;
        case encodings::EncodingType::RunLengthEncoding:         return kCpuDecodeAllRLE;
        case encodings::EncodingType::DictionaryEncoding:        return kCpuDecodeAllDict;
        case encodings::EncodingType::AdaptiveDictionaryEncoding: return kCpuDecodeAllAdaptiveDict;
        case encodings::EncodingType::HuffmanEncoding:           return kCpuDecodeAllHuffman;
        case encodings::EncodingType::FSEEncoding:               return kCpuDecodeAllFSE;
        case encodings::EncodingType::LZ4:                       return kCpuDecodeAllLZ4;
        case encodings::EncodingType::FrequencyPartitionEncoding:return kCpuDecodeAllFreqPartition;
        default:                                                  return kCpuDecodeAllRaw;
    }
}

// Returns the CPU-only cost for one random access for O(1) decoders.
// Returns 0.0 for sequential encodings (caller handles them separately).
inline double estimateCpuDecodeAtNs(encodings::EncodingType type) {
    switch (type) {
        case encodings::EncodingType::RawEncoding:               return kCpuDecodeAtRaw;
        case encodings::EncodingType::BitPacking:                return kCpuDecodeAtBitPacking;
        case encodings::EncodingType::FrameOfReference:          return kCpuDecodeAtFOR;
        case encodings::EncodingType::AdaptiveFrameOfReference:  return kCpuDecodeAtAdaptiveFOR;
        case encodings::EncodingType::AdaptiveFramedBitPrefix:   return kCpuDecodeAtAdaptiveBitPrefix;
        case encodings::EncodingType::RunLengthEncoding:         return kCpuDecodeAtRLE;
        case encodings::EncodingType::DictionaryEncoding:        return kCpuDecodeAtDict;
        case encodings::EncodingType::AdaptiveDictionaryEncoding: return kCpuDecodeAtAdaptiveDict;
        case encodings::EncodingType::FrequencyPartitionEncoding:return kCpuDecodeAtFreqPartition;
        default:                                                  return kCpuDecodeAtRaw;
    }
}

// Huffman, FSE, LZ4 must decode from position 0 for any random access.
inline bool isSequentialDecodeAt(encodings::EncodingType type) {
    return type == encodings::EncodingType::HuffmanEncoding
        || type == encodings::EncodingType::FSEEncoding
        || type == encodings::EncodingType::LZ4;
}

// Which SegmentMetrics fields a speed model for the given encoding type needs
// (beyond whatever the compression sub-model already requires).
inline MetricFlags requiredMetricsForSpeed(encodings::EncodingType type) {
    switch (type) {
        case encodings::EncodingType::DictionaryEncoding:
        case encodings::EncodingType::AdaptiveDictionaryEncoding:
        case encodings::EncodingType::HuffmanEncoding:
        case encodings::EncodingType::FSEEncoding:
        case encodings::EncodingType::FrequencyPartitionEncoding:
            return static_cast<MetricFlags>(MetricFlag::FreqStats);
        case encodings::EncodingType::RunLengthEncoding:
            return static_cast<MetricFlags>(MetricFlag::RunStats);
        default:
            return static_cast<MetricFlags>(MetricFlag::None);
    }
}

// Raw encoded size in bits (no compression fallback).
inline double rawEncodedBits(size_t numValues, size_t bitWidth) {
    return static_cast<double>(numValues) * static_cast<double>(storageWidthBits(bitWidth));
}

} // namespace detail

// ---------------------------------------------------------------------------
// Speed cost model classes (one per dimension)
// ---------------------------------------------------------------------------
//
// Each class models ONE encoding type (set at construction) for its speed
// dimension.  computeCost() returns total estimated nanoseconds for the
// segment — the same "total cost for N elements" convention as the
// compression cost models, so the DP's effectiveCount scaling applies
// uniformly.
//
// Each class holds a compression sub-model to estimate encoded_bits, which
// drives the I/O component of the cost.

class EncodeSpeedCostModel final : public EncodingCostModel {
    encodings::EncodingType encodingType_;
    std::unique_ptr<EncodingCostModel> compressionModel_;
public:
    explicit EncodeSpeedCostModel(encodings::EncodingType et)
        : encodingType_(et), compressionModel_(makeCompressionCostModel(et)) {}

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        const double encodedBits = compressionModel_
            ? compressionModel_->computeCost(metrics, numValues, bitWidth)
            : detail::rawEncodedBits(numValues, bitWidth);
        const double io  = encodedBits * detail::kBandwidthNsPerBit;
        const double cpu = static_cast<double>(numValues) *
                           detail::estimateCpuEncodeNs(encodingType_, metrics);
        return io + cpu;
    }

    encodings::EncodingType encodingType() const override { return encodingType_; }

    CostModelDimension costModelDimension() const override {
        return CostModelDimension::EncodeSpeed;
    }

    MetricFlags requiredMetrics() const override {
        MetricFlags flags = detail::requiredMetricsForSpeed(encodingType_);
        if (compressionModel_) flags |= compressionModel_->requiredMetrics();
        return flags;
    }
};

class DecodeAllSpeedCostModel final : public EncodingCostModel {
    encodings::EncodingType encodingType_;
    std::unique_ptr<EncodingCostModel> compressionModel_;
public:
    explicit DecodeAllSpeedCostModel(encodings::EncodingType et)
        : encodingType_(et), compressionModel_(makeCompressionCostModel(et)) {}

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        const double encodedBits = compressionModel_
            ? compressionModel_->computeCost(metrics, numValues, bitWidth)
            : detail::rawEncodedBits(numValues, bitWidth);
        const double io  = encodedBits * detail::kBandwidthNsPerBit;
        const double cpu = static_cast<double>(numValues) *
                           detail::estimateCpuDecodeAllNs(encodingType_, metrics);
        return io + cpu;
    }

    encodings::EncodingType encodingType() const override { return encodingType_; }

    CostModelDimension costModelDimension() const override {
        return CostModelDimension::DecodeAllSpeed;
    }

    MetricFlags requiredMetrics() const override {
        MetricFlags flags = detail::requiredMetricsForSpeed(encodingType_);
        if (compressionModel_) flags |= compressionModel_->requiredMetrics();
        return flags;
    }
};

class DecodeAtSpeedCostModel final : public EncodingCostModel {
    encodings::EncodingType encodingType_;
    std::unique_ptr<EncodingCostModel> compressionModel_;
public:
    explicit DecodeAtSpeedCostModel(encodings::EncodingType et)
        : encodingType_(et), compressionModel_(makeCompressionCostModel(et)) {}

    // Total cost for numValues uniform random accesses across the segment.
    //
    // O(1) encodings: N accesses × cpu_ns_per_access (no I/O scan needed).
    //
    // Sequential encodings (Huffman, FSE, LZ4): each access must scan from
    // position 0 to the target.  For N uniform accesses the average scan
    // depth is numValues/2 elements ≡ 0.5 × encoded_bits of I/O.  Both the
    // I/O and CPU scan components are proportional to the encoded segment
    // size, so better compression directly speeds up random access.
    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        const double N = static_cast<double>(numValues);

        if (detail::isSequentialDecodeAt(encodingType_)) {
            const double encodedBits = compressionModel_
                ? compressionModel_->computeCost(metrics, numValues, bitWidth)
                : detail::rawEncodedBits(numValues, bitWidth);
            const double ioPerAccess  = 0.5 * encodedBits * detail::kBandwidthNsPerBit;
            const double cpuPerAccess = 0.5 * N * detail::estimateCpuDecodeAllNs(encodingType_, metrics);
            return N * (ioPerAccess + cpuPerAccess);
        }

        // O(1): no I/O scan, just CPU cost per access.
        return N * detail::estimateCpuDecodeAtNs(encodingType_);
    }

    encodings::EncodingType encodingType() const override { return encodingType_; }

    CostModelDimension costModelDimension() const override {
        return CostModelDimension::DecodeAtSpeed;
    }

    MetricFlags requiredMetrics() const override {
        MetricFlags flags = detail::requiredMetricsForSpeed(encodingType_);
        if (compressionModel_) flags |= compressionModel_->requiredMetrics();
        return flags;
    }
};

class DecodeRangeSpeedCostModel final : public EncodingCostModel {
    encodings::EncodingType encodingType_;
    std::unique_ptr<EncodingCostModel> compressionModel_;
public:
    explicit DecodeRangeSpeedCostModel(encodings::EncodingType et)
        : encodingType_(et), compressionModel_(makeCompressionCostModel(et)) {}

    // Same as DecodeAll: range length is unknown to the cost model, so the
    // worst case (full segment) is used.  Sequential encodings' start-seek
    // cost is folded into the same per-element formula.
    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        const double encodedBits = compressionModel_
            ? compressionModel_->computeCost(metrics, numValues, bitWidth)
            : detail::rawEncodedBits(numValues, bitWidth);
        const double io  = encodedBits * detail::kBandwidthNsPerBit;
        const double cpu = static_cast<double>(numValues) *
                           detail::estimateCpuDecodeAllNs(encodingType_, metrics);
        return io + cpu;
    }

    encodings::EncodingType encodingType() const override { return encodingType_; }

    CostModelDimension costModelDimension() const override {
        return CostModelDimension::DecodeRangeSpeed;
    }

    MetricFlags requiredMetrics() const override {
        MetricFlags flags = detail::requiredMetricsForSpeed(encodingType_);
        if (compressionModel_) flags |= compressionModel_->requiredMetrics();
        return flags;
    }
};

// ---------------------------------------------------------------------------
// Per-dimension factory functions
// ---------------------------------------------------------------------------

inline std::vector<std::unique_ptr<EncodingCostModel>>
makeEncodeSpeedCostModels(const std::vector<encodings::EncodingType>& types) {
    std::vector<std::unique_ptr<EncodingCostModel>> v;
    v.reserve(types.size());
    for (auto et : types)
        v.emplace_back(std::make_unique<EncodeSpeedCostModel>(et));
    return v;
}

inline std::vector<std::unique_ptr<EncodingCostModel>>
makeDecodeAllSpeedCostModels(const std::vector<encodings::EncodingType>& types) {
    std::vector<std::unique_ptr<EncodingCostModel>> v;
    v.reserve(types.size());
    for (auto et : types)
        v.emplace_back(std::make_unique<DecodeAllSpeedCostModel>(et));
    return v;
}

inline std::vector<std::unique_ptr<EncodingCostModel>>
makeDecodeAtSpeedCostModels(const std::vector<encodings::EncodingType>& types) {
    std::vector<std::unique_ptr<EncodingCostModel>> v;
    v.reserve(types.size());
    for (auto et : types)
        v.emplace_back(std::make_unique<DecodeAtSpeedCostModel>(et));
    return v;
}

inline std::vector<std::unique_ptr<EncodingCostModel>>
makeDecodeRangeSpeedCostModels(const std::vector<encodings::EncodingType>& types) {
    std::vector<std::unique_ptr<EncodingCostModel>> v;
    v.reserve(types.size());
    for (auto et : types)
        v.emplace_back(std::make_unique<DecodeRangeSpeedCostModel>(et));
    return v;
}

} // namespace encodings::encoders::selectors::costs
