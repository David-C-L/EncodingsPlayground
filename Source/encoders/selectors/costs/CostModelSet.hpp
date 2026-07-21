#pragma once

#include "encoders/selectors/costs/EncodingCostModel.hpp"
#include "encoders/selectors/costs/SpeedCostModel.hpp"
#include "encodings/EncodingType.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace encodings::encoders::selectors::costs {

// ---------------------------------------------------------------------------
// WeightedCompositeEncodingCostModel
// ---------------------------------------------------------------------------
//
// Combines multiple cost dimensions for ONE encoding type into a single scalar
// that the DP can minimise.  Each dimension's cost is normalised by the raw
// encoding's cost in that same dimension, yielding a dimensionless ratio:
//
//   ratio_d = cost_d(E) / cost_d(Raw)
//
// The combined cost is:
//
//   combined = (Σ w_d × ratio_d / Σ w_d) × compression_baseline_bits
//
// where compression_baseline_bits = numValues × bitWidth.
//
// This keeps the output in "bits" so the DP's splitPenalty (also in bits)
// remains meaningful.  A single-dimension Compression-only composite reduces
// to the original compression cost because:
//   ratio_compression = comp_cost / raw_bits × raw_bits = comp_cost.
//
// Note: WeightedCompositeEncodingCostModel is used internally by CostModelSet.
// Users should prefer CostModelSet::build() rather than constructing this
// class directly.

class WeightedCompositeEncodingCostModel final : public EncodingCostModel {
public:
    struct Dimension {
        double weight;
        std::unique_ptr<EncodingCostModel> model;    // for this encoding type
        std::unique_ptr<EncodingCostModel> baseline; // Raw for same dimension
    };

    WeightedCompositeEncodingCostModel(
        encodings::EncodingType et,
        std::vector<Dimension> dims)
        : encodingType_(et), dims_(std::move(dims)) {}

    double computeCost(
        const SegmentMetrics& metrics,
        size_t numValues,
        size_t bitWidth) const override
    {
        const double compressionBaseline =
            static_cast<double>(numValues) * static_cast<double>(bitWidth);

        double weightSum = 0.0;
        double ratioSum  = 0.0;
        for (const auto& d : dims_) {
            const double raw  = std::max(1.0, d.baseline->computeCost(metrics, numValues, bitWidth));
            const double cost = d.model->computeCost(metrics, numValues, bitWidth);
            ratioSum  += d.weight * (cost / raw);
            weightSum += d.weight;
        }
        const double combinedRatio = (weightSum > 0.0) ? ratioSum / weightSum : 1.0;
        return combinedRatio * compressionBaseline;
    }

    encodings::EncodingType encodingType() const override { return encodingType_; }

    CostModelDimension costModelDimension() const override {
        return CostModelDimension::Compression; // composite; not a single dimension
    }

    MetricFlags requiredMetrics() const override {
        MetricFlags flags = static_cast<MetricFlags>(MetricFlag::None);
        for (const auto& d : dims_) {
            flags |= d.model->requiredMetrics();
            flags |= d.baseline->requiredMetrics();
        }
        return flags;
    }

private:
    encodings::EncodingType encodingType_;
    std::vector<Dimension> dims_;
};

// ---------------------------------------------------------------------------
// CostModelSet — fluent builder for composable cost model vectors
// ---------------------------------------------------------------------------
//
// Usage:
//
//   // Compression only (same as the existing default):
//   auto models = CostModelSet{}.add(CostModelDimension::Compression).build();
//
//   // DecodeAt speed only:
//   auto models = CostModelSet{}.add(CostModelDimension::DecodeAtSpeed).build();
//
//   // Compression + DecodeAt with equal weight (default):
//   auto models = CostModelSet{}
//       .add(CostModelDimension::Compression)
//       .add(CostModelDimension::DecodeAtSpeed)
//       .build();
//
//   // Compression + DecodeAt, decodeAt weighted 3×:
//   auto models = CostModelSet{}
//       .add(CostModelDimension::Compression, 1.0)
//       .add(CostModelDimension::DecodeAtSpeed, 3.0)
//       .build();
//
//   // Restrict the encoding types considered:
//   auto models = CostModelSet{}
//       .forEncodings({EncodingType::RawEncoding, EncodingType::BitPacking})
//       .add(CostModelDimension::DecodeAtSpeed)
//       .build();
//
// build() returns a vector<unique_ptr<EncodingCostModel>> ready to pass to
// IDSubStreamEncodingSelector or SubIntSplitAutoEncoderConfig::costModels.

class CostModelSet {
public:
    // Default encoding types (mirrors defaultAutoSubIntSplitCostModelTypes()).
    static std::vector<encodings::EncodingType> defaultEncodings() {
        return {
            encodings::EncodingType::RawEncoding,
            encodings::EncodingType::BitPacking,
            encodings::EncodingType::RunLengthEncoding,
            encodings::EncodingType::AdaptiveFrameOfReference,
            encodings::EncodingType::DictionaryEncoding,
            encodings::EncodingType::AdaptiveDictionaryEncoding,
            encodings::EncodingType::FrequencyPartitionEncoding,
        };
    }

    // Add a cost dimension with an optional weight (default 1.0).
    // Multiple calls to add() with the same dimension are additive (their
    // weights are summed), though typically each dimension is added once.
    CostModelSet& add(CostModelDimension dim, double weight = 1.0) {
        if (weight <= 0.0)
            throw std::invalid_argument("CostModelSet::add: weight must be > 0");
        dims_.push_back({dim, weight});
        return *this;
    }

    // Override which encoding types the models cover.
    CostModelSet& forEncodings(std::vector<encodings::EncodingType> types) {
        encodingTypes_ = std::move(types);
        return *this;
    }

    // Return the fraction of total weight assigned to speed dimensions [0, 1].
    double speedWeightFraction() const {
        double speedW = 0.0, totalW = 0.0;
        for (const auto& [dim, w] : dims_) {
            totalW += w;
            if (dim != CostModelDimension::Compression) speedW += w;
        }
        return (totalW > 0.0) ? speedW / totalW : 0.0;
    }

    // Recommended splitPenalty for the DP when this CostModelSet is used.
    //
    // The compression component (10 bits) is fixed — it represents the header
    // overhead of one extra sub-stream.
    //
    // The speed component scales with numSamples × speedWeightFraction to
    // capture the per-element decode overhead of having one more sub-stream in
    // the reconstruction pipeline (~1.5 ns/elem/stream, normalized to "bits").
    // Without this term the DP tends to create many fine-grained sub-streams
    // (e.g. 32 × 1-bit streams) that have low individual cost but collectively
    // dominate the total decode time via repeated full-segment passes.
    double recommendedSplitPenalty(size_t numSamples) const {
        static constexpr double kCompressionPenalty    = 10.0;
        static constexpr double kSpeedSplitFactor      = 1.5;
        const double speedPenalty = static_cast<double>(numSamples)
                                    * kSpeedSplitFactor
                                    * speedWeightFraction();
        return kCompressionPenalty + speedPenalty;
    }

    // Build the flat model vector for use in the DP.
    //
    // If exactly one dimension is registered, models are returned directly
    // (no composite wrapper) — this is the zero-overhead path for the common
    // single-dimension case and preserves the exact cost values from the
    // underlying model.
    //
    // If multiple dimensions are registered, one WeightedCompositeEncodingCostModel
    // is created per encoding type.
    std::vector<std::unique_ptr<EncodingCostModel>> build() const {
        if (dims_.empty())
            throw std::invalid_argument("CostModelSet::build: no dimensions added; call add() first");

        const auto& types = encodingTypes_.empty() ? defaultEncodings() : encodingTypes_;

        // Fast path: single dimension, return models directly.
        if (dims_.size() == 1) {
            return buildSingleDimension(dims_.front().dim, types);
        }

        // Multi-dimension: build one composite per encoding type.
        std::vector<std::unique_ptr<EncodingCostModel>> result;
        result.reserve(types.size());

        for (auto et : types) {
            std::vector<WeightedCompositeEncodingCostModel::Dimension> dimModels;
            dimModels.reserve(dims_.size());

            for (const auto& [dim, weight] : dims_) {
                auto model    = makeDimensionModel(dim, et);
                auto baseline = makeDimensionModel(dim, encodings::EncodingType::RawEncoding);
                if (!model || !baseline) continue; // unsupported combination, skip

                dimModels.push_back({weight, std::move(model), std::move(baseline)});
            }

            if (dimModels.empty()) continue;
            result.emplace_back(
                std::make_unique<WeightedCompositeEncodingCostModel>(et, std::move(dimModels)));
        }

        if (result.empty())
            throw std::runtime_error("CostModelSet::build: no models could be constructed");
        return result;
    }

private:
    struct DimWeight {
        CostModelDimension dim;
        double weight;
    };

    std::vector<DimWeight> dims_;
    std::vector<encodings::EncodingType> encodingTypes_; // empty = use defaultEncodings()

    // Create one model for a given (dimension, encoding_type) pair.
    // Returns nullptr for unsupported combinations.
    static std::unique_ptr<EncodingCostModel>
    makeDimensionModel(CostModelDimension dim, encodings::EncodingType et) {
        switch (dim) {
            case CostModelDimension::Compression:
                return makeCompressionCostModel(et);
            case CostModelDimension::EncodeSpeed:
                return std::make_unique<EncodeSpeedCostModel>(et);
            case CostModelDimension::DecodeAllSpeed:
                return std::make_unique<DecodeAllSpeedCostModel>(et);
            case CostModelDimension::DecodeAtSpeed:
                return std::make_unique<DecodeAtSpeedCostModel>(et);
            case CostModelDimension::DecodeRangeSpeed:
                return std::make_unique<DecodeRangeSpeedCostModel>(et);
        }
        return nullptr;
    }

    // Build a flat vector of models for a single dimension.
    static std::vector<std::unique_ptr<EncodingCostModel>>
    buildSingleDimension(
        CostModelDimension dim,
        const std::vector<encodings::EncodingType>& types)
    {
        std::vector<std::unique_ptr<EncodingCostModel>> result;
        result.reserve(types.size());
        for (auto et : types) {
            auto m = makeDimensionModel(dim, et);
            if (m) result.push_back(std::move(m));
        }
        if (result.empty())
            throw std::runtime_error("CostModelSet::build: no models for requested dimension");
        return result;
    }
};

} // namespace encodings::encoders::selectors::costs
