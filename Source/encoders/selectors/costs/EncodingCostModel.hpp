#pragma once

#include "encoders/selectors/MetricCollector.hpp"
#include "encodings/EncodingType.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

namespace encodings::encoders::selectors::costs {

#ifndef ENCODINGS_DEBUG_DICT_COST
#define ENCODINGS_DEBUG_DICT_COST 0
#endif

using encodings::encoders::selectors::SegmentMetrics;

// Helper: actual storage width (bits) given a logical bit width.
// Mirrors the section factory selection (8/16/32/64-bit underlying types).
inline constexpr uint8_t storageWidthBits(uint8_t bitWidth) {
	if (bitWidth <= 8) return 8;
	if (bitWidth <= 16) return 16;
	if (bitWidth <= 32) return 32;
	return 64;
}

class EncodingCostModel {
public:
	EncodingCostModel() = default;
    virtual ~EncodingCostModel() = default;

	// Compute the cost of encoding a segment based on its metrics.
	virtual double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const = 0;

    virtual encodings::EncodingType encodingType() const = 0;

};

namespace detail {
inline uint32_t ceil_log2_u64(uint64_t value) {
	if (value <= 1) {
		return 0;
	}
	return static_cast<uint32_t>(std::bit_width(value - 1));
}

inline uint32_t clamp_dict_index_width(uint32_t width) {
	// Allow non-power-of-two widths up to 32 bits to mirror DictionaryEncoder's default behavior.
	if (width == 0) return 1;
	return width <= 32 ? width : 32;
}

inline double hll_alpha(size_t m) {
	if (m == 16) return 0.673;
	if (m == 32) return 0.697;
	if (m == 64) return 0.709;
	return 0.7213 / (1.0 + 1.079 / static_cast<double>(m));
}

inline double estimate_hll_cardinality(const SegmentMetrics::HyperLogLog& hll) {
	const size_t m = hll.registers.size();
	if (m == 0) return 0.0;
	double sum = 0.0;
	size_t zeros = 0;
	for (uint8_t r : hll.registers) {
		if (r == 0) ++zeros;
		sum += std::ldexp(1.0, -static_cast<int>(r));
	}
	const double alpha = hll_alpha(m);
	double estimate = alpha * static_cast<double>(m) * static_cast<double>(m) / sum;
	if (estimate <= 2.5 * static_cast<double>(m) && zeros > 0) {
		estimate = static_cast<double>(m) * std::log(static_cast<double>(m) / static_cast<double>(zeros));
	}
	return estimate;
}

inline double estimate_chao1(size_t observed, size_t f1, size_t f2) {
	const double s = static_cast<double>(observed);
	const double f1d = static_cast<double>(f1);
	const double f2d = static_cast<double>(f2);
	if (f1 == 0) return s;
	if (f2 > 0) {
		return s + (f1d * f1d) / (2.0 * f2d);
	}
	return s + (f1d * (f1d - 1.0)) / 2.0;
}

inline void log_dictionary_cost(
	size_t numValues,
	size_t bitWidth,
	const SegmentMetrics& metrics,
	double observedUniques,
	double hllEstimate,
	double chao1Estimate,
	double blended,
	double estimatedUniques,
	uint32_t indexWidth,
	double entropyBits,
	double f1Ratio,
	double confidence,
	double dictBits,
	double indexBits,
	double headerBits,
	double rawBits,
	double dictPenalty,
	double finalCost) {
#if ENCODINGS_DEBUG_DICT_COST
	std::cout << std::fixed << std::setprecision(2)
		<< "[DictCost] n=" << numValues
		<< " bw=" << bitWidth
		<< " uniques(obs/hll/chao/blend/est)=" << observedUniques
		<< "/" << hllEstimate
		<< "/" << chao1Estimate
		<< "/" << blended
		<< "/" << estimatedUniques
		<< " f1=" << metrics.f1
		<< " f2=" << metrics.f2
		<< " f1r=" << f1Ratio
		<< " conf=" << confidence
		<< " capped=" << (metrics.uniqueCountCapped ? 1 : 0)
		<< " entropy=" << entropyBits
		<< " idxW=" << indexWidth
		<< " dictBits=" << dictBits
		<< " idxBits=" << indexBits
		<< " hdr=" << headerBits
		<< " raw=" << rawBits
		<< " penalty=" << dictPenalty
		<< " cost=" << finalCost
		<< std::endl;
#else
	(void)numValues;
	(void)bitWidth;
	(void)metrics;
	(void)observedUniques;
	(void)hllEstimate;
	(void)chao1Estimate;
	(void)blended;
	(void)estimatedUniques;
	(void)indexWidth;
	(void)entropyBits;
	(void)entropyBits;
	(void)f1Ratio;
	(void)confidence;
	(void)dictBits;
	(void)indexBits;
	(void)headerBits;
	(void)rawBits;
	(void)dictPenalty;
	(void)finalCost;
#endif
}
} // namespace detail

class RawCostModel final : public EncodingCostModel {
public:
	double computeCost(const SegmentMetrics& /*metrics*/, size_t numValues, size_t bitWidth) const override {
		const double storedBits = static_cast<double>(storageWidthBits(static_cast<uint8_t>(bitWidth)));
		return static_cast<double>(numValues) * storedBits;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::RawEncoding;
	}
};

class FORCostModel final : public EncodingCostModel {
public:
	static constexpr size_t kBlockSize = 1024;

	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (numValues == 0) {
			return 0.0;
		}

		// Mirror the runtime FOREncoder layout: 5x uint64_t header (N, FrameSize, numFrames, refBytes, resBytes)
		static constexpr double kHeaderBits = static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

		const size_t blocks = (numValues + kBlockSize - 1) / kBlockSize;
		// Use the precomputed span-based residual width for the 1024-frame candidate (index 6 in MetricCollector).
		constexpr size_t kFrameIdx = 6; // kFrameCandidates[6] == 1024
		const uint8_t spanBits = metrics.frameMaxResidualBits[kFrameIdx];
		// Reference values stored at rounded storage width per block (one ref per frame)
		const size_t refWidthBytes = static_cast<size_t>(storageWidthBits(static_cast<uint8_t>(bitWidth))) / 8;
		const double refBits = static_cast<double>(blocks * refWidthBytes) * 8.0;
		// Residuals stored per value with spanBits (if spanBits==0, treat as 1 bit like all-zero residuals)
		const uint8_t resBitsPerVal = spanBits == 0 ? 1 : spanBits;
		const double resBits = static_cast<double>(numValues) * static_cast<double>(resBitsPerVal);

		return kHeaderBits + refBits + resBits;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::FrameOfReference;
	}
};

class DictionaryCostModel final : public EncodingCostModel {
public:
	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (metrics.uniqueCount == 0 || numValues == 0) {
			return 0.0;
		}

			const double observedUniques = static_cast<double>(metrics.uniqueCount);
			const double hllEstimate = detail::estimate_hll_cardinality(metrics.hll);
			const double chao1Estimate = detail::estimate_chao1(metrics.uniqueCount, metrics.f1, metrics.f2);
			const double f1Ratio = (metrics.uniqueCount > 0)
				? static_cast<double>(metrics.f1) / static_cast<double>(metrics.uniqueCount)
				: 0.0;

			const double chaoWeight = (metrics.f2 < 5) ? 0.1 : 0.3;
			double blended = (1.0 - chaoWeight) * hllEstimate + chaoWeight * chao1Estimate;
			// blended = std::min(blended, observedUniques);

			const double confidence = std::min(1.0, std::sqrt(static_cast<double>(numValues) / 10000.0));
			double estimatedUniques = (1.0 - confidence) * observedUniques + confidence * blended;
			estimatedUniques = std::max(estimatedUniques, observedUniques);
			estimatedUniques = std::min(estimatedUniques, static_cast<double>(numValues));
			estimatedUniques = std::min(
				estimatedUniques,
				static_cast<double>(MetricCollector<uint64_t>::kUniqueCountCap)
			);

			const uint64_t uniques = static_cast<uint64_t>(std::ceil(estimatedUniques));
			const uint32_t rawWidth = detail::ceil_log2_u64(uniques);
			const uint32_t indexWidth = detail::clamp_dict_index_width(rawWidth);
			const double effectiveIndexWidth = std::min<double>(32.0, std::max(1.0, std::log2(std::max(estimatedUniques, 1.0))));
			const uint8_t valueStorageBits = storageWidthBits(static_cast<uint8_t>(bitWidth));
			const double dictBits = static_cast<double>(uniques) * static_cast<double>(valueStorageBits);
			// Bit-packed keys rounded up to whole bytes
			const double indexBitsWorst = static_cast<double>((numValues * static_cast<size_t>(indexWidth) + 7) / 8 * 8);
			const double indexBitsEntropy = (metrics.entropyEstimate > 0.0)
				? static_cast<double>(numValues) * metrics.entropyEstimate
				: static_cast<double>(numValues) * effectiveIndexWidth;
			double indexBits = std::min(indexBitsWorst, indexBitsEntropy);
			indexBits = indexBitsWorst; // disable entropy-based index size for now since estimates can be unreliable and lead to bad choices
			if (f1Ratio > 0.5) {
				indexBits *= (1.0 + 0.35 * f1Ratio * confidence);
			}
		// dict size + key-bytes size + key-bit-width byte
		const double headerBits = static_cast<double>(2 * sizeof(size_t) + 1) * 8.0;
		const double rawBits = static_cast<double>(numValues) * static_cast<double>(valueStorageBits);

		double dictPenalty = 1.0;
		if (indexWidth >= valueStorageBits) {
			const double widthOver = static_cast<double>(indexWidth) / static_cast<double>(valueStorageBits);
			dictPenalty = 1.0 + 0.15 * std::max(0.0, widthOver - 1.0);
		}

		double dictCost = headerBits + dictPenalty * (dictBits + indexBits);

		detail::log_dictionary_cost(
			numValues,
			bitWidth,
			metrics,
			observedUniques,
			hllEstimate,
			chao1Estimate,
			blended,
			estimatedUniques,
			indexWidth,
			metrics.entropyEstimate,
			f1Ratio,
			confidence,
			dictBits,
			indexBits,
			headerBits,
			rawBits,
			dictPenalty,
			dictCost
		);

		return dictCost;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::DictionaryEncoding;
	}
};

class RLECostModel final : public EncodingCostModel {
public:
	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (numValues == 0 || metrics.avgRunLength <= 0.0) {
			return 0.0;
		}

		// Extrapolate run count to the full stream using avgRunLength, which is a scale-invariant
		// ratio and therefore accurate regardless of sample size or sampling strategy.
		// Using raw metrics.runCount is wrong with strided sampling (runCount ≈ sampleSize there).
		const double estimatedRunCount = static_cast<double>(numValues) / metrics.avgRunLength;

		// Matches RunLengthEncoder layout: header (3 × sizeof(size_t)), runStarts, runValues.
		const double headerBits = static_cast<double>(3 * sizeof(size_t)) * 8.0;
		const double runStartsBits = estimatedRunCount * static_cast<double>(sizeof(size_t) * 8u);
		const double runValuesBits = estimatedRunCount * static_cast<double>(storageWidthBits(static_cast<uint8_t>(bitWidth)));

		return headerBits + runStartsBits + runValuesBits;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::RunLengthEncoding;
	}
};

class RawBitPackedCostModel final : public EncodingCostModel {
public:
	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (numValues == 0) {
			return 0.0;
		}

		// Use observed range to estimate payload width (min-based packing in encoder).
		const uint64_t range = metrics.range;
		const uint8_t observedWidth = range == 0 ? 0 : static_cast<uint8_t>(std::bit_width(range));
		const uint8_t payloadBitsPerVal = std::min<uint8_t>(static_cast<uint8_t>(bitWidth), observedWidth);
		const double packedBits = static_cast<double>(((static_cast<size_t>(payloadBitsPerVal) * numValues) + 7) / 8 * 8);
		const double baseBytes = static_cast<double>(storageWidthBits(static_cast<uint8_t>(bitWidth))) / 8.0; // store base at storage width
		const double headerBits = static_cast<double>(sizeof(size_t) + sizeof(uint8_t)) * 8.0 + baseBytes * 8.0;
		return headerBits + packedBits;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::BitPacking;
	}
};

class AdaptiveFORCostModel final : public EncodingCostModel {
public:
	static constexpr size_t kHeaderBytes = 32;

	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (numValues == 0) {
			return 0.0;
		}

		// Reference width: use the substream bitWidth, capped at 8 bytes (real AdaptiveFOR stores TIn, but for bit-sliced segments
		// this is a better estimate of achievable size and helps the selector pick it when appropriate).
		const size_t refWidthBytes = static_cast<size_t>(storageWidthBits(static_cast<uint8_t>(bitWidth))) / 8;

		size_t bestBytes = std::numeric_limits<size_t>::max();
		for (size_t i = 0; i < SegmentMetrics::kFrameCandidateCount; ++i) {
			const size_t frame = MetricCollector<uint64_t>::kFrameCandidates[i];
			const size_t numFrames = (numValues + frame - 1) / frame;
			const size_t refBytes = numFrames * refWidthBytes;

			const double avgBits = metrics.frameAvgResidualBits[i];
			const uint8_t spanBits = metrics.frameMaxResidualBits[i];
			uint8_t neededBits = avgBits > 0.0
				? static_cast<uint8_t>(std::ceil(avgBits))
				: (spanBits == 0 ? 1 : spanBits);
			neededBits = (spanBits == 0) ? 1 : spanBits; // for cost estimation, use max-based width since avg can be unreliable and lead to bad choices

			size_t packedBytes = std::numeric_limits<size_t>::max();
			if (neededBits <= 32) {
				packedBytes = ((static_cast<size_t>(neededBits) * numValues + 7) / 8) + 1; // pad/flush overhead
			}

			const uint8_t widthBytes = (neededBits <= 8) ? 1 : (neededBits <= 16) ? 2 : (neededBits <= 32) ? 4 : 8;
			const size_t typedBytes = numValues * static_cast<size_t>(widthBytes);

			const size_t resBytes = std::min(packedBytes, typedBytes);
			const size_t totalBytes = kHeaderBytes + refBytes + resBytes;
			bestBytes = std::min(bestBytes, totalBytes);
		}

		return static_cast<double>(bestBytes) * 8.0;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::AdaptiveFrameOfReference;
	}
};

class AdaptiveFramedBitPrefixCostModel final : public EncodingCostModel {
public:
	static constexpr size_t kHeaderBytes = 20; // matches encoder header
	static constexpr size_t kFrameRecordBytes = 1 + sizeof(uint64_t) + 4; // prefixBits + prefixValue + suffixBytes

	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (numValues == 0) {
			return 0.0;
		}

		size_t bestBytes = std::numeric_limits<size_t>::max();
		const uint8_t cappedWidth = static_cast<uint8_t>(std::min<size_t>(bitWidth, 64));
		// Only consider the frame sizes the encoder supports: 8,16,32,64,128 (indices 0-4)
		for (size_t i = 0; i < 5 && i < SegmentMetrics::kFrameCandidateCount; ++i) {
			const size_t frame = MetricCollector<uint64_t>::kFrameCandidates[i];
			const size_t numFrames = (numValues + frame - 1) / frame;

			const double avgSuffix = metrics.frameAvgSuffixBits[i];
			const uint8_t maxSuffix = metrics.frameMaxSuffixBits[i];
			uint8_t suffixBits = avgSuffix > 0.0
				? static_cast<uint8_t>(std::ceil(avgSuffix))
				: (maxSuffix == 0 ? 0 : maxSuffix);
			suffixBits = maxSuffix == 0 ? 0 : maxSuffix; // for cost estimation, use max-based width since avg can be unreliable and lead to bad choices
			suffixBits = std::min<uint8_t>(suffixBits, cappedWidth);

			const size_t payloadBytes = suffixBits == 0
				? 0
				: ((static_cast<size_t>(suffixBits) * numValues) + 7) / 8;
			const size_t totalBytes = kHeaderBytes + numFrames * kFrameRecordBytes + payloadBytes;
			bestBytes = std::min(bestBytes, totalBytes);
		}

		return static_cast<double>(bestBytes) * 8.0;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::AdaptiveFramedBitPrefix;
	}
};

} // namespace encodings::encoders::selectors::costs