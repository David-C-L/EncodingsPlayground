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

using encodings::encoders::selectors::MetricFlag;
using encodings::encoders::selectors::MetricFlags;

class EncodingCostModel {
public:
	EncodingCostModel() = default;
    virtual ~EncodingCostModel() = default;

	// Compute the cost of encoding a segment based on its metrics.
	virtual double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const = 0;

    virtual encodings::EncodingType encodingType() const = 0;

	// Declare which SegmentMetrics fields this model reads.
	// IDSubStreamEncodingSelector unions these across all registered models and
	// passes the result to MetricCollector::compute() to skip unused work.
	virtual MetricFlags requiredMetrics() const = 0;
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

// Precomputed 2^(-r) table for r in [0, 64]. Replaces std::ldexp(1.0, -r) in the HLL
// harmonic-mean sum — avoids a libm call per register on the hot path.
// The maximum rank stored in a register is 64 - p + 1 where p is the HLL precision,
// so index 64 is the safe upper bound for any precision ≥ 1.
inline constexpr std::array<double, 65> kPow2Neg = []() constexpr {
	std::array<double, 65> t{};
	t[0] = 1.0;
	for (int i = 1; i < 65; ++i) t[i] = t[i - 1] * 0.5;
	return t;
}();


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

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::None);
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
		// 1024-element frame is at index 2 in kResidualFrameCandidates = {256,512,1024,2048,4096}.
		constexpr size_t kFrameIdx = 2; // kResidualFrameCandidates[2] == 1024
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

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::ResidualFrames);
	}
};

class DictionaryCostModel final : public EncodingCostModel {
public:
	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (metrics.uniqueCount == 0 || numValues == 0) {
			return 0.0;
		}

			const double observedUniques = static_cast<double>(metrics.uniqueCount);
			// Use the cardinality estimate precomputed in MetricCollector::compute() — avoids
			// re-running the 1024-iteration harmonic-mean sum in every cost model call.
			const double hllEstimate = metrics.hllEstimatedCardinality;
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

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::FreqStats);
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

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::RunStats);
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

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::MinMax);
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
		for (size_t i = 0; i < SegmentMetrics::kResidualFrameCandidateCount; ++i) {
			const size_t frame = MetricCollector<uint64_t>::kResidualFrameCandidates[i];
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

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::ResidualFrames);
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
		for (size_t i = 0; i < SegmentMetrics::kBitPrefixFrameCandidateCount; ++i) {
			const size_t frame = MetricCollector<uint64_t>::kBitPrefixFrameCandidates[i];
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

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::SuffixFrames);
	}
};

// Cost model for canonical Huffman entropy coding (sequential decode only).
// Wire format: 20-byte fixed header + numSymbols*(sizeof(T)+1) symbol table + payload.
// Cost estimate: header + symbol-table bytes + entropy * N bits for payload.
class HuffmanCostModel final : public EncodingCostModel {
public:
	// Fixed header: 8 (numElements) + 4 (numSymbols) + 8 (payloadBits) = 20 bytes.
	static constexpr size_t kHeaderFixed = 20;

	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (numValues == 0) return 0.0;

		// Symbol-table entry: sizeof(storage_type) bytes for the symbol + 1 byte for code length.
		const size_t storageBytes = static_cast<size_t>(storageWidthBits(static_cast<uint8_t>(bitWidth))) / 8;
		const size_t numSymbols   = metrics.uniqueCountCapped
			? MetricCollector<uint64_t>::kUniqueCountCap
			: metrics.uniqueCount;

		const double headerBits      = static_cast<double>(kHeaderFixed) * 8.0;
		const double symTableBits    = static_cast<double>(numSymbols) *
		                               static_cast<double>(storageBytes + 1) * 8.0;
		// Huffman achieves ~entropy bits/symbol on the payload.
		const double payloadBits     = metrics.entropyEstimate * static_cast<double>(numValues);

		return headerBits + symTableBits + payloadBits;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::HuffmanEncoding;
	}

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::FreqStats);
	}
};

// Cost model for LZ4 fast block compression.
// Uses only existing metrics:
// - entropyEstimate (FreqStats) as incompressibility signal
// - avgRunLength (RunStats) as local-repeat signal
// - range/bitWidth (MinMax) as spread signal
//
// This is intentionally conservative: for high-entropy data, cost stays near raw +
// lightweight framing overhead; for repetitive/low-entropy data, projected ratio improves.
class LZ4CostModel final : public EncodingCostModel {
public:
	static constexpr size_t kHeaderBytes = 24; // [N][uncompressedBytes][compressedBytes]

	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (numValues == 0) return 0.0;

		const double rawBits = static_cast<double>(numValues) *
			static_cast<double>(storageWidthBits(static_cast<uint8_t>(bitWidth)));
		const double headerBits = static_cast<double>(kHeaderBytes) * 8.0;

		// Normalize signals into [0,1].
		const double maxEntropy = std::max(1.0, static_cast<double>(bitWidth));
		const double entropyNorm = std::clamp(metrics.entropyEstimate / maxEntropy, 0.0, 1.0);
		const double repeatScore = std::clamp((metrics.avgRunLength - 1.0) / 16.0, 0.0, 1.0);
		const double rangeNorm = (bitWidth == 0)
			? 0.0
			: std::clamp(static_cast<double>(std::bit_width(metrics.range)) / static_cast<double>(bitWidth), 0.0, 1.0);

		// Lower is better (smaller compressed payload relative to raw).
		double ratio = 0.99;
		ratio -= 0.32 * (1.0 - entropyNorm); // low entropy helps
		ratio -= 0.18 * repeatScore;         // longer runs help match-finding
		ratio -= 0.10 * (1.0 - rangeNorm);   // tighter value spread tends to help

		// Keep estimate conservative for selector stability.
		ratio = std::clamp(ratio, 0.14, 1.03);

		return headerBits + rawBits * ratio;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::LZ4;
	}

	MetricFlags requiredMetrics() const override {
		return static_cast<MetricFlags>(MetricFlag::FreqStats)
			| MetricFlag::RunStats
			| MetricFlag::MinMax;
	}
};

} // namespace encodings::encoders::selectors::costs