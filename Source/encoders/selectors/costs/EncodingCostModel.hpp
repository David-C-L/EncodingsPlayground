#pragma once

#include "encoders/selectors/MetricCollector.hpp"
#include "encoders/selectors/SubStreamReordererType.hpp"
#include "encodings/EncodingType.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

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

enum class CostModelDimension {
    Compression,      // bits (compressed size)
    EncodeSpeed,      // ns (total encode time)
    DecodeAllSpeed,   // ns (total sequential decode time)
    DecodeAtSpeed,    // ns (total cost for N uniform random accesses)
    DecodeRangeSpeed, // ns (total cost for N-element range decode with seek)
};

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

    // Which cost dimension this model measures (default: Compression).
    virtual CostModelDimension costModelDimension() const {
        return CostModelDimension::Compression;
    }
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

class AdaptiveDictionaryCostModel final : public EncodingCostModel {
public:
    static constexpr size_t kFileHeaderSize   = 16;
    static constexpr size_t kBlockDescSize    = 13;
    static constexpr size_t kKeysPaddingBytes =  8;
    static constexpr uint32_t kCandidates[]   = {
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
    };

    double computeCost(const SegmentMetrics& metrics,
                       size_t numValues,
                       size_t bitWidth) const override {
        if (numValues == 0 || metrics.uniqueCount == 0) return 0.0;

        // Replicate DictionaryCostModel's cardinality estimation.
        const double observedUniques  = static_cast<double>(metrics.uniqueCount);
        const double hllEstimate      = metrics.hllEstimatedCardinality;
        const double chao1Estimate    = detail::estimate_chao1(
                                            metrics.uniqueCount, metrics.f1, metrics.f2);
        const double chaoWeight       = (metrics.f2 < 5) ? 0.1 : 0.3;
        const double blended          = (1.0 - chaoWeight) * hllEstimate
                                      + chaoWeight * chao1Estimate;
        const double confidence       = std::min(
                                            1.0,
                                            std::sqrt(static_cast<double>(numValues) / 10000.0));
        double estimatedUniques       = (1.0 - confidence) * observedUniques
                                      + confidence * blended;
        estimatedUniques = std::max(estimatedUniques, observedUniques);
        estimatedUniques = std::min(estimatedUniques, static_cast<double>(numValues));
        estimatedUniques = std::min(
            estimatedUniques,
            static_cast<double>(MetricCollector<uint64_t>::kUniqueCountCap));
        const double C = std::max(1.0, estimatedUniques);

        const double storageBits = static_cast<double>(
            storageWidthBits(static_cast<uint8_t>(bitWidth)));

        double bestBits = std::numeric_limits<double>::max();

        for (const uint32_t bs : kCandidates) {
            const size_t numBlocks = (numValues + bs - 1) / bs;

            // Expected distinct values per block: balls-into-bins approximation.
            // Converges to bs when bs << C, to C when bs >> C.
            const double blockCard = std::max(1.0,
                std::min(C, C * (1.0 - std::exp(-static_cast<double>(bs) / C))));
            const uint64_t blockCardInt = static_cast<uint64_t>(std::ceil(blockCard));
            const uint32_t rawWidth = detail::ceil_log2_u64(blockCardInt);
            const uint32_t keyWidth = detail::clamp_dict_index_width(
                                          rawWidth == 0 ? 1u : rawWidth);

            const double headerBits      = static_cast<double>(kFileHeaderSize) * 8.0;
            const double indexBits       = static_cast<double>(numBlocks * kBlockDescSize) * 8.0;
            const double dictBitsPerBlk  = blockCard * storageBits;
            const double keyBitsPerBlk   = static_cast<double>(bs) * static_cast<double>(keyWidth);
            const double padBitsPerBlk   = static_cast<double>(kKeysPaddingBytes) * 8.0;
            const double totalBits       = headerBits + indexBits
                + static_cast<double>(numBlocks) * (dictBitsPerBlk + keyBitsPerBlk + padBitsPerBlk);

            bestBits = std::min(bestBits, totalBits);
        }

        return bestBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::AdaptiveDictionaryEncoding;
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

// Cost model for Finite State Entropy (tANS) entropy coding (sequential decode only).
// Wire format: 17-byte fixed header + numSymbols*(sizeof(T)+2) symbol table + payload.
// FSE achieves essentially the same entropy limit as Huffman, so cost estimates are
// very similar; the header is slightly larger (stores uint16_t normFreqs instead of
// uint8_t code lengths), but FSE avoids the power-of-2 code-length rounding loss on
// skewed distributions, giving a small advantage at high compression ratios.
class FSECostModel final : public EncodingCostModel {
public:
    // Fixed header: numElements(8) + tableLog(1) + numSymbols(4) + initState(4) = 17 bytes.
    static constexpr size_t kHeaderFixed = 17;

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;

        // Symbol-table entry: sizeof(T) + 2 bytes (sym + uint16_t normFreq).
        const size_t storageBytes = static_cast<size_t>(storageWidthBits(static_cast<uint8_t>(bitWidth))) / 8;
        const size_t numSymbols   = metrics.uniqueCountCapped
            ? MetricCollector<uint64_t>::kUniqueCountCap
            : metrics.uniqueCount;

		// Keep a "reasonable" FSE regime up to 65,536 symbols.
		// Above that, allow benchmarking but heavily penalize the model so the
		// selector strongly prefers alternatives unless no better option exists.
		constexpr size_t kReasonableFSESymbols = 1u << 16; // 65,536
		constexpr size_t kImplMaxFSESymbols    = 1u << 20; // kMaxTableLog = 20

		const double headerBits   = static_cast<double>(kHeaderFixed) * 8.0;
		const double symTableBits = static_cast<double>(numSymbols) *
									static_cast<double>(storageBytes + 2) * 8.0;
		const double rawBits      = static_cast<double>(numValues) * static_cast<double>(bitWidth);
        // FSE achieves ~entropy bits/symbol on the payload (same limit as Huffman).
        const double payloadBits  = metrics.entropyEstimate * static_cast<double>(numValues);

		if (numSymbols > kImplMaxFSESymbols) {
			return rawBits * 16.0;
		}

		if (numSymbols > kReasonableFSESymbols) {
			const double over =
				static_cast<double>(numSymbols - kReasonableFSESymbols)
				/ static_cast<double>(kReasonableFSESymbols);
			const double multiplier = 1.0 + 12.0 * over * over;
			return (headerBits + symTableBits + payloadBits) * multiplier;
		}

        return headerBits + symTableBits + payloadBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::FSEEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::FreqStats);
    }
};

// Cost model for FrequencyPartitionEncoding.
// Tiers use non-power-of-2 key widths: tier t gets key width ceil(log2(tierSize)).
// Tiers are pruned when their bitmap+dict cost exceeds savings over raw fallback storage.
// Cost = header + included-tier bitmaps + dict + keys + fallback raw storage.
class FrequencyPartitionCostModel final : public EncodingCostModel {
public:
    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;

        const size_t storageTypeBits = storageWidthBits(static_cast<uint8_t>(bitWidth));
        // TEMP DEBUG
        // if (bitWidth >= 33 && bitWidth <= 64) {
        //     std::cout << "[FreqPartCost] bw=" << bitWidth << " N=" << numValues
        //               << " unique=" << metrics.uniqueCount << " capped=" << metrics.uniqueCountCapped
        //               << " entropy=" << metrics.entropyEstimate << std::endl;
        // }
        const size_t maxKeyBits      = storageTypeBits / 2;

        // Tiers use key widths 1..maxKeyBits (non-power-of-2).
        // Tier t (0-indexed) has capacity 2^(t+1) and key width (t+1) bits.
        const size_t numTierDefs = maxKeyBits;
        if (numTierDefs == 0) return static_cast<double>(numValues * bitWidth);

        const size_t uniqueEst = metrics.uniqueCountCapped
            ? static_cast<size_t>(size_t{1} << std::min(bitWidth, size_t{20}))
            : metrics.uniqueCount;

        // 2^entropy gives the "active alphabet size" for skewed distributions.
        // When uniqueCount is capped the entropy estimate is computed from a binary
        // approximation (H(cap/n, 1-cap/n)) which severely underestimates the true
        // entropy. Fall back to uniqueEst so the coverage fraction stays sensible.
        const double effectiveUnique = metrics.uniqueCountCapped
            ? static_cast<double>(uniqueEst)
            : std::max(1.0, std::exp2(metrics.entropyEstimate));
        const double N               = static_cast<double>(numValues);
        const double storageBits     = static_cast<double>(storageTypeBits);

        // Simulate the greedy tier-fill and marginal-cost pruning the encoder applies.
        double totalCost   = (8 + 1 + 1 + 4) * 8.0; // header: numElements+numTiers+indexType+fallbackCount
        double coveredFrac = 0.0;   // fraction of element occurrences in included tiers
        size_t cumCapacity = 0;

        for (size_t t = 0; t < numTierDefs; ++t) {
            const size_t tierCap  = size_t{1} << (t + 1);
            const double kb       = static_cast<double>(t + 1); // key width in bits
            // Estimate how many unique values are newly assigned to this tier.
            const double prevCumUnique = std::min(static_cast<double>(cumCapacity),
                                                   static_cast<double>(uniqueEst));
            cumCapacity += tierCap;
            const double newCumUnique  = std::min(static_cast<double>(cumCapacity),
                                                   static_cast<double>(uniqueEst));
            const double tierDictSize  = newCumUnique - prevCumUnique;
            if (tierDictSize <= 0.0) break; // no more unique values to cover

            // Coverage fraction of element occurrences for this tier.
            const double newCumFrac = std::min(1.0, static_cast<double>(cumCapacity) / effectiveUnique);
            const double tierFrac   = newCumFrac - coveredFrac;
            const double tierCount  = tierFrac * N;

            // Marginal cost vs savings (mirror of encoder's includeTier logic).
            // numWords × 64 approximated as N (bitmap bits = N bits per tier).
            const double tierCostBits    = N + tierDictSize * storageBits + tierCount * kb;
            const double tierSavingsBits = tierCount * storageBits;
            if (tierSavingsBits <= tierCostBits) continue; // encoder would prune this tier

            // Include this tier.
            totalCost   += tierCostBits;
            coveredFrac  = newCumFrac;
        }

        // When no tier survived the marginal-cost check, FPE degrades to raw
        // fallback storage plus a fixed header — strictly worse than any direct
        // encoding.  Return infinity so the selector never chooses it.
        if (coveredFrac == 0.0)
            return std::numeric_limits<double>::infinity();

        // Fallback cost for uncovered element occurrences.
        const double fallbackBits = N * (1.0 - coveredFrac) * storageBits;
        totalCost += fallbackBits;

        return totalCost;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::FreqStats);
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::FrequencyPartitionEncoding;
    }
};

// ---------------------------------------------------------------------------
// BlockFrequencyPartitionCostModel
//
// Estimates compressed size for BlockFrequencyPartitionEncoder.  Like the
// global FrequencyPartitionCostModel but replaces per-tier N-bit bitmaps with
// a compact per-block tier-tag bitfield (1–2 bits/element).  Also adds per-
// block descriptor overhead (24 bytes × numBlocks) which is negligible at
// typical block sizes.
// ---------------------------------------------------------------------------
class BlockFrequencyPartitionCostModel final : public EncodingCostModel {
    // Mirrors BlockFrequencyPartitionEncoder constants
    static constexpr size_t   kBlockDescBits   = 24 * 8;  // 24 bytes per block descriptor
    static constexpr size_t   kFileHeaderBits  = 16 * 8;  // 16-byte file header
    static constexpr double   kDefaultBlockSz  = 256.0;
    static constexpr uint32_t kTierCaps[]      = {2, 4, 16};
    static constexpr uint32_t kTierKeyBits[]   = {1, 2, 4};
    static constexpr size_t   kNumTiers        = 3;

public:
    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;

        const size_t storageTypeBits = storageWidthBits(static_cast<uint8_t>(bitWidth));
        // BlockFP's tiers are fixed at 1-/2-/4-bit keys; no point going deeper than
        // the storage type allows.
        const size_t tierDefs = std::min(kNumTiers, storageTypeBits / 2);
        if (tierDefs == 0) return static_cast<double>(numValues * bitWidth);

        const size_t uniqueEst = metrics.uniqueCountCapped
            ? static_cast<size_t>(size_t{1} << std::min(bitWidth, size_t{20}))
            : metrics.uniqueCount;
        const double effectiveUnique = metrics.uniqueCountCapped
            ? static_cast<double>(uniqueEst)
            : std::max(1.0, std::exp2(metrics.entropyEstimate));
        const double N           = static_cast<double>(numValues);
        const double storageBits = static_cast<double>(storageTypeBits);

        // Simulate greedy tier-fill with marginal-cost pruning (same logic as FP,
        // but without the per-tier N-bit bitmap).
        double totalCost     = 0.0;
        double coveredFrac   = 0.0;
        size_t cumCapacity   = 0;
        uint8_t numActiveTiers = 0;

        for (size_t t = 0; t < tierDefs; ++t) {
            const size_t tierCap  = kTierCaps[t];
            const double kb       = static_cast<double>(kTierKeyBits[t]);
            const double prevCum  = std::min(static_cast<double>(cumCapacity),
                                             static_cast<double>(uniqueEst));
            cumCapacity += tierCap;
            const double newCum   = std::min(static_cast<double>(cumCapacity),
                                             static_cast<double>(uniqueEst));
            const double dictSize = newCum - prevCum;
            if (dictSize <= 0.0) break;

            const double newFrac  = std::min(1.0, static_cast<double>(cumCapacity) / effectiveUnique);
            const double tierFrac = newFrac - coveredFrac;
            const double tierCnt  = tierFrac * N;

            // No per-tier bitmap (BlockFP uses a shared tag bitfield instead).
            const double tierCost    = dictSize * storageBits + tierCnt * kb;
            const double tierSavings = tierCnt * storageBits;
            if (tierSavings <= tierCost) continue;

            totalCost += tierCost;
            coveredFrac = newFrac;
            numActiveTiers++;
        }

        if (coveredFrac == 0.0)
            return std::numeric_limits<double>::infinity();

        // Fallback cost for uncovered elements
        totalCost += N * (1.0 - coveredFrac) * storageBits;

        // Tag bitfield: 1–2 bits/element shared across all active tiers + fallback
        const uint8_t numTotalCodes = numActiveTiers + (coveredFrac < 1.0 ? 1u : 0u);
        const uint8_t tagBW = (numTotalCodes <= 1u) ? 0u : (numTotalCodes == 2u) ? 1u : 2u;
        totalCost += N * static_cast<double>(tagBW);

        // Per-block descriptor overhead + file header
        const double numBlocks = std::ceil(N / kDefaultBlockSz);
        totalCost += numBlocks * static_cast<double>(kBlockDescBits)
                   + static_cast<double>(kFileHeaderBits);

        return totalCost;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::FreqStats);
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::BlockFrequencyPartitionEncoding;
    }
};

// ---------------------------------------------------------------------------
// BlockFrequencyPartitionFORCostModel
//
// Estimates compressed size for BlockFrequencyPartitionEncoder<T, FORPrepass::GlobalFOR>.
//
// The GlobalFOR prepass subtracts the global minimum before block-wise FPE, but
// the encoder still stores residuals at full sizeof(T) width (no bit-packing).
// Consequently the tier-fill cost is identical to BlockFrequencyPartitionCostModel;
// the only structural difference is sizeof(T) extra bytes in the file header for
// the stored global minimum — negligible for typical stream sizes.
//
// The model exists as a distinct class so it:
//   (a) returns BlockFrequencyPartitionFOREncoding from encodingType(), letting the
//       AutoSubIntSplit DP distinguish it from the None variant, and
//   (b) declares MinMax as a required metric, signalling to the MetricCollector that
//       range information is needed (useful for future refinements that exploit the
//       narrowed residual range when a type-narrowing step is also applied).
// ---------------------------------------------------------------------------
class BlockFrequencyPartitionFORCostModel final : public EncodingCostModel {
    static constexpr size_t   kBlockDescBits  = 24 * 8;
    static constexpr size_t   kFileHeaderBits = 16 * 8; // globalMin overhead is negligible
    static constexpr double   kDefaultBlockSz = 256.0;
    static constexpr uint32_t kTierCaps[]     = {2, 4, 16};
    static constexpr uint32_t kTierKeyBits[]  = {1, 2, 4};
    static constexpr size_t   kNumTiers       = 3;

public:
    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;

        const size_t storageTypeBits = storageWidthBits(static_cast<uint8_t>(bitWidth));
        const size_t tierDefs = std::min(kNumTiers, storageTypeBits / 2);
        if (tierDefs == 0) return static_cast<double>(numValues * bitWidth);

        const size_t uniqueEst = metrics.uniqueCountCapped
            ? static_cast<size_t>(size_t{1} << std::min(bitWidth, size_t{20}))
            : metrics.uniqueCount;
        const double effectiveUnique = metrics.uniqueCountCapped
            ? static_cast<double>(uniqueEst)
            : std::max(1.0, std::exp2(metrics.entropyEstimate));
        const double N           = static_cast<double>(numValues);
        const double storageBits = static_cast<double>(storageTypeBits);

        double totalCost     = 0.0;
        double coveredFrac   = 0.0;
        size_t cumCapacity   = 0;
        uint8_t numActiveTiers = 0;

        for (size_t t = 0; t < tierDefs; ++t) {
            const size_t tierCap  = kTierCaps[t];
            const double kb       = static_cast<double>(kTierKeyBits[t]);
            const double prevCum  = std::min(static_cast<double>(cumCapacity),
                                             static_cast<double>(uniqueEst));
            cumCapacity += tierCap;
            const double newCum   = std::min(static_cast<double>(cumCapacity),
                                             static_cast<double>(uniqueEst));
            const double dictSize = newCum - prevCum;
            if (dictSize <= 0.0) break;

            const double newFrac  = std::min(1.0, static_cast<double>(cumCapacity) / effectiveUnique);
            const double tierFrac = newFrac - coveredFrac;
            const double tierCnt  = tierFrac * N;

            const double tierCost    = dictSize * storageBits + tierCnt * kb;
            const double tierSavings = tierCnt * storageBits;
            if (tierSavings <= tierCost) continue;

            totalCost += tierCost;
            coveredFrac = newFrac;
            numActiveTiers++;
        }

        if (coveredFrac == 0.0)
            return std::numeric_limits<double>::infinity();

        totalCost += N * (1.0 - coveredFrac) * storageBits;

        const uint8_t numTotalCodes = numActiveTiers + (coveredFrac < 1.0 ? 1u : 0u);
        const uint8_t tagBW = (numTotalCodes <= 1u) ? 0u : (numTotalCodes == 2u) ? 1u : 2u;
        totalCost += N * static_cast<double>(tagBW);

        const double numBlocks = std::ceil(N / kDefaultBlockSz);
        totalCost += numBlocks * static_cast<double>(kBlockDescBits)
                   + static_cast<double>(kFileHeaderBits);

        return totalCost;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::FreqStats)
             | static_cast<MetricFlags>(MetricFlag::MinMax);
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::BlockFrequencyPartitionFOREncoding;
    }
};

// ---------------------------------------------------------------------------
// BlockFSECostModel
//
// Estimates compressed size for BlockFSEEncoder.
// Each block carries an independent FSE header + symbol table + ANS payload.
// The structural overhead (file header + block index) is amortised over all
// blocks; the per-block FSE header dominates for small blockSizes.
//
// Formula (all in bits):
//   structural   = (kFileHeaderSize + numBlocks x kBlockDescSize) x 8
//   fse_headers  = numBlocks x (kFSEHeaderFixed + uniqueEst x (storageBytes+2)) x 8
//   payload      = N x entropyEstimate
// where numBlocks = ceil(N / kEstBlockSize).
// ---------------------------------------------------------------------------
class BlockFSECostModel final : public EncodingCostModel {
    static constexpr size_t kFileHeaderSize = 16;  // N(8)+blockSize(4)+numBlocks(4)
    static constexpr size_t kBlockDescSize  =  8;  // payloadByteOffset(8) only
    static constexpr size_t kFSEHeaderFixed = 17;  // numElements(8)+tableLog(1)+numSymbols(4)+initState(4)
    static constexpr double kEstBlockSize   = 1024.0;

public:
    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;

        const size_t storageBytes = static_cast<size_t>(storageWidthBits(static_cast<uint8_t>(bitWidth))) / 8;
        const size_t numSymbols   = metrics.uniqueCountCapped
            ? MetricCollector<uint64_t>::kUniqueCountCap
            : metrics.uniqueCount;

        // Heavy penalty above 65536 symbols: FSE table grows unboundedly.
        constexpr size_t kReasonableFSESymbols = 1u << 16;
        constexpr size_t kImplMaxFSESymbols    = 1u << 20;
        if (numSymbols > kImplMaxFSESymbols) {
            return static_cast<double>(numValues) * static_cast<double>(bitWidth) * 16.0;
        }

        const double numBlocks = std::ceil(static_cast<double>(numValues) / kEstBlockSize);

        const double structuralBits = static_cast<double>(
            kFileHeaderSize + static_cast<size_t>(numBlocks) * kBlockDescSize) * 8.0;

        const double fseHeaderBits = numBlocks *
            static_cast<double>(kFSEHeaderFixed + numSymbols * (storageBytes + 2)) * 8.0;

        const double payloadBits = metrics.entropyEstimate * static_cast<double>(numValues);

        double cost = structuralBits + fseHeaderBits + payloadBits;

        if (numSymbols > kReasonableFSESymbols) {
            const double over = static_cast<double>(numSymbols - kReasonableFSESymbols)
                              / static_cast<double>(kReasonableFSESymbols);
            cost *= 1.0 + 12.0 * over * over;
        }
        return cost;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::FreqStats);
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::BlockFSEEncoding;
    }
};

// ---------------------------------------------------------------------------
// BlockFORFPECostModel
//
// Estimates compressed size for BlockFORFPEEncoder. Sweeps the five block-size
// candidates aligned to MetricCollector::kResidualFrameCandidates and returns
// the minimum expected cost.
// ---------------------------------------------------------------------------

class BlockFORFPECostModel final : public EncodingCostModel {
    static constexpr size_t   kFileHeaderBytes  = 16;
    static constexpr uint32_t kRankSampleStride = 64;
    // Block-size candidates aligned to kResidualFrameCandidates = {256,512,1024,2048,4096}
    static constexpr uint32_t kCandidates[5]    = {256, 512, 1024, 2048, 4096};

    // kBlockDescBytes per TIn type width (mirrors BlockFORFPEEncoder constexpr)
    static constexpr size_t descBytesForStorage(size_t storageBytes) {
        if (storageBytes <= 2) return 36;   // uint8_t or uint16_t
        if (storageBytes <= 4) return 40;   // uint32_t
        return 44;                          // uint64_t
    }

public:
    double computeCost(const SegmentMetrics& m, size_t N, size_t bitWidth) const override {
        if (N == 0) return 0.0;
        const size_t storageBytes = storageWidthBits(bitWidth) / 8;
        const size_t descBytes    = descBytesForStorage(storageBytes);

        double bestBits = std::numeric_limits<double>::max();

        for (size_t ci = 0; ci < 5; ++ci) {
            const uint32_t bs        = kCandidates[ci];
            const size_t   numBlocks = (N + bs - 1) / bs;

            // residualBits after FOR: max residual in a frame of this size.
            const uint8_t rawRB       = m.frameMaxResidualBits[ci];
            const double  residualBits = (rawRB == 0) ? 1.0 : static_cast<double>(rawRB);

            // Structural overhead (file header + per-block descriptors).
            const double structBits =
                static_cast<double>((kFileHeaderBytes + numBlocks * descBytes) * 8);

            // Effective bits/element after FPE tier partitioning on residuals.
            // For small cardinality: all residuals fit in ≤ kMaxTiers tiers → key width ≈ log2(uniq).
            // Otherwise: most elements go to fallback at residualBits each; use entropy
            // re-scaled to the residual space as a proxy for payload cost.
            double effectiveBitsPerElem;
            if (m.uniqueCount <= 16) {
                const double avgKeyBits = (m.uniqueCount <= 1)
                    ? 0.0
                    : std::ceil(std::log2(static_cast<double>(m.uniqueCount)));
                effectiveBitsPerElem = std::max(1.0, std::min(residualBits, avgKeyBits));
            } else {
                const double scaledEntropy = (bitWidth > 0)
                    ? m.entropyEstimate * residualBits / static_cast<double>(bitWidth)
                    : residualBits;
                effectiveBitsPerElem = std::max(1.0, std::min(residualBits, scaledEntropy));
            }

            // Rank sample table: ceil(bs/kRankSampleStride) × numTiers × uint16_t per block.
            const size_t estNumTiers = (m.uniqueCount <= 1) ? 0u
                : (m.uniqueCount <= 2)  ? 1u
                : (m.uniqueCount <= 6)  ? 2u
                : 3u;
            const double rankBits = static_cast<double>(
                numBlocks
                * ((bs + kRankSampleStride - 1) / kRankSampleStride)
                * estNumTiers
                * sizeof(uint16_t) * 8);

            const double totalBits =
                structBits
                + static_cast<double>(N) * effectiveBitsPerElem
                + rankBits;
            if (totalBits < bestBits) bestBits = totalBits;
        }
        return bestBits;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::FreqStats)
             | static_cast<MetricFlags>(MetricFlag::ResidualFrames);
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::BlockFORFPEEncoding;
    }
};

// ---------------------------------------------------------------------------
// Reorderer cost model interface and implementations
// ---------------------------------------------------------------------------

// Interface for sub-stream reorderer cost models.
// The selector evaluates (reorderer × base-encoding) combinations by applying
// a multiplicative cost discount + fixed overhead on top of each base cost.
class ISubStreamReordererCostModel {
public:
    virtual ~ISubStreamReordererCostModel() = default;

    virtual encodings::encoders::selectors::SubStreamReordererType reordererType() const = 0;

    // Fixed overhead in bits added by the reorderer (e.g. BWT primary indices).
    virtual double overheadBits(size_t numValues) const = 0;

    // Multiplicative discount applied to the base encoding cost after reordering.
    // < 1.0 = the reorderer improves compression; depends on data characteristics
    // and the base encoding chosen.
    virtual double costMultiplier(const SegmentMetrics& metrics,
                                  const EncodingCostModel& base,
                                  uint8_t bitWidth) const = 0;

    // Which SegmentMetrics fields this reorderer cost model reads.
    virtual MetricFlags requiredMetrics() const = 0;
};

// BWTReordererCostModel: estimates the benefit of windowed BWT (W=512) as a
// pre-processing layer before any base encoding.
class BWTReordererCostModel : public ISubStreamReordererCostModel {
    static constexpr size_t W = 512;
public:
    encodings::encoders::selectors::SubStreamReordererType reordererType() const override {
        return encodings::encoders::selectors::SubStreamReordererType::BWT512;
    }

    // BWT permutation overhead: one uint64_t primaryIndex per window.
    double overheadBits(size_t numValues) const override {
        return static_cast<double>((numValues / W + 1) * 64);
    }

    // BWT discount:
    //   - normalizedCardinality: 0 = all values are the same, 1 = all unique
    //   - Multiplier approaches 0.5 for very low cardinality (BWT creates long runs
    //     → significant compression gain over raw entropy coding).
    //   - Approaches 1.0 for fully random data (BWT provides no benefit).
    //   - Extra discount for RLE and Dictionary which exploit run structure post-BWT.
    double costMultiplier(const SegmentMetrics& metrics,
                          const EncodingCostModel& base,
                          uint8_t bitWidth) const override {
        const double maxCard = static_cast<double>(
            uint64_t{1} << std::min(bitWidth, static_cast<uint8_t>(62)));
        const double normCard =
            std::min(metrics.hllEstimatedCardinality, maxCard) / maxCard;
        // Base multiplier: 0.5 (low card) → 1.0 (high card)
        double mult = 0.5 + 0.5 * normCard;
        // Extra discount for encodings whose performance is most amplified by BWT's
        // context-grouping effect (RLE and Dictionary see longer runs and denser dictionaries)
        const auto enc = base.encodingType();
        if (enc == encodings::EncodingType::RunLengthEncoding ||
            enc == encodings::EncodingType::DictionaryEncoding) {
            mult *= 0.85;
        }
        return mult;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::FreqStats);
    }
};

// Cost model for MainlyConstantEncoding.
// Estimates based on a dense isCommon bitmap (64-bit-aligned) plus raw storage
// for the uncommon values.  The dominant fraction is estimated from Shannon
// entropy: p_max ≈ 2^(-H), which is a conservative lower bound.
class MainlyConstantCostModel final : public EncodingCostModel {
public:
    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;

        // Estimate dominant fraction from entropy: p_max ≈ 2^(-H).
        // Conservative lower bound — underestimates MC's benefit, avoiding false positives.
        const double dominantFraction = (metrics.entropyEstimate > 0.0)
            ? std::min(1.0, std::pow(2.0, -metrics.entropyEstimate))
            : 1.0;
        const double uncommonCount = static_cast<double>(numValues) * (1.0 - dominantFraction);

        // Dense isCommon bitmap: rounded up to 64-bit words.
        const double numWords = std::ceil(static_cast<double>(numValues) / 64.0);
        const double bitmapBits = numWords * 64.0;

        // Uncommon values stored at storage width.
        const uint8_t storageBits = storageWidthBits(static_cast<uint8_t>(bitWidth));
        const double otherValuesBits = uncommonCount * static_cast<double>(storageBits);

        // Header: elementCount(8B) + bitmapByteCount(4B) + uncommonCount(4B) +
        //         otherValuesEncodedSize(4B) + commonValue(sizeof(T))
        const double headerBits = (8 + 4 + 4 + 4) * 8.0 + static_cast<double>(storageBits);

        return headerBits + bitmapBits + otherValuesBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::MainlyConstantEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::FreqStats);
    }
};

// ---------------------------------------------------------------------------
// RangePackCostModel — decorator mirroring RangePackSectionCodec's encoder-side
// design (Source/encoders/RangePackEncoder.hpp): re-invokes an inner cost
// model at the narrower bit width implied by the segment's actual value
// range, rather than the section's nominal bitWidth, plus a small fixed
// header overhead (minVal storage + narrowedBits/N/innerBytes fields).
// ---------------------------------------------------------------------------

class RangePackCostModel final : public EncodingCostModel {
public:
    RangePackCostModel(std::unique_ptr<EncodingCostModel> inner, encodings::EncodingType selfType)
        : inner_(std::move(inner)), selfType_(selfType) {}

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        const uint32_t narrowedBits = std::max<uint32_t>(
            1, detail::ceil_log2_u64(metrics.range + 1));
        const double headerBits = static_cast<double>(storageWidthBits(static_cast<uint8_t>(bitWidth))) // minVal
                                 + 72.0; // N(8B)+narrowedBits(1B)+innerBytes(8B) header fields, in bits
        return headerBits + inner_->computeCost(metrics, numValues, std::min<size_t>(narrowedBits, bitWidth));
    }

    encodings::EncodingType encodingType() const override { return selfType_; }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::MinMax) | inner_->requiredMetrics();
    }

private:
    std::unique_ptr<EncodingCostModel> inner_;
    encodings::EncodingType selfType_;
};

// ---------------------------------------------------------------------------
// CascadingFORCostModel
//
// Estimates compressed size for CascadingFOREncoder<int64_t> with the default
// schedule used by makeCascadingFORSection (residualSchedule={{512,MIN}},
// referenceSchedule={{64,MIN}}, RawBitPackedEncoder<int64_t> leaves). Reuses
// SegmentMetrics::frameMaxResidualBits at the 512-frame-size candidate index
// (no new metric needed) for the residual term; the reference-array term is
// an acknowledged approximation (no dedicated reference-stream-width metric
// exists, so the same residual-width stat stands in) -- could mis-rank
// CascadingFrameOfReference for data where residual-width and reference-width
// behave very differently; worth validating empirically like every other
// cost model here.
// ---------------------------------------------------------------------------
class CascadingFORCostModel final : public EncodingCostModel {
public:
    // Index into kResidualFrameCandidates={256,512,1024,2048,4096}, selecting
    // which precomputed residual-width bucket approximates this instance's
    // assumed frameSize. Defaults to index 1 (512), matching
    // makeCascadingFORSection's default residualSchedule[0].frameSize=512.
    // RunLengthCascadingFORStartsCostModel constructs a SEPARATE instance with
    // frameSizeCandidateIdx=0 (256), the closest available bucket to
    // makeCascadingFORSectionForRunStarts's empirically-tuned frameSize=128
    // (no exact candidate exists for 128; kResidualFrameCandidates only goes
    // down to 256) -- still an approximation, just a smaller mismatch than
    // leaving it at 512.
    explicit CascadingFORCostModel(size_t frameSizeCandidateIdx = 1)
        : frameSizeCandidateIdx_(frameSizeCandidateIdx) {}

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        // Mirror the runtime layout: top header (3x uint64_t) + one level header (5x uint64_t).
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        const size_t frameSize = MetricCollector<uint64_t>::kResidualFrameCandidates[frameSizeCandidateIdx_];
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        const uint8_t spanBits = metrics.frameMaxResidualBits[frameSizeCandidateIdx_];
        const uint8_t resBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double resBits = static_cast<double>(numValues) * static_cast<double>(resBitsPerVal);

        // APPROXIMATION: the reference array (length numFrames) is itself
        // cascaded through referenceSchedule={64} then leaf-encoded via
        // RawBitPackedEncoder<int64_t>. No dedicated reference-array-width
        // metric exists, so its storage is approximated as numFrames values
        // at the same resBitsPerVal-derived width, capped to bitWidth.
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(resBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override { return encodings::EncodingType::CascadingFrameOfReference; }

    MetricFlags requiredMetrics() const override { return static_cast<MetricFlags>(MetricFlag::ResidualFrames); }

private:
    size_t frameSizeCandidateIdx_;
};

// ---------------------------------------------------------------------------
// CascadingFORBlockFPECostModel
//
// Same header/reference-array structure as CascadingFORCostModel, but
// delegates the residual-stream term to the already-existing
// BlockFrequencyPartitionCostModel instead of a flat raw-bitwidth estimate,
// since makeCascadingFORBlockFrequencyPartitionSection's residual leaf
// genuinely is BlockFrequencyPartitionEncoder, not RawBitPackedEncoder -- this
// is a MORE accurate estimate than CascadingFORCostModel's own, not an
// additional approximation.
// ---------------------------------------------------------------------------
class CascadingFORBlockFPECostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        const double resBits = blockFpeInner_.computeCost(metrics, numValues, bitWidth);

        // Reference-array term: same approximation as CascadingFORCostModel
        // (reference cascade + RawBitPackedEncoder leaf, not BlockFPE).
        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORBlockFrequencyPartitionEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames) | blockFpeInner_.requiredMetrics();
    }

private:
    BlockFrequencyPartitionCostModel blockFpeInner_{};
};

// ---------------------------------------------------------------------------
// CascadingFORFSECostModel / CascadingFORBlockFSECostModel / CascadingFORHuffmanCostModel
//
// Same header/reference-array structure as CascadingFORCostModel, delegating
// the residual-stream term to the already-existing FSECostModel /
// BlockFSECostModel / HuffmanCostModel respectively instead of a flat
// raw-bitwidth estimate, mirroring CascadingFORBlockFPECostModel's pattern.
// ---------------------------------------------------------------------------
class CascadingFORFSECostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        const double resBits = fseInner_.computeCost(metrics, numValues, bitWidth);

        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORFSEEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames) | fseInner_.requiredMetrics();
    }

private:
    FSECostModel fseInner_{};
};

class CascadingFORBlockFSECostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        const double resBits = blockFseInner_.computeCost(metrics, numValues, bitWidth);

        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORBlockFSEEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames) | blockFseInner_.requiredMetrics();
    }

private:
    BlockFSECostModel blockFseInner_{};
};

class CascadingFORHuffmanCostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        const double resBits = huffmanInner_.computeCost(metrics, numValues, bitWidth);

        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORHuffmanEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames) | huffmanInner_.requiredMetrics();
    }

private:
    HuffmanCostModel huffmanInner_{};
};

// ---------------------------------------------------------------------------
// CascadingFORPrevFSECostModel / CascadingFORPrevBlockFSECostModel /
// CascadingFORPrevHuffmanCostModel
//
// PREV-policy siblings of CascadingFORFSECostModel/CascadingFORBlockFSECostModel/
// CascadingFORHuffmanCostModel above -- same header/reference-array structure,
// but the residual term is computed by handing the inner cost model a PROXY
// SegmentMetrics whose frameMaxResidualBits/frameAvgResidualBits have been
// overwritten with frameMaxDeltaBits/frameAvgDeltaBits (from
// MetricFlag::DeltaFrames) before delegating.
//
// APPROXIMATION (flagged, matching this session's practice for every other
// new cost model): FSECostModel/BlockFSECostModel/HuffmanCostModel's entropy
// term is still driven by MetricFlag::FreqStats computed over RAW segment
// values, not deltas -- so the estimated entropy-coding benefit may be
// systematically off for data whose delta distribution's skew differs a lot
// from its raw distribution's. This is a reasonable first-cut proxy for the
// bit-width portion of those inner models' formulas; validate/refine via
// this benchmark's existing cost-model-accuracy diagnostic rather than
// building a delta-aware FreqStats metric upfront.
// ---------------------------------------------------------------------------
class CascadingFORPrevFSECostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        SegmentMetrics proxy = metrics;
        proxy.frameMaxResidualBits[kOuterFrameIdx] = metrics.frameMaxDeltaBits[kOuterFrameIdx];
        proxy.frameAvgResidualBits[kOuterFrameIdx] = metrics.frameAvgDeltaBits[kOuterFrameIdx];
        const double resBits = fseInner_.computeCost(proxy, numValues, bitWidth);

        // Reference-array term: UNCHANGED -- reference array stores raw
        // data[lo] anchors, not deltas.
        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORPrevFSEEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames)
             | static_cast<MetricFlags>(MetricFlag::DeltaFrames)
             | fseInner_.requiredMetrics();
    }

private:
    FSECostModel fseInner_{};
};

class CascadingFORPrevBlockFSECostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        SegmentMetrics proxy = metrics;
        proxy.frameMaxResidualBits[kOuterFrameIdx] = metrics.frameMaxDeltaBits[kOuterFrameIdx];
        proxy.frameAvgResidualBits[kOuterFrameIdx] = metrics.frameAvgDeltaBits[kOuterFrameIdx];
        const double resBits = blockFseInner_.computeCost(proxy, numValues, bitWidth);

        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORPrevBlockFSEEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames)
             | static_cast<MetricFlags>(MetricFlag::DeltaFrames)
             | blockFseInner_.requiredMetrics();
    }

private:
    BlockFSECostModel blockFseInner_{};
};

class CascadingFORPrevHuffmanCostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        SegmentMetrics proxy = metrics;
        proxy.frameMaxResidualBits[kOuterFrameIdx] = metrics.frameMaxDeltaBits[kOuterFrameIdx];
        proxy.frameAvgResidualBits[kOuterFrameIdx] = metrics.frameAvgDeltaBits[kOuterFrameIdx];
        const double resBits = huffmanInner_.computeCost(proxy, numValues, bitWidth);

        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORPrevHuffmanEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames)
             | static_cast<MetricFlags>(MetricFlag::DeltaFrames)
             | huffmanInner_.requiredMetrics();
    }

private:
    HuffmanCostModel huffmanInner_{};
};

// ---------------------------------------------------------------------------
// CascadingFORPrevFrequencyPartitionCostModel /
// CascadingFORPrevBlockFrequencyPartitionCostModel
//
// PREV-policy compositions with FrequencyPartitionEncoder/
// BlockFrequencyPartitionEncoder leaves. Unlike the entropy-coder trio above,
// FrequencyPartitionCostModel/BlockFrequencyPartitionCostModel (checked
// directly) never read frameMaxResidualBits at all -- both are driven
// entirely by MetricFlag::FreqStats (metrics.uniqueCount/entropyEstimate/
// uniqueCountCapped), simulating the encoders' own greedy tier-fill against
// the segment's frequency distribution. Substituting frameMaxDeltaBits (as
// above) would do nothing here, so these two instead substitute
// MetricFlag::DeltaFreqStats's deltaUniqueCount/deltaEntropyEstimate/
// deltaUniqueCountCapped fields into the proxy's uniqueCount/entropyEstimate/
// uniqueCountCapped slots.
//
// APPROXIMATION (flagged): DeltaFreqStats collects at ONE fixed frame size
// (MetricCollector::kDeltaFreqStatsFrameSize) rather than comparing multiple
// candidates the way DeltaFrames does -- see that constant's own doc for why.
// Keep it in sync with whatever frame size ends up baked into
// makeCascadingFORPrevFrequencyPartitionSection/
// makeCascadingFORPrevBlockFrequencyPartitionSection.
// ---------------------------------------------------------------------------
class CascadingFORPrevFrequencyPartitionCostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        SegmentMetrics proxy = metrics;
        proxy.uniqueCount       = metrics.deltaUniqueCount;
        proxy.uniqueCountCapped = metrics.deltaUniqueCountCapped;
        proxy.entropyEstimate   = metrics.deltaEntropyEstimate;
        const double resBits = fpeInner_.computeCost(proxy, numValues, bitWidth);

        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORPrevFrequencyPartitionEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames)
             | static_cast<MetricFlags>(MetricFlag::DeltaFreqStats)
             | fpeInner_.requiredMetrics();
    }

private:
    FrequencyPartitionCostModel fpeInner_{};
};

class CascadingFORPrevBlockFrequencyPartitionCostModel final : public EncodingCostModel {
public:
    static constexpr size_t kOuterFrameIdx = 1; // candidates[1] == 512

    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0) return 0.0;
        static constexpr double kHeaderBits = static_cast<double>(3 * sizeof(uint64_t)) * 8.0
                                             + static_cast<double>(5 * sizeof(uint64_t)) * 8.0;

        constexpr size_t frameSize = 512;
        const size_t numFrames = (numValues + frameSize - 1) / frameSize;

        SegmentMetrics proxy = metrics;
        proxy.uniqueCount       = metrics.deltaUniqueCount;
        proxy.uniqueCountCapped = metrics.deltaUniqueCountCapped;
        proxy.entropyEstimate   = metrics.deltaEntropyEstimate;
        const double resBits = blockFpeInner_.computeCost(proxy, numValues, bitWidth);

        const uint8_t spanBits = metrics.frameMaxResidualBits[kOuterFrameIdx];
        const uint8_t refBitsPerVal = spanBits == 0 ? 1 : spanBits;
        const double refBits = static_cast<double>(numFrames) *
            static_cast<double>(std::min<uint8_t>(refBitsPerVal, static_cast<uint8_t>(bitWidth)));

        return kHeaderBits + resBits + refBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding;
    }

    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::ResidualFrames)
             | static_cast<MetricFlags>(MetricFlag::DeltaFreqStats)
             | blockFpeInner_.requiredMetrics();
    }

private:
    BlockFrequencyPartitionCostModel blockFpeInner_{};
};

// ---------------------------------------------------------------------------
// RunLengthCascadingFORStartsCostModel
//
// Reuses RLECostModel's estimatedRunCount = numValues/avgRunLength logic for
// the runValues term (reimplemented inline rather than delegated, since
// RLECostModel::computeCost is monolithic and doesn't expose a decomposable
// sub-call the way RangePackCostModel's wrapped inner_ does), and delegates
// the runStarts term to CascadingFORCostModel evaluated on the runStarts
// stream's OWN size (numValues=estimatedRunCount, bitWidth=ceil_log2(numValues))
// -- NOT the original segment's numValues/bitWidth, since run-start positions
// have a completely different domain than run values. The inner
// CascadingFORCostModel call still reads frameMaxResidualBits computed over
// the ORIGINAL N-element segment as a proxy for the (unmaterialized)
// R-element run-starts stream's own residual-width behavior -- an
// acknowledged, likely-imperfect proxy (run-start positions are monotonically
// increasing with different bit-width dynamics than run values); worth
// revisiting if this measurably misleads the DP once benchmarked. Constructed
// with frameSizeCandidateIdx=0 (256) rather than the default 1 (512), the
// closest available bucket to makeCascadingFORSectionForRunStarts's actual,
// empirically-tuned frameSize=128 (see that factory's comment in
// SubIntEncodingUtils.hpp for the frame-size sweep this was based on).
// ---------------------------------------------------------------------------
class RunLengthCascadingFORStartsCostModel final : public EncodingCostModel {
public:
    double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
        if (numValues == 0 || metrics.avgRunLength <= 0.0) return 0.0;

        const double estimatedRunCountD = static_cast<double>(numValues) / metrics.avgRunLength;
        const size_t estimatedRunCount  = std::max<size_t>(1, static_cast<size_t>(std::llround(estimatedRunCountD)));

        const double headerBits    = static_cast<double>(3 * sizeof(size_t)) * 8.0;  // RunLengthEncoder's own header
        const double runValuesBits = estimatedRunCountD *
            static_cast<double>(storageWidthBits(static_cast<uint8_t>(bitWidth)));   // unchanged from RLECostModel

        const uint32_t startsBitWidth = std::max<uint32_t>(1,
            detail::ceil_log2_u64(numValues == 0 ? 0 : numValues - 1));
        const double runStartsBits = cascadingForInner_.computeCost(
            metrics, estimatedRunCount, std::min<uint32_t>(startsBitWidth, 64));

        return headerBits + runStartsBits + runValuesBits;
    }

    encodings::EncodingType encodingType() const override {
        return encodings::EncodingType::RunLengthCascadingFOREncoding;
    }
    MetricFlags requiredMetrics() const override {
        return static_cast<MetricFlags>(MetricFlag::RunStats) | static_cast<MetricFlags>(MetricFlag::ResidualFrames);
    }
private:
    CascadingFORCostModel cascadingForInner_{/*frameSizeCandidateIdx=*/0};
};

// ---------------------------------------------------------------------------
// Single-encoding compression model factory
// ---------------------------------------------------------------------------
// Creates the appropriate compression cost model for a given encoding type.
// Returns nullptr for unsupported types so callers can skip them gracefully.
inline std::unique_ptr<EncodingCostModel>
makeCompressionCostModel(encodings::EncodingType type) {
    switch (type) {
        case encodings::EncodingType::RawEncoding:
            return std::make_unique<RawCostModel>();
        case encodings::EncodingType::BitPacking:
            return std::make_unique<RawBitPackedCostModel>();
        case encodings::EncodingType::RunLengthEncoding:
            return std::make_unique<RLECostModel>();
        case encodings::EncodingType::FrameOfReference:
            return std::make_unique<FORCostModel>();
        case encodings::EncodingType::AdaptiveFrameOfReference:
            return std::make_unique<AdaptiveFORCostModel>();
        case encodings::EncodingType::AdaptiveFramedBitPrefix:
            return std::make_unique<AdaptiveFramedBitPrefixCostModel>();
        case encodings::EncodingType::DictionaryEncoding:
            return std::make_unique<DictionaryCostModel>();
        case encodings::EncodingType::AdaptiveDictionaryEncoding:
            return std::make_unique<AdaptiveDictionaryCostModel>();
        case encodings::EncodingType::HuffmanEncoding:
            return std::make_unique<HuffmanCostModel>();
        case encodings::EncodingType::LZ4:
            return std::make_unique<LZ4CostModel>();
        case encodings::EncodingType::FSEEncoding:
            return std::make_unique<FSECostModel>();
        case encodings::EncodingType::FrequencyPartitionEncoding:
            return std::make_unique<FrequencyPartitionCostModel>();
        case encodings::EncodingType::BlockFORFPEEncoding:
            return std::make_unique<BlockFORFPECostModel>();
        case encodings::EncodingType::BlockFrequencyPartitionFOREncoding:
            return std::make_unique<BlockFrequencyPartitionFORCostModel>();
        case encodings::EncodingType::RangePackFrequencyPartitionEncoding:
            return std::make_unique<RangePackCostModel>(
                std::make_unique<FrequencyPartitionCostModel>(),
                encodings::EncodingType::RangePackFrequencyPartitionEncoding);
        case encodings::EncodingType::RangePackBlockFrequencyPartitionEncoding:
            return std::make_unique<RangePackCostModel>(
                std::make_unique<BlockFrequencyPartitionCostModel>(),
                encodings::EncodingType::RangePackBlockFrequencyPartitionEncoding);
        case encodings::EncodingType::CascadingFrameOfReference:
            return std::make_unique<CascadingFORCostModel>();
        case encodings::EncodingType::CascadingFORBlockFrequencyPartitionEncoding:
            return std::make_unique<CascadingFORBlockFPECostModel>();
        case encodings::EncodingType::RunLengthCascadingFOREncoding:
            return std::make_unique<RunLengthCascadingFORStartsCostModel>();
        case encodings::EncodingType::CascadingFORFSEEncoding:
            return std::make_unique<CascadingFORFSECostModel>();
        case encodings::EncodingType::CascadingFORBlockFSEEncoding:
            return std::make_unique<CascadingFORBlockFSECostModel>();
        case encodings::EncodingType::CascadingFORHuffmanEncoding:
            return std::make_unique<CascadingFORHuffmanCostModel>();
        case encodings::EncodingType::CascadingFORPrevFSEEncoding:
            return std::make_unique<CascadingFORPrevFSECostModel>();
        case encodings::EncodingType::CascadingFORPrevBlockFSEEncoding:
            return std::make_unique<CascadingFORPrevBlockFSECostModel>();
        case encodings::EncodingType::CascadingFORPrevHuffmanEncoding:
            return std::make_unique<CascadingFORPrevHuffmanCostModel>();
        case encodings::EncodingType::CascadingFORPrevFrequencyPartitionEncoding:
            return std::make_unique<CascadingFORPrevFrequencyPartitionCostModel>();
        case encodings::EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding:
            return std::make_unique<CascadingFORPrevBlockFrequencyPartitionCostModel>();
        default:
            return nullptr;
    }
}

} // namespace encodings::encoders::selectors::costs