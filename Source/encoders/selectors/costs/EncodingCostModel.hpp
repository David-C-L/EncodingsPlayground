#pragma once

#include "encoders/selectors/MetricCollector.hpp"
#include "encodings/EncodingType.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace encodings::encoders::selectors::costs {

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
		if (metrics.uniqueCountCapped) {
			return std::numeric_limits<double>::infinity();
		}

		const uint64_t uniques = static_cast<uint64_t>(metrics.uniqueCount);
		const uint32_t rawWidth = detail::ceil_log2_u64(uniques);
		const uint32_t indexWidth = detail::clamp_dict_index_width(rawWidth);
		const uint8_t valueStorageBits = storageWidthBits(static_cast<uint8_t>(bitWidth));
		const double dictBits = static_cast<double>(uniques) * static_cast<double>(valueStorageBits);
		// Bit-packed keys rounded up to whole bytes
		const double indexBits = static_cast<double>((numValues * static_cast<size_t>(indexWidth) + 7) / 8 * 8);
		// dict size + key-bytes size + key-bit-width byte
		const double headerBits = static_cast<double>(2 * sizeof(size_t) + 1) * 8.0;

		// If the index width is not narrower than the value storage width, dictionary cannot beat raw; bias toward raw.
		if (indexWidth >= valueStorageBits) {
			// Raw storage cost for this segment
			const double rawBits = static_cast<double>(numValues) * static_cast<double>(valueStorageBits);
			// Add header+dict overhead to ensure raw wins in this case.
			return rawBits + headerBits + dictBits;
		}

		return headerBits + dictBits + indexBits;
	}

	encodings::EncodingType encodingType() const override {
		return encodings::EncodingType::DictionaryEncoding;
	}
};

class RLECostModel final : public EncodingCostModel {
public:
	double computeCost(const SegmentMetrics& metrics, size_t numValues, size_t bitWidth) const override {
		if (numValues == 0) {
			return 0.0;
		}

		// Matches RunLengthEncoder layout: header (3 * sizeof(size_t)), runStarts (runCount * sizeof(size_t)), runValues (runCount * sizeof(T))
		const uint64_t runCount = metrics.runCount;
		const double headerBits = static_cast<double>(3 * sizeof(size_t)) * 8.0;
		const double runStartsBits = static_cast<double>(runCount) * static_cast<double>(sizeof(size_t) * 8u);
		const double runValuesBits = static_cast<double>(runCount) * static_cast<double>(storageWidthBits(static_cast<uint8_t>(bitWidth)));

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
			const uint8_t neededBits = avgBits > 0.0
				? static_cast<uint8_t>(std::ceil(avgBits))
				: (spanBits == 0 ? 1 : spanBits);

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