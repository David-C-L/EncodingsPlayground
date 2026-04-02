#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <ankerl/unordered_dense.h>

namespace encodings::encoders::selectors {

struct SegmentMetrics {
	uint64_t min{0};
	uint64_t max{0};
	uint64_t range{0};
	double mean{0.0};
	double variance{0.0};

	size_t uniqueCount{0};
	bool uniqueCountCapped{false};
	size_t runCount{0};

	// Additional metrics useful for estimating encoder sizes.
	uint64_t minDelta{0};
	uint64_t maxDelta{0};
    uint64_t minDelta1024Window{0};
    uint64_t maxDelta1024Window{0};
	double avgRunLength{0.0};
	size_t maxRunLength{0};
	size_t zeroCount{0};
	size_t nonzeroCount{0};
	uint8_t bitWidth{0};

	// Per-frame residual widths for AdaptiveFOR and suffix widths for prefix codecs (candidates below).
	static constexpr size_t kFrameCandidateCount = 10;
	std::array<uint8_t, kFrameCandidateCount> frameMaxResidualBits{};
	std::array<double, kFrameCandidateCount> frameAvgResidualBits{};
	std::array<uint8_t, kFrameCandidateCount> frameMaxSuffixBits{};
	std::array<double, kFrameCandidateCount> frameAvgSuffixBits{};

	// Optional
	double entropyEstimate{0.0};
};

// MAY WANT INPUT CONFIG TO SELECT WINDOW SIZE AND UNIQUE CAP
template <typename T = uint64_t>
    requires std::is_integral_v<T>
class MetricCollector {
public:
	static constexpr size_t kUniqueCountCap = 1 << 16; // stop tracking if too many uniques
	inline static constexpr std::array<size_t, SegmentMetrics::kFrameCandidateCount> kFrameCandidates{
		8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
	};

	SegmentMetrics compute(const std::vector<T>& values) const {
		SegmentMetrics out;
		out.frameMaxResidualBits.fill(0);
		out.frameAvgResidualBits.fill(0.0);
		out.frameMaxSuffixBits.fill(0);
		out.frameAvgSuffixBits.fill(0.0);
		if (values.empty()) {
			return out;
		}

		const size_t n = values.size();
		out.min = static_cast<uint64_t>(values[0]);
		out.max = static_cast<uint64_t>(values[0]);
		out.zeroCount = (values[0] == 0) ? 1 : 0;
		out.nonzeroCount = (values[0] == 0) ? 0 : 1;
		out.runCount = 1;
		out.uniqueCount = 0;
		out.minDelta = std::numeric_limits<uint64_t>::max();
		out.maxDelta = 0;
		out.minDelta1024Window = std::numeric_limits<uint64_t>::max();
		out.maxDelta1024Window = 0;
		out.maxRunLength = 1;

		ankerl::unordered_dense::set<T> uniqueSet;
		uniqueSet.reserve(std::min(n, kUniqueCountCap));
		bool uniqueCapped = false;
		uniqueSet.insert(values[0]);

		// Per-frame min/max trackers for each candidate frame size (AdaptiveFOR-style).
		std::array<uint64_t, SegmentMetrics::kFrameCandidateCount> frameMin;
		std::array<uint64_t, SegmentMetrics::kFrameCandidateCount> frameMax;
		std::array<uint64_t, SegmentMetrics::kFrameCandidateCount> frameFirst{};
		std::array<uint64_t, SegmentMetrics::kFrameCandidateCount> frameXor{};
		std::array<size_t, SegmentMetrics::kFrameCandidateCount> frameCount{};
		std::array<double, SegmentMetrics::kFrameCandidateCount> frameBitSum{};
		std::array<size_t, SegmentMetrics::kFrameCandidateCount> frameBitSeen{};
		std::array<std::vector<uint64_t>, SegmentMetrics::kFrameCandidateCount> frameXorHistory;
		std::array<std::vector<size_t>, SegmentMetrics::kFrameCandidateCount> frameSizeHistory;
		frameMin.fill(std::numeric_limits<uint64_t>::max());
		frameMax.fill(std::numeric_limits<uint64_t>::min());
		frameFirst.fill(0);
		frameXor.fill(0);

		const auto finalizeFrame = [&](size_t candidateIdx) {
			if (frameCount[candidateIdx] == 0) {
				return;
			}
			const uint64_t span = frameMax[candidateIdx] - frameMin[candidateIdx];
			const uint8_t bits = span == 0 ? 0 : static_cast<uint8_t>(std::bit_width(span));
			out.frameMaxResidualBits[candidateIdx] = std::max(out.frameMaxResidualBits[candidateIdx], bits);
			frameBitSum[candidateIdx] += static_cast<double>(bits);
			++frameBitSeen[candidateIdx];
			frameXorHistory[candidateIdx].push_back(frameXor[candidateIdx]);
			frameSizeHistory[candidateIdx].push_back(frameCount[candidateIdx]);
			frameCount[candidateIdx] = 0;
			frameMin[candidateIdx] = std::numeric_limits<uint64_t>::max();
			frameMax[candidateIdx] = std::numeric_limits<uint64_t>::min();
			frameXor[candidateIdx] = 0;
		};

		// Welford's algorithm
		double mean = static_cast<double>(values[0]);
		double m2 = 0.0;

		T prev = values[0];
		size_t currentRunLength = 1;
		size_t windowCount = 0;
		uint64_t windowMinDelta = std::numeric_limits<uint64_t>::max();
		uint64_t windowMaxDelta = 0;

		for (size_t i = 0; i < n; ++i) {
			const T v = values[i];
			const uint64_t uv = static_cast<uint64_t>(v);

			if (i > 0) {
				out.min = std::min(out.min, uv);
				out.max = std::max(out.max, uv);
				if (v == 0) {
					++out.zeroCount;
				} else {
					++out.nonzeroCount;
				}

				// Welford update
				double delta = static_cast<double>(v) - mean;
				mean += delta / static_cast<double>(i + 1);
				double delta2 = static_cast<double>(v) - mean;
				m2 += delta * delta2;

				// Runs
				if (v == prev) {
					++currentRunLength;
				} else {
					++out.runCount;
					out.maxRunLength = std::max(out.maxRunLength, currentRunLength);
					currentRunLength = 1;
				}

				// Deltas
				uint64_t deltaAbs = (uv >= static_cast<uint64_t>(prev))
					? (uv - static_cast<uint64_t>(prev))
					: (static_cast<uint64_t>(prev) - uv);
				out.minDelta = std::min(out.minDelta, deltaAbs);
				out.maxDelta = std::max(out.maxDelta, deltaAbs);

				windowMinDelta = std::min(windowMinDelta, deltaAbs);
				windowMaxDelta = std::max(windowMaxDelta, deltaAbs);
				++windowCount;
				if (windowCount == 1024) {
					out.minDelta1024Window = std::min(out.minDelta1024Window, windowMinDelta);
					out.maxDelta1024Window = std::max(out.maxDelta1024Window, windowMaxDelta);
					windowCount = 0;
					windowMinDelta = std::numeric_limits<uint64_t>::max();
					windowMaxDelta = 0;
				}

				// Unique values (capped)
				if (!uniqueCapped) {
					uniqueSet.insert(v);
					if (uniqueSet.size() > kUniqueCountCap) {
						uniqueCapped = true;
					}
				}
			}

			// Frame tracking (always runs for i==0 too)
			for (size_t c = 0; c < SegmentMetrics::kFrameCandidateCount; ++c) {
				if (frameCount[c] == 0) {
					frameFirst[c] = uv;
				}
				frameMin[c] = std::min(frameMin[c], uv);
				frameMax[c] = std::max(frameMax[c], uv);
				frameXor[c] |= (uv ^ frameFirst[c]);
				++frameCount[c];
				if (frameCount[c] == kFrameCandidates[c]) {
					finalizeFrame(c);
				}
			}

			prev = v;
		}

		// Finalize any partial frames
		for (size_t c = 0; c < SegmentMetrics::kFrameCandidateCount; ++c) {
			finalizeFrame(c);
		}

		for (size_t c = 0; c < SegmentMetrics::kFrameCandidateCount; ++c) {
			if (frameBitSeen[c] > 0) {
				out.frameAvgResidualBits[c] = frameBitSum[c] / static_cast<double>(frameBitSeen[c]);
			}
		}

		const uint8_t bitWidth = static_cast<uint8_t>(std::bit_width(out.max));
		for (size_t c = 0; c < SegmentMetrics::kFrameCandidateCount; ++c) {
			uint8_t maxSuffix = 0;
			double sumWeighted = 0.0;
			double totalVals = 0.0;
			for (size_t idx = 0; idx < frameXorHistory[c].size(); ++idx) {
				const uint64_t xorv = frameXorHistory[c][idx];
				const size_t fsize = frameSizeHistory[c][idx];
				uint8_t prefixBits;
				if (xorv == 0) {
					prefixBits = bitWidth;
				} else if (bitWidth == 64) {
					prefixBits = static_cast<uint8_t>(std::countl_zero(xorv));
				} else {
					prefixBits = std::min<uint8_t>(bitWidth, static_cast<uint8_t>(std::countl_zero(xorv << (64 - bitWidth))));
				}
				const uint8_t suffixBits = bitWidth > prefixBits ? static_cast<uint8_t>(bitWidth - prefixBits) : 0;
				maxSuffix = std::max<uint8_t>(maxSuffix, suffixBits);
				sumWeighted += static_cast<double>(suffixBits) * static_cast<double>(fsize);
				totalVals += static_cast<double>(fsize);
			}
			out.frameMaxSuffixBits[c] = maxSuffix;
			out.frameAvgSuffixBits[c] = (totalVals > 0.0) ? (sumWeighted / totalVals) : 0.0;
		}

		out.maxRunLength = std::max(out.maxRunLength, currentRunLength);
		out.mean = mean;
		out.variance = (n > 1) ? (m2 / static_cast<double>(n - 1)) : 0.0;
		out.range = out.max - out.min;
		out.avgRunLength = static_cast<double>(n) / static_cast<double>(out.runCount);
		out.uniqueCount = uniqueCapped ? (kUniqueCountCap + 1) : uniqueSet.size();
		out.uniqueCountCapped = uniqueCapped;
		out.bitWidth = static_cast<uint8_t>(std::bit_width(out.max));

		if (windowCount > 0) {
			out.minDelta1024Window = std::min(out.minDelta1024Window, windowMinDelta);
			out.maxDelta1024Window = std::max(out.maxDelta1024Window, windowMaxDelta);
		}

		if (out.minDelta == std::numeric_limits<uint64_t>::max()) {
			out.minDelta = 0;
		}
		if (out.minDelta1024Window == std::numeric_limits<uint64_t>::max()) {
			out.minDelta1024Window = 0;
		}

		return out;
	}
};

} // namespace encodings::encoders::selectors