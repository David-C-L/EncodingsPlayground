#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>

namespace encodings::generators::samplers {

template <typename TIn = uint64_t>
class StreamSampler {
public:
	struct Config {
        double_t maxPercentage{0}; // 0-1, overrides maxSamples if set
		size_t maxSamples{0};
		size_t stride{1};
		size_t blockSize{0}; // if > 0, sample contiguous blocks of this size instead of single strided elements
	};

	static std::vector<TIn> sample(
		const std::vector<TIn>& input,
		const Config& config
	) {
		return sample(std::span<const TIn>(input), config);
	}

	static std::vector<TIn> sample(
		std::span<const TIn> input,
		const Config& config
	) {
		if (input.empty() || (config.maxSamples == 0 && config.maxPercentage <= 0.0)) {
			return {};
		}

		double_t pct = config.maxPercentage;
		if (pct > 1.0) pct = 1.0;
		else if (pct < 0.0) pct = 0.0;
		const size_t pctSamples = static_cast<size_t>(std::ceil(pct * static_cast<double_t>(input.size())));
		const size_t maxCount = config.maxSamples == 0 ? pctSamples : config.maxSamples;

		std::vector<TIn> out;
		out.reserve(std::min(maxCount, input.size()));

		// Block sampling: take contiguous windows to preserve local temporal structure.
		// This allows the MetricCollector to compute accurate per-frame residuals for
		// patterns that only appear within short windows (e.g. sorted Snowflake IDs).
		if (config.blockSize > 0) {
			const size_t numBlocks = std::max<size_t>(1, maxCount / config.blockSize);
			const size_t blockStride = std::max<size_t>(1, input.size() / numBlocks);
			for (size_t b = 0; b < numBlocks && out.size() < maxCount; ++b) {
				const size_t start = b * blockStride;
				const size_t end = std::min(start + config.blockSize, input.size());
				for (size_t j = start; j < end && out.size() < maxCount; ++j) {
					out.push_back(input[j]);
				}
			}
			return out;
		}

		// Stride sampling: take every step-th element uniformly across the stream.
		const size_t step = std::max<size_t>(1, config.stride);
		for (size_t i = 0; i < input.size() && out.size() < maxCount; i += step) {
			out.push_back(input[i]);
		}

		return out;
	}
};

} // namespace encodings::generators::samplers
