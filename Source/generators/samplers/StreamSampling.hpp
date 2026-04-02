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
	};

	static std::vector<TIn> sample(
		const std::vector<TIn>& input,
		const Config& config
	) {
		if (input.empty() || (config.maxSamples == 0 && config.maxPercentage <= 0.0)) {
			return {};
		}

		const size_t step = std::max<size_t>(1, config.stride);
		const size_t estimated = (input.size() + step - 1) / step;
		double_t pct = config.maxPercentage;
		if (pct > 1.0) {
			pct = 1.0;
		} else if (pct < 0.0) {
			pct = 0.0;
		}
		const size_t pctSamples = static_cast<size_t>(std::ceil(pct * input.size()));
		const size_t maxCount = config.maxSamples == 0 ? pctSamples : config.maxSamples;
		const size_t reserve = std::min(maxCount, estimated);

		std::vector<TIn> out;
		out.reserve(reserve);

		for (size_t i = 0; i < input.size() && out.size() < maxCount; i += step) {
			out.push_back(input[i]);
		}

		return out;
	}

	static std::vector<TIn> sample(
		std::span<const TIn> input,
		const Config& config
	) {
		if (input.empty() || (config.maxSamples == 0 && config.maxPercentage <= 0.0)) {
			return {};
		}

		const size_t step = std::max<size_t>(1, config.stride);
		const size_t estimated = (input.size() + step - 1) / step;
		double_t pct = config.maxPercentage;
		if (pct > 1.0) {
			pct = 1.0;
		} else if (pct < 0.0) {
			pct = 0.0;
		}
		const size_t pctSamples = static_cast<size_t>(std::ceil(pct * input.size()));
		const size_t maxCount = config.maxSamples == 0 ? pctSamples : config.maxSamples;
		const size_t reserve = std::min(maxCount, estimated);

		std::vector<TIn> out;
		out.reserve(reserve);

		for (size_t i = 0; i < input.size() && out.size() < maxCount; i += step) {
			out.push_back(input[i]);
		}

		return out;
	}
};

} // namespace encodings::generators::samplers
