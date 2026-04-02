#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace encodings::encoders::selectors {

template <typename TIn = uint64_t, typename TOut = uint64_t>
class BitRangeSegmentBuilder {
public:
	explicit BitRangeSegmentBuilder(const std::vector<TIn>& data)
		: data_(data), currentValues_(data.size(), 0), bitStart_(-1), bitEnd_(-1) {}

	void reset(int bitStart) {
		if (bitStart < 0 || bitStart >= static_cast<int>(sizeof(TIn) * 8)) {
			throw std::out_of_range("SegmentBuilder::reset: bit_start out of range");
		}
		bitStart_ = bitStart;
		bitEnd_ = bitStart;

		const size_t n = data_.size();
		currentValues_.assign(n, 0);
		for (size_t i = 0; i < n; ++i) {
			currentValues_[i] = static_cast<TOut>((data_[i] >> bitStart_) & static_cast<TIn>(1));
		}
	}

	void extend(int bitEnd) {
		if (bitStart_ < 0) {
			throw std::logic_error("SegmentBuilder::extend: call reset() first");
		}
		if (bitEnd < bitStart_ || bitEnd >= static_cast<int>(sizeof(TIn) * 8)) {
			throw std::out_of_range("SegmentBuilder::extend: bit_end out of range");
		}
		if (bitEnd <= bitEnd_) {
			return; // no-op if already extended to this bit or beyond
		}

		const size_t n = data_.size();
		for (int b = bitEnd_ + 1; b <= bitEnd; ++b) {
			const int shift = b - bitStart_;
			const TOut mask_shift = static_cast<TOut>(1) << shift;
			for (size_t i = 0; i < n; ++i) {
				const TOut bit = static_cast<TOut>((data_[i] >> b) & static_cast<TIn>(1));
				currentValues_[i] |= (bit * mask_shift);
			}
		}

		bitEnd_ = bitEnd;
	}

	const std::vector<TOut>& values() const {
		return currentValues_;
	}

private:
	const std::vector<TIn>& data_;
	std::vector<TOut> currentValues_;
	int bitStart_;
	int bitEnd_;
};

} // namespace encodings::encoders::selectors
