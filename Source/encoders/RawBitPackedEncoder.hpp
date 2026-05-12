#pragma once

#include <span>
#include <vector>
#include <optional>
#include <cstring>
#include <type_traits>
#include <bit>
#include <limits>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

/**
 * @brief Bit-packed raw encoder for integral types.
 *
 * Detects the minimum number of bits needed to represent the value range
 * (using min as the base), then packs deltas into contiguous uint64_t words.
 * Format:
 *   [element_count (8 bytes)]
 *   [bit_width (1 byte)]
 *   [base_value (sizeof(T) bytes)]
 *   [packed words (ceil(n * bit_width / 64) * 8 bytes, little-endian)]
 *
 * bit_width == 0 means all values equal to base_value (no payload).
 */
template<typename T>
	requires std::is_integral_v<T>
class RawBitPackedEncoder : public Codec<T> {
public:
	EncodedData encode(std::span<const T> data) override {
		EncodedData result;

		const size_t n = data.size();
		const size_t headerSize = sizeof(size_t) + sizeof(uint8_t) + sizeof(T);

		if (n == 0) {
			result.data().resize(headerSize);
			writeHeader(result.data().data(), 0, 0, static_cast<T>(0));
			fillMetadata(result.metadata(), n, headerSize, 0);
			return result;
		}

		const auto [minVal, maxVal] = minMax(data);
		const uint64_t range = valueRange(minVal, maxVal);
		const uint8_t bitWidth = range == 0 ? 0 : static_cast<uint8_t>(std::bit_width(range));

		const size_t bitsTotal = static_cast<size_t>(bitWidth) * n;
		const size_t wordCount = bitWidth == 0 ? 0 : ((bitsTotal + 63) / 64);
		const size_t payloadBytes = wordCount * sizeof(uint64_t);
		result.data().resize(headerSize + payloadBytes, 0);

		uint8_t* writePtr = result.data().data();
		writeHeader(writePtr, n, bitWidth, minVal);
		writePtr += headerSize;

		if (bitWidth > 0) {
			std::vector<uint64_t> words(wordCount, 0);
			packValues(data, minVal, bitWidth, words.data());
			std::memcpy(writePtr, words.data(), payloadBytes);
		}

		fillMetadata(result.metadata(), n, headerSize + payloadBytes, bitWidth);
		return result;
	}

	std::vector<T> decodeAll(const EncodedData& encoded) override {
		if (encoded.size() < headerSize()) {
			return {};
		}

		size_t count;
		uint8_t bitWidth;
		T base;
		readHeader(encoded.data().data(), count, bitWidth, base);

		if (count == 0) {
			return {};
		}

		if (bitWidth == 0) {
			return std::vector<T>(count, base);
		}

		const uint8_t* payload = encoded.data().data() + headerSize();
		return unpackValues(payload, count, bitWidth, base);
	}

	void decodeAllInto(const EncodedData& encoded, T* dst, size_t n) override {
		if (encoded.size() < headerSize()) {
			if (n != 0) throw std::runtime_error("RawBitPackedEncoder::decodeAllInto: empty buffer, n!=0");
			return;
		}
		size_t count;
		uint8_t bitWidth;
		T base;
		readHeader(encoded.data().data(), count, bitWidth, base);
		if (count != n) [[unlikely]]
			throw std::runtime_error("RawBitPackedEncoder::decodeAllInto: size mismatch");
		if (bitWidth == 0) { std::fill(dst, dst + n, base); return; }
		const uint8_t* payload = encoded.data().data() + headerSize();
		unpackValuesInto(payload, count, bitWidth, base, dst);
	}

	std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
		if (encoded.size() < headerSize()) {
			return std::nullopt;
		}

		size_t count;
		uint8_t bitWidth;
		T base;
		readHeader(encoded.data().data(), count, bitWidth, base);

		if (index >= count) {
			return std::nullopt;
		}

		if (bitWidth == 0) {
			return base;
		}

		const uint8_t* payload = encoded.data().data() + headerSize();
		const uint64_t value = extractValue(payload, bitWidth, index);
		return addBase(value, base);
	}

	std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
		if (encoded.size() < headerSize()) {
			return {};
		}

		size_t count;
		uint8_t bitWidth;
		T base;
		readHeader(encoded.data().data(), count, bitWidth, base);

		if (start >= count) {
			return {};
		}
		end = std::min(end, count);
		const size_t outCount = end - start;

		if (bitWidth == 0) {
			return std::vector<T>(outCount, base);
		}

		const uint8_t* payload = encoded.data().data() + headerSize();
		std::vector<T> out;
		out.reserve(outCount);
		size_t bitPos = static_cast<size_t>(bitWidth) * start;
		const uint64_t mask = bitMask(bitWidth);
		for (size_t i = 0; i < outCount; ++i, bitPos += bitWidth) {
			out.push_back(addBase(extractAt(payload, bitPos, bitWidth, mask), base));
		}
		return out;
	}

	EncodingType encodingType() const override {
		return EncodingType::BitPacking;
	}

	std::string name() const override {
		return "RawBitPacked";
	}

	EncodingProperties properties() const override {
		return EncodingProperties(EncodingProperty::RandomAccess)
			| EncodingProperty::Lossless
			| EncodingProperty::PreservesOrder
			| EncodingProperty::Vectorizable
			| EncodingProperty::LowMemoryOverhead;
	}

	size_t estimateEncodedSize(size_t elementCount) const override {
		// Pessimistic: assume 64-bit width
		const uint8_t bitWidth = 64;
		const size_t bitsTotal = static_cast<size_t>(bitWidth) * elementCount;
		const size_t words = (bitsTotal + 63) / 64;
		return headerSize() + words * sizeof(uint64_t);
	}

private:
	static constexpr size_t headerSize() {
		return sizeof(size_t) + sizeof(uint8_t) + sizeof(T);
	}

	static void writeHeader(uint8_t* dst, size_t count, uint8_t bitWidth, T base) {
		std::memcpy(dst, &count, sizeof(size_t));
		dst += sizeof(size_t);
		std::memcpy(dst, &bitWidth, sizeof(uint8_t));
		dst += sizeof(uint8_t);
		std::memcpy(dst, &base, sizeof(T));
	}

	static void readHeader(const uint8_t* src, size_t& count, uint8_t& bitWidth, T& base) {
		std::memcpy(&count, src, sizeof(size_t));
		src += sizeof(size_t);
		std::memcpy(&bitWidth, src, sizeof(uint8_t));
		src += sizeof(uint8_t);
		std::memcpy(&base, src, sizeof(T));
	}

	void fillMetadata(encodings::EncodingMetadata& meta, size_t count, size_t compressedSize, uint8_t bitWidth) {
		meta.encodingName = "RawBitPacked";
		meta.dataType = Codec<T>::dataType();
		meta.elementCount = count;
		meta.compressedSize = compressedSize;
		meta.uncompressedSize = count * sizeof(T);
		meta.supportsRandomAccess = true;
		meta.customMetadata["bit_width"] = std::to_string(bitWidth);
	}

	template<typename U>
	static auto minMax(std::span<const U> data) {
		U minV = data[0];
		U maxV = data[0];
		for (const auto& v : data) {
			if (v < minV) minV = v;
			if (v > maxV) maxV = v;
		}
		return std::pair<U, U>{minV, maxV};
	}

	static uint64_t valueRange(T minV, T maxV) {
		using U = std::make_unsigned_t<T>;
		const U uMin = static_cast<U>(minV);
		const U uMax = static_cast<U>(maxV);
		return static_cast<uint64_t>(uMax - uMin);
	}

	static uint64_t bitMask(uint8_t width) {
		if (width == 64) return std::numeric_limits<uint64_t>::max();
		return (uint64_t{1} << width) - 1u;
	}

	static size_t payloadWordCount(size_t n, uint8_t bitWidth) {
		if (bitWidth == 0) return 0;
		return ((static_cast<size_t>(bitWidth) * n) + 63) / 64;
	}

	static void packValues(std::span<const T> data, T base, uint8_t bitWidth, uint64_t* outWords) {
		const uint64_t mask = bitMask(bitWidth);
		size_t bitPos = 0;

		for (const auto& v : data) {
			const uint64_t delta = deltaFromBase(v, base);

			if (bitWidth == 64) {
				const size_t wordIdx = bitPos >> 6;
				outWords[wordIdx] = delta;
				bitPos += 64;
				continue;
			}

			const size_t wordIdx = bitPos >> 6;
			const uint32_t offset = static_cast<uint32_t>(bitPos & 63u);
			const uint64_t value = delta & mask;

			outWords[wordIdx] |= value << offset;
			const uint32_t spill = offset + bitWidth;
			if (spill > 64) {
				outWords[wordIdx + 1] |= value >> (64 - offset);
			}

			bitPos += bitWidth;
		}
	}

	static uint64_t extractAt(const uint64_t* words, size_t bitPos, uint8_t bitWidth, uint64_t mask) {
		if (bitWidth == 64) {
			const size_t wordIdx = bitPos >> 6;
			return words[wordIdx];
		}

		const size_t wordIdx = bitPos >> 6;
		const uint32_t offset = static_cast<uint32_t>(bitPos & 63u);
		uint64_t value = words[wordIdx] >> offset;
		const uint32_t spill = offset + bitWidth;
		if (spill > 64) {
			value |= words[wordIdx + 1] << (64 - offset);
		}
		return value & mask;
	}

	static uint64_t loadWord(const uint8_t* payload, size_t wordIdx) {
		uint64_t w;
		std::memcpy(&w, payload + wordIdx * sizeof(uint64_t), sizeof(uint64_t));
		return w;
	}

	// Overload that reads directly from the raw byte payload via unaligned loads
	// (memcpy-based, compiles to a single movq on x86-64). Allows callers to skip
	// copying the payload into an intermediate uint64_t[] buffer.
	static uint64_t extractAt(const uint8_t* payload, size_t bitPos, uint8_t bitWidth, uint64_t mask) {
		if (bitWidth == 64) {
			return loadWord(payload, bitPos >> 6);
		}
		const size_t wordIdx = bitPos >> 6;
		const uint32_t offset = static_cast<uint32_t>(bitPos & 63u);
		uint64_t value = loadWord(payload, wordIdx) >> offset;
		const uint32_t spill = offset + bitWidth;
		if (spill > 64) {
			value |= loadWord(payload, wordIdx + 1) << (64 - offset);
		}
		return value & mask;
	}

	static uint64_t extractValue(const uint8_t* payload, uint8_t bitWidth, size_t index) {
		const size_t bitPos = static_cast<size_t>(bitWidth) * index;
		const uint64_t mask = bitMask(bitWidth);

		if (bitWidth == 64) {
			const size_t wordIdx = bitPos >> 6;
			return loadWord(payload, wordIdx);
		}

		const size_t wordIdx = bitPos >> 6;
		const uint32_t offset = static_cast<uint32_t>(bitPos & 63u);
		uint64_t value = loadWord(payload, wordIdx) >> offset;
		const uint32_t spill = offset + bitWidth;
		if (spill > 64) {
			value |= loadWord(payload, wordIdx + 1) << (64 - offset);
		}
		return value & mask;
	}

	static void unpackValuesInto(const uint8_t* payload, size_t count, uint8_t bitWidth, T base, T* dst) {
		const uint64_t mask = bitMask(bitWidth);
		size_t bitPos = 0;
		for (size_t i = 0; i < count; ++i, bitPos += bitWidth) {
			dst[i] = addBase(extractAt(payload, bitPos, bitWidth, mask), base);
		}
	}

	static std::vector<T> unpackValues(const uint8_t* payload, size_t count, uint8_t bitWidth, T base) {
		// reserve avoids zero-initialisation that vector<T>(count) would perform.
		std::vector<T> out;
		out.reserve(count);
		const uint64_t mask = bitMask(bitWidth);
		size_t bitPos = 0;
		for (size_t i = 0; i < count; ++i, bitPos += bitWidth) {
			out.push_back(addBase(extractAt(payload, bitPos, bitWidth, mask), base));
		}
		return out;
	}

	static uint64_t deltaFromBase(T value, T base) {
		using U = std::make_unsigned_t<T>;
		return static_cast<uint64_t>(static_cast<U>(value) - static_cast<U>(base));
	}

	static T addBase(uint64_t delta, T base) {
		using U = std::make_unsigned_t<T>;
		return static_cast<T>(static_cast<U>(base) + static_cast<U>(delta));
	}
};

} // namespace encodings::encoders
