#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace encodings::core {

/**
 * @brief Bit order convention used by BitWriter / BitReader.
 *
 * LSB  - bit 0 of each byte is the first logical bit written/read (default).
 * MSB  - bit 7 of each byte is the first logical bit written/read.
 */
enum class BitOrder { LSB, MSB };

// ============================================================================
// BitWriter
// ============================================================================

/**
 * @brief Packs arbitrarily-wide values (power-of-two bits, 1–16) into a
 *        byte buffer, one value at a time.
 *
 * Values are written left-to-right logically; within each byte the bit
 * order follows the BitOrder policy.
 *
 * Usage:
 *   BitWriter w(buf, BitOrder::LSB);
 *   w.write(val, 4);   // write a 4-bit code
 *   w.write(val, 12);  // write a 12-bit code
 *   w.flush();         // zero-pad the last partial byte
 */
class BitWriter {
public:
    /**
     * @param buf    Output buffer. Bytes are appended; the buffer need not be
     *               pre-sized (it grows via push_back / back-filling).
     * @param order  Bit order within each byte (default LSB).
     */
    explicit BitWriter(std::vector<uint8_t>& buf,
                       BitOrder order = BitOrder::LSB) noexcept
        : buf_(buf), order_(order) {}

    /**
     * @brief Write the @p width lowest bits of @p value into the stream.
     *
     * @param value  Value to write (only the lowest @p width bits are used).
     * @param width  Number of bits to write. Must be 1, 2, 4, 8, or 16.
     */
    void write(uint32_t value, uint32_t width) {
        // Mask to exactly `width` bits
        const uint32_t mask = (width == 32) ? ~0u : ((1u << width) - 1u);
        value &= mask;

        uint32_t bitsLeft = width;
        while (bitsLeft > 0) {
            // Ensure current byte exists
            if (bitOffset_ == 0) {
                buf_.push_back(0);
            }

            const uint32_t room = 8u - bitOffset_;   // bits free in current byte
            const uint32_t take = (bitsLeft < room) ? bitsLeft : room;

            if (order_ == BitOrder::LSB) {
                // Place the next `take` bits from LSB of value into current byte
                const uint8_t bits = static_cast<uint8_t>(value & ((1u << take) - 1u));
                buf_.back() |= static_cast<uint8_t>(bits << bitOffset_);
                value >>= take;
            } else {
                // MSB: place the top `take` bits of the remaining `bitsLeft` into
                // the current byte starting at position (7 - bitOffset_) downward.
                const uint32_t shift = bitsLeft - take;
                const uint8_t bits = static_cast<uint8_t>((value >> shift) & ((1u << take) - 1u));
                const uint32_t byteShift = room - take;
                buf_.back() |= static_cast<uint8_t>(bits << byteShift);
                // Clear the top bits we've consumed
                value &= (1u << shift) - 1u;
            }

            bitOffset_ = (bitOffset_ + take) & 7u;
            bitsLeft  -= take;
        }
    }

    /**
     * @brief Zero-pad any remaining bits in the current byte.
     * Safe to call multiple times; a no-op if already byte-aligned.
     */
    void flush() noexcept {
        bitOffset_ = 0;
    }

    /** Bit offset within the current (last) byte [0, 7]. */
    uint32_t bitOffset() const noexcept { return bitOffset_; }

    /** True when the writer is currently byte-aligned. */
    bool byteAligned() const noexcept { return bitOffset_ == 0; }

private:
    std::vector<uint8_t>& buf_;
    BitOrder              order_;
    uint32_t              bitOffset_{0};  // next free bit in buf_.back()
};

// ============================================================================
// BitReader
// ============================================================================

/**
 * @brief Reads arbitrarily-wide values (power-of-two bits, 1–16) from a
 *        byte buffer, mirroring the layout written by BitWriter.
 *
 * Usage:
 *   BitReader r(buf.data(), buf.size(), BitOrder::LSB);
 *   uint32_t code = r.read(4);
 */
class BitReader {
public:
    /**
     * @param data   Pointer to the byte buffer to read from.
     * @param size   Number of bytes in the buffer.
     * @param order  Bit order (must match the BitWriter that produced the data).
     */
    BitReader(const uint8_t* data, size_t size,
              BitOrder order = BitOrder::LSB) noexcept
        : data_(data), size_(size), order_(order) {}

    /**
     * @brief Read @p width bits from the stream and return them as uint32_t.
     *
     * @param width  Number of bits to read. Must be 1, 2, 4, 8, or 16.
     * @throws std::out_of_range if the buffer is exhausted.
     */
    uint32_t read(uint32_t width) {
        // Fast path: width fits in one byte (most common case for small codes)
        if (width <= 8u - bitOffset_) [[likely]] {
            if (bytePos_ >= size_) [[unlikely]]
                throw std::out_of_range("BitReader: read past end of buffer");
            uint32_t result;
            if (order_ == BitOrder::LSB) {
                result = (static_cast<uint32_t>(data_[bytePos_]) >> bitOffset_)
                         & ((1u << width) - 1u);
            } else {
                const uint32_t shift = 8u - bitOffset_ - width;
                result = (static_cast<uint32_t>(data_[bytePos_]) >> shift)
                         & ((1u << width) - 1u);
            }
            bitOffset_ += width;
            if (bitOffset_ == 8u) { bitOffset_ = 0; ++bytePos_; }
            return result;
        }

        // Slow path: spans two or more bytes
        uint32_t result   = 0;
        uint32_t bitsLeft = width;
        uint32_t outShift = 0;  // only used for LSB accumulation

        while (bitsLeft > 0) {
            if (bytePos_ >= size_) {
                throw std::out_of_range("BitReader: read past end of buffer");
            }

            const uint32_t avail = 8u - bitOffset_;  // bits remaining in current byte
            const uint32_t take  = (bitsLeft < avail) ? bitsLeft : avail;

            if (order_ == BitOrder::LSB) {
                const uint32_t bits =
                    (static_cast<uint32_t>(data_[bytePos_]) >> bitOffset_) &
                    ((1u << take) - 1u);
                result    |= bits << outShift;
                outShift  += take;
            } else {
                // MSB: the first logical bits are the high bits of the byte.
                const uint32_t shift = avail - take;
                const uint32_t bits =
                    (static_cast<uint32_t>(data_[bytePos_]) >> shift) &
                    ((1u << take) - 1u);
                result = (result << take) | bits;
            }

            bitOffset_ += take;
            bitsLeft   -= take;

            if (bitOffset_ == 8) {
                bitOffset_ = 0;
                ++bytePos_;
            }
        }

        return result;
    }

    /**
     * @brief Seek to an absolute bit position in the buffer.
     * Enables O(1) random access when combined with a pre-computed bit offset.
     *
     * @param bitPos  Absolute bit index (0 = first bit of first byte).
     * @throws std::out_of_range if bitPos is beyond the buffer.
     */
    void seekToBit(size_t bitPos) {
        bytePos_   = bitPos >> 3;
        bitOffset_ = static_cast<uint32_t>(bitPos & 7u);
        if (bytePos_ > size_) {
            throw std::out_of_range("BitReader::seekToBit: position out of range");
        }
    }

    /** Current absolute bit position. */
    size_t bitPosition() const noexcept {
        return (bytePos_ << 3) + bitOffset_;
    }

    /** True when the reader is currently byte-aligned. */
    bool byteAligned() const noexcept { return bitOffset_ == 0; }

private:
    const uint8_t* data_;
    size_t         size_;
    BitOrder       order_;
    size_t         bytePos_{0};
    uint32_t       bitOffset_{0};
};

} // namespace encodings::core
