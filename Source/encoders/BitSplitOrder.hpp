#pragma once

#include <cstdint>

namespace encodings::encoders {

// Bit split order (used by sub-integer / sectioned encodings)
enum class BitSplitOrder : uint8_t {
    LSB_TO_MSB = 0,  // A = lowest bits, then B, then C highest
    MSB_TO_LSB = 1,  // A = highest bits, then B, then C lowest
};

} // namespace encodings::encoders
