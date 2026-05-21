#pragma once

#include <cstdint>

namespace encodings::reorderers {

// Sub-classification for EncodingType::ReorderingEncoding.
// Stored inside ReorderingCodec and exposed via reorderingType().
enum class ReorderingType : uint8_t {
    Sort         = 0,
    GrayCode     = 1,
    WindowedSort = 2,
    MTF          = 3,
    BWT          = 4,
    BitShuffle   = 5,
};

constexpr const char* reorderingTypeToString(ReorderingType t) noexcept {
    switch (t) {
        case ReorderingType::Sort:         return "Sort";
        case ReorderingType::GrayCode:     return "GrayCode";
        case ReorderingType::WindowedSort: return "WindowedSort";
        case ReorderingType::MTF:          return "MTF";
        case ReorderingType::BWT:          return "BWT";
        case ReorderingType::BitShuffle:   return "BitShuffle";
    }
    return "Unknown";
}

} // namespace encodings::reorderers
