#pragma once

#include <cstdint>

namespace encodings::encoders::selectors {

// Reorderer types that can be applied as a pre-processing layer within a
// SubIntSplit section codec.  Stored in SegmentPlan so the section factory
// knows whether to wrap the base codec with a reordering transform.
enum class SubStreamReordererType : uint8_t {
    None   = 0,    // No reordering; use base encoding directly
    BWT512 = 1,    // Windowed Burrows-Wheeler Transform, window size W = 512
};

constexpr const char* subStreamReordererTypeToString(SubStreamReordererType r) noexcept {
    switch (r) {
        case SubStreamReordererType::None:   return "None";
        case SubStreamReordererType::BWT512: return "BWT512";
    }
    return "Unknown";
}

} // namespace encodings::encoders::selectors
