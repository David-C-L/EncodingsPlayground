#pragma once

#include "encodings/EncodingType.hpp"

namespace encodings::generators::samplers {

// =============================================================================
//  SamplingProfile / preferredSamplingProfile
//
//  Classifies each EncodingType by which kind of sample its cost estimate
//  should be drawn from, for the oracle grid DP in explore_best_encoding.cpp:
//
//    - Random:         representative, well-scattered sample. Right for
//                      order-independent codecs (global value distribution
//                      or bit width matters, not row position) and for
//                      block-local codecs whose block size already matches
//                      SubIntSplit's own default blockSize (BlockFrequencyPartition
//                      family) -- the "random" sample is itself built from many
//                      small scattered contiguous blocks at that same granularity.
//    - Consecutive:     contiguous windows sized to match ~512-scale per-frame
//                      codecs (FrameOfReference's FrameSize, and adaptive
//                      frame-based variants) -- a random sample scatters frames
//                      across the whole timeline and sees inflated residuals.
//    - WideContiguous:  large contiguous windows for genuinely order-dependent
//                      codecs whose benefit comes from real, long runs across
//                      consecutive rows (RunLengthEncoding, and any delta-style
//                      codec) -- invisible to both of the above at their
//                      current block granularities.
//
//  Every EncodingType not explicitly listed defaults to Random, which
//  reproduces today's behavior exactly for all of them.
// =============================================================================

enum class SamplingProfile {
    Random,
    Consecutive,
    WideContiguous,
};

constexpr SamplingProfile preferredSamplingProfile(encodings::EncodingType type) {
    using ET = encodings::EncodingType;
    switch (type) {
        case ET::RunLengthEncoding:
        case ET::DeltaEncoding:
        case ET::DeltaVarIntEncoding:
        case ET::DeltaPrepassEncoding:
            return SamplingProfile::WideContiguous;
        case ET::FrameOfReference:
        case ET::AdaptiveFrameOfReference:
        case ET::AdaptiveFramedBitPrefix:
            return SamplingProfile::Consecutive;
        default:
            return SamplingProfile::Random;
    }
}

} // namespace encodings::generators::samplers
