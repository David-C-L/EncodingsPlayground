#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>
#include <map>
#include <concepts>
#include <type_traits>
#include "core/DataType.hpp"

namespace encodings {

    enum class EncodingType {
        RawEncoding,
        RunLengthEncoding,
        DeltaEncoding,
        DictionaryEncoding,
        BitPacking,
        Zstd,
        SphericalEncoding,
        SubIntEncoding,  // Sub-integer byte-group dictionary + bit-packing
        VarIntEncoding,      // LEB128 variable-length integer with optional ZigZag
        DeltaVarIntEncoding, // Delta pre-pass then LEB128 VarInt
        Composed,            // Sequential composition (e.g., Delta | RLE)
        Structural,      // Structural composition (e.g., different encoders per field)
        FrameOfReference, // Frame-of-reference: store residuals relative to a per-frame ref
        AdaptiveFramedBitPrefix, // Adaptive framed bit-prefix encoding
        AdaptiveFrameOfReference, // Adaptive frame-of-reference: dynamic per-frame ref selection and residual width
        OpenZL,           // OpenZL external codec
        HuffmanEncoding,  // Canonical Huffman entropy coding (sequential decode only)
        LZ4,              // LZ4 block compression (fast mode)
        FSEEncoding,      // Finite State Entropy (tANS) entropy coding (sequential decode only)
        FrequencyPartitionEncoding, // Frequency-partitioned fixed-width keys with per-tier bitmaps
        ReorderingEncoding,         // Reordering pre-pass wrapping any inner codec; sub-classified by ReorderingType
        MainlyConstantEncoding,     // One dominant value stored inline; others in a sub-stream
        AdaptiveDictionaryEncoding,  // Block-partitioned dictionary with per-block key widths
        BlockFrequencyPartitionEncoding, // Block-local frequency-partitioned encoding with compact tier-tag bitfield
        BlockFSEEncoding,                // Block-indexed Finite State Entropy with per-block byte-offset seek
        BlockFORFPEEncoding,             // Block-local Frame-of-Reference + flexible-tier FPE with sampled rank index
        BlockFrequencyPartitionFOREncoding, // Block-local FPE with global FOR prepass (subtract global min before FPE)
        CascadingFrameOfReference,        // Recursive FOR cascade on residual AND reference streams, runtime frame-size schedule per stream
        DeltaPrepassEncoding,             // First-order delta prepass with a pluggable leaf codec for the delta stream
        RangePackEncoding,                // Generic global Frame-of-Reference + narrowest-width repack, composed with any inner section codec
        RangePackFrequencyPartitionEncoding, // RangePack composed with FrequencyPartitionEncoding (type-narrowed)
        RangePackBlockFrequencyPartitionEncoding, // RangePack composed with BlockFrequencyPartitionEncoding (type-narrowed)
        CascadingFORBlockFrequencyPartitionEncoding, // CascadingFOR with BlockFrequencyPartitionEncoding on the residual stream
        RunLengthCascadingFOREncoding, // RunLengthEncoding with a CascadingFOR-compressed runStarts stream
        CascadingFORFSEEncoding,      // CascadingFOR with FSEEncoder on the residual stream (NOT random-access safe -- see registration site)
        CascadingFORBlockFSEEncoding, // CascadingFOR with BlockFSEEncoder on the residual stream (random-access safe)
        CascadingFORHuffmanEncoding,  // CascadingFOR with HuffmanEncoder on the residual stream (NOT random-access safe -- see registration site)
        CascadingFORPrevHuffmanEncoding,               // CascadingFOR with FORReferencePolicy::PREV residual schedule + HuffmanEncoder leaf (NOT random-access safe)
        CascadingFORPrevFSEEncoding,                   // CascadingFOR with FORReferencePolicy::PREV residual schedule + FSEEncoder leaf (NOT random-access safe)
        CascadingFORPrevBlockFSEEncoding,               // CascadingFOR with FORReferencePolicy::PREV residual schedule + BlockFSEEncoder leaf (random-access safe)
        CascadingFORPrevFrequencyPartitionEncoding,      // CascadingFOR with FORReferencePolicy::PREV residual schedule + FrequencyPartitionEncoder leaf
        CascadingFORPrevBlockFrequencyPartitionEncoding, // CascadingFOR with FORReferencePolicy::PREV residual schedule + BlockFrequencyPartitionEncoder leaf
        ExceptionWrappedEncoding                         // Patched/exception pre-pass wrapping any inner codec; non-conforming values pulled into a side stream
    };

    /** Convert EncodingType enum to human-readable string */
    constexpr const char* encodingTypeToString(EncodingType type) {
        switch (type) {
            case EncodingType::RawEncoding:              return "RawEncoding";
            case EncodingType::RunLengthEncoding:        return "RunLengthEncoding";
            case EncodingType::DeltaEncoding:            return "DeltaEncoding";
            case EncodingType::DictionaryEncoding:       return "DictionaryEncoding";
            case EncodingType::BitPacking:               return "BitPacking";
            case EncodingType::Zstd:                     return "Zstd";
            case EncodingType::SphericalEncoding:        return "SphericalEncoding";
            case EncodingType::SubIntEncoding:           return "SubIntEncoding";
            case EncodingType::VarIntEncoding:           return "VarIntEncoding";
            case EncodingType::DeltaVarIntEncoding:      return "DeltaVarIntEncoding";
            case EncodingType::Composed:                 return "Composed";
            case EncodingType::Structural:               return "Structural";
            case EncodingType::FrameOfReference:         return "FrameOfReference";
            case EncodingType::AdaptiveFramedBitPrefix: return "AdaptiveFramedBitPrefix";
            case EncodingType::AdaptiveFrameOfReference: return "AdaptiveFrameOfReference";
            case EncodingType::OpenZL:                   return "OpenZL";
            case EncodingType::HuffmanEncoding:          return "HuffmanEncoding";
            case EncodingType::LZ4:                      return "LZ4";
            case EncodingType::FSEEncoding:              return "FSEEncoding";
            case EncodingType::FrequencyPartitionEncoding: return "FrequencyPartitionEncoding";
            case EncodingType::ReorderingEncoding:         return "ReorderingEncoding";
            case EncodingType::MainlyConstantEncoding:       return "MainlyConstantEncoding";
            case EncodingType::AdaptiveDictionaryEncoding:           return "AdaptiveDictionaryEncoding";
            case EncodingType::BlockFrequencyPartitionEncoding:     return "BlockFrequencyPartitionEncoding";
            case EncodingType::BlockFSEEncoding:                     return "BlockFSEEncoding";
            case EncodingType::BlockFORFPEEncoding:                  return "BlockFORFPEEncoding";
            case EncodingType::BlockFrequencyPartitionFOREncoding:   return "BlockFrequencyPartitionFOREncoding";
            case EncodingType::CascadingFrameOfReference:            return "CascadingFrameOfReference";
            case EncodingType::DeltaPrepassEncoding:                 return "DeltaPrepassEncoding";
            case EncodingType::RangePackEncoding:                    return "RangePackEncoding";
            case EncodingType::RangePackFrequencyPartitionEncoding:  return "RangePackFrequencyPartitionEncoding";
            case EncodingType::RangePackBlockFrequencyPartitionEncoding: return "RangePackBlockFrequencyPartitionEncoding";
            case EncodingType::CascadingFORBlockFrequencyPartitionEncoding: return "CascadingFORBlockFrequencyPartitionEncoding";
            case EncodingType::RunLengthCascadingFOREncoding: return "RunLengthCascadingFOREncoding";
            case EncodingType::CascadingFORFSEEncoding:       return "CascadingFORFSEEncoding";
            case EncodingType::CascadingFORBlockFSEEncoding:  return "CascadingFORBlockFSEEncoding";
            case EncodingType::CascadingFORHuffmanEncoding:   return "CascadingFORHuffmanEncoding";
            case EncodingType::CascadingFORPrevHuffmanEncoding: return "CascadingFORPrevHuffmanEncoding";
            case EncodingType::CascadingFORPrevFSEEncoding:     return "CascadingFORPrevFSEEncoding";
            case EncodingType::CascadingFORPrevBlockFSEEncoding: return "CascadingFORPrevBlockFSEEncoding";
            case EncodingType::CascadingFORPrevFrequencyPartitionEncoding: return "CascadingFORPrevFrequencyPartitionEncoding";
            case EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding: return "CascadingFORPrevBlockFrequencyPartitionEncoding";
            case EncodingType::ExceptionWrappedEncoding:      return "ExceptionWrappedEncoding";
        }
        return "Unknown";
    }

} // namespace encodings