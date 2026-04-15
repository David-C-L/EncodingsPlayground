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
        LZ4               // LZ4 block compression (fast mode)
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
        }
        return "Unknown";
    }

} // namespace encodings