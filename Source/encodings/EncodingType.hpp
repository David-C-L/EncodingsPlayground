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
        Structural       // Structural composition (e.g., different encoders per field)
    };

    /** Convert EncodingType enum to human-readable string */
    constexpr const char* encodingTypeToString(EncodingType type) {
        switch (type) {
            case EncodingType::RawEncoding:          return "RawEncoding";
            case EncodingType::RunLengthEncoding:    return "RunLengthEncoding";
            case EncodingType::DeltaEncoding:        return "DeltaEncoding";
            case EncodingType::DictionaryEncoding:   return "DictionaryEncoding";
            case EncodingType::BitPacking:           return "BitPacking";
            case EncodingType::Zstd:                 return "Zstd";
            case EncodingType::SphericalEncoding:    return "SphericalEncoding";
            case EncodingType::SubIntEncoding:       return "SubIntEncoding";
            case EncodingType::VarIntEncoding:       return "VarIntEncoding";
            case EncodingType::DeltaVarIntEncoding:  return "DeltaVarIntEncoding";
            case EncodingType::Composed:             return "Composed";
            case EncodingType::Structural:           return "Structural";
        }
        return "Unknown";
    }

} // namespace encodings