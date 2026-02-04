#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>
#include <map>
#include <concepts>
#include <type_traits>

namespace encodings::core {

/**
 * @brief Enumeration of all supported primitive and composite data types
 */
enum class DataType {
    // Integer types
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    
    // Floating point types
    Float32,
    Float64,
    
    // Other primitives
    Bool,
    String,
    
    // Composite types
    Array,
    Map
};

/**
 * @brief Convert DataType enum to human-readable string
 */
constexpr const char* dataTypeToString(DataType type) {
    switch (type) {
        case DataType::Int8:    return "Int8";
        case DataType::Int16:   return "Int16";
        case DataType::Int32:   return "Int32";
        case DataType::Int64:   return "Int64";
        case DataType::UInt8:   return "UInt8";
        case DataType::UInt16:  return "UInt16";
        case DataType::UInt32:  return "UInt32";
        case DataType::UInt64:  return "UInt64";
        case DataType::Float32: return "Float32";
        case DataType::Float64: return "Float64";
        case DataType::Bool:    return "Bool";
        case DataType::String:  return "String";
        case DataType::Array:   return "Array";
        case DataType::Map:     return "Map";
    }
    return "Unknown";
}

/**
 * @brief Get the size in bytes for fixed-size types
 */
constexpr size_t dataTypeSize(DataType type) {
    switch (type) {
        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Bool:
            return 1;
        case DataType::Int16:
        case DataType::UInt16:
            return 2;
        case DataType::Int32:
        case DataType::UInt32:
        case DataType::Float32:
            return 4;
        case DataType::Int64:
        case DataType::UInt64:
        case DataType::Float64:
            return 8;
        case DataType::String:
        case DataType::Array:
        case DataType::Map:
            return 0; // Variable size
    }
    return 0;
}

// Type concepts for compile-time type checking
template<typename T>
concept IntegralType = std::is_integral_v<T> && !std::is_same_v<T, bool>;

template<typename T>
concept FloatingPointType = std::is_floating_point_v<T>;

template<typename T>
concept NumericType = IntegralType<T> || FloatingPointType<T>;

template<typename T>
concept PrimitiveType = NumericType<T> || std::is_same_v<T, bool> || std::is_same_v<T, std::string>;

template<typename T>
concept ArrayType = requires(T t) {
    typename T::value_type;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<size_t>;
};

template<typename T>
concept MapType = requires(T t) {
    typename T::key_type;
    typename T::mapped_type;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
};

/**
 * @brief Type traits to map C++ types to DataType enum
 */
template<typename T>
struct TypeToDataType;

template<> struct TypeToDataType<int8_t>   { static constexpr DataType value = DataType::Int8; };
template<> struct TypeToDataType<int16_t>  { static constexpr DataType value = DataType::Int16; };
template<> struct TypeToDataType<int32_t>  { static constexpr DataType value = DataType::Int32; };
template<> struct TypeToDataType<int64_t>  { static constexpr DataType value = DataType::Int64; };
template<> struct TypeToDataType<uint8_t>  { static constexpr DataType value = DataType::UInt8; };
template<> struct TypeToDataType<uint16_t> { static constexpr DataType value = DataType::UInt16; };
template<> struct TypeToDataType<uint32_t> { static constexpr DataType value = DataType::UInt32; };
template<> struct TypeToDataType<uint64_t> { static constexpr DataType value = DataType::UInt64; };
template<> struct TypeToDataType<float>    { static constexpr DataType value = DataType::Float32; };
template<> struct TypeToDataType<double>   { static constexpr DataType value = DataType::Float64; };
template<> struct TypeToDataType<bool>     { static constexpr DataType value = DataType::Bool; };
template<> struct TypeToDataType<std::string> { static constexpr DataType value = DataType::String; };

template<typename T>
inline constexpr DataType typeToDataType = TypeToDataType<T>::value;

} // namespace encodings::core
