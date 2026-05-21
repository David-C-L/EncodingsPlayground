#pragma once

#include <cstdint>
#include <string>
#include <bitset>
#include <type_traits>
#include <concepts>

namespace encodings {

/**
 * @brief Individual encoding properties as bit flags
 * 
 * These properties describe characteristics and capabilities of encoding schemes.
 * Multiple properties can be combined using bitwise OR.
 */
enum class EncodingProperty : uint32_t {
    None = 0,
    
    // Access patterns
    RandomAccess       = 1 << 0,  // Supports O(1) random access to elements
    SequentialOnly     = 1 << 1,  // Requires sequential decoding
    FastSkip           = 1 << 2,  // Can skip elements efficiently without full decode
    
    // Data preservation
    Lossless           = 1 << 3,  // Preserves exact data
    Lossy              = 1 << 4,  // May lose precision/information
    
    // Ordering
    PreservesOrder     = 1 << 5,  // Maintains original element order
    ReordersData       = 1 << 6,  // May reorder elements (e.g., sorting, clustering)
    
    // Size characteristics
    FixedSize          = 1 << 7,  // Encoded size is predictable/fixed per element
    VariableSize       = 1 << 8,  // Encoded size varies per element
    
    // Compression approach
    DictionaryBased    = 1 << 9,  // Uses dictionary/lookup table
    DeltaBased         = 1 << 10, // Uses delta encoding
    RunLengthBased     = 1 << 11, // Uses run-length encoding
    BitPackingBased    = 1 << 12, // Uses bit-level packing
    EntropyCoding      = 1 << 13, // Uses entropy coding (Huffman, arithmetic, etc.)
    
    // Update capabilities
    SupportsUpdates    = 1 << 14, // Can update individual elements in-place
    ImmutableOnly      = 1 << 15, // Requires full re-encoding for updates
    
    // Streaming
    StreamingFriendly  = 1 << 16, // Can encode/decode in streaming fashion
    RequiresFullData   = 1 << 17, // Needs all data upfront (e.g., for dictionary)
    
    // Vectorization
    Vectorizable       = 1 << 18, // Benefits from SIMD operations
    
    // Null handling
    NullableSupport    = 1 << 19, // Explicitly supports null/missing values
    
    // Sorted data optimization
    OptimizedForSorted = 1 << 20, // Works better on sorted data
    
    // Memory characteristics
    LowMemoryOverhead  = 1 << 21, // Minimal extra memory during encode/decode
    HighMemoryOverhead = 1 << 22, // Significant extra memory needed
    
    // Cascadable
    Composable         = 1 << 23, // Can be chained with other encodings

    // Reordering-specific
    ValueTransform     = 1 << 24, // Bijection on values only; positions unchanged (e.g. GrayCode)
};

/**
 * @brief Set of encoding properties (bitset wrapper)
 */
class EncodingProperties {
public:
    constexpr EncodingProperties() : flags_(0) {}
    constexpr EncodingProperties(EncodingProperty prop) : flags_(static_cast<uint32_t>(prop)) {}
    constexpr EncodingProperties(uint32_t flags) : flags_(flags) {}
    
    // Add a property
    constexpr EncodingProperties& add(EncodingProperty prop) {
        flags_ |= static_cast<uint32_t>(prop);
        return *this;
    }
    
    // Remove a property
    constexpr EncodingProperties& remove(EncodingProperty prop) {
        flags_ &= ~static_cast<uint32_t>(prop);
        return *this;
    }
    
    // Check if has property
    constexpr bool has(EncodingProperty prop) const {
        return (flags_ & static_cast<uint32_t>(prop)) != 0;
    }
    
    // Check if has all properties
    constexpr bool hasAll(EncodingProperties props) const {
        return (flags_ & props.flags_) == props.flags_;
    }
    
    // Check if has any property
    constexpr bool hasAny(EncodingProperties props) const {
        return (flags_ & props.flags_) != 0;
    }
    
    // Operators
    constexpr EncodingProperties operator|(EncodingProperty prop) const {
        return EncodingProperties(flags_ | static_cast<uint32_t>(prop));
    }
    
    constexpr EncodingProperties operator|(EncodingProperties other) const {
        return EncodingProperties(flags_ | other.flags_);
    }
    
    constexpr EncodingProperties& operator|=(EncodingProperty prop) {
        return add(prop);
    }
    
    constexpr EncodingProperties& operator|=(EncodingProperties other) {
        flags_ |= other.flags_;
        return *this;
    }
    
    constexpr bool operator==(EncodingProperties other) const {
        return flags_ == other.flags_;
    }
    
    // Get raw flags
    constexpr uint32_t raw() const { return flags_; }
    
    // Common property combinations
    static constexpr EncodingProperties fastRandomAccess() {
        return EncodingProperties(EncodingProperty::RandomAccess) 
            | EncodingProperty::Lossless 
            | EncodingProperty::PreservesOrder;
    }
    
    static constexpr EncodingProperties basicCompression() {
        return EncodingProperties(EncodingProperty::Lossless) 
            | EncodingProperty::PreservesOrder 
            | EncodingProperty::SequentialOnly;
    }
    
    static constexpr EncodingProperties heavyCompression() {
        return EncodingProperties(EncodingProperty::Lossless) 
            | EncodingProperty::RequiresFullData 
            | EncodingProperty::SequentialOnly;
    }
    
private:
    uint32_t flags_;
};

/**
 * @brief Concept to check if a type has encoding properties
 */
template<typename T>
concept HasEncodingProperties = requires(const T& t) {
    { t.properties() } -> std::convertible_to<EncodingProperties>;
};

/**
 * @brief Trait to get properties from an encoder type at compile-time
 * Encoders can specialize this to provide static property information
 */
template<typename EncoderType>
struct EncoderTraits {
    // Default: unknown properties
    static constexpr EncodingProperties properties() {
        return EncodingProperties();
    }
};

/**
 * @brief Helper to get properties from an encoder instance or type
 */
template<typename T>
constexpr EncodingProperties getEncodingProperties() {
    if constexpr (HasEncodingProperties<T>) {
        return T{}.properties();
    } else {
        return EncoderTraits<T>::properties();
    }
}

/**
 * @brief Concept for encoders that support random access
 */
template<typename T>
concept RandomAccessEncoder = HasEncodingProperties<T> && 
    requires(const T& enc) {
        requires getEncodingProperties<T>().has(EncodingProperty::RandomAccess);
    };

/**
 * @brief Concept for lossless encoders
 */
template<typename T>
concept LosslessEncoder = HasEncodingProperties<T> && 
    requires(const T& enc) {
        requires getEncodingProperties<T>().has(EncodingProperty::Lossless);
    };

/**
 * @brief Concept for order-preserving encoders
 */
template<typename T>
concept OrderPreservingEncoder = HasEncodingProperties<T> && 
    requires(const T& enc) {
        requires getEncodingProperties<T>().has(EncodingProperty::PreservesOrder);
    };

} // namespace encodings