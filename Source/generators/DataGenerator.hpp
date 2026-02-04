#pragma once

#include <vector>
#include <span>
#include <memory>
#include <string>
#include <concepts>
#include <ranges>
#include "DataType.hpp"

namespace encodings::datagen {

    using core::DataType;
    using core::PrimitiveType;
    using core::MapType;
    using core::typeToDataType;
/**
 * @brief Abstract base class for data generation strategies
 * 
 * DataGenerators produce test data for benchmarking encodings.
 * They can generate data in bulk or as a stream.
 * 
 * @tparam T The type of data to generate
 */
template<typename T>
class DataGenerator {
public:
    virtual ~DataGenerator() = default;
    
    /**
     * @brief Generate a vector of data elements
     * 
     * @param count Number of elements to generate
     * @return Vector containing generated elements
     */
    virtual std::vector<T> generate(size_t count) = 0;
    
    /**
     * @brief Get the name/description of this generator
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Get the data type this generator produces
     */
    virtual DataType dataType() const {
        if constexpr (PrimitiveType<T>) {
            return typeToDataType<T>;
        } else if constexpr (MapType<T>) {
            return DataType::Map;
        } else {
            return DataType::Array; // Default for composite types
        }
    }
    
    /**
     * @brief Reset the generator's internal state (e.g., RNG seed)
     * Useful for reproducible benchmarks
     */
    virtual void reset() = 0;
    
    /**
     * @brief Get configuration parameters of this generator
     * Returns key-value pairs describing the generator's settings
     */
    virtual std::map<std::string, std::string> getConfig() const {
        return {};
    }
};

/**
 * @brief Configuration for data generator behavior
 */
struct GeneratorConfig {
    size_t seed = 42;  // Random seed for reproducibility
    bool reproducible = true;  // Whether to use fixed seed
    
    // Type-specific bounds (for numeric types)
    std::optional<int64_t> minValueInt;
    std::optional<int64_t> maxValueInt;
    std::optional<double> minValueFloat;
    std::optional<double> maxValueFloat;
    
    // String generation parameters
    std::optional<size_t> minStringLength;
    std::optional<size_t> maxStringLength;
    std::optional<std::string> charset;
    
    // Array/collection parameters
    std::optional<size_t> minArrayLength;
    std::optional<size_t> maxArrayLength;
    
    // Distribution parameters
    std::map<std::string, double> distributionParams;
};

} // namespace encodings::datagen
