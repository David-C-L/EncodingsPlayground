cmake_minimum_required(VERSION 3.28)
project(EncodingsPlayground VERSION 0.1.0 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

# Sanitizers
if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Add subdirectories
add_subdirectory(Source/core)
add_subdirectory(Source/encodings)
add_subdirectory(Source/generators)
add_subdirectory(Source/encoders)
add_subdirectory(Source/benchmark)

if(BUILD_BENCHMARKS)
    add_subdirectory(Benchmarks)
endif()

if(BUILD_TESTS)
    enable_testing()
    # add_subdirectory(Tests)
    
    # Add tests to CTest
    add_test(NAME test_delta_rle COMMAND test_delta_rle)
    add_test(NAME test_bit_parentheses COMMAND test_bit_parentheses)
    add_test(NAME test_tpch_partkey COMMAND test_tpch_partkey)
    add_test(NAME test_spherical_encoder COMMAND test_spherical_encoder)
    add_test(NAME test_vector_generators COMMAND test_vector_generators)
    add_test(NAME test_bitset COMMAND test_bitset)
    add_test(NAME test_subint_encoder COMMAND test_subint_encoder)
    add_test(NAME test_fse_encoder COMMAND test_fse_encoder)
    add_test(NAME test_frequency_partition_encoder COMMAND test_frequency_partition_encoder)
endif()

# Simple test for DeltaRunLengthEncoder
add_executable(test_delta_rle test_delta_rle.cpp)
target_include_directories(test_delta_rle PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_delta_rle PRIVATE encodings_core encodings_encodings encodings_encoders)
set_target_properties(test_delta_rle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Test for BitParenthesesEncoder
add_executable(test_bit_parentheses test_bit_parentheses.cpp)
target_include_directories(test_bit_parentheses PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_bit_parentheses PRIVATE encodings_core encodings_encodings encodings_encoders)
set_target_properties(test_bit_parentheses PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Example for BitParenthesesEncoder
add_executable(example_bit_parentheses example_bit_parentheses.cpp)
target_include_directories(example_bit_parentheses PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(example_bit_parentheses PRIVATE encodings_core encodings_encodings encodings_encoders)
set_target_properties(example_bit_parentheses PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Performance benchmark for BitParenthesesEncoder
add_executable(benchmark_bit_parentheses_speed benchmark_bit_parentheses_speed.cpp)
target_include_directories(benchmark_bit_parentheses_speed PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(benchmark_bit_parentheses_speed PRIVATE encodings_core encodings_encodings encodings_encoders)
set_target_properties(benchmark_bit_parentheses_speed PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Test for TPC-H Lineitem PartKey Generator
add_executable(test_tpch_partkey test_tpch_partkey.cpp)
target_include_directories(test_tpch_partkey PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_tpch_partkey PRIVATE encodings_core encodings_generators)
set_target_properties(test_tpch_partkey PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Test for SphericalEncoder
add_executable(test_spherical_encoder test_spherical_encoder.cpp)
target_include_directories(test_spherical_encoder PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_spherical_encoder PRIVATE encodings_core encodings_encodings encodings_encoders)
set_target_properties(test_spherical_encoder PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Test for Vector Generators
add_executable(test_vector_generators test_vector_generators.cpp)
target_include_directories(test_vector_generators PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_vector_generators PRIVATE encodings_core encodings_encodings encodings_encoders encodings_generators)
set_target_properties(test_vector_generators PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Demo for Compile-Time Dimension Benefits
add_executable(demo_compile_time_dimension demo_compile_time_dimension.cpp)
target_include_directories(demo_compile_time_dimension PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(demo_compile_time_dimension PRIVATE encodings_core encodings_encodings encodings_encoders encodings_generators)
set_target_properties(demo_compile_time_dimension PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Test for FastBitset
add_executable(test_bitset test_bitset.cpp)
target_include_directories(test_bitset PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_bitset PRIVATE encodings_core encodings_encodings encodings_encoders)
set_target_properties(test_bitset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Tests for SubIntEncoder (int32 + int64)
add_executable(test_subint_encoder test_subint_encoder.cpp)
target_include_directories(test_subint_encoder PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_subint_encoder PRIVATE encodings_core encodings_encodings encodings_encoders encodings_generators)
set_target_properties(test_subint_encoder PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Debug build variant with SUBINT_DEBUG=1
add_executable(test_subint_encoder_dbg test_subint_encoder.cpp)
target_compile_definitions(test_subint_encoder_dbg PRIVATE SUBINT_DEBUG=1)
target_include_directories(test_subint_encoder_dbg PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_subint_encoder_dbg PRIVATE encodings_core encodings_encodings encodings_encoders encodings_generators)
set_target_properties(test_subint_encoder_dbg PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Tests for FSEEncoder and SubIntSplit FSE integration
add_executable(test_fse_encoder test_fse_encoder.cpp)
target_include_directories(test_fse_encoder PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_fse_encoder PRIVATE encodings_core encodings_encodings encodings_encoders encodings_generators)
set_target_properties(test_fse_encoder PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Tests for FrequencyPartitionEncoder (bitmap/tag index modes)
add_executable(test_frequency_partition_encoder test_frequency_partition_encoder.cpp)
target_include_directories(test_frequency_partition_encoder PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(test_frequency_partition_encoder PRIVATE encodings_core encodings_encodings encodings_encoders encodings_generators)
set_target_properties(test_frequency_partition_encoder PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
