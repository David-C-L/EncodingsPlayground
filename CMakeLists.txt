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
add_subdirectory(Source/reorderers)
add_subdirectory(Source/benchmark)
add_subdirectory(Source/experiments)

if(BUILD_BENCHMARKS)
    add_subdirectory(Benchmarks)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()

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

# Demo for Compile-Time Dimension Benefits
add_executable(demo_compile_time_dimension demo_compile_time_dimension.cpp)
target_include_directories(demo_compile_time_dimension PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(demo_compile_time_dimension PRIVATE encodings_core encodings_encodings encodings_encoders encodings_generators)
set_target_properties(demo_compile_time_dimension PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
