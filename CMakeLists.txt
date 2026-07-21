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
