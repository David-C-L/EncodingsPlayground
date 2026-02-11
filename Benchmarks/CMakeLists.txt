cmake_minimum_required(VERSION 3.28)

# Create benchmark executable for basic types
add_executable(run_benchmarks
    run_benchmarks.cpp
)

target_link_libraries(run_benchmarks
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_generators
        encodings_encoders
        encodings_benchmark
)

target_include_directories(run_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Source
)

# Create map benchmark executable
add_executable(map_benchmarks
    map_benchmarks.cpp
)

target_link_libraries(map_benchmarks
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_generators
        encodings_encoders
        encodings_benchmark
)

target_include_directories(map_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Source
)

# Create refactored map benchmark executable
add_executable(map_benchmarks_refactored
    map_benchmarks_refactored.cpp
)

target_link_libraries(map_benchmarks_refactored
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_generators
        encodings_encoders
        encodings_benchmark
)

target_include_directories(map_benchmarks_refactored
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Source
)


# Create results directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark_results)

# Set output directory for the executables
set_target_properties(run_benchmarks map_benchmarks map_benchmarks_refactored PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add as tests (optional)
if(BUILD_TESTS)
    add_test(NAME BenchmarkSuite COMMAND run_benchmarks)
    add_test(NAME MapBenchmarkSuite COMMAND map_benchmarks)
endif()

