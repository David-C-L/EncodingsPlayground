cmake_minimum_required(VERSION 3.28)

# Create benchmark executable
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

# Create results directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark_results)

# Set output directory for the executable
set_target_properties(run_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add as a test (optional)
if(BUILD_TESTS)
    add_test(NAME BenchmarkSuite COMMAND run_benchmarks)
endif()
