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

# Create sweep executable for SubInt auto sample size study
add_executable(sweep_subint_samples
    sweep_subint_samples.cpp
)

target_link_libraries(sweep_subint_samples
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_generators
        encodings_encoders
        encodings_benchmark
)

target_include_directories(sweep_subint_samples
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

# Create vector benchmark executable
add_executable(vector_benchmarks
    vector_benchmarks.cpp
)

target_link_libraries(vector_benchmarks
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_generators
        encodings_encoders
        encodings_benchmark
)

target_include_directories(vector_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Source
)

# Create results directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark_results)

# Set output directory for the executables
set_target_properties(run_benchmarks sweep_subint_samples map_benchmarks map_benchmarks_refactored vector_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable native ISA (AVX2/AVX-512 etc.) and full optimisation for all benchmark targets.
# These are research binaries always run on the build machine, so -march=native is appropriate.
if(NOT MSVC)
    target_compile_options(run_benchmarks         PRIVATE -march=native -O3)
    target_compile_options(sweep_subint_samples   PRIVATE -march=native -O3)
    target_compile_options(map_benchmarks         PRIVATE -march=native -O3)
    target_compile_options(map_benchmarks_refactored PRIVATE -march=native -O3)
    target_compile_options(vector_benchmarks      PRIVATE -march=native -O3)
endif()

# Add as tests (optional)
if(BUILD_TESTS)
    add_test(NAME BenchmarkSuite COMMAND run_benchmarks)
    add_test(NAME MapBenchmarkSuite COMMAND map_benchmarks)
endif()

