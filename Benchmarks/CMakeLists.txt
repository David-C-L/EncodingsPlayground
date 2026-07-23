cmake_minimum_required(VERSION 3.28)

# Create benchmark executable for basic types
add_executable(run_benchmarks
    run_benchmarks.cpp
    allocation_hooks.cpp   # global operator new/delete overrides for allocation peak tracking
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
    allocation_hooks.cpp
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
    allocation_hooks.cpp
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
    allocation_hooks.cpp
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
    allocation_hooks.cpp
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

# Range-access heatmap benchmark (SubIntSplit motivation figure)
add_executable(heatmap_benchmark
    heatmap_benchmark.cpp
    allocation_hooks.cpp
)

target_link_libraries(heatmap_benchmark
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_generators
        encodings_encoders
        encodings_benchmark
)

target_include_directories(heatmap_benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Source
)

# Oracle vs AutoSIS comparison benchmark
add_executable(explore_best_encoding
    explore_best_encoding.cpp
    allocation_hooks.cpp
)

target_link_libraries(explore_best_encoding
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_generators
        encodings_encoders
        encodings_benchmark
)

target_include_directories(explore_best_encoding
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Source
)

# Optional VTune ITT pause/resume instrumentation.
# Build with -DVTUNE_ENABLED=ON to bracket selected benchmark phases with
# __itt_pause()/__itt_resume() so VTune only collects samples inside those phases.
option(VTUNE_ENABLED "Enable VTune ITT pause/resume instrumentation" OFF)

if(VTUNE_ENABLED)
    find_path(ITT_INCLUDE_DIR ittnotify.h
        HINTS
            $ENV{VTUNE_PROFILER_DIR}/sdk/include
            /opt/intel/oneapi/vtune/latest/sdk/include
            /opt/intel/vtune_amplifier/sdk/include)
    find_library(ITT_LIBRARY ittnotify
        HINTS
            $ENV{VTUNE_PROFILER_DIR}/sdk/lib64
            /opt/intel/oneapi/vtune/latest/sdk/lib64
            /opt/intel/vtune_amplifier/sdk/lib64)
    if(NOT ITT_INCLUDE_DIR OR NOT ITT_LIBRARY)
        message(FATAL_ERROR
            "VTUNE_ENABLED=ON but ittnotify.h / libittnotify not found. "
            "Set VTUNE_PROFILER_DIR or install Intel VTune.")
    endif()
    foreach(_tgt run_benchmarks sweep_subint_samples map_benchmarks map_benchmarks_refactored vector_benchmarks)
        target_include_directories(${_tgt} PRIVATE ${ITT_INCLUDE_DIR})
        target_link_libraries(${_tgt} PRIVATE ${ITT_LIBRARY} dl)
        target_compile_definitions(${_tgt} PRIVATE VTUNE_ENABLED)
    endforeach()
    message(STATUS "VTune ITT instrumentation enabled (${ITT_LIBRARY})")
endif()

# Reordering benchmarks
add_executable(reordering_benchmarks
    reordering_benchmarks.cpp
    allocation_hooks.cpp
)

target_link_libraries(reordering_benchmarks
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_generators
        encodings_encoders
        encodings_reorderers
        encodings_benchmark
)

target_include_directories(reordering_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Source
)

# Exception-handling benchmarks (see Source/exceptions/)
add_executable(exception_benchmarks
    exception_benchmarks.cpp
)

target_link_libraries(exception_benchmarks
    PRIVATE
        encodings_core
        encodings_encodings
        encodings_encoders
        encodings_exceptions
)

target_include_directories(exception_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Source
)

# Create results directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark_results)

# Set output directory for the executables
set_target_properties(run_benchmarks sweep_subint_samples map_benchmarks map_benchmarks_refactored vector_benchmarks reordering_benchmarks explore_best_encoding heatmap_benchmark exception_benchmarks PROPERTIES
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
    target_compile_options(explore_best_encoding  PRIVATE -march=native -O3)
    target_compile_options(heatmap_benchmark      PRIVATE -march=native -O3)
endif()

# Add as tests (optional)
if(BUILD_TESTS)
    add_test(NAME BenchmarkSuite COMMAND run_benchmarks)
    add_test(NAME MapBenchmarkSuite COMMAND map_benchmarks)
endif()

