# Benchmark library - benchmarking framework and metrics

add_library(encodings_benchmark INTERFACE)

target_sources(encodings_benchmark INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        AllocationTracker.hpp
        ArtifactCache.hpp
        Axes.hpp
        BenchmarkMetrics.hpp
        BenchmarkOutput.hpp
        BenchmarkRunner.hpp
        CachePolicy.hpp
        Cli.hpp
        DatasetCache.hpp
        GatherTraceGen.hpp
        MeasureLoop.hpp
        MemoryTracker.hpp
        OperatorGraphJson.hpp
        PointTraceGen.hpp
        ResultWriter.hpp
        RunManifest.hpp
        SelectiveTraceGen.hpp
        TimingStats.hpp
        registry/DatasetRegistry.hpp
        registry/EncoderRegistry.hpp
        targets/BenchTarget.hpp
        targets/PlaygroundTarget.hpp
)

target_include_directories(encodings_benchmark INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

# ResultWriter.hpp builds Arrow arrays and writes Parquet, and the registries
# construct encoders and parquet-backed generators.  Parquet arrives transitively
# via encodings_generators today, but this target now owns headers that need it,
# so it declares the dependency itself rather than relying on a sibling.
target_link_libraries(encodings_benchmark INTERFACE
    encodings_core
    encodings_encodings
    encodings_encoders
    encodings_generators
    Parquet::parquet_shared
)

target_compile_features(encodings_benchmark INTERFACE cxx_std_23)

# ---------------------------------------------------------------------------
# Build provenance, stamped in at configure time.
#
# Every driver writes a run manifest alongside its results; the manifest must
# not shell out to git at runtime, both because that is slow inside a sweep and
# because the working tree may have moved on since the binary was built.  These
# definitions record the commit the binary was built from, not the one checked
# out when it ran.
# ---------------------------------------------------------------------------
find_package(Git QUIET)
set(_encodings_git_sha "unknown")
set(_encodings_git_dirty "unknown")
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE _encodings_git_sha
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT _encodings_git_sha)
        set(_encodings_git_sha "unknown")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE _encodings_git_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_encodings_git_status STREQUAL "")
        set(_encodings_git_dirty "clean")
    else()
        set(_encodings_git_dirty "dirty")
    endif()
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _encodings_build_type_upper)
set(_encodings_cxx_flags "${CMAKE_CXX_FLAGS}")
if(_encodings_build_type_upper)
    string(APPEND _encodings_cxx_flags " ${CMAKE_CXX_FLAGS_${_encodings_build_type_upper}}")
endif()
string(STRIP "${_encodings_cxx_flags}" _encodings_cxx_flags)

target_compile_definitions(encodings_benchmark INTERFACE
    ENCODINGS_GIT_SHA="${_encodings_git_sha}"
    ENCODINGS_GIT_DIRTY="${_encodings_git_dirty}"
    ENCODINGS_CXX_FLAGS="${_encodings_cxx_flags}"
    ENCODINGS_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

# Enable warnings
if(MSVC)
    target_compile_options(encodings_benchmark INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_benchmark INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()
