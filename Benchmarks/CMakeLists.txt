cmake_minimum_required(VERSION 3.28)

# ---------------------------------------------------------------------------
# Benchmark drivers.
#
# Every driver is built the same way: its own .cpp plus allocation_hooks.cpp
# (global operator new/delete overrides for allocation peak tracking), the
# common set of interface libraries, -march=native -O3, and output into bin/.
# They used to be ten copy-pasted add_executable blocks, which is how
# reordering_benchmarks ended up as the one target missing the optimisation
# flags and how three targets ended up missing VTune instrumentation.  Adding a
# driver now means adding one line to encodings_benchmark_drivers.
# ---------------------------------------------------------------------------

set(encodings_benchmark_drivers
    run_benchmarks
    sweep_subint_samples
    map_benchmarks
    map_benchmarks_refactored
    vector_benchmarks
    heatmap_benchmark
    gather_heatmap_benchmark
    explore_best_encoding
    reordering_benchmarks
    bench_decode_gather
    bench_decode_bulk
    bench_decode_range
    bench_decode_point
)

# Drivers needing libraries beyond the common set, as "<driver>:<lib>;<lib>".
set(encodings_benchmark_extra_libs
    "reordering_benchmarks:encodings_reorderers"
)

function(encodings_add_benchmark_driver name)
    # A driver's source is either beside this CMakeLists (the pre-refactor
    # drivers) or under drivers/ (the refactored ones).  An explicit second
    # argument wins, so a driver can live anywhere without a convention change.
    if(ARGV1)
        set(_src "${ARGV1}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp")
        set(_src "${name}.cpp")
    else()
        set(_src "drivers/${name}.cpp")
    endif()

    add_executable(${name}
        ${_src}
        allocation_hooks.cpp
    )

    target_link_libraries(${name}
        PRIVATE
            encodings_core
            encodings_encodings
            encodings_generators
            encodings_encoders
            encodings_benchmark
    )

    foreach(_entry IN LISTS encodings_benchmark_extra_libs)
        string(REPLACE ":" ";" _parts "${_entry}")
        list(GET _parts 0 _driver)
        if(_driver STREQUAL name)
            list(REMOVE_AT _parts 0)
            target_link_libraries(${name} PRIVATE ${_parts})
        endif()
    endforeach()

    target_include_directories(${name}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/Source
    )

    # Native ISA (AVX2/AVX-512 etc.) and full optimisation.  These are research
    # binaries always run on the build machine, so -march=native is appropriate.
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -march=native -O3)
    endif()

    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

foreach(_driver IN LISTS encodings_benchmark_drivers)
    encodings_add_benchmark_driver(${_driver})
endforeach()

# ---------------------------------------------------------------------------
# Optional VTune ITT pause/resume instrumentation.
# Build with -DVTUNE_ENABLED=ON to bracket selected benchmark phases with
# __itt_pause()/__itt_resume() so VTune only collects samples inside those
# phases.  Applied to every driver: previously three of them were omitted, so
# a VTune build silently profiled them end to end.
# ---------------------------------------------------------------------------
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
    foreach(_tgt IN LISTS encodings_benchmark_drivers)
        target_include_directories(${_tgt} PRIVATE ${ITT_INCLUDE_DIR})
        target_link_libraries(${_tgt} PRIVATE ${ITT_LIBRARY} dl)
        target_compile_definitions(${_tgt} PRIVATE VTUNE_ENABLED)
    endforeach()
    message(STATUS "VTune ITT instrumentation enabled (${ITT_LIBRARY})")
endif()

# Create results directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark_results)

# Add as tests (optional)
if(BUILD_TESTS)
    add_test(NAME BenchmarkSuite COMMAND run_benchmarks)
    add_test(NAME MapBenchmarkSuite COMMAND map_benchmarks)
endif()
