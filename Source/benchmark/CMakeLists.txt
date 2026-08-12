# Benchmark library - benchmarking framework and metrics

add_library(encodings_benchmark INTERFACE)

target_sources(encodings_benchmark INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        BenchmarkMetrics.hpp
        BenchmarkRunner.hpp
        BenchmarkOutput.hpp
        SelectiveTraceGen.hpp
        GatherTraceGen.hpp
        TimingStats.hpp
)

target_include_directories(encodings_benchmark INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_benchmark INTERFACE encodings_core)

target_compile_features(encodings_benchmark INTERFACE cxx_std_23)

# Enable warnings
if(MSVC)
    target_compile_options(encodings_benchmark INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_benchmark INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()
