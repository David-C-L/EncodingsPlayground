# Exceptions library — PFOR-style exception ("patched") handling: a generic
# ExceptionEncoder<T> decorator plus pluggable per-encoding-family detectors.
# See ~/.claude/plans/could-you-analyse-the-misty-widget.md for the design.

add_library(encodings_exceptions INTERFACE)

target_sources(encodings_exceptions INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        BitmapRank.hpp
        ExceptionDetector.hpp
        ExceptionMetricUtils.hpp
        FORExceptionDetector.hpp
        ExceptionEncoder.hpp
)

target_include_directories(encodings_exceptions INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_exceptions INTERFACE
    encodings_core
    encodings_encodings
)

target_compile_features(encodings_exceptions INTERFACE cxx_std_23)

if(MSVC)
    target_compile_options(encodings_exceptions INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_exceptions INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()
