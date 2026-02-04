# Core library - fundamental types and interfaces

add_library(encodings_core INTERFACE)

target_sources(encodings_core INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        DataType.hpp
)

target_include_directories(encodings_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(encodings_core INTERFACE cxx_std_23)

# Enable warnings
if(MSVC)
    target_compile_options(encodings_core INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_core INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()
