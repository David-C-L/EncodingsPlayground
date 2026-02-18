# Core library - fundamental types and interfaces

add_library(encodings_core 
    Bitset.cpp
)

target_sources(encodings_core PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        DataType.hpp
        Bitset.hpp
)

target_include_directories(encodings_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(encodings_core PUBLIC cxx_std_23)

# Enable warnings
if(MSVC)
    target_compile_options(encodings_core PRIVATE /W4 /WX)
else()
    target_compile_options(encodings_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
