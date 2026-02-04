# Generators library - generator interfaces and implementations

add_library(encodings_generators INTERFACE)

target_sources(encodings_generators INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        DataGenerator.hpp
        MapGenerator.hpp
)

target_include_directories(encodings_generators INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_generators INTERFACE encodings_core)

target_compile_features(encodings_generators INTERFACE cxx_std_23)

# Enable warnings
if(MSVC)
    target_compile_options(encodings_generators INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_generators INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()