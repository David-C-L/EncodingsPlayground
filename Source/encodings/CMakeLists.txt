# Encodings library - encoding/decoding interfaces and implementations

add_library(encodings_encodings INTERFACE)

target_sources(encodings_encodings INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        EncodedData.hpp
        EncodingProperty.hpp
        EncodingType.hpp
        RowRange.hpp
        Encoder.hpp
        ComposedEncoder.hpp
)

target_include_directories(encodings_encodings INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_encodings INTERFACE encodings_core)

target_compile_features(encodings_encodings INTERFACE cxx_std_23)

# Enable warnings
if(MSVC)
    target_compile_options(encodings_encodings INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_encodings INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()
