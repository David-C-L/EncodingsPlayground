# Encoders library - concrete encoder implementations

add_library(encodings_encoders INTERFACE)

target_sources(encodings_encoders INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        RawEncoder.hpp
        RunLengthEncoder.hpp
        DeltaEncoder.hpp
        DictionaryEncoder.hpp
)

target_include_directories(encodings_encoders INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_encoders INTERFACE 
    encodings_core
    encodings_encodings
)

target_compile_features(encodings_encoders INTERFACE cxx_std_23)

# Enable warnings
if(MSVC)
    target_compile_options(encodings_encoders INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_encoders INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()
