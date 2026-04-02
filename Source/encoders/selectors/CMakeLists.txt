add_library(encodings_selectors INTERFACE)

target_sources(encodings_selectors INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        BitRangeSegmentBuilder.hpp
        IDSubStreamEncodingSelector.hpp
        MetricCollector.hpp
        costs/EncodingCostModel.hpp
)

target_include_directories(encodings_selectors INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_selectors INTERFACE
    encodings_encoders
)

add_executable(selectors_test_selector
    test_selector.cpp
)

target_link_libraries(selectors_test_selector
    PRIVATE encodings_selectors
)

set_target_properties(selectors_test_selector PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Inherit C++ standard and warning flags from encodings_encoders
