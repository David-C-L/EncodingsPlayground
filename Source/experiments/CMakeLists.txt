# Cardinality-analysis experiment drivers for CascadingFOREncoder.
# Kept separate from Benchmarks/ (compression-ratio/speed benchmarking).

add_executable(cascade_cardinality_experiment cascade_cardinality_experiment.cpp)
target_include_directories(cascade_cardinality_experiment PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(cascade_cardinality_experiment PRIVATE
    encodings_core
    encodings_encodings
    encodings_encoders
    encodings_generators
    encodings_reorderers
)
set_target_properties(cascade_cardinality_experiment PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
