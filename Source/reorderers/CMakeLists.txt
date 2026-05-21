# Reorderers library — data reordering algorithms for compression pre-processing

add_library(encodings_reorderers INTERFACE)

target_sources(encodings_reorderers INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        reorderers/ReorderingType.hpp
        reorderers/PermutationStore.hpp
        reorderers/Reorderer.hpp
        reorderers/ReorderingCodec.hpp
        reorderers/SortReorderer.hpp
        reorderers/GrayCodeReorderer.hpp
        reorderers/WindowedSortReorderer.hpp
        reorderers/MTFReorderer.hpp
        reorderers/BWTReorderer.hpp
        reorderers/BitShuffleCodec.hpp
)

target_include_directories(encodings_reorderers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_reorderers INTERFACE
    encodings_core
    encodings_encodings
    encodings_encoders
)

target_compile_features(encodings_reorderers INTERFACE cxx_std_23)

if(MSVC)
    target_compile_options(encodings_reorderers INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_reorderers INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()
