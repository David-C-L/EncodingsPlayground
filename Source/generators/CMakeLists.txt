# Generators library - generator interfaces and implementations

# ---------------------------------------------------------------------------
# Arrow + Parquet via ExternalProject
#   - Uses bundled Thrift to avoid the system libthrift version conflict
#   - Only builds the arrow core and parquet libraries (no Flight/Gandiva/ORC)
#   - Installed into <binary_dir>/arrow-install so it doesn't touch the system
# ---------------------------------------------------------------------------
include(ExternalProject)

set(ARROW_INSTALL_DIR "${CMAKE_BINARY_DIR}/arrow-install")

ExternalProject_Add(arrow_ep
    GIT_REPOSITORY  https://github.com/apache/arrow.git
    GIT_TAG         apache-arrow-20.0.0
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   cpp
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${ARROW_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_STANDARD=17
        # Only what we need
        -DARROW_PARQUET=ON
        -DARROW_WITH_SNAPPY=ON
        -DARROW_WITH_ZSTD=ON
        -DARROW_WITH_LZ4=ON
        -DARROW_WITH_ZLIB=ON
        -DARROW_WITH_BROTLI=OFF
        # Use bundled Thrift to avoid system version conflicts
        -DARROW_DEPENDENCY_SOURCE=BUNDLED
        -DThrift_SOURCE=BUNDLED
        # Disable everything we don't need
        -DARROW_BUILD_TESTS=OFF
        -DARROW_BUILD_BENCHMARKS=OFF
        -DARROW_BUILD_EXAMPLES=OFF
        -DARROW_BUILD_UTILITIES=OFF
        -DARROW_FLIGHT=OFF
        -DARROW_GANDIVA=OFF
        -DARROW_ORC=OFF
        -DARROW_DATASET=OFF
        -DARROW_COMPUTE=OFF
        -DARROW_CSV=OFF
        -DARROW_JSON=OFF
        -DARROW_JEMALLOC=OFF
        -DARROW_PLASMA=OFF
    BUILD_COMMAND   ${CMAKE_COMMAND} --build . --parallel
    INSTALL_COMMAND ${CMAKE_COMMAND} --install .
    LOG_DOWNLOAD  ON
    LOG_CONFIGURE ON
    LOG_BUILD     ON
    LOG_INSTALL   ON
)

# IMPORTED GLOBAL so the targets are visible to any consumer of encodings_generators
add_library(Arrow::arrow_shared SHARED IMPORTED GLOBAL)
set_target_properties(Arrow::arrow_shared PROPERTIES
    IMPORTED_LOCATION             "${ARROW_INSTALL_DIR}/lib/libarrow.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ARROW_INSTALL_DIR}/include"
)
add_dependencies(Arrow::arrow_shared arrow_ep)

add_library(Parquet::parquet_shared SHARED IMPORTED GLOBAL)
set_target_properties(Parquet::parquet_shared PROPERTIES
    IMPORTED_LOCATION             "${ARROW_INSTALL_DIR}/lib/libparquet.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ARROW_INSTALL_DIR}/include"
    INTERFACE_LINK_LIBRARIES      "Arrow::arrow_shared"
)
add_dependencies(Parquet::parquet_shared arrow_ep)

# ---------------------------------------------------------------------------

add_library(encodings_generators INTERFACE)

target_sources(encodings_generators INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        DataGenerator.hpp
        MapGenerator.hpp
        CommonGenerators.hpp
        VectorGenerator.hpp
        ParquetColumnGenerator.hpp
        GeneratorUtils.hpp
        SnowflakeIDGenerator.hpp
)

target_include_directories(encodings_generators INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_generators INTERFACE
    encodings_core
    Parquet::parquet_shared
)

target_compile_features(encodings_generators INTERFACE cxx_std_23)

# Enable warnings (no -Werror: Arrow headers can trigger warnings we don't control)
if(MSVC)
    target_compile_options(encodings_generators INTERFACE /W4)
else()
    target_compile_options(encodings_generators INTERFACE -Wall -Wextra -Wpedantic)
endif()