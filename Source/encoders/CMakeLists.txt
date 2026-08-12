# Encoders library - concrete encoder implementations

# ---------------------------------------------------------------------------
# ankerl::unordered_dense  (header-only, faster than std::unordered_map)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG        v4.5.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(unordered_dense)

find_package(ZSTD QUIET)
if(NOT ZSTD_FOUND)
    # Try pkg-config as a fallback (Debian/Ubuntu libzstd-dev provides zstd.pc)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(ZSTD_PKG zstd)
        if(ZSTD_PKG_FOUND)
            message(STATUS "Found zstd via pkg-config: ${ZSTD_PKG_VERSION}")
            # Create an IMPORTED interface target so the rest of the file can link
            # against ZSTD::libzstd as before.
            add_library(ZSTD::libzstd INTERFACE IMPORTED)
            set_target_properties(ZSTD::libzstd PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_PKG_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${ZSTD_PKG_LIBRARIES}")
            set(ZSTD_FOUND TRUE)
        endif()
    endif()
endif()

find_package(LZ4 QUIET)
if(NOT LZ4_FOUND)
    # Try pkg-config fallback (Debian/Ubuntu: liblz4-dev provides liblz4.pc)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(LZ4_PKG liblz4)
        if(LZ4_PKG_FOUND)
            message(STATUS "Found LZ4 via pkg-config: ${LZ4_PKG_VERSION}")
            add_library(LZ4::lz4 INTERFACE IMPORTED)
            set_target_properties(LZ4::lz4 PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${LZ4_PKG_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${LZ4_PKG_LIBRARIES}")
            set(LZ4_FOUND TRUE)
        endif()
    endif()
endif()
if(NOT LZ4_FOUND)
    # Manual fallback
    find_path(LZ4_INCLUDE_DIR lz4.h)
    find_library(LZ4_LIBRARY NAMES lz4 liblz4)
    if(LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
        message(STATUS "Found LZ4 manually: include=${LZ4_INCLUDE_DIR} lib=${LZ4_LIBRARY}")
        add_library(LZ4::lz4 UNKNOWN IMPORTED)
        set_target_properties(LZ4::lz4 PROPERTIES
            IMPORTED_LOCATION "${LZ4_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}")
        set(LZ4_FOUND TRUE)
    else()
        message(FATAL_ERROR "Could not find LZ4. Install liblz4-dev or set CMAKE_PREFIX_PATH.")
    endif()
endif()
if(NOT ZSTD_FOUND)
    # Try a manual search for headers and library if pkg-config/CMake config are not available
    find_path(ZSTD_INCLUDE_DIR zstd.h)
    find_library(ZSTD_LIBRARY NAMES zstd zstd_static libzstd)
    if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
        message(STATUS "Found zstd manually: include=${ZSTD_INCLUDE_DIR} lib=${ZSTD_LIBRARY}")
        # Create an IMPORTED target so other code can link to ZSTD::libzstd as before.
        add_library(ZSTD::libzstd UNKNOWN IMPORTED)
        set_target_properties(ZSTD::libzstd PROPERTIES
            IMPORTED_LOCATION "${ZSTD_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}")
        set(ZSTD_FOUND TRUE)
    else()
        message(FATAL_ERROR "Could not find ZSTD. Install libzstd-dev or set ZSTD_DIR/CMAKE_PREFIX_PATH")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenZL (prebuilt; state-of-the-art baseline and a SubIntSplit section codec)
#
# HAVE_OPENZL is defined here, at the build level, and NOT by OpenZLEncoder.hpp.
# It used to be #defined inside that header behind a __has_include probe, which
# made it depend on include order: a translation unit that reached its own
# `#ifdef HAVE_OPENZL` before transitively including OpenZLEncoder.hpp silently
# compiled with OpenZL disabled.  Defining it on the interface target makes it
# uniform across every TU, so reordering includes can no longer change which
# encoders a driver registers.
# ---------------------------------------------------------------------------
option(ENCODINGS_ENABLE_OPENZL "Build with the OpenZL codec and baselines" ON)

add_library(encodings_encoders INTERFACE)

if(ENCODINGS_ENABLE_OPENZL)
    # Resolution order: -DOPENZL_REPO, then $OPENZL_REPO, then a search of
    # locations relative to this repo.  The relative probe replaces what used to
    # be a hardcoded absolute path under a specific home directory, which was
    # invisible until the build failed to link on any other machine.
    if(NOT DEFINED OPENZL_REPO OR NOT OPENZL_REPO)
        set(OPENZL_REPO "$ENV{OPENZL_REPO}")
    endif()
    if(NOT OPENZL_REPO)
        foreach(_candidate
                "${PROJECT_SOURCE_DIR}/../openzl"
                "${PROJECT_SOURCE_DIR}/../../openzl"
                "${PROJECT_SOURCE_DIR}/../../../openzl")
            if(EXISTS "${_candidate}/include/openzl/openzl.h")
                get_filename_component(OPENZL_REPO "${_candidate}" ABSOLUTE)
                message(STATUS "OpenZL found relative to the repo: ${OPENZL_REPO}")
                break()
            endif()
        endforeach()
    endif()
    if(NOT OPENZL_REPO)
        message(FATAL_ERROR
            "ENCODINGS_ENABLE_OPENZL is ON but no OpenZL checkout was found.  Point "
            "OPENZL_REPO at a prebuilt checkout containing include/openzl/openzl.h and "
            "libopenzl${CMAKE_SHARED_LIBRARY_SUFFIX}, e.g. -DOPENZL_REPO=/path/to/openzl "
            "(or export OPENZL_REPO), or configure with -DENCODINGS_ENABLE_OPENZL=OFF.")
    endif()

    set(OPENZL_INCLUDE_DIR "${OPENZL_REPO}/include")
    set(OPENZL_LIBRARY "${OPENZL_REPO}/libopenzl${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${OPENZL_INCLUDE_DIR}/openzl/openzl.h")
        message(FATAL_ERROR "OPENZL_REPO=${OPENZL_REPO} has no include/openzl/openzl.h")
    endif()
    if(NOT EXISTS "${OPENZL_LIBRARY}")
        message(FATAL_ERROR "OPENZL_REPO=${OPENZL_REPO} has no libopenzl${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    add_library(OpenZL::openzl SHARED IMPORTED)
    set_target_properties(OpenZL::openzl PROPERTIES
        IMPORTED_LOCATION "${OPENZL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENZL_INCLUDE_DIR}"
    )
    target_link_libraries(encodings_encoders INTERFACE OpenZL::openzl)
    target_compile_definitions(encodings_encoders INTERFACE HAVE_OPENZL)
    message(STATUS "OpenZL enabled: ${OPENZL_REPO}")
else()
    message(STATUS "OpenZL disabled (ENCODINGS_ENABLE_OPENZL=OFF); OpenZLCodec will throw if used")
endif()

target_sources(encodings_encoders INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/Source
    FILES
        RawEncoder.hpp
        RawBitPackedEncoder.hpp
        RunLengthEncoder.hpp
        DeltaEncoder.hpp
        DeltaCodec.hpp
        DictionaryEncoder.hpp
        AdaptiveFramedBitPrefixEncoder.hpp
        ZstdEncoder.hpp
        SphericalEncoder.hpp
        MapEncoders.hpp
        SubIntEncoder.hpp
        VarIntEncoder.hpp
        DeltaVarIntEncoder.hpp
        FOREncoder.hpp
        CascadingFOREncoder.hpp
        OpenZLEncoder.hpp
        LZ4Encoder.hpp
        FSEEncoder.hpp
        BWTSectionEncoder.hpp
        selectors/SubStreamReordererType.hpp
        analysis/CascadeCardinalityAnalyzer.hpp
)

target_include_directories(encodings_encoders INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_encoders INTERFACE
    encodings_core
    encodings_encodings
    ZSTD::libzstd
    LZ4::lz4
    unordered_dense
)

target_compile_features(encodings_encoders INTERFACE cxx_std_23)

# Enable warnings
if(MSVC)
    target_compile_options(encodings_encoders INTERFACE /W4 /WX)
else()
    target_compile_options(encodings_encoders INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()

add_subdirectory(selectors)
