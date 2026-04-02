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

# Assumes a prebuilt openzl lib is available and provided
if (NOT DEFINED OPENZL_REPO)
    set(OPENZL_REPO "/home/david/Documents/PhD/symbol-store/openzl")
endif()
set(OPENZL_INCLUDE_DIR "${OPENZL_REPO}/include")
set(OPENZL_LIB_DIR "${OPENZL_REPO}")


add_library(encodings_encoders INTERFACE)

add_library(OpenZL::openzl SHARED IMPORTED)
set_target_properties(OpenZL::openzl PROPERTIES
    IMPORTED_LOCATION "${OPENZL_LIB_DIR}/libopenzl${CMAKE_SHARED_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENZL_INCLUDE_DIR}"
)

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
        OpenZLEncoder.hpp
)

target_include_directories(encodings_encoders INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(encodings_encoders INTERFACE 
    encodings_core
    encodings_encodings
    ZSTD::libzstd
    OpenZL::openzl
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
