cmake_minimum_required(VERSION 3.21)

project(cpptorch
    VERSION 0.1.0
    DESCRIPTION "Minimal neural networks library with numpy-style array primitives"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(cpptorch_BUILD_TESTS "Build the CppTorch test suite" ${PROJECT_IS_TOP_LEVEL})
option(cpptorch_BUILD_EXAMPLES "Build the CppTorch examples" ${PROJECT_IS_TOP_LEVEL})
option(cpptorch_INSTALL "Generate the CppTorch install rules" ${PROJECT_IS_TOP_LEVEL})

# Directory holding the bundled CSV datasets. Forward slashes keep the path
# usable inside a -D definition on every platform.
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/resources/" CPPTORCH_RESOURCES_DIR)

# ---------------------------------------------------------------------------
# The library
#
# CppTorch is header-only, so the CMake target is an INTERFACE library. Every
# consumer inherits the include path, the C++20 requirement and the BLAS /
# OpenMP dependencies through the exported target.
# ---------------------------------------------------------------------------
add_library(cpptorch INTERFACE)
add_library(CppTorch::cpptorch ALIAS cpptorch)

target_compile_features(cpptorch INTERFACE cxx_std_20)

target_include_directories(cpptorch INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# OpenMP powers the parallel kernels.
find_package(OpenMP REQUIRED)

# A CBLAS provider is required for the dense matrix products.
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

# cblas.h lives in a separate development package and is not exposed by the
# BLAS::BLAS imported target, so it is looked up on its own.
find_path(CBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATH_SUFFIXES openblas-pthread openblas x86_64-linux-gnu/openblas-pthread x86_64-linux-gnu
)
if(NOT CBLAS_INCLUDE_DIR)
    message(FATAL_ERROR
        "cblas.h was not found. Install the OpenBLAS development package or "
        "point CBLAS_INCLUDE_DIR at the directory that contains it.")
endif()

target_include_directories(cpptorch SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIR}>
)

target_link_libraries(cpptorch INTERFACE
    OpenMP::OpenMP_CXX
    BLAS::BLAS
)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
if(cpptorch_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(cpptorch_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
if(cpptorch_INSTALL)
    install(TARGETS cpptorch EXPORT cpptorchTargets)

    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/cpptorch"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            FILES_MATCHING PATTERN "*.h")

    install(EXPORT cpptorchTargets
            FILE cpptorchTargets.cmake
            NAMESPACE CppTorch::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cpptorch")

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpptorchConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cpptorchConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cpptorch"
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/cpptorchConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/cpptorchConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/cpptorchConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cpptorch")
endif()
