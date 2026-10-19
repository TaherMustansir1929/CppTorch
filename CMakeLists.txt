cmake_minimum_required(VERSION 3.20)

set(PRODUCTION_BUILD OFF CACHE BOOL "Make this a production build" FORCE)

project(CppTorch LANGUAGES CXX)

# Enable OpenMP
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(CppTorch ${MY_SOURCES} main.cpp)

# OpenBLAS is required for dense matrix multiplication.
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_path(CBLAS_INCLUDE_DIR NAMES cblas.h
    PATH_SUFFIXES x86_64-linux-gnu/openblas-pthread openblas-pthread x86_64-linux-gnu REQUIRED)
target_include_directories(CppTorch PRIVATE ${CBLAS_INCLUDE_DIR})
target_link_libraries(CppTorch PRIVATE BLAS::BLAS)

# Link OpenMP to the executable
target_link_libraries(CppTorch PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(CppTorch PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(CppTorch PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(CppTorch PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2>
    )
else()
    target_compile_options(CppTorch PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O2>
        $<$<CONFIG:Release>:-march=native>
    )
endif()

if(PRODUCTION_BUILD)
    target_compile_definitions(CppTorch PUBLIC RESOURCES_PATH="./resources/")
    target_compile_definitions(CppTorch PUBLIC PRODUCTION_BUILD=1)
    target_compile_definitions(CppTorch PUBLIC DEVELOPMENT_BUILD=0)
else()
    target_compile_definitions(CppTorch PUBLIC
        RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/"
    )
    target_compile_definitions(CppTorch PUBLIC PRODUCTION_BUILD=0)
    target_compile_definitions(CppTorch PUBLIC DEVELOPMENT_BUILD=1)
endif()

if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)

    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,
        $<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,
        $<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>"
    )
endif()
