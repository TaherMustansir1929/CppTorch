cmake_minimum_required(VERSION 3.20)

set(PRODUCTION_BUILD OFF CACHE BOOL "Make this a production build" FORCE)

project(CppTorch LANGUAGES CXX)

file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(CppTorch ${MY_SOURCES} main.cpp)

target_include_directories(CppTorch PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(CppTorch PRIVATE cxx_std_20)

if(PRODUCTION_BUILD)
	# setup the ASSETS_PATH macro to be in the root folder of your exe
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC RESOURCES_PATH="./resources/") 

	# remove the option to debug asserts.
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC PRODUCTION_BUILD=1) 
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC DEVELOPMENT_BUILD=0) 

else()
	# This is useful to get an ASSETS_PATH in your IDE during development
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/")
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC PRODUCTION_BUILD=0) 
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC DEVELOPMENT_BUILD=1) 

endif()

if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)

    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,
        $<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,
        $<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>"
    )
endif()