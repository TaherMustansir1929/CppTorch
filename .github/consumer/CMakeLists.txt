cmake_minimum_required(VERSION 3.21)
project(cpptorch_consumer LANGUAGES CXX)

# Smoke test for the installed package: it must be consumable through
# find_package() + the exported CppTorch::cpptorch target alone.
find_package(cpptorch 0.1 REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE CppTorch::cpptorch)
