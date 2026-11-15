add_executable(cpptorch_example_quickstart quickstart.cpp)
target_link_libraries(cpptorch_example_quickstart PRIVATE CppTorch::cpptorch)

add_executable(cpptorch_example_gender_audio_classification gender_audio_classification.cpp)
target_link_libraries(cpptorch_example_gender_audio_classification PRIVATE CppTorch::cpptorch)

# Point the examples at the in-tree dataset directory.
target_compile_definitions(cpptorch_example_gender_audio_classification PRIVATE
    CPPTORCH_DEFAULT_RESOURCES_PATH="${CPPTORCH_RESOURCES_DIR}")
