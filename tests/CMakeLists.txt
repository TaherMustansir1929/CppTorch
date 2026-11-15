set(CPPTORCH_TEST_NAMES
    backward_pass_on_one_layer
    forward_pass_with_loss_cce
    full_forward_backward_pass
    full_pass_with_optimizer_sgd
    full_pass_with_optimizer_adagrad
    full_pass_with_optimizer_rmsprop
    full_pass_with_optimizer_adam
    full_pass_with_regularization
    full_pass_with_dropout_layer
    full_pass_with_mean_squared_error
    california_housing_dataset_test
    train_california_housing
    california_housing_with_cpptorch
    test_softmax_backward
    gender_audio_classification
)

add_executable(cpptorch_tests
    main.cpp
    backward_pass_on_one_layer.cpp
    forward_pass_with_loss_cce.cpp
    full_forward_backward_pass.cpp
    full_pass_with_optimizer_sgd.cpp
    full_pass_with_optimizer_adagrad.cpp
    full_pass_with_optimizer_rmsprop.cpp
    full_pass_with_optimizer_adam.cpp
    full_pass_with_regularization.cpp
    full_pass_with_dropout_layer.cpp
    full_pass_with_mean_squared_error.cpp
    california_housing_dataset_test.cpp
    train_california_housing.cpp
    california_housing_with_cpptorch.cpp
    test_softmax_backward.cpp
    gender_audio_classification.cpp
)

target_link_libraries(cpptorch_tests PRIVATE CppTorch::cpptorch)

# Point the tests at the in-tree dataset directory.
target_compile_definitions(cpptorch_tests PRIVATE
    CPPTORCH_DEFAULT_RESOURCES_PATH="${CPPTORCH_RESOURCES_DIR}")

foreach(test_name IN LISTS CPPTORCH_TEST_NAMES)
    add_test(NAME ${test_name} COMMAND cpptorch_tests ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 900)
endforeach()
