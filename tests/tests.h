#pragma once

#include <cpptorch/cpptorch.h>

namespace cpptorch::tests
{

void backward_pass_on_one_layer();
void forward_pass_with_loss_cce();
void full_forward_backward_pass();
void full_pass_with_optimizer_sgd();
void full_pass_with_optimizer_adagrad();
void full_pass_with_optimizer_rmsprop();
void full_pass_with_optimizer_adam();
void full_pass_with_regularization();
void full_pass_with_dropout_layer();
void full_pass_with_mean_squared_error();
void california_housing_dataset_test();
void train_california_housing();
void california_housing_with_cpptorch();
void test_softmax_backward();
void gender_audio_classification();

} // namespace cpptorch::tests
