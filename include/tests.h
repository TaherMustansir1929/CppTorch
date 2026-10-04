#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"
#include "datasets.h"
#include "layers/activations/relu.h"
#include "layers/activations/softmax.h"
#include "layers/activations/linear.h"
#include "layers/dense.h"
#include "layers/dropout.h"
#include "losses/categorical_crossentropy.h"
#include "losses/loss.h"
#include "losses/mean_squared_error.h"
#include "losses/softmax_categoricalcrossentropy.h"
#include "optimizers/adagrad.h"
#include "optimizers/adam.h"
#include "optimizers/rmsprop.h"
#include "optimizers/sgd.h"
#include "utils/scaler.h"

#include <cmath>
#include <cstdio>
#include <iostream>

class Tests
{
public:
    static void forwardPass_with_Loss_CCE();
    static void backwardPass_on_one_Layer();
    static void full_forward_backward_pass();
    static void full_pass_with_optimizer_vgd();
    static void full_pass_with_optimizer_adagrad();
    static void full_pass_with_optimizer_rmsprop();
    static void full_pass_with_optimizer_adam();
    static void full_pass_with_regularization();
    static void full_pass_with_dropout_layer();
    static void full_pass_with_mean_squared_error();
    static void california_housing_dataset_test();
    static void train_california_housing();
    static void california_housing_with_cpptorch();
    static void test_softmax_backward();
};