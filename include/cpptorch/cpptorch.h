#pragma once

/**
 * @file cpptorch.h
 * @brief Umbrella header for the CppTorch library.
 *
 * Including this single header pulls in the whole public API:
 *
 * @code
 * #include <cpptorch/cpptorch.h>
 *
 * cpptorch::Model model(cpptorch::ModelType::Classification);
 * @endcode
 *
 * The library is organised in five groups:
 *
 * - **Core** – @ref cpptorch::Matrix and @ref cpptorch::NumCpp.
 * - **Layers** – dense layers, dropout and activation functions.
 * - **Losses** – regression and classification objectives.
 * - **Optimizers** – SGD, Adagrad, RMSprop and Adam.
 * - **Model** – @ref cpptorch::Model, the high-level training API.
 *
 * Individual headers can be included instead when only part of the library is
 * needed; every header is self-contained.
 */

#include "cpptorch/core/matrix.h"
#include "cpptorch/core/numcpp.h"

#include "cpptorch/layers/activations/activation.h"
#include "cpptorch/layers/activations/activation_registry.h"
#include "cpptorch/layers/activations/linear.h"
#include "cpptorch/layers/activations/relu.h"
#include "cpptorch/layers/activations/softmax.h"
#include "cpptorch/layers/dense.h"
#include "cpptorch/layers/dropout.h"
#include "cpptorch/layers/layer.h"

#include "cpptorch/losses/categorical_crossentropy.h"
#include "cpptorch/losses/loss.h"
#include "cpptorch/losses/loss_registry.h"
#include "cpptorch/losses/mean_squared_error.h"
#include "cpptorch/losses/softmax_categoricalcrossentropy.h"

#include "cpptorch/optimizers/adagrad.h"
#include "cpptorch/optimizers/adam.h"
#include "cpptorch/optimizers/optimizer.h"
#include "cpptorch/optimizers/optimizer_registry.h"
#include "cpptorch/optimizers/rmsprop.h"
#include "cpptorch/optimizers/sgd.h"

#include "cpptorch/datasets.h"
#include "cpptorch/model.h"

#include "cpptorch/utils/exceptions.h"
#include "cpptorch/utils/resources.h"
#include "cpptorch/utils/scaler.h"
#include "cpptorch/utils/utils.h"
