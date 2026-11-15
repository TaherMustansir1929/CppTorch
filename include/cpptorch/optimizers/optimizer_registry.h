#pragma once

#include "cpptorch/optimizers/adagrad.h"
#include "cpptorch/optimizers/adam.h"
#include "cpptorch/optimizers/optimizer.h"
#include "cpptorch/optimizers/rmsprop.h"
#include "cpptorch/optimizers/sgd.h"

#include <memory>

namespace cpptorch
{

/**
 * @brief Creates the optimizer identified by @p type.
 *
 * This is the **optimizer plug-in point**. To expose a new optimizer to the
 * rest of the library:
 *
 * 1. Create `include/cpptorch/optimizers/<name>.h` with a class deriving
 *    from @ref cpptorch::Optimizer.
 * 2. Add a value to @ref cpptorch::OptimizerType.
 * 3. Add a `case` to the switch below.
 * 4. Add the new header to `include/cpptorch/cpptorch.h`.
 *
 * @param type   The optimizer to instantiate.
 * @param params Hyper-parameters forwarded to the optimizer constructor.
 * @return A newly constructed optimizer.
 * @throws std::invalid_argument if @p type has no registered factory.
 */
inline std::unique_ptr<Optimizer> make_optimizer(OptimizerType type, const HyperParameters_Optimizer &params)
{
    switch (type)
    {
    case OptimizerType::SGD:
        return std::make_unique<Optimizer_SGD>(params.learning_rate, params.decay, params.momentum);
    case OptimizerType::Adagrad:
        return std::make_unique<Optimizer_Adagrad>(params.learning_rate, params.decay, params.epsilon);
    case OptimizerType::RMSprop:
        return std::make_unique<Optimizer_RMSprop>(params.learning_rate, params.decay, params.epsilon, params.rho);
    case OptimizerType::Adam:
        return std::make_unique<Optimizer_Adam>(params.learning_rate,
                                                 params.decay,
                                                 params.epsilon,
                                                 params.beta_1,
                                                 params.beta_2);
    }
    throw std::invalid_argument("Unknown OptimizerType in make_optimizer");
}

} // namespace cpptorch
