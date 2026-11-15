#pragma once

#include "cpptorch/layers/activations/activation.h"
#include "cpptorch/layers/activations/linear.h"
#include "cpptorch/layers/activations/relu.h"
#include "cpptorch/layers/activations/softmax.h"

#include <memory>

namespace cpptorch
{

/**
 * @brief Creates the activation function identified by @p type.
 *
 * This is the **activation plug-in point**. To expose a new activation to the
 * rest of the library:
 *
 * 1. Create `include/cpptorch/layers/activations/<name>.h` with a class
 *    deriving from @ref cpptorch::Activation.
 * 2. Add a value to @ref cpptorch::ActivationType.
 * 3. Add a `case` to the switch below.
 * 4. Add the new header to `include/cpptorch/cpptorch.h`.
 *
 * @param type The activation to instantiate.
 * @return A newly constructed activation.
 * @throws std::invalid_argument if @p type has no registered factory.
 */
inline std::unique_ptr<Activation> make_activation(ActivationType type)
{
    switch (type)
    {
    case ActivationType::Linear:
        return std::make_unique<Activation_Linear>();
    case ActivationType::ReLU:
        return std::make_unique<Activation_ReLU>();
    case ActivationType::Softmax:
        return std::make_unique<Activation_Softmax>();
    }
    throw std::invalid_argument("Unknown ActivationType in make_activation");
}

} // namespace cpptorch
