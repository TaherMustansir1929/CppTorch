#pragma once

#include "cpptorch/losses/categorical_crossentropy.h"
#include "cpptorch/losses/loss.h"
#include "cpptorch/losses/mean_squared_error.h"

#include <memory>

namespace cpptorch
{

/**
 * @brief Creates the loss function identified by @p type.
 *
 * This is the **loss plug-in point**. To expose a new loss to the rest of the
 * library:
 *
 * 1. Create `include/cpptorch/losses/<name>.h` with a class deriving from
 *    @ref cpptorch::Loss.
 * 2. Add a value to @ref cpptorch::LossType.
 * 3. Add a `case` to the switch below.
 * 4. Add the new header to `include/cpptorch/cpptorch.h`.
 *
 * @param type The loss to instantiate.
 * @return A newly constructed loss.
 * @throws std::invalid_argument if @p type has no registered factory.
 */
inline std::unique_ptr<Loss> make_loss(LossType type)
{
    switch (type)
    {
    case LossType::MeanSquaredError:
        return std::make_unique<Loss_MeanSquaredError>();
    case LossType::CategoricalCrossEntropy:
        return std::make_unique<Loss_CategoricalCrossEntropy>();
    }
    throw std::invalid_argument("Unknown LossType in make_loss");
}

} // namespace cpptorch
