#pragma once

#include "cpptorch/core/matrix.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace cpptorch
{

/**
 * @brief Identifies the activation function applied at the end of a layer.
 *
 * The enum is the single entry point of the activation plug-in system: add a
 * value here and a matching case in @ref cpptorch::make_activation to expose
 * a new activation to @ref cpptorch::Model.
 */
enum class ActivationType : std::uint8_t
{
    Linear = 0, ///< Identity activation, f(x) = x.
    ReLU = 1,   ///< Rectified linear unit, f(x) = max(0, x).
    Softmax = 2 ///< Softmax normalisation across the feature axis.
};

/**
 * @brief Base class for activation functions.
 *
 * An activation owns the buffers produced during the forward pass and the
 * gradients computed during the backward pass.
 *
 * @note To add a new activation, derive from this class, implement
 *       forward() and backward(), then register it in
 *       @ref cpptorch::make_activation.
 */
class Activation
{
public:
    /// Outputs produced by the most recent forward() call.
    Matrix output;
    /// Inputs cached by the most recent forward() call.
    Matrix inputs;
    /// Gradients with respect to the inputs, filled by backward().
    Matrix dinputs;

    virtual ~Activation() = default;

    // Enable Rule of Five Move Semantics for zero-allocation transfers
    Activation(const Activation &) = default;
    Activation(Activation &&) noexcept = default;
    Activation &operator=(const Activation &) = default;
    Activation &operator=(Activation &&) noexcept = default;

    Activation() = default;

    /**
     * @brief Constructs an activation with pre-allocated buffers.
     */
    Activation(Matrix output, Matrix inputs, Matrix dinputs)
        : output(std::move(output)), inputs(std::move(inputs)), dinputs(std::move(dinputs))
    {
    }

    /**
     * @brief Runs the activation forward.
     *
     * @param inputs       Input matrix of shape (samples, features).
     * @param cache_inputs When true, @ref inputs is stored so that backward()
     *                     can use it.
     */
    virtual void forward(const Matrix &inputs, bool cache_inputs = true) = 0;

    /**
     * @brief Computes the gradient of the loss with respect to the inputs.
     *
     * @param dvalues Gradient of the loss with respect to @ref output.
     */
    virtual void backward(const Matrix &dvalues) = 0;
};

} // namespace cpptorch
