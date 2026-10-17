#pragma once

#include "core/matrix.h"
#include <utility>

class Activation
{
public:
    Matrix output, inputs;
    Matrix dinputs;

    virtual ~Activation() = default;

    // Enable Rule of Five Move Semantics for zero-allocation transfers
    Activation(const Activation &) = default;
    Activation(Activation &&) noexcept = default;
    Activation &operator=(const Activation &) = default;
    Activation &operator=(Activation &&) noexcept = default;

    Activation() = default;
    Activation(Matrix output, Matrix inputs, Matrix dinputs)
        : output(std::move(output)), inputs(std::move(inputs)), dinputs(std::move(dinputs))
    {
    }

    virtual void forward(const Matrix &inputs, bool cache_inputs = true) = 0;
    virtual void backward(const Matrix &dvalues) = 0;
};
