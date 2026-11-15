#pragma once

#include "cpptorch/layers/activations/activation.h"

namespace cpptorch
{

/**
 * @brief Identity activation, `f(x) = x`.
 */
class Activation_Linear : public Activation
{
public:
    void forward(const Matrix &inputs, bool cache_inputs = true) override
    {
        if (cache_inputs)
            this->inputs = inputs;
        this->output = inputs;
    }

    void backward(const Matrix &dvalues) override
    {
        this->dinputs = dvalues;
    }
};

} // namespace cpptorch
