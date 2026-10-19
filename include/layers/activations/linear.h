#pragma once

#include "activation.h"

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
