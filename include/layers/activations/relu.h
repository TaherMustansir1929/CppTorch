#pragma once

#include "activation.h"
#include "core/numcpp.h"


class Activation_ReLU : public Activation
{
public:
    void forward(const Matrix &inputs) override
    {
        // remember inputs
        this->inputs = inputs;

        this->output = NumCpp::maximum(0, inputs);
    }

    void backward(const Matrix &dvalues) override
    {
        // copy dvalues
        this->dinputs = dvalues;
        // Zero gradient where input values were negative
        for (int i = 0; i < this->inputs.size(); i++)
        {
            if (this->inputs[i] <= 0)
            {
                this->dinputs[i] = 0;
            }
        }
    }
};