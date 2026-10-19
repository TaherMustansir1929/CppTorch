#pragma once

#include "activation.h"
#include "core/numcpp.h"
#include <omp.h>

class Activation_ReLU : public Activation
{
public:
    void forward(const Matrix &inputs, bool cache_inputs = true) override
    {
        if (cache_inputs)
            this->inputs = inputs;
        this->output.resize(inputs.rows(), inputs.cols());
#pragma omp parallel for simd if (inputs.size() > 16384)
        for (size_t i = 0; i < inputs.size(); ++i)
            this->output[i] = std::max(0.0, inputs[i]);
    }

    void backward(const Matrix &dvalues) override
    {
        this->dinputs.resize(dvalues.rows(), dvalues.cols());

#pragma omp parallel for simd if (this->output.size() > 16384)
        for (size_t i = 0; i < this->output.size(); i++)
        {
            this->dinputs[i] = (this->output[i] <= 0.0) ? 0.0 : dvalues[i];
        }
    }
};
