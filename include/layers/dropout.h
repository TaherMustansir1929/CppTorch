#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"

class Layer_Dropout
{
public:
    double rate;
    Matrix inputs, output, binary_mask;
    Matrix dinputs;

    explicit Layer_Dropout(double rate) : rate(1.0 - rate)
    {
        inputs = Matrix(0, 0);
        output = Matrix(0, 0);
        binary_mask = Matrix(0, 0);
        dinputs = Matrix(0, 0);
    }

    void forward(const Matrix &inputs, bool cache_inputs = true)
    {
        if (cache_inputs)
            this->inputs = inputs;
        NumCpp::binomial_into(1, rate, inputs.shape(), this->binary_mask);
        this->output.resize(inputs.rows(), inputs.cols());
#pragma omp parallel for simd if (inputs.size() > 16384)
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            this->binary_mask[i] /= this->rate;
            this->output[i] = inputs[i] * this->binary_mask[i];
        }
    }

    void backward(const Matrix &dvalues)
    {
        this->dinputs.resize(dvalues.rows(), dvalues.cols());
#pragma omp parallel for simd if (dvalues.size() > 16384)
        for (size_t i = 0; i < dvalues.size(); ++i)
            this->dinputs[i] = dvalues[i] * this->binary_mask[i];
    }
};
