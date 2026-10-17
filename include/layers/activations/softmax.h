#pragma once

#include "activation.h"
#include "core/numcpp.h"
#include <omp.h>

class Activation_Softmax : public Activation
{
public:
    void forward(const Matrix &inputs, bool cache_inputs = true) override
    {
        if (cache_inputs)
            this->inputs = inputs;
        const size_t samples = inputs.rows();
        const size_t features = inputs.cols();
        this->output.resize(samples, features);
#pragma omp parallel for if (samples * features > 16384)
        for (size_t i = 0; i < samples; ++i)
        {
            const size_t offset = i * features;
            double max_value = inputs[offset];
            for (size_t j = 1; j < features; ++j)
                max_value = std::max(max_value, inputs[offset + j]);

            double total = 0.0;
#pragma omp simd reduction(+ : total)
            for (size_t j = 0; j < features; ++j)
            {
                this->output[offset + j] = std::exp(inputs[offset + j] - max_value);
                total += this->output[offset + j];
            }

#pragma omp simd
            for (size_t j = 0; j < features; ++j)
                this->output[offset + j] /= total;
        }
    }

    void backward(const Matrix &dvalues) override
    {
        size_t samples = dvalues.rows();
        size_t features = dvalues.cols();
        this->dinputs.resize(samples, features);

        // Mathematical Simplification:
        // J = diag(s) - s * s.T
        // dinputs = J * d = s * d - s * (s.T * d)
        // Element-wise: dinputs_i = s_i * (d_i - sum(s_j * d_j))

#pragma omp parallel for if (samples * features > 16384)
        for (size_t i = 0; i < samples; i++)
        {
            double dot_product = 0.0;
            size_t row_offset = i * features;

// 1. Calculate the scalar dot product of output and dvalues for this sample
#pragma omp simd reduction(+ : dot_product)
            for (size_t j = 0; j < features; j++)
            {
                dot_product += this->output[row_offset + j] * dvalues[row_offset + j];
            }

// 2. Apply the simplified Jacobian element-wise
#pragma omp simd
            for (size_t j = 0; j < features; j++)
            {
                this->dinputs[row_offset + j] = this->output[row_offset + j] * (dvalues[row_offset + j] - dot_product);
            }
        }
    }
};
