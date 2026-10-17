#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"
#include <omp.h>

const double WEIGHT_SCALAR = 0.01;

class Layer_Dense
{
private:
    static constexpr size_t OMP_THRESHOLD = 8192;

public:
    Matrix weights, biases, inputs, output;
    Matrix dweights, dinputs, dbiases;
    Matrix weight_momentums, bias_momentums;
    Matrix weight_cache, bias_cache;

    double weight_regularizer_l1 = 0;
    double weight_regularizer_l2 = 0;
    double bias_regularizer_l1 = 0;
    double bias_regularizer_l2 = 0;

    Layer_Dense(size_t n_inputs,
                size_t n_neurons,
                double weight_regularizer_l1 = 0,
                double weight_regularizer_l2 = 0,
                double bias_regularizer_l1 = 0,
                double bias_regularizer_l2 = 0)
        : weights(NumCpp::randn({n_inputs, n_neurons}) * WEIGHT_SCALAR), biases(NumCpp::zeroes({1, n_neurons})),
          output(Matrix(1, n_neurons)), weight_regularizer_l1(weight_regularizer_l1),
          weight_regularizer_l2(weight_regularizer_l2), bias_regularizer_l1(bias_regularizer_l1),
          bias_regularizer_l2(bias_regularizer_l2)
    {
        this->weight_momentums = NumCpp::zeroes({n_inputs, n_neurons});
        this->bias_momentums = NumCpp::zeroes({1, n_neurons});
        this->weight_cache = NumCpp::zeroes({n_inputs, n_neurons});
        this->bias_cache = NumCpp::zeroes({1, n_neurons});
    }

    void forward(const Matrix &inputs)
    {
        this->inputs = inputs;
        NumCpp::dot_into(inputs, this->weights, this->output);
        this->output += this->biases;
    }

    void backward(const Matrix &dvalues)
    {
        NumCpp::dot_transpose_left_into(this->inputs, dvalues, this->dweights);
        this->dbiases = NumCpp::sum(dvalues, 0);

        // ==========================================
        // ===== GRADIENTS ON REGULARIZATION ========
        // ==========================================

        // Zero-allocation SIMD Penalty Integrations
        if (this->weight_regularizer_l1 > 0)
        {
#pragma omp parallel for simd if (this->weights.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < this->weights.size(); ++i)
            {
                this->dweights[i] += this->weight_regularizer_l1 * ((this->weights[i] < 0.0) ? -1.0 : 1.0);
            }
        }

        if (this->weight_regularizer_l2 > 0)
        {
#pragma omp parallel for simd if (this->weights.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < this->weights.size(); ++i)
            {
                this->dweights[i] += 2.0 * this->weight_regularizer_l2 * this->weights[i];
            }
        }

        if (this->bias_regularizer_l1 > 0)
        {
#pragma omp parallel for simd if (this->biases.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < this->biases.size(); ++i)
            {
                this->dbiases[i] += this->bias_regularizer_l1 * ((this->biases[i] < 0.0) ? -1.0 : 1.0);
            }
        }

        if (this->bias_regularizer_l2 > 0)
        {
#pragma omp parallel for simd if (this->biases.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < this->biases.size(); ++i)
            {
                this->dbiases[i] += 2.0 * this->bias_regularizer_l2 * this->biases[i];
            }
        }

        // Gradient on values
        NumCpp::dot_transpose_right_into(dvalues, this->weights, this->dinputs);
    }
};
