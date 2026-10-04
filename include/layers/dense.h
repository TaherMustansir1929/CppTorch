#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"


const double WEIGHT_SCALAR = 0.01;

class Layer_Dense
{
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
        // Initialize momentums
        this->weight_momentums = NumCpp::zeroes({n_inputs, n_neurons});
        this->bias_momentums = NumCpp::zeroes({1, n_neurons});
        // Initialize caches
        this->weight_cache = NumCpp::zeroes({n_inputs, n_neurons});
        this->bias_cache = NumCpp::zeroes({1, n_neurons});
    }

    void forward(const Matrix &inputs)
    {
        // store inputs
        this->inputs = inputs;

        this->output = NumCpp::dot(inputs, this->weights) + this->biases;
    }

    void backward(const Matrix &dvalues)
    {
        // Gradient on parameters
        this->dweights = NumCpp::dot(this->inputs.T(), dvalues);
        this->dbiases = NumCpp::sum(dvalues, 0);

        // Gradients on regularization
        // ---------------------------
        // L1 on weights
        if (this->weight_regularizer_l1 > 0)
        {
            Matrix dL1 = NumCpp::where(this->weights < 0, -1, 1);
            this->dweights += this->weight_regularizer_l1 * dL1;
        }
        // L2 on weights
        if (this->weight_regularizer_l2 > 0)
        {
            this->dweights += 2 * this->weight_regularizer_l2 * this->weights;
        }
        // L1 on biases
        if (this->bias_regularizer_l1 > 0)
        {
            Matrix dL1 = NumCpp::where(this->biases < 0, -1, 1);
            this->dbiases += this->bias_regularizer_l1 * dL1;
        }
        // L2 on biases
        if (this->bias_regularizer_l2 > 0)
        {
            this->dbiases += 2 * this->bias_regularizer_l2 * this->biases;
        }

        // Gradient on values
        this->dinputs = NumCpp::dot(dvalues, this->weights.T());
    }
};