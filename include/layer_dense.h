#pragma once

#include "matrix.h"
#include "numcpp.h"

const double WEIGHT_SCALAR = 0.01;

class Layer_Dense
{
public:
    Matrix weights, biases, inputs, output;
    Matrix dweights, dinputs, dbiases;
    Matrix weight_momentums, bias_momentums;
    Matrix weight_cache, bias_cache;


    Layer_Dense(int n_inputs, int n_neurons)
        : weights(NumCpp::randn({n_inputs, n_neurons}) * WEIGHT_SCALAR), biases(NumCpp::zeroes({1, n_neurons})),
          output(Matrix(1, n_neurons))
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

        // Gradient on values
        this->dinputs = NumCpp::dot(dvalues, this->weights.T());
    }
};