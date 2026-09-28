#pragma once

#include "numcpp.h"

using namespace std;

const double WEIGHT_SCALAR = 0.01;

class LayerDense
{
public:
    Matrix weights, biases;
    Matrix output;


    LayerDense(int n_inputs, int n_neurons)
        : weights(NumCpp::randn({n_inputs, n_neurons}) * WEIGHT_SCALAR), biases(NumCpp::zeroes({1, n_neurons})),
          output(Matrix(1, n_neurons))
    {
    }

    void forward(const Matrix &inputs)
    {
        this->output = NumCpp::dot(inputs, this->weights) + this->biases;
    }
};