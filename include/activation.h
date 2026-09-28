#pragma once

#include "matrix.h"
#include "numcpp.h"

class Activation_ReLU
{
public:
    Matrix output;

    void forward(const Matrix &input)
    {
        output = NumCpp::maximum(0, input);
    }
};

class Activation_Softmax
{
public:
    Matrix output;

    void forward(const Matrix &input)
    {
        Matrix exp_values = NumCpp::exp(input - NumCpp::max(input, 1));
        Matrix probabilities = exp_values / NumCpp::sum(exp_values, 1);

        output = probabilities;
    }
};