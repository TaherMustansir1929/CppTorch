#pragma once

#include <algorithm>

#include "matrix.h"
#include "numcpp.h"

class Activation_ReLU
{
public:
    Matrix output, inputs;
    Matrix dinputs;


    void forward(const Matrix &inputs)
    {
        // remember inputs
        this->inputs = inputs;

        this->output = NumCpp::maximum(0, inputs);
    }

    void backward(const Matrix &dvalues)
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

class Activation_Softmax
{
public:
    Matrix output;

    void forward(const Matrix &inputs)
    {
        Matrix exp_values = NumCpp::exp(inputs - NumCpp::max(inputs, 1));
        Matrix probabilities = exp_values / NumCpp::sum(exp_values, 1);

        this->output = probabilities;
    }

    // backward pass combined with categorical cross entropy loss class <loss.h> file
};

class Activation_Linear
{
public:
    Matrix output, inputs;
    Matrix dinputs;

    void forward(const Matrix &inputs)
    {
        this->inputs = inputs;
        this->output = inputs;
    }

    void backward(const Matrix &dvalues)
    {
        this->dinputs = dvalues;
    }
};
