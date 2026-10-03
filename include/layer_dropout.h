#pragma once

#include "matrix.h"
#include "numcpp.h"


class Layer_Dropout
{
public:
    double rate;
    Matrix inputs, output, binary_mask;
    Matrix dinputs;


    explicit Layer_Dropout(double rate) : rate(1 - rate)
    {
        inputs = output = binary_mask = dinputs = Matrix(0, 0);
    }

    void forward(const Matrix &inputs)
    {
        this->inputs = inputs;

        // generate and save the scaled binary mask
        this->binary_mask = NumCpp::binomial(1, rate, inputs.shape()) / this->rate;

        // apply mask to output values
        this->output = inputs * this->binary_mask;
    }

    void backward(const Matrix &dvalues)
    {
        this->dinputs = dvalues * this->binary_mask;
    }
};