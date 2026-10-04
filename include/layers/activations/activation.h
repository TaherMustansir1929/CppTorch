#pragma once

#include "core/matrix.h"


class Activation
{
public:
    Matrix output, inputs;
    Matrix dinputs;


    virtual ~Activation() = default;
    Activation(const Activation &) = default;
    Activation(Activation &&) = delete;
    Activation &operator=(const Activation &) = default;
    Activation &operator=(Activation &&) = delete;

    Activation() = default;
    Activation(Matrix output, Matrix inputs, Matrix dinputs)
        : output(std::move(output)), inputs(std::move(inputs)), dinputs(std::move(dinputs))
    {
    }

    virtual void forward(const Matrix &inputs) = 0;
    virtual void backward(const Matrix &dvalues) = 0;
};
