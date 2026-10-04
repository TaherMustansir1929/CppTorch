#pragma once

#include "optimizer.h"


class Optimizer_SGD : public Optimizer
{
public:
    explicit Optimizer_SGD(double learning_rate = 0.01, double decay = 0, double momentum = 0)
        : Optimizer(learning_rate, decay, momentum, {}, {}, {}, {})
    {
    }

    void update_params(Layer_Dense &layer) const override
    {
        Matrix weight_updates = this->momentum * layer.weight_momentums - this->current_learning_rate * layer.dweights;
        layer.weight_momentums = weight_updates;

        Matrix bias_updates = this->momentum * layer.bias_momentums - this->current_learning_rate * layer.dbiases;
        layer.bias_momentums = bias_updates;

        layer.weights += weight_updates;
        layer.biases += bias_updates;
    }
};