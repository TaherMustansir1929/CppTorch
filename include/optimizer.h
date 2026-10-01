#pragma once

#include "layer_dense.h"
class Optimizer_VGD
{
public:
    double learning_rate;
    double current_learning_rate = 0;
    double decay;
    double momentum;
    int iteration = 0;

    explicit Optimizer_VGD(double learning_rate = 1, double decay = 0, double momentum = 0)
        : learning_rate(learning_rate), decay(decay), momentum(momentum)
    {
    }

    void pre_update_params()
    {
        this->current_learning_rate = this->learning_rate * (1 / (1 + (this->decay * this->iteration)));
    }

    void update_params(Layer_Dense &layer) const
    {
        Matrix weight_updates = this->momentum * layer.weight_momentums - this->current_learning_rate * layer.dweights;
        layer.weight_momentums = weight_updates;

        Matrix bias_updates = this->momentum * layer.bias_momentums - this->current_learning_rate * layer.dbiases;
        layer.bias_momentums = bias_updates;

        layer.weights += weight_updates;
        layer.biases += bias_updates;
    }

    void post_update_params()
    {
        this->iteration++;
    }
};