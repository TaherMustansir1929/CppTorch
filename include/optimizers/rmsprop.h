#pragma once

#include "optimizer.h"


class Optimizer_RMSprop : public Optimizer
{
public:
    explicit Optimizer_RMSprop(double learning_rate = 0.001, double decay = 0, double epsilon = 1e-7, double rho = 0.9)
        : Optimizer(learning_rate, decay, {}, epsilon, rho, {}, {})
    {
    }

    void update_params(Layer_Dense &layer) const override
    {
        layer.weight_cache = this->rho * layer.weight_cache + (1 - this->rho) * layer.dweights.sq();
        layer.bias_cache = this->rho * layer.bias_cache + (1 - this->rho) * layer.dbiases.sq();

        // Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights +=
            -1 * this->current_learning_rate * layer.dweights / (NumCpp::sqrt(layer.weight_cache) + this->epsilon);

        layer.biases +=
            -1 * this->current_learning_rate * layer.dbiases / (NumCpp::sqrt(layer.bias_cache) + this->epsilon);
    }
};