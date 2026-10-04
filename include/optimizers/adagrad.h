#pragma once

#include "optimizer.h"


class Optimizer_Adagrad : public Optimizer
{
public:
    explicit Optimizer_Adagrad(double learning_rate = 0.01, double decay = 0, double epsilon = 1e-7)
        : Optimizer(learning_rate, decay, {}, epsilon, {}, {}, {})
    {
    }

    void update_params(Layer_Dense &layer) const override
    {
        // Update cache with squared current gradients
        layer.weight_cache += layer.dweights.sq();
        layer.bias_cache += layer.dbiases.sq();

        // Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights +=
            -1 * this->current_learning_rate * layer.dweights / (NumCpp::sqrt(layer.weight_cache) + this->epsilon);

        layer.biases +=
            -1 * this->current_learning_rate * layer.dbiases / (NumCpp::sqrt(layer.bias_cache) + this->epsilon);
    }
};