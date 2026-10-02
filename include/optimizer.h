#pragma once

#include "layer_dense.h"
#include "numcpp.h"


class Optimizer_SGD
{
public:
    double learning_rate;
    double current_learning_rate = 0;
    double decay;
    double momentum;
    int iteration = 0;

    explicit Optimizer_SGD(double learning_rate = 1, double decay = 0, double momentum = 0)
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


class Optimizer_Adagrad
{
public:
    double learning_rate = 1;
    double current_learning_rate = 0;
    double decay = 0;
    double epsilon = 1e-7;
    int iteration = 0;


    explicit Optimizer_Adagrad(double learning_rate = 1, double decay = 0, double epsilon = 1e-7)
        : learning_rate(learning_rate), current_learning_rate(learning_rate), decay(decay), epsilon(epsilon)
    {
    }

    void pre_update_params()
    {
        this->current_learning_rate = this->learning_rate * (1 / (1 + (this->decay * this->iteration)));
    }

    void update_params(Layer_Dense &layer) const
    {
        // Update cache with squared current gradients
        layer.weight_cache += (layer.dweights ^ 2);
        layer.bias_cache += (layer.dbiases ^ 2);

        // Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights +=
            -1 * this->current_learning_rate * layer.dweights / (NumCpp::sqrt(layer.weight_cache) + this->epsilon);

        layer.biases +=
            -1 * this->current_learning_rate * layer.dbiases / (NumCpp::sqrt(layer.bias_cache) + this->epsilon);
    }

    void post_update_params()
    {
        this->iteration++;
    }
};


class Optimizer_RMSprop
{
public:
    double learning_rate = 1;
    double current_learning_rate = 0;
    double decay = 0;
    double epsilon = 1e-7;
    double rho = 0.9;
    int iteration = 0;


    explicit Optimizer_RMSprop(double learning_rate = 0.001, double decay = 0, double epsilon = 1e-7, double rho = 0.9)
        : learning_rate(learning_rate), current_learning_rate(learning_rate), decay(decay), epsilon(epsilon), rho(rho)
    {
    }

    void pre_update_params()
    {
        this->current_learning_rate = this->learning_rate * (1 / (1 + (this->decay * this->iteration)));
    }

    void update_params(Layer_Dense &layer) const
    {
        // Fix 1: Added parentheses around (layer.dweights ^ 2) to bypass C++ precedence
        layer.weight_cache = this->rho * layer.weight_cache + (1 - this->rho) * (layer.dweights ^ 2);

        // Fix 2: Changed += to = and added parentheses
        layer.bias_cache = this->rho * layer.bias_cache + (1 - this->rho) * (layer.dbiases ^ 2);

        // Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights +=
            -1 * this->current_learning_rate * layer.dweights / (NumCpp::sqrt(layer.weight_cache) + this->epsilon);

        layer.biases +=
            -1 * this->current_learning_rate * layer.dbiases / (NumCpp::sqrt(layer.bias_cache) + this->epsilon);
    }

    void post_update_params()
    {
        this->iteration++;
    }
};