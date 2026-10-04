#pragma once

#include "optimizer.h"


class Optimizer_Adam : public Optimizer
{
public:
    explicit Optimizer_Adam(double learning_rate = 0.001,
                            double decay = 0,
                            double epsilon = 1e-7,
                            double beta_1 = 0.9,
                            double beta_2 = 0.999)
        : Optimizer(learning_rate, decay, {}, epsilon, {}, beta_1, beta_2)
    {
    }

    void update_params(Layer_Dense &layer) const override
    {
        // Update momentum with current gradients
        layer.weight_momentums = this->beta_1 * layer.weight_momentums + (1 - this->beta_1) * layer.dweights;
        layer.bias_momentums = this->beta_1 * layer.bias_momentums + (1 - this->beta_1) * layer.dbiases;

        // Get corrected momentums
        // iteration is 0 at first pass and we need to start with 1 here
        Matrix weight_momentums_corrected =
            layer.weight_momentums / (1 - std::pow(this->beta_1, (this->iteration + 1)));
        Matrix bias_momentums_corrected = layer.bias_momentums / (1 - std::pow(this->beta_1, (this->iteration + 1)));

        // Update cache with squared current gradients
        layer.weight_cache = this->beta_2 * layer.weight_cache + (1 - this->beta_2) * layer.dweights.sq();
        layer.bias_cache = this->beta_2 * layer.bias_cache + (1 - this->beta_2) * layer.dbiases.sq();

        // Get corrected caches
        Matrix weight_cache_corrected = layer.weight_cache / (1 - std::pow(this->beta_2, (this->iteration + 1)));
        Matrix bias_cache_corrected = layer.bias_cache / (1 - std::pow(this->beta_2, (this->iteration + 1)));

        // Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -1 * this->current_learning_rate * weight_momentums_corrected /
                         (NumCpp::sqrt(weight_cache_corrected) + this->epsilon);
        layer.biases += -1 * this->current_learning_rate * bias_momentums_corrected /
                        (NumCpp::sqrt(bias_cache_corrected) + this->epsilon);
    }
};