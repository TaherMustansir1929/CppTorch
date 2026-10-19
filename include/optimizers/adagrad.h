#pragma once

#include "optimizer.h"
#include <cmath>
#include <omp.h>

class Optimizer_Adagrad : public Optimizer
{
public:
    explicit Optimizer_Adagrad(double learning_rate = 0.01, double decay = 0, double epsilon = 1e-7)
        : Optimizer(learning_rate, decay, {}, epsilon, {}, {}, {})
    {
    }

    void update_params(Layer_Dense &layer) const override
    {
        update_params(layer, false);
    }

    void update_params(Layer_Dense &layer, bool include_regularization) const override
    {
#pragma omp parallel for simd if (layer.weights.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < layer.weights.size(); ++i)
        {
            const double gradient = weight_gradient(layer, i, layer.dweights[i], include_regularization);
            layer.weight_cache[i] += gradient * gradient;
            layer.weights[i] -=
                (this->current_learning_rate * gradient) / (std::sqrt(layer.weight_cache[i]) + this->epsilon);
        }

#pragma omp parallel for simd if (layer.biases.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < layer.biases.size(); ++i)
        {
            const double gradient = bias_gradient(layer, i, layer.dbiases[i], include_regularization);
            layer.bias_cache[i] += gradient * gradient;
            layer.biases[i] -=
                (this->current_learning_rate * gradient) / (std::sqrt(layer.bias_cache[i]) + this->epsilon);
        }
    }
};
