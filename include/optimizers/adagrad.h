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
#pragma omp parallel for simd if (layer.weights.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < layer.weights.size(); ++i)
        {
            layer.weight_cache[i] += (layer.dweights[i] * layer.dweights[i]);
            layer.weights[i] -=
                (this->current_learning_rate * layer.dweights[i]) / (std::sqrt(layer.weight_cache[i]) + this->epsilon);
        }

#pragma omp parallel for simd if (layer.biases.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < layer.biases.size(); ++i)
        {
            layer.bias_cache[i] += (layer.dbiases[i] * layer.dbiases[i]);
            layer.biases[i] -=
                (this->current_learning_rate * layer.dbiases[i]) / (std::sqrt(layer.bias_cache[i]) + this->epsilon);
        }
    }
};