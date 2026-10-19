#pragma once

#include "optimizer.h"
#include <omp.h>

class Optimizer_SGD : public Optimizer
{
public:
    explicit Optimizer_SGD(double learning_rate = 0.01, double decay = 0, double momentum = 0)
        : Optimizer(learning_rate, decay, momentum, {}, {}, {}, {})
    {
    }

    void update_params(Layer_Dense &layer) const override
    {
        update_params(layer, false);
    }

    void update_params(Layer_Dense &layer, bool include_regularization) const override
    {
// Fused SIMD loops prevent intermediate weight_updates allocations
#pragma omp parallel for simd if (layer.weights.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < layer.weights.size(); ++i)
        {
            const double gradient = weight_gradient(layer, i, layer.dweights[i], include_regularization);
            double update = this->momentum * layer.weight_momentums[i] - this->current_learning_rate * gradient;
            layer.weight_momentums[i] = update;
            layer.weights[i] += update;
        }

#pragma omp parallel for simd if (layer.biases.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < layer.biases.size(); ++i)
        {
            const double gradient = bias_gradient(layer, i, layer.dbiases[i], include_regularization);
            double update = this->momentum * layer.bias_momentums[i] - this->current_learning_rate * gradient;
            layer.bias_momentums[i] = update;
            layer.biases[i] += update;
        }
    }
};
