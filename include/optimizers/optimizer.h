#pragma once

#include "layers/dense.h"

class Optimizer
{
protected:
    double learning_rate = 0;
    double current_learning_rate = 0;
    double decay = 0;
    double momentum = 0;
    double epsilon = 1e-7;
    double rho = 0.9;
    double beta_1 = 0.9;
    double beta_2 = 0.999;
    int iteration = 0;

    // Threshold to prevent thread-spawn overhead on small layers
    static constexpr size_t OMP_THRESHOLD = 8192;

public:
    virtual ~Optimizer() = default;

    // Enable zero-allocation relocations via explicit noexcept defaults
    Optimizer(const Optimizer &) = default;
    Optimizer(Optimizer &&) noexcept = default;
    Optimizer &operator=(const Optimizer &) = default;
    Optimizer &operator=(Optimizer &&) noexcept = default;

    explicit Optimizer(double learning_rate = 0.01,
                       double decay = 0,
                       double momentum = 0,
                       double epsilon = 1e-7,
                       double rho = 0.9,
                       double beta_1 = 0.9,
                       double beta_2 = 0.999)
        : learning_rate(learning_rate), current_learning_rate(learning_rate), decay(decay), momentum(momentum),
          epsilon(epsilon), rho(rho), beta_1(beta_1), beta_2(beta_2)
    {
    }

    [[nodiscard]] double get_current_learning_rate() const
    {
        return current_learning_rate;
    }

    static double weight_gradient(const Layer_Dense &layer, size_t index, double gradient, bool include_regularization)
    {
        if (!include_regularization)
            return gradient;
        if (layer.weight_regularizer_l1 > 0.0)
            gradient += layer.weight_regularizer_l1 * ((layer.weights[index] < 0.0) ? -1.0 : 1.0);
        if (layer.weight_regularizer_l2 > 0.0)
            gradient += 2.0 * layer.weight_regularizer_l2 * layer.weights[index];
        return gradient;
    }

    static double bias_gradient(const Layer_Dense &layer, size_t index, double gradient, bool include_regularization)
    {
        if (!include_regularization)
            return gradient;
        if (layer.bias_regularizer_l1 > 0.0)
            gradient += layer.bias_regularizer_l1 * ((layer.biases[index] < 0.0) ? -1.0 : 1.0);
        if (layer.bias_regularizer_l2 > 0.0)
            gradient += 2.0 * layer.bias_regularizer_l2 * layer.biases[index];
        return gradient;
    }

    virtual void pre_update_params()
    {
        this->current_learning_rate = this->learning_rate * (1.0 / (1.0 + (this->decay * this->iteration)));
    }

    void post_update_params()
    {
        this->iteration++;
    }

    virtual void update_params(Layer_Dense &layer) const = 0;

    virtual void update_params(Layer_Dense &layer, bool include_regularization) const
    {
        if (include_regularization)
        {
#pragma omp parallel for simd if (layer.weights.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < layer.weights.size(); ++i)
                layer.dweights[i] = weight_gradient(layer, i, layer.dweights[i], true);
#pragma omp parallel for simd if (layer.biases.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < layer.biases.size(); ++i)
                layer.dbiases[i] = bias_gradient(layer, i, layer.dbiases[i], true);
        }
        update_params(layer);
    }
};
