#pragma once

#include "optimizer.h"
#include <cmath>
#include <omp.h>

class Optimizer_Adam : public Optimizer
{
    double beta_1_correction;
    double beta_2_correction;

public:
    explicit Optimizer_Adam(double learning_rate = 0.001,
                            double decay = 0,
                            double epsilon = 1e-7,
                            double beta_1 = 0.9,
                            double beta_2 = 0.999)
        : Optimizer(learning_rate, decay, {}, epsilon, {}, beta_1, beta_2), beta_1_correction(1.0 - beta_1),
          beta_2_correction(1.0 - beta_2)
    {
    }

    void pre_update_params() override
    {
        Optimizer::pre_update_params();
        beta_1_correction = 1.0 - std::pow(this->beta_1, this->iteration + 1);
        beta_2_correction = 1.0 - std::pow(this->beta_2, this->iteration + 1);
    }

    void update_params(Layer_Dense &layer) const override
    {
        update_params(layer, false);
    }

    void update_params(Layer_Dense &layer, bool include_regularization) const override
    {
        double inv_beta_1 = 1.0 - this->beta_1;
        double inv_beta_2 = 1.0 - this->beta_2;

#pragma omp parallel for simd if (layer.weights.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < layer.weights.size(); ++i)
        {
            const double gradient = weight_gradient(layer, i, layer.dweights[i], include_regularization);
            layer.weight_momentums[i] = this->beta_1 * layer.weight_momentums[i] + inv_beta_1 * gradient;
            layer.weight_cache[i] =
                this->beta_2 * layer.weight_cache[i] + inv_beta_2 * (gradient * gradient);

            double m_corr = layer.weight_momentums[i] / beta_1_correction;
            double c_corr = layer.weight_cache[i] / beta_2_correction;

            layer.weights[i] -= (this->current_learning_rate * m_corr) / (std::sqrt(c_corr) + this->epsilon);
        }

#pragma omp parallel for simd if (layer.biases.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < layer.biases.size(); ++i)
        {
            const double gradient = bias_gradient(layer, i, layer.dbiases[i], include_regularization);
            layer.bias_momentums[i] = this->beta_1 * layer.bias_momentums[i] + inv_beta_1 * gradient;
            layer.bias_cache[i] =
                this->beta_2 * layer.bias_cache[i] + inv_beta_2 * (gradient * gradient);

            double m_corr = layer.bias_momentums[i] / beta_1_correction;
            double c_corr = layer.bias_cache[i] / beta_2_correction;

            layer.biases[i] -= (this->current_learning_rate * m_corr) / (std::sqrt(c_corr) + this->epsilon);
        }
    }
};
