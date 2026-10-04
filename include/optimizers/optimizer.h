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

public:
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer &) = default;
    Optimizer(Optimizer &&) = delete;
    Optimizer &operator=(const Optimizer &) = default;
    Optimizer &operator=(Optimizer &&) = delete;

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

    void pre_update_params()
    {
        this->current_learning_rate = this->learning_rate * (1 / (1 + (this->decay * this->iteration)));
    }

    void post_update_params()
    {
        this->iteration++;
    }

    virtual void update_params(Layer_Dense &layer) const = 0;
};