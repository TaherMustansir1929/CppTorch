#pragma once

#include "cpptorch/layers/dense.h"

#include <cstdint>
#include <memory>

namespace cpptorch
{

/**
 * @brief Identifies the optimization algorithm used to update the parameters.
 *
 * The enum is the single entry point of the optimizer plug-in system: add a
 * value here and a matching case in @ref cpptorch::make_optimizer to expose a
 * new optimizer to @ref cpptorch::Model.
 */
enum class OptimizerType : std::uint8_t
{
    SGD = 0,    ///< Stochastic gradient descent, with optional momentum.
    Adagrad = 1, ///< Adagrad, per-parameter squared-gradient accumulation.
    RMSprop = 2, ///< RMSprop, per-parameter exponentially averaged squared gradient.
    Adam = 3    ///< Adam, momentum plus adaptive learning rates.
};

/**
 * @brief Hyper-parameters accepted by every optimizer.
 */
struct HyperParameters_Optimizer
{
    /// The optimization algorithm to use.
    OptimizerType optmzr_type = OptimizerType::SGD;
    /// Base learning rate.
    double learning_rate = 0.01;
    /// Learning-rate decay applied once per iteration.
    double decay = 0.0;
    /// Momentum factor (used by SGD).
    double momentum = 0.0;
    /// Numerical stability constant.
    double epsilon = 1e-7;
    /// Decay factor of the squared-gradient cache (RMSprop).
    double rho = 0.9;
    /// Exponential decay rate of the first moment (Adam).
    double beta_1 = 0.9;
    /// Exponential decay rate of the second moment (Adam).
    double beta_2 = 0.999;
};

/**
 * @brief Base class for optimizers.
 *
 * An optimizer owns the learning-rate schedule and knows how to apply a
 * gradient step to a @ref cpptorch::Layer_Dense.
 *
 * @note To add a new optimizer, derive from this class, implement
 *       update_params(), then register it in @ref cpptorch::make_optimizer.
 */
class Optimizer
{
protected:
    /// Base learning rate.
    double learning_rate = 0;
    /// Learning rate after the decay schedule has been applied.
    double current_learning_rate = 0;
    /// Learning-rate decay applied once per iteration.
    double decay = 0;
    /// Momentum factor.
    double momentum = 0;
    /// Numerical stability constant.
    double epsilon = 1e-7;
    /// Decay factor of the squared-gradient cache.
    double rho = 0.9;
    /// Exponential decay rate of the first moment.
    double beta_1 = 0.9;
    /// Exponential decay rate of the second moment.
    double beta_2 = 0.999;
    /// Number of parameter updates performed so far.
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

    /**
     * @brief Constructs an optimizer.
     */
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

    /**
     * @brief Returns the learning rate that will be used for the next update.
     */
    [[nodiscard]] double get_current_learning_rate() const
    {
        return current_learning_rate;
    }

    /**
     * @brief Returns the weight gradient with the regularization penalty applied.
     *
     * @param layer                    Layer that owns the weight.
     * @param index                    Index of the weight element.
     * @param gradient                 Raw gradient of the loss.
     * @param include_regularization   When false, the raw gradient is returned unchanged.
     */
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

    /**
     * @brief Returns the bias gradient with the regularization penalty applied.
     *
     * @param layer                    Layer that owns the bias.
     * @param index                    Index of the bias element.
     * @param gradient                 Raw gradient of the loss.
     * @param include_regularization   When false, the raw gradient is returned unchanged.
     */
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

    /**
     * @brief Applies the learning-rate decay before a parameter update.
     *
     * Called once per iteration, before the layers are updated.
     */
    virtual void pre_update_params()
    {
        this->current_learning_rate = this->learning_rate * (1.0 / (1.0 + (this->decay * this->iteration)));
    }

    /**
     * @brief Advances the iteration counter after a parameter update.
     */
    void post_update_params()
    {
        this->iteration++;
    }

    /**
     * @brief Applies a gradient step to @p layer.
     */
    virtual void update_params(Layer_Dense &layer) const = 0;

    /**
     * @brief Applies a gradient step to @p layer, optionally folding the
     *        regularization penalty into the gradient.
     */
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

} // namespace cpptorch
