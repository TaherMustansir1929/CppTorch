#pragma once

#include "cpptorch/core/matrix.h"
#include "cpptorch/core/numcpp.h"
#include <omp.h>

namespace cpptorch
{

/// Scale factor applied to the randomly initialised weights.
const double WEIGHT_SCALAR = 0.01;

/**
 * @brief A fully connected layer: `output = inputs * weights + biases`.
 *
 * The layer owns its weights, biases and gradient buffers. Gradients are
 * accumulated into @ref dweights and @ref dbiases by backward() and consumed
 * by the optimizer.
 */
class Layer_Dense
{
private:
    static constexpr size_t OMP_THRESHOLD = 8192;
    const Matrix *m_inputs_view = nullptr;
    const MatrixView *m_inputs_slice = nullptr;

public:
    /// Weights of shape (n_inputs, n_neurons).
    Matrix weights, biases, inputs, output;
    /// Gradients of the loss with respect to the weights, inputs and biases.
    Matrix dweights, dinputs, dbiases;
    /// Optimizer state: first and second moment estimates.
    Matrix weight_momentums, bias_momentums;
    /// Optimizer state: accumulated squared gradients.
    Matrix weight_cache, bias_cache;

    /// L1 penalty applied to the weights.
    double weight_regularizer_l1 = 0;
    /// L2 penalty applied to the weights.
    double weight_regularizer_l2 = 0;
    /// L1 penalty applied to the biases.
    double bias_regularizer_l1 = 0;
    /// L2 penalty applied to the biases.
    double bias_regularizer_l2 = 0;

private:
    friend class Layer_Wrapper;

    void forward_impl(const Matrix &inputs, bool cache_inputs)
    {
        if (cache_inputs)
        {
            this->inputs = inputs;
            m_inputs_view = nullptr;
            m_inputs_slice = nullptr;
        }
        else
        {
            m_inputs_view = &inputs;
            m_inputs_slice = nullptr;
        }
        NumCpp::dot_into(inputs, this->weights, this->output);
        this->output += this->biases;
    }

    void forward_impl(const MatrixView &inputs)
    {
        m_inputs_view = nullptr;
        m_inputs_slice = &inputs;
        NumCpp::dot_into(inputs, this->weights, this->output);
        this->output += this->biases;
    }

    void forward_borrowed(const Matrix &inputs)
    {
        forward_impl(inputs, false);
    }

    void forward_borrowed(const MatrixView &inputs)
    {
        forward_impl(inputs);
    }

    void cache_current_inputs()
    {
        if (m_inputs_view)
        {
            this->inputs = *m_inputs_view;
            m_inputs_view = nullptr;
        }
        else if (m_inputs_slice)
        {
            this->inputs.copy_from(*m_inputs_slice);
            m_inputs_slice = nullptr;
        }
    }

public:

    /**
     * @brief Constructs a dense layer with randomly initialised weights.
     *
     * @param n_inputs             Number of input features.
     * @param n_neurons            Number of neurons in the layer.
     * @param weight_regularizer_l1 L1 penalty applied to the weights.
     * @param weight_regularizer_l2 L2 penalty applied to the weights.
     * @param bias_regularizer_l1  L1 penalty applied to the biases.
     * @param bias_regularizer_l2  L2 penalty applied to the biases.
     */
    Layer_Dense(size_t n_inputs,
                size_t n_neurons,
                double weight_regularizer_l1 = 0,
                double weight_regularizer_l2 = 0,
                double bias_regularizer_l1 = 0,
                double bias_regularizer_l2 = 0)
        : weights(NumCpp::randn({n_inputs, n_neurons}) * WEIGHT_SCALAR), biases(NumCpp::zeroes({1, n_neurons})),
          output(Matrix(1, n_neurons)), weight_regularizer_l1(weight_regularizer_l1),
          weight_regularizer_l2(weight_regularizer_l2), bias_regularizer_l1(bias_regularizer_l1),
          bias_regularizer_l2(bias_regularizer_l2)
    {
        this->weight_momentums = NumCpp::zeroes({n_inputs, n_neurons});
        this->bias_momentums = NumCpp::zeroes({1, n_neurons});
        this->weight_cache = NumCpp::zeroes({n_inputs, n_neurons});
        this->bias_cache = NumCpp::zeroes({1, n_neurons});
    }

    /**
     * @brief Computes the forward pass and caches the inputs.
     * @param inputs Input matrix of shape (samples, n_inputs).
     */
    void forward(const Matrix &inputs)
    {
        forward_impl(inputs, true);
    }

    /**
     * @brief Computes the parameter and input gradients.
     *
     * @param dvalues              Gradient of the loss with respect to the output.
     * @param apply_regularization When true, the L1/L2 penalties are added to
     *                              the parameter gradients.
     */
    void backward(const Matrix &dvalues, bool apply_regularization = true)
    {
        if (m_inputs_slice)
            NumCpp::dot_transpose_left_into(*m_inputs_slice, dvalues, this->dweights);
        else
        {
            const Matrix &cached_inputs = m_inputs_view ? *m_inputs_view : this->inputs;
            NumCpp::dot_transpose_left_into(cached_inputs, dvalues, this->dweights);
        }
        NumCpp::sum_into(dvalues, 0, this->dbiases);

        // ==========================================
        // ===== GRADIENTS ON REGULARIZATION ========
        // ==========================================

        // Zero-allocation SIMD Penalty Integrations
        if (apply_regularization && this->weight_regularizer_l1 > 0)
        {
#pragma omp parallel for simd if (this->weights.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < this->weights.size(); ++i)
            {
                this->dweights[i] += this->weight_regularizer_l1 * ((this->weights[i] < 0.0) ? -1.0 : 1.0);
            }
        }

        if (apply_regularization && this->weight_regularizer_l2 > 0)
        {
#pragma omp parallel for simd if (this->weights.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < this->weights.size(); ++i)
            {
                this->dweights[i] += 2.0 * this->weight_regularizer_l2 * this->weights[i];
            }
        }

        if (apply_regularization && this->bias_regularizer_l1 > 0)
        {
#pragma omp parallel for simd if (this->biases.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < this->biases.size(); ++i)
            {
                this->dbiases[i] += this->bias_regularizer_l1 * ((this->biases[i] < 0.0) ? -1.0 : 1.0);
            }
        }

        if (apply_regularization && this->bias_regularizer_l2 > 0)
        {
#pragma omp parallel for simd if (this->biases.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < this->biases.size(); ++i)
            {
                this->dbiases[i] += 2.0 * this->bias_regularizer_l2 * this->biases[i];
            }
        }

        // Gradient on values
        NumCpp::dot_transpose_right_into(dvalues, this->weights, this->dinputs);
    }
};

} // namespace cpptorch
