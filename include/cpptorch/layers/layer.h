#pragma once

#include "cpptorch/layers/activations/activation_registry.h"
#include "cpptorch/layers/dense.h"
#include "cpptorch/layers/dropout.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace cpptorch
{

/**
 * @brief Hyper-parameters of a single layer of a @ref cpptorch::Model.
 */
struct HyperParameters_Layer_Wrapper
{
    /// Number of input features.
    size_t n_inputs = 0;
    /// Number of neurons in the layer.
    size_t n_neurons = 0;
    /// L1 penalty applied to the weights.
    double weight_regularizer_l1 = 0.0;
    /// L2 penalty applied to the weights.
    double weight_regularizer_l2 = 0.0;
    /// L1 penalty applied to the biases.
    double bias_regularizer_l1 = 0.0;
    /// L2 penalty applied to the biases.
    double bias_regularizer_l2 = 0.0;

    /// Activation function applied after the dense projection.
    ActivationType activation_type = ActivationType::Linear;

    /// Dropout rate, 0.0 disables dropout.
    double dropout_rate = 0.0;
};

/**
 * @brief Bundles a dense layer, an optional dropout stage and an activation.
 *
 * A @ref cpptorch::Model owns a sequence of layer wrappers. The wrapper owns
 * the parameters, the dropout mask and the activation, and exposes a single
 * forward/backward pair to the training loop.
 */
class Layer_Wrapper
{
    Layer_Dense m_dense;
    Layer_Dropout m_dropout;
    std::unique_ptr<Activation> m_activation;
    ActivationType m_activation_type = ActivationType::Linear;
    bool m_has_dropout = false;

public:
    /**
     * @brief Constructs a layer.
     *
     * @param params Layer hyper-parameters.
     */
    explicit Layer_Wrapper(HyperParameters_Layer_Wrapper params)
        : m_dense(Layer_Dense(params.n_inputs,
                              params.n_neurons,
                              params.weight_regularizer_l1,
                              params.weight_regularizer_l2,
                              params.bias_regularizer_l1,
                              params.bias_regularizer_l2)),
          m_dropout(Layer_Dropout(params.dropout_rate)), m_activation_type(params.activation_type),
          m_has_dropout(params.dropout_rate > 0.0)
    {
        m_activation = make_activation(m_activation_type);
    }

    Layer_Wrapper(Layer_Wrapper &&) noexcept = default;
    Layer_Wrapper &operator=(Layer_Wrapper &&) noexcept = default;
    Layer_Wrapper(const Layer_Wrapper &) = delete;
    Layer_Wrapper &operator=(const Layer_Wrapper &) = delete;

    /**
     * @brief Returns the output of the layer.
     *
     * @param is_training When false, dropout is bypassed.
     */
    [[nodiscard]] const Matrix &output(bool is_training = true) const
    {
        if (m_has_dropout && is_training)
            return m_dropout.output;
        if (m_activation_type == ActivationType::Linear)
            return m_dense.output;
        return m_activation->output;
    }

    /**
     * @brief Returns the gradient of the loss with respect to the layer input.
     */
    [[nodiscard]] const Matrix &dinputs() const
    {
        return m_dense.dinputs;
    }

    /**
     * @brief Returns the dense layer owned by this wrapper.
     */
    [[nodiscard]] Layer_Dense &dense()
    {
        return m_dense;
    }

    /**
     * @brief Returns the dense layer owned by this wrapper.
     */
    [[nodiscard]] const Layer_Dense &dense() const
    {
        return m_dense;
    }

    /**
     * @brief Returns the activation applied by this layer.
     */
    [[nodiscard]] ActivationType activation_type() const
    {
        return m_activation_type;
    }

    /**
     * @brief Returns true when the layer applies dropout.
     */
    [[nodiscard]] bool has_dropout() const
    {
        return m_has_dropout;
    }

private:
    void forward_impl(const Matrix &inputs, bool is_training, bool cache_inputs)
    {
        if (cache_inputs)
            m_dense.forward(inputs);
        else
            m_dense.forward_borrowed(inputs);
        if (m_activation_type != ActivationType::Linear)
            m_activation->forward(m_dense.output, false);
        if (m_has_dropout && is_training)
            m_dropout.forward(m_activation_type == ActivationType::Linear ? m_dense.output : m_activation->output,
                              false);
    }

public:
    /**
     * @brief Runs the layer forward.
     *
     * @param inputs      Input matrix of shape (samples, n_inputs).
     * @param is_training When false, dropout is bypassed.
     */
    void forward(const Matrix &inputs, bool is_training = true)
    {
        forward_impl(inputs, is_training, true);
    }

private:
    friend class Model;

    void forward_borrowed(const Matrix &inputs, bool is_training = true)
    {
        forward_impl(inputs, is_training, false);
    }

    void forward_borrowed(const MatrixView &inputs, bool is_training = true)
    {
        m_dense.forward_borrowed(inputs);
        if (m_activation_type != ActivationType::Linear)
            m_activation->forward(m_dense.output, false);
        if (m_has_dropout && is_training)
            m_dropout.forward(m_activation_type == ActivationType::Linear ? m_dense.output : m_activation->output,
                              false);
    }

    void cache_current_inputs()
    {
        m_dense.cache_current_inputs();
    }

public:
    /**
     * @brief Computes the gradient of the loss with respect to the layer input.
     *
     * @param dvalues              Gradient of the loss with respect to the output.
     * @param apply_regularization When true, the regularization penalty is added
     *                              to the parameter gradients.
     */
    void backward(const Matrix &dvalues, bool apply_regularization = true)
    {
        if (m_has_dropout)
        {
            m_dropout.backward(dvalues);
            if (m_activation_type == ActivationType::Linear)
                m_dense.backward(m_dropout.dinputs, apply_regularization);
            else
            {
                m_activation->backward(m_dropout.dinputs);
                m_dense.backward(m_activation->dinputs, apply_regularization);
            }
        }
        else if (m_activation_type == ActivationType::Linear)
            m_dense.backward(dvalues, apply_regularization);
        else
        {
            m_activation->backward(dvalues);
            m_dense.backward(m_activation->dinputs, apply_regularization);
        }
    }
};

} // namespace cpptorch
