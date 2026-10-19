#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/matrix.h"
#include "core/numcpp.h"
#include "layers/activations/activation.h"
#include "layers/activations/linear.h"
#include "layers/activations/relu.h"
#include "layers/activations/softmax.h"
#include "layers/dense.h"
#include "layers/dropout.h"
#include "losses/categorical_crossentropy.h"
#include "losses/loss.h"
#include "losses/mean_squared_error.h"
#include "optimizers/adagrad.h"
#include "optimizers/adam.h"
#include "optimizers/optimizer.h"
#include "optimizers/rmsprop.h"
#include "optimizers/sgd.h"
#include <omp.h>

enum class ActivationType : std::uint8_t
{
    Linear,
    ReLU,
    Softmax,
};

struct HyperParameters_Layer_Wrapper
{
    size_t n_inputs = 0;
    size_t n_neurons = 0;
    double weight_regularizer_l1 = 0.0;
    double weight_regularizer_l2 = 0.0;
    double bias_regularizer_l1 = 0.0;
    double bias_regularizer_l2 = 0.0;

    ActivationType activation_type = ActivationType::Linear;

    double dropout_rate = 0.0;
};

enum class OptimizerType : std::uint8_t
{
    SGD,
    Adagrad,
    RMSprop,
    Adam
};

struct HyperParameters_Optimizer
{
    OptimizerType optmzr_type = OptimizerType::SGD;

    double learning_rate = 0.01;
    double decay = 0.0;
    double momentum = 0.0;
    double epsilon = 1e-7;
    double rho = 0.9;
    double beta_1 = 0.9;
    double beta_2 = 0.999;
};

class Layer_Wrapper
{
    Layer_Dense m_dense;
    Layer_Dropout m_dropout;
    std::unique_ptr<Activation> m_activation;
    ActivationType m_activation_type = ActivationType::Linear;
    bool m_has_dropout = false;

public:
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
        switch (params.activation_type)
        {
        case ActivationType::Linear:
            m_activation = std::make_unique<Activation_Linear>();
            break;
        case ActivationType::ReLU:
            m_activation = std::make_unique<Activation_ReLU>();
            break;
        case ActivationType::Softmax:
            m_activation = std::make_unique<Activation_Softmax>();
            break;
        }
    }

    Layer_Wrapper(Layer_Wrapper &&) noexcept = default;
    Layer_Wrapper &operator=(Layer_Wrapper &&) noexcept = default;
    Layer_Wrapper(const Layer_Wrapper &) = delete;
    Layer_Wrapper &operator=(const Layer_Wrapper &) = delete;

    [[nodiscard]] const Matrix &output(bool is_training = true) const
    {
        if (m_has_dropout && is_training)
            return m_dropout.output;
        if (m_activation_type == ActivationType::Linear)
            return m_dense.output;
        return m_activation->output;
    }

    [[nodiscard]] const Matrix &dinputs() const
    {
        return m_dense.dinputs;
    }
    [[nodiscard]] Layer_Dense &dense()
    {
        return m_dense;
    }
    [[nodiscard]] const Layer_Dense &dense() const
    {
        return m_dense;
    }
    [[nodiscard]] ActivationType activation_type() const
    {
        return m_activation_type;
    }
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

enum class ModelType : std::uint8_t
{
    Classification,
    Regression
};

class Model
{
    ModelType m_model_type = ModelType::Regression;
    std::unique_ptr<Loss> m_loss;
    std::vector<Layer_Wrapper> m_seq_layers;
    std::unique_ptr<Optimizer> m_optimizer;

    [[nodiscard]] static std::unique_ptr<Loss> create_loss_for_type(ModelType model_type)
    {
        switch (model_type)
        {
        case ModelType::Classification:
            return std::make_unique<Loss_CategoricalCrossEntropy>();
        case ModelType::Regression:
            return std::make_unique<Loss_MeanSquaredError>();
        }
        throw std::invalid_argument("Unknown ModelType in create_loss_for_type");
    }

    [[nodiscard]] Matrix format_target(const Matrix &y) const
    {
        if (m_model_type == ModelType::Classification && y.cols() == 1 &&
            dynamic_cast<const Loss_CategoricalCrossEntropy *>(m_loss.get()) == nullptr)
        {
            size_t n_classes = m_seq_layers.back().dense().weights.cols();
            return NumCpp::one_hot(y, static_cast<int>(n_classes));
        }
        return y;
    }

public:
    explicit Model(ModelType model_type = ModelType::Regression)
        : m_model_type(model_type), m_loss(create_loss_for_type(model_type)),
          m_optimizer(std::make_unique<Optimizer_SGD>())
    {
    }

    void set_model_type(ModelType model_type)
    {
        m_model_type = model_type;
        m_loss = create_loss_for_type(model_type);
    }

    [[nodiscard]] ModelType model_type() const
    {
        return m_model_type;
    }

    void set_loss(std::unique_ptr<Loss> loss)
    {
        if (!loss)
            throw std::invalid_argument("Cannot assign a null loss function.");
        m_loss = std::move(loss);
    }

    [[nodiscard]] Loss &loss()
    {
        return *m_loss;
    }
    void add_layer(HyperParameters_Layer_Wrapper params)
    {
        m_seq_layers.emplace_back(params);
    }

    void set_optimizer(HyperParameters_Optimizer params)
    {
        switch (params.optmzr_type)
        {
        case OptimizerType::SGD:
            m_optimizer = std::make_unique<Optimizer_SGD>(params.learning_rate, params.decay, params.momentum);
            break;
        case OptimizerType::Adagrad:
            m_optimizer = std::make_unique<Optimizer_Adagrad>(params.learning_rate, params.decay, params.epsilon);
            break;
        case OptimizerType::RMSprop:
            m_optimizer =
                std::make_unique<Optimizer_RMSprop>(params.learning_rate, params.decay, params.epsilon, params.rho);
            break;
        case OptimizerType::Adam:
            m_optimizer = std::make_unique<Optimizer_Adam>(params.learning_rate,
                                                           params.decay,
                                                           params.epsilon,
                                                           params.beta_1,
                                                           params.beta_2);
            break;
        }
    }

    [[nodiscard]] Matrix predict(const Matrix &X)
    {
        if (m_seq_layers.empty())
            throw std::runtime_error("Model has no layers added.");
        if (X.cols() != m_seq_layers[0].dense().weights.rows())
        {
            throw std::invalid_argument("Input feature dimension (" + std::to_string(X.cols()) +
                                        ") does not match model input dimension (" +
                                        std::to_string(m_seq_layers[0].dense().weights.rows()) + ").");
        }

        m_seq_layers[0].forward(X, false);
        for (size_t i = 1; i < m_seq_layers.size(); ++i)
        {
            m_seq_layers[i].forward(m_seq_layers[i - 1].output(false), false);
        }
        return m_seq_layers.back().output(false);
    }

    [[nodiscard]] Matrix predict_classes(const Matrix &X)
    {
        Matrix output = predict(X);
        if (m_model_type == ModelType::Classification)
            return NumCpp::argmax(output, 1);
        return output;
    }

    struct EvaluationResult
    {
        double loss = 0.0;
        double accuracy = 0.0;
        double rmse = 0.0;

        void display(ModelType type) const
        {
            if (type == ModelType::Classification)
            {
                printf("Evaluation -> Loss: %.4f | Accuracy: %.3f (%.2f%%)\n", loss, accuracy, accuracy * 100.0);
            }
            else
            {
                printf("Evaluation -> Loss (MSE): %.4f | RMSE: %.4f\n", loss, rmse);
            }
        }
    };

    [[nodiscard]] EvaluationResult evaluate(const Matrix &X, const Matrix &y, bool verbose = true)
    {
        if (m_seq_layers.empty())
            throw std::runtime_error("Model has no layers added.");
        if (!m_loss)
            throw std::runtime_error("Loss function has not been initialized.");

        Matrix y_pred = predict(X);
        Matrix y_target = format_target(y);

        double eval_loss = m_loss->calculate(y_pred, y_target);
        double eval_acc = (m_model_type == ModelType::Classification) ? Loss::accuracy(y_pred, y_target) : 0.0;
        double eval_rmse = (m_model_type == ModelType::Regression) ? std::sqrt(eval_loss) : 0.0;

        EvaluationResult result{eval_loss, eval_acc, eval_rmse};
        if (verbose)
            result.display(m_model_type);

        return result;
    }

    static void displayEpoch(size_t n_epochs,
                             size_t epoch,
                             double train_loss,
                             double val_loss,
                             double train_acc,
                             double val_acc,
                             double lr,
                             bool has_val,
                             ModelType model_type)
    {
        printf("Epoch %3zu/%zu - ", epoch, n_epochs);
        if (model_type == ModelType::Classification)
        {
            printf("Loss: %.4f - Acc: %.3f", train_loss, train_acc);
            if (has_val)
                printf(" - Val Loss: %.4f - Val Acc: %.3f", val_loss, val_acc);
        }
        else
        {
            printf("Loss (MSE): %.4f", train_loss);
            if (has_val)
                printf(" - Val Loss (MSE): %.4f", val_loss);
        }
        printf(" - lr: %.6f\n", lr);
    }

    void train(const Matrix &X_train,
               const Matrix &y_train,
               size_t n_epochs,
               const Matrix &X_val = Matrix(),
               const Matrix &y_val = Matrix(),
               size_t print_every = 1,
               size_t batch_size = 0,
               size_t validation_every = 1)
    {
        if (m_seq_layers.empty())
            throw std::runtime_error("No layers have been added to the model.");
        if (!m_loss)
            throw std::runtime_error("Loss function has not been initialized.");
        if (!m_optimizer)
            throw std::runtime_error("Optimizer has not been initialized.");
        if (X_train.rows() == 0 || X_train.rows() != y_train.rows())
            throw std::invalid_argument("Training inputs and targets must have the same non-zero row count.");
        if (print_every == 0 || validation_every == 0)
            throw std::invalid_argument("Metric intervals must be greater than zero.");

        Matrix y_train_target = format_target(y_train);
        bool has_val = (X_val.rows() > 0 && y_val.rows() > 0);
        Matrix y_val_target = has_val ? format_target(y_val) : Matrix();

        bool use_fused_softmax_cce = (m_model_type == ModelType::Classification &&
                                      m_seq_layers.back().activation_type() == ActivationType::Softmax);

        size_t final_layer_idx = m_seq_layers.size() - 1;
        Matrix fused_grad;
        MatrixView X_batch_view;
        Matrix y_batch;
        const size_t effective_batch_size = batch_size == 0 ? X_train.rows() : std::min(batch_size, X_train.rows());
        const bool use_batches = effective_batch_size < X_train.rows();
        double val_loss = 0.0;
        double val_accuracy = 0.0;

        for (size_t epoch = 1; epoch <= n_epochs; ++epoch)
        {
            const bool should_report = epoch % print_every == 0 || epoch == 1 || epoch == n_epochs;
            double weighted_train_loss = 0.0;
            double weighted_train_accuracy = 0.0;

            for (size_t batch_begin = 0; batch_begin < X_train.rows(); batch_begin += effective_batch_size)
            {
                const size_t batch_rows = std::min(effective_batch_size, X_train.rows() - batch_begin);
                const Matrix *batch_targets = &y_train_target;
                if (use_batches)
                {
                    X_batch_view = X_train.rows_view(batch_begin, batch_rows);
                    y_batch.copy_rows_from(y_train_target, batch_begin, batch_rows);
                    batch_targets = &y_batch;
                }

                // 1. FORWARD PASS
                if (use_batches)
                    m_seq_layers[0].forward_borrowed(X_batch_view, true);
                else
                    m_seq_layers[0].forward_borrowed(X_train, true);
                for (size_t i = 1; i <= final_layer_idx; ++i)
                    m_seq_layers[i].forward_borrowed(m_seq_layers[i - 1].output(true), true);
                const Matrix &train_output = m_seq_layers.back().output(true);

                if (should_report)
                {
                    const double batch_loss = m_loss->calculate(train_output, *batch_targets);
                    const double batch_accuracy = m_model_type == ModelType::Classification
                                                      ? Loss::accuracy(train_output, *batch_targets)
                                                      : 0.0;
                    weighted_train_loss += batch_loss * static_cast<double>(batch_rows);
                    weighted_train_accuracy += batch_accuracy * static_cast<double>(batch_rows);

                }

                // 2. BACKWARD PASS
                if (use_fused_softmax_cce)
                {
                    const size_t samples = train_output.rows();
                    const size_t cols = train_output.cols();
                    const double inv_samples = 1.0 / static_cast<double>(samples);
                    fused_grad.resize(samples, cols);

// SIMD mathematical mapping directly scales the fused gradient without intermediate matrices
#pragma omp parallel for if (train_output.size() > 8192)
                    for (size_t s = 0; s < samples; ++s)
                    {
                        const size_t row_offset = s * cols;
                        size_t target_idx = static_cast<size_t>((*batch_targets)[s]);
                        if (batch_targets->cols() > 1)
                        {
                            target_idx = 0;
                            const size_t target_offset = s * batch_targets->cols();
                            for (size_t j = 1; j < batch_targets->cols(); ++j)
                            {
                                if ((*batch_targets)[target_offset + j] > (*batch_targets)[target_offset + target_idx])
                                    target_idx = j;
                            }
                        }

#pragma omp simd
                        for (size_t j = 0; j < cols; ++j)
                        {
                            double val = train_output[row_offset + j];
                            if (j == target_idx)
                                val -= 1.0;
                            fused_grad[row_offset + j] = val * inv_samples;
                        }
                    }

                    m_seq_layers.back().dense().backward(fused_grad, false);

                    const Matrix *upstream_grad = &m_seq_layers.back().dinputs();
                    for (int i = static_cast<int>(final_layer_idx) - 1; i >= 0; --i)
                    {
                        m_seq_layers[i].backward(*upstream_grad, false);
                        upstream_grad = &m_seq_layers[i].dinputs();
                    }
                }
                else
                {
                    m_loss->backward(train_output, *batch_targets);
                    const Matrix *upstream_grad = &m_loss->dinputs;

                    for (int i = static_cast<int>(final_layer_idx); i >= 0; --i)
                    {
                        m_seq_layers[i].backward(*upstream_grad, false);
                        upstream_grad = &m_seq_layers[i].dinputs();
                    }
                }

                // 3. PARAMETER UPDATE
                m_optimizer->pre_update_params();
                for (auto &layer : m_seq_layers)
                    m_optimizer->update_params(layer.dense(), true);
                m_optimizer->post_update_params();
            }

            if (should_report)
            {
                double reg_loss = 0.0;
                for (const auto &layer : m_seq_layers)
                    reg_loss += Loss::regularization_loss(layer.dense());
                weighted_train_loss += reg_loss * static_cast<double>(X_train.rows());
            }

            const bool evaluate_validation = has_val &&
                                             (epoch % validation_every == 0 || epoch == 1 || epoch == n_epochs);
            if (evaluate_validation)
            {
                EvaluationResult val_res = evaluate(X_val, y_val, false);
                val_loss = val_res.loss;
                val_accuracy = val_res.accuracy;
            }

            // 4. PROGRESS REPORTING
            if (should_report)
            {
                const double total_train_loss = weighted_train_loss / static_cast<double>(X_train.rows());
                const double train_accuracy = weighted_train_accuracy / static_cast<double>(X_train.rows());
                displayEpoch(n_epochs,
                             epoch,
                             total_train_loss,
                             val_loss,
                             train_accuracy,
                             val_accuracy,
                             m_optimizer->get_current_learning_rate(),
                             evaluate_validation,
                             m_model_type);
            }
        }

        for (auto &layer : m_seq_layers)
            layer.cache_current_inputs();
    }
};
