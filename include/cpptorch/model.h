#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cpptorch/core/matrix.h"
#include "cpptorch/core/numcpp.h"
#include "cpptorch/layers/layer.h"
#include "cpptorch/losses/loss_registry.h"
#include "cpptorch/optimizers/optimizer_registry.h"
#include <omp.h>

namespace cpptorch
{

/**
 * @brief Selects the default loss function and the metrics reported by a model.
 */
enum class ModelType : std::uint8_t
{
    Classification, ///< Cross-entropy loss and accuracy metrics.
    Regression      ///< Mean squared error loss and RMSE metrics.
};

/**
 * @brief A stack of layers trained with an optimizer.
 *
 * The model owns the layer sequence, the loss function and the optimizer, and
 * implements the forward, backward and parameter-update passes. It is the
 * entry point of the library for most users.
 *
 * @code
 * cpptorch::Model model(cpptorch::ModelType::Classification);
 * model.add_layer({.n_inputs = 2, .n_neurons = 64, .activation_type = cpptorch::ActivationType::ReLU});
 * model.add_layer({.n_inputs = 64, .n_neurons = 3, .activation_type = cpptorch::ActivationType::Softmax});
 * model.set_optimizer({.optmzr_type = cpptorch::OptimizerType::Adam, .learning_rate = 0.01});
 * model.train(X, y, 50);
 * @endcode
 */
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
            return make_loss(LossType::CategoricalCrossEntropy);
        case ModelType::Regression:
            return make_loss(LossType::MeanSquaredError);
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
    /**
     * @brief Constructs a model.
     *
     * @param model_type Selects the default loss function: categorical
     *                   cross-entropy for classification, mean squared error
     *                   for regression.
     */
    explicit Model(ModelType model_type = ModelType::Regression)
        : m_model_type(model_type), m_loss(create_loss_for_type(model_type)),
          m_optimizer(std::make_unique<Optimizer_SGD>())
    {
    }

    /**
     * @brief Switches the model type and resets the loss function accordingly.
     */
    void set_model_type(ModelType model_type)
    {
        m_model_type = model_type;
        m_loss = create_loss_for_type(model_type);
    }

    /**
     * @brief Returns the model type.
     */
    [[nodiscard]] ModelType model_type() const
    {
        return m_model_type;
    }

    /**
     * @brief Replaces the loss function.
     *
     * @param loss The new loss; must not be null.
     * @throws std::invalid_argument if @p loss is null.
     */
    void set_loss(std::unique_ptr<Loss> loss)
    {
        if (!loss)
            throw std::invalid_argument("Cannot assign a null loss function.");
        m_loss = std::move(loss);
    }

    /**
     * @brief Returns the loss function owned by the model.
     */
    [[nodiscard]] Loss &loss()
    {
        return *m_loss;
    }
    /**
     * @brief Appends a layer to the model.
     *
     * Layers are evaluated in the order they are added, so the number of
     * inputs of a layer must match the number of neurons of the previous one.
     */
    void add_layer(HyperParameters_Layer_Wrapper params)
    {
        m_seq_layers.emplace_back(params);
    }

    /**
     * @brief Replaces the optimizer used during training.
     */
    void set_optimizer(HyperParameters_Optimizer params)
    {
        m_optimizer = make_optimizer(params.optmzr_type, params);
    }

    /**
     * @brief Runs a forward pass without updating any parameter.
     *
     * @param X Input matrix of shape (samples, n_inputs).
     * @return The output of the last layer, of shape (samples, n_neurons).
     * @throws std::runtime_error if the model has no layers.
     * @throws std::invalid_argument if the feature count of @p X does not match
     *         the input dimension of the first layer.
     */
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

    /**
     * @brief Predicts the class index of every sample.
     *
     * @param X Input matrix of shape (samples, n_inputs).
     * @return For classification models, a column vector of class indices; for
     *         regression models, the raw model output.
     */
    [[nodiscard]] Matrix predict_classes(const Matrix &X)
    {
        Matrix output = predict(X);
        if (m_model_type == ModelType::Classification)
            return NumCpp::argmax(output, 1);
        return output;
    }

    /**
     * @brief Metrics produced by @ref cpptorch::Model::evaluate.
     */
    struct EvaluationResult
    {
        /// Loss value; the mean squared error for regression models.
        double loss = 0.0;
        /// Fraction of correctly classified samples; zero for regression models.
        double accuracy = 0.0;
        /// Root mean squared error; zero for classification models.
        double rmse = 0.0;

        /**
         * @brief Prints the metrics to stdout.
         *
         * @param type The model type, used to select the metrics to display.
         */
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

    /**
     * @brief Scores the model on a dataset.
     *
     * @param X       Input matrix of shape (samples, n_inputs).
     * @param y       Target values.
     * @param verbose When true, the metrics are printed to stdout.
     * @return The loss, accuracy and RMSE of the model.
     * @throws std::runtime_error if the model has no layers or no loss.
     */
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

    /**
     * @brief Prints the progress of a single epoch to stdout.
     *
     * @param n_epochs    Total number of epochs.
     * @param epoch       Current epoch, starting at 1.
     * @param train_loss  Loss over the training set.
     * @param val_loss    Loss over the validation set.
     * @param train_acc   Accuracy over the training set.
     * @param val_acc     Accuracy over the validation set.
     * @param lr          Learning rate used during the epoch.
     * @param has_val     Whether a validation set was supplied.
     * @param model_type  The model type, used to select the metrics to display.
     */
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

    /**
     * @brief Trains the model.
     *
     * Runs the forward, backward and parameter-update passes for every epoch.
     * When the last layer uses a softmax activation and the loss is
     * categorical cross-entropy, a fused gradient kernel is used instead of
     * the generic path.
     *
     * @param X_train         Training inputs of shape (samples, n_inputs).
     * @param y_train         Training targets.
     * @param n_epochs        Number of epochs to run.
     * @param X_val           Optional validation inputs.
     * @param y_val           Optional validation targets.
     * @param print_every     Print progress every @p print_every epochs.
     * @param batch_size      Batch size; 0 trains on the full dataset.
     * @param validation_every Evaluate the validation set every
     *                        @p validation_every epochs.
     * @throws std::runtime_error if the model has no layers, no loss or no
     *         optimizer.
     * @throws std::invalid_argument if the training inputs and targets do not
     *         have the same non-zero row count, or if an interval is zero.
     */
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

} // namespace cpptorch
