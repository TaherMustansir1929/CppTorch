#pragma once

#include "cpptorch/core/matrix.h"
#include "cpptorch/core/numcpp.h"
#include "cpptorch/layers/dense.h"

#include <cmath>
#include <cstdint>
#include <omp.h>

namespace cpptorch
{

/**
 * @brief Identifies the loss function used to score a model.
 *
 * The enum is the single entry point of the loss plug-in system: add a value
 * here and a matching case in @ref cpptorch::make_loss to expose a new loss to
 * @ref cpptorch::Model.
 */
enum class LossType : std::uint8_t
{
    MeanSquaredError = 0,        ///< Mean squared error, for regression.
    CategoricalCrossEntropy = 1  ///< Categorical cross-entropy, for classification.
};

/**
 * @brief Base class for loss functions.
 *
 * A loss scores a prediction against a target and produces the gradient of the
 * loss with respect to the prediction.
 *
 * @note To add a new loss, derive from this class, implement forward() and
 *       backward(), then register it in @ref cpptorch::make_loss.
 */
class Loss
{
private:
    static constexpr size_t OMP_THRESHOLD = 8192;

public:
    /// Gradient of the loss with respect to the prediction, filled by backward().
    Matrix dinputs;

    virtual ~Loss() = default;

    // Explicit noexcept guarantees zero-allocation transfers during vector scaling
    Loss(const Loss &) = default;
    Loss &operator=(const Loss &) = default;
    Loss(Loss &&) noexcept = default;
    Loss &operator=(Loss &&) noexcept = default;
    Loss() = default;

    /**
     * @brief Computes the per-sample loss.
     *
     * @param y_pred Predicted values of shape (samples, outputs).
     * @param y_true Target values, either class indices (samples, 1) or a
     *               one-hot matrix of shape (samples, outputs).
     * @return A column vector holding one loss value per sample.
     */
    virtual Matrix forward(const Matrix &y_pred, const Matrix &y_true) = 0;

    /**
     * @brief Computes the gradient of the loss with respect to the prediction.
     *
     * @param dvalues Gradient of the loss with respect to the prediction.
     * @param y_true  Target values.
     */
    virtual void backward(const Matrix &dvalues, const Matrix &y_true) = 0;

    /**
     * @brief Computes the mean loss over all samples.
     *
     * @param output Predicted values.
     * @param y      Target values.
     * @return The average of the per-sample losses.
     */
    double calculate(const Matrix &output, const Matrix &y)
    {
        Matrix sample_losses = forward(output, y);
        return NumCpp::mean(sample_losses);
    }

    /**
     * @brief Computes the classification accuracy.
     *
     * @param output Predicted probabilities of shape (samples, classes).
     * @param y      Target values, either class indices or a one-hot matrix.
     * @return The fraction of samples whose arg-max matches the target.
     */
    static double accuracy(const Matrix &output, const Matrix &y)
    {
        double correct = 0.0;
#pragma omp parallel for reduction(+ : correct) if (output.size() > 8192)
        for (size_t i = 0; i < output.rows(); ++i)
        {
            const size_t offset = i * output.cols();
            size_t predicted = 0;
            size_t expected = 0;
            for (size_t j = 1; j < output.cols(); ++j)
            {
                if (output[offset + j] > output[offset + predicted])
                    predicted = j;
                if (y.cols() != 1 && y[offset + j] > y[offset + expected])
                    expected = j;
            }
            if (y.cols() == 1)
                expected = static_cast<size_t>(y[i]);
            if (predicted == expected)
            {
                correct += 1.0;
            }
        }

        return correct / static_cast<double>(output.rows());
    }

    /**
     * @brief Computes the total regularization penalty of a layer.
     *
     * @param layer The layer whose L1 and L2 penalties are summed.
     * @return The weighted sum of the L1 and L2 penalties.
     */
    static double regularization_loss(const Layer_Dense &layer)
    {
        double reg_loss = 0.0;

        // O(1) Memory Manual Reduction Loops: Prevents full-size Matrix allocations
        if (layer.weight_regularizer_l1 > 0)
        {
            double penalty = 0.0;
#pragma omp parallel for simd reduction(+ : penalty) if (layer.weights.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < layer.weights.size(); ++i)
                penalty += std::abs(layer.weights[i]);
            reg_loss += layer.weight_regularizer_l1 * penalty;
        }

        if (layer.weight_regularizer_l2 > 0)
        {
            double penalty = 0.0;
#pragma omp parallel for simd reduction(+ : penalty) if (layer.weights.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < layer.weights.size(); ++i)
                penalty += (layer.weights[i] * layer.weights[i]);
            reg_loss += layer.weight_regularizer_l2 * penalty;
        }

        if (layer.bias_regularizer_l1 > 0)
        {
            double penalty = 0.0;
#pragma omp parallel for simd reduction(+ : penalty) if (layer.biases.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < layer.biases.size(); ++i)
                penalty += std::abs(layer.biases[i]);
            reg_loss += layer.bias_regularizer_l1 * penalty;
        }

        if (layer.bias_regularizer_l2 > 0)
        {
            double penalty = 0.0;
#pragma omp parallel for simd reduction(+ : penalty) if (layer.biases.size() > OMP_THRESHOLD)
            for (size_t i = 0; i < layer.biases.size(); ++i)
                penalty += (layer.biases[i] * layer.biases[i]);
            reg_loss += layer.bias_regularizer_l2 * penalty;
        }

        return reg_loss;
    }
};

} // namespace cpptorch
