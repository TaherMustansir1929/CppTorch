#pragma once

#include "cpptorch/losses/categorical_crossentropy.h"
#include "cpptorch/layers/activations/softmax.h"
#include <omp.h>

namespace cpptorch
{

/**
 * @brief Fused softmax activation and categorical cross-entropy loss.
 *
 * Combining the two allows the gradient of the softmax to be folded into the
 * loss gradient, which avoids materialising an intermediate Jacobian and is
 * noticeably faster than running the two steps separately.
 */
class Activation_Softmax_Loss_CategoricalCrossentropy
{
public:
    /// The softmax activation.
    Activation_Softmax activation;
    /// The categorical cross-entropy loss.
    Loss_CategoricalCrossEntropy loss;

    /// Output of the softmax, of shape (samples, classes).
    Matrix output;
    /// Gradient of the loss with respect to the layer input.
    Matrix dinputs;

    /**
     * @brief Runs the softmax forward and scores the result.
     * @param inputs  Input matrix of shape (samples, classes).
     * @param y_true  Target values.
     * @return The loss value; also stored in the output of the activation.
     */
    [[nodiscard("data_loss value ignored")]] double forward(const Matrix &inputs, const Matrix &y_true)
    {
        this->activation.forward(inputs);
        this->output = this->activation.output;
        return this->loss.calculate(this->output, y_true);
    }

    /**
     * @brief Alias of forward() that only returns the loss value.
     * @param inputs Input matrix of shape (samples, classes).
     * @param y_true Target values.
     * @return The loss value.
     */
    double calculate(const Matrix &inputs, const Matrix &y_true)
    {
        return forward(inputs, y_true);
    }

    /**
     * @brief Computes the fused gradient with respect to the layer input.
     * @param y_true Target values.
     */
    void backward(const Matrix &y_true)
    {
        int samples = static_cast<int>(this->output.rows());
        int cols = static_cast<int>(this->output.cols());
        double inv_samples = 1.0 / static_cast<double>(samples);

        Matrix y_true_discrete = (y_true.cols() > 1) ? NumCpp::argmax(y_true, 1) : y_true;
        this->dinputs.resize(samples, cols);

// Assigns, evaluates target index logic, and normalizes by samples all in one pass
#pragma omp parallel for if (samples > 64)
        for (int i = 0; i < samples; i++)
        {
            size_t target_idx = static_cast<size_t>(y_true_discrete[i]);
            size_t row_offset = i * cols;

#pragma omp simd
            for (int j = 0; j < cols; ++j)
            {
                double val = this->output[row_offset + j];
                if (j == target_idx)
                {
                    val -= 1.0;
                }
                this->dinputs[row_offset + j] = val * inv_samples;
            }
        }
    }
};

} // namespace cpptorch
