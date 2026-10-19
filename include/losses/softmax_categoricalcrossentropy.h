#pragma once

#include "categorical_crossentropy.h"
#include "layers/activations/softmax.h"
#include <omp.h>

class Activation_Softmax_Loss_CategoricalCrossentropy
{
public:
    Activation_Softmax activation;
    Loss_CategoricalCrossEntropy loss;

    Matrix output, dinputs;

    [[nodiscard("data_loss value ignored")]] double forward(const Matrix &inputs, const Matrix &y_true)
    {
        this->activation.forward(inputs);
        this->output = this->activation.output;
        return this->loss.calculate(this->output, y_true);
    }

    double calculate(const Matrix &inputs, const Matrix &y_true)
    {
        return forward(inputs, y_true);
    }

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
