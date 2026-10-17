#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"
#include "layers/dense.h"
#include <cmath>
#include <omp.h>

class Loss
{
private:
    static constexpr size_t OMP_THRESHOLD = 8192;

public:
    Matrix dinputs;

    virtual ~Loss() = default;

    // Explicit noexcept guarantees zero-allocation transfers during vector scaling
    Loss(const Loss &) = default;
    Loss &operator=(const Loss &) = default;
    Loss(Loss &&) noexcept = default;
    Loss &operator=(Loss &&) noexcept = default;
    Loss() = default;

    virtual Matrix forward(const Matrix &y_pred, const Matrix &y_true) = 0;
    virtual void backward(const Matrix &dvalues, const Matrix &y_true) = 0;

    double calculate(const Matrix &output, const Matrix &y)
    {
        Matrix sample_losses = forward(output, y);
        return NumCpp::mean(sample_losses);
    }

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
