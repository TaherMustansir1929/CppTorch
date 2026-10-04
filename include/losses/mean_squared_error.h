#pragma once

#include "loss.h"


class Loss_MeanSquaredError : public Loss
{
public:

    Matrix forward(const Matrix &y_pred, const Matrix &y_true) override
    {
        // Calculate squared difference
        Matrix diff = y_pred - y_true;
        Matrix diff_sq = diff * diff;

        // Sample losses: mean across outputs (axis 1)
        Matrix sample_losses = NumCpp::sum(diff_sq, 1) / static_cast<double>(y_pred.cols());
        return sample_losses;
    }

    void backward(const Matrix &dvalues, const Matrix &y_true) override
    {
        // Number of samples
        auto samples = static_cast<double>(dvalues.rows());
        // Number of outputs in every sample
        auto outputs = static_cast<double>(dvalues.cols());

        // Gradient on values: -2 * (y_true - dvalues) / outputs / samples
        this->dinputs = -2.0 * (y_true - dvalues) / outputs;
        this->dinputs /= samples;
    }
};