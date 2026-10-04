#pragma once

#include "loss.h"


class Loss_CategoricalCrossEntropy : public Loss
{
public:


    Matrix forward(const Matrix &y_pred, const Matrix &y_true) override
    {
        Matrix y_pred_clipped = NumCpp::clip(y_pred, 1e-7, 1 - 1e-7);
        Matrix correct_confidences = NumCpp::sum(y_pred_clipped * y_true, 1);
        Matrix negative_log_likelihoods = -1 * NumCpp::log(correct_confidences);
        return negative_log_likelihoods;
    }

    void backward(const Matrix &dvalues, const Matrix &y_true) override
    {
        // Number of samples
        int samples = static_cast<int>(dvalues.rows());

        // Calculate gradient
        this->dinputs = -1 * y_true / dvalues;
        // Normalize gradient
        this->dinputs /= samples;
    }
};