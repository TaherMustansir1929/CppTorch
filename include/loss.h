#pragma once

#include "numcpp.h"


class Loss
{
public:
    virtual ~Loss() = default;

    virtual Matrix forward(const Matrix &y_pred, const Matrix &y_true) = 0;

    double calculate(const Matrix &output, const Matrix &y)
    {
        Matrix sample_losses = forward(output, y);
        double data_loss = NumCpp::mean(sample_losses);
        return data_loss;
    }

    static double accuracy(const Matrix &output, const Matrix &y)
    {
        Matrix predictions = NumCpp::argmax(output, 1);
        Matrix y_max = NumCpp::argmax(y, 1);
        return NumCpp::mean(predictions == y_max);
    }
};


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
};