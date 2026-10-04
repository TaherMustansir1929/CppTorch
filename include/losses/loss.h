#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"
#include "layers/dense.h"


class Loss
{
public:
    Matrix dinputs;

    virtual ~Loss() = default;
    Loss(const Loss &) = default;
    Loss &operator=(const Loss &) = default;
    Loss(Loss &&) = default;
    Loss &operator=(Loss &&) = default;
    Loss() = default;

    virtual Matrix forward(const Matrix &y_pred, const Matrix &y_true) = 0;
    virtual void backward(const Matrix &dvalues, const Matrix &y_true) = 0;

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

    static double regularization_loss(Layer_Dense &layer)
    {
        double reg_loss = 0;

        if (layer.weight_regularizer_l1 > 0)
        {
            reg_loss += layer.weight_regularizer_l1 * NumCpp::sum(NumCpp::abs(layer.weights));
        }
        if (layer.weight_regularizer_l2 > 0)
        {
            reg_loss += layer.weight_regularizer_l2 * NumCpp::sum(layer.weights.sq());
        }
        if (layer.bias_regularizer_l1 > 0)
        {
            reg_loss += layer.bias_regularizer_l1 * NumCpp::sum(NumCpp::abs(layer.biases));
        }
        if (layer.bias_regularizer_l2 > 0)
        {
            reg_loss += layer.bias_regularizer_l2 * NumCpp::sum(layer.biases.sq());
        }

        return reg_loss;
    }
};
