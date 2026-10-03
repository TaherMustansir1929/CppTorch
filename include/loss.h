#pragma once

#include "activation.h"
#include "layer_dense.h"
#include "matrix.h"
#include "numcpp.h"


class Loss
{
public:
    virtual ~Loss() = default;

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
            reg_loss += layer.weight_regularizer_l2 * NumCpp::sum((layer.weights ^ 2));
        }
        if (layer.bias_regularizer_l1 > 0)
        {
            reg_loss += layer.bias_regularizer_l1 * NumCpp::sum(NumCpp::abs(layer.biases));
        }
        if (layer.bias_regularizer_l2 > 0)
        {
            reg_loss += layer.bias_regularizer_l2 * NumCpp::sum((layer.biases ^ 2));
        }

        return reg_loss;
    }
};


class Loss_CategoricalCrossEntropy : public Loss
{
public:
    Matrix dinputs;


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
        int samples = dvalues.rows();

        // Calculate gradient
        this->dinputs = -1 * y_true / dvalues;
        // Normalize gradient
        this->dinputs /= samples;
    }
};


class Activation_Softmax_Loss_CategoricalCrossentropy
{
public:
    Activation_Softmax activation;
    Loss_CategoricalCrossEntropy loss;

    Matrix output, dinputs;


    [[nodiscard("data_loss value ignored")]] double forward(const Matrix &inputs, const Matrix &y_true)
    {
        // Output layer's activation function
        this->activation.forward(inputs);
        // Set the output
        this->output = this->activation.output;
        // Calculate and return loss value
        return this->loss.calculate(this->output, y_true);
    }

    void backward(const Matrix &y_true)
    {
        // Use the cached output to determine the number of samples
        int samples = this->output.rows();

        // Turn labels from one-hot to discrete values
        Matrix y_true_discrete = NumCpp::argmax(y_true, 1);

        // Initialize the gradient using the predicted probabilities from the forward pass
        this->dinputs = this->output;

        // Calculate gradient
        for (int i = 0; i < samples; i++)
        {
            // Note: Added a static_cast to safely use the float as a matrix index
            this->dinputs(i, static_cast<size_t>(y_true_discrete[i])) -= 1;
        }

        // Normalize gradient
        this->dinputs /= samples;
    }
};