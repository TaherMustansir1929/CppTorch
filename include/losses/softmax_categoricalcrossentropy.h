#pragma once

#include "categorical_crossentropy.h"
#include "layers/activations/softmax.h"


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
        // Set the output probabilities
        this->output = this->activation.output;
        // Calculate and return scalar loss value
        return this->loss.calculate(this->output, y_true);
    }

    // Alias for consistency with Loss::calculate
    double calculate(const Matrix &inputs, const Matrix &y_true)
    {
        return forward(inputs, y_true);
    }

    void backward(const Matrix &y_true)
    {
        int samples = static_cast<int>(this->output.rows());

        // If labels are one-hot encoded (cols > 1), turn them into discrete values
        Matrix y_true_discrete = (y_true.cols() > 1) ? NumCpp::argmax(y_true, 1) : y_true;

        // Initialize the gradient using the predicted probabilities
        this->dinputs = this->output;

        // Calculate gradient: (y_pred - y_true)
        for (int i = 0; i < samples; i++)
        {
            auto target_idx = static_cast<size_t>(y_true_discrete[i]);
            this->dinputs(i, target_idx) -= 1.0;
        }

        // Normalize gradient over samples
        this->dinputs /= samples;
    }
};