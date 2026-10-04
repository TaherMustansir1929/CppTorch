#pragma once

#include "activation.h"
#include "core/numcpp.h"


class Activation_Softmax : public Activation
{
public:
    void forward(const Matrix &inputs) override
    {
        // Remember input values
        this->inputs = inputs;

        Matrix exp_values = NumCpp::exp(inputs - NumCpp::max(inputs, 1));
        Matrix probabilities = exp_values / NumCpp::sum(exp_values, 1);

        this->output = probabilities;
    }

    void backward(const Matrix &dvalues) override
    {
        // Create uninitialized array for sample gradients
        this->dinputs = Matrix(dvalues.rows(), dvalues.cols());

        size_t samples = dvalues.rows();
        size_t features = dvalues.cols();

        // Enumerate outputs and gradients for each sample
        for (size_t i = 0; i < samples; i++)
        {
            // Extract single output and single dvalues as column vectors (features x 1)
            Matrix single_output(features, 1);
            Matrix single_dvalues(features, 1);
            for (size_t j = 0; j < features; j++)
            {
                single_output[j] = this->output(i, j);
                single_dvalues[j] = dvalues(i, j);
            }

            // Calculate Jacobian matrix of the output:
            // J = diagflat(single_output) - dot(single_output, single_output.T)
            Matrix jacobian_matrix = NumCpp::diagflat(single_output) - NumCpp::dot(single_output, single_output.T());

            // Calculate sample-wise gradient:
            // sample_dinputs = dot(jacobian_matrix, single_dvalues)
            Matrix sample_dinputs = NumCpp::dot(jacobian_matrix, single_dvalues);

            // Store into dinputs
            for (size_t j = 0; j < features; j++)
            {
                this->dinputs(i, j) = sample_dinputs[j];
            }
        }
    }
};
