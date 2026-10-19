#pragma once

#include "loss.h"
#include <omp.h>

class Loss_MeanSquaredError : public Loss
{
public:
    Matrix forward(const Matrix &y_pred, const Matrix &y_true) override
    {
        size_t samples = y_pred.rows();
        size_t outputs = y_pred.cols();
        Matrix sample_losses(samples, 1);

// Avoids `diff` and `diff_sq` matrix creations by calculating directly to sum
#pragma omp parallel for if (samples * outputs > 8192)
        for (size_t i = 0; i < samples; ++i)
        {
            double loss_sum = 0.0;
            size_t row_offset = i * outputs;

#pragma omp simd reduction(+ : loss_sum)
            for (size_t j = 0; j < outputs; ++j)
            {
                double diff = y_pred[row_offset + j] - y_true[row_offset + j];
                loss_sum += (diff * diff);
            }

            sample_losses[i] = loss_sum / static_cast<double>(outputs);
        }
        return sample_losses;
    }

    void backward(const Matrix &dvalues, const Matrix &y_true) override
    {
        double samples = static_cast<double>(dvalues.rows());
        double outputs = static_cast<double>(dvalues.cols());
        double factor = -2.0 / (outputs * samples);

        this->dinputs.resize(dvalues.rows(), dvalues.cols());

// Scales factors uniformly inside a flat SIMD loop
#pragma omp parallel for simd if (dvalues.size() > 1024)
        for (size_t i = 0; i < dvalues.size(); ++i)
        {
            this->dinputs[i] = (y_true[i] - dvalues[i]) * factor;
        }
    }
};
