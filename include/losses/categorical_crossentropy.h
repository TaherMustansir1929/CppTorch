#pragma once

#include "loss.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

class Loss_CategoricalCrossEntropy : public Loss
{
public:
    Matrix forward(const Matrix &y_pred, const Matrix &y_true) override
    {
        size_t samples = y_pred.rows();
        size_t classes = y_pred.cols();
        Matrix sample_losses(samples, 1);

// Eliminates 3 intermediate matrices (clipped, confidences, neg_log)
#pragma omp parallel for if (samples * classes > 8192)
        for (size_t i = 0; i < samples; ++i)
        {
            double conf = 0.0;
            size_t row_offset = i * classes;

            if (y_true.cols() == 1)
            {
                const size_t target_idx = static_cast<size_t>(y_true[i]);
                conf = std::clamp(y_pred[row_offset + target_idx], 1e-7, 1.0 - 1e-7);
                sample_losses[i] = -std::log(conf);
                continue;
            }

#pragma omp simd reduction(+ : conf)
            for (size_t j = 0; j < classes; ++j)
            {
                double clipped = std::clamp(y_pred[row_offset + j], 1e-7, 1.0 - 1e-7);
                conf += clipped * y_true[row_offset + j];
            }

            sample_losses[i] = -std::log(conf);
        }
        return sample_losses;
    }

    void backward(const Matrix &dvalues, const Matrix &y_true) override
    {
        double samples = static_cast<double>(dvalues.rows());
        this->dinputs.resize(dvalues.rows(), dvalues.cols());

        if (y_true.cols() == 1)
        {
#pragma omp parallel for if (dvalues.size() > 8192)
            for (size_t i = 0; i < dvalues.rows(); ++i)
            {
                const size_t row_offset = i * dvalues.cols();
                const size_t target_idx = static_cast<size_t>(y_true[i]);
                for (size_t j = 0; j < dvalues.cols(); ++j)
                    this->dinputs[row_offset + j] = 0.0;
                this->dinputs[row_offset + target_idx] =
                    -1.0 / dvalues[row_offset + target_idx] / samples;
            }
            return;
        }

// Single pass execution avoids chaining allocation operations
#pragma omp parallel for simd if (dvalues.size() > 1024)
        for (size_t i = 0; i < dvalues.size(); ++i)
        {
            this->dinputs[i] = (-y_true[i] / dvalues[i]) / samples;
        }
    }
};
