#pragma once

#include "core/matrix.h"
#include "core/numcpp.h"
class StandardScaler
{
    Matrix mean, std_dev;

public:
    void fit(const Matrix &X)
    {
        mean = NumCpp::sum(X, 0) / static_cast<double>(X.rows());
        Matrix diff = X - mean;
        Matrix variance = NumCpp::sum(diff * diff, 0) / static_cast<double>(X.rows());
        std_dev = NumCpp::sqrt(variance) + 1e-7;
    }

    void fit_transform(Matrix &X)
    {
        fit(X);
        X = (X - mean) / std_dev;
    }

    void transform(Matrix &X)
    {
        X = (X - mean) / std_dev;
    }
};