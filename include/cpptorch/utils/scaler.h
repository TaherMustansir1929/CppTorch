#pragma once

#include "cpptorch/core/matrix.h"
#include "cpptorch/core/numcpp.h"

namespace cpptorch
{
/**
 * @brief Standardises features to zero mean and unit variance.
 *
 * The scaler is fitted on the training set and then applied unchanged to the
 * validation and test sets.
 */
class StandardScaler
{
    /// Per-feature mean used by the scaler.
    Matrix mean;
    /// Per-feature standard deviation used by the scaler.
    Matrix std_dev;

public:
    /**
     * @brief Computes the mean and standard deviation of @p X.
     * @param X The feature matrix to fit on.
     */
    void fit(const Matrix &X)
    {
        mean = NumCpp::sum(X, 0) / static_cast<double>(X.rows());
        Matrix diff = X - mean;
        Matrix variance = NumCpp::sum(diff * diff, 0) / static_cast<double>(X.rows());
        std_dev = NumCpp::sqrt(variance) + 1e-7;
    }

    /**
     * @brief Fits the scaler on @p X and standardises it in place.
     * @param X The feature matrix to standardise.
     */
    void fit_transform(Matrix &X)
    {
        fit(X);
        X = (X - mean) / std_dev;
    }

    /**
     * @brief Standardises @p X in place using the fitted statistics.
     * @param X The feature matrix to standardise.
     */
    void transform(Matrix &X)
    {
        X = (X - mean) / std_dev;
    }
};

} // namespace cpptorch
