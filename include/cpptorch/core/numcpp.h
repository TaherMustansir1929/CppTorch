#pragma once

#include "cpptorch/core/matrix.h"
#include "cpptorch/utils/exceptions.h"
#include "cpptorch/utils/utils.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <omp.h>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include <cblas.h>

namespace cpptorch
{

/**
 * @brief numpy-style operations on @ref cpptorch::Matrix.
 *
 * All operations are static and return new matrices; nothing is modified in
 * place unless the name ends with `_into` or `_to`. The `*_into` variants
 * write into a caller-provided buffer to avoid allocations in hot loops.
 *
 * @code
 * cpptorch::Matrix a = cpptorch::NumCpp::randn({100, 3});
 * cpptorch::Matrix b = cpptorch::NumCpp::dot(a, a.T());
 * @endcode
 */
class NumCpp
{
private:
    static constexpr size_t OMP_THRESHOLD = 8192;
    static constexpr size_t AXIS_OMP_THRESHOLD = 128;

    // Helper template replaces the broken macro to cleanly execute SIMD math operations
    template <typename Func>
    static Matrix element_wise_op(const Matrix &nc_arr, Func func)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
#pragma omp parallel for simd if (nc_arr.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < nc_arr.size(); ++i)
        {
            result[i] = func(nc_arr[i]);
        }
        return result;
    }

public:
    // =============================
    // ===== MATRIX ASSIGNMENT =====
    // =============================

    /**
     * @brief Creates a matrix filled with zeros.
     * @param shape The (rows, cols) shape of the result.
     */
    static Matrix zeroes(std::tuple<size_t, size_t> shape)
    {
        return Matrix(std::get<0>(shape), std::get<1>(shape));
    }

    /**
     * @brief Creates a matrix filled with ones.
     * @param shape The (rows, cols) shape of the result.
     */
    static Matrix ones(std::tuple<int, int> shape)
    {
        auto [row, col] = shape;
        return Matrix(row, col) + 1.0;
    }

    /**
     * @brief Builds a diagonal matrix from a flat vector.
     * @param nc_arr A matrix with a single element per row.
     */
    static Matrix diagflat(const Matrix &nc_arr)
    {
        size_t n = nc_arr.size();
        Matrix result(n, n);

#pragma omp parallel for if (n > OMP_THRESHOLD)
        for (size_t i = 0; i < n; i++)
        {
            result(i, i) = nc_arr[i];
        }
        return result;
    }

    // ========================================
    // ===== RANDOM: NORMAL DISTRIBUTION ======
    // ========================================

    /**
     * @brief Creates a matrix of samples from the standard normal distribution.
     * @param shape The (rows, cols) shape of the result.
     */
    static Matrix randn(std::tuple<size_t, size_t> shape)
    {
        auto [row, col] = shape;
        Matrix nc_arr(row, col);

        for (double &i : nc_arr)
        {
            i = randomNormalValue();
        }

        return nc_arr;
    }

    // ==========================================
    // ===== RANDOM: BINOMIAL DISTRIBUTION ======
    // ==========================================

    /**
     * @brief Creates a matrix of samples from the binomial distribution.
     * @param n     Number of trials.
     * @param p     Probability of success, between 0 and 1.
     * @param shape The (rows, cols) shape of the result.
     * @throws std::invalid_argument if @p n is negative or @p p is outside [0, 1].
     */
    static Matrix binomial(int n, double p, std::tuple<size_t, size_t> shape)
    {
        if (n < 0 || p < 0.0 || p > 1.0)
        {
            throw std::invalid_argument("n must be >= 0 and p must be between 0.0 and 1.0 inclusive.");
        }

        auto [rows, cols] = shape;
        Matrix result(rows, cols);

        for (double &i : result)
        {
            i = randomBinomialValue(n, p);
        }

        return result;
    }

    /**
     * @brief Fills @p result with samples from the binomial distribution.
     * @param n      Number of trials.
     * @param p      Probability of success, between 0 and 1.
     * @param shape  The (rows, cols) shape of @p result.
     * @param result The matrix to fill; resized when needed.
     * @throws std::invalid_argument if @p n is negative or @p p is outside [0, 1].
     */
    static void binomial_into(int n, double p, std::tuple<size_t, size_t> shape, Matrix &result)
    {
        if (n < 0 || p < 0.0 || p > 1.0)
            throw std::invalid_argument("n must be >= 0 and p must be between 0.0 and 1.0 inclusive.");

        auto [rows, cols] = shape;
        result.resize(rows, cols);
#pragma omp parallel for if (result.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = randomBinomialValue(n, p);
    }

    // ==============================
    // ===== LINSPACE MATRIX ========
    // ==============================

    /**
     * @brief Creates @p num evenly spaced samples over the closed interval.
     * @param start The first value.
     * @param stop  The last value.
     * @param num   The number of samples.
     */
    static Matrix linspace(double start, double stop, size_t num)
    {
        if (num == 0)
            return {};
        if (num == 1)
            return Matrix(1, 1, {start});

        Matrix result(num, 1);
        double step = (stop - start) / static_cast<double>(num - 1);

#pragma omp parallel for simd if (num > OMP_THRESHOLD)
        for (size_t i = 0; i < num; i++)
        {
            result[i] = start + (i * step);
        }
        return result;
    }

    // =============================
    // ===== DOT PRODUCT ===========
    // =============================

    /**
     * @brief Computes the matrix product `A * B` into @p result.
     * @param A      Left operand of shape (m, k).
     * @param B      Right operand of shape (k, n).
     * @param result The matrix to fill; resized when needed.
     * @throws cpptorch::ShapeInvalidForOperation if the inner dimensions differ.
     */
    static void dot_into(const Matrix &A, const Matrix &B, Matrix &result)
    {
        if (A.cols() != B.rows())
        {
            throw ShapeInvalidForOperation("dot product", A.shape(), B.shape());
        }

        size_t rowSize = A.rows();
        size_t colSize = B.cols();
        size_t innerSize = A.cols();

        result.resize(rowSize, colSize);
        if (rowSize > INT_MAX || colSize > INT_MAX || innerSize > INT_MAX)
            throw std::length_error("BLAS matrix dimensions exceed the supported integer range.");
        if (rowSize == 0 || colSize == 0 || innerSize == 0)
        {
            std::fill(result.begin(), result.end(), 0.0);
            return;
        }
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    static_cast<int>(rowSize),
                    static_cast<int>(colSize),
                    static_cast<int>(innerSize),
                    1.0,
                    &A[0],
                    static_cast<int>(innerSize),
                    &B[0],
                    static_cast<int>(colSize),
                    0.0,
                    &result[0],
                    static_cast<int>(colSize));
    }

    /**
     * @brief Computes the matrix product `A * B` into @p result, reading A as a view.
     * @param A      Left operand view of shape (m, k).
     * @param B      Right operand of shape (k, n).
     * @param result The matrix to fill; resized when needed.
     * @throws cpptorch::ShapeInvalidForOperation if the inner dimensions differ.
     */
    static void dot_into(const MatrixView &A, const Matrix &B, Matrix &result)
    {
        if (A.cols() != B.rows())
            throw ShapeInvalidForOperation("dot product",
                                           {static_cast<int>(A.rows()), static_cast<int>(A.cols())},
                                           {static_cast<int>(B.rows()), static_cast<int>(B.cols())});
        result.resize(A.rows(), B.cols());
        if (A.rows() > INT_MAX || B.cols() > INT_MAX || A.cols() > INT_MAX)
            throw std::length_error("BLAS matrix dimensions exceed the supported integer range.");
        if (A.rows() == 0 || B.cols() == 0 || A.cols() == 0)
        {
            std::fill(result.begin(), result.end(), 0.0);
            return;
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(A.rows()), static_cast<int>(B.cols()), static_cast<int>(A.cols()),
                    1.0, A.data(), static_cast<int>(A.cols()), &B[0], static_cast<int>(B.cols()),
                    0.0, &result[0], static_cast<int>(B.cols()));
    }

    /**
     * @brief Computes the matrix product `A * B`.
     * @param A Left operand of shape (m, k).
     * @param B Right operand of shape (k, n).
     * @return The product of shape (m, n).
     * @throws cpptorch::ShapeInvalidForOperation if the inner dimensions differ.
     */
    static Matrix dot(const Matrix &A, const Matrix &B)
    {
        Matrix result;
        dot_into(A, B, result);
        return result;
    }

    // Computes A^T * B without materializing a transposed copy of A.
    /**
     * @brief Computes `A^T * B` into @p result without transposing A.
     * @param A      Left operand of shape (k, m); it is transposed internally.
     * @param B      Right operand of shape (k, n).
     * @param result The matrix to fill; resized when needed.
     * @throws cpptorch::ShapeInvalidForOperation if the row counts differ.
     */
    static void dot_transpose_left_into(const Matrix &A, const Matrix &B, Matrix &result)
    {
        if (A.rows() != B.rows())
        {
            throw ShapeInvalidForOperation("dot product", A.shape(), B.shape());
        }

        const size_t samples = A.rows();
        const size_t features = A.cols();
        const size_t neurons = B.cols();
        result.resize(features, neurons);
        if (features > INT_MAX || neurons > INT_MAX || samples > INT_MAX)
            throw std::length_error("BLAS matrix dimensions exceed the supported integer range.");
        if (features == 0 || neurons == 0 || samples == 0)
        {
            std::fill(result.begin(), result.end(), 0.0);
            return;
        }
        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<int>(features),
                    static_cast<int>(neurons),
                    static_cast<int>(samples),
                    1.0,
                    &A[0],
                    static_cast<int>(features),
                    &B[0],
                    static_cast<int>(neurons),
                    0.0,
                    &result[0],
                    static_cast<int>(neurons));
    }

    /**
     * @brief Computes `A^T * B` into @p result, reading A as a view.
     * @param A      Left operand view of shape (k, m); it is transposed internally.
     * @param B      Right operand of shape (k, n).
     * @param result The matrix to fill; resized when needed.
     * @throws cpptorch::ShapeInvalidForOperation if the row counts differ.
     */
    static void dot_transpose_left_into(const MatrixView &A, const Matrix &B, Matrix &result)
    {
        if (A.rows() != B.rows())
            throw ShapeInvalidForOperation("dot product",
                                           {static_cast<int>(A.rows()), static_cast<int>(A.cols())},
                                           {static_cast<int>(B.rows()), static_cast<int>(B.cols())});
        result.resize(A.cols(), B.cols());
        if (A.cols() > INT_MAX || B.cols() > INT_MAX || A.rows() > INT_MAX)
            throw std::length_error("BLAS matrix dimensions exceed the supported integer range.");
        if (A.cols() == 0 || B.cols() == 0 || A.rows() == 0)
        {
            std::fill(result.begin(), result.end(), 0.0);
            return;
        }
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    static_cast<int>(A.cols()), static_cast<int>(B.cols()), static_cast<int>(A.rows()),
                    1.0, A.data(), static_cast<int>(A.cols()), &B[0], static_cast<int>(B.cols()),
                    0.0, &result[0], static_cast<int>(B.cols()));
    }

    /**
     * @brief Computes `A^T * B` without materialising the transpose of A.
     * @param A Left operand of shape (k, m).
     * @param B Right operand of shape (k, n).
     * @return The product of shape (m, n).
     * @throws cpptorch::ShapeInvalidForOperation if the row counts differ.
     */
    static Matrix dot_transpose_left(const Matrix &A, const Matrix &B)
    {
        Matrix result;
        dot_transpose_left_into(A, B, result);
        return result;
    }

    // Computes A * B^T without materializing a transposed copy of B.
    /**
     * @brief Computes `A * B^T` into @p result without transposing B.
     * @param A      Left operand of shape (m, k).
     * @param B      Right operand of shape (n, k); it is transposed internally.
     * @param result The matrix to fill; resized when needed.
     * @throws cpptorch::ShapeInvalidForOperation if the column counts differ.
     */
    static void dot_transpose_right_into(const Matrix &A, const Matrix &B, Matrix &result)
    {
        if (A.cols() != B.cols())
        {
            throw ShapeInvalidForOperation("dot product", A.shape(), B.shape());
        }

        const size_t samples = A.rows();
        const size_t features = A.cols();
        const size_t neurons = B.rows();
        result.resize(samples, neurons);
        if (samples > INT_MAX || neurons > INT_MAX || features > INT_MAX)
            throw std::length_error("BLAS matrix dimensions exceed the supported integer range.");
        if (samples == 0 || neurons == 0 || features == 0)
        {
            std::fill(result.begin(), result.end(), 0.0);
            return;
        }
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<int>(samples),
                    static_cast<int>(neurons),
                    static_cast<int>(features),
                    1.0,
                    &A[0],
                    static_cast<int>(features),
                    &B[0],
                    static_cast<int>(features),
                    0.0,
                    &result[0],
                    static_cast<int>(neurons));
    }

    /**
     * @brief Computes `A * B^T` without materialising the transpose of B.
     * @param A Left operand of shape (m, k).
     * @param B Right operand of shape (n, k).
     * @return The product of shape (m, n).
     * @throws cpptorch::ShapeInvalidForOperation if the column counts differ.
     */
    static Matrix dot_transpose_right(const Matrix &A, const Matrix &B)
    {
        Matrix result;
        dot_transpose_right_into(A, B, result);
        return result;
    }

    // =============================
    // ===== MATRIX SUMMATION ======
    // =============================

    /**
     * @brief Returns the sum of all elements.
     * @param A The matrix to reduce.
     */
    static double sum(const Matrix &A)
    {
        double total = 0.0;

#pragma omp parallel for simd reduction(+ : total) if (A.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < A.size(); i++)
        {
            total += A[i];
        }
        return total;
    }

    /**
     * @brief Returns the sum along one axis.
     * @param A    The matrix to reduce.
     * @param axis 0 to sum over the rows, 1 to sum over the columns.
     * @return A (1, cols) matrix for axis 0, a (rows, 1) matrix for axis 1.
     * @throws std::invalid_argument if @p axis is neither 0 nor 1.
     */
    static Matrix sum(const Matrix &A, int axis)
    {
        if (axis == 0)
        {
            Matrix result(1, A.cols());
            for (size_t j = 0; j < A.rows(); j++)
            {
                size_t row_offset = j * A.cols();
#pragma omp simd
                for (size_t i = 0; i < A.cols(); i++)
                {
                    result[i] += A[row_offset + i];
                }
            }
            return result;
        }

        if (axis == 1)
        {
            Matrix result(A.rows(), 1);
#pragma omp parallel for if (A.rows() > AXIS_OMP_THRESHOLD)
            for (size_t i = 0; i < A.rows(); i++)
            {
                double total = 0.0;
                size_t row_offset = i * A.cols();
#pragma omp simd reduction(+ : total)
                for (size_t j = 0; j < A.cols(); j++)
                {
                    total += A[row_offset + j];
                }
                result[i] = total;
            }
            return result;
        }

        throw std::invalid_argument("axis must be 0 or 1");
    }

    /**
     * @brief Computes the sum along an axis into @p result.
     * @param A      The matrix to reduce.
     * @param axis   Only axis 0 is currently supported.
     * @param result The matrix to fill; resized when needed.
     * @throws std::invalid_argument if @p axis is not 0.
     */
    static void sum_into(const Matrix &A, int axis, Matrix &result)
    {
        if (axis != 0)
            throw std::invalid_argument("sum_into currently supports axis 0 only");
        result.resize(1, A.cols());
        std::fill(result.begin(), result.end(), 0.0);
        for (size_t row = 0; row < A.rows(); ++row)
        {
            const size_t offset = row * A.cols();
#pragma omp simd
            for (size_t col = 0; col < A.cols(); ++col)
                result[col] += A[offset + col];
        }
    }

    // =============================
    // ===== MATRIX MAXIMUM ========
    // =============================

    /**
     * @brief Returns the element-wise maximum of a scalar and a matrix.
     * @param operand The scalar operand.
     * @param nc_arr  The matrix operand.
     */
    static Matrix maximum(double operand, const Matrix &nc_arr)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
#pragma omp parallel for simd if (nc_arr.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            result[i] = std::max(operand, nc_arr[i]);
        }
        return result;
    }

    /**
     * @brief Returns the largest element of a matrix.
     * @param nc_arr The matrix to reduce.
     * @throws std::out_of_range if the matrix is empty.
     */
    static double max(const Matrix &nc_arr)
    {
        if (nc_arr.size() == 0)
            throw std::out_of_range("Matrix is empty");

        double maxValue = nc_arr[0];
#pragma omp parallel for simd reduction(max : maxValue) if (nc_arr.size() > OMP_THRESHOLD)
        for (size_t i = 1; i < nc_arr.size(); i++)
        {
            maxValue = std::max(maxValue, nc_arr[i]);
        }
        return maxValue;
    }

    /**
     * @brief Returns the maximum along one axis.
     * @param nc_arr The matrix to reduce.
     * @param axis   0 to reduce over the rows, 1 to reduce over the columns.
     * @return A (1, cols) matrix for axis 0, a (rows, 1) matrix for axis 1.
     * @throws std::invalid_argument if @p axis is neither 0 nor 1.
     */
    static Matrix max(const Matrix &nc_arr, int axis)
    {
        if (axis == 0)
        {
            Matrix result(1, nc_arr.cols());
            for (size_t i = 0; i < nc_arr.cols(); i++)
                result[i] = nc_arr[i];

            for (size_t j = 1; j < nc_arr.rows(); j++)
            {
                size_t row_offset = j * nc_arr.cols();
#pragma omp simd
                for (size_t i = 0; i < nc_arr.cols(); i++)
                {
                    result[i] = std::max(result[i], nc_arr[row_offset + i]);
                }
            }
            return result;
        }

        if (axis == 1)
        {
            Matrix result(nc_arr.rows(), 1);
#pragma omp parallel for if (nc_arr.rows() > AXIS_OMP_THRESHOLD)
            for (size_t i = 0; i < nc_arr.rows(); i++)
            {
                size_t row_offset = i * nc_arr.cols();
                double maxValue = nc_arr[row_offset];
#pragma omp simd reduction(max : maxValue)
                for (size_t j = 1; j < nc_arr.cols(); j++)
                {
                    maxValue = std::max(maxValue, nc_arr[row_offset + j]);
                }
                result[i] = maxValue;
            }
            return result;
        }

        throw std::invalid_argument("axis must be 0 or 1");
    }

    /**
     * @brief Returns the position of the largest element.
     * @param nc_arr The matrix to search.
     * @return The (row, col) coordinates of the maximum.
     */
    static std::tuple<size_t, size_t> argmax(const Matrix &nc_arr)
    {
        if (nc_arr.size() == 0)
            return {0, 0};

        size_t maxIdx = 0;
        double maxValue = nc_arr[0];

        for (size_t i = 1; i < nc_arr.size(); i++)
        {
            if (nc_arr[i] > maxValue)
            {
                maxValue = nc_arr[i];
                maxIdx = i;
            }
        }
        return {maxIdx / nc_arr.cols(), maxIdx % nc_arr.cols()};
    }

    /**
     * @brief Returns the index of the maximum along one axis.
     * @param nc_arr The matrix to search.
     * @param axis   0 to reduce over the rows, 1 to reduce over the columns.
     * @return A (1, cols) matrix for axis 0, a (rows, 1) matrix for axis 1.
     * @throws std::invalid_argument if @p axis is neither 0 nor 1.
     */
    static Matrix argmax(const Matrix &nc_arr, int axis)
    {
        if (axis == 0)
        {
            Matrix result(1, nc_arr.cols());
            std::vector<double> max_vals(nc_arr.cols());

            for (size_t i = 0; i < nc_arr.cols(); i++)
            {
                max_vals[i] = nc_arr[i];
                result[i] = 0.0;
            }

            for (size_t j = 1; j < nc_arr.rows(); j++)
            {
                size_t row_offset = j * nc_arr.cols();
#pragma omp simd
                for (size_t i = 0; i < nc_arr.cols(); i++)
                {
                    if (nc_arr[row_offset + i] > max_vals[i])
                    {
                        max_vals[i] = nc_arr[row_offset + i];
                        result[i] = static_cast<double>(j);
                    }
                }
            }
            return result;
        }

        if (axis == 1)
        {
            Matrix result(nc_arr.rows(), 1);
#pragma omp parallel for if (nc_arr.rows() > AXIS_OMP_THRESHOLD)
            for (size_t i = 0; i < nc_arr.rows(); i++)
            {
                size_t row_offset = i * nc_arr.cols();
                size_t maxIdx = 0;
                double maxValue = nc_arr[row_offset];

                for (size_t j = 1; j < nc_arr.cols(); j++)
                {
                    if (nc_arr[row_offset + j] > maxValue)
                    {
                        maxValue = nc_arr[row_offset + j];
                        maxIdx = j;
                    }
                }
                result[i] = static_cast<double>(maxIdx);
            }
            return result;
        }

        throw std::invalid_argument("axis must be 0 or 1");
    }

    // =============================================
    // ===== ELEMENT-WISE OPENMP OPERATIONS ========
    // =============================================

    /// Returns the element-wise exponential.
    static Matrix exp(const Matrix &nc_arr)
    {
        return element_wise_op(nc_arr, [](double x) { return std::exp(x); });
    }
    /// Returns the element-wise natural logarithm.
    static Matrix log(const Matrix &nc_arr)
    {
        return element_wise_op(nc_arr, [](double x) { return std::log(x); });
    }
    /// Returns the element-wise square root.
    static Matrix sqrt(const Matrix &nc_arr)
    {
        return element_wise_op(nc_arr, [](double x) { return std::sqrt(x); });
    }
    /// Returns the element-wise absolute value.
    static Matrix abs(const Matrix &nc_arr)
    {
        return element_wise_op(nc_arr, [](double x) { return std::abs(x); });
    }
    /// Returns the element-wise sine.
    static Matrix sin(const Matrix &nc_arr)
    {
        return element_wise_op(nc_arr, [](double x) { return std::sin(x); });
    }
    /// Returns the element-wise cosine.
    static Matrix cos(const Matrix &nc_arr)
    {
        return element_wise_op(nc_arr, [](double x) { return std::cos(x); });
    }

    // ===========================
    // ===== MATRIX MEAN AVG =====
    // ===========================

    /**
     * @brief Returns the arithmetic mean of all elements.
     * @param nc_arr The matrix to reduce; an empty matrix averages to zero.
     */
    static double mean(const Matrix &nc_arr)
    {
        if (nc_arr.size() == 0)
            return 0.0;
        double meanVal = 0.0;

#pragma omp parallel for simd reduction(+ : meanVal) if (nc_arr.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            meanVal += nc_arr[i];
        }

        return meanVal / static_cast<double>(nc_arr.size());
    }

    // ===========================
    // ===== MATRIX CLIPPING =====
    // ===========================

    /**
     * @brief Clamps every element to the closed interval [min_val, max_val].
     * @param nc_arr  The matrix to clamp.
     * @param min_val The lower bound.
     * @param max_val The upper bound.
     */
    static Matrix clip(const Matrix &nc_arr, double min_val, double max_val)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());

#pragma omp parallel for simd if (nc_arr.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            result[i] = std::clamp(nc_arr[i], min_val, max_val);
        }
        return result;
    }

    // ============================
    // ===== ONE HOT ENCODING =====
    // ============================

    /**
     * @brief Converts a column of class indices into a one-hot matrix.
     * @param y         A matrix of class indices.
     * @param n_classes The number of classes.
     * @return A matrix of shape (y.size(), n_classes).
     */
    static Matrix one_hot(const Matrix &y, size_t n_classes)
    {
        Matrix Y_encoded(y.size(), n_classes);

#pragma omp parallel for if (y.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < y.size(); i++)
        {
            Y_encoded(i, static_cast<size_t>(y[i])) = 1.0;
        }
        return Y_encoded;
    }

    // ==================
    // ===== WHERE ======
    // ==================

    /**
     * @brief Selects between two values element-wise.
     * @param condition A matrix where a non-zero element selects @p ifTrue.
     * @param ifTrue    The value used where the condition is non-zero.
     * @param ifFalse   The value used where the condition is zero.
     */
    static Matrix where(const Matrix &condition, double ifTrue, double ifFalse)
    {
        Matrix result(condition.rows(), condition.cols());

#pragma omp parallel for simd if (condition.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < condition.size(); i++)
        {
            result[i] = (condition[i] != 0.0) ? ifTrue : ifFalse;
        }
        return result;
    }

    // ===================================
    // ===== OUTER PRODUCT ===============
    // ===================================

    /**
     * @brief Computes the outer product `A * B^T` of two flat matrices.
     * @param A A matrix with a single element per row.
     * @param B A matrix with a single element per row.
     * @return The outer product of shape (A.size(), B.size()).
     * @throws std::invalid_argument if either operand is empty.
     */
    static Matrix outer(const Matrix &A, const Matrix &B)
    {
        size_t out_rows = A.size();
        size_t out_cols = B.size();

        if (out_rows == 0 || out_cols == 0)
        {
            throw std::invalid_argument("Cannot compute outer product of empty matrices.");
        }

        Matrix result(out_rows, out_cols);

#pragma omp parallel for collapse(2) if (out_rows * out_cols > OMP_THRESHOLD)
        for (size_t i = 0; i < out_rows; i++)
        {
            for (size_t j = 0; j < out_cols; j++)
            {
                result(i, j) = A[i] * B[j];
            }
        }
        return result;
    }

    // =============================
    // ===== UNIQUE_COUNT ==========
    // =============================

    /**
     * @brief Counts the distinct values in a matrix.
     * @param nc_arr The matrix to inspect.
     */
    static size_t unique_count(const Matrix &nc_arr)
    {
        std::unordered_set<double> unique_vals(nc_arr.begin(), nc_arr.end());
        return unique_vals.size();
    }
};

} // namespace cpptorch
