#pragma once

#include "exceptions.h"
#include "matrix.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <tuple>


class NumCpp
{
public:
    // =============================
    // ===== MATRIX ASSIGNMENT =====
    // =============================

    static Matrix zeroes(std::tuple<size_t, size_t> shape)
    {
        return Matrix(std::get<0>(shape), std::get<1>(shape));
    }

    static Matrix ones(std::tuple<int, int> shape)
    {
        auto [row, col] = shape;
        return Matrix(row, col) + 1.0;
    }

    // ========================================
    // ===== RANDOM: NORMAL DISTRIBUTION ======
    // ========================================

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

    // ==============================
    // ===== LINSPACE MATRIX ========
    // ==============================

    static Matrix linspace(double start, double stop, size_t num)
    {
        if (num == 0)
        {
            return {}; // returns an empty Matrix(0,0) obj
        }
        if (num == 1)
        {
            return Matrix(1, 1, {start});
        }

        // Creates a column vector
        Matrix result(num, 1);
        double step = (stop - start) / static_cast<double>(num - 1);
        for (size_t i = 0; i < num; i++)
        {
            result[i] = start + (i * step);
        }
        return result;
    }

    // =============================
    // ===== MATRIX OPERATIONS =====
    // =============================

    // =============================
    // ===== DOT PRODUCT ===========
    // =============================

    static Matrix dot(const Matrix &A, const Matrix &B)
    {
        if (A.cols() != B.rows())
        {
            throw ShapeInvalidForOperation("dot product", A.shape(), B.shape());
        }

        size_t rowSize = A.rows();
        size_t colSize = B.cols();
        size_t innerSize = A.cols();

        // result constructor auto-initializes all elements to 0.0
        Matrix result(rowSize, colSize);

        // Optimized i-k-j cache-friendly loop sequencing
        for (size_t i = 0; i < rowSize; i++)
        {
            size_t offset_A = i * innerSize;
            size_t offset_result = i * colSize;

            for (size_t k = 0; k < innerSize; k++)
            {
                double a_ik = A[offset_A + k];
                size_t offset_B = k * colSize;

                for (size_t j = 0; j < colSize; j++)
                {
                    // Sequential row-major memory access natively avoids cache misses
                    result[offset_result + j] += a_ik * B[offset_B + j];
                }
            }
        }

        return result;
    }

    // =============================
    // ===== MATRIX SUMMATION ======
    // =============================

    static double sum(const Matrix &A)
    {
        double total = 0;
        // Leveraging the flat 1D iterators for max efficiency
        for (double val : A)
        {
            total += val;
        }
        return total;
    }

    static Matrix sum(const Matrix &A, int axis)
    {
        if (axis == 0)
        {
            Matrix result(1, A.cols());
            for (size_t i = 0; i < A.cols(); i++)
            {
                double total = 0;
                for (size_t j = 0; j < A.rows(); j++)
                {
                    total += A(j, i);
                }
                result(0, i) = total;
            }
            return result;
        }

        if (axis == 1)
        {
            Matrix result(A.rows(), 1);
            for (size_t i = 0; i < A.rows(); i++)
            {
                double total = 0;
                for (size_t j = 0; j < A.cols(); j++)
                {
                    total += A(i, j);
                }
                result(i, 0) = total;
            }
            return result;
        }

        throw std::invalid_argument("axis must be 0 or 1");
    }

    // =============================
    // ===== MATRIX MAXIMUM ========
    // =============================

    static Matrix maximum(double operand, const Matrix &nc_arr)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            result[i] = std::max(operand, nc_arr[i]);
        }
        return result;
    }

    static double max(const Matrix &nc_arr)
    {
        if (nc_arr.size() == 0)
        {
            throw std::out_of_range("Matrix is empty");
        }

        double maxValue = nc_arr[0];
        for (size_t i = 1; i < nc_arr.size(); i++)
        {
            maxValue = std::max(maxValue, nc_arr[i]);
        }
        return maxValue;
    }

    static Matrix max(const Matrix &nc_arr, int axis)
    {
        if (axis == 0)
        {
            Matrix result(1, nc_arr.cols());
            for (size_t i = 0; i < nc_arr.cols(); i++)
            {
                double maxValue = nc_arr(0, i);
                for (size_t j = 1; j < nc_arr.rows(); j++)
                {
                    maxValue = std::max(maxValue, nc_arr(j, i));
                }
                result(0, i) = maxValue;
            }
            return result;
        }

        if (axis == 1)
        {
            Matrix result(nc_arr.rows(), 1);
            for (size_t i = 0; i < nc_arr.rows(); i++)
            {
                double maxValue = nc_arr(i, 0);
                for (size_t j = 1; j < nc_arr.cols(); j++)
                {
                    maxValue = std::max(maxValue, nc_arr(i, j));
                }
                result(i, 0) = maxValue;
            }
            return result;
        }

        throw std::invalid_argument("axis must be 0 or 1");
    }

    static std::tuple<size_t, size_t> argmax(const Matrix &nc_arr)
    {
        std::tuple<size_t, size_t> maxIdx = {0, 0};
        if (nc_arr.size() == 0)
        {
            return maxIdx;
        }

        double maxValue = nc_arr(0, 0);

        for (size_t i = 0; i < nc_arr.rows(); i++)
        {
            for (size_t j = 0; j < nc_arr.cols(); j++)
            {
                if (nc_arr(i, j) > maxValue)
                {
                    maxValue = nc_arr(i, j);
                    maxIdx = {i, j};
                }
            }
        }
        return maxIdx;
    }

    static Matrix argmax(const Matrix &nc_arr, int axis)
    {
        if (axis == 0)
        {
            Matrix result(1, nc_arr.cols());
            for (size_t i = 0; i < nc_arr.cols(); i++)
            {
                size_t maxIdx = 0;
                double maxValue = nc_arr(0, i);
                for (size_t j = 1; j < nc_arr.rows(); j++)
                {
                    if (nc_arr(j, i) > maxValue)
                    {
                        maxValue = nc_arr(j, i);
                        maxIdx = j;
                    }
                }
                result(0, i) = static_cast<double>(maxIdx);
            }
            return result;
        }

        if (axis == 1)
        {
            Matrix result(nc_arr.rows(), 1);
            for (size_t i = 0; i < nc_arr.rows(); i++)
            {
                size_t maxIdx = 0;
                double maxValue = nc_arr(i, 0);
                for (size_t j = 1; j < nc_arr.cols(); j++)
                {
                    if (nc_arr(i, j) > maxValue)
                    {
                        maxValue = nc_arr(i, j);
                        maxIdx = j;
                    }
                }
                result(i, 0) = static_cast<double>(maxIdx);
            }
            return result;
        }

        throw std::invalid_argument("axis must be 0 or 1");
    }

    // =============================================
    // ===== MATRIX EXPONENTIATION & LOGARITHM =====
    // =============================================

    static Matrix exp(const Matrix &nc_arr)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            result[i] = std::exp(nc_arr[i]);
        }
        return result;
    }

    static Matrix log(const Matrix &nc_arr)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            result[i] = std::log(nc_arr[i]);
        }
        return result;
    }

    // ===========================
    // ===== MATRIX MEAN AVG =====
    // ===========================

    static double mean(const Matrix &nc_arr)
    {
        if (nc_arr.size() == 0)
        {
            return 0.0;
        }
        double meanVal = 0;

        for (double val : nc_arr)
        {
            meanVal += val;
        }

        return meanVal / static_cast<int>(nc_arr.size());
    }

    // ===========================
    // ===== MATRIX CLIPPING =====
    // ===========================

    static Matrix clip(const Matrix &nc_arr, double min_val, double max_val)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());

        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            result[i] = std::clamp(nc_arr[i], min_val, max_val);
        }

        return result;
    }

    // ============================
    // ===== ONE HOT ENCODING =====
    // ============================

    static Matrix one_hot(const Matrix &y, int n_classes)
    {
        // Matrix defaults to 0.0, so we only need to assign the hot indices
        Matrix Y_encoded(y.size(), n_classes);

        for (size_t i = 0; i < y.size(); i++)
        {
            Y_encoded(i, static_cast<size_t>(y[i])) = 1.0;
        }

        return Y_encoded;
    }

    // ==================
    // ===== WHERE ======
    // ==================

    static Matrix where(const Matrix &condition, double ifTrue, double ifFalse)
    {
        Matrix result(condition.rows(), condition.cols());
        for (size_t i = 0; i < condition.size(); i++)
        {
            result[i] = (condition[i] != 0.0) ? ifTrue : ifFalse;
        }

        return result;
    }

    // ===================================
    // ===== OUTER PRODUCT ===============
    // ===================================

    // (all elements muliplication: A_ij * B_entire)
    static Matrix outer(const Matrix &A, const Matrix &B)
    {
        size_t out_rows = A.size();
        size_t out_cols = B.size();

        if (out_rows == 0 || out_cols == 0)
        {
            throw std::invalid_argument("Cannot compute outer product of empty matrices.");
        }

        Matrix result(out_rows, out_cols);

        for (size_t i = 0; i < out_rows; i++)
        {
            for (size_t j = 0; j < out_cols; j++)
            {
                // A[i] and B[j] automatically use the flat 1D indexing
                result(i, j) = A[i] * B[j];
            }
        }

        return result;
    }

    // ========================================
    // ===== MATRIX SQUARE ROOT ===============
    // ========================================

    static Matrix sqrt(const Matrix &nc_arr)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
        for (int i = 0; i < result.size(); i++)
        {
            result[i] = std::sqrt(nc_arr[i]);
        }
        return result;
    }

    // =====================================
    // ===== MATRIX ABSOLUTE ===============
    // =====================================

    static Matrix abs(const Matrix &nc_arr)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
        for (int i = 0; i < result.size(); i++)
        {
            result[i] = std::abs(nc_arr[i]);
        }
        return result;
    }

    // =============================
    // ===== TRIGONOMETRY ==========
    // =============================

    static Matrix sin(const Matrix &nc_arr)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            result[i] = std::sin(nc_arr[i]);
        }
        return result;
    }

    static Matrix cos(const Matrix &nc_arr)
    {
        Matrix result(nc_arr.rows(), nc_arr.cols());
        for (size_t i = 0; i < nc_arr.size(); i++)
        {
            result[i] = std::cos(nc_arr[i]);
        }
        return result;
    }
};