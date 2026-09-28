#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

class Matrix
{
private:
    size_t m_rows;
    size_t m_cols;
    std::vector<double> m_data; // Flat, fixed-size data

public:
    // ==========================================
    // ===== CONSTRUCTORS & INITIALIZATIONS =====
    // ==========================================

    // 1. Standard constructor (fills with 0.0)
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0)
    {
    }

    // 2. Default constructor (creates an empty 0x0 matrix)
    Matrix() : m_rows(0), m_cols(0)
    {
    }

    // 3. Initializer list constructor: Matrix m(2, 2, {1, 2, 3, 4});
    Matrix(size_t rows, size_t cols, std::initializer_list<double> list) : m_rows(rows), m_cols(cols), m_data(list)
    {
        if (m_data.size() != rows * cols)
        {
            throw std::invalid_argument("Initializer list size does not match matrix dimensions.");
        }
    }

    // 4. Constructor from an existing flat std::vector
    Matrix(size_t rows, size_t cols, const std::vector<double> &flat_data)
        : m_rows(rows), m_cols(cols), m_data(flat_data)
    {
        if (m_data.size() != rows * cols)
        {
            throw std::invalid_argument("Vector size does not match matrix dimensions.");
        }
    }

    // ===========================
    // ===== VECTOR FEATURES =====
    // ===========================

    auto begin()
    {
        return m_data.begin();
    }
    auto end()
    {
        return m_data.end();
    }
    auto begin() const
    {
        return m_data.begin();
    }
    auto end() const
    {
        return m_data.end();
    }

    // 1D Indexing
    double &operator[](size_t i)
    {
        return m_data[i];
    }
    const double &operator[](size_t i) const
    {
        return m_data[i];
    }

    // ===========================
    // ===== MATRIX FEATURES =====
    // ===========================

    // 2D Indexing: matrix(row, col)
    double &operator()(size_t r, size_t c)
    {
        if (r >= m_rows || c >= m_cols)
        {
            throw std::out_of_range("Index out of bounds");
        }
        return m_data[(r * m_cols) + c];
    }

    const double &operator()(size_t r, size_t c) const
    {
        if (r >= m_rows || c >= m_cols)
        {
            throw std::out_of_range("Index out of bounds");
        }
        return m_data[(r * m_cols) + c];
    }

    [[nodiscard]] std::tuple<size_t, size_t> shape() const
    {
        return {m_rows, m_cols};
    }
    [[nodiscard]] size_t rows() const
    {
        return m_rows;
    }
    [[nodiscard]] size_t cols() const
    {
        return m_cols;
    }
    [[nodiscard]] size_t size() const
    {
        return m_data.size();
    }

    [[nodiscard]] const std::vector<double> &getFlatData() const
    {
        return m_data;
    }

    // Transpose generates and returns a completely new Matrix
    [[nodiscard]] Matrix T() const
    {
        Matrix transposed(m_cols, m_rows);
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                transposed(j, i) = (*this)(i, j);
            }
        }
        return transposed;
    }

    void display(int truncateX = -1, int truncateY = -1) const
    {
        size_t trX = (truncateX == -1) ? m_rows : (size_t)truncateX;
        size_t trY = (truncateY == -1) ? m_cols : (size_t)truncateY;

        std::cout << "[\n";
        for (size_t i = 0; i < trX; ++i)
        {
            std::cout << "  [";
            for (size_t j = 0; j < trY; ++j)
            {
                std::cout << (*this)(i, j);
                std::cout << ((j == trY - 1) ? "" : ", ");
            }
            std::cout << ((i == trX - 1) ? "]" : "],\n");
        }
        std::cout << "\n]\n";
    }

    // =======================================
    // ============ BROADCASTING  ============
    // =======================================

    static std::tuple<Matrix, Matrix> broadcast(const Matrix &A, const Matrix &B)
    {
        if (A.rows() == B.rows() && A.cols() == B.cols())
        {
            return {A, B};
        }

        if ((A.rows() != B.rows() && A.rows() != 1 && B.rows() != 1) ||
            (A.cols() != B.cols() && A.cols() != 1 && B.cols() != 1))
        {
            throw std::invalid_argument("Shapes invalid for broadcasting.");
        }

        size_t out_rows = std::max(A.rows(), B.rows());
        size_t out_cols = std::max(A.cols(), B.cols());

        auto expand = [&](const Matrix &X) -> Matrix {
            if (X.rows() == out_rows && X.cols() == out_cols)
            {
                return X;
            }

            // Generate a NEW matrix for the broadcasted shape
            Matrix result(out_rows, out_cols);
            for (size_t i = 0; i < out_rows; ++i)
            {
                for (size_t j = 0; j < out_cols; ++j)
                {
                    size_t srcRow = (X.rows() == 1) ? 0 : i;
                    size_t srcCol = (X.cols() == 1) ? 0 : j;
                    result(i, j) = X(srcRow, srcCol);
                }
            }
            return result;
        };

        return {expand(A), expand(B)};
    }

    // ==========================================
    // ===== ARITHMETIC: MATRIX-BY-MATRIX =======
    // ==========================================

    Matrix operator+(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = A[i] + B[i];
        }
        return result;
    }

    Matrix operator-(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = A[i] - B[i];
        }
        return result;
    }

    Matrix operator*(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = A[i] * B[i];
        }
        return result;
    }

    Matrix operator/(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = A[i] / B[i];
        }
        return result;
    }

    Matrix operator%(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = std::fmod(A[i], B[i]);
        }
        return result;
    }

    // ==========================================
    // ===== ARITHMETIC: MATRIX-BY-SCALAR =======
    // ==========================================

    Matrix operator+(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = m_data[i] + scalar;
        }
        return result;
    }

    Matrix operator-(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = m_data[i] - scalar;
        }
        return result;
    }

    Matrix operator*(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = m_data[i] * scalar;
        }
        return result;
    }

    Matrix operator/(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = m_data[i] / scalar;
        }
        return result;
    }

    Matrix operator%(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = std::fmod(m_data[i], scalar);
        }
        return result;
    }

    // ==========================================
    // ===== ARITHMETIC: SCALAR-BY-MATRIX =======
    // ==========================================

    friend Matrix operator+(double scalar, const Matrix &mat)
    {
        return mat + scalar;
    } // Commutative
    friend Matrix operator*(double scalar, const Matrix &mat)
    {
        return mat * scalar;
    } // Commutative

    friend Matrix operator-(double scalar, const Matrix &mat)
    {
        Matrix result(mat.rows(), mat.cols());
        for (size_t i = 0; i < mat.size(); i++)
        {
            result[i] = scalar - mat[i];
        }
        return result;
    }

    friend Matrix operator/(double scalar, const Matrix &mat)
    {
        Matrix result(mat.rows(), mat.cols());
        for (size_t i = 0; i < mat.size(); i++)
        {
            result[i] = scalar / mat[i];
        }
        return result;
    }

    friend Matrix operator%(double scalar, const Matrix &mat)
    {
        Matrix result(mat.rows(), mat.cols());
        for (size_t i = 0; i < mat.size(); i++)
        {
            result[i] = std::fmod(scalar, mat[i]);
        }
        return result;
    }

    // ==========================================
    // ===== RELATIONAL: MATRIX-BY-MATRIX =======
    // ==========================================

    Matrix operator==(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = (A[i] == B[i]) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator!=(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = (A[i] != B[i]) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator<(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = (A[i] < B[i]) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator>(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = (A[i] > B[i]) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator<=(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = (A[i] <= B[i]) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator>=(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = (A[i] >= B[i]) ? 1.0 : 0.0;
        }
        return result;
    }

    // ==========================================
    // ===== RELATIONAL: MATRIX-BY-SCALAR =======
    // ==========================================

    Matrix operator==(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = (m_data[i] == scalar) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator!=(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = (m_data[i] != scalar) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator<(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = (m_data[i] < scalar) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator>(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = (m_data[i] > scalar) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator<=(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = (m_data[i] <= scalar) ? 1.0 : 0.0;
        }
        return result;
    }

    Matrix operator>=(double scalar) const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); i++)
        {
            result[i] = (m_data[i] >= scalar) ? 1.0 : 0.0;
        }
        return result;
    }

    // ==========================================
    // ===== RELATIONAL: SCALAR-BY-MATRIX =======
    // ==========================================

    friend Matrix operator==(double scalar, const Matrix &mat)
    {
        return mat == scalar;
    }
    friend Matrix operator!=(double scalar, const Matrix &mat)
    {
        return mat != scalar;
    }

    // Note: scalar < matrix is equivalent to matrix > scalar
    friend Matrix operator<(double scalar, const Matrix &mat)
    {
        return mat > scalar;
    }
    friend Matrix operator>(double scalar, const Matrix &mat)
    {
        return mat < scalar;
    }
    friend Matrix operator<=(double scalar, const Matrix &mat)
    {
        return mat >= scalar;
    }
    friend Matrix operator>=(double scalar, const Matrix &mat)
    {
        return mat <= scalar;
    }

    // ==========================================
    // ===== ASSIGNMENT: MATRIX-BY-MATRIX =======
    // ==========================================

    Matrix &operator+=(const Matrix &matB)
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        if (A.rows() != m_rows || A.cols() != m_cols)
        {
            throw std::invalid_argument("In-place assignment cannot change matrix dimensions.");
        }
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            m_data[i] += B[i];
        }
        return *this;
    }

    Matrix &operator-=(const Matrix &matB)
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        if (A.rows() != m_rows || A.cols() != m_cols)
        {
            throw std::invalid_argument("In-place assignment cannot change matrix dimensions.");
        }
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            m_data[i] -= B[i];
        }
        return *this;
    }

    Matrix &operator*=(const Matrix &matB)
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        if (A.rows() != m_rows || A.cols() != m_cols)
        {
            throw std::invalid_argument("In-place assignment cannot change matrix dimensions.");
        }
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            m_data[i] *= B[i];
        }
        return *this;
    }

    Matrix &operator/=(const Matrix &matB)
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        if (A.rows() != m_rows || A.cols() != m_cols)
        {
            throw std::invalid_argument("In-place assignment cannot change matrix dimensions.");
        }
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            m_data[i] /= B[i];
        }
        return *this;
    }

    Matrix &operator%=(const Matrix &matB)
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        if (A.rows() != m_rows || A.cols() != m_cols)
        {
            throw std::invalid_argument("In-place assignment cannot change matrix dimensions.");
        }
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            m_data[i] = std::fmod(m_data[i], B[i]);
        }
        return *this;
    }

    // ==========================================
    // ===== ASSIGNMENT: MATRIX-BY-SCALAR =======
    // ==========================================

    Matrix &operator+=(double scalar)
    {
        for (double &i : m_data)
        {
            i += scalar;
        }
        return *this;
    }

    Matrix &operator-=(double scalar)
    {
        for (double &i : m_data)
        {
            i -= scalar;
        }
        return *this;
    }

    Matrix &operator*=(double scalar)
    {
        for (double &i : m_data)
        {
            i *= scalar;
        }
        return *this;
    }

    Matrix &operator/=(double scalar)
    {
        for (double &i : m_data)
        {
            i /= scalar;
        }
        return *this;
    }

    Matrix &operator%=(double scalar)
    {
        for (double &i : m_data)
        {
            i = std::fmod(i, scalar);
        }
        return *this;
    }

    // ==========================================
    // ===== BITWISE: MATRIX-BY-MATRIX ==========
    // ==========================================

    // use XOR as power operator
    Matrix operator^(const Matrix &matB) const
    {
        auto [A, B] = Matrix::broadcast(*this, matB);
        Matrix result(A.rows(), A.cols());
        for (int i = 0; i < result.size(); i++)
        {
            result[i] = std::pow(A[i], B[i]);
        }
        return result;
    }

    // ==========================================
    // ===== BITWISE: MATRIX-BY-SCALAR ==========
    // ==========================================

    // use XOR as power operator
    Matrix operator^(double exp)
    {
        Matrix result(m_rows, m_cols);
        for (int i = 0; i < result.size(); i++)
        {
            result[i] = std::pow(m_data[i], exp);
        }
        return result;
    }

    // ==========================================
    // ===== BITWISE: SCALAR-BY-MATRIX ==========
    // ==========================================

    // use XOR as power operator
    friend Matrix operator^(double x, const Matrix &matA)
    {
        Matrix result(matA.rows(), matA.cols());
        for (int i = 0; i < result.size(); i++)
        {
            result[i] = std::pow(x, matA[i]);
        }
        return result;
    }
};