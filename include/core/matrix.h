#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <omp.h> // Added OpenMP support
#include <stdexcept>
#include <tuple>
#include <vector>

class MatrixView;

class Matrix
{
private:
    size_t m_rows;
    size_t m_cols;
    std::vector<double> m_data; // Flat, fixed-size data

    // Minimum matrix size to trigger OpenMP multithreading (prevents thread-spawn overhead on small layers/biases)
    static constexpr size_t OMP_THRESHOLD = 8192;

    // ==========================================
    // ===== INTERNAL BROADCASTING HELPERS ======
    // ==========================================

    template <typename Op>
    static Matrix broadcast_op(const Matrix &A, const Matrix &B, Op op)
    {
        if ((A.m_rows != B.m_rows && A.m_rows != 1 && B.m_rows != 1) ||
            (A.m_cols != B.m_cols && A.m_cols != 1 && B.m_cols != 1))
        {
            throw std::invalid_argument("Shapes invalid for broadcasting.");
        }

        size_t out_rows = std::max(A.m_rows, B.m_rows);
        size_t out_cols = std::max(A.m_cols, B.m_cols);

        Matrix result(out_rows, out_cols);

        size_t step_rA = (A.m_rows == 1) ? 0 : 1;
        size_t step_cA = (A.m_cols == 1) ? 0 : 1;
        size_t step_rB = (B.m_rows == 1) ? 0 : 1;
        size_t step_cB = (B.m_cols == 1) ? 0 : 1;

#pragma omp parallel for collapse(2) if (out_rows * out_cols > OMP_THRESHOLD)
        for (size_t i = 0; i < out_rows; ++i)
        {
            for (size_t j = 0; j < out_cols; ++j)
            {
                size_t rA = i * step_rA;
                size_t cA = j * step_cA;
                size_t rB = i * step_rB;
                size_t cB = j * step_cB;

                result.m_data[i * out_cols + j] = op(A.m_data[rA * A.m_cols + cA], B.m_data[rB * B.m_cols + cB]);
            }
        }
        return result;
    }

    template <typename Op>
    Matrix &broadcast_op_assign(const Matrix &B, Op op)
    {
        if ((m_rows != B.m_rows && B.m_rows != 1) || (m_cols != B.m_cols && B.m_cols != 1))
        {
            throw std::invalid_argument("In-place assignment cannot change matrix dimensions.");
        }

        size_t step_rB = (B.m_rows == 1) ? 0 : 1;
        size_t step_cB = (B.m_cols == 1) ? 0 : 1;

#pragma omp parallel for collapse(2) if (m_rows * m_cols > OMP_THRESHOLD)
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                size_t rB = i * step_rB;
                size_t cB = j * step_cB;

                size_t idx = i * m_cols + j;
                m_data[idx] = op(m_data[idx], B.m_data[rB * B.m_cols + cB]);
            }
        }
        return *this;
    }

    // New Helper: Clean replacement for macros to handle optimized scalar operations
    template <typename Op>
    Matrix scalar_op(double scalar, Op op) const
    {
        Matrix result(m_rows, m_cols);
#pragma omp parallel for simd if (m_data.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = op(m_data[i], scalar);
        }
        return result;
    }

    template <typename Op>
    Matrix &scalar_assign_op(double scalar, Op op)
    {
#pragma omp parallel for simd if (m_data.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            m_data[i] = op(m_data[i], scalar);
        }
        return *this;
    }

public:
    // Guarantees zero-allocation transfers during computation chains
    Matrix(Matrix &&) noexcept = default;
    Matrix &operator=(Matrix &&) noexcept = default;
    Matrix(const Matrix &) = default;
    Matrix &operator=(const Matrix &) = default;

    // ==========================================
    // ===== CONSTRUCTORS & INITIALIZATIONS =====
    // ==========================================

    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0)
    {
    }
    Matrix() : m_rows(0), m_cols(0)
    {
    }

    Matrix(size_t rows, size_t cols, std::initializer_list<double> list) : m_rows(rows), m_cols(cols), m_data(list)
    {
        if (m_data.size() != rows * cols)
            throw std::invalid_argument("Initializer list size does not match matrix dimensions.");
    }

    Matrix(size_t rows, size_t cols, const std::vector<double> &flat_data)
        : m_rows(rows), m_cols(cols), m_data(flat_data)
    {
        if (m_data.size() != rows * cols)
            throw std::invalid_argument("Vector size does not match matrix dimensions.");
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

    double &operator()(size_t r, size_t c)
    {
        if (r >= m_rows || c >= m_cols)
            throw std::out_of_range("Index out of bounds");
        return m_data[r * m_cols + c];
    }

    const double &operator()(size_t r, size_t c) const
    {
        if (r >= m_rows || c >= m_cols)
            throw std::out_of_range("Index out of bounds");
        return m_data[r * m_cols + c];
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
    [[nodiscard]] const double *data() const
    {
        return m_data.data();
    }
    [[nodiscard]] MatrixView rows_view(size_t first_row, size_t row_count) const;
    void copy_from(const MatrixView &source);
    void resize(size_t rows, size_t cols)
    {
        m_rows = rows;
        m_cols = cols;
        m_data.resize(rows * cols);
    }
    void copy_rows_from(const Matrix &source, size_t first_row, size_t row_count)
    {
        if (first_row > source.m_rows || row_count > source.m_rows - first_row)
            throw std::out_of_range("Row range is outside the source matrix.");
        resize(row_count, source.m_cols);
        std::copy_n(source.m_data.begin() + first_row * source.m_cols, row_count * source.m_cols, m_data.begin());
    }
    [[nodiscard]] const std::vector<double> &getFlatData() const
    {
        return m_data;
    }

    [[nodiscard]] Matrix T() const
    {
        Matrix transposed(m_cols, m_rows);
#pragma omp parallel for collapse(2) if (m_rows * m_cols > OMP_THRESHOLD)
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                transposed.m_data[j * m_rows + i] = m_data[i * m_cols + j];
            }
        }
        return transposed;
    }

    [[nodiscard]] Matrix sq() const
    {
        Matrix result(m_rows, m_cols);
#pragma omp parallel for simd if (m_data.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] * m_data[i];
        }
        return result;
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

    // ==========================================
    // ===== ARITHMETIC: MATRIX-BY-MATRIX =======
    // ==========================================

    Matrix operator+(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return a + b; });
    }
    Matrix operator-(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return a - b; });
    }
    Matrix operator*(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return a * b; });
    }
    Matrix operator/(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return a / b; });
    }
    Matrix operator%(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return std::fmod(a, b); });
    }

    // ==========================================
    // ===== ARITHMETIC: MATRIX-BY-SCALAR =======
    // ==========================================

    Matrix operator+(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return a + b; });
    }
    Matrix operator-(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return a - b; });
    }
    Matrix operator*(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return a * b; });
    }
    Matrix operator/(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return a / b; });
    }
    Matrix operator%(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return std::fmod(a, b); });
    }

    // ==========================================
    // ===== ARITHMETIC: SCALAR-BY-MATRIX =======
    // ==========================================

    friend Matrix operator+(double scalar, const Matrix &mat)
    {
        return mat + scalar;
    }
    friend Matrix operator*(double scalar, const Matrix &mat)
    {
        return mat * scalar;
    }

    friend Matrix operator-(double scalar, const Matrix &mat)
    {
        Matrix result(mat.rows(), mat.cols());
#pragma omp parallel for simd if (mat.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < mat.size(); i++)
        {
            result.m_data[i] = scalar - mat[i];
        }
        return result;
    }

    friend Matrix operator/(double scalar, const Matrix &mat)
    {
        Matrix result(mat.rows(), mat.cols());
#pragma omp parallel for simd if (mat.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < mat.size(); i++)
        {
            result.m_data[i] = scalar / mat[i];
        }
        return result;
    }

    friend Matrix operator%(double scalar, const Matrix &mat)
    {
        Matrix result(mat.rows(), mat.cols());
#pragma omp parallel for simd if (mat.size() > OMP_THRESHOLD)
        for (size_t i = 0; i < mat.size(); i++)
        {
            result.m_data[i] = std::fmod(scalar, mat[i]);
        }
        return result;
    }

    // ==========================================
    // ===== RELATIONAL: MATRIX-BY-MATRIX =======
    // ==========================================

    Matrix operator==(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return (a == b) ? 1.0 : 0.0; });
    }
    Matrix operator!=(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return (a != b) ? 1.0 : 0.0; });
    }
    Matrix operator<(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return (a < b) ? 1.0 : 0.0; });
    }
    Matrix operator>(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return (a > b) ? 1.0 : 0.0; });
    }
    Matrix operator<=(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return (a <= b) ? 1.0 : 0.0; });
    }
    Matrix operator>=(const Matrix &matB) const
    {
        return broadcast_op(*this, matB, [](double a, double b) { return (a >= b) ? 1.0 : 0.0; });
    }

    // ==========================================
    // ===== RELATIONAL: MATRIX-BY-SCALAR =======
    // ==========================================

    Matrix operator==(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return (a == b) ? 1.0 : 0.0; });
    }
    Matrix operator!=(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return (a != b) ? 1.0 : 0.0; });
    }
    Matrix operator<(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return (a < b) ? 1.0 : 0.0; });
    }
    Matrix operator>(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return (a > b) ? 1.0 : 0.0; });
    }
    Matrix operator<=(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return (a <= b) ? 1.0 : 0.0; });
    }
    Matrix operator>=(double scalar) const
    {
        return scalar_op(scalar, [](double a, double b) { return (a >= b) ? 1.0 : 0.0; });
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
        return broadcast_op_assign(matB, [](double a, double b) { return a + b; });
    }
    Matrix &operator-=(const Matrix &matB)
    {
        return broadcast_op_assign(matB, [](double a, double b) { return a - b; });
    }
    Matrix &operator*=(const Matrix &matB)
    {
        return broadcast_op_assign(matB, [](double a, double b) { return a * b; });
    }
    Matrix &operator/=(const Matrix &matB)
    {
        return broadcast_op_assign(matB, [](double a, double b) { return a / b; });
    }
    Matrix &operator%=(const Matrix &matB)
    {
        return broadcast_op_assign(matB, [](double a, double b) { return std::fmod(a, b); });
    }

    // ==========================================
    // ===== ASSIGNMENT: MATRIX-BY-SCALAR =======
    // ==========================================

    Matrix &operator+=(double scalar)
    {
        return scalar_assign_op(scalar, [](double a, double b) { return a + b; });
    }
    Matrix &operator-=(double scalar)
    {
        return scalar_assign_op(scalar, [](double a, double b) { return a - b; });
    }
    Matrix &operator*=(double scalar)
    {
        return scalar_assign_op(scalar, [](double a, double b) { return a * b; });
    }
    Matrix &operator/=(double scalar)
    {
        return scalar_assign_op(scalar, [](double a, double b) { return a / b; });
    }
    Matrix &operator%=(double scalar)
    {
        return scalar_assign_op(scalar, [](double a, double b) { return std::fmod(a, b); });
    }
};

class MatrixView
{
    const double *m_data = nullptr;
    size_t m_rows = 0;
    size_t m_cols = 0;

public:
    MatrixView() = default;
    MatrixView(const double *data, size_t rows, size_t cols) : m_data(data), m_rows(rows), m_cols(cols)
    {
    }

    [[nodiscard]] const double *data() const { return m_data; }
    [[nodiscard]] size_t rows() const { return m_rows; }
    [[nodiscard]] size_t cols() const { return m_cols; }
    [[nodiscard]] size_t size() const { return m_rows * m_cols; }
};

inline MatrixView Matrix::rows_view(size_t first_row, size_t row_count) const
{
    if (first_row > m_rows || row_count > m_rows - first_row)
        throw std::out_of_range("Row range is outside the source matrix.");
    return MatrixView(m_data.data() + first_row * m_cols, row_count, m_cols);
}

inline void Matrix::copy_from(const MatrixView &source)
{
    resize(source.rows(), source.cols());
    std::copy_n(source.data(), source.size(), m_data.begin());
}
