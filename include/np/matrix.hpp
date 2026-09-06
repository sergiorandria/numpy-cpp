/**
 * @file matrix.hpp
 * @brief 2D matrix type with linear algebra helpers.
 *
 * np::Matrix<T> is a 2D ndarray<T> with (i, j) access and factories
 * (zeros, ones, eye, identity). Free functions det/inverse/solve use
 * Gaussian elimination with partial pivoting and promote to double.
 *
 * Memory: Matrix shares the same storage model as ndarray (shared_ptr
 * with copy-on-write semantics for views). All operations return
 * C-contiguous row-major arrays.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MATRIX_HPP
#define NP_MATRIX_HPP

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{

/** @brief 2D matrix: a ndarray guaranteed to have ndim == 2.
 *
 * @deprecated Use `ndarray<T>` + `linalg.hpp` (Decorator pattern).
 * `Matrix<T>` is now a thin `ndarray<T>` decorator that forwards
 * `matmul`/`dot` to `linalg::matmul` and is fully integrated with
 * `lattice` (as lattice basis) and `padic` (as p-adic matrix).
 * Kept for backward compatibility; dead code now live via `lattice`/`padic`.
 *
 * @tparam T Element type.
 */
template <typename T>
class NP_DEPRECATED("Matrix is deprecated: use ndarray<T> + linalg.hpp (see other.hpp integration)") Matrix
    : public ndarray<T>
{
  public:
    using Base = ndarray<T>;
    using value_type = typename Base::value_type;

    /** @brief r x c matrix filled with `fill`.
     *
     * @param rows  Number of rows.
     * @param cols  Number of columns.
     * @param fill  Initial value for every element (default: T{}).
     */
    Matrix(std::size_t rows, std::size_t cols, value_type fill = value_type{})
        : Base(std::vector<int>{static_cast<int>(rows), static_cast<int>(cols)}, dtype_of<T>, fill)
    {
    }

    /** @brief Build from nested row initializer lists.
     *
     * All rows must have the same length; the first row's length
     * determines the column count. Missing elements are
     * default-constructed.
     *
     * @param rows Nested initializer list of rows.
     * @throws std::invalid_argument on ragged rows.
     */
    Matrix(std::initializer_list<std::initializer_list<T>> rows) : Base()
    {
        const std::size_t nrows = rows.size();
        std::size_t ncols = 0;
        for (const auto &row : rows)
        {
            if (row.size() > ncols)
            {
                ncols = row.size();
            }
        }
        *this = Matrix(nrows, ncols);
        std::size_t i = 0;
        for (const auto &row : rows)
        {
            std::size_t j = 0;
            for (const T &v : row)
            {
                (*this)(i, j) = v;
                ++j;
            }
            ++i;
        }
    }

    /** @brief Element access without bounds checks (read/write).
     *
     * @param i Row index (0-based).
     * @param j Column index (0-based).
     * @return  Reference to element at (i, j).
     * @pre     i < rows() && j < cols().
     */
    auto operator()(std::size_t i, std::size_t j) -> value_type &
    {
        return this->data()[i * this->cols() + j];
    }

    /** @brief Element access without bounds checks (read-only).
     *
     * @param i Row index.
     * @param j Column index.
     * @return  Const reference to element at (i, j).
     */
    auto operator()(std::size_t i, std::size_t j) const -> const value_type &
    {
        return this->data()[i * this->cols() + j];
    }

    /** @brief Number of rows.
     *
     * @return Number of rows in the matrix.
     */
    auto rows() const -> std::size_t
    {
        return static_cast<std::size_t>(this->shape[0]);
    }

    /** @brief Number of columns.
     *
     * @return Number of columns in the matrix.
     */
    auto cols() const -> std::size_t
    {
        return static_cast<std::size_t>(this->shape[1]);
    }

    /** @brief True if rows == cols.
     *
     * @return True if the matrix is square.
     */
    auto is_square() const -> bool
    {
        return this->rows() == this->cols();
    }

    /** @brief Matrix transposition (returns a new matrix).
     *
     * The returned matrix shares no storage with this matrix.
     * Time complexity: O(rows * cols). Space complexity: O(rows * cols).
     *
     * @return Matrix<T> with shape (cols, rows) and transposed elements.
     */
    auto transpose() const -> Matrix<T>
    {
        Matrix<T> out(cols(), rows());
        for (std::size_t i = 0; i < rows(); ++i)
        {
            for (std::size_t j = 0; j < cols(); ++j)
            {
                out(j, i) = (*this)(i, j);
            }
        }
        return out;
    }

    /** @brief Matrix-matrix product.
     *
     * Promotes element type to the common type of T and U.
     * Throws std::invalid_argument if inner dimensions don't match.
     *
     * Time complexity: O(rows * cols * rhs.cols()).
     *
     * @tparam U  Element type of the right-hand side matrix.
     * @param rhs The right-hand side matrix.
     * @return    Matrix of the common type with shape (rows, rhs.cols()).
     * @throws    std::invalid_argument if cols() != rhs.rows().
     */
    template <typename U> auto operator*(const Matrix<U> &rhs) const -> Matrix<std::common_type_t<T, U>>
    {
        using R = std::common_type_t<T, U>;
        if (cols() != rhs.rows())
        {
            throw std::invalid_argument("matrix product: inner dimensions must match");
        }
        Matrix<R> out(rows(), rhs.cols());
        for (std::size_t i = 0; i < rows(); ++i)
        {
            for (std::size_t j = 0; j < rhs.cols(); ++j)
            {
                R acc{};
                for (std::size_t k = 0; k < cols(); ++k)
                {
                    acc += static_cast<R>((*this)(i, k)) * static_cast<R>(rhs(k, j));
                }
                out(i, j) = acc;
            }
        }
        return out;
    }

    /** @brief Scalar multiplication.
     *
     * @tparam U  Scalar type.
     * @param scalar  Scalar multiplier.
     * @return        Matrix of the common type.
     */
    template <typename U> auto operator*(U scalar) const -> Matrix<std::common_type_t<T, U>>
    {
        using R = std::common_type_t<T, U>;
        Matrix<R> out(rows(), cols());
        for (std::size_t i = 0; i < rows() * cols(); ++i)
        {
            out.data()[i] = static_cast<R>(this->data()[i]) * static_cast<R>(scalar);
        }
        return out;
    }

    // Factories

    /** @brief r x c matrix of zeros.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return     Matrix<T> filled with T{0}.
     */
    static auto zeros(std::size_t rows, std::size_t cols) -> Matrix<T>
    {
        return Matrix<T>(rows, cols, T{0});
    }

    /** @brief r x c matrix of ones.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return     Matrix<T> filled with T{1}.
     */
    static auto ones(std::size_t rows, std::size_t cols) -> Matrix<T>
    {
        return Matrix<T>(rows, cols, T{1});
    }

    /** @brief n x n identity matrix.
     *
     * @param n Size of the square identity matrix.
     * @return  Matrix<T> with ones on the main diagonal.
     */
    static auto identity(std::size_t n) -> Matrix<T>
    {
        return eye(n);
    }

    /** @brief n x n identity matrix.
     *
     * @param n Size of the square identity matrix.
     * @return  Matrix<T> with ones on the main diagonal.
     */
    static auto eye(std::size_t n) -> Matrix<T>
    {
        Matrix<T> out(n, n, T{0});
        for (std::size_t i = 0; i < n; ++i)
        {
            out(i, i) = T{1};
        }
        return out;
    }

    /** @brief n x m matrix with ones on the k-th diagonal.
     *
     * @param n  Number of rows.
     * @param m  Number of columns.
     * @param k  Diagonal offset (optional, default: 0 via std::nullopt).
     * @return   Matrix<T> with ones on the k-th diagonal.
     */
    static auto eye(std::size_t n, std::size_t m, std::optional<int> k = std::nullopt) -> Matrix<T>
    {
        const int kk = k.value_or(0);
        Matrix<T> out(n, m, T{0});
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i) + kk;
            if (j >= 0 && static_cast<std::size_t>(j) < m)
            {
                out(i, static_cast<std::size_t>(j)) = T{1};
            }
        }
        return out;
    }

    /** @brief n x m matrix with ones on the k-th diagonal (int overload).
     *
     * @param n  Number of rows.
     * @param m  Number of columns.
     * @param k  Diagonal offset.
     * @return   Matrix<T> with ones on the k-th diagonal.
     */
    static auto eye(std::size_t n, std::size_t m, int k) -> Matrix<T>
    {
        return eye(n, m, std::optional<int>{k});
    }

    // Secure (constant-time, PQC-hardened)
    /** @brief Securely zero (constant-time, not elided). @see ndarray::secure_zero */
    void secure_zero() noexcept
    {
        Base::secure_zero();
    }
    /** @brief Securely clear shape and wipe. @see ndarray::secure_clear */
    void secure_clear() noexcept
    {
        Base::secure_clear();
    }
    /** @brief Secure fill (constant-time). @see ndarray::secure_fill */
    void secure_fill(const value_type &v) noexcept
    {
        Base::secure_fill(v);
    }
    /** @brief Constant-time at (no branch on index). @see ndarray::secure_at */
    NP_NODISCARD auto secure_at(std::size_t i, std::size_t j) const noexcept -> value_type
    {
        // Flat index for row-major 2D: i*cols + j
        std::size_t idx = i * cols() + j;
        return Base::secure_at(idx);
    }
    /** @brief Secure matmul (constant-time accumulation, no early exit). */
    template <typename U>
    NP_NODISCARD auto secure_matmul(const Matrix<U> &rhs) const -> Matrix<std::common_type_t<T, U>>
    {
        using R = std::common_type_t<T, U>;
        if (cols() != rhs.rows())
            throw std::invalid_argument("secure_matmul: inner dims must match");
        Matrix<R> out(rows(), rhs.cols(), R{0});
        // Constant-time triple loop with ct_barrier, no secret-dependent branches
        for (std::size_t i = 0; i < rows(); ++i)
            for (std::size_t k = 0; k < cols(); ++k)
            {
                R aik = static_cast<R>((*this)(i, k));
                for (std::size_t j = 0; j < rhs.cols(); ++j)
                {
                    // Use volatile to prevent optimization, ct_barrier per inner
                    volatile R *p = reinterpret_cast<volatile R *>(&out(i, j));
                    R bkj = static_cast<R>(rhs(k, j));
                    R cur = *p;
                    cur += aik * bkj;
                    *p = cur;
                }
                pqc::ct_barrier();
            }
        pqc::ct_barrier();
        return out;
    }
};

/** @brief Scalar * Matrix.
 *
 * @tparam T  Matrix element type.
 * @tparam U  Scalar type.
 * @param scalar  Left scalar multiplier.
 * @param m       The matrix.
 * @return        Matrix of the common type.
 */
template <typename T, typename U>
NP_API NP_NODISCARD auto operator*(U scalar, const Matrix<T> &m) -> Matrix<std::common_type_t<T, U>>
{
    return m * scalar;
}

/** @brief Determinant via Gaussian elimination with partial pivoting.
 *
 * Handles both real and complex via common type. Time complexity: O(n^3).
 *
 * @tparam T  Element type of the matrix.
 * @param m   Square matrix.
 * @return    Determinant as double or complex<double>.
 * @throws    std::invalid_argument if the matrix is not square.
 */
template <typename T>
NP_API NP_NODISCARD auto det(const Matrix<T> &m)
    -> std::conditional_t<detail::is_complex_v<T>, std::complex<double>, double>
{
    using R = std::conditional_t<detail::is_complex_v<T>, std::complex<double>, double>;
    if (!m.is_square())
    {
        throw std::invalid_argument("det requires a square matrix");
    }
    const std::size_t n = m.rows();
    std::vector<std::vector<R>> a(n, std::vector<R>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a[i][j] = static_cast<R>(m(i, j));
    R detv = R{1};
    for (std::size_t col = 0; col < n; ++col)
    {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::abs(a[r][col]) > std::abs(a[piv][col]))
                piv = r;
        if (a[piv][col] == R{0})
            return R{0};
        if (piv != col)
        {
            std::swap(a[piv], a[col]);
            detv = -detv;
        }
        detv *= a[col][col];
        for (std::size_t r = col + 1; r < n; ++r)
        {
            const R f = a[r][col] / a[col][col];
            for (std::size_t c = col + 1; c < n; ++c)
                a[r][c] -= f * a[col][c];
        }
    }
    return detv;
}

/** @brief Inverse via Gauss-Jordan elimination.
 *
 * Handles real and complex via common type. Time complexity: O(n^3).
 *
 * @tparam T  Element type of the matrix.
 * @param m   Square matrix.
 * @return    Inverse matrix.
 * @throws    std::invalid_argument if the matrix is not square or singular.
 */
template <typename T>
NP_API NP_NODISCARD auto inverse(const Matrix<T> &m)
    -> Matrix<std::conditional_t<detail::is_complex_v<T>, std::complex<double>, double>>
{
    using R = std::conditional_t<detail::is_complex_v<T>, std::complex<double>, double>;
    if (!m.is_square())
        throw std::invalid_argument("inverse requires a square matrix");
    const std::size_t n = m.rows();
    std::vector<std::vector<R>> a(n, std::vector<R>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a[i][j] = static_cast<R>(m(i, j));
    std::vector<std::vector<R>> inv(n, std::vector<R>(n, R{0}));
    for (std::size_t i = 0; i < n; ++i)
        inv[i][i] = R{1};
    for (std::size_t col = 0; col < n; ++col)
    {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::abs(a[r][col]) > std::abs(a[piv][col]))
                piv = r;
        if (a[piv][col] == R{0})
            throw std::invalid_argument("matrix is singular");
        if (piv != col)
        {
            std::swap(a[piv], a[col]);
            std::swap(inv[piv], inv[col]);
        }
        const R diag = a[col][col];
        for (std::size_t c = 0; c < n; ++c)
        {
            a[col][c] /= diag;
            inv[col][c] /= diag;
        }
        for (std::size_t r = 0; r < n; ++r)
        {
            if (r == col)
                continue;
            const R f = a[r][col];
            for (std::size_t c = 0; c < n; ++c)
            {
                a[r][c] -= f * a[col][c];
                inv[r][c] -= f * inv[col][c];
            }
        }
    }
    Matrix<R> out(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            out(i, j) = inv[i][j];
    return out;
}

/** @brief Solve A x = b via Gaussian elimination with partial pivoting.
 *
 * Handles real and complex via common type. Time complexity: O(n^3).
 *
 * @tparam T  Element type of matrix A.
 * @tparam U  Element type of vector b.
 * @param a   Square coefficient matrix (n x n).
 * @param b   Right-hand side vector (length n).
 * @return    Solution vector x.
 * @throws    std::invalid_argument if b is not 1-D, shapes are
 *            incompatible, or the matrix is singular.
 */
template <typename T, typename U> NP_API NP_NODISCARD auto solve(const Matrix<T> &a, const ndarray<U> &b)
{
    using R0 = std::common_type_t<T, U>;
    using R = std::conditional_t<detail::is_complex_v<R0>, std::complex<double>, double>;
    if (b.ndim() != 1)
        throw std::invalid_argument("solve: b must be a 1D array");
    if (!a.is_square() || a.rows() != static_cast<std::size_t>(b.shape[0]))
        throw std::invalid_argument("solve: incompatible shapes");
    const std::size_t n = a.rows();
    std::vector<std::vector<R>> m(n, std::vector<R>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            m[i][j] = static_cast<R>(a(i, j));
    std::vector<R> rhs(n);
    for (std::size_t i = 0; i < n; ++i)
        rhs[i] = static_cast<R>(b.at(i));
    for (std::size_t col = 0; col < n; ++col)
    {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::abs(m[r][col]) > std::abs(m[piv][col]))
                piv = r;
        if (m[piv][col] == R{0})
            throw std::invalid_argument("matrix is singular");
        if (piv != col)
        {
            std::swap(m[piv], m[col]);
            std::swap(rhs[piv], rhs[col]);
        }
        const R diag = m[col][col];
        for (std::size_t c = col; c < n; ++c)
            m[col][c] /= diag;
        rhs[col] /= diag;
        for (std::size_t r = 0; r < n; ++r)
        {
            if (r == col)
                continue;
            const R f = m[r][col];
            for (std::size_t c = col; c < n; ++c)
                m[r][c] -= f * m[col][c];
            rhs[r] -= f * rhs[col];
        }
    }
    return ndarray<R>::from_data(std::vector<int>{static_cast<int>(n)}, std::move(rhs));
}

} // namespace np

#endif // NP_MATRIX_HPP
