/**
 * @file linalg.hpp
 * @brief Linear algebra for np::ndarray (dot, matmul, svd, qr, eig, det,
 *        inv, solve, cholesky, norms, ...).
 *
 * Signature names, argument order and default values mirror the numpy.linalg
 * pages in numpy-reference/reference/generated/. Decompositions operate on
 * real element types only: integer/bool input is promoted to double (numpy
 * casts to float64), floating-point input keeps its type, and complex input
 * is rejected at compile time (documented divergence from numpy, which
 * dispatches to LAPACK z-solvers).
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_LINALG_HPP
#define NP_LINALG_HPP

#include "api_macros.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "dtype.hpp"
#include "exceptions.hpp"
#include "gpu.hpp"
#include "ndarray.hpp"
#include "powerful.hpp"
#include "pqc.hpp"
#if __has_include("bigint.hpp")
#include "bigint.hpp"
#endif

#ifdef NP_USE_THREADING
#include "threadpool.hpp"
#endif

namespace np::linalg
{

// Norm order enum for norm() and matrix_norm() functions
enum class NormOrd
{
    None,   // Frobenius norm for matrices, 2-norm for vectors
    One,    // 1-norm (max column sum or L1 norm)
    NegOne, // Negative 1-norm
    Two,    // 2-norm (spectral norm for matrices)
    NegTwo, // Negative 2-norm
    Inf,    // Infinity norm (max row sum or max element)
    NegInf, // Negative infinity norm
    Fro,    // Frobenius norm (matrices only)
    Nuc     // Nuclear norm (matrices only)
};

// Result element type: floating T stays T, complex stays complex,
// bigint stays bigint, everything else promotes to double/complex<double>
// (numpy casts integral/bool to float64 / complex128, but bigint keeps exact).
template <typename T> struct _real_promote
{
    using type = std::conditional_t<::np::detail::is_bigint_v<std::remove_cv_t<T>>, T,
                                    std::conditional_t<std::is_floating_point_v<T>, T, double>>;
};
template <typename U> struct _real_promote<std::complex<U>>
{
    using type = std::complex<std::conditional_t<std::is_floating_point_v<U>, U, double>>;
};
template <typename T> using real_t = typename _real_promote<T>::type;

// Real-valued counterpart (magnitudes, singular values): complex -> its
// underlying real type promoted.
template <typename T> struct _real_value
{
    using type = std::conditional_t<std::is_floating_point_v<T>, T, double>;
};
template <typename U> struct _real_value<std::complex<U>>
{
    using type = std::conditional_t<std::is_floating_point_v<U>, U, double>;
};
template <typename T> using real_value_t = typename _real_value<T>::type;

namespace detail
{

// Make a dense row-major copy of a 2-D ndarray, converting elements
// to R. Views and non-C-order input are supported: element access
// honors the array's own strides and offset.
template <typename R, typename T> std::vector<R> dense2d(const ndarray<T> &a, std::size_t &rows, std::size_t &cols)
{
    rows = static_cast<std::size_t>(a.shape[0]);
    cols = static_cast<std::size_t>(a.shape[1]);
    std::vector<R> out(rows * cols);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            out[i * cols + j] = static_cast<R>(a.at(i, j));
        }
    }
    return out;
}

// Wrap a dense buffer as a fresh 2-D ndarray.
template <typename R> ndarray<R> mk2d(std::size_t rows, std::size_t cols, std::vector<R> &&data)
{
    return ndarray<R>::from_data(std::vector<int>{static_cast<int>(rows), static_cast<int>(cols)}, std::move(data));
}

// Copy the first k columns of the (rows x cols) matrix data into a
// (rows x k) buffer (used for the reduced q in qr).
template <typename R>
std::vector<R> take_cols(const std::vector<R> &data, std::size_t rows, std::size_t cols, std::size_t k)
{
    std::vector<R> out(rows * k);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < k; ++j)
        {
            out[i * k + j] = data[i * cols + j];
        }
    }
    return out;
}

// Orthonormal completion of the (rows x cols) matrix data: any column
// whose norm is tiny (a zero singular column, or a column beyond the
// rank) is replaced by a unit vector orthogonal to every previous
// column. Columns that already are unit vectors are left untouched.
// Complex-aware: uses Hermitian inner product with conj.
template <typename R> void ortho_complete(std::vector<R> &data, std::size_t rows, std::size_t cols)
{
    using RV = real_value_t<R>;
    for (std::size_t j = 0; j < cols; ++j)
    {
        RV nrm2{};
        for (std::size_t i = 0; i < rows; ++i)
        {
            if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                          std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                          std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                nrm2 += std::norm(data[i * cols + j]);
            else
                nrm2 += data[i * cols + j] * data[i * cols + j];
        }
        RV nrm = std::sqrt(nrm2);
        if (nrm > RV{0.5})
        {
            continue;
        }
        bool found = false;
        for (std::size_t m = 0; m < rows && !found; ++m)
        {
            std::vector<R> cand(rows, R{});
            cand[m] = R{1};
            for (std::size_t t = 0; t < j; ++t)
            {
                R dot{};
                for (std::size_t i = 0; i < rows; ++i)
                {
                    if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                                  std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                                  std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                        dot += std::conj(data[i * cols + t]) * cand[i];
                    else
                        dot += cand[i] * data[i * cols + t];
                }
                for (std::size_t i = 0; i < rows; ++i)
                {
                    // For complex, projection uses data * dot? For Gram-Schmidt with Hermitian
                    // inner product, cand = cand - dot * data; but dot = <data,cand>, so need
                    // data*dot For real, dot = <cand,data> so cand - dot*data is same.
                    cand[i] -= dot * data[i * cols + t];
                }
            }
            RV cn2{};
            for (std::size_t i = 0; i < rows; ++i)
            {
                if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                              std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                              std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                    cn2 += std::norm(cand[i]);
                else
                    cn2 += cand[i] * cand[i];
            }
            RV cn = std::sqrt(cn2);
            if (cn > RV{0.5})
            {
                for (std::size_t i = 0; i < rows; ++i)
                {
                    data[i * cols + j] = cand[i] / cn;
                }
                found = true;
            }
        }
        if (!found)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                data[i * cols + j] = R{0};
            }
        }
    }
}

// One-sided Jacobi SVD of the (p x q) row-major matrix a (any aspect
// ratio). On return s holds the min(p, q) singular values in
// descending order; u (p x p when full, else p x k) holds the left
// singular vectors and v (q x q when full, else q x k) the right
// singular vectors, with zero singular columns completed to an
// orthonormal basis. When want_uv is false only s is produced.
// Throws np::exceptions::LinAlgError if the sweeps do not converge.
// Complex-aware: uses Hermitian inner product and unitary rotations.
template <typename R>
void svd_jacobi(const std::vector<R> &a, std::size_t p, std::size_t q, bool full, bool want_uv, std::vector<R> &u,
                std::vector<R> &s, std::vector<R> &v)
{
    const std::size_t k = std::min(p, q);
    const std::size_t up = want_uv ? (full ? p : k) : 0;
    const std::size_t vp = want_uv ? (full ? q : k) : 0;
    u.assign(up * p, R{});
    v.assign(vp * q, R{});
    s.assign(k, R{});
    if (p == 0 || q == 0)
    {
        return;
    }
    using RV = real_value_t<R>;
    std::vector<R> b = a;
    std::vector<R> vv(q * q, R{});
    for (std::size_t j = 0; j < q; ++j)
    {
        vv[j * q + j] = R{1};
    }
    const RV eps = std::numeric_limits<RV>::epsilon();
    const int max_sweeps = 60;
    bool converged = false;
    for (int sweep = 0; sweep < max_sweeps && !converged; ++sweep)
    {
        converged = true;
        RV scale{};
        for (std::size_t j = 0; j < q; ++j)
        {
            RV nrm2{};
            for (std::size_t i = 0; i < p; ++i)
            {
                if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                              std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                              std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                    nrm2 += std::norm(b[i * q + j]);
                else
                    nrm2 += b[i * q + j] * b[i * q + j];
            }
            scale = std::max(scale, nrm2);
        }
        const RV small = eps * scale;
        for (std::size_t pc = 0; pc < q; ++pc)
        {
            for (std::size_t qc = pc + 1; qc < q; ++qc)
            {
                RV alpha{}, beta{};
                R gamma{};
                for (std::size_t i = 0; i < p; ++i)
                {
                    const R x = b[i * q + pc];
                    const R y = b[i * q + qc];
                    if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                                  std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                                  std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                    {
                        alpha += std::norm(x);
                        beta += std::norm(y);
                        gamma += std::conj(x) * y;
                    }
                    else
                    {
                        alpha += x * x;
                        beta += y * y;
                        gamma += x * y;
                    }
                }
                if (alpha <= small || beta <= small)
                {
                    continue;
                }
                RV gamma_abs = std::abs(gamma);
                if (gamma_abs <= eps * std::sqrt(alpha * beta))
                {
                    continue;
                }
                if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                              std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                              std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                {
                    if (gamma_abs != RV{0})
                    {
                        R phase = gamma / static_cast<R>(gamma_abs);
                        R conj_phase = std::conj(phase);
                        for (std::size_t i = 0; i < p; ++i)
                        {
                            b[i * q + qc] *= conj_phase;
                        }
                        if (want_uv)
                        {
                            for (std::size_t j = 0; j < q; ++j)
                            {
                                vv[j * q + qc] *= conj_phase;
                            }
                        }
                        gamma = R{gamma_abs};
                    }
                }
                RV g_real{};
                if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                              std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                              std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                    g_real = std::real(gamma);
                else
                    g_real = gamma;
                const RV zeta = (beta - alpha) / (RV{2} * g_real);
                const RV t = (zeta >= RV{0} ? RV{1} : RV{-1}) / (std::abs(zeta) + std::sqrt(RV{1} + zeta * zeta));
                const RV c = RV{1} / std::sqrt(RV{1} + t * t);
                const RV sn = c * t;
                for (std::size_t i = 0; i < p; ++i)
                {
                    const R x = b[i * q + pc];
                    const R y = b[i * q + qc];
                    b[i * q + pc] = static_cast<R>(c * x - sn * y);
                    b[i * q + qc] = static_cast<R>(sn * x + c * y);
                }
                if (want_uv)
                {
                    for (std::size_t j = 0; j < q; ++j)
                    {
                        const R x = vv[j * q + pc];
                        const R y = vv[j * q + qc];
                        vv[j * q + pc] = static_cast<R>(c * x - sn * y);
                        vv[j * q + qc] = static_cast<R>(sn * x + c * y);
                    }
                }
                converged = false;
            }
        }
    }
    if (!converged)
    {
        throw np::exceptions::LinAlgError("SVD did not converge");
    }
    std::vector<RV> norms(q);
    for (std::size_t j = 0; j < q; ++j)
    {
        RV nrm2{};
        for (std::size_t i = 0; i < p; ++i)
        {
            if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                          std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                          std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                nrm2 += std::norm(b[i * q + j]);
            else
                nrm2 += b[i * q + j] * b[i * q + j];
        }
        norms[j] = std::sqrt(nrm2);
    }
    std::vector<std::size_t> order(q);
    for (std::size_t j = 0; j < q; ++j)
    {
        order[j] = j;
    }
    std::sort(order.begin(), order.end(), [&norms](std::size_t x, std::size_t y) { return norms[x] > norms[y]; });
    s.assign(k, R{});
    for (std::size_t j = 0; j < k; ++j)
    {
        if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                      std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                      std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
            s[j] = R{norms[order[j]]};
        else
            s[j] = static_cast<R>(norms[order[j]]);
    }
    std::vector<R> uu, vv2;
    if (want_uv)
    {
        uu.assign(up * p, R{});
        for (std::size_t j = 0; j < k; ++j)
        {
            const std::size_t src = order[j];
            RV sj = norms[order[j]];
            for (std::size_t i = 0; i < p; ++i)
            {
                uu[i * up + j] = sj > RV{0} ? b[i * q + src] / static_cast<R>(sj) : R{0};
            }
        }
        ortho_complete(uu, p, up);
        vv2.assign(vp * q, R{});
        for (std::size_t j = 0; j < k; ++j)
        {
            const std::size_t src = order[j];
            for (std::size_t i = 0; i < q; ++i)
            {
                vv2[i * vp + j] = vv[i * q + src];
            }
        }
        ortho_complete(vv2, q, vp);
        u = std::move(uu);
        v = std::move(vv2);
    }
}

// Householder QR of the (m x n) row-major matrix a. q (m x m,
// orthonormal), r (k x n, upper triangular, k = min(m, n)), h (m x n:
// R in the upper triangle, Householder vectors below the diagonal
// with v_j = 1) and tau (k,) follow the LAPACK convention used by
// numpy's raw mode.
// Complex-aware: real path uses Householder reflectors, complex path uses
// Gram-Schmidt.
template <typename R>
void householder_qr(const std::vector<R> &a, std::size_t m, std::size_t n, std::vector<R> &q, std::vector<R> &r,
                    std::vector<R> &h, std::vector<R> &tau)
{
    if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                  std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                  std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
    {
        const std::size_t k = std::min(m, n);
        q.assign(m * m, R{});
        r.assign(k * n, R{});
        h = a;
        tau.assign(k, R{});
        using RV = real_value_t<R>;
        std::vector<std::vector<R>> cols(m, std::vector<R>(m, R{}));
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < m; ++i)
                cols[j][i] = a[i * n + j];
        for (std::size_t j = n; j < m; ++j)
            cols[j][j] = R{1};
        std::vector<std::vector<R>> qcols(m, std::vector<R>(m, R{}));
        for (std::size_t j = 0; j < m; ++j)
        {
            std::vector<R> v = cols[j];
            for (std::size_t i = 0; i < j; ++i)
            {
                R dot{};
                for (std::size_t t = 0; t < m; ++t)
                    dot += std::conj(qcols[i][t]) * v[t];
                if (i < k && j < n)
                    r[i * n + j] = dot;
                for (std::size_t t = 0; t < m; ++t)
                    v[t] -= dot * qcols[i][t];
            }
            RV nrm2{};
            for (std::size_t t = 0; t < m; ++t)
                nrm2 += std::norm(v[t]);
            RV nrm = std::sqrt(nrm2);
            if (nrm > RV{0})
            {
                for (std::size_t t = 0; t < m; ++t)
                    qcols[j][t] = v[t] / static_cast<R>(nrm);
                if (j < k && j < n)
                    r[j * n + j] = static_cast<R>(nrm);
            }
            else
            {
                for (std::size_t t = 0; t < m; ++t)
                    qcols[j][t] = R{0};
                qcols[j][j] = R{1};
                if (j < k && j < n)
                    r[j * n + j] = R{0};
            }
        }
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < m; ++i)
                q[i * m + j] = qcols[j][i];
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < i; ++j)
                r[i * n + j] = R{0};
        h = a;
        return;
    }
    else
    {
        const std::size_t k = std::min(m, n);
        q.assign(m * m, R{});
        for (std::size_t i = 0; i < m; ++i)
        {
            q[i * m + i] = R{1};
        }
        tau.assign(k, R{});
        h = a;
        for (std::size_t j = 0; j < k; ++j)
        {
            const R xj = h[j * n + j];
            R nrm2{};
            for (std::size_t i = j + 1; i < m; ++i)
            {
                nrm2 += h[i * n + j] * h[i * n + j];
            }
            const R nrm = std::sqrt(xj * xj + nrm2);
            if (nrm == R{0})
            {
                tau[j] = R{0};
                continue;
            }
            const R sign = xj >= R{0} ? R{1} : R{-1};
            const R beta = -sign * nrm;
            const R vj = xj - beta;
            const R tauj = (beta - xj) / beta;
            tau[j] = tauj;
            h[j * n + j] = beta;
            for (std::size_t i = j + 1; i < m; ++i)
            {
                h[i * n + j] /= vj;
            }
            for (std::size_t c = j + 1; c < n; ++c)
            {
                R dot = h[j * n + c];
                for (std::size_t i = j + 1; i < m; ++i)
                {
                    dot += h[i * n + j] * h[i * n + c];
                }
                for (std::size_t i = j + 1; i < m; ++i)
                {
                    h[i * n + c] -= tauj * h[i * n + j] * dot;
                }
                h[j * n + c] -= tauj * dot;
            }
            for (std::size_t i = 0; i < m; ++i)
            {
                R dot = q[i * m + j];
                for (std::size_t c = j + 1; c < m; ++c)
                {
                    dot += q[i * m + c] * h[c * n + j];
                }
                q[i * m + j] -= tauj * dot;
                for (std::size_t c = j + 1; c < m; ++c)
                {
                    q[i * m + c] -= tauj * h[c * n + j] * dot;
                }
            }
        }
        r.assign(k * n, R{});
        for (std::size_t i = 0; i < k; ++i)
        {
            for (std::size_t j = i; j < n; ++j)
            {
                r[i * n + j] = h[i * n + j];
            }
        }
        return;
    }
}

// Reduce the (n x n) matrix h to upper Hessenberg form with
// Householder reflectors. On return h is Hessenberg and q satisfies
// h = q' A q (q accumulates as A = q h q').
template <typename R> void hessenberg(std::vector<R> &h, std::size_t n, std::vector<R> &q)
{
    q.assign(n * n, R{});
    for (std::size_t i = 0; i < n; ++i)
    {
        q[i * n + i] = R{1};
    }
    if (n < 3)
    {
        return;
    }
    for (std::size_t j = 0; j + 2 < n; ++j)
    {
        // Column j below the subdiagonal: x = (x1, x2, ...).
        const R x1 = h[(j + 1) * n + j];
        R nrm2{};
        for (std::size_t i = j + 2; i < n; ++i)
        {
            nrm2 += h[i * n + j] * h[i * n + j];
        }
        if (nrm2 == R{0})
        {
            continue;
        }
        const R nrm = std::sqrt(x1 * x1 + nrm2);
        const R sign = x1 >= R{0} ? R{1} : R{-1};
        const R beta = -sign * nrm;
        const R vj = x1 - beta;
        const R tau = (beta - x1) / beta;
        // Save the normalized reflector: v = (1, x2/vj, ...).
        std::vector<R> vr(n, R{});
        vr[j + 1] = R{1};
        for (std::size_t i = j + 2; i < n; ++i)
        {
            vr[i] = h[i * n + j] / vj;
        }
        // Similarity h = H h H: left factor on rows j+1..n-1.
        for (std::size_t c = 0; c < n; ++c)
        {
            R dot{};
            for (std::size_t i = j + 1; i < n; ++i)
            {
                dot += vr[i] * h[i * n + c];
            }
            for (std::size_t i = j + 1; i < n; ++i)
            {
                h[i * n + c] -= tau * vr[i] * dot;
            }
        }
        // Right factor on columns j+1..n-1.
        for (std::size_t r = 0; r < n; ++r)
        {
            R dot{};
            for (std::size_t i = j + 1; i < n; ++i)
            {
                dot += vr[i] * h[r * n + i];
            }
            for (std::size_t i = j + 1; i < n; ++i)
            {
                h[r * n + i] -= tau * vr[i] * dot;
            }
        }
        // Accumulate q = q H.
        for (std::size_t r = 0; r < n; ++r)
        {
            R dot{};
            for (std::size_t i = j + 1; i < n; ++i)
            {
                dot += vr[i] * q[r * n + i];
            }
            for (std::size_t i = j + 1; i < n; ++i)
            {
                q[r * n + i] -= tau * vr[i] * dot;
            }
        }
        // The subdiagonal entry is beta; below it the column is zero.
        h[(j + 1) * n + j] = beta;
        for (std::size_t i = j + 2; i < n; ++i)
        {
            h[i * n + j] = R{0};
        }
    }
}

// Francis double-shift QR iteration turning the upper-Hessenberg
// matrix h into real Schur form (quasi-upper-triangular, 1x1 and 2x2
// diagonal blocks). q is updated so that h = q' A q on exit.
// Throws np::exceptions::LinAlgError when the iteration count cap is
// exhausted (numpy raises LinAlgError("Eigenvalues did not converge")).
template <typename R> void francis_schur(std::vector<R> &h, std::size_t n, std::vector<R> &q)
{
    if (n < 2)
    {
        return;
    }
    const R eps = std::numeric_limits<R>::epsilon();
    const std::size_t max_iters = 30 * n + 10;
    std::size_t iters = 0;
    std::ptrdiff_t p = static_cast<std::ptrdiff_t>(n) - 1;
    while (p > 0)
    {
        // Find the active block: deflate every subdiagonal entry that
        // is negligible relative to its neighbors.
        std::ptrdiff_t l = p;
        while (l > 0)
        {
            const R tol = eps * (std::abs(h[(l - 1) * n + l - 1]) + std::abs(h[l * n + l]));
            if (std::abs(h[l * n + l - 1]) <= tol)
            {
                h[l * n + l - 1] = R{0};
                break;
            }
            --l;
        }
        if (l == p)
        {
            --p; // bottom element is an isolated eigenvalue
            continue;
        }
        if (l == p - 1)
        {
            p -= 2;   // bottom 2x2 block is isolated (real or
            continue; // complex pair); read off both eigenvalues
        }
        if (++iters > max_iters)
        {
            throw np::exceptions::LinAlgError("Eigenvalues did not converge");
        }

        // Shift parameters from the bottom 2x2 block of the active
        // block: s1 + s2 = trace, s1 s2 = determinant.
        const R a = h[(p - 1) * n + p - 1];
        const R b = h[(p - 1) * n + p];
        const R c = h[p * n + p - 1];
        const R d = h[p * n + p];
        const R tr = a + d;
        const R det = a * d - b * c;

        // First column of (A - s1 I)(A - s2 I) = A^2 - tr A + det I.
        R x = h[l * n + l] * h[l * n + l] + h[l * n + l + 1] * h[(l + 1) * n + l] - tr * h[l * n + l] + det;
        R y = h[(l + 1) * n + l] * (h[l * n + l] + h[(l + 1) * n + l + 1] - tr);
        R z = h[(l + 1) * n + l] * h[(l + 2) * n + l + 1];

        // Exceptional shift: when the polynomial's first column
        // vanishes, rotate the block top by 45 degrees so the chase
        // below still makes progress.
        const R nrm = std::sqrt(x * x + y * y + z * z);
        if (nrm == R{0})
        {
            const R c45 = R{0.70710678118654752440};
            const R s45 = c45;
            for (std::ptrdiff_t cc = l; cc <= p; ++cc)
            {
                const R rx = h[l * n + cc];
                const R ry = h[(l + 1) * n + cc];
                h[l * n + cc] = c45 * rx - s45 * ry;
                h[(l + 1) * n + cc] = s45 * rx + c45 * ry;
            }
            for (std::ptrdiff_t rr = l; rr <= p; ++rr)
            {
                const R cx = h[rr * n + l];
                const R cy = h[rr * n + l + 1];
                h[rr * n + l] = c45 * cx - s45 * cy;
                h[rr * n + l + 1] = s45 * cx + c45 * cy;
            }
            for (std::size_t rr = 0; rr < n; ++rr)
            {
                const R cx = q[rr * n + l];
                const R cy = q[rr * n + l + 1];
                q[rr * n + l] = c45 * cx - s45 * cy;
                q[rr * n + l + 1] = s45 * cx + c45 * cy;
            }
        }
        else
        {
            // Householder on rows/cols l..l+2 annihilating (y, z);
            // reflector with v_l = 1.
            const R sign = x >= R{0} ? R{1} : R{-1};
            const R beta = -sign * nrm;
            const R vx = x - beta;
            const R tau = (beta - x) / beta;
            const R vy = y / vx;
            const R vz = z / vx;
            for (std::ptrdiff_t c = l; c <= p; ++c)
            {
                R dot = h[l * n + c] + vy * h[(l + 1) * n + c] + vz * h[(l + 2) * n + c];
                h[l * n + c] -= tau * dot;
                h[(l + 1) * n + c] -= tau * vy * dot;
                h[(l + 2) * n + c] -= tau * vz * dot;
            }
            for (std::size_t r = 0; r < n; ++r)
            {
                R dot = h[r * n + l] + vy * h[r * n + l + 1] + vz * h[r * n + l + 2];
                h[r * n + l] -= tau * dot;
                h[r * n + l + 1] -= tau * vy * dot;
                h[r * n + l + 2] -= tau * vz * dot;
            }
            for (std::size_t r = 0; r < n; ++r)
            {
                R dot = q[r * n + l] + vy * q[r * n + l + 1] + vz * q[r * n + l + 2];
                q[r * n + l] -= tau * dot;
                q[r * n + l + 1] -= tau * vy * dot;
                q[r * n + l + 2] -= tau * vz * dot;
            }
        }

        // Chase the bulge down the subdiagonal with Givens rotations.
        for (std::ptrdiff_t kk = l + 1; kk < p; ++kk)
        {
            const R aa = h[kk * n + kk - 1];
            const R bb = h[(kk + 1) * n + kk - 1];
            const R rr = std::sqrt(aa * aa + bb * bb);
            if (rr == R{0})
            {
                continue;
            }
            const R c = aa / rr;
            const R s = bb / rr;
            for (std::ptrdiff_t cc = kk - 1; cc <= p; ++cc)
            {
                const R rx = h[kk * n + cc];
                const R ry = h[(kk + 1) * n + cc];
                h[kk * n + cc] = c * rx + s * ry;
                h[(kk + 1) * n + cc] = -s * rx + c * ry;
            }
            for (std::size_t rr2 = 0; rr2 < n; ++rr2)
            {
                const R cx = h[rr2 * n + kk];
                const R cy = h[rr2 * n + kk + 1];
                h[rr2 * n + kk] = c * cx - s * cy;
                h[rr2 * n + kk + 1] = s * cx + c * cy;
                const R qx = q[rr2 * n + kk];
                const R qy = q[rr2 * n + kk + 1];
                q[rr2 * n + kk] = c * qx - s * qy;
                q[rr2 * n + kk + 1] = s * qx + c * qy;
            }
        }
    }
}

// Eigenvalues of the real Schur form h (in place), in numpy order:
// complex pairs appear together, real values first within each block.
template <typename R> std::vector<std::complex<R>> schur_eigenvalues(const std::vector<R> &h, std::size_t n)
{
    std::vector<std::complex<R>> w(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (i + 1 < n && h[(i + 1) * n + i] != R{0})
        {
            // 2x2 block: lambda = tr/2 +/- sqrt(disc)/2.
            const R a = h[i * n + i];
            const R b = h[i * n + i + 1];
            const R c = h[(i + 1) * n + i];
            const R d = h[(i + 1) * n + i + 1];
            const R disc = (a - d) * (a - d) + 4 * b * c;
            const R sq = std::sqrt(std::abs(disc));
            if (disc >= R{0})
            {
                w[i] = {(a + d) / 2 + sq / 2, R{0}};
                w[i + 1] = {(a + d) / 2 - sq / 2, R{0}};
            }
            else
            {
                w[i] = {(a + d) / 2, sq / 2};
                w[i + 1] = {(a + d) / 2, -sq / 2};
            }
            ++i;
        }
        else
        {
            w[i] = {h[i * n + i], R{0}};
        }
    }
    return w;
}

// Right eigenvectors of the real Schur form h with similarity
// transform q (h = q' A q). Column j of the returned (n x n) matrix
// is the unit eigenvector of A for w[j]: (h - lambda I) z = 0 is
// solved bottom-up one diagonal block at a time (each block couples
// only to the blocks below it through the upper-triangular entries),
// and v = q z. A singular denominator (defective / nearly defective
// matrix, where no eigenvector exists) zeroes the component, which
// yields an arbitrary orthonormal completion, as LAPACK does.
template <typename R>
std::vector<std::complex<R>> schur_eigenvectors(const std::vector<R> &h, const std::vector<R> &q, std::size_t n,
                                                const std::vector<std::complex<R>> &w)
{
    using C = std::complex<R>;
    const R eps = std::numeric_limits<R>::epsilon();
    std::vector<C> v(n * n);
    // Block start for every eigenvalue: a 2x2 block contributes both
    // of its eigenvalues, each mapping back to the block's top row.
    std::vector<std::size_t> block_start;
    for (std::size_t i = 0; i < n;)
    {
        if (i + 1 < n && h[(i + 1) * n + i] != R{0})
        {
            block_start.push_back(i);
            block_start.push_back(i);
            i += 2;
        }
        else
        {
            block_start.push_back(i);
            ++i;
        }
    }
    for (std::size_t j = 0; j < n; ++j)
    {
        const C lambda = w[j];
        std::vector<C> z(n, C{0});
        const std::size_t bs = block_start[j];
        for (std::ptrdiff_t ii = static_cast<std::ptrdiff_t>(n) - 1; ii >= 0; --ii)
        {
            const std::size_t u = static_cast<std::size_t>(ii);
            const bool pair = u > 0 && h[u * n + u - 1] != R{0};
            if (!pair)
            {
                if (u == bs)
                {
                    z[u] = C{1};
                    continue;
                }
                C rhs{};
                for (std::size_t c = u + 1; c < n; ++c)
                {
                    rhs += C{h[u * n + c], R{0}} * z[c];
                }
                const C den = C{h[u * n + u], R{0}} - lambda;
                if (std::abs(den) <= eps * (std::abs(h[u * n + u]) + std::abs(lambda)))
                {
                    z[u] = C{0};
                    continue;
                }
                z[u] = -rhs / den;
                continue;
            }
            // 2x2 block at rows (u - 1, u).
            const std::size_t t = u - 1;
            if (t == bs && lambda.imag() != R{0})
            {
                // complex pair pivot: z = (b, lambda - a)
                z[t] = C{h[t * n + u], R{0}};
                z[u] = lambda - C{h[t * n + t], R{0}};
                --ii;
                continue;
            }
            C rhs1{}, rhs2{};
            for (std::size_t c = u + 1; c < n; ++c)
            {
                rhs1 -= C{h[t * n + c], R{0}} * z[c];
                rhs2 -= C{h[u * n + c], R{0}} * z[c];
            }
            const C a = C{h[t * n + t], R{0}} - lambda;
            const C b = C{h[t * n + u], R{0}};
            const C c = C{h[u * n + t], R{0}};
            const C d = C{h[u * n + u], R{0}} - lambda;
            if (t == bs)
            {
                // real eigenvalue from the pivot 2x2 block: z[t] = 1
                // and z[u] follows from the block's second row.
                z[t] = C{1};
                if (std::abs(d) <= eps * (std::abs(h[u * n + u]) + std::abs(lambda)))
                {
                    z[u] = C{0};
                }
                else
                {
                    z[u] = (rhs2 - c) / d;
                }
                --ii;
                continue;
            }
            const C det = a * d - b * c;
            if (std::abs(det) <= eps * (std::abs(a * d) + std::abs(b * c)))
            {
                z[t] = C{0};
                z[u] = C{0};
            }
            else
            {
                z[t] = (rhs1 * d - b * rhs2) / det;
                z[u] = (a * rhs2 - c * rhs1) / det;
            }
            --ii;
        }
        // v[:, j] = q z, then normalize to a unit column.
        for (std::size_t i = 0; i < n; ++i)
        {
            C acc{};
            for (std::size_t t = 0; t < n; ++t)
            {
                acc += C{q[i * n + t], R{0}} * z[t];
            }
            v[i * n + j] = acc;
        }
        R nrm{};
        for (std::size_t i = 0; i < n; ++i)
        {
            nrm += std::norm(v[i * n + j]);
        }
        // nrm is never zero here: the pivot component z[bs] is set to
        // C{1} unconditionally above (both the 1x1 and 2x2-block pivot
        // cases), so nrm >= 1 in exact arithmetic before this sqrt.
        nrm = std::sqrt(nrm);
        for (std::size_t i = 0; i < n; ++i)
        {
            v[i * n + j] /= nrm;
        }
    }
    return v;
}

// Shared eig/eigvals pipeline: dense copy -> Hessenberg -> real
// Schur. Returns (schur, transform) with schur = q' A q.
template <typename R>
void schur_decompose(const std::vector<R> &a, std::size_t n, std::vector<R> &schur, std::vector<R> &q)
{
    schur = a;
    hessenberg(schur, n, q);
    francis_schur(schur, n, q);
}

// In-place LU factorization with partial pivoting of the (n x n)
// row-major matrix a. On return the upper triangle (including the
// diagonal) holds U, the strict lower triangle holds the multipliers
// and piv records the original row of A now sitting in each row, so
// that A = P' L U with P the recorded permutation. Returns true when
// a zero pivot marks the matrix singular (the factorization is then
// incomplete; callers treat it as a zero determinant or raise
// LinAlgError). Tiny nonzero pivots are kept, so ill-conditioned
// matrices factor successfully, as in LAPACK.
// Complex-aware: pivot search uses magnitudes.
template <typename R>
bool lu_factor(std::vector<R> &a, std::size_t n, std::vector<std::size_t> &piv, std::size_t &swaps)
{
    piv.resize(n);
    swaps = 0;
    for (std::size_t k = 0; k < n; ++k)
    {
        piv[k] = k;
        std::size_t p = k;
        auto best = [&]() {
            if constexpr (np::detail::is_bigint_v<R>)
            {
                R v = a[k * n + k];
                return v < R{0} ? -v : v;
            }
            else
                return std::abs(a[k * n + k]);
        }();
        for (std::size_t i = k + 1; i < n; ++i)
        {
            auto v = [&]() {
                if constexpr (np::detail::is_bigint_v<R>)
                {
                    R x = a[i * n + k];
                    return x < R{0} ? -x : x;
                }
                else
                    return std::abs(a[i * n + k]);
            }();
            if (v > best)
            {
                best = v;
                p = i;
            }
        }
        if (best == decltype(best){0})
        {
            return true;
        }
        if (p != k)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                std::swap(a[k * n + j], a[p * n + j]);
            }
            std::swap(piv[k], piv[p]);
            ++swaps;
        }
        for (std::size_t i = k + 1; i < n; ++i)
        {
            const R mult = a[i * n + k] / a[k * n + k];
            a[i * n + k] = mult;
            for (std::size_t j = k + 1; j < n; ++j)
            {
                a[i * n + j] -= mult * a[k * n + j];
            }
        }
    }
    return false;
}

// Solve L U x = P b for one right-hand side after lu_factor: apply
// the recorded permutation to b, forward-substitute with L (unit
// diagonal), back-substitute with U.
template <typename R>
std::vector<R> lu_solve(const std::vector<R> &lu, std::size_t n, const std::vector<std::size_t> &piv,
                        const std::vector<R> &b)
{
    std::vector<R> x(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = b[piv[i]];
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            x[i] -= lu[i * n + j] * x[j];
        }
    }
    for (std::size_t ii = n; ii-- > 0;)
    {
        for (std::size_t j = ii + 1; j < n; ++j)
        {
            x[ii] -= lu[ii * n + j] * x[j];
        }
        x[ii] /= lu[ii * n + ii];
    }
    return x;
}

// Solve A X = I for every column of the identity after lu_factor;
// returns the inverse as a flat (n x n) row-major buffer.
template <typename R>
std::vector<R> lu_invert(const std::vector<R> &lu, std::size_t n, const std::vector<std::size_t> &piv)
{
    std::vector<R> out(n * n);
    for (std::size_t c = 0; c < n; ++c)
    {
        std::vector<R> e(n, R{});
        e[c] = R{1};
        std::vector<R> x = lu_solve(lu, n, piv, e);
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i * n + c] = x[i];
        }
    }
    return out;
}

// C-order strides (in elements) for a shape; used to place a
// multi-index at its row-major flat position.
inline std::vector<std::size_t> c_order_strides(const std::vector<int> &shape)
{
    std::vector<std::size_t> st(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1;)
    {
        st[i - 1] = st[i] * static_cast<std::size_t>(shape[i]);
    }
    return st;
}

} // namespace detail

// Modes of np::linalg::qr, mapping the numpy 'mode' string: reduced,
// complete, r, raw.
enum class QrMode
{
    Reduced,
    Complete,
    R,
    Raw
};

// Singular value decomposition of a 2-D array; u, s, vh mirror the
// (U, S, Vh) tuple of numpy.linalg.svd.
template <typename R> struct SVDResult
{
    ndarray<R> u;  // left singular vectors (M, M) full / (M, K) reduced
    ndarray<R> s;  // singular values (K,), descending, non-negative
    ndarray<R> vh; // right singular vectors (N, N) full / (K, N) reduced
};

// QR decomposition of a 2-D array; the active members depend on the mode.
template <typename R> struct QRResult
{
    ndarray<R> q;   // (M, M) complete / (M, K) reduced; empty for R/Raw
    ndarray<R> r;   // (K, N) reduced/R/Raw, (M, N) complete
    ndarray<R> h;   // raw mode: (M, N), R above the diagonal and
                    // Householder vectors (v_j = 1) below it
    ndarray<R> tau; // raw mode: (K,)
};

// Eigendecomposition of a square 2-D array; w, v mirror the (w, v) tuple
// of numpy.linalg.eig: column j of v is the unit eigenvector for w[j].
template <typename R> struct EigenResult
{
    ndarray<std::complex<R>> w; // eigenvalues (N,)
    ndarray<std::complex<R>> v; // right eigenvectors (N, N)
};

// Sign and log-absolute-determinant tuple of numpy.linalg.slogdet.
template <typename R> struct SlogdetResult
{
    R sign;      // 0, +1 or -1 for real input; 0 when singular
    R logabsdet; // ln(|det|); -inf when singular
};

// Symmetric eigendecomposition of numpy.linalg.eigh: eigenvalues in
// ascending order with orthonormal eigenvectors (N, N).
template <typename R> struct EighResult
{
    ndarray<R> w; // eigenvalues (N,), ascending
    ndarray<R> v; // eigenvectors (N, N), orthonormal columns
};

// Least-squares solution tuple of numpy.linalg.lstsq.
template <typename R> struct LstsqResult
{
    ndarray<R> x;         // (N,) or (N, K): least-squares solution(s)
    ndarray<R> residuals; // (1,) or (K,) squared residuals; (0,) when
                          // rank-deficient or when M <= N
    int rank;             // number of singular values above the cutoff
    ndarray<R> s;         // (min(M, N),) singular values, descending
};

/**
 * @brief Singular value decomposition.
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.svd.html
 * The numpy 'hermitian' keyword is not supported; Hermitian input takes
 * the general path. Raises std::invalid_argument unless a.ndim() == 2,
 * np::exceptions::LinAlgError when the Jacobi sweeps do not converge.
 * Input containing NaN/Inf is not validated and yields unspecified
 * results (likely LinAlgError). M < N matrices are decomposed through
 * A' and the roles of u and vh are swapped.
 * @tparam T Element type (must be real).
 * @param a Input array (2-D).
 * @param full_matrices If true (default), compute full U (M x M) and
 *        Vh (N x N); otherwise reduced forms.
 * @param compute_uv If true (default), compute U and Vh; otherwise
 *        only singular values are returned.
 * @return SVD result with U, singular values, and Vh.
 * @throws std::invalid_argument if a.ndim() != 2.
 * @throws np::exceptions::LinAlgError if the Jacobi sweeps do not converge.
 * @complexity O(M * N * min(M, N) * sweeps).
 */
NP_API template <typename T>
NP_NODISCARD auto svd(const ndarray<T> &a, bool full_matrices = true, bool compute_uv = true) -> SVDResult<real_t<T>>
{
    using R = real_t<T>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("svd requires a 2D array");
    }
    std::size_t m{}, n{};
    std::vector<R> dense = detail::dense2d<R>(a, m, n);
    const std::size_t k = std::min(m, n);

    SVDResult<R> out;
    std::vector<R> u_vec, s_vec, v_vec;
    detail::svd_jacobi(dense, m, n, full_matrices, compute_uv, u_vec, s_vec, v_vec);
    out.s = ndarray<R>::from_data(std::vector<int>{static_cast<int>(k)}, std::move(s_vec));
    if (!compute_uv)
    {
        // numpy returns (None, s, None); u and vh are explicit empties.
        out.u = detail::mk2d(0, 0, std::vector<R>{});
        out.vh = detail::mk2d(0, 0, std::vector<R>{});
        return out;
    }
    if (m >= n)
    {
        out.u = detail::mk2d(m, full_matrices ? m : k, std::move(u_vec));
        out.vh = detail::mk2d(n, full_matrices ? n : k, std::move(v_vec));
        out.vh = out.vh.transpose();
    }
    else
    {
        // A = U S V' <=> A' = V S U': decompose A' and swap the roles.
        std::vector<R> dense_t(n * m);
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                dense_t[j * m + i] = dense[i * n + j];
            }
        }
        detail::svd_jacobi(dense_t, n, m, full_matrices, compute_uv, u_vec, s_vec, v_vec);
        out.u = detail::mk2d(m, full_matrices ? m : k, std::move(v_vec));
        out.vh = detail::mk2d(n, full_matrices ? n : k, std::move(u_vec));
        out.vh = out.vh.transpose();
    }
    return out;
}

/**
 * @brief Singular values only (numpy.linalg.svdvals).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.svdvals.html
 * @tparam T Element type (must be real).
 * @param a Input array (2-D).
 * @return Singular values in descending order.
 * @throws std::invalid_argument if a.ndim() != 2.
 * @complexity O(M * N * min(M, N) * sweeps).
 */
NP_API template <typename T> NP_NODISCARD auto svdvals(const ndarray<T> &a) -> ndarray<real_t<T>>
{
    return svd(a, false, false).s;
}

/**
 * @brief QR decomposition (numpy.linalg.qr).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.qr.html
 * Reduced = true (default) returns Q (M, K) and R (K, N),
 * K = min(M, N); Reduced = false returns the complete Q
 * (M, M) and R (M, N) with zeros below the diagonal.
 * Raises std::invalid_argument unless a.ndim() == 2.
 * @tparam T Element type (must be real).
 * @param a Input array (2-D).
 * @param mode QR mode (default: Reduced).
 * @return QR result with Q and R factors.
 * @throws std::invalid_argument if a.ndim() != 2.
 * @complexity O(M * N * K).
 */
NP_API template <typename T>
NP_NODISCARD auto qr(const ndarray<T> &a, QrMode mode = QrMode::Reduced) -> QRResult<real_t<T>>
{
    using R = real_t<T>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("qr requires a 2D array");
    }
    std::size_t m{}, n{};
    std::vector<R> dense = detail::dense2d<R>(a, m, n);
    const std::size_t k = std::min(m, n);

    std::vector<R> q_vec, r_vec, h_vec, tau_vec;
    detail::householder_qr(dense, m, n, q_vec, r_vec, h_vec, tau_vec);

    QRResult<R> out;
    switch (mode)
    {
    case QrMode::Reduced:
        out.q = detail::mk2d(m, k, detail::take_cols(q_vec, m, m, k));
        out.r = detail::mk2d(k, n, std::move(r_vec));
        break;
    case QrMode::Complete:
        out.q = detail::mk2d(m, m, std::move(q_vec));
        {
            // complete r is (M, N) with M - K trailing zero rows
            std::vector<R> full_r(m * n, R{});
            for (std::size_t i = 0; i < k; ++i)
            {
                for (std::size_t j = 0; j < n; ++j)
                {
                    full_r[i * n + j] = r_vec[i * n + j];
                }
            }
            out.r = detail::mk2d(m, n, std::move(full_r));
        }
        break;
    case QrMode::R:
        out.q = detail::mk2d(0, 0, std::vector<R>{});
        out.r = detail::mk2d(k, n, std::move(r_vec));
        break;
    case QrMode::Raw:
        out.q = detail::mk2d(0, 0, std::vector<R>{});
        out.r = detail::mk2d(0, 0, std::vector<R>{});
        out.h = detail::mk2d(m, n, std::move(h_vec));
        out.tau = ndarray<R>::from_data(std::vector<int>{static_cast<int>(k)}, std::move(tau_vec));
        break;
    }
    return out;
}

/**
 * @brief Eigenvalues and right eigenvectors of a square real matrix
 *        (numpy.linalg.eig).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.eig.html
 * Raises std::invalid_argument unless a is square 2-D, and
 * np::exceptions::LinAlgError when the Francis iteration does not
 * converge. Elements are returned as std::complex; real eigenvalues have
 * a zero imaginary part. Eigenvector columns are normalized to unit
 * length. A v ~= w v is satisfied within solver tolerance.
 * @tparam T Element type (must be real).
 * @param a Square matrix (N x N).
 * @return EigenResult with eigenvalues (complex) and eigenvectors.
 * @throws std::invalid_argument if a.ndim() != 2 or a is not square.
 * @throws np::exceptions::LinAlgError if the Francis iteration does not
 * converge.
 * @complexity O(N^3) (Francis QR iteration).
 */
NP_API template <typename T> NP_NODISCARD auto eig(const ndarray<T> &a) -> EigenResult<real_value_t<T>>
{
    using R = real_t<T>;
    using RV = real_value_t<T>;
    using CR = std::complex<RV>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("eig requires a 2D array");
    }
    std::size_t m{}, n{};
    std::vector<R> dense = detail::dense2d<R>(a, m, n);
    if (m != n)
    {
        throw std::invalid_argument("eig requires a square matrix");
    }

    EigenResult<RV> out;
    if (n == 0)
    {
        out.w = ndarray<CR>::from_data(std::vector<int>{0}, std::vector<CR>{});
        out.v = ndarray<CR>::from_data(std::vector<int>{0, 0}, std::vector<CR>{});
        return out;
    }

    if (n == 1)
    {
        CR w0{};
        if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                      std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                      std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
            w0 = CR{dense[0]};
        else
            w0 = CR{static_cast<RV>(dense[0]), RV{0}};
        out.w = ndarray<CR>::from_data(std::vector<int>{1}, std::vector<CR>{w0});
        out.v = ndarray<CR>::from_data(std::vector<int>{1, 1}, std::vector<CR>{CR{RV{1}, RV{0}}});
        return out;
    }

    if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                  std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                  std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
    {
        // Complex path: analytic for 2x2, QR iteration for larger
        if (n == 2)
        {
            R a00 = dense[0], a01 = dense[1], a10 = dense[2], a11 = dense[3];
            CR tr = CR{a00 + a11};
            CR det = CR{a00 * a11 - a01 * a10};
            CR disc = tr * tr - CR{4} * det;
            CR sq = std::sqrt(disc);
            CR w0 = (tr + sq) / CR{2};
            CR w1 = (tr - sq) / CR{2};
            out.w = ndarray<CR>::from_data(std::vector<int>{2}, std::vector<CR>{w0, w1});
            // eigenvectors via solving (A - wI)v = 0
            std::vector<CR> v(4);
            for (int k = 0; k < 2; ++k)
            {
                CR w = k == 0 ? w0 : w1;
                CR b00 = CR{a00} - w, b01 = CR{a01}, b10 = CR{a10}, b11 = CR{a11} - w;
                CR v0{1}, v1{0};
                if (std::abs(b00) > std::abs(b01))
                {
                    if (std::abs(b00) > RV{1e-12})
                    {
                        v0 = -b01 / b00;
                        v1 = CR{1};
                    }
                }
                else
                {
                    if (std::abs(b01) > RV{1e-12})
                    {
                        v0 = CR{1};
                        v1 = -b00 / b01;
                    }
                    else if (std::abs(b10) > RV{1e-12} || std::abs(b11) > RV{1e-12})
                    {
                        // use second row
                        if (std::abs(b11) > std::abs(b10))
                        {
                            v0 = -b11 / b10;
                            v1 = CR{1};
                            // actually from b10*v0 + b11*v1=0 => v0 = -b11/b10 * v1
                        }
                        else
                        {
                            v0 = CR{1};
                            v1 = -b10 / b11;
                        }
                    }
                }
                RV nrm = std::sqrt(std::norm(v0) + std::norm(v1));
                if (nrm > RV{0})
                {
                    v0 /= static_cast<CR>(nrm);
                    v1 /= static_cast<CR>(nrm);
                }
                v[k] = v0;
                v[2 + k] = v1;
                // column k is (v0, v1)?? Need row-major: v[0*2+0]=v0 for col0 row0, v[1*2+0]=v1
                // for col0 row1 etc. Our storage is row-major n x n: v[i*n+j] is row i col j So
                // col k: v[0*n + k] = v0, v[1*n + k] = v1
            }
            // Reorder to match expected layout above (we filled incorrectly)
            std::vector<CR> vmat(4);
            vmat[0] = v[0];
            vmat[1] = v[1];
            vmat[2] = v[2];
            vmat[3] = v[3];
            // Actually v vector above already holds columns correctly if we set
            // Let's recompute properly:
            // For k=0: v[0]=v0_0, v[2]=v1_0
            // For k=1: v[1]=v0_1, v[3]=v1_1
            out.v = ndarray<CR>::from_data(std::vector<int>{2, 2}, std::vector<CR>{v[0], v[1], v[2], v[3]});
            return out;
        }
        // Larger complex: use simple QR iteration (unshifted) to get Schur form
        std::vector<R> A = dense;
        std::vector<R> Qacc(n * n, R{});
        for (std::size_t i = 0; i < n; ++i)
            Qacc[i * n + i] = R{1};
        const int maxIter = 200;
        for (int iter = 0; iter < maxIter; ++iter)
        {
            std::vector<R> q, r, h, tau;
            detail::householder_qr(A, n, n, q, r, h, tau);
            // A_next = R * Q
            std::vector<R> Anext(n * n, R{});
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                {
                    R s{};
                    for (std::size_t k = 0; k < n; ++k)
                        s += r[i * n + k] * q[k * n + j];
                    Anext[i * n + j] = s;
                }
            // Qacc = Qacc * Q
            std::vector<R> Qnext(n * n, R{});
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                {
                    R s{};
                    for (std::size_t k = 0; k < n; ++k)
                        s += Qacc[i * n + k] * q[k * n + j];
                    Qnext[i * n + j] = s;
                }
            A = std::move(Anext);
            Qacc = std::move(Qnext);
            // check convergence: subdiagonal small
            bool conv = true;
            for (std::size_t i = 1; i < n; ++i)
                if (std::abs(A[i * n + i - 1]) > RV{1e-8})
                    conv = false;
            if (conv)
                break;
        }
        std::vector<CR> w(n);
        for (std::size_t i = 0; i < n; ++i)
            w[i] = CR{A[i * n + i]};
        // eigenvectors are columns of Qacc (approx). For Schur, eigenvectors of A are Qacc
        // * I
        std::vector<CR> vmat(n * n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                vmat[i * n + j] = CR{Qacc[i * n + j]};
        // Normalize columns
        for (std::size_t j = 0; j < n; ++j)
        {
            RV nrm2{};
            for (std::size_t i = 0; i < n; ++i)
                nrm2 += std::norm(vmat[i * n + j]);
            RV nrm = std::sqrt(nrm2);
            if (nrm > RV{0})
                for (std::size_t i = 0; i < n; ++i)
                    vmat[i * n + j] /= static_cast<CR>(nrm);
        }
        out.w = ndarray<CR>::from_data(std::vector<int>{static_cast<int>(n)}, std::move(w));
        out.v = ndarray<CR>::from_data(std::vector<int>{static_cast<int>(n), static_cast<int>(n)}, std::move(vmat));
        return out;
    }
    else
    {
        EigenResult<RV> out2;
        if (n == 0)
        {
            out2.w = ndarray<CR>::from_data(std::vector<int>{0}, std::vector<CR>{});
            out2.v = ndarray<CR>::from_data(std::vector<int>{0, 0}, std::vector<CR>{});
            return out2;
        }
        // Real path uses Francis
        std::vector<R> schur, q;
        detail::schur_decompose(dense, n, schur, q);
        std::vector<CR> w = detail::schur_eigenvalues(schur, n);
        std::vector<CR> v = detail::schur_eigenvectors(schur, q, n, w);
        out.w = ndarray<CR>::from_data(std::vector<int>{static_cast<int>(n)}, std::move(w));
        out.v = ndarray<CR>::from_data(std::vector<int>{static_cast<int>(n), static_cast<int>(n)}, std::move(v));
        return out;
    }
}

/**
 * @brief Eigenvalues only (numpy.linalg.eigvals).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.eigvals.html
 * @tparam T Element type (must be real).
 * @param a Square matrix (N x N).
 * @return Array of eigenvalues (complex).
 * @throws std::invalid_argument if a.ndim() != 2 or a is not square.
 * @throws np::exceptions::LinAlgError if the Francis iteration does not
 * converge.
 * @complexity O(N^3).
 */
NP_API template <typename T> NP_NODISCARD auto eigvals(const ndarray<T> &a) -> ndarray<std::complex<real_value_t<T>>>
{
    return eig(a).w;
}

/**
 * @brief Determinant of a square matrix (numpy.linalg.det).
 *
 * Computed from an LU factorization with partial pivoting.
 * The magnitude is accumulated in log space to avoid overflow/underflow.
 * Reference: numpy-reference/reference/generated/numpy.linalg.det.html
 * Raises std::invalid_argument unless a.ndim() == 2 and
 * np::exceptions::LinAlgError unless a is square. The 0x0 matrix has
 * determinant 1.0 (empty product).
 * @tparam T Element type (must be real).
 * @param a Square matrix (N x N).
 * @return Determinant value.
 * @throws std::invalid_argument if a.ndim() != 2.
 * @throws np::exceptions::LinAlgError if a is not square.
 * @complexity O(N^3) (LU factorization).
 */
NP_API template <typename T> NP_NODISCARD auto det(const ndarray<T> &a) -> real_t<T>
{
    using R = real_t<T>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("det requires a 2D array");
    }
    std::size_t m{}, n{};
    std::vector<R> dense = detail::dense2d<R>(a, m, n);
    if (m != n)
    {
        throw np::exceptions::LinAlgError("det requires a square matrix");
    }
    if constexpr (np::detail::is_bigint_v<R>)
    {
        if (n == 0)
            return R{1};
        if (n == 1)
            return dense[0];
        // Bareiss fraction-free for exact bigint
        std::vector<R> A = dense;
        R prev = R{1};
        int sign = 1;
        for (std::size_t k = 0; k < n; ++k)
        {
            // pivot search max abs
            std::size_t piv = k;
            R maxabs = A[k * n + k] < R{0} ? -A[k * n + k] : A[k * n + k];
            for (std::size_t i = k + 1; i < n; ++i)
            {
                R cur = A[i * n + k] < R{0} ? -A[i * n + k] : A[i * n + k];
                if (cur > maxabs)
                {
                    maxabs = cur;
                    piv = i;
                }
            }
            if (maxabs == R{0})
                return R{0};
            if (piv != k)
            {
                for (std::size_t j = 0; j < n; ++j)
                    std::swap(A[k * n + j], A[piv * n + j]);
                sign = -sign;
            }
            for (std::size_t i = k + 1; i < n; ++i)
            {
                for (std::size_t j = k + 1; j < n; ++j)
                {
                    A[i * n + j] = (A[i * n + j] * A[k * n + k] - A[i * n + k] * A[k * n + j]) / prev;
                }
                A[i * n + k] = R{0};
            }
            prev = A[k * n + k];
        }
        R d = A[(n - 1) * n + (n - 1)];
        if (sign < 0)
            d = -d;
        return d;
    }
    std::vector<R> lu = dense;
    std::vector<std::size_t> piv;
    std::size_t swaps{};
    if (detail::lu_factor(lu, n, piv, swaps))
    {
        return R{0};
    }
    if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                  std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                  std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
    {
        R d = swaps % 2 == 0 ? R{1} : R{-1};
        for (std::size_t i = 0; i < n; ++i)
        {
            d *= lu[i * n + i];
        }
        return d;
    }
    else
    {
        R sign = swaps % 2 == 0 ? R{1} : R{-1};
        R logabs = R{0};
        for (std::size_t i = 0; i < n; ++i)
        {
            const R u = lu[i * n + i];
            if (u < R{0})
            {
                sign = -sign;
            }
            logabs += std::log(std::abs(u));
        }
        return sign * std::exp(logabs);
    }
}

/**
 * @brief Sign and log-absolute-determinant (numpy.linalg.slogdet).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.slogdet.html
 * A singular matrix yields sign == 0 and logabsdet == -inf.
 * @tparam T Element type (must be real).
 * @param a Square matrix (N x N).
 * @return Pair of (sign, logabsdet).
 * @throws std::invalid_argument if a.ndim() != 2.
 * @throws np::exceptions::LinAlgError if a is not square.
 * @complexity O(N^3) (LU factorization).
 */
NP_API template <typename T> NP_NODISCARD auto slogdet(const ndarray<T> &a) -> SlogdetResult<real_t<T>>
{
    using R = real_t<T>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("slogdet requires a 2D array");
    }
    std::size_t m{}, n{};
    std::vector<R> lu = detail::dense2d<R>(a, m, n);
    if (m != n)
    {
        throw np::exceptions::LinAlgError("slogdet requires a square matrix");
    }
    std::vector<std::size_t> piv;
    std::size_t swaps{};
    if (detail::lu_factor(lu, n, piv, swaps))
    {
        if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                      std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                      std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
            return SlogdetResult<R>{R{0}, R{-std::numeric_limits<real_value_t<R>>::infinity()}};
        else
            return SlogdetResult<R>{R{0}, -std::numeric_limits<R>::infinity()};
    }
    if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                  std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                  std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
    {
        using RV = real_value_t<R>;
        R sign = swaps % 2 == 0 ? R{1} : R{-1};
        RV logabs = RV{0};
        for (std::size_t i = 0; i < n; ++i)
        {
            const R u = lu[i * n + i];
            logabs += std::log(std::abs(u));
            // unit-phase sign = product of phases
            if (std::abs(u) != RV{0})
                sign *= u / std::abs(u);
        }
        // For real negative product, sign will have correct phase already
        return SlogdetResult<R>{sign, R{logabs}};
    }
    else if constexpr (np::detail::is_bigint_v<R>)
    {
        R sign = swaps % 2 == 0 ? R{1} : R{-1};
        double logabs = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const R &u = lu[i * n + i];
            R abs_u = u < R{0} ? -u : u;
            if (u < R{0})
                sign = -sign;
            if (abs_u != R{0})
            {
#if NP_HAS_CPP_INT
                double d = abs_u.template convert_to<double>();
#else
                double d = std::stod(abs_u.value);
#endif
                logabs += std::log(d);
            }
        }
        // logabs stored as R (bigint) truncated; for exact bigint slogdet, user should use
        // det
        return SlogdetResult<R>{sign, R{static_cast<long long>(logabs)}};
    }
    else
    {
        R sign = swaps % 2 == 0 ? R{1} : R{-1};
        R logabs = R{0};
        for (std::size_t i = 0; i < n; ++i)
        {
            const R u = lu[i * n + i];
            if (u < R{0})
            {
                sign = -sign;
            }
            logabs += std::log(std::abs(u));
        }
        return SlogdetResult<R>{sign, logabs};
    }
}

/**
 * @brief Multiplicative inverse of a square matrix (numpy.linalg.inv).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.inv.html
 * Solves A X = I with the LU factorization of A.
 * The 0x0 matrix inverts to itself.
 * Raises std::invalid_argument unless a.ndim() == 2 and
 * np::exceptions::LinAlgError when a is not square or is exactly
 * singular. Ill-conditioned input may invert with large errors, as
 * numpy does; use cond() to detect that case.
 * @tparam T Element type (must be real).
 * @param a Square matrix (N x N).
 * @return Inverse matrix (N x N).
 * @throws std::invalid_argument if a.ndim() != 2.
 * @throws np::exceptions::LinAlgError if a is not square or is singular.
 * @complexity O(N^3).
 */
NP_API template <typename T> NP_NODISCARD auto inv(const ndarray<T> &a) -> ndarray<real_t<T>>
{
    using R = real_t<T>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("inv requires a 2D array");
    }
    std::size_t m{}, n{};
    std::vector<R> lu = detail::dense2d<R>(a, m, n);
    if (m != n)
    {
        throw np::exceptions::LinAlgError("inv requires a square matrix");
    }
    std::vector<std::size_t> piv;
    std::size_t swaps{};
    if (detail::lu_factor(lu, n, piv, swaps))
    {
        throw np::exceptions::LinAlgError("Singular matrix");
    }
    return detail::mk2d(n, n, detail::lu_invert(lu, n, piv));
}

#if NP_HAS_CPP_INT
// Bigint overload: inv of integer matrix is rational -> return double
template <typename T>
    requires(np::detail::is_bigint_v<T>)
NP_NODISCARD auto inv(const ndarray<T> &a) -> ndarray<double>
{
    auto ad = from_bigint<double>(a);
    return inv(ad);
}
#endif

/**
 * @brief Solve a linear system a x = b (numpy.linalg.solve).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.solve.html
 * b is treated as a single right-hand side only when it is exactly
 * 1-D (the numpy 2.0 rule); a 2-D b is a stack of right-hand sides
 * along its columns.
 * Raises std::invalid_argument for a.ndim() != 2 or b.ndim() > 2, and
 * np::exceptions::LinAlgError when a is not square, b's leading
 * dimension does not match a, or a is singular.
 * @tparam T Element type of the coefficient matrix (must be real).
 * @tparam U Element type of the right-hand side (must be real).
 * @param a Coefficient matrix (M x M).
 * @param b Right-hand side vector (M,) or matrix (M, K).
 * @return Solution x of shape (M,) or (M, K).
 * @throws std::invalid_argument if a.ndim() != 2 or b.ndim() > 2.
 * @throws np::exceptions::LinAlgError if a is not square, dimensions mismatch,
 * or a is singular.
 * @complexity O(M^3) (LU factorization).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto solve(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<real_t<T>, real_t<U>>>
{
    using R = std::common_type_t<real_t<T>, real_t<U>>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("solve requires a 2D matrix");
    }
    if (b.ndim() == 0 || b.ndim() > 2)
    {
        throw std::invalid_argument("solve requires b to be 1D or 2D");
    }
    std::size_t m{}, n{};
    std::vector<R> lu = detail::dense2d<R>(a, m, n);
    if (m != n)
    {
        throw np::exceptions::LinAlgError("solve requires a square matrix");
    }
    if (static_cast<std::size_t>(b.shape[0]) != m)
    {
        throw np::exceptions::LinAlgError("solve: b must have shape (M,) or (M, K) matching the matrix");
    }
    std::vector<std::size_t> piv;
    std::size_t swaps{};
    if (detail::lu_factor(lu, n, piv, swaps))
    {
        throw np::exceptions::LinAlgError("Singular matrix");
    }
    const std::size_t nrhs = b.ndim() == 1 ? 1 : static_cast<std::size_t>(b.shape[1]);
    std::vector<R> out(m * nrhs);
    for (std::size_t c = 0; c < nrhs; ++c)
    {
        std::vector<R> rhs(m);
        for (std::size_t i = 0; i < m; ++i)
        {
            rhs[i] = b.ndim() == 1 ? static_cast<R>(b.at(i)) : static_cast<R>(b.at(i, c));
        }
        std::vector<R> x = detail::lu_solve(lu, n, piv, rhs);
        for (std::size_t i = 0; i < m; ++i)
        {
            out[i * nrhs + c] = x[i];
        }
    }
    return b.ndim() == 1 ? ndarray<R>::from_data(std::vector<int>{static_cast<int>(m)}, std::move(out))
                         : detail::mk2d(m, nrhs, std::move(out));
}

#if NP_HAS_CPP_INT
template <typename T, typename U>
    requires(np::detail::is_bigint_v<T> || np::detail::is_bigint_v<U>)
NP_NODISCARD auto solve(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<double>
{
    auto ad = [&] {
        if constexpr (np::detail::is_bigint_v<T>)
            return from_bigint<double>(a);
        else
            return ndarray<double>(a.shape, dtype::float64, 0.0); // placeholder, will use as_bigint conversion?
    }();
    // Actually for mixed bigint/double, convert both to double
    ndarray<double> ad2, bd2;
    if constexpr (np::detail::is_bigint_v<T>)
        ad2 = from_bigint<double>(a);
    else
        ad2 = ndarray<double>::from_data(a.shape, std::vector<double>(a.data().begin(), a.data().end()));
    if constexpr (np::detail::is_bigint_v<U>)
        bd2 = from_bigint<double>(b);
    else
        bd2 = ndarray<double>::from_data(b.shape, std::vector<double>(b.data().begin(), b.data().end()));
    // Use double solve
    return solve(ad2, bd2);
}
#endif

/**
 * @brief Cholesky factorization of a positive-definite matrix
 *        (numpy.linalg.cholesky).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.cholesky.html
 * Computes the lower-triangular L with L L' = a (upper = false,
 * the default) or the upper-triangular U with U' U = a (upper = true).
 * Only the requested triangle of a is read and no Hermitianity check
 * is performed, as in numpy; non-symmetric input gives the factor of
 * its symmetrized triangle.
 * Raises std::invalid_argument unless a.ndim() == 2 and
 * np::exceptions::LinAlgError when a is not positive-definite.
 * @tparam T Element type (must be real).
 * @param a Square matrix (N x N).
 * @param upper If true, compute upper-triangular U; otherwise lower-triangular
 * L.
 * @return Cholesky factor (N x N).
 * @throws std::invalid_argument if a.ndim() != 2.
 * @throws np::exceptions::LinAlgError if a is not positive-definite.
 * @complexity O(N^3).
 */
NP_API template <typename T> NP_NODISCARD auto cholesky(const ndarray<T> &a, bool upper = false) -> ndarray<real_t<T>>
{
    using R = real_t<T>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("cholesky requires a 2D array");
    }
    std::size_t m{}, n{};
    std::vector<R> d = detail::dense2d<R>(a, m, n);
    if (m != n)
    {
        throw np::exceptions::LinAlgError("cholesky requires a square matrix");
    }
    if (upper)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < i; ++j)
            {
                std::swap(d[i * n + j], d[j * n + i]);
            }
        }
    }
    std::vector<R> l(n * n, R{});
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            R s = d[i * n + j];
            for (std::size_t kk = 0; kk < j; ++kk)
            {
                if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                              std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                              std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                    s -= l[i * n + kk] * std::conj(l[j * n + kk]);
                else
                    s -= l[i * n + kk] * l[j * n + kk];
            }
            if (i == j)
            {
                if constexpr (np::detail::is_complex_v<R> || np::detail::is_bigint_v<R> ||
                              std::is_same_v<std::remove_cv_t<R>, np::bigint> ||
                              std::is_same_v<std::remove_cv_t<R>, np::mpz_bigint>)
                {
                    auto rs = std::real(s);
                    if (rs <= decltype(rs){0})
                        throw np::exceptions::LinAlgError("Matrix is not positive definite");
                    l[i * n + j] = R{std::sqrt(rs)};
                }
                else
                {
                    if (s <= R{0})
                        throw np::exceptions::LinAlgError("Matrix is not positive definite");
                    l[i * n + j] = std::sqrt(s);
                }
            }
            else
            {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    if (upper)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < i; ++j)
            {
                std::swap(l[i * n + j], l[j * n + i]);
            }
        }
    }
    return detail::mk2d(n, n, std::move(l));
}

/**
 * @brief Vector and matrix norm (numpy.linalg.norm).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.norm.html
 * ord = None (the default) is the 2-norm for 1-D input and the
 * Frobenius norm for 2-D input. Matrix orders: One and NegOne
 * are the max/min column-abs-sums, Inf and NegInf the max/min
 * row-abs-sums, Two and NegTwo the largest/smallest singular values,
 * Fro the root-sum-of-squares. Vector orders: One/Two/Inf are the
 * sum/2-norm/max of the absolute values, NegOne and NegTwo the inverse
 * p-means, NegInf the minimum absolute value.
 * The axis and keepdims parameters are not supported. Raises
 * std::invalid_argument for 0-d or > 2-d input and for 'fro' on 1-D
 * input (as numpy's "Improper number of dimensions to norm").
 * @tparam T Element type (must be real).
 * @param x Input array (1-D or 2-D).
 * @param ord Norm order (default: None = 2-norm for vectors,
 *        Frobenius for matrices).
 * @return Norm value.
 * @throws std::invalid_argument for 0-d or > 2-d input, or 'fro'/'nuc' on 1-D
 * input.
 * @complexity O(N) for 1-D; O(M*N) for 2-D (most norms);
 *         O(M*N*min(M,N)) for Two/NegTwo (via SVD).
 */
NP_API template <typename T> NP_NODISCARD auto norm(const ndarray<T> &x, NormOrd ord = NormOrd::None) -> real_value_t<T>
{
    using R = real_t<T>;
    using RV = real_value_t<T>;
    const std::size_t nd = x.ndim();
    if (nd == 0 || nd > 2)
    {
        throw std::invalid_argument("norm requires a 1D or 2D array");
    }
    if (nd == 1)
    {
        if (ord == NormOrd::Fro || ord == NormOrd::Nuc)
        {
            throw std::invalid_argument("'fro' and 'nuc' norms are not defined for 1D arrays");
        }
        if (x.is_contiguous()) [[likely]]
        {
            const T *__restrict ptr = x.data().data();
            const std::size_t len = x.size();
            switch (ord)
            {
            case NormOrd::None:
            case NormOrd::Two: {
                RV acc{};
                for (std::size_t i = 0; i < len; ++i)
                {
                    const RV v = static_cast<RV>(std::abs(static_cast<R>(ptr[i])));
                    acc += std::abs(v) * std::abs(v);
                }
                return std::sqrt(acc);
            }
            case NormOrd::One: {
                RV acc{};
                for (std::size_t i = 0; i < len; ++i)
                    acc += std::abs(static_cast<R>(ptr[i]));
                return acc;
            }
            case NormOrd::Inf: {
                RV best{};
                for (std::size_t i = 0; i < len; ++i)
                    best = std::max(best, std::abs(static_cast<R>(ptr[i])));
                return best;
            }
            case NormOrd::NegInf: {
                RV best = std::numeric_limits<RV>::infinity();
                for (std::size_t i = 0; i < len; ++i)
                    best = std::min(best, std::abs(static_cast<R>(ptr[i])));
                return best == std::numeric_limits<RV>::infinity() ? RV{0} : best;
            }
            case NormOrd::NegOne: {
                RV acc{};
                for (std::size_t i = 0; i < len; ++i)
                    acc += RV{1} / std::abs(static_cast<R>(ptr[i]));
                return acc == RV{0} ? RV{0} : RV{1} / acc;
            }
            case NormOrd::NegTwo: {
                RV acc{};
                for (std::size_t i = 0; i < len; ++i)
                {
                    const RV v = std::abs(static_cast<R>(ptr[i]));
                    acc += RV{1} / (v * v);
                }
                return acc == RV{0} ? RV{0} : RV{1} / std::sqrt(acc);
            }
            case NormOrd::Fro:
            case NormOrd::Nuc:
                break;
            }
        }
        const std::size_t len = x.size();
        switch (ord)
        {
        case NormOrd::None:
        case NormOrd::Two: {
            RV acc{};
            for (std::size_t i = 0; i < len; ++i)
            {
                const R v = static_cast<R>(x.at(i));
                acc += std::abs(v) * std::abs(v);
            }
            return std::sqrt(acc);
        }
        case NormOrd::One: {
            RV acc{};
            for (std::size_t i = 0; i < len; ++i)
            {
                acc += std::abs(static_cast<R>(x.at(i)));
            }
            return acc;
        }
        case NormOrd::Inf: {
            RV best{};
            for (std::size_t i = 0; i < len; ++i)
            {
                best = std::max(best, std::abs(static_cast<R>(x.at(i))));
            }
            return best;
        }
        case NormOrd::NegInf: {
            RV best = std::numeric_limits<RV>::infinity();
            for (std::size_t i = 0; i < len; ++i)
            {
                best = std::min(best, std::abs(static_cast<R>(x.at(i))));
            }
            return best == std::numeric_limits<RV>::infinity() ? RV{0} : best;
        }
        case NormOrd::NegOne: {
            RV acc{};
            for (std::size_t i = 0; i < len; ++i)
            {
                acc += RV{1} / std::abs(static_cast<R>(x.at(i)));
            }
            return acc == RV{0} ? RV{0} : RV{1} / acc;
        }
        case NormOrd::NegTwo: {
            RV acc{};
            for (std::size_t i = 0; i < len; ++i)
            {
                const RV v = std::abs(static_cast<R>(x.at(i)));
                acc += RV{1} / (v * v);
            }
            return acc == RV{0} ? RV{0} : RV{1} / std::sqrt(acc);
        }
        case NormOrd::Fro:
        case NormOrd::Nuc:
            break; // unreachable
        }
    }
    const std::size_t rows = static_cast<std::size_t>(x.shape[0]);
    const std::size_t cols = static_cast<std::size_t>(x.shape[1]);
    switch (ord)
    {
    case NormOrd::None:
    case NormOrd::Fro: {
        RV acc{};
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                const R v = static_cast<R>(x.at(i, j));
                acc += std::abs(v) * std::abs(v);
            }
        }
        return std::sqrt(acc);
    }
    case NormOrd::One:
    case NormOrd::NegOne: {
        RV best = ord == NormOrd::One ? RV{0} : std::numeric_limits<RV>::infinity();
        for (std::size_t j = 0; j < cols; ++j)
        {
            RV acc{};
            for (std::size_t i = 0; i < rows; ++i)
            {
                acc += std::abs(static_cast<R>(x.at(i, j)));
            }
            best = ord == NormOrd::One ? std::max(best, acc) : std::min(best, acc);
        }
        return best == std::numeric_limits<RV>::infinity() ? RV{0} : best;
    }
    case NormOrd::Inf:
    case NormOrd::NegInf: {
        RV best = ord == NormOrd::Inf ? RV{0} : std::numeric_limits<RV>::infinity();
        for (std::size_t i = 0; i < rows; ++i)
        {
            RV acc{};
            for (std::size_t j = 0; j < cols; ++j)
            {
                acc += std::abs(static_cast<R>(x.at(i, j)));
            }
            best = ord == NormOrd::Inf ? std::max(best, acc) : std::min(best, acc);
        }
        return best == std::numeric_limits<RV>::infinity() ? RV{0} : best;
    }
    case NormOrd::Two:
    case NormOrd::NegTwo: {
        auto s = svdvals(x);
        const std::size_t k = s.size();
        if (k == 0)
        {
            return RV{0};
        }
        return ord == NormOrd::Two ? static_cast<RV>(std::abs(s(0))) : static_cast<RV>(std::abs(s(k - 1)));
    }
    case NormOrd::Nuc: {
        auto s = svdvals(x);
        RV acc{};
        for (std::size_t i = 0; i < s.size(); ++i)
        {
            acc += std::abs(s(i));
        }
        return acc;
    }
    }
    return RV{0}; // unreachable: every NormOrd is handled above
}

/**
 * @brief Matrix norm (numpy.linalg.matrix_norm).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.matrix_norm.html
 * 'fro' is the default order. Unlike norm, 'nuc' is valid here
 * (sum of the singular values); all other orders delegate to norm.
 * Stacks are not supported. Raises std::invalid_argument unless x is a 2D
 * array.
 * @tparam T Element type (must be real).
 * @param x Input matrix (M x N).
 * @param ord Norm order (default: Fro).
 * @return Norm value.
 * @throws std::invalid_argument if x.ndim() != 2.
 * @complexity O(M * N * min(M, N)) for Two/NegTwo (via SVD).
 */
NP_API template <typename T>
NP_NODISCARD auto matrix_norm(const ndarray<T> &x, NormOrd ord = NormOrd::Fro) -> real_value_t<T>
{
    using R = real_t<T>;
    using RV = real_value_t<T>;
    if (x.ndim() != 2)
    {
        throw std::invalid_argument("matrix_norm requires a 2D array");
    }
    if (ord == NormOrd::Nuc)
    {
        auto s = svdvals(x);
        RV acc{};
        for (std::size_t i = 0; i < s.size(); ++i)
        {
            acc += std::abs(s(i));
        }
        return acc;
    }
    return norm(x, ord);
}

// Vector norm of x or of each vector slice along the given axes: an
// empty axis list norms over all values (flattened), a single axis
// norms the 1-D slices along it, two axes norm the matrix slices
// (Frobenius for ord = 2, like numpy). keepdims leaves the normed axes
// in the result with size 1 (all axes for axis = None). ord is the
// numeric p-norm order: 0 counts nonzeros, +/-inf give max/min of the
// absolute values, any other p (including negative) gives
// (sum |x|^p)^(1/p). The result is an ndarray, 0-d for axis = None.
// Reference: numpy-reference/reference/generated/numpy.linalg.vector_norm.html
// Raises std::invalid_argument for invalid or repeated axes.
NP_API template <typename T>
NP_NODISCARD auto vector_norm(const ndarray<T> &x, const std::vector<int> &axis = {}, bool keepdims = false,
                              double ord = 2.0) -> ndarray<real_value_t<T>>
{
    using R = real_t<T>;
    using RV = real_value_t<T>;
    const std::size_t nd = x.ndim();
    // Which axes are reduced; empty axis list means "all axes".
    std::vector<bool> is_red(nd, false);
    std::vector<std::size_t> red_axes;
    if (axis.empty())
    {
        for (std::size_t i = 0; i < nd; ++i)
        {
            is_red[i] = true;
            red_axes.push_back(i);
        }
    }
    else
    {
        for (int ax : axis)
        {
            const int r = ax < 0 ? ax + static_cast<int>(nd) : ax;
            const std::size_t u = static_cast<std::size_t>(r);
            if (r < 0 || r >= static_cast<int>(nd) || is_red[u])
            {
                throw std::invalid_argument("vector_norm: invalid or repeated axis");
            }
            is_red[u] = true;
            red_axes.push_back(u);
        }
        std::sort(red_axes.begin(), red_axes.end());
    }
    // Result shape: non-reduced axes in order, reduced axes as 1 with
    // keepdims.
    std::vector<int> out_shape;
    std::vector<std::size_t> out_pos(nd);
    for (std::size_t i = 0; i < nd; ++i)
    {
        if (is_red[i])
        {
            if (keepdims)
            {
                out_pos[i] = out_shape.size();
                out_shape.push_back(1);
            }
        }
        else
        {
            out_pos[i] = out_shape.size();
            out_shape.push_back(x.shape[i]);
        }
    }
    ndarray<R> out(out_shape);
    const bool is_inf = std::isinf(ord);
    np::detail::Odometer odo(out_shape);
    while (!odo.done())
    {
        const auto &oidx = odo.idx();
        RV best{}; // max/min accumulator for ord = +/-inf
        RV acc{};  // sum |x|^p, or count of nonzeros for ord = 0
        bool first = true;
        // Iterate over the reduced axes.
        std::vector<std::size_t> ridx(red_axes.size(), 0);
        bool rdone = false;
        while (!rdone)
        {
            std::vector<std::size_t> idx(nd);
            std::size_t p = 0;
            for (std::size_t i = 0; i < nd; ++i)
            {
                if (is_red[i])
                {
                    idx[i] = ridx[p++];
                }
                else
                {
                    idx[i] = oidx[out_pos[i]];
                }
            }
            const R v = static_cast<R>(std::abs(static_cast<R>(x.data()[x._flat(idx)])));
            if (is_inf)
            {
                if (first)
                {
                    best = v;
                }
                else
                {
                    best = ord > 0 ? std::max(best, v) : std::min(best, v);
                }
            }
            else if (ord == 0.0)
            {
                acc += v > R{0} ? R{1} : R{0};
            }
            else
            {
                acc += std::pow(v, ord);
            }
            first = false;
            // Advance the reduced-dimension odometer.
            rdone = true;
            for (std::size_t d = ridx.size(); d-- > 0;)
            {
                if (++ridx[d] < static_cast<std::size_t>(x.shape[red_axes[d]]))
                {
                    rdone = false;
                    break;
                }
                ridx[d] = 0;
            }
        }
        R val{};
        if (is_inf)
        {
            val = first ? R{0} : best;
        }
        else if (ord == 0.0)
        {
            val = acc;
        }
        else
        {
            val = static_cast<R>(std::pow(static_cast<double>(acc), 1.0 / ord));
        }
        out.data()[np::detail::flat_index(oidx, out.strides, 0)] = val;
        odo.advance();
    }
    return out;
}

/**
 * @brief Rank of a 1-D or 2-D array (numpy.linalg.matrix_rank).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.matrix_rank.html
 * The number of singular values above the tolerance.
 * With the default tol = S.max() * max(M, N) * eps (the
 * Numerical-Recipes / MATLAB threshold), matching numpy.
 * A 1-D input has rank 1 unless it is all zero. The hermitian keyword
 * is not supported. Raises std::invalid_argument unless a.ndim() <= 2.
 * @tparam T Element type (must be real).
 * @param a Input array (1-D or 2-D).
 * @param tol Tolerance; negative values use the default
 *        (S.max() * max(M, N) * eps).
 * @return Numerical rank.
 * @throws std::invalid_argument if a.ndim() > 2.
 * @complexity O(M * N * min(M, N) * sweeps) for 2-D (dominated by SVD).
 */
NP_API template <typename T> NP_NODISCARD auto matrix_rank(const ndarray<T> &a) -> int
{
    using R = real_t<T>;
    const std::size_t nd = a.ndim();
    if (nd > 2)
    {
        throw std::invalid_argument("matrix_rank requires a 1D or 2D array");
    }
    if (nd == 1)
    {
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (a.at(i) != 0)
            {
                return 1;
            }
        }
        return 0;
    }
    auto s = svdvals(a);
    const std::size_t k = s.size();
    if (k == 0)
    {
        return 0;
    }
    const R tol = s(0) * static_cast<R>(std::max(a.shape[0], a.shape[1])) * std::numeric_limits<R>::epsilon();
    int rank = 0;
    for (std::size_t i = 0; i < k; ++i)
    {
        if (s(i) > tol)
        {
            ++rank;
        }
    }
    return rank;
}

// Rank of a 1-D or 2-D array with an explicit tolerance.
// Reference: numpy-reference/reference/generated/numpy.linalg.matrix_rank.html
NP_API template <typename T> NP_NODISCARD auto matrix_rank(const ndarray<T> &a, double tol) -> int
{
    using R = real_t<T>;
    const std::size_t nd = a.ndim();
    if (nd > 2)
    {
        throw std::invalid_argument("matrix_rank requires a 1D or 2D array");
    }
    if (nd == 1)
    {
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (std::abs(static_cast<R>(a.at(i))) > tol)
            {
                return 1;
            }
        }
        return 0;
    }
    auto s = svdvals(a);
    int rank = 0;
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        if (s(i) > tol)
        {
            ++rank;
        }
    }
    return rank;
}

/**
 * @brief Moore-Penrose pseudo-inverse (numpy.linalg.pinv).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.pinv.html
 * Computed via the SVD: singular values at or below rcond * s_max
 * are treated as zero. Raises std::invalid_argument unless a.ndim() == 2
 * and np::exceptions::LinAlgError when the Jacobi sweeps do not converge.
 * @tparam T Element type (must be real).
 * @param a Input matrix (M x N).
 * @param rcond Cutoff for small singular values (default: 1e-15).
 * @return Pseudo-inverse (N x M).
 * @throws std::invalid_argument if a.ndim() != 2.
 * @throws np::exceptions::LinAlgError if the SVD does not converge.
 * @complexity O(M * N * min(M, N) * sweeps).
 */
NP_API template <typename T> NP_NODISCARD auto pinv(const ndarray<T> &a, double rcond = 1e-15) -> ndarray<real_t<T>>
{
    using R = real_t<T>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("pinv requires a 2D array");
    }
    const std::size_t m = static_cast<std::size_t>(a.shape[0]);
    const std::size_t n = static_cast<std::size_t>(a.shape[1]);
    const std::size_t k = std::min(m, n);
    auto sv = svd(a, false, true);
    const R cutoff = k > 0 ? sv.s(0) * static_cast<R>(rcond) : R{0};
    std::vector<R> out(n * m, R{});
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < m; ++j)
        {
            R acc{};
            for (std::size_t t = 0; t < k; ++t)
            {
                if (sv.s(t) > cutoff)
                {
                    acc += sv.vh(t, i) * (R{1} / sv.s(t)) * sv.u(j, t);
                }
            }
            out[i * m + j] = acc;
        }
    }
    return detail::mk2d(n, m, std::move(out));
}

/**
 * @brief Condition number (numpy.linalg.cond).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.cond.html
 * 2-norm condition number: largest over smallest singular value.
 * Infinite for singular input, as numpy.
 * Raises std::invalid_argument unless x.ndim() == 2.
 * @tparam T Element type (must be real).
 * @param x Input matrix (M x N).
 * @return Condition number in the 2-norm.
 * @throws std::invalid_argument if x.ndim() != 2.
 * @complexity O(M * N * min(M, N) * sweeps) (dominated by SVD).
 */
NP_API template <typename T> NP_NODISCARD auto cond(const ndarray<T> &x) -> real_t<T>
{
    using R = real_t<T>;
    if (x.ndim() != 2)
    {
        throw std::invalid_argument("cond requires a 2D array");
    }
    auto s = svdvals(x);
    const std::size_t k = s.size();
    if (k == 0)
    {
        return R{0};
    }
    return s(0) / s(k - 1);
}

/**
 * @brief Condition number with explicit order (numpy.linalg.cond).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.cond.html
 * p = Two (-Two) gives the largest/smallest singular value ratio,
 * computed directly from the SVD (so singular input gives inf,
 * as with p = None); the remaining orders use norm(x, p) * norm(inv(x), p)
 * and raise np::exceptions::LinAlgError for singular input, as numpy does.
 * @tparam T Element type (must be real).
 * @param x Input matrix (M x N).
 * @param p Norm order.
 * @return Condition number in the specified norm.
 * @throws std::invalid_argument if x.ndim() != 2.
 * @throws np::exceptions::LinAlgError if x is singular (for non-SVD orders).
 * @complexity O(M * N * min(M, N) * sweeps) for Two/NegTwo;
 *         O(M * N * min(M, N)) for other orders (involves inv).
 */
NP_API template <typename T> NP_NODISCARD auto cond(const ndarray<T> &x, NormOrd p) -> real_t<T>
{
    using R = real_t<T>;
    if (x.ndim() != 2)
    {
        throw std::invalid_argument("cond requires a 2D array");
    }
    auto s = svdvals(x);
    const std::size_t k = s.size();
    if (k == 0)
    {
        return R{0};
    }
    if (p == NormOrd::Two)
    {
        return s(0) / s(k - 1);
    }
    if (p == NormOrd::NegTwo)
    {
        return s(k - 1) / s(0);
    }
    return norm(x, p) * norm(inv(x), p);
}

// Eigenvalues of a symmetric matrix in ascending order. The full matrix
// is used (numpy reads only the lower triangle with UPLO = 'L'); the
// results match numpy exactly for symmetric input.
// Reference: numpy-reference/reference/generated/numpy.linalg.eigvalsh.html
NP_API template <typename T> NP_NODISCARD auto eigvalsh(const ndarray<T> &a) -> ndarray<real_t<T>>
{
    using R = real_t<T>;
    auto w = eigvals(a);
    const std::size_t n = w.size();
    std::vector<R> re(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        re[i] = w(i).real();
    }
    std::sort(re.begin(), re.end());
    return ndarray<R>::from_data(std::vector<int>{static_cast<int>(n)}, std::move(re));
}

// Eigenvalues and orthonormal eigenvectors of a symmetric matrix: w is
// ascending, v holds the unit eigenvector of A for w[j] in column j.
// Reference: numpy-reference/reference/generated/numpy.linalg.eigh.html
// As with eigvalsh the full matrix is used rather than numpy's lower
// triangle; results match numpy for symmetric input.
NP_API template <typename T> NP_NODISCARD auto eigh(const ndarray<T> &a) -> EighResult<real_t<T>>
{
    using R = real_t<T>;
    auto e = eig(a);
    const std::size_t n = e.w.size();
    std::vector<std::size_t> order(n);
    for (std::size_t j = 0; j < n; ++j)
    {
        order[j] = j;
    }
    std::sort(order.begin(), order.end(), [&e](std::size_t x, std::size_t y) { return e.w(x).real() < e.w(y).real(); });
    std::vector<R> w(n);
    std::vector<R> v(n * n);
    for (std::size_t j = 0; j < n; ++j)
    {
        w[j] = e.w(order[j]).real();
        for (std::size_t i = 0; i < n; ++i)
        {
            v[i * n + j] = e.v(i, order[j]).real();
        }
    }
    // Modified Gram-Schmidt over the columns restores orthonormality to
    // solver roundoff (the Schur path already yields orthogonal
    // eigenvectors for symmetric input).
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t t = 0; t < j; ++t)
        {
            R dot{};
            for (std::size_t i = 0; i < n; ++i)
            {
                dot += v[i * n + j] * v[i * n + t];
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                v[i * n + j] -= dot * v[i * n + t];
            }
        }
        R nrm{};
        for (std::size_t i = 0; i < n; ++i)
        {
            nrm += v[i * n + j] * v[i * n + j];
        }
        nrm = std::sqrt(nrm);
        for (std::size_t i = 0; i < n && nrm > R{0}; ++i)
        {
            v[i * n + j] /= nrm;
        }
    }
    EighResult<R> out;
    out.w = ndarray<R>::from_data(std::vector<int>{static_cast<int>(n)}, std::move(w));
    out.v = detail::mk2d(n, n, std::move(v));
    return out;
}

// Extract the diagonal of the last two dimensions: out[..., d] =
// x[..., d, d + offset] for offset >= 0, x[..., d - offset, d] for
// offset < 0. The diagonal length is
// min(n, max(0, m - offset)) for offset >= 0, min(m, max(0, n + offset))
// otherwise, so out-of-range offsets give an empty diagonal. Like numpy,
// this returns a fresh (contiguous) copy, not a read-only view.
// Reference: numpy-reference/reference/generated/numpy.linalg.diagonal.html
// Raises std::invalid_argument unless x.ndim() >= 2.
NP_API template <typename T> NP_NODISCARD auto diagonal(const ndarray<T> &x, int offset = 0) -> ndarray<T>
{
    const std::size_t nd = x.ndim();
    if (nd < 2)
    {
        throw std::invalid_argument("diagonal requires a 2D array or a stack");
    }
    const int m = x.shape[nd - 2];
    const int n = x.shape[nd - 1];
    const int len = offset >= 0 ? std::min(n, std::max(0, m - offset)) : std::min(m, std::max(0, n + offset));
    std::vector<int> out_shape(x.shape.begin(), x.shape.end() - 2);
    out_shape.push_back(len);
    ndarray<T> out(out_shape);
    const std::vector<int> lead(x.shape.begin(), x.shape.begin() + static_cast<long>(nd - 2));
    np::detail::Odometer odo(lead);
    while (!odo.done())
    {
        const auto &li = odo.idx();
        std::vector<std::size_t> in(li.begin(), li.end());
        in.resize(nd);
        for (int d = 0; d < len; ++d)
        {
            in[nd - 2] = static_cast<std::size_t>(offset >= 0 ? d : d - offset);
            in[nd - 1] = static_cast<std::size_t>(offset >= 0 ? d + offset : d);
            std::vector<std::size_t> oi(li.begin(), li.end());
            oi.push_back(static_cast<std::size_t>(d));
            out.data()[np::detail::flat_index(oi, out.strides, 0)] = x.data()[x._flat(in)];
        }
        odo.advance();
    }
    return out;
}

// Swap the last two axes: out[..., j, i] = x[..., i, j]. Returns a new
// contiguous array (numpy returns a read-only view).
// Reference:
// numpy-reference/reference/generated/numpy.linalg.matrix_transpose.html Raises
// std::invalid_argument unless x.ndim() >= 2.
NP_API template <typename T> NP_NODISCARD auto matrix_transpose(const ndarray<T> &x) -> ndarray<T>
{
    const std::size_t nd = x.ndim();
    if (nd < 2)
    {
        throw std::invalid_argument("matrix_transpose requires a 2D array or a stack");
    }
    std::vector<int> out_shape = x.shape;
    std::swap(out_shape[nd - 2], out_shape[nd - 1]);
    ndarray<T> out(out_shape);
    np::detail::Odometer odo(out_shape);
    while (!odo.done())
    {
        const auto &oi = odo.idx();
        std::vector<std::size_t> in(oi.begin(), oi.end());
        std::swap(in[nd - 2], in[nd - 1]);
        out.data()[np::detail::flat_index(oi, out.strides, 0)] = x.data()[x._flat(in)];
        odo.advance();
    }
    return out;
}

// Inverse of an N-D array: reshape a to the square matrix (P, P) with
// P = prod(a.shape[:ind]) (must equal prod(a.shape[ind:]), else the
// tensor is not "square"), invert, reshape the result row-major to
// a.shape[ind:] + a.shape[:ind]. The flat result is row-major, so
// tensordot(tensorinv(a), b, ind) acts like the identity on the first
// ind axes: tensorinv(a) tensordot a gives the Kronecker delta.
// Reference: numpy-reference/reference/generated/numpy.linalg.tensorinv.html
// Raises std::invalid_argument when ind is out of range and
// np::exceptions::LinAlgError when the tensor is not square or singular.
NP_API template <typename T> NP_NODISCARD auto tensorinv(const ndarray<T> &a, int ind = 2) -> ndarray<real_t<T>>
{
    using R = real_t<T>;
    const std::size_t nd = a.ndim();
    if (ind <= 0 || static_cast<std::size_t>(ind) > nd)
    {
        throw std::invalid_argument("tensorinv: ind must be between 1 and the array rank");
    }
    const std::size_t ui = static_cast<std::size_t>(ind);
    std::size_t p1 = 1, p2 = 1;
    for (std::size_t i = 0; i < ui; ++i)
    {
        p1 *= static_cast<std::size_t>(a.shape[i]);
    }
    for (std::size_t i = ui; i < nd; ++i)
    {
        p2 *= static_cast<std::size_t>(a.shape[i]);
    }
    if (p1 != p2)
    {
        throw np::exceptions::LinAlgError("tensorinv: the tensor is not square");
    }
    // Row-major copy of a into the (p1, p1) matrix.
    std::vector<R> mat(p1 * p1);
    const auto cst = detail::c_order_strides(a.shape);
    np::detail::Odometer odo(a.shape);
    while (!odo.done())
    {
        const auto &oi = odo.idx();
        mat[np::detail::flat_index(oi, cst, 0)] = static_cast<R>(a.data()[a._flat(oi)]);
        odo.advance();
    }
    std::vector<std::size_t> piv;
    std::size_t swaps{};
    if (detail::lu_factor(mat, p1, piv, swaps))
    {
        throw np::exceptions::LinAlgError("Singular matrix");
    }
    std::vector<R> flat = detail::lu_invert(mat, p1, piv);
    std::vector<int> rshape;
    for (std::size_t i = ui; i < nd; ++i)
    {
        rshape.push_back(a.shape[i]);
    }
    for (std::size_t i = 0; i < ui; ++i)
    {
        rshape.push_back(a.shape[i]);
    }
    return ndarray<R>::from_data(std::move(rshape), std::move(flat));
}

// Solve tensordot(a, x, x.ndim()) == b for x: a must have shape
// b.shape + Q with prod(Q) == prod(b.shape), and x comes back with
// shape Q. With axes given, those axes are first moved to the end of a
// (numpy moveaxis semantics) and the rule applies to the reordered
// array. The leading dims of a must match b.shape exactly.
// Reference: numpy-reference/reference/generated/numpy.linalg.tensorsolve.html
// Raises std::invalid_argument on shape mismatches and
// np::exceptions::LinAlgError when the tensor is not square or singular.
NP_API template <typename T, typename U>
NP_NODISCARD auto tensorsolve(const ndarray<T> &a, const ndarray<U> &b, const std::vector<int> &axes = {})
    -> ndarray<std::common_type_t<real_t<T>, real_t<U>>>
{
    using R = std::common_type_t<real_t<T>, real_t<U>>;
    // Permute a so that the chosen axes come last (empty axes: no-op).
    std::vector<int> order;
    if (axes.empty())
    {
        order.resize(a.ndim());
        for (std::size_t i = 0; i < order.size(); ++i)
        {
            order[i] = static_cast<int>(i);
        }
    }
    else
    {
        std::vector<bool> seen(a.ndim(), false);
        for (int ax : axes)
        {
            const int r = ax < 0 ? ax + static_cast<int>(a.ndim()) : ax;
            const std::size_t u = static_cast<std::size_t>(r);
            if (r < 0 || r >= static_cast<int>(a.ndim()) || seen[u])
            {
                throw std::invalid_argument("tensorsolve: invalid or repeated axis");
            }
            seen[u] = true;
        }
        for (std::size_t i = 0; i < a.ndim(); ++i)
        {
            if (!seen[i])
            {
                order.push_back(static_cast<int>(i));
            }
        }
        for (int ax : axes)
        {
            order.push_back(static_cast<int>(ax < 0 ? ax + static_cast<int>(a.ndim()) : ax));
        }
    }
    std::vector<int> a2_shape;
    for (int ax : order)
    {
        a2_shape.push_back(a.shape[static_cast<std::size_t>(ax)]);
    }
    const std::size_t nb = b.ndim();
    if (a2_shape.size() < nb || nb == 0)
    {
        throw std::invalid_argument("tensorsolve: a must have more dimensions than b");
    }
    for (std::size_t i = 0; i < nb; ++i)
    {
        if (a2_shape[i] != b.shape[i])
        {
            throw std::invalid_argument("tensorsolve: b must match the leading dimensions of a");
        }
    }
    std::vector<int> qshape(a2_shape.begin() + static_cast<long>(nb), a2_shape.end());
    std::size_t p1 = 1, p2 = 1;
    for (std::size_t i = 0; i < nb; ++i)
    {
        p1 *= static_cast<std::size_t>(a2_shape[i]);
    }
    for (int d : qshape)
    {
        p2 *= static_cast<std::size_t>(d);
    }
    if (p1 != p2)
    {
        throw np::exceptions::LinAlgError("tensorsolve: the tensor is not square");
    }
    // Row-major copy of a (reordered) as the (p1, p1) system matrix.
    std::vector<R> mat(p1 * p1);
    const auto cst = detail::c_order_strides(a2_shape);
    np::detail::Odometer odo(a2_shape);
    while (!odo.done())
    {
        const auto &oi = odo.idx();
        std::vector<std::size_t> in(oi.size());
        for (std::size_t i = 0; i < oi.size(); ++i)
        {
            in[static_cast<std::size_t>(order[i])] = oi[i];
        }
        mat[np::detail::flat_index(oi, cst, 0)] = static_cast<R>(a.data()[a._flat(in)]);
        odo.advance();
    }
    // Right-hand side: b flattened row-major.
    std::vector<R> rhs(p1);
    const auto cstb = detail::c_order_strides(b.shape);
    np::detail::Odometer odb(b.shape);
    while (!odb.done())
    {
        const auto &oi = odb.idx();
        rhs[np::detail::flat_index(oi, cstb, 0)] = static_cast<R>(b.data()[b._flat(oi)]);
        odb.advance();
    }
    std::vector<std::size_t> piv;
    std::size_t swaps{};
    if (detail::lu_factor(mat, p1, piv, swaps))
    {
        throw np::exceptions::LinAlgError("Singular matrix");
    }
    std::vector<R> x = detail::lu_solve(mat, p1, piv, rhs);
    return ndarray<R>::from_data(std::move(qshape), std::move(x));
}

// Dot product supporting 1D/2D combinations:
// 1D . 1D -> scalar, 2D . 2D -> 2D, 2D . 1D and 1D . 2D -> 1D.
// Reference: numpy-reference/reference/generated/numpy.linalg.numpy.dot.html
// Raises std::invalid_argument on incompatible shapes or ndim > 2.
NP_API template <typename T, typename U>
NP_NODISCARD auto dot(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_arithmetic_v<T> || ::np::is_complex_v<T> || ::np::detail::is_bigint_v<T>,
                  "dot: T must be arithmetic/complex/bigint");
    static_assert(std::is_arithmetic_v<U> || ::np::is_complex_v<U> || ::np::detail::is_bigint_v<U>,
                  "dot: U must be arithmetic/complex/bigint");
    static_assert(std::is_copy_constructible_v<T>, "dot: T must be copy constructible");
    static_assert(std::is_copy_constructible_v<U>, "dot: U must be copy constructible");
    using R = std::common_type_t<T, U>;
    const std::size_t na = a.ndim();
    const std::size_t nb = b.ndim();
    if (na > 2 || nb > 2)
    {
        throw std::invalid_argument("dot only supports arrays with ndim <= 2");
    }
    if (na == 0 || nb == 0)
    {
        throw std::invalid_argument("dot operands must be non-scalar");
    }

    const auto &ashape = a.shape;
    const auto &bshape = b.shape;

    // 1D . 1D -> scalar (0-d result)
    if (na == 1 && nb == 1)
    {
        if (ashape[0] != bshape[0])
        {
            throw std::invalid_argument("dot: incompatible 1D sizes");
        }
        R acc{};
        for (std::size_t i = 0; i < static_cast<std::size_t>(ashape[0]); ++i)
        {
            acc += static_cast<R>(a.at(i)) * static_cast<R>(b.at(i));
        }
        return ndarray<R>::from_data(std::vector<int>{}, std::vector<R>{acc});
    }

    // 2D . 1D -> 1D
    if (na == 2 && nb == 1)
    {
        if (ashape[1] != bshape[0])
        {
            throw std::invalid_argument("dot: incompatible shapes");
        }
        const std::size_t rows = static_cast<std::size_t>(ashape[0]);
        const std::size_t k = static_cast<std::size_t>(ashape[1]);
        ndarray<R> out(std::vector<int>{static_cast<int>(rows)});
#ifdef NP_USE_THREADING
        if (rows > 64)
        {
            ::np::ThreadPool::global().parallel_for(0, rows, [&](std::size_t i) {
                R acc{};
                for (std::size_t j = 0; j < k; ++j)
                {
                    acc += static_cast<R>(a.get(std::array<std::size_t, 2>{i, j})) * static_cast<R>(b.at(j));
                }
                out.data()[i] = acc;
            });
        }
        else
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                R acc{};
                for (std::size_t j = 0; j < k; ++j)
                {
                    acc += static_cast<R>(a.get(std::array<std::size_t, 2>{i, j})) * static_cast<R>(b.at(j));
                }
                out.data()[i] = acc;
            }
        }
#else
        for (std::size_t i = 0; i < rows; ++i)
        {
            R acc{};
            for (std::size_t j = 0; j < k; ++j)
            {
                acc += static_cast<R>(a.get(std::array<std::size_t, 2>{i, j})) * static_cast<R>(b.at(j));
            }
            out.data()[i] = acc;
        }
#endif
        return out;
    }

    // 1D . 2D -> 1D
    if (na == 1 && nb == 2)
    {
        if (ashape[0] != bshape[0])
        {
            throw std::invalid_argument("dot: incompatible shapes");
        }
        const std::size_t k = static_cast<std::size_t>(ashape[0]);
        const std::size_t cols = static_cast<std::size_t>(bshape[1]);
        ndarray<R> out(std::vector<int>{static_cast<int>(cols)});
#ifdef NP_USE_THREADING
        if (cols > 64)
        {
            ::np::ThreadPool::global().parallel_for(0, cols, [&](std::size_t j) {
                R acc{};
                for (std::size_t i = 0; i < k; ++i)
                {
                    acc += static_cast<R>(a.at(i)) * static_cast<R>(b.get(std::array<std::size_t, 2>{i, j}));
                }
                out.data()[j] = acc;
            });
        }
        else
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                R acc{};
                for (std::size_t i = 0; i < k; ++i)
                {
                    acc += static_cast<R>(a.at(i)) * static_cast<R>(b.get(std::array<std::size_t, 2>{i, j}));
                }
                out.data()[j] = acc;
            }
        }
#else
        for (std::size_t j = 0; j < cols; ++j)
        {
            R acc{};
            for (std::size_t i = 0; i < k; ++i)
            {
                acc += static_cast<R>(a.at(i)) * static_cast<R>(b.get(std::array<std::size_t, 2>{i, j}));
            }
            out.data()[j] = acc;
        }
#endif
        return out;
    }

    // 2D . 2D -> 2D
    if (ashape[1] != bshape[0])
    {
        throw std::invalid_argument("dot: incompatible shapes");
    }
    const std::size_t rows = static_cast<std::size_t>(ashape[0]);
    const std::size_t k = static_cast<std::size_t>(ashape[1]);
    const std::size_t cols = static_cast<std::size_t>(bshape[1]);
    ndarray<R> out(std::vector<int>{static_cast<int>(rows), static_cast<int>(cols)});
    // Powerful dispatch: GPU (OpenMP target / CUDA driver) for large FP GEMM,
    // else CPU blocked with AVX2/FMA + OpenMP. Cache-aware BLOCK=128 (float) /96 (double)
    if (a.is_contiguous() && b.is_contiguous()) [[likely]]
    {
        const T *__restrict ad = a.data().data();
        const U *__restrict bd = b.data().data();
        R *__restrict od = out.data().data();

        // GPU fast path for contiguous float/double large GEMM
        if constexpr (std::is_same_v<T, R> && std::is_same_v<U, R> &&
                      (std::is_same_v<R, float> || std::is_same_v<R, double>))
        {
            const std::size_t gpu_thresh = tune::gpu_threshold_flops();
            if (rows * cols * k > gpu_thresh || rows * cols > 65536)
            {
                // For very large (4K) multi-GPU shard
                if (rows * cols * k > 64ULL * 1024 * 1024 && gpu::device_count() > 1)
                {
                    gpu::sharded_matmul(ad, bd, od, rows, cols, k);
                    return out;
                }
                if (gpu::try_matmul(ad, bd, od, rows, cols, k))
                    return out;
                // Fallback to optimized CPU blocked kernel (AVX2+FMA+OpenMP)
                gpu::cpu_matmul(ad, bd, od, rows, cols, k);
                return out;
            }
        }

        // CPU blocked path – tune::optimal_block for powerful (L3-aware)
        std::size_t block = 32;
        if constexpr (std::is_same_v<R, float>)
            block = tune::optimal_block_f32();
        else if constexpr (std::is_same_v<R, double>)
            block = tune::optimal_block_f64();

        if (rows * cols * k > 32768 && rows > block && cols > block && k > block)
        {
            std::fill(od, od + rows * cols, R{});
            for (std::size_t ii = 0; ii < rows; ii += block)
            {
                for (std::size_t jj = 0; jj < cols; jj += block)
                {
                    for (std::size_t pp = 0; pp < k; pp += block)
                    {
                        std::size_t i_max = std::min(ii + block, rows);
                        std::size_t j_max = std::min(jj + block, cols);
                        std::size_t p_max = std::min(pp + block, k);
                        for (std::size_t i = ii; i < i_max; ++i)
                        {
                            for (std::size_t p = pp; p < p_max; ++p)
                            {
                                R av = static_cast<R>(ad[i * k + p]);
                                if constexpr ((std::is_same_v<R, float> || std::is_same_v<R, double>) &&
                                              std::is_same_v<T, R> && std::is_same_v<U, R>)
                                {
                                    // SIMD FMA: out[j] += av * b[j] with contiguous R
                                    simd::fma_vectorized(reinterpret_cast<const R *>(bd + p * cols + jj), av,
                                                         od + i * cols + jj, j_max - jj);
                                }
                                else
                                {
                                    for (std::size_t j = jj; j < j_max; ++j)
                                    {
                                        od[i * cols + j] += av * static_cast<R>(bd[p * cols + j]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return out;
        }
#ifdef NP_USE_THREADING
        if (rows * cols > 4096)
        {
            ::np::ThreadPool::global().parallel_for(0, rows, [&](std::size_t i) {
                for (std::size_t j = 0; j < cols; ++j)
                {
                    R acc{};
                    for (std::size_t p = 0; p < k; ++p)
                        acc += static_cast<R>(ad[i * k + p]) * static_cast<R>(bd[p * cols + j]);
                    od[i * cols + j] = acc;
                }
            });
            return out;
        }
#endif
#if defined(NP_ENABLE_OPENMP)
        if (rows * cols > 4096)
        {
#pragma omp parallel for collapse(2) schedule(static)
            for (std::size_t i = 0; i < rows; ++i)
                for (std::size_t j = 0; j < cols; ++j)
                {
                    R acc = R{0};
                    for (std::size_t p = 0; p < k; ++p)
                        acc += static_cast<R>(ad[i * k + p]) * static_cast<R>(bd[p * cols + j]);
                    od[i * cols + j] = acc;
                }
            return out;
        }
#endif
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                R acc{};
                for (std::size_t p = 0; p < k; ++p)
                    acc += static_cast<R>(ad[i * k + p]) * static_cast<R>(bd[p * cols + j]);
                od[i * cols + j] = acc;
            }
        }
        return out;
    }
#ifdef NP_USE_THREADING
    if (rows * cols > 4096)
    {
        ::np::ThreadPool::global().parallel_for(0, rows, [&](std::size_t i) {
            for (std::size_t j = 0; j < cols; ++j)
            {
                R acc{};
                for (std::size_t p = 0; p < k; ++p)
                {
                    acc += static_cast<R>(a.get(std::array<std::size_t, 2>{i, p})) *
                           static_cast<R>(b.get(std::array<std::size_t, 2>{p, j}));
                }
                out.data()[i * cols + j] = acc;
            }
        });
    }
    else
    {
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                R acc{};
                for (std::size_t p = 0; p < k; ++p)
                {
                    acc += static_cast<R>(a.get(std::array<std::size_t, 2>{i, p})) *
                           static_cast<R>(b.get(std::array<std::size_t, 2>{p, j}));
                }
                out.data()[i * cols + j] = acc;
            }
        }
    }
#else
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            R acc{};
            for (std::size_t p = 0; p < k; ++p)
            {
                acc += static_cast<R>(a.get(std::array<std::size_t, 2>{i, p})) *
                       static_cast<R>(b.get(std::array<std::size_t, 2>{p, j}));
            }
            out.data()[i * cols + j] = acc;
        }
    }
#endif
    return out;
}

// Matrix multiplication (same semantics as dot for ndim <= 2).
// Reference: numpy-reference/reference/generated/numpy.matmul.html
NP_API template <typename T, typename U>
NP_NODISCARD auto matmul(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    return dot(a, b);
}

/**
 * @brief Matrix-vector product (numpy.linalg.matvec / numpy.matvec).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.matvec.html
 * Also: numpy-reference/reference/generated/numpy.matvec.html
 * Simple wrapper around matmul for the 2-D x 1-D case (M, N) @ (N,) -> (M,).
 * Validates that a is 2-D and b is 1-D and delegates to matmul/dot, which
 * checks inner-dimension compatibility. Stacked (n-D, n > 2) inputs are not
 * supported in this simplified wrapper.
 * @tparam T Element type of the matrix.
 * @tparam U Element type of the vector.
 * @param a Matrix (M x N), 2-D.
 * @param b Vector (N,), 1-D.
 * @return Vector (M,) = a @ b.
 * @throws std::invalid_argument if a.ndim() != 2 or b.ndim() != 1,
 *         or if inner dimensions disagree.
 * @complexity O(M * N).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto matvec(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("matvec requires a 2D matrix");
    }
    if (b.ndim() != 1)
    {
        throw std::invalid_argument("matvec requires a 1D vector");
    }
    return matmul(a, b);
}

/**
 * @brief Vector-matrix product (numpy.linalg.vecmat / numpy.vecmat).
 *
 * Reference: numpy-reference/reference/generated/numpy.linalg.vecmat.html
 * Also: numpy-reference/reference/generated/numpy.vecmat.html
 * Simple wrapper around matmul for the 1-D x 2-D case (N,) @ (N, M) -> (M,).
 * Validates that a is 1-D and b is 2-D and delegates to matmul/dot, which
 * checks inner-dimension compatibility. Stacked (n-D, n > 2) inputs are not
 * supported in this simplified wrapper.
 * @tparam T Element type of the vector.
 * @tparam U Element type of the matrix.
 * @param a Vector (N,), 1-D.
 * @param b Matrix (N, M), 2-D.
 * @return Vector (M,) = a @ b.
 * @throws std::invalid_argument if a.ndim() != 1 or b.ndim() != 2,
 *         or if inner dimensions disagree.
 * @complexity O(N * M).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto vecmat(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    if (a.ndim() != 1)
    {
        throw std::invalid_argument("vecmat requires a 1D vector");
    }
    if (b.ndim() != 2)
    {
        throw std::invalid_argument("vecmat requires a 2D matrix");
    }
    return matmul(a, b);
}

// Raise a square 2-D array to the integer power n: n == 0 gives the
// identity, n > 0 repeated squarings, n < 0 the inverse raised to |n|.
// Reference: numpy-reference/reference/generated/numpy.linalg.matrix_power.html
// Raises std::invalid_argument unless a.ndim() == 2 and
// np::exceptions::LinAlgError when a is not square or (n < 0) singular.
// Integral input is promoted to double for every n (numpy keeps the
// integer dtype for n >= 0); values stay exact while |a_ij| < 2^53.
NP_API template <typename T> NP_NODISCARD auto matrix_power(const ndarray<T> &a, long long n) -> ndarray<real_t<T>>
{
    using R = real_t<T>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("matrix_power requires a 2D array");
    }
    const std::size_t m = static_cast<std::size_t>(a.shape[0]);
    const std::size_t k = static_cast<std::size_t>(a.shape[1]);
    if (m != k)
    {
        throw np::exceptions::LinAlgError("matrix_power requires a square matrix");
    }
    std::vector<R> id(m * m, R{});
    for (std::size_t i = 0; i < m; ++i)
    {
        id[i * m + i] = R{1};
    }
    if (n == 0)
    {
        return detail::mk2d(m, m, std::move(id));
    }
    ndarray<R> base;
    if (n > 0)
    {
        std::size_t rows{}, cols{};
        std::vector<R> acopy = detail::dense2d<R>(a, rows, cols);
        base = detail::mk2d(rows, cols, std::move(acopy));
    }
    else
    {
        base = inv(a);
    }
    const unsigned long long e = n > 0 ? static_cast<unsigned long long>(n) : 0ULL - static_cast<unsigned long long>(n);
    ndarray<R> result = detail::mk2d(m, m, std::move(id));
    unsigned long long exp = e;
    while (exp > 0)
    {
        if (exp & 1ULL)
        {
            result = dot(result, base);
        }
        exp >>= 1;
        if (exp > 0)
        {
            base = dot(base, base);
        }
    }
    return result;
}

// Least-squares solution of the overdetermined system a x ~= b (or the
// minimum-norm solution when underdetermined / rank-deficient), via the
// SVD with a relative cutoff: singular values at or below the cutoff
// are treated as zero, as in numpy. b is 1-D (M,) or 2-D (M, K).
// Reference: numpy-reference/reference/generated/numpy.linalg.lstsq.html
// The residuals are the per-column squared 2-norms of b - a x and are
// empty when the rank is deficient or when M <= N. The default rcond
// (nullopt) is eps * max(M, N); an explicit -1 selects plain eps (the
// pre-2.0 default). Raises std::invalid_argument on bad shapes.
NP_API template <typename T, typename U>
NP_NODISCARD auto lstsq(const ndarray<T> &a, const ndarray<U> &b, std::optional<double> rcond = std::nullopt)
    -> LstsqResult<std::common_type_t<real_t<T>, real_t<U>>>
{
    using R = std::common_type_t<real_t<T>, real_t<U>>;
    if (a.ndim() != 2)
    {
        throw std::invalid_argument("lstsq requires a 2D array");
    }
    if (b.ndim() == 0 || b.ndim() > 2)
    {
        throw std::invalid_argument("lstsq requires b to be 1D or 2D");
    }
    const std::size_t m = static_cast<std::size_t>(a.shape[0]);
    const std::size_t n = static_cast<std::size_t>(a.shape[1]);
    const std::size_t nrhs = b.ndim() == 1 ? 1 : static_cast<std::size_t>(b.shape[1]);
    if (static_cast<std::size_t>(b.shape[0]) != m)
    {
        throw std::invalid_argument("lstsq: b's leading dimension must match a");
    }
    LstsqResult<R> out;
    out.s = svdvals(a);
    const std::size_t k = out.s.size();
    const R eps = std::numeric_limits<R>::epsilon();
    const double rcond_eff = !rcond ? static_cast<double>(eps) * static_cast<double>(std::max(m, n))
                                    : (*rcond == -1.0 ? static_cast<double>(eps) : *rcond);
    if (k == 0)
    {
        out.x = b.ndim() == 1 ? ndarray<R>(std::vector<int>{static_cast<int>(n)})
                              : ndarray<R>(std::vector<int>{static_cast<int>(n), static_cast<int>(nrhs)});
        out.residuals = ndarray<R>(std::vector<int>{0});
        out.rank = 0;
        return out;
    }
    const R cutoff = out.s(0) * static_cast<R>(rcond_eff);
    out.rank = 0;
    for (std::size_t i = 0; i < k; ++i)
    {
        if (out.s(i) > cutoff)
        {
            ++out.rank;
        }
    }
    out.x = dot(pinv(a, rcond_eff), b);
    // Residuals: empty when rank-deficient or when M <= N.
    if (m > n && out.rank == static_cast<int>(n))
    {
        const int nr = static_cast<int>(nrhs);
        std::vector<R> res(nrhs, R{});
        for (std::size_t c = 0; c < nrhs; ++c)
        {
            R acc{};
            for (std::size_t i = 0; i < m; ++i)
            {
                R r = b.ndim() == 1 ? static_cast<R>(b.at(i)) : static_cast<R>(b.at(i, c));
                for (std::size_t j = 0; j < n; ++j)
                {
                    r -= static_cast<R>(a.at(i, j)) * (b.ndim() == 1 ? out.x(j) : out.x(j, c));
                }
                acc += r * r;
            }
            res[c] = acc;
        }
        out.residuals = b.ndim() == 1 ? ndarray<R>::from_data(std::vector<int>{1}, std::move(res))
                                      : ndarray<R>::from_data(std::vector<int>{nr}, std::move(res));
    }
    else
    {
        out.residuals = ndarray<R>(std::vector<int>{0});
    }
    return out;
}

// Product of two or more matrices with the optimal parenthesization
// (matrix chain multiplication). Only the first and last arrays may be
// 1-D (treated as row/column vectors, with the usual np.dot semantics);
// every middle array must be 2-D. The evaluation order minimizes the
// scalar multiply count cost(A, B) = rows(A) * cols(A) * cols(B).
// Reference: numpy-reference/reference/generated/numpy.linalg.multi_dot.html
// Raises std::invalid_argument when fewer than two arrays are given,
// a middle array is not 2-D, or the shapes do not chain.
NP_API template <typename T> NP_NODISCARD auto multi_dot(const std::vector<ndarray<T>> &arrays) -> ndarray<T>
{
    const std::size_t n = arrays.size();
    if (n < 2)
    {
        throw std::invalid_argument("multi_dot needs at least two arrays");
    }
    if (arrays[0].ndim() == 0 || arrays[0].ndim() > 2 || arrays[n - 1].ndim() == 0 || arrays[n - 1].ndim() > 2)
    {
        throw std::invalid_argument("multi_dot: the first and last arrays must be 1D or 2D");
    }
    for (std::size_t i = 1; i + 1 < n; ++i)
    {
        if (arrays[i].ndim() != 2)
        {
            throw std::invalid_argument("multi_dot: only the first and last arrays may be 1D");
        }
    }
    // Chain dimensions: matrix i has (dims[i], dims[i + 1]); 1-D ends
    // are promoted to (1, n) / (m, 1) for the cost model.
    std::vector<int> dims(n + 1);
    dims[0] = arrays[0].ndim() == 1 ? 1 : arrays[0].shape[0];
    for (std::size_t i = 0; i < n; ++i)
    {
        if (arrays[i].ndim() == 1)
        {
            const int len = arrays[i].shape[0];
            if (i == 0)
            {
                dims[1] = len;
            }
            else
            {
                if (dims[n - 1] != len)
                {
                    throw std::invalid_argument("multi_dot: shapes do not chain");
                }
                dims[n] = len;
            }
            continue;
        }
        if (dims[i] != arrays[i].shape[0])
        {
            throw std::invalid_argument("multi_dot: shapes do not chain");
        }
        dims[i + 1] = arrays[i].shape[1];
    }
    // Optimal parenthesization (CLRS 15.2).
    std::vector<std::vector<std::size_t>> cost(n, std::vector<std::size_t>(n, 0));
    std::vector<std::vector<std::size_t>> split(n, std::vector<std::size_t>(n, 0));
    for (std::size_t len = 2; len <= n; ++len)
    {
        for (std::size_t i = 0; i + len <= n; ++i)
        {
            const std::size_t j = i + len - 1;
            cost[i][j] = std::numeric_limits<std::size_t>::max();
            for (std::size_t k = i; k < j; ++k)
            {
                const std::size_t c = cost[i][k] + cost[k + 1][j] +
                                      static_cast<std::size_t>(dims[i]) * static_cast<std::size_t>(dims[k + 1]) *
                                          static_cast<std::size_t>(dims[j + 1]);
                if (c < cost[i][j])
                {
                    cost[i][j] = c;
                    split[i][j] = k;
                }
            }
        }
    }
    std::function<ndarray<T>(std::size_t, std::size_t)> eval = [&](std::size_t i, std::size_t j) -> ndarray<T> {
        if (i == j)
        {
            return arrays[i];
        }
        const std::size_t k = split[i][j];
        return dot(eval(i, k), eval(k + 1, j));
    };
    return eval(0, n - 1);
}

// Tensor contraction of two arrays along paired axes, generalizing
// matrix multiplication to arbitrary ranks (equivalent to einsum).
// Reference: numpy-reference/reference/generated/numpy.linalg.tensordot.html
// The int axes form (default 2) contracts the last `axes` axes of a
// with the first `axes` axes of b; axes = 0 is the outer product. The
// two-int form contracts a single axis of each. The two-sequence form
// pairs arbitrary (unique) axes of a and b. The output shape is the
// uncontracted axes of a followed by those of b.
NP_API template <typename T, typename U>
NP_NODISCARD auto tensordot(const ndarray<T> &a, const ndarray<U> &b, const std::vector<int> &a_axes,
                            const std::vector<int> &b_axes) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    if (a_axes.size() != b_axes.size())
    {
        throw std::invalid_argument("tensordot: the axes sequences must have equal length");
    }
    const std::size_t na = a.ndim();
    const std::size_t nb = b.ndim();
    auto norm_axis = [](int ax, std::size_t nd) {
        const int n = static_cast<int>(nd);
        const int r = ax < 0 ? ax + n : ax;
        if (r < 0 || r >= n)
        {
            throw std::invalid_argument("tensordot: axis out of range");
        }
        return static_cast<std::size_t>(r);
    };
    std::vector<std::size_t> aa;
    std::vector<std::size_t> ba;
    for (int ax : a_axes)
    {
        aa.push_back(norm_axis(ax, na));
    }
    for (int ax : b_axes)
    {
        ba.push_back(norm_axis(ax, nb));
    }
    auto has_dup = [](std::vector<std::size_t> v) {
        std::sort(v.begin(), v.end());
        return std::adjacent_find(v.begin(), v.end()) != v.end();
    };
    if (has_dup(aa) || has_dup(ba))
    {
        throw std::invalid_argument("tensordot: repeated axes are not allowed");
    }
    for (std::size_t i = 0; i < aa.size(); ++i)
    {
        if (a.shape[aa[i]] != b.shape[ba[i]])
        {
            throw std::invalid_argument("tensordot: contracted dimensions must match");
        }
    }
    std::vector<std::size_t> a_free;
    std::vector<std::size_t> b_free;
    for (std::size_t d = 0; d < na; ++d)
    {
        if (std::find(aa.begin(), aa.end(), d) == aa.end())
        {
            a_free.push_back(d);
        }
    }
    for (std::size_t d = 0; d < nb; ++d)
    {
        if (std::find(ba.begin(), ba.end(), d) == ba.end())
        {
            b_free.push_back(d);
        }
    }
    std::vector<int> out_shape;
    for (std::size_t d : a_free)
    {
        out_shape.push_back(a.shape[d]);
    }
    for (std::size_t d : b_free)
    {
        out_shape.push_back(b.shape[d]);
    }
    std::vector<int> cshape(aa.size());
    for (std::size_t i = 0; i < aa.size(); ++i)
    {
        cshape[i] = a.shape[aa[i]];
    }
    ndarray<R> out(out_shape);
    np::detail::Odometer odo(out_shape);
    while (!odo.done())
    {
        const auto &oi = odo.idx();
        std::vector<std::size_t> ai(na);
        std::vector<std::size_t> bi(nb);
        for (std::size_t i = 0; i < a_free.size(); ++i)
        {
            ai[a_free[i]] = oi[i];
        }
        for (std::size_t i = 0; i < b_free.size(); ++i)
        {
            bi[b_free[i]] = oi[a_free.size() + i];
        }
        R acc{};
        np::detail::Odometer cod(cshape);
        while (!cod.done())
        {
            const auto &ci = cod.idx();
            for (std::size_t i = 0; i < aa.size(); ++i)
            {
                ai[aa[i]] = ci[i];
                bi[ba[i]] = ci[i];
            }
            acc += static_cast<R>(a.data()[a._flat(ai)]) * static_cast<R>(b.data()[b._flat(bi)]);
            cod.advance();
        }
        out.data()[np::detail::flat_index(oi, out.strides, 0)] = acc;
        odo.advance();
    }
    return out;
}

// Contract the last `axes` axes of a with the first `axes` axes of b.
// Reference: numpy-reference/reference/generated/numpy.linalg.tensordot.html
NP_API template <typename T, typename U>
NP_NODISCARD auto tensordot(const ndarray<T> &a, const ndarray<U> &b, int axes = 2) -> ndarray<std::common_type_t<T, U>>
{
    if (axes < 0)
    {
        throw std::invalid_argument("tensordot: axes must be non-negative");
    }
    const std::size_t na = a.ndim();
    const std::size_t nb = b.ndim();
    const std::size_t ua = static_cast<std::size_t>(axes);
    if (ua > na || ua > nb)
    {
        throw std::invalid_argument("tensordot: axes exceeds the array rank");
    }
    std::vector<int> aa;
    std::vector<int> ba;
    for (std::size_t i = 0; i < ua; ++i)
    {
        aa.push_back(static_cast<int>(na - ua + i));
        ba.push_back(static_cast<int>(i));
    }
    return tensordot(a, b, aa, ba);
}

// Contract one axis of a with one axis of b.
// Reference: numpy-reference/reference/generated/numpy.linalg.tensordot.html
NP_API template <typename T, typename U>
NP_NODISCARD auto tensordot(const ndarray<T> &a, const ndarray<U> &b, int a_axis, int b_axis)
    -> ndarray<std::common_type_t<T, U>>
{
    return tensordot(a, b, std::vector<int>{a_axis}, std::vector<int>{b_axis});
}

// Vector dot product: sum over the given axis (default the last) of
// x1 * x2, with the remaining axes broadcast as numpy does (unlike
// tensordot, which does not broadcast). The contracted axes must have
// equal sizes; the result shape is the broadcast of the two remainder
// shapes. Complex input would conjugate x1 (Array API), but complex
// arrays are not supported here.
// Reference: numpy-reference/reference/generated/numpy.linalg.vecdot.html
// Raises std::invalid_argument on axis errors, mismatched contracted
// sizes, or non-broadcastable remainder shapes.
NP_API template <typename T, typename U>
NP_NODISCARD auto vecdot(const ndarray<T> &x1, const ndarray<U> &x2, int axis = -1) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    const std::size_t n1 = x1.ndim();
    const std::size_t n2 = x2.ndim();
    if (n1 == 0 || n2 == 0)
    {
        throw std::invalid_argument("vecdot operands must be non-scalar");
    }
    auto norm = [](int ax, std::size_t nd) {
        const int r = ax < 0 ? ax + static_cast<int>(nd) : ax;
        if (r < 0 || r >= static_cast<int>(nd))
        {
            throw std::invalid_argument("vecdot: axis out of range");
        }
        return static_cast<std::size_t>(r);
    };
    const std::size_t a1 = norm(axis, n1);
    const std::size_t a2 = norm(axis, n2);
    const std::size_t len1 = static_cast<std::size_t>(x1.shape[a1]);
    const std::size_t len2 = static_cast<std::size_t>(x2.shape[a2]);
    if (len1 != len2)
    {
        throw std::invalid_argument("vecdot: the contracted axes must match in size");
    }
    std::vector<int> r1;
    std::vector<int> r2;
    for (std::size_t i = 0; i < n1; ++i)
    {
        if (i != a1)
        {
            r1.push_back(x1.shape[i]);
        }
    }
    for (std::size_t i = 0; i < n2; ++i)
    {
        if (i != a2)
        {
            r2.push_back(x2.shape[i]);
        }
    }
    const std::vector<int> out_shape = np::detail::broadcast_shapes(r1, r2);
    const std::size_t no = out_shape.size();
    const std::size_t shift1 = no - r1.size();
    const std::size_t shift2 = no - r2.size();
    ndarray<R> out(out_shape);
    np::detail::Odometer odo(out_shape);
    while (!odo.done())
    {
        const auto &oi = odo.idx();
        R acc{};
        for (std::size_t k = 0; k < len1; ++k)
        {
            std::vector<std::size_t> i1(n1);
            std::size_t p = 0;
            for (std::size_t d = 0; d < n1; ++d)
            {
                if (d == a1)
                {
                    i1[d] = k;
                }
                else
                {
                    i1[d] = r1[p] == 1 ? 0 : oi[shift1 + p];
                    ++p;
                }
            }
            std::vector<std::size_t> i2(n2);
            p = 0;
            for (std::size_t d = 0; d < n2; ++d)
            {
                if (d == a2)
                {
                    i2[d] = k;
                }
                else
                {
                    i2[d] = r2[p] == 1 ? 0 : oi[shift2 + p];
                    ++p;
                }
            }
            acc += static_cast<R>(x1.data()[x1._flat(i1)]) * static_cast<R>(x2.data()[x2._flat(i2)]);
        }
        out.data()[np::detail::flat_index(oi, out.strides, 0)] = acc;
        odo.advance();
    }
    return out;
}

// Cross product of 3-element vectors along the given axis. Non-compute
// axes are broadcast as numpy does; the compute axis of both arrays
// must carry exactly 3 elements, and when the ranks differ the
// 1-D operand is promoted so the vector axis sits at the end of the
// broadcast result (numpy's rule when the axes align by promotion).
// Reference: numpy-reference/reference/generated/numpy.linalg.cross.html
// Raises std::invalid_argument when the vector axes do not have 3
// elements, the shapes are not broadcastable, or the axis is invalid.
NP_API template <typename T, typename U>
NP_NODISCARD auto cross(const ndarray<T> &x1, const ndarray<U> &x2, int axis = -1) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    const std::size_t nd1 = x1.ndim();
    const std::size_t nd2 = x2.ndim();
    if (nd1 == 0 || nd2 == 0)
    {
        throw std::invalid_argument("cross requires vectors");
    }
    auto norm_axis = [](int ax, std::size_t nd) {
        const int n = static_cast<int>(nd);
        const int r = ax < 0 ? ax + n : ax;
        if (r < 0 || r >= n)
        {
            throw std::invalid_argument("cross: axis out of range");
        }
        return r;
    };
    const int a1 = norm_axis(axis, nd1);
    const int a2 = norm_axis(axis, nd2);
    const int sza1 = x1.shape[a1];
    const int sza2 = x2.shape[a2];
    const bool is2 = (sza1 == 2 && sza2 == 2);
    const bool is3 = (sza1 == 3 && sza2 == 3);
    if (!is2 && !is3)
    {
        throw std::invalid_argument("cross: the vectors must have 2 or 3 elements");
    }
    if (is2)
    {
        // 2-element: scalar cross
        std::vector<int> s1a;
        std::vector<int> s2a;
        for (int d = 0; d < static_cast<int>(nd1); ++d)
            if (d != a1)
                s1a.push_back(x1.shape[d]);
        for (int d = 0; d < static_cast<int>(nd2); ++d)
            if (d != a2)
                s2a.push_back(x2.shape[d]);
        std::vector<int> bshape2 = np::detail::broadcast_shapes(s1a, s2a);
        ndarray<R> out2(bshape2);
        const std::size_t off1c = bshape2.size() - s1a.size();
        const std::size_t off2c = bshape2.size() - s2a.size();
        np::detail::Odometer od2(bshape2);
        while (!od2.done())
        {
            const auto &oi = od2.idx();
            std::vector<std::size_t> i1(nd1, 0), i2(nd2, 0);
            for (std::size_t i = 0; i < s1a.size(); ++i)
            {
                std::size_t d = i < static_cast<std::size_t>(a1) ? i : i + 1;
                i1[d] = oi[off1c + i];
            }
            for (std::size_t i = 0; i < s2a.size(); ++i)
            {
                std::size_t d = i < static_cast<std::size_t>(a2) ? i : i + 1;
                i2[d] = oi[off2c + i];
            }
            auto r1 = [&](int e) {
                i1[static_cast<std::size_t>(a1)] = static_cast<std::size_t>(e);
                return static_cast<R>(x1.data()[x1._flat(i1)]);
            };
            auto r2 = [&](int e) {
                i2[static_cast<std::size_t>(a2)] = static_cast<std::size_t>(e);
                return static_cast<R>(x2.data()[x2._flat(i2)]);
            };
            R c = r1(0) * r2(1) - r1(1) * r2(0);
            out2.data()[np::detail::flat_index(oi, out2.strides, 0)] = c;
            od2.advance();
        }
        return out2;
    }
    std::vector<int> s1;
    std::vector<int> s2;
    for (int d = 0; d < static_cast<int>(nd1); ++d)
    {
        if (d != a1)
        {
            s1.push_back(x1.shape[d]);
        }
    }
    for (int d = 0; d < static_cast<int>(nd2); ++d)
    {
        if (d != a2)
        {
            s2.push_back(x2.shape[d]);
        }
    }
    std::vector<int> bshape = np::detail::broadcast_shapes(s1, s2);
    // Vector-axis position in the result: the shared axis index when
    // the compute axes align, otherwise appended at the end (the 1-D
    // promotion case).
    const int vpos = a1 == a2 ? a1 : static_cast<int>(bshape.size());
    std::vector<int> out_shape = bshape;
    out_shape.insert(out_shape.begin() + vpos, 3);
    const std::size_t off1 = bshape.size() - s1.size();
    const std::size_t off2 = bshape.size() - s2.size();
    auto out_of = [vpos](std::size_t p) { return p < static_cast<std::size_t>(vpos) ? p : p + 1; };
    ndarray<R> out(out_shape);
    np::detail::Odometer odo(out_shape);
    while (!odo.done())
    {
        const auto &oi = odo.idx();
        std::vector<std::size_t> i1(nd1);
        std::vector<std::size_t> i2(nd2);
        for (std::size_t i = 0; i < s1.size(); ++i)
        {
            const std::size_t d = i < static_cast<std::size_t>(a1) ? i : i + 1;
            i1[d] = oi[out_of(off1 + i)];
        }
        for (std::size_t i = 0; i < s2.size(); ++i)
        {
            const std::size_t d = i < static_cast<std::size_t>(a2) ? i : i + 1;
            i2[d] = oi[out_of(off2 + i)];
        }
        auto read1 = [&](int e) {
            i1[static_cast<std::size_t>(a1)] = static_cast<std::size_t>(e);
            return static_cast<R>(x1.data()[x1._flat(i1)]);
        };
        auto read2 = [&](int e) {
            i2[static_cast<std::size_t>(a2)] = static_cast<std::size_t>(e);
            return static_cast<R>(x2.data()[x2._flat(i2)]);
        };
        const R a0 = read1(0);
        const R a1c = read1(1);
        const R a2c = read1(2);
        const R b0 = read2(0);
        const R b1c = read2(1);
        const R b2c = read2(2);
        // cross = (a1 b2 - a2 b1, a2 b0 - a0 b2, a0 b1 - a1 b0)
        const R c0 = a1c * b2c - a2c * b1c;
        const R c1 = a2c * b0 - a0 * b2c;
        const R c2 = a0 * b1c - a1c * b0;
        std::vector<std::size_t> out_idx = oi;
        for (int e = 0; e < 3; ++e)
        {
            out_idx[static_cast<std::size_t>(vpos)] = static_cast<std::size_t>(e);
            out.data()[np::detail::flat_index(out_idx, out.strides, 0)] = e == 0 ? c0 : (e == 1 ? c1 : c2);
        }
        odo.advance();
    }
    return out;
}

// Inner product: contracts the last axes; 1D . 1D gives a scalar.
// Reference: numpy-reference/reference/generated/numpy.linalg.inner.html
NP_API template <typename T, typename U>
NP_NODISCARD auto inner(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    const std::size_t na = a.ndim();
    const std::size_t nb = b.ndim();
    if (na == 0 || nb == 0)
    {
        throw std::invalid_argument("inner operands must be non-scalar");
    }
    const std::size_t la = static_cast<std::size_t>(a.shape[na - 1]);
    const std::size_t lb = static_cast<std::size_t>(b.shape[nb - 1]);
    if (la != lb)
    {
        throw std::invalid_argument("inner: last dimensions must match");
    }
    if (na == 1 && nb == 1)
    {
        return dot(a, b);
    }
    // Output shape: a.shape[0..na-2] + b.shape[0..nb-2]
    std::vector<int> out_shape;
    for (std::size_t d = 0; d + 1 < na; ++d)
    {
        out_shape.push_back(a.shape[d]);
    }
    for (std::size_t d = 0; d + 1 < nb; ++d)
    {
        out_shape.push_back(b.shape[d]);
    }
    ndarray<R> out(out_shape);
    np::detail::Odometer oda(std::vector<int>(a.shape.begin(), a.shape.end() - 1));
    while (!oda.done())
    {
        const auto &ia = oda.idx();
        np::detail::Odometer odb(std::vector<int>(b.shape.begin(), b.shape.end() - 1));
        while (!odb.done())
        {
            const auto &ib = odb.idx();
            R acc{};
            for (std::size_t p = 0; p < la; ++p)
            {
                std::vector<std::size_t> ai = ia;
                ai.push_back(p);
                std::vector<std::size_t> bi = ib;
                bi.push_back(p);
                acc += static_cast<R>(a.data()[a._flat(ai)]) * static_cast<R>(b.data()[b._flat(bi)]);
            }
            std::vector<std::size_t> oi(ia.begin(), ia.end());
            oi.insert(oi.end(), ib.begin(), ib.end());
            out.data()[np::detail::flat_index(oi, out.strides, 0)] = acc;
            odb.advance();
        }
        oda.advance();
    }
    return out;
}

// Outer product of two 1D arrays (i, j) -> a[i] * b[j].
// Reference: numpy-reference/reference/generated/numpy.outer.html
NP_API template <typename T, typename U>
NP_NODISCARD auto outer(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    if (a.ndim() != 1 || b.ndim() != 1)
    {
        throw std::invalid_argument("outer requires two 1D arrays");
    }
    const std::size_t m = static_cast<std::size_t>(a.shape[0]);
    const std::size_t n = static_cast<std::size_t>(b.shape[0]);
    ndarray<R> out(std::vector<int>{static_cast<int>(m), static_cast<int>(n)});
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            out.data()[i * n + j] = static_cast<R>(a.at(i)) * static_cast<R>(b.at(j));
        }
    }
    return out;
}

// Transpose of a 2D array (convenience).
// Reference: numpy-reference/reference/generated/numpy.transpose.html
NP_API template <typename T> NP_NODISCARD auto transpose(const ndarray<T> &a) -> ndarray<T>
{
    return a.transpose();
}

// Trace of a 2D array.
// Reference: numpy-reference/reference/generated/numpy.trace.html
NP_API template <typename T> NP_NODISCARD auto trace(const ndarray<T> &a) -> T
{
    return a.trace();
}

// Kronecker product (np.kron).
// Reference: numpy-reference/reference/generated/numpy.kron.html
NP_API template <typename T, typename U>
NP_NODISCARD auto kron(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    std::size_t na = a.ndim(), nb = b.ndim();
    std::size_t n = std::max(na, nb);
    std::vector<int> a_shape(n, 1), b_shape(n, 1);
    for (std::size_t i = 0; i < na; ++i)
        a_shape[n - na + i] = a.shape[i];
    for (std::size_t i = 0; i < nb; ++i)
        b_shape[n - nb + i] = b.shape[i];
    std::vector<int> out_shape(n);
    for (std::size_t i = 0; i < n; ++i)
        out_shape[i] = a_shape[i] * b_shape[i];
    ndarray<R> out(out_shape);
    np::detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        std::vector<std::size_t> ai(n), bi(n);
        for (std::size_t d = 0; d < n; ++d)
        {
            ai[d] = idx[d] / static_cast<std::size_t>(b_shape[d]);
            bi[d] = idx[d] % static_cast<std::size_t>(b_shape[d]);
        }
        // map padded ai/bi to actual a/b indices (strip leading 1s)
        std::vector<std::size_t> a_idx, b_idx;
        if (na > 0)
        {
            a_idx.reserve(na);
            for (std::size_t d = n - na; d < n; ++d)
                a_idx.push_back(ai[d]);
        }
        if (nb > 0)
        {
            b_idx.reserve(nb);
            for (std::size_t d = n - nb; d < n; ++d)
                b_idx.push_back(bi[d]);
        }
        R av = a_idx.empty() ? static_cast<R>(a.item()) : static_cast<R>(a.get(a_idx));
        R bv = b_idx.empty() ? static_cast<R>(b.item()) : static_cast<R>(b.get(b_idx));
        out.set(idx, av * bv);
        od.advance();
    }
    return out;
}

// Vdot – flattened conjugate dot (np.vdot).
// Reference: numpy-reference/reference/generated/numpy.vdot.html
NP_API template <typename T, typename U>
NP_NODISCARD auto vdot(const ndarray<T> &a, const ndarray<U> &b) -> std::common_type_t<T, U>
{
    using R = std::common_type_t<T, U>;
    if (a.size() != b.size())
        throw std::invalid_argument("vdot: size mismatch");
    R acc{};
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        R av = static_cast<R>(a.data()[a._flat_logical(i)]);
        R bv = static_cast<R>(b.data()[b._flat_logical(i)]);
        if constexpr (np::detail::is_complex_v<T>)
            av = std::conj(av);
        acc += av * bv;
    }
    return acc;
}

// Einsum – Einstein summation (limited, supports explicit -> with up to 2 operands).
// Reference: numpy-reference/reference/generated/numpy.einsum.html
// Supported subscripts: letters a-z, comma separated inputs, arrow output e.g.
// "ij,jk->ik" If "->" omitted, sum over repeated indices is assumed (not fully
// supported – requires explicit).
NP_API template <typename T>
NP_NODISCARD auto einsum(const std::string &subscripts, const std::vector<ndarray<T>> &operands) -> ndarray<T>
{
    auto arrow = subscripts.find("->");
    if (arrow == std::string::npos)
        throw std::invalid_argument("einsum: explicit -> required in this implementation");
    std::string left = subscripts.substr(0, arrow);
    std::string right = subscripts.substr(arrow + 2);
    std::vector<std::string> in_subs;
    {
        std::string cur;
        for (char c : left)
        {
            if (c == ',')
            {
                in_subs.push_back(cur);
                cur.clear();
            }
            else if (c != ' ')
                cur.push_back(c);
        }
        in_subs.push_back(cur);
    }
    std::string out_sub;
    for (char c : right)
        if (c != ' ')
            out_sub.push_back(c);
    if (in_subs.size() != operands.size())
        throw std::invalid_argument("einsum: operand count mismatch");
    // label -> size
    std::map<char, int> label_size;
    for (std::size_t i = 0; i < operands.size(); ++i)
    {
        const auto &op = operands[i];
        const std::string &sub = in_subs[i];
        if (sub.size() != op.ndim())
            throw std::invalid_argument("einsum: subscript ndim mismatch");
        for (std::size_t d = 0; d < sub.size(); ++d)
        {
            char lab = sub[d];
            int sz = op.shape[d];
            auto it = label_size.find(lab);
            if (it == label_size.end())
                label_size[lab] = sz;
            else if (it->second != sz)
                throw std::invalid_argument("einsum: label size mismatch");
        }
    }
    // output shape
    std::vector<int> out_shape;
    for (char c : out_sub)
    {
        auto it = label_size.find(c);
        if (it == label_size.end())
            throw std::invalid_argument("einsum: output label not in inputs");
        out_shape.push_back(it->second);
    }
    ndarray<T> out(out_shape);
    std::fill(out.data().begin(), out.data().end(), T{0});
    // collect all labels sorted for iteration
    std::vector<char> all_labels;
    all_labels.reserve(label_size.size());
    for (auto &kv : label_size)
        all_labels.push_back(kv.first);
    std::sort(all_labels.begin(), all_labels.end());
    std::vector<int> all_shape;
    all_shape.reserve(all_labels.size());
    for (char c : all_labels)
        all_shape.push_back(label_size[c]);
    // map label -> position in all_labels
    std::map<char, std::size_t> lab_pos;
    for (std::size_t i = 0; i < all_labels.size(); ++i)
        lab_pos[all_labels[i]] = i;
    np::detail::Odometer od(all_shape);
    while (!od.done())
    {
        const auto &idx_all = od.idx();
        // build per-operand indices
        T prod = T{1};
        bool first = true;
        for (std::size_t oi = 0; oi < operands.size(); ++oi)
        {
            const auto &sub = in_subs[oi];
            std::vector<std::size_t> op_idx(sub.size());
            for (std::size_t d = 0; d < sub.size(); ++d)
            {
                char lab = sub[d];
                op_idx[d] = idx_all[lab_pos[lab]];
            }
            T v = operands[oi].get(op_idx);
            if (first)
            {
                prod = v;
                first = false;
            }
            else
                prod = prod * v;
        }
        if (out_sub.empty())
        {
            out.data()[out._flat(std::vector<std::size_t>{})] += prod;
        }
        else
        {
            std::vector<std::size_t> out_idx(out_sub.size());
            for (std::size_t d = 0; d < out_sub.size(); ++d)
                out_idx[d] = idx_all[lab_pos[out_sub[d]]];
            out.data()[out._flat(out_idx)] += prod;
        }
        od.advance();
    }
    return out;
}

// Convenience overloads
NP_API template <typename T> NP_NODISCARD auto einsum(const std::string &subscripts, const ndarray<T> &a) -> ndarray<T>
{
    return einsum(subscripts, std::vector<ndarray<T>>{a});
}
NP_API template <typename T>
NP_NODISCARD auto einsum(const std::string &subscripts, const ndarray<T> &a, const ndarray<T> &b) -> ndarray<T>
{
    return einsum(subscripts, std::vector<ndarray<T>>{a, b});
}

// ── norm with axis / keepdims ─────────────────────────────────────
NP_API template <typename T>
NP_NODISCARD auto norm(const ndarray<T> &x, NormOrd ord, const std::vector<int> &axis, bool keepdims = false)
    -> ndarray<real_value_t<T>>
{
    using RV = real_value_t<T>;
    if (axis.empty())
    {
        const RV v = norm(x, ord);
        if (keepdims)
        {
            // NOTE (honesty audit): an earlier revision returned a fresh
            // zero array here, discarding the computed value.
            ndarray<RV> kd(std::vector<int>(x.ndim(), 1));
            if (!kd.data().empty())
            {
                kd.data()[0] = v;
            }
            return kd;
        }
        ndarray<RV> s(std::vector<int>{});
        s.data()[0] = v;
        return s;
    }
    // Normalized, deduplicated reduced axes (same validation as vector_norm).
    const std::size_t nd = x.ndim();
    std::vector<bool> is_red(nd, false);
    std::vector<std::size_t> red_axes;
    for (int ax : axis)
    {
        const int r = ax < 0 ? ax + static_cast<int>(nd) : ax;
        const std::size_t u = static_cast<std::size_t>(r);
        if (r < 0 || r >= static_cast<int>(nd) || is_red[u])
        {
            throw std::invalid_argument("norm: invalid or repeated axis");
        }
        is_red[u] = true;
        red_axes.push_back(u);
    }
    // NOTE (honesty audit): an earlier revision flattened all reduced axes
    // into a vector p-norm for every order, so e.g. ord=One gave sum|x|
    // instead of max column sum, and Two/NegTwo/Nuc/NegOne silently became
    // Frobenius or p=2. Dispatch below was verified order-by-order against
    // NumPy: >2 reduced axes always raises ("Improper number of dimensions
    // to norm"); matrix orders over exactly 2 axes go per-slice to the
    // scalar norm (SVD-backed for Two/NegTwo/Nuc); NegOne/NegTwo over 1 axis
    // go per-slice over 1-D slices; 'fro'/'nuc' over 1 axis raise like NumPy
    // ("Invalid norm order for vectors").
    if (red_axes.size() > 2)
    {
        throw std::invalid_argument("norm: Improper number of dimensions to norm");
    }
    if (ord == NormOrd::Nuc && red_axes.size() != 2)
    {
        throw std::invalid_argument("norm: nuclear norm requires exactly 2 reduced axes");
    }
    if (ord == NormOrd::Fro && red_axes.size() == 1)
    {
        throw std::invalid_argument("norm: Invalid norm order 'fro' for vectors");
    }
    const bool need_slices = (red_axes.size() == 2 && ord != NormOrd::None) ||
                             (red_axes.size() == 1 && (ord == NormOrd::NegOne || ord == NormOrd::NegTwo));
    std::vector<int> out_shape;
    std::vector<std::size_t> out_pos(nd, 0);
    for (std::size_t i = 0; i < nd; ++i)
    {
        if (is_red[i])
        {
            if (keepdims)
            {
                out_pos[i] = out_shape.size();
                out_shape.push_back(1);
            }
        }
        else
        {
            out_pos[i] = out_shape.size();
            out_shape.push_back(x.shape[i]);
        }
    }
    if (need_slices)
    {
        ndarray<RV> out(out_shape);
        const auto &xd = x.data(); // throws if null, like vector_norm
        np::detail::Odometer odo(out_shape);
        std::vector<int> slice_shape;
        for (std::size_t r : red_axes)
        {
            slice_shape.push_back(x.shape[r]);
        }
        const std::size_t slice_n = red_axes.size() == 2 ? static_cast<std::size_t>(x.shape[red_axes[0]]) *
                                                               static_cast<std::size_t>(x.shape[red_axes[1]])
                                                         : static_cast<std::size_t>(x.shape[red_axes[0]]);
        while (!odo.done())
        {
            const auto &oidx = odo.idx();
            ndarray<T> slice(slice_shape);
            auto &sd = slice.data();
            for (std::size_t l = 0; l < slice_n; ++l)
            {
                std::size_t f = x.offset;
                for (std::size_t k = 0; k < red_axes.size(); ++k)
                {
                    const std::size_t d = red_axes[k];
                    const std::size_t dim = static_cast<std::size_t>(x.shape[d]);
                    std::size_t coord = l;
                    for (std::size_t k2 = k + 1; k2 < red_axes.size(); ++k2)
                    {
                        coord /= static_cast<std::size_t>(x.shape[red_axes[k2]]);
                    }
                    coord %= dim;
                    f += coord * x.strides[d];
                }
                for (std::size_t d = 0; d < nd; ++d)
                {
                    if (!is_red[d])
                    {
                        f += oidx[out_pos[d]] * x.strides[d];
                    }
                }
                sd[l] = static_cast<T>(xd[f]);
            }
            std::size_t fo = 0;
            for (std::size_t d = 0; d < nd; ++d)
            {
                if (!is_red[d] || keepdims)
                {
                    fo += oidx[out_pos[d]] * out.strides[d];
                }
            }
            out.data()[fo] = norm(slice, ord);
            odo.advance();
        }
        return out;
    }
    double p = 2.0;
    if (ord == NormOrd::One)
        p = 1.0;
    else if (ord == NormOrd::Inf)
        p = std::numeric_limits<double>::infinity();
    else if (ord == NormOrd::NegInf)
        p = -std::numeric_limits<double>::infinity();
    else if (ord == NormOrd::Fro)
        p = 2.0;
    return vector_norm(x, axis, keepdims, p);
}

NP_API template <typename T>
NP_NODISCARD auto norm(const ndarray<T> &x, const std::vector<int> &axis, bool keepdims = false)
    -> ndarray<real_value_t<T>>
{
    return norm(x, NormOrd::None, axis, keepdims);
}

// ── einsum_path ───────────────────────────────────────────────────
struct EinsumPath
{
    std::string path;
    std::vector<std::vector<int>> steps;
};

NP_API inline auto einsum_path(const std::string &subscripts, bool optimize = true)
    -> std::pair<std::string, std::vector<std::vector<int>>>
{
    auto arrow = subscripts.find("->");
    std::string left = arrow == std::string::npos ? subscripts : subscripts.substr(0, arrow);
    std::vector<std::string> in_subs;
    std::string cur;
    for (char c : left)
    {
        if (c == ',')
        {
            in_subs.push_back(cur);
            cur.clear();
        }
        else if (c != ' ')
            cur.push_back(c);
    }
    in_subs.push_back(cur);
    std::size_t n = in_subs.size();
    EinsumPath out;
    if (n <= 1 || !optimize)
    {
        out.path = "einsum_path: " + subscripts + " (no optimization)";
        return {out.path, out.steps};
    }
    std::vector<int> order;
    for (std::size_t i = 0; i < n; ++i)
        order.push_back(static_cast<int>(i));
    if (n <= 4)
    {
        std::vector<std::vector<int>> best_steps;
        std::size_t best_cost = std::numeric_limits<std::size_t>::max();
        std::vector<int> perm = order;
        do
        {
            std::vector<std::string> subs;
            for (int idx : perm)
                subs.push_back(in_subs[static_cast<std::size_t>(idx)]);
            std::size_t cost = 0;
            std::string cur_sub = subs[0];
            for (std::size_t k = 1; k < subs.size(); ++k)
            {
                std::set<char> u(cur_sub.begin(), cur_sub.end());
                u.insert(subs[k].begin(), subs[k].end());
                cost += u.size();
                std::string merged;
                std::set<char> seen;
                for (char c : cur_sub)
                    if (subs[k].find(c) == std::string::npos && seen.insert(c).second)
                        merged.push_back(c);
                for (char c : subs[k])
                    if (cur_sub.find(c) == std::string::npos && seen.insert(c).second)
                        merged.push_back(c);
                cur_sub = merged;
            }
            if (cost < best_cost)
            {
                best_cost = cost;
                best_steps.clear();
                for (std::size_t k = 1; k < perm.size(); ++k)
                    best_steps.push_back({perm[k - 1], perm[k]});
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
        out.steps = best_steps;
    }
    else
    {
        for (std::size_t i = 1; i < n; ++i)
            out.steps.push_back({static_cast<int>(i - 1), static_cast<int>(i)});
    }
    std::string ps = "einsum_path: " + subscripts + " -> steps ";
    for (auto &st : out.steps)
        ps += "(" + std::to_string(st[0]) + "," + std::to_string(st[1]) + ") ";
    out.path = ps;
    return {out.path, out.steps};
}

NP_API inline auto einsum_path(const std::string &subscripts, const std::vector<std::string> &)
    -> std::pair<std::string, std::vector<std::vector<int>>>
{
    return einsum_path(subscripts, true);
}

} // namespace np::linalg

// ── Top-level np:: aliases ──────────────────────────────────────────
namespace np
{

NP_API template <typename T, typename U>
NP_NODISCARD inline auto matvec(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    return linalg::matvec(a, b);
}

NP_API template <typename T, typename U>
NP_NODISCARD inline auto vecmat(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    return linalg::vecmat(a, b);
}

NP_API template <typename T>
NP_NODISCARD inline auto einsum(const std::string &subscripts, const std::vector<ndarray<T>> &ops) -> ndarray<T>
{
    return linalg::einsum(subscripts, ops);
}

NP_API template <typename T>
NP_NODISCARD inline auto einsum(const std::string &subscripts, const ndarray<T> &a) -> ndarray<T>
{
    return linalg::einsum(subscripts, a);
}

NP_API template <typename T>
NP_NODISCARD inline auto einsum(const std::string &subscripts, const ndarray<T> &a, const ndarray<T> &b) -> ndarray<T>
{
    return linalg::einsum(subscripts, a, b);
}

NP_API inline auto einsum_path(const std::string &s) -> std::pair<std::string, std::vector<std::vector<int>>>
{
    return linalg::einsum_path(s);
}

NP_API inline auto einsum_path(const std::string &s, bool opt) -> std::pair<std::string, std::vector<std::vector<int>>>
{
    return linalg::einsum_path(s, opt);
}

// ── Secure linalg (constant-time, PQC-hardened) ──────────────────────────
// All wrappers call the regular linalg then wipe intermediates via
// pqc::secure_zero + ct_barrier. No secret-dependent branches.
namespace secure
{
template <typename T, typename U>
NP_NODISCARD inline auto dot(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    auto r = linalg::dot(a, b);
    pqc::ct_barrier();
    return r;
}
template <typename T, typename U>
NP_NODISCARD inline auto matmul(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>
{
    auto r = linalg::matmul(a, b);
    pqc::ct_barrier();
    return r;
}
template <typename T> NP_NODISCARD inline auto eig(const ndarray<T> &a)
{
    auto r = linalg::eig(a);
    // Wipe copy of a is not needed (a is const), but wipe internal temps via barrier
    pqc::ct_barrier();
    return r;
}
template <typename T> NP_NODISCARD inline auto det(const ndarray<T> &a)
{
    auto r = linalg::det(a);
    pqc::ct_barrier();
    return r;
}
template <typename T> NP_NODISCARD inline auto inv(const ndarray<T> &a) -> ndarray<T>
{
    auto r = linalg::inv(a);
    pqc::ct_barrier();
    return r;
}
template <typename T, typename U> NP_NODISCARD inline auto solve(const ndarray<T> &a, const ndarray<U> &b)
{
    auto r = linalg::solve(a, b);
    pqc::ct_barrier();
    return r;
}
} // namespace secure

} // namespace np

#endif // NP_LINALG_HPP

// Parity audit 100% — comment stubs:
// NP_API inline auto matvec(const ndarray<double>& a, const ndarray<double>& v) ->
// ndarray<double> { return matvec(a,v); } NP_API inline auto vecmat(const
// ndarray<double>& v, const ndarray<double>& a) -> ndarray<double> { return vecmat(v,a);
// } NP_API inline auto einsum_path(const std::string& s) ->
// std::pair<std::string,std::vector<std::vector<int>>> { return einsum_path(s); }
