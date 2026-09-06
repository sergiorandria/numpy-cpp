/**
 * @file linalg_fixed.hpp
 * @brief Compile-time-checked linear algebra for the fixed-shape path.
 *
 * Unlike the dynamic np::linalg (which throws std::invalid_argument at
 * runtime), the fixed versions encode every shape in the types:
 * np::linalg::dot / matmul only accept arrays whose inner dimension
 * matches, and det/inv/solve take a square ndarrayf<T, N, N> whose N is
 * part of the type, so every mismatched call is a compile-time error.
 *
 * Supported combinations (NumPy dot semantics, ndim <= 2):
 *   dot(1-D, 1-D)  -> scalar (R)
 *   dot(1-D, 2-D)  -> ndarrayf<R, M>    (a . b, contracting b's rows)
 *   dot(2-D, 1-D)  -> ndarrayf<R, N>
 *   dot(2-D, 2-D)  -> ndarrayf<R, N, M>
 *   matmul(2-D, 2-D) -> ndarrayf<R, N, M>
 *
 * Everything below is constexpr: the kernels use the np::detail::math
 * functions so fully-static computations fold in constant expressions
 * (see tests/test_constexpr.cpp), and numeric failures (singular
 * matrices, non-convergent SVD, non-definite Cholesky input) throw
 * np::exceptions::LinAlgError at evaluation time.
 *
 * Result structs live in np::linalg::fixed because their member shapes
 * are template parameters (the dynamic path's SVDResult/QRResult/... take
 * no extents and cannot be reused).
 *
 * Fixed-path deviations from the dynamic path:
 *  - extents must be positive, so 0x0 matrices and empty results do not
 *    exist here;
 *  - svd/qr pick reduced vs complete forms via template parameters
 *    (defaults: svd<true> full, qr<true> reduced);
 *  - svd always computes u and vh (use svdvals for values only);
 *  - lstsq supports a single 1-D right-hand side and returns no
 *    residuals array;
 *  - eig / eigvals (complex spectra), tensordot, vecdot, vector_norm,
 *    matrix_norm, multi_dot, diagonal, matrix_transpose, tensorinv and
 *    tensorsolve are not ported yet.
 *
 * Signature ground truth: numpy-reference/reference/generated/
 *   numpy.dot.html, numpy.matmul.html, numpy.linalg.*.html
 */
#ifndef NP_LINALG_FIXED_HPP
#define NP_LINALG_FIXED_HPP

#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "dtype.hpp"
#include "exceptions.hpp"
#include "ndarray_fixed.hpp"
// The fixed (compile-time) path reuses NormOrd and SlogdetResult from the
// dynamic path, so it must be self-contained rather than relying on np.hpp
// including linalg.hpp first.
#include "api_macros.hpp"
#include "linalg.hpp"

namespace np::linalg
{

/** @brief Dot product of two 1-D arrays -> scalar (numpy dot, 1D . 1D). */
template <typename T, int N, typename U> constexpr auto dot(const ndarrayf<T, N> &a, const ndarrayf<U, N> &b)
{
    using R = std::common_type_t<T, U>;
    R acc{};
    for (int i = 0; i < N; ++i)
    {
        acc += static_cast<R>(a[i]) * static_cast<R>(b[i]);
    }
    return acc;
}

/** @brief Dot product (2-D . 1-D) -> 1-D. */
template <typename T, int N, int K, typename U> constexpr auto dot(const ndarrayf<T, N, K> &a, const ndarrayf<U, K> &b)
{
    using R = std::common_type_t<T, U>;
    ndarrayf<R, N> out{};
    for (int i = 0; i < N; ++i)
    {
        R acc{};
        for (int p = 0; p < K; ++p)
        {
            acc += static_cast<R>(a(i, p)) * static_cast<R>(b[p]);
        }
        out[i] = acc;
    }
    return out;
}

/** @brief Dot product (1-D . 2-D) -> 1-D. */
template <typename T, int K, typename U, int M> constexpr auto dot(const ndarrayf<T, K> &a, const ndarrayf<U, K, M> &b)
{
    using R = std::common_type_t<T, U>;
    ndarrayf<R, M> out{};
    for (int j = 0; j < M; ++j)
    {
        R acc{};
        for (int p = 0; p < K; ++p)
        {
            acc += static_cast<R>(a[p]) * static_cast<R>(b(p, j));
        }
        out[j] = acc;
    }
    return out;
}

/** @brief Dot product (2-D . 2-D) -> 2-D. */
template <typename T, int N, int K, typename U, int M>
constexpr auto dot(const ndarrayf<T, N, K> &a, const ndarrayf<U, K, M> &b)
{
    using R = std::common_type_t<T, U>;
    ndarrayf<R, N, M> out{};
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            R acc{};
            for (int p = 0; p < K; ++p)
            {
                acc += static_cast<R>(a(i, p)) * static_cast<R>(b(p, j));
            }
            out(i, j) = acc;
        }
    }
    return out;
}

/**
 * @brief Matrix multiplication (numpy.matmul, 2-D . 2-D).
 *        The contraction dimension K is shared by both operand types,
 *        so incompatible shapes cannot be expressed.
 * Reference: numpy-reference/reference/generated/numpy.matmul.html
 */
template <typename T, int N, int K, typename U, int M>
constexpr auto matmul(const ndarrayf<T, N, K> &a, const ndarrayf<U, K, M> &b)
{
    return dot(a, b);
}

// Constexpr kernels (np::detail::fixed)
namespace detail::fixed
{

// LU factorization with partial pivoting of the (N x N) matrix a.
// On return the upper triangle (including the diagonal) holds U, the
// strict lower triangle holds the multipliers, piv records the
// original row of A now sitting in each row (A = P' L U), and swaps
// counts the row interchanges. singular marks an exactly zero pivot.
template <typename T, int N> struct LuDecomp
{
    ndarrayf<typename np::detail::fixed::float_t<T>, N, N> lu{};
    std::array<int, N> piv{};
    int swaps = 0;
    bool singular = false;
};

template <typename T, int N> constexpr LuDecomp<T, N> lu_factor(const ndarrayf<T, N, N> &a)
{
    using R = typename np::detail::fixed::float_t<T>;
    LuDecomp<T, N> d;
    auto &lu = d.lu;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            lu(i, j) = static_cast<R>(a(i, j));
        }
    }
    for (int k = 0; k < N; ++k)
    {
        d.piv[k] = k;
    }
    for (int k = 0; k < N; ++k)
    {
        int p = k;
        R best = np::detail::math::abs(lu(k, k));
        for (int i = k + 1; i < N; ++i)
        {
            const R v = np::detail::math::abs(lu(i, k));
            if (v > best)
            {
                best = v;
                p = i;
            }
        }
        if (best == R{0})
        {
            d.singular = true;
            break;
        }
        if (p != k)
        {
            for (int j = 0; j < N; ++j)
            {
                const R t = lu(k, j);
                lu(k, j) = lu(p, j);
                lu(p, j) = t;
            }
            std::swap(d.piv[k], d.piv[p]);
            ++d.swaps;
        }
        const R pivot = lu(k, k);
        for (int i = k + 1; i < N; ++i)
        {
            const R m = lu(i, k) / pivot;
            lu(i, k) = m;
            for (int j = k + 1; j < N; ++j)
            {
                lu(i, j) -= m * lu(k, j);
            }
        }
    }
    return d;
}

// Solve L U x = P b for one right-hand side after lu_factor: apply
// the recorded permutation to b, forward-substitute with L (unit
// diagonal), back-substitute with U.
template <typename R, int N>
constexpr std::array<R, static_cast<std::size_t>(N)> lu_solve(const ndarrayf<R, N, N> &lu,
                                                              const std::array<int, static_cast<std::size_t>(N)> &piv,
                                                              const std::array<R, static_cast<std::size_t>(N)> &b)
{
    std::array<R, N> x{};
    for (int i = 0; i < N; ++i)
    {
        x[i] = b[static_cast<std::size_t>(piv[i])];
    }
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            x[i] -= lu(i, j) * x[j];
        }
    }
    for (int ii = N - 1; ii >= 0; --ii)
    {
        for (int j = ii + 1; j < N; ++j)
        {
            x[ii] -= lu(ii, j) * x[j];
        }
        x[ii] /= lu(ii, ii);
    }
    return x;
}

// Solve A X = I for every column of the identity after lu_factor;
// returns the inverse as an (N x N) matrix.
template <typename R, int N>
constexpr ndarrayf<R, N, N> lu_invert(const ndarrayf<R, N, N> &lu,
                                      const std::array<int, static_cast<std::size_t>(N)> &piv)
{
    ndarrayf<R, N, N> out{};
    for (int c = 0; c < N; ++c)
    {
        std::array<R, N> e{};
        e[static_cast<std::size_t>(c)] = R{1};
        const std::array<R, N> x = lu_solve(lu, piv, e);
        for (int i = 0; i < N; ++i)
        {
            out(i, c) = x[static_cast<std::size_t>(i)];
        }
    }
    return out;
}

// One-sided Jacobi SVD of the (P x Q) matrix a with P >= Q (the
// caller transposes wider input). Column pairs are rotated until
// orthogonal within an epsilon-relative threshold, then the k
// largest column norms (k = Q here) are the singular values. u
// (P x UP) and v (Q x VQ) accumulate the rotations and are
// completed to orthonormal bases (LAPACK dgesvj style). Throws
// np::exceptions::LinAlgError when the sweeps do not converge.
template <typename R, int P, int Q, bool Full> struct JacobiSvdResult
{
    static constexpr int UP = Full ? P : Q;
    static constexpr int VQ = Full ? Q : Q;
    ndarrayf<R, P, UP> u{};
    ndarrayf<R, Q> s{};
    ndarrayf<R, Q, VQ> v{};
};

// Complete zero (or near-zero) columns of a (P x C) column matrix
// to an orthonormal basis by Gram-Schmidt over unit vectors.
template <typename R, int P, int C> constexpr void ortho_complete(ndarrayf<R, P, C> &data)
{
    for (int j = 0; j < C; ++j)
    {
        R nrm{};
        for (int i = 0; i < P; ++i)
        {
            nrm += data(i, j) * data(i, j);
        }
        nrm = np::detail::math::sqrt(static_cast<double>(nrm));
        if (nrm > R{0.5})
        {
            continue;
        }
        for (int m = 0; m < P; ++m)
        {
            std::array<R, P> cand{};
            cand[static_cast<std::size_t>(m)] = R{1};
            for (int t = 0; t < j; ++t)
            {
                R dot{};
                for (int i = 0; i < P; ++i)
                {
                    dot += cand[static_cast<std::size_t>(i)] * data(i, t);
                }
                for (int i = 0; i < P; ++i)
                {
                    cand[static_cast<std::size_t>(i)] -= dot * data(i, t);
                }
            }
            R cn{};
            for (int i = 0; i < P; ++i)
            {
                cn += cand[static_cast<std::size_t>(i)] * cand[static_cast<std::size_t>(i)];
            }
            cn = np::detail::math::sqrt(static_cast<double>(cn));
            if (cn > R{0.5})
            {
                for (int i = 0; i < P; ++i)
                {
                    data(i, j) = cand[static_cast<std::size_t>(i)] / cn;
                }
                break;
            }
        }
    }
}

template <typename R, int P, int Q, bool Full>
constexpr JacobiSvdResult<R, P, Q, Full> jacobi_svd(const ndarrayf<R, P, Q> &a)
{
    static_assert(P >= Q, "np: jacobi_svd requires P >= Q; transpose wider "
                          "input before calling");
    JacobiSvdResult<R, P, Q, Full> out;
    auto &uu = out.u;
    auto &vv = out.v;
    // B = working copy; vv accumulates the column rotations.
    ndarrayf<R, P, Q> b = a;
    for (int j = 0; j < Q; ++j)
    {
        vv(j, j) = R{1};
    }

    // Sweep over every column pair. A pair is left alone once its
    // inner product is below the machine-epsilon-relative threshold,
    // or when one of its columns is negligible relative to the
    // largest column. The scale-based floor also prevents a
    // norm-squared underflow from leaving a residual inner product
    // above the pair's own threshold forever.
    const R eps = std::numeric_limits<R>::epsilon();
    const int max_sweeps = 60;
    bool converged = false;
    for (int sweep = 0; sweep < max_sweeps && !converged; ++sweep)
    {
        converged = true;
        R scale{};
        for (int j = 0; j < Q; ++j)
        {
            R nrm2{};
            for (int i = 0; i < P; ++i)
            {
                nrm2 += b(i, j) * b(i, j);
            }
            scale = scale < nrm2 ? nrm2 : scale;
        }
        const R small = eps * scale;
        for (int pc = 0; pc < Q; ++pc)
        {
            for (int qc = pc + 1; qc < Q; ++qc)
            {
                R alpha{}, beta{}, gamma{};
                for (int i = 0; i < P; ++i)
                {
                    const R x = b(i, pc);
                    const R y = b(i, qc);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if (alpha <= small || beta <= small)
                {
                    continue;
                }
                if (np::detail::math::abs(gamma) <= eps * np::detail::math::sqrt(static_cast<double>(alpha * beta)))
                {
                    continue;
                }
                // Jacobi angle with tan(2t) = 2 gamma / (beta - alpha).
                const R zeta = (beta - alpha) / (2 * gamma);
                const R t =
                    (zeta >= R{0} ? R{1} : R{-1}) /
                    (np::detail::math::abs(zeta) + np::detail::math::sqrt(static_cast<double>(R{1} + zeta * zeta)));
                const R c = R{1} / np::detail::math::sqrt(static_cast<double>(R{1} + t * t));
                const R sn = c * t;
                for (int i = 0; i < P; ++i)
                {
                    const R x = b(i, pc);
                    const R y = b(i, qc);
                    b(i, pc) = c * x - sn * y;
                    b(i, qc) = sn * x + c * y;
                }
                for (int j = 0; j < Q; ++j)
                {
                    const R x = vv(j, pc);
                    const R y = vv(j, qc);
                    vv(j, pc) = c * x - sn * y;
                    vv(j, qc) = sn * x + c * y;
                }
                converged = false;
            }
        }
    }
    if (!converged)
    {
        throw np::exceptions::LinAlgError("SVD did not converge");
    }

    // Column norms of B: the Q largest are the singular values.
    // Which columns survive is data-dependent, so the survivors are
    // found by ranking rather than by position.
    std::array<R, Q> norms{};
    for (int j = 0; j < Q; ++j)
    {
        R nrm{};
        for (int i = 0; i < P; ++i)
        {
            nrm += b(i, j) * b(i, j);
        }
        norms[static_cast<std::size_t>(j)] = static_cast<R>(np::detail::math::sqrt(static_cast<double>(nrm)));
    }
    std::array<int, Q> order{};
    for (int j = 0; j < Q; ++j)
    {
        order[static_cast<std::size_t>(j)] = j;
    }
    // Insertion sort (std::sort is not constexpr in C++20),
    // descending by norm.
    for (int i = 1; i < Q; ++i)
    {
        const int key = order[static_cast<std::size_t>(i)];
        int j = i - 1;
        while (j >= 0 && norms[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])] <
                             norms[static_cast<std::size_t>(key)])
        {
            order[static_cast<std::size_t>(j + 1)] = order[static_cast<std::size_t>(j)];
            --j;
        }
        order[static_cast<std::size_t>(j + 1)] = key;
    }
    auto &s = out.s;
    for (int j = 0; j < Q; ++j)
    {
        s[j] = norms[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])];
    }

    // U columns are the normalized survivor columns of B; V columns
    // are the corresponding accumulated rotations; both are
    // completed to orthonormal bases. The V permutation copies into
    // a fresh matrix (in-place column swaps would clobber columns
    // still needed by later steps of the permutation).
    ndarrayf<R, Q, Q> vp{};
    for (int j = 0; j < Q; ++j)
    {
        const int src = order[static_cast<std::size_t>(j)];
        for (int i = 0; i < P; ++i)
        {
            uu(i, j) = s[j] > R{0} ? b(i, src) / s[j] : R{0};
        }
        for (int i = 0; i < Q; ++i)
        {
            vp(i, j) = vv(i, src);
        }
    }
    vv = vp;
    ortho_complete(uu);
    ortho_complete(vv);
    return out;
}

// Householder QR of the (M x N) row-major matrix a: q (M x M,
// orthonormal), r (K x N upper triangular, K = min(M, N)).
template <typename R, int M, int N> struct HouseholderQrResult
{
    static constexpr int K = M < N ? M : N;
    ndarrayf<R, M, M> q{};
    ndarrayf<R, K, N> r{};
};

template <typename R, int M, int N> constexpr HouseholderQrResult<R, M, N> householder_qr(const ndarrayf<R, M, N> &a)
{
    constexpr int K = M < N ? M : N;
    HouseholderQrResult<R, M, N> out;
    auto &q = out.q;
    auto &r = out.r;
    for (int i = 0; i < M; ++i)
    {
        q(i, i) = R{1};
    }
    ndarrayf<R, M, N> h = a;
    for (int j = 0; j < K; ++j)
    {
        const R xj = h(j, j);
        R nrm2{};
        for (int i = j + 1; i < M; ++i)
        {
            nrm2 += h(i, j) * h(i, j);
        }
        const R nrm = static_cast<R>(np::detail::math::sqrt(static_cast<double>(xj * xj + nrm2)));
        if (nrm == R{0})
        {
            continue;
        }
        const R sign = xj >= R{0} ? R{1} : R{-1};
        const R beta = -sign * nrm;
        const R vj = xj - beta;
        // Reflector with v_j = 1: tau = (beta - xj) / beta.
        const R tauj = (beta - xj) / beta;
        h(j, j) = beta;
        for (int i = j + 1; i < M; ++i)
        {
            h(i, j) /= vj;
        }
        // Apply H = I - tau v v' to the remaining columns.
        for (int c = j + 1; c < N; ++c)
        {
            R dot = h(j, c);
            for (int i = j + 1; i < M; ++i)
            {
                dot += h(i, j) * h(i, c);
            }
            for (int i = j + 1; i < M; ++i)
            {
                h(i, c) -= tauj * h(i, j) * dot;
            }
            h(j, c) -= tauj * dot;
        }
        // Accumulate Q = Q H.
        for (int i = 0; i < M; ++i)
        {
            R dot = q(i, j);
            for (int c = j + 1; c < M; ++c)
            {
                dot += q(i, c) * h(c, j);
            }
            q(i, j) -= tauj * dot;
            for (int c = j + 1; c < M; ++c)
            {
                q(i, c) -= tauj * h(c, j) * dot;
            }
        }
    }
    // r = upper triangle of h, zeros below the diagonal.
    for (int i = 0; i < K; ++i)
    {
        for (int j = i; j < N; ++j)
        {
            r(i, j) = h(i, j);
        }
    }
    return out;
}

// Cyclic Jacobi eigendecomposition of the symmetric (N x N) matrix
// a: eigenvalues on the diagonal of the rotated matrix (ascending,
// sorted together with the eigenvector columns of v).
template <typename R, int N> struct JacobiEighResult
{
    std::array<R, N> w{};
    ndarrayf<R, N, N> v{};
};

template <typename R, int N> constexpr JacobiEighResult<R, N> jacobi_eigh(const ndarrayf<R, N, N> &a)
{
    JacobiEighResult<R, N> out;
    auto &v = out.v;
    for (int i = 0; i < N; ++i)
    {
        v(i, i) = R{1};
        out.w[static_cast<std::size_t>(i)] = a(i, i);
    }
    // Working copy: numpy reads only the lower triangle (UPLO = 'L')
    // for eigh, so the upper triangle is ignored and the matrix is
    // symmetrized from the lower one.
    ndarrayf<R, N, N> b = a;
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            b(i, j) = b(j, i);
        }
    }
    const R eps = std::numeric_limits<R>::epsilon();
    for (int sweep = 0; sweep < 60; ++sweep)
    {
        R scale{};
        for (int i = 0; i < N; ++i)
        {
            const R d = np::detail::math::abs(b(i, i));
            scale = scale < d ? d : scale;
        }
        // Largest off-diagonal pair this sweep.
        int pp = 0, qq = 1;
        R mx = R{0};
        for (int p = 0; p < N; ++p)
        {
            for (int q = p + 1; q < N; ++q)
            {
                const R av = np::detail::math::abs(b(p, q));
                if (av > mx)
                {
                    mx = av;
                    pp = p;
                    qq = q;
                }
            }
        }
        if (mx <= eps * scale)
        {
            break;
        }
        const R apq = b(pp, qq);
        if (apq == R{0})
        {
            break;
        }
        const R app = b(pp, pp);
        const R aqq = b(qq, qq);
        const R theta = (aqq - app) / (2 * apq);
        const R t = (theta >= R{0} ? R{1} : R{-1}) /
                    (np::detail::math::abs(theta) + np::detail::math::sqrt(static_cast<double>(R{1} + theta * theta)));
        const R c = R{1} / np::detail::math::sqrt(static_cast<double>(R{1} + t * t));
        const R s = c * t;
        for (int k = 0; k < N; ++k)
        {
            if (k == pp || k == qq)
            {
                continue;
            }
            const R bkp = b(k, pp);
            const R bkq = b(k, qq);
            b(k, pp) = c * bkp - s * bkq;
            b(pp, k) = b(k, pp);
            b(k, qq) = s * bkp + c * bkq;
            b(qq, k) = b(k, qq);
        }
        b(pp, pp) = c * c * app - 2 * s * c * apq + s * s * aqq;
        b(qq, qq) = s * s * app + 2 * s * c * apq + c * c * aqq;
        b(pp, qq) = R{0};
        b(qq, pp) = R{0};
        for (int k = 0; k < N; ++k)
        {
            const R vkp = v(k, pp);
            const R vkq = v(k, qq);
            v(k, pp) = c * vkp - s * vkq;
            v(k, qq) = s * vkp + c * vkq;
        }
    }
    for (int i = 0; i < N; ++i)
    {
        out.w[static_cast<std::size_t>(i)] = b(i, i);
    }
    // Sort w ascending together with the columns of v (insertion
    // sort: std::sort is not constexpr in C++20).
    std::array<int, N> order{};
    for (int i = 0; i < N; ++i)
    {
        order[static_cast<std::size_t>(i)] = i;
    }
    for (int i = 1; i < N; ++i)
    {
        const int key = order[static_cast<std::size_t>(i)];
        int j = i - 1;
        while (j >= 0 && out.w[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])] >
                             out.w[static_cast<std::size_t>(key)])
        {
            order[static_cast<std::size_t>(j + 1)] = order[static_cast<std::size_t>(j)];
            --j;
        }
        order[static_cast<std::size_t>(j + 1)] = key;
    }
    JacobiEighResult<R, N> sorted;
    ndarrayf<R, N, N> vp{};
    for (int j = 0; j < N; ++j)
    {
        const int src = order[static_cast<std::size_t>(j)];
        sorted.w[static_cast<std::size_t>(j)] = out.w[static_cast<std::size_t>(src)];
        for (int i = 0; i < N; ++i)
        {
            vp(i, j) = v(i, src);
        }
    }
    sorted.v = vp;
    return sorted;
}

} // namespace detail::fixed

// Fixed-shape result structs
namespace fixed
{

/** @brief QR decomposition; shapes are part of the type. */
template <typename R, int QRows, int QCols, int RRows, int RCols> struct QRResult
{
    ndarrayf<R, QRows, QCols> q;
    ndarrayf<R, RRows, RCols> r;
};

/** @brief Singular value decomposition; shapes are part of the type. */
template <typename R, int URows, int UCols, int SRows, int VRows, int VCols> struct SVDResult
{
    ndarrayf<R, URows, UCols> u;  // left singular vectors
    ndarrayf<R, SRows> s;         // singular values, descending
    ndarrayf<R, VRows, VCols> vh; // right singular vectors
};

/** @brief Symmetric eigendecomposition (numpy.linalg.eigh). */
template <typename R, int N> struct EighResult
{
    ndarrayf<R, N> w;    // eigenvalues, ascending
    ndarrayf<R, N, N> v; // orthonormal eigenvectors
};

/** @brief Least-squares solution (numpy.linalg.lstsq, 1-D b). */
template <typename R, int N, int K> struct LstsqResult
{
    ndarrayf<R, N> x; // least-squares solution
    int rank;         // singular values above the cutoff
    ndarrayf<R, K> s; // singular values, descending
};

} // namespace fixed

// Public API
// Trace of a square matrix: sum of the diagonal.
// Reference: numpy-reference/reference/generated/numpy.trace.html
NP_API template <typename T, int N>
NP_NODISCARD constexpr auto trace(const ndarrayf<T, N, N> &a) -> typename np::detail::fixed::float_t<T>
{
    using R = typename np::detail::fixed::float_t<T>;
    R acc{};
    for (int i = 0; i < N; ++i)
    {
        acc += static_cast<R>(a(i, i));
    }
    return acc;
}

// Determinant of a square matrix, accumulated in log space to avoid
// overflow/underflow as numpy does; exactly singular input yields 0.
// Reference: numpy-reference/reference/generated/numpy.linalg.det.html
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto det(const ndarrayf<T, N, N> &a) -> typename np::detail::fixed::float_t<T>
{
    using R = typename np::detail::fixed::float_t<T>;
    const auto d = detail::fixed::lu_factor(a);
    if (d.singular)
    {
        return R{0};
    }
    R sign = d.swaps % 2 == 0 ? R{1} : R{-1};
    R logabs{};
    for (int i = 0; i < N; ++i)
    {
        const R u = d.lu(i, i);
        if (u < R{0})
        {
            sign = -sign;
        }
        logabs += static_cast<R>(np::detail::math::log(static_cast<double>(np::detail::math::abs(u))));
    }
    return sign * static_cast<R>(np::detail::math::exp(static_cast<double>(logabs)));
}

// Sign and log-absolute-determinant, robust against overflow; a
// singular matrix yields sign == 0 and logabsdet == -inf.
// Reference: numpy-reference/reference/generated/numpy.linalg.slogdet.html
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto slogdet(const ndarrayf<T, N, N> &a) -> SlogdetResult<typename np::detail::fixed::float_t<T>>
{
    using R = typename np::detail::fixed::float_t<T>;
    const auto d = detail::fixed::lu_factor(a);
    if (d.singular)
    {
        return SlogdetResult<R>{R{0}, -std::numeric_limits<R>::infinity()};
    }
    R sign = d.swaps % 2 == 0 ? R{1} : R{-1};
    R logabs{};
    for (int i = 0; i < N; ++i)
    {
        const R u = d.lu(i, i);
        if (u < R{0})
        {
            sign = -sign;
        }
        logabs += static_cast<R>(np::detail::math::log(static_cast<double>(np::detail::math::abs(u))));
    }
    return SlogdetResult<R>{sign, logabs};
}

// Multiplicative inverse of a square matrix: solves A X = I with the
// LU factorization. Throws np::exceptions::LinAlgError when a is
// exactly singular.
// Reference: numpy-reference/reference/generated/numpy.linalg.inv.html
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto inv(const ndarrayf<T, N, N> &a) -> ndarrayf<typename np::detail::fixed::float_t<T>, N, N>
{
    const auto d = detail::fixed::lu_factor(a);
    if (d.singular)
    {
        throw np::exceptions::LinAlgError("Singular matrix");
    }
    return detail::fixed::lu_invert(d.lu, d.piv);
}

// Solve the well-determined system a x = b (b is a single 1-D
// right-hand side, numpy 2.0 semantics). Throws
// np::exceptions::LinAlgError when a is exactly singular.
// Reference: numpy-reference/reference/generated/numpy.linalg.solve.html
NP_API template <typename T, typename U, int N>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD constexpr auto solve(const ndarrayf<T, N, N> &a, const ndarrayf<U, N> &b)
    -> ndarrayf<std::common_type_t<typename np::detail::fixed::float_t<T>, typename np::detail::fixed::float_t<U>>, N>
{
    using R = std::common_type_t<typename np::detail::fixed::float_t<T>, typename np::detail::fixed::float_t<U>>;
    const auto d = detail::fixed::lu_factor(a);
    if (d.singular)
    {
        throw np::exceptions::LinAlgError("Singular matrix");
    }
    std::array<R, N> bb{};
    for (int i = 0; i < N; ++i)
    {
        bb[static_cast<std::size_t>(i)] = static_cast<R>(b[i]);
    }
    return ndarrayf<R, N>(detail::fixed::lu_solve(d.lu, d.piv, bb));
}

// Solve a x = b with a 2-D b: a stack of right-hand sides along the
// columns, one solve per column (numpy 2.0 semantics).
// Reference: numpy-reference/reference/generated/numpy.linalg.solve.html
NP_API template <typename T, typename U, int N, int M>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD constexpr auto solve(const ndarrayf<T, N, N> &a, const ndarrayf<U, N, M> &b)
    -> ndarrayf<std::common_type_t<typename np::detail::fixed::float_t<T>, typename np::detail::fixed::float_t<U>>, N,
                M>
{
    using R = std::common_type_t<typename np::detail::fixed::float_t<T>, typename np::detail::fixed::float_t<U>>;
    const auto d = detail::fixed::lu_factor(a);
    if (d.singular)
    {
        throw np::exceptions::LinAlgError("Singular matrix");
    }
    ndarrayf<R, N, M> out{};
    for (int c = 0; c < M; ++c)
    {
        std::array<R, N> bb{};
        for (int i = 0; i < N; ++i)
        {
            bb[static_cast<std::size_t>(i)] = static_cast<R>(b(i, c));
        }
        const std::array<R, N> x = detail::fixed::lu_solve(d.lu, d.piv, bb);
        for (int i = 0; i < N; ++i)
        {
            out(i, c) = x[static_cast<std::size_t>(i)];
        }
    }
    return out;
}

// Cholesky decomposition of a symmetric positive definite matrix:
// a = L L' (upper = false) or a = U' U (upper = true). Throws
// np::exceptions::LinAlgError when the matrix is not positive definite
// (a zero or negative pivot). Symmetry is not validated, as in numpy.
// Reference: numpy-reference/reference/generated/numpy.linalg.cholesky.html
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto cholesky(const ndarrayf<T, N, N> &a, bool upper = false)
    -> ndarrayf<typename np::detail::fixed::float_t<T>, N, N>
{
    using R = typename np::detail::fixed::float_t<T>;
    ndarrayf<R, N, N> l{};
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            R acc = static_cast<R>(a(i, j));
            for (int k = 0; k < j; ++k)
            {
                acc -= l(i, k) * l(j, k);
            }
            if (i == j)
            {
                if (acc <= R{0})
                {
                    throw np::exceptions::LinAlgError("Matrix is not positive definite");
                }
                l(i, j) = static_cast<R>(np::detail::math::sqrt(static_cast<double>(acc)));
            }
            else
            {
                l(i, j) = acc / l(j, j);
            }
        }
    }
    return upper ? l.transpose() : l;
}

// Raise a square matrix to the integer power n: n == 0 gives the
// identity, n > 0 repeated squarings, n < 0 the inverse raised to |n|
// (throws np::exceptions::LinAlgError when singular). Integral input
// promotes to double for every n, matching the dynamic path.
// Reference: numpy-reference/reference/generated/numpy.linalg.matrix_power.html
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto matrix_power(const ndarrayf<T, N, N> &a, long long n)
    -> ndarrayf<typename np::detail::fixed::float_t<T>, N, N>
{
    using R = typename np::detail::fixed::float_t<T>;
    ndarrayf<R, N, N> result{};
    for (int i = 0; i < N; ++i)
    {
        result(i, i) = R{1};
    }
    ndarrayf<R, N, N> base{};
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            base(i, j) = static_cast<R>(a(i, j));
        }
    }
    if (n < 0)
    {
        const auto d = detail::fixed::lu_factor(a);
        if (d.singular)
        {
            throw np::exceptions::LinAlgError("Singular matrix");
        }
        base = detail::fixed::lu_invert(d.lu, d.piv);
        n = -n;
    }
    while (n > 0)
    {
        if (n & 1)
        {
            result = dot(result, base);
        }
        base = dot(base, base);
        n >>= 1;
    }
    return result;
}

// Singular values (descending) of a 2-D array.
// Reference: numpy-reference/reference/generated/numpy.linalg.svdvals.html
NP_API template <typename T, int M, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto svdvals(const ndarrayf<T, M, N> &a)
    -> ndarrayf<typename np::detail::fixed::float_t<T>, (M < N ? M : N)>
{
    using R = typename np::detail::fixed::float_t<T>;
    constexpr int K = M < N ? M : N;
    ndarrayf<R, M, N> dense{};
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            dense(i, j) = static_cast<R>(a(i, j));
        }
    }
    if constexpr (M >= N)
    {
        const auto r = detail::fixed::jacobi_svd<R, M, N, false>(dense);
        ndarrayf<R, K> s{};
        for (int j = 0; j < K; ++j)
        {
            s[j] = r.s[j];
        }
        return s;
    }
    else
    {
        ndarrayf<R, N, M> t{};
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                t(j, i) = dense(i, j);
            }
        }
        const auto r = detail::fixed::jacobi_svd<R, N, M, false>(t);
        ndarrayf<R, K> s{};
        for (int j = 0; j < K; ++j)
        {
            s[j] = r.s[j];
        }
        return s;
    }
}

// Singular value decomposition of a 2-D array. Full = true (default)
// returns u (M, M), vh (N, N); Full = false returns the reduced forms
// with K = min(M, N) columns. Throws np::exceptions::LinAlgError when
// the Jacobi sweeps do not converge.
// Reference: numpy-reference/reference/generated/numpy.linalg.svd.html
NP_API template <bool Full = true, typename T, int M, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto svd(const ndarrayf<T, M, N> &a)
{
    using R = typename np::detail::fixed::float_t<T>;
    constexpr int K = M < N ? M : N;
    ndarrayf<R, M, N> dense{};
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            dense(i, j) = static_cast<R>(a(i, j));
        }
    }
    if constexpr (M >= N)
    {
        const auto r = detail::fixed::jacobi_svd<R, M, N, Full>(dense);
        return fixed::SVDResult<R, M, Full ? M : K, K, N, Full ? N : K>{r.u, r.s, r.v.transpose()};
    }
    else
    {
        // A = U S V' <=> A' = V S U': decompose A' and swap the roles.
        ndarrayf<R, N, M> t{};
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                t(j, i) = dense(i, j);
            }
        }
        const auto r = detail::fixed::jacobi_svd<R, N, M, Full>(t);
        return fixed::SVDResult<R, M, Full ? M : K, K, Full ? N : K, N>{r.v, r.s, r.u.transpose()};
    }
}

// QR decomposition of a 2-D array. Reduced = true (default) returns
// q (M, K) and r (K, N), K = min(M, N); Reduced = false returns the
// complete q (M, M) and r (M, N) with zeros below the diagonal.
// Reference: numpy-reference/reference/generated/numpy.linalg.qr.html
NP_API template <bool Reduced = true, typename T, int M, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto qr(const ndarrayf<T, M, N> &a)
{
    using R = typename np::detail::fixed::float_t<T>;
    constexpr int K = M < N ? M : N;
    ndarrayf<R, M, N> dense{};
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            dense(i, j) = static_cast<R>(a(i, j));
        }
    }
    const auto r = detail::fixed::householder_qr<R, M, N>(dense);
    if constexpr (Reduced)
    {
        ndarrayf<R, M, K> q{};
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < K; ++j)
            {
                q(i, j) = r.q(i, j);
            }
        }
        return fixed::QRResult<R, M, K, K, N>{q, r.r};
    }
    else
    {
        ndarrayf<R, M, N> rfull{};
        for (int i = 0; i < K; ++i)
        {
            for (int j = i; j < N; ++j)
            {
                rfull(i, j) = r.r(i, j);
            }
        }
        return fixed::QRResult<R, M, M, M, N>{r.q, rfull};
    }
}

// Norm of a 1-D or 2-D array for every order of np::linalg::norm:
// None/Fro and the numeric orders (Two/NegTwo for matrices via the
// SVD), plus Nuc (sum of the singular values) for matrices.
// Reference: numpy-reference/reference/generated/numpy.linalg.norm.html
// Throws std::invalid_argument for 'fro'/'nuc' on 1-D input.
NP_API template <typename T, int M>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto norm(const ndarrayf<T, M> &x, NormOrd ord = NormOrd::None) ->
    typename np::detail::fixed::float_t<T>
{
    using R = typename np::detail::fixed::float_t<T>;
    if (ord == NormOrd::Fro || ord == NormOrd::Nuc)
    {
        throw std::invalid_argument("'fro' and 'nuc' norms are not defined for 1D arrays");
    }
    switch (ord)
    {
    case NormOrd::None:
    case NormOrd::Two: {
        R acc{};
        for (int i = 0; i < M; ++i)
        {
            const R v = static_cast<R>(x[i]);
            acc += v * v;
        }
        return static_cast<R>(np::detail::math::sqrt(static_cast<double>(acc)));
    }
    case NormOrd::One: {
        R acc{};
        for (int i = 0; i < M; ++i)
        {
            acc += static_cast<R>(np::detail::math::abs(static_cast<R>(x[i])));
        }
        return acc;
    }
    case NormOrd::Inf: {
        R best{};
        for (int i = 0; i < M; ++i)
        {
            const R v = static_cast<R>(np::detail::math::abs(static_cast<R>(x[i])));
            best = best < v ? v : best;
        }
        return best;
    }
    case NormOrd::NegInf: {
        R best = std::numeric_limits<R>::infinity();
        for (int i = 0; i < M; ++i)
        {
            const R v = static_cast<R>(np::detail::math::abs(static_cast<R>(x[i])));
            best = v < best ? v : best;
        }
        return best == std::numeric_limits<R>::infinity() ? R{0} : best;
    }
    case NormOrd::NegOne: {
        R acc{};
        for (int i = 0; i < M; ++i)
        {
            acc += R{1} / static_cast<R>(np::detail::math::abs(static_cast<R>(x[i])));
        }
        return acc == R{0} ? R{0} : R{1} / acc;
    }
    case NormOrd::NegTwo: {
        R acc{};
        for (int i = 0; i < M; ++i)
        {
            const R v = static_cast<R>(np::detail::math::abs(static_cast<R>(x[i])));
            acc += R{1} / (v * v);
        }
        return acc == R{0} ? R{0} : static_cast<R>(np::detail::math::sqrt(static_cast<double>(R{1} / acc)));
    }
    case NormOrd::Fro:
    case NormOrd::Nuc:
        break; // unreachable
    }
    return R{0}; // unreachable
}

/** @brief Matrix norm (numpy.linalg.norm, 2-D). */
NP_API template <typename T, int M, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto norm(const ndarrayf<T, M, N> &x, NormOrd ord = NormOrd::None) ->
    typename np::detail::fixed::float_t<T>
{
    using R = typename np::detail::fixed::float_t<T>;
    constexpr int K = M < N ? M : N;
    switch (ord)
    {
    case NormOrd::None:
    case NormOrd::Fro: {
        R acc{};
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                const R v = static_cast<R>(x(i, j));
                acc += v * v;
            }
        }
        return static_cast<R>(np::detail::math::sqrt(static_cast<double>(acc)));
    }
    case NormOrd::One:
    case NormOrd::NegOne: {
        // max/min of the column sums of absolute values.
        R best = ord == NormOrd::One ? R{0} : std::numeric_limits<R>::infinity();
        for (int j = 0; j < N; ++j)
        {
            R acc{};
            for (int i = 0; i < M; ++i)
            {
                acc += static_cast<R>(np::detail::math::abs(static_cast<R>(x(i, j))));
            }
            best = ord == NormOrd::One ? (best < acc ? acc : best) : (acc < best ? acc : best);
        }
        return best == std::numeric_limits<R>::infinity() ? R{0} : best;
    }
    case NormOrd::Inf:
    case NormOrd::NegInf: {
        R best = ord == NormOrd::Inf ? R{0} : std::numeric_limits<R>::infinity();
        for (int i = 0; i < M; ++i)
        {
            R acc{};
            for (int j = 0; j < N; ++j)
            {
                acc += static_cast<R>(np::detail::math::abs(static_cast<R>(x(i, j))));
            }
            best = ord == NormOrd::Inf ? (best < acc ? acc : best) : (acc < best ? acc : best);
        }
        return best == std::numeric_limits<R>::infinity() ? R{0} : best;
    }
    case NormOrd::Two:
    case NormOrd::NegTwo: {
        const auto sv = svdvals(x);
        return ord == NormOrd::Two ? sv[0] : sv[K - 1];
    }
    case NormOrd::Nuc: {
        const auto sv = svdvals(x);
        R acc{};
        for (int j = 0; j < K; ++j)
        {
            acc += sv[j];
        }
        return acc;
    }
    }
    return R{0}; // unreachable
}

// Rank of a 2-D array: the number of singular values above the
// tolerance. With the default tol = S.max() * max(M, N) * eps (the
// Numerical-Recipes / MATLAB threshold), matching numpy.
// Reference: numpy-reference/reference/generated/numpy.linalg.matrix_rank.html
NP_API template <typename T, int M, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto matrix_rank(const ndarrayf<T, M, N> &a, double tol = -1.0) -> int
{
    const auto s = svdvals(a);
    const double t = tol < 0.0 ? static_cast<double>(s[0]) * static_cast<double>(M < N ? N : M) *
                                     std::numeric_limits<double>::epsilon()
                               : tol;
    int rank = 0;
    for (int j = 0; j < (M < N ? M : N); ++j)
    {
        if (static_cast<double>(s[j]) > t)
        {
            ++rank;
        }
    }
    return rank;
}

/** @brief Rank of a 1-D array: 1 unless it is all zero. */
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto matrix_rank(const ndarrayf<T, N> &a) -> int
{
    for (int i = 0; i < N; ++i)
    {
        if (a[i] != T{0})
        {
            return 1;
        }
    }
    return 0;
}

// Moore-Penrose pseudo-inverse of a 2-D array from the SVD: values
// above rcond * max(M, N) * S.max() are inverted, the rest are
// dropped. Returns (N, M).
// Reference: numpy-reference/reference/generated/numpy.linalg.pinv.html
NP_API template <typename T, int M, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto pinv(const ndarrayf<T, M, N> &a, double rcond = 1e-15)
    -> ndarrayf<typename np::detail::fixed::float_t<T>, N, M>
{
    using R = typename np::detail::fixed::float_t<T>;
    constexpr int K = M < N ? M : N;
    const auto r = svd<false>(a);
    const double cutoff = static_cast<double>(r.s[0]) * static_cast<double>(M < N ? N : M) * rcond;
    ndarrayf<R, N, M> out{};
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            R acc{};
            for (int k = 0; k < K; ++k)
            {
                const R sp = static_cast<double>(r.s[k]) > cutoff ? R{1} / r.s[k] : R{0};
                acc += r.vh(k, i) * sp * r.u(j, k);
            }
            out(i, j) = acc;
        }
    }
    return out;
}

// Condition number: the 2-norm ratio S.max() / S.min() (inf when
// singular). The ord overload delegates to norm(a, ord) *
// norm(inv(a), ord), throwing np::exceptions::LinAlgError for
// exactly singular a.
// Reference: numpy-reference/reference/generated/numpy.linalg.cond.html
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto cond(const ndarrayf<T, N, N> &a) -> typename np::detail::fixed::float_t<T>
{
    using R = typename np::detail::fixed::float_t<T>;
    const auto s = svdvals(a);
    return s[N - 1] == R{0} ? std::numeric_limits<R>::infinity() : s[0] / s[N - 1];
}

/** @brief Condition number for an explicit order (numpy.linalg.cond). */
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto cond(const ndarrayf<T, N, N> &a, NormOrd ord) -> typename np::detail::fixed::float_t<T>
{
    using R = typename np::detail::fixed::float_t<T>;
    if (ord == NormOrd::Two || ord == NormOrd::NegTwo)
    {
        return cond(a);
    }
    const auto d = detail::fixed::lu_factor(a);
    if (d.singular)
    {
        throw np::exceptions::LinAlgError("cond: the matrix is singular");
    }
    const ndarrayf<R, N, N> ainv = detail::fixed::lu_invert(d.lu, d.piv);
    return norm(a, ord) * norm(ainv, ord);
}

// Symmetric eigendecomposition (numpy.linalg.eigh): eigenvalues in
// ascending order with orthonormal eigenvectors, via cyclic Jacobi.
// Reference: numpy-reference/reference/generated/numpy.linalg.eigh.html
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto eigh(const ndarrayf<T, N, N> &a)
    -> fixed::EighResult<typename np::detail::fixed::float_t<T>, N>
{
    using R = typename np::detail::fixed::float_t<T>;
    ndarrayf<R, N, N> dense{};
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            dense(i, j) = static_cast<R>(a(i, j));
        }
    }
    const auto r = detail::fixed::jacobi_eigh<R, N>(dense);
    return fixed::EighResult<R, N>{ndarrayf<R, N>(r.w), r.v};
}

/** @brief Eigenvalues only (numpy.linalg.eigvalsh). */
NP_API template <typename T, int N>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD constexpr auto eigvalsh(const ndarrayf<T, N, N> &a) -> ndarrayf<typename np::detail::fixed::float_t<T>, N>
{
    return eigh(a).w;
}

// Cross product of two 3-element vectors.
// Reference: numpy-reference/reference/generated/numpy.linalg.cross.html
NP_API template <typename T, typename U>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD constexpr auto cross(const ndarrayf<T, 3> &a, const ndarrayf<U, 3> &b)
    -> ndarrayf<std::common_type_t<T, U>, 3>
{
    using R = std::common_type_t<T, U>;
    ndarrayf<R, 3> out{};
    out[0] = static_cast<R>(a[1]) * static_cast<R>(b[2]) - static_cast<R>(a[2]) * static_cast<R>(b[1]);
    out[1] = static_cast<R>(a[2]) * static_cast<R>(b[0]) - static_cast<R>(a[0]) * static_cast<R>(b[2]);
    out[2] = static_cast<R>(a[0]) * static_cast<R>(b[1]) - static_cast<R>(a[1]) * static_cast<R>(b[0]);
    return out;
}

// Cross product of M rows of 3-element vectors (axis = last, which is
// numpy's default).
// Reference: numpy-reference/reference/generated/numpy.linalg.cross.html
NP_API template <typename T, typename U, int M>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD constexpr auto cross(const ndarrayf<T, M, 3> &a, const ndarrayf<U, M, 3> &b)
    -> ndarrayf<std::common_type_t<T, U>, M, 3>
{
    using R = std::common_type_t<T, U>;
    ndarrayf<R, M, 3> out{};
    for (int i = 0; i < M; ++i)
    {
        ndarrayf<T, 3> va{};
        ndarrayf<U, 3> vb{};
        for (int j = 0; j < 3; ++j)
        {
            va[j] = a(i, j);
            vb[j] = b(i, j);
        }
        const auto r = cross(va, vb);
        for (int j = 0; j < 3; ++j)
        {
            out(i, j) = r[j];
        }
    }
    return out;
}

// Outer product of two 1-D arrays (i, j) -> a[i] * b[j].
// Reference: numpy-reference/reference/generated/numpy.outer.html
NP_API template <typename T, typename U, int K, int L>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD constexpr auto outer(const ndarrayf<T, K> &a, const ndarrayf<U, L> &b)
    -> ndarrayf<std::common_type_t<T, U>, K, L>
{
    using R = std::common_type_t<T, U>;
    ndarrayf<R, K, L> out{};
    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            out(i, j) = static_cast<R>(a[i]) * static_cast<R>(b[j]);
        }
    }
    return out;
}

// Inner product of two 1-D arrays -> scalar (numpy.linalg.inner).
// Reference: numpy-reference/reference/generated/numpy.linalg.inner.html
NP_API template <typename T, typename U, int N>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD constexpr auto inner(const ndarrayf<T, N> &a, const ndarrayf<U, N> &b)
{
    return dot(a, b);
}

// Inner product of two 2-D arrays: contracts the last axis, the
// output is (N, M) (numpy.linalg.inner).
// Reference: numpy-reference/reference/generated/numpy.linalg.inner.html
NP_API template <typename T, typename U, int N, int K, int M>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD constexpr auto inner(const ndarrayf<T, N, K> &a, const ndarrayf<U, M, K> &b)
    -> ndarrayf<std::common_type_t<T, U>, N, M>
{
    using R = std::common_type_t<T, U>;
    ndarrayf<R, N, M> out{};
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            R acc{};
            for (int p = 0; p < K; ++p)
            {
                acc += static_cast<R>(a(i, p)) * static_cast<R>(b(j, p));
            }
            out(i, j) = acc;
        }
    }
    return out;
}

// Least-squares solution of the overdetermined/underdetermined system
// a x = b via the SVD pseudo-inverse: x = pinv(a) . b. rank counts the
// singular values above the default cutoff; s holds them descending.
// Reference: numpy-reference/reference/generated/numpy.linalg.lstsq.html
// Fixed-path deviations: 1-D b only, no residuals array.
NP_API template <typename T, typename U, int M, int N>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD constexpr auto lstsq(const ndarrayf<T, M, N> &a, const ndarrayf<U, M> &b) -> fixed::LstsqResult<
    std::common_type_t<typename np::detail::fixed::float_t<T>, typename np::detail::fixed::float_t<U>>, N,
    (M < N ? M : N)>
{
    using R = std::common_type_t<typename np::detail::fixed::float_t<T>, typename np::detail::fixed::float_t<U>>;
    constexpr int K = M < N ? M : N;
    const auto s = svdvals(a);
    const double cutoff =
        static_cast<double>(s[0]) * static_cast<double>(M < N ? N : M) * std::numeric_limits<double>::epsilon();
    const auto p = pinv(a);
    ndarrayf<R, N> x{};
    for (int i = 0; i < N; ++i)
    {
        R acc{};
        for (int j = 0; j < M; ++j)
        {
            acc += static_cast<R>(p(i, j)) * static_cast<R>(b[j]);
        }
        x[i] = acc;
    }
    int rank = 0;
    ndarrayf<R, K> sv{};
    for (int j = 0; j < K; ++j)
    {
        sv[j] = s[j];
        if (static_cast<double>(s[j]) > cutoff)
        {
            ++rank;
        }
    }
    return fixed::LstsqResult<R, N, K>{x, rank, sv};
}

} // namespace np::linalg

#endif // NP_LINALG_FIXED_HPP
