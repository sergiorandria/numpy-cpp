/**
 * @file test_linalg.cpp
 * @brief Tests for np::linalg functions.
 */
#include <cmath>
#include <complex>
#include <stdexcept>

#include "np/np.hpp"
#include "test_util.hpp"

using namespace np;

// Verify eig output against A v = w v, unit columns, and the numpy pairing
// of w[j] with v[:, j]; returns the result for further invariant checks.
static np::linalg::EigenResult<double> check_eig(const ndarray<int> &a)
{
    auto e = linalg::eig(a);
    const int n = a.shape[0];
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            std::complex<double> lhs{};
            for (int t = 0; t < n; ++t)
            {
                lhs += std::complex<double>(static_cast<double>(a(i, t)), 0.0) * e.v(t, j);
            }
            std::complex<double> rhs = e.w(j) * e.v(i, j);
            test::check(std::abs(lhs - rhs) < 1e-7 * (std::abs(lhs) + std::abs(rhs) + 1.0), "eig residual A v = w v");
        }
        double nrm = 0.0;
        for (int i = 0; i < n; ++i)
        {
            nrm += std::norm(e.v(i, j));
        }
        test::check(test::approx(std::sqrt(nrm), 1.0), "eig unit column");
    }
    return e;
}

int main()
{
    // dot 1D . 1D -> scalar
    {
        ndarray<int> a{1, 2, 3};
        ndarray<int> b{4, 5, 6};
        auto d = linalg::dot(a, b);
        test::check(d.ndim() == 0, "dot scalar ndim");
        test::check(d.item() == 32, "dot 1D value");
    }

    // dot 2D . 2D
    {
        ndarray<int> a{{1, 2}, {3, 4}};
        ndarray<int> b{{5, 6}, {7, 8}};
        auto d = linalg::dot(a, b);
        test::check(d.shape[0] == 2 && d.shape[1] == 2, "dot 2D shape");
        test::check(d(0, 0) == 19 && d(0, 1) == 22, "dot 2D values");
        test::check(d(1, 0) == 43 && d(1, 1) == 50, "dot 2D values 2");
    }

    // dot mixed 1D/2D
    {
        ndarray<int> m{{1, 2}, {3, 4}};
        ndarray<int> v{1, 1};
        auto mv = linalg::dot(m, v);
        test::check(mv.ndim() == 1 && mv.size() == 2, "dot 2D.1D shape");
        test::check(mv(0) == 3 && mv(1) == 7, "dot 2D.1D values");
        auto vm = linalg::dot(v, m);
        test::check(vm.size() == 2 && vm(0) == 4 && vm(1) == 6, "dot 1D.2D values");
    }

    // matmul
    {
        ndarray<int> a{{1, 0}, {0, 1}};
        ndarray<int> b{{2, 3}, {4, 5}};
        auto mm = linalg::matmul(a, b);
        test::check(mm(1, 1) == 5, "matmul identity");
    }

    // inner
    {
        ndarray<int> a{1, 2, 3};
        ndarray<int> b{4, 5, 6};
        auto in = linalg::inner(a, b);
        test::check(in.ndim() == 0 && in.item() == 32, "inner 1D");
    }

    // outer
    {
        ndarray<int> a{1, 2};
        ndarray<int> b{3, 4};
        auto o = linalg::outer(a, b);
        test::check(o.shape[0] == 2 && o.shape[1] == 2, "outer shape");
        test::check(o(1, 0) == 6 && o(0, 1) == 4, "outer values");
    }

    // mixed types promote to common type
    {
        ndarray<int> a{1, 2};
        ndarray<double> b{0.5, 0.5};
        auto d = linalg::dot(a, b);
        test::check(test::approx(d.item(), 1.5), "dot mixed type");
    }

    // --- SVD ----------------------------------------------------------------

    // 2x3, full matrices: shapes, ordering, orthonormality, reconstruction
    {
        ndarray<int> a{{1, 2, 3}, {4, 5, 6}};
        auto r = linalg::svd(a);
        test::check(r.u.shape[0] == 2 && r.u.shape[1] == 2, "svd 2x3 u shape");
        test::check(r.s.shape[0] == 2, "svd 2x3 s shape");
        test::check(r.vh.shape[0] == 3 && r.vh.shape[1] == 3, "svd 2x3 vh shape");
        test::check(r.s(0) > r.s(1) && r.s(1) > 0.0, "svd values descending");
        // Frobenius norm squared = 91 = s0^2 + s1^2; product = sqrt(54).
        test::check(test::approx(r.s(0) * r.s(0) + r.s(1) * r.s(1), 91.0, 1e-12), "svd 2x3 frobenius invariant");
        test::check(test::approx(r.s(0) * r.s(1), std::sqrt(54.0), 1e-12), "svd 2x3 product invariant");
        auto utu = linalg::dot(r.u.transpose(), r.u);
        test::check(test::approx(utu(0, 0), 1.0) && test::approx(utu(0, 1), 0.0) && test::approx(utu(1, 1), 1.0),
                    "svd u orthonormal");
        auto vvt = linalg::dot(r.vh, r.vh.transpose());
        test::check(test::approx(vvt(0, 0), 1.0) && test::approx(vvt(0, 1), 0.0) && test::approx(vvt(1, 1), 1.0),
                    "svd vh orthonormal");
        ndarray<double> s_mat{{r.s(0), 0.0, 0.0}, {0.0, r.s(1), 0.0}};
        auto rec = linalg::dot(linalg::dot(r.u, s_mat), r.vh);
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                test::check(test::approx(rec(i, j), static_cast<double>(a(i, j)), 1e-12), "svd 2x3 reconstruction");
            }
        }
    }

    // rank-deficient: A = [[1,2],[2,4]] has sigma = {5, 0}
    {
        ndarray<int> a{{1, 2}, {2, 4}};
        auto r = linalg::svd(a);
        test::check(test::approx(r.s(0), 5.0), "svd rank-deficient sigma 1");
        test::check(std::abs(r.s(1)) < 1e-12, "svd rank-deficient sigma 2");
        auto rec = linalg::dot(linalg::dot(r.u, ndarray<double>{{r.s(0), 0.0}, {0.0, r.s(1)}}), r.vh);
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                test::check(test::approx(rec(i, j), static_cast<double>(a(i, j)), 1e-12),
                            "svd rank-deficient reconstruction");
            }
        }
    }

    // 3x2 (m > n) and the m < n transpose path
    {
        ndarray<int> a{{1, 2}, {3, 4}, {5, 6}};
        auto full = linalg::svd(a, true, true);
        test::check(full.u.shape[0] == 3 && full.u.shape[1] == 3, "svd 3x2 u full shape");
        test::check(full.vh.shape[0] == 2 && full.vh.shape[1] == 2, "svd 3x2 vh full shape");
        auto red = linalg::svd(a, false, true);
        test::check(red.u.shape[0] == 3 && red.u.shape[1] == 2, "svd 3x2 u reduced shape");
        test::check(red.vh.shape[0] == 2 && red.vh.shape[1] == 2, "svd 3x2 vh reduced shape");
        test::check(test::approx(full.s(0), red.s(0)) && test::approx(full.s(1), red.s(1)),
                    "svd full/reduced same sigma");
        ndarray<double> s_mat{{full.s(0), 0.0}, {0.0, full.s(1)}};
        auto rec = linalg::dot(linalg::dot(red.u, s_mat), red.vh);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                test::check(test::approx(rec(i, j), static_cast<double>(a(i, j)), 1e-12), "svd 3x2 reconstruction");
            }
        }
    }

    // singular values of a diagonal matrix; svdvals
    {
        ndarray<int> a{{3, 0}, {0, 4}};
        auto s = linalg::svdvals(a);
        test::check(s.shape[0] == 2, "svdvals shape");
        test::check(test::approx(s(0), 4.0) && test::approx(s(1), 3.0), "svdvals values");
    }

    // compute_uv = false leaves u and vh empty
    {
        ndarray<int> a{{1, 2}, {3, 4}};
        auto r = linalg::svd(a, true, false);
        test::check(r.s.shape[0] == 2, "svd no-uv s");
        test::check(r.u.size() == 0 && r.vh.size() == 0, "svd no-uv empties");
    }

    // error paths
    {
        ndarray<int> v{1, 2, 3};
        bool threw = false;
        try
        {
            (void)linalg::svd(v);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "svd rejects 1D input");
    }

    // --- QR -----------------------------------------------------------------

    // reduced mode on 3x2: reconstruction, orthonormality, triangularity
    {
        ndarray<int> a{{1, 2}, {3, 4}, {5, 6}};
        auto r = linalg::qr(a);
        test::check(r.q.shape[0] == 3 && r.q.shape[1] == 2, "qr reduced q shape");
        test::check(r.r.shape[0] == 2 && r.r.shape[1] == 2, "qr reduced r shape");
        test::check(r.q.size() > 0 && r.r.size() > 0, "qr reduced populated");
        test::check(test::approx(r.r(1, 0), 0.0, 1e-14), "qr r upper triangular");
        auto rec = linalg::dot(r.q, r.r);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                test::check(test::approx(rec(i, j), static_cast<double>(a(i, j)), 1e-12), "qr reduced reconstruction");
            }
        }
        auto qtq = linalg::dot(r.q.transpose(), r.q);
        test::check(test::approx(qtq(0, 0), 1.0) && test::approx(qtq(0, 1), 0.0) && test::approx(qtq(1, 1), 1.0),
                    "qr q orthonormal");
    }

    // complete mode: q (M, M), r (M, N)
    {
        ndarray<int> a{{1, 2}, {3, 4}, {5, 6}};
        auto r = linalg::qr(a, linalg::QrMode::Complete);
        test::check(r.q.shape[0] == 3 && r.q.shape[1] == 3, "qr complete q shape");
        test::check(r.r.shape[0] == 3 && r.r.shape[1] == 2, "qr complete r shape");
        test::check(test::approx(r.r(2, 0), 0.0) && test::approx(r.r(2, 1), 0.0), "qr complete r trailing zeros");
        auto rec = linalg::dot(r.q, r.r);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                test::check(test::approx(rec(i, j), static_cast<double>(a(i, j)), 1e-12), "qr complete reconstruction");
            }
        }
    }

    // r mode and raw mode
    {
        ndarray<int> a{{1, 2}, {3, 4}, {5, 6}};
        auto r = linalg::qr(a, linalg::QrMode::R);
        test::check(r.r.shape[0] == 2 && r.r.shape[1] == 2, "qr r mode shape");
        test::check(r.q.size() == 0, "qr r mode no q");
        auto raw = linalg::qr(a, linalg::QrMode::Raw);
        test::check(raw.h.shape[0] == 3 && raw.h.shape[1] == 2, "qr raw h shape");
        test::check(raw.tau.shape[0] == 2, "qr raw tau shape");
        for (int i = 0; i < 2; ++i)
        {
            for (int j = i; j < 2; ++j)
            {
                test::check(test::approx(raw.h(i, j), r.r(i, j), 1e-14), "qr raw upper triangle matches r");
            }
        }
        // reconstruct q from (h, tau) and verify A = q r
        const int m = 3;
        std::vector<std::vector<double>> qq(m, std::vector<double>(m, 0.0));
        for (int i = 0; i < m; ++i)
        {
            qq[i][i] = 1.0;
        }
        for (int j = 0; j < 2; ++j)
        {
            std::vector<double> v(m, 0.0);
            v[j] = 1.0;
            for (int i = j + 1; i < m; ++i)
            {
                v[i] = raw.h(i, j);
            }
            const double tau = raw.tau(j);
            for (int rr = 0; rr < m; ++rr)
            {
                double dot = 0.0;
                for (int c = j; c < m; ++c)
                {
                    dot += qq[rr][c] * v[c];
                }
                for (int c = j; c < m; ++c)
                {
                    qq[rr][c] -= tau * v[c] * dot;
                }
            }
        }
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                double rec = 0.0;
                for (int t = 0; t < 2; ++t)
                {
                    rec += qq[i][t] * r.r(t, j);
                }
                test::check(test::approx(rec, static_cast<double>(a(i, j)), 1e-12), "qr raw h/tau reconstruction");
            }
        }
    }

    // empty 0x3 input
    {
        ndarray<double> a(std::vector<int>{0, 3});
        auto r = linalg::qr(a);
        test::check(r.q.shape[0] == 0 && r.q.shape[1] == 0, "qr empty q shape");
        test::check(r.r.shape[0] == 0 && r.r.shape[1] == 3, "qr empty r shape");
    }

    // --- eig ----------------------------------------------------------------

    // symmetric 2x2: eigenvalues {3, 1}
    {
        ndarray<int> a{{2, 1}, {1, 2}};
        auto e = linalg::eig(a);
        test::check(e.w.shape[0] == 2, "eig w shape");
        test::check(e.v.shape[0] == 2 && e.v.shape[1] == 2, "eig v shape");
        const double hi = std::max(e.w(0).real(), e.w(1).real());
        const double lo = std::min(e.w(0).real(), e.w(1).real());
        test::check(test::approx(hi, 3.0) && test::approx(lo, 1.0), "eig symmetric values");
        check_eig(a);
    }

    // complex pair from a rotation block
    {
        ndarray<int> a{{0, -1, 0}, {1, 0, 0}, {0, 0, 3}};
        auto e = linalg::eig(a);
        int complex_count = 0;
        for (int j = 0; j < 3; ++j)
        {
            if (std::abs(e.w(j).imag()) > 0.5)
            {
                ++complex_count;
            }
        }
        test::check(complex_count == 2, "eig rotation complex pair count");
        std::complex<double> sum{};
        for (int j = 0; j < 3; ++j)
        {
            sum += e.w(j);
        }
        test::check(test::approx(sum.real(), 3.0) && test::approx(sum.imag(), 0.0), "eig rotation trace invariant");
        check_eig(a);
    }

    // defective 2x2 Jordan block: only one eigenvector exists, so the
    // A v = w v residual cannot hold for a full basis (numpy behaves the
    // same); here we only verify the double eigenvalue.
    {
        ndarray<int> a{{1, 1}, {0, 1}};
        auto e = linalg::eig(a);
        test::check(test::approx(e.w(0).real(), 1.0) && test::approx(e.w(0).imag(), 0.0) &&
                        test::approx(e.w(1).real(), 1.0) && test::approx(e.w(1).imag(), 0.0),
                    "eig defective values");
    }

    // 4x4 block diagonal with two complex pairs: trace and det invariants
    {
        ndarray<int> a{{0, -1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, -2}, {0, 0, 2, 0}};
        auto e = check_eig(a);
        std::complex<double> sum{}, prod{1.0};
        for (int j = 0; j < 4; ++j)
        {
            sum += e.w(j);
            prod *= e.w(j);
        }
        test::check(test::approx(sum.real(), 0.0) && test::approx(sum.imag(), 0.0), "eig 4x4 trace invariant");
        test::check(test::approx(prod.real(), 4.0) && test::approx(prod.imag(), 0.0), "eig 4x4 det invariant");
    }

    // eigvals and the 1x1 / empty / triangular edge cases
    {
        ndarray<int> a{{3, 0}, {0, 3}};
        auto w = linalg::eigvals(a);
        test::check(w.shape[0] == 2, "eigvals shape");
        test::check(test::approx(w(0).real(), 3.0) && test::approx(w(1).real(), 3.0), "eigvals values");
    }
    {
        ndarray<int> a = ndarray<int>::from_data(std::vector<int>{1, 1}, std::vector<int>{7});
        auto e = linalg::eig(a);
        test::check(test::approx(e.w(0).real(), 7.0) && test::approx(e.w(0).imag(), 0.0) &&
                        test::approx(e.v(0, 0).real(), 1.0),
                    "eig 1x1");
    }
    {
        ndarray<int> a{{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
        auto e = check_eig(a);
        test::check(test::approx(e.w(0).real(), 1.0) && test::approx(e.w(1).real(), 4.0) &&
                        test::approx(e.w(2).real(), 6.0),
                    "eig triangular values");
    }
    {
        ndarray<double> a(std::vector<int>{0, 0});
        auto e = linalg::eig(a);
        test::check(e.w.size() == 0 && e.v.size() == 0, "eig empty");
    }

    // error paths
    {
        ndarray<int> m{{1, 2, 3}, {4, 5, 6}};
        bool threw = false;
        try
        {
            (void)linalg::eig(m);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "eig rejects non-square");
        ndarray<int> v{1, 2};
        threw = false;
        try
        {
            (void)linalg::eig(v);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "eig rejects 1D input");
    }

    // --- det / slogdet ------------------------------------------------------

    {
        ndarray<int> a{{1, 2}, {3, 4}};
        test::check(test::approx(linalg::det(a), -2.0), "det 2x2");
        auto sl = linalg::slogdet(a);
        test::check(test::approx(sl.sign, -1.0) && test::approx(sl.logabsdet, std::log(2.0)), "slogdet 2x2");
    }
    {
        ndarray<int> a{{1, 1, 2}, {3, 5, 8}, {13, 21, 34}};
        // 1*(5*34-8*21) - 1*(3*34-8*13) + 2*(3*21-5*13) = 2 - (-2) + 2*(-2) = 0
        test::check(test::approx(linalg::det(a), 0.0), "det singular 3x3");
        auto sl = linalg::slogdet(a);
        test::check(sl.sign == 0.0 && std::isinf(sl.logabsdet) && sl.logabsdet < 0.0, "slogdet singular");
    }
    {
        ndarray<double> a(std::vector<int>{0, 0});
        test::check(test::approx(linalg::det(a), 1.0), "det 0x0");
        auto sl = linalg::slogdet(a);
        test::check(test::approx(sl.sign, 1.0) && test::approx(sl.logabsdet, 0.0), "slogdet 0x0");
    }
    {
        ndarray<int> a = ndarray<int>::from_data(std::vector<int>{1, 1}, std::vector<int>{7});
        test::check(test::approx(linalg::det(a), 7.0), "det 1x1");
    }
    {
        ndarray<int> a{{0, 0}, {0, 0}};
        test::check(test::approx(linalg::det(a), 0.0), "det zero matrix");
    }

    // --- inv ----------------------------------------------------------------

    {
        ndarray<int> a{{4, 7}, {2, 6}};
        auto i = linalg::inv(a);
        test::check(test::approx(i(0, 0), 0.6) && test::approx(i(0, 1), -0.7) && test::approx(i(1, 0), -0.2) &&
                        test::approx(i(1, 1), 0.4),
                    "inv 2x2 values");
        auto id = linalg::matmul(a, i);
        test::check(test::approx(id(0, 0), 1.0) && test::approx(id(0, 1), 0.0) && test::approx(id(1, 0), 0.0) &&
                        test::approx(id(1, 1), 1.0),
                    "inv round trip");
    }
    {
        ndarray<int> a{{1, 2}, {2, 4}};
        bool threw = false;
        try
        {
            (void)linalg::inv(a);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "inv rejects singular");
    }
    {
        ndarray<double> a(std::vector<int>{0, 0});
        test::check(linalg::inv(a).size() == 0, "inv 0x0 empty");
    }

    // --- solve --------------------------------------------------------------

    {
        ndarray<int> a{{3, 1}, {1, 2}};
        ndarray<int> b{9, 8};
        auto x = linalg::solve(a, b);
        test::check(x.ndim() == 1 && test::approx(x(0), 2.0) && test::approx(x(1), 3.0), "solve 1D rhs");
        ndarray<int> b2{{9, 9}, {8, 8}};
        auto x2 = linalg::solve(a, b2);
        test::check(x2.ndim() == 2 && test::approx(x2(0, 0), 2.0) && test::approx(x2(0, 1), 2.0) &&
                        test::approx(x2(1, 0), 3.0) && test::approx(x2(1, 1), 3.0),
                    "solve 2D rhs");
        auto chk = linalg::matmul(a, x2);
        test::check(test::approx(chk(0, 1), 9.0) && test::approx(chk(1, 0), 8.0), "solve round trip");
    }
    {
        ndarray<int> a{{1, 2}, {2, 4}};
        ndarray<int> b{1, 1};
        bool threw = false;
        try
        {
            (void)linalg::solve(a, b);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "solve rejects singular");
    }

    // --- matrix_power -------------------------------------------------------

    {
        ndarray<int> a{{1, 1}, {1, 0}};
        auto p = linalg::matrix_power(a, 10);
        test::check(test::approx(p(0, 0), 89.0) && test::approx(p(1, 0), 55.0), "matrix_power fibonacci");
        auto z = linalg::matrix_power(a, 0);
        test::check(test::approx(z(0, 0), 1.0) && test::approx(z(1, 0), 0.0), "matrix_power zero");
        auto m = linalg::matrix_power(a, -1);
        test::check(test::approx(m(0, 0), 0.0) && test::approx(m(0, 1), 1.0) && test::approx(m(1, 0), 1.0) &&
                        test::approx(m(1, 1), -1.0),
                    "matrix_power negative");
    }

    // --- cholesky -----------------------------------------------------------

    {
        ndarray<int> a{{4, 2}, {2, 3}};
        auto l = linalg::cholesky(a);
        test::check(test::approx(l(0, 0), 2.0) && test::approx(l(1, 0), 1.0) && test::approx(l(0, 1), 0.0) &&
                        test::approx(l(1, 1), std::sqrt(2.0)),
                    "cholesky lower");
        auto rec = linalg::matmul(l, l.transpose());
        test::check(test::approx(rec(1, 0), 2.0) && test::approx(rec(1, 1), 3.0), "cholesky lower round trip");
        auto u = linalg::cholesky(a, true);
        test::check(test::approx(u(0, 0), 2.0) && test::approx(u(0, 1), 1.0) && test::approx(u(1, 1), std::sqrt(2.0)),
                    "cholesky upper");
    }
    {
        ndarray<int> a{{1, 2}, {2, 1}};
        bool threw = false;
        try
        {
            (void)linalg::cholesky(a);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "cholesky rejects indefinite");
    }

    // --- norm ---------------------------------------------------------------

    {
        ndarray<int> v{3, 4};
        test::check(test::approx(linalg::norm(v), 5.0), "norm vector 2");
        test::check(test::approx(linalg::norm(v, linalg::NormOrd::One), 7.0), "norm vector 1");
        test::check(test::approx(linalg::norm(v, linalg::NormOrd::Inf), 4.0), "norm vector inf");
        test::check(test::approx(linalg::norm(v, linalg::NormOrd::NegInf), 3.0), "norm vector -inf");
        test::check(test::approx(linalg::norm(v, linalg::NormOrd::NegOne), 12.0 / 7.0), "norm vector -1");
    }
    {
        ndarray<int> m{{1, 2}, {3, 4}};
        test::check(test::approx(linalg::norm(m), std::sqrt(30.0)), "norm matrix fro");
        test::check(test::approx(linalg::norm(m, linalg::NormOrd::One), 6.0), "norm matrix 1");
        test::check(test::approx(linalg::norm(m, linalg::NormOrd::Inf), 7.0), "norm matrix inf");
        test::check(test::approx(linalg::norm(m, linalg::NormOrd::NegOne), 4.0), "norm matrix -1");
        test::check(test::approx(linalg::norm(m, linalg::NormOrd::NegInf), 3.0), "norm matrix -inf");
        const double s0 = std::sqrt(15.0 + std::sqrt(221.0));
        const double s1 = std::sqrt(15.0 - std::sqrt(221.0));
        test::check(test::approx(linalg::norm(m, linalg::NormOrd::Two), s0), "norm matrix 2");
        test::check(test::approx(linalg::norm(m, linalg::NormOrd::NegTwo), s1), "norm matrix -2");
    }

    // --- matrix_rank --------------------------------------------------------

    {
        ndarray<int> a{{1, 2, 3}, {2, 4, 6}};
        test::check(linalg::matrix_rank(a) == 1, "rank dependent rows");
        ndarray<int> b{{1, 2}, {3, 4}};
        test::check(linalg::matrix_rank(b) == 2, "rank full");
        ndarray<int> z{{0, 0}, {0, 0}};
        test::check(linalg::matrix_rank(z) == 0, "rank zero");
        ndarray<int> v{1, 0, 0};
        test::check(linalg::matrix_rank(v) == 1, "rank 1D nonzero");
        ndarray<int> w{0, 0};
        test::check(linalg::matrix_rank(w) == 0, "rank 1D zero");
        ndarray<double> s{{1e-10, 0.0}, {0.0, 1.0}};
        test::check(linalg::matrix_rank(s, 1e-15) == 2, "rank tol loose");
        test::check(linalg::matrix_rank(s, 1e-5) == 1, "rank tol strict");
    }

    // --- pinv ---------------------------------------------------------------

    {
        ndarray<int> a{{1, 2}, {3, 4}};
        auto p = linalg::pinv(a);
        test::check(test::approx(p(0, 0), -2.0) && test::approx(p(0, 1), 1.0) && test::approx(p(1, 0), 1.5) &&
                        test::approx(p(1, 1), -0.5),
                    "pinv 2x2");
        ndarray<int> t{{1, 0}, {0, 1}, {0, 0}};
        auto pt = linalg::pinv(t);
        test::check(pt.shape[0] == 2 && pt.shape[1] == 3 && test::approx(pt(0, 0), 1.0) &&
                        test::approx(pt(1, 1), 1.0) && test::approx(pt(0, 2), 0.0),
                    "pinv 3x2");
        ndarray<double> s{{1e-10, 0.0}, {0.0, 1.0}};
        auto ps = linalg::pinv(s, 1e-5);
        test::check(test::approx(ps(0, 0), 0.0) && test::approx(ps(1, 1), 1.0), "pinv rcond cutoff");
        auto pa = linalg::pinv(a);
        auto chk = linalg::matmul(linalg::matmul(a, pa), a);
        test::check(test::approx(chk(0, 0), 1.0) && test::approx(chk(1, 1), 4.0), "pinv idempotence a pinv a = a");
    }

    // --- cond ---------------------------------------------------------------

    {
        ndarray<int> i2{{1, 0}, {0, 1}};
        test::check(test::approx(linalg::cond(i2), 1.0), "cond identity");
        ndarray<int> d{{1, 0}, {0, 2}};
        test::check(test::approx(linalg::cond(d), 2.0), "cond diagonal");
        ndarray<int> m{{1, 2}, {3, 4}};
        test::check(test::approx(linalg::cond(m, linalg::NormOrd::One), 21.0), "cond p=1");
        test::check(test::approx(linalg::cond(m, linalg::NormOrd::Inf), 21.0), "cond p=inf");
        ndarray<int> s{{1, 2}, {2, 4}};
        test::check(std::isinf(linalg::cond(s)), "cond singular inf");
    }

    // --- eigh / eigvalsh ----------------------------------------------------

    {
        ndarray<int> a{{2, 1}, {1, 2}};
        auto e = linalg::eigh(a);
        test::check(test::approx(e.w(0), 1.0) && test::approx(e.w(1), 3.0), "eigh values ascending");
        for (int j = 0; j < 2; ++j)
        {
            double lhs = e.w(j) * e.v(0, j);
            double rhs = 2.0 * e.v(0, j) + e.v(1, j);
            test::check(test::approx(lhs, rhs), "eigh residual row 0");
        }
        auto vtv = linalg::matmul(e.v.transpose(), e.v);
        test::check(test::approx(vtv(0, 0), 1.0) && test::approx(vtv(0, 1), 0.0) && test::approx(vtv(1, 1), 1.0),
                    "eigh orthonormal");
        auto w = linalg::eigvalsh(a);
        test::check(test::approx(w(0), 1.0) && test::approx(w(1), 3.0), "eigvalsh values");
    }

    // --- tensordot ----------------------------------------------------------

    {
        ndarray<int> a{{1, 2}, {3, 4}};
        ndarray<int> b{{5, 6}, {7, 8}};
        // axes = 1: matrix product
        auto t1 = linalg::tensordot(a, b, 1);
        test::check(t1.ndim() == 2 && t1(0, 0) == 19 && t1(1, 1) == 50, "tensordot axes 1");
        // axes = 0: outer product, shape (2, 2, 2, 2)
        auto t0 = linalg::tensordot(a, b, 0);
        test::check(t0.ndim() == 4 && t0.get(std::vector<std::size_t>{0, 0, 0, 0}) == 5 &&
                        t0.get(std::vector<std::size_t>{1, 0, 1, 0}) == 21,
                    "tensordot axes 0");
        // axes = 2: double contraction -> scalar (a's last axes pair with
        // b's first axes in order: (0,0) and (1,1))
        auto t2 = linalg::tensordot(a, b, 2);
        test::check(t2.ndim() == 0 && t2.item() == 70, "tensordot axes 2");
        // explicit axis pair (contract middle axes of a 3-D with a 2-D)
        ndarray<int> c(std::vector<int>{2, 2, 2});
        c.data()[0] = 1;
        c.data()[1] = 2;
        c.data()[2] = 3;
        c.data()[3] = 4;
        c.data()[4] = 5;
        c.data()[5] = 6;
        c.data()[6] = 7;
        c.data()[7] = 8;
        auto t3 = linalg::tensordot(c, a, std::vector<int>{1, 2}, std::vector<int>{0, 1});
        test::check(t3.ndim() == 1 && t3(0) == 30 && t3(1) == 70, "tensordot axis sequences");
    }

    // --- cross --------------------------------------------------------------

    {
        ndarray<int> a{1, 2, 3};
        ndarray<int> b{4, 5, 6};
        auto c = linalg::cross(a, b);
        test::check(c.ndim() == 1 && c(0) == -3 && c(1) == 6 && c(2) == -3, "cross 1D");
        // axis along 0
        ndarray<int> m{{1, 4}, {2, 5}, {3, 6}};
        auto cm = linalg::cross(m, m, 0);
        test::check(cm.ndim() == 2 && cm(0, 0) == 0 && cm(1, 0) == 0 && cm(2, 0) == 0, "cross axis 0 self");
        // 1-D x 2-D broadcast
        ndarray<int> row{1, 0, 0};
        ndarray<int> mat{{0, 1, 0}, {0, 0, 1}};
        auto cb = linalg::cross(row, mat);
        test::check(cb.shape[0] == 2 && cb(0, 2) == 1 && cb(1, 1) == -1, "cross broadcast");
    }

    // --- multi_dot ----------------------------------------------------------

    {
        ndarray<int> a{{1, 0}, {0, 1}};
        ndarray<int> b{{2, 3}, {4, 5}};
        ndarray<int> c{{1, 1}, {1, 1}};
        auto md = linalg::multi_dot(std::vector<ndarray<int>>{a, b, c});
        test::check(md.shape[0] == 2 && md(0, 0) == 5 && md(1, 1) == 9, "multi_dot 3 matrices");
        auto md2 = linalg::multi_dot(std::vector<ndarray<int>>{a, b});
        test::check(md2(1, 0) == 4, "multi_dot 2 matrices");
        // 1-D ends follow dot semantics: (K,) (K, M) (M,)
        ndarray<int> v{1, 1};
        auto md3 = linalg::multi_dot(std::vector<ndarray<int>>{v, b, v});
        test::check(md3.ndim() == 0 && md3.item() == 14, "multi_dot 1D ends");
        // optimal ordering picks the cheap parenthesization: (AB)C vs A(BC)
        // 10x100, 100x5, 5x50 -> result (10, 50), value 10 at (0, 0)
        ndarray<int> big(std::vector<int>{10, 100});
        ndarray<int> mid(std::vector<int>{100, 5});
        ndarray<int> sml(std::vector<int>{5, 50});
        big.data()[0] = 1;
        mid.data()[0] = 1;
        sml.data()[0] = 1;
        auto mo = linalg::multi_dot(std::vector<ndarray<int>>{big, mid, sml});
        test::check(mo.shape[0] == 10 && mo.shape[1] == 50 && mo(0, 0) == 1, "multi_dot optimal order");
    }

    // --- lstsq --------------------------------------------------------------

    {
        // exact 2x2 system -> x = [2, 3], full rank, M == N so no residuals
        ndarray<int> a{{3, 1}, {1, 2}};
        ndarray<int> b{9, 8};
        auto r = linalg::lstsq(a, b);
        test::check(test::approx(r.x(0), 2.0) && test::approx(r.x(1), 3.0), "lstsq exact");
        test::check(r.rank == 2, "lstsq rank");
        test::check(r.residuals.size() == 0, "lstsq residuals empty M==N");
    }
    {
        // overdetermined consistent: y = 2x + 1 through (0, 1), (1, 3), (2, 5)
        ndarray<int> a{{0, 1}, {1, 1}, {2, 1}};
        ndarray<int> b{1, 3, 5};
        auto r = linalg::lstsq(a, b);
        test::check(test::approx(r.x(0), 2.0) && test::approx(r.x(1), 1.0), "lstsq overdetermined solution");
        test::check(r.rank == 2, "lstsq overdetermined rank");
        test::check(r.residuals.size() == 1 && test::approx(r.residuals(0), 0.0, 1e-10), "lstsq consistent residuals");
        // 2-D b: two systems sharing the same a
        ndarray<int> b2{{1, 3}, {3, 7}, {5, 11}};
        auto r2 = linalg::lstsq(a, b2);
        test::check(r2.x.ndim() == 2 && test::approx(r2.x(0, 1), 4.0) && test::approx(r2.x(1, 1), 3.0) &&
                        r2.residuals.size() == 2,
                    "lstsq 2D rhs");
    }
    {
        // rank-deficient: singular values below the cutoff are dropped
        ndarray<double> a{{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
        ndarray<double> b{1.0, 2.0, 3.0};
        auto r = linalg::lstsq(a, b);
        test::check(r.rank == 1, "lstsq rank deficient");
        test::check(r.residuals.size() == 0, "lstsq residuals empty rank");
        auto chk = linalg::matmul(a, r.x);
        test::check(test::approx(chk(0), 1.0) && test::approx(chk(1), 2.0) && test::approx(chk(2), 3.0),
                    "lstsq rank deficient fit");
    }

    // --- diagonal -----------------------------------------------------------

    {
        ndarray<int> a{{0, 1}, {2, 3}};
        auto d0 = linalg::diagonal(a);
        test::check(d0.ndim() == 1 && d0(0) == 0 && d0(1) == 3, "diagonal main");
        ndarray<int> m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        auto d1 = linalg::diagonal(m, 1);
        test::check(d1.ndim() == 1 && d1(0) == 2 && d1(1) == 6, "diagonal offset +1");
        auto dm1 = linalg::diagonal(m, -1);
        test::check(dm1(0) == 4 && dm1(1) == 8, "diagonal offset -1");
        auto d3 = linalg::diagonal(m, 3);
        test::check(d3.ndim() == 1 && d3.size() == 0, "diagonal offset out of range");
        auto dm3 = linalg::diagonal(m, -3);
        test::check(dm3.size() == 0, "diagonal offset -3 out of range");
        // 3-D stack: leading dims are preserved
        ndarray<int> s(std::vector<int>{2, 2, 2});
        for (std::size_t i = 0; i < 8; ++i)
        {
            s.data()[i] = static_cast<int>(i);
        }
        auto ds = linalg::diagonal(s);
        test::check(ds.shape[0] == 2 && ds.shape[1] == 2 && ds(0, 0) == 0 && ds(0, 1) == 3 && ds(1, 0) == 4 &&
                        ds(1, 1) == 7,
                    "diagonal 3D stack");
    }

    // --- matrix_transpose ---------------------------------------------------

    {
        ndarray<int> m{{1, 2, 3}, {4, 5, 6}};
        auto t = linalg::matrix_transpose(m);
        test::check(t.shape[0] == 3 && t.shape[1] == 2 && t(0, 1) == 4 && t(2, 0) == 3, "matrix_transpose 2D");
        ndarray<int> s(std::vector<int>{2, 2, 2});
        for (std::size_t i = 0; i < 8; ++i)
        {
            s.data()[i] = static_cast<int>(i);
        }
        auto ts = linalg::matrix_transpose(s);
        test::check(
            ts.shape[0] == 2 && ts.shape[1] == 2 && ts.get(std::vector<std::size_t>{0, 0, 0}) == 0 &&
                ts.get(std::vector<std::size_t>{0, 0, 1}) == 2 && ts.get(std::vector<std::size_t>{0, 1, 0}) == 1 &&
                ts.get(std::vector<std::size_t>{1, 0, 0}) == 4 && ts.get(std::vector<std::size_t>{1, 1, 1}) == 7,
            "matrix_transpose 3D stack");
    }

    // --- matrix_norm --------------------------------------------------------

    {
        ndarray<int> m{{1, 2}, {3, 4}};
        test::check(test::approx(linalg::matrix_norm(m), 5.477225575051661), "matrix_norm default fro");
        test::check(test::approx(linalg::matrix_norm(m, linalg::NormOrd::Fro), linalg::norm(m, linalg::NormOrd::Fro)),
                    "matrix_norm fro == norm fro");
        test::check(test::approx(linalg::matrix_norm(m, linalg::NormOrd::Nuc), 5.830951894845301), "matrix_norm nuc");
        test::check(test::approx(linalg::matrix_norm(m, linalg::NormOrd::Inf), linalg::norm(m, linalg::NormOrd::Inf)),
                    "matrix_norm inf");
        test::check(test::approx(linalg::matrix_norm(m, linalg::NormOrd::Two), linalg::norm(m, linalg::NormOrd::Two)),
                    "matrix_norm two");
        ndarray<int> v{1, 2};
        bool threw = false;
        try
        {
            (void)linalg::matrix_norm(v);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "matrix_norm rejects 1D");
    }

    // --- tensorinv ----------------------------------------------------------

    {
        // eye(24) reshaped to (4, 6, 8, 3): a(i, j, k, l) = 1 iff i*6+j == k*3+l
        ndarray<int> a(std::vector<int>{4, 6, 8, 3});
        for (std::size_t i = 0; i < 4; ++i)
        {
            for (std::size_t j = 0; j < 6; ++j)
            {
                for (std::size_t k = 0; k < 8; ++k)
                {
                    for (std::size_t l = 0; l < 3; ++l)
                    {
                        a.data()[((i * 6 + j) * 8 + k) * 3 + l] = i * 6 + j == k * 3 + l ? 1 : 0;
                    }
                }
            }
        }
        auto ai = linalg::tensorinv(a, 2);
        test::check(ai.shape[0] == 8 && ai.shape[1] == 3 && ai.shape[2] == 4 && ai.shape[3] == 6,
                    "tensorinv result shape");
        test::check(ai.get(std::vector<std::size_t>{0, 0, 0, 0}) == 1.0 &&
                        ai.get(std::vector<std::size_t>{0, 0, 1, 0}) == 0.0 &&
                        ai.get(std::vector<std::size_t>{0, 0, 0, 1}) == 0.0 &&
                        ai.get(std::vector<std::size_t>{0, 2, 0, 2}) == 1.0 &&
                        ai.get(std::vector<std::size_t>{2, 0, 1, 0}) == 1.0 &&
                        ai.get(std::vector<std::size_t>{2, 1, 2, 1}) == 0.0 &&
                        ai.get(std::vector<std::size_t>{2, 1, 3, 1}) == 0.0,
                    "tensorinv identity structure");
        // ind = 1 on (24, 8, 3): out(j, k, i) = 1 iff 3*j + k == i
        ndarray<int> a24(std::vector<int>{24, 8, 3});
        for (std::size_t f = 0; f < 24 * 8 * 3; ++f)
        {
            a24.data()[f] = f / 24 == f % 24 ? 1 : 0;
        }
        auto ai1 = linalg::tensorinv(a24, 1);
        test::check(ai1.shape[0] == 8 && ai1.shape[1] == 3 && ai1.shape[2] == 24, "tensorinv ind 1 shape");
        test::check(
            ai1.get(std::vector<std::size_t>{0, 0, 0}) == 1.0 && ai1.get(std::vector<std::size_t>{0, 0, 1}) == 0.0 &&
                ai1.get(std::vector<std::size_t>{0, 1, 1}) == 1.0 && ai1.get(std::vector<std::size_t>{1, 0, 1}) == 0.0,
            "tensorinv ind 1 structure");
        // errors: not square, singular, bad ind
        ndarray<int> ns{{1, 2, 3}, {4, 5, 6}};
        bool threw = false;
        try
        {
            (void)linalg::tensorinv(ns);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "tensorinv non-square");
        ndarray<int> z{{0, 0}, {0, 0}};
        threw = false;
        try
        {
            (void)linalg::tensorinv(z);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "tensorinv singular");
        threw = false;
        try
        {
            (void)linalg::tensorinv(z, 0);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "tensorinv ind out of range");
    }

    // --- tensorsolve --------------------------------------------------------

    {
        // a = eye(24) reshaped to (6, 4, 2, 3, 4), b = ones((6, 4)):
        // the system matrix is the identity, so x = ones((2, 3, 4))
        ndarray<int> a(std::vector<int>{6, 4, 2, 3, 4});
        for (std::size_t i = 0; i < 6; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                for (std::size_t k = 0; k < 2; ++k)
                {
                    for (std::size_t l = 0; l < 3; ++l)
                    {
                        for (std::size_t m = 0; m < 4; ++m)
                        {
                            a.data()[(((i * 4 + j) * 2 + k) * 3 + l) * 4 + m] =
                                i * 4 + j == (k * 3 + l) * 4 + m ? 1 : 0;
                        }
                    }
                }
            }
        }
        ndarray<int> b(std::vector<int>{6, 4});
        for (std::size_t i = 0; i < 24; ++i)
        {
            b.data()[i] = 1;
        }
        auto x = linalg::tensorsolve(a, b);
        test::check(x.shape[0] == 2 && x.shape[1] == 3 && x.shape[2] == 4 &&
                        x.get(std::vector<std::size_t>{0, 0, 0}) == 1.0 &&
                        x.get(std::vector<std::size_t>{1, 2, 3}) == 1.0,
                    "tensorsolve doc example");
        // axes = {0, 1} moves those dims last: (4, 6, 8, 3) -> (8, 3, 4, 6);
        // the reordered system matrix is again the identity, so x is b
        // reshaped to (4, 6)
        ndarray<int> a2(std::vector<int>{4, 6, 8, 3});
        for (std::size_t i = 0; i < 4; ++i)
        {
            for (std::size_t j = 0; j < 6; ++j)
            {
                for (std::size_t k = 0; k < 8; ++k)
                {
                    for (std::size_t l = 0; l < 3; ++l)
                    {
                        a2.data()[((i * 6 + j) * 8 + k) * 3 + l] = i * 6 + j == k * 3 + l ? 1 : 0;
                    }
                }
            }
        }
        ndarray<int> b2(std::vector<int>{8, 3});
        for (std::size_t i = 0; i < 24; ++i)
        {
            b2.data()[i] = static_cast<int>(i);
        }
        auto x2 = linalg::tensorsolve(a2, b2, std::vector<int>{0, 1});
        test::check(x2.shape[0] == 4 && x2.shape[1] == 6 && x2(0, 1) == 1 && x2(1, 2) == 8 && x2(2, 0) == 12 &&
                        x2(3, 3) == 21,
                    "tensorsolve with axes");
        // errors: not square, singular, shape mismatch
        ndarray<int> ns(std::vector<int>{4, 6, 8, 1});
        ndarray<int> bs(std::vector<int>{4, 6, 8});
        bool threw = false;
        try
        {
            (void)linalg::tensorsolve(ns, bs);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "tensorsolve not square");
        ndarray<int> z(std::vector<int>{4, 6, 8, 3});
        ndarray<int> bz(std::vector<int>{4, 6});
        threw = false;
        try
        {
            (void)linalg::tensorsolve(z, bz);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "tensorsolve singular");
        ndarray<int> m4(std::vector<int>{4, 6, 8, 3});
        ndarray<int> wb(std::vector<int>{6, 8});
        threw = false;
        try
        {
            (void)linalg::tensorsolve(m4, wb);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "tensorsolve leading mismatch");
    }

    // --- vecdot ------------------------------------------------------------

    {
        // doc example: projected size along a normal for an array of vectors
        ndarray<int> v{{0, 5, 0}, {0, 0, 10}, {0, 6, 8}};
        ndarray<double> n{0.0, 0.6, 0.8};
        auto p = linalg::vecdot(v, n);
        test::check(p.ndim() == 1 && test::approx(p(0), 3.0) && test::approx(p(1), 8.0) && test::approx(p(2), 10.0),
                    "vecdot doc example");
        // axis = 0 contracts the first axis
        ndarray<int> a{{1, 2, 3}, {4, 5, 6}};
        ndarray<int> one{1, 1};
        auto c0 = linalg::vecdot(a, one, 0);
        test::check(c0.ndim() == 1 && c0(0) == 5 && c0(1) == 7 && c0(2) == 9, "vecdot axis 0");
        // broadcasting of the remainder: (2, 3, 4) . (4,) -> (2, 3)
        ndarray<int> t(std::vector<int>{2, 3, 4});
        for (std::size_t i = 0; i < 24; ++i)
        {
            t.data()[i] = static_cast<int>(i);
        }
        ndarray<int> ones(std::vector<int>{4});
        for (std::size_t i = 0; i < 4; ++i)
        {
            ones.data()[i] = 1;
        }
        auto bc = linalg::vecdot(t, ones);
        test::check(bc.shape[0] == 2 && bc.shape[1] == 3 && bc(0, 0) == 6 && bc(1, 2) == 86, "vecdot broadcast");
        bool threw = false;
        try
        {
            (void)linalg::vecdot(a, one, -1);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "vecdot mismatched sizes");
        threw = false;
        try
        {
            ndarray<int> w(std::vector<int>{2, 3, 4});
            ndarray<int> u(std::vector<int>{2, 5});
            (void)linalg::vecdot(w, u);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "vecdot non-broadcastable");
    }

    // --- vector_norm -------------------------------------------------------

    {
        // doc example: arange(1..10).reshape(3, 3)
        ndarray<int> b(std::vector<int>{3, 3});
        for (std::size_t i = 0; i < 9; ++i)
        {
            b.data()[i] = static_cast<int>(i) + 1;
        }
        auto all = linalg::vector_norm(b);
        test::check(all.ndim() == 0 && test::approx(all.item(), 16.881943016134134), "vector_norm default");
        test::check(test::approx(linalg::vector_norm(b, {}, false, 0.0).item(), 9.0), "vector_norm ord 0");
        test::check(test::approx(linalg::vector_norm(b, {}, false, 1.0).item(), 45.0), "vector_norm ord 1");
        test::check(test::approx(linalg::vector_norm(b, {}, false, -1.0).item(), 0.3534857623790153),
                    "vector_norm ord -1");
        test::check(test::approx(linalg::vector_norm(b, {}, false, 2.0).item(), 16.881943016134134),
                    "vector_norm ord 2");
        test::check(test::approx(linalg::vector_norm(b, {}, false, -2.0).item(), 0.8058837395885292),
                    "vector_norm ord -2");
        test::check(
            test::approx(linalg::vector_norm(b, {}, false, std::numeric_limits<double>::infinity()).item(), 9.0),
            "vector_norm ord inf");
        test::check(
            test::approx(linalg::vector_norm(b, {}, false, -std::numeric_limits<double>::infinity()).item(), 1.0),
            "vector_norm ord -inf");
        // axis: column norms (axis 0) and row norms (axis -1)
        auto cols = linalg::vector_norm(b, std::vector<int>{0});
        test::check(cols.ndim() == 1 && test::approx(cols(0), 8.12403840463596) &&
                        test::approx(cols(1), 9.643650760992955) && test::approx(cols(2), 11.224972160321824),
                    "vector_norm axis 0");
        auto rows = linalg::vector_norm(b, std::vector<int>{-1});
        test::check(rows.ndim() == 1 && test::approx(rows(0), 3.7416573867739413) &&
                        test::approx(rows(2), 13.92838827718412),
                    "vector_norm axis -1");
        // two axes reduce the matrix slices; keepdims leaves 1s behind
        auto both = linalg::vector_norm(b, std::vector<int>{0, 1});
        test::check(both.ndim() == 0 && test::approx(both.item(), 16.881943016134134), "vector_norm two axes");
        auto kd = linalg::vector_norm(b, std::vector<int>{0}, true);
        test::check(kd.shape[0] == 1 && kd.shape[1] == 3 && test::approx(kd(0, 0), 8.12403840463596),
                    "vector_norm keepdims");
        auto kdall = linalg::vector_norm(b, {}, true);
        test::check(kdall.shape[0] == 1 && kdall.shape[1] == 1 && test::approx(kdall(0, 0), 16.881943016134134),
                    "vector_norm keepdims all axes");
        bool threw = false;
        try
        {
            (void)linalg::vector_norm(b, std::vector<int>{5});
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "vector_norm bad axis");
    }

    return test::failures() ? 1 : 0;
}
