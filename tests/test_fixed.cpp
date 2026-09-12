/**
 * @file test_fixed.cpp
 * @brief Runtime tests for the fixed-shape np::ndarrayf<T, Extents...> path
 *        (construction, reductions, manipulation, elementwise operators,
 *        broadcasting, joins). Compile-time guarantees are exercised in
 *        test_compile_time.cpp / test_constexpr.cpp.
 */
#include <stdexcept>

#include "np/np.hpp"
#include "test_util.hpp"

int main()
{
    // Construction and access
    {
        np::ndarrayf<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        test::check(a.rank == 2, "rank");
        test::check(a.size_v == 6 && a.size() == 6, "size");
        test::check(a.static_shape[0] == 2 && a.static_shape[1] == 3, "static_shape");
        test::check(a(0, 1) == 2, "operator()");
        test::check(a[4] == 5, "flat operator[]");
        const auto &ca = a;
        test::check(ca(1, 2) == 6 && ca[0] == 1, "const access");
    }

    // Construction errors (ragged / wrong row count)
    {
        bool threw = false;
        try
        {
            np::ndarrayf<int, 2, 3> bad{{1, 2}, {3, 4, 5}};
            (void)bad;
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "ragged rows throw");

        threw = false;
        try
        {
            np::ndarrayf<int, 2, 3> bad{{1, 2, 3}};
            (void)bad;
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "wrong row count throws");

        np::ndarrayf<int, 2, 3> ok{{1, 2, 3}, {4, 5, 6}};
        test::check(ok(1, 0) == 4, "valid nested init");
        np::ndarrayf<int, 4> flat{1, 2, 3, 4};
        test::check(flat[3] == 4, "rank-1 init list");
        np::ndarray<double> scalar{2.5};
        test::check(scalar[0] == 2.5, "rank-0 init list");
    }

    // std::array construction
    {
        np::ndarrayf<int, 3> b{std::array<int, 3>{7, 8, 9}};
        test::check(b[1] == 8, "std::array ctor");
    }

    // fill
    {
        np::ndarrayf<int, 2, 2> f;
        f.fill(5);
        test::check(f(1, 1) == 5 && f(0, 0) == 5, "fill");
    }

    // Reductions
    {
        np::ndarrayf<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        test::check(a.sum() == 21, "sum all");
        test::check(a.prod() == 720, "prod all");
        test::check(a.min() == 1 && a.max() == 6, "min/max");
        test::check(a.argmin() == 0 && a.argmax() == 5, "argmin/argmax");
        test::check(a.mean() == 3.5, "mean all (double)");
        test::check(a.std() > 1.7 && a.std() < 1.8, "std all");

        auto rowsum = a.sum<1>();
        test::check(rowsum.rank == 1 && rowsum[0] == 6 && rowsum[1] == 15, "sum axis=1");
        auto colsum = a.sum<0>();
        test::check(colsum.rank == 1 && colsum[0] == 5 && colsum[2] == 9, "sum axis=0");
        auto rowmean = a.mean<1>();
        test::check(rowmean[0] == 2.0 && rowmean[1] == 5.0, "mean axis=1");
        auto colmin = a.min<0>();
        test::check(colmin[1] == 2 && colmin[0] == 1, "min axis=0");
        auto colmax = a.max<1>();
        test::check(colmax[0] == 3 && colmax[1] == 6, "max axis=1");

        np::ndarrayf<bool, 2, 2> tb{{true, true}, {true, false}};
        test::check(tb.all() == false && tb.any() == true, "bool all/any");
        test::check(tb.all<1>()[0] == true && tb.all<1>()[1] == false, "bool all axis");
        test::check(tb.sum() == true, "bool sum");
    }

    // Manipulation
    {
        np::ndarrayf<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        auto t = a.transpose();
        test::check(t.rank == 2 && t.static_shape[0] == 3 && t.static_shape[1] == 2, "transpose shape");
        test::check(t(0, 1) == 4 && t(1, 0) == 2, "transpose values");
        test::check(a(0, 1) == 2, "transpose does not mutate source");

        auto r = a.reshape<3, 2>();
        test::check(r.static_shape[0] == 3 && r(2, 1) == 6, "reshape values");
        auto r1 = a.reshape<1, 6>();
        test::check(r1(0, 5) == 6, "reshape to rank 2");

        auto f = a.flatten();
        test::check(f.rank == 1 && f.size_v == 6 && f[5] == 6, "flatten");

        np::ndarrayf<int, 1, 3, 1> sq_src{{1, 2, 3}};
        auto sq = sq_src.squeeze();
        test::check(sq.rank == 1 && sq.size_v == 3 && sq[2] == 3, "squeeze all");
        auto sq1 = sq_src.squeeze<0>();
        test::check(sq1.rank == 2 && sq1.static_shape[0] == 3, "squeeze axis 0");
        auto sq2 = sq_src.squeeze<2>();
        test::check(sq2.rank == 2 && sq2.static_shape[1] == 3, "squeeze axis 2");

        auto ex = a.expand_dims<0>();
        test::check(ex.rank == 3 && ex.static_shape[0] == 1 && ex(0, 1, 2) == 6, "expand_dims front");
        auto ex2 = a.expand_dims<2>();
        test::check(ex2.rank == 3 && ex2.static_shape[2] == 1 && ex2(1, 0, 0) == 4, "expand_dims back");
    }

    // Elementwise expressions
    {
        np::ndarrayf<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        auto e = a + a * 2;
        test::check(e(0, 0) == 3 && e(1, 2) == 18, "lazy chain");
        np::ndarrayf<int, 2, 3> conv = a + a;
        test::check(conv(0, 2) == 6, "implicit eval conversion");
        auto mat = e.eval();
        test::check(mat.rank == 2 && mat(1, 1) == 15, "explicit eval");

        np::ndarrayf<int, 2, 3> dm = a - 1;
        test::check(dm(1, 0) == 3, "scalar subtraction");
        np::ndarrayf<int, 2, 3> m2 = a * a;
        test::check(m2(1, 2) == 36, "elementwise multiply");
        np::ndarrayf<int, 2, 3> d = a / np::ndarrayf<int, 2, 3>{{1, 1, 1}, {2, 2, 2}};
        test::check(d(1, 0) == 2, "elementwise divide");
        np::ndarrayf<int, 2, 3> mm = a % 2;
        test::check(mm(0, 0) == 1 && mm(1, 0) == 0, "modulo");

        np::ndarrayf<bool, 2, 3> gt = a > 3;
        test::check(gt(0, 0) == false && gt(1, 0) == true, "comparison");
        np::ndarrayf<bool, 2, 3> le = a <= 3;
        test::check(le(1, 2) == false && le(0, 2) == true, "le comparison");

        auto neg = -a;
        test::check(neg(0, 0) == -1 && neg(1, 2) == -6, "unary minus");

        np::ndarrayf<int, 2, 3> ands = a & np::ndarrayf<int, 2, 3>{{1, 1, 1}, {0, 0, 0}};
        test::check(ands(1, 0) == 0 && ands(0, 0) == 1, "bitwise and");
        np::ndarrayf<int, 2, 3> shifts = a << 1;
        test::check(shifts(0, 0) == 2 && shifts(1, 2) == 12, "left shift");
    }

    // Broadcasting
    {
        np::ndarrayf<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        np::ndarrayf<int, 3> col{10, 20, 30};
        np::ndarrayf<int, 2, 1> row{{100}, {200}};
        auto bc = a + col + row;
        test::check(bc.rank == 2 && bc.static_shape[0] == 2 && bc.static_shape[1] == 3, "broadcast shape");
        test::check(bc(0, 0) == 111 && bc(1, 0) == 214 && bc(1, 2) == 236, "broadcast values");
        auto scalar = a + 1;
        test::check(scalar(0, 2) == 4, "scalar broadcast");
    }

    // Elementwise math functions
    {
        np::ndarrayf<double, 2> s{4.0, 9.0};
        auto sr = np::sqrt(s).eval();
        test::check(sr[0] == 2.0 && sr[1] == 3.0, "np::sqrt");
        np::ndarrayf<int, 2> ai{-3, 4};
        test::check(np::abs(ai).eval()[0] == 3, "np::abs");
        test::check(np::square(np::ndarrayf<int, 2>{2, 3}).eval()[1] == 9, "np::square");
        test::check(np::power(np::ndarrayf<int, 2>{2, 3}, 3).eval()[1] == 27, "np::power");
        auto se = np::exp(np::ndarrayf<double, 1>{0.0}).eval();
        test::check(se[0] == 1.0, "np::exp");
        auto sl = np::log(np::ndarrayf<double, 1>{1.0}).eval();
        test::check(sl[0] == 0.0, "np::log");
        auto ss = np::sin(np::ndarrayf<double, 1>{0.0}).eval();
        test::check(ss[0] == 0.0, "np::sin");
        auto sf = np::floor(np::ndarrayf<double, 1>{2.7}).eval();
        test::check(sf[0] == 2.0, "np::floor");
    }

    // concatenate / stack
    {
        np::ndarrayf<int, 3> c1{{0, 1, 2}};
        np::ndarrayf<int, 2> c2{{0, 1}};
        auto cat = np::concatenate(c1, c2);
        test::check(cat.rank == 1 && cat.size_v == 5 && cat[3] == 0 && cat[4] == 1, "concatenate");
        np::ndarrayf<int, 2, 2> m1{{1, 2}, {3, 4}};
        auto st = np::stack<0>(m1, m1);
        test::check(st.rank == 3 && st.static_shape[0] == 2 && st(1, 0, 1) == 2, "stack axis 0");
        auto st1 = np::stack<2>(m1, m1);
        test::check(st1.rank == 3 && st1.static_shape[2] == 2 && st1(0, 1, 1) == 2, "stack axis 2");
    }

    // Copy semantics
    {
        np::ndarrayf<int, 2, 2> a{{1, 2}, {3, 4}};
        auto b = a;
        b(0, 0) = 99;
        test::check(a(0, 0) == 1, "copy is deep");
        b = a;
        test::check(b(0, 0) == 1, "assignment");
    }

    // Fixed-shape creators
    {
        auto z = np::zeros<2, 3>();
        test::check(z(1, 2) == 0 && z.size_v == 6, "zeros");
        auto o = np::ones<int, 3>();
        test::check(o[0] == 1 && o[2] == 1, "ones");
        auto f = np::full<2, 2>(7);
        test::check(f(1, 1) == 7, "full");
        auto e = np::eye<3>();
        test::check(e(0, 0) == 1.0 && e(1, 2) == 0.0 && e(2, 2) == 1.0, "eye");
        auto e1 = np::eye<3, 4, 1, int>();
        test::check(e1(0, 1) == 1 && e1(0, 0) == 0 && e1(2, 3) == 1, "eye k=1");
        auto e2 = np::eye<3, 3, -1, int>();
        test::check(e2(1, 0) == 1 && e2(0, 0) == 0, "eye k=-1");
        auto i3 = np::identity<3, int>();
        test::check(i3(2, 2) == 1 && i3(1, 0) == 0, "identity");
        auto rng = np::arange<6, int>();
        test::check(rng[0] == 0 && rng[5] == 5, "arange");
        auto rng2 = np::arange<6>(1, 7, 2);
        test::check(rng2[0] == 1 && rng2[3] == 7 && rng2[5] == 11, "arange s/s");
        auto ls = np::linspace<5>(0.0, 1.0);
        test::check(ls[0] == 0.0 && ls[2] == 0.5 && ls[4] == 1.0, "linspace");
        auto ls1 = np::linspace<1>(5.0, 9.0);
        test::check(ls1[0] == 5.0, "linspace num=1");
    }

    // Fixed-shape linalg
    {
        np::ndarrayf<int, 3> u{1, 2, 3};
        np::ndarrayf<int, 3> v{4, 5, 6};
        test::check(np::linalg::dot(u, v) == 32, "dot 1D . 1D");

        np::ndarrayf<int, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
        auto mv = np::linalg::dot(m, v);
        test::check(mv.rank == 1 && mv[0] == 32 && mv[1] == 77, "dot 2D . 1D");
        np::ndarrayf<int, 3, 2> n{{1, 2}, {3, 4}, {5, 6}};
        auto vm = np::linalg::dot(v, n);
        test::check(vm.rank == 1 && vm[0] == 49 && vm[1] == 64, "dot 1D . 2D");

        auto mm = np::linalg::dot(m, n);
        test::check(mm.rank == 2 && mm(0, 0) == 22 && mm(1, 1) == 64, "dot 2D . 2D");
        auto mat = np::linalg::matmul(m, n);
        test::check(mat(0, 1) == 28 && mat(1, 0) == 49, "matmul");
        auto roundtrip = np::linalg::matmul(m, np::eye<3, 3, 0, int>());
        test::check(roundtrip.rank == 2 && roundtrip(1, 1) == 5, "matmul eye");
    }

    // Fixed-shape linalg: decompositions and inverses
    {
        np::ndarrayf<double, 2, 2> a{{1, 2}, {3, 4}};
        test::check(np::linalg::trace(a) == 5.0, "trace");
        test::check(test::approx(np::linalg::det(a), -2.0), "det");
        const auto sd = np::linalg::slogdet(a);
        test::check(sd.sign == -1.0 && test::approx(sd.logabsdet, std::log(2.0)), "slogdet");

        np::ndarrayf<double, 2, 2> singular{{1, 2}, {2, 4}};
        test::check(np::linalg::det(singular) == 0.0, "det singular is zero");
        const auto sds = np::linalg::slogdet(singular);
        test::check(sds.sign == 0.0 && sds.logabsdet == -std::numeric_limits<double>::infinity(), "slogdet singular");

        const auto iv = np::linalg::inv(a);
        test::check(test::approx(iv(0, 0), -2.0) && test::approx(iv(0, 1), 1.0) && test::approx(iv(1, 0), 1.5) &&
                        test::approx(iv(1, 1), -0.5),
                    "inv");
        bool threw = false;
        try
        {
            (void)np::linalg::inv(singular);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "inv singular throws");

        np::ndarrayf<double, 2> b{1, 0};
        const auto x = np::linalg::solve(a, b);
        test::check(test::approx(x[0], -2.0) && test::approx(x[1], 1.5), "solve 1-D rhs");
        np::ndarrayf<double, 2, 3> B{{1, 0, 2}, {0, 1, 3}};
        const auto X = np::linalg::solve(a, B);
        const auto rec = np::linalg::matmul(a, X);
        test::check(test::approx(rec(0, 0), 1.0) && test::approx(rec(1, 1), 1.0) && test::approx(rec(0, 2), 2.0) &&
                        test::approx(rec(1, 2), 3.0),
                    "solve 2-D rhs roundtrip");
        threw = false;
        try
        {
            (void)np::linalg::solve(singular, b);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "solve singular throws");

        np::ndarrayf<double, 2, 2> pd{{4, 0}, {0, 9}};
        const auto L = np::linalg::cholesky(pd);
        test::check(test::approx(L(0, 0), 2.0) && test::approx(L(1, 1), 3.0), "cholesky lower");
        const auto U = np::linalg::cholesky(pd, true);
        test::check(test::approx(U(0, 0), 2.0) && test::approx(U(1, 1), 3.0), "cholesky upper");
        np::ndarrayf<double, 2, 2> nonpd{{1, 2}, {2, 1}};
        threw = false;
        try
        {
            (void)np::linalg::cholesky(nonpd);
        }
        catch (const np::exceptions::LinAlgError &)
        {
            threw = true;
        }
        test::check(threw, "cholesky non-PD throws");

        np::ndarrayf<int, 2, 2> ai{{1, 2}, {3, 4}};
        test::check(test::approx(np::linalg::det(ai), -2.0), "det promotes int");
        const auto p2 = np::linalg::matrix_power(ai, 2);
        test::check(p2(0, 0) == 7.0 && p2(0, 1) == 10.0 && p2(1, 1) == 22.0, "matrix_power 2 (int promotes to double)");
        const auto p0 = np::linalg::matrix_power(ai, 0);
        test::check(p0(0, 0) == 1.0 && p0(1, 0) == 0.0, "matrix_power 0 is identity");
        const auto pm1 = np::linalg::matrix_power(ai, -1);
        const auto ivf = np::linalg::inv(ai);
        test::check(test::approx(pm1(0, 0), ivf(0, 0)) && test::approx(pm1(1, 1), ivf(1, 1)) &&
                        test::approx(pm1(0, 1), ivf(0, 1)),
                    "matrix_power -1 equals inv");
    }

    // Fixed-shape linalg: norms
    {
        np::ndarrayf<double, 4> v{1, 2, 3, 4};
        test::check(test::approx(np::linalg::norm(v), std::sqrt(30.0)), "norm default (2-norm)");
        test::check(test::approx(np::linalg::norm(v, np::linalg::NormOrd::Two), std::sqrt(30.0)), "norm ord=2");
        test::check(np::linalg::norm(v, np::linalg::NormOrd::One) == 10.0, "norm ord=1");
        test::check(np::linalg::norm(v, np::linalg::NormOrd::Inf) == 4.0, "norm ord=inf");
        test::check(np::linalg::norm(v, np::linalg::NormOrd::NegInf) == 1.0, "norm ord=-inf");
        test::check(test::approx(np::linalg::norm(v, np::linalg::NormOrd::NegOne), 0.48), "norm ord=-1");
        test::check(test::approx(np::linalg::norm(v, np::linalg::NormOrd::NegTwo), 12.0 / std::sqrt(205.0)),
                    "norm ord=-2");
        bool threw = false;
        try
        {
            (void)np::linalg::norm(v, np::linalg::NormOrd::Fro);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "norm fro on 1-D throws");

        np::ndarrayf<double, 2, 2> m{{1, 2}, {3, 4}};
        test::check(test::approx(np::linalg::norm(m), std::sqrt(30.0)), "matrix norm default (fro)");
        test::check(test::approx(np::linalg::norm(m, np::linalg::NormOrd::Fro), std::sqrt(30.0)), "matrix norm fro");
        test::check(np::linalg::norm(m, np::linalg::NormOrd::One) == 6.0, "matrix norm 1 (max column sum)");
        test::check(np::linalg::norm(m, np::linalg::NormOrd::NegOne) == 4.0, "matrix norm -1");
        test::check(np::linalg::norm(m, np::linalg::NormOrd::Inf) == 7.0, "matrix norm inf (max row sum)");
        test::check(np::linalg::norm(m, np::linalg::NormOrd::NegInf) == 3.0, "matrix norm -inf");
        test::check(test::approx(np::linalg::norm(m, np::linalg::NormOrd::Two), 5.464985704219043), "matrix norm 2");
        test::check(test::approx(np::linalg::norm(m, np::linalg::NormOrd::NegTwo), 0.3659661906262574),
                    "matrix norm -2");
        test::check(test::approx(np::linalg::norm(m, np::linalg::NormOrd::Nuc), 5.830951894845301), "matrix norm nuc");

        np::ndarrayf<double, 2, 3> wide{{1, 0, 0}, {0, 2, 0}};
        test::check(test::approx(np::linalg::norm(wide, np::linalg::NormOrd::Two), 2.0) &&
                        test::approx(np::linalg::norm(wide, np::linalg::NormOrd::NegTwo), 1.0),
                    "matrix norms on M<N input");
    }

    // Fixed-shape linalg: SVD, QR, rank, pinv, cond
    {
        np::ndarrayf<double, 3, 2> r1{{1, 2}, {0, 0}, {0, 0}};
        const auto s = np::linalg::svdvals(r1);
        test::check(s.rank == 1 && test::approx(s[0], std::sqrt(5.0)), "svdvals rank-1");
        np::ndarrayf<int, 2, 2> ai{{1, 2}, {3, 4}};
        const auto si = np::linalg::svdvals(ai);
        test::check(test::approx(si[0], 5.464985704219043) && test::approx(si[1], 0.3659661906262574),
                    "svdvals int promotes, descending");

        // Full SVD on M >= N and M < N: reconstruction and orthonormality.
        np::ndarrayf<double, 2, 3> wide{{1, 0, 0}, {0, 2, 0}};
        const auto sw = np::linalg::svd(wide);
        test::check(sw.u.rank == 2 && sw.vh.rank == 2 && sw.u.static_shape[0] == 2 && sw.u.static_shape[1] == 2 &&
                        sw.vh.static_shape[0] == 3 && sw.vh.static_shape[1] == 3,
                    "svd full shapes on M<N");
        test::check(test::approx(sw.s[0], 2.0) && test::approx(sw.s[1], 1.0), "svd singular values on M<N");
        double acc = 0;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                acc +=
                    std::pow(sw.u(i, 0) * sw.s[0] * sw.vh(0, j) + sw.u(i, 1) * sw.s[1] * sw.vh(1, j) - wide(i, j), 2);
            }
        }
        test::check(test::approx(acc, 0.0, 1e-12), "svd reconstruction M<N");
        const auto wr = np::linalg::svd<false>(wide);
        test::check(wr.vh.static_shape[0] == 2 && wr.vh.static_shape[1] == 3, "svd reduced vh shape (K, N)");
        test::check(test::approx(wr.vh(0, 0), 0.0) && test::approx(wr.vh(1, 1), 0.0) && test::approx(wr.vh(1, 0), 1.0),
                    "svd reduced vh values");

        np::ndarrayf<double, 3, 2> tall{{1, 0}, {0, 2}, {0, 0}};
        const auto st = np::linalg::svd<false>(tall);
        test::check(st.u.static_shape[0] == 3 && st.u.static_shape[1] == 2, "svd reduced u shape (M, K)");
        acc = 0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                acc +=
                    std::pow(st.u(i, 0) * st.s[0] * st.vh(0, j) + st.u(i, 1) * st.s[1] * st.vh(1, j) - tall(i, j), 2);
            }
        }
        test::check(test::approx(acc, 0.0, 1e-12), "svd reduced reconstruction M>N");

        np::ndarrayf<double, 2, 2> m{{1, 2}, {3, 4}};
        const auto sm = np::linalg::svd(m);
        test::check(test::approx(sm.s[0], 5.464985704219043) && test::approx(sm.s[1], 0.3659661906262574),
                    "svd 2x2 values");

        // QR: q orthonormal and q . r == a for both modes and both shapes.
        auto check_qr = [](const char *what, double err, bool ortho) {
            test::check(err < 1e-9, what);
            test::check(ortho, what, "q orthonormal");
        };
        {
            const auto q = np::linalg::qr(wide); // reduced: q (2,2), r (2,3)
            double err = 0;
            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    err += std::pow(q.q(i, 0) * q.r(0, j) + q.q(i, 1) * q.r(1, j) - wide(i, j), 2);
                }
            }
            bool ortho = test::approx(q.q(0, 0) * q.q(0, 0) + q.q(1, 0) * q.q(1, 0), 1.0) &&
                         test::approx(q.q(0, 1) * q.q(0, 1) + q.q(1, 1) * q.q(1, 1), 1.0);
            check_qr("qr reduced reconstruction M<N", err, ortho);
        }
        {
            const auto q = np::linalg::qr<false>(tall); // complete: q (3,3), r (3,2)
            double err = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    err +=
                        std::pow(q.q(i, 0) * q.r(0, j) + q.q(i, 1) * q.r(1, j) + q.q(i, 2) * q.r(2, j) - tall(i, j), 2);
                }
            }
            bool ortho = test::approx(q.q(0, 2), 0.0) && test::approx(q.q(2, 2), 1.0);
            check_qr("qr complete reconstruction M>N", err, ortho);
        }
        {
            const auto q = np::linalg::qr(m);
            double err = 0;
            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    err += std::pow(q.q(i, 0) * q.r(0, j) + q.q(i, 1) * q.r(1, j) - m(i, j), 2);
                }
            }
            bool ortho = test::approx(q.q(0, 0) * q.q(0, 0) + q.q(1, 0) * q.q(1, 0), 1.0);
            check_qr("qr reconstruction square", err, ortho);
        }

        test::check(np::linalg::matrix_rank(m) == 2, "matrix_rank 2x2");
        np::ndarrayf<double, 2, 2> rank1{{1, 2}, {2, 4}};
        test::check(np::linalg::matrix_rank(rank1) == 1, "matrix_rank rank-1");
        test::check(np::linalg::matrix_rank(rank1, 1e-9) == 1, "matrix_rank explicit tol");
        test::check(np::linalg::matrix_rank(rank1, 1e9) == 0, "matrix_rank huge tol");
        np::ndarrayf<double, 3> z{0, 0, 0};
        test::check(np::linalg::matrix_rank(z) == 0, "matrix_rank 1-D zero");
        np::ndarrayf<double, 3> nz{0, 5, 0};
        test::check(np::linalg::matrix_rank(nz) == 1, "matrix_rank 1-D");

        const auto pv = np::linalg::pinv(r1);
        test::check(test::approx(pv(0, 0), 0.2) && test::approx(pv(1, 0), 0.4) && test::approx(pv(0, 1), 0.0) &&
                        pv.rank == 2 && pv.static_shape[0] == 2 && pv.static_shape[1] == 3,
                    "pinv rank-1 M>N");
        const auto pp = np::linalg::pinv(wide); // 2x3, full row rank
        const auto id = np::linalg::matmul(wide, pp);
        test::check(test::approx(id(0, 0), 1.0) && test::approx(id(1, 1), 1.0) && test::approx(id(0, 1), 0.0),
                    "pinv right-inverse on M<N");

        test::check(np::linalg::cond(np::identity<2>()) == 1.0, "cond identity");
        test::check(test::approx(np::linalg::cond(m), 14.933034373659252), "cond 2-norm");
        test::check(test::approx(np::linalg::cond(m, np::linalg::NormOrd::Fro), 15.0, 1e-9), "cond fro");
        np::ndarrayf<double, 2, 2> cs{{1, 0}, {0, 0}};
        test::check(np::linalg::cond(cs) == std::numeric_limits<double>::infinity(), "cond singular is inf");
    }

    // Fixed-shape linalg: eigendecomposition, cross, outer, inner, lstsq
    {
        np::ndarrayf<double, 3, 3> s{{2, 0, 0}, {0, 3, 0}, {0, 0, 5}};
        const auto e = np::linalg::eigh(s);
        test::check(test::approx(e.w[0], 2.0) && test::approx(e.w[1], 3.0) && test::approx(e.w[2], 5.0),
                    "eigh ascending eigenvalues");
        test::check(test::approx(e.v(0, 0), 1.0) && test::approx(e.v(1, 1), 1.0) && test::approx(e.v(2, 2), 1.0),
                    "eigh diagonal eigenvectors");
        const auto ev = np::linalg::eigvalsh(s);
        test::check(ev.rank == 1 && test::approx(ev[0], 2.0) && test::approx(ev[2], 5.0), "eigvalsh");

        np::ndarrayf<double, 2, 2> sym{{1, 3}, {3, 4}};
        const auto e2 = np::linalg::eigh(sym);
        test::check(test::approx(e2.w[0], -0.8541019662496847) && test::approx(e2.w[1], 5.854101966249685),
                    "eigh nontrivial eigenvalues");
        double err = 0;
        for (int j = 0; j < 2; ++j)
        {
            for (int i = 0; i < 2; ++i)
            {
                err += std::pow(sym(i, 0) * e2.v(0, j) + sym(i, 1) * e2.v(1, j) - e2.w[j] * e2.v(i, j), 2);
            }
        }
        test::check(test::approx(err, 0.0, 1e-12), "eigh A v = w v");
        test::check(test::approx(e2.v(0, 0) * e2.v(0, 0) + e2.v(1, 0) * e2.v(1, 0), 1.0), "eigh eigenvectors unit");
        np::ndarrayf<int, 2, 2> asym{{1, 2}, {3, 4}};
        const auto e3 = np::linalg::eigh(asym);
        test::check(test::approx(e3.w[0], -0.8541019662496847) && test::approx(e3.w[1], 5.854101966249685),
                    "eigh reads lower triangle (numpy UPLO='L')");

        const auto c = np::linalg::cross(np::ndarrayf<double, 3>{1, 0, 0}, np::ndarrayf<double, 3>{0, 1, 0});
        test::check(test::approx(c[0], 0.0) && test::approx(c[1], 0.0) && test::approx(c[2], 1.0), "cross 3-vectors");
        np::ndarrayf<double, 2, 3> ca{{1, 0, 0}, {0, 1, 0}};
        np::ndarrayf<double, 2, 3> cb{{0, 1, 0}, {0, 0, 1}};
        const auto c2 = np::linalg::cross(ca, cb);
        test::check(test::approx(c2(0, 2), 1.0) && test::approx(c2(1, 0), 1.0), "cross rows");

        np::ndarrayf<double, 2> oa{1, 2};
        np::ndarrayf<double, 3> ob{3, 4, 5};
        const auto ot = np::linalg::outer(oa, ob);
        test::check(test::approx(ot(0, 0), 3.0) && test::approx(ot(1, 2), 10.0), "outer");
        test::check(np::linalg::inner(oa, np::ndarrayf<double, 2>{3, 4}) == 11.0, "inner 1-D");
        np::ndarrayf<double, 2, 3> ia{{1, 2, 3}, {4, 5, 6}};
        np::ndarrayf<double, 2, 3> ib{{1, 0, 0}, {0, 1, 0}};
        const auto inn = np::linalg::inner(ia, ib);
        test::check(test::approx(inn(0, 0), 1.0) && test::approx(inn(0, 1), 2.0) && test::approx(inn(1, 1), 5.0),
                    "inner 2-D contracts last axis");

        // lstsq: overdetermined (3x2) and underdetermined (2x3).
        np::ndarrayf<double, 3, 2> ov{{1, 0}, {0, 1}, {0, 0}};
        np::ndarrayf<double, 3> obv{1, 2, 0};
        const auto ls = np::linalg::lstsq(ov, obv);
        test::check(test::approx(ls.x[0], 1.0) && test::approx(ls.x[1], 2.0) && ls.rank == 2 &&
                        test::approx(ls.s[0], 1.0) && test::approx(ls.s[1], 1.0),
                    "lstsq overdetermined");
        np::ndarrayf<double, 2, 3> und{{1, 0, 0}, {0, 2, 0}};
        np::ndarrayf<double, 2> ubv{1, 0};
        const auto lu = np::linalg::lstsq(und, ubv);
        test::check(test::approx(lu.x[0], 1.0) && test::approx(lu.x[1], 0.0) && test::approx(lu.x[2], 0.0) &&
                        lu.rank == 2,
                    "lstsq underdetermined");
    }

    return test::failures() ? 1 : 0;
}
