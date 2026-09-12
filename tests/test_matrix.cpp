/**
 * @file test_matrix.cpp
 * @brief Tests for np::Matrix and det/inverse/solve.
 */
#include <cmath>

#include "np/np.hpp"
#include "test_util.hpp"

using namespace np;

int main()
{
    // Construction and access
    {
        Matrix<double> m(2, 3);
        test::check(m.rows() == 2 && m.cols() == 3, "rows/cols");
        m(0, 1) = 5.0;
        test::check(m(0, 1) == 5.0, "(i, j) access");
        const auto &cm = m;
        test::check(cm(0, 1) == 5.0, "const (i, j) access");
        test::check(m.is_square() == false, "is_square false");
    }

    // Nested initializer lists
    {
        Matrix<int> m{{1, 2}, {3, 4}};
        test::check(m.rows() == 2 && m.cols() == 2, "init list shape");
        test::check(m(1, 0) == 3, "init list value");
    }

    // Factories
    {
        auto id = Matrix<int>::identity(3);
        test::check(id(0, 0) == 1 && id(0, 1) == 0 && id(2, 2) == 1, "identity");
        auto z = Matrix<double>::zeros(2, 2);
        test::check(z.sum() == 0.0, "zeros");
        auto o = Matrix<int>::ones(2, 1);
        test::check(o(1, 0) == 1, "ones");
        auto e = Matrix<int>::eye(2, 3, 1);
        test::check(e(0, 1) == 1, "eye offset");
    }

    // transpose
    {
        Matrix<int> m{{1, 2, 3}, {4, 5, 6}};
        auto t = m.transpose();
        test::check(t.rows() == 3 && t.cols() == 2, "transpose shape");
        test::check(t(2, 1) == 6, "transpose value");
    }

    // Matrix product and scalar multiply
    {
        Matrix<int> a{{1, 2}, {3, 4}};
        Matrix<int> b{{5, 6}, {7, 8}};
        auto p = a * b;
        test::check(p(0, 0) == 19 && p(1, 1) == 50, "matrix product");
        auto s = a * 2;
        test::check(s(1, 0) == 6, "scalar multiply");
        auto sl = 3 * a;
        test::check(sl(0, 1) == 6, "scalar-left multiply");
        Matrix<int> i{{1, 0}, {0, 1}};
        auto ip = a * i;
        test::check(ip(1, 1) == 4, "identity product");
    }

    // det
    {
        Matrix<double> m{{2.0, 0.0}, {0.0, 3.0}};
        test::check(test::approx(det(m), 6.0), "det diagonal");
        Matrix<double> n{{1.0, 2.0}, {3.0, 4.0}};
        test::check(test::approx(det(n), -2.0), "det 2x2");
        Matrix<int> s{{1, 2}, {2, 4}};
        test::check(test::approx(det(s), 0.0), "det singular");
        Matrix<double> g{{2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
        test::check(test::approx(det(g), 4.0), "det 3x3");
    }

    // inverse
    {
        Matrix<double> m{{2.0, 0.0}, {0.0, 4.0}};
        auto inv = inverse(m);
        test::check(test::approx(inv(0, 0), 0.5), "inverse diag");
        test::check(test::approx(inv(1, 1), 0.25), "inverse diag 2");
        auto prod = m * inv;
        test::check(test::approx(prod(0, 0), 1.0), "m * m^-1 = I");
        Matrix<double> n{{1.0, 2.0}, {3.0, 4.0}};
        auto p2 = n * inverse(n);
        test::check(test::approx(p2(0, 0), 1.0) && test::approx(p2(1, 1), 1.0), "n * n^-1 = I");
    }

    // solve
    {
        Matrix<double> a{{2.0, 0.0}, {0.0, 4.0}};
        ndarray<double> b{4.0, 8.0};
        auto x = solve(a, b);
        test::check(x.size() == 2, "solve shape");
        test::check(test::approx(x(0), 2.0) && test::approx(x(1), 2.0), "solve values");
    }

    // Matrix inherits ndarray API
    {
        Matrix<int> m{{1, 2}, {3, 4}};
        test::check(m.sum() == 10, "inherited sum");
        test::check(m.mean() == 2.5, "inherited mean");
        auto flat = m.flatten();
        test::check(flat.size() == 4, "inherited flatten");
        test::check(m.ndim() == 2, "inherited ndim");
    }

    return test::failures() ? 1 : 0;
}
