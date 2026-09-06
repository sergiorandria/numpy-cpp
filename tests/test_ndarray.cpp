/**
 * @file test_ndarray.cpp
 * @brief Core tests for np::ndarray.
 */
#include <cstdint>
#include <sstream>

#include "np/np.hpp"
#include "test_util.hpp"

int main()
{
    // Construction
    {
        np::ndarray<int> a(std::vector<int>{2, 3});
        test::check(a.ndim() == 2, "ndim");
        test::check(a.size() == 6, "size");
        test::check(a.shape[0] == 2 && a.shape[1] == 3, "shape");
        test::check(a.type == np::dtype::int32, "default dtype");
    }

    // Fill + copy semantics (deep copy)
    {
        np::ndarray<double> a(std::vector<int>{3});
        a.fill(1.0);
        auto b = a; // copy
        b[0] = 99.0;
        test::check(a[0] == 1.0, "copy is deep");
        test::check(b[0] == 99.0, "copy modification");
    }

    // Nested initializer-list construction
    {
        np::ndarray<int> a{{1, 2, 3}, {4, 5, 6}};
        test::check(a.ndim() == 2 && a.shape[0] == 2 && a.shape[1] == 3, "nested init list shape");
        test::check(a(1, 2) == 6, "nested init list value");
        np::ndarray<int> b{1, 2, 3};
        test::check(b.ndim() == 1 && b.size() == 3, "flat init list");
    }

    // Reductions
    {
        np::ndarray<int> a(std::vector<int>{4});
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        test::check(a.sum() == 10, "sum");
        test::check(a.prod() == 24, "prod");
        test::check(a.min() == 1, "min");
        test::check(a.max() == 4, "max");
        test::check(a.mean() == 2.5, "mean promotes to double");
        test::check(a.var() == 1.25, "var");
        test::check(test::approx(a.std(), std::sqrt(1.25)), "std");
        test::check(a.all() == true, "all true values");
        test::check(a.any() == true, "any");
        test::check(a.argmax() == 3, "argmax");
        test::check(a.argmin() == 0, "argmin");

        auto c = np::ndarray<int>{{1, 2, 3}, {4, 5, 6}};
        test::check(c.sum(0).size() == 3, "sum(0) shape");
        test::check(c.sum(0)[0] == 5, "sum(0) value");
        test::check(c.sum(1)[1] == 15, "sum(1) value");
        test::check(c.max(0)[2] == 6, "max(0)");
        test::check(c.min(1)[0] == 1, "min(1)");
        test::check(c.mean(0)[0] == 2.5, "mean(0)");
        auto cm = c.cumsum(1);
        test::check(cm(0, 2) == 6 && cm(1, 2) == 15, "cumsum(1)");
        test::check(c.cumsum()[5] == 21, "cumsum flat");
    }

    // Elementwise arithmetic with broadcasting
    {
        np::ndarray<double> a(std::vector<int>{2, 3});
        a.fill(2.0);
        auto b = a * 3.0;
        test::check(b(1, 2) == 6.0, "scalar multiply");
        auto d = a + b;
        test::check(d(0, 0) == 8.0, "elementwise add");

        np::ndarray<double> row(std::vector<int>{3});
        row.fill(10.0);
        auto e = a + row; // broadcast (2,3) + (3,)
        test::check(e.shape[0] == 2 && e.shape[1] == 3, "broadcast shape");
        test::check(e(1, 2) == 12.0, "broadcast value");
        test::check((2.0 * a)(0, 1) == 4.0, "scalar-left multiply");
    }

    // Comparisons
    {
        np::ndarray<int> a{{1, 2}, {3, 4}};
        auto gt = a > 2;
        test::check(gt.type == np::dtype::bool_, "comparison dtype");
        test::check(gt(1, 0) == true && gt(0, 0) == false, "comparison values");
        test::check(a.all_equal(5) == false, "all_equal false");
        np::ndarray<int> b{{1, 2}, {3, 4}};
        test::check(a.all_equal(b) == true, "all_equal true");
    }

    // Views: transpose, swapaxes, squeeze, ravel, reshape
    {
        np::ndarray<int> a{{1, 2, 3}, {4, 5, 6}};
        auto t = a.transpose();
        test::check(t.shape[0] == 3 && t.shape[1] == 2, "transpose shape");
        test::check(t(2, 1) == 6, "transpose value");
        t(1, 0) = 99; // view: write through
        test::check(a(0, 1) == 99, "transpose writes through");

        auto r = a.reshape({6});
        test::check(r.ndim() == 1 && r.size() == 6, "reshape 1D");
        r(0) = 7;
        test::check(a(0, 0) == 7, "reshape writes through (contiguous)");

        auto s = np::ndarray<int>(std::vector<int>{1, 3, 1});
        s.fill(1);
        auto sq = s.squeeze();
        test::check(sq.shape[0] == 3 && sq.ndim() == 1, "squeeze");

        auto f = a.flatten();
        test::check(f.size() == 6 && f(1) == 99, "flatten copies values");
        f(0) = -1;
        test::check(a(0, 0) == 7, "flatten does not write through");

        auto sw = a.swapaxes(0, 1);
        test::check(sw(1, 0) == 99, "swapaxes value");
    }

    // Sorting
    {
        np::ndarray<int> a{3, 1, 2};
        auto s = a.sorted();
        test::check(s(0) == 1 && s(1) == 2 && s(2) == 3, "sort");
        auto o = a.argsort();
        test::check(o(0) == 1 && o(1) == 2 && o(2) == 0, "argsort");
        auto p = a.argpartition(1);
        test::check(p(1) == 2, "argpartition pivot at index 1");
        np::ndarray<int> sorted{1, 3, 5};
        test::check(sorted.searchsorted(4) == 2, "searchsorted value");
        test::check(sorted.searchsorted(np::ndarray<int>{0, 4, 9})[1] == 2, "searchsorted vector");
    }

    // take / put / repeat / clip / round
    {
        np::ndarray<int> a{10, 20, 30};
        auto t = a.take(std::vector<std::size_t>{2, 0});
        test::check(t(0) == 30 && t(1) == 10, "take");
        auto r = a.repeat(2);
        test::check(r.size() == 6 && r(2) == 20 && r(3) == 20, "repeat");
        auto c = a.clip(15, 25);
        test::check(c(0) == 15 && c(1) == 20 && c(2) == 25, "clip");
        np::ndarray<double> d{1.6, -2.4};
        auto rd = d.round();
        test::check(rd(0) == 2.0 && rd(1) == -2.0, "round");
        a.put(std::vector<std::size_t>{1}, std::vector<int>{55});
        test::check(a(1) == 55, "put");
    }

    // diagonal / trace / nonzero
    {
        np::ndarray<int> a{{1, 2}, {3, 4}};
        auto d = a.diagonal();
        test::check(d.size() == 2 && d(0) == 1 && d(1) == 4, "diagonal");
        test::check(a.trace() == 5, "trace");
        np::ndarray<int> z{{0, 1}, {0, 2}};
        auto nz = z.nonzero();
        test::check(nz.size() == 2 && nz[0](0) == 0 && nz[1](0) == 1 && nz[0](1) == 1 && nz[1](1) == 1, "nonzero");
    }

    // tolist / tobytes / tofile
    {
        np::ndarray<int> a{{1, 2}, {3, 4}};
        auto lst = a.tolist();
        test::check(lst.size() == 4 && lst[3] == 4, "tolist");
        auto bytes = a.tobytes();
        test::check(bytes.size() == 4 * 4, "tobytes size");
        std::stringstream ss;
        a.tofile(ss);
        test::check(ss.str().size() == 4 * 4, "tofile size");
    }

    // Printing
    {
        np::ndarray<int> a{{1, 2}, {3, 4}};
        std::ostringstream os;
        os << a;
        test::check(os.str().find("1") != std::string::npos, "operator<<");
    }

    // Iterator
    {
        np::ndarray<int> a{{1, 2}, {3, 4}};
        long total = 0;
        for (int v : a)
        {
            total += v;
        }
        test::check(total == 10, "range-based iteration");
    }

    // Bool arrays
    {
        np::ndarray<bool> a(std::vector<int>{3});
        a.fill(true);
        test::check(a.sum() == 3, "bool sum");
        test::check(a.all() == true, "bool all");
    }

    return test::failures() ? 1 : 0;
}
