/**
 * @file test_ndarray.cpp
 * @brief Core tests for np::ndarray.
 */
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
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

    // Negative indices (NumPy semantics)
    {
        np::ndarray<int> a{10, 20, 30};
        test::check(a(-1) == 30 && a.at(-1) == 30, "negative 1-D index");
        test::check(a[-1] == 30, "negative subscript");
        np::ndarray<int> m{{1, 2}, {3, 4}};
        test::check(m(-1, -1) == 4 && m.at(-2, 0) == 1, "negative 2-D index");
        test::check(m[-1][-1] == 4, "negative chained subscript");
        bool threw = false;
        try
        {
            a(-4);
        }
        catch (const std::out_of_range &)
        {
            threw = true;
        }
        test::check(threw, "negative index out of bounds throws");
    }

    // Ragged nested double lists are rejected
    {
        bool threw = false;
        try
        {
            np::ndarray<int> r{{1.0, 2.0}, {3.0}};
            (void)r;
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "ragged double init list throws");
    }

    // Span construction copies data
    {
        std::array<int, 4> raw{1, 2, 3, 4};
        np::ndarray<int> a(std::span<const int>(raw), std::vector<int>{2, 2});
        test::check(a(1, 1) == 4 && a.size() == 4, "span ctor");
        bool threw = false;
        try
        {
            np::ndarray<int> b(std::span<const int>(raw), std::vector<int>{3});
            (void)b;
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "span ctor size mismatch throws");
    }

    // In-place ops write through views and keep shape
    {
        np::ndarray<int> a{{1, 2}, {3, 4}};
        auto t = a.transpose();
        t += 10;
        test::check(a(0, 1) == 12 && t.shape[0] == 2, "in-place writes through view");
        np::ndarray<int> b{{1, 2}, {3, 4}};
        np::ndarray<double> d{{0.5, 0.5}, {0.5, 0.5}};
        b += d; // heterogeneous: converts
        test::check(b(0, 0) == 1 && b(1, 1) == 4, "heterogeneous in-place add");
        test::check((2.0 == b * 1.0)(0, 1) == true, "scalar-left comparison");
        test::check((10 > b)(0, 0) == true, "scalar-left greater");
    }

    // True division promotes integral pairs to double (NumPy semantics)
    {
        np::ndarray<int> a{1, 2, 3, 4};
        auto q = a / 2;
        test::check(std::abs(q(0) - 0.5) < 1e-12 && std::abs(q(3) - 2.0) < 1e-12, "int/scalar promotes");
        auto r = a / a;
        test::check(std::abs(r(2) - 1.0) < 1e-12, "int/int promotes");
        auto f = a.floordiv(2);
        test::check(f(0) == 0 && f(3) == 2, "floordiv still floors");
    }

    // floored mod/div edge cases
    {
        test::check(np::detail::floored_mod(-4, 3) == 2, "floored mod negative");
        test::check(np::detail::floored_div(-4, 3) == -2, "floored div negative");
        test::check(np::detail::floored_mod(-4, 3u) == 2u, "mixed-sign mod");
        bool threw = false;
        try
        {
            (void)np::detail::floored_div(1, 0);
        }
        catch (const std::domain_error &)
        {
            threw = true;
        }
        test::check(threw, "integer divide by zero throws");
        // Negative int exponents truncate (pinned repo contract, see test_math).
        test::check(np::detail::power_elem(2, -1) == 0, "negative int power truncates");
        test::check(np::detail::power_elem(2, 10) == 1024, "binary power");
    }

    // NaN propagation + complex guards
    {
        np::ndarray<double> a{1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
        test::check(std::isnan(a.max()) && std::isnan(a.min()), "NaN propagates in min/max");
        test::check(a.argmax() == 1, "NaN wins argmax");
        np::ndarray<std::complex<double>> c(std::vector<int>{2});
        c.fill({1.0, 2.0});
        bool threw = false;
        try
        {
            (void)c.min();
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "complex min throws");
        test::check(std::abs(c.mean().real() - 1.0) < 1e-12, "complex mean");
        test::check(std::abs(c.var() - 0.0) < 1e-12, "complex var is real zero");
        np::ndarray<std::complex<double>> d{{3.0, 1.0}, {1.0, 5.0}};
        test::check(d.argsort(1)(0, 0) == 1, "complex argsort by real part");
    }

    // Empty-slice reductions
    {
        np::ndarray<int> e(std::vector<int>{2, 0});
        bool threw = false;
        try
        {
            (void)e.min(1);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "min of empty slice throws");
        threw = false;
        try
        {
            (void)e.argmax(1);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "argmax of empty slice throws");
        auto s = e.sum(1); // seeded reductions yield identity
        test::check(s.size() == 2 && s(0) == 0, "sum of empty slice is zero");
        auto v = e.var(1);
        test::check(std::isnan(v(0)), "var of empty slice is NaN");
        bool cum_ok = true;
        try
        {
            auto cs = e.cumsum(1);
            cum_ok = cs.size() == 0;
        }
        catch (...)
        {
            cum_ok = false;
        }
        test::check(cum_ok, "cumsum of empty axis returns empty");
    }

    // diagonal(-1), abs(complex), round half-even
    {
        np::ndarray<int> a{{1, 2}, {3, 4}};
        auto d = a.diagonal(-1);
        test::check(d.size() == 1 && d(0) == 3, "diagonal(-1)");
        test::check(a.trace(-1) == 3, "trace(-1)");
        np::ndarray<std::complex<double>> c(std::vector<int>{2});
        c.fill({3.0, 4.0});
        auto m = c.abs();
        test::check(std::abs(m(0) - 5.0) < 1e-12, "complex abs magnitude");
        np::ndarray<double> r{2.5, 3.5, -2.5};
        auto rd = r.round();
        test::check(rd(0) == 2.0 && rd(1) == 4.0 && rd(2) == -2.0, "banker's rounding");
    }

    // Bool iteration + const access + sorting
    {
        np::ndarray<bool> a{true, false, true};
        long n = 0;
        for (bool v : a)
            n += v ? 1 : 0;
        test::check(n == 2, "range-for over bool array");
        test::check(a.tolist().size() == 3, "bool tolist");
        const auto &ca = a;
        test::check(ca(0) == true && ca.at(2) == true, "const bool access");
        np::ndarray<bool> b{true, false, true, false, false};
        b.sort();
        test::check(b(0) == false && b(4) == true, "bool sort");
        test::check(b.argsort().size() == 5, "bool argsort runs");
        auto by = a.tobytes();
        test::check(by.size() == 3 && by[0] == 1 && by[1] == 0, "bool tobytes is 1 byte/elem");
    }

    // take/sorted/argsort None-flatten + templated searchsorted
    {
        np::ndarray<int> a{{3, 1}, {2, 0}};
        auto t = a.take(std::vector<std::size_t>{0, 3}, std::nullopt);
        test::check(t.size() == 2 && t(0) == 3 && t(1) == 0, "take(None) flattens");
        auto s = a.sorted(std::nullopt);
        test::check(s.size() == 4 && s(0) == 0 && s(3) == 3, "sorted(None) flattens");
        auto o = a.argsort(std::nullopt);
        test::check(o.size() == 4 && o(0) == 3, "argsort(None) flattens");
        np::ndarray<double> d{1.0, 3.0, 5.0};
        np::ndarray<double> needles{0.5, 4.0};
        auto idx = d.searchsorted(needles);
        test::check(idx(0) == 0 && idx(1) == 2, "searchsorted templated needles");
        auto st = a.argsort(1); // stable ties keep input order
        (void)st;
        np::ndarray<int> ties{2, 1, 2};
        auto so = ties.argsort();
        test::check(so(0) == 1 && so(1) == 0 && so(2) == 2, "argsort stable ties");
    }

    // Validation errors
    {
        np::ndarray<int> a{1, 2, 3};
        bool threw = false;
        try
        {
            a.put(std::vector<std::size_t>{0}, std::vector<int>{9}); // ok
            np::ndarray<int> e(std::vector<int>{0});
            e.put(std::vector<std::size_t>{0}, std::vector<int>{9});
        }
        catch (const std::out_of_range &)
        {
            threw = true;
        }
        test::check(threw, "put into empty array throws");
        threw = false;
        try
        {
            a.resize(std::vector<int>{-1, 2});
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "resize negative dim throws");
        threw = false;
        try
        {
            np::ndarray<int> e;
            (void)e(0);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        test::check(threw, "access on empty array throws");
        threw = false;
        try
        {
            np::ndarray<int> v{1, 2};
            (void)(v << -1);
        }
        catch (const std::out_of_range &)
        {
            threw = true;
        }
        test::check(threw, "negative shift throws");
        threw = false;
        try
        {
            np::ndarray<int> v{{1, 2}, {3}};
            (void)v;
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "ragged init list still throws");
    }

    // tofile stream failure is reported
    {
        np::ndarray<int> a{1, 2, 3};
        std::ostringstream os;
        os.setstate(std::ios::badbit);
        bool threw = false;
        try
        {
            a.tofile(os);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        test::check(threw, "tofile on bad stream throws");
    }

    return test::failures() ? 1 : 0;
}
