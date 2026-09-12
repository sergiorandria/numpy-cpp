/**
 * @file test_creation.cpp
 * @brief Tests for array creation routines (np/creation.hpp).
 */
#include <cmath>

#include "np/np.hpp"
#include "test_util.hpp"

int main()
{
    // zeros / ones / full / empty
    {
        auto z = np::zeros<int>({2, 3});
        test::check(z.shape[0] == 2 && z.shape[1] == 3, "zeros shape");
        test::check(z.sum() == 0, "zeros values");
        auto o = np::ones<double>({4});
        test::check(o.sum() == 4.0, "ones values");
        auto f = np::full(std::vector<int>{2, 2}, 7);
        test::check(f(1, 1) == 7, "full values");
        auto e = np::empty<int>({3});
        test::check(e.size() == 3, "empty size");
    }

    // *_like
    {
        np::ndarray<int> a(std::vector<int>{2, 3});
        auto zl = np::zeros_like(a);
        test::check(zl.shape == a.shape && zl.type == a.type, "zeros_like");
        test::check(zl.sum() == 0, "zeros_like values");
        auto ol = np::ones_like(a);
        test::check(ol.sum() == 6, "ones_like values");
        auto el = np::empty_like(a);
        test::check(el.shape == a.shape, "empty_like");
        auto fl = np::full_like(a, 3);
        test::check(fl(1, 2) == 3, "full_like values");
    }

    // arange
    {
        auto a = np::arange(0, 5);
        test::check(a.size() == 5 && a(4) == 4, "arange stop");
        auto b = np::arange(0.0, 1.0, 0.25);
        test::check(b.size() == 4, "arange float count");
        test::check(test::approx(b(2), 0.5), "arange float value");
        auto c = np::arange(5, 0, -2);
        test::check(c.size() == 3 && c(0) == 5 && c(2) == 1, "arange negative step");
        auto d = np::arange(10);
        test::check(d.size() == 10, "arange single arg");
        auto e = np::arange(5, 5);
        test::check(e.size() == 0, "arange empty");
    }

    // linspace / logspace
    {
        auto l = np::linspace(0.0, 1.0, 5);
        test::check(l.size() == 5, "linspace size");
        test::check(test::approx(l(0), 0.0) && test::approx(l(4), 1.0), "linspace endpoints");
        test::check(test::approx(l(2), 0.5), "linspace midpoint");
        auto n = np::linspace(0, 10, 3);
        test::check(n(2) == 10.0, "linspace int->double");
        auto lg = np::logspace(0.0, 2.0, 3);
        test::check(test::approx(lg(0), 1.0) && test::approx(lg(2), 100.0), "logspace values");
    }

    // eye / identity
    {
        auto i = np::eye<int>(3);
        test::check(i.shape[0] == 3 && i.shape[1] == 3, "eye square");
        test::check(i(0, 0) == 1 && i(0, 1) == 0 && i(2, 2) == 1, "eye values");
        auto id = np::identity(2);
        test::check(id(1, 1) == 1.0, "identity default dtype");
        auto off = np::eye<int>(2, 3, 1);
        test::check(off(0, 1) == 1 && off(0, 0) == 0, "eye k offset");
    }

    // asarray
    {
        std::vector<int> v{1, 2, 3};
        auto a = np::asarray(v);
        test::check(a.ndim() == 1 && a.size() == 3 && a(2) == 3, "asarray vector");
        std::array<int, 2> arr{{5, 6}};
        auto b = np::asarray(arr);
        test::check(b.size() == 2 && b(1) == 6, "asarray std::array");
        auto c = np::asarray(v, {3});
        test::check(c.ndim() == 1 && c.size() == 3, "asarray with shape");
    }

    return test::failures() ? 1 : 0;
}
