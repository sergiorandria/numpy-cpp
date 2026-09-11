/**
 * @file test_emath.cpp
 * @brief Tests for np::emath (complex-promoting math).
 */
#include "test_util.hpp"
#include <np/np.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

int main()
{
    using namespace np;
    using C = std::complex<double>;

    // --- sqrt: in-domain, negative, zero, complex ---
    {
        auto x = ndarray<double>::from_data({4}, std::vector<double>{4.0, -1.0, 0.0, 2.25});
        auto r = emath::sqrt(x);
        test::check(test::approx_c(r.at(0), C(2.0, 0.0)), "sqrt(4)");
        test::check(test::approx_c(r.at(1), C(0.0, 1.0)), "sqrt(-1)");
        test::check(test::approx_c(r.at(2), C(0.0, 0.0)), "sqrt(0)");
        test::check(test::approx_c(r.at(3), C(1.5, 0.0)), "sqrt(2.25)");
    }
    {
        auto x = ndarray<C>::from_data({1}, std::vector<C>{C(-1.0, 0.0)});
        auto r = emath::sqrt(x);
        test::check(test::approx_c(r.at(0), C(0.0, 1.0)), "sqrt complex(-1)");
    }

    // --- log: positive, -1 -> i*pi, complex ---
    {
        auto x = ndarray<double>::from_data({2}, std::vector<double>{1.0, -1.0});
        auto r = emath::log(x);
        test::check(test::approx_c(r.at(0), C(0.0, 0.0)), "log(1)");
        test::check(test::approx_c(r.at(1), C(0.0, std::numbers::pi)), "log(-1)");
    }
    {
        auto x = ndarray<C>::from_data({1}, std::vector<C>{C(1.0, 0.0)});
        test::check(test::approx_c(emath::log(x).at(0), C(0.0, 0.0)), "log complex(1)");
    }

    // --- log2 / log10 / logn: single-pass values + complex ---
    {
        auto x = ndarray<double>::from_data({3}, std::vector<double>{8.0, -2.0, 1.0});
        auto r2 = emath::log2(x);
        test::check(test::approx_c(r2.at(0), C(3.0, 0.0)), "log2(8)");
        test::check(test::approx_c(r2.at(2), C(0.0, 0.0)), "log2(1)");
        test::check(test::approx_c(r2.at(1), std::log(C(-2.0, 0.0)) / std::log(2.0)), "log2(-2)");
        auto r10 = emath::log10(x);
        test::check(test::approx_c(r10.at(0), C(std::log10(8.0), 0.0)), "log10(8)");
        test::check(test::approx_c(r10.at(1), std::log10(C(-2.0, 0.0))), "log10(-2)");
        auto rn = emath::logn(8.0, x);
        test::check(test::approx_c(rn.at(0), C(1.0, 0.0)), "logn base 8 of 8");
        bool threw = false;
        try
        {
            (void)emath::logn(1.0, x);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "logn base 1 throws");
    }
    {
        auto x = ndarray<C>::from_data({1}, std::vector<C>{C(8.0, 0.0)});
        test::check(test::approx_c(emath::log2(x).at(0), C(3.0, 0.0)), "log2 complex(8)");
        test::check(test::approx_c(emath::log10(x).at(0), std::log10(C(8.0, 0.0))), "log10 complex(8)");
        test::check(test::approx_c(emath::logn(2.0, x).at(0), C(3.0, 0.0)), "logn complex");
    }

    // --- power: scalar exponent, negative base, broadcast ---
    {
        auto x = ndarray<double>::from_data({3}, std::vector<double>{4.0, -1.0, 9.0});
        auto r = emath::power(x, 0.5);
        test::check(test::approx_c(r.at(0), C(2.0, 0.0)), "power(4, 0.5)");
        test::check(test::approx_c(r.at(1), C(0.0, 1.0)), "power(-1, 0.5)");
        test::check(test::approx_c(r.at(2), C(3.0, 0.0)), "power(9, 0.5)");
    }
    {
        auto x = ndarray<double>::from_data({2}, std::vector<double>{2.0, 3.0});
        auto p = ndarray<double>::from_data({2}, std::vector<double>{3.0, 2.0});
        auto r = emath::power(x, p);
        test::check(test::approx_c(r.at(0), C(8.0, 0.0)), "power array [2^3]");
        test::check(test::approx_c(r.at(1), C(9.0, 0.0)), "power array [3^2]");
    }

    // --- arccos / arcsin / arctanh: in and out of domain + complex ---
    {
        auto x = ndarray<double>::from_data({3}, std::vector<double>{0.0, 2.0, -2.0});
        auto ac = emath::arccos(x);
        test::check(test::approx_c(ac.at(0), C(std::numbers::pi / 2, 0.0)), "arccos(0)");
        test::check(test::approx_c(ac.at(1), std::acos(C(2.0, 0.0))), "arccos(2)");
        auto as = emath::arcsin(x);
        test::check(test::approx_c(as.at(0), C(0.0, 0.0)), "arcsin(0)");
        test::check(test::approx_c(as.at(2), std::asin(C(-2.0, 0.0))), "arcsin(-2)");
        auto at = emath::arctanh(ndarray<double>::from_data({2}, std::vector<double>{0.0, 2.0}));
        test::check(test::approx_c(at.at(0), C(0.0, 0.0)), "arctanh(0)");
        test::check(test::approx_c(at.at(1), std::atanh(C(2.0, 0.0))), "arctanh(2)");
    }
    {
        auto x = ndarray<C>::from_data({1}, std::vector<C>{C(0.0, 0.0)});
        test::check(test::approx_c(emath::arccos(x).at(0), C(std::numbers::pi / 2, 0.0)), "arccos complex(0)");
        test::check(test::approx_c(emath::arcsin(x).at(0), C(0.0, 0.0)), "arcsin complex(0)");
        test::check(test::approx_c(emath::arctanh(x).at(0), C(0.0, 0.0)), "arctanh complex(0)");
    }

    // --- int input promotes through double ---
    {
        auto x = ndarray<int>::from_data({2}, std::vector<int>{9, -4});
        auto r = emath::sqrt(x);
        test::check(test::approx_c(r.at(0), C(3.0, 0.0)), "sqrt int 9");
        test::check(test::approx_c(r.at(1), C(0.0, 2.0)), "sqrt int -4");
    }

    // --- strided (non-contiguous) view takes the fallback path ---
    {
        auto m = ndarray<double>::from_data({2, 2}, std::vector<double>{1.0, -1.0, 4.0, -4.0});
        auto t = m.transpose(); // view, non-contiguous
        test::check(!t.is_contiguous(), "transpose is a strided view");
        auto r = emath::sqrt(t);
        // t = [[1, 4], [-1, -4]]
        test::check(test::approx_c(r.at(0, 0), C(1.0, 0.0)), "sqrt strided (0,0)");
        test::check(test::approx_c(r.at(0, 1), C(2.0, 0.0)), "sqrt strided (0,1)");
        test::check(test::approx_c(r.at(1, 0), C(0.0, 1.0)), "sqrt strided (1,0)");
        test::check(test::approx_c(r.at(1, 1), C(0.0, 2.0)), "sqrt strided (1,1)");
        auto rl = emath::log(t);
        test::check(test::approx_c(rl.at(0, 0), C(0.0, 0.0)), "log strided (0,0)");
        test::check(test::approx_c(rl.at(1, 0), C(0.0, std::numbers::pi)), "log strided (1,0)");
        auto rp = emath::power(t, 2.0);
        test::check(test::approx_c(rp.at(0, 1), C(16.0, 0.0)), "power strided scalar");
    }

    return test::failures() ? 1 : 0;
}
