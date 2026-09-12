/**
 * @file test_logic.cpp
 * @brief Tests for logic functions (logic.hpp).
 *
 * Verifies type checks, logical operations, and array comparisons.
 */
#include "test_util.hpp"
#include <np/np.hpp>

#include <cmath>
#include <limits>

int main()
{
    using namespace np;

    // --- Type checks ---
    {
        auto x = asarray(std::vector<double>{1.0, 2.0, INFINITY, -INFINITY, NAN});

        auto fin = isfinite(x);
        test::check(fin.at(0) == true, "1.0 is finite");
        test::check(fin.at(2) == false, "inf is not finite");
        test::check(fin.at(4) == false, "nan is not finite");

        auto inf = isinf(x);
        test::check(inf.at(0) == false, "1.0 is not inf");
        test::check(inf.at(2) == true, "inf is inf");
        test::check(inf.at(3) == true, "-inf is inf");

        auto nan = isnan(x);
        test::check(nan.at(0) == false, "1.0 is not nan");
        test::check(nan.at(4) == true, "nan is nan");

        auto posinf = isposinf(x);
        test::check(posinf.at(2) == true, "inf is positive inf");
        test::check(posinf.at(3) == false, "-inf is not positive inf");

        auto neginf = isneginf(x);
        test::check(neginf.at(2) == false, "inf is not negative inf");
        test::check(neginf.at(3) == true, "-inf is negative inf");
    }

    // --- Logical operations ---
    {
        auto a = asarray(std::vector<int>{0, 0, 1, 1});
        auto b = asarray(std::vector<int>{0, 1, 0, 1});

        auto land = logical_and(a, b);
        test::check(land.at(0) == false, "0 AND 0 = false");
        test::check(land.at(1) == false, "0 AND 1 = false");
        test::check(land.at(2) == false, "1 AND 0 = false");
        test::check(land.at(3) == true, "1 AND 1 = true");

        auto lor = logical_or(a, b);
        test::check(lor.at(0) == false, "0 OR 0 = false");
        test::check(lor.at(1) == true, "0 OR 1 = true");
        test::check(lor.at(2) == true, "1 OR 0 = true");
        test::check(lor.at(3) == true, "1 OR 1 = true");

        auto lxor = logical_xor(a, b);
        test::check(lxor.at(0) == false, "0 XOR 0 = false");
        test::check(lxor.at(1) == true, "0 XOR 1 = true");
        test::check(lxor.at(2) == true, "1 XOR 0 = true");
        test::check(lxor.at(3) == false, "1 XOR 1 = false");

        auto lnot = logical_not(a);
        test::check(lnot.at(0) == true, "NOT 0 = true");
        test::check(lnot.at(2) == false, "NOT 1 = false");
    }

    // --- Comparison functions ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3, 4, 5});
        auto b = asarray(std::vector<int>{1, 3, 2, 4, 6});

        auto eq = equal(a, b);
        test::check(eq.at(0) == true, "1 == 1");
        test::check(eq.at(1) == false, "2 != 3");

        auto neq = not_equal(a, b);
        test::check(neq.at(0) == false, "1 == 1");
        test::check(neq.at(1) == true, "2 != 3");

        auto gt = greater(a, b);
        test::check(gt.at(2) == true, "3 > 2");
        test::check(gt.at(1) == false, "2 < 3");

        auto lt = less(a, b);
        test::check(lt.at(1) == true, "2 < 3");
        test::check(lt.at(2) == false, "3 > 2");

        auto ge = greater_equal(a, b);
        test::check(ge.at(0) == true, "1 >= 1");
        test::check(ge.at(2) == true, "3 >= 2");

        auto le = less_equal(a, b);
        test::check(le.at(0) == true, "1 <= 1");
        test::check(le.at(1) == true, "2 <= 3");
    }

    // --- Array comparison ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3});
        auto b = asarray(std::vector<int>{1, 2, 3});
        auto c = asarray(std::vector<int>{1, 2, 4});

        test::check(array_equal(a, b), "equal arrays are equal");
        test::check(!array_equal(a, c), "different arrays not equal");
    }

    // --- isclose and allclose ---
    {
        auto a = asarray(std::vector<double>{1.0, 2.0, 3.0});
        auto b = asarray(std::vector<double>{1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9});

        auto close = isclose(a, b, 1e-5, 1e-8);
        test::check(close.at(0) == true, "1.0 is close to 1.0+1e-9");
        test::check(close.at(1) == true, "2.0 is close to 2.0+1e-9");

        test::check(allclose(a, b, 1e-5, 1e-8), "arrays are allclose");

        auto c = asarray(std::vector<double>{1.0, 2.0, 3.1});
        test::check(!allclose(a, c, 1e-5, 1e-8), "arrays not allclose");
    }

    // --- array_equiv with broadcasting ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3});
        auto b = asarray(std::vector<int>{1, 2, 3});
        test::check(array_equiv(a, b), "same arrays are equiv");

        // Broadcasting test
        auto scalar = full({1}, 5);
        auto vec = full({3}, 5);
        test::check(array_equiv(scalar, vec), "broadcast equiv");
    }

    return test::failures() ? 1 : 0;
}
