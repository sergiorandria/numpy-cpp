/**
 * @file test_masked_array.cpp
 * @brief Tests for masked_array.hpp — the subsystem previously had zero
 * behavioral coverage, which is how scalar-total count(axis), unmasked dot,
 * and element-0 hard-mask checks shipped unnoticed.
 */
#include "test_util.hpp"
#include <np/masked_array.hpp>
#include <np/np.hpp>

int main()
{
    using np::ma::count;
    using np::ma::count_axis;
    using np::ma::dot;
    using np::ma::MaskedArray;
    using np::ma::put;

    // count() scalar total; count_axis() per-axis array.
    {
        np::ndarray<double> d{{1.0, 2.0}, {3.0, 4.0}};
        np::ndarray<bool> m{{false, true}, {false, false}};
        MaskedArray<double> a(d, m);
        test::check(count(a) == 3, "masked count total");
        auto c0 = count_axis(a, 0);
        test::check(c0.size() == 2 && c0.data()[0] == 2 && c0.data()[1] == 1, "count_axis 0");
        auto c1 = count_axis(a, 1);
        test::check(c1.size() == 2 && c1.data()[0] == 1 && c1.data()[1] == 2, "count_axis 1");
        // Explicit axis on scalar count() must throw (NumPy returns an
        // array; the old code silently returned the scalar total).
        bool threw = false;
        try
        {
            (void)count(a, 0);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "count(axis) throws, use count_axis");
    }

    // dot(): masked entries contribute 0, contaminated outputs are masked.
    {
        np::ndarray<double> ad{{1.0, 2.0}, {3.0, 4.0}};
        np::ndarray<bool> am{{false, true}, {false, false}}; // a[0,1] masked
        np::ndarray<double> bd{{5.0, 6.0}, {7.0, 8.0}};
        np::ndarray<bool> bm(std::vector<int>{2, 2});
        bm.fill(false);
        MaskedArray<double> a(ad, am), b(bd, bm);
        MaskedArray<double> c = dot(a, b);
        // Row 0: [1*5+0*7, 1*6+0*8] = [5, 6], both contaminated by a[0,1].
        // Row 1: [3*5+4*7, 3*6+4*8] = [43, 50], clean.
        test::check(std::abs(c.data.data()[0] - 5.0) < 1e-9, "masked dot value 00");
        test::check(std::abs(c.data.data()[3] - 50.0) < 1e-9, "masked dot value 11");
        test::check(c.mask.data()[0] == true && c.mask.data()[1] == true, "masked dot row 0 masked");
        test::check(c.mask.data()[2] == false && c.mask.data()[3] == false, "masked dot row 1 clean");
    }

    // put(): hard mask guards the destination element, not element 0.
    {
        np::ndarray<double> d{1.0, 2.0, 3.0};
        np::ndarray<bool> m{false, true, false};
        MaskedArray<double> a(d, m);
        a.hard_mask = true;
        np::ndarray<std::size_t> idx{2};
        np::ndarray<double> vals{9.0};
        put(a, idx, vals); // dst=2 unmasked -> must write
        test::check(a.data.data()[2] == 9.0, "put writes unmasked dst under hard mask");
        np::ndarray<std::size_t> idx2{1};
        put(a, idx2, vals); // dst=1 masked -> must skip
        test::check(a.data.data()[1] == 2.0, "put skips masked dst under hard mask");
    }

    return test::failures() ? 1 : 0;
}
