/**
 * @file test_proxy.cpp
 * @brief Tests for proxy-based indexing (np/detail/proxy.hpp).
 */
#include <cstdint>

#include "np/np.hpp"
#include "test_util.hpp"

int main()
{
    // 1D proxy read/write
    {
        np::ndarray<int> a(std::vector<int>{4});
        a.fill(0);
        a[0] = 10;
        a[1] = 20;
        test::check(a[0] == 10, "1D proxy read a[0]");
        test::check(a[1] == 20, "1D proxy read a[1]");
        a[2] = a[0] + a[1];
        test::check(a[2] == 30, "1D proxy arithmetic");
        const auto &ca = a;
        test::check(ca[3] == 0, "const 1D proxy");
    }

    // 2D proxy chained indexing
    {
        np::ndarray<int> a(std::vector<int>{2, 3});
        a.fill(0);
        a[1][2] = 42;
        test::check(a[1][2] == 42, "2D chained proxy write");
        test::check(a(1, 2) == 42, "2D operator() write");
        test::check(a(0, 0) == 0, "2D operator() read");
        const auto &ca = a;
        test::check(ca[1][2] == 42, "2D const chained proxy");
    }

    // 3D proxy
    {
        np::ndarray<int> a(std::vector<int>{2, 2, 2});
        a.fill(7);
        a[1][0][1] = -1;
        test::check(a[1][0][1] == -1, "3D chained proxy");
        const auto &ca = a;
        test::check(ca[1][0][1] == -1, "3D const proxy");
    }

    // Proxy to scalar conversion / comparison
    {
        np::ndarray<double> a(std::vector<int>{3});
        a[0] = 1.5;
        double v = a[0];
        test::check(v == 1.5, "proxy -> T conversion");
        test::check(a[0] == 1.5, "proxy == scalar");
        test::check(a[0] != 2.5, "proxy != scalar");
    }

    // get/set/at
    {
        np::ndarray<int> a(std::vector<int>{2, 3});
        a.fill(0);
        a.set(std::array<std::size_t, 2>{1, 1}, 5);
        test::check(a.get(std::array<std::size_t, 2>{1, 1}) == 5, "get/set");
        test::check(a.at(1, 1) == 5, "at(i, j)");
    }

    return test::failures() ? 1 : 0;
}
