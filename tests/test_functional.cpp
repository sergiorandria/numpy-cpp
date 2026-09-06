/**
 * @file test_functional.cpp
 * @brief Tests for functional.hpp (apply_along_axis, vectorize, piecewise)
 */
#include "test_util.hpp"
#include <cmath>
#include <np/functional.hpp>
#include <np/np.hpp>

int main()
{
    using namespace np;
    // apply_along_axis
    {
        auto a = ndarray<int>::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
        // sum along axis 1 -> [6,15] (may be shape [2,1] or [2] depending on scalar detection)
        auto res = apply_along_axis(
            [](const ndarray<int> &s) {
                auto sum = 0;
                for (auto v : s.data())
                    sum += v;
                ndarray<int> o(std::vector<int>{1});
                o(0) = sum;
                return o;
            },
            1, a);
        test::check(res.size() == 2 && res.data()[0] == 6 && res.data()[1] == 15, "apply_along_axis axis1");
        auto res0 = apply_along_axis(
            [](const ndarray<int> &s) {
                ndarray<int> o(std::vector<int>{1});
                o(0) = (int)s.size();
                return o;
            },
            0, a);
        test::check(res0.size() == 3, "apply_along_axis axis0");
        // negative axis
        auto resNeg = apply_along_axis(
            [](const ndarray<int> &s) {
                ndarray<int> o(std::vector<int>{1});
                o(0) = s.data()[0];
                return o;
            },
            -1, a);
        test::check(resNeg.size() == 2, "apply_along_axis neg axis");
    }
    // apply_over_axes
    {
        auto a = ndarray<int>::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
        auto r = apply_over_axes([](const ndarray<int> &x, int ax) { return x; }, a, {0});
        test::check(r.shape == a.shape, "apply_over_axes");
    }
    // vectorize
    {
        auto v = make_vectorize([](int x) { return x * x; });
        auto a = ndarray<int>::from_data({3}, {1, 2, 3});
        auto b = v(a);
        test::check(b.at(2) == 9, "vectorize 1 arg");
        auto v2 = make_vectorize([](int x, int y) { return x + y; });
        auto c = ndarray<int>::from_data({3}, {10, 20, 30});
        auto d = v2(a, c);
        test::check(d.at(1) == 22, "vectorize 2 args broadcast");
        auto sc = v(5);
        test::check(sc.at(0) == 25, "vectorize scalar");
        auto f = frompyfunc([](int x) { return x + 1; }, 1, 1);
        auto e = f(a);
        test::check(e.at(0) == 2, "frompyfunc");
    }
    // piecewise
    {
        auto x = ndarray<int>::from_data({5}, {-2, -1, 0, 1, 2});
        auto cond1 = ndarray<bool>::from_data({5}, {true, true, false, false, false});  // x<0
        auto cond2 = ndarray<bool>::from_data({5}, {false, false, true, false, false}); // x==0
        // funclist as arrays: -1 where cond1, 0 where cond2, 1 otherwise
        auto f1 = ndarray<int>::from_data({1}, {-1});
        auto f2 = ndarray<int>::from_data({1}, {0});
        auto f3 = ndarray<int>::from_data({1}, {1});
        auto out = piecewise(x, {cond1, cond2}, std::vector<ndarray<int>>{f1, f2, f3});
        test::check(out.at(0) == -1 && out.at(2) == 0 && out.at(4) == 1, "piecewise array");
        // funclist as callables
        auto out2 =
            piecewise<int>(x, {cond1, cond2},
                           std::vector<std::function<int(int)>>{[](int v) { return v * 2; }, [](int v) { return 0; },
                                                                [](int v) { return v * 3; }});
        test::check(out2.at(0) == -4 && out2.at(3) == 3, "piecewise func");
        // condlist size mismatch -> throws
        bool threw = false;
        try
        {
            piecewise(x, {cond1}, std::vector<ndarray<int>>{f1});
        }
        catch (...)
        {
            threw = true;
        }
        // Actually condlist 1, funclist 1 is valid (size equal) -> not throw, test with size mismatch 1 vs 3
        threw = false;
        try
        {
            auto bad = piecewise(x, {cond1, cond2}, std::vector<ndarray<int>>{f1});
        }
        catch (...)
        {
            threw = true;
        }
        test::check(threw, "piecewise mismatch");
    }
    if (test::failures() == 0)
        std::printf("OK functional\n");
    return test::failures() ? 1 : 0;
}
