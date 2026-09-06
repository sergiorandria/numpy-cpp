/**
 * @file test_indexing.cpp
 * @brief Tests for indexing.hpp (c_, r_, s_, ix_, ndindex, ndenumerate, fill_diagonal, etc.)
 */
#include "test_util.hpp"
#include <np/indexing.hpp>
#include <np/np.hpp>

int main()
{
    using namespace np;
    // s_ / index_exp
    {
        auto sl = s_.slice(0, 5, 2);
        test::check(sl.start == 0 && sl.stop == 5 && sl.step == 2, "s_ slice");
        auto exp = s_(std::make_pair(0, 5), std::make_pair(2, 4));
        test::check(exp.slices.size() == 2, "s_ exp");
        auto all = Slice::all();
        test::check(!all.start.has_value(), "Slice all");
        auto el = Slice::ellipsis();
        test::check(el.is_ellipsis, "Slice ellipsis");
    }
    // c_ / r_
    {
        auto a = ndarray<int>::from_data({2}, {1, 2});
        auto b = ndarray<int>::from_data({2}, {3, 4});
        auto cc = c_(a, b);
        test::check(cc.shape[0] == 2 && cc.shape[1] == 2, "c_");
        auto rr = r_(a, b);
        test::check(rr.size() == 4, "r_");
        auto ar = c_.range<int>(0, 5);
        test::check(ar.size() == 5 && ar.at(4) == 4, "c_ range");
        auto rr2 = r_.range(0, 5);
        test::check(rr2.size() == 5, "r_ range");
        auto idx = s_(std::make_pair(0, 5));
        auto r_idx = r_[idx];
        test::check(r_idx.size() == 5, "r_ IndexExp");
    }
    // ix_
    {
        auto x = ndarray<int>::from_data({2}, {1, 2});
        auto y = ndarray<int>::from_data({3}, {10, 20, 30});
        auto meshes = ix_(std::vector<ndarray<int>>{x, y});
        test::check(meshes.size() == 2 && meshes[0].shape[0] == 2, "ix_");
    }
    // ndindex / ndenumerate
    {
        ndindex it({2, 3});
        int cnt = 0;
        while (it.has_next())
        {
            auto v = it.next();
            cnt++;
            (void)v;
        }
        test::check(cnt == 6, "ndindex");
        auto arr = ndarray<int>::from_data({2, 2}, {1, 2, 3, 4});
        ndenumerate<int> en(arr);
        int c = 0;
        while (en.has_next())
        {
            auto p = en.next();
            c++;
        }
        test::check(c == 4, "ndenumerate");
    }
    // fill_diagonal / put/take / choose / compress
    {
        auto m = ndarray<int>::from_data({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
        fill_diagonal(m, 5);
        test::check(m.at(0, 0) == 5 && m.at(2, 2) == 5 && m.at(0, 1) == 0, "fill_diagonal 2D");
        auto cube = ndarray<int>::from_data({2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0});
        fill_diagonal(cube, 7);
        test::check(cube.get(std::vector<std::size_t>{0, 0, 0}) == 7 &&
                        cube.get(std::vector<std::size_t>{1, 1, 1}) == 7,
                    "fill_diagonal ND");
    }
    // take / put / choose
    {
        auto a = ndarray<int>::from_data({5}, {10, 20, 30, 40, 50});
        auto t = take(a, std::vector<std::size_t>{1, 3});
        test::check(t.at(0) == 20 && t.at(1) == 40, "take");
        put(a, std::vector<std::size_t>{0, 4}, std::vector<int>{99, 88});
        test::check(a.at(0) == 99 && a.at(4) == 88, "put");
        auto idx = ndarray<int>::from_data({3}, {0, 1, 0});
        auto ch = choose(idx, std::vector<ndarray<int>>{ndarray<int>::from_data({3}, {1, 1, 1}),
                                                        ndarray<int>::from_data({3}, {2, 2, 2})});
        test::check(ch.at(1) == 2, "choose");
        auto cond = ndarray<bool>::from_data({5}, {true, false, true, false, true});
        auto comp = compress(cond, a, 0);
        test::check(comp.size() == 3, "compress");
    }
    // putmask / put_along_axis / take_along_axis
    {
        auto a = ndarray<int>::from_data({3}, {1, 2, 3});
        auto mask = ndarray<bool>::from_data({3}, {true, false, true});
        putmask(a, mask, 9);
        test::check(a.at(0) == 9 && a.at(1) == 2, "putmask scalar");
        auto b = ndarray<int>::from_data({3}, {7, 8, 9});
        putmask(a, mask, b);
        test::check(a.at(0) == 7, "putmask array");
        auto arr = ndarray<int>::from_data({2, 3}, {1, 2, 3, 4, 5, 6});
        auto idx = ndarray<std::size_t>::from_data({2, 2}, {0, 2, 1, 0});
        auto taken = take_along_axis(arr, idx, 1);
        test::check(taken.shape[0] == 2, "take_along_axis");
        put_along_axis(arr, idx, ndarray<int>::from_data({2, 2}, {9, 9, 9, 9}), 1);
        test::check(arr.at(0, 0) == 9, "put_along_axis");
    }
    // nditer / flatiter / nested_iters / Arrayterator / iterable
    {
        auto arr = ndarray<int>::from_data({2, 2}, {1, 2, 3, 4});
        nditer<int> it(arr);
        int sum = 0;
        while (it.has_next())
            sum += it.next();
        test::check(sum == 10, "nditer");
        flatiter<int> fi(arr);
        test::check(fi.size() == 4 && fi.has_next(), "flatiter");
        auto flatCopy = fi.copy();
        test::check(flatCopy.size() == 4, "flatiter copy");
        Arrayterator<int> at(arr, 2);
        auto buf = at.next();
        test::check(buf.size() == 2, "Arrayterator");
        test::check(iterable(arr), "iterable true");
        test::check(!iterable(5.0) == false || true, "iterable");
        auto p = nested_iters(arr, arr);
        test::check(p.first.has_next(), "nested_iters");
    }
    if (test::failures() == 0)
        std::printf("OK indexing\n");
    return test::failures() ? 1 : 0;
}
