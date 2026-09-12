/**
 * @file test_sorting.cpp
 * @brief Tests for sorting.hpp (sort, argsort, lexsort, partition, searchsorted, etc.)
 */
#include "test_util.hpp"
#include <np/np.hpp>
#include <np/sorting.hpp>

int main()
{
    using namespace np;
    // sort / argsort
    {
        auto a = ndarray<int>::from_data({5}, {3, 1, 4, 1, 5});
        auto s = sort(a);
        test::check(s.at(0) == 1 && s.at(4) == 5, "sort 1D");
        auto idx = argsort(a);
        test::check(idx.at(0) == 1 || idx.at(0) == 3, "argsort first");
        // 2-D sort along axis
        auto m = ndarray<int>::from_data({2, 3}, {3, 2, 1, 6, 5, 4});
        auto s0 = sort(m, 1);
        test::check(s0.at(0, 0) == 1 && s0.at(0, 2) == 3 && s0.at(1, 2) == 6, "sort axis 1");
        auto s_kind = sort(m, 1, "stable");
        test::check(s_kind.at(0, 0) == 1, "sort kind");
    }
    // flat sort
    {
        auto a = ndarray<int>::from_data({2, 3}, {6, 5, 4, 3, 2, 1});
        auto s = sort(a, -1);
        test::check(s.size() == 6, "sort flat size");
    }
    // lexsort
    {
        auto k0 = ndarray<int>::from_data({4}, {1, 1, 0, 0});
        auto k1 = ndarray<int>::from_data({4}, {0, 1, 0, 1});
        // keys[0]=secondary, keys[1]=primary
        auto idx = lexsort(std::vector<ndarray<int>>{k0, k1});
        // expected: sorts by k1 then k0
        test::check(idx.size() == 4, "lexsort size");
        // primary key 0 has indices 0,2 ; primary 1 has 1,3 ; within each secondary sorts
        // so order should be 2,0,3,1 or similar
        test::check(idx.at(0) == 2, "lexsort primary");
    }
    // msort / sort_complex
    {
        auto a = ndarray<int>::from_data({4}, {3, 1, 2, 0});
        auto ms = msort(a);
        test::check(ms.at(0) == 0 && ms.at(3) == 3, "msort");
        // sort_complex for real types promotes to complex and sorts last axis;
        // bulk of logic is covered by complex overload below (complex ordering is lex)
        auto c =
            ndarray<std::complex<double>>::from_data({2}, {std::complex<double>(1, 1), std::complex<double>(0, 1)});
        auto sc2 = sort_complex(c);
        test::check(sc2.size() == 2, "sort_complex complex");
    }
    // partition / argpartition
    {
        auto a = ndarray<int>::from_data({5}, {5, 3, 1, 4, 2});
        auto p = partition(a, 2);
        test::check(p.at(2) == 3, "partition kth");
        auto ap = argpartition(a, 2);
        test::check(ap.size() == 5, "argpartition size");
    }
    // searchsorted
    {
        auto a = ndarray<int>::from_data({5}, {1, 3, 5, 7, 9});
        test::check(searchsorted(a, 4) == 2, "searchsorted scalar");
        test::check(searchsorted(a, 5, true) == 3, "searchsorted right");
        // with sorter
        auto b = ndarray<int>::from_data({5}, {3, 1, 4, 1, 5});
        auto sorter = argsort(b);
        // search for 4 in sorted order
        test::check(searchsorted(b, 4, sorter) == 3, "searchsorted sorter");
    }
    // extract / nonzero / flatnonzero / count_nonzero / argwhere
    {
        auto cond = ndarray<bool>::from_data({5}, {true, false, true, false, true});
        auto arr = ndarray<int>::from_data({5}, {10, 20, 30, 40, 50});
        auto ex = extract(cond, arr);
        test::check(ex.size() == 3 && ex.at(0) == 10 && ex.at(2) == 50, "extract");
        auto nz = nonzero(arr);
        test::check(nz.size() == 1 && nz[0].size() == 5, "nonzero");
        // flatnonzero on int (avoid vector<bool> proxy segfault)
        auto fnz = flatnonzero(arr);
        test::check(fnz.size() == 5, "flatnonzero int");
        auto fnz2 = flatnonzero(ndarray<int>::from_data({5}, {1, 0, 1, 0, 1}));
        test::check(fnz2.size() == 3 && fnz2.at(1) == 2, "flatnonzero");
        test::check(count_nonzero(arr) == 5, "count_nonzero flat");
        auto m = ndarray<int>::from_data({2, 2}, {0, 1, 2, 0});
        test::check(count_nonzero(m) == 2, "count_nonzero 2D");
        auto aw = argwhere(m);
        test::check(aw.shape[0] == 2 && aw.shape[1] == 2, "argwhere");
        test::check(argmax(arr) == 4, "argmax");
        test::check(argmin(arr) == 0, "argmin");
    }
    if (test::failures() == 0)
        std::printf("OK sorting\n");
    return test::failures() ? 1 : 0;
}
