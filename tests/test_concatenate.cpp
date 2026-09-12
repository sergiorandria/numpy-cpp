/**
 * @file test_concatenate.cpp
 * @brief Tests for array concatenation and stacking (concatenate.hpp).
 *
 * Verifies concatenate, stack, vstack, hstack, dstack, column_stack, row_stack.
 */
#include "test_util.hpp"
#include <np/concatenate.hpp>
#include <np/creation.hpp>
#include <np/ndarray.hpp>

int main()
{
    using namespace np;

    // --- concatenate along axis 0 ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3});
        auto b = asarray(std::vector<int>{4, 5, 6});

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = concatenate(arrays, 0);
        test::check(c.shape[0] == 6, "concatenate axis 0: shape");
        test::check(c.at(0) == 1, "first element");
        test::check(c.at(3) == 4, "fourth element");
        test::check(c.at(5) == 6, "last element");
    }

    // --- concatenate 2D along axis 0 ---
    {
        ndarray<int> a{{1, 2}, {3, 4}};
        ndarray<int> b{{5, 6}, {7, 8}};

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = concatenate(arrays, 0);
        test::check(c.shape[0] == 4 && c.shape[1] == 2, "concat 2D axis 0: shape");
        test::check(c.at(0, 0) == 1, "element [0,0]");
        test::check(c.at(2, 0) == 5, "element [2,0]");
        test::check(c.at(3, 1) == 8, "element [3,1]");
    }

    // --- concatenate 2D along axis 1 ---
    {
        ndarray<int> a{{1, 2}, {3, 4}};
        ndarray<int> b{{5, 6}, {7, 8}};

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = concatenate(arrays, 1);
        test::check(c.shape[0] == 2 && c.shape[1] == 4, "concat 2D axis 1: shape");
        test::check(c.at(0, 0) == 1, "element [0,0]");
        test::check(c.at(0, 2) == 5, "element [0,2]");
        test::check(c.at(1, 3) == 8, "element [1,3]");
    }

    // --- stack along new axis ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3});
        auto b = asarray(std::vector<int>{4, 5, 6});

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = stack(arrays, 0);
        test::check(c.shape[0] == 2 && c.shape[1] == 3, "stack axis 0: shape");
        test::check(c.at(0, 0) == 1, "element [0,0]");
        test::check(c.at(1, 0) == 4, "element [1,0]");
        test::check(c.at(1, 2) == 6, "element [1,2]");

        auto d = stack(arrays, 1);
        test::check(d.shape[0] == 3 && d.shape[1] == 2, "stack axis 1: shape");
        test::check(d.at(0, 0) == 1, "element [0,0]");
        test::check(d.at(0, 1) == 4, "element [0,1]");
        test::check(d.at(2, 1) == 6, "element [2,1]");
    }

    // --- vstack (vertical) ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3});
        auto b = asarray(std::vector<int>{4, 5, 6});

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = vstack(arrays);
        test::check(c.shape[0] == 2 && c.shape[1] == 3, "vstack 1D: shape");
        test::check(c.at(0, 0) == 1, "element [0,0]");
        test::check(c.at(1, 0) == 4, "element [1,0]");

        ndarray<int> d{{1, 2}};
        ndarray<int> e{{3, 4}};
        std::vector<ndarray<int>> arrays2 = {d, e};
        auto f = vstack(arrays2);
        test::check(f.shape[0] == 2 && f.shape[1] == 2, "vstack 2D: shape");
        test::check(f.at(0, 0) == 1, "element [0,0]");
        test::check(f.at(1, 1) == 4, "element [1,1]");
    }

    // --- hstack (horizontal) ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3});
        auto b = asarray(std::vector<int>{4, 5, 6});

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = hstack(arrays);
        test::check(c.shape[0] == 6, "hstack 1D: shape");
        test::check(c.at(0) == 1, "first element");
        test::check(c.at(3) == 4, "fourth element");

        ndarray<int> d = ndarray<int>::from_data({2, 1}, {1, 2});
        ndarray<int> e = ndarray<int>::from_data({2, 1}, {3, 4});
        std::vector<ndarray<int>> arrays2 = {d, e};
        auto f = hstack(arrays2);
        test::check(f.shape[0] == 2 && f.shape[1] == 2, "hstack 2D: shape");
        test::check(f.at(0, 0) == 1, "element [0,0]");
        test::check(f.at(0, 1) == 3, "element [0,1]");
        test::check(f.at(1, 0) == 2, "element [1,0]");
        test::check(f.at(1, 1) == 4, "element [1,1]");
    }

    // --- dstack (depth) ---
    {
        ndarray<int> a{{1, 2}, {3, 4}};
        ndarray<int> b{{5, 6}, {7, 8}};

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = dstack(arrays);
        test::check(c.ndim() == 3, "dstack: 3D");
        test::check(c.shape[0] == 2 && c.shape[1] == 2 && c.shape[2] == 2, "dstack: shape");
        test::check(c.get(std::vector<std::size_t>{0, 0, 0}) == 1, "element [0,0,0]");
        test::check(c.get(std::vector<std::size_t>{0, 0, 1}) == 5, "element [0,0,1]");
    }

    // --- column_stack ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3});
        auto b = asarray(std::vector<int>{4, 5, 6});

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = column_stack(arrays);
        test::check(c.shape[0] == 3 && c.shape[1] == 2, "column_stack: shape");
        test::check(c.at(0, 0) == 1, "element [0,0]");
        test::check(c.at(0, 1) == 4, "element [0,1]");
        test::check(c.at(2, 1) == 6, "element [2,1]");
    }

    // --- row_stack (alias for vstack) ---
    {
        auto a = asarray(std::vector<int>{1, 2, 3});
        auto b = asarray(std::vector<int>{4, 5, 6});

        std::vector<ndarray<int>> arrays = {a, b};
        auto c = row_stack(arrays);
        test::check(c.shape[0] == 2 && c.shape[1] == 3, "row_stack: shape");
        test::check(c.at(0, 0) == 1, "element [0,0]");
        test::check(c.at(1, 0) == 4, "element [1,0]");
    }

    return test::failures() ? 1 : 0;
}
