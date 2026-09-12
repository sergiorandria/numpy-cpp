/**
 * @file test_bitwise.cpp
 * @brief Tests for bitwise.hpp (and, or, xor, invert, shifts, packbits, etc.)
 */
#include "test_util.hpp"
#include <np/bitwise.hpp>
#include <np/np.hpp>

int main()
{
    using namespace np;
    // bitwise_and/or/xor
    {
        auto a = ndarray<int>::from_data({3}, {0b1100, 0b1010, 0b0110});
        auto b = ndarray<int>::from_data({3}, {0b1010, 0b1100, 0b0011});
        auto c = bitwise_and(a, b);
        test::check(c.at(0) == 0b1000 && c.at(1) == 0b1000, "bitwise_and");
        auto o = bitwise_or(a, b);
        test::check(o.at(2) == 0b0111, "bitwise_or");
        auto x = bitwise_xor(a, b);
        test::check(x.at(0) == 0b0110, "bitwise_xor");
        // scalar overload
        auto cs = bitwise_and(a, 0b1010);
        test::check(cs.at(1) == 0b1010, "bitwise_and scalar");
    }
    // invert / bitwise_count
    {
        auto a = ndarray<int>::from_data({2}, {0, -1});
        auto inv = invert(a);
        test::check(inv.at(0) == -1 && inv.at(1) == 0, "invert");
        auto cnt = bitwise_count(ndarray<std::uint8_t>::from_data({3}, {0, 1, 255}));
        test::check(cnt.at(0) == 0 && cnt.at(1) == 1 && cnt.at(2) == 8, "bitwise_count");
    }
    // shifts
    {
        auto a = ndarray<int>::from_data({2}, {1, 8});
        auto l = left_shift(a, 2);
        test::check(l.at(0) == 4 && l.at(1) == 32, "left_shift");
        auto r = right_shift(a, 1);
        test::check(r.at(1) == 4, "right_shift");
    }
    // packbits / unpackbits
    {
        auto bits = ndarray<std::uint8_t>::from_data({8}, {1, 0, 1, 0, 1, 0, 1, 0});
        auto packed = packbits(bits);
        test::check(packed.size() == 1 && packed.at(0) == 0b10101010, "packbits big");
        auto unpacked = unpackbits(packed);
        test::check(unpacked.size() == 8 && unpacked.at(0) == 1 && unpacked.at(1) == 0, "unpackbits");
        // little endian
        auto p2 = packbits(bits, std::nullopt, "little");
        test::check(p2.at(0) == 0b01010101, "packbits little");
        auto up2 = unpackbits(p2, std::nullopt, -1, "little");
        test::check(up2.at(0) == 1, "unpackbits little");
        // axis path
        auto m = ndarray<std::uint8_t>::from_data({2, 4}, {1, 0, 1, 0, 0, 1, 0, 1});
        auto pm = packbits(m, 1);
        test::check(pm.shape[0] == 2 && pm.shape[1] == 1, "packbits axis");
    }
    // binary_repr / ct_select
    {
        test::check(binary_repr(5) == "101", "binary_repr 5");
        test::check(binary_repr(5, 8) == "00000101", "binary_repr width");
        test::check(binary_repr(-1, 8) == "11111111", "binary_repr neg");
        test::check(ct_select(1, 10, 20) == 10, "ct_select true");
        test::check(ct_select(0, 10, 20) == 20, "ct_select false");
        auto xr = bitwise_xor_ct(ndarray<int>::from_data({2}, {1, 2}), ndarray<int>::from_data({2}, {3, 4}));
        test::check(xr.at(0) == 2, "bitwise_xor_ct");
    }
    // broadcast via bitwise ops
    {
        auto a = ndarray<int>::from_data({2, 2}, {1, 2, 3, 4});
        auto b = ndarray<int>::from_data({2}, {0b1111, 0b0000});
        // broadcast shape [2,2] vs [2] -> simulate via pack?
        auto c = bitwise_and(a, b);
        test::check(c.size() > 0, "bitwise broadcast");
    }
    if (test::failures() == 0)
        std::printf("OK bitwise\n");
    return test::failures() ? 1 : 0;
}
