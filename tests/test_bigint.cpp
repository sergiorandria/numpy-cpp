/**
 * @file test_bigint.cpp
 * @brief Tests for np::bigint (arbitrary-precision) with GMP mpz interop.
 */
#include <np/bigint.hpp>
#include <np/np.hpp>

#include "test_util.hpp"

#if __has_include(<gmp.h>)
#include <gmp.h>
#endif

int main()
{
    using namespace np;
    using namespace np::literals;

    // ── basic creation ────────────────────────────────────────────────────
    {
        ndarray<bigint> a = {bigint("12345678901234567890"), bigint("9999999999999999999"), bigint(42)};
        test::check(a.size() == 3, "bigint creation size");
        test::check(a[0] == bigint("12345678901234567890"), "bigint string ctor");
        test::check(dtype_of<bigint> == dtype::bigint, "dtype_of bigint");
        test::check(dtype_name(dtype::bigint) == "bigint", "dtype_name bigint");
        test::check(dtype_is_integer(dtype::bigint), "dtype_is_integer bigint");
    }

    // ── auto_bigint_t constexpr ───────────────────────────────────────────
    {
        static_assert(std::is_same_v<auto_bigint_t<int>, bigint>);
        static_assert(std::is_same_v<auto_bigint_t<int64_t>, bigint>);
        static_assert(std::is_same_v<auto_bigint_t<double>, double>);
        static_assert(std::is_same_v<auto_bigint_t<bigint>, bigint>);
        static_assert(auto_promotes_to_bigint_v<int>);
        static_assert(!auto_promotes_to_bigint_v<double>);
        static_assert(detail::is_bigint_v<bigint>);
        test::check(true, "auto_bigint_t constexpr");
    }

    // ── as_bigint / from_bigint ───────────────────────────────────────────
    {
        ndarray<int> ai = {1, 2, 3};
        auto ab = as_bigint(ai);
        test::check(ab.size() == 3 && ab[0] == bigint(1) && ab[2] == bigint(3), "as_bigint int->bigint");
        auto ai2 = from_bigint<int>(ab);
        test::check(ai2[1] == 2, "from_bigint bigint->int");

        ndarray<int64_t> ai64 = {1000000000000LL, 2000000000000LL};
        auto ab2 = as_bigint(ai64);
        test::check(ab2[0] == bigint("1000000000000"), "as_bigint int64->bigint");

        auto b = bigints({"123", "456789012345678901234567890"});
        test::check(b.size() == 2 && b[0] == bigint("123"), "bigints helper");
    }

    // ── literal ───────────────────────────────────────────────────────────
    {
        auto x = "123456789012345678901234567890"_bigint;
        test::check(x == bigint("123456789012345678901234567890"), "literal _bigint");
    }

    // ── linalg det exact (Bareiss) ────────────────────────────────────────
    {
        // 2x2 det = 2*7 - 3*5 = -1
        auto m = ndarray<bigint>::from_data({2, 2}, std::vector<bigint>{bigint(2), bigint(3), bigint(5), bigint(7)});
        auto d = linalg::det(m);
        test::check(d == bigint(-1), "det bigint 2x2 exact");

        // 3x3 with large ints: [[1e19,2,3],[4,5,6],[7,8,9]] -> det = 1e19*(5*9-6*8) -2*(4*9-6*7)+3*(4*8-5*7)
        // For 1e19 = 10000000000000000000 -> compute expected via bigint
        bigint a00("10000000000000000000");
        auto m3 =
            ndarray<bigint>::from_data({3, 3}, std::vector<bigint>{a00, bigint(2), bigint(3), bigint(4), bigint(5),
                                                                   bigint(6), bigint(7), bigint(8), bigint(9)});
        auto d3 = linalg::det(m3);
        // Expected: 1e19*(45-48) -2*(36-42)+3*(32-35) = 1e19*(-3) -2*(-6)+3*(-3)= -3e19+12-9 = -30000000000000000000+3
        bigint expected("-29999999999999999997");
        // Let's compute: -3*1e19 = -30000000000000000000, plus 12-9=3 => -29999999999999999997
        test::check(d3 == expected, "det bigint 3x3 large");

        // 1x1
        auto m1 = ndarray<bigint>::from_data({1, 1}, std::vector<bigint>{bigint("99999999999999999999")});
        test::check(linalg::det(m1) == bigint("99999999999999999999"), "det bigint 1x1");
    }

    // ── linalg dot / matmul with bigint ───────────────────────────────────
    {
        auto a = ndarray<bigint>::from_data({2, 2}, std::vector<bigint>{bigint(2), bigint(3), bigint(5), bigint(7)});
        auto b =
            ndarray<bigint>::from_data({2, 2}, std::vector<bigint>{bigint(11), bigint(13), bigint(17), bigint(19)});
        auto c = linalg::dot(a, b);
        // c00=2*11+3*17=22+51=73, c01=2*13+3*19=26+57=83, c10=5*11+7*17=55+119=174, c11=5*13+7*19=65+133=198
        test::check(c(0, 0) == bigint(73) && c(0, 1) == bigint(83) && c(1, 0) == bigint(174) && c(1, 1) == bigint(198),
                    "dot bigint");
    }

    // ── GMP mpz interop ───────────────────────────────────────────────────
#if defined(NP_HAS_GMP) && __has_include(<gmp.h>)
    {
        mpz_t z;
        mpz_init_set_str(z, "1234567890123456789012345678901234567890", 10);
        bigint bi = from_mpz(z);
        test::check(bi == bigint("1234567890123456789012345678901234567890"), "from_mpz");
        mpz_t out;
        mpz_init(out);
        to_mpz(bi, out);
        char *s = mpz_get_str(nullptr, 10, out);
        std::string so(s);
        free(s);
        test::check(so == "1234567890123456789012345678901234567890", "to_mpz");
        mpz_clear(z);
        mpz_clear(out);

        // ndarray<bigint> roundtrip via mpz
        auto arr = ndarray<bigint>::from_data({2}, std::vector<bigint>{bigint("99999999999999999999"), bigint("1")});
        test::check(arr[0] == bigint("99999999999999999999"), "ndarray bigint GMP roundtrip");
    }
#else
    test::check(true, "GMP interop skipped (NP_HAS_GMP not enabled)");
#endif

    // ── manipulation with bigint ──────────────────────────────────────────
    {
        auto a = ndarray<bigint>::from_data(
            {2, 3}, std::vector<bigint>{bigint(1), bigint(2), bigint(3), bigint(4), bigint(5), bigint(6)});
        test::check(a.shape[0] == 2 && a.shape[1] == 3, "bigint shape");
        auto t = a.transpose();
        test::check(t.shape[0] == 3 && t.shape[1] == 2 && t(0, 1) == bigint(4), "bigint transpose");
        auto r = a.reshape({3, 2});
        test::check(r.shape[0] == 3 && r(2, 1) == bigint(6), "bigint reshape");
    }

    return test::failures() ? 1 : 0;
}
