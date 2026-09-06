/**
 * @file test_scalar_custom.cpp
 * @brief Runtime tests proving the fixed-shape np::ndarrayf<T, Extents...>
 *        works with custom element types via the internal
 *        np::detail::fixed::scalar_traits<T> backend.
 *
 * Two kinds of custom types are exercised:
 *  1. The _Np_dtype storage-classifier types (self-describing dtype
 *     scalars defined in dtype.hpp, backed by scalar_custom.hpp).
 *  2. A user-defined scalar type specialized on scalar_traits in the test
 *     itself, demonstrating how any third-party scalar plugs into the same
 *     array business logic without touching ndarray_fixed.hpp.
 */
#include <cstddef>
#include <string>
#include <type_traits>

#include "np/np.hpp"
#include "test_util.hpp"

// A user-defined scalar: routes through the scalar_traits customization
// point exactly like the _Np_dtype classifiers.
struct temperature
{
    double value = 0.0;
};

namespace np::detail::fixed
{

template <> struct scalar_traits<::temperature>
{
    static constexpr bool is_custom = true;
    using value_type = double;
    static constexpr const double &get(const ::temperature &v) noexcept
    {
        return v.value;
    }
    static constexpr ::temperature make(const double &v) noexcept
    {
        return ::temperature{v};
    }
    static constexpr double zero() noexcept
    {
        return 0.0;
    }
    static constexpr double one() noexcept
    {
        return 1.0;
    }
    static constexpr bool truthy(const ::temperature &v) noexcept
    {
        return v.value != 0.0;
    }
};

} // namespace np::detail::fixed

int main()
{
    using i64 = np::_Np_dtype::_Np_int64;
    using f64 = np::_Np_dtype::_Np_float64;

    // Construction and access (rank-1 and rank-2).
    {
        np::ndarrayf<i64, 4> a{1, 2, 3, 4};
        test::check(a.size() == 4, "custom 1-D size");
        test::check(a[0].value() == 1 && a[3].value() == 4, "custom flat access");
        test::check(a[0] == a[0] && a[1].value() == 2, "custom flat access");
        test::check(a.data()[2].value() == 3, "custom data()");

        np::ndarrayf<i64, 2, 3> b{{1, 2, 3}, {4, 5, 6}};
        test::check(b(1, 2).value() == 6, "custom operator() 2-D");
        const auto &cb = b;
        test::check(cb(0, 1).value() == 2, "custom const access");
    }

    // Reductions: computations run on the unwrapped core, results are
    // re-wrapped into the classifier (dtype preserved).
    {
        np::ndarrayf<i64, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        test::check(a.sum().value() == 21, "custom sum all");
        test::check(a.prod().value() == 720, "custom prod all");
        test::check(a.min().value() == 1 && a.max().value() == 6, "custom min/max");
        test::check(a.argmin() == 0 && a.argmax() == 5, "custom argmin/argmax");
        test::check(test::approx(a.mean(), 3.5), "custom mean promotes to double");

        const auto rowsum = a.sum<1>();
        test::check(rowsum.rank == 1 && rowsum[0] == 6 && rowsum[1] == 15, "custom sum axis");

        const auto rowmax = a.max<1>();
        test::check(rowmax[0].value() == 3 && rowmax[1].value() == 6, "custom max axis");

        test::check(a.all(), "custom all");
        np::ndarrayf<i64, 3> zero{0, 0, 0};
        test::check(!zero.any(), "custom any false");

        np::ndarrayf<f64, 3> f{1.5, 2.5, 3.5};
        test::check(test::approx(f.sum().value(), 7.5), "custom float sum");
        test::check(test::approx(f.mean(), 2.5), "custom float mean");
    }

    // Elementwise expressions (lazy nodes route through binary_apply /
    // unary_apply and re-wrap into the classifier).
    {
        np::ndarrayf<i64, 2, 3> a{{1, 2, 3}, {4, 5, 6}};

        const auto sum2 = a + a;
        test::check(sum2.rank == 2 && sum2(1, 2).value() == 12, "custom a + a");

        // Mixed custom/builtin resolves to the plain promoted core.
        const auto plus1 = a + 1;
        test::check(plus1(0, 0) == 2 && plus1(1, 2) == 7, "custom a + scalar");
        const auto times2 = a * 2;
        test::check(times2(1, 0) == 8, "custom a * scalar");
        const auto div2 = a / 2;
        test::check(div2(0, 1) == 1, "custom a / scalar");

        // Negation keeps the classifier (unary core is the classifier core).
        const auto neg = -a;
        test::check(neg(0, 1) == -2, "custom unary minus");
        const auto squares = a * a;
        test::check(squares(1, 1) == 25, "custom a * a");

        // Comparisons/logical kernels yield bool (NumPy semantics).
        const auto eq = a == a;
        static_assert(std::is_same_v<decltype(eq.eval()), np::ndarrayf<bool, 2, 3>>,
                      "custom == materializes a bool array");
        test::check(eq[0] == true && eq[5] == true, "custom a == a");
        const auto lt = a < 3;
        test::check(lt(0, 2) == false && lt(0, 1) == true, "custom a < scalar");
    }

    // Broadcasting against custom classifiers mirrors the builtin rules.
    {
        np::ndarrayf<i64, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        np::ndarrayf<i64, 3> row{10, 20, 30};
        const auto out = a + row;
        test::check(out(0, 2).value() == 33 && out(1, 2).value() == 36, "custom broadcast row");
        np::ndarrayf<i64, 2, 1> mx{{1}, {100}};
        const auto col = a * mx;
        test::check(col(0, 1).value() == 2 && col(1, 2).value() == 600, "custom broadcast col");
    }

    // Manipulation retains the custom element type.
    {
        np::ndarrayf<i64, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        auto t = a.transpose();
        static_assert(std::is_same_v<decltype(t), np::ndarrayf<i64, 3, 2>>, "custom transpose keeps the dtype");
        test::check(t(1, 0).value() == 2, "custom transpose value");
        const auto fl = a.flatten();
        test::check(fl[5].value() == 6, "custom flatten value");
    }

    // A fully user-defined scalar type works through the same code path.
    {
        np::ndarrayf<::temperature, 3> temps{::temperature{1.0}, ::temperature{2.5}, ::temperature{-0.5}};
        test::check(test::approx(temps.sum().value, 3.0), "user scalar sum");
        test::check(test::approx(temps.mean(), 1.0), "user scalar mean");
        const auto d = temps * 2;
        test::check(test::approx(d[1].value, 5.0), "user scalar * scalar");
        np::ndarrayf<::temperature, 2> t2{::temperature{1.0}, ::temperature{0.0}};
        test::check(t2.all() == false, "user scalar truthiness");
    }

    // String-branch classifiers: get/make/truthy operate on the text core.
    {
        np::ndarrayf<np::_Np_dtype::_Np_string, 2> s{std::string{"ab"}, std::string{"cd"}};
        test::check(s[0].value() == "ab", "string element");
        test::check(s.all(), "string all");
    }

    return test::failures() ? 1 : 0;
}