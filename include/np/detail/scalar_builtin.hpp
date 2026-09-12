/**
 * @file scalar_builtin.hpp
 * @brief Internal scalar backend for plain (builtin) element types.
 *
 * The array business logic (ndarray_fixed.hpp and detail/expr.hpp) routes
 * every per-element computation through the internal class
 * `np::detail::fixed::scalar_traits<T>`, so one code path serves both the
 * builtin C++ scalars and the custom `dtype_storage` storage-classifier types
 * from dtype.hpp.
 *
 * This header ships the primary template: the identity behaviour for the
 * plain scalars (arithmetic, bool and std::complex), plus the elementwise
 * `binary_apply` / `unary_apply` dispatch for the builtin branch. The
 * custom backend for the `dtype_storage` classifier types lives in
 * scalar_custom.hpp.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DETAIL_SCALAR_BUILTIN_HPP
#define NP_DETAIL_SCALAR_BUILTIN_HPP

#include "../api_macros.hpp"

#include <complex>
#include <type_traits>
#include <utility>

namespace np::detail::fixed
{

/** @brief True when T is a std::complex instantiation. */
template <typename T> struct is_complex_instance : std::false_type
{
};
template <typename T> struct is_complex_instance<std::complex<T>> : std::true_type
{
};
template <typename T> inline constexpr bool is_complex_instance_v = is_complex_instance<T>::value;

/** @brief True for a plain scalar element type (arithmetic / complex). */
template <typename T> inline constexpr bool is_plain_scalar_v = std::is_arithmetic_v<T> || is_complex_instance_v<T>;

/**
 * @brief Scalar backend for a plain (builtin) element type.
 *
 * For the builtin scalars everything is the identity: `value_type == T`,
 * `get`/`make` are passthroughs and `zero()/one()` match the current
 * `T{}` / `T{1}` defaults used by the array business logic, so a builtin
 * instantiation compiles to the exact same code as before.
 *
 * @tparam T  A plain scalar element type.
 */
template <typename T> struct scalar_traits
{
    static constexpr bool is_custom = false;

    /** @brief Numeric core that reductions and kernels compute in. */
    using value_type = std::remove_cv_t<T>;

    /** @brief Read the computation core out of a stored element. */
    static constexpr const value_type &get(const value_type &v) noexcept
    {
        return v;
    }

    /** @brief Store a computation result into an element slot. */
    static constexpr value_type make(const value_type &v) noexcept
    {
        return v;
    }

    /** @brief Additive identity used by sum()/axis reductions. */
    static constexpr value_type zero() noexcept
    {
        return value_type{};
    }

    /** @brief Multiplicative identity used by prod(). */
    static constexpr value_type one() noexcept
    {
        return value_type{1};
    }

    /** @brief Truthiness used by all()/any(). */
    static constexpr bool truthy(const value_type &v) noexcept
    {
        if constexpr (std::is_arithmetic_v<value_type>)
        {
            return v != value_type{0};
        }
        else
        {
            return v != value_type{};
        }
    }
};

/** @brief Computation core of any element type (identity for plain). */
template <typename T> using scalar_core_t = typename scalar_traits<std::remove_cv_t<T>>::value_type;

// Elementwise kernel dispatch (used by detail/expr.hpp nodes).
//
// `binary_apply<Op, A, B, Custom>` maps an elementwise functor over a pair
// of stored element types A, B to its value type and its call kernel. The
// builtin branch (this header) feeds the stored values straight to the
// functor; scalar_custom.hpp supplies the `true` specialization that
// unwraps custom classifiers first.
/** @brief True when a binary kernel must unwrap at least one operand. */
template <typename A, typename B>
inline constexpr bool needs_custom_kernel = scalar_traits<A>::is_custom || scalar_traits<B>::is_custom;

/** @brief Builtin binary evaluation (stored values pass through). */
template <typename Op, typename A, typename B, bool Custom = needs_custom_kernel<A, B>> struct binary_apply
{
    /** @brief Stored element type the expression materializes. */
    using type = std::invoke_result_t<Op, A, B>;

    static constexpr type call(const A &a, const B &b)
    {
        return Op{}(a, b);
    }
};

/** @brief Builtin unary evaluation (stored value passes through). */
template <typename Op, typename A, bool Custom = scalar_traits<A>::is_custom> struct unary_apply
{
    /** @brief Stored element type the expression materializes. */
    using type = std::invoke_result_t<Op, A>;

    static constexpr type call(const A &a)
    {
        return Op{}(a);
    }
};

} // namespace np::detail::fixed

#endif // NP_DETAIL_SCALAR_BUILTIN_HPP
