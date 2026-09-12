/**
 * @file scalar_custom.hpp
 * @brief Internal scalar backend for the `dtype_storage` storage-classifier
 *        element types.
 *
 * Specializes `np::detail::fixed::scalar_traits<T>` for every
 * `dtype_storage::storage_classifier<D, ...>` instantiation: the classifier
 * stores its computation core (an arithmetic/string `value_type`) behind a
 * union, so `get` unwraps it, `make` re-wraps a computed result, and
 * `zero()/one()/truthy()` operate on the core. The array business logic
 * (ndarray_fixed.hpp and detail/expr.hpp) is identical to the builtin case.
 *
 * This is also the model for user-defined scalar types: specialize
 * `scalar_traits` for your own element type with the same five members and
 * the whole fixed-shape API (reductions, elementwise ops, broadcasting)
 * works unchanged.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DETAIL_SCALAR_CUSTOM_HPP
#define NP_DETAIL_SCALAR_CUSTOM_HPP

#include "../api_macros.hpp"

#include <type_traits>
#include <utility>

#include "../dtype.hpp"
#include "scalar_builtin.hpp"

namespace np::detail::fixed
{

// scalar_traits for the dtype_storage storage classifiers
/**
 * @brief Numeric branch: the classifier holds a contiguous scalar core.
 *
 * @tparam D  A np::dtype enumeration value.
 */
template <auto D> struct scalar_traits<dtype_storage::storage_classifier<D, false>>
{
    using classifier = dtype_storage::storage_classifier<D, false>;
    static constexpr bool is_custom = true;

    /** @brief Numeric core that reductions and kernels compute in. */
    using value_type = std::remove_cv_t<typename classifier::value_type>;

    static constexpr const value_type &get(const classifier &v) noexcept
    {
        return static_cast<const value_type &>(v.value());
    }
    static constexpr classifier make(const value_type &v) noexcept
    {
        return classifier{v};
    }
    static constexpr value_type zero() noexcept
    {
        return value_type{};
    }
    static constexpr value_type one() noexcept
    {
        return value_type{1};
    }
    static constexpr bool truthy(const classifier &v) noexcept
    {
        return static_cast<bool>(v.value());
    }
};

/**
 * @brief String branch: the classifier holds a text attribute.
 *
 * @tparam T  A np::dtype enumeration value (`string_` / `unicode_`).
 */
template <auto D> struct scalar_traits<dtype_storage::storage_classifier<D, true>>
{
    using classifier = dtype_storage::storage_classifier<D, true>;
    static constexpr bool is_custom = true;

    /** @brief Text core that reductions and kernels compute in. */
    using value_type = typename classifier::value_type;

    static constexpr const value_type &get(const classifier &v) noexcept
    {
        return v.value();
    }
    static constexpr classifier make(const value_type &v) noexcept
    {
        return classifier{v};
    }
    static constexpr value_type zero() noexcept
    {
        return value_type{};
    }
    static constexpr value_type one() noexcept
    {
        return value_type{};
    }
    static constexpr bool truthy(const classifier &v) noexcept
    {
        return !v.value().empty();
    }
};

// Elementwise kernel dispatch for the custom branch.
// Operands are unwrapped to their computation cores, the functor runs on
// the cores, and the result is re-wrapped into the custom element type.
// Comparisons/logical kernels that yield bool stay bool (NumPy semantics).
/** @brief Result element type of a binary expression over custom operands.
 *
 * When both operands are the same custom classifier, the classifier is
 * kept (NumPy keeps the array dtype). Mixed custom/builtin operands and
 * kernels that promote past the core (e.g. int sqrt -> double) resolve to
 * the plain promoted core type, exactly like the builtin branch.
 */
template <typename Op, typename A, typename B> struct custom_binary_value
{
    using CA = typename scalar_traits<A>::value_type;
    using CB = typename scalar_traits<B>::value_type;
    using Core = std::invoke_result_t<Op, CA, CB>;

    /** @brief Both handles are the same custom scalar type. */
    static constexpr bool both_same_custom =
        scalar_traits<A>::is_custom && scalar_traits<B>::is_custom && std::is_same_v<A, B>;

    /** @brief Keep the custom scalar when it can hold the core result. */
    static constexpr bool a_holds = scalar_traits<A>::is_custom && std::is_same_v<Core, CA>;
    static constexpr bool b_holds = scalar_traits<B>::is_custom && std::is_same_v<Core, CB>;

    using type =
        std::conditional_t<std::is_same_v<Core, bool>, bool,
                           std::conditional_t<both_same_custom && a_holds, A,
                                              std::conditional_t<a_holds, A, std::conditional_t<b_holds, B, Core>>>>;
};

/** @brief Result element type of a unary expression over a custom operand. */
template <typename Op, typename A> struct custom_unary_value
{
    using CA = typename scalar_traits<A>::value_type;
    using Core = std::invoke_result_t<Op, CA>;

    using type =
        std::conditional_t<std::is_same_v<Core, bool>, bool, std::conditional_t<std::is_same_v<Core, CA>, A, Core>>;
};

/** @brief Custom binary evaluation: unwrap, run the functor, re-wrap. */
template <typename Op, typename A, typename B> struct binary_apply<Op, A, B, true>
{
    using type = typename custom_binary_value<Op, A, B>::type;

    static constexpr type call(const A &a, const B &b)
    {
        if constexpr (std::is_same_v<type, bool>)
        {
            return Op{}(scalar_traits<A>::get(a), scalar_traits<B>::get(b));
        }
        else
        {
            return scalar_traits<type>::make(Op{}(scalar_traits<A>::get(a), scalar_traits<B>::get(b)));
        }
    }
};

/** @brief Custom unary evaluation: unwrap, run the re-wrap. */
template <typename Op, typename A> struct unary_apply<Op, A, true>
{
    using type = typename custom_unary_value<Op, A>::type;

    static constexpr type call(const A &a)
    {
        if constexpr (std::is_same_v<type, bool>)
        {
            return Op{}(scalar_traits<A>::get(a));
        }
        else
        {
            return scalar_traits<type>::make(Op{}(scalar_traits<A>::get(a)));
        }
    }
};

} // namespace np::detail::fixed

#endif // NP_DETAIL_SCALAR_CUSTOM_HPP
