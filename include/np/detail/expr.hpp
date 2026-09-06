/**
 * @file expr.hpp
 * @brief Compile-time expression templates for lazy, fused elementwise
 *        evaluation over fixed-shape sources.
 *
 * Design contract (module `np::detail::expr`):
 *  - Every expression node models `fixed_source`: it exposes
 *      `value_type`, `rank`, `static_shape` and a constexpr `operator[]`.
 *  - Broadcasting legality is decided at compile time by `merged<L, R>`;
 *      invalid combinations are rejected through `broadcast_valid` BEFORE
 *      the node is ever instantiated (SFINAE-friendly) and additionally
 *      guarded by `static_assert` for a readable diagnostic.
 *  - `eval()` materializes the whole tree into one pass over the merged
 *      (broadcast) index space; no intermediate temporaries are allocated.
 *  - All nodes are `constexpr`: a fully literal expression can be folded in
 *      a constant expression (see tests/test_constexpr.cpp).
 *
 * NumPy reference (ground truth for broadcasting semantics):
 *   numpy-reference/user/basics.broadcasting.html
 *   numpy-reference/reference/arrays.promotion.html
 */
#ifndef NP_DETAIL_EXPR_HPP
#define NP_DETAIL_EXPR_HPP

#include "../api_macros.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "scalar_builtin.hpp"

#include "scalar_custom.hpp"

namespace np
{

/**
 * @brief Fixed-shape array with compile-time extents (defined in
 *        ndarray_fixed.hpp; only declared here so that expression nodes
 *        can name the materialization target).
 */
template <typename T, int... Extents> class ndarrayf;

} // namespace np

namespace np::detail::expr
{

/**
 * @brief Compile-time shape descriptor: a pack of extents.
 */
template <int... E> struct shape_tag
{
    static constexpr std::size_t rank = sizeof...(E);
    static constexpr std::array<int, rank> arr{E...};
};

/**
 * @brief Static shape of any fixed-shape source (array, scalar wrapper or
 *        expression node).
 */
template <typename S> struct shape_tag_of;

template <typename T, int... E> struct shape_tag_of<ndarrayf<T, E...>>
{
    using type = shape_tag<E...>;
};

template <typename S> using shape_tag_t = typename shape_tag_of<std::remove_cv_t<std::remove_reference_t<S>>>::type;

/**
 * @brief Concept satisfied by any lazy/eager element source whose shape
 *        and element type are fully known at compile time.
 */
template <typename S>
concept fixed_source = requires(const S &s, std::size_t i) {
    typename S::value_type;
    S::rank;
    S::static_shape;
    { s[i] } -> std::convertible_to<typename S::value_type>;
};

/**
 * @brief Elementwise operation kernel (stateless, constexpr).
 */
template <typename Op> struct op_tag
{
};

// Compile-time broadcasting (numpy-reference/user/basics.broadcasting.html)
template <int A, int B> struct dim_merge
{
    static constexpr int value = (A == B) ? A : (A == 1) ? B : (B == 1) ? A : -1;
    static constexpr bool ok = value != -1;
};

/** @brief Prepend `Head` to a shape_tag. */
template <int Head, typename Tag> struct prepend;

template <int Head, int... E> struct prepend<Head, shape_tag<E...>>
{
    using type = shape_tag<Head, E...>;
};

/**
 * @brief Pad a shape_tag on the left with 1s up to `Target` rank
 *        (NumPy right-aligns shapes when broadcasting).
 */
template <typename Tag, int Cur, int Target> struct pad;

template <int... E, int Cur, int Target>
struct pad<shape_tag<E...>, Cur, Target> : pad<shape_tag<1, E...>, Cur + 1, Target>
{
};

template <int... E, int Target> struct pad<shape_tag<E...>, Target, Target>
{
    using type = shape_tag<E...>;
};

/** @brief Elementwise merge of two equal-rank padded shape_tags. */
template <typename TA, typename TB> struct zip_merge;

template <int... A, int... B> struct zip_merge<shape_tag<A...>, shape_tag<B...>>
{
    using type = shape_tag<dim_merge<A, B>::value...>;
    static constexpr bool valid = (dim_merge<A, B>::ok && ...);
};

/**
 * @brief Broadcast-merge of two sources' static shapes.
 *        `type` is the merged shape_tag; `valid` is false when the
 *        shapes cannot be broadcast together.
 */
template <typename LA, typename LB> struct merged
{
    using A = shape_tag_t<LA>;
    using B = shape_tag_t<LB>;
    static constexpr std::size_t ra = A::rank;
    static constexpr std::size_t rb = B::rank;
    static constexpr std::size_t rank = ra > rb ? ra : rb;
    using PA = typename pad<A, static_cast<int>(ra), static_cast<int>(rank)>::type;
    using PB = typename pad<B, static_cast<int>(rb), static_cast<int>(rank)>::type;
    using type = typename zip_merge<PA, PB>::type;
    static constexpr bool valid = zip_merge<PA, PB>::valid;
};

/** @brief True when L and R can be broadcast together (SFINAE helper). */
template <typename L, typename R> inline constexpr bool broadcast_valid = merged<L, R>::valid;

// Index arithmetic (constexpr)
/** @brief Row-major unflattening of a flat index. */
template <std::size_t R>
constexpr std::array<std::size_t, R> unflatten(std::size_t flat, const std::array<int, R> &shape)
{
    static_assert(R > 0, "unflatten: rank must be positive");
    static_assert(std::is_unsigned_v<std::size_t>, "unflatten: size_t must be unsigned");
    std::array<std::size_t, R> coords{};
    for (std::size_t d = R; d-- > 0;)
    {
        coords[d] = flat % static_cast<std::size_t>(shape[d]);
        flat /= static_cast<std::size_t>(shape[d]);
    }
    return coords;
}

/** @brief Row-major flattening of multi-dimensional coordinates. */
template <std::size_t R>
constexpr std::size_t flatten(const std::array<int, R> &shape, const std::array<std::size_t, R> &coords)
{
    static_assert(R > 0, "flatten: rank must be positive");
    static_assert(std::is_same_v<decltype(shape), const std::array<int, R> &>, "flatten: shape type check");
    std::size_t idx = 0;
    for (std::size_t d = 0; d < R; ++d)
    {
        static_assert(std::is_unsigned_v<std::size_t>, "flatten: coord must be unsigned");
        idx = idx * static_cast<std::size_t>(shape[d]) + coords[d];
    }
    return idx;
}

/**
 * @brief Read one element of `src` under the broadcast coordinates of a
 *        merged space of rank R: right-aligned, extent-1 dims read 0.
 */
template <fixed_source S, std::size_t R>
constexpr typename S::value_type gather(const S &src, const std::array<std::size_t, R> &coords)
{
    constexpr std::size_t r = S::rank;
    constexpr std::array<int, r> shp = S::static_shape;
    std::size_t idx = 0;
    for (std::size_t j = 0; j < r; ++j)
    {
        std::size_t dim = coords[R - r + j];
        if (shp[j] == 1)
        {
            dim = 0;
        }
        idx = idx * static_cast<std::size_t>(shp[j]) + dim;
    }
    return src[idx];
}

// Scalar wrapper: a rank-0 source broadcastable to anything.
template <typename T> struct scalar_expr
{
    using value_type = T;
    static constexpr std::size_t rank = 0;
    static constexpr std::array<int, 0> static_shape{};
    T value;

    constexpr explicit scalar_expr(const T &v) : value(v)
    {
    }
    constexpr T operator[](std::size_t) const
    {
        return value;
    }
};

template <typename T> struct shape_tag_of<scalar_expr<T>>
{
    using type = shape_tag<>;
};

// Nodes
/**
 * @brief Lazy elementwise binary operation over two broadcast sources.
 */
template <typename Op, typename L, typename R> class binary_expr
{
    static_assert(merged<L, R>::valid, "np: incompatible shapes for elementwise operation "
                                       "(not broadcastable; see "
                                       "numpy-reference/user/basics.broadcasting.html)");

  public:
    using lhs_type = L;
    using rhs_type = R;
    using lhs_element = typename L::value_type;
    using rhs_element = typename R::value_type;
    static constexpr bool is_custom =
        detail::fixed::scalar_traits<lhs_element>::is_custom || detail::fixed::scalar_traits<rhs_element>::is_custom;
    using value_type = typename detail::fixed::binary_apply<Op, lhs_element, rhs_element, is_custom>::type;
    static constexpr std::size_t rank = merged<L, R>::rank;
    using tag = typename merged<L, R>::type;
    static constexpr std::array<int, rank> static_shape = tag::arr;

    L lhs;
    R rhs;

    constexpr binary_expr(const L &l, const R &r) : lhs(l), rhs(r)
    {
    }

    /** @brief Element under a flat (row-major) broadcast index. */
    constexpr value_type operator[](std::size_t i) const
    {
        const auto coords = unflatten(i, static_shape);
        return detail::fixed::binary_apply<Op, lhs_element, rhs_element, is_custom>::call(gather(lhs, coords),
                                                                                          gather(rhs, coords));
    }

    /** @brief Element under explicit multi-dimensional coordinates. */
    template <typename... Idx>
        requires(sizeof...(Idx) == rank && rank >= 2)
    constexpr value_type operator()(Idx... idx) const
    {
        const std::array<std::size_t, rank> coords{static_cast<std::size_t>(idx)...};
        return detail::fixed::binary_apply<Op, lhs_element, rhs_element, is_custom>::call(gather(lhs, coords),
                                                                                          gather(rhs, coords));
    }

    /** @brief Single fused pass into a fresh fixed-shape array. */
    template <typename V = value_type, int... E> constexpr ndarrayf<V, E...> eval_impl(shape_tag<E...>) const
    {
        ndarrayf<V, E...> out{};
        for (std::size_t i = 0; i < out.size_v; ++i)
        {
            out[i] = (*this)[i];
        }
        return out;
    }

    constexpr auto eval() const
    {
        return eval_impl(tag{});
    }

    using eval_type = decltype(std::declval<binary_expr>().eval_impl(tag{}));

    constexpr operator eval_type() const
    {
        return eval();
    }
};

/**
 * @brief Lazy elementwise unary operation over one source.
 */
template <typename Op, typename S> class unary_expr
{
  public:
    using source_type = S;
    using element_type = typename S::value_type;
    static constexpr bool is_custom = detail::fixed::scalar_traits<element_type>::is_custom;
    using value_type = typename detail::fixed::unary_apply<Op, element_type, is_custom>::type;
    static constexpr std::size_t rank = S::rank;
    using tag = shape_tag_t<S>;
    static constexpr std::array<int, rank> static_shape = tag::arr;

    S src;

    constexpr unary_expr(const S &s) : src(s)
    {
    }

    constexpr value_type operator[](std::size_t i) const
    {
        return detail::fixed::unary_apply<Op, element_type, is_custom>::call(src[i]);
    }

    template <typename... Idx>
        requires(sizeof...(Idx) == rank && rank >= 2)
    constexpr value_type operator()(Idx... idx) const
    {
        const std::array<std::size_t, rank> coords{static_cast<std::size_t>(idx)...};
        return detail::fixed::unary_apply<Op, element_type, is_custom>::call(src[flatten(static_shape, coords)]);
    }

    template <typename V = value_type, int... E> constexpr ndarrayf<V, E...> eval_impl(shape_tag<E...>) const
    {
        ndarrayf<V, E...> out{};
        for (std::size_t i = 0; i < out.size_v; ++i)
        {
            out[i] = (*this)[i];
        }
        return out;
    }

    constexpr auto eval() const
    {
        return eval_impl(tag{});
    }

    using eval_type = decltype(std::declval<unary_expr>().eval_impl(tag{}));

    constexpr operator eval_type() const
    {
        return eval();
    }
};

// Shape metaprogramming helpers shared with ndarray_fixed.hpp
/** @brief Static shape of a lazily evaluated expression node. */
template <typename Op, typename L, typename R> struct shape_tag_of<binary_expr<Op, L, R>>
{
    using type = typename binary_expr<Op, L, R>::tag;
};

template <typename Op, typename S> struct shape_tag_of<unary_expr<Op, S>>
{
    using type = shape_tag_t<S>;
};

/** @brief Reverse a shape_tag (used by ndarrayf::transpose). */
template <typename Acc, typename Tag> struct rev_impl;

template <int... Acc> struct rev_impl<shape_tag<Acc...>, shape_tag<>>
{
    using type = shape_tag<Acc...>;
};

template <int... Acc, int Head, int... Tail>
struct rev_impl<shape_tag<Acc...>, shape_tag<Head, Tail...>> : rev_impl<shape_tag<Head, Acc...>, shape_tag<Tail...>>
{
};

template <typename Tag> struct reverse : rev_impl<shape_tag<>, Tag>
{
};

/** @brief Remove the extent at zero-based position `Axis`. */
template <int Axis, int Cur, typename Tag> struct remove_at;

template <int Axis, int Cur> struct remove_at<Axis, Cur, shape_tag<>>
{
    using type = shape_tag<>;
};

template <int Axis, int Cur, int Head, int... Tail> struct remove_at<Axis, Cur, shape_tag<Head, Tail...>>
{
    using type =
        std::conditional_t<Cur == Axis, shape_tag<Tail...>,
                           typename prepend<Head, typename remove_at<Axis, Cur + 1, shape_tag<Tail...>>::type>::type>;
};

/** @brief Drop every extent equal to 1 (used by ndarrayf::squeeze). */
template <typename Tag> struct squeeze_tag;

template <> struct squeeze_tag<shape_tag<>>
{
    using type = shape_tag<>;
};

template <int Head, int... Tail> struct squeeze_tag<shape_tag<Head, Tail...>>
{
    using rest = typename squeeze_tag<shape_tag<Tail...>>::type;
    using type = std::conditional_t<Head == 1, rest, typename prepend<Head, rest>::type>;
};

/**
 * @brief Insert extent `Value` at zero-based position `Axis` (used by
 *        ndarrayf::expand_dims and np::stack).
 */
template <int Value, int Axis, int Cur, typename Tag> struct insert;

template <int Value, int Axis, int Cur> struct insert<Value, Axis, Cur, shape_tag<>>
{
    using type = shape_tag<Value>;
};

template <int Value, int Axis, int Cur, int Head, int... Tail> struct insert<Value, Axis, Cur, shape_tag<Head, Tail...>>
{
    using rest = typename insert<Value, Axis, Cur + 1, shape_tag<Tail...>>::type;
    using type = std::conditional_t<Cur == Axis, shape_tag<Value, Head, Tail...>, typename prepend<Head, rest>::type>;
};

/** @brief First extent of a shape_tag (rank must be >= 1). */
template <typename Tag> struct head;

template <int Head, int... Tail> struct head<shape_tag<Head, Tail...>>
{
    static constexpr int value = Head;
};

/** @brief shape_tag without its first extent. */
template <typename Tag> struct tail;

template <int Head, int... Tail> struct tail<shape_tag<Head, Tail...>>
{
    using type = shape_tag<Tail...>;
};

/** @brief True when two shape_tags are identical. */
template <typename A, typename B> struct same_tag : std::false_type
{
};

template <int... A> struct same_tag<shape_tag<A...>, shape_tag<A...>> : std::true_type
{
};

} // namespace np::detail::expr

#endif // NP_DETAIL_EXPR_HPP
