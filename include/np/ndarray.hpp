/**
 * @file ndarray.hpp
 * @brief The np::ndarray class -- a NumPy-compatible multidimensional array.
 *
 * Features:
 *  - N-dimensional storage with C-order (row-major) strides
 *  - Chained subscript access via stack-based proxies (arr[i][j][k])
 *  - Views (transpose, swapaxes, squeeze, reshape) that share storage
 *  - Reductions with optional axis (sum, mean, var, std, min, max, all, any)
 *  - Sorting / indexing helpers (sort, argsort, searchsorted, take, put)
 *  - Element-wise arithmetic with NumPy-style broadcasting
 *  - Logical iterators that honor strides (correct for views)
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_NDARRAY_HPP
#define NP_NDARRAY_HPP

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "api_macros.hpp"
#include "detail/proxy.hpp"
#include "dtype.hpp"
#include "exceptions.hpp"
#include "pqc.hpp"
#include "simd.hpp"

#ifdef NP_USE_THREADING
#include "threadpool.hpp"
#endif

namespace np
{
namespace matrix
{
/**
 * @brief Memory layout order.
 */
enum class Order : std::uint8_t
{
    C, // Row-major (C style)
    F  // Column-major (Fortran style)
};
} // namespace matrix

namespace detail
{

#if __cpp_initializer_lists >= 200806L
// True when _ElementType is a std::initializer_list instantiation.
// Most of the time, when _ElementType will be range, the user automatically
// uses std::initializer_list without noticing the default type of braces
// initializer list in __cplusplus >= 11

template <typename _ElementType> NP_INTERNAL struct _Np_is_init_list : std::false_type
{
};

template <typename _ElementType>
NP_INTERNAL struct _Np_is_init_list<std::initializer_list<_ElementType>> : std::true_type
{
};
#endif

// The real element type of a (possibly complex) scalar type.
// For std::complex<_ElementType> this is _ElementType,
// for everything else the type itself (used by real()/imag()).
template <typename _ElementType> struct _Np_real_of
{
    using type = _ElementType;
};
template <typename _ElementType> struct _Np_real_of<std::complex<_ElementType>>
{
    using type = _ElementType;
};

/**
 * @brief NaN test without tautological-compare warnings.
 *
 * `v != v` on an integer type trips `-Wtautological-compare` under
 * `-Werror`; this helper keeps NaN-propagation branches warning-free for
 * all dtypes (non-floating types simply never report NaN).
 */
template <typename V> NP_NODISCARD inline bool isnan_val(const V &v) noexcept
{
    if constexpr (std::is_floating_point_v<V>)
    {
        return v != v;
    }
    else
    {
        return false;
    }
}

/**
 * @brief Computation type for floored `%` / `//`.
 *
 * `std::common_type_t` alone is wrong for mixed-sign integers
 * (`common_type<int, unsigned>` is unsigned, so `-4` wraps before the
 * floored adjustment can see it). Mixed-sign pairs smaller than 64 bit
 * compute exactly in `int64_t`; pairs involving 64-bit types follow NumPy
 * and compute in `double`. Same-sign pairs keep `common_type`.
 */
/**
 * NOTE: the primary has NO `type` member on purpose. `std::common_type_t`
 * of unrelated types (e.g. an array type deduced into a scalar overload
 * during overload resolution) must fail as "no member named type" in the
 * immediate context (SFINAE-friendly, like `std::common_type` itself) —
 * never as a hard error inside an eagerly-instantiated branch. An eager
 * nested-`conditional_t` formulation broke every array/scalar operator pair
 * sharing a name (verified: `a / a` selected the scalar overload's return
 * type and hard-errored instead of SFINAE-discarding it).
 */
template <typename A, typename B, typename = void> struct floored_common
{
};
template <typename A, typename B>
struct floored_common<
    A, B,
    std::enable_if_t<std::is_integral_v<A> && std::is_integral_v<B> && (std::is_signed_v<A> != std::is_signed_v<B>) &&
                     (sizeof(A) < 8 && sizeof(B) < 8)>>
{
    using type = std::int64_t; // narrow mixed-sign: exact in int64
};
template <typename A, typename B>
struct floored_common<
    A, B,
    std::enable_if_t<std::is_integral_v<A> && std::is_integral_v<B> && (std::is_signed_v<A> != std::is_signed_v<B>) &&
                     (sizeof(A) >= 8 || sizeof(B) >= 8)>>
{
    using type = double; // wide mixed-sign: NumPy promotes like float64
};
template <typename A, typename B>
struct floored_common<
    A, B,
    std::enable_if_t<!(std::is_integral_v<A> && std::is_integral_v<B> && (std::is_signed_v<A> != std::is_signed_v<B>)),
                     std::void_t<std::common_type_t<A, B>>>>
{
    using type = std::common_type_t<A, B>; // non-integral or same-sign
};
template <typename A, typename B> using floored_common_t = typename floored_common<A, B>::type;

/**
 * @brief NumPy `%` (mod): remainder with the sign of the divisor
 *        (complementary to floor division). C's `%` truncates toward
 *        zero, so this adjusts when the signs differ.
 * @throws std::domain_error on integral division by zero (NumPy raises
 *         `ZeroDivisionError`; floating point follows `fmod`, i.e. NaN).
 */
template <typename A, typename B> inline auto floored_mod(A a, B b) -> std::common_type_t<A, B>
{
    using R = std::common_type_t<A, B>;
    using P = floored_common_t<A, B>;
    const P x = static_cast<P>(a);
    const P y = static_cast<P>(b);
    if constexpr (std::is_integral_v<P>)
    {
        if (y == P{0})
        {
            throw std::domain_error("floored_mod: integer division by zero");
        }
    }
    P m;
    if constexpr (std::is_floating_point_v<P>)
    {
        m = std::fmod(x, y);
    }
    else
    {
        m = x % y;
    }
    if (m != P{0} && ((m < P{0}) != (y < P{0})))
    {
        m += y;
    }
    return static_cast<R>(m);
}

/**
 * @brief NumPy `//` (floor_divide): largest integer <= x / y, and the
 *        floor for floating point (y = floor(x1 / x2)).
 * @throws std::domain_error on integral division by zero (NumPy raises
 *         `ZeroDivisionError`; floating point follows IEEE, i.e. inf/NaN).
 */
template <typename A, typename B> inline auto floored_div(A a, B b) -> std::common_type_t<A, B>
{
    using R = std::common_type_t<A, B>;
    using P = floored_common_t<A, B>;
    const P x = static_cast<P>(a);
    const P y = static_cast<P>(b);
    if constexpr (std::is_floating_point_v<P>)
    {
        return static_cast<R>(std::floor(x / y));
    }
    else
    {
        if (y == P{0})
        {
            throw std::domain_error("floored_div: integer division by zero");
        }
        P q = x / y;
        if ((x % y) != P{0} && ((x < P{0}) != (y < P{0})))
        {
            q -= P{1};
        }
        return static_cast<R>(q);
    }
}

/**
 * @brief NumPy-ish `**` (power): binary exponentiation for non-negative
 *        integral exponents, `std::pow` otherwise.
 *
 * Negative exponents with integral result truncate toward zero (e.g.
 * `2**-1 == 0`), pinned by test_math ("int truncation"). This diverges
 * from NumPy (which raises) but is the established contract here; use a
 * floating-point base for fractional results.
 */
template <typename A, typename B> inline auto power_elem(A a, B b) -> std::common_type_t<A, B>
{
    using R = std::common_type_t<A, B>;
    if constexpr (std::is_integral_v<R> && std::is_integral_v<B>)
    {
        // (Unsigned B can never be negative; guard avoids -Wtype-limits.)
        if constexpr (std::is_signed_v<B>)
        {
            if (b < B{0})
            {
                return static_cast<R>(std::pow(static_cast<double>(a), static_cast<double>(b)));
            }
        }
        // Binary exponentiation: O(log e) instead of O(e).
        R result = R{1};
        R base = static_cast<R>(a);
        B e = b;
        while (e > B{0})
        {
            if ((e & B{1}) != B{0})
            {
                result *= base;
            }
            base *= base;
            e >>= B{1};
        }
        return result;
    }
    else
    {
        // No forced double round-trip: std::pow overloads handle float
        // and std::complex directly.
        return static_cast<R>(std::pow(a, b));
    }
}

/**
 * @brief Validated shift count for `<<` / `>>`.
 *
 * A negative count or a count >= the value width is UB in C++; NumPy
 * raises on out-of-range counts instead of silently wrapping.
 * @throws std::out_of_range if the count is negative or too large.
 */
template <typename A, typename B> inline B checked_shift_count(A /*value*/, B count)
{
    static_assert(std::is_integral_v<A> && std::is_integral_v<B>, "shifts require integral operands");
    using W = std::make_unsigned_t<B>;
    if constexpr (std::is_signed_v<B>)
    {
        if (count < B{0})
        {
            throw std::out_of_range("negative shift count");
        }
    }
    if (static_cast<W>(count) >= static_cast<W>(sizeof(A) * CHAR_BIT))
    {
        throw std::out_of_range("shift count exceeds value width");
    }
    return count;
}

/**
 * @brief Division result type following NumPy `true_divide` semantics:
 *        integral / integral promotes to `double`; anything else keeps
 *        `std::common_type_t` (float and complex division unchanged).
 */
template <typename T, typename U, typename = void> struct div_result
{
};
template <typename T, typename U>
struct div_result<T, U, std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
{
    using type = double;
};
template <typename T, typename U>
struct div_result<
    T, U, std::enable_if_t<!(std::is_integral_v<T> && std::is_integral_v<U>), std::void_t<std::common_type_t<T, U>>>>
{
    using type = std::common_type_t<T, U>;
};
template <typename T, typename U> using div_result_t = typename div_result<T, U>::type;

/**
 * @brief Checked `size_t` → `int` cast for shape storage.
 * @throws std::length_error if the value does not fit in an `int`.
 */
inline int checked_int(std::size_t v)
{
    if (v > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    {
        throw std::length_error("dimension exceeds int range");
    }
    return static_cast<int>(v);
}

#ifdef NP_USE_THREADING
constexpr std::size_t kParallelThreshold = 10000;

template <typename Func> inline void maybe_parallel_for(std::size_t begin, std::size_t end, Func &&f)
{
    const std::size_t n = (end > begin) ? end - begin : 0;
    if (n > kParallelThreshold)
    {
        ::np::ThreadPool::global().parallel_for(begin, end, std::forward<Func>(f));
    }
    else
    {
        for (std::size_t i = begin; i < end; ++i)
        {
            f(i);
        }
    }
}
#endif

} // namespace detail

namespace detail
{

// ── ND initializer_list proxy helpers (arbitrary depth) ─────────────────
template <typename T> struct is_init_list : std::false_type
{
};
template <typename U> struct is_init_list<std::initializer_list<U>> : std::true_type
{
};
template <typename T> constexpr bool is_init_list_v = is_init_list<T>::value;

template <typename T> struct init_depth
{
    static constexpr std::size_t value = 0;
};
template <typename U> struct init_depth<std::initializer_list<U>>
{
    static constexpr std::size_t value = 1 + init_depth<U>::value;
};
template <typename T> constexpr std::size_t init_depth_v = init_depth<T>::value;

template <typename T> struct init_value_type
{
    using type = T;
};
template <typename U> struct init_value_type<std::initializer_list<U>>
{
    using type = typename init_value_type<U>::type;
};
template <typename T> using init_value_type_t = typename init_value_type<T>::type;

template <typename List> inline std::vector<int> nested_shape(const List &lst)
{
    std::vector<int> s;
    s.push_back(static_cast<int>(lst.size()));
    if constexpr (is_init_list_v<typename List::value_type>)
    {
        if (lst.size() > 0)
        {
            auto sub = nested_shape(*lst.begin());
            for (auto &sub_lst : lst)
            {
                auto cur = nested_shape(sub_lst);
                if (cur != sub)
                {
                    throw std::invalid_argument("ragged nested initializer list");
                }
            }
            s.insert(s.end(), sub.begin(), sub.end());
        }
    }
    return s;
}

template <typename List, typename T> inline void nested_flatten(const List &lst, std::vector<T> &out)
{
    if constexpr (is_init_list_v<typename List::value_type>)
    {
        for (auto &sub : lst)
        {
            nested_flatten(sub, out);
        }
    }
    else
    {
        for (auto &v : lst)
        {
            out.push_back(static_cast<T>(v));
        }
    }
}

// NDProxy for arbitrary-depth braced-init (proxy pattern)
// Suppress -Wbraced-scalar-init only around this proxy (e.g.
// {{{1},{2},{3}},{{1},{2},{3}}} shape 2x3x1); scoped push/pop so user
// code including this header keeps the warning enabled.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbraced-scalar-init"
#endif
template <typename T> struct NDProxy
{
    std::vector<NDProxy> children;
    T value{};
    bool is_leaf = false;
    NDProxy() = default;
    template <typename U>
        requires std::is_convertible_v<U, T>
    NDProxy(U v) : value(static_cast<T>(v)), is_leaf(true)
    {
    }
    NDProxy(std::initializer_list<NDProxy> lst) : children(lst), is_leaf(false)
    {
    }
};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

template <typename T> inline std::vector<int> proxy_shape(const NDProxy<T> &p)
{
    if (p.is_leaf)
    {
        return {};
    }
    std::vector<int> s;
    s.push_back(static_cast<int>(p.children.size()));
    if (p.children.empty())
    {
        return s; // empty branch node (e.g. from `{{}}`): shape is [0]
    }
    if (!p.children[0].is_leaf)
    {
        auto sub = proxy_shape(p.children[0]);
        for (auto &c : p.children)
        {
            auto cur = proxy_shape(c);
            if (cur != sub)
            {
                throw std::invalid_argument("ragged nested initializer list");
            }
        }
        s.insert(s.end(), sub.begin(), sub.end());
    }
    else
    {
        // Check all children are leaf for 1-D branch; for deeper, leaf branch is 1-D
        for (auto &c : p.children)
        {
            if (c.is_leaf != p.children[0].is_leaf)
            {
                throw std::invalid_argument("ragged nested initializer list");
            }
        }
    }
    return s;
}

template <typename T> inline std::vector<int> proxy_shape_list(const std::initializer_list<NDProxy<T>> &lst)
{
    std::vector<int> s;
    s.push_back(static_cast<int>(lst.size()));
    if (lst.size() == 0)
    {
        return s;
    }
    auto sub = proxy_shape(*lst.begin());
    for (auto &p : lst)
    {
        auto cur = proxy_shape(p);
        if (cur != sub)
        {
            throw std::invalid_argument("ragged nested initializer list");
        }
    }
    s.insert(s.end(), sub.begin(), sub.end());
    return s;
}

template <typename T> inline void proxy_flatten(const NDProxy<T> &p, std::vector<T> &out)
{
    if (p.is_leaf)
    {
        out.push_back(p.value);
    }
    else
    {
        for (auto &c : p.children)
        {
            proxy_flatten(c, out);
        }
    }
}

} // namespace detail

/**
 * @brief Result type of mean/var/std reductions.
 *
 * Floating and complex inputs keep their type; integer and boolean
 * inputs promote to double (NumPy semantics).
 */
template <typename T> struct _mean_type
{
    using type = std::conditional_t<std::is_floating_point_v<T> || detail::is_complex_v<T>, T, double>;
};

/**
 * @brief Result type of var/std reductions.
 *
 * Like `_mean_type`, except complex inputs yield real floating output
 * (NumPy: `var(complex128) -> float64`), since variance is mean squared
 * magnitude. Floating and other inputs behave exactly like `_mean_type`.
 */
template <typename T> struct _var_type
{
    using real = typename detail::_Np_real_of<T>::type;
    using type = typename _mean_type<real>::type;
};

template <typename T> class Matrix;

// Logical iterator (stride-aware, correct for views)
/**
 * @brief Forward iterator visiting array elements in logical (C) order.
 *
 * Iterates over the logical (row-major) element order, correctly
 * handling views with non-trivial strides.
 *
 * @tparam T Element type; instantiate with `const T` for read-only access.
 * @complexity O(1) per increment, O(n) total for a full traversal.
 */
template <typename T> class ndarray_iterator
{
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::remove_const_t<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

    /**
     * @brief Constructs an iterator.
     * @param base Pointer to the start of the data buffer.
     * @param shape Logical shape of the array.
     * @param strides Stride vector in elements.
     * @param at_end If true, constructs the end sentinel.
     */
    ndarray_iterator(T *base, std::vector<std::size_t> shape, std::vector<std::size_t> strides, bool at_end)
        : base_(base), shape_(std::move(shape)), strides_(std::move(strides)), idx_(shape_.size(), 0), done_(at_end)
    {
    }

    /**
     * @brief Dereference: returns the element at the current logical position.
     * @return Reference to the element.
     */
    NP_NODISCARD reference operator*() const
    {
        return base_[detail::flat_index(idx_, strides_, 0)];
    }

    /**
     * @brief Member access: returns a pointer to the current element.
     * @return Pointer to the element.
     */
    NP_NODISCARD pointer operator->() const
    {
        return &base_[detail::flat_index(idx_, strides_, 0)];
    }

    /**
     * @brief Pre-increment: advances to the next logical element.
     * @return Reference to this iterator.
     * @complexity O(1) amortised.
     */
    ndarray_iterator &operator++()
    {
        _advance();
        return *this;
    }

    /**
     * @brief Post-increment: advances to the next logical element.
     * @return Copy of the iterator before incrementing.
     * @complexity O(1) amortised.
     */
    ndarray_iterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    /**
     * @brief Equality comparison.
     * @param o Other iterator.
     * @return true if both iterators refer to the same position.
     */
    NP_NODISCARD bool operator==(const ndarray_iterator &o) const noexcept
    {
        if (base_ != o.base_ || done_ != o.done_)
        {
            return false;
        }
        return done_ || idx_ == o.idx_;
    }

    /**
     * @brief Inequality comparison.
     * @param o Other iterator.
     * @return true if the iterators refer to different positions.
     */
    NP_NODISCARD bool operator!=(const ndarray_iterator &o) const noexcept
    {
        return !(*this == o);
    }

  private:
    /**
     * @brief Advances the multi-index by one position in C order.
     *
     * Carries across dimensions like an odometer. Sets `done_` when
     * the index overflows all dimensions.
     * @complexity O(ndim) worst case, O(1) amortised.
     */
    void _advance() noexcept
    {
        if (shape_.empty())
        {
            done_ = true;
            return;
        }
        for (std::size_t d = shape_.size(); d-- > 0;)
        {
            if (++idx_[d] < shape_[d])
            {
                return;
            }
            idx_[d] = 0;
        }
        done_ = true;
    }

    T *base_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<std::size_t> idx_;
    bool done_;
};

/**
 * @brief Forward iterator for `bool` arrays (bit-packed storage).
 *
 * `std::vector<bool>` has no addressable elements, so the primary template
 * (raw `T*` base) cannot work: `ndarray<bool>::_raw_ptr()` is null by
 * design. This specialization indexes the bit-packed vector directly.
 * There is intentionally no `operator->` — packed bits are not addressable.
 */
template <> class ndarray_iterator<bool>
{
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = bool;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = std::vector<bool>::reference;

    ndarray_iterator(std::vector<bool> *vec, std::vector<std::size_t> shape, std::vector<std::size_t> strides,
                     std::size_t offset, bool at_end)
        : vec_(vec), shape_(std::move(shape)), strides_(std::move(strides)), idx_(shape_.size(), 0), offset_(offset),
          done_(at_end)
    {
    }

    NP_NODISCARD reference operator*() const
    {
        return (*vec_)[offset_ + detail::flat_index(idx_, strides_, 0)];
    }

    ndarray_iterator &operator++()
    {
        _advance();
        return *this;
    }

    ndarray_iterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    NP_NODISCARD bool operator==(const ndarray_iterator &o) const noexcept
    {
        if (vec_ != o.vec_ || done_ != o.done_)
        {
            return false;
        }
        return done_ || idx_ == o.idx_;
    }

    NP_NODISCARD bool operator!=(const ndarray_iterator &o) const noexcept
    {
        return !(*this == o);
    }

  private:
    void _advance() noexcept
    {
        if (shape_.empty())
        {
            done_ = true;
            return;
        }
        for (std::size_t d = shape_.size(); d-- > 0;)
        {
            if (++idx_[d] < shape_[d])
            {
                return;
            }
            idx_[d] = 0;
        }
        done_ = true;
    }

    std::vector<bool> *vec_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<std::size_t> idx_;
    std::size_t offset_;
    bool done_;
};

/** @brief Read-only variant of the `bool` iterator (yields `bool` by value). */
template <> class ndarray_iterator<const bool>
{
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = bool;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = bool;

    ndarray_iterator(const std::vector<bool> *vec, std::vector<std::size_t> shape, std::vector<std::size_t> strides,
                     std::size_t offset, bool at_end)
        : vec_(vec), shape_(std::move(shape)), strides_(std::move(strides)), idx_(shape_.size(), 0), offset_(offset),
          done_(at_end)
    {
    }

    NP_NODISCARD reference operator*() const
    {
        return (*vec_)[offset_ + detail::flat_index(idx_, strides_, 0)];
    }

    ndarray_iterator &operator++()
    {
        _advance();
        return *this;
    }

    ndarray_iterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    NP_NODISCARD bool operator==(const ndarray_iterator &o) const noexcept
    {
        if (vec_ != o.vec_ || done_ != o.done_)
        {
            return false;
        }
        return done_ || idx_ == o.idx_;
    }

    NP_NODISCARD bool operator!=(const ndarray_iterator &o) const noexcept
    {
        return !(*this == o);
    }

  private:
    void _advance() noexcept
    {
        if (shape_.empty())
        {
            done_ = true;
            return;
        }
        for (std::size_t d = shape_.size(); d-- > 0;)
        {
            if (++idx_[d] < shape_[d])
            {
                return;
            }
            idx_[d] = 0;
        }
        done_ = true;
    }

    const std::vector<bool> *vec_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<std::size_t> idx_;
    std::size_t offset_;
    bool done_;
};

// ndarray
/**
 * @brief A NumPy-style multidimensional array container.
 *
 * Stores elements in a shared `std::vector<T>` buffer, enabling
 * zero-copy views (transpose, swapaxes, squeeze, reshape) that
 * share storage with the parent array.
 *
 * @tparam T Element type (numeric or `std::complex`).
 * @note Memory ownership is reference-counted via `std::shared_ptr`.
 *       Views set `is_view_ = true` and hold a pointer to the parent's
 *       buffer; `base()` returns the parent's `shared_ptr` raw pointer.
 * @note Strides are always in *elements*, not bytes.
 */
template <typename T = double> class ndarray
{
  public:
    using value_type = typename dtype_tag_to_type<T>::type;
    using size_type = std::size_t;
    using iterator = ndarray_iterator<value_type>;
    using const_iterator = ndarray_iterator<const value_type>;
    /**
     * @brief Reference type returned by non-const element accessors.
     *
     * `std::vector<bool>` is specialised, so its element access yields a
     * proxy type rather than `bool&`; this alias keeps the `ndarray` API
     * uniform for `bool` arrays.
     */
    using reference = std::conditional_t<std::is_same_v<value_type, bool>, std::vector<bool>::reference, value_type &>;
    /**
     * @brief Const element reference type.
     *
     * `std::vector<bool>` has no addressable `const bool&` (its accessors
     * yield prvalue proxies), so const accessors return `bool` by value
     * for `bool` arrays and `const value_type&` otherwise. Returning
     * `const bool&` would dangle.
     */
    using const_reference = std::conditional_t<std::is_same_v<value_type, bool>, bool, const value_type &>;

    // Attributes (mirror ndarray.shape / strides / dtype / order)
    std::vector<int> shape;                 ///< Dimensions of the array
    std::vector<std::size_t> strides;       ///< Strides in elements
    np::dtype type = dtype::void_;          ///< Data type
    matrix::Order order = matrix::Order::C; ///< Memory layout
    std::size_t offset = 0;                 ///< Element offset into storage (views)

    // Construction
    /**
     * @brief Default constructor: empty 0-dimensional array.
     */
    ndarray() = default;

    /**
     * @brief Constructs an array of the given shape, filled with `fill`.
     *
     * @param shape Dimensions of the new array.
     * @param type Data type (default `dtype::void_`, deduced from `T`).
     * @param fill Initial value for every element.
     * @post `this->shape == shape` and `this->size() == product(shape)`.
     */
    explicit ndarray(const std::vector<int> &shape, np::dtype type = dtype::void_,
                     const value_type &fill = typename ndarray<T>::value_type{});

    /**
     * @brief Builds an array from an owned data buffer.
     *
     * Provided as a static factory (instead of a constructor) so that
     * nested-brace construction like `ndarray<int> a{{1,2},{3,4}}`
     * unambiguously selects the nested initializer-list constructor.
     *
     * @param shape Dimensions of the new array.
     * @param data Owned element data; moved into the array's buffer.
     * @return New `ndarray` with the given shape and data.
     * @throws std::invalid_argument if `data.size() != product(shape)`.
     */
    static ndarray from_data(const std::vector<int> &shape, std::vector<value_type> data);

    /**
     * @brief 1D construction from a flat initializer list.
     * @param list Flat list of elements; length becomes `shape[0]`.
     */

    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U, value_type> ||
                                                      std::is_same_v<U, std::initializer_list<double>> ||
                                                      std::is_same_v<U, std::initializer_list<int>>>>
    ndarray(std::initializer_list<U> list);

    /**
     * @brief 2D construction from nested initializer lists, e.g.
     *        `ndarray<int> a{{1, 2}, {3, 4}}`.
     * @tparam U Element type of the inner lists (deduced).
     * @param rows Nested initializer list; all rows must have the same
     *        length, otherwise `std::invalid_argument` is thrown.
     * @throws std::invalid_argument on ragged (inconsistent) rows.
     */
    template <typename U> ndarray(std::initializer_list<std::initializer_list<U>> rows);

    ndarray(std::initializer_list<std::initializer_list<double>> rows);

    /**
     * @brief ND construction via proxy for arbitrary depth,
     *        e.g. `ndarray<int> a{{{1,2},{3,4}},{{5,6},{7,8}}}` (2×2×2).
     *        Uses `detail::NDProxy<T>` so a single overload handles
     *        any depth >=1; ragged inputs throw.
     */
    ndarray(std::initializer_list<detail::NDProxy<T>> nested);

    /**
     * @brief Construction from std::span with explicit shape.
     *        e.g. `std::array<int,4> arr{1,2,3,4}; ndarray<int> a(arr, {2,2});`
     */
    ndarray(std::span<const value_type> data, const std::vector<int> &shape);

    /**
     * @brief Construction from std::array (1-D).
     */
    template <std::size_t N>
    ndarray(const std::array<value_type, N> &arr)
        : ndarray(std::span<const value_type>(arr), std::vector<int>{static_cast<int>(N)})
    {
    }

    /**
     * @brief Construction from C-array.
     */
    template <std::size_t N>
    ndarray(const value_type (&arr)[N])
        : ndarray(std::span<const value_type>(arr, N), std::vector<int>{static_cast<int>(N)})
    {
    }

    /**
     * @brief Range construction — any contiguous range with explicit shape.
     *        e.g. `std::vector<int> v{1,2,3,4}; ndarray<int> a(v, {2,2});`
     */
    // Additional shape-flexible overloads (C++20)
    /** @brief 1-D from std::span (explicit). */
    explicit ndarray(std::span<const value_type> sp)
        : ndarray(std::vector<int>{static_cast<int>(sp.size())}, dtype_of<T>, value_type{})
    {
        std::copy(sp.begin(), sp.end(), data().begin());
    }
    /** @brief From any contiguous range + explicit shape. */
    template <std::ranges::contiguous_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, value_type>
    ndarray(const R &rng, const std::vector<int> &shape);
    /** @brief From vector + explicit shape (e.g. ndarr({1,2,3,4},{2,2})). */
    ndarray(const std::vector<value_type> &vec, const std::vector<int> &shape_)
        : ndarray(shape_, dtype_of<T>, value_type{})
    {
        if (vec.size() != size())
            throw std::invalid_argument("vector size != product(shape)");
        std::copy(vec.begin(), vec.end(), data().begin());
    }

    /**
     * @brief Deep-copying copy constructor (value semantics).
     * @param other Array to copy.
     * @post `this` owns a separate copy of `other`'s logical elements,
     *       stored C-contiguous (views are compacted, not cloned whole).
     */
    ndarray(const ndarray &other);

    /** @brief Move constructor: transfers storage in O(1). */
    ndarray(ndarray &&) noexcept = default;

    /**
     * @brief Deep-copying copy assignment (value semantics).
     * @param other Array to copy.
     * @return Reference to `*this`.
     * @post `this` owns a separate copy of `other`'s data.
     */
    ndarray &operator=(const ndarray &other);

    /** @brief Move assignment: transfers storage in O(1). */
    ndarray &operator=(ndarray &&) noexcept = default;

    // Attributes
    /**
     * @brief Total number of elements.
     * @return `product(shape)`.
     * @complexity O(ndim).
     */
    NP_NODISCARD std::size_t size() const noexcept;

    /**
     * @brief Number of dimensions.
     * @return `shape.size()`.
     * @complexity O(1).
     */
    NP_NODISCARD std::size_t ndim() const noexcept;

    /**
     * @brief Bytes per element.
     * @return `sizeof(T)`.
     * @complexity O(1).
     */
    NP_NODISCARD std::size_t itemsize() const noexcept;

    /**
     * @brief Total bytes consumed by the logical elements.
     * @return `size() * sizeof(T)`.
     * @complexity O(1).
     */
    NP_NODISCARD std::size_t nbytes() const noexcept;

    /**
     * @brief True if the array has no elements.
     * @return `size() == 0`.
     * @complexity O(1).
     */
    NP_NODISCARD bool empty() const noexcept;

    /**
     * @brief True when the logical elements are laid out contiguously
     *        in C (row-major) order.
     * @return true if strides match C-order strides for the shape and
     *         `offset == 0`.
     * @complexity O(ndim).
     */
    NP_NODISCARD bool is_contiguous() const noexcept;

    /**
     * @brief True when the logical elements are laid out
     *        column-major (Fortran) contiguously.
     * @return true if strides match F-order strides for the shape and
     *         `offset == 0`.
     * @complexity O(ndim).
     */
    NP_NODISCARD bool is_f_contiguous() const noexcept;

    /**
     * @brief Writable access to the underlying storage buffer.
     *
     * Lazily allocates the buffer if it is empty.
     * @return Reference to the internal `std::vector<value_type>`.
     */
    std::vector<value_type> &data();

    /**
     * @brief Read-only access to the underlying storage buffer.
     * @return Const reference to the internal `std::vector<value_type>`.
     * @throws std::runtime_error if the array has no data buffer.
     */
    NP_NODISCARD const std::vector<value_type> &data() const;

    /**
     * @brief Product of the shape (total element count).
     * @return `size()`.
     * @complexity O(ndim).
     */
    NP_NODISCARD std::size_t _numel() const noexcept;

    /**
     * @brief Flat logical offset of a multi-index.
     * @param idx Multi-dimensional index; size must equal `ndim()`.
     * @return Linear offset into the storage buffer (accounting for
     *         strides and `offset`).
     * @complexity O(ndim).
     */
    NP_NODISCARD std::size_t _flat(const std::vector<std::size_t> &idx) const noexcept;

    /**
     * @brief Physical storage offset of flat logical position `i`.
     * @param i Flat logical index (C-order).
     * @return Physical offset into the storage buffer.
     * @complexity O(ndim).
     */
    NP_NODISCARD std::size_t _flat_logical(std::size_t i) const noexcept;

    // Iterators
    /**
     * @brief Returns a mutable iterator to the first element.
     * @return Iterator pointing to the first logical element.
     */
    iterator begin();

    /**
     * @brief Returns a mutable iterator past the last element.
     * @return Iterator pointing one past the last logical element.
     */
    iterator end();

    /**
     * @brief Returns a const iterator to the first element.
     * @return Const iterator pointing to the first logical element.
     */
    const_iterator begin() const;

    /**
     * @brief Returns a const iterator past the last element.
     * @return Const iterator pointing one past the last logical element.
     */
    const_iterator end() const;

    /** @brief Const iterator to the first element. */
    const_iterator cbegin() const
    {
        return begin();
    }

    /** @brief Const iterator past the last element. */
    const_iterator cend() const
    {
        return end();
    }

    // Element access
    /**
     * @brief Chained subscript access (read/write).
     *
     * Each call reduces one dimension; e.g. `a[i][j]` for a 2-D
     * array returns a reference to the element at `(i, j)`.
     * @param index Index into the first (outermost) dimension.
     * @return A `Proxy<T>` that can be further subscripted or
     *         implicitly converted to a reference.
     */
    auto operator[](std::ptrdiff_t index) -> Proxy<T>;

    /**
     * @brief Chained subscript access (read-only).
     * @param index Index into the first (outermost) dimension; negative
     *        counts from the end (NumPy semantics).
     * @return A `ConstProxy<T>` that can be further subscripted.
     */
    auto operator[](std::ptrdiff_t index) const -> ConstProxy<T>;

    /**
     * @brief Compile-time-size index access (reference).
     * @tparam N Number of dimensions (deduced from the array).
     * @param idx Fixed-size array of indices, one per dimension.
     * @return Reference to the element at `idx`.
     * @throws std::invalid_argument if `N != ndim()`.
     * @throws std::out_of_range if any index is out of bounds.
     */
    template <std::size_t N> auto get(const std::array<std::size_t, N> &idx) -> reference;

    /**
     * @brief Compile-time-size index access (const reference).
     * @tparam N Number of dimensions.
     * @param idx Fixed-size array of indices.
     * @return Const reference to the element at `idx`.
     * @throws std::invalid_argument if `N != ndim()`.
     * @throws std::out_of_range if any index is out of bounds.
     */
    template <std::size_t N> auto get(const std::array<std::size_t, N> &idx) const -> const_reference;

    /**
     * @brief Runtime index container access (by value).
     * @tparam Container Type of the index container (e.g.
     *         `std::vector<std::size_t>`).
     * @param idx Index container; size must equal `ndim()`.
     * @return Element value at `idx`.
     * @throws std::invalid_argument if `idx.size() != ndim()`.
     * @throws std::out_of_range if any index is out of bounds.
     */
    template <typename Container>
        requires std::ranges::sized_range<Container> &&
                 std::convertible_to<std::ranges::range_value_t<Container>, std::size_t>
    auto get(const Container &idx) const -> value_type;

    /**
     * @brief Runtime index container access (reference, read/write).
     * @tparam Container Type of the index container (e.g.
     *         `std::vector<std::size_t>`).
     * @param idx Index container; size must equal `ndim()`.
     * @return Reference to the element at `idx`.
     * @throws std::invalid_argument if `idx.size() != ndim()`.
     * @throws std::out_of_range if any index is out of bounds.
     */
    template <typename Container>
        requires std::ranges::sized_range<Container> &&
                 std::convertible_to<std::ranges::range_value_t<Container>, std::size_t>
    auto get(const Container &idx) -> reference;

    /**
     * @brief Write a value at runtime index container position.
     * @tparam Container Type of the index container.
     * @param idx Index container; size must equal `ndim()`.
     * @param value Value to write.
     * @throws std::invalid_argument if `idx.size() != ndim()`.
     * @throws std::out_of_range if any index is out of bounds.
     */
    template <typename Container>
        requires std::ranges::sized_range<Container> &&
                 std::convertible_to<std::ranges::range_value_t<Container>, std::size_t>
    void set(const Container &idx, const value_type &value);

    /**
     * @brief 1D bounds-checked access.
     * @param i Row index.
     * @return Reference to the element.
     * @throws std::invalid_argument if `ndim() != 1`.
     * @throws std::out_of_range if `i >= shape[0]`.
     */
    auto at(std::ptrdiff_t i) -> reference;

    /**
     * @brief 1D bounds-checked access (const).
     * @param i Row index; negative counts from the end.
     * @return Const reference to the element (`bool` by value).
     * @throws std::invalid_argument if `ndim() != 1`.
     * @throws std::out_of_range if `i` is out of bounds.
     */
    auto at(std::ptrdiff_t i) const -> const_reference;

    /**
     * @brief Single-index access for 1D arrays (read/write).
     * @param i Element index; negative counts from the end.
     * @return Reference to the element.
     * @throws std::invalid_argument if `ndim() != 1`.
     * @throws std::out_of_range if `i` is out of bounds.
     */
    auto operator()(std::ptrdiff_t i) -> reference;

    /**
     * @brief Single-index access for 1D arrays (const).
     * @param i Element index; negative counts from the end.
     * @return Const reference to the element (`bool` by value).
     * @throws std::invalid_argument if `ndim() != 1`.
     * @throws std::out_of_range if `i` is out of bounds.
     */
    auto operator()(std::ptrdiff_t i) const -> const_reference;

    /**
     * @brief 2D index access (read/write).
     * @param i Row index; negative counts from the end.
     * @param j Column index; negative counts from the end.
     * @return Reference to the element.
     * @throws std::invalid_argument if `ndim() != 2`.
     * @throws std::out_of_range if either index is out of bounds.
     */
    auto operator()(std::ptrdiff_t i, std::ptrdiff_t j) -> reference;

    /**
     * @brief 2D index access (const).
     * @param i Row index; negative counts from the end.
     * @param j Column index; negative counts from the end.
     * @return Const reference to the element (`bool` by value).
     * @throws std::invalid_argument if `ndim() != 2`.
     * @throws std::out_of_range if either index is out of bounds.
     */
    auto operator()(std::ptrdiff_t i, std::ptrdiff_t j) const -> const_reference;

    /**
     * @brief ND index access (read/write) for `ndim() >= 3`.
     *        e.g. `a(0,1,2)` for 3-D, `a(0,1,2,3)` for 4-D.
     *        Negative indices count from the end of each dimension.
     * @tparam Args Integral index types; count must equal `ndim()`.
     *         (Floating-point indices no longer silently truncate.)
     */
    template <typename... Args>
        requires(sizeof...(Args) >= 3 && (std::integral<Args> && ...))
    auto operator()(Args... args) -> reference;
    template <typename... Args>
        requires(sizeof...(Args) >= 3 && (std::integral<Args> && ...))
    auto operator()(Args... args) const -> const_reference;

    /**
     * @brief 2D bounds-checked access.
     * @param i Row index; negative counts from the end.
     * @param j Column index; negative counts from the end.
     * @return Reference to the element.
     * @throws std::invalid_argument if `ndim() != 2`.
     * @throws std::out_of_range if either index is out of bounds.
     */
    auto at(std::ptrdiff_t i, std::ptrdiff_t j) -> reference;

    /**
     * @brief 2D bounds-checked access (const).
     * @param i Row index; negative counts from the end.
     * @param j Column index; negative counts from the end.
     * @return Const reference to the element (`bool` by value).
     * @throws std::invalid_argument if `ndim() != 2`.
     * @throws std::out_of_range if either index is out of bounds.
     */
    auto at(std::ptrdiff_t i, std::ptrdiff_t j) const -> const_reference;

    /**
     * @brief ND bounds-checked access for `ndim() >= 3`.
     * @tparam Args Integral index types; negatives count from the end.
     */
    template <typename... Args>
        requires(sizeof...(Args) >= 3 && (std::integral<Args> && ...))
    auto at(Args... args) -> reference;
    template <typename... Args>
        requires(sizeof...(Args) >= 3 && (std::integral<Args> && ...))
    auto at(Args... args) const -> const_reference;

    /**
     * @brief Returns the single element of a 0-d/1-element array.
     * @return The single element value.
     * @throws std::invalid_argument if `size() != 1`.
     */
    value_type item() const;

    /**
     * @brief Scalar conversion for single-element arrays (numpy `bool()`).
     * @return `true` if the single element is non-zero.
     * @throws std::invalid_argument if `size() != 1`.
     */
    explicit operator bool() const;

    /**
     * @brief Scalar conversion for single-element arrays (numpy `int()`).
     * @return The element cast to `long long`.
     * @throws std::invalid_argument if `size() != 1`.
     */
    explicit operator long long() const;

    /**
     * @brief Scalar conversion for single-element arrays (numpy `float()`).
     * @return The element cast to `double`.
     * @throws std::invalid_argument if `size() != 1`.
     */
    explicit operator double() const;

    /**
     * @brief Scalar conversion for single-element arrays (numpy `complex()`).
     * @return The element as `std::complex<double>`.
     * @throws std::invalid_argument if `size() != 1`.
     */
    explicit operator std::complex<double>() const;

    // Reductions
    /**
     * @brief Sum over all elements.
     * @return Sum of all elements. For `bool` arrays the return type
     *         is `std::int64_t` (NumPy semantics).
     * @complexity O(n), where n = size().
     */
    auto sum() const -> std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>;

    /**
     * @brief Sum along an axis.
     * @tparam Acc Accumulator type (default: element type, or `int64_t`
     *         for `bool`).
     * @param axis Axis along which to reduce. Negative indices count
     *        from the last axis.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n), where n = size().
     */
    template <typename Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
    auto sum(std::optional<int> axis, bool keepdims = false) const -> ndarray<Acc>;

    template <typename Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
    auto sum(int axis, bool keepdims = false) const -> ndarray<Acc>;

    /**
     * @brief Product over all elements.
     * @return Product of all elements. For `bool` arrays the return
     *         type is `std::int64_t`.
     * @complexity O(n), where n = size().
     */
    auto prod() const -> std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>;

    /**
     * @brief Product along an axis.
     * @tparam Acc Accumulator type.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    template <typename Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
    auto prod(std::optional<int> axis, bool keepdims = false) const -> ndarray<Acc>;

    template <typename Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
    auto prod(int axis, bool keepdims = false) const -> ndarray<Acc>;

    /**
     * @brief Minimum over all elements.
     * @return Smallest element.
     * @throws std::runtime_error if the array is empty.
     * @complexity O(n).
     */
    value_type min() const;

    /**
     * @brief Minimum along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto min(std::optional<int> axis, bool keepdims = false) const -> ndarray<T>;

    auto min(int axis, bool keepdims = false) const -> ndarray<T>;

    /**
     * @brief Maximum over all elements.
     * @return Largest element.
     * @throws std::runtime_error if the array is empty.
     * @complexity O(n).
     */
    value_type max() const;

    /**
     * @brief Maximum along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto max(std::optional<int> axis, bool keepdims = false) const -> ndarray<T>;

    auto max(int axis, bool keepdims = false) const -> ndarray<T>;

    /**
     * @brief Peak-to-peak (max - min) over all elements.
     * @return `max() - min()`.
     * @throws std::runtime_error if the array is empty.
     * @complexity O(n).
     */
    value_type ptp() const;

    /**
     * @brief Peak-to-peak (max - min) along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto ptp(std::optional<int> axis, bool keepdims = false) const -> ndarray<T>;

    auto ptp(int axis, bool keepdims = false) const -> ndarray<T>;

    /**
     * @brief Arithmetic mean over all elements.
     * @return Mean value. Integer and boolean inputs promote to
     *         `double`; floating-point and complex inputs keep
     *         their type (NumPy semantics).
     * @throws std::runtime_error if the array is empty.
     * @complexity O(n).
     */
    auto mean() const -> typename _mean_type<value_type>::type;

    /**
     * @brief Arithmetic mean along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto mean(std::optional<int> axis, bool keepdims = false) const -> ndarray<typename _mean_type<T>::type>;

    auto mean(int axis, bool keepdims = false) const -> ndarray<typename _mean_type<T>::type>;

    /**
     * @brief Population variance over all elements.
     * @return Variance. Uses Welford's online algorithm for numerical
     *         stability.
     * @throws std::runtime_error if the array is empty.
     * @complexity O(n).
     */
    auto var() const -> typename _var_type<value_type>::type;

    /**
     * @brief Population variance along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto var(std::optional<int> axis, bool keepdims = false) const -> ndarray<typename _var_type<T>::type>;

    auto var(int axis, bool keepdims = false) const -> ndarray<typename _var_type<T>::type>;

    /**
     * @brief Population standard deviation over all elements.
     * @return Standard deviation (`sqrt(var())`).
     * @complexity O(n).
     */
    auto std() const -> typename _var_type<value_type>::type;

    /**
     * @brief Population standard deviation along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto std(std::optional<int> axis, bool keepdims = false) const -> ndarray<typename _var_type<T>::type>;

    auto std(int axis, bool keepdims = false) const -> ndarray<typename _var_type<T>::type>;

    /**
     * @brief True when every element is non-zero.
     * @return true if all elements are truthy.
     * @complexity O(n).
     */
    bool all() const;

    /**
     * @brief All along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Boolean array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto all(std::optional<int> axis, bool keepdims = false) const -> ndarray<bool>;

    auto all(int axis, bool keepdims = false) const -> ndarray<bool>;

    /**
     * @brief True when any element is non-zero.
     * @return true if any element is truthy.
     * @complexity O(n).
     */
    bool any() const;

    /**
     * @brief Any along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Boolean array with one fewer dimension (or same rank if
     *         `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto any(std::optional<int> axis, bool keepdims = false) const -> ndarray<bool>;

    auto any(int axis, bool keepdims = false) const -> ndarray<bool>;

    /**
     * @brief Flat logical index of the maximum element.
     * @return Linear index of the first occurrence of the maximum.
     * @throws std::runtime_error if the array is empty.
     * @complexity O(n).
     */
    std::size_t argmax() const;

    /**
     * @brief Indices of maxima along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array of indices with one fewer dimension (or same rank
     *         if `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto argmax(std::optional<int> axis, bool keepdims = false) const -> ndarray<std::size_t>;

    auto argmax(int axis, bool keepdims = false) const -> ndarray<std::size_t>;

    /**
     * @brief Flat logical index of the minimum element.
     * @return Linear index of the first occurrence of the minimum.
     * @throws std::runtime_error if the array is empty.
     * @complexity O(n).
     */
    std::size_t argmin() const;

    /**
     * @brief Indices of minima along an axis.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, the reduced axis is retained with size 1.
     * @return Array of indices with one fewer dimension (or same rank
     *         if `keepdims`).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto argmin(std::optional<int> axis, bool keepdims = false) const -> ndarray<std::size_t>;

    auto argmin(int axis, bool keepdims = false) const -> ndarray<std::size_t>;

    /**
     * @brief Cumulative sum (flattened when no axis is given).
     * @return 1-D array of cumulative sums.
     * @complexity O(n).
     */
    auto cumsum() const -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>;

    /**
     * @brief Cumulative sum along an axis.
     * @param axis Axis along which to accumulate.
     * @return Array of the same shape with cumulative sums.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto cumsum(int axis) const -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>;

    /**
     * @brief Cumulative sum with optional axis (flattened when std::nullopt).
     * @param axis Optional axis; std::nullopt means flattened.
     * @return Array of cumulative sums.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto cumsum(std::optional<int> axis) const
        -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>;

    /**
     * @brief Cumulative product (flattened when no axis is given).
     * @return 1-D array of cumulative products.
     * @complexity O(n).
     */
    auto cumprod() const -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>;

    /**
     * @brief Cumulative product along an axis.
     * @param axis Axis along which to accumulate.
     * @return Array of the same shape with cumulative products.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto cumprod(int axis) const -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>;

    /**
     * @brief Cumulative product with optional axis (flattened when std::nullopt).
     * @param axis Optional axis; std::nullopt means flattened.
     * @return Array of cumulative products.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    auto cumprod(std::optional<int> axis) const
        -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>;

    // Sorting / searching
    /**
     * @brief In-place sort along an axis (default: last axis).
     * @param axis Axis along which to sort. Negative indices
     *        count from the last axis.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n log n) per slice, where n is the axis
     *         length.
     */
    void sort(int axis = -1);

    /**
     * @brief In-place sort with optional axis (std::nullopt = last axis).
     * @param axis Optional axis.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n log n).
     */
    void sort(std::optional<int> axis);

    /**
     * @brief Sorted copy of the array along an axis (default: last).
     * @param axis Axis along which to sort.
     * @return A new sorted array.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n log n).
     */
    auto sorted(int axis = -1) const -> ndarray<T>;

    /**
     * @brief Sorted copy with optional axis (std::nullopt = flatten, NumPy np.sort(None)).
     * @param axis Optional axis.
     * @return A new sorted array.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n log n).
     */
    auto sorted(std::optional<int> axis) const -> ndarray<T>;

    /**
     * @brief Indices that would sort the array along an axis.
     * @param axis Axis along which to sort.
     * @return Array of indices that would sort this array along
     *         the given axis.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n log n).
     */
    auto argsort(int axis = -1) const -> ndarray<std::size_t>;

    /**
     * @brief Indices that would sort with optional axis (std::nullopt = flatten, NumPy np.argsort(None)).
     * @param axis Optional axis.
     * @return Array of indices.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n log n).
     */
    auto argsort(std::optional<int> axis) const -> ndarray<std::size_t>;

    /**
     * @brief Indices that would partition at position k along
     *        an axis.
     * @param kth Partition index; the element at position k
     *        will be in its final sorted position.
     * @param axis Axis along which to partition.
     * @return Array of partition indices.
     * @throws np::AxisError if the axis is out of bounds.
     * @throws std::out_of_range if `kth >= axis_len`.
     * @complexity O(n) average (introselect).
     */
    auto argpartition(std::size_t kth, int axis = -1) const -> ndarray<std::size_t>;

    /**
     * @brief Argpartition with optional axis (std::nullopt = flatten).
     * @param kth Partition index.
     * @param axis Optional axis.
     * @return Array of partition indices.
     * @throws np::AxisError if the axis is out of bounds.
     * @throws std::out_of_range if `kth >= axis_len`.
     * @complexity O(n) average (introselect).
     */
    auto argpartition(std::size_t kth, std::optional<int> axis) const -> ndarray<std::size_t>;

    /**
     * @brief Binary search for a value in a sorted 1D array.
     * @param value Value to search for.
     * @param side_right If true, return the rightmost
     *        insertion point; otherwise the leftmost.
     * @return Index where `value` would be inserted.
     * @throws std::invalid_argument if the array is not 1-D.
     * @complexity O(log n).
     */
    std::size_t searchsorted(const value_type &value, bool side_right = false) const;

    std::size_t searchsorted(const value_type &value, std::optional<bool> side_right) const;

    /**
     * @brief Searchsorted applied to every element of `values`.
     * @tparam U Needle element type (converted to `value_type`).
     * @param values 1-D array of search values.
     * @return Array of insertion indices.
     * @throws std::invalid_argument if the array is not 1-D.
     * @complexity O(m log n), where m = values.size().
     */
    template <typename U> auto searchsorted(const ndarray<U> &values) const -> ndarray<std::size_t>;

    // Shape manipulation
    /**
     * @brief View (when contiguous) or copy with a new shape.
     *
     * If the array is C-contiguous, returns a view sharing
     * storage. Otherwise, returns a copy. At most one
     * dimension may be `-1`, in which case it is inferred.
     * @param shape New shape; at most one dimension may be -1.
     * @return Reshaped array (view or copy).
     * @throws std::invalid_argument if the total element count
     *         does not match, or if more than one dimension is -1.
     * @complexity O(n) for the copy path, O(ndim) for the view path.
     */
    auto reshape(const std::vector<int> &shape) const -> ndarray;

    /**
     * @brief View with reversed dimensions.
     * @return Array with shape and strides reversed.
     * @complexity O(ndim).
     */
    auto transpose() const -> ndarray;

    /**
     * @brief View with a permutation of the dimensions.
     * @param perm Permutation of axis indices; length must
     *        equal `ndim()`.
     * @return Array with axes reordered.
     * @throws std::invalid_argument if `perm` is not a valid
     *         permutation of `[0, ndim())`.
     * @complexity O(ndim).
     */
    auto transpose(const std::vector<int> &perm) const -> ndarray;

    /**
     * @brief View with two axes swapped.
     * @param axis1 First axis index.
     * @param axis2 Second axis index.
     * @return Array with the two axes swapped.
     * @throws np::AxisError if either axis is out of bounds.
     * @complexity O(ndim).
     */
    auto swapaxes(int axis1, int axis2) const -> ndarray;

    /**
     * @brief View removing all size-1 dimensions.
     * @return Array with all singleton dimensions removed.
     * @complexity O(ndim).
     */
    auto squeeze() const -> ndarray;

    /**
     * @brief View removing a specific dimension.
     * @param axis Axis to remove; must have extent 1.
     * @return Array with the specified axis removed.
     * @throws np::AxisError if the axis is out of bounds.
     * @throws std::invalid_argument if `shape[axis] != 1`.
     * @complexity O(ndim).
     */
    auto squeeze(int axis) const -> ndarray;

    /**
     * @brief View (contiguous) or copy flattened in C order.
     * @return 1-D array containing all logical elements.
     * @complexity O(n) for the copy path, O(ndim) for the view path.
     */
    auto ravel() const -> ndarray;

    /**
     * @brief Copy flattened in C order.
     * @return 1-D copy of all elements.
     * @complexity O(n).
     */
    auto flatten() const -> ndarray;

    /**
     * @brief Resize in place to a new total number of elements.
     *
     * Truncates or pads with default-constructed values.
     * @param new_shape New shape; total element count may
     *        differ from the current size.
     * @complexity O(n).
     */
    void resize(const std::vector<int> &new_shape);

    // Manipulation
    /**
     * @brief Fill every element with a value.
     * @param value Value to fill with.
     * @complexity O(n).
     */
    void fill(const value_type &value);

    /**
     * @brief Securely zero the underlying storage (constant-time, not elided).
     *
     * Uses `pqc::secure_zero` (volatile + compiler fence) to ensure the
     * zeroing is not optimized away. For `vector<bool>` each bit is cleared
     * via fill(false). Intended for key material (ML-KEM/ML-DSA).
     *
     * Reference: pqc.hpp:secure_zero, NIST FIPS 203/204
     * @complexity O(n).
     */
    void secure_zero() noexcept;

    /**
     * @brief Securely zero and reset shape to empty (constant-time erasure).
     *
     * Zeroes the backing store via `secure_zero()` then clears shape/strides
     * and releases the buffer. Use when an array held secret key material
     * and must not linger in memory.
     * @complexity O(n).
     */
    void secure_clear() noexcept;

    /**
     * @brief Securely fill every element (constant-time, not elided).
     *
     * Uses volatile store + `pqc::ct_barrier` to ensure not optimized away.
     * For zero value, delegates to `secure_zero()`.
     * @complexity O(n).
     */
    // NOTE: not noexcept — copying an arbitrary T can throw.
    void secure_fill(const value_type &value);

    /**
     * @brief Constant-time element access (no secret-dependent branches).
     *
     * Returns element at `i` if in bounds, else zero, without branching on `i`.
     * Uses `pqc::ct_select` + `ct_barrier` to avoid timing leaks.
     * For ND arrays, `i` is flat logical index.
     * @complexity O(ndim).
     */
    // NOTE: not noexcept — allocation (unravel buffer) can throw.
    NP_NODISCARD value_type secure_at(std::size_t i) const;

    /**
     * @brief Deep copy of the array.
     * @return New array with the same data and shape.
     * @complexity O(n).
     */
    auto copy() const -> ndarray;

    /**
     * @brief View sharing the same storage.
     * @return New array that shares `data_` with `*this`.
     * @note The returned view is always mutable, even when called on a
     *       `const` array (same for `transpose`/`reshape`/`squeeze`/
     *       `ravel`/`diagonal`/`real`/`imag`/`mT`/`flat`). True
     *       const-propagation would require `ndarray<const T>` storage,
     *       which is ill-formed for `std::vector<const T>` — NumPy has no
     *       const arrays either, so mutability-through-views is by design.
     * @complexity O(ndim).
     */
    auto view() const -> ndarray;

    /**
     * @brief Element-wise conversion to another type.
     * @tparam U Target element type.
     * @return New array with elements cast to `U`.
     * @complexity O(n).
     */
    template <typename U> auto astype() const -> ndarray<U>;

    /**
     * @brief Gather elements along an axis.
     * @param indices Indices to gather (must be in range; negatives are
     *        not accepted here, unlike element accessors).
     * @param axis Axis along which to gather (default: 0); the
     *        `std::optional<int>` overload flattens first when nullopt
     *        (NumPy `take` with `axis=None`).
     * @return New array with gathered elements.
     * @throws np::AxisError if the axis is out of bounds.
     * @throws std::out_of_range if any index is out of bounds.
     * @complexity O(n).
     */
    auto take(const std::vector<std::size_t> &indices, int axis = 0) const -> ndarray;

    auto take(const std::vector<std::size_t> &indices, std::optional<int> axis) const -> ndarray;

    /**
     * @brief Set elements at flat logical positions.
     * @param indices Flat logical positions to write to.
     * @param values Values to write (cycled if shorter than
     *        `indices`).
     * @param mode `'r'` raise on out-of-bounds (default),
     *        `'w'` wrap, `'c'` clip.
     * @throws std::out_of_range if an index is out of bounds
     *         and `mode == 'r'`.
     * @complexity O(indices.size()).
     */
    void put(const std::vector<std::size_t> &indices, const std::vector<typename ndarray<T>::value_type> &values,
             char mode = 'r');

    void put(const std::vector<std::size_t> &indices, const std::vector<typename ndarray<T>::value_type> &values,
             std::optional<char> mode);

    /**
     * @brief Repeat elements (flattened when no axis given).
     * @param repeats Number of repetitions per element.
     * @return Array with repeated elements.
     * @complexity O(n * repeats).
     */
    auto repeat(std::size_t repeats) const -> ndarray;

    /**
     * @brief Repeat elements along an axis.
     * @param repeats Number of repetitions per element.
     * @param axis Axis along which to repeat.
     * @return Array with repeated elements along the axis.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n * repeats).
     */
    auto repeat(std::size_t repeats, int axis) const -> ndarray;

    /**
     * @brief Repeat with optional axis (std::nullopt = flattened).
     * @param repeats Number of repetitions per element.
     * @param axis Optional axis.
     * @return Array with repeated elements.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n * repeats).
     */
    auto repeat(std::size_t repeats, std::optional<int> axis) const -> ndarray;

    /**
     * @brief Clip values into [min_value, max_value].
     * @param min_value Lower bound.
     * @param max_value Upper bound.
     * @return New array with clipped values.
     * @complexity O(n).
     */
    auto clip(const value_type &min_value, const value_type &max_value) const -> ndarray;

    /**
     * @brief Round to `decimals` places.
     * @param decimals Number of decimal places (default: 0).
     * @return New array with rounded values.
     * @note Only affects floating-point element types; integer
     *       arrays are returned unchanged.
     * @complexity O(n).
     */
    auto round(int decimals = 0) const -> ndarray;

    auto round(std::optional<int> decimals) const -> ndarray;

    /**
     * @brief Diagonal of a 2D+ array.
     * @param offset Diagonal offset (0 = main diagonal,
     *        positive = above, negative = below).
     * @return 1-D array of diagonal elements.
     * @throws np::AxisError if `ndim() < 2`.
     * @complexity O(min(shape[0], shape[1])).
     */
    auto diagonal(int offset = 0) const -> ndarray;

    auto diagonal(std::optional<int> offset) const -> ndarray;

    /**
     * @brief Sum along the diagonal.
     * @param offset Diagonal offset.
     * @return Sum of diagonal elements.
     * @throws np::AxisError if `ndim() < 2`.
     * @complexity O(min(shape[0], shape[1])).
     */
    value_type trace(int offset = 0) const;

    value_type trace(std::optional<int> offset) const;

    /**
     * @brief Indices of non-zero elements (one array per dimension).
     * @return Vector of 1-D arrays, one per dimension.
     * @complexity O(n).
     */
    auto nonzero() const -> std::vector<ndarray<std::size_t>>;

    /**
     * @brief Element-wise complex conjugate.
     * @return New array with conjugated elements.
     * @complexity O(n).
     */
    auto conj() const -> ndarray;

    /**
     * @brief Swap the byte order of every element, in place.
     * @complexity O(n).
     */
    void byteswap();

    // Selection / manipulation (numpy.ndarray.choose / compress / ...)
    /**
     * @brief Element-wise absolute value.
     * @return New array with absolute values.
     * @complexity O(n).
     */
    auto abs() const -> ndarray<typename detail::_Np_real_of<T>::type>;

    /**
     * @brief Alias of conj() (numpy.ndarray.conjugate).
     * @return New array with conjugated elements.
     * @complexity O(n).
     */
    auto conjugate() const -> ndarray;

    /**
     * @brief Build an array from an index array and a list of
     *        choices.
     *
     * The i-th output element is `choices[a[i]][i]` with
     * broadcast indexing.
     * @tparam U Element type of the choice arrays.
     * @param choices Choice arrays; all must be broadcastable
     *        to the same shape.
     * @param mode `'r'` raise (default), `'w'` wrap,
     *        `'c'` clip.
     * @return New array assembled from the choices.
     * @throws std::invalid_argument if `choices` is empty or
     *         if an out-of-range index is encountered with
     *         `mode == 'r'`.
     * @complexity O(n * choices.size()).
     */
    template <typename U> auto choose(const std::vector<ndarray<U>> &choices, char mode = 'r') const -> ndarray<U>;

    template <typename U>
    auto choose(const std::vector<ndarray<U>> &choices, std::optional<char> mode) const -> ndarray<U>;

    /**
     * @brief Return selected slices along an axis.
     *
     * When `condition` is a 1-D bool array whose length
     * matches the axis length, the matching slices are kept.
     * A `nullopt` axis works on the flattened array (numpy
     * default).
     * @param condition 1-D bool array of selectors.
     * @param axis Axis along which to filter (default:
     *        flattened).
     * @return New array with selected slices.
     * @throws std::invalid_argument if `condition` is not 1-D
     *         or if its length does not match the axis length.
     * @complexity O(n).
     */
    auto compress(const ndarray<bool> &condition, std::optional<int> axis = std::nullopt) const -> ndarray;

    /**
     * @brief Matrix product (delegates to np::dot).
     * @tparam U Right-hand operand element type.
     * @param b Right-hand operand.
     * @return Result of the matrix product.
     * @see np::linalg::dot
     * @complexity O(n^3) for 2-D arrays (standard matrix
     *         multiplication).
     */
    template <typename U> auto dot(const ndarray<U> &b) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Matrix multiply (numpy `@`, delegates to np::matmul).
     * @tparam U Right-hand operand element type.
     * @param b Right-hand operand.
     * @return Result of the matrix multiplication.
     * @see np::linalg::matmul
     * @complexity O(n^3) for 2-D arrays.
     */
    template <typename U> auto matmul(const ndarray<U> &b) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief In-place partial sort so that a[kth] is in its
     *        sorted position along an axis.
     * @param kth Partition index.
     * @param axis Axis along which to partition (default: last).
     * @throws np::AxisError if the axis is out of bounds.
     * @throws std::out_of_range if `kth >= axis_len`.
     * @complexity O(n) average (introselect).
     */
    void partition(std::size_t kth, int axis = -1);

    void partition(std::size_t kth, std::optional<int> axis);

    /**
     * @brief Real part: for complex element types the extracted
     *        real components; for real types a view of the array
     *        itself.
     * @return New array of real components, or a view for real
     *         types.
     * @complexity O(n).
     */
    auto real() const -> ndarray<typename detail::_Np_real_of<T>::type>;

    /**
     * @brief Imaginary part: for complex element types the
     *        extracted imaginary components; for real types an
     *        all-zero array.
     * @return New array of imaginary components, or a zero array
     *         for real types.
     * @complexity O(n).
     */
    auto imag() const -> ndarray<typename detail::_Np_real_of<T>::type>;

    /**
     * @brief View transposing the last two dimensions
     *        (ndim >= 2).
     * @return Array with the last two axes swapped.
     * @throws np::AxisError if `ndim() < 2`.
     * @complexity O(ndim).
     */
    auto mT() const -> ndarray;

    /**
     * @brief Set the WRITEABLE flag (numpy.ndarray.setflags).
     * @param writeable If false, the array becomes read-only.
     */
    void setflags(bool writeable);

    /**
     * @brief Current WRITEABLE flag.
     * @return true if the array is writable.
     */
    NP_NODISCARD bool writeable() const noexcept;

    /**
     * @brief Base storage pointer when the array borrows memory
     *        from a parent view, nullptr when it owns its data
     *        (numpy.ndarray.base).
     * @return Raw pointer to the parent's data buffer, or
     *         nullptr if this array owns its data.
     */
    NP_NODISCARD const void *base() const noexcept;

    /**
     * @brief True when the array owns its own data buffer.
     * @return true if `data_` is owned exclusively.
     */
    NP_NODISCARD bool owns_data() const noexcept;

    /**
     * @brief True when the array shares storage with a parent
     *        view.
     * @return true if this is a view.
     */
    NP_NODISCARD bool is_view() const noexcept;

    /**
     * @brief 1-D view of the logical elements (numpy.ndarray.flat).
     * @return 1-D array of all logical elements.
     */
    auto flat() const -> ndarray;

    /**
     * @brief Size of the first axis (numpy `__len__`).
     * @return `shape[0]`.
     * @throws std::invalid_argument if the array is 0-D.
     */
    NP_NODISCARD std::size_t len() const;

    /**
     * @brief True when any element equals `value` (numpy `in`).
     * @param value Value to search for.
     * @return true if `value` is found.
     * @complexity O(n).
     */
    NP_NODISCARD bool contains(const value_type &value) const;

    /**
     * @brief Element-wise floor division (numpy `//`).
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand array.
     * @return Array of floored division results.
     * @complexity O(n).
     */
    template <typename U> auto floordiv(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise floor division by a scalar.
     * @tparam U Scalar type.
     * @param scalar Divisor.
     * @return Array of floored division results.
     * @complexity O(n).
     */
    template <typename U> auto floordiv(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief (floor_divide, remainder) pair (numpy `divmod`).
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand array.
     * @return Pair of (floordiv, remainder) arrays.
     * @complexity O(n).
     */
    template <typename U>
    auto divmod(const ndarray<U> &rhs) const
        -> std::pair<ndarray<std::common_type_t<T, U>>, ndarray<std::common_type_t<T, U>>>;

    /**
     * @brief (floor_divide, remainder) pair by a scalar.
     * @tparam U Scalar type.
     * @param scalar Divisor.
     * @return Pair of (floordiv, remainder) arrays.
     * @complexity O(n).
     */
    template <typename U>
    auto divmod(const U &scalar) const
        -> std::pair<ndarray<std::common_type_t<T, U>>, ndarray<std::common_type_t<T, U>>>;

    /**
     * @brief Element-wise power (numpy `**`).
     * @tparam U Right-hand operand element type.
     * @param rhs Exponent array.
     * @return Array of element-wise powers.
     * @complexity O(n).
     */
    template <typename U> auto pow(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise power by a scalar.
     * @tparam U Scalar exponent type.
     * @param scalar Exponent.
     * @return Array of element-wise powers.
     * @complexity O(n).
     */
    template <typename U> auto pow(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    // Conversions / IO
    /**
     * @brief Flat logical elements as a std::vector.
     * @return Vector of all logical elements in C order.
     * @note Deliberately flat for every `ndim` (unlike NumPy, which nests
     *       lists per dimension); kept flat for API stability — a nested
     *       return type would break all existing callers.
     * @complexity O(n).
     */
    auto tolist() const -> std::vector<value_type>;

    /**
     * @brief Native-endian byte dump of the logical elements.
     * @return Byte vector of all elements in native endianness.
     * @complexity O(n * sizeof(T)).
     */
    auto tobytes() const -> std::vector<std::uint8_t>;

    /**
     * @brief Write the raw bytes to a binary file.
     * @param filename Output file path.
     * @throws std::runtime_error if the file cannot be opened.
     * @complexity O(n * sizeof(T)).
     */
    void tofile(const std::string &filename) const;

    /**
     * @brief Write the raw bytes to an output stream.
     * @param os Output stream.
     * @complexity O(n * sizeof(T)).
     */
    void tofile(std::ostream &os) const;

    /**
     * @brief Human-readable representation to stdout.
     * @param os Output stream (default: `std::cout`).
     * @complexity O(n).
     */
    void print(std::ostream &os = std::cout) const;

    // Element-wise arithmetic (broadcasting)
    /**
     * @brief Element-wise addition with another array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Broadcast sum.
     * @complexity O(n), where n = broadcast size.
     */
    template <typename U> auto operator+(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise subtraction with another array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Broadcast difference.
     * @complexity O(n).
     */
    template <typename U> auto operator-(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise multiplication with another array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Broadcast product.
     * @complexity O(n).
     */
    template <typename U> auto operator*(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise division with another array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Broadcast quotient.
     * @complexity O(n).
     */
    template <typename U> auto operator/(const ndarray<U> &rhs) const -> ndarray<detail::div_result_t<T, U>>;

    /**
     * @brief Element-wise addition with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Array with each element incremented.
     * @complexity O(n).
     */
    template <typename U> auto operator+(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise subtraction of a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Array with each element decremented.
     * @complexity O(n).
     */
    template <typename U> auto operator-(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise multiplication by a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Array with each element scaled.
     * @complexity O(n).
     */
    template <typename U> auto operator*(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise division by a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar divisor.
     * @return Array with each element divided.
     * @complexity O(n).
     */
    template <typename U> auto operator/(const U &scalar) const -> ndarray<detail::div_result_t<T, U>>;

    /**
     * @brief Unary negation (element-wise).
     * @return Array with negated elements.
     * @complexity O(n).
     */
    auto operator-() const -> ndarray;

    /**
     * @brief Unary plus (numpy `+a`): identity copy.
     * @return Copy of the array.
     * @complexity O(n).
     */
    auto operator+() const -> ndarray;

    // Element-wise floored remainder (numpy `%`)

    /**
     * @brief Element-wise floored remainder with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Array of floored remainders.
     * @complexity O(n).
     */
    template <typename U> auto operator%(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise floored remainder by a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar divisor.
     * @return Array of floored remainders.
     * @complexity O(n).
     */
    template <typename U> auto operator%(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    // Bitwise ops (integral/bool element types only, numpy semantics)

    /**
     * @brief Element-wise bitwise AND with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Bitwise AND result.
     * @pre Both element types must be integral or bool.
     * @complexity O(n).
     */
    template <typename U> auto operator&(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise bitwise AND with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Bitwise AND result.
     * @pre Both element types must be integral or bool.
     * @complexity O(n).
     */
    template <typename U> auto operator&(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise bitwise OR with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Bitwise OR result.
     * @pre Both element types must be integral or bool.
     * @complexity O(n).
     */
    template <typename U> auto operator|(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise bitwise OR with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Bitwise OR result.
     * @pre Both element types must be integral or bool.
     * @complexity O(n).
     */
    template <typename U> auto operator|(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise bitwise XOR with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Bitwise XOR result.
     * @pre Both element types must be integral or bool.
     * @complexity O(n).
     */
    template <typename U> auto operator^(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise bitwise XOR with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Bitwise XOR result.
     * @pre Both element types must be integral or bool.
     * @complexity O(n).
     */
    template <typename U> auto operator^(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise bitwise NOT (numpy `~`).
     * @return Bitwise complement of each element.
     * @pre Element type must be integral.
     * @complexity O(n).
     */
    auto operator~() const -> ndarray;

    // Element-wise shifts (integral element types only)

    /**
     * @brief Element-wise left shift with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Shift amounts.
     * @return Left-shifted result.
     * @pre Both element types must be integral.
     * @complexity O(n).
     */
    template <typename U> auto operator<<(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise left shift by a scalar.
     * @tparam U Scalar type.
     * @param scalar Shift amount.
     * @return Left-shifted result.
     * @pre Both element types must be integral.
     * @complexity O(n).
     */
    template <typename U> auto operator<<(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise right shift with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Shift amounts.
     * @return Right-shifted result.
     * @pre Both element types must be integral.
     * @complexity O(n).
     */
    template <typename U> auto operator>>(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Element-wise right shift by a scalar.
     * @tparam U Scalar type.
     * @param scalar Shift amount.
     * @return Right-shifted result.
     * @pre Both element types must be integral.
     * @complexity O(n).
     */
    template <typename U> auto operator>>(const U &scalar) const -> ndarray<std::common_type_t<T, U>>;

    // Comparisons (element-wise, NumPy semantics)

    /**
     * @brief Element-wise equality with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Boolean array of element-wise equality.
     * @complexity O(n).
     */
    template <typename U> auto operator==(const ndarray<U> &rhs) const -> ndarray<bool>;

    /**
     * @brief Element-wise inequality with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Boolean array of element-wise inequality.
     * @complexity O(n).
     */
    template <typename U> auto operator!=(const ndarray<U> &rhs) const -> ndarray<bool>;

    /**
     * @brief Element-wise less-than with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator<(const ndarray<U> &rhs) const -> ndarray<bool>;

    /**
     * @brief Element-wise less-than-or-equal with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator<=(const ndarray<U> &rhs) const -> ndarray<bool>;

    /**
     * @brief Element-wise greater-than with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator>(const ndarray<U> &rhs) const -> ndarray<bool>;

    /**
     * @brief Element-wise greater-than-or-equal with an array.
     * @tparam U Right-hand operand element type.
     * @param rhs Right-hand operand.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator>=(const ndarray<U> &rhs) const -> ndarray<bool>;

    /**
     * @brief Element-wise equality with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator==(const U &scalar) const -> ndarray<bool>;

    /**
     * @brief Element-wise inequality with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator!=(const U &scalar) const -> ndarray<bool>;

    /**
     * @brief Element-wise less-than with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator<(const U &scalar) const -> ndarray<bool>;

    /**
     * @brief Element-wise less-than-or-equal with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator<=(const U &scalar) const -> ndarray<bool>;

    /**
     * @brief Element-wise greater-than with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator>(const U &scalar) const -> ndarray<bool>;

    /**
     * @brief Element-wise greater-than-or-equal with a scalar.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @return Boolean array of element-wise comparisons.
     * @complexity O(n).
     */
    template <typename U> auto operator>=(const U &scalar) const -> ndarray<bool>;

    /**
     * @brief True if same shape and all elements equal.
     * @param other Array to compare against.
     * @return true if shapes match and all elements are equal.
     * @complexity O(n).
     */
    // NOTE: not noexcept — element comparison and odometer allocation can throw.
    bool all_equal(const ndarray &other) const;

    /**
     * @brief True if all elements equal the given value.
     * @param value Value to compare against.
     * @return true if every element equals `value`.
     * @complexity O(n).
     */
    bool all_equal(const typename ndarray<T>::value_type &value) const;

    // In-place arithmetic (same shape, or broadcast for += etc.)

    // In-place operators write through to shared storage (views included)
    // and never change shape: `rhs` must broadcast to `*this`'s shape.
    // Unlike the value-returning operators, `/=` keeps truncating division
    // on integral arrays (NumPy same-kind casting; true division cannot be
    // stored back into an integer buffer).

    /**
     * @brief In-place element-wise addition with an array.
     * @tparam U Right-hand element type (converted to `T`).
     * @param rhs Right-hand operand, broadcastable to `*this`.
     * @return Reference to `*this`.
     * @complexity O(n).
     */
    template <typename U> ndarray &operator+=(const ndarray<U> &rhs);

    /**
     * @brief In-place element-wise subtraction with an array.
     * @tparam U Right-hand element type (converted to `T`).
     * @param rhs Right-hand operand, broadcastable to `*this`.
     * @return Reference to `*this`.
     * @complexity O(n).
     */
    template <typename U> ndarray &operator-=(const ndarray<U> &rhs);

    /**
     * @brief In-place element-wise multiplication with an array.
     * @tparam U Right-hand element type (converted to `T`).
     * @param rhs Right-hand operand, broadcastable to `*this`.
     * @return Reference to `*this`.
     * @complexity O(n).
     */
    template <typename U> ndarray &operator*=(const ndarray<U> &rhs);

    /**
     * @brief In-place element-wise division with an array (truncating for
     *        integral `T`; see note above).
     * @tparam U Right-hand element type (converted to `T`).
     * @param rhs Right-hand operand, broadcastable to `*this`.
     * @return Reference to `*this`.
     * @complexity O(n).
     */
    template <typename U> ndarray &operator/=(const ndarray<U> &rhs);

    /**
     * @brief In-place addition of a scalar.
     * @tparam U Scalar type (converted to `T`).
     * @param scalar Scalar value.
     * @return Reference to `*this`.
     * @complexity O(n).
     */
    template <typename U> ndarray &operator+=(const U &scalar);

    /**
     * @brief In-place subtraction of a scalar.
     * @tparam U Scalar type (converted to `T`).
     * @param scalar Scalar value.
     * @return Reference to `*this`.
     * @complexity O(n).
     */
    template <typename U> ndarray &operator-=(const U &scalar);

    /**
     * @brief In-place multiplication by a scalar.
     * @tparam U Scalar type (converted to `T`).
     * @param scalar Scalar value.
     * @return Reference to `*this`.
     * @complexity O(n).
     */
    template <typename U> ndarray &operator*=(const U &scalar);

    /**
     * @brief In-place division by a scalar (truncating for integral `T`).
     * @tparam U Scalar type (converted to `T`).
     * @param scalar Scalar divisor.
     * @return Reference to `*this`.
     * @complexity O(n).
     */
    template <typename U> ndarray &operator/=(const U &scalar);

    // In-place floored remainder / bitwise / shifts (heterogeneous, write-through)

    /** @brief In-place floored remainder with an array. */
    template <typename U> ndarray &operator%=(const ndarray<U> &rhs);
    /** @brief In-place floored remainder by a scalar. */
    template <typename U> ndarray &operator%=(const U &scalar);
    /** @brief In-place bitwise AND with an array. */
    template <typename U> ndarray &operator&=(const ndarray<U> &rhs);
    /** @brief In-place bitwise AND with a scalar. */
    template <typename U> ndarray &operator&=(const U &scalar);
    /** @brief In-place bitwise OR with an array. */
    template <typename U> ndarray &operator|=(const ndarray<U> &rhs);
    /** @brief In-place bitwise OR with a scalar. */
    template <typename U> ndarray &operator|=(const U &scalar);
    /** @brief In-place bitwise XOR with an array. */
    template <typename U> ndarray &operator^=(const ndarray<U> &rhs);
    /** @brief In-place bitwise XOR with a scalar. */
    template <typename U> ndarray &operator^=(const U &scalar);
    /** @brief In-place left shift with an array. */
    template <typename U> ndarray &operator<<=(const ndarray<U> &rhs);
    /** @brief In-place left shift by a scalar. */
    template <typename U> ndarray &operator<<=(const U &scalar);
    /** @brief In-place right shift with an array. */
    template <typename U> ndarray &operator>>=(const ndarray<U> &rhs);
    /** @brief In-place right shift by a scalar. */
    template <typename U> ndarray &operator>>=(const U &scalar);

    // In-place floor division / power (no C++ operator spelling)

    /** @brief In-place floored division by an array. */
    template <typename U> ndarray &floordiv_eq(const ndarray<U> &rhs);
    /** @brief In-place floored division by a scalar. */
    template <typename U> ndarray &floordiv_eq(const U &scalar);
    /** @brief In-place element-wise power by an array. */
    template <typename U> ndarray &pow_eq(const ndarray<U> &rhs);
    /** @brief In-place element-wise power by a scalar. */
    template <typename U> ndarray &pow_eq(const U &scalar);

    // Scalar-on-the-left friends

    /**
     * @brief Scalar + array (commutative with array + scalar).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast sum.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator+(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr + scalar;
    }

    /**
     * @brief Scalar - array (non-commutative).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast difference.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator-(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr._scalar_left_op(scalar, [](const U &a, const T &b) { return a - b; });
    }

    /**
     * @brief Scalar * array (commutative with array * scalar).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast product.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator*(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr._scalar_left_op(scalar, [](const U &a, const T &b) { return a * b; });
    }

    /**
     * @brief Scalar / array (non-commutative).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast quotient.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator/(const U &scalar, const ndarray &arr) -> ndarray<detail::div_result_t<U, T>>
    {
        // NumPy true_divide: integral / integral promotes to double.
        using R = detail::div_result_t<U, T>;
        ndarray<R> out(arr.shape);
        std::size_t i = 0;
        arr._for_each_logical([&](const typename ndarray<T>::value_type &v) {
            if constexpr (std::is_integral_v<U> && std::is_integral_v<T>)
            {
                out.data()[i++] = static_cast<double>(scalar) / static_cast<double>(v);
            }
            else
            {
                out.data()[i++] = scalar / v;
            }
        });
        return out;
    }

    /**
     * @brief Scalar == array, !=, <, <=, >, >= (element-wise, NumPy-style).
     * @tparam U Scalar type.
     * @return Boolean array; true where the comparison holds.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator==(const U &scalar, const ndarray &arr) -> ndarray<bool>
    {
        return arr._cmp_scalar_left(scalar, [](const U &a, const T &b) { return a == b; });
    }
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator!=(const U &scalar, const ndarray &arr) -> ndarray<bool>
    {
        return arr._cmp_scalar_left(scalar, [](const U &a, const T &b) { return a != b; });
    }
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator<(const U &scalar, const ndarray &arr) -> ndarray<bool>
    {
        return arr._cmp_scalar_left(scalar, [](const U &a, const T &b) { return a < b; });
    }
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator<=(const U &scalar, const ndarray &arr) -> ndarray<bool>
    {
        return arr._cmp_scalar_left(scalar, [](const U &a, const T &b) { return a <= b; });
    }
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator>(const U &scalar, const ndarray &arr) -> ndarray<bool>
    {
        return arr._cmp_scalar_left(scalar, [](const U &a, const T &b) { return a > b; });
    }
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator>=(const U &scalar, const ndarray &arr) -> ndarray<bool>
    {
        return arr._cmp_scalar_left(scalar, [](const U &a, const T &b) { return a >= b; });
    }

    /**
     * @brief Scalar % array (non-commutative).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast floored remainder.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator%(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr._scalar_left_op(scalar, [](const U &a, const T &b) { return detail::floored_mod(a, b); });
    }

    /**
     * @brief Scalar & array (bitwise AND).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast bitwise AND.
     * @pre Both types must be integral.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator&(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr._scalar_left_op(scalar, [](const U &a, const T &b) { return a & b; });
    }

    /**
     * @brief Scalar | array (bitwise OR).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast bitwise OR.
     * @pre Both types must be integral.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator|(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr._scalar_left_op(scalar, [](const U &a, const T &b) { return a | b; });
    }

    /**
     * @brief Scalar ^ array (bitwise XOR).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast bitwise XOR.
     * @pre Both types must be integral.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator^(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr._scalar_left_op(scalar, [](const U &a, const T &b) { return a ^ b; });
    }

    /**
     * @brief Scalar << array (left shift).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast left shift.
     * @pre Both types must be integral.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator<<(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr._scalar_left_op(scalar,
                                   [](const U &a, const T &b) { return a << detail::checked_shift_count(a, b); });
    }

    /**
     * @brief Scalar >> array (right shift).
     * @tparam U Scalar type.
     * @param scalar Left-hand scalar operand.
     * @param arr Right-hand array operand.
     * @return Broadcast right shift.
     * @pre Both types must be integral.
     * @complexity O(n).
     */
    template <typename U>
        requires std::is_arithmetic_v<U> || detail::is_complex_v<U>
    friend auto operator>>(const U &scalar, const ndarray &arr) -> ndarray<std::common_type_t<U, T>>
    {
        return arr._scalar_left_op(scalar,
                                   [](const U &a, const T &b) { return a >> detail::checked_shift_count(a, b); });
    }

    /**
     * @brief Stream output in NumPy repr style.
     * @param os Output stream.
     * @param arr Array to output.
     * @return Reference to `os`.
     * @complexity O(n).
     */
    friend auto operator<<(std::ostream &os, const ndarray &arr) -> std::ostream &
    {
        arr._print_to(os);
        return os;
    }

  private:
    // Internals
    template <typename U> friend class ndarray;

    std::shared_ptr<std::vector<value_type>> data_; ///< Shared storage (enables views)

    /** WRITEABLE flag (numpy.ndarray.setflags). */
    bool writeable_ = true;

    /** True when this array shares storage with a parent view. */
    bool is_view_ = false;

    /**
     * @brief View constructor (shares storage).
     * @param data Shared data buffer.
     * @param shape Dimensions.
     * @param strides Stride vector in elements.
     * @param type Data type.
     * @param order Memory layout.
     * @param offset Element offset into storage.
     */
    ndarray(std::shared_ptr<std::vector<value_type>> data, std::vector<int> shape, std::vector<std::size_t> strides,
            np::dtype type, matrix::Order order, std::size_t offset);

    /**
     * @brief C-order strides for a shape.
     * @param shape Shape vector.
     * @return Stride vector in elements for C-order layout.
     * @complexity O(ndim).
     */
    NP_NODISCARD static std::vector<std::size_t> _c_strides(const std::vector<int> &shape);

    /**
     * @brief Validate that every shape dimension is non-negative.
     * @param s Shape to validate.
     * @throws std::invalid_argument if any dimension is negative.
     * @note A negative dimension would otherwise silently wrap to a huge
     *       (or, for certain combinations, incorrectly small) value when
     *       cast to `std::size_t` in `_numel()`/`_c_strides()`, producing
     *       a buffer/stride mismatch and out-of-bounds access.
     */
    static void _validate_shape(const std::vector<int> &s);

    /**
     * @brief Validated element count: throws before any allocation if
     *        `s` contains a negative dimension.
     * @param s Shape to measure.
     * @return `product(s)`.
     * @throws std::invalid_argument if any dimension is negative.
     */
    NP_NODISCARD static std::size_t _checked_numel(const std::vector<int> &s);

    /**
     * @brief shape as std::size_t vector.
     * @return Shape converted to `std::size_t`.
     * @complexity O(ndim).
     */
    NP_NODISCARD std::vector<std::size_t> _shape_u() const;

    /**
     * @brief Normalize a possibly negative axis.
     * @param axis Axis index (may be negative).
     * @return Normalized axis in [0, ndim()).
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(1).
     */
    NP_NODISCARD int _normalize_axis(int axis) const;

    /**
     * @brief Normalize a possibly negative element index (NumPy semantics).
     * @param i Index (negative counts from the end).
     * @param dim Dimension extent; must be non-negative.
     * @return Normalized index in [0, dim).
     * @throws std::out_of_range if the normalized index is out of bounds.
     * @complexity O(1).
     */
    NP_NODISCARD static std::size_t _norm_idx(std::ptrdiff_t i, std::ptrdiff_t dim);

    /**
     * @brief Throw if this array has no data buffer.
     *
     * Default-constructed arrays (and views thereof) own no storage; most
     * element accessors would otherwise segfault on the null buffer.
     * @throws std::runtime_error if `data_` is null.
     * @complexity O(1).
     */
    void _require_data() const;

    /**
     * @brief Visit every logical element.
     * @tparam Fn Callable accepting `const T&`.
     * @param fn Function to call for each element.
     * @complexity O(n).
     */
    template <typename Fn> void _for_each_logical(Fn &&fn) const;

    /**
     * @brief Visit every logical element with its multi-index.
     * @tparam Fn Callable accepting `(const std::vector<std::size_t>&, const
     * T&)`.
     * @param fn Function to call for each element.
     * @complexity O(n).
     */
    template <typename Fn> void _for_each_indexed(Fn &&fn) const;

    /**
     * @brief Generic axis reduction.
     * @tparam Acc Accumulator type.
     * @tparam StepFn Callable accepting `(Acc&, const T&)`.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, retain the reduced axis with size 1.
     * @param seed Optional initial accumulator value.
     * @param step Reduction step function.
     * @return Reduced array.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    template <typename Acc, typename Fn>
    auto _reduce_axis(int axis, bool keepdims, std::optional<Acc> seed, Fn &&step) const -> ndarray<Acc>;

    /**
     * @brief Welford-based variance along an axis.
     * @tparam MeanT Accumulator/promoted type.
     * @param axis Axis along which to compute variance.
     * @param keepdims If true, retain the reduced axis with size 1.
     * @return Variance array.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    template <typename MeanT> auto _var_axis(int axis, bool keepdims) const -> ndarray<MeanT>;

    /**
     * @brief Generic extrema/arg reduction along an axis.
     * @tparam Cmp Comparison callable returning bool.
     * @param axis Axis along which to reduce.
     * @param keepdims If true, retain the reduced axis with size 1.
     * @param cmp Comparison function (returns true if first arg is
     *        "better").
     * @return Array of indices of the extrema along the axis.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    template <typename Cmp> auto _arg_reduce_axis(int axis, bool keepdims, Cmp &&cmp) const -> ndarray<std::size_t>;

    /**
     * @brief Internal flat write used by cumsum/cumprod.
     * @tparam Acc Accumulator type.
     * @tparam Fn Reduction callable.
     * @param axis Axis along which to accumulate.
     * @param fn Accumulation function.
     * @return Array of accumulated values.
     * @throws np::AxisError if the axis is out of bounds.
     * @complexity O(n).
     */
    template <typename Acc, typename Fn> auto _cum_axis(int axis, Fn &&fn) const -> ndarray<Acc>;

    /**
     * @brief Scalar element-wise operation over own shape.
     * @tparam U Scalar type.
     * @tparam Fn Binary operation callable.
     * @param scalar Scalar operand.
     * @param fn Operation `(T, U) -> R`.
     * @return Result array.
     * @complexity O(n).
     */
    template <typename U, typename Fn>
    auto _scalar_op(const U &scalar, Fn &&fn) const -> ndarray<std::common_type_t<T, U>>;

    /**
     * @brief Scalar on the left (a op b[i] with a first).
     * @tparam U Scalar type.
     * @tparam Fn Binary operation callable.
     * @param scalar Left scalar operand.
     * @param fn Operation `(U, T) -> R`.
     * @return Result array.
     * @complexity O(n).
     */
    template <typename U, typename Fn>
    auto _scalar_left_op(const U &scalar, Fn &&fn) const -> ndarray<std::common_type_t<U, T>>;

    /**
     * @brief Scalar comparison producing a bool array.
     * @tparam U Scalar type.
     * @tparam Fn Comparison callable.
     * @param scalar Scalar operand.
     * @param fn Comparison `(T, U) -> bool`.
     * @return Boolean result array.
     * @complexity O(n).
     */
    template <typename U, typename Fn> auto _cmp_scalar(const U &scalar, Fn &&fn) const -> ndarray<bool>;

    /**
     * @brief In-place element-wise op with another array (write-through).
     * @tparam U Right-hand element type.
     * @tparam Fn Pure callable `(T, T) -> T`; the result is converted back
     *         to `T` and assigned (assignment, not compound assignment, so
     *         `vector<bool>` proxies work).
     * @param rhs Right-hand operand, broadcastable to `*this`'s shape.
     * @throws std::invalid_argument if `rhs` cannot broadcast to `*this`.
     * @throws std::runtime_error if `*this` is not writeable or has no data.
     * @complexity O(n).
     */
    template <typename U, typename Fn> void _inplace_op(const ndarray<U> &rhs, Fn &&fn);

    /**
     * @brief In-place element-wise op with a scalar (write-through).
     * @tparam U Scalar type.
     * @tparam Fn Pure callable `(T, U) -> T` (see above for proxy note).
     * @throws std::runtime_error if `*this` is not writeable or has no data.
     * @complexity O(n).
     */
    template <typename U, typename Fn> void _inplace_scalar(const U &scalar, Fn &&fn);

    /**
     * @brief Left-scalar comparison producing a bool array.
     * @tparam U Scalar type.
     * @tparam Fn Comparison callable.
     * @param scalar Left scalar operand.
     * @param fn Comparison `(U, T) -> bool`.
     * @return Boolean result array.
     * @complexity O(n).
     */
    template <typename U, typename Fn> auto _cmp_scalar_left(const U &scalar, Fn &&fn) const -> ndarray<bool>;

    /**
     * @brief Recursive printing helper.
     * @param dim Current dimension depth.
     * @param flat_offset Flat offset into the data buffer.
     * @param os Output stream.
     */
    void _print_recursive(std::size_t dim, std::size_t flat_offset, std::ostream &os) const;

    /**
     * @brief Full repr: `array(..., dtype=...)`.
     * @param os Output stream.
     */
    void _print_to(std::ostream &os) const;

    /**
     * @brief Storage pointer for iterators.
     * @return Raw pointer to the data buffer + offset.
     */
    value_type *_raw_ptr() noexcept;

    /**
     * @brief Const storage pointer for iterators.
     * @return Const raw pointer to the data buffer + offset.
     */
    const value_type *_raw_ptr() const noexcept;

    /**
     * @brief Finalize strides/type after construction.
     *
     * Computes C-order strides, allocates the data buffer
     * if empty, and deduces the dtype from `T` if unset.
     */
    void _finalize();

    template <typename U> void _flatten_initializer(std::initializer_list<U> list);

    /**
     * @brief Valid scalar type constraint.
     * @tparam U Type to check.
     */
    template <typename U> static constexpr bool _is_valid_scalar = std::is_arithmetic_v<U> || detail::is_complex_v<U>;
};

// Forward declarations so ndarray<T>::dot/matmul can delegate to the
// free functions defined in linalg.hpp (which includes this header).
namespace linalg
{
template <typename T, typename U>
auto dot(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>;
template <typename T, typename U>
auto matmul(const ndarray<T> &a, const ndarray<U> &b) -> ndarray<std::common_type_t<T, U>>;
} // namespace linalg

// Broadcasting helpers
namespace detail
{

/**
 * @brief NumPy-style broadcast of two shapes.
 *
 * Aligns shapes from the right and expands dimensions
 * of size 1 to match the other shape.
 * @param a First shape.
 * @param b Second shape.
 * @return Broadcast shape.
 * @throws std::invalid_argument if the shapes cannot be
 *         broadcast together.
 * @complexity O(max(a.size(), b.size())).
 */
NP_NODISCARD inline std::vector<int> broadcast_shapes(const std::vector<int> &a, const std::vector<int> &b)
{
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    const int nr = std::max(na, nb);
    std::vector<int> r(nr);
    for (int d = 0; d < nr; ++d)
    {
        const int ia = na - nr + d;
        const int ib = nb - nr + d;
        const int sa = ia < 0 ? 1 : a[ia];
        const int sb = ib < 0 ? 1 : b[ib];
        if (sa == sb)
        {
            r[d] = sa;
        }
        else if (sa == 1)
        {
            r[d] = sb;
        }
        else if (sb == 1)
        {
            r[d] = sa;
        }
        else
        {
            throw std::invalid_argument("operands could not be broadcast together");
        }
    }
    return r;
}

NP_NODISCARD inline std::vector<std::size_t> broadcast_index(const std::vector<int> &in_shape,
                                                             const std::vector<int> &out_shape,
                                                             const std::vector<std::size_t> &out_idx)
{
    if (in_shape.size() > out_shape.size())
    {
        throw std::invalid_argument("broadcast_index: input rank exceeds output rank");
    }
    std::vector<std::size_t> in_idx(in_shape.size(), 0);
    const std::ptrdiff_t out_nd = static_cast<std::ptrdiff_t>(out_shape.size());
    const std::ptrdiff_t in_nd = static_cast<std::ptrdiff_t>(in_shape.size());
    for (std::ptrdiff_t d = 0; d < out_nd; ++d)
    {
        const std::ptrdiff_t in_d = d - (out_nd - in_nd);
        if (in_d < 0)
        {
            continue;
        }
        const std::size_t id = static_cast<std::size_t>(in_d);
        const std::size_t od = static_cast<std::size_t>(d);
        if (in_shape[id] == 1)
        {
            in_idx[id] = 0;
        }
        else
        {
            in_idx[id] = out_idx[od];
        }
    }
    return in_idx;
}

/* Micro-optimized radix sort for integral types: O(n) vs O(n log n)
 * Uses 4-pass 8-bit counting sort (LSD) with sign-bit flipping for signed.
 * For floating point, reinterprets bits as integer with sign handling.
 * Code is intentionally long (explicit loops) for max throughput.
 */
template <typename T> inline void radix_sort_integral(std::vector<T> &a)
{
    // bool has no unsigned counterpart (make_unsigned_t<bool> is ill-formed)
    // and sorts trivially; route it to std::sort explicitly.
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                  "radix_sort_integral requires a non-bool integral type");
    if (a.size() < 64)
    {
        std::stable_sort(a.begin(), a.end());
        return;
    }
    using U = std::make_unsigned_t<T>;
    const std::size_t n = a.size();
    std::vector<T> b(n);
    std::vector<U> cur(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        U u = static_cast<U>(a[i]);
        if constexpr (std::is_signed_v<T>)
        {
            u ^= U(1) << (sizeof(T) * 8 - 1);
        }
        cur[i] = u;
    }
    constexpr int BITS = 8;
    constexpr int BUCKETS = 1 << BITS;
    constexpr int PASSES = sizeof(T) * 8 / BITS;
    std::array<std::size_t, BUCKETS> cnt{};
    std::array<std::size_t, BUCKETS> pos{};
    std::vector<U> tmp(n);
    for (int pass = 0; pass < PASSES; ++pass)
    {
        cnt.fill(0);
        int shift = pass * BITS;
        for (std::size_t i = 0; i < n; ++i)
            ++cnt[(cur[i] >> shift) & (BUCKETS - 1)];
        pos[0] = 0;
        for (int i = 1; i < BUCKETS; ++i)
            pos[i] = pos[i - 1] + cnt[i - 1];
        for (std::size_t i = 0; i < n; ++i)
        {
            int bucket = (cur[i] >> shift) & (BUCKETS - 1);
            tmp[pos[bucket]++] = cur[i];
        }
        cur.swap(tmp);
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        U u = cur[i];
        if constexpr (std::is_signed_v<T>)
        {
            u ^= U(1) << (sizeof(T) * 8 - 1);
        }
        b[i] = static_cast<T>(u);
    }
    a.swap(b);
}

/**
 * @brief Strict weak ordering on `pair<index, value>` by value, NumPy-style.
 *
 * Plain `<` for arithmetic types; lexicographic real-then-imag for complex
 * (matching `sort()`'s complex branch — `std::complex` has no `operator<`).
 */
/**
 * @brief Strict weak ordering on values, NumPy-style.
 *
 * Plain `<` for arithmetic types; lexicographic real-then-imag for complex
 * (matching `sort()`'s complex branch — `std::complex` has no `operator<`).
 */
template <typename V> inline bool value_less(const V &a, const V &b)
{
    if constexpr (is_complex_v<V>)
    {
        if (a.real() != b.real())
        {
            return a.real() < b.real();
        }
        return a.imag() < b.imag();
    }
    else
    {
        return a < b;
    }
}

template <typename P> inline bool pair_second_less(const P &a, const P &b)
{
    using V = typename P::second_type;
    if constexpr (is_complex_v<V>)
    {
        if (a.second.real() != b.second.real())
        {
            return a.second.real() < b.second.real();
        }
        return a.second.imag() < b.second.imag();
    }
    else
    {
        return a.second < b.second;
    }
}

template <typename T> inline void radix_sort_pair(std::vector<std::pair<std::size_t, T>> &a)
{
    // Constraint first: make_unsigned_t<non-integral> (and <bool>) would
    // otherwise hard-error before any static_assert fires.
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                  "radix_sort_pair requires a non-bool integral key type");
    if (a.size() < 64)
    {
        // Stable, matching the LSD radix path below (previously std::sort,
        // whose tie order differed by input size).
        std::stable_sort(a.begin(), a.end(), [](auto &x, auto &y) { return x.second < y.second; });
        return;
    }
    // Radix sort pairs by T (second) while carrying index (first)
    using U = std::make_unsigned_t<T>;
    const std::size_t n = a.size();
    std::vector<std::pair<std::size_t, T>> b(n);
    std::vector<U> keys(n);
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        U u = static_cast<U>(a[i].second);
        if constexpr (std::is_signed_v<T>)
            u ^= U(1) << (sizeof(T) * 8 - 1);
        keys[i] = u;
        idx[i] = a[i].first;
    }
    constexpr int BITS = 8, BUCKETS = 256, PASSES = sizeof(T) * 8 / 8;
    std::array<std::size_t, BUCKETS> cnt{}, pos{};
    std::vector<U> tmpk(n);
    std::vector<std::size_t> tmpi(n);
    for (int pass = 0; pass < PASSES; ++pass)
    {
        cnt.fill(0);
        int shift = pass * BITS;
        for (std::size_t i = 0; i < n; ++i)
            ++cnt[(keys[i] >> shift) & 255];
        pos[0] = 0;
        for (int i = 1; i < BUCKETS; ++i)
            pos[i] = pos[i - 1] + cnt[i - 1];
        for (std::size_t i = 0; i < n; ++i)
        {
            int buck = (keys[i] >> shift) & 255;
            tmpk[pos[buck]] = keys[i];
            tmpi[pos[buck]] = idx[i];
            ++pos[buck];
        }
        keys.swap(tmpk);
        idx.swap(tmpi);
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        U u = keys[i];
        if constexpr (std::is_signed_v<T>)
            u ^= U(1) << (sizeof(T) * 8 - 1);
        b[i].first = idx[i];
        b[i].second = static_cast<T>(u);
    }
    a.swap(b);
}

/**
 * @brief Element-wise operation with broadcasting.
 *
 * Computes the broadcast shape, then iterates over
 * every logical element applying `fn(a[i], b[i])`.
 * @tparam R Element type of `a`.
 * @tparam S Element type of `b`.
 * @tparam Fn Callable accepting `(const R&, const S&)`
 *        and returning the output type.
 * @param a First operand.
 * @param b Second operand.
 * @param fn Element-wise operation.
 * @return New array with the broadcast shape and
 *         element-wise results.
 * @throws std::invalid_argument if shapes cannot be
 *         broadcast.
 * @complexity O(n), where n = broadcast size.
 */
template <typename R, typename S, typename Fn> auto elementwise(const ndarray<R> &a, const ndarray<S> &b, Fn &&fn)
{
    // Decay: callables invoked on references/proxies must not deduce
    // reference or proxy output types (ndarray<U&> is ill-formed).
    using OutT = std::decay_t<std::invoke_result_t<Fn, const R &, const S &>>;
    // Null buffers throw here via the const data() accessor (runtime_error),
    // never segfault later. (detail::elementwise is not a friend, so only
    // public accessors are used throughout.)
    const auto &ad = a.data();
    const auto &bd = b.data();
    const std::vector<int> out_shape = broadcast_shapes(a.shape, b.shape);
    ndarray<OutT> out(out_shape);

    const int nr = static_cast<int>(out_shape.size());
    const int shift_a = nr - static_cast<int>(a.shape.size());
    const int shift_b = nr - static_cast<int>(b.shape.size());

    std::vector<std::size_t> adj_a(nr), adj_b(nr);
    for (int d = 0; d < nr; ++d)
    {
        const int ka = d - shift_a;
        const int kb = d - shift_b;
        adj_a[d] = (ka < 0 || a.shape[ka] == 1) ? 0 : a.strides[ka];
        adj_b[d] = (kb < 0 || b.shape[kb] == 1) ? 0 : b.strides[kb];
    }
    const std::vector<std::size_t> &out_strides = out.strides;

    // Fast path: identical dense layouts iterate linearly (offsets included).
    // vector<bool> is excluded: bit-packed RMW on shared words races.
    if constexpr (!std::is_same_v<R, bool> && !std::is_same_v<S, bool> && !std::is_same_v<OutT, bool>)
    {
        if (a.shape == b.shape && a.is_contiguous() && b.is_contiguous())
        {
            const R *__restrict pa = ad.data() + a.offset;
            const S *__restrict pb = bd.data() + b.offset;
            OutT *__restrict po = out.data().data();
            const std::size_t n = out.size();
#ifdef NP_USE_THREADING
            if (n > detail::kParallelThreshold)
            {
                detail::maybe_parallel_for(0, n, [&](std::size_t i) { po[i] = fn(pa[i], pb[i]); });
                return out;
            }
#endif
            for (std::size_t i = 0; i < n; ++i)
            {
                po[i] = fn(pa[i], pb[i]);
            }
            return out;
        }
    }

    auto at = [&](std::size_t fo, const std::vector<std::size_t> &idx) {
        std::size_t fa = a.offset, fb = b.offset;
        for (int d = 0; d < nr; ++d)
        {
            fa += idx[d] * adj_a[d];
            fb += idx[d] * adj_b[d];
        }
        out.data()[fo] = fn(ad[fa], bd[fb]);
    };

#ifdef NP_USE_THREADING
    // vector<bool> output cannot run in parallel (shared-word read-modify-write).
    const std::size_t n_elem = out.size();
    if constexpr (!std::is_same_v<OutT, bool>)
    {
        if (n_elem > detail::kParallelThreshold)
        {
            // Shard the flat output range (n_elem > 0 here, so no dim is 0);
            // unravel per task from the dense output layout instead of
            // materializing n_elem index vectors up front.
            detail::maybe_parallel_for(0, n_elem, [&](std::size_t fo) {
                std::size_t rem = fo, fa = a.offset, fb = b.offset;
                for (int d = nr - 1; d >= 0; --d)
                {
                    const std::size_t dim = static_cast<std::size_t>(out_shape[d]);
                    const std::size_t coord = rem % dim;
                    rem /= dim;
                    fa += coord * adj_a[d];
                    fb += coord * adj_b[d];
                }
                out.data()[fo] = fn(ad[fa], bd[fb]);
            });
            return out;
        }
    }
#endif
    Odometer od(out_shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        std::size_t fo = 0;
        for (int d = 0; d < nr; ++d)
        {
            fo += idx[d] * out_strides[d];
        }
        at(fo, idx);
        od.advance();
    }
    return out;
}

/**
 * @brief Flat offset of array element at a broadcast
 *        position.
 *
 * Computes the physical storage offset for element
 * `idx` of array `a` when `a` is broadcast to
 * `out_shape`.
 * @tparam R Element type of the source array.
 * @param a Source array.
 * @param out_shape Broadcast shape (rank >= a's rank).
 * @param idx Multi-index into the broadcast shape.
 * @return Physical flat offset into `a`'s storage.
 * @complexity O(out_shape.size()).
 */
template <typename R>
NP_NODISCARD inline std::size_t broadcast_offset(const ndarray<R> &a, const std::vector<int> &out_shape,
                                                 const std::vector<std::size_t> &idx)
{
    if (a.shape.size() > out_shape.size())
    {
        throw std::invalid_argument("broadcast_offset: input rank exceeds output rank");
    }
    const int nr = static_cast<int>(out_shape.size());
    const int shift = nr - static_cast<int>(a.shape.size());
    std::size_t f = a.offset;
    for (int d = 0; d < nr; ++d)
    {
        const int ka = d - shift;
        if (ka < 0 || a.shape[ka] == 1)
        {
            continue;
        }
        f += idx[d] * a.strides[ka];
    }
    return f;
}

} // namespace detail

// Implementation
template <typename T>
ndarray<T>::ndarray(const std::vector<int> &shape, np::dtype type, const value_type &fill)
    : shape(shape), type(type), data_(std::make_shared<std::vector<value_type>>(_checked_numel(shape), fill))
{
    _finalize();
}

template <typename T>
auto ndarray<T>::from_data(const std::vector<int> &shape, std::vector<typename ndarray<T>::value_type> data) -> ndarray
{
    _validate_shape(shape);
    ndarray out;
    out.shape = shape;
    out.data_ = std::make_shared<std::vector<value_type>>(std::move(data));
    if (out.data_->size() != out._numel())
    {
        throw std::invalid_argument("data size does not match the array shape");
    }
    out._finalize();
    return out;
}

template <typename T>
template <typename U, typename>
ndarray<T>::ndarray(std::initializer_list<U> list)
    : data_(std::make_shared<std::vector<typename ndarray<T>::value_type>>())
{
    // Flatten and convert
    _flatten_initializer(list);
    shape = {static_cast<int>(list.size())};
    _finalize();
}

template <typename T> template <typename U> ndarray<T>::ndarray(std::initializer_list<std::initializer_list<U>> rows)
{
    const int n_rows = static_cast<int>(rows.size());
    const int n_cols = n_rows > 0 ? static_cast<int>(rows.begin()->size()) : 0;
    shape = {n_rows, n_cols};
    data_ = std::make_shared<std::vector<value_type>>(_numel(), typename ndarray<T>::value_type{});
    std::size_t k = 0;
    for (const auto &row : rows)
    {
        if (static_cast<int>(row.size()) != n_cols)
        {
            throw std::invalid_argument("ragged rows in nested initializer list");
        }
        for (const U &v : row)
        {
            (*data_)[k++] = static_cast<value_type>(v);
        }
    }
    _finalize();
}

template <typename T>
ndarray<T>::ndarray(std::initializer_list<std::initializer_list<double>> rows)
    : data_(std::make_shared<std::vector<typename ndarray<T>::value_type>>())
{
    // Store the shape
    shape = {static_cast<int>(rows.size()), 0};

    // Reserve space
    size_t total_elements = 0;
    for (const auto &row : rows)
    {
        total_elements += row.size();
    }
    data_->reserve(total_elements);

    // Fill with converted values; ragged rows are rejected like the
    // generic nested-list constructor (previously silently mis-shaped).
    bool first_row = true;
    for (const auto &row : rows)
    {
        if (first_row)
        {
            shape[1] = static_cast<int>(row.size());
            first_row = false;
        }
        else if (static_cast<int>(row.size()) != shape[1])
        {
            throw std::invalid_argument("ragged rows in nested initializer list");
        }
        for (const auto &val : row)
        {
            data_->push_back(static_cast<typename ndarray<T>::value_type>(val));
        }
    }
    _finalize();
}

template <typename T> ndarray<T>::ndarray(std::initializer_list<detail::NDProxy<T>> nested)
{
    shape = detail::proxy_shape_list(nested);
    std::vector<value_type> flat;
    flat.reserve(_checked_numel(shape));
    for (auto &p : nested)
    {
        detail::proxy_flatten(p, flat);
    }
    if (flat.size() != _checked_numel(shape))
    {
        throw std::invalid_argument("ragged nested initializer list");
    }
    data_ = std::make_shared<std::vector<value_type>>(std::move(flat));
    _finalize();
}

template <typename T>
ndarray<T>::ndarray(std::span<const value_type> data, const std::vector<int> &shape) : shape(shape)
{
    if (_checked_numel(shape) != data.size())
        throw std::invalid_argument("shape/data size mismatch");
    data_ = std::make_shared<std::vector<value_type>>(data.begin(), data.end());
    _finalize();
}

template <typename T>
template <std::ranges::contiguous_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, typename ndarray<T>::value_type>
ndarray<T>::ndarray(const R &range, const std::vector<int> &shape_) : shape(shape_)
{
    std::vector<typename ndarray<T>::value_type> tmp(std::ranges::begin(range), std::ranges::end(range));
    if (_checked_numel(shape_) != tmp.size())
        throw std::invalid_argument("shape/data size mismatch");
    data_ = std::make_shared<std::vector<typename ndarray<T>::value_type>>(std::move(tmp));
    _finalize();
}

template <typename T>
ndarray<T>::ndarray(const ndarray &other)
    : shape(other.shape), type(other.type), order(matrix::Order::C), offset(0), writeable_(other.writeable_),
      is_view_(false)
{
    // Compact copy: only the logical elements are duplicated (a view of a
    // huge parent no longer clones the whole buffer), stored C-contiguous.
    strides = _c_strides(shape);
    if (other.data_)
    {
        auto buf = std::make_shared<std::vector<value_type>>();
        buf->reserve(other._numel());
        other._for_each_logical([&](const value_type &v) { buf->push_back(v); });
        data_ = std::move(buf);
    }
}

template <typename T> ndarray<T> &ndarray<T>::operator=(const ndarray &other)
{
    if (this != &other)
    {
        // Copy-and-swap: allocate the new buffer before touching `this`,
        // so a throwing element copy cannot leave a half-assigned object.
        ndarray tmp(other);
        shape = std::move(tmp.shape);
        strides = std::move(tmp.strides);
        type = tmp.type;
        order = tmp.order;
        offset = tmp.offset;
        writeable_ = tmp.writeable_;
        is_view_ = tmp.is_view_;
        data_ = std::move(tmp.data_);
    }
    return *this;
}

template <typename T>
ndarray<T>::ndarray(std::shared_ptr<std::vector<typename ndarray<T>::value_type>> data, std::vector<int> shape,
                    std::vector<std::size_t> strides, np::dtype type, matrix::Order order, std::size_t offset)
    : shape(std::move(shape)), strides(std::move(strides)), type(type), order(order), offset(offset),
      data_(std::move(data)), is_view_(true)
{
}

// Attributes
template <typename T> auto ndarray<T>::size() const noexcept -> std::size_t
{
    return _numel();
}

template <typename T> auto ndarray<T>::ndim() const noexcept -> std::size_t
{
    return shape.size();
}

template <typename T> auto ndarray<T>::itemsize() const noexcept -> std::size_t
{
    return sizeof(T);
}

template <typename T> auto ndarray<T>::nbytes() const noexcept -> std::size_t
{
    return _numel() * sizeof(T);
}

template <typename T> bool ndarray<T>::empty() const noexcept
{
    return _numel() == 0;
}

template <typename T> bool ndarray<T>::is_contiguous() const noexcept
{
    // NumPy semantics: a view with nonzero offset but dense C strides (e.g.
    // `a[1:3]`) IS C-contiguous. Callers must still add `offset` to the base
    // pointer (all fast paths in this file do).
    if (strides.size() != shape.size()) [[unlikely]]
        return false;
    std::size_t exp = 1;
    for (std::size_t i = shape.size(); i-- > 0;)
    {
        if (strides[i] != exp) [[unlikely]]
            return false;
        exp *= static_cast<std::size_t>(shape[i]);
    }
    if (!data_)
    {
        return true;
    }
    const std::size_t n = _numel();
    return offset <= data_->size() && n <= data_->size() - offset;
}

template <typename T> bool ndarray<T>::is_f_contiguous() const noexcept
{
    if (strides.size() != shape.size()) [[unlikely]]
        return false;
    std::size_t stride = 1;
    for (std::size_t d = 0; d < shape.size(); ++d)
    {
        if (strides[d] != stride)
        {
            return false;
        }
        stride *= static_cast<std::size_t>(shape[d]);
    }
    if (!data_)
    {
        return true;
    }
    const std::size_t n = _numel();
    return offset <= data_->size() && n <= data_->size() - offset;
}

template <typename T> auto ndarray<T>::data() -> std::vector<typename ndarray<T>::value_type> &
{
    if (!data_)
    {
        // Allocate room for the offset as well: element (i) lives at
        // storage [offset + ...], so a bare _numel() buffer would OOB.
        data_ = std::make_shared<std::vector<typename ndarray<T>::value_type>>(offset + _numel(),
                                                                               typename ndarray<T>::value_type{});
    }
    return *data_;
}

template <typename T> auto ndarray<T>::data() const -> const std::vector<typename ndarray<T>::value_type> &
{
    if (!data_)
    {
        throw std::runtime_error("ndarray has no data buffer");
    }
    return *data_;
}

// Iterators
template <typename T> auto ndarray<T>::_raw_ptr() noexcept -> typename ndarray<T>::value_type *
{
    return data_ ? data_->data() + offset : nullptr;
}

template <typename T> auto ndarray<T>::_raw_ptr() const noexcept -> const typename ndarray<T>::value_type *
{
    return data_ ? data_->data() + offset : nullptr;
}

template <> inline auto ndarray<bool>::_raw_ptr() noexcept -> bool *
{
    return nullptr;
}

template <> inline auto ndarray<bool>::_raw_ptr() const noexcept -> const bool *
{
    return nullptr;
}

template <typename T> auto ndarray<T>::begin() -> iterator
{
    // done from the start on empty or buffer-less arrays (never deref null)
    return iterator(_raw_ptr(), _shape_u(), strides, _numel() == 0 || !data_);
}

template <typename T> auto ndarray<T>::end() -> iterator
{
    return iterator(_raw_ptr(), _shape_u(), strides, true);
}

template <typename T> auto ndarray<T>::begin() const -> const_iterator
{
    return const_iterator(_raw_ptr(), _shape_u(), strides, _numel() == 0 || !data_);
}

// Bit-packed storage has no raw pointer: bool arrays iterate by index.
template <> inline auto ndarray<bool>::begin() -> iterator
{
    return iterator(data_.get(), _shape_u(), strides, offset, _numel() == 0 || !data_);
}
template <> inline auto ndarray<bool>::end() -> iterator
{
    return iterator(data_.get(), _shape_u(), strides, offset, true);
}
template <> inline auto ndarray<bool>::begin() const -> const_iterator
{
    return const_iterator(data_.get(), _shape_u(), strides, offset, _numel() == 0 || !data_);
}
template <> inline auto ndarray<bool>::end() const -> const_iterator
{
    return const_iterator(data_.get(), _shape_u(), strides, offset, true);
}

template <typename T> auto ndarray<T>::end() const -> const_iterator
{
    return const_iterator(_raw_ptr(), _shape_u(), strides, true);
}

// Element access
template <typename T> auto ndarray<T>::operator[](std::ptrdiff_t index) -> Proxy<T>
{
    detail::IndexStack<> idx;
    idx.push_back(shape.empty() ? static_cast<std::size_t>(index) : _norm_idx(index, shape[0]));
    return Proxy<T>(*this, idx);
}

template <typename T> auto ndarray<T>::operator[](std::ptrdiff_t index) const -> ConstProxy<T>
{
    detail::IndexStack<> idx;
    idx.push_back(shape.empty() ? static_cast<std::size_t>(index) : _norm_idx(index, shape[0]));
    return ConstProxy<T>(*this, idx);
}

template <typename T> template <std::size_t N> auto ndarray<T>::get(const std::array<std::size_t, N> &idx) -> reference
{
    _require_data();
    if (N != shape.size())
    {
        throw std::invalid_argument("index dimensionality does not match array dimensions");
    }
    std::size_t flat = offset;
    for (std::size_t i = 0; i < N; ++i)
    {
        if (idx[i] >= static_cast<std::size_t>(shape[i]))
        {
            throw std::out_of_range("index out of bounds");
        }
        flat += idx[i] * strides[i];
    }
    return (*data_)[flat];
}

template <typename T>
template <std::size_t N>
auto ndarray<T>::get(const std::array<std::size_t, N> &idx) const -> const_reference
{
    _require_data();
    if (N != shape.size())
    {
        throw std::invalid_argument("index dimensionality does not match array dimensions");
    }
    std::size_t flat = offset;
    for (std::size_t i = 0; i < N; ++i)
    {
        if (idx[i] >= static_cast<std::size_t>(shape[i]))
        {
            throw std::out_of_range("index out of bounds");
        }
        flat += idx[i] * strides[i];
    }
    return (*data_)[flat];
}

template <typename T>
template <typename Container>
    requires std::ranges::sized_range<Container> &&
             std::convertible_to<std::ranges::range_value_t<Container>, std::size_t>
auto ndarray<T>::get(const Container &idx) const -> typename ndarray<T>::value_type
{
    _require_data();
    if (idx.size() != shape.size())
    {
        throw std::invalid_argument("index dimensionality does not match array dimensions");
    }
    std::size_t flat = offset;
    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        if (idx[i] >= static_cast<std::size_t>(shape[i]))
        {
            throw std::out_of_range("index out of bounds");
        }
        flat += idx[i] * strides[i];
    }
    return (*data_)[flat];
}

template <typename T>
template <typename Container>
    requires std::ranges::sized_range<Container> &&
             std::convertible_to<std::ranges::range_value_t<Container>, std::size_t>
auto ndarray<T>::get(const Container &idx) -> reference
{
    _require_data();
    if (idx.size() != shape.size())
    {
        throw std::invalid_argument("index dimensionality does not match array dimensions");
    }
    std::size_t flat = offset;
    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        if (idx[i] >= static_cast<std::size_t>(shape[i]))
        {
            throw std::out_of_range("index out of bounds");
        }
        flat += idx[i] * strides[i];
    }
    return (*data_)[flat];
}

template <typename T>
template <typename Container>
    requires std::ranges::sized_range<Container> &&
             std::convertible_to<std::ranges::range_value_t<Container>, std::size_t>
void ndarray<T>::set(const Container &idx, const typename ndarray<T>::value_type &value)
{
    _require_data();
    if (idx.size() != shape.size())
    {
        throw std::invalid_argument("index dimensionality does not match array dimensions");
    }
    std::size_t flat = offset;
    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        if (idx[i] >= static_cast<std::size_t>(shape[i]))
        {
            throw std::out_of_range("index out of bounds");
        }
        flat += idx[i] * strides[i];
    }
    (*data_)[flat] = value;
}

template <typename T> auto ndarray<T>::at(std::ptrdiff_t i) -> reference
{
    _require_data();
    if (shape.size() != 1) [[unlikely]]
    {
        throw std::invalid_argument("at() requires a 1D array");
    }
    const std::size_t ni = _norm_idx(i, shape[0]);
    if constexpr (std::is_same_v<T, bool>)
    {
        return (*data_)[offset + ni * strides[0]];
    }
    else
    {
        T *__restrict d = data_->data();
        return d[offset + ni * strides[0]];
    }
}

template <typename T> auto ndarray<T>::at(std::ptrdiff_t i) const -> const_reference
{
    _require_data();
    if (shape.size() != 1) [[unlikely]]
    {
        throw std::invalid_argument("at() requires a 1D array");
    }
    const std::size_t ni = _norm_idx(i, shape[0]);
    // NB: const_reference is `bool` (by value) for bool arrays — returning
    // the vector<bool> proxy prvalue converts safely instead of dangling.
    return (*data_)[offset + ni * strides[0]];
}

template <typename T> typename ndarray<T>::value_type ndarray<T>::item() const
{
    if (_numel() != 1)
    {
        throw std::invalid_argument("can only convert an array of size 1 to a scalar");
    }
    _require_data();
    return (*data_)[offset];
}

template <typename T> auto ndarray<T>::operator()(std::ptrdiff_t i) -> reference
{
    _require_data();
    if (shape.size() != 1)
    {
        throw std::invalid_argument("operator()(i) requires a 1D array");
    }
    const std::size_t ni = _norm_idx(i, shape[0]);
    return (*data_)[offset + ni * strides[0]];
}

template <typename T> auto ndarray<T>::operator()(std::ptrdiff_t i) const -> const_reference
{
    _require_data();
    if (shape.size() != 1)
    {
        throw std::invalid_argument("operator()(i) requires a 1D array");
    }
    const std::size_t ni = _norm_idx(i, shape[0]);
    return (*data_)[offset + ni * strides[0]];
}

template <typename T> auto ndarray<T>::operator()(std::ptrdiff_t i, std::ptrdiff_t j) -> reference
{
    _require_data();
    if (shape.size() != 2)
    {
        throw std::invalid_argument("operator()(i, j) requires a 2D array");
    }
    const std::size_t ni = _norm_idx(i, shape[0]);
    const std::size_t nj = _norm_idx(j, shape[1]);
    return (*data_)[offset + ni * strides[0] + nj * strides[1]];
}

template <typename T> auto ndarray<T>::operator()(std::ptrdiff_t i, std::ptrdiff_t j) const -> const_reference
{
    _require_data();
    if (shape.size() != 2)
    {
        throw std::invalid_argument("operator()(i, j) requires a 2D array");
    }
    const std::size_t ni = _norm_idx(i, shape[0]);
    const std::size_t nj = _norm_idx(j, shape[1]);
    return (*data_)[offset + ni * strides[0] + nj * strides[1]];
}

template <typename T> auto ndarray<T>::at(std::ptrdiff_t i, std::ptrdiff_t j) -> reference
{
    _require_data();
    if (shape.size() != 2) [[unlikely]]
    {
        throw std::invalid_argument("at(i, j) requires a 2D array");
    }
    const std::size_t ni = _norm_idx(i, shape[0]);
    const std::size_t nj = _norm_idx(j, shape[1]);
    if constexpr (std::is_same_v<T, bool>)
    {
        return (*data_)[offset + ni * strides[0] + nj * strides[1]];
    }
    else
    {
        T *__restrict d = data_->data();
        return d[offset + ni * strides[0] + nj * strides[1]];
    }
}

template <typename T> auto ndarray<T>::at(std::ptrdiff_t i, std::ptrdiff_t j) const -> const_reference
{
    _require_data();
    if (shape.size() != 2) [[unlikely]]
    {
        throw std::invalid_argument("at(i, j) requires a 2D array");
    }
    const std::size_t ni = _norm_idx(i, shape[0]);
    const std::size_t nj = _norm_idx(j, shape[1]);
    return (*data_)[offset + ni * strides[0] + nj * strides[1]];
}

template <typename T>
template <typename... Args>
    requires(sizeof...(Args) >= 3 && (std::integral<Args> && ...))
auto ndarray<T>::operator()(Args... args) -> reference
{
    constexpr std::size_t N = sizeof...(Args);
    const std::ptrdiff_t raw[N] = {static_cast<std::ptrdiff_t>(args)...};
    std::array<std::size_t, N> idx{};
    if (N != shape.size())
    {
        throw std::invalid_argument("index dimensionality does not match array dimensions");
    }
    for (std::size_t d = 0; d < N; ++d)
    {
        idx[d] = _norm_idx(raw[d], shape[d]);
    }
    return get(idx);
}

template <typename T>
template <typename... Args>
    requires(sizeof...(Args) >= 3 && (std::integral<Args> && ...))
auto ndarray<T>::operator()(Args... args) const -> const_reference
{
    constexpr std::size_t N = sizeof...(Args);
    const std::ptrdiff_t raw[N] = {static_cast<std::ptrdiff_t>(args)...};
    std::array<std::size_t, N> idx{};
    if (N != shape.size())
    {
        throw std::invalid_argument("index dimensionality does not match array dimensions");
    }
    for (std::size_t d = 0; d < N; ++d)
    {
        idx[d] = _norm_idx(raw[d], shape[d]);
    }
    return get(idx);
}

template <typename T>
template <typename... Args>
    requires(sizeof...(Args) >= 3 && (std::integral<Args> && ...))
auto ndarray<T>::at(Args... args) -> reference
{
    return (*this)(args...);
}

template <typename T>
template <typename... Args>
    requires(sizeof...(Args) >= 3 && (std::integral<Args> && ...))
auto ndarray<T>::at(Args... args) const -> const_reference
{
    return (*this)(args...);
}

// Internals
template <typename T> void ndarray<T>::_validate_shape(const std::vector<int> &s)
{
    for (int d : s)
    {
        if (d < 0)
        {
            throw std::invalid_argument("ndarray: shape dimensions must be non-negative (got " + std::to_string(d) +
                                        ")");
        }
    }
}

template <typename T> auto ndarray<T>::_checked_numel(const std::vector<int> &s) -> std::size_t
{
    _validate_shape(s);
    std::size_t n = 1;
    for (int d : s)
    {
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

template <typename T> auto ndarray<T>::_numel() const noexcept -> std::size_t
{
    std::size_t n = 1;
    for (int d : shape)
    {
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

template <typename T> auto ndarray<T>::_c_strides(const std::vector<int> &s) -> std::vector<std::size_t>
{
    std::vector<std::size_t> st(s.size(), 1);
    std::size_t stride = 1;
    for (std::size_t i = s.size(); i-- > 0;)
    {
        st[i] = stride;
        stride *= static_cast<std::size_t>(s[i]);
    }
    return st;
}

template <typename T> auto ndarray<T>::_flat(const std::vector<std::size_t> &idx) const noexcept -> std::size_t
{
    return detail::flat_index(idx, strides, offset);
}

template <typename T> auto ndarray<T>::_flat_logical(std::size_t i) const noexcept -> std::size_t
{
    if (shape.empty() || i == 0) [[unlikely]]
        return offset;
    if (is_contiguous()) [[likely]]
        return offset + i;
    // Non-contiguous: compute flat without allocating vector
    std::size_t rem = i;
    std::size_t flat = offset;
    for (std::size_t d = shape.size(); d-- > 0;)
    {
        const std::size_t dim = static_cast<std::size_t>(shape[d]);
        if (dim == 0) [[unlikely]]
        {
            return offset; // unreachable for i < _numel(); fail safe, not loud
        }
        std::size_t coord = rem % dim;
        rem /= dim;
        flat += coord * strides[d];
    }
    return flat;
}

template <typename T> auto ndarray<T>::_shape_u() const -> std::vector<std::size_t>
{
    std::vector<std::size_t> u(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        u[i] = static_cast<std::size_t>(shape[i]);
    }
    return u;
}

template <typename T> auto ndarray<T>::_normalize_axis(int axis) const -> int
{
    const int nd = static_cast<int>(shape.size());
    if (axis < 0)
    {
        axis += nd;
    }
    if (axis < 0 || axis >= nd)
    {
        throw np::AxisError("axis " + std::to_string(axis - (axis < 0 ? nd : 0)) +
                            " is out of bounds for array of dimension " + std::to_string(nd));
    }
    return axis;
}

template <typename T> auto ndarray<T>::_norm_idx(std::ptrdiff_t i, std::ptrdiff_t dim) -> std::size_t
{
    if (dim < 0)
    {
        throw std::invalid_argument("_norm_idx: negative dimension extent");
    }
    if (i < 0)
    {
        i += dim;
    }
    if (i < 0 || i >= dim)
    {
        throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for dimension with size " +
                                std::to_string(dim));
    }
    return static_cast<std::size_t>(i);
}

template <typename T> void ndarray<T>::_require_data() const
{
    if (!data_) [[unlikely]]
    {
        throw std::runtime_error("ndarray has no data buffer (default-constructed array)");
    }
}

template <typename T> template <typename Fn> void ndarray<T>::_for_each_logical(Fn &&fn) const
{
    if (!data_) [[unlikely]]
        return;
    if (is_contiguous()) [[likely]]
    {
        // Logical range only: the buffer may be larger than the view, and
        // `offset` may be nonzero (offset views are contiguous by layout).
        const std::size_t n = _numel();
        if constexpr (std::is_same_v<T, bool>)
        {
            for (std::size_t i = 0; i < n; ++i)
                fn((*data_)[offset + i]);
        }
        else
        {
            const T *__restrict p = data_->data() + offset;
            for (std::size_t i = 0; i < n; ++i)
                fn(p[i]);
        }
        return;
    }
    detail::Odometer od(shape);
    while (!od.done())
    {
        fn((*data_)[_flat(od.idx())]);
        od.advance();
    }
}

template <typename T> template <typename Fn> void ndarray<T>::_for_each_indexed(Fn &&fn) const
{
    if (!data_)
    {
        return;
    }
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        fn(idx, (*data_)[_flat(idx)]);
        od.advance();
    }
}

template <typename T> void ndarray<T>::_finalize()
{
    _validate_shape(shape);
    strides = _c_strides(shape);
    if (!data_)
    {
        data_ =
            std::make_shared<std::vector<typename ndarray<T>::value_type>>(_numel(), typename ndarray<T>::value_type{});
    }
    if (type == dtype::void_)
    {
        type = dtype_of<T>;
    }
    order = matrix::Order::C;
}

template <typename T> template <typename U> void ndarray<T>::_flatten_initializer(std::initializer_list<U> list)
{
    for (const auto &val : list)
    {
        if constexpr (std::is_convertible_v<U, value_type>)
        {
            data_->push_back(static_cast<typename ndarray<T>::value_type>(val));
        }
        else if constexpr (std::is_same_v<U, std::initializer_list<double>> ||
                           std::is_same_v<U, std::initializer_list<int>>)
        {
            // Recursively flatten nested lists
            _flatten_initializer(val);
        }
        else
        {
            static_assert(std::is_convertible_v<U, value_type>, "Element type must be convertible to T");
        }
    }
}

// Reductions
template <typename T>
template <typename Acc, typename StepFn>
auto ndarray<T>::_reduce_axis(int axis, bool keepdims, std::optional<Acc> seed, StepFn &&step) const -> ndarray<Acc>
{
    axis = _normalize_axis(axis);
    const int nd = static_cast<int>(shape.size());

    std::vector<int> out_shape = shape;
    out_shape.erase(out_shape.begin() + axis);
    if (keepdims)
    {
        out_shape.insert(out_shape.begin() + axis, 1);
    }

    _require_data();
    if (shape[axis] == 0 && !seed.has_value())
    {
        // NumPy raises on min/max/arg-reduction of an empty slice; sum/prod
        // carry an explicit seed and correctly yield the identity instead.
        throw std::invalid_argument("reduction of empty slice with no seed (min/max/argmin/argmax)");
    }
    ndarray<Acc> out(out_shape);
    if (seed.has_value())
    {
        std::fill(out.data().begin(), out.data().end(), *seed);
    }
    std::vector<std::uint8_t> first(out.size(), seed.has_value() ? 0u : 1u);

    std::vector<std::size_t> out_idx;
    out_idx.reserve(nd - 1);
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        out_idx.clear();
        for (int d = 0; d < nd; ++d)
        {
            if (d != axis)
            {
                out_idx.push_back(idx[d]);
            }
            else if (keepdims)
            {
                out_idx.push_back(0);
            }
        }
        const std::size_t of = detail::flat_index(out_idx, out.strides, 0);
        const value_type value = (*data_)[_flat(idx)];
        if (first[of])
        {
            out.data()[of] = static_cast<Acc>(value);
            first[of] = 0;
        }
        else
        {
            step(out.data()[of], value);
        }
        od.advance();
    }
    return out;
}

template <typename T>
auto ndarray<T>::sum() const -> std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>
{
    using Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        if (is_contiguous() && data_)
        {
            // SIMD sum is O(n) with 2-8x speedup for contiguous float/double
            return static_cast<Acc>(simd::sum_vectorized(data_->data() + offset, _numel()));
        }
    }
    Acc total{};
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { total += v; });
    return total;
}

template <typename T> template <typename Acc> auto ndarray<T>::sum(int axis, bool keepdims) const -> ndarray<Acc>
{
    return _reduce_axis<Acc>(axis, keepdims, Acc(0),
                             [](Acc &acc, const typename ndarray<T>::value_type &v) { acc += v; });
}

template <typename T>
template <typename Acc>
auto ndarray<T>::sum(std::optional<int> axis, bool keepdims) const -> ndarray<Acc>
{
    if (!axis.has_value())
    {
        ndarray<Acc> out(std::vector<int>{});
        out.data()[0] = sum();
        return out;
    }
    return sum<Acc>(*axis, keepdims);
}

template <typename T>
auto ndarray<T>::prod() const -> std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>
{
    using Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>;
    Acc total{1};
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { total *= v; });
    return total;
}

template <typename T> template <typename Acc> auto ndarray<T>::prod(int axis, bool keepdims) const -> ndarray<Acc>
{
    return _reduce_axis<Acc>(axis, keepdims, Acc(1),
                             [](Acc &acc, const typename ndarray<T>::value_type &v) { acc *= v; });
}

template <typename T>
template <typename Acc>
auto ndarray<T>::prod(std::optional<int> axis, bool keepdims) const -> ndarray<Acc>
{
    if (!axis.has_value())
    {
        ndarray<Acc> out(std::vector<int>{});
        out.data()[0] = prod();
        return out;
    }
    return prod<Acc>(*axis, keepdims);
}

template <typename T> typename ndarray<T>::value_type ndarray<T>::min() const
{
    if constexpr (detail::is_complex_v<value_type>)
    {
        // NumPy raises TypeError ordering complex values; fail loudly
        // instead of a hard < operator error.
        throw std::invalid_argument("min() is not defined for complex arrays (no total order)");
    }
    else
    {
        if (_numel() == 0)
        {
            throw std::runtime_error("min() on empty array");
        }
        std::optional<T> best;
        _for_each_logical([&](const typename ndarray<T>::value_type &v) {
            // NaN propagates (NumPy semantics); first NaN sticks.
            if (!best.has_value() || v < *best || (detail::isnan_val(v) && !detail::isnan_val(*best)))
            {
                best = v;
            }
        });
        return *best;
    }
}

template <typename T> auto ndarray<T>::min(int axis, bool keepdims) const -> ndarray<T>
{
    if constexpr (detail::is_complex_v<value_type>)
    {
        throw std::invalid_argument("min() is not defined for complex arrays (no total order)");
    }
    else
    {
        // Custom step (not std::min): NaN must propagate like the scalar path.
        return _reduce_axis<T>(axis, keepdims, std::nullopt, [](T &acc, const typename ndarray<T>::value_type &v) {
            if (detail::isnan_val(acc))
                return;
            if (detail::isnan_val(v) || v < acc)
                acc = v;
        });
    }
}

template <typename T> auto ndarray<T>::min(std::optional<int> axis, bool keepdims) const -> ndarray<T>
{
    if (!axis.has_value())
    {
        ndarray<T> out(std::vector<int>{});
        out.data()[0] = min();
        return out;
    }
    return min(*axis, keepdims);
}

template <typename T> typename ndarray<T>::value_type ndarray<T>::max() const
{
    if constexpr (detail::is_complex_v<value_type>)
    {
        throw std::invalid_argument("max() is not defined for complex arrays (no total order)");
    }
    else
    {
        if (_numel() == 0)
        {
            throw std::runtime_error("max() on empty array");
        }
        std::optional<T> best;
        _for_each_logical([&](const typename ndarray<T>::value_type &v) {
            // NaN propagates (NumPy semantics); first NaN sticks.
            if (!best.has_value() || v > *best || (detail::isnan_val(v) && !detail::isnan_val(*best)))
            {
                best = v;
            }
        });
        return *best;
    }
}

template <typename T> auto ndarray<T>::max(int axis, bool keepdims) const -> ndarray<T>
{
    if constexpr (detail::is_complex_v<value_type>)
    {
        throw std::invalid_argument("max() is not defined for complex arrays (no total order)");
    }
    else
    {
        // Custom step (not std::max): NaN must propagate like the scalar path.
        return _reduce_axis<T>(axis, keepdims, std::nullopt, [](T &acc, const typename ndarray<T>::value_type &v) {
            if (detail::isnan_val(acc))
                return;
            if (detail::isnan_val(v) || v > acc)
                acc = v;
        });
    }
}

template <typename T> auto ndarray<T>::max(std::optional<int> axis, bool keepdims) const -> ndarray<T>
{
    if (!axis.has_value())
    {
        ndarray<T> out(std::vector<int>{});
        out.data()[0] = max();
        return out;
    }
    return max(*axis, keepdims);
}

template <typename T> typename ndarray<T>::value_type ndarray<T>::ptp() const
{
    return max() - min();
}

template <typename T> auto ndarray<T>::ptp(int axis, bool keepdims) const -> ndarray<T>
{
    const ndarray<T> mx = max(axis, keepdims);
    const ndarray<T> mn = min(axis, keepdims);
    ndarray<T> out(mx.shape);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        out.data()[i] = mx.data()[i] - mn.data()[i];
    }
    return out;
}

template <typename T> auto ndarray<T>::ptp(std::optional<int> axis, bool keepdims) const -> ndarray<T>
{
    if (!axis.has_value())
    {
        ndarray<T> out(std::vector<int>{});
        out.data()[0] = ptp();
        return out;
    }
    return ptp(*axis, keepdims);
}

template <typename T> auto ndarray<T>::mean() const -> typename _mean_type<typename ndarray<T>::value_type>::type
{
    using MeanT = typename _mean_type<T>::type;
    if (_numel() == 0)
    {
        throw std::runtime_error("mean() on empty array");
    }
    // Complex accumulation so complex inputs average correctly (NumPy:
    // mean(complex) is complex); real inputs are unaffected.
    std::complex<long double> total(0.0L, 0.0L);
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        if constexpr (detail::is_complex_v<value_type>)
        {
            total += std::complex<long double>(static_cast<long double>(v.real()), static_cast<long double>(v.imag()));
        }
        else
        {
            total += std::complex<long double>(static_cast<long double>(v), 0.0L);
        }
    });
    const std::complex<long double> avg = total / static_cast<long double>(_numel());
    if constexpr (detail::is_complex_v<MeanT>)
    {
        return MeanT(static_cast<typename MeanT::value_type>(avg.real()),
                     static_cast<typename MeanT::value_type>(avg.imag()));
    }
    else
    {
        return static_cast<MeanT>(avg.real());
    }
}

template <typename T> auto ndarray<T>::mean(int axis, bool keepdims) const -> ndarray<typename _mean_type<T>::type>
{
    using MeanT = typename _mean_type<T>::type;
    axis = _normalize_axis(axis);
    const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);
    auto s = _reduce_axis<MeanT>(axis, keepdims, MeanT(0),
                                 [](MeanT &acc, const typename ndarray<T>::value_type &v) { acc += v; });
    for (auto &v : s.data())
    {
        v /= static_cast<MeanT>(axis_len);
    }
    return s;
}

template <typename T>
auto ndarray<T>::mean(std::optional<int> axis, bool keepdims) const -> ndarray<typename _mean_type<T>::type>
{
    if (!axis.has_value())
    {
        ndarray<typename _mean_type<T>::type> out(std::vector<int>{});
        out.data()[0] = mean();
        return out;
    }
    return mean(*axis, keepdims);
}

template <typename T>
template <typename MeanT>
auto ndarray<T>::_var_axis(int axis, bool keepdims) const -> ndarray<MeanT>
{
    axis = _normalize_axis(axis);
    const int nd = static_cast<int>(shape.size());

    std::vector<int> out_shape = shape;
    out_shape.erase(out_shape.begin() + axis);
    if (keepdims)
    {
        out_shape.insert(out_shape.begin() + axis, 1);
    }
    _require_data();
    ndarray<MeanT> out(out_shape);
    const std::size_t n_out = out.size();
    // Complex-capable Welford: complex running mean, real M2 (variance is
    // mean squared magnitude, always real).
    std::vector<std::complex<long double>> m(n_out);
    std::vector<long double> m2(n_out, 0.0L);
    std::vector<std::size_t> count(n_out, 0);

    std::vector<std::size_t> out_idx;
    out_idx.reserve(nd - 1);
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        out_idx.clear();
        for (int d = 0; d < nd; ++d)
        {
            if (d != axis)
            {
                out_idx.push_back(idx[d]);
            }
            else if (keepdims)
            {
                out_idx.push_back(0);
            }
        }
        const std::size_t of = detail::flat_index(out_idx, out.strides, 0);
        const value_type &vv = (*data_)[_flat(idx)];
        std::complex<long double> x;
        if constexpr (detail::is_complex_v<value_type>)
        {
            x = std::complex<long double>(static_cast<long double>(vv.real()), static_cast<long double>(vv.imag()));
        }
        else
        {
            x = std::complex<long double>(static_cast<long double>(vv), 0.0L);
        }
        ++count[of];
        const std::complex<long double> delta = x - m[of];
        m[of] += delta / static_cast<long double>(count[of]);
        m2[of] += std::real(delta * std::conj(x - m[of]));
        od.advance();
    }
    for (std::size_t i = 0; i < n_out; ++i)
    {
        if (count[i] == 0)
        {
            // Empty slice: NaN like NumPy (previously silent 0).
            out.data()[i] = static_cast<MeanT>(std::numeric_limits<double>::quiet_NaN());
        }
        else
        {
            out.data()[i] = static_cast<MeanT>(m2[i] / static_cast<long double>(count[i]));
        }
    }
    return out;
}

template <typename T> auto ndarray<T>::var() const -> typename _var_type<typename ndarray<T>::value_type>::type
{
    using VarT = typename _var_type<T>::type;
    if (_numel() == 0)
    {
        throw std::runtime_error("var() on empty array");
    }
    // Welford with complex mean: variance is mean squared magnitude, so the
    // result is always real (NumPy: var(complex128) -> float64).
    std::complex<long double> m(0.0L, 0.0L);
    long double m2 = 0.0L;
    std::size_t count = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        ++count;
        std::complex<long double> x;
        if constexpr (detail::is_complex_v<value_type>)
        {
            x = std::complex<long double>(static_cast<long double>(v.real()), static_cast<long double>(v.imag()));
        }
        else
        {
            x = std::complex<long double>(static_cast<long double>(v), 0.0L);
        }
        const std::complex<long double> delta = x - m;
        m += delta / static_cast<long double>(count);
        m2 += std::real(delta * std::conj(x - m));
    });
    return static_cast<VarT>(m2 / static_cast<long double>(count));
}

template <typename T> auto ndarray<T>::var(int axis, bool keepdims) const -> ndarray<typename _var_type<T>::type>
{
    return _var_axis<typename _var_type<T>::type>(axis, keepdims);
}

template <typename T>
auto ndarray<T>::var(std::optional<int> axis, bool keepdims) const -> ndarray<typename _var_type<T>::type>
{
    if (!axis.has_value())
    {
        ndarray<typename _mean_type<T>::type> out(std::vector<int>{});
        out.data()[0] = var();
        return out;
    }
    return var(*axis, keepdims);
}

template <typename T> auto ndarray<T>::std() const -> typename _var_type<typename ndarray<T>::value_type>::type
{
    using VarT = typename _var_type<T>::type;
    return static_cast<VarT>(std::sqrt(var()));
}

template <typename T> auto ndarray<T>::std(int axis, bool keepdims) const -> ndarray<typename _var_type<T>::type>
{
    using VarT = typename _var_type<T>::type;
    auto v = _var_axis<VarT>(axis, keepdims);
    for (auto &x : v.data())
    {
        x = static_cast<VarT>(std::sqrt(x));
    }
    return v;
}

template <typename T>
auto ndarray<T>::std(std::optional<int> axis, bool keepdims) const -> ndarray<typename _var_type<T>::type>
{
    using VarT = typename _var_type<T>::type;
    if (!axis.has_value())
    {
        ndarray<VarT> out(std::vector<int>{});
        out.data()[0] = std();
        return out;
    }
    return std(*axis, keepdims);
}

template <typename T> bool ndarray<T>::all() const
{
    bool result = true;
    _for_each_logical(
        [&](const typename ndarray<T>::value_type &v) { result = result && (v != typename ndarray<T>::value_type{}); });
    return result;
}

template <typename T> auto ndarray<T>::all(int axis, bool keepdims) const -> ndarray<bool>
{
    return _reduce_axis<bool>(axis, keepdims, std::optional<bool>(true),
                              [](bool &acc, const typename ndarray<T>::value_type &v) {
                                  acc = acc && (v != typename ndarray<T>::value_type{});
                              });
}

template <typename T> auto ndarray<T>::all(std::optional<int> axis, bool keepdims) const -> ndarray<bool>
{
    if (!axis.has_value())
    {
        ndarray<bool> out(std::vector<int>{});
        out.data()[0] = all();
        return out;
    }
    return all(*axis, keepdims);
}

template <typename T> bool ndarray<T>::any() const
{
    bool result = false;
    _for_each_logical(
        [&](const typename ndarray<T>::value_type &v) { result = result || (v != typename ndarray<T>::value_type{}); });
    return result;
}

template <typename T> auto ndarray<T>::any(int axis, bool keepdims) const -> ndarray<bool>
{
    return _reduce_axis<bool>(axis, keepdims, std::optional<bool>(false),
                              [](bool &acc, const typename ndarray<T>::value_type &v) {
                                  acc = acc || (v != typename ndarray<T>::value_type{});
                              });
}

template <typename T> auto ndarray<T>::any(std::optional<int> axis, bool keepdims) const -> ndarray<bool>
{
    if (!axis.has_value())
    {
        ndarray<bool> out(std::vector<int>{});
        out.data()[0] = any();
        return out;
    }
    return any(*axis, keepdims);
}

template <typename T>
template <typename Cmp>
auto ndarray<T>::_arg_reduce_axis(int axis, bool keepdims, Cmp &&cmp) const -> ndarray<std::size_t>
{
    _require_data();
    axis = _normalize_axis(axis);
    if (shape[axis] == 0)
    {
        // NumPy raises on argmin/argmax of an empty sequence (previously
        // silently returned index 0).
        throw std::invalid_argument("argmin/argmax of empty slice");
    }
    const int nd = static_cast<int>(shape.size());

    std::vector<int> out_shape = shape;
    out_shape.erase(out_shape.begin() + axis);
    if (keepdims)
    {
        out_shape.insert(out_shape.begin() + axis, 1);
    }
    ndarray<std::size_t> out(out_shape);
    std::vector<std::uint8_t> first(out.size(), 1u);
    std::vector<value_type> best_val(out.size(), typename ndarray<T>::value_type{});
    std::vector<std::size_t> best_pos(out.size(), 0);

    std::vector<std::size_t> out_idx;
    out_idx.reserve(nd - 1);
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        out_idx.clear();
        for (int d = 0; d < nd; ++d)
        {
            if (d != axis)
            {
                out_idx.push_back(idx[d]);
            }
            else if (keepdims)
            {
                out_idx.push_back(0);
            }
        }
        const std::size_t of = detail::flat_index(out_idx, out.strides, 0);
        const value_type value = (*data_)[_flat(idx)];
        if (first[of] || cmp(value, best_val[of]))
        {
            first[of] = 0;
            best_val[of] = value;
            best_pos[of] = idx[axis];
        }
        od.advance();
    }
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        out.data()[i] = best_pos[i];
    }
    return out;
}

template <typename T> std::size_t ndarray<T>::argmax() const
{
    _require_data();
    if (_numel() == 0)
    {
        throw std::runtime_error("argmax() on empty array");
    }
    std::size_t best = 0;
    std::size_t pos = 0;
    std::optional<T> best_val;
    detail::Odometer od(shape);
    while (!od.done())
    {
        const T v = (*data_)[_flat(od.idx())];
        // NumPy treats NaN as larger than everything (first NaN sticks);
        // complex orders lexicographically by (real, imag).
        const bool wins = !best_val.has_value() || detail::value_less(*best_val, v) ||
                          (detail::isnan_val(v) && !detail::isnan_val(*best_val));
        if (wins)
        {
            best_val = v;
            best = pos;
        }
        ++pos;
        od.advance();
    }
    return best;
}

template <typename T> auto ndarray<T>::argmax(int axis, bool keepdims) const -> ndarray<std::size_t>
{
    // NaN beats everything (first NaN sticks); complex orders (real, imag).
    return _arg_reduce_axis(axis, keepdims, [](const typename ndarray<T>::value_type &v, const T &b) {
        return detail::value_less(b, v) || (detail::isnan_val(v) && !detail::isnan_val(b));
    });
}

template <typename T> auto ndarray<T>::argmax(std::optional<int> axis, bool keepdims) const -> ndarray<std::size_t>
{
    if (!axis.has_value())
    {
        ndarray<std::size_t> out(std::vector<int>{});
        out.data()[0] = argmax();
        return out;
    }
    return argmax(*axis, keepdims);
}

template <typename T> std::size_t ndarray<T>::argmin() const
{
    _require_data();
    if (_numel() == 0)
    {
        throw std::runtime_error("argmin() on empty array");
    }
    std::size_t best = 0;
    std::size_t pos = 0;
    std::optional<T> best_val;
    detail::Odometer od(shape);
    while (!od.done())
    {
        const T v = (*data_)[_flat(od.idx())];
        const bool wins = !best_val.has_value() || detail::value_less(v, *best_val) ||
                          (detail::isnan_val(v) && !detail::isnan_val(*best_val));
        if (wins)
        {
            best_val = v;
            best = pos;
        }
        ++pos;
        od.advance();
    }
    return best;
}

template <typename T> auto ndarray<T>::argmin(int axis, bool keepdims) const -> ndarray<std::size_t>
{
    // NaN beats everything (first NaN sticks); complex orders (real, imag).
    return _arg_reduce_axis(axis, keepdims, [](const typename ndarray<T>::value_type &v, const T &b) {
        return detail::value_less(v, b) || (detail::isnan_val(v) && !detail::isnan_val(b));
    });
}

template <typename T> auto ndarray<T>::argmin(std::optional<int> axis, bool keepdims) const -> ndarray<std::size_t>
{
    if (!axis.has_value())
    {
        ndarray<std::size_t> out(std::vector<int>{});
        out.data()[0] = argmin();
        return out;
    }
    return argmin(*axis, keepdims);
}

template <typename T>
template <typename Acc, typename Fn>
auto ndarray<T>::_cum_axis(int axis, Fn &&fn) const -> ndarray<Acc>
{
    _require_data();
    axis = _normalize_axis(axis);
    const int nd = static_cast<int>(shape.size());
    const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);
    if (axis_len == 0)
    {
        // Empty axis: nothing to accumulate (previously divided by zero).
        return ndarray<Acc>(shape);
    }

    ndarray<Acc> out(shape);
    std::vector<int> reduced_shape = shape;
    reduced_shape.erase(reduced_shape.begin() + axis);
    const std::vector<std::size_t> red_strides = _c_strides(reduced_shape);
    const std::size_t n_slots = _numel() / axis_len;
    std::vector<Acc> acc(n_slots, Acc{});

    std::vector<std::size_t> slot;
    slot.reserve(nd - 1);
    std::vector<std::size_t> out_idx;
    out_idx.reserve(nd);
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        out_idx = idx;
        slot.clear();
        for (int d = 0; d < nd; ++d)
        {
            if (d != axis)
            {
                slot.push_back(idx[d]);
            }
        }
        const std::size_t slot_of = detail::flat_index(slot, red_strides, 0);
        acc[slot_of] = fn(acc[slot_of], (*data_)[_flat(idx)]);
        out.data()[detail::flat_index(out_idx, out.strides, 0)] = acc[slot_of];
        od.advance();
    }
    return out;
}

template <typename T>
auto ndarray<T>::cumsum() const -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
{
    using Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>;
    ndarray<Acc> out(std::vector<int>{static_cast<int>(_numel())});
    Acc running{};
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        running += v;
        out.data()[i++] = running;
    });
    return out;
}

template <typename T>
auto ndarray<T>::cumsum(int axis) const
    -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
{
    using Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>;
    return _cum_axis<Acc>(axis, [](Acc &acc, const typename ndarray<T>::value_type &v) { return acc + v; });
}

template <typename T>
auto ndarray<T>::cumsum(std::optional<int> axis) const
    -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
{
    if (!axis.has_value())
    {
        return cumsum();
    }
    return cumsum(*axis);
}

template <typename T>
auto ndarray<T>::cumprod() const -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
{
    using Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>;
    ndarray<Acc> out(std::vector<int>{static_cast<int>(_numel())});
    Acc running{1};
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        running *= v;
        out.data()[i++] = running;
    });
    return out;
}

template <typename T>
auto ndarray<T>::cumprod(int axis) const
    -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
{
    using Acc = std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>;
    return _cum_axis<Acc>(axis, [](Acc &acc, const typename ndarray<T>::value_type &v) { return acc * v; });
}

template <typename T>
auto ndarray<T>::cumprod(std::optional<int> axis) const
    -> ndarray<std::conditional_t<std::is_same_v<value_type, bool>, std::int64_t, T>>
{
    if (!axis.has_value())
    {
        return cumprod();
    }
    return cumprod(*axis);
}

// Sorting / searching
template <typename T> void ndarray<T>::sort(std::optional<int> axis)
{
    sort(axis.value_or(-1));
}

template <typename T> void ndarray<T>::sort(int axis)
{
    static_assert(std::is_copy_constructible_v<value_type>, "sort: value_type must be copy constructible");
    _require_data();
    static_assert(std::is_move_constructible_v<value_type>, "sort: value_type must be move constructible");
    static_assert(std::is_default_constructible_v<value_type>, "sort: value_type must be default constructible");
    static_assert(std::is_copy_assignable_v<value_type>, "sort: value_type must be copy assignable");
    axis = _normalize_axis(axis);
    const int nd = static_cast<int>(shape.size());
    const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);

    std::vector<int> slice_shape = shape;
    slice_shape.erase(slice_shape.begin() + axis);

#ifdef NP_USE_THREADING
    // Collect slices for parallel dispatch
    std::vector<std::vector<std::size_t>> all_slices;
    {
        detail::Odometer od(slice_shape);
        while (!od.done())
        {
            all_slices.push_back(od.idx());
            od.advance();
        }
        if (all_slices.empty())
        {
            all_slices.push_back({});
        }
    }
    const std::size_t n_slices = all_slices.size();
    auto do_slice = [&](std::size_t si) {
        const auto &s = all_slices[si];
        std::vector<std::size_t> full(nd);
        std::vector<typename ndarray<T>::value_type> work(axis_len);
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                full[d] = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += full[d] * strides[d];
            }
            work[p] = (*data_)[offset + f];
        }
        if constexpr (std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
        {
            if (axis_len >= 64)
            {
                detail::radix_sort_integral(work);
            }
            else
            {
                std::sort(work.begin(), work.end());
            }
        }
        else if constexpr (detail::is_complex_v<T>)
        {
            std::sort(work.begin(), work.end(), [](const value_type &a, const value_type &b) {
                if (a.real() != b.real())
                    return a.real() < b.real();
                return a.imag() < b.imag();
            });
        }
        else
        {
            std::sort(work.begin(), work.end());
        }
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                full[d] = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += full[d] * strides[d];
            }
            (*data_)[offset + f] = work[p];
        }
    };
    if (n_slices > 4)
    {
        detail::maybe_parallel_for(0, n_slices, do_slice);
    }
    else
    {
        for (std::size_t si = 0; si < n_slices; ++si)
        {
            do_slice(si);
        }
    }
#else
    std::vector<std::size_t> full(nd);
    detail::Odometer od(slice_shape);
    while (!od.done())
    {
        const auto &s = od.idx();
        std::vector<typename ndarray<T>::value_type> work(axis_len);
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                full[d] = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += full[d] * strides[d];
            }
            work[p] = (*data_)[offset + f];
        }
        // Micro-optimized: radix O(n) for integral, pdqsort O(n log n) otherwise
        if constexpr (std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
        {
            if (axis_len >= 64)
            {
                detail::radix_sort_integral(work);
            }
            else
            {
                std::sort(work.begin(), work.end());
            }
        }
        else if constexpr (detail::is_complex_v<T>)
        {
            std::sort(work.begin(), work.end(), [](const value_type &a, const value_type &b) {
                if (a.real() != b.real())
                    return a.real() < b.real();
                return a.imag() < b.imag();
            });
        }
        else
        {
            std::sort(work.begin(), work.end());
        }
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                full[d] = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += full[d] * strides[d];
            }
            (*data_)[offset + f] = work[p];
        }
        od.advance();
    }
#endif
}

template <typename T> auto ndarray<T>::sorted(int axis) const -> ndarray<T>
{
    ndarray<T> out = *this;
    out.sort(axis);
    return out;
}

template <typename T> auto ndarray<T>::sorted(std::optional<int> axis) const -> ndarray<T>
{
    if (!axis.has_value())
    {
        // NumPy np.sort(None) flattens; sort a flattened copy (ravel() may
        // alias storage, so flatten() guarantees we never mutate *this).
        ndarray flat = flatten();
        flat.sort(0);
        return flat;
    }
    return sorted(*axis);
}

template <typename T> auto ndarray<T>::argsort(std::optional<int> axis) const -> ndarray<std::size_t>
{
    if (!axis.has_value())
    {
        return flatten().argsort(0);
    }
    return argsort(*axis);
}

template <typename T> auto ndarray<T>::argsort(int axis) const -> ndarray<std::size_t>
{
    static_assert(std::is_copy_constructible_v<value_type>, "argsort: value_type must be copy constructible");
    _require_data();
    static_assert(std::is_move_constructible_v<value_type>, "argsort: value_type must be move constructible");
    static_assert(std::is_default_constructible_v<value_type>, "argsort: value_type must be default constructible");
    axis = _normalize_axis(axis);
    const int nd = static_cast<int>(shape.size());
    const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);

    ndarray<std::size_t> out(shape);
    std::vector<int> slice_shape = shape;
    slice_shape.erase(slice_shape.begin() + axis);

#ifdef NP_USE_THREADING
    // Collect slices for parallel dispatch
    std::vector<std::vector<std::size_t>> all_slices;
    {
        detail::Odometer od(slice_shape);
        while (!od.done())
        {
            all_slices.push_back(od.idx());
            od.advance();
        }
        if (all_slices.empty())
        {
            all_slices.push_back({});
        }
    }
    const std::size_t n_slices = all_slices.size();
    auto do_slice = [&](std::size_t si) {
        const auto &s = all_slices[si];
        std::vector<std::pair<std::size_t, value_type>> work;
        work.reserve(axis_len);
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t coord = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += coord * strides[d];
            }
            work.emplace_back(p, (*data_)[offset + f]);
        }
        if constexpr (std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
        {
            if (axis_len >= 64)
            {
                detail::radix_sort_pair(work);
            }
            else
            {
                std::sort(work.begin(), work.end(), detail::pair_second_less<typename decltype(work)::value_type>);
            }
        }
        else
        {
            std::sort(work.begin(), work.end(), detail::pair_second_less<typename decltype(work)::value_type>);
        }
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t coord = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += coord * out.strides[d];
            }
            out.data()[f] = work[p].first;
        }
    };
    if (n_slices > 4)
    {
        detail::maybe_parallel_for(0, n_slices, do_slice);
    }
    else
    {
        for (std::size_t si = 0; si < n_slices; ++si)
        {
            do_slice(si);
        }
    }
    return out;
#else
    std::vector<std::pair<std::size_t, value_type>> work;
    work.reserve(axis_len);
    detail::Odometer od(slice_shape);
    while (!od.done())
    {
        const auto &s = od.idx();
        work.clear();
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t coord = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += coord * strides[d];
            }
            work.emplace_back(p, (*data_)[offset + f]);
        }
        // Micro-optimized: radix for integral keys, pdqsort otherwise
        if constexpr (std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
        {
            if (axis_len >= 64)
            {
                detail::radix_sort_pair(work);
            }
            else
            {
                std::sort(work.begin(), work.end(), detail::pair_second_less<typename decltype(work)::value_type>);
            }
        }
        else
        {
            std::sort(work.begin(), work.end(), detail::pair_second_less<typename decltype(work)::value_type>);
        }
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t coord = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += coord * out.strides[d];
            }
            out.data()[f] = work[p].first;
        }
        od.advance();
    }
    return out;
#endif
}

template <typename T>
auto ndarray<T>::argpartition(std::size_t kth, std::optional<int> axis) const -> ndarray<std::size_t>
{
    if (!axis.has_value())
    {
        return flatten().argpartition(kth, 0);
    }
    return argpartition(kth, *axis);
}

template <typename T> auto ndarray<T>::argpartition(std::size_t kth, int axis) const -> ndarray<std::size_t>
{
    axis = _normalize_axis(axis);
    const int nd = static_cast<int>(shape.size());
    const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);
    if (kth >= axis_len)
    {
        throw std::out_of_range("kth out of bounds");
    }
    _require_data();

    ndarray<std::size_t> out(shape);
    std::vector<int> slice_shape = shape;
    slice_shape.erase(slice_shape.begin() + axis);

    std::vector<std::pair<std::size_t, T>> work;
    work.reserve(axis_len);
    detail::Odometer od(slice_shape);
    while (!od.done())
    {
        const auto &s = od.idx();
        work.clear();
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t coord = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += coord * strides[d];
            }
            work.emplace_back(p, (*data_)[offset + f]);
        }
        std::nth_element(work.begin(), work.begin() + kth, work.end(),
                         detail::pair_second_less<typename decltype(work)::value_type>);
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t coord = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                f += coord * out.strides[d];
            }
            out.data()[f] = work[p].first;
        }
        od.advance();
    }
    return out;
}

template <typename T>
std::size_t ndarray<T>::searchsorted(const typename ndarray<T>::value_type &value, bool side_right) const
{
    static_assert(std::is_copy_constructible_v<value_type>, "searchsorted: value_type must be copy constructible");
    static_assert(std::is_default_constructible_v<value_type>,
                  "searchsorted: value_type must be default constructible");
    static_assert(std::is_copy_assignable_v<value_type>, "searchsorted: value_type must be copy assignable");
    _require_data();
    if (shape.size() != 1)
    {
        throw std::invalid_argument("searchsorted requires a 1D array");
    }
    // Ordering matches sort(): plain `<` except complex, which orders
    // lexicographically by (real, imag) — std::complex has no operator<.
    constexpr auto cmp = detail::value_less<value_type>;
    // Micro-optimized: O(log n) binary search, contiguous fast path with raw pointer
    // (vector<bool> has no data pointer — strided path below handles it).
    const std::size_t n = static_cast<std::size_t>(shape[0]);
    if constexpr (!std::is_same_v<T, bool>)
    {
        if (is_contiguous())
        {
            const T *base = data().data() + offset;
            const T *lo = base;
            const T *hi = base + n;
            const T *it = side_right ? std::upper_bound(lo, hi, value, cmp) : std::lower_bound(lo, hi, value, cmp);
            return static_cast<std::size_t>(it - lo);
        }
    }
    // Non-contiguous (view) -> manual binary search via strided access, still O(log n)
    std::size_t lo = 0, hi = n;
    while (lo < hi)
    {
        std::size_t mid = lo + (hi - lo) / 2;
        const value_type mid_val = (*data_)[_flat_logical(mid)];
        const bool advance = side_right ? !cmp(value, mid_val) : cmp(mid_val, value);
        if (advance)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

template <typename T>
std::size_t ndarray<T>::searchsorted(const typename ndarray<T>::value_type &value, std::optional<bool> side_right) const
{
    return searchsorted(value, side_right.value_or(false));
}

template <typename T>
template <typename U>
auto ndarray<T>::searchsorted(const ndarray<U> &values) const -> ndarray<std::size_t>
{
    static_assert(std::is_copy_constructible_v<value_type>, "searchsorted: value_type must be copy constructible");
    static_assert(std::is_default_constructible_v<value_type>,
                  "searchsorted: value_type must be default constructible");
    if (shape.size() != 1)
    {
        throw std::invalid_argument("searchsorted requires a 1D array");
    }
    ndarray<std::size_t> out(std::vector<int>{static_cast<int>(values.size())});
#ifdef NP_USE_THREADING
    auto do_search = [&](std::size_t i) { out.data()[i] = searchsorted(values.data()[values._flat_logical(i)]); };
    detail::maybe_parallel_for(0, values.size(), do_search);
#else
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        out.data()[i] = searchsorted(values.data()[values._flat_logical(i)]);
    }
#endif
    return out;
}

// Shape manipulation
template <typename T> auto ndarray<T>::reshape(const std::vector<int> &new_shape) const -> ndarray
{
    std::vector<int> resolved = new_shape;
    int neg_count = 0;
    for (int d : resolved)
    {
        if (d == -1)
        {
            ++neg_count;
        }
        else if (d < 0)
        {
            throw std::invalid_argument("reshape: shape dimensions must be non-negative (except a "
                                        "single -1), got " +
                                        std::to_string(d));
        }
    }
    if (neg_count > 1)
    {
        throw std::invalid_argument("at most one dimension may be -1");
    }
    if (neg_count == 1)
    {
        std::size_t known = 1;
        int neg_at = 0;
        for (std::size_t i = 0; i < resolved.size(); ++i)
        {
            if (resolved[i] == -1)
            {
                neg_at = static_cast<int>(i);
            }
            else
            {
                known *= static_cast<std::size_t>(resolved[i]);
            }
        }
        if (known == 0 || _numel() % known != 0)
        {
            throw std::invalid_argument("cannot infer -1 dimension");
        }
        resolved[neg_at] = static_cast<int>(_numel() / known);
    }
    std::size_t total = 1;
    for (int d : resolved)
    {
        total *= static_cast<std::size_t>(d);
    }
    if (total != _numel())
    {
        throw std::invalid_argument("cannot reshape array of size " + std::to_string(_numel()) +
                                    " into shape with total size " + std::to_string(total));
    }
    if (is_contiguous())
    {
        // View sharing storage
        return ndarray(data_, resolved, _c_strides(resolved), type, order, offset);
    }
    // Copy path
    ndarray out(resolved, type);
    std::copy(begin(), end(), out.begin());
    return out;
}

template <typename T> auto ndarray<T>::transpose() const -> ndarray
{
    if (shape.empty())
    {
        return *this;
    }
    std::vector<int> p(shape.size());
    std::vector<std::size_t> s(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        p[i] = shape[shape.size() - 1 - i];
        s[i] = strides[strides.size() - 1 - i];
    }
    matrix::Order o = (order == matrix::Order::C) ? matrix::Order::F : matrix::Order::C;
    return ndarray(data_, std::move(p), std::move(s), type, o, offset);
}

template <typename T> auto ndarray<T>::transpose(const std::vector<int> &perm) const -> ndarray
{
    if (perm.size() != shape.size())
    {
        throw std::invalid_argument("permutation length must equal ndim");
    }
    std::vector<int> p(perm.size());
    std::vector<std::size_t> s(perm.size());
    std::vector<std::uint8_t> seen(perm.size(), 0);
    for (std::size_t i = 0; i < perm.size(); ++i)
    {
        int a = perm[i];
        if (a < 0)
        {
            a += static_cast<int>(perm.size());
        }
        if (a < 0 || a >= static_cast<int>(perm.size()) || seen[a])
        {
            throw std::invalid_argument("invalid permutation");
        }
        seen[a] = 1;
        p[i] = shape[a];
        s[i] = strides[a];
    }
    return ndarray(data_, std::move(p), std::move(s), type, order, offset);
}

template <typename T> auto ndarray<T>::swapaxes(int axis1, int axis2) const -> ndarray
{
    axis1 = _normalize_axis(axis1);
    axis2 = _normalize_axis(axis2);
    std::vector<int> p = shape;
    std::vector<std::size_t> s = strides;
    std::swap(p[axis1], p[axis2]);
    std::swap(s[axis1], s[axis2]);
    return ndarray(data_, std::move(p), std::move(s), type, order, offset);
}

template <typename T> auto ndarray<T>::squeeze() const -> ndarray
{
    std::vector<int> p;
    std::vector<std::size_t> s;
    p.reserve(shape.size());
    s.reserve(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] != 1)
        {
            p.push_back(shape[i]);
            s.push_back(strides[i]);
        }
    }
    if (p == shape)
    {
        return *this;
    }
    return ndarray(data_, std::move(p), std::move(s), type, order, offset);
}

template <typename T> auto ndarray<T>::squeeze(int axis) const -> ndarray
{
    axis = _normalize_axis(axis);
    if (shape[axis] != 1)
    {
        throw std::invalid_argument("cannot squeeze a dimension that is not of size 1");
    }
    std::vector<int> p = shape;
    std::vector<std::size_t> s = strides;
    p.erase(p.begin() + axis);
    s.erase(s.begin() + axis);
    return ndarray(data_, std::move(p), std::move(s), type, order, offset);
}

template <typename T> auto ndarray<T>::ravel() const -> ndarray
{
    if (is_contiguous())
    {
        return ndarray(data_, {static_cast<int>(_numel())}, {std::size_t{1}}, type, order, offset);
    }
    return flatten();
}

template <typename T> auto ndarray<T>::flatten() const -> ndarray
{
    ndarray out({static_cast<int>(_numel())}, type);
    std::copy(begin(), end(), out.begin());
    return out;
}

template <typename T> void ndarray<T>::resize(const std::vector<int> &new_shape)
{
    // Validate before multiplying: a negative dim would wrap to a huge
    // size_t and fail far away with bad_alloc instead of here.
    _validate_shape(new_shape);
    std::size_t total = 1;
    for (int d : new_shape)
    {
        total *= static_cast<std::size_t>(d);
    }
    std::vector<value_type> flat;
    flat.reserve(total);
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        if (flat.size() < total)
        {
            flat.push_back(v);
        }
    });
    flat.resize(total, typename ndarray<T>::value_type{});
    shape = new_shape;
    strides = _c_strides(new_shape);
    offset = 0;
    data_ = std::make_shared<std::vector<value_type>>(std::move(flat));
    type = type;
}

// Manipulation
template <typename T> void ndarray<T>::fill(const typename ndarray<T>::value_type &value)
{
    static_assert(std::is_copy_constructible_v<value_type>, "fill: value_type must be copy constructible");
    static_assert(std::is_copy_assignable_v<value_type>, "fill: value_type must be copy assignable");
    static_assert(std::is_default_constructible_v<value_type>, "fill: value_type must be default constructible");
    if (!data_)
    {
        data_ = std::make_shared<std::vector<value_type>>(_numel(), value);
        return;
    }
    if (is_contiguous() && data_)
    {
        // Logical range only: never touch sibling storage past _numel().
        const std::size_t n = _numel();
        if constexpr (std::is_same_v<T, bool>)
        {
            for (std::size_t i = 0; i < n; ++i)
                (*data_)[offset + i] = value;
            return;
        }
        else
        {
#ifdef NP_USE_THREADING
            if (n > detail::kParallelThreshold)
            {
                auto *base = data_->data() + offset;
                detail::maybe_parallel_for(0, n, [&](std::size_t i) { base[i] = value; });
                return;
            }
#endif
            std::fill_n(data_->data() + offset, n, value);
            return;
        }
    }
    if (!data_)
    {
        // Null storage has no layout worth preserving: normalize to dense.
        strides = _c_strides(shape);
        offset = 0;
        data_ = std::make_shared<std::vector<value_type>>(_numel(), value);
        return;
    }
    _for_each_indexed([&](const std::vector<std::size_t> &idx, const value_type &) { (*data_)[_flat(idx)] = value; });
}

template <typename T> void ndarray<T>::secure_zero() noexcept
{
    if (!data_ || data_->empty())
    {
        return;
    }
    pqc::ct_barrier();
    if constexpr (std::is_same_v<value_type, bool>)
    {
        // vector<bool> is bit-packed — fill via volatile proxy to avoid elision
        std::fill(data_->begin(), data_->end(), false);
        pqc::ct_barrier();
    }
    else
    {
        static_assert(std::is_trivially_copyable_v<value_type> || std::is_same_v<value_type, std::string> == false,
                      "secure_zero: non-trivially-copyable type");
        pqc::secure_zero(data_->data(), data_->size() * sizeof(value_type));
    }
    pqc::ct_barrier();
}

template <typename T> void ndarray<T>::secure_clear() noexcept
{
    secure_zero();
    shape = std::vector<int>{0};
    strides.clear();
    offset = 0;
    // Use secure wipe for shape/strides vectors as well (contain no secrets but keep
    // consistent)
    pqc::ct_barrier();
    data_.reset();
    pqc::ct_barrier();
}

// ── Secure fill (constant-time, not elided) ──────────────────────────────────
template <typename T> void ndarray<T>::secure_fill(const typename ndarray<T>::value_type &value)
{
    if (!data_ || _numel() == 0)
        return;
    pqc::ct_barrier();
    if constexpr (std::is_same_v<value_type, bool>)
    {
        std::fill(data_->begin(), data_->end(), static_cast<bool>(value));
        pqc::ct_barrier();
    }
    else if (value == value_type{0})
    {
        // Zero is special: use secure_zero (volatile + fence) to guarantee not elided
        secure_zero();
    }
    else
    {
        // For non-zero, use volatile fill + fence to avoid optimization
        if (is_contiguous() && data_)
        {
            // Logical range only (see fill()): never wipe sibling storage.
            const std::size_t n = _numel();
            if constexpr (std::is_same_v<value_type, bool>)
            {
                for (std::size_t i = 0; i < n; ++i)
                    (*data_)[offset + i] = value;
            }
            else
            {
                volatile value_type *p = reinterpret_cast<volatile value_type *>(data_->data() + offset);
                for (std::size_t i = 0; i < n; ++i)
                    p[i] = value;
            }
            pqc::ct_barrier();
        }
        else
        {
            // Non-contiguous: use indexed path with barrier
            _for_each_indexed([&](const std::vector<std::size_t> &idx, const value_type &) {
                volatile value_type *vp = reinterpret_cast<volatile value_type *>(&(*data_)[_flat(idx)]);
                *vp = value;
            });
            pqc::ct_barrier();
        }
    }
}

// ── Secure constant-time access (no secret-dependent branches) ────────────────
template <typename T> typename ndarray<T>::value_type ndarray<T>::secure_at(std::size_t i) const
{
    // Constant-time bounds check: return 0 if out of bounds, but still do not branch on
    // secret Use pqc::ct_select to avoid timing leak on index
    const std::size_t n = _numel();
    if (!data_ || n == 0)
    {
        return value_type{}; // nothing to read (previously segfaulted/OOB)
    }
    // Clamp index to [0, n-1] via ct_select (branch-free)
    std::size_t idx = i;
    int in_range = (i < n) ? 1 : 0;
    // Use ct_select for index: if in_range then i else 0
    // For size_t, we can use mask
    std::size_t mask = static_cast<std::size_t>(-static_cast<std::int64_t>(in_range));
    idx = (idx & mask) | (0 & ~mask);
    // Always do a valid access (0) then select
    value_type v0 = (*data_)[offset + 0 * (strides.empty() ? 0 : strides[0])]; // dummy to keep cache
    (void)v0;
    value_type res{};
    if constexpr (std::is_same_v<T, bool>)
    {
        // Packed bits have no addressable (volatile) storage; plain read.
        res = (*data_)[_flat_logical(idx)];
    }
    else if (is_contiguous())
    {
        // Use volatile load to prevent optimization
        const volatile value_type *p = reinterpret_cast<const volatile value_type *>(data_->data());
        res = p[offset + idx * (strides.empty() ? 1 : 1)]; // simplified for 1D; for ND use _flat
        // For ND, use _flat_logical with constant-time odometer (still O(n) but no branch
        // on i)
        if (ndim() != 1)
        {
            // Unravel the clamped flat index (n > 0 here, so no dim is 0).
            std::vector<std::size_t> cidx(shape.size(), 0);
            std::size_t rem = idx;
            for (std::size_t d = shape.size(); d-- > 0;)
            {
                std::size_t dim = static_cast<std::size_t>(shape[d]);
                cidx[d] = rem % dim;
                rem /= dim;
            }
            res = (*data_)[_flat(cidx)];
        }
    }
    else
    {
        // Non-contiguous: unravel exactly like above (the old code passed a
        // 1-element vector to ND get(), which threw invalid_argument).
        std::vector<std::size_t> cidx(shape.size(), 0);
        std::size_t rem = idx;
        for (std::size_t d = shape.size(); d-- > 0;)
        {
            const std::size_t dim = static_cast<std::size_t>(shape[d]);
            cidx[d] = dim == 0 ? 0 : rem % dim;
            rem = dim == 0 ? 0 : rem / dim;
        }
        res = (*data_)[_flat(cidx)];
    }
    // If out of bounds, return 0 via ct_select (branch-free)
    // For arithmetic types, use pqc::ct_select
    if constexpr (std::is_arithmetic_v<value_type>)
    {
        // Use ct_select: if in_range then res else 0
        // Need to handle different sizes; use generic via pqc::ct_select for 32/64, else
        // branch
        if constexpr (sizeof(value_type) == 4 || sizeof(value_type) == 8)
        {
            // Use pqc::ct_select for 4/8 byte types
            // For float/double, it will use memcpy trick
            value_type zero{};
            res = pqc::ct_select(in_range, res, zero);
        }
        else
        {
            res = in_range ? res : value_type{};
        }
    }
    else
    {
        res = in_range ? res : value_type{};
    }
    pqc::ct_barrier();
    return res;
}

template <typename T> auto ndarray<T>::copy() const -> ndarray
{
    ndarray out(shape, type);
    if constexpr (std::is_same_v<T, bool>)
    {
        detail::Odometer od(shape);
        while (!od.done())
        {
            out.set(od.idx(), get(od.idx()));
            od.advance();
        }
    }
    else
    {
        std::copy(begin(), end(), out.begin());
    }
    return out;
}

template <typename T> auto ndarray<T>::view() const -> ndarray
{
    return ndarray(data_, shape, strides, type, order, offset);
}

template <typename T> template <typename U> auto ndarray<T>::astype() const -> ndarray<U>
{
    _require_data();
    ndarray<U> out(shape);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = static_cast<U>(v); });
    return out;
}

template <typename T>
auto ndarray<T>::take(const std::vector<std::size_t> &indices, std::optional<int> axis) const -> ndarray
{
    if (!axis.has_value())
    {
        return flatten().take(indices, 0);
    }
    return take(indices, *axis);
}

template <typename T> auto ndarray<T>::take(const std::vector<std::size_t> &indices, int axis) const -> ndarray
{
    _require_data();
    const int nd = static_cast<int>(shape.size());
    axis = _normalize_axis(axis);
    std::vector<int> out_shape = shape;
    out_shape[axis] = static_cast<int>(indices.size());
    ndarray out(out_shape, type);

    const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);
    for (std::size_t k = 0; k < indices.size(); ++k)
    {
        if (indices[k] >= axis_len)
        {
            throw std::out_of_range("take index out of bounds");
        }
    }

    std::vector<int> slice_shape = shape;
    slice_shape.erase(slice_shape.begin() + axis);

    detail::Odometer od(slice_shape);
    while (!od.done())
    {
        const auto &s = od.idx();
        for (std::size_t k = 0; k < indices.size(); ++k)
        {
            std::size_t in_f = 0, out_f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t coord = (d < axis) ? s[d] : (d == axis ? indices[k] : s[d - 1]);
                in_f += coord * strides[d];
                const std::size_t out_coord = (d < axis) ? s[d] : (d == axis ? k : s[d - 1]);
                out_f += out_coord * out.strides[d];
            }
            out.data()[out_f] = (*data_)[offset + in_f];
        }
        od.advance();
    }
    return out;
}

template <typename T>
void ndarray<T>::put(const std::vector<std::size_t> &indices,
                     const std::vector<typename ndarray<T>::value_type> &values, std::optional<char> mode)
{
    put(indices, values, mode.value_or('r'));
}

template <typename T>
void ndarray<T>::put(const std::vector<std::size_t> &indices,
                     const std::vector<typename ndarray<T>::value_type> &values, char mode)
{
    _require_data();
    const std::size_t n = _numel();
    if (n == 0 && !indices.empty())
    {
        throw std::out_of_range("put into empty array");
    }
    for (std::size_t k = 0; k < indices.size(); ++k)
    {
        std::size_t p = indices[k];
        if (mode == 'w')
        {
            p %= n;
        }
        else if (mode == 'c')
        {
            p = std::min(p, n - 1);
        }
        else if (p >= n)
        {
            throw std::out_of_range("put index out of bounds");
        }
        const typename ndarray<T>::value_type &v =
            values.empty() ? typename ndarray<T>::value_type{} : values[k % values.size()];
        // logical flat index -> multi-index -> flat storage offset
        std::vector<std::size_t> idx = _shape_u();
        std::size_t rem = p;
        for (std::size_t d = shape.size(); d-- > 0;)
        {
            idx[d] = rem % static_cast<std::size_t>(shape[d]);
            rem /= static_cast<std::size_t>(shape[d]);
        }
        (*data_)[_flat(idx)] = v;
    }
}

template <typename T> auto ndarray<T>::repeat(std::size_t repeats) const -> ndarray
{
    ndarray out({static_cast<int>(_numel() * repeats)}, type);
    std::size_t o = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        for (std::size_t r = 0; r < repeats; ++r)
        {
            out.data()[o++] = v;
        }
    });
    return out;
}

template <typename T> auto ndarray<T>::repeat(std::size_t repeats, std::optional<int> axis) const -> ndarray
{
    if (!axis.has_value())
    {
        return repeat(repeats);
    }
    return repeat(repeats, *axis);
}

template <typename T> auto ndarray<T>::repeat(std::size_t repeats, int axis) const -> ndarray
{
    axis = _normalize_axis(axis);
    const int nd = static_cast<int>(shape.size());
    std::vector<int> out_shape = shape;
    out_shape[axis] = static_cast<int>(static_cast<std::size_t>(shape[axis]) * repeats);
    ndarray out(out_shape, type);

    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        for (std::size_t r = 0; r < repeats; ++r)
        {
            std::size_t in_f = _flat(idx);
            std::size_t out_f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t coord = (d == axis) ? idx[d] * repeats + r : idx[d];
                out_f += coord * out.strides[d];
            }
            out.data()[out_f] = (*data_)[in_f];
        }
        od.advance();
    }
    return out;
}

template <typename T>
auto ndarray<T>::clip(const typename ndarray<T>::value_type &min_value,
                      const typename ndarray<T>::value_type &max_value) const -> ndarray
{
    ndarray out(shape, type);
    std::size_t i = 0;
    _for_each_logical(
        [&](const typename ndarray<T>::value_type &v) { out.data()[i++] = std::clamp(v, min_value, max_value); });
    return out;
}

template <typename T> auto ndarray<T>::round(int decimals) const -> ndarray
{
    ndarray out(shape, type);
    // Hoisted once: per-element pow() was O(n) wasted work, and narrowing
    // the factor to T lost precision for float. std::nearbyint rounds
    // half-to-even (NumPy banker's rounding); std::round was half-away.
    const double factor = std::pow(10.0, static_cast<double>(decimals));
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        if constexpr (std::is_floating_point_v<T>)
        {
            out.data()[i++] = static_cast<T>(std::nearbyint(static_cast<double>(v) * factor) / factor);
        }
        else if constexpr (detail::is_complex_v<T>)
        {
            using R = typename detail::_Np_real_of<T>::type;
            const auto re = static_cast<R>(std::nearbyint(static_cast<double>(v.real()) * factor) / factor);
            const auto im = static_cast<R>(std::nearbyint(static_cast<double>(v.imag()) * factor) / factor);
            out.data()[i++] = value_type(re, im);
        }
        else
        {
            out.data()[i++] = v;
        }
    });
    return out;
}

template <typename T> auto ndarray<T>::round(std::optional<int> decimals) const -> ndarray
{
    return round(decimals.value_or(0));
}

template <typename T> auto ndarray<T>::diagonal(std::optional<int> offset) const -> ndarray
{
    return diagonal(offset.value_or(0));
}

template <typename T> auto ndarray<T>::diagonal(int offset) const -> ndarray
{
    _require_data();
    if (shape.size() < 2)
    {
        throw np::AxisError("diagonal requires an array with ndim >= 2");
    }
    const std::size_t n0 = static_cast<std::size_t>(shape[0]);
    const std::size_t n1 = static_cast<std::size_t>(shape[1]);

    std::size_t len = 0;
    if (offset >= 0)
    {
        const std::size_t o = static_cast<std::size_t>(offset);
        len = (n1 > o) ? std::min(n0, n1 - o) : 0;
    }
    else
    {
        const std::size_t o = static_cast<std::size_t>(-offset);
        len = (n0 > o) ? std::min(n1, n0 - o) : 0;
    }

    std::vector<int> out_shape;
    out_shape.push_back(static_cast<int>(len));
    out_shape.insert(out_shape.end(), shape.begin() + 2, shape.end());
    ndarray out(out_shape, type);

    detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &oi = od.idx();
        std::vector<std::size_t> in_idx(shape.size());
        if (offset >= 0)
        {
            in_idx[0] = oi[0];
            in_idx[1] = oi[0] + static_cast<std::size_t>(offset);
        }
        else
        {
            // Negative offset: diagonal runs below the main one, so the row
            // leads (previously in_idx[0] stayed 0 while in_idx[1] wrapped).
            in_idx[0] = oi[0] + static_cast<std::size_t>(-offset);
            in_idx[1] = oi[0];
        }
        for (std::size_t d = 2; d < shape.size(); ++d)
        {
            in_idx[d] = oi[d - 1];
        }
        out.data()[detail::flat_index(oi, out.strides, 0)] = (*data_)[_flat(in_idx)];
        od.advance();
    }
    return out;
}

template <typename T> typename ndarray<T>::value_type ndarray<T>::trace(std::optional<int> offset) const
{
    return trace(offset.value_or(0));
}

template <typename T> typename ndarray<T>::value_type ndarray<T>::trace(int offset) const
{
    _require_data();
    if (shape.size() < 2)
    {
        throw np::AxisError("trace requires an array with ndim >= 2");
    }
    // Accumulate along the diagonal directly instead of materializing it.
    const std::size_t n0 = static_cast<std::size_t>(shape[0]);
    const std::size_t n1 = static_cast<std::size_t>(shape[1]);
    std::size_t len = 0, r0 = 0, c0 = 0;
    if (offset >= 0)
    {
        const std::size_t o = static_cast<std::size_t>(offset);
        len = (n1 > o) ? std::min(n0, n1 - o) : 0;
        c0 = o;
    }
    else
    {
        const std::size_t o = static_cast<std::size_t>(-static_cast<std::ptrdiff_t>(offset));
        len = (n0 > o) ? std::min(n1, n0 - o) : 0;
        r0 = o;
    }
    T total{};
    // NB: the `offset` parameter (diagonal shift) shadows member `offset`
    // (storage origin) — qualify explicitly.
    const std::size_t base = this->offset;
    for (std::size_t k = 0; k < len; ++k)
    {
        total += (*data_)[base + (r0 + k) * strides[0] + (c0 + k) * strides[1]];
    }
    return total;
}

template <typename T> auto ndarray<T>::nonzero() const -> std::vector<ndarray<std::size_t>>
{
    std::vector<ndarray<std::size_t>> result(shape.size());
    std::vector<std::vector<std::size_t>> per_dim(shape.size());
    _for_each_indexed([&](const std::vector<std::size_t> &idx, const typename ndarray<T>::value_type &v) {
        if (v != typename ndarray<T>::value_type{})
        {
            for (std::size_t d = 0; d < idx.size(); ++d)
            {
                per_dim[d].push_back(idx[d]);
            }
        }
    });
    for (std::size_t d = 0; d < result.size(); ++d)
    {
        const int n_coords = static_cast<int>(per_dim[d].size());
        result[d] = ndarray<std::size_t>::from_data(std::vector<int>{n_coords}, std::move(per_dim[d]));
    }
    return result;
}

template <typename T> auto ndarray<T>::conj() const -> ndarray
{
    ndarray out(shape, type);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        if constexpr (detail::is_complex_v<T>)
        {
            out.data()[i++] = std::conj(v);
        }
        else
        {
            out.data()[i++] = v;
        }
    });
    return out;
}

template <typename T> void ndarray<T>::byteswap()
{
    if (!data_)
    {
        return;
    }
    if constexpr (std::is_same_v<T, bool>)
    {
        return; // single-bit elements have no byte order
    }
    else if (is_contiguous())
    {
        // Logical range only: an oversized shared buffer must not be touched
        // past this view's elements.
        const std::size_t n = _numel();
        T *__restrict base = data_->data() + offset;
        for (std::size_t i = 0; i < n; ++i)
        {
            char *p = reinterpret_cast<char *>(&base[i]);
            std::reverse(p, p + sizeof(T));
        }
        return;
    }
    else
    {
        _for_each_indexed([&](const std::vector<std::size_t> &idx, const T &) {
            T &v = (*data_)[_flat(idx)];
            char *p = reinterpret_cast<char *>(&v);
            std::reverse(p, p + sizeof(T));
        });
    }
}

// Selection / manipulation
template <typename T> auto ndarray<T>::abs() const -> ndarray<typename detail::_Np_real_of<T>::type>
{
    // NumPy abs() of complex returns a real array (magnitudes), not complex.
    using R = typename detail::_Np_real_of<T>::type;
    ndarray<R> out(shape);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = static_cast<R>(std::abs(v)); });
    return out;
}

template <typename T> auto ndarray<T>::conjugate() const -> ndarray
{
    return conj();
}

template <typename T>
template <typename U>
auto ndarray<T>::choose(const std::vector<ndarray<U>> &choices, std::optional<char> mode) const -> ndarray<U>
{
    return choose(choices, mode.value_or('r'));
}

template <typename T>
template <typename U>
auto ndarray<T>::choose(const std::vector<ndarray<U>> &choices, char mode) const -> ndarray<U>
{
    // Index arrays must be integral: float->long long is UB out of range
    // and complex has no such conversion at all.
    static_assert(std::is_integral_v<T>, "choose: index array must have integral dtype");
    _require_data();
    if (choices.empty())
    {
        throw std::invalid_argument("choose requires at least one choice");
    }
    std::vector<int> bshape = shape;
    for (const auto &c : choices)
    {
        bshape = detail::broadcast_shapes(bshape, c.shape);
    }
    const std::size_t n = choices.size();
    ndarray<U> out(bshape);
    detail::Odometer od(bshape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        const T a_v = (*data_)[detail::broadcast_offset(*this, bshape, idx)];
        long long k = static_cast<long long>(a_v);
        if (mode == 'w')
        {
            k = ((k % static_cast<long long>(n)) + static_cast<long long>(n)) % static_cast<long long>(n);
        }
        else if (mode == 'c')
        {
            k = std::clamp(k, 0LL, static_cast<long long>(n) - 1);
        }
        else if (k < 0 || k >= static_cast<long long>(n))
        {
            throw std::out_of_range("choose index out of range");
        }
        const auto &ch = choices[static_cast<std::size_t>(k)];
        out.data()[detail::flat_index(idx, out.strides, 0)] = ch.data()[detail::broadcast_offset(ch, bshape, idx)];
        od.advance();
    }
    return out;
}

template <typename T>
auto ndarray<T>::compress(const ndarray<bool> &condition, std::optional<int> axis) const -> ndarray
{
    if (condition.ndim() != 1)
    {
        throw std::invalid_argument("condition must be 1-D");
    }
    const std::size_t cond_len = static_cast<std::size_t>(condition.shape[0]);
    if (!axis.has_value())
    {
        if (cond_len != _numel())
        {
            throw std::invalid_argument("condition length must match the array size");
        }
        const auto flat = ravel();
        std::vector<value_type> picked;
        picked.reserve(cond_len);
        for (std::size_t i = 0; i < cond_len; ++i)
        {
            if (condition.data()[condition._flat_logical(i)])
            {
                picked.push_back(flat.data()[flat._flat_logical(i)]);
            }
        }
        const std::vector<int> out_shp{static_cast<int>(picked.size())};
        return ndarray::from_data(out_shp, std::move(picked));
    }
    const int ax = _normalize_axis(*axis);
    const std::size_t axis_len = static_cast<std::size_t>(shape[ax]);
    if (cond_len != axis_len)
    {
        throw std::invalid_argument("condition length must match the array's axis length");
    }
    std::vector<std::size_t> keep;
    keep.reserve(cond_len);
    for (std::size_t i = 0; i < axis_len; ++i)
    {
        if (condition.data()[condition._flat_logical(i)])
        {
            keep.push_back(i);
        }
    }
    std::vector<int> out_shape = shape;
    out_shape[ax] = static_cast<int>(keep.size());
    ndarray out(out_shape, type);
    std::vector<int> rest = shape;
    rest.erase(rest.begin() + ax);
    detail::Odometer od(rest);
    const int nd = static_cast<int>(shape.size());
    while (!od.done())
    {
        const auto &s = od.idx();
        for (std::size_t k = 0; k < keep.size(); ++k)
        {
            std::size_t in_f = offset, out_f = 0;
            for (int d = 0; d < nd; ++d)
            {
                const std::size_t ic = d == ax ? keep[k] : (d < ax ? s[d] : s[d - 1]);
                const std::size_t oc = d == ax ? k : (d < ax ? s[d] : s[d - 1]);
                in_f += ic * strides[d];
                out_f += oc * out.strides[d];
            }
            out.data()[out_f] = (*data_)[in_f];
        }
        od.advance();
    }
    return out;
}

template <typename T>
template <typename U>
auto ndarray<T>::dot(const ndarray<U> &b) const -> ndarray<std::common_type_t<T, U>>
{
    return np::linalg::dot(*this, b);
}

template <typename T>
template <typename U>
auto ndarray<T>::matmul(const ndarray<U> &b) const -> ndarray<std::common_type_t<T, U>>
{
    return np::linalg::matmul(*this, b);
}

template <typename T> void ndarray<T>::partition(std::size_t kth, std::optional<int> axis)
{
    partition(kth, axis.value_or(-1));
}

template <typename T> void ndarray<T>::partition(std::size_t kth, int axis)
{
    axis = _normalize_axis(axis);
    const int nd = static_cast<int>(shape.size());
    const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);
    if (kth >= axis_len)
    {
        throw std::out_of_range("kth out of bounds");
    }
    _require_data();
    std::vector<int> rest = shape;
    rest.erase(rest.begin() + axis);
    detail::Odometer od(rest);
    std::vector<typename ndarray<T>::value_type> work(axis_len);
    std::vector<std::size_t> full(nd);
    while (!od.done())
    {
        const auto &s = od.idx();
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = offset;
            for (int d = 0; d < nd; ++d)
            {
                full[d] = d == axis ? p : (d < axis ? s[d] : s[d - 1]);
                f += full[d] * strides[d];
            }
            work[p] = (*data_)[f];
        }
        std::nth_element(work.begin(), work.begin() + kth, work.end(), detail::value_less<value_type>);
        for (std::size_t p = 0; p < axis_len; ++p)
        {
            std::size_t f = offset;
            for (int d = 0; d < nd; ++d)
            {
                full[d] = d == axis ? p : (d < axis ? s[d] : s[d - 1]);
                f += full[d] * strides[d];
            }
            (*data_)[f] = work[p];
        }
        od.advance();
    }
}

template <typename T> auto ndarray<T>::real() const -> ndarray<typename detail::_Np_real_of<T>::type>
{
    using R = typename detail::_Np_real_of<T>::type;
    if constexpr (detail::is_complex_v<T>)
    {
        ndarray<R> out(shape);
        std::size_t i = 0;
        _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = v.real(); });
        return out;
    }
    else
    {
        return view();
    }
}

template <typename T> auto ndarray<T>::imag() const -> ndarray<typename detail::_Np_real_of<T>::type>
{
    using R = typename detail::_Np_real_of<T>::type;
    if constexpr (detail::is_complex_v<T>)
    {
        ndarray<R> out(shape);
        std::size_t i = 0;
        _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = v.imag(); });
        return out;
    }
    else
    {
        return ndarray<R>(shape);
    }
}

template <typename T> auto ndarray<T>::mT() const -> ndarray
{
    if (shape.size() < 2)
    {
        throw np::AxisError("mT requires an array with ndim >= 2");
    }
    const std::size_t nd = shape.size();
    std::vector<int> p = shape;
    std::vector<std::size_t> s = strides;
    std::swap(p[nd - 1], p[nd - 2]);
    std::swap(s[nd - 1], s[nd - 2]);
    return ndarray(data_, std::move(p), std::move(s), type, order, offset);
}

template <typename T> void ndarray<T>::setflags(bool writeable)
{
    writeable_ = writeable;
}

template <typename T> bool ndarray<T>::writeable() const noexcept
{
    return writeable_;
}

template <typename T> const void *ndarray<T>::base() const noexcept
{
    return is_view_ ? static_cast<const void *>(data_.get()) : nullptr;
}

template <typename T> bool ndarray<T>::owns_data() const noexcept
{
    return !is_view_;
}

template <typename T> bool ndarray<T>::is_view() const noexcept
{
    return is_view_;
}

template <typename T> auto ndarray<T>::flat() const -> ndarray
{
    return ravel();
}

template <typename T> std::size_t ndarray<T>::len() const
{
    if (shape.empty())
    {
        throw std::invalid_argument("len() of a 0-d array is undefined");
    }
    return static_cast<std::size_t>(shape[0]);
}

template <typename T> bool ndarray<T>::contains(const typename ndarray<T>::value_type &value) const
{
    // Early exit on first match (previously scanned the whole array even
    // after finding it); _for_each_logical cannot break out, so loop here.
    if (!data_)
    {
        return false;
    }
    if (is_contiguous())
    {
        const std::size_t n = _numel();
        if constexpr (std::is_same_v<T, bool>)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                if ((*data_)[offset + i] == value)
                {
                    return true;
                }
            }
        }
        else
        {
            const T *__restrict p = data_->data() + offset;
            for (std::size_t i = 0; i < n; ++i)
            {
                if (p[i] == value)
                {
                    return true;
                }
            }
        }
        return false;
    }
    detail::Odometer od(shape);
    while (!od.done())
    {
        if ((*data_)[_flat(od.idx())] == value)
        {
            return true;
        }
        od.advance();
    }
    return false;
}

template <typename T>
template <typename U>
auto ndarray<T>::floordiv(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return detail::floored_div(a, b); });
}

template <typename T>
template <typename U>
auto ndarray<T>::floordiv(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic");
    return _scalar_op(scalar, [](const T &a, const U &b) { return detail::floored_div(a, b); });
}

template <typename T>
template <typename U>
auto ndarray<T>::divmod(const ndarray<U> &rhs) const
    -> std::pair<ndarray<std::common_type_t<T, U>>, ndarray<std::common_type_t<T, U>>>
{
    // Single traversal computing both halves (previously two full passes).
    using R = std::common_type_t<T, U>;
    if (!data_ || !rhs.data_)
    {
        throw std::runtime_error("divmod: operand has no data buffer");
    }
    const std::vector<int> out_shape = detail::broadcast_shapes(shape, rhs.shape);
    ndarray<R> q(out_shape), r(out_shape);
    const int nr = static_cast<int>(out_shape.size());
    const int shift_a = nr - static_cast<int>(shape.size());
    const int shift_b = nr - static_cast<int>(rhs.shape.size());
    detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        std::size_t fa = offset, fb = rhs.offset, fo = 0;
        for (int d = 0; d < nr; ++d)
        {
            const int ka = d - shift_a;
            const int kb = d - shift_b;
            fa += (ka < 0 || shape[ka] == 1) ? 0 : idx[d] * strides[ka];
            fb += (kb < 0 || rhs.shape[kb] == 1) ? 0 : idx[d] * rhs.strides[kb];
            fo += idx[d] * q.strides[d];
        }
        // Explicit T/U (not auto): vector<bool> accessors yield proxy
        // prvalues, which would poison common_type deduction.
        const T av = (*data_)[fa];
        const U bv = (*rhs.data_)[fb];
        q.data()[fo] = detail::floored_div(av, bv);
        r.data()[fo] = detail::floored_mod(av, bv);
        od.advance();
    }
    return {std::move(q), std::move(r)};
}

template <typename T>
template <typename U>
auto ndarray<T>::divmod(const U &scalar) const
    -> std::pair<ndarray<std::common_type_t<T, U>>, ndarray<std::common_type_t<T, U>>>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    // Single traversal (previously floordiv + remainder = two passes).
    using R = std::common_type_t<T, U>;
    _require_data();
    ndarray<R> q(shape), r(shape);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        q.data()[i] = detail::floored_div(v, scalar);
        r.data()[i] = detail::floored_mod(v, scalar);
        ++i;
    });
    return {std::move(q), std::move(r)};
}

template <typename T>
template <typename U>
auto ndarray<T>::pow(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return detail::power_elem(a, b); });
}

template <typename T>
template <typename U>
auto ndarray<T>::pow(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic");
    return _scalar_op(scalar, [](const T &a, const U &b) { return detail::power_elem(a, b); });
}

// Conversions
template <typename T> ndarray<T>::operator bool() const
{
    if (_numel() != 1)
    {
        throw std::invalid_argument("bool() of a non-single-element array");
    }
    return static_cast<bool>(item());
}

template <typename T> ndarray<T>::operator long long() const
{
    if (_numel() != 1)
    {
        throw std::invalid_argument("int() of a non-single-element array");
    }
    return static_cast<long long>(item());
}

template <typename T> ndarray<T>::operator double() const
{
    if (_numel() != 1)
    {
        throw std::invalid_argument("float() of a non-single-element array");
    }
    return static_cast<double>(item());
}

template <typename T> ndarray<T>::operator std::complex<double>() const
{
    if (_numel() != 1)
    {
        throw std::invalid_argument("complex() of a non-single-element array");
    }
    return std::complex<double>(item());
}

// Element-wise operators
template <typename T> auto ndarray<T>::operator+() const -> ndarray
{
    return *this;
}

template <typename T>
template <typename U>
auto ndarray<T>::operator%(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return detail::floored_mod(a, b); });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator%(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic");
    return _scalar_op(scalar, [](const T &a, const U &b) { return detail::floored_mod(a, b); });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator&(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise AND requires integral element types");
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a & b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator&(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise AND requires integral element types");
    return _scalar_op(scalar, [](const T &a, const U &b) { return a & b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator|(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise OR requires integral element types");
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a | b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator|(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise OR requires integral element types");
    return _scalar_op(scalar, [](const T &a, const U &b) { return a | b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator^(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise XOR requires integral element types");
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a ^ b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator^(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise XOR requires integral element types");
    return _scalar_op(scalar, [](const T &a, const U &b) { return a ^ b; });
}

template <typename T> auto ndarray<T>::operator~() const -> ndarray
{
    static_assert(std::is_integral_v<T>, "bitwise NOT requires an integral element type");
    ndarray out(shape, type);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = ~v; });
    return out;
}

template <typename T>
template <typename U>
auto ndarray<T>::operator<<(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "left shift requires integral element types");
    return detail::elementwise(*this, rhs,
                               [](const T &a, const U &b) { return a << detail::checked_shift_count(a, b); });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator<<(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "left shift requires integral element types");
    return _scalar_op(scalar, [](const T &a, const U &b) { return a << detail::checked_shift_count(a, b); });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator>>(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "right shift requires integral element types");
    return detail::elementwise(*this, rhs,
                               [](const T &a, const U &b) { return a >> detail::checked_shift_count(a, b); });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator>>(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "right shift requires integral element types");
    return _scalar_op(scalar, [](const T &a, const U &b) { return a >> detail::checked_shift_count(a, b); });
}

// In-place operators (recompute from the element-wise form).

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator%=(const ndarray<U> &rhs)
{
    _inplace_op(rhs, [&](const T &a, const U &b) { return detail::floored_mod(a, b); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator%=(const U &scalar)
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return detail::floored_mod(a, b); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator&=(const ndarray<U> &rhs)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_op(rhs, [&](const T &a, const U &b) { return a & b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator&=(const U &scalar)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return a & b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator|=(const ndarray<U> &rhs)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_op(rhs, [&](const T &a, const U &b) { return a | b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator|=(const U &scalar)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return a | b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator^=(const ndarray<U> &rhs)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_op(rhs, [&](const T &a, const U &b) { return a ^ b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator^=(const U &scalar)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return a ^ b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator<<=(const ndarray<U> &rhs)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_op(rhs, [&](const T &a, const U &b) { return static_cast<T>(a << detail::checked_shift_count(a, b)); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator<<=(const U &scalar)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_scalar(scalar,
                    [&](const T &a, const U &b) { return static_cast<T>(a << detail::checked_shift_count(a, b)); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator>>=(const ndarray<U> &rhs)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_op(rhs, [&](const T &a, const U &b) { return static_cast<T>(a >> detail::checked_shift_count(a, b)); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator>>=(const U &scalar)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "bitwise/shift in-place ops require integral types");
    _inplace_scalar(scalar,
                    [&](const T &a, const U &b) { return static_cast<T>(a >> detail::checked_shift_count(a, b)); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::floordiv_eq(const ndarray<U> &rhs)
{
    _inplace_op(rhs, [&](const T &a, const U &b) { return detail::floored_div(a, b); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::floordiv_eq(const U &scalar)
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return detail::floored_div(a, b); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::pow_eq(const ndarray<U> &rhs)
{
    _inplace_op(rhs, [&](const T &a, const U &b) { return detail::power_elem(a, b); });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::pow_eq(const U &scalar)
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return detail::power_elem(a, b); });
    return *this;
}

// Conversions / IO
template <typename T> auto ndarray<T>::tolist() const -> std::vector<typename ndarray<T>::value_type>
{
    return std::vector<value_type>(begin(), end());
}

template <typename T> auto ndarray<T>::tobytes() const -> std::vector<std::uint8_t>
{
    // NumPy bool arrays dump one byte per element (not bit-packed).
    if constexpr (std::is_same_v<T, bool>)
    {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(_numel());
        _for_each_logical([&](const typename ndarray<T>::value_type &v) { bytes.push_back(v ? 1 : 0); });
        return bytes;
    }
    else
    {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(_numel() * sizeof(value_type));
        if (is_contiguous() && data_)
        {
            // Single memcpy for dense storage (offsets included).
            const std::uint8_t *p = reinterpret_cast<const std::uint8_t *>(data_->data() + offset);
            bytes.insert(bytes.end(), p, p + _numel() * sizeof(value_type));
            return bytes;
        }
        _for_each_logical([&](const typename ndarray<T>::value_type &v) {
            const std::uint8_t *p = reinterpret_cast<const std::uint8_t *>(&v);
            bytes.insert(bytes.end(), p, p + sizeof(value_type));
        });
        return bytes;
    }
}

template <typename T> void ndarray<T>::tofile(const std::string &filename) const
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("cannot open file: " + filename);
    }
    tofile(out);
    out.flush();
    if (!out)
    {
        throw std::runtime_error("failed writing file: " + filename);
    }
}

template <typename T> void ndarray<T>::tofile(std::ostream &os) const
{
    const auto bytes = tobytes();
    if (bytes.size() > static_cast<std::size_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        throw std::length_error("tofile: array too large for a single stream write");
    }
    os.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!os)
    {
        throw std::runtime_error("tofile: stream write failed (disk full?)");
    }
}

template <typename T> void ndarray<T>::print(std::ostream &os) const
{
    _print_to(os);
    os << '\n';
}

template <typename T>
void ndarray<T>::_print_recursive(std::size_t dim, std::size_t flat_offset, std::ostream &os) const
{
    if (shape.empty())
    {
        os << (*data_)[offset];
        return;
    }
    if (dim == shape.size() - 1)
    {
        os << "[";
        for (std::size_t i = 0; i < static_cast<std::size_t>(shape[dim]); ++i)
        {
            if (i != 0)
            {
                os << ", ";
            }
            os << (*data_)[flat_offset + i * strides[dim]];
        }
        os << "]";
        return;
    }
    os << "[";
    for (std::size_t i = 0; i < static_cast<std::size_t>(shape[dim]); ++i)
    {
        if (i != 0)
        {
            os << ",\n ";
        }
        _print_recursive(dim + 1, flat_offset + i * strides[dim], os);
    }
    os << "]";
}

template <typename T> void ndarray<T>::_print_to(std::ostream &os) const
{
    if (!data_)
    {
        os << "array([])";
        return;
    }
    os << "array(";
    _print_recursive(0, offset, os);
    os << ", dtype=" << dtype_name(type) << ")";
}

// Element-wise arithmetic
template <typename T>
template <typename U>
auto ndarray<T>::operator+(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    // SIMD fast path: contiguous, same shape, float/double
    if constexpr ((std::is_same_v<R, float> || std::is_same_v<R, double>) && std::is_same_v<T, R> &&
                  std::is_same_v<U, R>)
    {
        if (data_ && rhs.data_ && is_contiguous() && rhs.is_contiguous() && shape == rhs.shape &&
            std::is_same_v<T, R> && std::is_same_v<U, R>)
        {
            ndarray<R> out(shape);
            simd::add_vectorized(data_->data() + offset, rhs.data_->data() + rhs.offset, out.data_->data(), _numel());
            return out;
        }
    }
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a + b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator-(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    if constexpr ((std::is_same_v<R, float> || std::is_same_v<R, double>) && std::is_same_v<T, R> &&
                  std::is_same_v<U, R>)
    {
        if (data_ && rhs.data_ && is_contiguous() && rhs.is_contiguous() && shape == rhs.shape &&
            std::is_same_v<T, R> && std::is_same_v<U, R>)
        {
            ndarray<R> out(shape);
            simd::sub_vectorized(data_->data() + offset, rhs.data_->data() + rhs.offset, out.data_->data(), _numel());
            return out;
        }
    }
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a - b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator*(const ndarray<U> &rhs) const -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    if constexpr ((std::is_same_v<R, float> || std::is_same_v<R, double>) && std::is_same_v<T, R> &&
                  std::is_same_v<U, R>)
    {
        if (data_ && rhs.data_ && is_contiguous() && rhs.is_contiguous() && shape == rhs.shape &&
            std::is_same_v<T, R> && std::is_same_v<U, R>)
        {
            ndarray<R> out(shape);
            simd::mul_vectorized(data_->data() + offset, rhs.data_->data() + rhs.offset, out.data_->data(), _numel());
            return out;
        }
    }
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a * b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator/(const ndarray<U> &rhs) const -> ndarray<detail::div_result_t<T, U>>
{
    // NumPy true_divide: integral / integral promotes to double instead of
    // truncating (use floordiv() for floor semantics).
    using R = detail::div_result_t<T, U>;
    if constexpr ((std::is_same_v<R, float> || std::is_same_v<R, double>) && std::is_same_v<T, R> &&
                  std::is_same_v<U, R>)
    {
        if (data_ && rhs.data_ && is_contiguous() && rhs.is_contiguous() && shape == rhs.shape &&
            std::is_same_v<T, R> && std::is_same_v<U, R>)
        {
            ndarray<R> out(shape);
            simd::div_vectorized(data_->data() + offset, rhs.data_->data() + rhs.offset, out.data_->data(), _numel());
            return out;
        }
    }
    if constexpr (std::is_integral_v<T> && std::is_integral_v<U>)
    {
        return detail::elementwise(
            *this, rhs, [](const T &a, const U &b) { return static_cast<double>(a) / static_cast<double>(b); });
    }
    else
    {
        return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a / b; });
    }
}

template <typename T>
template <typename U, typename Fn>
auto ndarray<T>::_scalar_op(const U &scalar, Fn &&fn) const -> ndarray<std::common_type_t<T, U>>
{
    _require_data();
    using R = std::common_type_t<T, U>;
    ndarray<R> out(shape);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = fn(v, scalar); });
    return out;
}

template <typename T>
template <typename U, typename Fn>
auto ndarray<T>::_scalar_left_op(const U &scalar, Fn &&fn) const -> ndarray<std::common_type_t<U, T>>
{
    _require_data();
    using R = std::common_type_t<U, T>;
    ndarray<R> out(shape);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = fn(scalar, v); });
    return out;
}

template <typename T>
template <typename U, typename Fn>
auto ndarray<T>::_cmp_scalar(const U &scalar, Fn &&fn) const -> ndarray<bool>
{
    _require_data();
    ndarray<bool> out(shape, dtype::bool_);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = fn(v, scalar); });
    return out;
}

template <typename T>
template <typename U, typename Fn>
auto ndarray<T>::_cmp_scalar_left(const U &scalar, Fn &&fn) const -> ndarray<bool>
{
    _require_data();
    ndarray<bool> out(shape, dtype::bool_);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = fn(scalar, v); });
    return out;
}

template <typename T> template <typename U, typename Fn> void ndarray<T>::_inplace_op(const ndarray<U> &rhs, Fn &&fn)
{
    _require_data();
    if (!rhs.data_)
    {
        throw std::runtime_error("in-place op: right-hand operand has no data buffer");
    }
    if (!writeable_)
    {
        throw std::runtime_error("in-place op: array is not writeable (see setflags)");
    }
    // rhs must broadcast TO *this: every rhs dim is 1 or equals ours, and
    // rhs rank <= our rank (right-aligned, NumPy rules).
    if (rhs.shape.size() > shape.size())
    {
        throw std::invalid_argument("in-place op: right-hand side has higher rank than target");
    }
    const std::size_t nr = shape.size();
    const std::size_t shift = nr - rhs.shape.size();
    std::vector<std::size_t> adj(nr, 0);
    for (std::size_t d = 0; d < nr; ++d)
    {
        const int dim = shape[d];
        if (d < shift)
        {
            continue; // broadcast leading dim
        }
        const int rd = rhs.shape[d - shift];
        if (rd != 1 && rd != dim)
        {
            throw std::invalid_argument("in-place op: right-hand side is not broadcastable to target shape");
        }
        adj[d] = (rd == 1) ? 0 : rhs.strides[d - shift];
    }
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        std::size_t fself = offset, frhs = rhs.offset;
        for (std::size_t d = 0; d < nr; ++d)
        {
            fself += idx[d] * strides[d];
            frhs += idx[d] * adj[d];
        }
        // Assign through operator[] (not compound assignment): vector<bool>
        // element proxies have operator= but no operator+= etc.
        (*data_)[fself] = static_cast<T>(fn(static_cast<T>((*data_)[fself]), (*rhs.data_)[frhs]));
        od.advance();
    }
}

template <typename T> template <typename U, typename Fn> void ndarray<T>::_inplace_scalar(const U &scalar, Fn &&fn)
{
    _require_data();
    if (!writeable_)
    {
        throw std::runtime_error("in-place op: array is not writeable (see setflags)");
    }
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        const std::size_t f = _flat(idx);
        (*data_)[f] = static_cast<T>(fn(static_cast<T>((*data_)[f]), scalar));
        od.advance();
    }
}

template <typename T>
template <typename U>
auto ndarray<T>::operator+(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _scalar_op(scalar, [](const T &a, const U &b) { return a + b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator-(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _scalar_op(scalar, [](const T &a, const U &b) { return a - b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator*(const U &scalar) const -> ndarray<std::common_type_t<T, U>>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _scalar_op(scalar, [](const T &a, const U &b) { return a * b; });
}

template <typename T>
template <typename U>
auto ndarray<T>::operator/(const U &scalar) const -> ndarray<detail::div_result_t<T, U>>
{
    // NumPy true_divide: integral / integral promotes to double (see above).
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    _require_data();
    using R = detail::div_result_t<T, U>;
    ndarray<R> out(shape);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<U>)
        {
            out.data()[i++] = static_cast<double>(v) / static_cast<double>(scalar);
        }
        else
        {
            out.data()[i++] = v / scalar;
        }
    });
    return out;
}

template <typename T> auto ndarray<T>::operator-() const -> ndarray
{
    ndarray out(shape, type);
    std::size_t i = 0;
    _for_each_logical([&](const typename ndarray<T>::value_type &v) { out.data()[i++] = -v; });
    return out;
}

template <typename T> template <typename U> auto ndarray<T>::operator==(const ndarray<U> &rhs) const -> ndarray<bool>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a == b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator!=(const ndarray<U> &rhs) const -> ndarray<bool>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a != b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator<(const ndarray<U> &rhs) const -> ndarray<bool>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a < b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator<=(const ndarray<U> &rhs) const -> ndarray<bool>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a <= b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator>(const ndarray<U> &rhs) const -> ndarray<bool>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a > b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator>=(const ndarray<U> &rhs) const -> ndarray<bool>
{
    return detail::elementwise(*this, rhs, [](const T &a, const U &b) { return a >= b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator==(const U &scalar) const -> ndarray<bool>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _cmp_scalar(scalar, [](const T &a, const U &b) { return a == b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator!=(const U &scalar) const -> ndarray<bool>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _cmp_scalar(scalar, [](const T &a, const U &b) { return a != b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator<(const U &scalar) const -> ndarray<bool>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _cmp_scalar(scalar, [](const T &a, const U &b) { return a < b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator<=(const U &scalar) const -> ndarray<bool>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _cmp_scalar(scalar, [](const T &a, const U &b) { return a <= b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator>(const U &scalar) const -> ndarray<bool>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _cmp_scalar(scalar, [](const T &a, const U &b) { return a > b; });
}

template <typename T> template <typename U> auto ndarray<T>::operator>=(const U &scalar) const -> ndarray<bool>
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    return _cmp_scalar(scalar, [](const T &a, const U &b) { return a >= b; });
}

template <typename T> bool ndarray<T>::all_equal(const ndarray &other) const
{
    if (shape != other.shape || !data_ || !other.data_)
    {
        return false;
    }
    // No catch-all: allocation/user-comparison failures must propagate
    // (swallowing them turned OOM into a silent `false`).
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        if (!((*data_)[_flat(idx)] == (*other.data_)[other._flat(idx)]))
        {
            return false;
        }
        od.advance();
    }
    return true;
}

template <typename T> bool ndarray<T>::all_equal(const typename ndarray<T>::value_type &value) const
{
    if (!data_)
    {
        return _numel() == 0;
    }
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        if (!((*data_)[_flat(idx)] == value))
        {
            return false;
        }
        od.advance();
    }
    return true;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator+=(const ndarray<U> &rhs)
{
    _inplace_op(rhs, [&](const T &a, const U &b) { return a + b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator-=(const ndarray<U> &rhs)
{
    _inplace_op(rhs, [&](const T &a, const U &b) { return a - b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator*=(const ndarray<U> &rhs)
{
    _inplace_op(rhs, [&](const T &a, const U &b) { return a * b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator/=(const ndarray<U> &rhs)
{
    _inplace_op(rhs, [&](const T &a, const U &b) { return a / b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator+=(const U &scalar)
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return a + b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator-=(const U &scalar)
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return a - b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator*=(const U &scalar)
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return a * b; });
    return *this;
}

template <typename T> template <typename U> ndarray<T> &ndarray<T>::operator/=(const U &scalar)
{
    static_assert(_is_valid_scalar<U>, "scalar operand must be arithmetic or complex");
    _inplace_scalar(scalar, [&](const T &a, const U &b) { return a / b; });
    return *this;
}

} // namespace np

#endif // NP_NDARRAY_HPP
