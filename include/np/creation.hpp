/**
 * @file creation.hpp
 * @brief Array creation routines (np::zeros, np::ones, np::arange, ...).
 *
 * Mirrors numpy's creation API:
 *   zeros, ones, full, empty, empty_like/zeros_like/ones_like,
 *   arange, linspace, logspace, eye, identity, asarray.
 *
 * All functions return C-contiguous arrays with row-major strides.
 * The dynamic path throws std::invalid_argument on shape mismatches;
 * the fixed-shape path (creation_fixed.hpp) encodes shape in the
 * type and rejects mismatches at compile time.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_CREATION_HPP
#define NP_CREATION_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"
#include "pqc.hpp"
#include <map>
#include <type_traits>
#include <variant>
#if __cplusplus >= 202302L && __has_include(<expected>)
#include <expected>
#endif

namespace np
{

/** @brief Array of zeros with the given shape.
 *
 * @tparam T  Element type (default: double).
 * @param shape  Shape vector; must have at least one element.
 * @return       ndarray<T> of the given shape, filled with T{0}.
 * @throws       std::invalid_argument if shape is empty.
 *
 * Reference: numpy-reference/reference/generated/numpy.zeros.html
 */
#ifdef __NUMPY_RANGES_CONTAINER_CONCEPT
template <typename R>
concept _RangeStructure = std::ranges::input_range<R> && std::convertible_to<std::ranges::range_value_t<R>, int> &&
                          std::is_integral_v<std::ranges::range_value_t<R>>;
#define RangeStructure _RangeStructure
#endif

template <typename T>
concept Fillable = std::is_copy_constructible_v<T> && std::is_default_constructible_v<T>;

// Standard zeros method
NP_API template <Fillable T = double> NP_NODISCARD auto zeros(const std::vector<int> &shape) -> ndarray<T>
{
    constexpr bool is_void = (dtype_of<T> == dtype::void_);
    static_assert(!is_void || std::is_class_v<T>,
                  "zeros: T has no dtype mapping – will be stored as object_/void_ with sizeof(T) "
                  "storage. "
                  "Define cxx_to_np_type specialization or use dtype::object_ explicitly");
    dtype d = (dtype_of<T> == dtype::void_ ? dtype::object_ : dtype_of<T>);
#if __cplusplus >= 202302L && __has_include(<expected>)
    if (shape.empty())
        throw std::invalid_argument("zeros: empty shape");
#endif
    if constexpr (pqc::secure_enabled)
    {
        // Secure path: use secure_buffer + secure_zero to guarantee
        // constant-time, non-elided zeroing (PQC-hardened). For trivially
        // copyable types byte-zero == T{0}; for others fall back to fill
        // then barrier. Reference: pqc.hpp:secure_zero, pqc.hpp:secure_buffer
        ndarray<T> out(shape, d, T{0});
        out.secure_zero();
        if constexpr (!std::is_trivially_copyable_v<T>)
        {
            out.fill(T{0});
            pqc::ct_barrier();
        }
        return out;
    }
    else
    {
        return ndarray<T>(shape, d, T{0});
    }
}

template <Fillable T = double>
[[nodiscard]] inline auto try_zeros(const std::vector<int> &shape)
#if __cplusplus >= 202302L && __has_include(<expected>)
    -> std::expected<ndarray<T>, std::string>
#else
    -> ndarray<T>
#endif
{
#if __cplusplus >= 202302L && __has_include(<expected>)
    try
    {
        return zeros<T>(shape);
    }
    catch (const std::exception &e)
    {
        return std::unexpected<std::string>(e.what());
    }
#else
    return zeros<T>(shape);
#endif
}

#ifdef __NUMPY_RANGES_CONTAINER_CONCEPT
template <typename T = double, _RangeStructure R>
#else
template <typename T = double, std::ranges::input_range R>
    requires(!std::is_same_v<std::decay_t<R>, std::initializer_list<int>>) &&
            std::convertible_to<std::ranges::range_value_t<R>, int>
#endif
NP_NODISCARD NP_SYMBOL_VISIBILITY(hidden) auto __np_builtin_zeros(const R &shape) -> ndarray<T>
{
    std::vector<int> s{std::ranges::begin(shape), std::ranges::end(shape)};
    if (s.empty())
        throw std::invalid_argument("zeros: empty shape");
    if constexpr (pqc::secure_enabled)
    {
        // Secure path: allocate via secure_buffer then harden with secure_zero.
        // Mirrors numpy zeros but guarantees volatile zeroing not optimized away.
        if constexpr (std::is_same_v<T, bool>)
        {
            ndarray<T> out(s, dtype_of<T>, false);
            out.secure_zero();
            return out;
        }
        else
        {
            std::size_t n = 1;
            for (int d : s)
                n *= static_cast<std::size_t>(d);
            // Isolated secure_buffer: locked + MADV_DONTDUMP, wiped on destruction
            // and slack. Explicit secure_zero keeps volatile + fence in creation.
            pqc::secure_buffer<T> sbuf(n);
            if (n != 0)
                pqc::secure_zero(sbuf.get().data(), n * sizeof(T));
            if constexpr (!std::is_trivially_copyable_v<T>)
            {
                std::fill(sbuf.get().begin(), sbuf.get().end(), T{0});
                pqc::ct_barrier();
            }
            // release() de-isolates (munlock + allow dump) and hands off ownership
            // to ndarray; sbuf is left empty and will not double-unlock.
            return ndarray<T>::from_data(s, sbuf.release());
        }
    }
    else
    {
        return ndarray<T>(s, dtype_of<T>, T{0});
    }
}

template <typename T = double> NP_NODISCARD auto zeros(std::initializer_list<int> shape) -> ndarray<T>
{
    std::vector<int> s(shape);
    if (s.empty())
        throw std::invalid_argument("zeros: empty shape");
    if constexpr (pqc::secure_enabled)
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            ndarray<T> out(s, dtype_of<T>, false);
            out.secure_zero();
            return out;
        }
        else
        {
            std::size_t n = 1;
            for (int d : s)
                n *= static_cast<std::size_t>(d);
            pqc::secure_buffer<T> sbuf(n);
            if (n != 0)
                pqc::secure_zero(sbuf.get().data(), n * sizeof(T));
            if constexpr (!std::is_trivially_copyable_v<T>)
            {
                std::fill(sbuf.get().begin(), sbuf.get().end(), T{0});
                pqc::ct_barrier();
            }
            return ndarray<T>::from_data(s, sbuf.release());
        }
    }
    else
    {
        return ndarray<T>(s, dtype_of<T>, T{0});
    }
}

template <typename T = double, std::size_t N> NP_NODISCARD auto zeros(const int (&shape)[N]) -> ndarray<T>
{
    return __np_builtin_zeros<T>(std::span<const int, N>(shape));
}

// Generic range overload for any array-like structure (std::array, std::deque, etc.)
// Compatible with every C++ array like structure
template <typename T = double, std::ranges::input_range R>
    requires(!std::is_same_v<std::decay_t<R>, std::vector<int>>) &&
            (!std::is_same_v<std::decay_t<R>, std::initializer_list<int>>) &&
            std::convertible_to<std::ranges::range_value_t<R>, int>
NP_NODISCARD auto zeros(const R &shape) -> ndarray<T>
{
    return __np_builtin_zeros<T>(shape);
}

/** @brief Array of ones with the given shape.
 *
 * @tparam T  Element type (default: double).
 * @param shape  Shape vector; must have at least one element.
 * @return       ndarray<T> of the given shape, filled with T{1}.
 * @throws       std::invalid_argument if shape is empty.
 *
 * Reference: numpy-reference/reference/generated/numpy.ones.html
 */
NP_API template <Fillable T = double> NP_NODISCARD auto ones(const std::vector<int> &shape) -> ndarray<T>
{
    constexpr bool is_void = (dtype_of<T> == dtype::void_);
    static_assert(!is_void || std::is_class_v<T>, "ones: T has no dtype mapping – will be stored as object_/void_");
    dtype d = (dtype_of<T> == dtype::void_ ? dtype::object_ : dtype_of<T>);
    return ndarray<T>(shape, d, T{1});
}

#ifdef __NUMPY_RANGES_CONTAINER_CONCEPT
template <typename T = double, _RangeStructure R>
#else
template <typename T = double, std::ranges::input_range R>
    requires(!std::is_same_v<std::decay_t<R>, std::initializer_list<int>>) &&
            std::convertible_to<std::ranges::range_value_t<R>, int>
#endif
NP_NODISCARD NP_SYMBOL_VISIBILITY(hidden) auto __np_builtin_ones(const R &shape) -> ndarray<T>
{
    std::vector<int> s{std::ranges::begin(shape), std::ranges::end(shape)};
    if (s.empty())
        throw std::invalid_argument("ones: empty shape");
    return ndarray<T>(s, dtype_of<T>, T{1});
}

template <typename T = double> NP_NODISCARD auto ones(std::initializer_list<int> shape) -> ndarray<T>
{
    std::vector<int> s(shape);
    if (s.empty())
        throw std::invalid_argument("ones: empty shape");
    return ndarray<T>(s, dtype_of<T>, T{1});
}

template <typename T = double, std::size_t N> NP_NODISCARD auto ones(const int (&shape)[N]) -> ndarray<T>
{
    return __np_builtin_ones<T>(std::span<const int, N>(shape));
}

template <typename T = double, std::ranges::input_range R>
    requires(!std::is_same_v<std::decay_t<R>, std::vector<int>>) &&
            (!std::is_same_v<std::decay_t<R>, std::initializer_list<int>>) &&
            std::convertible_to<std::ranges::range_value_t<R>, int>
NP_NODISCARD auto ones(const R &shape) -> ndarray<T>
{
    return __np_builtin_ones<T>(shape);
}

/** @brief Array filled with a constant value.
 *
 * @tparam T  Element type (deduced from fill_value).
 * @param shape      Shape vector.
 * @param fill_value Value to fill every element with.
 * @return           ndarray<T> of the given shape, filled with fill_value.
 * @throws           std::invalid_argument if shape is empty.
 *
 * Reference: numpy-reference/reference/generated/numpy.full.html
 */
NP_API template <Fillable T> NP_NODISCARD auto full(const std::vector<int> &shape, const T &fill_value) -> ndarray<T>
{
    constexpr bool is_void = (dtype_of<T> == dtype::void_);
    static_assert(!is_void || std::is_class_v<T>, "full: T has no dtype mapping – will be stored as object_/void_");
    dtype d = (dtype_of<T> == dtype::void_ ? dtype::object_ : dtype_of<T>);
    return ndarray<T>(shape, d, fill_value);
}

#ifdef __NUMPY_RANGES_CONTAINER_CONCEPT
template <typename T, _RangeStructure R>
#else
template <typename T, std::ranges::input_range R>
    requires(!std::is_same_v<std::decay_t<R>, std::initializer_list<int>>) &&
            std::convertible_to<std::ranges::range_value_t<R>, int>
#endif
NP_NODISCARD NP_SYMBOL_VISIBILITY(hidden) auto __np_builtin_full(const R &shape, const T &fill_value) -> ndarray<T>
{
    std::vector<int> s{std::ranges::begin(shape), std::ranges::end(shape)};
    if (s.empty())
        throw std::invalid_argument("full: empty shape");
    return ndarray<T>(s, dtype_of<T>, fill_value);
}

template <typename T> NP_NODISCARD auto full(std::initializer_list<int> shape, const T &fill_value) -> ndarray<T>
{
    std::vector<int> s(shape);
    if (s.empty())
        throw std::invalid_argument("full: empty shape");
    return ndarray<T>(s, dtype_of<T>, fill_value);
}

template <typename T, std::size_t N> NP_NODISCARD auto full(const int (&shape)[N], const T &fill_value) -> ndarray<T>
{
    return __np_builtin_full(std::span<const int, N>(shape), fill_value);
}

template <typename T, std::ranges::input_range R>
    requires(!std::is_same_v<std::decay_t<R>, std::vector<int>>) &&
            (!std::is_same_v<std::decay_t<R>, std::initializer_list<int>>) &&
            std::convertible_to<std::ranges::range_value_t<R>, int>
NP_NODISCARD auto full(const R &shape, const T &fill_value) -> ndarray<T>
{
    return __np_builtin_full(shape, fill_value);
}

/** @brief Uninitialized array (values are default-constructed in C++).
 *
 * The memory is allocated but not zeroed; elements hold their
 * default-constructed values. For scalar types this means
 * indeterminate values for built-in types (same as `new T[n]`).
 *
 * @tparam T  Element type (default: double).
 * @param shape  Shape vector.
 * @return       ndarray<T> of the given shape with default-constructed
 * elements.
 * @throws       std::invalid_argument if shape is empty.
 *
 * Reference: numpy-reference/reference/generated/numpy.empty.html
 */
NP_API template <Fillable T = double> NP_NODISCARD auto empty(const std::vector<int> &shape) -> ndarray<T>
{
    constexpr bool is_void = (dtype_of<T> == dtype::void_);
    static_assert(!is_void || std::is_class_v<T>, "empty: T has no dtype mapping – will be stored as object_/void_");
    dtype d = (dtype_of<T> == dtype::void_ ? dtype::object_ : dtype_of<T>);
    return ndarray<T>(shape, d, T{});
}

#ifdef __NUMPY_RANGES_CONTAINER_CONCEPT
template <typename T = double, _RangeStructure R>
#else
template <typename T = double, std::ranges::input_range R>
    requires(!std::is_same_v<std::decay_t<R>, std::initializer_list<int>>) &&
            std::convertible_to<std::ranges::range_value_t<R>, int>
#endif
NP_NODISCARD NP_SYMBOL_VISIBILITY(hidden) auto __np_builtin_empty(const R &shape) -> ndarray<T>
{
    std::vector<int> s{std::ranges::begin(shape), std::ranges::end(shape)};
    if (s.empty())
        throw std::invalid_argument("empty: empty shape");
    return ndarray<T>(s, dtype_of<T>, T{});
}

template <typename T = double> NP_NODISCARD auto empty(std::initializer_list<int> shape) -> ndarray<T>
{
    std::vector<int> s(shape);
    if (s.empty())
        throw std::invalid_argument("empty: empty shape");
    return ndarray<T>(s, dtype_of<T>, T{});
}

template <typename T = double, std::size_t N> NP_NODISCARD auto empty(const int (&shape)[N]) -> ndarray<T>
{
    return __np_builtin_empty<T>(std::span<const int, N>(shape));
}

template <typename T = double, std::ranges::input_range R>
    requires(!std::is_same_v<std::decay_t<R>, std::vector<int>>) &&
            (!std::is_same_v<std::decay_t<R>, std::initializer_list<int>>) &&
            std::convertible_to<std::ranges::range_value_t<R>, int>
NP_NODISCARD auto empty(const R &shape) -> ndarray<T>
{
    return __np_builtin_empty<T>(shape);
}

/** @brief New array with the same shape as `a` (uninitialized).
 *
 * @tparam T  Element type of `a`.
 * @param a   Source array whose shape is copied.
 * @return    ndarray<T> with the same shape as `a`, default-constructed
 * elements.
 */
NP_API template <typename T> NP_NODISCARD auto empty_like(const ndarray<T> &a) -> ndarray<T>
{
    return ndarray<T>(a.shape, a.type);
}

/** @brief Zeros with the same shape as `a`.
 *
 * @tparam T  Element type of `a`.
 * @param a   Source array whose shape is copied.
 * @return    ndarray<T> with the same shape as `a`, filled with T{0}.
 */
NP_API template <typename T> NP_NODISCARD auto zeros_like(const ndarray<T> &a) -> ndarray<T>
{
    if constexpr (pqc::secure_enabled)
    {
        ndarray<T> out(a.shape, a.type, T{0});
        out.secure_zero();
        if constexpr (!std::is_trivially_copyable_v<T>)
        {
            out.fill(T{0});
            pqc::ct_barrier();
        }
        return out;
    }
    else
    {
        return ndarray<T>(a.shape, a.type, T{0});
    }
}

/** @brief Ones with the same shape as `a`.
 *
 * @tparam T  Element type of `a`.
 * @param a   Source array whose shape is copied.
 * @return    ndarray<T> with the same shape as `a`, filled with T{1}.
 */
NP_API template <typename T> NP_NODISCARD auto ones_like(const ndarray<T> &a) -> ndarray<T>
{
    return ndarray<T>(a.shape, a.type, T{1});
}

/** @brief Filled with `fill_value` using the shape and dtype of `a`.
 *
 * @tparam T       Element type of `a` and `fill_value`.
 * @param a        Source array whose shape and dtype are copied.
 * @param fill_value Value to fill every element with.
 * @return         ndarray<T> with the shape and dtype of `a`, filled with
 * fill_value.
 */
NP_API template <typename T> NP_NODISCARD auto full_like(const ndarray<T> &a, const T &fill_value) -> ndarray<T>
{
    return ndarray<T>(a.shape, a.type, fill_value);
}

/** @brief Values evenly spaced from start (inclusive) to stop (exclusive).
 *
 * Computes the number of elements as ceil((stop - start) / step),
 * then generates start + step * i for i in [0, n). If step > 0 and
 * stop <= start, returns a 1-D array of size 0. If step < 0 and
 * stop >= start, returns a 1-D array of size 0.
 *
 * @tparam T  Element type (deduced from arguments).
 * @param start  Start value (inclusive).
 * @param stop   Stop value (exclusive).
 * @param step   Step size (default: T{1}); must not be zero.
 * @return       1-D ndarray<T> of the computed length.
 * @throws       std::invalid_argument if step is zero.
 *
 * Reference: numpy-reference/reference/generated/numpy.arange.html
 */
NP_API template <typename T> NP_NODISCARD auto arange(T start, T stop, T step = T{1}) -> ndarray<T>
{
    if (step == T{0})
    {
        throw std::invalid_argument("arange step cannot be zero");
    }
    std::vector<T> out;
    if (step > T{0})
    {
        if (stop <= start)
        {
            return ndarray<T>({0}, dtype_of<T>, T{});
        }
        const std::size_t n = static_cast<std::size_t>(
            std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(step)));
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            out.push_back(start + step * static_cast<T>(i));
        }
    }
    else
    {
        if (stop >= start)
        {
            return ndarray<T>({0}, dtype_of<T>, T{});
        }
        const std::size_t n = static_cast<std::size_t>(
            std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(step)));
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            out.push_back(start + step * static_cast<T>(i));
        }
    }
    const int n_elems = static_cast<int>(out.size());
    return ndarray<T>::from_data(std::vector<int>{n_elems}, std::move(out));
}

/** @brief Values from 0 to stop (exclusive).
 *
 * Equivalent to arange(T{0}, stop, T{1}).
 *
 * @tparam T  Element type (deduced from stop).
 * @param stop  Exclusive upper bound.
 * @return      1-D ndarray<T> of length max(0, ceil(stop)).
 */
NP_API template <typename T> NP_NODISCARD auto arange(T stop) -> ndarray<T>
{
    return arange(T{0}, stop, T{1});
}

/** @brief num evenly spaced values from start to stop (inclusive).
 *
 * When endpoint is true (default), the sequence includes stop.
 * When endpoint is false, stop is excluded and the step is
 * (stop - start) / num. Integer inputs are promoted to double.
 *
 * @tparam T  Element type (deduced from start/stop).
 * @param start    Start value (inclusive).
 * @param stop     Stop value (inclusive when endpoint is true).
 * @param num      Number of samples (default: 50); must be > 0.
 * @param endpoint Whether to include stop in the sequence.
 * @return         1-D ndarray<R> where R is double if T is integral,
 *                 otherwise T.
 *
 * Reference: numpy-reference/reference/generated/numpy.linspace.html
 */
NP_API template <typename T>
NP_NODISCARD auto linspace(T start, T stop, std::size_t num = 50, bool endpoint = true)
    -> ndarray<std::conditional_t<std::is_floating_point_v<T>, T, double>>
{
    using R = std::conditional_t<std::is_floating_point_v<T>, T, double>;
    if (num == 0)
    {
        return ndarray<R>(std::vector<int>{0});
    }
    std::vector<R> out;
    out.reserve(num);
    if (num == 1)
    {
        out.push_back(static_cast<R>(start));
        return ndarray<R>::from_data(std::vector<int>{1}, std::move(out));
    }
    const R delta = endpoint ? (static_cast<R>(stop) - static_cast<R>(start)) / static_cast<R>(num - 1)
                             : (static_cast<R>(stop) - static_cast<R>(start)) / static_cast<R>(num);
    for (std::size_t i = 0; i < num; ++i)
    {
        out.push_back(static_cast<R>(start) + delta * static_cast<R>(i));
    }
    return ndarray<R>::from_data(std::vector<int>{static_cast<int>(num)}, std::move(out));
}

/** @brief Logarithmically spaced values from base^start to base^stop.
 *
 * Uses linspace internally to generate the exponent values, then
 * applies std::pow(base, exponent) element-wise.
 *
 * @tparam T  Element type (deduced from start/stop).
 * @param start  Start exponent (inclusive).
 * @param stop   Stop exponent (inclusive).
 * @param num    Number of samples (default: 50).
 * @param base   The base of the logarithm (default: T{10}).
 * @return       1-D ndarray<double> of num elements.
 *
 * Reference: numpy-reference/reference/generated/numpy.logspace.html
 */
NP_API template <typename T>
NP_NODISCARD auto logspace(T start, T stop, std::size_t num = 50, T base = T{10}) -> ndarray<double>
{
    auto powers = linspace(start, stop, num);
    ndarray<double> out(std::vector<int>{static_cast<int>(num)});
    for (std::size_t i = 0; i < num; ++i)
    {
        out.data()[i] = std::pow(static_cast<double>(base), static_cast<double>(powers.data()[i]));
    }
    return out;
}

/** @brief Identity matrix of size n x n with optional offset k.
 *
 * The diagonal at offset k is set to T{1}. k > 0 places the
 * diagonal above the main diagonal; k < 0 below it.
 *
 * @tparam T  Element type (default: double).
 * @param n  Number of rows.
 * @param m  Number of columns (optional, default n – square).
 * @param k  Diagonal offset (default: 0).
 * @return   ndarray<T> of shape (n, m) with ones on the k-th diagonal.
 *
 * Reference: numpy-reference/reference/generated/numpy.eye.html
 */
NP_API template <typename T = double>
NP_NODISCARD auto eye(std::size_t n, std::optional<std::size_t> m = std::nullopt, std::optional<int> k = std::nullopt)
    -> ndarray<T>
{
    std::size_t cols = m.has_value() ? *m : n;
    if (cols == 0)
        cols = n; // keep backward compat: eye(n,0) -> n x n
    const int kk = k.value_or(0);
    std::vector<int> shape = {static_cast<int>(n), static_cast<int>(cols)};
    ndarray<T> out(shape, dtype_of<T>, T{0});
    const std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(n);
    const std::ptrdiff_t cols_p = static_cast<std::ptrdiff_t>(cols);
    for (std::ptrdiff_t i = 0; i < rows; ++i)
    {
        const std::ptrdiff_t j = i + kk;
        if (j >= 0 && j < cols_p)
        {
            out.set(std::array<std::size_t, 2>{static_cast<std::size_t>(i), static_cast<std::size_t>(j)}, T{1});
        }
    }
    return out;
}

/** @brief Identity matrix of size n x n.
 *
 * Equivalent to eye<T>(n, n, 0).
 *
 * @tparam T  Element type (default: double).
 * @param n  Size of the square matrix.
 * @return   ndarray<T> of shape (n, n) with ones on the main diagonal.
 *
 * Reference: numpy-reference/reference/generated/numpy.identity.html
 */
NP_API template <typename T = double> NP_NODISCARD auto identity(std::size_t n) -> ndarray<T>
{
    return eye<T>(n, n, 0);
}

// Runtime dtype factory: create ndarray with dtype enum at runtime
// Returns variant of common dtypes; for void/string etc. returns empty
NP_API inline auto ndarray_from_dtype(dtype type, const std::vector<int> &shape)
    -> std::variant<ndarray<std::int8_t>, ndarray<std::int16_t>, ndarray<std::int32_t>, ndarray<std::int64_t>,
                    ndarray<std::uint8_t>, ndarray<std::uint16_t>, ndarray<std::uint32_t>, ndarray<std::uint64_t>,
                    ndarray<float>, ndarray<double>, ndarray<long double>, ndarray<std::complex<float>>,
                    ndarray<std::complex<double>>, ndarray<std::complex<long double>>, ndarray<bool>>
{
    switch (type)
    {
    case dtype::int8:
        return ndarray<std::int8_t>(shape);
    case dtype::int16:
        return ndarray<std::int16_t>(shape);
    case dtype::int32:
        return ndarray<std::int32_t>(shape);
    case dtype::int64:
        return ndarray<std::int64_t>(shape);
    case dtype::uint8:
        return ndarray<std::uint8_t>(shape);
    case dtype::uint16:
        return ndarray<std::uint16_t>(shape);
    case dtype::uint32:
        return ndarray<std::uint32_t>(shape);
    case dtype::uint64:
        return ndarray<std::uint64_t>(shape);
    case dtype::float32:
        return ndarray<float>(shape);
    case dtype::float64:
        return ndarray<double>(shape);
    case dtype::longdouble:
        return ndarray<long double>(shape);
    case dtype::complex64:
        return ndarray<std::complex<float>>(shape);
    case dtype::complex128:
        return ndarray<std::complex<double>>(shape);
    case dtype::clongdouble:
        return ndarray<std::complex<long double>>(shape);
    case dtype::bool_:
        return ndarray<bool>(shape);
    default:
        return ndarray<std::int8_t>(shape);
    }
}

// Overload for dtype_tag (compile-time dtype as np::complex128 etc.)
template <dtype D>
NP_API inline auto ndarray_from_dtype(dtype_tag<D>, const std::vector<int> &shape)
    -> ndarray<typename dtype_tag<D>::type>
{
    return ndarray<typename dtype_tag<D>::type>(shape);
}

/** @brief 1D array from a std::vector (copies).
 *
 * @tparam T  Element type.
 * @param values  Source vector (copied).
 * @return        1-D ndarray<T> with the same elements as values.
 */
NP_API template <typename T> NP_NODISCARD auto asarray(const std::vector<T> &values) -> ndarray<T>
{
    return ndarray<T>::from_data(std::vector<int>{static_cast<int>(values.size())}, std::vector<T>(values));
}

/** @brief 1D array from a std::array (copies).
 *
 * @tparam T  Element type.
 * @tparam N  Size of the source array.
 * @param values  Source std::array (copied).
 * @return        1-D ndarray<T> with the same elements.
 */
NP_API template <typename T, std::size_t N> NP_NODISCARD auto asarray(const std::array<T, N> &values) -> ndarray<T>
{
    return ndarray<T>::from_data(std::vector<int>{static_cast<int>(N)}, std::vector<T>(values.begin(), values.end()));
}

/** @brief Array of the given shape from a contiguous std::vector.
 *
 * The total number of elements in values must equal the product
 * of the shape dimensions. The data is copied.
 *
 * @tparam T  Element type.
 * @param values  Source vector (copied).
 * @param shape   Target shape; must have at least one element.
 * @return        ndarray<T> of the given shape.
 * @throws        std::invalid_argument if sizes do not match.
 *
 * Reference: numpy-reference/reference/generated/numpy.asarray.html
 */
NP_API template <typename T>
NP_NODISCARD auto asarray(const std::vector<T> &values, const std::vector<int> &shape) -> ndarray<T>
{
    return ndarray<T>::from_data(shape, std::vector<T>(values));
}

/** @brief Geometrically spaced values (log-spaced between start and stop).
 *
 * Equivalent to `np.geomspace` with base 10 geometric progression.
 * For integer inputs the result is double. Requires start and stop
 * non-zero and with the same sign; otherwise std::invalid_argument
 * is thrown (mirrors NumPy's behaviour for negative inputs when
 * num samples would cross zero via log).
 *
 * @tparam T Element type (floating-point promoted to double for integers).
 * @param start Start value (inclusive, non-zero).
 * @param stop Stop value (inclusive, non-zero).
 * @param num Number of samples (default 50).
 * @param endpoint Whether to include stop.
 * @return 1-D ndarray<double> of num elements.
 *
 * Reference: numpy-reference/reference/generated/numpy.geomspace.html
 */
NP_API template <typename T>
NP_NODISCARD auto geomspace(T start, T stop, std::size_t num = 50, bool endpoint = true) -> ndarray<double>
{
    if (num == 0)
    {
        return ndarray<double>(std::vector<int>{0});
    }
    double s = static_cast<double>(start);
    double e = static_cast<double>(stop);
    if (s == 0.0 || e == 0.0)
    {
        throw std::invalid_argument("geomspace: start and stop must be non-zero");
    }
    // Numpy allows negative start/stop only if they share the sign
    if ((s < 0) != (e < 0))
    {
        throw std::invalid_argument("geomspace: start and stop must have same sign");
    }
    bool neg = s < 0;
    double ls = std::log10(std::abs(s));
    double le = std::log10(std::abs(e));
    auto p = linspace(ls, le, num, endpoint);
    ndarray<double> out(std::vector<int>{static_cast<int>(num)});
    for (std::size_t i = 0; i < num; ++i)
    {
        double v = std::pow(10.0, p.data()[i]);
        out.data()[i] = neg ? -v : v;
    }
    return out;
}

/** @brief Indices of an N-dimensional array.
 *
 * Returns an array of shape (ndim, dim0, dim1, ...) where
 * result(0, ...) contains row indices, result(1, ...) column, etc.
 * Mirrors `np.indices`.
 *
 * @param dimensions Shape of the desired index grid.
 * @param dtype Ignored – always int.
 * @return ndarray<int> with ndim leading dimension.
 *
 * Reference: numpy-reference/reference/generated/numpy.indices.html
 */
NP_API inline auto indices(const std::vector<int> &dimensions) -> ndarray<int>
{
    if (dimensions.empty())
    {
        throw std::invalid_argument("indices: dimensions must be non-empty");
    }
    std::vector<int> out_shape;
    out_shape.push_back(static_cast<int>(dimensions.size()));
    out_shape.insert(out_shape.end(), dimensions.begin(), dimensions.end());
    ndarray<int> out(out_shape);
    // Fill using odometer
    std::vector<std::size_t> idx(dimensions.size(), 0);
    // Iterate over all positions in the grid
    detail::Odometer od(dimensions);
    while (!od.done())
    {
        const auto &pos = od.idx();
        for (std::size_t d = 0; d < dimensions.size(); ++d)
        {
            std::vector<std::size_t> full(dimensions.size() + 1, 0);
            full[0] = d;
            for (std::size_t k = 0; k < dimensions.size(); ++k)
                full[k + 1] = pos[k];
            out.set(full, static_cast<int>(pos[d]));
        }
        od.advance();
    }
    return out;
}

/** @brief Construct an array from a function over indices.
 *
 * Calls `func` for every index tuple and stores the result.
 * The callable receives `std::vector<std::size_t>` of length
 * `shape.size()` and returns T.
 *
 * @tparam T Element type.
 * @tparam Fn Callable `T(const std::vector<std::size_t>&)`.
 * @param shape Desired shape.
 * @param func Function generating elements.
 * @return ndarray<T> filled via func.
 *
 * Reference: numpy-reference/reference/generated/numpy.fromfunction.html
 */
NP_API template <typename T, typename Fn>
NP_NODISCARD auto fromfunction(const std::vector<int> &shape, Fn &&func) -> ndarray<T>
{
    ndarray<T> out(shape);
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &pos = od.idx();
        out.set(pos, func(pos));
        od.advance();
    }
    return out;
}

/** @brief Meshgrid for 1-D coordinate vectors.
 *
 * Supports 2-input Cartesian meshgrids (the most common NumPy use).
 * indexing='ij' (matrix) gives shape (len(x), len(y)) transposed,
 * indexing='xy' (default, Cartesian) gives shape (len(y), len(x)).
 * For N>2 use the vector overload below.
 *
 * @tparam T Element type (common type of inputs).
 * @param x First coordinate vector (1-D).
 * @param y Second coordinate vector (1-D).
 * @param indexing 'xy' or 'ij' (default 'xy').
 * @return Pair {X, Y} broadcast grids.
 *
 * Reference: numpy-reference/reference/generated/numpy.meshgrid.html
 */
NP_API template <typename T>
NP_NODISCARD auto meshgrid(const ndarray<T> &x, const ndarray<T> &y, const std::string &indexing = "xy")
    -> std::pair<ndarray<T>, ndarray<T>>
{
    if (x.ndim() != 1 || y.ndim() != 1)
    {
        throw std::invalid_argument("meshgrid: inputs must be 1-D");
    }
    if (indexing != "xy" && indexing != "ij")
    {
        throw std::invalid_argument("meshgrid: indexing must be 'xy' or 'ij'");
    }
    std::size_t nx = x.size();
    std::size_t ny = y.size();
    ndarray<T> X, Y;
    if (indexing == "xy")
    {
        X = ndarray<T>(std::vector<int>{static_cast<int>(ny), static_cast<int>(nx)});
        Y = ndarray<T>(std::vector<int>{static_cast<int>(ny), static_cast<int>(nx)});
        for (std::size_t i = 0; i < ny; ++i)
        {
            for (std::size_t j = 0; j < nx; ++j)
            {
                X.at(i, j) = x.at(j);
                Y.at(i, j) = y.at(i);
            }
        }
    }
    else
    {
        X = ndarray<T>(std::vector<int>{static_cast<int>(nx), static_cast<int>(ny)});
        Y = ndarray<T>(std::vector<int>{static_cast<int>(nx), static_cast<int>(ny)});
        for (std::size_t i = 0; i < nx; ++i)
        {
            for (std::size_t j = 0; j < ny; ++j)
            {
                X.at(i, j) = x.at(i);
                Y.at(i, j) = y.at(j);
            }
        }
    }
    return {X, Y};
}

/** @brief N-dimensional meshgrid (vector version).
 *
 * @tparam T Element type.
 * @param arrays Vector of 1-D coordinate arrays.
 * @param indexing 'xy' or 'ij'.
 * @return Vector of N broadcast grids, each with shape
 *         (len0, len1, ...) with xy swap on first two axes when
 *         indexing=='xy'.
 *
 * Reference: numpy-reference/reference/generated/numpy.meshgrid.html
 */
NP_API template <typename T>
NP_NODISCARD auto meshgrid(const std::vector<ndarray<T>> &arrays, const std::string &indexing = "xy")
    -> std::vector<ndarray<T>>
{
    if (arrays.empty())
    {
        throw std::invalid_argument("meshgrid: at least one array required");
    }
    for (auto &a : arrays)
    {
        if (a.ndim() != 1)
            throw std::invalid_argument("meshgrid: inputs must be 1-D");
    }
    std::size_t N = arrays.size();
    // Build shape: lens in order, swapping first two when xy and N>=2
    std::vector<int> base_shape;
    base_shape.reserve(N);
    for (auto &a : arrays)
        base_shape.push_back(static_cast<int>(a.size()));
    std::vector<int> out_shape = base_shape;
    if (indexing == "xy" && N >= 2)
        std::swap(out_shape[0], out_shape[1]);

    std::vector<ndarray<T>> grids;
    grids.reserve(N);
    for (std::size_t n = 0; n < N; ++n)
    {
        grids.emplace_back(out_shape);
    }
    // For each output position, source index per grid is position mapped
    // through shape permutation.
    detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &pos = od.idx();
        for (std::size_t n = 0; n < N; ++n)
        {
            std::size_t src;
            if (indexing == "xy" && N >= 2)
            {
                if (n == 0)
                    src = pos[1];
                else if (n == 1)
                    src = pos[0];
                else
                    src = pos[n];
            }
            else
            {
                src = pos[n];
            }
            grids[n].set(pos, arrays[n].at(src));
        }
        od.advance();
    }
    return grids;
}

/** @brief Convert input to ndarray – passes through if already ndarray (np.asanyarray).
 * Reference: numpy.asanyarray
 */
NP_API template <typename T> NP_NODISCARD auto asanyarray(const ndarray<T> &a) -> ndarray<T>
{
    return a;
}
NP_API template <typename T> NP_NODISCARD auto asanyarray(const std::vector<T> &v) -> ndarray<T>
{
    return asarray(v);
}

/** @brief Ensure C-contiguous (np.ascontiguousarray). */
NP_API template <typename T> NP_NODISCARD auto ascontiguousarray(const ndarray<T> &a) -> ndarray<T>
{
    if (a.is_contiguous())
        return a;
    return a.copy();
}

/** @brief Create array from buffer (np.frombuffer). Copies data.
 * @param buffer raw vector of bytes reinterpreted as T
 * @param dtype ignored – T determines dtype
 * @param count number of items to read (-1 all)
 * @param offset bytes to skip
 */
NP_API template <typename T>
NP_NODISCARD auto frombuffer(const std::vector<char> &buffer, int count = -1, std::size_t offset = 0) -> ndarray<T>
{
    if (offset > buffer.size())
        throw std::invalid_argument("frombuffer: offset out of range");
    std::size_t avail = (buffer.size() - offset) / sizeof(T);
    std::size_t n = count < 0 ? avail : static_cast<std::size_t>(count);
    if (n > avail)
        throw std::invalid_argument("frombuffer: count exceeds buffer");
    ndarray<T> out(std::vector<int>{static_cast<int>(n)});
    std::memcpy(out.data().data(), buffer.data() + offset, n * sizeof(T));
    return out;
}

/** @brief Alias to asarray with order flag (np.require). Only C flag honored. */
NP_API template <typename T>
NP_NODISCARD auto require(const ndarray<T> &a, const std::string &requirements = "C") -> ndarray<T>
{
    if (requirements.find('C') != std::string::npos || requirements.find('A') != std::string::npos)
    {
        if (a.is_contiguous())
            return a;
        return a.copy();
    }
    return a;
}

// Normal comment: additional creation helpers

/** @brief Create array from object – alias to asarray (np.array). */
NP_API template <typename T> NP_NODISCARD auto array(const std::vector<T> &v) -> ndarray<T>
{
    return asarray(v);
}
NP_API template <typename T> NP_NODISCARD auto array(const ndarray<T> &a) -> ndarray<T>
{
    return a.copy();
}

/** @brief Deep copy (np.copy). */
NP_API template <typename T> NP_NODISCARD auto copy(const ndarray<T> &a) -> ndarray<T>
{
    return a.copy();
}

/** @brief Create array from iterable (np.fromiter) – copies count elements. */
NP_API template <typename T, typename Iter>
NP_NODISCARD auto fromiter(Iter begin, Iter end, int count = -1) -> ndarray<T>
{
    std::vector<T> out;
    for (auto it = begin; it != end && (count < 0 || static_cast<int>(out.size()) < count); ++it)
        out.push_back(static_cast<T>(*it));
    if (count >= 0 && static_cast<int>(out.size()) < count)
        throw std::invalid_argument("fromiter: not enough elements");
    int n = static_cast<int>(out.size());
    return ndarray<T>::from_data(std::vector<int>{n}, std::move(out));
}

/** @brief Create array from string (np.fromstring) – splits by sep. */
NP_API template <typename T>
NP_NODISCARD auto fromstring(const std::string &s, const std::string &sep = " ") -> ndarray<T>
{
    std::vector<T> out;
    if (sep == " " || sep.empty())
    {
        std::istringstream iss(s);
        T v;
        while (iss >> v)
            out.push_back(v);
    }
    else
    {
        std::string cur;
        for (char c : s)
        {
            if (sep.find(c) != std::string::npos)
            {
                if (!cur.empty())
                {
                    std::istringstream iss(cur);
                    T v;
                    iss >> v;
                    out.push_back(v);
                    cur.clear();
                }
            }
            else
                cur.push_back(c);
        }
        if (!cur.empty())
        {
            std::istringstream iss(cur);
            T v;
            iss >> v;
            out.push_back(v);
        }
    }
    int n = static_cast<int>(out.size());
    return ndarray<T>::from_data(std::vector<int>{n}, std::move(out));
}

/** @brief Check finite and convert (np.asarray_chkfinite). */
NP_API template <typename T> NP_NODISCARD auto asarray_chkfinite(const ndarray<T> &a) -> ndarray<T>
{
    for (auto v : a)
        if (!std::isfinite(static_cast<double>(v)))
            throw std::invalid_argument("asarray_chkfinite: non-finite value");
    return a;
}

/** @brief Convert to float array (np.asfarray) – promotes to double if needed. */
NP_API template <typename T> NP_NODISCARD auto asfarray(const ndarray<T> &a) -> ndarray<double>
{
    return a.template astype<double>();
}

/**
 * @brief Return an array laid out in Fortran order (np.asfortranarray).
 *
 * In NumPy this guarantees F-contiguity (column-major). Here we
 * materialise a copy whose data is laid out so that
 * `is_f_contiguous()` is true – i.e. strides follow Fortran order
 * while remaining correct under the ndarray's strided view model.
 * For 0-D/1-D the layout is trivially both C and F contiguous.
 *
 * Reference: numpy-reference/reference/generated/numpy.asfortranarray.html
 */
NP_API template <typename T> NP_NODISCARD auto asfortranarray(const ndarray<T> &a) -> ndarray<T>
{
    if (a.ndim() <= 1)
    {
        return a.copy();
    }
    if (a.is_f_contiguous())
    {
        return a.copy();
    }
    ndarray<T> out(a.shape);
    // Build F-order strides: stride[0]=1, stride[d]=prod_{k<d} shape[k]
    std::vector<std::size_t> fstr(a.ndim());
    std::size_t s = 1;
    for (std::size_t d = 0; d < a.ndim(); ++d)
    {
        fstr[d] = s;
        s *= static_cast<std::size_t>(a.shape[d]);
    }
    out.strides = fstr;
    out.offset = 0;
    // Copy logical values so physical layout becomes F-contiguous.
    detail::Odometer od(a.shape);
    while (!od.done())
    {
        out.set(od.idx(), a.get(od.idx()));
        od.advance();
    }
    return out;
}

/**
 * @brief Create an array from an object implementing __dlpack__.
 *
 * Python NumPy's `from_dlpack(x)` consumes the DLPack capsule.
 * In C++ there is no capsule protocol; the closest equivalent is an
 * `ndarray<U>` (or anything convertible to it). This overload passes
 * through `ndarray<T>` unchanged and converts convertible ranges via
 * `asarray`. For truly opaque DLPack objects the caller should
 * materialise an `ndarray` first. The overload exists for API parity.
 *
 * Reference: numpy-reference/reference/generated/numpy.from_dlpack.html
 */
NP_API template <typename T> NP_NODISCARD auto from_dlpack(const ndarray<T> &x) -> ndarray<T>
{
    return x.copy();
}

NP_API template <typename T> NP_NODISCARD auto from_dlpack(ndarray<T> &&x) -> ndarray<T>
{
    return std::move(x);
}

NP_API template <typename T> NP_NODISCARD auto from_dlpack(const std::vector<T> &obj) -> ndarray<T>
{
    return asarray(obj);
}

/**
 * @brief Interpret input as a matrix – legacy alias (np.asmatrix).
 *
 * NumPy's `matrix` subclass is deprecated; here it is an alias to
 * a 2-D `ndarray` (row-major). If input is 1-D it becomes (1, N).
 *
 * Reference: numpy-reference/reference/generated/numpy.asmatrix.html
 */
NP_API template <typename T> NP_NODISCARD auto asmatrix(const ndarray<T> &data) -> ndarray<T>
{
    if (data.ndim() == 0)
    {
        return data.reshape(std::vector<int>{1, 1});
    }
    if (data.ndim() == 1)
    {
        return data.reshape(std::vector<int>{1, static_cast<int>(data.size())});
    }
    return data.copy();
}

/**
 * @brief Build a matrix from string/nested sequence (np.bmat).
 *
 * The Python form parses strings like "1 2; 3 4". Here we provide the
 * array form: `bmat({{a, b}, {c, d}})` assembles via `block` semantics.
 * For API parity a 1-D string overload is also accepted but simply
 * throws – callers should use the nested-vector form.
 *
 * Reference: numpy-reference/reference/generated/numpy.bmat.html
 */
NP_API template <typename T> NP_NODISCARD auto bmat(const std::vector<std::vector<ndarray<T>>> &obj) -> ndarray<T>
{
    // Delegate to block in manipulation.hpp would be ideal; to avoid
    // circular dependency we implement a minimal 2-D block here and
    // let manipulation.hpp's richer `block` be preferred for general use.
    if (obj.empty() || obj[0].empty())
    {
        throw std::invalid_argument("bmat: empty object");
    }
    int rows0 = obj[0][0].shape[0];
    for (auto &row : obj)
    {
        for (auto &b : row)
        {
            if (b.ndim() != 2)
            {
                throw std::invalid_argument("bmat: all blocks must be 2-D");
            }
        }
        int h = row[0].shape[0];
        (void)rows0;
        (void)h;
    }
    // Reuse block logic by including manipulation would be circular;
    // fallback: horizontal stack rows then vertical stack – rely on caller
    // to include manipulation.hpp for richer path; here do manual.
    std::vector<ndarray<T>> row_stacked;
    row_stacked.reserve(obj.size());
    for (auto &row : obj)
    {
        int h = row[0].shape[0];
        int total_w = 0;
        for (auto &b : row)
        {
            if (b.shape[0] != h)
            {
                throw std::invalid_argument("bmat: row blocks must agree in rows");
            }
            total_w += b.shape[1];
        }
        ndarray<T> r(std::vector<int>{h, total_w});
        int col_off = 0;
        for (auto &b : row)
        {
            for (int i = 0; i < h; ++i)
            {
                for (int j = 0; j < b.shape[1]; ++j)
                {
                    r.at(static_cast<std::size_t>(i), static_cast<std::size_t>(col_off + j)) =
                        b.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
                }
            }
            col_off += b.shape[1];
        }
        row_stacked.push_back(std::move(r));
    }
    int total_h = 0;
    int w = row_stacked[0].shape[1];
    for (auto &r : row_stacked)
    {
        if (r.shape[1] != w)
        {
            throw std::invalid_argument("bmat: rows must agree in width");
        }
        total_h += r.shape[0];
    }
    ndarray<T> out(std::vector<int>{total_h, w});
    int row_off = 0;
    for (auto &r : row_stacked)
    {
        for (int i = 0; i < r.shape[0]; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                out.at(static_cast<std::size_t>(row_off + i), static_cast<std::size_t>(j)) =
                    r.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            }
        }
        row_off += r.shape[0];
    }
    return out;
}

// Normal comment: index creation helpers

NP_API inline auto diag_indices(int n, int ndim = 2) -> std::vector<ndarray<int>>
{
    if (n < 0 || ndim <= 0)
        throw std::invalid_argument("diag_indices: invalid n/ndim");
    std::vector<ndarray<int>> out;
    out.reserve(ndim);
    for (int d = 0; d < ndim; ++d)
    {
        ndarray<int> idx(std::vector<int>{n});
        for (int i = 0; i < n; ++i)
            idx.data()[i] = i;
        out.push_back(std::move(idx));
    }
    return out;
}

NP_API inline auto tril_indices(int n, int k = 0, int m = -1) -> std::pair<ndarray<int>, ndarray<int>>
{
    if (m == -1)
        m = n;
    if (n < 0 || m < 0)
        throw std::invalid_argument("tril_indices: negative dimension");
    std::vector<int> rows, cols;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if (j <= i + k)
            {
                rows.push_back(i);
                cols.push_back(j);
            }
    ndarray<int> r(std::vector<int>{static_cast<int>(rows.size())});
    ndarray<int> c(std::vector<int>{static_cast<int>(cols.size())});
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        r.data()[i] = rows[i];
        c.data()[i] = cols[i];
    }
    return {r, c};
}

NP_API inline auto triu_indices(int n, int k = 0, int m = -1) -> std::pair<ndarray<int>, ndarray<int>>
{
    if (m == -1)
        m = n;
    auto p = tril_indices(n, k - 1, m);
    // complement of tril(k-1) is triu(k)
    // Instead of set difference, generate directly
    std::vector<int> rows, cols;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if (j >= i + k)
            {
                rows.push_back(i);
                cols.push_back(j);
            }
    ndarray<int> r(std::vector<int>{static_cast<int>(rows.size())});
    ndarray<int> c(std::vector<int>{static_cast<int>(cols.size())});
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        r.data()[i] = rows[i];
        c.data()[i] = cols[i];
    }
    return {r, c};
}

NP_API inline auto mask_indices(int n, bool (*mask_func)(int, int), int k = 0) -> std::pair<ndarray<int>, ndarray<int>>
{
    std::vector<int> rows, cols;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (mask_func(i, j))
            {
                rows.push_back(i);
                cols.push_back(j);
            }
    (void)k;
    ndarray<int> r(std::vector<int>{static_cast<int>(rows.size())});
    ndarray<int> c(std::vector<int>{static_cast<int>(cols.size())});
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        r.data()[i] = rows[i];
        c.data()[i] = cols[i];
    }
    return {r, c};
}

NP_API inline auto unravel_index(const ndarray<int> &indices, const std::vector<int> &dims) -> std::vector<ndarray<int>>
{
    if (dims.empty())
        throw std::invalid_argument("unravel_index: dims empty");
    std::size_t total = 1;
    for (int d : dims)
        total *= static_cast<std::size_t>(d);
    std::vector<ndarray<int>> out(dims.size(), ndarray<int>(std::vector<int>{static_cast<int>(indices.size())}));
    for (std::size_t idx = 0; idx < indices.size(); ++idx)
    {
        int flat = indices.data()[indices._flat_logical(idx)];
        if (flat < 0)
            flat += static_cast<int>(total);
        if (flat < 0 || static_cast<std::size_t>(flat) >= total)
            throw std::invalid_argument("unravel_index: flat out of bounds");
        int rem = flat;
        for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d)
        {
            int dim = dims[d];
            out[d].data()[idx] = rem % dim;
            rem /= dim;
        }
    }
    return out;
}

NP_API inline auto unravel_index(int flat, const std::vector<int> &dims) -> std::vector<int>
{
    ndarray<int> idx(std::vector<int>{1});
    idx.data()[0] = flat;
    auto res = unravel_index(idx, dims);
    std::vector<int> out;
    out.reserve(res.size());
    for (auto &arr : res)
        out.push_back(arr.data()[0]);
    return out;
}

NP_API inline auto ravel_multi_index(const std::vector<ndarray<int>> &indices, const std::vector<int> &dims,
                                     const std::string &mode = "raise", const std::string &order = "C") -> ndarray<int>
{
    if (indices.empty())
        throw std::invalid_argument("ravel_multi_index: empty indices");
    std::size_t n = indices[0].size();
    for (auto &arr : indices)
        if (arr.size() != n)
            throw std::invalid_argument("ravel_multi_index: indices size mismatch");
    if (indices.size() != dims.size())
        throw std::invalid_argument("ravel_multi_index: dims size mismatch");
    ndarray<int> out(std::vector<int>{static_cast<int>(n)});
    for (std::size_t i = 0; i < n; ++i)
    {
        int flat = 0;
        if (order == "C" || order == "c")
        {
            for (std::size_t d = 0; d < dims.size(); ++d)
            {
                int idx = indices[d].data()[indices[d]._flat_logical(i)];
                if (idx < 0)
                    idx += dims[d];
                if (mode == "clip")
                    idx = std::clamp(idx, 0, dims[d] - 1);
                else if (idx < 0 || idx >= dims[d])
                    throw std::invalid_argument("ravel_multi_index: out of bounds");
                flat = flat * dims[d] + idx;
            }
        }
        else // Fortran
        {
            int stride = 1;
            for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d)
            {
                int idx = indices[d].data()[indices[d]._flat_logical(i)];
                if (idx < 0)
                    idx += dims[d];
                flat += idx * stride;
                stride *= dims[d];
            }
        }
        out.data()[i] = flat;
    }
    return out;
}

// Normal comment: remaining index helpers

NP_API inline auto diag_indices_from(const ndarray<int> &arr) -> std::vector<ndarray<int>>
{
    if (arr.ndim() < 2)
        throw std::invalid_argument("diag_indices_from: need at least 2-D");
    int n = std::min(arr.shape[0], arr.shape[1]);
    return diag_indices(n, 2);
}

NP_API inline auto tril_indices_from(const ndarray<int> &arr, int k = 0) -> std::pair<ndarray<int>, ndarray<int>>
{
    if (arr.ndim() != 2)
        throw std::invalid_argument("tril_indices_from: need 2-D");
    return tril_indices(arr.shape[0], k, arr.shape[1]);
}

NP_API inline auto triu_indices_from(const ndarray<int> &arr, int k = 0) -> std::pair<ndarray<int>, ndarray<int>>
{
    if (arr.ndim() != 2)
        throw std::invalid_argument("triu_indices_from: need 2-D");
    return triu_indices(arr.shape[0], k, arr.shape[1]);
}

// Normal comment: mgrid / ogrid dense and sparse meshgrids

NP_API inline auto mgrid(const std::vector<std::pair<int, int>> &ranges) -> std::vector<ndarray<int>>
{
    std::vector<ndarray<int>> axes;
    axes.reserve(ranges.size());
    for (auto &r : ranges)
        axes.push_back(arange(r.first, r.second));
    return meshgrid(axes, "ij");
}

NP_API inline auto ogrid(const std::vector<std::pair<int, int>> &ranges) -> std::vector<ndarray<int>>
{
    std::vector<ndarray<int>> out;
    out.reserve(ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i)
    {
        auto arr = arange(ranges[i].first, ranges[i].second);
        std::vector<int> shape(ranges.size(), 1);
        shape[i] = static_cast<int>(arr.size());
        out.push_back(arr.reshape(shape));
    }
    return out;
}

namespace rec
{
/**
 * @brief Record array helpers (np.rec.*).
 *
 * Reference: numpy-reference/reference/generated/numpy.rec.array.html
 *
 * Minimal stubs – structured arrays are modeled as vector<map<string, double>>
 * in this header-only port. These helpers provide API parity.
 */
NP_API inline auto array(const std::vector<std::map<std::string, double>> &records)
    -> std::vector<std::map<std::string, double>>
{
    return records;
}

NP_API inline auto fromarrays(const std::vector<ndarray<double>> &arrays, const std::vector<std::string> &names = {})
    -> std::vector<std::map<std::string, double>>
{
    if (arrays.empty())
        return {};
    std::size_t n = arrays[0].size();
    std::vector<std::map<std::string, double>> out(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < arrays.size(); ++j)
        {
            std::string key = j < names.size() ? names[j] : "f" + std::to_string(j);
            out[i][key] = arrays[j].data()[arrays[j]._flat_logical(i)];
        }
    return out;
}

NP_API inline auto fromrecords(const std::vector<std::vector<double>> &records,
                               const std::vector<std::string> &names = {}) -> std::vector<std::map<std::string, double>>
{
    std::vector<std::map<std::string, double>> out;
    out.reserve(records.size());
    for (auto &rec : records)
    {
        std::map<std::string, double> m;
        for (std::size_t i = 0; i < rec.size(); ++i)
        {
            std::string key = i < names.size() ? names[i] : "f" + std::to_string(i);
            m[key] = rec[i];
        }
        out.push_back(std::move(m));
    }
    return out;
}

NP_API inline auto fromstring(const std::string &s, const std::string &dtype = "float64")
    -> std::vector<std::map<std::string, double>>
{
    (void)dtype;
    auto vals = ::np::fromstring<double>(s);
    std::vector<std::map<std::string, double>> out;
    out.reserve(vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
    {
        std::map<std::string, double> m;
        m["f0"] = vals.data()[i];
        out.push_back(std::move(m));
    }
    return out;
}

NP_API inline auto fromfile(const std::string &filename) -> std::vector<std::map<std::string, double>>
{
    std::ifstream is(filename);
    std::vector<std::map<std::string, double>> out;
    if (!is)
        return out;
    double v;
    while (is >> v)
    {
        std::map<std::string, double> m;
        m["f0"] = v;
        out.push_back(std::move(m));
    }
    return out;
}
} // namespace rec

/**
 * @brief Cast array to new dtype (np.astype free-function wrapper).
 *
 * NumPy exposes `ndarray.astype(dtype)` as a method; this free
 * function mirrors the functional form `np.astype(a, dtype)` for
 * API parity and to support generic dispatch where the target
 * type is a template parameter rather than a runtime `dtype`
 * enum.
 *
 * Reference: numpy-reference/reference/generated/numpy.ndarray.astype.html
 */
NP_API template <typename U, typename T> NP_NODISCARD auto astype(const ndarray<T> &a) -> ndarray<U>
{
    return a.template astype<U>();
}

NP_API template <typename T>
NP_NODISCARD auto astype(const ndarray<T> &a, dtype dt)
    -> std::variant<ndarray<std::int8_t>, ndarray<std::int16_t>, ndarray<std::int32_t>, ndarray<std::int64_t>,
                    ndarray<std::uint8_t>, ndarray<std::uint16_t>, ndarray<std::uint32_t>, ndarray<std::uint64_t>,
                    ndarray<float>, ndarray<double>, ndarray<long double>, ndarray<std::complex<float>>,
                    ndarray<std::complex<double>>, ndarray<std::complex<long double>>, ndarray<bool>>
{
    // Runtime-dtype form – materialises the correct C++ typed array
    // via the existing `ndarray_from_dtype` + elementwise cast.
    auto fallback = ndarray_from_dtype(dt, a.shape);
    return std::visit(
        [&](auto &out)
            -> std::variant<ndarray<std::int8_t>, ndarray<std::int16_t>, ndarray<std::int32_t>, ndarray<std::int64_t>,
                            ndarray<std::uint8_t>, ndarray<std::uint16_t>, ndarray<std::uint32_t>,
                            ndarray<std::uint64_t>, ndarray<float>, ndarray<double>, ndarray<long double>,
                            ndarray<std::complex<float>>, ndarray<std::complex<double>>,
                            ndarray<std::complex<long double>>, ndarray<bool>> {
            using U = typename std::decay_t<decltype(out)>::value_type;
            out = a.template astype<U>();
            return out;
        },
        fallback);
}

/**
 * @brief Return an array with the same shape as `a` but filled with
 *        `fill_value` cast to `U` (np.full_like with dtype override).
 *
 * Reference: numpy-reference/reference/generated/numpy.full_like.html
 */
NP_API template <typename U, typename T> NP_NODISCARD auto full_like(const ndarray<T> &a, U fill_value) -> ndarray<U>
{
    return ndarray<U>(a.shape, dtype_of<U>, fill_value);
}

/**
 * @brief Zeros with dtype override (np.zeros_like with dtype).
 */
NP_API template <typename U, typename T> NP_NODISCARD auto zeros_like(const ndarray<T> &a, dtype) -> ndarray<U>
{
    return ndarray<U>(a.shape, dtype_of<U>, U{0});
}

/**
 * @brief Ones with dtype override (np.ones_like with dtype).
 */
NP_API template <typename U, typename T> NP_NODISCARD auto ones_like(const ndarray<T> &a, dtype) -> ndarray<U>
{
    return ndarray<U>(a.shape, dtype_of<U>, U{1});
}

/**
 * @brief Empty with dtype override (np.empty_like with dtype).
 */
NP_API template <typename U, typename T> NP_NODISCARD auto empty_like(const ndarray<T> &a, dtype) -> ndarray<U>
{
    return ndarray<U>(a.shape, dtype_of<U>, U{});
}

} // namespace np

#endif // NP_CREATION_HPP
