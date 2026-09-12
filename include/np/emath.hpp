/**
 * @file emath.hpp
 * @brief Mathematical functions with automatic domain (np.emath / lib.scimath).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.emath.html
 *
 * Wrappers that return complex results when the real domain is exceeded,
 * e.g. sqrt(-1) -> 1j, log(-1) -> pi*j, arccos(2) -> -j*log(...).
 * Each function has overloads for real ndarrays (returning complex) and
 * for already-complex inputs (delegating to std::complex math).
 *
 * Performance notes: every loop has a contiguous fast path (raw pointers,
 * single is_contiguous() check hoisted out of the loop) plus a strided
 * fallback via _flat_logical. Outputs are freshly allocated (hence always
 * contiguous) and written linearly in both paths.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_EMATH_HPP
#define NP_EMATH_HPP

#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{
namespace emath
{

namespace detail
{
// 1/ln(2) for the complex log2 fallback (std::log2 has no complex overload).
inline constexpr double kInvLn2 = 1.4426950408889634;

// Shared element kernel for power: real fast path for non-negative bases
// (NaN and negative bases still go through complex pow, as before).
inline auto pow_elem(double xv, double pv) -> std::complex<double>
{
    if (xv >= 0.0) [[likely]]
    {
        return std::complex<double>(std::pow(xv, pv), 0.0);
    }
    return std::pow(std::complex<double>(xv, 0.0), std::complex<double>(pv, 0.0));
}
} // namespace detail

/**
 * @brief Square root with complex promotion (np.emath.sqrt).
 * Reference: numpy-reference/reference/generated/numpy.emath.sqrt.html
 */
NP_API template <typename T>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD auto sqrt(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    ndarray<Out> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            const double v = static_cast<double>(src[i]);
            if (v >= 0.0) [[likely]]
            {
                dst[i] = Out(std::sqrt(v), 0.0);
            }
            else if (v < 0.0)
            {
                // Analytic sqrt of a negative real: 0 + i*sqrt(-v).
                dst[i] = Out(0.0, std::sqrt(-v));
            }
            else
            {
                dst[i] = std::sqrt(Out(v, 0.0)); // NaN: preserve complex-sqrt behavior
            }
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(src[x._flat_logical(i)]);
        if (v >= 0.0) [[likely]]
        {
            dst[i] = Out(std::sqrt(v), 0.0);
        }
        else if (v < 0.0)
        {
            dst[i] = Out(0.0, std::sqrt(-v));
        }
        else
        {
            dst[i] = std::sqrt(Out(v, 0.0)); // NaN
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto sqrt(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = std::sqrt(src[i]);
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        dst[i] = std::sqrt(src[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Natural log with complex promotion (np.emath.log).
 * Reference: numpy-reference/reference/generated/numpy.emath.log.html
 */
NP_API template <typename T>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD auto log(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    ndarray<Out> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            const double v = static_cast<double>(src[i]);
            if (v > 0.0 || std::isnan(v)) [[likely]]
            {
                dst[i] = Out(std::log(v), 0.0);
            }
            else
            {
                dst[i] = std::log(Out(v, 0.0));
            }
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(src[x._flat_logical(i)]);
        if (v > 0.0 || std::isnan(v)) [[likely]]
        {
            dst[i] = Out(std::log(v), 0.0);
        }
        else
        {
            dst[i] = std::log(Out(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto log(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = std::log(src[i]);
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        dst[i] = std::log(src[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Log base 2 with complex promotion (np.emath.log2).
 * Reference: numpy-reference/reference/generated/numpy.emath.log2.html
 *
 * Single pass: uses std::log2 directly instead of log-then-divide.
 */
NP_API template <typename T>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD auto log2(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    ndarray<Out> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            const double v = static_cast<double>(src[i]);
            if (v > 0.0 || std::isnan(v)) [[likely]]
            {
                dst[i] = Out(std::log2(v), 0.0);
            }
            else
            {
                dst[i] = std::log(Out(v, 0.0)) * detail::kInvLn2;
            }
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(src[x._flat_logical(i)]);
        if (v > 0.0 || std::isnan(v)) [[likely]]
        {
            dst[i] = Out(std::log2(v), 0.0);
        }
        else
        {
            dst[i] = std::log(Out(v, 0.0)) * detail::kInvLn2;
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto log2(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = std::log(src[i]) * detail::kInvLn2;
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        dst[i] = std::log(src[x._flat_logical(i)]) * detail::kInvLn2;
    }
    return out;
}

/**
 * @brief Log base 10 with complex promotion (np.emath.log10).
 * Reference: numpy-reference/reference/generated/numpy.emath.log10.html
 *
 * Single pass: uses std::log10 directly instead of log-then-divide.
 */
NP_API template <typename T>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD auto log10(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    ndarray<Out> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            const double v = static_cast<double>(src[i]);
            if (v > 0.0 || std::isnan(v)) [[likely]]
            {
                dst[i] = Out(std::log10(v), 0.0);
            }
            else
            {
                dst[i] = std::log10(Out(v, 0.0));
            }
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(src[x._flat_logical(i)]);
        if (v > 0.0 || std::isnan(v)) [[likely]]
        {
            dst[i] = Out(std::log10(v), 0.0);
        }
        else
        {
            dst[i] = std::log10(Out(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto log10(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = std::log10(src[i]);
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        dst[i] = std::log10(src[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Log base n with complex promotion (np.emath.logn).
 * Reference: numpy-reference/reference/generated/numpy.emath.logn.html
 *
 * Single pass with a precomputed reciprocal (multiply, not divide).
 */
NP_API template <typename T>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD auto logn(double n, const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    if (n <= 0.0 || n == 1.0)
    {
        throw std::invalid_argument("emath::logn: base must be >0 and !=1");
    }
    using Out = std::complex<double>;
    const double inv_ln = 1.0 / std::log(n);
    ndarray<Out> out(x.shape);
    const std::size_t sz = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < sz; ++i)
        {
            const double v = static_cast<double>(src[i]);
            if (v > 0.0 || std::isnan(v)) [[likely]]
            {
                dst[i] = Out(std::log(v) * inv_ln, 0.0);
            }
            else
            {
                dst[i] = std::log(Out(v, 0.0)) * inv_ln;
            }
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < sz; ++i)
    {
        const double v = static_cast<double>(src[x._flat_logical(i)]);
        if (v > 0.0 || std::isnan(v)) [[likely]]
        {
            dst[i] = Out(std::log(v) * inv_ln, 0.0);
        }
        else
        {
            dst[i] = std::log(Out(v, 0.0)) * inv_ln;
        }
    }
    return out;
}

NP_API template <typename T>
NP_NODISCARD auto logn(double n, const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    if (n <= 0.0 || n == 1.0)
    {
        throw std::invalid_argument("emath::logn: base must be >0 and !=1");
    }
    const double inv_ln = 1.0 / std::log(n);
    ndarray<std::complex<T>> out(x.shape);
    const std::size_t sz = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < sz; ++i)
        {
            dst[i] = std::log(src[i]) * inv_ln;
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < sz; ++i)
    {
        dst[i] = std::log(src[x._flat_logical(i)]) * inv_ln;
    }
    return out;
}

/**
 * @brief Power with complex promotion (np.emath.power).
 * Reference: numpy-reference/reference/generated/numpy.emath.power.html
 */
NP_API template <typename T, typename U>
    requires(!np::detail::is_complex_v<T> && !np::detail::is_complex_v<U>)
NP_NODISCARD auto power(const ndarray<T> &x, const ndarray<U> &p) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    if (x.shape == p.shape && x.is_contiguous() && p.is_contiguous())
    {
        ndarray<Out> out(x.shape);
        const std::size_t n = x.size();
        auto *dst = out.data().data();
        const auto *xs = x.data().data();
        const auto *ps = p.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = detail::pow_elem(static_cast<double>(xs[i]), static_cast<double>(ps[i]));
        }
        return out;
    }
    std::vector<int> out_shape = np::detail::broadcast_shapes(x.shape, p.shape);
    ndarray<Out> out(out_shape);
    np::detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        const double xv = static_cast<double>(x.get(np::detail::broadcast_index(x.shape, out_shape, idx)));
        const double pv = static_cast<double>(p.get(np::detail::broadcast_index(p.shape, out_shape, idx)));
        out.set(idx, detail::pow_elem(xv, pv));
        od.advance();
    }
    return out;
}

NP_API template <typename T, typename U>
    requires(!np::detail::is_complex_v<T> && std::is_arithmetic_v<U>)
NP_NODISCARD auto power(const ndarray<T> &x, U p) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    const double pv = static_cast<double>(p); // loop-invariant: hoist the conversion
    ndarray<Out> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = detail::pow_elem(static_cast<double>(src[i]), pv);
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        dst[i] = detail::pow_elem(static_cast<double>(src[x._flat_logical(i)]), pv);
    }
    return out;
}

/**
 * @brief Arccos with complex promotion (np.emath.arccos).
 * Reference: numpy-reference/reference/generated/numpy.emath.arccos.html
 */
NP_API template <typename T>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD auto arccos(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    ndarray<Out> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            const double v = static_cast<double>(src[i]);
            if (v >= -1.0 && v <= 1.0) [[likely]]
            {
                dst[i] = Out(std::acos(v), 0.0);
            }
            else
            {
                dst[i] = std::acos(Out(v, 0.0));
            }
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(src[x._flat_logical(i)]);
        if (v >= -1.0 && v <= 1.0) [[likely]]
        {
            dst[i] = Out(std::acos(v), 0.0);
        }
        else
        {
            dst[i] = std::acos(Out(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto arccos(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = std::acos(src[i]);
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        dst[i] = std::acos(src[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Arcsin with complex promotion (np.emath.arcsin).
 * Reference: numpy-reference/reference/generated/numpy.emath.arcsin.html
 */
NP_API template <typename T>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD auto arcsin(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    ndarray<Out> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            const double v = static_cast<double>(src[i]);
            if (v >= -1.0 && v <= 1.0) [[likely]]
            {
                dst[i] = Out(std::asin(v), 0.0);
            }
            else
            {
                dst[i] = std::asin(Out(v, 0.0));
            }
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(src[x._flat_logical(i)]);
        if (v >= -1.0 && v <= 1.0) [[likely]]
        {
            dst[i] = Out(std::asin(v), 0.0);
        }
        else
        {
            dst[i] = std::asin(Out(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto arcsin(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = std::asin(src[i]);
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        dst[i] = std::asin(src[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Arctanh with complex promotion (np.emath.arctanh).
 * Reference: numpy-reference/reference/generated/numpy.emath.arctanh.html
 */
NP_API template <typename T>
    requires(!np::detail::is_complex_v<T>)
NP_NODISCARD auto arctanh(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    using Out = std::complex<double>;
    ndarray<Out> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            const double v = static_cast<double>(src[i]);
            if (std::abs(v) < 1.0) [[likely]]
            {
                dst[i] = Out(std::atanh(v), 0.0);
            }
            else
            {
                dst[i] = std::atanh(Out(v, 0.0));
            }
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(src[x._flat_logical(i)]);
        if (std::abs(v) < 1.0) [[likely]]
        {
            dst[i] = Out(std::atanh(v), 0.0);
        }
        else
        {
            dst[i] = std::atanh(Out(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto arctanh(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    const std::size_t n = x.size();
    auto *dst = out.data().data();
    if (x.is_contiguous())
    {
        const auto *src = x.data().data();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = std::atanh(src[i]);
        }
        return out;
    }
    const auto &src = x.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        dst[i] = std::atanh(src[x._flat_logical(i)]);
    }
    return out;
}

} // namespace emath
} // namespace np

#endif // NP_EMATH_HPP
