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
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_EMATH_HPP
#define NP_EMATH_HPP

#include <cmath>
#include <complex>
#include <type_traits>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{
namespace emath
{

namespace detail
{
template <typename T> using cplx = std::complex<double>;

template <typename T> inline auto to_cplx(T v) -> cplx<T>
{
    return cplx<T>(static_cast<double>(v), 0.0);
}
} // namespace detail

/**
 * @brief Square root with complex promotion (np.emath.sqrt).
 * Reference: numpy-reference/reference/generated/numpy.emath.sqrt.html
 */
NP_API template <typename T> NP_NODISCARD auto sqrt(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    ndarray<std::complex<double>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double v = static_cast<double>(x.data()[x._flat_logical(i)]);
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>> ||
                      std::is_same_v<T, std::complex<long double>>)
        {
            auto c = x.data()[x._flat_logical(i)];
            out.data()[i] = std::sqrt(c);
        }
        else
        {
            if (v >= 0)
            {
                out.data()[i] = std::complex<double>(std::sqrt(v), 0.0);
            }
            else
            {
                out.data()[i] = std::sqrt(std::complex<double>(v, 0.0));
            }
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto sqrt(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        out.data()[i] = std::sqrt(x.data()[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Natural log with complex promotion (np.emath.log).
 * Reference: numpy-reference/reference/generated/numpy.emath.log.html
 */
NP_API template <typename T> NP_NODISCARD auto log(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    ndarray<std::complex<double>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double v = static_cast<double>(x.data()[x._flat_logical(i)]);
        if (v > 0 || std::isnan(v))
        {
            out.data()[i] = std::complex<double>(std::log(v), 0.0);
        }
        else
        {
            out.data()[i] = std::log(std::complex<double>(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto log(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        out.data()[i] = std::log(x.data()[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Log base 2 with complex promotion (np.emath.log2).
 * Reference: numpy-reference/reference/generated/numpy.emath.log2.html
 */
NP_API template <typename T> NP_NODISCARD auto log2(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    auto lg = log(x);
    const double ln2 = std::log(2.0);
    for (std::size_t i = 0; i < lg.size(); ++i)
    {
        lg.data()[i] /= ln2;
    }
    return lg;
}

NP_API template <typename T> NP_NODISCARD auto log2(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    auto lg = log(x);
    const double ln2 = std::log(2.0);
    for (std::size_t i = 0; i < lg.size(); ++i)
    {
        lg.data()[i] /= ln2;
    }
    return lg;
}

/**
 * @brief Log base 10 with complex promotion (np.emath.log10).
 * Reference: numpy-reference/reference/generated/numpy.emath.log10.html
 */
NP_API template <typename T> NP_NODISCARD auto log10(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    auto lg = log(x);
    const double ln10 = std::log(10.0);
    for (std::size_t i = 0; i < lg.size(); ++i)
    {
        lg.data()[i] /= ln10;
    }
    return lg;
}

NP_API template <typename T> NP_NODISCARD auto log10(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    auto lg = log(x);
    const double ln10 = std::log(10.0);
    for (std::size_t i = 0; i < lg.size(); ++i)
    {
        lg.data()[i] /= ln10;
    }
    return lg;
}

/**
 * @brief Log base n with complex promotion (np.emath.logn).
 * Reference: numpy-reference/reference/generated/numpy.emath.logn.html
 */
NP_API template <typename T> NP_NODISCARD auto logn(double n, const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    if (n <= 0 || n == 1.0)
    {
        throw std::invalid_argument("emath::logn: base must be >0 and !=1");
    }
    auto lg = log(x);
    double lnn = std::log(n);
    for (std::size_t i = 0; i < lg.size(); ++i)
    {
        lg.data()[i] /= lnn;
    }
    return lg;
}

NP_API template <typename T>
NP_NODISCARD auto logn(double n, const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    auto lg = log(x);
    double lnn = std::log(n);
    for (std::size_t i = 0; i < lg.size(); ++i)
    {
        lg.data()[i] /= lnn;
    }
    return lg;
}

/**
 * @brief Power with complex promotion (np.emath.power).
 * Reference: numpy-reference/reference/generated/numpy.emath.power.html
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto power(const ndarray<T> &x, const ndarray<U> &p) -> ndarray<std::complex<double>>
{
    std::vector<int> out_shape = np::detail::broadcast_shapes(x.shape, p.shape);
    ndarray<std::complex<double>> out(out_shape);
    np::detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        double xv = static_cast<double>(x.get(np::detail::broadcast_index(x.shape, out_shape, idx)));
        double pv = static_cast<double>(p.get(np::detail::broadcast_index(p.shape, out_shape, idx)));
        std::complex<double> c = std::pow(std::complex<double>(xv, 0.0), std::complex<double>(pv, 0.0));
        out.set(idx, c);
        od.advance();
    }
    return out;
}

NP_API template <typename T, typename U>
NP_NODISCARD auto power(const ndarray<T> &x, U p) -> ndarray<std::complex<double>>
{
    ndarray<std::complex<double>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double xv = static_cast<double>(x.data()[x._flat_logical(i)]);
        double pv = static_cast<double>(p);
        out.data()[i] = std::pow(std::complex<double>(xv, 0.0), std::complex<double>(pv, 0.0));
    }
    return out;
}

/**
 * @brief Arccos with complex promotion (np.emath.arccos).
 * Reference: numpy-reference/reference/generated/numpy.emath.arccos.html
 */
NP_API template <typename T> NP_NODISCARD auto arccos(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    ndarray<std::complex<double>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double v = static_cast<double>(x.data()[x._flat_logical(i)]);
        if (v >= -1.0 && v <= 1.0)
        {
            out.data()[i] = std::complex<double>(std::acos(v), 0.0);
        }
        else
        {
            out.data()[i] = std::acos(std::complex<double>(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto arccos(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        out.data()[i] = std::acos(x.data()[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Arcsin with complex promotion (np.emath.arcsin).
 * Reference: numpy-reference/reference/generated/numpy.emath.arcsin.html
 */
NP_API template <typename T> NP_NODISCARD auto arcsin(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    ndarray<std::complex<double>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double v = static_cast<double>(x.data()[x._flat_logical(i)]);
        if (v >= -1.0 && v <= 1.0)
        {
            out.data()[i] = std::complex<double>(std::asin(v), 0.0);
        }
        else
        {
            out.data()[i] = std::asin(std::complex<double>(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto arcsin(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        out.data()[i] = std::asin(x.data()[x._flat_logical(i)]);
    }
    return out;
}

/**
 * @brief Arctanh with complex promotion (np.emath.arctanh).
 * Reference: numpy-reference/reference/generated/numpy.emath.arctanh.html
 */
NP_API template <typename T> NP_NODISCARD auto arctanh(const ndarray<T> &x) -> ndarray<std::complex<double>>
{
    ndarray<std::complex<double>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double v = static_cast<double>(x.data()[x._flat_logical(i)]);
        if (std::abs(v) < 1.0)
        {
            out.data()[i] = std::complex<double>(std::atanh(v), 0.0);
        }
        else
        {
            out.data()[i] = std::atanh(std::complex<double>(v, 0.0));
        }
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD auto arctanh(const ndarray<std::complex<T>> &x) -> ndarray<std::complex<T>>
{
    ndarray<std::complex<T>> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        out.data()[i] = std::atanh(x.data()[x._flat_logical(i)]);
    }
    return out;
}

} // namespace emath
} // namespace np

#endif // NP_EMATH_HPP
