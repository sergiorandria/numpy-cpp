/**
 * @file math_constexpr.hpp
 * @brief Constexpr replacements for the C math library.
 *
 * std::sqrt/std::exp/... are not constexpr until C++26 (P0533), so the
 * fixed-shape path of `np` provides its own constexpr kernels. The error
 * budget (~1e-10 relative) is deliberately looser than libm: these functions
 * exist to fold compile-time expressions (see tests/test_constexpr.cpp), not
 * to replace the runtime fast paths (ndarray elementwise ops still use
 * std::cmath).
 */
#ifndef NP_DETAIL_MATH_CONSTEXPR_HPP
#define NP_DETAIL_MATH_CONSTEXPR_HPP

#include "../api_macros.hpp"

#include <cstddef>
#include <limits>

namespace np::detail::math
{

constexpr double pi_v = 3.141592653589793238462643383279502884;

/** @brief constexpr absolute value. */
template <typename V> constexpr V abs(V v)
{
    return v < V{0} ? -v : v;
}

/** @brief constexpr floor. */
constexpr double floor(double x)
{
    const double t = static_cast<double>(static_cast<long long>(x));
    return (x < 0.0 && t != x) ? t - 1.0 : t;
}

/** @brief constexpr ceil. */
constexpr double ceil(double x)
{
    const double t = static_cast<double>(static_cast<long long>(x));
    return (x > 0.0 && t != x) ? t + 1.0 : t;
}

/**
 * @brief constexpr round, half-to-even (banker's rounding, exactly like
 *        numpy.round: numpy-reference/reference/generated/numpy.round.html).
 */
constexpr double round(double x)
{
    if (x >= 0.0)
    {
        const double f = floor(x);
        const double frac = x - f;
        if (frac < 0.5)
        {
            return f;
        }
        if (frac > 0.5)
        {
            return f + 1.0;
        }
        const double h = f / 2.0;
        return (h == floor(h)) ? f : f + 1.0;
    }
    const double c = ceil(x);
    const double frac = x - c;
    if (frac > -0.5)
    {
        return c;
    }
    if (frac < -0.5)
    {
        return c - 1.0;
    }
    const double h = c / 2.0;
    return (h == floor(h)) ? c : c - 1.0;
}

/** @brief constexpr fractional part of x / y for y > 0. */
constexpr double fmod(double x, double y)
{
    double r = x;
    while (r >= y)
    {
        r -= y;
    }
    while (r < 0.0)
    {
        r += y;
    }
    return r;
}

/** @brief constexpr sqrt via Newton's method. */
constexpr double sqrt(double x)
{
    if (x < 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0)
    {
        return 0.0;
    }
    // Scale x into [0.25, 1) so that the initial guess (x itself) is
    // within a factor of two of sqrt(x); Newton's method then converges
    // in a few iterations regardless of the magnitude of x.
    double v = x;
    double scale = 1.0;
    while (v >= 1.0)
    {
        v *= 0.25;
        scale *= 2.0;
    }
    while (v < 0.25)
    {
        v *= 4.0;
        scale *= 0.5;
    }
    double guess = v;
    for (std::size_t i = 0; i < 12; ++i)
    {
        guess = 0.5 * (guess + v / guess);
    }
    return guess * scale;
}

/**
 * @brief constexpr exp via Taylor series with halving reduction.
 */
constexpr double exp(double x)
{
    if (x == 0.0)
    {
        return 1.0;
    }
    double reduced = x;
    int halvings = 0;
    while (reduced > 0.35)
    {
        reduced *= 0.5;
        ++halvings;
    }
    while (reduced < -0.35)
    {
        reduced *= 0.5;
        ++halvings;
    }
    double term = 1.0;
    double acc = 1.0;
    for (std::size_t k = 1; k <= 14; ++k)
    {
        term *= reduced / static_cast<double>(k);
        acc += term;
    }
    for (int h = 0; h < halvings; ++h)
    {
        acc *= acc;
    }
    return acc;
}

/**
 * @brief constexpr log via atanh series: log(x) = 2*atanh((x-1)/(x+1)).
 */
constexpr double log(double x)
{
    if (x <= 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 1.0)
    {
        return 0.0;
    }
    double y = (x - 1.0) / (x + 1.0);
    double y2 = y * y;
    double term = y;
    double acc = 0.0;
    for (std::size_t k = 1; k <= 20; ++k)
    {
        acc += term / static_cast<double>(2 * k - 1);
        term *= y2;
    }
    return 2.0 * acc;
}

/** @brief constexpr sin via Taylor series (argument reduced to [0, 2pi)). */
constexpr double sin(double x)
{
    const double r = fmod(x, 2.0 * pi_v);
    double term = r;
    double r2 = r * r;
    double acc = 0.0;
    for (std::size_t k = 0; k < 12; ++k)
    {
        acc += term;
        term = -term * r2 / static_cast<double>((2 * k + 2) * (2 * k + 3));
    }
    return acc;
}

/** @brief constexpr cos via Taylor series (argument reduced to [0, 2pi)). */
constexpr double cos(double x)
{
    const double r = fmod(x, 2.0 * pi_v);
    double term = 1.0;
    double r2 = r * r;
    double acc = 0.0;
    for (std::size_t k = 0; k < 12; ++k)
    {
        acc += term;
        term = -term * r2 / static_cast<double>((2 * k + 1) * (2 * k + 2));
    }
    return acc;
}

/** @brief constexpr tan. */
constexpr double tan(double x)
{
    return sin(x) / cos(x);
}

/** @brief Exact power for integer-valued exponents via squaring. */
constexpr double pow_sq(double base, long long e)
{
    double acc = 1.0;
    double b = e < 0 ? 1.0 / base : base;
    long long n = e < 0 ? -e : e;
    while (n > 0)
    {
        if (n % 2 == 1)
        {
            acc *= b;
        }
        b *= b;
        n /= 2;
    }
    return acc;
}

/**
 * @brief constexpr power: exact for integer-valued exponents (as numpy
 *        yields), otherwise x^y via exp(y*log(x)).
 */
constexpr double pow(double x, double y)
{
    if (x == 0.0)
    {
        return y == 0.0 ? 1.0 : 0.0;
    }
    const double f = floor(y);
    if (y == f && abs(y) <= 9.0e15)
    {
        return pow_sq(x, static_cast<long long>(y));
    }
    return exp(y * log(x));
}

/** @brief constexpr square. */
constexpr double square(double x)
{
    return x * x;
}

/** @brief NaN sentinel used by math kernels for invalid inputs. */
constexpr double nan()
{
    return std::numeric_limits<double>::quiet_NaN();
}

} // namespace np::detail::math

#endif // NP_DETAIL_MATH_CONSTEXPR_HPP
