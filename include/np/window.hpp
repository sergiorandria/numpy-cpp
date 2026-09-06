/**
 * @file window.hpp
 * @brief Window functions (np.bartlett/blackman/hamming/hanning/kaiser).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.window.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_WINDOW_HPP
#define NP_WINDOW_HPP

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{

/**
 * @brief Bartlett (triangular) window (np.bartlett).
 *
 * Reference: numpy-reference/reference/generated/numpy.bartlett.html
 */
NP_API inline auto bartlett(int M) -> ndarray<double>
{
    if (M < 0)
    {
        throw std::invalid_argument("bartlett: M must be non-negative");
    }
    if (M == 0)
    {
        return ndarray<double>(std::vector<int>{0});
    }
    if (M == 1)
    {
        ndarray<double> w(std::vector<int>{1});
        w(0) = 1.0;
        return w;
    }
    ndarray<double> w(std::vector<int>{M});
    for (int n = 0; n < M; ++n)
    {
        w(n) = 2.0 / (M - 1) * ((M - 1) / 2.0 - std::abs(n - (M - 1) / 2.0));
    }
    return w;
}

/**
 * @brief Blackman window (np.blackman).
 *
 * Reference: numpy-reference/reference/generated/numpy.blackman.html
 */
NP_API inline auto blackman(int M) -> ndarray<double>
{
    if (M < 0)
    {
        throw std::invalid_argument("blackman: M must be non-negative");
    }
    if (M == 0)
    {
        return ndarray<double>(std::vector<int>{0});
    }
    if (M == 1)
    {
        ndarray<double> w(std::vector<int>{1});
        w(0) = 1.0;
        return w;
    }
    ndarray<double> w(std::vector<int>{M});
    const double pi = std::numbers::pi_v<double>;
    for (int n = 0; n < M; ++n)
    {
        double a = 2.0 * pi * n / (M - 1);
        w(n) = 0.42 - 0.5 * std::cos(a) + 0.08 * std::cos(2 * a);
    }
    return w;
}

/**
 * @brief Hamming window (np.hamming).
 *
 * Reference: numpy-reference/reference/generated/numpy.hamming.html
 */
NP_API inline auto hamming(int M) -> ndarray<double>
{
    if (M < 0)
    {
        throw std::invalid_argument("hamming: M must be non-negative");
    }
    if (M == 0)
    {
        return ndarray<double>(std::vector<int>{0});
    }
    if (M == 1)
    {
        ndarray<double> w(std::vector<int>{1});
        w(0) = 1.0;
        return w;
    }
    ndarray<double> w(std::vector<int>{M});
    const double pi = std::numbers::pi_v<double>;
    for (int n = 0; n < M; ++n)
    {
        w(n) = 0.54 - 0.46 * std::cos(2.0 * pi * n / (M - 1));
    }
    return w;
}

/**
 * @brief Hanning (Hann) window (np.hanning).
 *
 * Reference: numpy-reference/reference/generated/numpy.hanning.html
 */
NP_API inline auto hanning(int M) -> ndarray<double>
{
    if (M < 0)
    {
        throw std::invalid_argument("hanning: M must be non-negative");
    }
    if (M == 0)
    {
        return ndarray<double>(std::vector<int>{0});
    }
    if (M == 1)
    {
        ndarray<double> w(std::vector<int>{1});
        w(0) = 1.0;
        return w;
    }
    ndarray<double> w(std::vector<int>{M});
    const double pi = std::numbers::pi_v<double>;
    for (int n = 0; n < M; ++n)
    {
        w(n) = 0.5 - 0.5 * std::cos(2.0 * pi * n / (M - 1));
    }
    return w;
}

/**
 * @brief Alias for hanning (NumPy also exposes `hann`).
 */
NP_API inline auto hann(int M) -> ndarray<double>
{
    return hanning(M);
}

/**
 * @brief Kaiser window (np.kaiser).
 *
 * Reference: numpy-reference/reference/generated/numpy.kaiser.html
 *
 * Uses std::cyl_bessel_i (C++17) for I0.
 */
NP_API inline auto kaiser(int M, double beta) -> ndarray<double>
{
    if (M < 0)
    {
        throw std::invalid_argument("kaiser: M must be non-negative");
    }
    if (M == 0)
    {
        return ndarray<double>(std::vector<int>{0});
    }
    if (M == 1)
    {
        ndarray<double> w(std::vector<int>{1});
        w(0) = 1.0;
        return w;
    }
    ndarray<double> w(std::vector<int>{M});
    double denom = std::cyl_bessel_i(0, beta);
    if (denom == 0.0)
    {
        throw std::runtime_error("kaiser: I0(beta) is zero");
    }
    for (int n = 0; n < M; ++n)
    {
        double r = (2.0 * n / (M - 1) - 1.0);
        double arg = beta * std::sqrt(1.0 - r * r);
        w(n) = std::cyl_bessel_i(0, arg) / denom;
    }
    return w;
}

} // namespace np

#endif // NP_WINDOW_HPP
