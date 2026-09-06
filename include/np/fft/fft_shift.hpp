/**
 * @file fft/fft_shift.hpp
 * @brief Frequency helpers and half-spectrum shifting utilities.
 *
 *  - fftfreq / rfftfreq: bin center frequencies for a given window length.
 *  - fftshift / ifftshift: move the zero-frequency component to the center
 *    of the spectrum (and back).
 *
 * The frequency formulas match NumPy exactly:
 *   f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)     if n is even
 *   f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n) if n is odd
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_FFT_SHIFT_HPP
#define NP_FFT_SHIFT_HPP

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../api_macros.hpp"
#include "../creation.hpp"
#include "../manipulation.hpp"
#include "../ndarray.hpp"

namespace np::fft
{

/**
 * @brief Return the Discrete Fourier Transform sample frequencies.
 *
 * `f[i]` is the frequency bin center in cycles per unit of the sample
 * spacing, with zero at the start.
 *
 * @param n  Window length (positive).
 * @param d  Sample spacing (inverse of the sampling rate); default 1.0.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
 */
NP_API NP_NODISCARD inline auto fftfreq(int n, double d = 1.0) -> ndarray<double>
{
    if (n <= 0)
    {
        throw std::invalid_argument("fftfreq: n should be positive");
    }
    ndarray<double> out(std::vector<int>{n});
    const int first = (n - 1) / 2 + 1;
    const double val = 1.0 / (static_cast<double>(n) * d);
    for (int i = 0; i < n; ++i)
    {
        out.data()[static_cast<std::size_t>(i)] = static_cast<double>(i < first ? i : i - n) * val;
    }
    return out;
}

/**
 * @brief Return the sample frequencies for use with rfft / irfft.
 *
 * Unlike fftfreq, the Nyquist frequency is considered positive, so the
 * returned array has length n/2+1.
 *
 * @param n  Window length (positive).
 * @param d  Sample spacing (default 1.0).
 *
 * Reference: numpy reference/generated/numpy.fft.rfftfreq.html
 */
NP_API NP_NODISCARD inline auto rfftfreq(int n, double d = 1.0) -> ndarray<double>
{
    if (n <= 0)
    {
        throw std::invalid_argument("rfftfreq: n should be positive");
    }
    ndarray<double> out(std::vector<int>{n / 2 + 1});
    const double val = 1.0 / (static_cast<double>(n) * d);
    for (int i = 0; i < n / 2 + 1; ++i)
    {
        out.data()[static_cast<std::size_t>(i)] = static_cast<double>(i) * val;
    }
    return out;
}

namespace detail
{

/** @brief Roll every listed axis of `arr` by the given raw shift. */
template <typename T>
NP_NODISCARD ndarray<T> shift_roll(const ndarray<T> &arr, const std::optional<std::vector<int>> &axes, int sign)
{
    ndarray<T> out = arr;
    if (axes)
    {
        for (int a : *axes)
        {
            int ax = normalize_axis(a, out.ndim());
            out = np::roll(out, sign * (out.shape[ax] / 2), ax);
        }
    }
    else
    {
        for (int a = 0; a < static_cast<int>(out.ndim()); ++a)
        {
            out = np::roll(out, sign * (out.shape[a] / 2), a);
        }
    }
    return out;
}

} // namespace detail

/**
 * @brief Shift the zero-frequency component to the center of the spectrum.
 *
 * This function swaps half-spaces for all axes listed (defaults to all).
 * Note that y[0] is the Nyquist component only if len(x) is even.
 *
 * @param x     Input array.
 * @param axes  Axes over which to shift (default: all).
 *
 * Reference: numpy reference/generated/numpy.fft.fftshift.html
 */
NP_API template <typename T>
NP_NODISCARD auto fftshift(const ndarray<T> &x, std::optional<std::vector<int>> axes = std::nullopt) -> ndarray<T>
{
    return detail::shift_roll(x, axes, +1);
}

/**
 * @brief The inverse of `fftshift`.
 *
 * Although identical to fftshift for even-length signals, the two differ by
 * one sample for odd-length signals.
 *
 * @param x     Input array.
 * @param axes  Axes over which to shift (default: all).
 *
 * @see fftshift
 */
NP_API template <typename T>
NP_NODISCARD auto ifftshift(const ndarray<T> &x, std::optional<std::vector<int>> axes = std::nullopt) -> ndarray<T>
{
    return detail::shift_roll(x, axes, -1);
}

} // namespace np::fft

#endif // NP_FFT_SHIFT_HPP