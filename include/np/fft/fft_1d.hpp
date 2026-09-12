/**
 * @file fft/fft_1d.hpp
 * @brief One-dimensional discrete Fourier transforms (np::fft::fft family).
 *
 * Complex-input transforms and the real-input family:
 *  - fft / ifft
 *  - rfft / irfft
 *  - hfft / ihfft
 *
 * All operate along a configurable axis with optional padding/truncation
 * (`n`) and a NumPy-compatible normalization mode (`norm`). Inputs are
 * promoted to std::complex<double>.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_FFT_1D_HPP
#define NP_FFT_1D_HPP

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../api_macros.hpp"
#include "../ndarray.hpp"
#include "fft_core.hpp"

namespace np::fft
{

/** @brief Element types accepted by the transform templates. */
template <typename T>
inline constexpr bool is_transform_element_v = std::is_arithmetic_v<T> || np::detail::is_complex_v<T>;

namespace detail
{

/** @brief Resolve the transform length, rejecting 0 (NumPy ValueError). */
NP_NODISCARD inline std::size_t deduced_len(bool has_n, std::size_t n, std::size_t axis_len)
{
    return has_n ? check_len(n) : check_len(axis_len);
}

/** @brief Default output length for the inverse real transforms
 *         (2*(m-1), throwing when m < 2). */
NP_NODISCARD inline std::size_t inverse_real_len(std::size_t axis_len)
{
    if (axis_len < 2)
    {
        throw std::invalid_argument("Invalid number of FFT data points (0).");
    }
    return (axis_len - 1) * 2;
}

} // namespace detail

/** @brief 1-D discrete Fourier transform of a complex sequence.
 *
 * Reference: numpy-reference/reference/generated/numpy.fft.fft.html
 */
NP_API NP_NODISCARD inline auto fft(const std::vector<Cplx> &x) -> std::vector<Cplx>
{
    std::vector<Cplx> out(x);
    detail::transform(out, false, 1.0, detail::twiddle_cache());
    return out;
}

/** @brief 1-D inverse discrete Fourier transform of a complex sequence.
 *
 * Reference: numpy-reference/reference/generated/numpy.fft.ifft.html
 */
NP_API NP_NODISCARD inline auto ifft(const std::vector<Cplx> &x) -> std::vector<Cplx>
{
    std::vector<Cplx> out(x);
    detail::transform(out, true, 1.0 / static_cast<double>(out.size()), detail::twiddle_cache());
    return out;
}

/**
 * @brief Compute the 1-D discrete Fourier Transform.
 *
 * @tparam T    Element type (arithmetic or std::complex).
 * @param x     Input array, can be complex.
 * @param n     Length of the transformed axis of the output (zero-pads or
 *              crops when given; defaults to the input axis length).
 * @param axis  Axis over which to compute the FFT (default: last).
 * @param norm  Normalization mode (default: Backward).
 * @return The transformed array (complex).
 *
 * Reference: numpy-reference/reference/generated/numpy.fft.fft.html
 */
NP_API template <typename T>
NP_NODISCARD auto fft(const ndarray<T> &x, std::optional<std::size_t> n = std::nullopt, int axis = -1,
                      Norm norm = Norm::Backward) -> ndarray<Cplx>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    if (x.ndim() == 0)
    {
        throw std::invalid_argument("fft: input must have at least one dimension");
    }
    const int ax = detail::normalize_axis(axis, x.ndim());
    const std::size_t len = detail::deduced_len(n.has_value(), n.value_or(0), static_cast<std::size_t>(x.shape[ax]));
    ndarray<Cplx> out(detail::with_axis_len(x.shape, ax, len));
    detail::transform_lines(x, ax, len, out, false, detail::scale_factor(norm, len, false), detail::twiddle_cache());
    return out;
}

/**
 * @brief Compute the 1-D inverse discrete Fourier Transform.
 *
 * Reference: numpy-reference/reference/generated/numpy.fft.ifft.html
 */
NP_API template <typename T>
NP_NODISCARD auto ifft(const ndarray<T> &x, std::optional<std::size_t> n = std::nullopt, int axis = -1,
                       Norm norm = Norm::Backward) -> ndarray<Cplx>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    if (x.ndim() == 0)
    {
        throw std::invalid_argument("ifft: input must have at least one dimension");
    }
    const int ax = detail::normalize_axis(axis, x.ndim());
    const std::size_t len = detail::deduced_len(n.has_value(), n.value_or(0), static_cast<std::size_t>(x.shape[ax]));
    ndarray<Cplx> out(detail::with_axis_len(x.shape, ax, len));
    detail::transform_lines(x, ax, len, out, true, detail::scale_factor(norm, len, true), detail::twiddle_cache());
    return out;
}

/**
 * @brief Compute the 1-D discrete Fourier Transform for real input.
 *
 * The output has length n/2+1 (non-negative frequencies); the imaginary
 * part of the input is silently discarded.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
 */
NP_API template <typename T>
NP_NODISCARD auto rfft(const ndarray<T> &x, std::optional<std::size_t> n = std::nullopt, int axis = -1,
                       Norm norm = Norm::Backward) -> ndarray<Cplx>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    if (x.ndim() == 0)
    {
        throw std::invalid_argument("rfft: input must have at least one dimension");
    }
    const int ax = detail::normalize_axis(axis, x.ndim());
    const std::size_t len = detail::deduced_len(n.has_value(), n.value_or(0), static_cast<std::size_t>(x.shape[ax]));
    ndarray<Cplx> out(detail::with_axis_len(x.shape, ax, len / 2 + 1));
    detail::rfft_lines(x, ax, len, out, detail::scale_factor(norm, len, false), detail::twiddle_cache());
    return out;
}

/**
 * @brief Compute the inverse of `rfft` (real output).
 *
 * The input holds the non-negative frequencies of a Hermitian-symmetric
 * spectrum; `n` is the length of the output (default 2*(m-1) where m is the
 * input axis length).
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html
 */
NP_API template <typename T>
NP_NODISCARD auto irfft(const ndarray<T> &x, std::optional<std::size_t> n = std::nullopt, int axis = -1,
                        Norm norm = Norm::Backward) -> ndarray<double>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    if (x.ndim() == 0)
    {
        throw std::invalid_argument("irfft: input must have at least one dimension");
    }
    const int ax = detail::normalize_axis(axis, x.ndim());
    const std::size_t m = static_cast<std::size_t>(x.shape[ax]);
    const std::size_t len = n.has_value() ? detail::check_len(*n) : detail::inverse_real_len(m);
    ndarray<double> out(detail::with_axis_len(x.shape, ax, len));
    ndarray<Cplx> cx = detail::to_complex(x);
    detail::irfft_lines(cx, ax, len, out, detail::scale_factor(norm, len, true), detail::twiddle_cache());
    return out;
}

/**
 * @brief Compute the FFT of a signal that has Hermitian symmetry (real output).
 *
 * This mirrors NumPy exactly: ``hfft(x, n, axis, norm)`` is
 * ``irfft(conj(x), n, axis, swapped(norm))``. `n` defaults to ``2*(m-1)``
 * with m the input axis length.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html
 */
NP_API template <typename T>
NP_NODISCARD auto hfft(const ndarray<T> &x, std::optional<std::size_t> n = std::nullopt, int axis = -1,
                       Norm norm = Norm::Backward) -> ndarray<double>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    ndarray<Cplx> cx = detail::conjugate_copy(x);
    return irfft(cx, n, axis, detail::swapped(norm));
}

/**
 * @brief Compute the inverse FFT of a signal with Hermitian symmetry
 *        (complex output of length n/2+1).
 *
 * Exactly as in NumPy: ``ihfft(x, n, axis, norm)`` is
 * ``conj(rfft(x, n, axis, swapped(norm)))``. `n` defaults to the input axis
 * length and the output has length n/2+1.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html
 */
NP_API template <typename T>
NP_NODISCARD auto ihfft(const ndarray<T> &x, std::optional<std::size_t> n = std::nullopt, int axis = -1,
                        Norm norm = Norm::Backward) -> ndarray<Cplx>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    ndarray<Cplx> out = rfft(x, n, axis, detail::swapped(norm));
    detail::conjugate_inplace(out);
    return out;
}

} // namespace np::fft

#endif // NP_FFT_1D_HPP