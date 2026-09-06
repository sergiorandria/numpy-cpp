/**
 * @file fft/fft_nd.hpp
 * @brief Multi-dimensional discrete Fourier transforms (np::fft::fftn family).
 *
 * Implements fftn / ifftn and the real-input variants rfftn / irfftn, plus
 * the 2-D convenience wrappers fft2 / ifft2 / rfft2 / irfft2.
 *
 * The transforms are applied axis by axis in the same order NumPy uses:
 *  - fftn / ifftn: every axis is transformed (order is irrelevant, all are
 *    complex transforms).
 *  - rfftn: the real transform runs on the last listed axis first, then the
 *    complex transforms proceed on the remaining axes in reverse order.
 *  - irfftn: complex inverse transforms run on all but the last axis first,
 *    then the real inverse runs on the last axis.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_FFT_ND_HPP
#define NP_FFT_ND_HPP

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../api_macros.hpp"
#include "../ndarray.hpp"
#include "fft_1d.hpp"
#include "fft_core.hpp"

namespace np::fft
{
namespace detail
{

/** @brief Resolved per-axis transform plan (normalized axes and lengths). */
struct NdPlan
{
    std::vector<int> axes;
    std::vector<std::size_t> lens;
};

/**
 * @brief Resolve `s`/`axes` the way numpy.fft._cook_nd_args does.
 *
 * @param invreal_last For the inverse real family (irfftn): when `s` is
 *          omitted the final axis default length becomes 2*(m-1).
 */
template <typename T>
NP_NODISCARD NdPlan cook_nd(const ndarray<T> &x, const std::optional<std::vector<int>> &s,
                            const std::optional<std::vector<int>> &axes, bool invreal_last)
{
    const std::size_t nd = x.ndim();
    std::vector<int> ax;
    if (axes)
    {
        ax.reserve(axes->size());
        for (int a : *axes)
        {
            ax.push_back(normalize_axis(a, nd));
        }
    }
    else
    {
        for (std::size_t i = 0; i < nd; ++i)
        {
            ax.push_back(static_cast<int>(i));
        }
    }
    {
        std::vector<bool> seen(nd, false);
        for (int a : ax)
        {
            if (seen[static_cast<std::size_t>(a)])
            {
                throw std::invalid_argument("axes contains duplicate axes");
            }
            seen[static_cast<std::size_t>(a)] = true;
        }
    }

    std::vector<std::size_t> lens;
    if (s)
    {
        if (s->size() != ax.size())
        {
            throw std::invalid_argument("Shape and axes have different lengths.");
        }
        lens.reserve(s->size());
        for (std::size_t i = 0; i < s->size(); ++i)
        {
            const int v = (*s)[i];
            if (v == -1)
            {
                lens.push_back(static_cast<std::size_t>(x.shape[ax[i]]));
            }
            else
            {
                lens.push_back(check_len(static_cast<std::size_t>(v)));
            }
        }
    }
    else
    {
        lens.reserve(ax.size());
        for (int a : ax)
        {
            lens.push_back(static_cast<std::size_t>(x.shape[a]));
        }
        if (invreal_last && !lens.empty())
        {
            const std::size_t m = lens.back();
            if (m < 2)
            {
                throw std::invalid_argument("Invalid number of FFT data points (0).");
            }
            lens.back() = (m - 1) * 2;
        }
    }

    NdPlan plan;
    plan.axes = std::move(ax);
    plan.lens = std::move(lens);
    return plan;
}

/** @brief Shape of `x` with dimension `axis` replaced by `len`. */
inline std::vector<int> adjust_axis(const ndarray<Cplx> &x, int axis, std::size_t len)
{
    return with_axis_len(x.shape, axis, len);
}

} // namespace detail

/**
 * @brief Compute the N-dimensional discrete Fourier Transform.
 *
 * @param x     Input array, can be complex.
 * @param s     Shape (length of each transformed axis); -1 keeps the full
 *          input length along that axis.
 * @param axes  Axes over which to compute the transform (default: all).
 * @param norm  Normalization mode (default: Backward).
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html
 */
NP_API template <typename T>
NP_NODISCARD auto fftn(const ndarray<T> &x, std::optional<std::vector<int>> s = std::nullopt,
                       std::optional<std::vector<int>> axes = std::nullopt, Norm norm = Norm::Backward) -> ndarray<Cplx>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    if (x.ndim() == 0)
    {
        throw std::invalid_argument("fftn: input must have at least one dimension");
    }
    detail::NdPlan plan = detail::cook_nd(x, s, axes, false);
    ndarray<Cplx> cur = detail::to_complex(x);
    const auto &cache = detail::twiddle_cache();
    for (std::size_t i = 0; i < plan.axes.size(); ++i)
    {
        const int a = plan.axes[i];
        const std::size_t len = plan.lens[i];
        ndarray<Cplx> nxt(detail::with_axis_len(cur.shape, a, len));
        detail::transform_lines(cur, a, len, nxt, false, detail::scale_factor(norm, len, false), cache);
        cur = std::move(nxt);
    }
    return cur;
}

/**
 * @brief Compute the N-dimensional inverse discrete Fourier Transform.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html
 */
NP_API template <typename T>
NP_NODISCARD auto ifftn(const ndarray<T> &x, std::optional<std::vector<int>> s = std::nullopt,
                        std::optional<std::vector<int>> axes = std::nullopt, Norm norm = Norm::Backward)
    -> ndarray<Cplx>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    if (x.ndim() == 0)
    {
        throw std::invalid_argument("ifftn: input must have at least one dimension");
    }
    detail::NdPlan plan = detail::cook_nd(x, s, axes, false);
    ndarray<Cplx> cur = detail::to_complex(x);
    const auto &cache = detail::twiddle_cache();
    for (std::size_t i = 0; i < plan.axes.size(); ++i)
    {
        const int a = plan.axes[i];
        const std::size_t len = plan.lens[i];
        ndarray<Cplx> nxt(detail::with_axis_len(cur.shape, a, len));
        detail::transform_lines(cur, a, len, nxt, true, detail::scale_factor(norm, len, true), cache);
        cur = std::move(nxt);
    }
    return cur;
}

/**
 * @brief Two-dimensional discrete Fourier Transform.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
 */
NP_API template <typename T>
NP_NODISCARD auto fft2(const ndarray<T> &x, std::optional<std::vector<int>> s = std::nullopt,
                       std::optional<std::vector<int>> axes = std::nullopt, Norm norm = Norm::Backward) -> ndarray<Cplx>
{
    if (!axes)
    {
        axes = std::vector<int>{-2, -1};
    }
    return fftn(x, s, axes, norm);
}

/**
 * @brief Two-dimensional inverse discrete Fourier Transform.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html
 */
NP_API template <typename T>
NP_NODISCARD auto ifft2(const ndarray<T> &x, std::optional<std::vector<int>> s = std::nullopt,
                        std::optional<std::vector<int>> axes = std::nullopt, Norm norm = Norm::Backward)
    -> ndarray<Cplx>
{
    if (!axes)
    {
        axes = std::vector<int>{-2, -1};
    }
    return ifftn(x, s, axes, norm);
}

/**
 * @brief N-dimensional discrete Fourier Transform for real input.
 *
 * The real transform is applied on the last listed axis (output length
 * s[-1]/2+1) and the remaining axes receive complex transforms.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html
 */
NP_API template <typename T>
NP_NODISCARD auto rfftn(const ndarray<T> &x, std::optional<std::vector<int>> s = std::nullopt,
                        std::optional<std::vector<int>> axes = std::nullopt, Norm norm = Norm::Backward)
    -> ndarray<Cplx>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    if (x.ndim() == 0)
    {
        throw std::invalid_argument("rfftn: input must have at least one dimension");
    }
    detail::NdPlan plan = detail::cook_nd(x, s, axes, false);
    if (plan.axes.empty())
    {
        return detail::to_complex(x);
    }
    const auto &cache = detail::twiddle_cache();

    // Real transform first, on the last listed axis.
    const int last_axis = plan.axes.back();
    const std::size_t last_len = plan.lens.back();
    ndarray<Cplx> cur(detail::with_axis_len(x.shape, last_axis, last_len / 2 + 1));
    detail::rfft_lines(x, last_axis, last_len, cur, detail::scale_factor(norm, last_len, false), cache);

    // Complex transforms on the remaining axes, in reverse order.
    for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(plan.axes.size()) - 2; j >= 0; --j)
    {
        const int a = plan.axes[static_cast<std::size_t>(j)];
        const std::size_t len = plan.lens[static_cast<std::size_t>(j)];
        ndarray<Cplx> nxt(detail::with_axis_len(cur.shape, a, len));
        detail::transform_lines(cur, a, len, nxt, false, detail::scale_factor(norm, len, false), cache);
        cur = std::move(nxt);
    }
    return cur;
}

/**
 * @brief Inverse of the N-dimensional real discrete Fourier Transform.
 *
 * Complex inverse transforms run on all axes but the last first; the final
 * Hermitian inverse runs on the last axis and produces a real array.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html
 */
NP_API template <typename T>
NP_NODISCARD auto irfftn(const ndarray<T> &x, std::optional<std::vector<int>> s = std::nullopt,
                         std::optional<std::vector<int>> axes = std::nullopt, Norm norm = Norm::Backward)
    -> ndarray<double>
{
    static_assert(is_transform_element_v<T>, "np::fft requires arithmetic or std::complex element types");
    if (x.ndim() == 0)
    {
        throw std::invalid_argument("irfftn: input must have at least one dimension");
    }
    detail::NdPlan plan = detail::cook_nd(x, s, axes, true);
    if (plan.axes.empty())
    {
        throw std::invalid_argument("irfftn: no axis to transform");
    }
    const auto &cache = detail::twiddle_cache();
    ndarray<Cplx> cur = detail::to_complex(x);
    for (std::size_t i = 0; i + 1 < plan.axes.size(); ++i)
    {
        const int a = plan.axes[i];
        const std::size_t len = plan.lens[i];
        ndarray<Cplx> nxt(detail::with_axis_len(cur.shape, a, len));
        detail::transform_lines(cur, a, len, nxt, true, detail::scale_factor(norm, len, true), cache);
        cur = std::move(nxt);
    }
    const int last_axis = plan.axes.back();
    const std::size_t last_len = plan.lens.back();
    ndarray<double> out(detail::with_axis_len(cur.shape, last_axis, last_len));
    detail::irfft_lines(cur, last_axis, last_len, out, detail::scale_factor(norm, last_len, true), cache);
    return out;
}

/**
 * @brief Two-dimensional real FFT.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft2.html
 */
NP_API template <typename T>
NP_NODISCARD auto rfft2(const ndarray<T> &x, std::optional<std::vector<int>> s = std::nullopt,
                        std::optional<std::vector<int>> axes = std::nullopt, Norm norm = Norm::Backward)
    -> ndarray<Cplx>
{
    if (!axes)
    {
        axes = std::vector<int>{-2, -1};
    }
    return rfftn(x, s, axes, norm);
}

/**
 * @brief Inverse of the two-dimensional real FFT.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html
 */
NP_API template <typename T>
NP_NODISCARD auto irfft2(const ndarray<T> &x, std::optional<std::vector<int>> s = std::nullopt,
                         std::optional<std::vector<int>> axes = std::nullopt, Norm norm = Norm::Backward)
    -> ndarray<double>
{
    if (!axes)
    {
        axes = std::vector<int>{-2, -1};
    }
    return irfftn(x, s, axes, norm);
}

} // namespace np::fft

#endif // NP_FFT_ND_HPP