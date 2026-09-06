/**
 * @file fft/fft_core.hpp
 * @brief Shared internal machinery for np::fft.
 *
 * Provides the normalization enum, the cached twiddle tables, the
 * un-normalized radix-2 / Bluestein engines driven by an explicit scale
 * factor (the inverse scaling is no longer baked into the engine), and the
 * helpers used to apply a 1-D transform along an arbitrary axis.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_FFT_CORE_HPP
#define NP_FFT_CORE_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../api_macros.hpp"
#include "../dtype.hpp"
#include "../exceptions.hpp"
#include "../gpu.hpp"
#include "../ndarray.hpp"

#ifdef NP_USE_THREADING
#include "../threadpool.hpp"
#endif

namespace np::fft
{

/** @brief Complex type used by the FFT routines. */
using Cplx = std::complex<double>;

/**
 * @brief Normalization convention for a forward/inverse pair of transforms.
 *
 * Mirrors NumPy's ``dft convention. For a transform of length n:
 *  - Backward: forward unscaled, inverse scaled by 1/n.
 *  - Ortho:    both scaled by 1/sqrt(n).
 *  - Forward:  forward scaled by 1/n, inverse unscaled.
 */
enum class Norm : std::uint8_t
{
    Backward,
    Ortho,
    Forward
};

namespace detail
{

/** @brief Smallest power of two >= n. */
inline std::size_t next_pow2(std::size_t n)
{
    std::size_t p = 1;
    while (p < n)
    {
        p <<= 1;
    }
    return p;
}

/** @brief Swap a normalization direction (NumPy ``_swap_direction``). */
inline constexpr Norm swapped(const Norm n) noexcept
{
    switch (n)
    {
    case Norm::Forward:
        return Norm::Backward;
    case Norm::Backward:
        return Norm::Forward;
    case Norm::Ortho:
        break;
    }
    return Norm::Ortho;
}

/**
 * @brief Scale applied to the raw (un-normalized) DFT result.
 *
 * For transforms running in the inverse direction the norm is swapped first,
 * matching NumPy's handling inside ``_raw_fft``.
 */
inline double scale_factor(const Norm norm, std::size_t len, bool inverse) noexcept
{
    const Norm n = inverse ? swapped(norm) : norm;
    switch (n)
    {
    case Norm::Ortho:
        return 1.0 / std::sqrt(static_cast<double>(len));
    case Norm::Forward:
        return 1.0 / static_cast<double>(len);
    case Norm::Backward:
        break;
    }
    return 1.0;
}

/** @brief Reject a non-positive transform length (NumPy ValueError). */
inline std::size_t check_len(std::size_t n)
{
    if (n <= 0)
    {
        throw std::invalid_argument("Invalid number of FFT data points (0).");
    }
    return n;
}

/** @brief Normalize an axis index; throws np::AxisError when out of bounds. */
NP_NODISCARD inline int normalize_axis(int axis, std::size_t nd)
{
    if (axis < 0)
    {
        axis += static_cast<int>(nd);
    }
    if (axis < 0 || axis >= static_cast<int>(nd))
    {
        throw np::AxisError("axis " + std::to_string(axis) + " is out of bounds for array of dimension " +
                            std::to_string(nd));
    }
    return axis;
}

/** @brief Compute a forward twiddle table t[k] = exp(-2*pi*i*k/n). */
inline std::vector<Cplx> make_radix_table(std::size_t n)
{
    std::vector<Cplx> t(n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k)
    {
        const double angle = -2.0 * std::numbers::pi_v<double> * static_cast<double>(k) / static_cast<double>(n);
        t[k] = Cplx{std::cos(angle), std::sin(angle)};
    }
    return t;
}

/**
 * @brief In-place iterative radix-2 butterflies using a precomputed table.
 *
 * @param n       Transform length (a power of two, must equal a.size()).
 * @param tbl     Forward table of length n/2+1 (see make_radix_table).
 * @param inverse Use conjugated twiddles (inverse DFT direction).
 */
inline void radix2_apply(std::vector<Cplx> &a, std::size_t n, const std::vector<Cplx> &tbl, bool inverse)
{
    for (std::size_t i = 1, j = 0; i < n; ++i)
    {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
        {
            j ^= bit;
        }
        j ^= bit;
        if (i < j)
        {
            std::swap(a[i], a[j]);
        }
    }

    for (std::size_t len = 2; len <= n; len <<= 1)
    {
        const std::size_t half = len >> 1;
        const std::size_t step = n / len;
        for (std::size_t i = 0; i < n; i += len)
        {
            // Micro-unrolled butterfly: 4-way unroll reduces loop overhead
            // and improves ILP for the complex multiply-add. Remainder
            // handled scalarly.
            std::size_t k = 0;
            const std::size_t unroll = 4;
            const std::size_t limit = half & ~(unroll - 1);
            for (; k < limit; k += unroll)
            {
                Cplx w0 = tbl[(k + 0) * step];
                Cplx w1 = tbl[(k + 1) * step];
                Cplx w2 = tbl[(k + 2) * step];
                Cplx w3 = tbl[(k + 3) * step];
                if (inverse)
                {
                    w0 = std::conj(w0);
                    w1 = std::conj(w1);
                    w2 = std::conj(w2);
                    w3 = std::conj(w3);
                }
                const Cplx u0 = a[i + k + 0];
                const Cplx u1 = a[i + k + 1];
                const Cplx u2 = a[i + k + 2];
                const Cplx u3 = a[i + k + 3];
                const Cplx v0 = a[i + k + 0 + half] * w0;
                const Cplx v1 = a[i + k + 1 + half] * w1;
                const Cplx v2 = a[i + k + 2 + half] * w2;
                const Cplx v3 = a[i + k + 3 + half] * w3;
                a[i + k + 0] = u0 + v0;
                a[i + k + 1] = u1 + v1;
                a[i + k + 2] = u2 + v2;
                a[i + k + 3] = u3 + v3;
                a[i + k + 0 + half] = u0 - v0;
                a[i + k + 1 + half] = u1 - v1;
                a[i + k + 2 + half] = u2 - v2;
                a[i + k + 3 + half] = u3 - v3;
            }
            for (; k < half; ++k)
            {
                Cplx w = tbl[k * step];
                if (inverse)
                {
                    w = std::conj(w);
                }
                const Cplx u = a[i + k];
                const Cplx v = a[i + k + half] * w;
                a[i + k] = u + v;
                a[i + k + half] = u - v;
            }
        }
    }
}

/** @brief Precomputed Bluestein machinery for a given transform length. */
struct BluesteinPlan
{
    std::vector<Cplx> chirp;  ///< exp(+/-(pi)*i*j*j/n), j = 0..n-1
    std::size_t conv_len = 0; ///< Power-of-two convolution buffer length
    std::vector<Cplx> conv;   ///< Forward FFT of the padded reversed chirp
};

/** @brief Per-length cached twiddle tables (not thread-safe). */
class TwiddleCache
{
  public:
    /** @brief Forward radix-2 table t[k] = exp(-2*pi*i*k/n), k = 0..n/2. */
    NP_NODISCARD const std::vector<Cplx> &radix_table(std::size_t n) const
    {
        auto it = fwd_.find(n);
        if (it == fwd_.end())
        {
            it = fwd_.emplace(n, make_radix_table(n)).first;
        }
        return it->second;
    }

    /** @brief Lazily-built Bluestein plan (chirp + kernel FFT). */
    NP_NODISCARD const BluesteinPlan &bluestein_plan(std::size_t n, bool inverse) const
    {
        auto &tbl = inverse ? bn_ : bf_;
        auto it = tbl.find(n);
        if (it == tbl.end())
        {
            const double s = inverse ? 1.0 : -1.0;
            const double inv_n = 1.0 / static_cast<double>(n);

            BluesteinPlan plan;
            plan.chirp.reserve(n);
            for (std::size_t j = 0; j < n; ++j)
            {
                const double angle = s * std::numbers::pi_v<double> * static_cast<double>(j * j) * inv_n;
                plan.chirp.push_back({std::cos(angle), std::sin(angle)});
            }

            plan.conv_len = next_pow2(3 * n - 2);
            plan.conv.assign(plan.conv_len, Cplx{0.0, 0.0});
            for (std::size_t j = 0; j < 2 * n - 1; ++j)
            {
                const long m = static_cast<long>(j) - static_cast<long>(n - 1);
                const double angle = -s * std::numbers::pi_v<double> * static_cast<double>(m * m) * inv_n;
                plan.conv[j] = {std::cos(angle), std::sin(angle)};
            }
            if (plan.conv_len > 1)
            {
                const std::vector<Cplx> &tbl2 = radix_table(plan.conv_len);
                radix2_apply(plan.conv, plan.conv_len, tbl2, false);
            }
            it = tbl.emplace(n, std::move(plan)).first;
        }
        return it->second;
    }

  private:
    mutable std::unordered_map<std::size_t, std::vector<Cplx>> fwd_;
    mutable std::unordered_map<std::size_t, BluesteinPlan> bf_, bn_;
};

/** @brief Returns the shared twiddle cache (not thread-safe). */
NP_NODISCARD inline const TwiddleCache &twiddle_cache()
{
    static const TwiddleCache cache;
    return cache;
}

/** @brief In-place radix-2 FFT (n = a.size() must be a power of two). */
inline void radix2(std::vector<Cplx> &a, bool inverse, double scale, const TwiddleCache &cache)
{
    const std::size_t n = a.size();
    if (n <= 1)
    {
        if (scale != 1.0)
        {
            for (auto &v : a)
            {
                v *= scale;
            }
        }
        return;
    }
    const std::vector<Cplx> &tbl = cache.radix_table(n);
    radix2_apply(a, n, tbl, inverse);
    if (scale != 1.0)
    {
        for (auto &v : a)
        {
            v *= scale;
        }
    }
}

/**
 * @brief In-place Bluestein FFT for an arbitrary length n.
 *
 * Computes X[k] = sum_j x[j] * exp(+-2*pi*i*j*k/n) (sign selects
 * forward/inverse); the result is scaled by `scale`.
 */
inline void bluestein(std::vector<Cplx> &x, bool inverse, double scale, const TwiddleCache &cache)
{
    const std::size_t n = x.size();
    if (n <= 1)
    {
        if (scale != 1.0)
        {
            for (auto &v : x)
            {
                v *= scale;
            }
        }
        return;
    }
    const BluesteinPlan &plan = cache.bluestein_plan(n, inverse);

    std::vector<Cplx> b(plan.conv_len, Cplx{0.0, 0.0});
    for (std::size_t j = 0; j < n; ++j)
    {
        b[j] = x[j] * plan.chirp[j];
    }

    radix2(b, false, 1.0, cache);
    for (std::size_t i = 0; i < plan.conv_len; ++i)
    {
        b[i] *= plan.conv[i];
    }
    radix2(b, true, 1.0, cache);

    // The radix-2 stages are un-normalized, so the circular convolution
    // carries a factor of 1/conv_len that must be removed here.
    const double inv_n = 1.0 / static_cast<double>(plan.conv_len);
    for (std::size_t k = 0; k < n; ++k)
    {
        x[k] = plan.chirp[k] * b[k + n - 1] * inv_n;
    }
    if (scale != 1.0)
    {
        for (auto &v : x)
        {
            v *= scale;
        }
    }
}

/** @brief FFT of a single 1-D sequence with the requested direction/scale. */
inline void transform(std::vector<Cplx> &a, bool inverse, double scale, const TwiddleCache &cache)
{
    const std::size_t n = a.size();
    if (n <= 1)
    {
        if (scale != 1.0)
        {
            for (auto &v : a)
            {
                v *= scale;
            }
        }
        return;
    }
    // GPU offload for large N (powerful workstation + GPU)
    if (n >= 8192 && ::np::gpu::is_available())
    {
        std::vector<Cplx> out(n);
        if (::np::gpu::try_fft(a.data(), out.data(), n, inverse))
        {
            for (std::size_t i = 0; i < n; ++i)
                a[i] = out[i] * scale;
            return;
        }
    }
    if ((n & (n - 1)) == 0)
    {
        radix2(a, inverse, scale, cache);
    }
    else
    {
        bluestein(a, inverse, scale, cache);
    }
}

/** @brief Copy any numeric array into a fresh complex array. */
template <typename T> NP_NODISCARD ndarray<Cplx> to_complex(const ndarray<T> &x)
{
    ndarray<Cplx> out(x.shape);
    for (std::size_t i = 0; i < x._numel(); ++i)
    {
        out.data()[i] = static_cast<Cplx>(x.data()[x._flat_logical(i)]);
    }
    return out;
}

/** @brief Copy any numeric array, conjugating each element. */
template <typename T> NP_NODISCARD ndarray<Cplx> conjugate_copy(const ndarray<T> &x)
{
    ndarray<Cplx> out(x.shape);
    for (std::size_t i = 0; i < x._numel(); ++i)
    {
        const Cplx v = static_cast<Cplx>(x.data()[x._flat_logical(i)]);
        out.data()[i] = {v.real(), -v.imag()};
    }
    return out;
}

/** @brief Replace dimension `axis` by `len` in a shape vector. */
inline std::vector<int> with_axis_len(const std::vector<int> &shape, int axis, std::size_t len)
{
    std::vector<int> s = shape;
    s[static_cast<std::size_t>(axis)] = static_cast<int>(len);
    return s;
}

// Applying a 1-D transform along an arbitrary axis
/**
 * @brief Transform every 1-D line of `src` along `axis`.
 *
 * Each line is (zero-)padded or truncated to length `n` before the complex
 * transform runs in-place in a temporary; the full `n` result is then
 * written to the line of `dst`. Handles strided views.
 *
 * @tparam T  Element type of the source array.
 */
template <typename T>
inline void transform_lines(const ndarray<T> &src, int axis, std::size_t n, ndarray<Cplx> &dst, bool inverse,
                            double scale, const TwiddleCache &cache)
{
    const std::size_t ax = static_cast<std::size_t>(axis);
    const std::size_t src_len = static_cast<std::size_t>(src.shape[static_cast<std::ptrdiff_t>(axis)]);
    const std::size_t read = std::min(n, src_len);
    const std::size_t nd = src.ndim();
    const std::size_t src_stride = src.strides[ax];
    const std::size_t dst_stride = dst.strides[ax];

    std::vector<int> od_dims;
    od_dims.reserve(nd == 0 ? 0 : nd - 1);
    for (std::size_t d = 0; d < nd; ++d)
    {
        if (d != ax)
        {
            od_dims.push_back(src.shape[d]);
        }
    }

    std::vector<Cplx> slice(n, Cplx{0.0, 0.0});
    std::vector<std::size_t> full(nd, 0);
#ifdef NP_USE_THREADING
    // Collect outer indices for parallel dispatch
    std::vector<std::vector<std::size_t>> all_oi;
    {
        np::detail::Odometer od_tmp(od_dims);
        while (!od_tmp.done())
        {
            all_oi.push_back(od_tmp.idx());
            od_tmp.advance();
        }
        if (all_oi.empty())
        {
            all_oi.push_back({});
        }
    }
    const std::size_t n_outer = all_oi.size();
    if (n_outer > 4)
    {
        ::np::ThreadPool::global().parallel_for(0, n_outer, [&](std::size_t oi_idx) {
            const auto &oi = all_oi[oi_idx];
            std::vector<Cplx> slice_local(n, Cplx{0.0, 0.0});
            std::vector<std::size_t> full_local(nd, 0);
            std::size_t p = 0;
            for (std::size_t d = 0; d < nd; ++d)
            {
                if (d != ax)
                {
                    full_local[d] = oi[p++];
                }
            }
            const std::size_t base = np::detail::flat_index(full_local, src.strides, src.offset);
            for (std::size_t k = 0; k < read; ++k)
            {
                slice_local[k] = static_cast<Cplx>(src.data()[base + k * src_stride]);
            }
            for (std::size_t k = read; k < n; ++k)
            {
                slice_local[k] = Cplx{0.0, 0.0};
            }
            transform(slice_local, inverse, scale, cache);
            const std::size_t db = np::detail::flat_index(full_local, dst.strides, dst.offset);
            for (std::size_t k = 0; k < n; ++k)
            {
                dst.data()[db + k * dst_stride] = slice_local[k];
            }
        });
        return;
    }
    // Fallback to serial for small n_outer
    np::detail::Odometer od(od_dims);
    while (!od.done())
    {
        const auto &oi = od.idx();
        std::size_t p = 0;
        for (std::size_t d = 0; d < nd; ++d)
        {
            if (d != ax)
            {
                full[d] = oi[p++];
            }
        }
        const std::size_t base = np::detail::flat_index(full, src.strides, src.offset);
        for (std::size_t k = 0; k < read; ++k)
        {
            slice[k] = static_cast<Cplx>(src.data()[base + k * src_stride]);
        }
        for (std::size_t k = read; k < n; ++k)
        {
            slice[k] = Cplx{0.0, 0.0};
        }
        transform(slice, inverse, scale, cache);

        const std::size_t db = np::detail::flat_index(full, dst.strides, dst.offset);
        for (std::size_t k = 0; k < n; ++k)
        {
            dst.data()[db + k * dst_stride] = slice[k];
        }
        od.advance();
    }
#else
    np::detail::Odometer od(od_dims);
    while (!od.done())
    {
        const auto &oi = od.idx();
        std::size_t p = 0;
        for (std::size_t d = 0; d < nd; ++d)
        {
            if (d != ax)
            {
                full[d] = oi[p++];
            }
        }
        const std::size_t base = np::detail::flat_index(full, src.strides, src.offset);
        for (std::size_t k = 0; k < read; ++k)
        {
            slice[k] = static_cast<Cplx>(src.data()[base + k * src_stride]);
        }
        for (std::size_t k = read; k < n; ++k)
        {
            slice[k] = Cplx{0.0, 0.0};
        }
        transform(slice, inverse, scale, cache);

        const std::size_t db = np::detail::flat_index(full, dst.strides, dst.offset);
        for (std::size_t k = 0; k < n; ++k)
        {
            dst.data()[db + k * dst_stride] = slice[k];
        }
        od.advance();
    }
#endif
}

/**
 * @brief Forward real-to-complex transform of every 1-D line of `src`.
 *
 * Computes the full complex spectrum of length `n` and stores only the
 * non-negative frequencies (0..n/2) into `dst` (whose axis length is n/2+1).
 * The imaginary part of the source is silently discarded. This is a faithful
 * (if not maximally efficient) implementation: a packed real FFT could halve
 * the work, but for clarity the redundant half-spectrum is computed here.
 */
template <typename T>
inline void rfft_lines(const ndarray<T> &src, int axis, std::size_t n, ndarray<Cplx> &dst, double scale,
                       const TwiddleCache &cache)
{
    const std::size_t half = n / 2 + 1;
    const std::size_t ax = static_cast<std::size_t>(axis);
    const std::size_t src_len = static_cast<std::size_t>(src.shape[static_cast<std::ptrdiff_t>(axis)]);
    const std::size_t read = std::min(n, src_len);
    const std::size_t nd = src.ndim();
    const std::size_t src_stride = src.strides[ax];
    const std::size_t dst_stride = dst.strides[ax];

    std::vector<int> od_dims;
    od_dims.reserve(nd == 0 ? 0 : nd - 1);
    for (std::size_t d = 0; d < nd; ++d)
    {
        if (d != ax)
        {
            od_dims.push_back(src.shape[d]);
        }
    }

    std::vector<Cplx> slice(n, Cplx{0.0, 0.0});
    std::vector<std::size_t> full(nd, 0);
#ifdef NP_USE_THREADING
    std::vector<std::vector<std::size_t>> all_oi_r;
    {
        np::detail::Odometer od_tmp(od_dims);
        while (!od_tmp.done())
        {
            all_oi_r.push_back(od_tmp.idx());
            od_tmp.advance();
        }
        if (all_oi_r.empty())
        {
            all_oi_r.push_back({});
        }
    }
    const std::size_t n_outer_r = all_oi_r.size();
    if (n_outer_r > 4)
    {
        ::np::ThreadPool::global().parallel_for(0, n_outer_r, [&](std::size_t oi_idx) {
            const auto &oi = all_oi_r[oi_idx];
            std::vector<Cplx> slice_local(n, Cplx{0.0, 0.0});
            std::vector<std::size_t> full_local(nd, 0);
            std::size_t p = 0;
            for (std::size_t d = 0; d < nd; ++d)
            {
                if (d != ax)
                {
                    full_local[d] = oi[p++];
                }
            }
            const std::size_t base = np::detail::flat_index(full_local, src.strides, src.offset);
            for (std::size_t k = 0; k < read; ++k)
            {
                const Cplx v = static_cast<Cplx>(src.data()[base + k * src_stride]);
                slice_local[k] = Cplx{v.real(), 0.0};
            }
            for (std::size_t k = read; k < n; ++k)
            {
                slice_local[k] = Cplx{0.0, 0.0};
            }
            transform(slice_local, false, scale, cache);
            const std::size_t db = np::detail::flat_index(full_local, dst.strides, dst.offset);
            for (std::size_t k = 0; k < half; ++k)
            {
                dst.data()[db + k * dst_stride] = slice_local[k];
            }
        });
        return;
    }
    np::detail::Odometer od(od_dims);
    while (!od.done())
    {
        const auto &oi = od.idx();
        std::size_t p = 0;
        for (std::size_t d = 0; d < nd; ++d)
        {
            if (d != ax)
            {
                full[d] = oi[p++];
            }
        }
        const std::size_t base = np::detail::flat_index(full, src.strides, src.offset);
        for (std::size_t k = 0; k < read; ++k)
        {
            // Discard any imaginary part, matching numpy.rfft.
            const Cplx v = static_cast<Cplx>(src.data()[base + k * src_stride]);
            slice[k] = Cplx{v.real(), 0.0};
        }
        for (std::size_t k = read; k < n; ++k)
        {
            slice[k] = Cplx{0.0, 0.0};
        }
        transform(slice, false, scale, cache);

        const std::size_t db = np::detail::flat_index(full, dst.strides, dst.offset);
        for (std::size_t k = 0; k < half; ++k)
        {
            dst.data()[db + k * dst_stride] = slice[k];
        }
        od.advance();
    }
#else
    np::detail::Odometer od(od_dims);
    while (!od.done())
    {
        const auto &oi = od.idx();
        std::size_t p = 0;
        for (std::size_t d = 0; d < nd; ++d)
        {
            if (d != ax)
            {
                full[d] = oi[p++];
            }
        }
        const std::size_t base = np::detail::flat_index(full, src.strides, src.offset);
        for (std::size_t k = 0; k < read; ++k)
        {
            // Discard any imaginary part, matching numpy.rfft.
            const Cplx v = static_cast<Cplx>(src.data()[base + k * src_stride]);
            slice[k] = Cplx{v.real(), 0.0};
        }
        for (std::size_t k = read; k < n; ++k)
        {
            slice[k] = Cplx{0.0, 0.0};
        }
        transform(slice, false, scale, cache);

        const std::size_t db = np::detail::flat_index(full, dst.strides, dst.offset);
        for (std::size_t k = 0; k < half; ++k)
        {
            dst.data()[db + k * dst_stride] = slice[k];
        }
        od.advance();
    }
#endif
}

/**
 * @brief Inverse complex-to-real FFT of every 1-D line of `src`.
 *
 * The source line holds the non-negative frequencies (length n/2+1) of a
 * Hermitian-symmetric spectrum; the missing negative frequencies are
 * reconstructed by conjugation before an inverse transform of length `n`
 * produces the real output line.
 */
inline void irfft_lines(const ndarray<Cplx> &src, int axis, std::size_t n, ndarray<double> &dst, double scale,
                        const TwiddleCache &cache)
{
    const std::size_t hp = n / 2 + 1;
    const std::size_t ax = static_cast<std::size_t>(axis);
    const std::size_t src_len = static_cast<std::size_t>(src.shape[static_cast<std::ptrdiff_t>(axis)]);
    const std::size_t read = std::min(hp, src_len);
    const std::size_t nd = src.ndim();
    const std::size_t src_stride = src.strides[ax];
    const std::size_t dst_stride = dst.strides[ax];

    std::vector<int> od_dims;
    od_dims.reserve(nd == 0 ? 0 : nd - 1);
    for (std::size_t d = 0; d < nd; ++d)
    {
        if (d != ax)
        {
            od_dims.push_back(src.shape[d]);
        }
    }

    std::vector<Cplx> spec(hp, Cplx{0.0, 0.0});
    std::vector<Cplx> full(n, Cplx{0.0, 0.0});
    std::vector<std::size_t> full_idx(nd, 0);
#ifdef NP_USE_THREADING
    std::vector<std::vector<std::size_t>> all_oi_i;
    {
        np::detail::Odometer od_tmp(od_dims);
        while (!od_tmp.done())
        {
            all_oi_i.push_back(od_tmp.idx());
            od_tmp.advance();
        }
        if (all_oi_i.empty())
        {
            all_oi_i.push_back({});
        }
    }
    const std::size_t n_outer_i = all_oi_i.size();
    if (n_outer_i > 4)
    {
        ::np::ThreadPool::global().parallel_for(0, n_outer_i, [&](std::size_t oi_idx) {
            const auto &oi = all_oi_i[oi_idx];
            std::vector<std::size_t> full_idx_local(nd, 0);
            std::vector<Cplx> spec_local(hp, Cplx{0.0, 0.0});
            std::vector<Cplx> full_local(n, Cplx{0.0, 0.0});
            std::size_t p = 0;
            for (std::size_t d = 0; d < nd; ++d)
            {
                if (d != ax)
                {
                    full_idx_local[d] = oi[p++];
                }
            }
            const std::size_t base = np::detail::flat_index(full_idx_local, src.strides, src.offset);
            for (std::size_t k = 0; k < read; ++k)
            {
                spec_local[k] = src.data()[base + k * src_stride];
            }
            for (std::size_t k = read; k < hp; ++k)
            {
                spec_local[k] = Cplx{0.0, 0.0};
            }
            const std::size_t mid = n / 2;
            std::fill(full_local.begin(), full_local.end(), Cplx{0.0, 0.0});
            full_local[0] = spec_local[0];
            for (std::size_t k = 1; k <= mid; ++k)
            {
                full_local[k] = spec_local[k];
                if (n - k != k)
                {
                    full_local[n - k] = std::conj(spec_local[k]);
                }
                else
                {
                    full_local[k] = std::conj(spec_local[k]);
                }
            }
            transform(full_local, true, scale, cache);
            const std::size_t db = np::detail::flat_index(full_idx_local, dst.strides, dst.offset);
            for (std::size_t k = 0; k < n; ++k)
            {
                dst.data()[db + k * dst_stride] = full_local[k].real();
            }
        });
        return;
    }
    np::detail::Odometer od(od_dims);
    while (!od.done())
    {
        const auto &oi = od.idx();
        std::size_t p = 0;
        for (std::size_t d = 0; d < nd; ++d)
        {
            if (d != ax)
            {
                full_idx[d] = oi[p++];
            }
        }
        const std::size_t base = np::detail::flat_index(full_idx, src.strides, src.offset);
        for (std::size_t k = 0; k < read; ++k)
        {
            spec[k] = src.data()[base + k * src_stride];
        }
        for (std::size_t k = read; k < hp; ++k)
        {
            spec[k] = Cplx{0.0, 0.0};
        }

        // Reconstruct the full Hermitian spectrum of length n.
        const std::size_t mid = n / 2;
        std::fill(full.begin(), full.end(), Cplx{0.0, 0.0});
        full[0] = spec[0];
        for (std::size_t k = 1; k <= mid; ++k)
        {
            full[k] = spec[k];
            if (n - k != k)
            {
                full[n - k] = std::conj(spec[k]);
            }
            else
            {
                full[k] = std::conj(spec[k]); // Nyquist bin, real by symmetry
            }
        }
        transform(full, true, scale, cache);

        const std::size_t db = np::detail::flat_index(full_idx, dst.strides, dst.offset);
        for (std::size_t k = 0; k < n; ++k)
        {
            dst.data()[db + k * dst_stride] = full[k].real();
        }
        od.advance();
    }
#else
    np::detail::Odometer od(od_dims);
    while (!od.done())
    {
        const auto &oi = od.idx();
        std::size_t p = 0;
        for (std::size_t d = 0; d < nd; ++d)
        {
            if (d != ax)
            {
                full_idx[d] = oi[p++];
            }
        }
        const std::size_t base = np::detail::flat_index(full_idx, src.strides, src.offset);
        for (std::size_t k = 0; k < read; ++k)
        {
            spec[k] = src.data()[base + k * src_stride];
        }
        for (std::size_t k = read; k < hp; ++k)
        {
            spec[k] = Cplx{0.0, 0.0};
        }

        // Reconstruct the full Hermitian spectrum of length n.
        const std::size_t mid = n / 2;
        std::fill(full.begin(), full.end(), Cplx{0.0, 0.0});
        full[0] = spec[0];
        for (std::size_t k = 1; k <= mid; ++k)
        {
            full[k] = spec[k];
            if (n - k != k)
            {
                full[n - k] = std::conj(spec[k]);
            }
            else
            {
                full[k] = std::conj(spec[k]); // Nyquist bin, real by symmetry
            }
        }
        transform(full, true, scale, cache);

        const std::size_t db = np::detail::flat_index(full_idx, dst.strides, dst.offset);
        for (std::size_t k = 0; k < n; ++k)
        {
            dst.data()[db + k * dst_stride] = full[k].real();
        }
        od.advance();
    }
#endif
}

/** @brief Conjugate every element of a complex array in place. */
inline void conjugate_inplace(ndarray<Cplx> &a)
{
    for (std::size_t i = 0; i < a._numel(); ++i)
    {
        a.data()[i] = std::conj(a.data()[i]);
    }
}

} // namespace detail
} // namespace np::fft

#endif // NP_FFT_CORE_HPP