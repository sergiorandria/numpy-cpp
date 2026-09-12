/**
 * @file functional.hpp
 * @brief Functional programming helpers (np.apply_along_axis, vectorize, ...).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.functional.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_FUNCTIONAL_HPP
#define NP_FUNCTIONAL_HPP

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{

/**
 * @brief Apply a 1-D function along an axis (np.apply_along_axis).
 *
 * @tparam F Callable `(const ndarray<T>& slice) -> ndarray<R>` or scalar `R`.
 * @tparam T Input element type.
 * @param func1d Callable applied to each 1-D slice.
 * @param axis Axis along which to apply.
 * @param arr Input array.
 * @return Array with `axis` replaced by result dimension.
 *
 * Reference: numpy-reference/reference/generated/numpy.apply_along_axis.html
 */
NP_API template <typename F, typename T> auto apply_along_axis(F &&func1d, int axis, const ndarray<T> &arr)
{
    int nd = static_cast<int>(arr.ndim());
    if (nd == 0)
    {
        throw AxisError("apply_along_axis: 0-D array");
    }
    if (axis < 0)
    {
        axis += nd;
    }
    if (axis < 0 || axis >= nd)
    {
        throw AxisError("apply_along_axis: axis out of bounds");
    }

    auto probe_slice = [&]() -> std::vector<T> {
        std::vector<T> s;
        s.reserve(arr.shape[axis]);
        std::vector<std::size_t> idx(nd, 0);
        for (int i = 0; i < arr.shape[axis]; ++i)
        {
            idx[axis] = static_cast<std::size_t>(i);
            s.push_back(arr.get(idx));
        }
        return s;
    };

    // Build a sample 1-D array for probing return type/shape.
    ndarray<T> sample(std::vector<int>{arr.shape[axis]});
    {
        auto v = probe_slice();
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            sample.data()[sample._flat_logical(i)] = v[i];
        }
    }
    auto sample_res = func1d(sample);
    using R_Arr = std::decay_t<decltype(sample_res)>;
    using R = typename R_Arr::value_type;

    // Determine output shape: copy arr shape, replace axis with res dim.
    // If func returns scalar-like (size 1 but 0-D notion), we treat as scalar?
    // NumPy: if func returns scalar, axis is removed. We mimic: if res is 0-D or 1
    // element, keep scalar?
    bool returns_scalar = false;
    std::vector<int> res_shape;
    if constexpr (requires {
                      sample_res.shape;
                      sample_res.ndim();
                  })
    {
        if (sample_res.ndim() == 0 || (sample_res.ndim() == 1 && sample_res.size() == 1 && nd == 1))
        {
            // Check if probe returned 0-D (shape empty) – scalar case
            if (sample_res.ndim() == 0)
            {
                returns_scalar = true;
            }
            else
            {
                // Keep as 1-D size-1 -> treat as vector of length 1
                res_shape = std::vector<int>{1};
            }
        }
        else
        {
            res_shape = sample_res.shape;
        }
    }

    std::vector<int> out_shape;
    if (returns_scalar)
    {
        out_shape.reserve(nd - 1);
        for (int d = 0; d < nd; ++d)
        {
            if (d == axis)
            {
                continue;
            }
            out_shape.push_back(arr.shape[d]);
        }
        if (out_shape.empty())
        {
            out_shape.push_back(1);
        }
    }
    else
    {
        int res_nd = static_cast<int>(res_shape.size());
        if (res_nd == 1)
        {
            out_shape.reserve(nd);
            for (int d = 0; d < nd; ++d)
            {
                if (d == axis)
                {
                    out_shape.push_back(res_shape[0]);
                }
                else
                {
                    out_shape.push_back(arr.shape[d]);
                }
            }
        }
        else if (res_nd == 0)
        {
            out_shape.reserve(nd - 1);
            for (int d = 0; d < nd; ++d)
            {
                if (d != axis)
                {
                    out_shape.push_back(arr.shape[d]);
                }
            }
        }
        else
        {
            // General: insert res_shape in place of axis
            out_shape.reserve(nd - 1 + res_nd);
            for (int d = 0; d < nd; ++d)
            {
                if (d == axis)
                {
                    for (int v : res_shape)
                    {
                        out_shape.push_back(v);
                    }
                }
                else
                {
                    out_shape.push_back(arr.shape[d]);
                }
            }
        }
    }

    // Allocate output and iterate over all positions not on axis.
    using OutT = R;
    ndarray<OutT> out(out_shape);

    // Build outer odometer over dimensions excluding axis
    std::vector<int> outer_shape;
    for (int d = 0; d < nd; ++d)
    {
        if (d != axis)
        {
            outer_shape.push_back(arr.shape[d]);
        }
    }
    if (outer_shape.empty())
    {
        outer_shape.push_back(1);
    }

    // Map linear outer index to full index
    detail::Odometer outer_od(outer_shape);
    while (!outer_od.done())
    {
        const auto &oidx = outer_od.idx();
        // Build slice
        ndarray<T> slice(std::vector<int>{arr.shape[axis]});
        for (int i = 0; i < arr.shape[axis]; ++i)
        {
            std::vector<std::size_t> full(nd);
            for (int d = 0, o = 0; d < nd; ++d)
            {
                if (d == axis)
                {
                    full[d] = static_cast<std::size_t>(i);
                }
                else
                {
                    full[d] = oidx[o++];
                }
            }
            slice.data()[slice._flat_logical(static_cast<std::size_t>(i))] = arr.get(full);
        }
        auto res = func1d(slice);
        if (returns_scalar)
        {
            std::vector<std::size_t> out_idx;
            out_idx.reserve(out_shape.size());
            for (std::size_t k = 0; k < oidx.size(); ++k)
            {
                out_idx.push_back(oidx[k]);
            }
            R val{};
            if constexpr (requires { res.item(); })
            {
                val = res.item();
            }
            else
            {
                val = res.data()[0];
            }
            if (out_shape.size() == 1 && nd == 2 && out_shape[0] == arr.shape[axis])
            {
                // fallback shouldn't happen
            }
            out.set(out_idx, val);
        }
        else
        {
            for (std::size_t i = 0; i < res.size(); ++i)
            {
                std::vector<std::size_t> res_idx;
                if (res.ndim() == 1)
                {
                    res_idx.push_back(i);
                }
                else
                {
                    // decode multi-index for general res
                    std::size_t rem = i;
                    for (int d = static_cast<int>(res.shape.size()) - 1; d >= 0; --d)
                    {
                        res_idx.push_back(rem % static_cast<std::size_t>(res.shape[d]));
                        rem /= static_cast<std::size_t>(res.shape[d]);
                    }
                    std::reverse(res_idx.begin(), res_idx.end());
                }
                std::vector<std::size_t> out_idx;
                out_idx.reserve(out_shape.size());
                for (int d = 0, o = 0, r = 0; d < nd; ++d)
                {
                    if (d == axis)
                    {
                        for (std::size_t rr = 0; rr < res_idx.size(); ++rr)
                        {
                            out_idx.push_back(res_idx[rr]);
                        }
                    }
                    else
                    {
                        out_idx.push_back(oidx[o++]);
                    }
                }
                out.set(out_idx, res.data()[res._flat_logical(i)]);
            }
        }
        outer_od.advance();
        if (outer_shape.size() == 1 && outer_shape[0] == 1 && nd == 1)
        {
            break;
        }
    }
    return out;
}

/**
 * @brief Apply func repeatedly over axes (np.apply_over_axes).
 *
 * Reference: numpy-reference/reference/generated/numpy.apply_over_axes.html
 */
NP_API template <typename F, typename T>
auto apply_over_axes(F &&func, const ndarray<T> &a, const std::vector<int> &axes) -> ndarray<T>
{
    ndarray<T> res = a;
    for (int ax : axes)
    {
        int nd = static_cast<int>(res.ndim());
        int nax = ax < 0 ? ax + nd : ax;
        if (nax < 0 || nax >= nd)
        {
            throw AxisError("apply_over_axes: axis out of bounds");
        }
        res = func(res, nax);
    }
    return res;
}

// ── vectorize ─────────────────────────────────────────────────────

namespace detail
{
template <typename F, typename... Args> using vec_invoke = std::invoke_result_t<F, Args...>;
}

/**
 * @brief Vectorized wrapper (np.vectorize).
 *
 * Wraps a scalar callable so it operates element-wise with broadcasting.
 *
 * Reference: numpy-reference/reference/generated/numpy.vectorize.html
 */
template <typename F> class vectorize
{
  public:
    explicit vectorize(F f) : func_(std::move(f))
    {
    }

    template <typename T> auto operator()(const ndarray<T> &a) const
    {
        using R = std::invoke_result_t<F, T>;
        ndarray<R> out(a.shape);
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            out.data()[i] = func_(a.data()[a._flat_logical(i)]);
        }
        return out;
    }

    template <typename T, typename U> auto operator()(const ndarray<T> &a, const ndarray<U> &b) const
    {
        using R = std::invoke_result_t<F, T, U>;
        std::vector<int> out_shape = detail::broadcast_shapes(a.shape, b.shape);
        ndarray<R> out(out_shape);
        detail::Odometer od(out_shape);
        while (!od.done())
        {
            const auto &idx = od.idx();
            auto av = a.get(detail::broadcast_index(a.shape, out_shape, idx));
            auto bv = b.get(detail::broadcast_index(b.shape, out_shape, idx));
            out.set(idx, func_(av, bv));
            od.advance();
        }
        return out;
    }

    template <typename T> auto operator()(T scalar) const
    {
        using R = std::invoke_result_t<F, T>;
        ndarray<R> out(std::vector<int>{1});
        out.data()[0] = func_(scalar);
        return out;
    }

    F func_;
};

template <typename F> vectorize(F) -> vectorize<F>;

/**
 * @brief Helper to create vectorize object (np.vectorize).
 */
NP_API template <typename F> auto make_vectorize(F &&f)
{
    return vectorize<std::decay_t<F>>(std::forward<F>(f));
}

/**
 * @brief From Python function to ufunc (np.frompyfunc).
 *
 * Simplified: identical to vectorize but enforces `nin`/`nout`.
 *
 * Reference: numpy-reference/reference/generated/numpy.frompyfunc.html
 */
NP_API template <typename F> auto frompyfunc(F &&func, std::size_t nin, std::size_t nout)
{
    if (nout != 1)
    {
        throw std::invalid_argument("frompyfunc: only nout==1 supported in this port");
    }
    (void)nin;
    return vectorize<std::decay_t<F>>(std::forward<F>(func));
}

// ── piecewise ─────────────────────────────────────────────────────

/**
 * @brief Evaluate piecewise function (np.piecewise).
 *
 * Reference: numpy-reference/reference/generated/numpy.piecewise.html
 *
 * @param x Input array.
 * @param condlist Vector of boolean arrays (same shape as x).
 * @param funclist Vector of arrays/scalars or callables `T(T)`.
 *
 * Two overloads:
 *  - funclist as `vector<ndarray<T>>` (each broadcast to x.shape)
 *  - funclist as `vector<function<T(T)>>`
 */
NP_API template <typename T>
NP_NODISCARD auto piecewise(const ndarray<T> &x, const std::vector<ndarray<bool>> &condlist,
                            const std::vector<ndarray<T>> &funclist) -> ndarray<T>
{
    if (condlist.size() != funclist.size() && condlist.size() + 1 != funclist.size())
    {
        throw std::invalid_argument("piecewise: condlist/funclist size mismatch");
    }
    ndarray<T> out(x.shape);
    detail::Odometer od(x.shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        bool assigned = false;
        for (std::size_t k = 0; k < condlist.size(); ++k)
        {
            if (condlist[k].get(idx))
            {
                // broadcast funclist[k] to x
                const auto &f = funclist[k];
                T val;
                if (f.shape == x.shape)
                {
                    val = f.get(idx);
                }
                else if (f.size() == 1)
                {
                    val = f.data()[0];
                }
                else
                {
                    val = f.get(detail::broadcast_index(f.shape, x.shape, idx));
                }
                out.set(idx, val);
                assigned = true;
                break;
            }
        }
        if (!assigned)
        {
            if (funclist.size() > condlist.size())
            {
                const auto &f = funclist.back();
                T val;
                if (f.shape == x.shape)
                {
                    val = f.get(idx);
                }
                else if (f.size() == 1)
                {
                    val = f.data()[0];
                }
                else
                {
                    val = f.get(detail::broadcast_index(f.shape, x.shape, idx));
                }
                out.set(idx, val);
            }
            else
            {
                out.set(idx, T{0});
            }
        }
        od.advance();
    }
    return out;
}

NP_API template <typename T, typename F>
NP_NODISCARD auto piecewise(const ndarray<T> &x, const std::vector<ndarray<bool>> &condlist,
                            const std::vector<F> &funclist) -> ndarray<T>
    requires(std::is_invocable_r_v<T, F, T>)
{
    std::vector<ndarray<bool>> cl = condlist;
    ndarray<T> out(x.shape);
    detail::Odometer od(x.shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        T xv = x.get(idx);
        bool assigned = false;
        for (std::size_t k = 0; k < cl.size(); ++k)
        {
            if (cl[k].get(idx))
            {
                out.set(idx, funclist[k](xv));
                assigned = true;
                break;
            }
        }
        if (!assigned)
        {
            if (funclist.size() > cl.size())
            {
                out.set(idx, funclist.back()(xv));
            }
            else
            {
                out.set(idx, T{0});
            }
        }
        od.advance();
    }
    return out;
}

} // namespace np

#endif // NP_FUNCTIONAL_HPP
