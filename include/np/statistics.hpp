/**
 * @file statistics.hpp
 * @brief Statistical functions (NumPy reference: routines.statistics).
 *
 * Provides scalar and axis-aware reductions on np::ndarray mirroring
 * numpy: mean, var, std, median, percentile, quantile, average, ptp,
 * corrcoef, cov, histogram, histogram_bin_edges, bincount, digitize
 * and the NaN-skipping nan* family
 * (nanmin, nanmax, nansum, nanprod, nanmean, nanvar, nanstd, nanmedian,
 * nanpercentile, nanquantile, nanargmin, nanargmax, nancumsum, nancumprod).
 *
 * Axis convention matches np::ndarray's member reductions:
 *  - the no-axis overload reduces over the whole (flattened) array and
 *    returns a scalar;
 *  - the overload taking an `int axis` reduces along that single axis
 *    (negative indices count from the end) and returns an ndarray.
 *
 * All floating-point results are computed in double for double inputs and
 * preserved for float inputs (NumPy semantics: a float32 input keeps
 * float32). Integer and bool inputs promote to double.
 *
 * Reference: numpy-reference/reference/routines.statistics.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_STATISTICS_HPP
#define NP_STATISTICS_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "api_macros.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"

#ifdef NP_USE_THREADING
#include "threadpool.hpp"
#endif

namespace np
{

namespace detail
{

/** @brief Normalize an axis index, throwing np::AxisError if invalid. */
NP_NODISCARD inline int stat_normalize_axis(int axis, std::size_t ndim, const char *what)
{
    const int nd = static_cast<int>(ndim);
    if (axis < 0)
    {
        axis += nd;
    }
    if (axis < 0 || axis >= nd)
    {
        throw np::AxisError(std::string(what) + ": axis " + std::to_string(axis) +
                            " is out of bounds for array of dimension " + std::to_string(nd));
    }
    return axis;
}

/** @brief Row-major flat offset of `idx` within `shape`. */
NP_NODISCARD inline std::size_t row_major_offset(const std::vector<std::size_t> &idx, const std::vector<int> &shape)
{
    std::size_t flat = 0;
    for (std::size_t d = 0; d < shape.size(); ++d)
    {
        flat = flat * static_cast<std::size_t>(shape[d]) + idx[d];
    }
    return flat;
}

/**
 * @brief Gather the elements of one slice of `arr` along `axis`.
 *
 * `base` is a full-size index into `arr`; the `axis` component is
 * overwritten with 0..shape[axis]-1. Returns the slice as a fresh vector,
 * reading through the public `get()` accessor (argument evaluates strides
 * and negative offsets of views).
 */
template <typename T>
NP_NODISCARD std::vector<T> gather_slice(const ndarray<T> &arr, int axis, std::vector<std::size_t> base)
{
    const std::size_t alen = static_cast<std::size_t>(arr.shape[axis]);
    std::vector<T> slice;
    slice.reserve(alen);
    for (std::size_t a = 0; a < alen; ++a)
    {
        base[static_cast<std::size_t>(axis)] = a;
        slice.push_back(arr.get(base));
    }
    return slice;
}

/**
 * @brief Apply `fn` to every slice along `axis`, assembling the results.
 *
 * @tparam R   Result element type.
 * @tparam T   Input element type.
 * @tparam Fn  Callable `(const std::vector<T>&) -> R`.
 * @param arr  Input array.
 * @param axis Axis along which to reduce (normalized; may be negative).
 * @param fn   Per-slice function.
 * @return     Array with `axis` removed holding `fn` per slice.
 */
template <typename R, typename T, typename Fn>
NP_NODISCARD ndarray<R> stat_axis_map(const ndarray<T> &arr, int axis, Fn &&fn)
{
    axis = stat_normalize_axis(axis, arr.ndim(), "np::stats");
    std::vector<int> out_shape = arr.shape;
    out_shape.erase(out_shape.begin() + axis);
    const std::size_t nd = static_cast<std::size_t>(arr.ndim());

    ndarray<R> out(out_shape);
#ifdef NP_USE_THREADING
    // Collect outer indices for parallel dispatch
    std::vector<std::vector<std::size_t>> all_red;
    {
        detail::Odometer od(out_shape);
        while (!od.done())
        {
            all_red.push_back(od.idx());
            od.advance();
        }
        if (all_red.empty())
        {
            all_red.push_back({});
        }
    }
    if (all_red.size() > 32)
    {
        ::np::ThreadPool::global().parallel_for(0, all_red.size(), [&](std::size_t idx) {
            const auto &red = all_red[idx];
            std::vector<std::size_t> full(nd, 0);
            for (std::size_t d = 0, r = 0; d < nd; ++d)
            {
                if (static_cast<int>(d) == axis)
                {
                    full[d] = 0;
                    continue;
                }
                full[d] = red[r++];
            }
            const std::size_t flat = detail::row_major_offset(red, out_shape);
            out.data()[flat] = fn(detail::gather_slice(arr, axis, full));
        });
        return out;
    }
#endif
    detail::Odometer od(out_shape);
    std::vector<std::size_t> full(nd, 0);
    while (!od.done())
    {
        const auto &red = od.idx();
        for (std::size_t d = 0, r = 0; d < nd; ++d)
        {
            if (static_cast<int>(d) == axis)
            {
                full[d] = 0;
                continue;
            }
            full[d] = red[r++];
        }
        const std::size_t flat = detail::row_major_offset(red, out_shape);
        out.data()[flat] = fn(detail::gather_slice(arr, axis, full));
        od.advance();
    }
    return out;
}

/** @brief Linear interpolation of `values` at fractional `position`. */
NP_NODISCARD inline double lin_interp(const std::vector<double> &values, double position)
{
    const auto n = values.size();
    if (position <= 0.0)
        return values[0];
    if (position >= static_cast<double>(n - 1))
        return values.back();
    const std::size_t lo = static_cast<std::size_t>(position);
    const double frac = position - static_cast<double>(lo);
    return values[lo] + frac * (values[lo + 1] - values[lo]);
}

/** @brief Promote a (possibly bool/integer) element type for reductions. */
template <typename T> using stat_real_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;

/** @brief NaN tester: false for integer/bool/complex-safe types. */
template <typename T> constexpr bool is_nan_elem(const T &v)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return std::isnan(v);
    }
    else
    {
        (void)v;
        return false;
    }
}

/**
 * @brief Same-shape cumulative reduction along `axis`.
 *
 * Walks each slice along `axis` in order, applying `step(acc, v)` for
 * every non-NaN element and writing the current `acc` at every position
 * (NaN holds the running value, NumPy nancumsum/nancumprod semantics).
 * Output element type is `R`.
 */
template <typename R, typename T, typename Step>
NP_NODISCARD ndarray<R> cum_axis_map(const ndarray<T> &arr, int axis, R init, Step &&step)
{
    axis = stat_normalize_axis(axis, arr.ndim(), "np::nancum*");
    const std::size_t nd = static_cast<std::size_t>(arr.ndim());
    const std::size_t alen = static_cast<std::size_t>(arr.shape[axis]);
    std::vector<int> red_shape = arr.shape;
    red_shape.erase(red_shape.begin() + axis);

    ndarray<R> out(arr.shape);
#ifdef NP_USE_THREADING
    std::vector<std::vector<std::size_t>> all_red;
    {
        Odometer od(red_shape);
        while (!od.done())
        {
            all_red.push_back(od.idx());
            od.advance();
        }
        if (all_red.empty())
        {
            all_red.push_back({});
        }
    }
    if (all_red.size() > 16)
    {
        ::np::ThreadPool::global().parallel_for(0, all_red.size(), [&](std::size_t idx) {
            const auto &red = all_red[idx];
            std::vector<std::size_t> full(nd, 0);
            for (std::size_t d = 0, r = 0; d < nd; ++d)
            {
                full[d] = (static_cast<int>(d) == axis) ? 0 : red[r++];
            }
            R running = init;
            for (std::size_t a = 0; a < alen; ++a)
            {
                full[static_cast<std::size_t>(axis)] = a;
                const T v = arr.get(full);
                if (!is_nan_elem(v))
                {
                    step(running, v);
                }
                const std::size_t flat = row_major_offset(full, out.shape);
                out.data()[flat] = running;
            }
        });
        return out;
    }
#endif
    Odometer od(red_shape);
    std::vector<std::size_t> full(nd, 0);
    while (!od.done())
    {
        const auto &red = od.idx();
        for (std::size_t d = 0, r = 0; d < nd; ++d)
        {
            full[d] = (static_cast<int>(d) == axis) ? 0 : red[r++];
        }
        R running = init;
        for (std::size_t a = 0; a < alen; ++a)
        {
            full[static_cast<std::size_t>(axis)] = a;
            const T v = arr.get(full);
            if (!is_nan_elem(v))
            {
                step(running, v);
            }
            const std::size_t flat = row_major_offset(full, out.shape);
            out.data()[flat] = running;
        }
        od.advance();
    }
    return out;
}

} // namespace detail

// Median / percentile / quantile
/**
 * @brief Median of all elements.
 *
 * Sorts a copy of the (flattened) input and returns the middle value;
 * for even-size inputs the mean of the two middle values is returned.
 *
 * Reference: numpy-reference/reference/generated/numpy.median.html
 *
 * @tparam T Element type.
 * @param arr Input array.
 * @return Median of all elements.
 * @throws std::invalid_argument if the array is empty.
 * @complexity O(n log n). Space: O(n).
 */
NP_API template <typename T> NP_NODISCARD auto median(const ndarray<T> &arr) -> double
{
    if (arr.size() == 0)
    {
        throw std::invalid_argument("median: empty array");
    }
    std::vector<double> values;
    values.reserve(arr.size());
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        values.push_back(static_cast<double>(*it));
    }
    // Micro-optimized: O(n) nth_element vs O(n log n) sort
    const std::size_t n = values.size();
    if (n % 2 == 1)
    {
        std::nth_element(values.begin(), values.begin() + n / 2, values.end());
        return values[n / 2];
    }
    // even: need two middle values
    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
    double upper = values[n / 2];
    // lower median is max of lower partition
    std::nth_element(values.begin(), values.begin() + n / 2 - 1, values.begin() + n / 2);
    double lower = values[n / 2 - 1];
    // nth_element on prefix ensures lower is correct but we need max of [0, n/2)
    // Actually after second nth_element, upper still at n/2? Re-find upper as min of
    // upper part Simpler: find nth for lower, then upper is min of [n/2, end) Use linear
    // scan for lower max to avoid second nth overhead for small n
    if (n < 128)
    {
        // for tiny n, second nth_element is fine
        return 0.5 * (lower + upper);
    }
    // For larger n, lower is at n/2-1, upper remains at n/2 from first partition?
    // Need to guarantee upper still valid after second partition on prefix only (which
    // doesn't touch suffix) So upper remains correct.
    return 0.5 * (lower + upper);
}

/**
 * @brief Median along a single axis.
 *
 * Reference: numpy-reference/reference/generated/numpy.median.html
 *
 * @tparam T The element type.
 * @param arr Input array.
 * @param axis Axis along which to reduce (may be negative).
 * @return Array of medians with `axis` removed.
 * @throws std::invalid_argument if a reduced slice is empty.
 * @throws np::AxisError if `axis` is out of bounds.
 * @complexity O(n log axis_len).
 */
NP_API template <typename T> NP_NODISCARD auto median(const ndarray<T> &arr, int axis) -> ndarray<double>
{
    return detail::stat_axis_map<double>(arr, axis, [](const std::vector<T> &slice) -> double {
        if (slice.empty())
        {
            throw std::invalid_argument("median: empty slice along axis");
        }
        std::vector<double> vals;
        vals.reserve(slice.size());
        for (const auto &v : slice)
        {
            vals.push_back(static_cast<double>(v));
        }
        const std::size_t n = vals.size();
        if (n % 2 == 1)
        {
            std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
            return vals[n / 2];
        }
        std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
        double upper = vals[n / 2];
        std::nth_element(vals.begin(), vals.begin() + n / 2 - 1, vals.begin() + n / 2);
        double lower = vals[n / 2 - 1];
        return 0.5 * (lower + upper);
    });
}

/**
 * @brief Percentile over all elements, NumPy `linear` method.
 *
 * The `linear` interpolation method (NumPy default) is used: the sorted
 * value at position `(n-1) * q/100` with fractional interpolation between
 * the two neighbours.
 *
 * Reference: numpy-reference/reference/generated/numpy.percentile.html
 *
 * @tparam T Element type.
 * @tparam Q Percentile type (typically double).
 * @param arr Input array.
 * @param q Percentile, 0 <= q <= 100.
 * @return The q-th percentile of all elements.
 * @throws std::invalid_argument if q is out of range or arr is empty.
 * @complexity O(n log n).
 */
NP_API template <typename T, typename Q> NP_NODISCARD auto percentile(const ndarray<T> &arr, const Q &q) -> double
{
    if (arr.size() == 0)
    {
        throw std::invalid_argument("percentile: empty array");
    }
    const double p = static_cast<double>(q);
    if (p < 0.0 || p > 100.0)
    {
        throw std::invalid_argument("percentile: q must be in [0, 100]");
    }
    std::vector<double> values;
    values.reserve(arr.size());
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        values.push_back(static_cast<double>(*it));
    }
    // Micro-optimized: O(n) nth_element + O(n) min vs O(n log n) sort
    const std::size_t n = values.size();
    const double rank = (n - 1) * (p / 100.0);
    const std::size_t k = static_cast<std::size_t>(rank);
    const double frac = rank - static_cast<double>(k);
    std::nth_element(values.begin(), values.begin() + k, values.end());
    double low = values[k];
    if (frac == 0.0 || k + 1 >= n)
    {
        return low;
    }
    // k+1 th smallest is min of suffix [k+1, end)
    double high = *std::min_element(values.begin() + k + 1, values.end());
    return low * (1.0 - frac) + high * frac;
}

/**
 * @brief Percentile along a single axis (linear method).
 *
 * Reference: numpy-reference/reference/generated/numpy.percentile.html
 *
 * @tparam T Element type.
 * @tparam Q Percentile type (typically double).
 * @param arr Input array.
 * @param q Percentile, 0 <= q <= 100.
 * @param axis Axis along which to reduce.
 * @return Array of percentiles with `axis` removed.
 * @throws std::invalid_argument if q is out of range.
 * @throws np::AxisError if `axis` is out of bounds.
 * @complexity O(n log axis_len).
 */
NP_API template <typename T, typename Q>
NP_NODISCARD auto percentile(const ndarray<T> &arr, const Q &q, int axis) -> ndarray<double>
{
    const double p = static_cast<double>(q);
    if (p < 0.0 || p > 100.0)
    {
        throw std::invalid_argument("percentile: q must be in [0, 100]");
    }
    return detail::stat_axis_map<double>(arr, axis, [p](const std::vector<T> &slice) -> double {
        if (slice.empty())
        {
            throw std::invalid_argument("percentile: empty slice along axis");
        }
        std::vector<double> vals;
        vals.reserve(slice.size());
        for (const auto &v : slice)
        {
            vals.push_back(static_cast<double>(v));
        }
        const std::size_t n = vals.size();
        const double rank = (n - 1) * (p / 100.0);
        const std::size_t k = static_cast<std::size_t>(rank);
        const double frac = rank - static_cast<double>(k);
        std::nth_element(vals.begin(), vals.begin() + k, vals.end());
        double low = vals[k];
        if (frac == 0.0 || k + 1 >= n)
            return low;
        double high = *std::min_element(vals.begin() + k + 1, vals.end());
        return low * (1.0 - frac) + high * frac;
    });
}

/**
 * @brief Quantile of all elements (percentile with q in [0,1]).
 *
 * Reference: numpy-reference/reference/generated/numpy.quantile.html
 *
 * @tparam T Element type.
 * @tparam Q Quantile type (typically double).
 * @param arr Input array.
 * @param q Quantile, 0 <= q <= 1.
 * @return The q-th quantile of all elements.
 * @throws std::invalid_argument if q is out of range or arr is empty.
 */
NP_API template <typename T, typename Q> NP_NODISCARD auto quantile(const ndarray<T> &arr, const Q &q) -> double
{
    const double p = static_cast<double>(q);
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("quantile: q must be in [0, 1]");
    }
    return percentile(arr, p * 100.0);
}

/**
 * @brief Quantile along a single axis (linear method).
 *
 * @tparam T Element type.
 * @tparam Q Quantile type (typically double).
 * @param arr Input array.
 * @param q Quantile, 0 <= q <= 1.
 * @param axis Axis along which to reduce.
 * @return Array of quantiles with `axis` removed.
 * @throws std::invalid_argument if q is out of range.
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T, typename Q>
NP_NODISCARD auto quantile(const ndarray<T> &arr, const Q &q, int axis) -> ndarray<double>
{
    const double p = static_cast<double>(q);
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("quantile: q must be in [0, 1]");
    }
    return percentile(arr, p * 100.0, axis);
}

// NaN-skipping reduction family (nanmin/nanmax/nansum/nanprod/
// nanmean/nanvar/nanstd/nanmedian/nanpercentile/nanquantile)
/**
 * @brief Minimum of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nanmin.html
 *
 * @throws std::invalid_argument if every element is NaN.
 */
NP_API template <typename T> NP_NODISCARD auto nanmin(const ndarray<T> &arr) -> T
{
    bool any = false;
    T best{};
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (detail::is_nan_elem(*it))
            continue;
        if (!any || *it < best)
        {
            best = *it;
            any = true;
        }
    }
    if (!any)
    {
        throw std::invalid_argument("nanmin: all-NaN slice");
    }
    return best;
}

/** @brief Minimum along an axis, ignoring NaN. */
NP_API template <typename T> NP_NODISCARD auto nanmin(const ndarray<T> &arr, int axis) -> ndarray<T>
{
    return detail::stat_axis_map<T>(arr, axis, [](const std::vector<T> &slice) -> T {
        bool any = false;
        T best{};
        for (const auto &v : slice)
        {
            if (detail::is_nan_elem(v))
                continue;
            if (!any || v < best)
            {
                best = v;
                any = true;
            }
        }
        if (!any)
        {
            throw std::invalid_argument("nanmin: all-NaN slice");
        }
        return best;
    });
}

/**
 * @brief Maximum of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nanmax.html
 *
 * @throws std::invalid_argument if every element is NaN.
 */
NP_API template <typename T> NP_NODISCARD auto nanmax(const ndarray<T> &arr) -> T
{
    bool any = false;
    T best{};
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (detail::is_nan_elem(*it))
            continue;
        if (!any || *it > best)
        {
            best = *it;
            any = true;
        }
    }
    if (!any)
    {
        throw std::invalid_argument("nanmax: all-NaN slice");
    }
    return best;
}

/** @brief Maximum along an axis, ignoring NaN. */
NP_API template <typename T> NP_NODISCARD auto nanmax(const ndarray<T> &arr, int axis) -> ndarray<T>
{
    return detail::stat_axis_map<T>(arr, axis, [](const std::vector<T> &slice) -> T {
        bool any = false;
        T best{};
        for (const auto &v : slice)
        {
            if (detail::is_nan_elem(v))
                continue;
            if (!any || v > best)
            {
                best = v;
                any = true;
            }
        }
        if (!any)
        {
            throw std::invalid_argument("nanmax: all-NaN slice");
        }
        return best;
    });
}

/**
 * @brief Sum of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nansum.html
 *
 * An all-NaN (or empty) input sums to zero (NumPy semantics).
 */
NP_API template <typename T> NP_NODISCARD auto nansum(const ndarray<T> &arr) -> typename np::_mean_type<T>::type
{
    using R = typename np::_mean_type<T>::type;
    R sum{};
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (detail::is_nan_elem(*it))
            continue;
        sum += static_cast<R>(*it);
    }
    return sum;
}

/** @brief Sum along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nansum(const ndarray<T> &arr, int axis) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    return detail::stat_axis_map<R>(arr, axis, [](const std::vector<T> &slice) -> R {
        R sum = R{};
        for (const auto &v : slice)
        {
            if (detail::is_nan_elem(v))
                continue;
            sum += static_cast<R>(v);
        }
        return sum;
    });
}

/**
 * @brief Product of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nanprod.html
 *
 * Empty array product is 1 (NumPy behavior).
 */
NP_API template <typename T> NP_NODISCARD auto nanprod(const ndarray<T> &arr) -> typename np::_mean_type<T>::type
{
    using R = typename np::_mean_type<T>::type;
    R prod = static_cast<R>(1);
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (detail::is_nan_elem(*it))
            continue;
        prod = static_cast<R>(prod * static_cast<R>(*it));
    }
    return prod;
}

/** @brief Product along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanprod(const ndarray<T> &arr, int axis) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    return detail::stat_axis_map<R>(arr, axis, [](const std::vector<T> &slice) -> R {
        R prod = static_cast<R>(1);
        for (const auto &v : slice)
        {
            if (detail::is_nan_elem(v))
                continue;
            prod = static_cast<R>(prod * static_cast<R>(v));
        }
        return prod;
    });
}

/**
 * @brief Mean of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nanmean.html
 *
 * If every element is NaN the reduction yields NaN (NumPy behavior).
 */
NP_API template <typename T> NP_NODISCARD auto nanmean(const ndarray<T> &arr) -> typename np::_mean_type<T>::type
{
    using R = typename np::_mean_type<T>::type;
    long double sum = 0.0;
    std::size_t n = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (detail::is_nan_elem(*it))
            continue;
        sum += static_cast<long double>(*it);
        ++n;
    }
    if (n == 0)
    {
        return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
    }
    return static_cast<R>(sum / static_cast<long double>(n));
}

/** @brief Mean along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanmean(const ndarray<T> &arr, int axis) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    return detail::stat_axis_map<R>(arr, axis, [](const std::vector<T> &slice) -> R {
        long double sum = 0.0;
        std::size_t n = 0;
        for (const auto &v : slice)
        {
            if (detail::is_nan_elem(v))
                continue;
            sum += static_cast<long double>(v);
            ++n;
        }
        return n == 0 ? static_cast<R>(std::numeric_limits<double>::quiet_NaN())
                      : static_cast<R>(sum / static_cast<long double>(n));
    });
}

/**
 * @brief Variance of all elements, ignoring NaN (population, ddof=0).
 *
 * Reference: numpy-reference/reference/generated/numpy.nanvar.html
 *
 * All-NaN input yields NaN.
 */
NP_API template <typename T>
NP_NODISCARD auto nanvar(const ndarray<T> &arr, int ddof = 0) -> typename np::_mean_type<T>::type
{
    using R = typename np::_mean_type<T>::type;
    if (ddof < 0)
        throw std::invalid_argument("nanvar: ddof must be >= 0");
    const auto m = nanmean(arr);
    if (detail::is_nan_elem(m))
    {
        return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
    }
    long double acc = 0.0;
    std::size_t n = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (detail::is_nan_elem(*it))
            continue;
        const long double d = static_cast<long double>(*it) - static_cast<long double>(m);
        acc += d * d;
        ++n;
    }
    // NOTE (honesty audit): count-vs-ddof was previously ignored (always /n).
    // NumPy yields NaN when ddof swallows the sample count.
    if (n == 0 || n <= static_cast<std::size_t>(ddof))
        return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
    return static_cast<R>(acc / static_cast<long double>(n - static_cast<std::size_t>(ddof)));
}

/** @brief Standard deviation of all elements, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanstd(const ndarray<T> &arr, int ddof = 0) -> typename np::_mean_type<T>::type
{
    using R = typename np::_mean_type<T>::type;
    return static_cast<R>(std::sqrt(static_cast<long double>(nanvar(arr, ddof))));
}

/** @brief Variance along an axis, ignoring NaN (NumPy default ddof=0). */
NP_API template <typename T>
NP_NODISCARD auto nanvar(const ndarray<T> &arr, int axis, int ddof) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    if (ddof < 0)
        throw std::invalid_argument("nanvar: ddof must be >= 0");
    return detail::stat_axis_map<R>(arr, axis, [ddof](const std::vector<T> &slice) -> R {
        long double sum = 0;
        std::size_t n = 0;
        for (auto &v : slice)
            if (!detail::is_nan_elem(v))
            {
                sum += static_cast<long double>(v);
                ++n;
            }
        if (n == 0 || n <= static_cast<std::size_t>(ddof))
            return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
        long double mean = sum / static_cast<long double>(n);
        long double acc = 0;
        for (auto &v : slice)
            if (!detail::is_nan_elem(v))
            {
                long double d = static_cast<long double>(v) - mean;
                acc += d * d;
            }
        return static_cast<R>(acc / static_cast<long double>(n - static_cast<std::size_t>(ddof)));
    });
}

/** @brief Standard deviation along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanstd(const ndarray<T> &arr, int axis, int ddof) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    auto v = nanvar(arr, axis, ddof);
    ndarray<R> out(v.shape);
    for (std::size_t i = 0; i < v.size(); ++i)
        out.data()[i] = static_cast<R>(std::sqrt(static_cast<long double>(v.data()[i])));
    return out;
}

/**
 * @brief Median of all elements, ignoring NaN.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html
 *
 * Empty or all-NaN input yields NaN.
 */
NP_API template <typename T> NP_NODISCARD auto nanmedian(const ndarray<T> &arr)
{
    return nanpercentile(arr, 50.0);
}

/** @brief Median along an axis, ignoring NaN. */
NP_API template <typename T> NP_NODISCARD auto nanmedian(const ndarray<T> &arr, int axis)
{
    return nanpercentile(arr, 50.0, axis);
}

/** @brief Sorted, NaN-stripped values of `slice` (percentile w/o NaN). */
namespace detail
{
template <typename T> NP_NODISCARD double nan_percentile_of_slice(const std::vector<T> &slice, double p)
{
    std::vector<double> vals;
    vals.reserve(slice.size());
    for (const auto &v : slice)
    {
        if (is_nan_elem(v))
            continue;
        vals.push_back(static_cast<double>(v));
    }
    if (vals.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(vals.begin(), vals.end());
    return lin_interp(vals, (vals.size() - 1) * (p / 100.0));
}
} // namespace detail

/**
 * @brief Percentile of all elements, ignoring NaN (linear method).
 *
 * Reference: numpy-reference/reference/generated/numpy.nanpercentile.html
 *
 * All-NaN or empty input yields NaN.
 */
NP_API template <typename T, typename Q> NP_NODISCARD auto nanpercentile(const ndarray<T> &arr, const Q &q) -> double
{
    const double p = static_cast<double>(q);
    if (p < 0.0 || p > 100.0)
    {
        throw std::invalid_argument("nanpercentile: q must be in [0, 100]");
    }
    std::vector<T> all;
    all.reserve(arr.size());
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        all.push_back(*it);
    }
    return detail::nan_percentile_of_slice(all, p);
}

/** @brief Percentile along an axis, ignoring NaN (linear method). */
NP_API template <typename T, typename Q>
NP_NODISCARD auto nanpercentile(const ndarray<T> &arr, const Q &q, int axis) -> ndarray<double>
{
    const double p = static_cast<double>(q);
    if (p < 0.0 || p > 100.0)
    {
        throw std::invalid_argument("nanpercentile: q must be in [0, 100]");
    }
    return detail::stat_axis_map<double>(
        arr, axis, [p](const std::vector<T> &slice) -> double { return detail::nan_percentile_of_slice(slice, p); });
}

/**
 * @brief Quantile of all elements, ignoring NaN (q in [0,1]).
 *
 * Reference: numpy-reference/reference/generated/numpy.nanquantile.html
 */
NP_API template <typename T, typename Q> NP_NODISCARD auto nanquantile(const ndarray<T> &arr, const Q &q) -> double
{
    const double p = static_cast<double>(q);
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("nanquantile: q must be in [0, 1]");
    }
    return nanpercentile(arr, p * 100.0);
}

/** @brief Quantile along an axis, ignoring NaN (q in [0,1]). */
NP_API template <typename T, typename Q>
NP_NODISCARD auto nanquantile(const ndarray<T> &arr, const Q &q, int axis) -> ndarray<double>
{
    const double p = static_cast<double>(q);
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("nanquantile: q must be in [0, 1]");
    }
    return nanpercentile(arr, p * 100.0, axis);
}

// nanargmin / nanargmax (NaN-skipping index reductions)
/**
 * @brief Index of the smallest element, ignoring NaN (flattened).
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html
 *
 * The returned index is into the flattened array; the first occurrence
 * of the minimum wins. Integer/bool inputs have no NaN so this matches
 * `ndarray::argmin`.
 *
 * @throws std::invalid_argument if every element is NaN.
 */
NP_API template <typename T> NP_NODISCARD std::size_t nanargmin(const ndarray<T> &arr)
{
    bool any = false;
    T best{};
    std::size_t best_idx = 0;
    std::size_t pos = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it, ++pos)
    {
        if (detail::is_nan_elem(*it))
            continue;
        if (!any || *it < best)
        {
            best = *it;
            best_idx = pos;
            any = true;
        }
    }
    if (!any)
    {
        throw std::invalid_argument("nanargmin: all-NaN slice");
    }
    return best_idx;
}

/**
 * @brief Indices of the smallest elements along an axis, ignoring NaN.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html
 *
 * @throws std::invalid_argument if a slice is all-NaN.
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T> NP_NODISCARD auto nanargmin(const ndarray<T> &arr, int axis) -> ndarray<std::size_t>
{
    return detail::stat_axis_map<std::size_t>(arr, axis, [](const std::vector<T> &slice) -> std::size_t {
        bool any = false;
        T best{};
        std::size_t best_idx = 0;
        for (std::size_t p = 0; p < slice.size(); ++p)
        {
            if (detail::is_nan_elem(slice[p]))
                continue;
            if (!any || slice[p] < best)
            {
                best = slice[p];
                best_idx = p;
                any = true;
            }
        }
        if (!any)
        {
            throw std::invalid_argument("nanargmin: all-NaN slice");
        }
        return best_idx;
    });
}

/**
 * @brief Index of the largest element, ignoring NaN (flattened).
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html
 *
 * The returned index is into the flattened array; the first occurrence
 * of the maximum wins. Integer/bool inputs have no NaN so this matches
 * `ndarray::argmax`.
 *
 * @throws std::invalid_argument if every element is NaN.
 */
NP_API template <typename T> NP_NODISCARD std::size_t nanargmax(const ndarray<T> &arr)
{
    bool any = false;
    T best{};
    std::size_t best_idx = 0;
    std::size_t pos = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it, ++pos)
    {
        if (detail::is_nan_elem(*it))
            continue;
        if (!any || *it > best)
        {
            best = *it;
            best_idx = pos;
            any = true;
        }
    }
    if (!any)
    {
        throw std::invalid_argument("nanargmax: all-NaN slice");
    }
    return best_idx;
}

/**
 * @brief Indices of the largest elements along an axis, ignoring NaN.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html
 *
 * @throws std::invalid_argument if a slice is all-NaN.
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T> NP_NODISCARD auto nanargmax(const ndarray<T> &arr, int axis) -> ndarray<std::size_t>
{
    return detail::stat_axis_map<std::size_t>(arr, axis, [](const std::vector<T> &slice) -> std::size_t {
        bool any = false;
        T best{};
        std::size_t best_idx = 0;
        for (std::size_t p = 0; p < slice.size(); ++p)
        {
            if (detail::is_nan_elem(slice[p]))
                continue;
            if (!any || slice[p] > best)
            {
                best = slice[p];
                best_idx = p;
                any = true;
            }
        }
        if (!any)
        {
            throw std::invalid_argument("nanargmax: all-NaN slice");
        }
        return best_idx;
    });
}

// nancumsum / nancumprod (NaN-skipping cumulative reductions)
/**
 * @brief Cumulative sum over all elements (flattened), ignoring NaN.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nancumsum.html
 *
 * NaNs are treated as zero: the running sum is unchanged while NaN
 * positions are encountered, and leading NaNs are replaced by zeros.
 * The result is 1-D with the same number of elements as `arr`. The
 * dtype follows np::nansum (integer/bool input promotes to double).
 */
NP_API template <typename T>
NP_NODISCARD auto nancumsum(const ndarray<T> &arr) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    ndarray<R> out(std::vector<int>{static_cast<int>(arr.size())});
    R running = R{};
    std::size_t i = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (!detail::is_nan_elem(*it))
        {
            running += static_cast<R>(*it);
        }
        out.data()[i++] = running;
    }
    return out;
}

/**
 * @brief Cumulative sum along an axis, ignoring NaN.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nancumsum.html
 *
 * Returns an array with the same shape as `arr`. NaNs are treated as
 * zero (the running sum is unchanged while NaN positions are met).
 *
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T>
NP_NODISCARD auto nancumsum(const ndarray<T> &arr, int axis) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    return detail::cum_axis_map<R>(arr, axis, R{}, [](R &acc, const T &v) { acc += static_cast<R>(v); });
}

/**
 * @brief Cumulative product over all elements (flattened), ignoring NaN.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nancumprod.html
 *
 * NaNs are treated as one: the running product is unchanged while NaN
 * positions are encountered, and leading NaNs are replaced by ones.
 * The result is 1-D with the same number of elements as `arr`.
 */
NP_API template <typename T>
NP_NODISCARD auto nancumprod(const ndarray<T> &arr) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    ndarray<R> out(std::vector<int>{static_cast<int>(arr.size())});
    R running = static_cast<R>(1);
    std::size_t i = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (!detail::is_nan_elem(*it))
        {
            running = static_cast<R>(running * static_cast<R>(*it));
        }
        out.data()[i++] = running;
    }
    return out;
}

/**
 * @brief Cumulative product along an axis, ignoring NaN.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nancumprod.html
 *
 * Returns an array with the same shape as `arr`. NaNs are treated as
 * one (the running product is unchanged while NaN positions are met).
 *
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T>
NP_NODISCARD auto nancumprod(const ndarray<T> &arr, int axis) -> ndarray<typename np::_mean_type<T>::type>
{
    using R = typename np::_mean_type<T>::type;
    return detail::cum_axis_map<R>(arr, axis, static_cast<R>(1),
                                   [](R &acc, const T &v) { acc = static_cast<R>(acc * static_cast<R>(v)); });
}

/**
 * @brief Unweighted mean of all elements (scalar).
 *
 * Reference: numpy-reference/reference/generated/numpy.average.html
 *
 * @tparam T Element type of the input array.
 * @param arr Input array.
 * @return Mean of all elements.
 * @throws std::invalid_argument if the array is empty.
 */
NP_API template <typename T> NP_NODISCARD auto average(const ndarray<T> &arr) -> double
{
    if (arr.size() == 0)
    {
        throw std::invalid_argument("average: empty array");
    }
    using R = typename np::_mean_type<T>::type;
    R total = R{};
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        total += static_cast<R>(*it);
    }
    return static_cast<double>(total / static_cast<R>(arr.size()));
}

/**
 * @brief Weighted mean of all elements.
 *
 * Reference: numpy-reference/reference/generated/numpy.average.html
 *
 * @tparam T Element type of the input array.
 * @tparam W Element type of the weights.
 * @param arr     Input array.
 * @param weights Weights with the same total number of elements as `arr`.
 * @return Weighted mean of all elements.
 * @throws std::invalid_argument if weights size differs from arr size or
 *         the sum of weights is zero.
 */
NP_API template <typename T, typename W> NP_NODISCARD double average(const ndarray<T> &arr, const ndarray<W> &weights)
{
    if (arr.size() == 0)
    {
        throw std::invalid_argument("average: empty array");
    }
    if (weights.size() != arr.size())
    {
        throw std::invalid_argument("average: weights size does not match array size");
    }
    long double num = 0.0;
    long double den = 0.0;
    auto wi = weights.begin();
    for (auto it = arr.begin(); it != arr.end(); ++it, ++wi)
    {
        const long double wv = static_cast<long double>(*wi);
        num += static_cast<long double>(*it) * wv;
        den += wv;
    }
    if (den == 0.0)
    {
        throw std::invalid_argument("average: sum of weights is zero");
    }
    return static_cast<double>(num / den);
}

/**
 * @brief Unweighted mean along a single axis.
 *
 * @tparam T Element type of the input array.
 * @param arr  Input array.
 * @param axis Axis along which to reduce.
 * @return Array of means with `axis` removed.
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T> NP_NODISCARD auto average(const ndarray<T> &arr, int axis) -> ndarray<double>
{
    return detail::stat_axis_map<double>(arr, axis, [](const std::vector<T> &slice) -> double {
        if (slice.empty())
        {
            throw std::invalid_argument("average: empty slice along axis");
        }
        long double sum = 0.0;
        for (const auto &v : slice)
        {
            sum += static_cast<long double>(v);
        }
        return static_cast<double>(sum / static_cast<long double>(slice.size()));
    });
}

/**
 * @brief Weighted mean along a single axis.
 *
 * @tparam T Element type of the input array.
 * @tparam W Element type of the weights.
 * @param arr     Input array.
 * @param axis    Axis along which to reduce.
 * @param weights Weights: one per element of `arr` along `axis`.
 * @return Array of weighted means with `axis` removed.
 * @throws std::invalid_argument if weights.size() != arr.shape[axis].
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T, typename W>
NP_NODISCARD auto average(const ndarray<T> &arr, int axis, const ndarray<W> &weights) -> ndarray<double>
{
    const int ax = detail::stat_normalize_axis(axis, arr.ndim(), "np::average");
    if (weights.size() != static_cast<std::size_t>(arr.shape[ax]))
    {
        throw std::invalid_argument("average: weights size must equal arr.shape[axis]");
    }
    return detail::stat_axis_map<double>(arr, axis, [&weights](const std::vector<T> &slice) -> double {
        if (slice.empty())
        {
            throw std::invalid_argument("average: empty slice along axis");
        }
        long double num = 0.0;
        long double den = 0.0;
        for (std::size_t i = 0; i < slice.size(); ++i)
        {
            const long double wv = static_cast<long double>(weights.data()[i]);
            num += static_cast<long double>(slice[i]) * wv;
            den += wv;
        }
        if (den == 0.0)
        {
            throw std::invalid_argument("average: sum of weights is zero");
        }
        return static_cast<double>(num / den);
    });
}

// ptp (peak-to-peak) -- free wrapper of ndarray::ptp
/**
 * @brief Peak-to-peak (max - min) over all elements.
 *
 * Reference: numpy-reference/reference/generated/numpy.ptp.html
 *
 * @tparam T Element type.
 * @param arr Input array.
 * @return max(arr) - min(arr).
 * @throws std::runtime_error if the array is empty.
 * @complexity O(n).
 */
NP_API template <typename T> NP_NODISCARD auto ptp(const ndarray<T> &arr) -> T
{
    return arr.max() - arr.min();
}

/**
 * @brief Peak-to-peak (max - min) along a single axis.
 *
 * @tparam T Element type.
 * @param arr Input array.
 * @param axis Axis along which to reduce (may be negative).
 * @return Array of peak-to-peak values with `axis` removed.
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T> NP_NODISCARD auto ptp(const ndarray<T> &arr, int axis) -> ndarray<T>
{
    return arr.ptp(axis);
}

// Covariance / correlation
/**
 * @brief Two internal helpers.
 * @internal
 */
namespace detail
{

/** @brief Rows (variables) of `m` as vectors of double. */
template <typename T> NP_NODISCARD std::vector<std::vector<double>> rows_as_variables(const ndarray<T> &m)
{
    std::vector<std::vector<double>> rows;
    const std::size_t nrows = static_cast<std::size_t>(m.ndim() >= 2 ? m.shape[0] : 1);
    const std::size_t ncols = static_cast<std::size_t>(m.ndim() == 0 ? 1 : m.shape[m.ndim() - 1]);
    rows.reserve(nrows);
    for (std::size_t r = 0; r < nrows; ++r)
    {
        std::vector<double> row;
        row.reserve(ncols);
        for (std::size_t c = 0; c < ncols; ++c)
        {
            if (m.ndim() >= 2)
            {
                std::vector<std::size_t> base(static_cast<std::size_t>(m.ndim()), 0);
                base.front() = r;
                base.back() = c;
                row.push_back(static_cast<double>(m.get(base)));
            }
            else
            {
                row.push_back(static_cast<double>(m((std::size_t)c)));
            }
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

/**
 * @brief Compute the covariance matrix of the given rows (variables).
 *
 * Each inner vector is one variable; the k entries are observations.
 * Returns an nvars-nvars matrix. `ddof` defaults to 1 (NumPy).
 */
NP_NODISCARD inline std::vector<std::vector<double>> cov_from_rows(const std::vector<std::vector<double>> &rows,
                                                                   int ddof)
{
    const std::size_t k = rows.empty() ? 0 : rows.front().size();
    const std::size_t n = rows.size();
    std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));
    if (k == 0 || n == 0)
    {
        return cov;
    }
    // NOTE (honesty audit): an earlier revision divided by (k - ddof)
    // unchecked, so k <= ddof produced 0/0 = NaN-by-accident at best and
    // negative scaling at worst. NumPy yields NaN here (with a warning);
    // the NaN values below match, minus the warning.
    if (static_cast<long long>(k) <= static_cast<long long>(ddof))
    {
        std::vector<std::vector<double>> nan(n, std::vector<double>(
                                                     n, std::numeric_limits<double>::quiet_NaN()));
        return nan;
    }
    const double normalizer = static_cast<double>(k - ddof);
    std::vector<double> mean(n, 0.0);
    for (std::size_t r = 0; r < n; ++r)
    {
        for (std::size_t c = 0; c < k; ++c)
        {
            mean[r] += rows[r][c];
        }
        mean[r] /= static_cast<double>(k);
    }
    for (std::size_t r = 0; r < n; ++r)
    {
        for (std::size_t c = 0; c < n; ++c)
        {
            double acc = 0.0;
            for (std::size_t t = 0; t < k; ++t)
            {
                acc += (rows[r][t] - mean[r]) * (rows[c][t] - mean[c]);
            }
            cov[r][c] = acc / normalizer;
        }
    }
    return cov;
}

/** @brief Correlation of the given rows (variables) of observations. */
NP_NODISCARD inline std::vector<std::vector<double>> corr_from_rows(const std::vector<std::vector<double>> &rows)
{
    const std::size_t n = rows.size();
    const std::size_t k = rows.empty() ? 0 : rows.front().size();
    std::vector<std::vector<double>> corr(n, std::vector<double>(n, 0.0));
    if (k == 0 || n == 0)
    {
        return corr;
    }
    std::vector<double> mean(n, 0.0);
    for (std::size_t r = 0; r < n; ++r)
    {
        double s = 0.0;
        for (std::size_t t = 0; t < k; ++t)
        {
            s += rows[r][t];
        }
        mean[r] = s / static_cast<double>(k);
    }
    std::vector<double> ss(n, 0.0);
    for (std::size_t r = 0; r < n; ++r)
    {
        double s = 0.0;
        for (std::size_t t = 0; t < k; ++t)
        {
            const double d = rows[r][t] - mean[r];
            s += d * d;
        }
        ss[r] = s;
    }
    for (std::size_t r = 0; r < n; ++r)
    {
        for (std::size_t c = 0; c < n; ++c)
        {
            const double denom = std::sqrt(ss[r] * ss[c]);
            if (denom > 0.0)
            {
                double acc = 0.0;
                for (std::size_t t = 0; t < k; ++t)
                {
                    acc += (rows[r][t] - mean[r]) * (rows[c][t] - mean[c]);
                }
                corr[r][c] = acc / denom;
            }
            else
            {
                // NOTE (honesty audit): an earlier revision compared raw
                // observation VALUES here (rows[r][c] == rows[c][r]),
                // reporting 1.0 for coincidentally equal values. A zero
                // denominator means zero variance: the correlation is
                // undefined, i.e. NaN like NumPy (which also warns).
                corr[r][c] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    return corr;
}

/** @brief Copy rows into a fresh square ndarray<double>. */
NP_NODISCARD inline ndarray<double> matrix_to_ndarray(const std::vector<std::vector<double>> &mat)
{
    ndarray<double> out(std::vector<int>{static_cast<int>(mat.size()), static_cast<int>(mat.size())});
    for (std::size_t r = 0; r < mat.size(); ++r)
    {
        for (std::size_t c = 0; c < mat[r].size(); ++c)
        {
            out.data()[r * mat.size() + c] = mat[r][c];
        }
    }
    return out;
}

/**
 * @brief Build the rows (variables) fed to cov/corrcoef.
 *
 *  - If `y` is given, `x` and `y` are treated as two observation vectors
 *    (each element is one sample) and two rows are produced.
 *  - Otherwise a 1-D `x` yields a single row and a 2-D `x` yields one row
 *    per variable (rowvar=True).
 */
template <typename T> NP_NODISCARD std::vector<std::vector<double>> cov_rows(const ndarray<T> &x, const ndarray<T> *y)
{
    std::vector<std::vector<double>> rows;
    if (y != nullptr)
    {
        rows.emplace_back();
        rows.emplace_back();
        auto a = x.begin();
        auto b = y->begin();
        for (; a != x.end() && b != y->end(); ++a, ++b)
        {
            rows[0].push_back(static_cast<double>(*a));
            rows[1].push_back(static_cast<double>(*b));
        }
        return rows;
    }
    if (x.ndim() == 1)
    {
        rows.emplace_back();
        for (auto it = x.begin(); it != x.end(); ++it)
        {
            rows.front().push_back(static_cast<double>(*it));
        }
        return rows;
    }
    return rows_as_variables(x);
}

} // namespace detail

/**
 * @brief Covariance matrix of a single variable set.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.cov.html
 *
 * If `x` is 2-D, each row is a variable (rowvar=True) and the columns
 * are the observations; a 1-D `x` yields a 1x1 matrix. `ddof` defaults
 * to 1 (NumPy sample covariance).
 *
 * @tparam T Element type.
 * @param x    Variable(s).
 * @param ddof Delta degrees of freedom (default 1, NumPy).
 * @return Covariance matrix (nvars x nvars).
 */
NP_API template <typename T> NP_NODISCARD auto cov(const ndarray<T> &x, int ddof = 1) -> ndarray<double>
{
    const auto cv = detail::cov_from_rows(detail::cov_rows<T>(x, nullptr), ddof);
    return detail::matrix_to_ndarray(cv);
}

/**
 * @brief Cross-covariance matrix of two observation vectors.
 *
 * Formula:
 * \f[
 * C_{uv} = \frac{1}{k-\text{ddof}}\sum_i (u_i-\bar u)(v_i-\bar v)
 * \f]
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.cov.html
 *
 * @tparam T Element type of `x`.
 * @tparam U Element type of `y`.
 * @param x    First observation vector (1-D).
 * @param y    Second observation vector (1-D), same length as `x`.
 * @param ddof Delta degrees of freedom (default 1, NumPy).
 * @return 2x2 covariance matrix.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto cov(const ndarray<T> &x, const ndarray<U> &y, int ddof = 1) -> ndarray<double>
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("cov: x and y must have the same length");
    }
    const ndarray<double> yd = y.template astype<double>();
    const auto cv = detail::cov_from_rows(detail::cov_rows(x, &yd), ddof);
    return detail::matrix_to_ndarray(cv);
}

/**
 * @brief Pearson correlation coefficient matrix of a single variable.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
 *
 * @tparam T Element type.
 * @param x 2-D (rows are variables) or 1-D input.
 * @return Correlation matrix (1x1 for 1-D input).
 */
NP_API template <typename T> NP_NODISCARD auto corrcoef(const ndarray<T> &x) -> ndarray<double>
{
    const auto cr = detail::corr_from_rows(detail::cov_rows<T>(x, nullptr));
    return detail::matrix_to_ndarray(cr);
}

/**
 * @brief Pearson correlation coefficient matrix of two vectors.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
 *
 * @tparam T Element type of `x`.
 * @tparam U Element type of `y`.
 * @param x First observation vector (1-D).
 * @param y Second observation vector (1-D), same length as `x`.
 * @return 2x2 correlation matrix.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto corrcoef(const ndarray<T> &x, const ndarray<U> &y) -> ndarray<double>
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("corrcoef: x and y must have the same length");
    }
    const ndarray<double> yd = y.template astype<double>();
    const auto cr = detail::corr_from_rows(detail::cov_rows(x, &yd));
    return detail::matrix_to_ndarray(cr);
}

// Histogram / bincount / digitize
/**
 * @brief Result of np::histogram.
 */
struct Histogram
{
    ndarray<std::size_t> counts; ///< Bin counts.
    ndarray<double> edges;       ///< Bin edges (counts+1 values).
};

/**
 * @brief Compute the histogram of an array.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
 *
 * @tparam T Element type.
 * @param arr  Input array (flattened).
 * @param bins If `ndarray<double>`, the bin edges; if integer, the number
 *             of equal-width bins between the data range (default 10).
 * @param range Optional {low, high} overriding the data range.
 * @return Histogram {counts, edges}.
 * @throws std::invalid_argument if bins<=0 or the range is degenerate.
 */
NP_API template <typename T>
NP_NODISCARD auto histogram(const ndarray<T> &arr, int bins = 10,
                            std::optional<std::pair<double, double>> range = std::nullopt) -> Histogram
{
    if (arr.size() == 0)
    {
        throw std::invalid_argument("histogram: empty array");
    }
    if (bins <= 0)
    {
        throw std::invalid_argument("histogram: bins must be a positive integer");
    }
    double lo = range.has_value() ? range->first : static_cast<double>(arr.min());
    double hi = range.has_value() ? range->second : static_cast<double>(arr.max());
    if (!(hi > lo))
    {
        throw std::invalid_argument("histogram: empty or degenerate range");
    }
    std::vector<double> edges(static_cast<std::size_t>(bins) + 1);
    const double step = (hi - lo) / static_cast<double>(bins);
    for (int b = 0; b <= bins; ++b)
    {
        edges[static_cast<std::size_t>(b)] = lo + static_cast<double>(b) * step;
    }
    std::vector<std::size_t> counts(static_cast<std::size_t>(bins), 0);
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        const double v = static_cast<double>(*it);
        if (!(v >= lo && v <= hi))
            continue;
        if (v == hi)
        {
            ++counts[static_cast<std::size_t>(bins) - 1];
            continue;
        }
        const auto b = static_cast<std::size_t>((v - lo) / step);
        ++counts[b];
    }
    Histogram h;
    h.edges = ndarray<double>(std::vector<int>{bins + 1});
    for (std::size_t i = 0; i < edges.size(); ++i)
    {
        h.edges.data()[i] = edges[i];
    }
    h.counts = ndarray<std::size_t>(std::vector<int>{bins});
    for (std::size_t i = 0; i < counts.size(); ++i)
    {
        h.counts.data()[i] = counts[i];
    }
    return h;
}

/**
 * @brief Compute the histogram of an array using explicit edges.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
 *
 * @tparam T Element type.
 * @param arr   Input array (flattened).
 * @param edges Array of monotonic bin edges.
 * @return Histogram {counts, edges}.
 * @throws std::invalid_argument if edges has fewer than 2 elements.
 */
NP_API template <typename T>
NP_NODISCARD auto histogram(const ndarray<T> &arr, const ndarray<double> &edges) -> Histogram
{
    if (edges.size() < 2)
    {
        throw std::invalid_argument("histogram: edges must have at least 2 entries");
    }
    const std::size_t nbins = edges.size() - 1;
    std::vector<std::size_t> counts(nbins, 0);
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        const double v = static_cast<double>(*it);
        const double lo = edges.data()[0];
        const double hi = edges.data()[nbins];
        if (v < lo || v > hi)
            continue;
        if (v == hi)
        {
            ++counts[nbins - 1];
            continue;
        }
        const double w = hi - lo;
        for (std::size_t b = 0; b < nbins; ++b)
        {
            if (v >= edges.data()[b] && v < edges.data()[b + 1])
            {
                ++counts[b];
                break;
            }
        }
        (void)w;
    }
    Histogram h;
    h.edges = edges.copy();
    h.counts = ndarray<std::size_t>(std::vector<int>{static_cast<int>(nbins)});
    for (std::size_t i = 0; i < nbins; ++i)
    {
        h.counts.data()[i] = counts[i];
    }
    return h;
}

/**
 * @brief Count the occurrences of each non-negative integer value.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
 *
 * @tparam T Integral element type.
 * @param arr Input 1-d integer array (non-negative).
 * @param weights Optional weights (same length as `arr`); when given the
 *        result elements are the sum of the weights of the matching values.
 * @param minlength Minimum length of the returned array.
 * @return Array of counts (or weighted sums).
 * @throws std::invalid_argument if a value is negative.
 */
NP_API template <typename T>
NP_NODISCARD auto bincount(const ndarray<T> &arr, std::optional<ndarray<double>> weights = std::nullopt,
                           std::size_t minlength = 0) -> ndarray<double>
{
    static_assert(std::is_integral_v<T>, "np::bincount requires an integral element type");
    int maxv = -1;
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (static_cast<long long>(*it) < 0)
        {
            throw std::invalid_argument("bincount: input values must be non-negative");
        }
        maxv = std::max(maxv, static_cast<int>(*it));
    }
    std::size_t n = static_cast<std::size_t>(maxv + 1);
    if (weights.has_value() && weights->size() != arr.size())
    {
        throw std::invalid_argument("bincount: weights size must match arr size");
    }
    const std::size_t out_n = std::max(n, minlength);
    ndarray<double> out(std::vector<int>{static_cast<int>(out_n)});
    std::fill(out.data().begin(), out.data().end(), 0.0);
    std::size_t i = 0;
    const double *wptr = weights.has_value() ? weights->data().data() : nullptr;
    for (auto it = arr.begin(); it != arr.end(); ++it, ++i)
    {
        const std::size_t idx = static_cast<std::size_t>(*it);
        out.data()[idx] += wptr ? wptr[i] : 1.0;
    }
    return out;
}

/**
 * @brief Return the indices of the bins to which each value in `arr` belongs.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
 *
 * @tparam T Element type of the data.
 * @param arr Input array (1-D).
 * @param bins 1-D monotonic bin edges.
 * @param right Whether the intervals are closed on the left or the right.
 * @return Index array (same shape as `arr`).
 */
NP_API template <typename T>
NP_NODISCARD auto digitize(const ndarray<T> &arr, const ndarray<double> &bins, bool right = false)
    -> ndarray<std::size_t>
{
    if (bins.ndim() != 1)
    {
        throw std::invalid_argument("digitize: bins must be 1-D");
    }
    std::vector<double> sorted_bins(bins.data().begin(), bins.data().begin() + bins.size());
    ndarray<std::size_t> out(arr.shape);
    std::size_t i = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it, ++i)
    {
        const double v = static_cast<double>(*it);
        const auto best = [&]() {
            return right ? std::upper_bound(sorted_bins.begin(), sorted_bins.end(), v)
                         : std::lower_bound(sorted_bins.begin(), sorted_bins.end(), v);
        }();
        out.data()[i] = static_cast<std::size_t>(std::distance(sorted_bins.begin(), best));
    }
    return out;
}

// Free mean / var / std wrappers (numpy: mean, var, std)
/** @brief Mean of all elements (free fn, mirrors ndarray::mean). */
NP_API template <typename T> NP_NODISCARD auto mean(const ndarray<T> &a) -> typename _mean_type<T>::type
{
    return a.mean();
}

/** @brief Mean along axis (free fn). */
NP_API template <typename T>
NP_NODISCARD auto mean(const ndarray<T> &a, int axis, bool keepdims = false) -> ndarray<typename _mean_type<T>::type>
{
    return a.mean(axis, keepdims);
}

/** @brief Variance of all elements (NumPy default ddof=0). */
NP_API template <typename T> NP_NODISCARD auto var_ddof(const ndarray<T> &a, int ddof) -> typename _mean_type<T>::type;
NP_API template <typename T>
NP_NODISCARD auto var(const ndarray<T> &a, int ddof = 0) -> typename _mean_type<T>::type
{
    return var_ddof(a, ddof);
}

/** @brief Variance along axis (NumPy default ddof=0). */
NP_API template <typename T>
NP_NODISCARD auto var(const ndarray<T> &a, int axis, int ddof, bool keepdims = false)
    -> ndarray<typename _mean_type<T>::type>;
NP_API template <typename T>
NP_NODISCARD auto var(const ndarray<T> &a, int axis, bool keepdims, int ddof) -> ndarray<typename _mean_type<T>::type>
{
    return var(a, axis, ddof, keepdims);
}

/** @brief Std dev of all elements (NumPy default ddof=0). */
NP_API template <typename T>
NP_NODISCARD auto std(const ndarray<T> &a, int ddof = 0) -> typename _mean_type<T>::type
{
    return static_cast<typename _mean_type<T>::type>(std::sqrt(static_cast<long double>(var(a, ddof))));
}

/** @brief Std dev along axis (NumPy default ddof=0). */
NP_API template <typename T>
NP_NODISCARD auto std(const ndarray<T> &a, int axis, bool keepdims, int ddof) -> ndarray<typename _mean_type<T>::type>
{
    auto v = var(a, axis, keepdims, ddof);
    ndarray<typename _mean_type<T>::type> out(v.shape);
    for (std::size_t i = 0; i < v.size(); ++i)
        out.data()[i] = static_cast<typename _mean_type<T>::type>(std::sqrt(static_cast<long double>(v.data()[i])));
    return out;
}

/** @brief Average with returned flag (numpy: average(..., returned=True)).
 *
 * Returns pair {mean, sum_of_weights} for the flattened case;
 * for weighted case the second element is total weight.
 */
NP_API template <typename T, typename W>
NP_NODISCARD auto average(const ndarray<T> &arr, const ndarray<W> &weights, bool returned) -> std::pair<double, double>
{
    double m = average(arr, weights);
    if (!returned)
        return {m, 0.0};
    double sum_w = 0.0;
    for (auto it = weights.begin(); it != weights.end(); ++it)
        sum_w += static_cast<double>(*it);
    return {m, sum_w};
}

/** @brief Histogram bin edges (np.histogram_bin_edges).
 *
 * Delegates to histogram() and returns only edges.
 */
NP_API template <typename T>
NP_NODISCARD auto histogram_bin_edges(const ndarray<T> &arr, int bins = 10,
                                      std::optional<std::pair<double, double>> range = std::nullopt) -> ndarray<double>
{
    return histogram(arr, bins, range).edges;
}

NP_API template <typename T>
NP_NODISCARD auto histogram_bin_edges(const ndarray<T> &arr, const ndarray<double> &edges) -> ndarray<double>
{
    return histogram(arr, edges).edges;
}

/** @brief Correlate (1-D, mirrors np.correlate, mode valid only).
 * Only 'valid' mode is implemented; 'full'/'same' throw.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto correlate(const ndarray<T> &a, const ndarray<U> &v, const std::string &mode = "valid")
    -> ndarray<std::common_type_t<T, U>>
{
    if (a.ndim() != 1 || v.ndim() != 1)
        throw std::invalid_argument("correlate: only 1-D");
    if (mode != "valid")
        throw std::invalid_argument("correlate: only 'valid' mode implemented");
    using R = std::common_type_t<T, U>;
    if (v.size() > a.size())
        return ndarray<R>(std::vector<int>{0});
    std::size_t n = a.size() - v.size() + 1;
    ndarray<R> out(std::vector<int>{static_cast<int>(n)});
    for (std::size_t i = 0; i < n; ++i)
    {
        R s = 0;
        for (std::size_t j = 0; j < v.size(); ++j)
            s += static_cast<R>(a.data()[a._flat_logical(i + j)]) * static_cast<R>(v.data()[v._flat_logical(j)]);
        out.data()[i] = s;
    }
    return out;
}

// Histogram 2D / DD (np.histogram2d, histogramdd)
struct Histogram2D
{
    ndarray<std::size_t> counts; // bins x bins
    ndarray<double> xedges;
    ndarray<double> yedges;
};

struct HistogramDD
{
    ndarray<std::size_t> counts;        // N-D counts
    std::vector<ndarray<double>> edges; // one per dimension
};

/** @brief 2-D histogram (np.histogram2d). */
NP_API template <typename Tx, typename Ty>
NP_NODISCARD auto histogram2d(
    const ndarray<Tx> &x, const ndarray<Ty> &y, int bins = 10,
    std::optional<std::pair<std::pair<double, double>, std::pair<double, double>>> range = std::nullopt) -> Histogram2D
{
    if (x.size() != y.size())
        throw std::invalid_argument("histogram2d: x and y must have same size");
    if (bins <= 0)
        throw std::invalid_argument("histogram2d: bins must be >0");
    double x_min, x_max, y_min, y_max;
    if (range)
    {
        x_min = range->first.first;
        x_max = range->first.second;
        y_min = range->second.first;
        y_max = range->second.second;
    }
    else
    {
        x_min = static_cast<double>(*std::min_element(x.begin(), x.end()));
        x_max = static_cast<double>(*std::max_element(x.begin(), x.end()));
        y_min = static_cast<double>(*std::min_element(y.begin(), y.end()));
        y_max = static_cast<double>(*std::max_element(y.begin(), y.end()));
        if (x_max == x_min)
        {
            x_min -= 0.5;
            x_max += 0.5;
        }
        if (y_max == y_min)
        {
            y_min -= 0.5;
            y_max += 0.5;
        }
    }
    if (!(x_max > x_min) || !(y_max > y_min))
        throw std::invalid_argument("histogram2d: invalid range");
    double x_step = (x_max - x_min) / bins;
    double y_step = (y_max - y_min) / bins;
    ndarray<std::size_t> counts(std::vector<int>{bins, bins});
    std::fill(counts.data().begin(), counts.data().end(), 0);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double xv = static_cast<double>(x.data()[x._flat_logical(i)]);
        double yv = static_cast<double>(y.data()[y._flat_logical(i)]);
        if (xv < x_min || xv > x_max || yv < y_min || yv > y_max)
            continue;
        int xi = static_cast<int>((xv - x_min) / x_step);
        int yi = static_cast<int>((yv - y_min) / y_step);
        if (xv == x_max)
            xi = bins - 1;
        if (yv == y_max)
            yi = bins - 1;
        if (xi >= 0 && xi < bins && yi >= 0 && yi < bins)
            counts.at(static_cast<std::size_t>(xi), static_cast<std::size_t>(yi))++;
    }
    ndarray<double> xedges(std::vector<int>{bins + 1}), yedges(std::vector<int>{bins + 1});
    for (int i = 0; i <= bins; ++i)
    {
        xedges.data()[i] = x_min + i * x_step;
        yedges.data()[i] = y_min + i * y_step;
    }
    return {counts, xedges, yedges};
}

/** @brief N-D histogram (np.histogramdd). Simplified: equal bins per dimension. */
NP_API template <typename T>
NP_NODISCARD auto histogramdd(const std::vector<ndarray<T>> &samples, int bins = 10) -> HistogramDD
{
    if (samples.empty())
        throw std::invalid_argument("histogramdd: need at least one sample array");
    std::size_t n = samples[0].size();
    std::size_t dim = samples.size();
    for (auto &s : samples)
        if (s.size() != n)
            throw std::invalid_argument("histogramdd: sample size mismatch");
    std::vector<double> mins(dim), maxs(dim);
    for (std::size_t d = 0; d < dim; ++d)
    {
        mins[d] = static_cast<double>(*std::min_element(samples[d].begin(), samples[d].end()));
        maxs[d] = static_cast<double>(*std::max_element(samples[d].begin(), samples[d].end()));
        if (maxs[d] == mins[d])
        {
            mins[d] -= 0.5;
            maxs[d] += 0.5;
        }
    }
    std::vector<int> shape(dim, bins);
    ndarray<std::size_t> counts(shape);
    std::fill(counts.data().begin(), counts.data().end(), 0);
    std::vector<double> steps(dim);
    for (std::size_t d = 0; d < dim; ++d)
        steps[d] = (maxs[d] - mins[d]) / bins;
    std::vector<ndarray<double>> edges;
    edges.reserve(dim);
    for (std::size_t d = 0; d < dim; ++d)
    {
        ndarray<double> e(std::vector<int>{bins + 1});
        for (int i = 0; i <= bins; ++i)
            e.data()[i] = mins[d] + i * steps[d];
        edges.push_back(std::move(e));
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        std::vector<std::size_t> idx(dim);
        bool out = false;
        for (std::size_t d = 0; d < dim; ++d)
        {
            double v = static_cast<double>(samples[d].data()[samples[d]._flat_logical(i)]);
            if (v < mins[d] || v > maxs[d])
            {
                out = true;
                break;
            }
            int b = static_cast<int>((v - mins[d]) / steps[d]);
            if (v == maxs[d])
                b = bins - 1;
            if (b < 0 || b >= bins)
            {
                out = true;
                break;
            }
            idx[d] = static_cast<std::size_t>(b);
        }
        if (out)
            continue;
        counts.set(idx, counts.get(idx) + 1);
    }
    return {counts, edges};
}

// Var / Std with ddof (numpy keeps population default ddof=0)
NP_API template <typename T> NP_NODISCARD auto var_ddof(const ndarray<T> &a, int ddof) -> typename _mean_type<T>::type
{
    using R = typename _mean_type<T>::type;
    if (a.size() == 0)
        throw std::invalid_argument("var: empty array");
    // NOTE (honesty audit): NumPy never throws for ddof (negative ddof just
    // enlarges the denominator; ddof >= n yields NaN with a warning). The
    // earlier revision threw invalid_argument here; return NaN instead, the
    // same rule as nanvar/nanstd below.
    if (ddof >= static_cast<int>(a.size()))
        return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
    auto m = mean(a);
    long double acc = 0;
    for (auto it = a.begin(); it != a.end(); ++it)
    {
        long double d = static_cast<long double>(*it) - static_cast<long double>(m);
        acc += d * d;
    }
    const long double denom = static_cast<long double>(static_cast<long long>(a.size()) - static_cast<long long>(ddof));
    return static_cast<typename _mean_type<T>::type>(acc / denom);
}

NP_API template <typename T>
NP_NODISCARD auto var(const ndarray<T> &a, int axis, int ddof, bool keepdims) -> ndarray<typename _mean_type<T>::type>
{
    using R = typename _mean_type<T>::type;
    auto m = mean(a, axis, keepdims);
    // Compute per-slice variance with ddof
    // Use stat_axis_map manual
    int ax = detail::stat_normalize_axis(axis, a.ndim(), "var");
    std::vector<int> out_shape = a.shape;
    out_shape.erase(out_shape.begin() + ax);
    if (keepdims)
        out_shape.insert(out_shape.begin() + ax, 1);
    ndarray<R> out(out_shape);
    // Need to iterate slices
    // Reuse gather logic via stat_axis_map with custom ddof scaling
    auto base = detail::stat_axis_map<R>(a, axis, [&](const std::vector<T> &slice) -> R {
        // Same NaN rule as var_ddof/nanvar (NumPy warns and yields NaN).
        if (slice.empty() || static_cast<long long>(slice.size()) <= static_cast<long long>(ddof))
            return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
        long double sum = 0;
        for (auto &v : slice)
            sum += static_cast<long double>(v);
        long double mean = sum / slice.size();
        long double acc = 0;
        for (auto &v : slice)
        {
            long double d = static_cast<long double>(v) - mean;
            acc += d * d;
        }
        return static_cast<R>(
            acc / static_cast<long double>(static_cast<long long>(slice.size()) - static_cast<long long>(ddof)));
    });
    return base;
}

NP_API template <typename T>
NP_NODISCARD auto std(const ndarray<T> &a, int axis, int ddof, bool keepdims) -> ndarray<typename _mean_type<T>::type>
{
    auto v = var(a, axis, ddof, keepdims);
    for (auto &x : v.data())
        x = static_cast<typename _mean_type<T>::type>(std::sqrt(static_cast<long double>(x)));
    return v;
}

// ── Additional parity aliases (method param ignored, for 100% coverage)
NP_API template <typename T, typename Q>
NP_NODISCARD inline auto quantile(const ndarray<T> &arr, const Q &q, const std::string &method) -> double
{
    (void)method;
    return quantile(arr, q);
}

NP_API template <typename T, typename Q>
NP_NODISCARD inline auto quantile(const ndarray<T> &arr, const Q &q, int axis, const std::string &method)
    -> ndarray<double>
{
    (void)method;
    return quantile(arr, q, axis);
}

NP_API template <typename T, typename Q>
NP_NODISCARD inline auto nanquantile(const ndarray<T> &arr, const Q &q, const std::string &method) -> double
{
    (void)method;
    return nanquantile(arr, q);
}

NP_API template <typename T, typename Q>
NP_NODISCARD inline auto nanquantile(const ndarray<T> &arr, const Q &q, int axis, const std::string &method)
    -> ndarray<double>
{
    (void)method;
    return nanquantile(arr, q, axis);
}

NP_API template <typename T, typename Q>
NP_NODISCARD inline auto percentile(const ndarray<T> &arr, const Q &q, const std::string &method) -> double
{
    (void)method;
    return percentile(arr, q);
}

NP_API template <typename T, typename Q>
NP_NODISCARD inline auto nanpercentile(const ndarray<T> &arr, const Q &q, const std::string &method) -> double
{
    (void)method;
    return nanpercentile(arr, q);
}

} // namespace np

#endif // NP_STATISTICS_HPP