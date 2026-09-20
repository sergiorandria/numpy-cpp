/**
 * @file sorting.hpp
 * @brief Sorting and searching: sort, argsort, lexsort, msort,
 *        sort_complex, partition, argpartition, argmin/max,
 *        searchsorted, extract, argwhere, flatnonzero, count_nonzero.
 *
 * Reference: numpy-reference/reference/routines.sort.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_SORTING_HPP
#define NP_SORTING_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np::inline v1
{

/** @brief Indirect stable sort using sequence of keys.
 *
 * `keys[0]` is secondary, `keys.back()` is primary (NumPy order).
 * All keys must be 1-D and same length. Returns permutation indices
 * that sort by the tuple (keys[0][i], ..., keys[n-1][i]).
 *
 * Reference: numpy-reference/reference/generated/numpy.lexsort.html
 *
 * @tparam T Element type of keys.
 * @param keys Vector of 1-D key arrays.
 * @return 1-D ndarray<std::size_t> permutation indices.
 * @throws std::invalid_argument if keys empty or length mismatch.
 */
NP_API template <typename T> NP_NODISCARD auto lexsort(const std::vector<ndarray<T>> &keys) -> ndarray<std::size_t>
{
    if (keys.empty())
    {
        throw std::invalid_argument("lexsort: at least one key required");
    }
    std::size_t n = keys[0].size();
    for (const auto &k : keys)
    {
        if (k.ndim() != 1)
            throw std::invalid_argument("lexsort: keys must be 1-D");
        if (k.size() != n)
            throw std::invalid_argument("lexsort: keys length mismatch");
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("lexsort: key length exceeds int range");

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        // primary key is last element
        for (std::ptrdiff_t ki = static_cast<std::ptrdiff_t>(keys.size()) - 1; ki >= 0; --ki)
        {
            T va = keys[static_cast<std::size_t>(ki)].at(a);
            T vb = keys[static_cast<std::size_t>(ki)].at(b);
            if (va < vb)
                return true;
            if (va > vb)
                return false;
        }
        return false;
    });
    ndarray<std::size_t> out(std::vector<int>{static_cast<int>(n)});
    for (std::size_t i = 0; i < n; ++i)
        out.data()[i] = idx[i];
    return out;
}

/** @brief Sort array along first axis (numpy.msort).
 *
 * Equivalent to `arr.sorted(0)` for 2-D and flatten sort for 1-D.
 * Deprecated in numpy but retained for coverage.
 *
 * Reference: numpy-reference/reference/generated/numpy.msort.html
 */
NP_API template <typename T> NP_NODISCARD auto msort(const ndarray<T> &arr) -> ndarray<T>
{
    if (arr.ndim() <= 1)
        return arr.sorted();
    return arr.sorted(0);
}

/** @brief Sort complex array by real then imag.
 *
 * Mirrors `np.sort_complex`. Real inputs are promoted to complex.
 *
 * Reference: numpy-reference/reference/generated/numpy.sort_complex.html
 */
NP_API template <typename T> NP_NODISCARD auto sort_complex(const ndarray<T> &arr) -> ndarray<std::complex<double>>
{
    // Promote to complex<double>
    ndarray<std::complex<double>> c(arr.shape);
    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        c.data()[c._flat_logical(i)] = std::complex<double>(static_cast<double>(arr.data()[arr._flat_logical(i)]), 0.0);
    }
    // sort flattened then reshape? numpy sorts along last axis; we mirror member sort
    // logic: sort last axis
    c.sort(-1);
    // For complex, we need real-then-imag comparison; std::sort uses operator< which does
    // lex order exactly.
    return c;
}

NP_API template <typename T>
NP_NODISCARD auto sort_complex(const ndarray<std::complex<T>> &arr) -> ndarray<std::complex<T>>
{
    auto out = arr.copy();
    out.sort(-1);
    return out;
}

/** @brief Indices of non-zero elements as 2-D array (N, ndim).
 *
 * Mirrors `np.argwhere`. Each row is the multi-index of a non-zero element.
 *
 * @param arr Input array (any dtype converts to bool).
 * @return ndarray<std::size_t> shape (N, ndim) row-major.
 */
NP_API template <typename T> NP_NODISCARD auto argwhere(const ndarray<T> &arr) -> ndarray<std::size_t>
{
    auto nz = arr.nonzero(); // vector per dim
    if (nz.empty())
    {
        return ndarray<std::size_t>(std::vector<int>{0, 0});
    }
    std::size_t N = nz[0].size();
    std::size_t nd = arr.ndim();
    ndarray<std::size_t> out(std::vector<int>{static_cast<int>(N), static_cast<int>(nd)});

    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t d = 0; d < nd; ++d)
        {
            out.at(i, d) = nz[d].at(i);
        }
    }
    return out;
}

/** @brief Indices of non-zero elements in flattened array.
 *
 * Mirrors `np.flatnonzero`.
 *
 * @param arr Input array.
 * @return 1-D array of flat indices.
 */
NP_API template <typename T> NP_NODISCARD auto flatnonzero(const ndarray<T> &arr) -> ndarray<std::size_t>
{
    std::vector<std::size_t> idx;
    std::size_t flat = 0;

    for (auto it = arr.begin(); it != arr.end(); ++it, ++flat)
    {
        if (static_cast<bool>(*it))
            idx.push_back(flat);
    }

    ndarray<std::size_t> out(std::vector<int>{static_cast<int>(idx.size())});

    for (std::size_t i = 0; i < idx.size(); ++i)
        out.data()[i] = idx[i];

    return out;
}

/** @brief Count non-zero elements.
 *
 * Mirrors `np.count_nonzero`. When axis is nullopt counts overall,
 * otherwise counts along axis.
 *
 * @param arr Input array.
 * @param axis Axis along which to count (nullopt = flattened).
 * @return Scalar count or reduced array.
 */
NP_API template <typename T> NP_NODISCARD std::size_t count_nonzero(const ndarray<T> &arr)
{
    std::size_t c = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it)
    {
        if (static_cast<bool>(*it))
            ++c;
    }
    return c;
}

NP_API template <typename T> NP_NODISCARD auto count_nonzero(const ndarray<T> &arr, int axis) -> ndarray<std::size_t>
{
    if (arr.ndim() == 0)
        throw std::invalid_argument("count_nonzero: 0-d array has no axis");
    if (axis < 0)
        axis += static_cast<int>(arr.ndim());
    if (axis < 0 || axis >= static_cast<int>(arr.ndim()))
        throw AxisError("count_nonzero: axis out of bounds");

    std::vector<int> out_shape = arr.shape;
    out_shape.erase(out_shape.begin() + axis);
    ndarray<std::size_t> out(out_shape);
    std::fill(out.data().begin(), out.data().end(), 0);

    // Row-major strides of the (contiguous) output, hoisted out of the loop.
    std::vector<std::size_t> out_strides(out_shape.size(), 1);
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(out_shape.size()) - 2; d >= 0; --d)
        out_strides[static_cast<std::size_t>(d)] =
            out_strides[static_cast<std::size_t>(d) + 1] * static_cast<std::size_t>(out_shape[static_cast<std::size_t>(d) + 1]);

    // Iterate over all elements and increment output bin
    detail::Odometer od(arr.shape);
    std::vector<std::size_t> oidx;
    oidx.reserve(out_shape.size());
    while (!od.done())
    {
        const auto &idx = od.idx();
        if (static_cast<bool>(arr.get(idx)))
        {
            oidx.clear();
            for (std::size_t d = 0; d < idx.size(); ++d)
                if (static_cast<int>(d) != axis)
                    oidx.push_back(idx[d]);

            if (oidx.empty())
            {
                out.data()[0] += 1;
            }
            else
            {
                std::size_t fo = 0;
                for (std::size_t d = 0; d < oidx.size(); ++d)
                    fo += oidx[d] * out_strides[d];
                out.data()[fo] += 1;
            }
        }
        od.advance();
    }
    return out;
}

// Free wrappers mirroring NumPy API
/** @brief Sorted copy (np.sort). */
NP_API template <typename T> NP_NODISCARD auto sort(const ndarray<T> &a, int axis = -1) -> ndarray<T>
{
    return a.sorted(axis);
}

/** @brief Argsort (np.argsort). */
NP_API template <typename T> NP_NODISCARD auto argsort(const ndarray<T> &a, int axis = -1) -> ndarray<std::size_t>
{
    return a.argsort(axis);
}

/** @brief Partition – kth element in sorted position (np.partition). */
NP_API template <typename T>
NP_NODISCARD auto partition(const ndarray<T> &a, std::size_t kth, int axis = -1) -> ndarray<T>
{
    auto out = a.copy();
    out.partition(kth, axis);
    return out;
}

/** @brief Argpartition (np.argpartition). */
NP_API template <typename T>
NP_NODISCARD auto argpartition(const ndarray<T> &a, std::size_t kth, int axis = -1) -> ndarray<std::size_t>
{
    return a.argpartition(kth, axis);
}

/** @brief Searchsorted scalar (np.searchsorted). */
NP_API template <typename T>
NP_NODISCARD std::size_t searchsorted(const ndarray<T> &a, const T &value, bool side_right = false)
{
    return a.searchsorted(value, side_right);
}

/** @brief Searchsorted array (np.searchsorted). */
NP_API template <typename T>
NP_NODISCARD auto searchsorted(const ndarray<T> &a, const ndarray<int> &values) -> ndarray<std::size_t>
{
    return a.searchsorted(values);
}

/** @brief Searchsorted with sorter index array (np.searchsorted with sorter).
 *
 * `sorter` is permutation that sorts `a`. This overload validates sizes.
 */
NP_API template <typename T>
NP_NODISCARD std::size_t searchsorted(const ndarray<T> &a, const T &value, const ndarray<std::size_t> &sorter,
                                      bool side_right = false)
{
    if (sorter.size() != a.size())
        throw std::invalid_argument("searchsorted: sorter size mismatch");
    // Binary search over sorter indices directly (no sorted copy, no
    // default-constructed T required).
    const auto at_sorted = [&](std::size_t i) -> const T & {
        return a.data()[a._flat_logical(sorter.data()[sorter._flat_logical(i)])];
    };
    std::size_t lo = 0, hi = a.size();
    if (!side_right)
    {
        while (lo < hi)
        {
            std::size_t mid = lo + (hi - lo) / 2;
            if (at_sorted(mid) < value)
                lo = mid + 1;
            else
                hi = mid;
        }
    }
    else
    {
        while (lo < hi)
        {
            std::size_t mid = lo + (hi - lo) / 2;
            if (value < at_sorted(mid))
                hi = mid;
            else
                lo = mid + 1;
        }
    }
    return lo;
}

/** @brief Extract elements where condition is true (np.extract).
 *
 * @param condition Bool array, broadcastable to `arr` shape; for simplicity
 *                  requires identical shape (NumPy broadcasts, we enforce exact).
 * @param arr Source array.
 * @return 1-D array of selected elements in C order.
 */
NP_API template <typename T>
NP_NODISCARD auto extract(const ndarray<bool> &condition, const ndarray<T> &arr) -> ndarray<T>
{
    if (condition.shape != arr.shape)
        throw std::invalid_argument("extract: condition and arr must have same shape");
    std::vector<T> out;
    out.reserve(arr.size());
    detail::Odometer od(condition.shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        if (condition.get(idx))
            out.push_back(arr.get(idx));
        od.advance();
    }
    ndarray<T> res(std::vector<int>{static_cast<int>(out.size())});
    for (std::size_t i = 0; i < out.size(); ++i)
        res.data()[i] = out[i];
    return res;
}

/** @brief Extract with scalar condition broadcast (convenience). */
NP_API template <typename T> NP_NODISCARD auto extract(bool condition, const ndarray<T> &arr) -> ndarray<T>
{
    if (!condition)
        return ndarray<T>(std::vector<int>{0});
    return arr.flatten();
}

// ── argmax / argmin (free wrappers, numpy.sort parity) ─────────────
NP_API template <typename T> NP_NODISCARD inline auto argmax(const ndarray<T> &a) -> std::size_t
{
    return a.argmax();
}

NP_API template <typename T> NP_NODISCARD inline auto argmax(const ndarray<T> &a, int axis) -> ndarray<std::size_t>
{
    return a.argmax(axis);
}

NP_API template <typename T> NP_NODISCARD inline auto argmin(const ndarray<T> &a) -> std::size_t
{
    return a.argmin();
}

NP_API template <typename T> NP_NODISCARD inline auto argmin(const ndarray<T> &a, int axis) -> ndarray<std::size_t>
{
    return a.argmin(axis);
}

// nanargmax / nanargmin are provided by statistics.hpp and re-exported
// via the umbrella header; no duplicate definition here to avoid ODR
// clash. Sorting parity is satisfied when both headers are included
// (see include/np/np.hpp:33 includes sorting + statistics).

// ── Missing parity: nonzero / where free wrappers ───────────────────
/**
 * @brief Non-zero indices (np.nonzero).
 * Returns vector of arrays per dimension (mirrors ndarray::nonzero).
 * Reference: numpy-reference/reference/generated/numpy.nonzero.html
 */
NP_API template <typename T> NP_NODISCARD inline auto nonzero(const ndarray<T> &a) -> std::vector<ndarray<std::size_t>>
{
    return a.nonzero();
}

// `where` lives in manipulation.hpp; counted for sorting taxonomy via
// umbrella header – no duplicate here to avoid ODR clash.

/**
 * @brief Sort with kind dispatch (np.sort with kind).
 *
 * Valid `kind` values are "quicksort", "mergesort", "heapsort", "stable"
 * (NumPy parity). quicksort maps to `std::sort`, mergesort/stable to
 * `std::stable_sort`, heapsort to `make_heap` + `sort_heap`. Sorting is
 * applied slice-wise along `axis` via the same layout as `ndarray::sorted`.
 * Reference: numpy-reference/reference/generated/numpy.sort.html
 */
NP_API template <typename T>
NP_NODISCARD inline auto sort(const ndarray<T> &a, int axis, const std::string &kind) -> ndarray<T>
{
    if (kind != "quicksort" && kind != "mergesort" && kind != "heapsort" && kind != "stable")
    {
        throw std::invalid_argument("sort: kind must be one of 'quicksort', 'mergesort', 'heapsort', 'stable'");
    }
    if (kind == "quicksort" || kind == "heapsort")
    {
        // Non-stable paths: sort each 1-D slice with the matching algorithm.
        ndarray<T> out = a.copy();
        const int ax = axis < 0 ? axis + static_cast<int>(out.ndim()) : axis;
        if (out.ndim() == 1)
        {
            auto &d = out.data();
            if (kind == "quicksort")
            {
                std::sort(d.begin(), d.end());
            }
            else
            {
                std::make_heap(d.begin(), d.end());
                std::sort_heap(d.begin(), d.end());
            }
            return out;
        }
        // N-D: gather slices along axis, sort each slice in place.
        std::vector<int> slice_shape = out.shape;
        slice_shape.erase(slice_shape.begin() + ax);
        np::detail::Odometer od(slice_shape);
        while (!od.done())
        {
            const auto &idx = od.idx();
            std::vector<T> vals;
            vals.reserve(static_cast<std::size_t>(out.shape[ax]));
            for (int k = 0; k < out.shape[ax]; ++k)
            {
                std::vector<std::size_t> full;
                full.reserve(out.shape.size());
                for (std::size_t d = 0; d < out.shape.size(); ++d)
                {
                    full.push_back(d == static_cast<std::size_t>(ax) ? static_cast<std::size_t>(k)
                                                                     : static_cast<std::size_t>(idx[d < static_cast<std::size_t>(ax) ? d : d - 1]));
                }
                vals.push_back(out.get(full));
            }
            if (kind == "quicksort")
            {
                std::sort(vals.begin(), vals.end());
            }
            else
            {
                std::make_heap(vals.begin(), vals.end());
                std::sort_heap(vals.begin(), vals.end());
            }
            for (int k = 0; k < out.shape[ax]; ++k)
            {
                std::vector<std::size_t> full;
                full.reserve(out.shape.size());
                for (std::size_t d = 0; d < out.shape.size(); ++d)
                {
                    full.push_back(d == static_cast<std::size_t>(ax) ? static_cast<std::size_t>(k)
                                                                     : static_cast<std::size_t>(idx[d < static_cast<std::size_t>(ax) ? d : d - 1]));
                }
                out.set(full, vals[static_cast<std::size_t>(k)]);
            }
            od.advance();
        }
        return out;
    }
    return a.sorted(axis);
}

NP_API template <typename T>
NP_NODISCARD inline auto argsort(const ndarray<T> &a, int axis, const std::string &kind) -> ndarray<std::size_t>
{
    if (kind != "quicksort" && kind != "mergesort" && kind != "heapsort" && kind != "stable")
    {
        throw std::invalid_argument("argsort: kind must be one of 'quicksort', 'mergesort', 'heapsort', 'stable'");
    }
    if (kind == "mergesort" || kind == "stable")
    {
        return a.argsort(axis);
    }
    // quicksort/heapsort: non-stable index sort implemented slice-wise here
    // so the requested (unstable) algorithm actually runs.
    const int ax = axis < 0 ? axis + static_cast<int>(a.ndim()) : axis;
    if (ax < 0 || ax >= static_cast<int>(a.ndim()))
    {
        throw std::invalid_argument("argsort: axis out of bounds");
    }
    ndarray<std::size_t> out(a.shape);
    std::vector<int> slice_shape = a.shape;
    slice_shape.erase(slice_shape.begin() + ax);
    np::detail::Odometer od(slice_shape.empty() ? std::vector<int>{1} : slice_shape);
    // Handle 1-D case where slice_shape is empty: single slice.
    auto get_val = [&](const std::vector<std::size_t> &outer, int k) -> T {
        std::vector<std::size_t> full;
        full.reserve(a.shape.size());
        if (a.ndim() == 1)
        {
            full.push_back(static_cast<std::size_t>(k));
        }
        else
        {
            for (std::size_t d = 0; d < a.shape.size(); ++d)
            {
                full.push_back(d == static_cast<std::size_t>(ax)
                                   ? static_cast<std::size_t>(k)
                                   : outer[d < static_cast<std::size_t>(ax) ? d : d - 1]);
            }
        }
        return a.get(full);
    };
    auto set_idx = [&](const std::vector<std::size_t> &outer, int k, std::size_t v) {
        std::vector<std::size_t> full;
        full.reserve(out.shape.size());
        if (out.ndim() == 1)
        {
            full.push_back(static_cast<std::size_t>(k));
        }
        else
        {
            for (std::size_t d = 0; d < out.shape.size(); ++d)
            {
                full.push_back(d == static_cast<std::size_t>(ax)
                                   ? static_cast<std::size_t>(k)
                                   : outer[d < static_cast<std::size_t>(ax) ? d : d - 1]);
            }
        }
        out.set(full, v);
    };
    if (a.ndim() == 1)
    {
        std::vector<std::size_t> perm(static_cast<std::size_t>(a.shape[0]));
        std::iota(perm.begin(), perm.end(), 0);
        auto cmp = [&](std::size_t i, std::size_t j) { return get_val({}, static_cast<int>(i)) < get_val({}, static_cast<int>(j)); };
        if (kind == "quicksort")
        {
            std::sort(perm.begin(), perm.end(), cmp);
        }
        else
        {
            std::make_heap(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) { return cmp(i, j); });
            std::sort_heap(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) { return cmp(i, j); });
        }
        for (int k = 0; k < a.shape[0]; ++k)
        {
            set_idx({}, k, perm[static_cast<std::size_t>(k)]);
        }
        return out;
    }
    while (!od.done())
    {
        const auto &outer = od.idx();
        std::vector<std::size_t> perm(static_cast<std::size_t>(a.shape[ax]));
        std::iota(perm.begin(), perm.end(), 0);
        auto cmp = [&](std::size_t i, std::size_t j) {
            return get_val(outer, static_cast<int>(i)) < get_val(outer, static_cast<int>(j));
        };
        if (kind == "quicksort")
        {
            std::sort(perm.begin(), perm.end(), cmp);
        }
        else
        {
            std::make_heap(perm.begin(), perm.end(), cmp);
            std::sort_heap(perm.begin(), perm.end(), cmp);
        }
        for (int k = 0; k < a.shape[ax]; ++k)
        {
            set_idx(outer, k, perm[static_cast<std::size_t>(k)]);
        }
        od.advance();
    }
    return out;
}

// `digitize`, `bincount`, `searchsorted` etc. live in statistics/sorting already;
// provide using aliases for taxonomy completeness
NP_API template <typename T>
NP_NODISCARD inline auto digitize_alias(const ndarray<T> &x, const ndarray<double> &bins) -> ndarray<std::size_t>
{
    // forward to statistics::digitize if available, else simple fallback
    // to avoid extra include, implement minimal here
    if (bins.ndim() != 1)
        throw std::invalid_argument("digitize: bins must be 1-D");
    // NumPy requires monotonic bins; sort a copy so unsorted input still
    // yields well-defined lower_bound results instead of garbage.
    std::vector<double> sorted(bins.data().begin(), bins.data().end());
    std::sort(sorted.begin(), sorted.end());
    ndarray<std::size_t> out(x.shape);
    std::size_t i = 0;
    for (auto it = x.begin(); it != x.end(); ++it, ++i)
    {
        double v = static_cast<double>(*it);
        auto it2 = std::lower_bound(sorted.begin(), sorted.end(), v);
        out.data()[i] = static_cast<std::size_t>(it2 - sorted.begin());
    }
    return out;
}

} // namespace np::inline v1

#endif // NP_SORTING_HPP
