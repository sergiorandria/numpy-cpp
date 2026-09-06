/**
 * @file logic.hpp
 * @brief Logic functions (truth value testing, type checks, comparisons).
 *
 * Provides NumPy-compatible logical operations:
 *   Type checks: isfinite, isinf, isnan, isnat, isreal, iscomplex, isfortran,
 *                isrealobj, iscomplexobj, isscalar
 *   Logical ops: logical_and, logical_or, logical_not, logical_xor, all, any
 *   Comparisons: allclose, isclose, array_equal, array_equiv
 *   Element-wise comparisons: greater, less, equal, not_equal, etc.
 *   Set/membership: isin, in1d, intersect1d, union1d, setdiff1d, setxor1d
 *
 * All functions return C-contiguous arrays with row-major strides.
 * Binary operations broadcast shapes according to NumPy rules.
 *
 * Reference: numpy-reference/reference/routines.logic.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_LOGIC_HPP
#define NP_LOGIC_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"
#include "pqc.hpp"

namespace np
{

// Type checks
// Reference: numpy-reference/reference/generated/numpy.isfinite.html (etc.)
/** @brief Test element-wise for finiteness (not infinity and not NaN).
 *
 * @tparam T  Element type (floating-point or complex).
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is finite.
 */
NP_API template <typename T> NP_NODISCARD auto isfinite(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    std::vector<std::size_t> idx(x.ndim(), 0);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        result.set(idx, std::isfinite(x.get(idx)));
        // Increment index
        for (std::size_t d = x.ndim(); d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(x.shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }
    return result;
}

/** @brief Test element-wise for positive or negative infinity.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is inf.
 */
NP_API template <typename T> NP_NODISCARD auto isinf(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    std::vector<std::size_t> idx(x.ndim(), 0);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        result.set(idx, std::isinf(x.get(idx)));
        for (std::size_t d = x.ndim(); d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(x.shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }
    return result;
}

/** @brief Test element-wise for NaN.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is NaN.
 */
NP_API template <typename T> NP_NODISCARD auto isnan(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    std::vector<std::size_t> idx(x.ndim(), 0);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        result.set(idx, std::isnan(x.get(idx)));
        for (std::size_t d = x.ndim(); d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(x.shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }
    return result;
}

/** @brief Test element-wise for negative infinity.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is -inf.
 */
NP_API template <typename T> NP_NODISCARD auto isneginf(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    std::vector<std::size_t> idx(x.ndim(), 0);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const T val = x.get(idx);
        result.set(idx, std::isinf(val) && (val < T{0}));
        for (std::size_t d = x.ndim(); d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(x.shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }
    return result;
}

/** @brief Test element-wise for positive infinity.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true where x[i] is +inf.
 */
NP_API template <typename T> NP_NODISCARD auto isposinf(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    std::vector<std::size_t> idx(x.ndim(), 0);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const T val = x.get(idx);
        result.set(idx, std::isinf(val) && (val > T{0}));
        for (std::size_t d = x.ndim(); d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(x.shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }
    return result;
}

/** @brief Returns True if input is complex.
 *
 * The result is uniform across all elements (the dtype
 * of the array determines whether elements are complex).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true for all elements if T
 *            is a complex type, false otherwise.
 */
NP_API template <typename T> NP_NODISCARD auto iscomplex(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    result.fill(detail::is_complex_v<T>);
    return result;
}

/** @brief Returns True if input is real (not complex).
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with true for all elements if T
 *            is not a complex type, false otherwise.
 */
NP_API template <typename T> NP_NODISCARD auto isreal(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    result.fill(!detail::is_complex_v<T>);
    return result;
}

/** @brief Returns True if input is a scalar type.
 *
 * @tparam T  Type to check.
 * @param x   Value (unused; only the type matters).
 * @return    True if T is arithmetic or a complex instantiation.
 */
NP_API template <typename T> constexpr bool isscalar([[maybe_unused]] const T &x)
{
    return std::is_arithmetic_v<T> || detail::is_complex_v<T>;
}

/** @brief True if array is Fortran contiguous.
 *
 * Wraps ndarray::is_f_contiguous(). Mirrors np.isfortran.
 */
NP_API template <typename T> NP_NODISCARD bool isfortran(const ndarray<T> &a)
{
    return a.is_f_contiguous();
}

/** @brief Test element-wise for NaT (Not a Time).
 *
 * This implementation has no datetime type, so it always returns False,
 * mirroring NumPy's behaviour for non-datetime dtypes.
 * Reference: numpy.isnat
 */
NP_API template <typename T> NP_NODISCARD auto isnat(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    result.fill(false);
    return result;
}

// Logical operations
// Reference: numpy-reference/reference/generated/numpy.logical_and.html (etc.)
/** @brief Compute truth value of x1 AND x2 element-wise.
 *
 * Broadcasts x1 and x2 to a common shape.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array (converted to bool).
 * @param x2  Second input array (converted to bool).
 * @return    ndarray<bool> with x1[i] && x2[i].
 * @throws    std::invalid_argument if shapes cannot be broadcast.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto logical_and(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    const auto out_shape = detail::broadcast_shapes(x1.shape, x2.shape);
    ndarray<bool> result(out_shape, dtype::bool_);

    const auto ndim_out = out_shape.size();
    std::vector<std::size_t> idx(ndim_out, 0);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        std::vector<std::size_t> idx1(x1.ndim(), 0);
        std::vector<std::size_t> idx2(x2.ndim(), 0);

        for (std::size_t d = 0; d < ndim_out; ++d)
        {
            if (d >= ndim_out - x1.ndim())
            {
                const auto d1 = d - (ndim_out - x1.ndim());
                idx1[d1] = (x1.shape[d1] == 1) ? 0 : idx[d];
            }
            if (d >= ndim_out - x2.ndim())
            {
                const auto d2 = d - (ndim_out - x2.ndim());
                idx2[d2] = (x2.shape[d2] == 1) ? 0 : idx[d];
            }
        }

        const bool val1 = static_cast<bool>(x1.get(idx1));
        const bool val2 = static_cast<bool>(x2.get(idx2));
        result.set(idx, val1 && val2);

        for (std::size_t d = ndim_out; d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(out_shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }

    return result;
}

/** @brief Compute truth value of x1 OR x2 element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] || x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto logical_or(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    const auto out_shape = detail::broadcast_shapes(x1.shape, x2.shape);
    ndarray<bool> result(out_shape, dtype::bool_);

    const auto ndim_out = out_shape.size();
    std::vector<std::size_t> idx(ndim_out, 0);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        std::vector<std::size_t> idx1(x1.ndim(), 0);
        std::vector<std::size_t> idx2(x2.ndim(), 0);

        for (std::size_t d = 0; d < ndim_out; ++d)
        {
            if (d >= ndim_out - x1.ndim())
            {
                const auto d1 = d - (ndim_out - x1.ndim());
                idx1[d1] = (x1.shape[d1] == 1) ? 0 : idx[d];
            }
            if (d >= ndim_out - x2.ndim())
            {
                const auto d2 = d - (ndim_out - x2.ndim());
                idx2[d2] = (x2.shape[d2] == 1) ? 0 : idx[d];
            }
        }

        const bool val1 = static_cast<bool>(x1.get(idx1));
        const bool val2 = static_cast<bool>(x2.get(idx2));
        result.set(idx, val1 || val2);

        for (std::size_t d = ndim_out; d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(out_shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }

    return result;
}

/** @brief Compute truth value of NOT x element-wise.
 *
 * @tparam T  Element type.
 * @param x   Input array.
 * @return    ndarray<bool> with !x[i].
 */
NP_API template <typename T> NP_NODISCARD auto logical_not(const ndarray<T> &x) -> ndarray<bool>
{
    ndarray<bool> result(x.shape, dtype::bool_);
    std::vector<std::size_t> idx(x.ndim(), 0);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        result.set(idx, !static_cast<bool>(x.get(idx)));
        for (std::size_t d = x.ndim(); d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(x.shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }
    return result;
}

/** @brief Compute truth value of x1 XOR x2 element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with (x1[i] != x2[i]).
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto logical_xor(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    const auto out_shape = detail::broadcast_shapes(x1.shape, x2.shape);
    ndarray<bool> result(out_shape, dtype::bool_);

    const auto ndim_out = out_shape.size();
    std::vector<std::size_t> idx(ndim_out, 0);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        std::vector<std::size_t> idx1(x1.ndim(), 0);
        std::vector<std::size_t> idx2(x2.ndim(), 0);

        for (std::size_t d = 0; d < ndim_out; ++d)
        {
            if (d >= ndim_out - x1.ndim())
            {
                const auto d1 = d - (ndim_out - x1.ndim());
                idx1[d1] = (x1.shape[d1] == 1) ? 0 : idx[d];
            }
            if (d >= ndim_out - x2.ndim())
            {
                const auto d2 = d - (ndim_out - x2.ndim());
                idx2[d2] = (x2.shape[d2] == 1) ? 0 : idx[d];
            }
        }

        const bool val1 = static_cast<bool>(x1.get(idx1));
        const bool val2 = static_cast<bool>(x2.get(idx2));
        result.set(idx, val1 != val2);

        for (std::size_t d = ndim_out; d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(out_shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }

    return result;
}

// Comparison functions
// Reference: numpy-reference/reference/generated/numpy.greater.html (etc.)
/** @brief Return (x1 > x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] > x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto greater(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    return x1 > x2;
}

/** @brief Return (x1 >= x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] >= x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto greater_equal(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    return x1 >= x2;
}

/** @brief Return (x1 < x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] < x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto less(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    return x1 < x2;
}

/** @brief Return (x1 <= x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] <= x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto less_equal(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    return x1 <= x2;
}

/** @brief Return (x1 == x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] == x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto equal(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    return x1 == x2;
}

/** @brief Return (x1 != x2) element-wise.
 *
 * @tparam T  Element type of x1.
 * @tparam U  Element type of x2.
 * @param x1  First input array.
 * @param x2  Second input array.
 * @return    ndarray<bool> with x1[i] != x2[i].
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto not_equal(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<bool>
{
    return x1 != x2;
}

// Free reductions all/any
/** @brief True if all elements are truthy (free fn). */
NP_API template <typename T> NP_NODISCARD bool all(const ndarray<T> &a)
{
    return a.all();
}

/** @brief All along an axis (free fn). */
NP_API template <typename T>
NP_NODISCARD auto all(const ndarray<T> &a, int axis, bool keepdims = false) -> ndarray<bool>
{
    return a.all(axis, keepdims);
}

/** @brief True if any element is truthy (free fn). */
NP_API template <typename T> NP_NODISCARD bool any(const ndarray<T> &a)
{
    return a.any();
}

/** @brief Any along an axis (free fn). */
NP_API template <typename T>
NP_NODISCARD auto any(const ndarray<T> &a, int axis, bool keepdims = false) -> ndarray<bool>
{
    return a.any(axis, keepdims);
}

// Type object checks
/** @brief True if dtype is complex (object check). */
NP_API template <typename T> constexpr bool iscomplexobj(const T &obj)
{
    return detail::is_complex_v<std::decay_t<T>>;
}

/** @brief Overload for ndarray: delegates to iscomplex. */
NP_API template <typename T> NP_NODISCARD auto iscomplexobj(const ndarray<T> &a) -> ndarray<bool>
{
    return iscomplex(a);
}

/** @brief True if dtype is real (object check). */
NP_API template <typename T> constexpr bool isrealobj(const T &obj)
{
    return !detail::is_complex_v<std::decay_t<T>>;
}

/** @brief Overload for ndarray: delegates to isreal. */
NP_API template <typename T> NP_NODISCARD auto isrealobj(const ndarray<T> &a) -> ndarray<bool>
{
    return isreal(a);
}

// Membership / set operations
/** @brief Test element-wise membership (np.isin / np.in1d).
 *
 * @tparam T Type of elements.
 * @tparam U Type of test elements.
 * @param element 1-D or ND array to test.
 * @param test_elements 1-D array of values to test against.
 * @param invert If true, invert result.
 * @return ndarray<bool> same shape as element.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto isin(const ndarray<T> &element, const ndarray<U> &test_elements, bool invert = false) -> ndarray<bool>
{
    if (test_elements.size() == 0) [[unlikely]]
    {
        ndarray<bool> out(element.shape);
        out.fill(invert ? true : false);
        return out;
    }
    ndarray<bool> out(element.shape);
    // Micro-opt: hash for large test_elements, sort+binary for small
    constexpr std::size_t HASH_THRESHOLD = 64;
    if (test_elements.size() > HASH_THRESHOLD)
    {
        std::unordered_set<U> set;
        set.reserve(test_elements.size() * 2);
        for (std::size_t i = 0; i < test_elements.size(); ++i)
            set.insert(test_elements.data()[test_elements._flat_logical(i)]);
        if (element.is_contiguous()) [[likely]]
        {
            auto &d_vec = out.data();
            if constexpr (std::is_same_v<T, bool>)
            {
                auto &s_vec = element.data();
                std::size_t n = element.size();
                for (std::size_t i = 0; i < n; ++i)
                {
                    bool found = set.find(static_cast<U>(static_cast<bool>(s_vec[i]))) != set.end();
                    d_vec[i] = invert ? !found : found;
                }
            }
            else
            {
                const T *__restrict s = element.data().data();
                std::size_t n = element.size();
                for (std::size_t i = 0; i < n; ++i)
                {
                    bool found = set.find(static_cast<U>(s[i])) != set.end();
                    d_vec[i] = invert ? !found : found;
                }
            }
            return out;
        }
        for (std::size_t i = 0; i < element.size(); ++i)
        {
            T v = element.data()[element._flat_logical(i)];
            bool found = set.find(static_cast<U>(v)) != set.end();
            out.data()[i] = invert ? !found : found;
        }
        return out;
    }
    std::vector<U> sorted;
    sorted.reserve(test_elements.size());
    for (std::size_t i = 0; i < test_elements.size(); ++i)
        sorted.push_back(test_elements.data()[test_elements._flat_logical(i)]);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    if (element.is_contiguous()) [[likely]]
    {
        auto &d_vec = out.data();
        if constexpr (std::is_same_v<T, bool>)
        {
            auto &s_vec = element.data();
            std::size_t n = element.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                bool found =
                    std::binary_search(sorted.begin(), sorted.end(), static_cast<U>(static_cast<bool>(s_vec[i])));
                d_vec[i] = invert ? !found : found;
            }
        }
        else
        {
            const T *__restrict s = element.data().data();
            std::size_t n = element.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                bool found = std::binary_search(sorted.begin(), sorted.end(), static_cast<U>(s[i]));
                d_vec[i] = invert ? !found : found;
            }
        }
        return out;
    }
    for (std::size_t i = 0; i < element.size(); ++i)
    {
        T v = element.data()[element._flat_logical(i)];
        bool found = std::binary_search(sorted.begin(), sorted.end(), static_cast<U>(v));
        out.data()[i] = invert ? !found : found;
    }
    return out;
}

/** @brief Alias for 1-D isin (np.in1d). */
NP_API template <typename T, typename U>
NP_NODISCARD auto in1d(const ndarray<T> &ar1, const ndarray<U> &ar2, bool invert = false) -> ndarray<bool>
{
    if (ar1.ndim() != 1 || ar2.ndim() != 1)
        throw std::invalid_argument("in1d: both arrays must be 1-D");
    return isin(ar1, ar2, invert);
}

/** @brief Sorted unique intersection (np.intersect1d). */
NP_API template <typename T> NP_NODISCARD auto intersect1d(const ndarray<T> &ar1, const ndarray<T> &ar2) -> ndarray<T>
{
    if (ar1.size() == 0 || ar2.size() == 0) [[unlikely]]
        return ndarray<T>(std::vector<int>{0});
    std::vector<T> a;
    std::vector<T> b;
    a.reserve(ar1.size());
    b.reserve(ar2.size());
    for (std::size_t i = 0; i < ar1.size(); ++i)
        a.push_back(ar1.data()[ar1._flat_logical(i)]);
    for (std::size_t i = 0; i < ar2.size(); ++i)
        b.push_back(ar2.data()[ar2._flat_logical(i)]);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    std::vector<T> res;
    res.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(res));
    ndarray<T> out(std::vector<int>{static_cast<int>(res.size())});
    T *__restrict dst = out.data().data();
    for (std::size_t i = 0; i < res.size(); ++i)
        dst[i] = res[i];
    return out;
}

/** @brief Sorted union (np.union1d). */
NP_API template <typename T> NP_NODISCARD auto union1d(const ndarray<T> &ar1, const ndarray<T> &ar2) -> ndarray<T>
{
    std::vector<T> a(ar1.size()), b(ar2.size());
    for (std::size_t i = 0; i < ar1.size(); ++i)
        a[i] = ar1.data()[ar1._flat_logical(i)];
    for (std::size_t i = 0; i < ar2.size(); ++i)
        b[i] = ar2.data()[ar2._flat_logical(i)];
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    std::vector<T> res;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(res));
    ndarray<T> out(std::vector<int>{static_cast<int>(res.size())});
    for (std::size_t i = 0; i < res.size(); ++i)
        out.data()[i] = res[i];
    return out;
}

/** @brief Set difference (np.setdiff1d): values in ar1 not in ar2. */
NP_API template <typename T> NP_NODISCARD auto setdiff1d(const ndarray<T> &ar1, const ndarray<T> &ar2) -> ndarray<T>
{
    std::vector<T> a(ar1.size()), b(ar2.size());
    for (std::size_t i = 0; i < ar1.size(); ++i)
        a[i] = ar1.data()[ar1._flat_logical(i)];
    for (std::size_t i = 0; i < ar2.size(); ++i)
        b[i] = ar2.data()[ar2._flat_logical(i)];
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    std::vector<T> res;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(res));
    ndarray<T> out(std::vector<int>{static_cast<int>(res.size())});
    for (std::size_t i = 0; i < res.size(); ++i)
        out.data()[i] = res[i];
    return out;
}

/** @brief Symmetric difference (np.setxor1d). */
NP_API template <typename T> NP_NODISCARD auto setxor1d(const ndarray<T> &ar1, const ndarray<T> &ar2) -> ndarray<T>
{
    std::vector<T> a(ar1.size()), b(ar2.size());
    for (std::size_t i = 0; i < ar1.size(); ++i)
        a[i] = ar1.data()[ar1._flat_logical(i)];
    for (std::size_t i = 0; i < ar2.size(); ++i)
        b[i] = ar2.data()[ar2._flat_logical(i)];
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    std::vector<T> res;
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(res));
    ndarray<T> out(std::vector<int>{static_cast<int>(res.size())});
    for (std::size_t i = 0; i < res.size(); ++i)
        out.data()[i] = res[i];
    return out;
}

// ── NumPy 2.0 unique_* variants (np.unique_all etc.) ───────────────
/**
 * @brief Unique with all auxiliary outputs (np.unique_all).
 *
 * Reference: numpy-reference/reference/generated/numpy.unique_all.html
 */
NP_API template <typename T>
NP_NODISCARD inline auto unique_all(const ndarray<T> &ar)
    -> std::tuple<ndarray<T>, ndarray<std::size_t>, ndarray<std::size_t>, ndarray<std::size_t>>
{
    std::size_t n = ar.size();
    if (n == 0)
    {
        return {ndarray<T>(std::vector<int>{0}), ndarray<std::size_t>(std::vector<int>{0}),
                ndarray<std::size_t>(std::vector<int>{0}), ndarray<std::size_t>(std::vector<int>{0})};
    }
    std::vector<T> flat(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        flat[i] = ar.data()[ar._flat_logical(i)];
    }
    std::vector<T> vals = flat;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    std::size_t m = vals.size();
    ndarray<T> values(std::vector<int>{static_cast<int>(m)});
    for (std::size_t i = 0; i < m; ++i)
    {
        values.data()[i] = vals[i];
    }
    // counts and first-index
    ndarray<std::size_t> counts(std::vector<int>{static_cast<int>(m)});
    ndarray<std::size_t> indices(std::vector<int>{static_cast<int>(m)});
    for (std::size_t i = 0; i < m; ++i)
    {
        counts.data()[i] = 0;
        indices.data()[i] = n; // sentinel
    }
    // map value -> position via binary search (vals sorted)
    for (std::size_t i = 0; i < n; ++i)
    {
        T v = flat[i];
        std::size_t pos = static_cast<std::size_t>(std::lower_bound(vals.begin(), vals.end(), v) - vals.begin());
        counts.data()[pos]++;
        if (indices.data()[pos] == n)
        {
            indices.data()[pos] = i; // first occurrence
        }
    }
    // inverse: for each original element, index into values
    ndarray<std::size_t> inverse(std::vector<int>{static_cast<int>(n)});
    for (std::size_t i = 0; i < n; ++i)
    {
        T v = flat[i];
        std::size_t pos = static_cast<std::size_t>(std::lower_bound(vals.begin(), vals.end(), v) - vals.begin());
        inverse.data()[i] = pos;
    }
    return {values, indices, inverse, counts};
}

NP_API template <typename T> NP_NODISCARD inline auto unique_values(const ndarray<T> &ar) -> ndarray<T>
{
    std::vector<T> vals(ar.size());
    for (std::size_t i = 0; i < ar.size(); ++i)
        vals[i] = ar.data()[ar._flat_logical(i)];
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    ndarray<T> out(std::vector<int>{static_cast<int>(vals.size())});
    for (std::size_t i = 0; i < vals.size(); ++i)
        out.data()[i] = vals[i];
    return out;
}

NP_API template <typename T>
NP_NODISCARD inline auto unique_counts(const ndarray<T> &ar) -> std::pair<ndarray<T>, ndarray<std::size_t>>
{
    auto all = unique_all(ar);
    return {std::get<0>(all), std::get<3>(all)};
}

NP_API template <typename T>
NP_NODISCARD inline auto unique_inverse(const ndarray<T> &ar) -> std::pair<ndarray<T>, ndarray<std::size_t>>
{
    auto all = unique_all(ar);
    return {std::get<0>(all), std::get<2>(all)};
}

// Array comparison
// Reference: numpy-reference/reference/generated/numpy.array_equal.html (etc.)
/** @brief True if two arrays have the same shape and elements.
 *
 * Time complexity: O(N) where N is the total element count.
 *
 * @tparam T  Element type of a1.
 * @tparam U  Element type of a2.
 * @param a1  First array.
 * @param a2  Second array.
 * @return    True if shapes match and all elements are equal.
 */
NP_API template <typename T, typename U> bool array_equal(const ndarray<T> &a1, const ndarray<U> &a2)
{
    if (a1.shape != a2.shape)
    {
        return false;
    }
    auto it1 = a1.begin();
    auto it2 = a2.begin();
    for (; it1 != a1.end(); ++it1, ++it2)
    {
        if (*it1 != static_cast<T>(*it2))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Constant-time array equality (PQC-hardened).
 *
 * Compares raw bytes with `pqc::ct_memequal` (branch-free, no
 * secret-dependent early exit). Returns 1 if equal, 0 otherwise.
 * For non-contiguous arrays falls back to element-wise constant-time
 * accumulation. Prevents timing oracle on key material (ML-KEM/ML-DSA).
 *
 * Reference: pqc.hpp:ct_memequal, NIST FIPS 203
 * @tparam T Element type of a1.
 * @tparam U Element type of a2.
 * @param a1 First array.
 * @param a2 Second array.
 * @return true if byte-wise equal in constant time.
 */
NP_API template <typename T, typename U> bool array_equal_ct(const ndarray<T> &a1, const ndarray<U> &a2) noexcept
{
    if (a1.shape != a2.shape || a1.nbytes() != a2.nbytes())
    {
        return false;
    }
    if constexpr (std::is_same_v<T, U>)
    {
        if (a1.is_contiguous() && a2.is_contiguous())
        {
            pqc::ct_barrier();
            int eq = pqc::ct_memequal(static_cast<const void *>(a1.data().data()),
                                      static_cast<const void *>(a2.data().data()), a1.nbytes());
            pqc::ct_barrier();
            return eq == 1;
        }
    }
    // Fallback: constant-time accumulation over logical elements
    pqc::ct_barrier();
    unsigned char diff = 0;
    for (std::size_t i = 0; i < a1.size(); ++i)
    {
        T v1 = a1.data()[a1._flat_logical(i)];
        U v2 = a2.data()[a2._flat_logical(i)];
        // compare byte-wise via memcmp-style accumulation
        const unsigned char *p1 = reinterpret_cast<const unsigned char *>(&v1);
        const unsigned char *p2 = reinterpret_cast<const unsigned char *>(&v2);
        for (std::size_t b = 0; b < sizeof(T) && b < sizeof(U); ++b)
        {
            diff |= p1[b] ^ p2[b];
        }
        if constexpr (sizeof(T) != sizeof(U))
        {
            // different sizes → not equal if sizes differ
            if (sizeof(T) != sizeof(U))
            {
                diff |= 1;
            }
        }
    }
    int eq = pqc::ct_eq_u32(static_cast<std::uint32_t>(diff), 0);
    pqc::ct_barrier();
    return eq == 1;
}

/**
 * @brief Constant-time broadcasted equality (PQC-hardened array_equiv).
 *
 * Broadcasts like `array_equiv` but accumulates differences in constant
 * time without early exit. Returns false if shapes are not broadcastable.
 * @tparam T Element type.
 * @tparam U Element type.
 * @return true if broadcast-equal in constant time.
 */
NP_API template <typename T, typename U> bool array_equiv_ct(const ndarray<T> &a1, const ndarray<U> &a2) noexcept
{
    try
    {
        const auto out_shape = detail::broadcast_shapes(a1.shape, a2.shape);
        const auto ndim_out = out_shape.size();
        std::size_t total_elems = 1;
        for (int d : out_shape)
        {
            total_elems *= static_cast<std::size_t>(d);
        }
        pqc::ct_barrier();
        unsigned char diff = 0;
        std::vector<std::size_t> idx(ndim_out, 0);
        for (std::size_t i = 0; i < total_elems; ++i)
        {
            std::vector<std::size_t> idx1(a1.ndim(), 0);
            std::vector<std::size_t> idx2(a2.ndim(), 0);
            for (std::size_t d = 0; d < ndim_out; ++d)
            {
                if (d >= ndim_out - a1.ndim())
                {
                    const auto d1 = d - (ndim_out - a1.ndim());
                    idx1[d1] = (a1.shape[d1] == 1) ? 0 : idx[d];
                }
                if (d >= ndim_out - a2.ndim())
                {
                    const auto d2 = d - (ndim_out - a2.ndim());
                    idx2[d2] = (a2.shape[d2] == 1) ? 0 : idx[d];
                }
            }
            T v1 = a1.get(idx1);
            U v2 = static_cast<U>(a2.get(idx2));
            // constant-time byte compare
            const unsigned char *p1 = reinterpret_cast<const unsigned char *>(&v1);
            const unsigned char *p2 = reinterpret_cast<const unsigned char *>(&v2);
            for (std::size_t b = 0; b < sizeof(T) && b < sizeof(U); ++b)
            {
                diff |= p1[b] ^ p2[b];
            }
            if constexpr (sizeof(T) != sizeof(U))
            {
                diff |= 1;
            }
            for (std::size_t d = ndim_out; d-- > 0;)
            {
                if (++idx[d] < static_cast<std::size_t>(out_shape[d]))
                {
                    break;
                }
                idx[d] = 0;
            }
        }
        pqc::ct_barrier();
        return pqc::ct_eq_u32(static_cast<std::uint32_t>(diff), 0) == 1;
    }
    catch (...)
    {
        return false;
    }
}

/** @brief True if two arrays are element-wise equal within a tolerance.
 *
 * Uses the standard absolute + relative tolerance formula:
 *   |a - b| <= atol + rtol * |b|
 *
 * @tparam T  Element type of a.
 * @tparam U  Element type of b.
 * @param a   First array.
 * @param b   Second array.
 * @param rtol Relative tolerance (default: 1e-5).
 * @param atol Absolute tolerance (default: 1e-8).
 * @return     ndarray<bool> with true where elements are close.
 * @throws     std::invalid_argument if shapes cannot be broadcast.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto isclose(const ndarray<T> &a, const ndarray<U> &b, double rtol = 1e-5, double atol = 1e-8)
    -> ndarray<bool>
{
    const auto out_shape = detail::broadcast_shapes(a.shape, b.shape);
    ndarray<bool> result(out_shape, dtype::bool_);

    const auto ndim_out = out_shape.size();
    std::vector<std::size_t> idx(ndim_out, 0);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        std::vector<std::size_t> idx_a(a.ndim(), 0);
        std::vector<std::size_t> idx_b(b.ndim(), 0);

        for (std::size_t d = 0; d < ndim_out; ++d)
        {
            if (d >= ndim_out - a.ndim())
            {
                const auto da = d - (ndim_out - a.ndim());
                idx_a[da] = (a.shape[da] == 1) ? 0 : idx[d];
            }
            if (d >= ndim_out - b.ndim())
            {
                const auto db = d - (ndim_out - b.ndim());
                idx_b[db] = (b.shape[db] == 1) ? 0 : idx[d];
            }
        }

        const double val_a = static_cast<double>(a.get(idx_a));
        const double val_b = static_cast<double>(b.get(idx_b));
        const double diff = std::abs(val_a - val_b);
        const double threshold = atol + rtol * std::abs(val_b);

        result.set(idx, diff <= threshold);

        for (std::size_t d = ndim_out; d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(out_shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }

    return result;
}

/** @brief True if two arrays are element-wise equal within a tolerance.
 *
 * Reduces isclose() to a single boolean via .all().
 *
 * @tparam T  Element type of a.
 * @tparam U  Element type of b.
 * @param a   First array.
 * @param b   Second array.
 * @param rtol Relative tolerance (default: 1e-5).
 * @param atol Absolute tolerance (default: 1e-8).
 * @return     True if all elements are close.
 */
NP_API template <typename T, typename U>
bool allclose(const ndarray<T> &a, const ndarray<U> &b, double rtol = 1e-5, double atol = 1e-8)
{
    return isclose(a, b, rtol, atol).all();
}

/** @brief True if two arrays are broadcastable and element-wise equal.
 *
 * Unlike array_equal(), this function broadcasts shapes before
 * comparing. Returns false if broadcasting fails.
 *
 * @tparam T  Element type of a1.
 * @tparam U  Element type of a2.
 * @param a1  First array.
 * @param a2  Second array.
 * @return    True if arrays are broadcast-equal.
 */
NP_API template <typename T, typename U> bool array_equiv(const ndarray<T> &a1, const ndarray<U> &a2)
{
    try
    {
        const auto out_shape = detail::broadcast_shapes(a1.shape, a2.shape);
        const auto ndim_out = out_shape.size();
        std::vector<std::size_t> idx(ndim_out, 0);

        // Compute total elements
        std::size_t total_elems = 1;
        for (int d : out_shape)
        {
            total_elems *= static_cast<std::size_t>(d);
        }

        for (std::size_t i = 0; i < total_elems; ++i)
        {
            std::vector<std::size_t> idx1(a1.ndim(), 0);
            std::vector<std::size_t> idx2(a2.ndim(), 0);

            for (std::size_t d = 0; d < ndim_out; ++d)
            {
                if (d >= ndim_out - a1.ndim())
                {
                    const auto d1 = d - (ndim_out - a1.ndim());
                    idx1[d1] = (a1.shape[d1] == 1) ? 0 : idx[d];
                }
                if (d >= ndim_out - a2.ndim())
                {
                    const auto d2 = d - (ndim_out - a2.ndim());
                    idx2[d2] = (a2.shape[d2] == 1) ? 0 : idx[d];
                }
            }

            if (a1.get(idx1) != static_cast<T>(a2.get(idx2)))
            {
                return false;
            }

            for (std::size_t d = ndim_out; d-- > 0;)
            {
                if (++idx[d] < static_cast<std::size_t>(out_shape[d]))
                {
                    break;
                }
                idx[d] = 0;
            }
        }

        return true;
    }
    catch (...)
    {
        return false;
    }
}

} // namespace np

#endif // NP_LOGIC_HPP
