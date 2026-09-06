/**
 * @file concatenate.hpp
 * @brief Array concatenation and stacking routines.
 *
 * Provides NumPy-compatible joining operations:
 *   concatenate, stack, vstack, hstack, dstack, column_stack, row_stack
 *
 * All functions return C-contiguous arrays with row-major strides.
 * Shape mismatches throw std::invalid_argument at runtime.
 *
 * Reference: numpy-reference/reference/routines.array-manipulation.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_CONCATENATE_HPP
#define NP_CONCATENATE_HPP

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{

// Internal helpers
namespace detail
{

/** @brief Advance an odometer-style multi-index through `shape`.
 *
 * Called after every element visit so a row-major traversal visits each
 * index once. Returns false when the index wraps past the last element
 * (all dimensions back to zero), signalling the end of the traversal.
 *
 * @param idx    Multi-index to advance (in place).
 * @param shape  Extents of each dimension.
 * @return       true while `idx` still addresses a valid element.
 */
inline bool advance_multi_index(std::vector<std::size_t> &idx, const std::vector<int> &shape) noexcept
{
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d)
    {
        const std::size_t dim = static_cast<std::size_t>(d);
        ++idx[dim];
        if (idx[dim] < static_cast<std::size_t>(shape[dim]))
        {
            return true;
        }
        idx[dim] = 0;
    }
    return false;
}

/** @brief Insert one axis value into a source multi-index.
 *
 * Builds the destination index produced by np::stack: the axes before
 * `axis` are copied from `idx_in`, the new axis gets `pos`, and the
 * remaining axes follow.
 */
inline std::vector<std::size_t> insert_axis_index(const std::vector<std::size_t> &idx_in, int ndim, int axis,
                                                  std::size_t pos)
{
    std::vector<std::size_t> out;
    out.reserve(static_cast<std::size_t>(ndim) + 1);
    for (int d = 0; d < axis; ++d)
    {
        out.push_back(idx_in[static_cast<std::size_t>(d)]);
    }
    out.push_back(pos);
    for (int d = axis; d < ndim; ++d)
    {
        out.push_back(idx_in[static_cast<std::size_t>(d)]);
    }
    return out;
}

} // namespace detail

// Concatenate
// Reference: numpy-reference/reference/generated/numpy.concatenate.html
/** @brief Join a sequence of arrays along an existing axis.
 *
 * All arrays must have the same shape except in the
 * concatenation axis. The output shape matches the input
 * shape with the concatenation axis size being the sum of
 * all input sizes along that axis.
 *
 * Time complexity: O(N) where N is the total number of
 * elements across all arrays. Space complexity: O(N) for
 * the output array.
 *
 * @tparam T  Element type.
 * @param arrays Sequence of arrays to concatenate.
 * @param axis   Axis along which to concatenate (default: 0).
 * @return       ndarray<T> with concatenated data.
 * @throws       std::invalid_argument if arrays is empty,
 *               shapes are incompatible, or axis is out of bounds.
 *
 * Reference: numpy-reference/reference/generated/numpy.concatenate.html
 */
template <typename T> NP_NODISCARD auto concatenate(const std::vector<ndarray<T>> &arrays, int axis = 0) -> ndarray<T>
{
    if (arrays.empty())
    {
        throw std::invalid_argument("concatenate: need at least one array");
    }

    const auto &first = arrays[0];
    const int ndim = static_cast<int>(first.ndim());

    if (axis < 0)
    {
        axis += ndim;
    }
    if (axis < 0 || axis >= ndim)
    {
        throw std::invalid_argument("concatenate: axis out of bounds");
    }

    // Check shape compatibility
    for (std::size_t i = 1; i < arrays.size(); ++i)
    {
        if (arrays[i].ndim() != first.ndim())
        {
            throw std::invalid_argument("concatenate: all arrays must have same ndim");
        }
        for (int d = 0; d < ndim; ++d)
        {
            if (d != axis && arrays[i].shape[d] != first.shape[d])
            {
                throw std::invalid_argument("concatenate: shapes must match on non-concat axis");
            }
        }
    }

    // Compute output shape
    std::vector<int> out_shape = first.shape;
    for (std::size_t i = 1; i < arrays.size(); ++i)
    {
        out_shape[axis] += arrays[i].shape[axis];
    }

    // Allocate output
    ndarray<T> result(out_shape, first.type);

    // Copy data
    std::size_t offset = 0;
    for (const auto &arr : arrays)
    {
        const std::size_t axis_size = static_cast<std::size_t>(arr.shape[axis]);

        // Copy all elements from this array
        std::vector<std::size_t> src_idx(ndim, 0);

        do
        {
            auto dst_idx = src_idx;
            dst_idx[axis] += offset;
            result.set(dst_idx, arr.get(src_idx));
        } while (detail::advance_multi_index(src_idx, arr.shape));

        offset += axis_size;
    }

    return result;
}

// Stack
// Reference: numpy-reference/reference/generated/numpy.stack.html
/** @brief Join a sequence of arrays along a new axis.
 *
 * All arrays must have the same shape. The output has one
 * additional dimension compared to the inputs.
 *
 * Time complexity: O(N * k) where N is the number of
 * elements per array and k is the number of arrays.
 * Space complexity: O(N * k) for the output array.
 *
 * @tparam T  Element type.
 * @param arrays Sequence of arrays to stack.
 * @param axis   Position where new axis is inserted (default: 0).
 * @return       ndarray<T> with stacked data.
 * @throws       std::invalid_argument if arrays is empty,
 *               shapes are incompatible, or axis is out of bounds.
 *
 * Reference: numpy-reference/reference/generated/numpy.stack.html
 */
template <typename T> NP_API NP_NODISCARD auto stack(const std::vector<ndarray<T>> &arrays, int axis = 0) -> ndarray<T>
{
    if (arrays.empty())
    {
        throw std::invalid_argument("stack: need at least one array");
    }

    const auto &first = arrays[0];
    const int ndim = static_cast<int>(first.ndim());

    // Normalize axis
    if (axis < 0)
    {
        axis += ndim + 1;
    }
    if (axis < 0 || axis > ndim)
    {
        throw std::invalid_argument("stack: axis out of bounds");
    }

    // Check all arrays have same shape
    for (std::size_t i = 1; i < arrays.size(); ++i)
    {
        if (arrays[i].shape != first.shape)
        {
            throw std::invalid_argument("stack: all arrays must have same shape");
        }
    }

    // Compute output shape (insert new dimension)
    std::vector<int> out_shape;
    out_shape.reserve(ndim + 1);
    for (int d = 0; d < axis; ++d)
    {
        out_shape.push_back(first.shape[d]);
    }
    out_shape.push_back(static_cast<int>(arrays.size()));
    for (int d = axis; d < ndim; ++d)
    {
        out_shape.push_back(first.shape[d]);
    }

    // Allocate output
    ndarray<T> result(out_shape, first.type);

    // Copy data
    for (std::size_t i = 0; i < arrays.size(); ++i)
    {
        std::vector<std::size_t> idx_in(ndim, 0);

        do
        {
            result.set(detail::insert_axis_index(idx_in, ndim, axis, i), arrays[i].get(idx_in));
        } while (detail::advance_multi_index(idx_in, first.shape));
    }

    return result;
}

// Convenience stacking functions
// Reference: numpy-reference/reference/generated/numpy.vstack.html (etc.)
/** @brief Stack arrays vertically (row-wise).
 *
 * Equivalent to concatenate(arrays, axis=0) for 2D+ arrays.
 * For 1D arrays, stacks them as rows into a 2D array.
 *
 * @tparam T  Element type.
 * @param arrays Sequence of arrays to stack.
 * @return       ndarray<T> with stacked data.
 * @throws       std::invalid_argument if arrays is empty.
 */
template <typename T> NP_API NP_NODISCARD auto vstack(const std::vector<ndarray<T>> &arrays) -> ndarray<T>
{
    if (arrays.empty())
    {
        throw std::invalid_argument("vstack: need at least one array");
    }

    // If 1D, reshape to (1, N) first
    std::vector<ndarray<T>> reshaped;
    reshaped.reserve(arrays.size());

    for (const auto &arr : arrays)
    {
        if (arr.ndim() == 1)
        {
            reshaped.push_back(arr.reshape({1, arr.shape[0]}));
        }
        else
        {
            reshaped.push_back(arr);
        }
    }

    return concatenate(reshaped, 0);
}

/** @brief Stack arrays horizontally (column-wise).
 *
 * Equivalent to concatenate(arrays, axis=1) for 2D+ arrays.
 * For 1D arrays, concatenates them into a single 1D array.
 *
 * @tparam T  Element type.
 * @param arrays Sequence of arrays to stack.
 * @return       ndarray<T> with stacked data.
 * @throws       std::invalid_argument if arrays is empty.
 */
template <typename T> NP_API NP_NODISCARD auto hstack(const std::vector<ndarray<T>> &arrays) -> ndarray<T>
{
    if (arrays.empty())
    {
        throw std::invalid_argument("hstack: need at least one array");
    }

    if (arrays[0].ndim() == 1)
    {
        return concatenate(arrays, 0);
    }

    return concatenate(arrays, 1);
}

/** @brief Stack arrays depth-wise (along third axis).
 *
 * Takes a sequence of arrays and stacks them along the
 * third axis. 1D or 2D arrays are first reshaped to
 * (M, N, 1).
 *
 * @tparam T  Element type.
 * @param arrays Sequence of arrays to stack.
 * @return       ndarray<T> with stacked data.
 * @throws       std::invalid_argument if arrays is empty
 *               or contains arrays with ndim > 2.
 */
template <typename T> NP_API NP_NODISCARD auto dstack(const std::vector<ndarray<T>> &arrays) -> ndarray<T>
{
    if (arrays.empty())
    {
        throw std::invalid_argument("dstack: need at least one array");
    }

    std::vector<ndarray<T>> reshaped;
    reshaped.reserve(arrays.size());

    for (const auto &arr : arrays)
    {
        if (arr.ndim() == 1)
        {
            reshaped.push_back(arr.reshape({1, arr.shape[0], 1}));
        }
        else if (arr.ndim() == 2)
        {
            reshaped.push_back(arr.reshape({arr.shape[0], arr.shape[1], 1}));
        }
        else
        {
            reshaped.push_back(arr);
        }
    }

    return concatenate(reshaped, 2);
}

/** @brief Stack 1D arrays as columns into a 2D array.
 *
 * Each 1D array becomes a column of the output 2D array.
 * 2D arrays are used as-is.
 *
 * @tparam T  Element type.
 * @param arrays Sequence of 1D or 2D arrays.
 * @return       ndarray<T> with shape (N, K) where K is
 *               the number of arrays and N is the length
 *               of each 1D array (or the row count of 2D arrays).
 * @throws       std::invalid_argument if arrays is empty
 *               or contains arrays with ndim > 2.
 */
#ifndef NP_MANIPULATION_HPP
template <typename T> NP_API NP_NODISCARD auto column_stack(const std::vector<ndarray<T>> &arrays) -> ndarray<T>
{
    if (arrays.empty())
    {
        throw std::invalid_argument("column_stack: need at least one array");
    }

    // Reshape 1D arrays to (N, 1)
    std::vector<ndarray<T>> reshaped;
    reshaped.reserve(arrays.size());

    for (const auto &arr : arrays)
    {
        if (arr.ndim() == 1)
        {
            reshaped.push_back(arr.reshape({arr.shape[0], 1}));
        }
        else if (arr.ndim() == 2)
        {
            reshaped.push_back(arr);
        }
        else
        {
            throw std::invalid_argument("column_stack: arrays must be 1D or 2D");
        }
    }

    return concatenate(reshaped, 1);
}

/** @brief Stack 1D arrays as rows into a 2D array.
 *
 * Equivalent to vstack for 1D arrays.
 *
 * @tparam T  Element type.
 * @param arrays Sequence of 1D arrays.
 * @return       ndarray<T> with shape (K, N) where K is
 *               the number of arrays and N is the length
 *               of each array.
 */
template <typename T> NP_API NP_NODISCARD auto row_stack(const std::vector<ndarray<T>> &arrays) -> ndarray<T>
{
    return vstack(arrays);
}

/**
 * @brief Alias `np.concat` for `np.concatenate` (NumPy 2.0).
 * Reference: numpy-reference/reference/generated/numpy.concat.html
 */
template <typename T>
NP_API NP_NODISCARD inline auto concat(const std::vector<ndarray<T>> &arrays, int axis = 0) -> ndarray<T>
{
    return concatenate(arrays, axis);
}
#endif // NP_MANIPULATION_HPP guard

} // namespace np

#endif // NP_CONCATENATE_HPP
