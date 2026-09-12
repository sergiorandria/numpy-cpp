/**
 * @file manipulation.hpp
 * @brief Array manipulation routines (tile, flip, roll, split, delete, insert,
 * etc.).
 *
 * Implements NumPy's array manipulation functions:
 *   - Rearranging: flip, fliplr, flipud, roll, rot90
 *   - Tiling: tile, repeat (already in ndarray.hpp)
 *   - Splitting: split, array_split, hsplit, vsplit, dsplit
 *   - Adding/removing: delete, insert, append, trim_zeros, unique
 *   - Building matrices: diag, diagflat, tri, tril, triu, vander
 *   - Other: where, select, choose
 *
 * Reference: numpy-reference/reference/routines.array-manipulation.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MANIPULATION_HPP
#define NP_MANIPULATION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "creation.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"

namespace np
{
// Rearranging Elements
/**
 * @brief Reverse the order of elements along the given axis.
 *
 * Reference: numpy-reference/reference/generated/numpy.flip.html
 *
 * @tparam T Element type
 * @param arr Input array
 * @param axis Axis along which to flip (if nullopt, flips all axes)
 * @return Flipped array (view with reversed strides)
 */
NP_API template <typename T>
NP_NODISCARD auto flip(const ndarray<T> &arr, std::optional<int> axis = std::nullopt) -> ndarray<T>
{
    ndarray<T> result = arr;

    if (!axis.has_value())
    {
        // Flip all axes
        for (std::size_t ax = 0; ax < result.ndim(); ++ax)
        {
            if (result.shape[ax] > 1)
            {
                result.offset += result.strides[ax] * (result.shape[ax] - 1);
                result.strides[ax] = -static_cast<std::ptrdiff_t>(result.strides[ax]);
            }
        }
    }
    else
    {
        // Normalize axis
        int ax = *axis;
        if (ax < 0)
        {
            ax += static_cast<int>(result.ndim());
        }
        if (ax < 0 || ax >= static_cast<int>(result.ndim()))
        {
            throw AxisError("axis " + std::to_string(*axis) + " is out of bounds");
        }

        // Flip specified axis
        if (result.shape[ax] > 1)
        {
            result.offset += result.strides[ax] * (result.shape[ax] - 1);
            result.strides[ax] = -static_cast<std::ptrdiff_t>(result.strides[ax]);
        }
    }

    return result;
}

/**
 * @brief Flip array left to right (horizontally).
 *
 * Reference: numpy-reference/reference/generated/numpy.fliplr.html
 *
 * @tparam T Element type
 * @param arr Input array (must be at least 2D)
 * @return Array with columns reversed
 */
NP_API template <typename T> NP_NODISCARD auto fliplr(const ndarray<T> &arr) -> ndarray<T>
{
    if (arr.ndim() < 2)
    {
        throw std::invalid_argument("fliplr requires at least 2 dimensions");
    }
    return flip(arr, 1);
}

/**
 * @brief Flip array up to down (vertically).
 *
 * Reference: numpy-reference/reference/generated/numpy.flipud.html
 *
 * @tparam T Element type
 * @param arr Input array (must be at least 1D)
 * @return Array with rows reversed
 */
NP_API template <typename T> NP_NODISCARD auto flipud(const ndarray<T> &arr) -> ndarray<T>
{
    if (arr.ndim() < 1)
    {
        throw std::invalid_argument("flipud requires at least 1 dimension");
    }
    return flip(arr, 0);
}

/**
 * @brief Roll array elements along a given axis.
 *
 * Reference: numpy-reference/reference/generated/numpy.roll.html
 *
 * @tparam T Element type
 * @param arr Input array
 * @param shift Number of places to shift
 * @param axis Axis along which to roll (if nullopt, flattens first)
 * @return Rolled array
 */
NP_API template <typename T>
NP_NODISCARD auto roll(const ndarray<T> &arr, int shift, std::optional<int> axis = std::nullopt) -> ndarray<T>
{
    if (!axis.has_value())
    {
        // Flatten, roll, reshape
        auto flat = arr.ravel();
        int n = static_cast<int>(flat.size());
        shift = ((shift % n) + n) % n; // Normalize shift

        // Create result array with original shape
        ndarray<T> result(arr.shape);
        for (int i = 0; i < n; ++i)
        {
            result.data()[i] = flat((i - shift + n) % n);
        }
        return result;
    }

    // Normalize axis
    int ax = *axis;
    if (ax < 0)
    {
        ax += static_cast<int>(arr.ndim());
    }
    if (ax < 0 || ax >= static_cast<int>(arr.ndim()))
    {
        throw AxisError("axis " + std::to_string(*axis) + " is out of bounds");
    }

    ndarray<T> result = arr.copy();
    int n = result.shape[ax];
    shift = ((shift % n) + n) % n; // Normalize shift

    if (shift == 0 || n <= 1)
    {
        return result;
    }

    // Create index arrays for source and destination
    std::vector<std::size_t> idx(arr.ndim(), 0);

    // Helper to iterate through all indices
    auto roll_recursive = [&](auto &self, std::size_t dim) -> void {
        if (static_cast<int>(dim) == ax)
        {
            // At the rolling axis, perform the shift
            std::vector<T> temp(n);
            for (int i = 0; i < n; ++i)
            {
                idx[dim] = i;
                temp[i] = arr.get(idx);
            }
            for (int i = 0; i < n; ++i)
            {
                idx[dim] = (i + shift) % n;
                result.set(idx, temp[i]);
            }
            return;
        }

        if (dim >= arr.ndim())
        {
            return;
        }

        for (std::size_t i = 0; i < static_cast<std::size_t>(arr.shape[dim]); ++i)
        {
            idx[dim] = i;
            if (dim + 1 < arr.ndim())
            {
                self(self, dim + 1);
            }
            else
            {
                self(self, ax);
            }
        }
    };

    if (ax == 0)
    {
        roll_recursive(roll_recursive, 0);
    }
    else
    {
        roll_recursive(roll_recursive, 0);
    }

    return result;
}

/**
 * @brief Rotate array by 90 degrees.
 *
 * Reference: numpy-reference/reference/generated/numpy.rot90.html
 *
 * @tparam T Element type
 * @param arr Input array (must be at least 2D)
 * @param k Number of times to rotate by 90 degrees
 * @param axes Axes defining the plane of rotation
 * @return Rotated array
 */
NP_API template <typename T>
NP_NODISCARD auto rot90(const ndarray<T> &arr, int k = 1, const std::vector<int> &axes = {0, 1}) -> ndarray<T>
{
    if (arr.ndim() < 2)
    {
        throw std::invalid_argument("rot90 requires at least 2 dimensions");
    }

    if (axes.size() != 2)
    {
        throw std::invalid_argument("axes must have exactly 2 elements");
    }

    // Normalize k to [0, 3]
    k = ((k % 4) + 4) % 4;

    if (k == 0)
    {
        return arr.copy();
    }

    // Normalize axes
    std::vector<int> norm_axes = axes;
    for (auto &ax : norm_axes)
    {
        if (ax < 0)
        {
            ax += static_cast<int>(arr.ndim());
        }
        if (ax < 0 || ax >= static_cast<int>(arr.ndim()))
        {
            throw AxisError("axis out of bounds");
        }
    }

    if (norm_axes[0] == norm_axes[1])
    {
        throw std::invalid_argument("axes must be different");
    }

    ndarray<T> result = arr.copy();

    for (int i = 0; i < k; ++i)
    {
        // Transpose the two axes
        result = result.swapaxes(norm_axes[0], norm_axes[1]);
        // Flip the second axis
        result = flip(result, norm_axes[1]);
    }

    return result;
}

// Tiling Arrays
/**
 * @brief Construct an array by repeating arr the number of times given by reps.
 *
 * Reference: numpy-reference/reference/generated/numpy.tile.html
 *
 * @tparam T Element type
 * @param arr Input array
 * @param reps Number of repetitions along each axis
 * @return Tiled array
 */
NP_API template <typename T> NP_NODISCARD auto tile(const ndarray<T> &arr, const std::vector<int> &reps) -> ndarray<T>
{
    if (reps.empty())
    {
        throw std::invalid_argument("reps cannot be empty");
    }

    // Expand dimensions if necessary
    std::vector<int> arr_shape = arr.shape;
    std::vector<int> tile_reps = reps;

    while (arr_shape.size() < tile_reps.size())
    {
        arr_shape.insert(arr_shape.begin(), 1);
    }
    while (tile_reps.size() < arr_shape.size())
    {
        tile_reps.insert(tile_reps.begin(), 1);
    }

    // Calculate result shape
    std::vector<int> result_shape(arr_shape.size());
    for (std::size_t i = 0; i < arr_shape.size(); ++i)
    {
        result_shape[i] = arr_shape[i] * tile_reps[i];
    }

    ndarray<T> result(result_shape);

    // Reshape input if dimensions were expanded
    ndarray<T> arr_expanded = arr.reshape(arr_shape);

    // Fill result with tiles
    std::vector<std::size_t> idx(result_shape.size(), 0);
    std::vector<std::size_t> src_idx(result_shape.size(), 0);

    auto tile_recursive = [&](auto &self, std::size_t dim) -> void {
        if (dim >= result_shape.size())
        {
            result.set(idx, arr_expanded.get(src_idx));
            return;
        }

        for (int i = 0; i < result_shape[dim]; ++i)
        {
            idx[dim] = i;
            src_idx[dim] = i % arr_shape[dim];
            self(self, dim + 1);
        }
    };

    tile_recursive(tile_recursive, 0);

    return result;
}

// Building Matrices
/**
 * @brief Extract or construct a diagonal array.
 *
 * Reference: numpy-reference/reference/generated/numpy.diag.html
 *
 * @tparam T Element type
 * @param v Input array (1D or 2D)
 * @param k Diagonal offset (0=main diagonal, >0 above, <0 below)
 * @return Diagonal array
 */
NP_API template <typename T> NP_NODISCARD auto diag(const ndarray<T> &v, int k = 0) -> ndarray<T>
{
    if (v.ndim() == 1)
    {
        // Construct 2D array with v on diagonal
        int n = static_cast<int>(v.size());
        int size = n + std::abs(k);

        // Create 2D array with correct shape [size, size]
        std::vector<int> shape_vec = {size, size};
        ndarray<T> result(shape_vec);
        // Fill with zeros
        for (std::size_t idx = 0; idx < result.size(); ++idx)
        {
            result.data()[idx] = T{0};
        }

        // Set diagonal
        for (int i = 0; i < n; ++i)
        {
            if (k >= 0)
            {
                result(i, i + k) = v(i);
            }
            else
            {
                result(i - k, i) = v(i);
            }
        }

        return result;
    }
    else if (v.ndim() == 2)
    {
        // Extract diagonal
        int rows = v.shape[0];
        int cols = v.shape[1];
        int diag_size;

        if (k >= 0)
        {
            // Upper diagonal or main diagonal
            diag_size = std::min(rows, cols - k);
        }
        else
        {
            // Lower diagonal
            diag_size = std::min(rows + k, cols);
        }

        // Ensure non-negative
        diag_size = std::max(0, diag_size);

        if (diag_size == 0)
        {
            std::vector<int> empty_shape = {0};
            return ndarray<T>(empty_shape);
        }

        std::vector<int> result_shape = {diag_size};
        ndarray<T> result(result_shape);

        for (int i = 0; i < diag_size; ++i)
        {
            if (k >= 0)
            {
                result(i) = v(i, i + k);
            }
            else
            {
                result(i) = v(i - k, i);
            }
        }

        return result;
    }
    else
    {
        throw std::invalid_argument("diag requires 1D or 2D array");
    }
}

/**
 * @brief Create a 2D array with flattened input on the diagonal.
 *
 * Reference: numpy-reference/reference/generated/numpy.diagflat.html
 *
 * @tparam T Element type
 * @param v Input array (flattened before use)
 * @param k Diagonal offset
 * @return 2D array with v on diagonal
 */
NP_API template <typename T> NP_NODISCARD auto diagflat(const ndarray<T> &v, int k = 0) -> ndarray<T>
{
    auto flat = v.ravel();
    return diag(flat, k);
}

/**
 * @brief Array with ones at and below the given diagonal and zeros elsewhere.
 *
 * Reference: numpy-reference/reference/generated/numpy.tri.html
 *
 * @tparam T Element type
 * @param n Number of rows
 * @param m Number of columns (default: n)
 * @param k Diagonal offset (0=main, >0 above, <0 below)
 * @return Lower triangular array
 */
NP_API template <typename T = double> NP_NODISCARD auto tri(int n, int m = -1, int k = 0) -> ndarray<T>
{
    if (m < 0)
    {
        m = n;
    }

    ndarray<T> result = zeros<T>({n, m});

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (j <= i + k)
            {
                result(i, j) = T{1};
            }
        }
    }

    return result;
}

/**
 * @brief Lower triangle of an array.
 *
 * Reference: numpy-reference/reference/generated/numpy.tril.html
 *
 * @tparam T Element type
 * @param arr Input array (must be at least 2D)
 * @param k Diagonal offset
 * @return Array with elements above kth diagonal zeroed
 */
NP_API template <typename T> NP_NODISCARD auto tril(const ndarray<T> &arr, int k = 0) -> ndarray<T>
{
    if (arr.ndim() < 2)
    {
        throw std::invalid_argument("tril requires at least 2 dimensions");
    }

    ndarray<T> result = arr.copy();
    int rows = result.shape[result.ndim() - 2];
    int cols = result.shape[result.ndim() - 1];

    // Create base indices
    std::vector<std::size_t> idx(result.ndim(), 0);

    auto apply_tril = [&](auto &self, std::size_t dim) -> void {
        if (dim == result.ndim() - 2)
        {
            // At the row dimension
            for (int i = 0; i < rows; ++i)
            {
                idx[dim] = i;
                for (int j = 0; j < cols; ++j)
                {
                    idx[dim + 1] = j;
                    if (j > i + k)
                    {
                        result.set(idx, T{0});
                    }
                }
            }
            return;
        }

        for (std::size_t i = 0; i < static_cast<std::size_t>(result.shape[dim]); ++i)
        {
            idx[dim] = i;
            self(self, dim + 1);
        }
    };

    if (result.ndim() == 2)
    {
        apply_tril(apply_tril, 0);
    }
    else
    {
        apply_tril(apply_tril, 0);
    }

    return result;
}

/**
 * @brief Upper triangle of an array.
 *
 * Reference: numpy-reference/reference/generated/numpy.triu.html
 *
 * @tparam T Element type
 * @param arr Input array (must be at least 2D)
 * @param k Diagonal offset
 * @return Array with elements below kth diagonal zeroed
 */
NP_API template <typename T> NP_NODISCARD auto triu(const ndarray<T> &arr, int k = 0) -> ndarray<T>
{
    if (arr.ndim() < 2)
    {
        throw std::invalid_argument("triu requires at least 2 dimensions");
    }

    ndarray<T> result = arr.copy();
    int rows = result.shape[result.ndim() - 2];
    int cols = result.shape[result.ndim() - 1];

    std::vector<std::size_t> idx(result.ndim(), 0);

    auto apply_triu = [&](auto &self, std::size_t dim) -> void {
        if (dim == result.ndim() - 2)
        {
            for (int i = 0; i < rows; ++i)
            {
                idx[dim] = i;
                for (int j = 0; j < cols; ++j)
                {
                    idx[dim + 1] = j;
                    if (j < i + k)
                    {
                        result.set(idx, T{0});
                    }
                }
            }
            return;
        }

        for (std::size_t i = 0; i < static_cast<std::size_t>(result.shape[dim]); ++i)
        {
            idx[dim] = i;
            self(self, dim + 1);
        }
    };

    if (result.ndim() == 2)
    {
        apply_triu(apply_triu, 0);
    }
    else
    {
        apply_triu(apply_triu, 0);
    }

    return result;
}

/**
 * @brief Generate a Vandermonde matrix.
 *
 * Reference: numpy-reference/reference/generated/numpy.vander.html
 *
 * @tparam T Element type
 * @param x Input 1D array
 * @param n Number of columns (default: x.size())
 * @param increasing Order of powers (false: decreasing, true: increasing)
 * @return Vandermonde matrix
 */
NP_API template <typename T>
NP_NODISCARD auto vander(const ndarray<T> &x, int n = -1, bool increasing = false) -> ndarray<T>
{
    if (x.ndim() != 1)
    {
        throw std::invalid_argument("vander requires 1D array");
    }

    int rows = static_cast<int>(x.size());
    if (n < 0)
    {
        n = rows;
    }

    ndarray<T> result(std::vector<int>{rows, n});

    for (int i = 0; i < rows; ++i)
    {
        T val = x(i);
        for (int j = 0; j < n; ++j)
        {
            int power = increasing ? j : (n - 1 - j);
            T powered = T{1};
            for (int p = 0; p < power; ++p)
            {
                powered *= val;
            }
            result(i, j) = powered;
        }
    }

    return result;
}

// Splitting Arrays
/**
 * @brief Split array into multiple sub-arrays.
 *
 * Reference: numpy-reference/reference/generated/numpy.split.html
 *
 * @tparam T Element type
 * @param arr Input array
 * @param indices_or_sections Indices where to split, or number of equal
 * sections
 * @param axis Axis along which to split
 * @return Vector of sub-arrays
 */
NP_API template <typename T>
NP_NODISCARD auto split(const ndarray<T> &arr, const std::vector<int> &indices_or_sections, int axis = 0)
    -> std::vector<ndarray<T>>
{
    // Normalize axis
    if (axis < 0)
    {
        axis += static_cast<int>(arr.ndim());
    }
    if (axis < 0 || axis >= static_cast<int>(arr.ndim()))
    {
        throw AxisError("axis out of bounds");
    }

    int n = arr.shape[axis];
    std::vector<ndarray<T>> result;

    if (indices_or_sections.size() == 1 && indices_or_sections[0] > 0)
    {
        // Split into equal sections
        int sections = indices_or_sections[0];
        if (n % sections != 0)
        {
            throw std::invalid_argument("array split does not result in equal division");
        }

        int section_size = n / sections;
        result.reserve(static_cast<std::size_t>(sections));
        for (int i = 0; i < sections; ++i)
        {
            std::vector<int> new_shape = arr.shape;
            new_shape[axis] = section_size;
            ndarray<T> sub(new_shape);

            // Copy data
            std::vector<std::size_t> src_idx(arr.ndim(), 0);
            std::vector<std::size_t> dst_idx(arr.ndim(), 0);

            auto copy_section = [&](auto &self, std::size_t dim) -> void {
                if (dim >= arr.ndim())
                {
                    sub.set(dst_idx, arr.get(src_idx));
                    return;
                }

                if (dim == static_cast<std::size_t>(axis))
                {
                    for (int j = 0; j < section_size; ++j)
                    {
                        src_idx[dim] = i * section_size + j;
                        dst_idx[dim] = j;
                        self(self, dim + 1);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < static_cast<std::size_t>(arr.shape[dim]); ++j)
                    {
                        src_idx[dim] = j;
                        dst_idx[dim] = j;
                        self(self, dim + 1);
                    }
                }
            };

            copy_section(copy_section, 0);
            result.push_back(sub);
        }
    }
    else
    {
        // Split at indices
        std::vector<int> split_points;
        split_points.reserve(indices_or_sections.size() + 2);
        split_points.push_back(0);
        split_points.insert(split_points.end(), indices_or_sections.begin(), indices_or_sections.end());
        split_points.push_back(n);
        result.reserve(split_points.size() - 1);
        for (std::size_t i = 0; i < split_points.size() - 1; ++i)
        {
            int start = split_points[i];
            int end = split_points[i + 1];

            if (start < 0 || end > n || start > end)
            {
                throw std::invalid_argument("invalid split indices");
            }

            std::vector<int> new_shape = arr.shape;
            new_shape[axis] = end - start;
            ndarray<T> sub(new_shape);

            std::vector<std::size_t> src_idx(arr.ndim(), 0);
            std::vector<std::size_t> dst_idx(arr.ndim(), 0);

            auto copy_section = [&](auto &self, std::size_t dim) -> void {
                if (dim >= arr.ndim())
                {
                    sub.set(dst_idx, arr.get(src_idx));
                    return;
                }

                if (dim == static_cast<std::size_t>(axis))
                {
                    for (int j = start; j < end; ++j)
                    {
                        src_idx[dim] = j;
                        dst_idx[dim] = j - start;
                        self(self, dim + 1);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < static_cast<std::size_t>(arr.shape[dim]); ++j)
                    {
                        src_idx[dim] = j;
                        dst_idx[dim] = j;
                        self(self, dim + 1);
                    }
                }
            };

            copy_section(copy_section, 0);
            result.push_back(sub);
        }
    }

    return result;
}

/**
 * @brief Split array into approximately equal pieces.
 *
 * Reference: numpy-reference/reference/generated/numpy.array_split.html
 */
NP_API template <typename T>
NP_NODISCARD auto array_split(const ndarray<T> &arr, int sections, int axis = 0) -> std::vector<ndarray<T>>
{
    if (axis < 0)
    {
        axis += static_cast<int>(arr.ndim());
    }
    if (axis < 0 || axis >= static_cast<int>(arr.ndim()))
    {
        throw AxisError("axis out of bounds");
    }

    int n = arr.shape[axis];
    int base_size = n / sections;
    int remainder = n % sections;

    std::vector<int> split_points;
    int pos = 0;
    for (int i = 0; i < sections; ++i)
    {
        int size = base_size + (i < remainder ? 1 : 0);
        pos += size;
        if (pos < n)
        {
            split_points.push_back(pos);
        }
    }

    return split(arr, split_points, axis);
}

/**
 * @brief Split array at explicit indices (np.array_split with indices).
 *
 * Reference: numpy-reference/reference/generated/numpy.array_split.html
 */
NP_API template <typename T>
NP_NODISCARD auto array_split(const ndarray<T> &arr, const std::vector<int> &indices, int axis = 0)
    -> std::vector<ndarray<T>>
{
    return split(arr, indices, axis);
}

/**
 * @brief Split array into multiple sub-arrays horizontally.
 */
NP_API template <typename T>
NP_NODISCARD auto hsplit(const ndarray<T> &arr, const std::vector<int> &indices_or_sections) -> std::vector<ndarray<T>>
{
    if (arr.ndim() == 0)
    {
        throw std::invalid_argument("hsplit requires at least 1D array");
    }
    int axis = arr.ndim() == 1 ? 0 : 1;
    return split(arr, indices_or_sections, axis);
}

/**
 * @brief Split array into multiple sub-arrays vertically.
 */
NP_API template <typename T>
NP_NODISCARD auto vsplit(const ndarray<T> &arr, const std::vector<int> &indices_or_sections) -> std::vector<ndarray<T>>
{
    if (arr.ndim() < 2)
    {
        throw std::invalid_argument("vsplit requires at least 2D array");
    }
    return split(arr, indices_or_sections, 0);
}

/**
 * @brief Split array into multiple sub-arrays along 3rd axis.
 */
NP_API template <typename T>
NP_NODISCARD auto dsplit(const ndarray<T> &arr, const std::vector<int> &indices_or_sections) -> std::vector<ndarray<T>>
{
    if (arr.ndim() < 3)
    {
        throw std::invalid_argument("dsplit requires at least 3D array");
    }
    return split(arr, indices_or_sections, 2);
}

// Adding/Removing Elements
/**
 * @brief Return a new array with sub-arrays along an axis deleted.
 *
 * Reference: numpy-reference/reference/generated/numpy.delete.html
 *
 * @tparam T Element type
 * @param arr Input array
 * @param indices Indices of sub-arrays to remove
 * @param axis Axis along which to delete (if nullopt, flattens first)
 * @return Array with deleted elements
 */
NP_API template <typename T>
NP_NODISCARD auto delete_arr(const ndarray<T> &arr, const std::vector<int> &indices,
                             std::optional<int> axis = std::nullopt) -> ndarray<T>
{
    if (indices.empty())
    {
        return arr.copy();
    }

    if (!axis.has_value())
    {
        // Flatten and delete
        auto flat = arr.ravel();
        int n = static_cast<int>(flat.size());

        // Normalize and sort indices
        std::set<int> to_delete;
        for (int idx : indices)
        {
            int norm_idx = idx < 0 ? idx + n : idx;
            if (norm_idx >= 0 && norm_idx < n)
            {
                to_delete.insert(norm_idx);
            }
        }

        ndarray<T> result(std::vector<int>{n - static_cast<int>(to_delete.size())});
        int j = 0;
        for (int i = 0; i < n; ++i)
        {
            if (to_delete.find(i) == to_delete.end())
            {
                result(j++) = flat(i);
            }
        }

        return result;
    }

    // Normalize axis
    int ax = *axis;
    if (ax < 0)
    {
        ax += static_cast<int>(arr.ndim());
    }
    if (ax < 0 || ax >= static_cast<int>(arr.ndim()))
    {
        throw AxisError("axis out of bounds");
    }

    int n = arr.shape[ax];

    // Normalize and sort indices
    std::set<int> to_delete;
    for (int idx : indices)
    {
        int norm_idx = idx < 0 ? idx + n : idx;
        if (norm_idx >= 0 && norm_idx < n)
        {
            to_delete.insert(norm_idx);
        }
    }

    std::vector<int> new_shape = arr.shape;
    new_shape[ax] = n - static_cast<int>(to_delete.size());

    ndarray<T> result(new_shape);

    std::vector<std::size_t> src_idx(arr.ndim(), 0);
    std::vector<std::size_t> dst_idx(arr.ndim(), 0);

    auto copy_without = [&](auto &self, std::size_t dim) -> void {
        if (dim >= arr.ndim())
        {
            result.set(dst_idx, arr.get(src_idx));
            return;
        }

        if (dim == static_cast<std::size_t>(ax))
        {
            int dst_i = 0;
            for (int i = 0; i < n; ++i)
            {
                if (to_delete.find(i) == to_delete.end())
                {
                    src_idx[dim] = i;
                    dst_idx[dim] = dst_i++;
                    self(self, dim + 1);
                }
            }
        }
        else
        {
            for (std::size_t i = 0; i < static_cast<std::size_t>(arr.shape[dim]); ++i)
            {
                src_idx[dim] = i;
                dst_idx[dim] = i;
                self(self, dim + 1);
            }
        }
    };

    copy_without(copy_without, 0);

    return result;
}

/**
 * @brief Insert values along the given axis before the given indices.
 *
 * Reference: numpy-reference/reference/generated/numpy.insert.html
 *
 * @tparam T Element type
 * @param arr Input array
 * @param indices Indices before which to insert
 * @param values Values to insert
 * @param axis Axis along which to insert (if nullopt, flattens first)
 * @return Array with inserted values
 */
NP_API template <typename T>
NP_NODISCARD auto insert(const ndarray<T> &arr, const std::vector<int> &indices, const ndarray<T> &values,
                         std::optional<int> axis = std::nullopt) -> ndarray<T>
{
    if (indices.empty())
    {
        return arr.copy();
    }

    if (!axis.has_value())
    {
        // Flatten and insert
        auto flat = arr.ravel();
        auto val_flat = values.ravel();
        int n = static_cast<int>(flat.size());

        // Build (position, value, order) list: value j goes before
        // indices[j % indices.size()]; equal positions keep value order.
        std::vector<std::tuple<int, T, std::size_t>> inserts;
        for (std::size_t j = 0; j < val_flat.size(); ++j)
        {
            int raw = indices[j % indices.size()];
            int idx = raw < 0 ? raw + n + 1 : raw;
            idx = std::max(0, std::min(n, idx));
            inserts.emplace_back(idx, val_flat(j), j);
        }
        std::sort(inserts.begin(), inserts.end(), [](const auto &x, const auto &y) {
            if (std::get<0>(x) != std::get<0>(y))
                return std::get<0>(x) < std::get<0>(y);
            return std::get<2>(x) < std::get<2>(y);
        });

        ndarray<T> result(std::vector<int>{n + static_cast<int>(val_flat.size())});
        int k = 0;
        std::size_t v = 0;
        for (int i = 0; i <= n; ++i)
        {
            while (v < inserts.size() && std::get<0>(inserts[v]) == i)
            {
                result(k++) = std::get<1>(inserts[v++]);
            }
            if (i < n)
            {
                result(k++) = flat(i);
            }
        }

        return result;
    }

    // Axis case – insert values along axis before sorted indices
    int ax = *axis;
    if (ax < 0)
        ax += static_cast<int>(arr.ndim());
    if (ax < 0 || ax >= static_cast<int>(arr.ndim()))
        throw AxisError("insert: axis out of bounds");
    int n = arr.shape[ax];
    // Normalize indices to 0..n range (numpy allows n as append)
    std::vector<int> norm_idx;
    norm_idx.reserve(indices.size());
    for (int v : indices)
    {
        int iv = v < 0 ? v + n + 1 : v;
        iv = std::clamp(iv, 0, n);
        norm_idx.push_back(iv);
    }
    // Pair each index with its insertion order to keep stable for equal indices
    std::vector<std::pair<int, std::size_t>> sorted;
    sorted.reserve(norm_idx.size());
    for (std::size_t i = 0; i < norm_idx.size(); ++i)
        sorted.emplace_back(norm_idx[i], i);
    std::sort(sorted.begin(), sorted.end(), [](auto &a, auto &b) {
        if (a.first != b.first)
            return a.first < b.first;
        return a.second < b.second;
    });
    std::vector<int> sorted_vals;
    sorted_vals.reserve(sorted.size());
    for (auto &p : sorted)
        sorted_vals.push_back(p.first);
    // values handling: if values size == 1, broadcast; else size must == indices size
    // For ND, values shape should be broadcastable – we handle 1-D values and ND arr
    std::size_t val_n = values.size();
    bool val_is_scalar = (val_n == 1);
    // Build output shape
    std::vector<int> out_shape = arr.shape;
    out_shape[ax] = n + static_cast<int>(sorted_vals.size());
    ndarray<T> out(out_shape);
    // is_inserted for each output axis position
    int out_n = out_shape[ax];
    std::vector<int> is_inserted(out_n, -1);
    int ins = 0;
    for (int k = 0; k < out_n; ++k)
    {
        if (ins < static_cast<int>(sorted_vals.size()) && k == sorted_vals[ins] + ins)
        {
            is_inserted[k] = static_cast<int>(sorted[ins].second);
            ++ins;
        }
        else
        {
            is_inserted[k] = -1;
        }
    }
    detail::Odometer od(out_shape);
    while (!od.done())
    {
        auto out_idx = od.idx();
        int k = static_cast<int>(out_idx[static_cast<std::size_t>(ax)]);
        int ins_idx = is_inserted[k];
        if (ins_idx >= 0)
        {
            // from values
            std::size_t val_pos = val_is_scalar ? 0 : static_cast<std::size_t>(ins_idx);
            // Map other dims from out_idx to values idx (values may be 1-D or ND)
            std::vector<std::size_t> val_idx(values.ndim(), 0);
            if (values.ndim() == 1)
            {
                val_idx[0] = val_pos % values.shape[0];
            }
            else if (values.ndim() == arr.ndim())
            {
                for (size_t d = 0, o = 0; d < values.ndim(); ++d)
                {
                    if (static_cast<int>(d) == ax)
                        val_idx[d] = val_pos;
                    else
                    {
                        // map other dims from out_idx
                        size_t out_d = d < static_cast<size_t>(ax) ? d : d;
                        // For ND values, other dims should match output other dims
                        // Use out_idx other dim directly, clamped to values shape
                        size_t ov = out_idx[d < static_cast<size_t>(ax) ? d : d];
                        // Actually need to map via output other dims
                        // Simplified: use out_idx other dim
                        val_idx[d] = out_idx[d];
                        if (val_idx[d] >= static_cast<size_t>(values.shape[d]))
                            val_idx[d] = values.shape[d] - 1;
                    }
                }
            }
            else
            {
                // fallback: use flat logical
                out.set(out_idx, values.data()[values._flat_logical(val_pos)]);
                od.advance();
                continue;
            }
            // Try to get from values, fallback to flat
            try
            {
                out.set(out_idx, values.get(val_idx));
            }
            catch (...)
            {
                out.set(out_idx, values.data()[values._flat_logical(val_pos)]);
            }
        }
        else
        {
            int cnt_before = 0;
            for (int i = 0; i < static_cast<int>(sorted_vals.size()); ++i)
                if (sorted_vals[i] + i < k)
                    ++cnt_before;
            int arr_k = k - cnt_before;
            std::vector<std::size_t> arr_idx(arr.ndim(), 0);
            for (size_t d = 0; d < arr.ndim(); ++d)
            {
                if (static_cast<int>(d) == ax)
                    arr_idx[d] = static_cast<size_t>(arr_k);
                else
                    arr_idx[d] = out_idx[d];
            }
            out.set(out_idx, arr.get(arr_idx));
        }
        od.advance();
    }
    return out;
}

/**
 * @brief Append values to the end of an array.
 *
 * Reference: numpy-reference/reference/generated/numpy.append.html
 *
 * @tparam T Element type
 * @param arr Input array
 * @param values Values to append
 * @param axis Axis along which to append (if nullopt, flattens both)
 * @return Array with appended values
 */
NP_API template <typename T>
NP_NODISCARD auto append(const ndarray<T> &arr, const ndarray<T> &values, std::optional<int> axis = std::nullopt)
    -> ndarray<T>
{
    if (!axis.has_value())
    {
        // Flatten both and concatenate
        auto arr_flat = arr.ravel();
        auto val_flat = values.ravel();

        ndarray<T> result(std::vector<int>{static_cast<int>(arr_flat.size() + val_flat.size())});

        for (std::size_t i = 0; i < arr_flat.size(); ++i)
        {
            result(i) = arr_flat(i);
        }
        for (std::size_t i = 0; i < val_flat.size(); ++i)
        {
            result(arr_flat.size() + i) = val_flat(i);
        }

        return result;
    }

    // Use concat for axis version
    int ax = *axis;
    if (ax < 0)
        ax += static_cast<int>(arr.ndim());
    if (ax < 0 || ax >= static_cast<int>(arr.ndim()))
        throw AxisError("append: axis out of bounds");
    std::vector<ndarray<T>> parts{arr, values};
    return concat(parts, ax);
}

/**
 * @brief Trim the leading and/or trailing zeros from a 1D array.
 *
 * Reference: numpy-reference/reference/generated/numpy.trim_zeros.html
 *
 * @tparam T Element type
 * @param arr Input 1D array
 * @param trim String with 'f' for leading, 'b' for trailing
 * @return Trimmed array
 */
NP_API template <typename T>
NP_NODISCARD auto trim_zeros(const ndarray<T> &arr, const std::string &trim = "fb") -> ndarray<T>
{
    if (arr.ndim() != 1)
    {
        throw std::invalid_argument("trim_zeros requires 1D array");
    }

    int n = static_cast<int>(arr.size());
    int start = 0;
    int end = n;

    if (trim.find('f') != std::string::npos)
    {
        while (start < n && arr(start) == T{0})
        {
            ++start;
        }
    }

    if (trim.find('b') != std::string::npos)
    {
        while (end > start && arr(end - 1) == T{0})
        {
            --end;
        }
    }

    if (start >= end)
    {
        return ndarray<T>(std::vector<int>{0});
    }

    ndarray<T> result(std::vector<int>{end - start});
    for (int i = start; i < end; ++i)
    {
        result(i - start) = arr(i);
    }

    return result;
}

/**
 * @brief Find the unique elements of an array.
 *
 * Reference: numpy-reference/reference/generated/numpy.unique.html
 *
 * @tparam T Element type
 * @param arr Input array
 * @param return_index If true, also return indices of first occurrences
 * @param return_inverse If true, also return indices to reconstruct input
 * @param return_counts If true, also return counts of each unique element
 * @return Unique sorted elements (and optionally indices, inverse, counts)
 */
NP_API template <typename T>
NP_NODISCARD auto unique(const ndarray<T> &arr, bool return_index = false, bool return_inverse = false,
                         bool return_counts = false)
    -> std::tuple<ndarray<T>, ndarray<std::size_t>, ndarray<std::size_t>, ndarray<std::size_t>>
{
    auto flat = arr.ravel();
    std::size_t n = flat.size();

    // Create vector of (value, original_index) pairs
    std::vector<std::pair<T, std::size_t>> pairs;
    for (std::size_t i = 0; i < n; ++i)
    {
        pairs.push_back({flat(i), i});
    }

    // Sort by value
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

    // Find unique values
    std::vector<T> unique_vals;
    std::vector<std::size_t> unique_indices;
    std::vector<std::size_t> inverse_indices(n);
    std::vector<std::size_t> counts;

    if (!pairs.empty())
    {
        unique_vals.push_back(pairs[0].first);
        unique_indices.push_back(pairs[0].second);
        std::size_t count = 1;

        for (std::size_t i = 1; i < pairs.size(); ++i)
        {
            if (pairs[i].first != pairs[i - 1].first)
            {
                counts.push_back(count);
                unique_vals.push_back(pairs[i].first);
                unique_indices.push_back(pairs[i].second);
                count = 1;
            }
            else
            {
                ++count;
            }
        }
        counts.push_back(count);

        // Build inverse mapping
        std::size_t unique_idx = 0;
        for (std::size_t i = 0; i < pairs.size(); ++i)
        {
            if (i > 0 && pairs[i].first != pairs[i - 1].first)
            {
                ++unique_idx;
            }
            inverse_indices[pairs[i].second] = unique_idx;
        }
    }

    // Build result arrays
    ndarray<T> result_vals(std::vector<int>{static_cast<int>(unique_vals.size())});
    for (std::size_t i = 0; i < unique_vals.size(); ++i)
    {
        result_vals(static_cast<int>(i)) = unique_vals[i];
    }

    int idx_size = return_index ? static_cast<int>(unique_indices.size()) : 0;
    int inv_size = return_inverse ? static_cast<int>(inverse_indices.size()) : 0;
    int cnt_size = return_counts ? static_cast<int>(counts.size()) : 0;

    ndarray<std::size_t> result_index(std::vector<int>{idx_size});
    if (return_index)
    {
        for (std::size_t i = 0; i < unique_indices.size(); ++i)
        {
            result_index(static_cast<int>(i)) = unique_indices[i];
        }
    }

    ndarray<std::size_t> result_inverse(std::vector<int>{inv_size});
    if (return_inverse)
    {
        for (std::size_t i = 0; i < inverse_indices.size(); ++i)
        {
            result_inverse(static_cast<int>(i)) = inverse_indices[i];
        }
    }

    ndarray<std::size_t> result_counts(std::vector<int>{cnt_size});
    if (return_counts)
    {
        for (std::size_t i = 0; i < counts.size(); ++i)
        {
            result_counts(static_cast<int>(i)) = counts[i];
        }
    }

    return {result_vals, result_index, result_inverse, result_counts};
}

// Conditional Selection
/**
 * @brief Return elements chosen from x or y depending on condition.
 *
 * Reference: numpy-reference/reference/generated/numpy.where.html
 *
 * @tparam T Element type
 * @param condition Boolean array
 * @param x Values where condition is true
 * @param y Values where condition is false
 * @return Array with elements from x where condition, y elsewhere
 */
NP_API template <typename T>
NP_NODISCARD auto where(const ndarray<bool> &condition, const ndarray<T> &x, const ndarray<T> &y) -> ndarray<T>
{
    // Broadcasting check (simplified)
    if (condition.shape != x.shape || condition.shape != y.shape)
    {
        throw std::invalid_argument("where: arrays must have compatible shapes");
    }

    ndarray<T> result(x.shape);

    for (std::size_t i = 0; i < x.size(); ++i)
    {
        result.data()[i] = condition.data()[i] ? x.data()[i] : y.data()[i];
    }

    return result;
}

/**
 * @brief Return indices where condition is true.
 *
 * Reference: numpy-reference/reference/generated/numpy.where.html (single
 * argument form)
 *
 * @param condition Boolean array
 * @return Tuple of arrays, one for each dimension
 */
inline auto where(const ndarray<bool> &condition) -> std::vector<ndarray<std::size_t>>
{
    auto indices = condition.nonzero();
    return indices;
}

/**
 * @brief Broadcast array to a new shape.
 *
 * Mirrors `np.broadcast_to`. The new shape must be broadcast-compatible
 * with the input shape.
 *
 * @tparam T Element type.
 * @param arr Input array.
 * @param shape Target shape.
 * @return View/copy broadcast to shape.
 *
 * Reference: numpy-reference/reference/generated/numpy.broadcast_to.html
 */
NP_API template <typename T>
NP_NODISCARD auto broadcast_to(const ndarray<T> &arr, const std::vector<int> &shape) -> ndarray<T>
{
    // Validate broadcast compatibility via detail::broadcast_shapes
    (void)detail::broadcast_shapes(arr.shape, shape);
    ndarray<T> out(shape);
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        // Map output index to input via broadcast rules
        std::vector<std::size_t> src_idx(arr.ndim(), 0);
        std::size_t out_dim = shape.size();
        std::size_t in_dim = arr.ndim();
        for (std::size_t d = 0; d < out_dim; ++d)
        {
            std::ptrdiff_t in_d = static_cast<std::ptrdiff_t>(d) - static_cast<std::ptrdiff_t>(out_dim - in_dim);
            if (in_d < 0)
                continue;
            if (arr.shape[static_cast<std::size_t>(in_d)] == 1)
                src_idx[static_cast<std::size_t>(in_d)] = 0;
            else
                src_idx[static_cast<std::size_t>(in_d)] = idx[d];
        }
        out.set(idx, arr.get(src_idx));
        od.advance();
    }
    return out;
}

/**
 * @brief Expand dimensions of an array.
 *
 * Inserts a new axis at `axis` (mirrors `np.expand_dims`).
 *
 * @tparam T Element type.
 * @param arr Input array.
 * @param axis Position of new axis (may be negative).
 * @return View with expanded shape.
 */
NP_API template <typename T> NP_NODISCARD auto expand_dims(const ndarray<T> &arr, int axis) -> ndarray<T>
{
    int nd = static_cast<int>(arr.ndim());
    if (axis < 0)
        axis += nd + 1;
    if (axis < 0 || axis > nd)
        throw AxisError("expand_dims: axis out of bounds");
    std::vector<int> new_shape = arr.shape;
    new_shape.insert(new_shape.begin() + axis, 1);
    return arr.reshape(new_shape);
}

/**
 * @brief Ensure array is at least 1-D.
 *
 * Mirrors `np.atleast_1d`. 0-D arrays become 1-D with single element.
 */
NP_API template <typename T> NP_NODISCARD auto atleast_1d(const ndarray<T> &arr) -> ndarray<T>
{
    if (arr.ndim() >= 1)
        return arr.copy();
    ndarray<T> out(std::vector<int>{1});
    out.data()[0] = arr.item();
    return out;
}

/**
 * @brief Ensure array is at least 2-D.
 *
 * Mirrors `np.atleast_2d`. 1-D becomes (1, N).
 */
NP_API template <typename T> NP_NODISCARD auto atleast_2d(const ndarray<T> &arr) -> ndarray<T>
{
    if (arr.ndim() >= 2) [[likely]]
        return arr; // zero-copy view (shared_ptr), was arr.copy()
    if (arr.ndim() == 1)
    {
        return arr.reshape(std::vector<int>{1, static_cast<int>(arr.size())});
    }
    // 0-D -> (1,1)
    ndarray<T> out(std::vector<int>{1, 1});
    out.data()[0] = arr.item();
    return out;
}

/**
 * @brief Ensure array is at least 3-D.
 *
 * Mirrors `np.atleast_3d`. 1-D (N,) -> (1,N,1), 2-D (M,N) -> (M,N,1).
 */
NP_API template <typename T> NP_NODISCARD auto atleast_3d(const ndarray<T> &arr) -> ndarray<T>
{
    if (arr.ndim() >= 3) [[likely]]
        return arr; // zero-copy
    if (arr.ndim() == 2)
    {
        return arr.reshape(std::vector<int>{arr.shape[0], arr.shape[1], 1});
    }
    if (arr.ndim() == 1)
    {
        return arr.reshape(std::vector<int>{1, static_cast<int>(arr.size()), 1});
    }
    ndarray<T> out(std::vector<int>{1, 1, 1});
    out.data()[0] = arr.item();
    return out;
}

/** @brief Move axis to new position (np.moveaxis). */
NP_API template <typename T> NP_NODISCARD auto moveaxis(const ndarray<T> &a, int source, int destination) -> ndarray<T>
{
    int nd = static_cast<int>(a.ndim());
    if (source < 0)
        source += nd;
    if (destination < 0)
        destination += nd;
    if (source < 0 || source >= nd || destination < 0 || destination >= nd)
        throw AxisError("moveaxis: axis out of bounds");
    std::vector<int> perm(nd);
    for (int i = 0; i < nd; ++i)
        perm[i] = i;
    int val = perm[source];
    perm.erase(perm.begin() + source);
    perm.insert(perm.begin() + destination, val);
    return a.transpose(perm);
}

/** @brief Roll axis to start position (np.rollaxis). */
NP_API template <typename T> NP_NODISCARD auto rollaxis(const ndarray<T> &a, int axis, int start = 0) -> ndarray<T>
{
    int nd = static_cast<int>(a.ndim());
    if (axis < 0)
        axis += nd;
    if (start < 0)
        start += nd;
    if (axis < 0 || axis >= nd || start < 0 || start > nd)
        throw AxisError("rollaxis: axis out of bounds");
    if (axis == start)
        return a.copy();
    // Equivalent to moveaxis with destination handling for start > axis case
    int dest = start;
    if (axis < start)
        dest = start - 1;
    return moveaxis(a, axis, dest);
}

/** @brief Stack 1-D arrays as columns (np.column_stack). */
#ifndef NP_CONCATENATE_HPP
NP_API template <typename T> NP_NODISCARD auto column_stack(const std::vector<ndarray<T>> &tup) -> ndarray<T>
{
    if (tup.empty())
        throw std::invalid_argument("column_stack: need at least one array");
    // If all 1-D, stack as columns -> shape (N, K)
    bool all1d = true;
    for (auto &arr : tup)
        if (arr.ndim() != 1)
            all1d = false;
    if (all1d)
    {
        int n = tup[0].shape[0];
        for (auto &arr : tup)
            if (arr.shape[0] != n)
                throw std::invalid_argument("column_stack: 1-D arrays must have same length");
        ndarray<T> out(std::vector<int>{n, static_cast<int>(tup.size())});
        for (std::size_t k = 0; k < tup.size(); ++k)
            for (int i = 0; i < n; ++i)
                out.at(static_cast<std::size_t>(i), k) = tup[k].at(static_cast<std::size_t>(i));
        return out;
    }
    // Otherwise hstack
    // Fallback to hstack via concatenate along axis 1 (requires same rows)
    int rows = tup[0].shape[0];
    for (auto &arr : tup)
        if (arr.shape[0] != rows)
            throw std::invalid_argument("column_stack: arrays must have same first dimension");
    // Concatenate along last axis (1 for 2-D)
    int total_cols = 0;
    for (auto &arr : tup)
        total_cols += arr.shape[1];
    ndarray<T> out(std::vector<int>{rows, total_cols});
    int col_off = 0;
    for (auto &arr : tup)
    {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < arr.shape[1]; ++j)
                out.at(static_cast<std::size_t>(i), static_cast<std::size_t>(col_off + j)) =
                    arr.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        col_off += arr.shape[1];
    }
    return out;
}

/** @brief Stack arrays vertically (np.row_stack / vstack). */
NP_API template <typename T> NP_NODISCARD auto row_stack(const std::vector<ndarray<T>> &tup) -> ndarray<T>
{
    if (tup.empty())
        throw std::invalid_argument("row_stack: need at least one array");
    // Promote 1-D to 2-D row (1, N) then vstack
    std::vector<ndarray<T>> tmp;
    tmp.reserve(tup.size());
    for (auto &arr : tup)
    {
        if (arr.ndim() == 1)
            tmp.push_back(arr.reshape(std::vector<int>{1, arr.shape[0]}));
        else
            tmp.push_back(arr);
    }
    int cols = tmp[0].shape[1];
    for (auto &arr : tmp)
        if (arr.shape[1] != cols)
            throw std::invalid_argument("row_stack: arrays must have same number of columns");
    int total_rows = 0;
    for (auto &arr : tmp)
        total_rows += arr.shape[0];
    ndarray<T> out(std::vector<int>{total_rows, cols});
    int row_off = 0;
    for (auto &arr : tmp)
    {
        for (int i = 0; i < arr.shape[0]; ++i)
            for (int j = 0; j < cols; ++j)
                out.at(static_cast<std::size_t>(row_off + i), static_cast<std::size_t>(j)) =
                    arr.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        row_off += arr.shape[0];
    }
    return out;
}
#endif // NP_CONCATENATE_HPP

/** @brief Assemble array from blocks (np.block) – 2-D case. */
NP_API template <typename T> NP_NODISCARD auto block(const std::vector<std::vector<ndarray<T>>> &blocks) -> ndarray<T>
{
    if (blocks.empty() || blocks[0].empty())
        throw std::invalid_argument("block: need at least one block");
    // First, hstack each row, then vstack rows
    std::vector<ndarray<T>> row_stacked;
    row_stacked.reserve(blocks.size());
    for (auto &row : blocks)
    {
        if (row.empty())
            throw std::invalid_argument("block: empty row");
        int h = row[0].shape[0];
        for (auto &b : row)
            if (b.shape[0] != h)
                throw std::invalid_argument("block: blocks in row must have same rows");
        int total_w = 0;
        for (auto &b : row)
            total_w += b.shape[1];
        ndarray<T> r(std::vector<int>{h, total_w});
        int col_off = 0;
        for (auto &b : row)
        {
            for (int i = 0; i < h; ++i)
                for (int j = 0; j < b.shape[1]; ++j)
                    r.at(static_cast<std::size_t>(i), static_cast<std::size_t>(col_off + j)) =
                        b.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            col_off += b.shape[1];
        }
        row_stacked.push_back(std::move(r));
    }
    // vstack
    int total_h = 0;
    int w = row_stacked[0].shape[1];
    for (auto &r : row_stacked)
    {
        if (r.shape[1] != w)
            throw std::invalid_argument("block: rows must have same width");
        total_h += r.shape[0];
    }
    ndarray<T> out(std::vector<int>{total_h, w});
    int row_off = 0;
    for (auto &r : row_stacked)
    {
        for (int i = 0; i < r.shape[0]; ++i)
            for (int j = 0; j < w; ++j)
                out.at(static_cast<std::size_t>(row_off + i), static_cast<std::size_t>(j)) =
                    r.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        row_off += r.shape[0];
    }
    return out;
}

/** @brief 1-D block – horizontal concatenation (np.block([a,b])).
 *
 * Reference: numpy-reference/reference/generated/numpy.block.html
 */
NP_API template <typename T> NP_NODISCARD auto block(const std::vector<ndarray<T>> &blocks) -> ndarray<T>
{
    if (blocks.empty())
    {
        throw std::invalid_argument("block: need at least one block");
    }
    // Defer to concatenate along last axis (horizontal for 1-D/2-D)
    int axis = blocks[0].ndim() == 0 ? 0 : static_cast<int>(blocks[0].ndim() - 1);
    return concat(blocks, axis);
}

/** @brief Broadcast arrays to common shape (np.broadcast_arrays). */
NP_API template <typename T>
NP_NODISCARD auto broadcast_arrays(const std::vector<ndarray<T>> &arrays) -> std::vector<ndarray<T>>
{
    if (arrays.empty())
        return {};
    std::vector<int> common = arrays[0].shape;
    for (std::size_t i = 1; i < arrays.size(); ++i)
        common = detail::broadcast_shapes(common, arrays[i].shape);
    std::vector<ndarray<T>> out;
    out.reserve(arrays.size());
    for (auto &arr : arrays)
    {
        out.push_back(broadcast_to(arr, common));
    }
    return out;
}

// Normal comment: free wrappers for ndarray shape methods

NP_API template <typename T> NP_NODISCARD auto reshape(const ndarray<T> &a, const std::vector<int> &shape) -> ndarray<T>
{
    return a.reshape(shape);
}
NP_API template <typename T> NP_NODISCARD auto ravel(const ndarray<T> &a) -> ndarray<T>
{
    return a.ravel();
}
NP_API template <typename T> NP_NODISCARD auto squeeze(const ndarray<T> &a) -> ndarray<T>
{
    return a.squeeze();
}
NP_API template <typename T> NP_NODISCARD auto squeeze(const ndarray<T> &a, int axis) -> ndarray<T>
{
    return a.squeeze(axis);
}
NP_API template <typename T> NP_NODISCARD auto transpose(const ndarray<T> &a) -> ndarray<T>
{
    return a.transpose();
}
NP_API template <typename T>
NP_NODISCARD auto transpose(const ndarray<T> &a, const std::vector<int> &axes) -> ndarray<T>
{
    return a.transpose(axes);
}
NP_API template <typename T> NP_NODISCARD auto swapaxes(const ndarray<T> &a, int axis1, int axis2) -> ndarray<T>
{
    return a.swapaxes(axis1, axis2);
}

// Normal comment: select and place

NP_API template <typename T>
NP_NODISCARD auto select(const std::vector<ndarray<bool>> &condlist, const std::vector<ndarray<T>> &choicelist,
                         T default_val = T{0}) -> ndarray<T>
{
    if (condlist.size() != choicelist.size())
        throw std::invalid_argument("select: condlist and choicelist size mismatch");
    if (condlist.empty())
        throw std::invalid_argument("select: empty condlist");
    std::vector<int> shape = condlist[0].shape;
    for (auto &c : condlist)
        if (c.shape != shape)
            throw std::invalid_argument("select: condlist shapes must match");
    for (auto &ch : choicelist)
        if (ch.shape != shape)
            throw std::invalid_argument("select: choicelist shapes must match condlist shape");
    ndarray<T> out(shape);
    detail::Odometer od(shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        bool found = false;
        for (std::size_t k = 0; k < condlist.size(); ++k)
        {
            if (condlist[k].get(idx))
            {
                out.set(idx, choicelist[k].get(idx));
                found = true;
                break;
            }
        }
        if (!found)
            out.set(idx, default_val);
        od.advance();
    }
    return out;
}

NP_API template <typename T> void place(ndarray<T> &arr, const ndarray<bool> &mask, const ndarray<T> &vals)
{
    if (arr.shape != mask.shape)
        throw std::invalid_argument("place: arr and mask shape mismatch");
    if (vals.size() == 0)
        throw std::invalid_argument("place: vals empty");
    std::size_t v_idx = 0;
    detail::Odometer od(arr.shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        if (mask.get(idx))
        {
            arr.set(idx, vals.data()[vals._flat_logical(v_idx % vals.size())]);
            ++v_idx;
        }
        od.advance();
    }
}

NP_API template <typename T> void place(ndarray<T> &arr, const ndarray<bool> &mask, const std::vector<T> &vals)
{
    if (vals.empty())
        throw std::invalid_argument("place: vals empty");
    ndarray<T> v(std::vector<int>{static_cast<int>(vals.size())});
    for (std::size_t i = 0; i < vals.size(); ++i)
        v.data()[i] = vals[i];
    place(arr, mask, v);
}

// ── Basic operations ────────────────────────────────────────────────

/**
 * @brief Copies values from src to dst broadcasting as necessary.
 *
 * Reference: numpy-reference/reference/generated/numpy.copyto.html
 */
NP_API template <typename T>
void copyto(ndarray<T> &dst, const ndarray<T> &src, const std::optional<ndarray<bool>> &where = std::nullopt)
{
    // Fast path: contiguous same-shape without mask → memcpy
    if (!where.has_value() && src.shape == dst.shape)
    {
        if (src.is_contiguous() && dst.is_contiguous() && src.size() == dst.size())
        {
            // Direct block copy, no per-element Odometer
            const T *__restrict s = src.data().data();
            T *__restrict d = dst.data().data();
            // Use memmove for overlapping views, memcpy otherwise
            if (s != d)
            {
                std::memcpy(d, s, src.size() * sizeof(T));
            }
            return;
        }
    }
    // Broadcast src to dst shape if needed
    ndarray<T> bsrc;
    if (src.shape == dst.shape)
    {
        bsrc = src;
    }
    else
    {
        bsrc = broadcast_to(src, dst.shape);
    }
    if (!where.has_value())
    {
        if (bsrc.is_contiguous() && dst.is_contiguous())
        {
            const T *__restrict s = bsrc.data().data();
            T *__restrict d = dst.data().data();
            std::memcpy(d, s, bsrc.size() * sizeof(T));
            return;
        }
        detail::Odometer od(dst.shape);
        while (!od.done())
        {
            dst.set(od.idx(), bsrc.get(od.idx()));
            od.advance();
        }
        return;
    }
    const auto &mask = *where;
    ndarray<bool> bmask;
    if (mask.shape == dst.shape)
    {
        bmask = mask;
    }
    else
    {
        bmask = broadcast_to(mask, dst.shape);
    }
    detail::Odometer od(dst.shape);
    while (!od.done())
    {
        if (bmask.get(od.idx()))
        {
            dst.set(od.idx(), bsrc.get(od.idx()));
        }
        od.advance();
    }
}

/**
 * @brief Number of dimensions (np.ndim).
 * Reference: numpy-reference/reference/generated/numpy.ndim.html
 */
NP_API template <typename T> NP_NODISCARD inline int ndim(const ndarray<T> &a) noexcept
{
    return static_cast<int>(a.ndim());
}

/**
 * @brief Shape of array (np.shape).
 * Reference: numpy-reference/reference/generated/numpy.shape.html
 */
NP_API template <typename T> NP_NODISCARD inline std::vector<int> shape(const ndarray<T> &a) noexcept
{
    return a.shape;
}

/**
 * @brief Size of array – total or along axis (np.size).
 * Reference: numpy-reference/reference/generated/numpy.size.html
 */
NP_API template <typename T> NP_NODISCARD inline std::size_t size(const ndarray<T> &a) noexcept
{
    return a.size();
}

NP_API template <typename T> NP_NODISCARD inline int size(const ndarray<T> &a, int axis)
{
    int ax = axis < 0 ? axis + static_cast<int>(a.ndim()) : axis;
    if (ax < 0 || ax >= static_cast<int>(a.ndim()))
    {
        throw AxisError("size: axis out of bounds");
    }
    return a.shape[static_cast<std::size_t>(ax)];
}

/**
 * @brief Alias for transpose (np.permute_dims).
 * Reference: numpy-reference/reference/generated/numpy.permute_dims.html
 */
NP_API template <typename T> NP_NODISCARD inline auto permute_dims(const ndarray<T> &a) -> ndarray<T>
{
    return a.transpose();
}

NP_API template <typename T>
NP_NODISCARD inline auto permute_dims(const ndarray<T> &a, const std::vector<int> &axes) -> ndarray<T>
{
    return a.transpose(axes);
}

/**
 * @brief Transpose last two axes (np.matrix_transpose).
 * Reference: numpy-reference/reference/generated/numpy.matrix_transpose.html
 */
NP_API template <typename T> NP_NODISCARD inline auto matrix_transpose(const ndarray<T> &x) -> ndarray<T>
{
    if (x.ndim() < 2)
    {
        throw std::invalid_argument("matrix_transpose: need at least 2-D");
    }
    int ax1 = static_cast<int>(x.ndim()) - 2;
    int ax2 = static_cast<int>(x.ndim()) - 1;
    return x.swapaxes(ax1, ax2);
}

/**
 * @brief Split array into sequence along axis, removing axis (np.unstack).
 * Reference: numpy-reference/reference/generated/numpy.unstack.html
 */
NP_API template <typename T> NP_NODISCARD auto unstack(const ndarray<T> &x, int axis = 0) -> std::vector<ndarray<T>>
{
    int ax = axis < 0 ? axis + static_cast<int>(x.ndim()) : axis;
    if (ax < 0 || ax >= static_cast<int>(x.ndim()))
    {
        throw AxisError("unstack: axis out of bounds");
    }
    int n = x.shape[static_cast<std::size_t>(ax)];
    std::vector<int> idx_spec;
    idx_spec.reserve(1);
    // Reuse split logic: slice along axis
    std::vector<ndarray<T>> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        std::vector<int> new_shape = x.shape;
        new_shape.erase(new_shape.begin() + ax);
        if (new_shape.empty())
        {
            new_shape = {1};
        }
        // Build slice by copying with fixed index on ax
        ndarray<T> sl(new_shape.empty() ? std::vector<int>{1} : new_shape);
        // Iterate over output indices and map to input
        detail::Odometer od(new_shape.empty() ? std::vector<int>{1} : new_shape);
        // Need shape for od: if we erased axis, od shape is new_shape
        std::vector<int> od_shape = new_shape;
        if (od_shape.empty())
        {
            od_shape = {1};
        }
        detail::Odometer od2(od_shape);
        while (!od2.done())
        {
            const auto &oidx = od2.idx();
            std::vector<std::size_t> src_idx(x.ndim(), 0);
            for (std::size_t d = 0, o = 0; d < x.ndim(); ++d)
            {
                if (static_cast<int>(d) == ax)
                {
                    src_idx[d] = static_cast<std::size_t>(i);
                }
                else
                {
                    src_idx[d] = oidx[o++];
                }
            }
            std::vector<std::size_t> dst_idx = oidx;
            // For scalar case where new_shape was empty
            if (new_shape.empty())
            {
                sl.data()[0] = x.get(src_idx);
                break;
            }
            sl.set(dst_idx, x.get(src_idx));
            od2.advance();
        }
        if (new_shape.empty())
        {
            // sl is shape {1} but unstack of 1-D with axis 0 and n elements
            // should produce 0-D scalars; keep as 0-D view: reshape to {}
            // Our ndarray doesn't support 0-D empty vector easily; keep 1-elem
            // but squeeze?
            out.push_back(sl);
            continue;
        }
        out.push_back(std::move(sl));
    }
    return out;
}

/**
 * @brief Return new array with specified shape, repeating data (np.resize).
 * Reference: numpy-reference/reference/generated/numpy.resize.html
 */
NP_API template <typename T>
NP_NODISCARD auto resize(const ndarray<T> &a, const std::vector<int> &new_shape) -> ndarray<T>
{
    if (new_shape.empty())
    {
        throw std::invalid_argument("resize: new_shape empty");
    }
    for (int d : new_shape)
    {
        if (d < 0)
        {
            throw std::invalid_argument("resize: negative dimension");
        }
    }
    std::size_t total = 1;
    for (int d : new_shape)
    {
        total *= static_cast<std::size_t>(d);
    }
    if (total == 0)
    {
        return ndarray<T>(new_shape);
    }
    ndarray<T> out(new_shape);
    auto flat = a.ravel();
    std::size_t src_n = flat.size();
    if (src_n == 0)
    {
        // Fill with zeros if source empty (numpy would fill with 0)
        for (std::size_t i = 0; i < total; ++i)
        {
            out.data()[i] = T{0};
        }
        return out;
    }
    for (std::size_t i = 0; i < total; ++i)
    {
        out.data()[i] = flat.data()[i % src_n];
    }
    return out;
}

/**
 * @brief Pad array (np.pad).
 *
 * Supports modes: "constant" (default), "edge", "wrap", "reflect",
 * "symmetric". `pad_width` may be an int (all axes/sides) or a vector
 * of (before, after) pairs per axis. For ND, each axis may have its
 * own pair.
 *
 * Reference: numpy-reference/reference/generated/numpy.pad.html
 */
NP_API template <typename T>
NP_NODISCARD auto pad(const ndarray<T> &array, const std::vector<std::pair<int, int>> &pad_width,
                      const std::string &mode = "constant", T constant_values = T{0}) -> ndarray<T>
{
    if (pad_width.size() != array.ndim())
    {
        throw std::invalid_argument("pad: pad_width size must match ndim");
    }
    std::vector<int> new_shape(array.ndim());
    for (std::size_t d = 0; d < array.ndim(); ++d)
    {
        int w0 = pad_width[d].first;
        int w1 = pad_width[d].second;
        if (w0 < 0 || w1 < 0)
        {
            throw std::invalid_argument("pad: negative pad_width");
        }
        new_shape[d] = array.shape[d] + w0 + w1;
    }
    ndarray<T> out(new_shape, array.type, constant_values);
    if (mode == "constant")
    {
        // already filled with constant_values, copy interior
        detail::Odometer od(array.shape);
        while (!od.done())
        {
            const auto &idx = od.idx();
            std::vector<std::size_t> dst_idx(idx.size());
            for (std::size_t d = 0; d < idx.size(); ++d)
            {
                dst_idx[d] = idx[d] + static_cast<std::size_t>(pad_width[d].first);
            }
            out.set(dst_idx, array.get(idx));
            od.advance();
        }
        return out;
    }
    if (mode == "edge" || mode == "wrap" || mode == "reflect" || mode == "symmetric")
    {
        detail::Odometer od(new_shape);
        while (!od.done())
        {
            const auto &didx = od.idx();
            std::vector<std::size_t> src_idx(array.ndim());
            bool in_interior = true;
            for (std::size_t d = 0; d < array.ndim(); ++d)
            {
                int p0 = pad_width[d].first;
                int n = array.shape[d];
                int pos = static_cast<int>(didx[d]);
                if (pos < p0 || pos >= p0 + n)
                {
                    in_interior = false;
                }
            }
            if (in_interior)
            {
                // Already handled by constant path? For non-constant we still need interior
                // copy
                std::vector<std::size_t> sidx(array.ndim());
                for (std::size_t d = 0; d < array.ndim(); ++d)
                {
                    sidx[d] = didx[d] - static_cast<std::size_t>(pad_width[d].first);
                }
                out.set(didx, array.get(sidx));
            }
            else
            {
                // Map didx to src per mode
                for (std::size_t d = 0; d < array.ndim(); ++d)
                {
                    int p0 = pad_width[d].first;
                    int n = array.shape[d];
                    int pos = static_cast<int>(didx[d]) - p0;
                    int src = 0;
                    if (mode == "edge")
                    {
                        src = std::clamp(pos, 0, n - 1);
                    }
                    else if (mode == "wrap")
                    {
                        src = ((pos % n) + n) % n;
                    }
                    else if (mode == "reflect")
                    {
                        // reflect without repeating edge: 0 1 2 | 1 0 | 1 2 ...
                        int period = 2 * n - 2;
                        if (period <= 0)
                        {
                            src = 0;
                        }
                        else
                        {
                            int p = ((pos % period) + period) % period;
                            src = p < n ? p : period - p;
                        }
                    }
                    else // symmetric (reflect with edge)
                    {
                        int period = 2 * n;
                        if (period == 0)
                        {
                            src = 0;
                        }
                        else
                        {
                            int p = ((pos % period) + period) % period;
                            src = p < n ? p : period - 1 - p;
                        }
                    }
                    src_idx[d] = static_cast<std::size_t>(src);
                }
                out.set(didx, array.get(src_idx));
            }
            od.advance();
        }
        return out;
    }
    throw std::invalid_argument("pad: unsupported mode '" + mode + "'");
}

NP_API template <typename T>
NP_NODISCARD inline auto pad(const ndarray<T> &array, int pad_width, const std::string &mode = "constant",
                             T constant_values = T{0}) -> ndarray<T>
{
    std::vector<std::pair<int, int>> pw(array.ndim(), {pad_width, pad_width});
    return pad(array, pw, mode, constant_values);
}

NP_API template <typename T>
NP_NODISCARD inline auto pad(const ndarray<T> &array, std::pair<int, int> pad_width,
                             const std::string &mode = "constant", T constant_values = T{0}) -> ndarray<T>
{
    std::vector<std::pair<int, int>> pw(array.ndim(), pad_width);
    return pad(array, pw, mode, constant_values);
}

// ── Broadcast object (np.broadcast) ─────────────────────────────────
/**
 * @brief Broadcast object mimicking `np.broadcast(*arrays)`.
 *
 * Holds the broadcast shape of all inputs and provides `nd`/`numiter`/
 * `shape`/`size` attributes. Iteration is via index counter.
 *
 * Reference: numpy-reference/reference/generated/numpy.broadcast.html
 */
struct broadcast
{
    int nd = 0;                           ///< Number of dimensions of broadcast
    std::vector<int> shape;               ///< Broadcast shape
    std::size_t size = 0;                 ///< Total elements in broadcast
    int numiter = 0;                      ///< Number of arrays
    std::size_t index = 0;                ///< Current flat index
    std::vector<std::vector<int>> shapes; ///< Shapes of input arrays

    broadcast() = default;

    explicit broadcast(const std::vector<std::vector<int>> &shapes_)
        : numiter(static_cast<int>(shapes_.size())), shapes(shapes_)
    {
        if (shapes.empty())
        {
            nd = 0;
            shape = {};
            size = 0;
            return;
        }
        shape = shapes[0];
        for (std::size_t i = 1; i < shapes.size(); ++i)
        {
            shape = detail::broadcast_shapes(shape, shapes[i]);
        }
        nd = static_cast<int>(shape.size());
        size = 1;
        for (int d : shape)
        {
            size *= static_cast<std::size_t>(d);
        }
    }

    template <typename T>
    explicit broadcast(const std::vector<ndarray<T>> &arrays) : numiter(static_cast<int>(arrays.size()))
    {
        std::vector<std::vector<int>> sh;
        sh.reserve(arrays.size());
        for (auto &a : arrays)
        {
            sh.push_back(a.shape);
        }
        shapes = sh;
        if (sh.empty())
        {
            nd = 0;
            shape = {};
            size = 0;
            return;
        }
        shape = sh[0];
        for (std::size_t i = 1; i < sh.size(); ++i)
        {
            shape = detail::broadcast_shapes(shape, sh[i]);
        }
        nd = static_cast<int>(shape.size());
        size = 1;
        for (int d : shape)
        {
            size *= static_cast<std::size_t>(d);
        }
    }

    void reset() noexcept
    {
        index = 0;
    }

    bool has_next() const noexcept
    {
        return index < size;
    }
};

/**
 * @brief Alias for `np.concatenate` (NumPy 2.0 `np.concat`).
 *
 * Reference: numpy-reference/reference/generated/numpy.concat.html
 */
#ifndef NP_CONCATENATE_HPP
NP_API template <typename T>
NP_NODISCARD inline auto concat(const std::vector<ndarray<T>> &arrays, int axis = 0) -> ndarray<T>
{
    // Forward to concatenate in concatenate.hpp when available; otherwise
    // perform a local implementation using broadcast_to/copy logic.
    // To avoid circular include, implement inline using detail helpers:
    if (arrays.empty())
    {
        throw std::invalid_argument("concat: need at least one array");
    }
    int ndim = static_cast<int>(arrays[0].ndim());
    int ax = axis < 0 ? axis + ndim : axis;
    if (ax < 0 || ax >= ndim)
    {
        throw std::invalid_argument("concat: axis out of bounds");
    }
    for (auto &a : arrays)
    {
        if (static_cast<int>(a.ndim()) != ndim)
        {
            throw std::invalid_argument("concat: ndim mismatch");
        }
        for (int d = 0; d < ndim; ++d)
        {
            if (d != ax && a.shape[d] != arrays[0].shape[d])
            {
                throw std::invalid_argument("concat: shape mismatch");
            }
        }
    }
    std::vector<int> out_shape = arrays[0].shape;
    for (std::size_t i = 1; i < arrays.size(); ++i)
    {
        out_shape[ax] += arrays[i].shape[ax];
    }
    ndarray<T> out(out_shape);
    std::size_t off = 0;
    for (auto &arr : arrays)
    {
        std::size_t axis_n = static_cast<std::size_t>(arr.shape[ax]);
        detail::Odometer od(arr.shape);
        while (!od.done())
        {
            auto sidx = od.idx();
            auto didx = sidx;
            didx[static_cast<std::size_t>(ax)] += off;
            out.set(didx, arr.get(sidx));
            od.advance();
        }
        off += axis_n;
    }
    return out;
}
#endif // NP_CONCATENATE_HPP

/**
 * @brief `np.delete` alias – C++ keyword workaround.
 *
 * The C++ keyword `delete` cannot be used as a function name, so the
 * implementation is `delete_arr`. This alias `np_delete` and the
 * overload `remove` provide the same semantics under a legal identifier.
 *
 * Reference: numpy-reference/reference/generated/numpy.delete.html
 */
NP_API template <typename T>
NP_NODISCARD inline auto np_delete(const ndarray<T> &arr, const std::vector<int> &indices,
                                   std::optional<int> axis = std::nullopt) -> ndarray<T>
{
    return delete_arr(arr, indices, axis);
}

// ── flat view helper (np.flatiter / np.ndarray.flat) ────────────
/**
 * @brief Flat iterator view (np.ndarray.flat).
 * Returns a 1-D copy that iterates in C order (logical flat).
 *
 * Reference: numpy-reference/reference/generated/numpy.ndarray.flat.html
 */
NP_API template <typename T> NP_NODISCARD inline auto flat(const ndarray<T> &a) -> ndarray<T>
{
    return a.ravel();
}

/**
 * @brief Strided view (np.lib.stride_tricks.as_strided).
 *
 * Reference:
 * numpy-reference/reference/generated/numpy.lib.stride_tricks.as_strided.html
 */
NP_API template <typename T>
NP_NODISCARD inline auto as_strided(const ndarray<T> &x, const std::vector<int> &shape,
                                    const std::vector<std::size_t> &strides) -> ndarray<T>
{
    ndarray<T> out = x;
    out.shape = shape;
    out.strides = strides;
    // keep same offset and shared storage (view)
    return out;
}

/**
 * @brief Sliding window view (np.lib.stride_tricks.sliding_window_view).
 *
 * Reference:
 * numpy-reference/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
 */
NP_API template <typename T>
NP_NODISCARD inline auto sliding_window_view(const ndarray<T> &x, int window_shape, int axis = -1) -> ndarray<T>
{
    if (x.ndim() == 0)
        throw std::invalid_argument("sliding_window_view: 0-d not supported");
    int ax = axis == -1 ? static_cast<int>(x.ndim() - 1) : axis;
    if (ax < 0)
        ax += static_cast<int>(x.ndim());
    if (ax < 0 || ax >= static_cast<int>(x.ndim()))
        throw AxisError("sliding_window_view: axis out of bounds");
    int n = x.shape[ax];
    if (window_shape <= 0 || window_shape > n)
        throw std::invalid_argument("sliding_window_view: invalid window_shape");
    std::vector<int> out_shape = x.shape;
    out_shape[ax] = n - window_shape + 1;
    out_shape.push_back(window_shape);
    // Simplified: return strided copy via as_strided semantics
    ndarray<T> out(out_shape);
    detail::Odometer od(out_shape);
    while (!od.done())
    {
        std::vector<std::size_t> out_idx = od.idx();
        int win_idx = static_cast<int>(out_idx.back());
        out_idx.pop_back();
        std::vector<std::size_t> src_idx(x.ndim(), 0);
        for (size_t d = 0, o = 0; d < x.ndim(); ++d)
        {
            if (static_cast<int>(d) == ax)
                src_idx[d] = out_idx[o++] + win_idx;
            else
                src_idx[d] = out_idx[o++];
        }
        out.set(od.idx(), x.get(src_idx));
        od.advance();
    }
    return out;
}

// Broadcast shapes helper exposed as np.broadcast_shapes
NP_API inline auto broadcast_shapes(const std::vector<int> &a, const std::vector<int> &b) -> std::vector<int>
{
    return detail::broadcast_shapes(a, b);
}

NP_API inline auto broadcast_shapes(const std::vector<int> &a, const std::vector<int> &b, const std::vector<int> &c)
    -> std::vector<int>
{
    return detail::broadcast_shapes(detail::broadcast_shapes(a, b), c);
}

} // namespace np

#endif // NP_MANIPULATION_HPP
