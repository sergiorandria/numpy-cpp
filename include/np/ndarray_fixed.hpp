/**
 * @file ndarray_fixed.hpp
 * @brief Compile-time fixed-shape arrays: ndarrayf<T, Extents...>.
 *
 * This is the compile-time-first axis of the library:
 *  - every operation validates shape compatibility and broadcast legality
 *    via static_assert / concept constraints at the call site;
 *  - reductions, reshaping and elementwise expressions are constexpr, so
 *    fully-static computations fold in constant expressions
 *    (see tests/test_constexpr.cpp);
 *  - elementwise operators build lazy expression trees (detail/expr.hpp)
 *    that fuse into a single pass at materialization (eval()).
 *
 * NumPy reference (semantics mirrored here):
 *   numpy-reference/reference/arrays.ndarray.html
 *   numpy-reference/reference/routines.array-manipulation.html
 */
#ifndef NP_NDARRAY_FIXED_HPP
#define NP_NDARRAY_FIXED_HPP

#include "api_macros.hpp"

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#if __cplusplus >= 202302L && __has_include(<mdspan>)
#include <mdspan>
#endif
#include <type_traits>
#include <utility>

#include "detail/expr.hpp"
#include "detail/math_constexpr.hpp"
#include "detail/scalar_custom.hpp"
#include "exceptions.hpp"
#include "ndarray.hpp"
#if __has_include("bigint.hpp")
#include "bigint.hpp"
#endif

namespace np::detail::fixed
{

/** @brief Floating-point promotion used by mean/std/linspace (NumPy: int ->
 * float64). */
template <typename V> using float_t = std::conditional_t<std::is_floating_point_v<V>, V, double>;

} // namespace np::detail::fixed

namespace np
{

/**
 * @brief Fixed-shape array with compile-time extents.
 *
 * Nested-brace construction matches NumPy nested-list literals:
 *     np::ndarrayf<int, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
 *
 * Documented deviations from the dynamic ndarray<T>:
 *  - indexing `a[i]` is the flat index; `a[i, j]` is multi-dimensional;
 *  - views do not exist here: transpose/squeeze/expand_dims return new
 *    arrays (value semantics) because the result shape must be static;
 *  - extents must be positive (NumPy allows empty dims; this class does
 *    not, so every shape check stays a constant expression);
 *  - reduction axes are template parameters (NumPy axis=None is
 *    Axis = -1), validated at compile time.
 *
 * @tparam T       element type
 * @tparam Extents compile-time extents (row-major)
 */
template <typename T, int... Extents> class ndarrayf
{
    static_assert(std::conjunction_v<std::bool_constant<(Extents > 0)>...>, "np: ndarrayf extents must be positive");

  public:
    using _RawT = T;
    using value_type = typename dtype_tag_to_type<T>::type;
    static constexpr std::size_t rank = sizeof...(Extents);
    static constexpr std::size_t size_v = (static_cast<std::size_t>(Extents) * ... * 1ull);
    static constexpr std::array<int, rank> static_shape{Extents...};

    /** @brief Row-major flat storage (public, std::array-style). */
    std::array<value_type, size_v> m_data{};

    constexpr ndarrayf() = default;

    /** @brief From a flat buffer (numpy.asarray semantics). */
    constexpr explicit ndarrayf(const std::array<value_type, size_v> &flat) : m_data(flat)
    {
    }

    /** @brief Rank-1 flat list: ndarrayf<int, 3>{1, 2, 3}. */
    template <typename U>
        requires(rank == 1 && std::convertible_to<U, T>)
    constexpr explicit ndarrayf(std::initializer_list<U> flat) : m_data()
    {
        if (flat.size() != size_v)
        {
            throw std::invalid_argument("np: initializer list size does not match the fixed extent");
        }
        std::size_t i = 0;
        for (const U &v : flat)
        {
            m_data[i++] = static_cast<T>(v);
        }
    }

    /**
     * @brief Rank-2 nested rows: ndarrayf<int, 2, 3>{{1, 2, 3}, {4, 5, 6}}.
     *        Ragged rows throw at runtime (init-list sizes are not
     *        compile-time constants); in constant evaluation a mismatch
     *        is a compile error.
     */
    template <typename U>
        requires(rank == 2 && std::convertible_to<U, T>)
    constexpr explicit ndarrayf(std::initializer_list<std::initializer_list<U>> rows) : m_data()
    {
        const std::size_t cols = static_cast<std::size_t>(static_shape[1]);
        if (rows.size() != static_cast<std::size_t>(static_shape[0]))
        {
            throw std::invalid_argument("np: row count does not match the fixed extent");
        }
        std::size_t r = 0;
        for (const auto &row : rows)
        {
            if (row.size() != cols)
            {
                throw std::invalid_argument("np: ragged rows in nested initializer list");
            }
            std::size_t c = 0;
            for (const U &v : row)
            {
                m_data[r * cols + c++] = static_cast<T>(v);
            }
            ++r;
        }
    }

    /** @brief Rank-0 scalar: ndarrayf<int>{5}. */
    template <typename U>
        requires(rank == 0 && std::convertible_to<U, T>)
    constexpr explicit ndarrayf(U value) : m_data{static_cast<T>(value)}
    {
    }

    // Factories
    /**
     * @brief Array from a flat std::array (C++ analog of numpy.asarray).
     * Reference: numpy-reference/reference/generated/numpy.asarray.html
     */
    static constexpr ndarrayf from_data(const std::array<value_type, size_v> &flat)
    {
        return ndarrayf{flat};
    }

    // Access
    constexpr std::size_t size() const
    {
        return size_v;
    }

#if __cplusplus >= 202302L && __has_include(<mdspan>)
    // C++23 mdspan view — zero-cost, non-owning, row-major
    [[nodiscard]] constexpr auto mdspan() noexcept
    {
        return std::mdspan<value_type, std::dextents<std::size_t, rank>>(m_data.data(),
                                                                         static_cast<std::size_t>(Extents)...);
    }
    [[nodiscard]] constexpr auto mdspan() const noexcept
    {
        return std::mdspan<const value_type, std::dextents<std::size_t, rank>>(m_data.data(),
                                                                               static_cast<std::size_t>(Extents)...);
    }
#endif

    constexpr value_type *data()
    {
        return m_data.data();
    }
    constexpr const value_type *data() const
    {
        return m_data.data();
    }

    constexpr auto begin()
    {
        return m_data.begin();
    }
    constexpr auto end()
    {
        return m_data.end();
    }
    constexpr auto begin() const
    {
        return m_data.begin();
    }
    constexpr auto end() const
    {
        return m_data.end();
    }

    /** @brief Flat (row-major) element access. */
    constexpr value_type &operator[](std::size_t i)
    {
        static_assert(size_v > 0, "ndarrayf: size must be positive");
        static_assert(rank >= 1, "operator[] requires rank >=1");
        return m_data[i];
    }
    constexpr const value_type &operator[](std::size_t i) const
    {
        static_assert(size_v > 0, "ndarrayf: size must be positive");
        static_assert(rank >= 1, "operator[] requires rank >=1");
        return m_data[i];
    }

    /**
     * @brief Multi-dimensional access: a(i0, i1, ..., iN-1).
     *        (The arr[i, j, k] subscript form needs C++23 P2128, so the
     *        fixed path uses operator() — the same convention as the
     *        dynamic ndarray.)
     */
    template <typename... Idx>
        requires(sizeof...(Idx) == rank && rank >= 2)
    constexpr value_type &operator()(Idx... idx)
    {
        static_assert(sizeof...(Idx) == rank, "ndarrayf: index count must equal rank");
        static_assert(rank >= 2, "operator() requires rank >=2");
        static_assert((std::is_convertible_v<Idx, std::size_t> && ...), "indices must be convertible to size_t");
        const std::array<std::size_t, rank> coords{static_cast<std::size_t>(idx)...};
        return m_data[detail::expr::flatten(static_shape, coords)];
    }

    template <typename... Idx>
        requires(sizeof...(Idx) == rank && rank >= 2)
    constexpr const value_type &operator()(Idx... idx) const
    {
        static_assert(sizeof...(Idx) == rank, "ndarrayf: index count must equal rank");
        static_assert(rank >= 2, "operator() requires rank >=2");
        static_assert((std::is_convertible_v<Idx, std::size_t> && ...), "indices must be convertible to size_t");
        const std::array<std::size_t, rank> coords{static_cast<std::size_t>(idx)...};
        return m_data[detail::expr::flatten(static_shape, coords)];
    }

    /** @brief Fill every element. */
    constexpr void fill(const value_type &v)
    {
        static_assert(std::is_copy_assignable_v<value_type>, "fill: value_type must be copy assignable");
        static_assert(size_v > 0, "fill: size must be positive");
        m_data.fill(v);
    }

    // Reductions
    // (numpy-reference/reference/routines.statistics.html)
    // NumPy axis=None is represented by Axis = -1; every axis is a
    // template parameter so out-of-range axes fail to compile.
    /** @brief Sum (numpy-reference/reference/generated/numpy.sum.html). */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto sum() const
    {
        static_assert(Axis >= -1 && Axis < static_cast<int>(rank), "sum: Axis out of range");
        static_assert(size_v > 0, "sum: size must be positive");
        static_assert(std::is_copy_constructible_v<value_type>, "sum: value_type must be copy constructible");
        using traits = detail::fixed::scalar_traits<T>;
        using V = typename traits::value_type;
        if constexpr (Axis == -1)
        {
            V acc = traits::zero();
            for (std::size_t i = 0; i < size_v; ++i)
            {
                acc += traits::get(m_data[i]);
            }
            return traits::make(acc);
        }
        else
        {
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return axis_reduce<V>(
                Axis, tag{}, [](V acc, const V &v) { return acc + v; }, traits::zero(),
                [](V v) { return traits::make(v); }, [](const T &v) { return traits::get(v); });
        }
    }

    /** @brief Product (numpy-reference/reference/generated/numpy.prod.html). */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto prod() const
    {
        static_assert(Axis >= -1 && Axis < static_cast<int>(rank), "prod: Axis out of range");
        static_assert(size_v > 0, "prod: size must be positive");
        static_assert(std::is_copy_constructible_v<value_type>, "prod: value_type must be copy constructible");
        using traits = detail::fixed::scalar_traits<T>;
        using V = typename traits::value_type;
        if constexpr (Axis == -1)
        {
            V acc = traits::one();
            for (std::size_t i = 0; i < size_v; ++i)
            {
                acc *= traits::get(m_data[i]);
            }
            return traits::make(acc);
        }
        else
        {
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return axis_reduce<V>(
                Axis, tag{}, [](V acc, const V &v) { return acc * v; }, traits::one(),
                [](V v) { return traits::make(v); }, [](const T &v) { return traits::get(v); });
        }
    }

    /**
     * @brief Arithmetic mean (numpy.mean.html). Integer inputs promote
     *        to double, as in NumPy.
     */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto mean() const
    {
        static_assert(Axis >= -1 && Axis < static_cast<int>(rank), "mean: Axis out of range");
        static_assert(size_v > 0, "mean: size must be positive");
        static_assert(std::is_copy_constructible_v<value_type>, "mean: value_type must be copy constructible");
        using traits = detail::fixed::scalar_traits<T>;
        using R = detail::fixed::float_t<typename traits::value_type>;
        if constexpr (Axis == -1)
        {
            R acc = R{0};
            for (std::size_t i = 0; i < size_v; ++i)
            {
                acc += static_cast<R>(traits::get(m_data[i]));
            }
            return acc / static_cast<R>(size_v);
        }
        else
        {
            constexpr std::size_t n = static_cast<std::size_t>(static_shape[Axis]);
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return axis_reduce<R>(
                Axis, tag{}, [](R acc, const R &v) { return acc + v; }, R{0},
                [n](R v) { return v / static_cast<R>(n); }, [](const T &v) { return static_cast<R>(traits::get(v)); });
        }
    }

    /**
     * @brief Standard deviation (numpy.std.html). Two-pass computation;
     *        ddof matches NumPy.
     */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto std(std::optional<int> ddof = std::nullopt) const
    {
        using traits = detail::fixed::scalar_traits<T>;
        using R = detail::fixed::float_t<typename traits::value_type>;
        if constexpr (Axis == -1)
        {
            R acc = R{0};
            for (std::size_t i = 0; i < size_v; ++i)
            {
                acc += static_cast<R>(traits::get(m_data[i]));
            }
            const R mean_v = acc / static_cast<R>(size_v);
            R sq = R{0};
            for (std::size_t i = 0; i < size_v; ++i)
            {
                const R d = static_cast<R>(traits::get(m_data[i])) - mean_v;
                sq += d * d;
            }
            return std_denom(sq, size_v, ddof.value_or(0));
        }
        else
        {
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return std_axis_impl(Axis, tag{}, ddof.value_or(0));
        }
    }

    /** @brief Minimum (numpy-reference/reference/generated/numpy.min.html). */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto min() const
    {
        using traits = detail::fixed::scalar_traits<T>;
        using V = typename traits::value_type;
        if constexpr (Axis == -1)
        {
            V best = traits::get(m_data[0]);
            for (std::size_t i = 1; i < size_v; ++i)
            {
                const V v = traits::get(m_data[i]);
                if (v < best)
                {
                    best = v;
                }
            }
            return traits::make(best);
        }
        else
        {
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return extremum_axis(Axis, tag{}, false);
        }
    }

    /** @brief Maximum (numpy-reference/reference/generated/numpy.max.html). */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto max() const
    {
        using traits = detail::fixed::scalar_traits<T>;
        using V = typename traits::value_type;
        if constexpr (Axis == -1)
        {
            V best = traits::get(m_data[0]);
            for (std::size_t i = 1; i < size_v; ++i)
            {
                const V v = traits::get(m_data[i]);
                if (best < v)
                {
                    best = v;
                }
            }
            return traits::make(best);
        }
        else
        {
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return extremum_axis(Axis, tag{}, true);
        }
    }

    /**
     * @brief Index of the first minimum (numpy.argmin.html). Axis = -1
     *        flattens, matching NumPy's default axis=None.
     */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto argmin() const
    {
        return argextremum<Axis>(false);
    }

    /**
     * @brief Index of the first maximum (numpy.argmax.html). Axis = -1
     *        flattens, matching NumPy's default axis=None.
     */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto argmax() const
    {
        return argextremum<Axis>(true);
    }

    /**
     * @brief True when every element is truthy (numpy.all.html).
     */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto all() const
    {
        using traits = detail::fixed::scalar_traits<T>;
        if constexpr (Axis == -1)
        {
            for (std::size_t i = 0; i < size_v; ++i)
            {
                if (!traits::truthy(m_data[i]))
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return axis_reduce<bool>(
                Axis, tag{}, [](bool acc, bool v) { return acc && v; }, true, [](bool v) { return v; },
                [](const T &v) { return traits::truthy(v); });
        }
    }

    /**
     * @brief True when any element is truthy (numpy.any.html).
     */
    template <int Axis = -1>
        requires(Axis >= -1 && Axis < static_cast<int>(rank))
    constexpr auto any() const
    {
        using traits = detail::fixed::scalar_traits<T>;
        if constexpr (Axis == -1)
        {
            for (std::size_t i = 0; i < size_v; ++i)
            {
                if (traits::truthy(m_data[i]))
                {
                    return true;
                }
            }
            return false;
        }
        else
        {
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return axis_reduce<bool>(
                Axis, tag{}, [](bool acc, bool v) { return acc || v; }, false, [](bool v) { return v; },
                [](const T &v) { return traits::truthy(v); });
        }
    }

    // Manipulation
    // (numpy-reference/reference/routines.array-manipulation.html)
    // All return fresh arrays with static result shapes.
    /**
     * @brief Reverse the axes (numpy.transpose.html), mirroring
     *        numpy.transpose(a) without an axis argument.
     */
    constexpr auto transpose() const
    {
        using tag = typename detail::expr::reverse<detail::expr::shape_tag<Extents...>>::type;
        return transpose_impl(tag{});
    }

    /**
     * @brief Same elements, new extents (numpy.reshape.html). The
     *        element count is verified at compile time.
     */
    template <int... NewExtents>
        requires((static_cast<std::size_t>(NewExtents) * ... * 1ull) == size_v &&
                 std::conjunction_v<std::bool_constant<(NewExtents > 0)>...>)
    constexpr auto reshape() const
    {
        static_assert((static_cast<std::size_t>(NewExtents) * ... * 1ull) == size_v,
                      "np: reshape cannot change the number of elements "
                      "(numpy-reference/reference/generated/numpy.reshape.html)");
        using tag = detail::expr::shape_tag<NewExtents...>;
        return reshape_impl(tag{});
    }

    /** @brief 1D copy in logical order (numpy.ndarray.flatten.html). */
    constexpr auto flatten() const
    {
        return reshape<static_cast<int>(size_v)>();
    }

    /**
     * @brief Remove every extent-1 axis (numpy.squeeze.html), mirroring
     *        numpy.squeeze(a) with axis=None.
     */
    constexpr auto squeeze() const
    {
        using tag = typename detail::expr::squeeze_tag<detail::expr::shape_tag<Extents...>>::type;
        return squeeze_impl(tag{});
    }

    /** @brief Remove one extent-1 axis (numpy.squeeze.html, axis=...). */
    template <int Axis>
        requires(Axis >= 0 && Axis < static_cast<int>(rank))
    constexpr auto squeeze() const
    {
        static_assert(Axis >= 0 && Axis < static_cast<int>(rank), "np: squeeze axis out of range");
        static_assert(static_shape[Axis] == 1, "np: squeeze(Axis) requires the axis extent to be 1 "
                                               "(numpy-reference/reference/generated/numpy.squeeze.html)");
        using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
        return squeeze_axis_impl<Axis>(tag{});
    }

    /**
     * @brief Squeeze with optional axis (std::nullopt = remove all).
     *        Runtime variant that returns a dynamic ndarray.
     * @param axis Optional axis to remove; std::nullopt removes all
     *             extent-1 axes.
     * @return Dynamic ndarray with squeezed shape.
     */
    auto squeeze(std::optional<int> axis) const -> np::ndarray<T>
    {
        if (!axis.has_value())
        {
            // Remove all 1-extents: build dynamic shape
            std::vector<int> out_shape;
            for (std::size_t d = 0; d < rank; ++d)
            {
                if (static_shape[d] != 1)
                {
                    out_shape.push_back(static_shape[d]);
                }
            }
            np::ndarray<T> out(out_shape);
            for (std::size_t i = 0; i < size_v; ++i)
            {
                out.data()[i] = m_data[i];
            }
            return out;
        }
        int ax = *axis;
        if (ax < 0)
        {
            ax += static_cast<int>(rank);
        }
        if (ax < 0 || ax >= static_cast<int>(rank))
        {
            throw np::AxisError("squeeze axis out of bounds");
        }
        if (static_shape[ax] != 1)
        {
            throw std::invalid_argument("squeeze axis must have extent 1");
        }
        std::vector<int> out_shape;
        out_shape.reserve(rank - 1);
        for (std::size_t d = 0; d < rank; ++d)
        {
            if (static_cast<int>(d) != ax)
            {
                out_shape.push_back(static_shape[d]);
            }
        }
        np::ndarray<T> out(out_shape);
        for (std::size_t i = 0; i < size_v; ++i)
        {
            out.data()[i] = m_data[i];
        }
        return out;
    }

    /**
     * @brief Insert a new extent-1 axis at position `Axis`
     *        (numpy.expand_dims.html). Axis may equal rank (append).
     */
    template <int Axis>
        requires(Axis >= 0 && Axis <= static_cast<int>(rank))
    constexpr auto expand_dims() const
    {
        static_assert(Axis >= 0 && Axis <= static_cast<int>(rank),
                      "np: expand_dims axis out of range "
                      "(numpy-reference/reference/generated/numpy.expand_dims.html)");
        using tag = typename detail::expr::insert<1, Axis, 0, detail::expr::shape_tag<Extents...>>::type;
        return expand_dims_impl<Axis>(tag{});
    }

    // Internal constexpr helpers
  private:
    constexpr const value_type &axis_elem(int axis, const std::array<std::size_t, rank - 1> &cr, std::size_t a) const
    {
        std::array<std::size_t, rank> full{};
        std::size_t k = 0;
        for (std::size_t d = 0; d < rank; ++d)
        {
            full[d] = (static_cast<int>(d) == axis) ? a : cr[k++];
        }
        return m_data[detail::expr::flatten(static_shape, full)];
    }

    template <typename R, int... E, typename Op, typename Finish, typename Convert>
    constexpr auto axis_reduce(int axis, detail::expr::shape_tag<E...>, Op &&op, R init, Finish &&finish,
                               Convert &&convert) const
    {
        using Out = std::remove_cv_t<std::invoke_result_t<Finish &, R>>;
        constexpr std::size_t red_numel = (static_cast<std::size_t>(E) * ... * 1ull);
        const std::size_t axis_ext = static_cast<std::size_t>(static_shape[axis]);
        ndarrayf<Out, E...> out{};
        for (std::size_t j = 0; j < red_numel; ++j)
        {
            const auto cr = detail::expr::unflatten(j, out.static_shape);
            R acc = init;
            for (std::size_t a = 0; a < axis_ext; ++a)
            {
                acc = op(acc, convert(axis_elem(axis, cr, a)));
            }
            out.m_data[j] = finish(acc);
        }
        return out;
    }

    template <typename R, int... E>
    constexpr ndarrayf<R, E...> std_axis_impl(int axis, detail::expr::shape_tag<E...>, int ddof) const
    {
        using traits = detail::fixed::scalar_traits<T>;
        constexpr std::size_t red_numel = (static_cast<std::size_t>(E) * ... * 1ull);
        const std::size_t n = static_cast<std::size_t>(static_shape[axis]);
        ndarrayf<R, E...> out{};
        for (std::size_t j = 0; j < red_numel; ++j)
        {
            const auto cr = detail::expr::unflatten(j, out.static_shape);
            R mean_acc = R{0};
            for (std::size_t a = 0; a < n; ++a)
            {
                mean_acc += static_cast<R>(traits::get(axis_elem(axis, cr, a)));
            }
            mean_acc /= static_cast<R>(n);
            R sq = R{0};
            for (std::size_t a = 0; a < n; ++a)
            {
                const R d = static_cast<R>(traits::get(axis_elem(axis, cr, a))) - mean_acc;
                sq += d * d;
            }
            out.m_data[j] = static_cast<R>(std_denom(sq, n, ddof));
        }
        return out;
    }

    template <int... E>
    constexpr ndarrayf<T, E...> extremum_axis(int axis, detail::expr::shape_tag<E...>, bool is_max) const
    {
        using traits = detail::fixed::scalar_traits<T>;
        using V = typename traits::value_type;
        constexpr std::size_t red_numel = (static_cast<std::size_t>(E) * ... * 1ull);
        const std::size_t n = static_cast<std::size_t>(static_shape[axis]);
        ndarrayf<T, E...> out{};
        for (std::size_t j = 0; j < red_numel; ++j)
        {
            const auto cr = detail::expr::unflatten(j, out.static_shape);
            V best = traits::get(axis_elem(axis, cr, 0));
            for (std::size_t a = 1; a < n; ++a)
            {
                const V v = traits::get(axis_elem(axis, cr, a));
                if (is_max ? (best < v) : (v < best))
                {
                    best = v;
                }
            }
            out.m_data[j] = traits::make(best);
        }
        return out;
    }

    template <int Axis> constexpr auto argextremum(bool is_max) const
    {
        using traits = detail::fixed::scalar_traits<T>;
        using V = typename traits::value_type;
        if constexpr (Axis == -1)
        {
            std::size_t best_i = 0;
            for (std::size_t i = 1; i < size_v; ++i)
            {
                const V cur = traits::get(m_data[i]);
                const V best = traits::get(m_data[best_i]);
                if (is_max ? (best < cur) : (cur < best))
                {
                    best_i = i;
                }
            }
            return best_i;
        }
        else
        {
            using tag = typename detail::expr::remove_at<Axis, 0, detail::expr::shape_tag<Extents...>>::type;
            return argextremum_axis(Axis, tag{}, is_max);
        }
    }

    template <int... E>
    constexpr ndarrayf<std::size_t, E...> argextremum_axis(int axis, detail::expr::shape_tag<E...>, bool is_max) const
    {
        using traits = detail::fixed::scalar_traits<T>;
        using V = typename traits::value_type;
        constexpr std::size_t red_numel = (static_cast<std::size_t>(E) * ... * 1ull);
        const std::size_t n = static_cast<std::size_t>(static_shape[axis]);
        ndarrayf<std::size_t, E...> out{};
        for (std::size_t j = 0; j < red_numel; ++j)
        {
            const auto cr = detail::expr::unflatten(j, out.static_shape);
            std::size_t best_i = 0;
            for (std::size_t a = 1; a < n; ++a)
            {
                const V cur = traits::get(axis_elem(axis, cr, a));
                const V best = traits::get(axis_elem(axis, cr, best_i));
                if (is_max ? (best < cur) : (cur < best))
                {
                    best_i = a;
                }
            }
            out.m_data[j] = best_i;
        }
        return out;
    }

    static constexpr double std_denom(double sq, std::size_t n, int ddof)
    {
        if (n <= static_cast<std::size_t>(ddof))
        {
            return detail::math::nan();
        }
        return detail::math::sqrt(sq / static_cast<double>(n - static_cast<std::size_t>(ddof)));
    }

    template <int... E> constexpr ndarrayf<T, E...> transpose_impl(detail::expr::shape_tag<E...>) const
    {
        constexpr std::array<int, rank> rshape{E...};
        ndarrayf<T, E...> out{};
        for (std::size_t i = 0; i < size_v; ++i)
        {
            const auto c = detail::expr::unflatten(i, static_shape);
            std::array<std::size_t, rank> rc{};
            for (std::size_t d = 0; d < rank; ++d)
            {
                rc[rank - 1 - d] = c[d];
            }
            out.m_data[detail::expr::flatten(rshape, rc)] = m_data[i];
        }
        return out;
    }

    template <int... E> constexpr ndarrayf<T, E...> reshape_impl(detail::expr::shape_tag<E...>) const
    {
        ndarrayf<T, E...> out{};
        for (std::size_t i = 0; i < size_v; ++i)
        {
            out.m_data[i] = m_data[i];
        }
        return out;
    }

    template <int... E> constexpr ndarrayf<T, E...> squeeze_impl(detail::expr::shape_tag<E...>) const
    {
        constexpr std::array<int, sizeof...(E)> rshape{E...};
        ndarrayf<T, E...> out{};
        for (std::size_t i = 0; i < size_v; ++i)
        {
            const auto c = detail::expr::unflatten(i, static_shape);
            std::array<std::size_t, sizeof...(E)> rc{};
            std::size_t m = 0;
            for (std::size_t d = 0; d < rank; ++d)
            {
                if (static_shape[d] != 1)
                {
                    rc[m++] = c[d];
                }
            }
            out.m_data[detail::expr::flatten(rshape, rc)] = m_data[i];
        }
        return out;
    }

    template <int Axis, int... E> constexpr ndarrayf<T, E...> squeeze_axis_impl(detail::expr::shape_tag<E...>) const
    {
        constexpr std::array<int, rank - 1> rshape{E...};
        ndarrayf<T, E...> out{};
        for (std::size_t i = 0; i < size_v; ++i)
        {
            const auto c = detail::expr::unflatten(i, static_shape);
            std::array<std::size_t, rank - 1> rc{};
            std::size_t m = 0;
            for (std::size_t d = 0; d < rank; ++d)
            {
                if (static_cast<int>(d) != Axis)
                {
                    rc[m++] = c[d];
                }
            }
            out.m_data[detail::expr::flatten(rshape, rc)] = m_data[i];
        }
        return out;
    }

    template <int Axis, int... E> constexpr ndarrayf<T, E...> expand_dims_impl(detail::expr::shape_tag<E...>) const
    {
        constexpr std::array<int, rank + 1> rshape{E...};
        ndarrayf<T, E...> out{};
        for (std::size_t i = 0; i < size_v; ++i)
        {
            const auto c = detail::expr::unflatten(i, static_shape);
            std::array<std::size_t, rank + 1> rc{};
            std::size_t m = 0;
            for (std::size_t d = 0; d < rank + 1; ++d)
            {
                rc[d] = (static_cast<int>(d) == Axis) ? 0 : c[m++];
            }
            out.m_data[detail::expr::flatten(rshape, rc)] = m_data[i];
        }
        return out;
    }
};

// Elementwise operators (lazy expressions, NumPy broadcasting)
// Reference: numpy-reference/user/basics.broadcasting.html and
//            numpy-reference/reference/ufuncs.html
namespace detail::fixed
{

/** @brief Wrap an arithmetic scalar in a rank-0 broadcast source. */
template <typename S> using wrap_t = std::conditional_t<detail::expr::fixed_source<S>, S, detail::expr::scalar_expr<S>>;

/** @brief True when the two operands (arrays or scalars) broadcast. */
template <typename L, typename R> inline constexpr bool binop_ok = detail::expr::broadcast_valid<wrap_t<L>, wrap_t<R>>;

/** @brief Common elementwise operation helper (see ufuncs.html). */
template <typename Op, typename L, typename R> constexpr auto make_binary(const L &l, const R &r)
{
    using LE = wrap_t<L>;
    using RE = wrap_t<R>;
    return detail::expr::binary_expr<Op, LE, RE>(LE(l), RE(r));
}

/**
 * @brief Bitwise left shift (numpy.left_shift.html). std::shift_left
 *        is not provided by this toolchain's libstdc++, so the
 *        kernel is defined here.
 */
struct left_shift_fn
{
    template <typename V1, typename V2> constexpr auto operator()(V1 x, V2 y) const -> std::common_type_t<V1, V2>
    {
        return static_cast<std::common_type_t<V1, V2>>(x) << static_cast<std::common_type_t<V1, V2>>(y);
    }
};

/**
 * @brief Bitwise right shift (numpy.right_shift.html).
 */
struct right_shift_fn
{
    template <typename V1, typename V2> constexpr auto operator()(V1 x, V2 y) const -> std::common_type_t<V1, V2>
    {
        return static_cast<std::common_type_t<V1, V2>>(x) >> static_cast<std::common_type_t<V1, V2>>(y);
    }
};

} // namespace detail::fixed

#define NP_FIXED_BINOP(op, stdop)                                                                                      \
    template <typename L, typename R>                                                                                  \
        requires((detail::expr::fixed_source<L> || std::is_arithmetic_v<L> || detail::is_bigint_v<L>) &&               \
                 (detail::expr::fixed_source<R> || std::is_arithmetic_v<R> || detail::is_bigint_v<R>) &&               \
                 !((std::is_arithmetic_v<L> || detail::is_bigint_v<L>) &&                                              \
                   (std::is_arithmetic_v<R> || detail::is_bigint_v<R>)) &&                                             \
                 detail::fixed::binop_ok<L, R>)                                                                        \
    constexpr auto operator op(const L &l, const R &r)                                                                 \
    {                                                                                                                  \
        return detail::fixed::make_binary<stdop>(l, r);                                                                \
    }

NP_FIXED_BINOP(+, std::plus<void>)
NP_FIXED_BINOP(-, std::minus<void>)
NP_FIXED_BINOP(*, std::multiplies<void>)
NP_FIXED_BINOP(/, std::divides<void>)
NP_FIXED_BINOP(%, std::modulus<void>)
NP_FIXED_BINOP(&, std::bit_and<void>)
NP_FIXED_BINOP(|, std::bit_or<void>)
NP_FIXED_BINOP(^, std::bit_xor<void>)
NP_FIXED_BINOP(<<, detail::fixed::left_shift_fn)
NP_FIXED_BINOP(>>, detail::fixed::right_shift_fn)
NP_FIXED_BINOP(==, std::equal_to<void>)
NP_FIXED_BINOP(!=, std::not_equal_to<void>)
NP_FIXED_BINOP(<, std::less<void>)
NP_FIXED_BINOP(<=, std::less_equal<void>)
NP_FIXED_BINOP(>, std::greater<void>)
NP_FIXED_BINOP(>=, std::greater_equal<void>)
NP_FIXED_BINOP(&&, std::logical_and<void>)
NP_FIXED_BINOP(||, std::logical_or<void>)

#undef NP_FIXED_BINOP

/** @brief Unary minus
 * (numpy-reference/reference/generated/numpy.negative.html). */
template <detail::expr::fixed_source S> constexpr auto operator-(const S &s)
{
    return detail::expr::unary_expr<std::negate<void>, S>(s);
}

/** @brief Unary plus: value copy
 * (numpy-reference/reference/generated/numpy.positive.html). */
template <detail::expr::fixed_source S> constexpr S operator+(const S &s)
{
    return s;
}

/** @brief Logical not
 * (numpy-reference/reference/generated/numpy.logical_not.html). */
template <detail::expr::fixed_source S> constexpr auto operator!(const S &s)
{
    return detail::expr::unary_expr<std::logical_not<void>, S>(s);
}

/** @brief Bitwise not (numpy-reference/reference/generated/numpy.invert.html).
 */
template <detail::expr::fixed_source S> constexpr auto operator~(const S &s)
{
    return detail::expr::unary_expr<std::bit_not<void>, S>(s);
}

// Elementwise math functions
// Reference: numpy-reference/reference/routines.math.html
// The constexpr kernels come from detail/math_constexpr.hpp so that
// static expressions fold at compile time.
namespace detail::fixed
{

struct abs_fn
{
    template <typename V> constexpr V operator()(V v) const
    {
        return detail::math::abs(v);
    }
};

struct sqrt_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return detail::math::sqrt(static_cast<R>(v));
    }
};

struct exp_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return static_cast<R>(detail::math::exp(static_cast<double>(v)));
    }
};

struct log_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return static_cast<R>(detail::math::log(static_cast<double>(v)));
    }
};

struct sin_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return static_cast<R>(detail::math::sin(static_cast<double>(v)));
    }
};

struct cos_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return static_cast<R>(detail::math::cos(static_cast<double>(v)));
    }
};

struct tan_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return static_cast<R>(detail::math::tan(static_cast<double>(v)));
    }
};

struct floor_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return static_cast<R>(detail::math::floor(static_cast<double>(v)));
    }
};

struct ceil_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return static_cast<R>(detail::math::ceil(static_cast<double>(v)));
    }
};

struct round_fn
{
    template <typename V> constexpr auto operator()(V v) const
    {
        using R = float_t<V>;
        return static_cast<R>(detail::math::round(static_cast<double>(v)));
    }
};

struct square_fn
{
    template <typename V> constexpr V operator()(V v) const
    {
        return v * v;
    }
};

struct power_fn
{
    template <typename V1, typename V2> constexpr auto operator()(V1 x, V2 y) const
    {
        using R = std::common_type_t<V1, V2>;
        if constexpr (std::is_integral_v<R>)
        {
            if (y < V2{0})
            {
                return R{0};
            }
            R acc = R{1};
            for (V2 e = V2{0}; e < y; ++e)
            {
                acc *= static_cast<R>(x);
            }
            return acc;
        }
        else
        {
            return static_cast<R>(detail::math::pow(static_cast<double>(x), static_cast<double>(y)));
        }
    }
};

} // namespace detail::fixed

#define NP_FIXED_UNARY_FN(name, fn_struct)                                                                             \
    template <detail::expr::fixed_source S> constexpr auto name(const S &s)                                            \
    {                                                                                                                  \
        return detail::expr::unary_expr<detail::fixed::fn_struct, S>(s);                                               \
    }

/** @brief Elementwise absolute value (numpy.absolute.html). */
NP_FIXED_UNARY_FN(abs, abs_fn)
/** @brief Elementwise sqrt (numpy.sqrt.html). */
NP_FIXED_UNARY_FN(sqrt, sqrt_fn)
/** @brief Elementwise exp (numpy.exp.html). */
NP_FIXED_UNARY_FN(exp, exp_fn)
/** @brief Elementwise natural log (numpy.log.html). */
NP_FIXED_UNARY_FN(log, log_fn)
/** @brief Elementwise sine (numpy.sin.html). */
NP_FIXED_UNARY_FN(sin, sin_fn)
/** @brief Elementwise cosine (numpy.cos.html). */
NP_FIXED_UNARY_FN(cos, cos_fn)
/** @brief Elementwise tangent (numpy.tan.html). */
NP_FIXED_UNARY_FN(tan, tan_fn)
/** @brief Elementwise floor (numpy.floor.html). */
NP_FIXED_UNARY_FN(floor, floor_fn)
/** @brief Elementwise ceil (numpy.ceil.html). */
NP_FIXED_UNARY_FN(ceil, ceil_fn)
/** @brief Elementwise round (numpy.round.html). */
NP_FIXED_UNARY_FN(round, round_fn)
/** @brief Elementwise square (numpy.square.html). */
NP_FIXED_UNARY_FN(square, square_fn)

#undef NP_FIXED_UNARY_FN

/**
 * @brief Elementwise power x1^x2 with NumPy broadcasting
 *        (numpy-reference/reference/generated/numpy.power.html).
 */
template <typename L, typename R>
    requires((detail::expr::fixed_source<L> || std::is_arithmetic_v<L>) &&
             (detail::expr::fixed_source<R> || std::is_arithmetic_v<R>) &&
             !(std::is_arithmetic_v<L> && std::is_arithmetic_v<R>) && detail::fixed::binop_ok<L, R>)
constexpr auto power(const L &l, const R &r)
{
    return detail::fixed::make_binary<detail::fixed::power_fn>(l, r);
}

// Joining (numpy-reference/reference/routines.array-manipulation.html)
namespace detail::fixed
{

/** @brief True when all arrays share rank >= 1 and the same tail. */
template <typename T, typename = void> struct is_ndarrayf : std::false_type
{
};
template <typename T> struct is_ndarrayf<T, std::void_t<typename T::value_type, decltype(T::rank)>> : std::true_type
{
};

template <typename A0, typename... Rest> constexpr bool concat_ok_v()
{
    if constexpr (!is_ndarrayf<A0>::value || !(is_ndarrayf<Rest>::value && ...))
    {
        return false;
    }
    else
    {
        using t0 = detail::expr::shape_tag_t<A0>;
        constexpr std::size_t r0 = A0::rank;
        return r0 >= 1 && ((Rest::rank == r0) && ...) &&
               ((detail::expr::same_tag<typename detail::expr::tail<t0>::type,
                                        typename detail::expr::tail<detail::expr::shape_tag_t<Rest>>::type>::value &&
                 ...));
    }
}

template <typename A0, typename... Rest> struct concat_ok
{
    static constexpr bool valid = concat_ok_v<A0, Rest...>();
};

/** @brief Cumulative axis-0 offsets across the concatenated arrays. */
template <typename... As> constexpr std::array<int, sizeof...(As) + 1> concat_offsets()
{
    std::array<int, sizeof...(As) + 1> offs{};
    int acc = 0;
    std::size_t k = 0;
    offs[0] = 0;
    ((offs[++k] = acc += detail::expr::head<detail::expr::shape_tag_t<As>>::value), ...);
    return offs;
}

/** @brief Result shape of concatenating along axis 0. */
template <typename Tag0, int Sum> struct concat_tag;

template <int Head, int... Tail, int Sum> struct concat_tag<detail::expr::shape_tag<Head, Tail...>, Sum>
{
    using type = detail::expr::shape_tag<Sum, Tail...>;
};

/** @brief Copy every element into its axis-0 slot of the result. */
template <typename R, int... E, typename... As>
constexpr ndarrayf<R, E...> concat_impl(detail::expr::shape_tag<E...>, const As &...as)
{
    constexpr std::size_t numel = (static_cast<std::size_t>(E) * ... * 1ull);
    constexpr std::array<int, sizeof...(As) + 1> offs = concat_offsets<As...>();
    ndarrayf<R, E...> out{};
    for (std::size_t i = 0; i < numel; ++i)
    {
        const auto c = detail::expr::unflatten(i, out.static_shape);
        const std::size_t ax = c[0];
        std::size_t sel = 0;
        while (sel + 1 < sizeof...(As) && ax >= static_cast<std::size_t>(offs[sel + 1]))
        {
            ++sel;
        }
        std::size_t k = 0;
        (
            [&](const auto &a) {
                if (k++ == sel)
                {
                    auto sc = c;
                    sc[0] -= static_cast<std::size_t>(offs[sel]);
                    out.m_data[detail::expr::flatten(out.static_shape, c)] =
                        static_cast<R>(a.m_data[detail::expr::flatten(a.static_shape, sc)]);
                }
            }(as),
            ...);
    }
    return out;
}

} // namespace detail::fixed

/**
 * @brief Join arrays along a new leading axis... (see below).
 */
template <typename A0, typename... Rest>
    requires detail::fixed::concat_ok<A0, Rest...>::valid
constexpr auto concatenate(const A0 &a0, const Rest &...rest)
{
    static_assert(detail::fixed::concat_ok<A0, Rest...>::valid,
                  "np: concatenate requires equal ranks and matching "
                  "non-axis-0 extents "
                  "(numpy-reference/reference/generated/numpy.concatenate.html)");
    using tag0 = detail::expr::shape_tag_t<A0>;
    constexpr int sum =
        detail::expr::head<tag0>::value + (detail::expr::head<detail::expr::shape_tag_t<Rest>>::value + ...);
    using rtag = typename detail::fixed::concat_tag<tag0, sum>::type;
    using R = std::common_type_t<typename A0::value_type, typename Rest::value_type...>;
    return detail::fixed::concat_impl<R>(rtag{}, a0, rest...);
}

namespace detail::fixed
{

/** @brief True when every array has exactly the same static shape. */
template <typename A0, typename... Rest> struct stack_ok
{
    using t0 = detail::expr::shape_tag_t<A0>;
    static constexpr bool valid = ((detail::expr::same_tag<t0, detail::expr::shape_tag_t<Rest>>::value) && ...);
};

/** @brief Copy every element into its new-axis slot of the result. */
template <typename R, int Axis, int... E, typename... As>
constexpr ndarrayf<R, E...> stack_impl(detail::expr::shape_tag<E...>, const As &...as)
{
    constexpr std::size_t numel = (static_cast<std::size_t>(E) * ... * 1ull);
    ndarrayf<R, E...> out{};
    for (std::size_t i = 0; i < numel; ++i)
    {
        const auto c = detail::expr::unflatten(i, out.static_shape);
        const std::size_t sel = c[Axis];
        std::size_t k = 0;
        (
            [&](const auto &a) {
                if (k++ == sel)
                {
                    std::array<std::size_t, sizeof...(E) - 1> sc{};
                    std::size_t m = 0;
                    for (std::size_t d = 0; d < sizeof...(E); ++d)
                    {
                        if (static_cast<int>(d) != Axis)
                        {
                            sc[m++] = c[d];
                        }
                    }
                    out.m_data[i] = static_cast<R>(a.m_data[detail::expr::flatten(a.static_shape, sc)]);
                }
            }(as),
            ...);
    }
    return out;
}

} // namespace detail::fixed

/**
 * @brief Join arrays along a new axis (numpy.stack.html). The new axis
 *        holds one slot per input array and its position `Axis` is a
 *        template parameter (NumPy default axis=0).
 */
template <int Axis = 0, typename A0, typename... Rest>
    requires detail::fixed::stack_ok<A0, Rest...>::valid
constexpr auto stack(const A0 &a0, const Rest &...rest)
{
    static_assert(detail::fixed::stack_ok<A0, Rest...>::valid,
                  "np: stack requires every input to have the same shape "
                  "(numpy-reference/reference/generated/numpy.stack.html)");
    constexpr std::size_t count = 1 + sizeof...(Rest);
    using tag0 = detail::expr::shape_tag_t<A0>;
    using rtag = typename detail::expr::insert<static_cast<int>(count), Axis, 0, tag0>::type;
    using R = std::common_type_t<typename A0::value_type, typename Rest::value_type...>;
    return detail::fixed::stack_impl<R, Axis>(rtag{}, a0, rest...);
}

} // namespace np

#endif // NP_NDARRAY_FIXED_HPP
