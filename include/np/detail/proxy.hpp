/**
 * @file detail/proxy.hpp
 * @brief Stack-based subscript proxies and index iteration helpers.
 *
 * Enables intuitive chained subscript syntax (`arr[i][j][k]`) using a
 * compile-time fixed-size index stack, avoiding heap allocations on the
 * hot indexing path.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DETAIL_PROXY_HPP
#define NP_DETAIL_PROXY_HPP

#include "../api_macros.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace np
{

template <typename T> class ndarray;

namespace detail
{

/**
 * @brief Stack-based index storage that avoids heap allocations.
 * @tparam MaxDims Maximum supported dimensionality (default 8).
 */
template <std::size_t MaxDims = 8> struct IndexStack
{
    std::array<std::size_t, MaxDims> m_data{};
    std::size_t m_count = 0;

    /**
     * @throws std::out_of_range if the stack already holds `MaxDims`
     *         indices (i.e. more than `MaxDims` chained `operator[]` calls).
     */
    constexpr void push_back(std::size_t v)
    {
        if (m_count >= MaxDims)
        {
            throw std::out_of_range("np::detail::IndexStack: chained operator[] depth exceeds "
                                    "MaxDims (increase the Proxy<T, MaxDims> template parameter "
                                    "for arrays with more dimensions)");
        }
        m_data[m_count++] = v;
    }

    NP_NODISCARD constexpr std::size_t size() const noexcept
    {
        return m_count;
    }

    NP_NODISCARD constexpr auto operator[](std::size_t i) const noexcept -> std::size_t
    {
        return m_data[i];
    }

    NP_NODISCARD constexpr const std::size_t *begin() const noexcept
    {
        return m_data.data();
    }

    NP_NODISCARD constexpr const std::size_t *end() const noexcept
    {
        return m_data.data() + m_count;
    }
};

/**
 * @brief Incrementing multi-dimensional counter (C order).
 *
 * Iterates every multi-index of the given shape exactly once.
 * An empty shape yields a single iteration (0-d array).
 */
class Odometer
{
  public:
    explicit Odometer(std::vector<int> dims) : dims_(dims.size())
    {
        for (std::size_t i = 0; i < dims.size(); ++i)
        {
            dims_[i] = static_cast<std::size_t>(dims[i]);
        }
        idx_.resize(dims_.size(), 0);
        done_ = !dims_.empty() && std::any_of(dims_.begin(), dims_.end(), [](std::size_t d) { return d == 0; });
    }

    NP_NODISCARD bool done() const noexcept
    {
        return done_;
    }

    NP_NODISCARD const std::vector<std::size_t> &idx() const noexcept
    {
        return idx_;
    }

    /** @brief Number of dimensions being iterated. */
    NP_NODISCARD std::size_t ndim() const noexcept
    {
        return dims_.size();
    }

    void advance() noexcept
    {
        if (dims_.empty())
        {
            done_ = true;
            return;
        }
        for (std::size_t d = dims_.size(); d-- > 0;)
        {
            if (++idx_[d] < dims_[d])
            {
                return;
            }
            idx_[d] = 0;
        }
        done_ = true;
    }

  private:
    std::vector<std::size_t> dims_;
    std::vector<std::size_t> idx_;
    bool done_ = false;
};

/**
 * @brief Flat offset of a multi-index given strides (in elements).
 */
NP_NODISCARD inline std::size_t flat_index(const std::vector<std::size_t> &idx, const std::vector<std::size_t> &strides,
                                           std::size_t offset = 0) noexcept
{
    std::size_t flat = offset;
    for (std::size_t i = 0; i < idx.size() && i < strides.size(); ++i)
    {
        flat += idx[i] * strides[i];
    }
    return flat;
}

} // namespace detail

/**
 * @brief Base class for multidimensional subscript proxies.
 *
 * @tparam T Element type of the array
 * @tparam IsConst Whether this proxy provides read-only access
 */
template <typename T, bool IsConst, std::size_t MaxDims = 8> class ProxyBase
{
    using Array = std::conditional_t<IsConst, const ndarray<T>, ndarray<T>>;
    using Stack = detail::IndexStack<MaxDims>;
    using Self = ProxyBase<T, IsConst, MaxDims>;

    Array &m_array;  ///< Underlying array (const or not)
    Stack m_indices; ///< Accumulated indices (stack-allocated)

  public:
    constexpr ProxyBase(Array &arr, Stack idx) noexcept : m_array(arr), m_indices(idx)
    {
    }

    // Assignment (write)

    template <bool C = IsConst>
    constexpr auto operator=(const T &v) noexcept -> Self &
        requires(!C)
    {
        m_array.set(m_indices, v);
        return *this;
    }

    template <bool C = IsConst>
    constexpr auto operator=(T &&v) noexcept -> Self &
        requires(!C)
    {
        m_array.set(m_indices, std::move(v));
        return *this;
    }

    constexpr auto operator=(const Self &other) -> Self &
    {
        if constexpr (*this != other)
        {
            T value = static_cast<T>(other);
            m_array.set(m_indices, value);
        }
        return *this;
    }

    // Reading

    NP_NODISCARD constexpr operator T() const noexcept
    {
        return m_array.get(m_indices);
    }

    // Fix for cpp-repl: ProxyBase with types that have convert_to (e.g. bigint)
    template <typename U>
        requires requires { std::declval<T>().template convert_to<U>(); }
    auto convert_to() const
    {
        return static_cast<T>(*this).template convert_to<U>();
    }

    // Support ap * a[n] where a[n] is ProxyBase
    friend auto operator*(const T &lhs, const Self &rhs) -> decltype(lhs * std::declval<T>())
    {
        return lhs * static_cast<T>(rhs);
    }
    friend auto operator*(const Self &lhs, const T &rhs) -> decltype(std::declval<T>() * rhs)
    {
        return static_cast<T>(lhs) * rhs;
    }

    NP_NODISCARD constexpr auto operator==(const T &v) const noexcept -> bool
    {
        return static_cast<T>(*this) == v;
    }

    NP_NODISCARD constexpr auto operator!=(const T &v) const noexcept -> bool
    {
        return static_cast<T>(*this) != v;
    }

    NP_NODISCARD constexpr auto operator==(const Self &other) const noexcept -> bool
    {
        return static_cast<T>(*this) == static_cast<T>(other);
    }

    NP_NODISCARD constexpr auto operator!=(const Self &other) const noexcept -> bool
    {
        return !(*this == other);
    }

    template <typename U> NP_NODISCARD constexpr auto operator==(const U &v) const noexcept -> bool
    {
        return static_cast<T>(*this) == static_cast<T>(v);
    }

    template <typename U> NP_NODISCARD constexpr auto operator!=(const U &v) const noexcept -> bool
    {
        return static_cast<T>(*this) != static_cast<T>(v);
    }

    NP_NODISCARD friend auto operator<<(std::ostream &os, const Self &proxy) -> std::ostream &
    {
        os << static_cast<T>(proxy);
        return os;
    }

    /**
     * @brief Descend one dimension, appending the new index.
     *
     * Negative indices count from the end of the current dimension
     * (NumPy semantics), matching `ndarray::operator[]`.
     * @throws std::out_of_range if this would exceed `MaxDims` chained
     *         subscripts (see `IndexStack::push_back`), or if the
     *         normalized index is out of bounds.
     */
    NP_NODISCARD constexpr auto operator[](std::ptrdiff_t idx) const -> Self
    {
        Stack next = m_indices; // trivial copy -- no heap touch
        const std::size_t depth = m_indices.size();
        if (depth < m_array.shape.size())
        {
            const std::ptrdiff_t dim = static_cast<std::ptrdiff_t>(m_array.shape[depth]);
            if (idx < 0)
            {
                idx += dim;
            }
            if (idx < 0 || idx >= dim)
            {
                throw std::out_of_range("proxy subscript out of bounds");
            }
        }
        next.push_back(static_cast<std::size_t>(idx));
        return Self(m_array, next);
    }
};

/** @brief Read-write proxy. */
template <typename T, std::size_t MaxDims = 8> using Proxy = ProxyBase<T, false, MaxDims>;

/** @brief Read-only proxy. */
template <typename T, std::size_t MaxDims = 8> using ConstProxy = ProxyBase<T, true, MaxDims>;

} // namespace np

#endif // NP_DETAIL_PROXY_HPP
