/**
 * @file bitwise.hpp
 * @brief Bit-wise operations (np.bitwise_and/or/xor, invert, shifts, packbits...).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.bitwise.html
 *
 * Elementwise ops broadcast and honour NumPy integer promotion.
 * packbits / unpackbits handle axis and bitorder; binary_repr returns
 * a std::string.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_BITWISE_HPP
#define NP_BITWISE_HPP

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"
#include "pqc.hpp"

namespace np
{

namespace detail
{
template <typename T> struct _is_int_like : std::bool_constant<std::is_integral_v<T> && !std::is_same_v<T, bool>>
{
};

template <typename T> inline constexpr bool _is_int_like_v = _is_int_like<T>::value;

// Broadcast binary elementwise for integral types
template <typename T, typename U, typename Fn>
auto _bitwise_broadcast(const ndarray<T> &a, const ndarray<U> &b, Fn &&fn) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    static_assert(_is_int_like_v<R> || std::is_integral_v<R>, "bitwise: integral required");
    std::vector<int> out_shape = broadcast_shapes(a.shape, b.shape);
    ndarray<R> out(out_shape);
    Odometer od(out_shape);
    while (!od.done())
    {
        const auto &idx = od.idx();
        R av = static_cast<R>(a.get(broadcast_index(a.shape, out_shape, idx)));
        R bv = static_cast<R>(b.get(broadcast_index(b.shape, out_shape, idx)));
        out.set(idx, fn(av, bv));
        od.advance();
    }
    return out;
}

} // namespace detail

// ── Elementwise bit operations ────────────────────────────────────

/**
 * @brief Bitwise AND element-wise (np.bitwise_and).
 * Reference: numpy-reference/reference/generated/numpy.bitwise_and.html
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto bitwise_and(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    static_assert(std::is_integral_v<R> && !std::is_same_v<R, bool>, "bitwise_and: integral types required");
    return detail::_bitwise_broadcast(x1, x2, [](R a, R b) -> R { return a & b; });
}

NP_API template <typename T, typename U>
NP_NODISCARD auto bitwise_and(const ndarray<T> &x1, U scalar) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    static_assert(std::is_integral_v<R>, "bitwise_and: integral required");
    ndarray<U> s(std::vector<int>{1});
    s.data()[0] = scalar;
    return bitwise_and(x1, s);
}

/**
 * @brief Bitwise OR element-wise (np.bitwise_or).
 * Reference: numpy-reference/reference/generated/numpy.bitwise_or.html
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto bitwise_or(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    static_assert(std::is_integral_v<R>, "bitwise_or: integral required");
    return detail::_bitwise_broadcast(x1, x2, [](R a, R b) -> R { return a | b; });
}

NP_API template <typename T, typename U>
NP_NODISCARD auto bitwise_or(const ndarray<T> &x1, U scalar) -> ndarray<std::common_type_t<T, U>>
{
    ndarray<U> s(std::vector<int>{1});
    s.data()[0] = scalar;
    return bitwise_or(x1, s);
}

/**
 * @brief Bitwise XOR element-wise (np.bitwise_xor).
 * Reference: numpy-reference/reference/generated/numpy.bitwise_xor.html
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto bitwise_xor(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    static_assert(std::is_integral_v<R>, "bitwise_xor: integral required");
    return detail::_bitwise_broadcast(x1, x2, [](R a, R b) -> R { return a ^ b; });
}

NP_API template <typename T, typename U>
NP_NODISCARD auto bitwise_xor(const ndarray<T> &x1, U scalar) -> ndarray<std::common_type_t<T, U>>
{
    ndarray<U> s(std::vector<int>{1});
    s.data()[0] = scalar;
    return bitwise_xor(x1, s);
}

/**
 * @brief Bitwise invert / NOT element-wise (np.invert, np.bitwise_invert).
 * Reference: numpy-reference/reference/generated/numpy.invert.html
 */
NP_API template <typename T> NP_NODISCARD auto invert(const ndarray<T> &x) -> ndarray<T>
{
    static_assert(std::is_integral_v<T>, "invert: integral required");
    ndarray<T> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        out.data()[i] = ~x.data()[x._flat_logical(i)];
    }
    return out;
}

NP_API template <typename T> NP_NODISCARD inline auto bitwise_invert(const ndarray<T> &x) -> ndarray<T>
{
    return invert(x);
}

/**
 * @brief Alias `np.bitwise_not` (NumPy 2.0 synonym for `invert`).
 * Reference: numpy-reference/reference/generated/numpy.bitwise_not.html
 */
NP_API template <typename T> NP_NODISCARD inline auto bitwise_not(const ndarray<T> &x) -> ndarray<T>
{
    return invert(x);
}

/**
 * @brief Count number of set bits per element (np.bitwise_count).
 *
 * Returns an array of `uint8`/`int` counts. For signed types the
 * two's-complement representation is counted (as in NumPy, which
 * counts bits of the underlying storage).
 *
 * Reference: numpy-reference/reference/generated/numpy.bitwise_count.html
 */
NP_API template <typename T> NP_NODISCARD auto bitwise_count(const ndarray<T> &x) -> ndarray<std::uint8_t>
{
    static_assert(std::is_integral_v<T>, "bitwise_count: integral required");
    ndarray<std::uint8_t> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        using U = std::make_unsigned_t<T>;
        U v = static_cast<U>(x.data()[x._flat_logical(i)]);
        int cnt = 0;
        if constexpr (sizeof(U) <= sizeof(unsigned int))
        {
            cnt = __builtin_popcount(static_cast<unsigned int>(v));
        }
        else if constexpr (sizeof(U) <= sizeof(unsigned long))
        {
            cnt = __builtin_popcountl(static_cast<unsigned long>(v));
        }
        else
        {
            cnt = __builtin_popcountll(static_cast<unsigned long long>(v));
        }
        out.data()[i] = static_cast<std::uint8_t>(cnt);
    }
    return out;
}

// NumPy also exposes `bit_count` as alias (Python 3.8 int.bit_count)
NP_API template <typename T> NP_NODISCARD inline auto bit_count(const ndarray<T> &x) -> ndarray<std::uint8_t>
{
    return bitwise_count(x);
}

/**
 * @brief Left shift bits (np.left_shift / np.bitwise_left_shift).
 * Reference: numpy-reference/reference/generated/numpy.left_shift.html
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto left_shift(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    static_assert(std::is_integral_v<R>, "left_shift: integral required");
    return detail::_bitwise_broadcast(x1, x2, [](R a, R b) -> R { return a << b; });
}

NP_API template <typename T, typename U>
NP_NODISCARD auto left_shift(const ndarray<T> &x1, U shift) -> ndarray<std::common_type_t<T, U>>
{
    ndarray<U> s(std::vector<int>{1});
    s.data()[0] = shift;
    return left_shift(x1, s);
}

NP_API template <typename T, typename U>
NP_NODISCARD inline auto bitwise_left_shift(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>>
{
    return left_shift(x1, x2);
}

NP_API template <typename T, typename U>
NP_NODISCARD inline auto bitwise_left_shift(const ndarray<T> &x1, U shift) -> ndarray<std::common_type_t<T, U>>
{
    return left_shift(x1, shift);
}

/**
 * @brief Right shift bits (np.right_shift / np.bitwise_right_shift).
 * Reference: numpy-reference/reference/generated/numpy.right_shift.html
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto right_shift(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<std::common_type_t<T, U>>
{
    using R = std::common_type_t<T, U>;
    static_assert(std::is_integral_v<R>, "right_shift: integral required");
    return detail::_bitwise_broadcast(x1, x2, [](R a, R b) -> R { return a >> b; });
}

NP_API template <typename T, typename U>
NP_NODISCARD auto right_shift(const ndarray<T> &x1, U shift) -> ndarray<std::common_type_t<T, U>>
{
    ndarray<U> s(std::vector<int>{1});
    s.data()[0] = shift;
    return right_shift(x1, s);
}

NP_API template <typename T, typename U>
NP_NODISCARD inline auto bitwise_right_shift(const ndarray<T> &x1, const ndarray<U> &x2)
    -> ndarray<std::common_type_t<T, U>>
{
    return right_shift(x1, x2);
}

NP_API template <typename T, typename U>
NP_NODISCARD inline auto bitwise_right_shift(const ndarray<T> &x1, U shift) -> ndarray<std::common_type_t<T, U>>
{
    return right_shift(x1, shift);
}

// ── Bit packing ───────────────────────────────────────────────────

/**
 * @brief Packs binary array into uint8 bits (np.packbits).
 *
 * Reference: numpy-reference/reference/generated/numpy.packbits.html
 *
 * @param a Binary-valued array (0/1).
 * @param axis Axis along which to pack (nullopt => flatten).
 * @param bitorder "big" (MSB first, default) or "little" (LSB first).
 * @return uint8 array with packed bits.
 */
NP_API template <typename T>
NP_NODISCARD auto packbits(const ndarray<T> &a, std::optional<int> axis = std::nullopt,
                           const std::string &bitorder = "big") -> ndarray<std::uint8_t>
{
    if (bitorder != "big" && bitorder != "little")
    {
        throw std::invalid_argument("packbits: bitorder must be 'big' or 'little'");
    }
    if (!axis.has_value())
    {
        auto flat = a.ravel();
        std::size_t n = flat.size();
        std::size_t out_n = (n + 7) / 8;
        ndarray<std::uint8_t> out(std::vector<int>{static_cast<int>(out_n)});
        for (std::size_t i = 0; i < out_n; ++i)
        {
            std::uint8_t v = 0;
            for (int b = 0; b < 8; ++b)
            {
                std::size_t idx = i * 8 + static_cast<std::size_t>(b);
                std::uint8_t bit = 0;
                if (idx < n)
                {
                    bit = flat.data()[flat._flat_logical(idx)] ? 1 : 0;
                }
                if (bitorder == "big")
                {
                    v |= static_cast<std::uint8_t>(bit << (7 - b));
                }
                else
                {
                    v |= static_cast<std::uint8_t>(bit << b);
                }
            }
            out.data()[i] = v;
        }
        return out;
    }
    int ax = *axis;
    if (ax < 0)
    {
        ax += static_cast<int>(a.ndim());
    }
    if (ax < 0 || ax >= static_cast<int>(a.ndim()))
    {
        throw AxisError("packbits: axis out of bounds");
    }
    int n = a.shape[static_cast<std::size_t>(ax)];
    int out_n = (n + 7) / 8;
    std::vector<int> out_shape = a.shape;
    out_shape[static_cast<std::size_t>(ax)] = out_n;
    ndarray<std::uint8_t> out(out_shape);
    // Iterate over all positions in output, expand to 8 source bits
    detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &oidx = od.idx();
        int out_pos = static_cast<int>(oidx[static_cast<std::size_t>(ax)]);
        std::uint8_t v = 0;
        for (int b = 0; b < 8; ++b)
        {
            int src_pos = out_pos * 8 + b;
            std::uint8_t bit = 0;
            if (src_pos < n)
            {
                std::vector<std::size_t> sidx(a.ndim());
                for (std::size_t d = 0; d < a.ndim(); ++d)
                {
                    if (static_cast<int>(d) == ax)
                    {
                        sidx[d] = static_cast<std::size_t>(src_pos);
                    }
                    else
                    {
                        sidx[d] = oidx[d];
                    }
                }
                bit = a.get(sidx) ? 1 : 0;
            }
            if (bitorder == "big")
            {
                v |= static_cast<std::uint8_t>(bit << (7 - b));
            }
            else
            {
                v |= static_cast<std::uint8_t>(bit << b);
            }
        }
        out.set(oidx, v);
        od.advance();
    }
    return out;
}

/**
 * @brief Unpacks uint8 array into binary array (np.unpackbits).
 *
 * Reference: numpy-reference/reference/generated/numpy.unpackbits.html
 *
 * @param a uint8 array to unpack.
 * @param axis Axis along which to unpack (nullopt => flatten).
 * @param count Number of elements to unpack (negative => all).
 * @param bitorder "big" or "little".
 * @return uint8 array of 0/1.
 */
NP_API inline auto unpackbits(const ndarray<std::uint8_t> &a, std::optional<int> axis = std::nullopt, int count = -1,
                              const std::string &bitorder = "big") -> ndarray<std::uint8_t>
{
    if (bitorder != "big" && bitorder != "little")
    {
        throw std::invalid_argument("unpackbits: bitorder must be 'big' or 'little'");
    }
    if (!axis.has_value())
    {
        auto flat = a.ravel();
        std::size_t n = flat.size() * 8;
        if (count >= 0)
        {
            n = std::min<std::size_t>(n, static_cast<std::size_t>(count));
        }
        ndarray<std::uint8_t> out(std::vector<int>{static_cast<int>(n)});
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t byte_idx = i / 8;
            int bit_idx = static_cast<int>(i % 8);
            std::uint8_t byte = flat.data()[flat._flat_logical(byte_idx)];
            std::uint8_t bit;
            if (bitorder == "big")
            {
                bit = (byte >> (7 - bit_idx)) & 1;
            }
            else
            {
                bit = (byte >> bit_idx) & 1;
            }
            out.data()[i] = bit;
        }
        return out;
    }
    int ax = *axis;
    if (ax < 0)
    {
        ax += static_cast<int>(a.ndim());
    }
    if (ax < 0 || ax >= static_cast<int>(a.ndim()))
    {
        throw AxisError("unpackbits: axis out of bounds");
    }
    int n = a.shape[static_cast<std::size_t>(ax)];
    int out_n = n * 8;
    if (count >= 0)
    {
        out_n = std::min(out_n, count);
    }
    std::vector<int> out_shape = a.shape;
    out_shape[static_cast<std::size_t>(ax)] = out_n;
    ndarray<std::uint8_t> out(out_shape);
    detail::Odometer od(out_shape);
    while (!od.done())
    {
        const auto &oidx = od.idx();
        int out_pos = static_cast<int>(oidx[static_cast<std::size_t>(ax)]);
        int byte_idx = out_pos / 8;
        int bit_idx = out_pos % 8;
        std::vector<std::size_t> sidx(a.ndim());
        for (std::size_t d = 0; d < a.ndim(); ++d)
        {
            if (static_cast<int>(d) == ax)
            {
                sidx[d] = static_cast<std::size_t>(byte_idx);
            }
            else
            {
                sidx[d] = oidx[d];
            }
        }
        std::uint8_t byte = 0;
        if (byte_idx < n)
        {
            byte = a.get(sidx);
        }
        std::uint8_t bit;
        if (bitorder == "big")
        {
            bit = (byte >> (7 - bit_idx)) & 1;
        }
        else
        {
            bit = (byte >> bit_idx) & 1;
        }
        out.set(oidx, bit);
        od.advance();
    }
    return out;
}

// ── Output formatting ─────────────────────────────────────────────

/**
 * @brief Binary representation of integer as string (np.binary_repr).
 * Reference: numpy-reference/reference/generated/numpy.binary_repr.html
 */
/**
 * @brief Constant-time select for bitwise mux (PQC).
 *
 * Uses `pqc::ct_select` to choose between two integral values without
 * branching. Suitable for McEliece conditional swaps on secret indices.
 * Reference: pqc.hpp:ct_select
 */
NP_API template <typename T> NP_NODISCARD inline T ct_select(int cond, T a, T b) noexcept
{
    return pqc::ct_select(cond, a, b);
}

/**
 * @brief Constant-time bitwise XOR for secret arrays (PQC-hardened).
 *
 * Wraps `bitwise_xor` with `ct_barrier` fences to prevent speculative
 * reordering. Use for key material (e.g., ML-KEM ciphertext xor).
 * Reference: pqc.hpp:ct_barrier
 */
NP_API template <typename T, typename U>
NP_NODISCARD inline auto bitwise_xor_ct(const ndarray<T> &x1, const ndarray<U> &x2) -> ndarray<std::common_type_t<T, U>>
{
    pqc::ct_barrier();
    auto out = bitwise_xor(x1, x2);
    pqc::ct_barrier();
    return out;
}

NP_API inline auto binary_repr(long long num, std::optional<int> width = std::nullopt) -> std::string
{
    if (!width.has_value())
    {
        if (num == 0)
        {
            return "0";
        }
        bool neg = num < 0;
        unsigned long long v = neg ? static_cast<unsigned long long>(-num) : static_cast<unsigned long long>(num);
        std::string s;
        while (v)
        {
            s.push_back((v & 1) ? '1' : '0');
            v >>= 1;
        }
        std::reverse(s.begin(), s.end());
        if (neg)
        {
            s = "-" + s;
        }
        return s;
    }
    int w = *width;
    if (w <= 0)
    {
        throw std::invalid_argument("binary_repr: width must be positive");
    }
    // Two's complement for negative with width
    unsigned long long mask = 0;
    if (w < 64)
    {
        mask = (1ULL << w) - 1;
    }
    else
    {
        mask = ~0ULL;
    }
    unsigned long long v = static_cast<unsigned long long>(num) & mask;
    std::string s;
    s.reserve(w);
    for (int i = w - 1; i >= 0; --i)
    {
        s.push_back(((v >> i) & 1) ? '1' : '0');
    }
    return s;
}

} // namespace np

#endif // NP_BITWISE_HPP
