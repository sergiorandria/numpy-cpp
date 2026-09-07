/**
 * @file half.hpp
 * @brief FP16 / BF16 half-precision for powerful GPU tensor cores.
 *
 * Provides np::half (float16) and np::bfloat16 wrappers with conversion to/from float.
 * Uses _Float16 on GCC/Clang (AVX512-FP16, ARMv8.2) or std::float16_t if C++23,
 * otherwise emulates via soft-float with correct 16-bit storage and rounding.
 * Header-only, for Hopper/Blackwell FP16 tensor cores.
 */
#ifndef NP_HALF_HPP
#define NP_HALF_HPP

#include "api_macros.hpp"
#include "ndarray.hpp"
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace np
{

#if defined(__FLT16_MAX__)
using half = _Float16;
#define NP_HAS_FLOAT16 1
#elif __has_include(<stdfloat>)
#include <stdfloat>
#if defined(__STDCPP_FLOAT16_T__)
using half = std::float16_t;
#define NP_HAS_FLOAT16 1
#endif
#endif

#ifndef NP_HAS_FLOAT16
// Fallback: distinct 16-bit soft-float (keeps sizeof 2 and correct is_half_v semantics)
struct half
{
    uint16_t bits = 0;
    constexpr half() noexcept = default;
    constexpr explicit half(float f) noexcept : bits(float_to_half_bits(f))
    {
    }

    constexpr operator float() const noexcept
    {
        return half_bits_to_float(bits);
    }

    // arithmetic via float round-trip (explicit construction keeps precision)
    friend constexpr half operator+(half a, half b) noexcept
    {
        return half(float(a) + float(b));
    }
    friend constexpr half operator-(half a, half b) noexcept
    {
        return half(float(a) - float(b));
    }
    friend constexpr half operator*(half a, half b) noexcept
    {
        return half(float(a) * float(b));
    }
    friend constexpr half operator/(half a, half b) noexcept
    {
        return half(float(a) / float(b));
    }
    constexpr half &operator+=(half o) noexcept
    {
        return *this = *this + o;
    }
    constexpr half &operator-=(half o) noexcept
    {
        return *this = *this - o;
    }
    constexpr half &operator*=(half o) noexcept
    {
        return *this = *this * o;
    }
    constexpr half &operator/=(half o) noexcept
    {
        return *this = *this / o;
    }
    friend constexpr bool operator==(half a, half b) noexcept
    {
        return a.bits == b.bits;
    }
    friend constexpr bool operator!=(half a, half b) noexcept
    {
        return a.bits != b.bits;
    }
    friend constexpr bool operator<(half a, half b) noexcept
    {
        return float(a) < float(b);
    }
    friend constexpr bool operator<=(half a, half b) noexcept
    {
        return float(a) <= float(b);
    }
    friend constexpr bool operator>(half a, half b) noexcept
    {
        return float(a) > float(b);
    }
    friend constexpr bool operator>=(half a, half b) noexcept
    {
        return float(a) >= float(b);
    }

  private:
    static constexpr uint16_t float_to_half_bits(float f) noexcept
    {
        uint32_t u = std::bit_cast<uint32_t>(f);
        uint32_t sign = (u >> 16) & 0x8000u;
        uint32_t exp = (u >> 23) & 0xFFu;
        uint32_t mant = u & 0x7FFFFFu;
        if (exp == 255u)
        {
            // Inf/NaN
            if (mant == 0)
                return static_cast<uint16_t>(sign | 0x7C00u);
            // NaN: keep payload, ensure quiet
            uint16_t h = static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13));
            if ((h & 0x03FFu) == 0)
                h |= 1u;
            return h;
        }
        if (exp > 142u)
        {
            // Overflow to Inf (142 = 127 -15 +30)
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        if (exp < 113u)
        {
            // Underflow to zero/subnormal
            if (exp < 103u)
                return static_cast<uint16_t>(sign);
            mant |= 0x800000u;
            uint32_t shift = 113u - exp;
            // round to nearest even
            uint32_t rounding = 0xFFFu + ((mant >> shift) & 1u);
            mant = (mant + rounding) >> shift;
            return static_cast<uint16_t>(sign | (mant >> 13));
        }
        // Normalized
        exp = exp - 127u + 15u;
        uint32_t rounding = 0xFFFu + ((mant >> 13) & 1u);
        mant += rounding;
        if (mant & 0x800000u)
        {
            mant = 0;
            ++exp;
        }
        if (exp >= 31u)
            return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
    }
    static constexpr float half_bits_to_float(uint16_t h) noexcept
    {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
        uint32_t mant = static_cast<uint32_t>(h) & 0x03FFu;
        uint32_t f = 0;
        if (exp == 0)
        {
            if (mant == 0)
            {
                f = sign;
            }
            else
            {
                // subnormal -> normalize
                exp = 1;
                while ((mant & 0x0400u) == 0)
                {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x03FFu;
                uint32_t exp32 = (127u - 15u + exp);
                f = sign | (exp32 << 23) | (mant << 13);
            }
        }
        else if (exp == 31u)
        {
            f = sign | 0x7F800000u | (mant << 13);
        }
        else
        {
            uint32_t exp32 = exp + (127u - 15u);
            f = sign | (exp32 << 23) | (mant << 13);
        }
        return std::bit_cast<float>(f);
    }
};
#define NP_HAS_FLOAT16 1
#endif
// Note: np::float16 tag is defined in dtype.hpp; use np::half for the actual FP16 type

struct bfloat16
{
    uint16_t bits = 0;
    constexpr bfloat16() noexcept = default;
    constexpr explicit bfloat16(float f) noexcept
    {
        uint32_t u = std::bit_cast<uint32_t>(f);
        if ((u & 0x7FFFFFFFu) > 0x7F800000u)
        {
            // NaN: force quiet
            bits = static_cast<uint16_t>((u >> 16) | 0x0040u);
            return;
        }
        uint32_t rounding_bias = 0x7FFFu + ((u >> 16) & 1u); // round-to-nearest-even
        bits = static_cast<uint16_t>((u + rounding_bias) >> 16);
    }
    constexpr operator float() const noexcept
    {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        return std::bit_cast<float>(u);
    }
    friend constexpr bfloat16 operator+(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(float(a) + float(b));
    }
    friend constexpr bfloat16 operator-(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(float(a) - float(b));
    }
    friend constexpr bfloat16 operator*(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(float(a) * float(b));
    }
    friend constexpr bfloat16 operator/(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(float(a) / float(b));
    }
    constexpr bfloat16 &operator+=(bfloat16 o) noexcept
    {
        return *this = *this + o;
    }
    constexpr bfloat16 &operator-=(bfloat16 o) noexcept
    {
        return *this = *this - o;
    }
    constexpr bfloat16 &operator*=(bfloat16 o) noexcept
    {
        return *this = *this * o;
    }
    constexpr bfloat16 &operator/=(bfloat16 o) noexcept
    {
        return *this = *this / o;
    }
    friend constexpr bool operator==(bfloat16 a, bfloat16 b) noexcept
    {
        return a.bits == b.bits;
    }
    friend constexpr bool operator!=(bfloat16 a, bfloat16 b) noexcept
    {
        return a.bits != b.bits;
    }
    friend constexpr bool operator<(bfloat16 a, bfloat16 b) noexcept
    {
        return float(a) < float(b);
    }
    friend constexpr bool operator<=(bfloat16 a, bfloat16 b) noexcept
    {
        return float(a) <= float(b);
    }
    friend constexpr bool operator>(bfloat16 a, bfloat16 b) noexcept
    {
        return float(a) > float(b);
    }
    friend constexpr bool operator>=(bfloat16 a, bfloat16 b) noexcept
    {
        return float(a) >= float(b);
    }
};

// Traits — C++23 consteval + inline constexpr
template <typename T> struct is_half : std::false_type
{
};
template <> struct is_half<half> : std::true_type
{
};
template <> struct is_half<bfloat16> : std::true_type
{
};
template <typename T> inline constexpr bool is_half_v = is_half<T>::value;

#if __cplusplus >= 202302L
consteval bool has_half_consteval() noexcept
{
    return is_half_v<half>;
}
static_assert(has_half_consteval(), "half trait broken");
#endif

// Scalar half/bfloat16 conversion
NP_NODISCARD inline ndarray<half> quantize_half(const ndarray<float> &a)
{
    ndarray<half> out(a.shape);
    auto &od = out.data();
    auto &ad = a.data();
    for (size_t i = 0; i < a.size(); ++i)
        od[i] = half(ad[i]);
    return out;
}
NP_NODISCARD inline ndarray<float> dequantize_half(const ndarray<half> &a)
{
    ndarray<float> out(a.shape);
    auto &od = out.data();
    auto &ad = a.data();
    for (size_t i = 0; i < a.size(); ++i)
        od[i] = float(ad[i]);
    return out;
}

NP_NODISCARD inline ndarray<bfloat16> quantize_bfloat16(const ndarray<float> &a)
{
    ndarray<bfloat16> out(a.shape);
    auto &od = out.data();
    auto &ad = a.data();
    for (size_t i = 0; i < a.size(); ++i)
        od[i] = bfloat16(ad[i]);
    return out;
}
NP_NODISCARD inline ndarray<float> dequantize_bfloat16(const ndarray<bfloat16> &a)
{
    ndarray<float> out(a.shape);
    auto &od = out.data();
    auto &ad = a.data();
    for (size_t i = 0; i < a.size(); ++i)
        od[i] = float(ad[i]);
    return out;
}

} // namespace np

// numeric_limits specializations omitted (see CLAUDE item 7)
#endif // NP_HALF_HPP
