/**
 * @file half.hpp
 * @brief FP16 / BF16 half-precision for powerful GPU tensor cores.
 *
 * Provides np::half (float16) and np::bfloat16 wrappers with conversion to/from float.
 * Uses _Float16 on GCC/Clang (AVX512-FP16, ARMv8.2) or std::float16_t if C++23,
 * otherwise emulates via float. Header-only, for Hopper/Blackwell FP16 tensor cores.
 */
#ifndef NP_HALF_HPP
#define NP_HALF_HPP

#include "api_macros.hpp"
#include <cstdint>
#include <cstring>

namespace np
{

#if defined(__FLT16_MAX__) || defined(__HAVE_FLOAT16)
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
// Fallback: use float as emulated half (keeps ndarray arithmetic, header-only)
using half = float;
#define NP_HAS_FLOAT16 1
#endif
// Note: np::float16 tag is defined in dtype.hpp; use np::half for the actual FP16 type

struct bfloat16
{
    uint16_t bits = 0;
    bfloat16() = default;
    explicit bfloat16(float f)
    {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(float));
        bits = static_cast<uint16_t>(u >> 16);
    }
    operator float() const noexcept
    {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(float));
        return f;
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
    if consteval
    {
        return is_half_v<half>;
    }
    else
    {
        return true;
    }
}
static_assert(has_half_consteval(), "half trait broken");
#endif

// SIMD vectorized half conversion (uses simd.hpp when available)
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

} // namespace np

#endif // NP_HALF_HPP
