/**
 * @file vecmath.hpp
 * @brief Portable SIMD vector math library (SLEEF/SVML-class ufunc kernels).
 *
 * Self-contained, header-only, dependency-free vector implementations of the
 * transcendental and rounding kernels used by np::simd / np math ufuncs:
 *   exp, expm1, exp2, log, log10, log2, log1p,
 *   sin, cos, sincos, tan, asin, acos, atan, atan2,
 *   sinh, cosh, tanh, asinh, acosh, atanh,
 *   sqrt, cbrt, pow, hypot, floor, ceil, trunc, rint, fabs
 * for `float` and `double`.
 *
 * Design (single-source portable SIMD, as in SLEEF):
 *   - Every kernel is written once against a tiny vector abstraction
 *     (`detail::Traits<T>`). The abstraction maps to the widest ISA tier
 *     available at compile time: AVX-512 > AVX2 > SSE2 > NEON > scalar.
 *     Plain AVX without AVX2 deliberately uses the 128-bit tier because AVX
 *     lacks 256-bit integer arithmetic needed for exact range reduction.
 *   - Range reduction is Cody-Waite style with exact bit tricks (fdlibm
 *     constants), so the hot path needs no libm calls. Rare lanes (huge trig
 *     arguments, subnormal log inputs, negative-base pow, ...) fall back to
 *     scalar `std::` per lane, keeping every edge case bit-compatible with
 *     the scalar path.
 *   - If SLEEF is enabled (NP_HAS_SLEEF), np::simd keeps preferring SLEEF's
 *     u10 kernels; this library is the dependency-free default underneath.
 *
 * Accuracy targets (worst case vs. correctly rounded over the fast path,
 * measured by the tests/test_vecmath.cpp sweep; without FMA add ~1 ULP):
 *   exp/exp2/expm1/log/log1p/log10/log2: <= 2 ULP (3 ULP for expm1)
 *   sin/cos/tan (|x| <= 1e6 double, 5e3 float): <= 4 ULP; larger |x| uses
 *     libm lane fallback (exact)
 *   asin/acos/atan/atan2: <= 6 ULP (typically <= 2)
 *   sinh/cosh/tanh/asinh/acosh/atanh: <= 4 ULP
 *   sqrt/floor/ceil/trunc/rint/fabs: correctly rounded (native instructions)
 *   cbrt/hypot: <= 3 ULP
 *   pow: exact for b in {0, 1, 2, -1, 1/2, -1/2}; <= 4 ULP for
 *     |b*log2(a)| <= 16, growing ~|b*log2(a)| * 2^-53 beyond that (the
 *     intermediate product is exact; the residual is the log kernel's
 *     absolute error amplified by |b|, inherent to exp-based pow)
 *
 * Special values follow IEEE 754 / NumPy semantics: NaN propagates, signed
 * zeros and infinities match scalar `std::` behavior (verified in tests).
 *
 * Thread safety: no mutable globals; every function is re-entrant.
 *
 * Reference: numpy-reference/reference/routines.math.html,
 *   S. L. Moshier (Cephes), Sun fdlibm, SLEEF design notes.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_VECMATH_HPP
#define NP_VECMATH_HPP
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>

#include "api_macros.hpp"

// Cumulative ISA detection. An elif chain would define only the top macro and
// silently disable SSE2-gated kernels on SSE4/AVX builds, so every level at or
// below the host capability is defined here.
#if defined(__AVX512F__)
#define NP_VECMATH_HAS_AVX512 1
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
#define NP_VECMATH_HAS_AVX2 1
#endif
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#define NP_VECMATH_HAS_AVX 1
#endif
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_2__)
#define NP_VECMATH_HAS_SSE42 1
#endif
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_2__) || defined(__SSE4_1__)
#define NP_VECMATH_HAS_SSE41 1
#endif
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_2__) || defined(__SSE4_1__) ||     \
    defined(__SSSE3__)
#define NP_VECMATH_HAS_SSSE3 1
#endif
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_2__) || defined(__SSE4_1__) ||     \
    defined(__SSSE3__) || defined(__SSE3__)
#define NP_VECMATH_HAS_SSE3 1
#endif
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_2__) || defined(__SSE4_1__) ||     \
    defined(__SSSE3__) || defined(__SSE3__) || defined(__SSE2__) || defined(_M_X64) ||                                 \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define NP_VECMATH_HAS_SSE2 1
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NP_VECMATH_HAS_NEON 1
#endif
#if defined(__aarch64__) || defined(__arm64__)
#define NP_VECMATH_HAS_NEON64 1
#endif

#if defined(NP_VECMATH_HAS_AVX512) || defined(NP_VECMATH_HAS_AVX2) || defined(NP_VECMATH_HAS_AVX) ||                   \
    defined(NP_VECMATH_HAS_SSE2)
#include <immintrin.h>
#endif
#if defined(NP_VECMATH_HAS_NEON)
#include <arm_neon.h>
#endif

namespace np
{
namespace vecmath
{

/** @brief Compile-time detection of the vector-math backend. */
struct Features
{
    static constexpr bool has_sse2 =
#ifdef NP_VECMATH_HAS_SSE2
        true;
#else
        false;
#endif
    static constexpr bool has_avx =
#ifdef NP_VECMATH_HAS_AVX
        true;
#else
        false;
#endif
    static constexpr bool has_avx2 =
#ifdef NP_VECMATH_HAS_AVX2
        true;
#else
        false;
#endif
    static constexpr bool has_avx512 =
#ifdef NP_VECMATH_HAS_AVX512
        true;
#else
        false;
#endif
    static constexpr bool has_neon =
#ifdef NP_VECMATH_HAS_NEON
        true;
#else
        false;
#endif
    static constexpr bool has_fma =
#if defined(__FMA__) || defined(__AVX512F__) || (defined(__aarch64__) && defined(NP_VECMATH_HAS_NEON))
        true;
#else
        false;
#endif
    static constexpr std::size_t width_f32 =
#if defined(NP_VECMATH_HAS_AVX512)
        16;
#elif defined(NP_VECMATH_HAS_AVX2)
        8;
#elif defined(NP_VECMATH_HAS_SSE2) || defined(NP_VECMATH_HAS_NEON)
                                4;
#else
                                1;
#endif
    static constexpr std::size_t width_f64 =
#if defined(NP_VECMATH_HAS_AVX512)
        8;
#elif defined(NP_VECMATH_HAS_AVX2)
        4;
#elif defined(NP_VECMATH_HAS_SSE2)
        2;
#elif defined(NP_VECMATH_HAS_NEON64)
        2;
#else
                                    1;
#endif
};

namespace detail
{

// Mathematical constants per precision. The reduction splits carry trailing
// zero bits so that k*HI is exact over the stated k range (fdlibm technique):
// double HI keeps 42/31 bits (exact for exp |k|<=2048, trig |k|<2^22),
// float HI keeps 12 bits (exact for |k|<4096, covering exp and |x|<=5000).
template <typename T> struct Consts;
template <> struct Consts<double>
{
    static constexpr double pi() noexcept
    {
        return std::numbers::pi_v<double>;
    }
    static constexpr double pi_2() noexcept
    {
        return 1.5707963267948966;
    }
    static constexpr double pio2_1() noexcept
    {
        return 1.57079632673412561417e+00;
    }
    static constexpr double pio2_1t() noexcept
    {
        return 6.07710050650619224932e-11;
    }
    static constexpr double inv_pio2() noexcept
    {
        return 6.36619772367581382433e-01;
    }
    static constexpr double ln2_hi() noexcept
    {
        return 6.93147180369123816490e-01;
    }
    static constexpr double ln2_lo() noexcept
    {
        return 1.90821492927058770002e-10;
    }
    static constexpr double ln2() noexcept
    {
        return std::numbers::ln2_v<double>;
    }
    static constexpr double inv_ln2() noexcept
    {
        return 1.44269504088896340736e+00;
    }
    static constexpr double inv_ln10() noexcept
    {
        return 4.34294481903251850931e-01;
    }
    static constexpr double sqrt2() noexcept
    {
        return std::numbers::sqrt2_v<double>;
    }
    static constexpr double exp_hi() noexcept
    {
        return 709.78271289338400;
    }
    static constexpr double exp_lo() noexcept
    {
        return -745.13321910194110;
    }
    static constexpr double trig_big() noexcept
    {
        return 1.0e6;
    } // |x| above this uses the scalar libm fallback
    static constexpr double exp2_hi() noexcept
    {
        return 1024.0;
    } // 2^1024 overflows to inf
    static constexpr double exp2_lo() noexcept
    {
        return -1075.0;
    } // 2^-1075 underflows to +0
    static constexpr double min_normal() noexcept
    {
        return 2.2250738585072014e-308;
    }
    static constexpr double asinh_big() noexcept
    {
        return 1.0e150;
    } // above this, asinh/acosh use ln2+log(x)
    static constexpr double tiny() noexcept
    {
        return 1.4901161193847656e-08;
    } // 2^-26: below, atan(x)/asinh(x) round to x
    static constexpr double veltkamp() noexcept
    {
        return 134217729.0;
    } // 2^27+1: Veltkamp split constant for f64
};
template <> struct Consts<float>
{
    static constexpr float pi() noexcept
    {
        return std::numbers::pi_v<float>;
    }
    static constexpr float pi_2() noexcept
    {
        return 1.5707963267948966f;
    }
    static constexpr float pio2_1() noexcept
    {
        return 1.5703125f;
    } // 0x3FC90000, 12 trailing zero bits
    static constexpr float pio2_1t() noexcept
    {
        return 4.838267948966e-04f;
    } // pi/2 - pio2_1
    static constexpr float inv_pio2() noexcept
    {
        return 6.3661977236758138e-01f;
    }
    static constexpr float ln2_hi() noexcept
    {
        return 0.693359375f;
    } // 0x3F317000, 12 trailing zero bits
    static constexpr float ln2_lo() noexcept
    {
        return -2.121944400547e-04f;
    } // ln2 - ln2_hi
    static constexpr float ln2() noexcept
    {
        return std::numbers::ln2_v<float>;
    }
    static constexpr float inv_ln2() noexcept
    {
        return 1.4426950408889634e+00f;
    }
    static constexpr float inv_ln10() noexcept
    {
        return 4.3429448190325182e-01f;
    }
    static constexpr float sqrt2() noexcept
    {
        return std::numbers::sqrt2_v<float>;
    }
    static constexpr float exp_hi() noexcept
    {
        return 88.722831726074218f;
    }
    static constexpr float exp_lo() noexcept
    {
        return -103.97208404541016f;
    }
    static constexpr float trig_big() noexcept
    {
        return 5.0e3f;
    }
    static constexpr float exp2_hi() noexcept
    {
        return 128.0f;
    }
    static constexpr float exp2_lo() noexcept
    {
        return -150.0f;
    }
    static constexpr float min_normal() noexcept
    {
        return 1.1754943508222875e-38f;
    }
    static constexpr float asinh_big() noexcept
    {
        return 1.0e19f;
    }
    static constexpr float tiny() noexcept
    {
        return 2.44140625e-04f;
    } // 2^-12: below, atan(x)/asinh(x) round to x
    static constexpr float veltkamp() noexcept
    {
        return 4097.0f;
    } // 2^12+1: Veltkamp split constant for f32
};

// Scalar traits: width-1 backend so kernels compile unchanged without SIMD.
template <typename T> struct ScalarTraits
{
    using scalar = T;
    using vec = T;
    using ivec = std::int64_t;
    using mask = bool;
    static constexpr std::size_t W = 1;
    static inline vec loadu(const T *p) noexcept
    {
        return *p;
    }
    static inline void storeu(vec v, T *p) noexcept
    {
        *p = v;
    }
    static inline vec set1(T v) noexcept
    {
        return v;
    }
    static inline vec setzero() noexcept
    {
        return T{0};
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return a + b;
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return a - b;
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return a * b;
    }
    static inline vec div(vec a, vec b) noexcept
    {
        return a / b;
    }
    static inline vec sqrt(vec a) noexcept
    {
        return std::sqrt(a);
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
        return std::fma(a, b, c);
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return std::fmin(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return std::fmax(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return std::fabs(a);
    }
    static inline vec neg(vec a) noexcept
    {
        return -a;
    }
    static inline vec floor(vec a) noexcept
    {
        return std::floor(a);
    }
    static inline vec ceil(vec a) noexcept
    {
        return std::ceil(a);
    }
    static inline vec trunc(vec a) noexcept
    {
        return std::trunc(a);
    }
    static inline vec nearbyint(vec a) noexcept
    {
        return std::nearbyint(a);
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return a == b;
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return a < b;
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return a <= b;
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return a && b;
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return a || b;
    }
    static inline mask not_mask(mask a) noexcept
    {
        return !a;
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
        return m ? b : a;
    }
    static inline bool any(mask m) noexcept
    {
        return m;
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        lanes[0] = m ? 1u : 0u;
    }
    static inline vec scale2k(vec k) noexcept
    {
        // The kernels clamp k into [-2048, 2048]; re-clamp defensively so the
        // float->int conversion below can never be undefined (UBSan-clean).
        T kc = k < T{-2048} ? T{-2048} : (k > T{2048} ? T{2048} : k);
        if (!std::isfinite(kc))
        {
            kc = T{0};
        }
        return std::ldexp(T{1}, static_cast<int>(kc));
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        int ei = 0;
        const vec m = std::frexp(x, &ei);
        e = static_cast<ivec>(ei);
        return m * T{2};
    }
    static inline vec from_int(ivec e) noexcept
    {
        return static_cast<T>(e);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        return std::copysign(mag, sgn);
    }
};

#if defined(NP_VECMATH_HAS_SSE2)
// 128-bit x86 tier (also used on AVX1 hosts, which lack 256-bit integer ops).
struct SseF64
{
    using scalar = double;
    using vec = __m128d;
    using ivec = __m128i; // 2x i32 in the low half
    using mask = __m128d;
    static constexpr std::size_t W = 2;
    static inline vec loadu(const double *p) noexcept
    {
        return _mm_loadu_pd(p);
    }
    static inline void storeu(vec v, double *p) noexcept
    {
        _mm_storeu_pd(p, v);
    }
    static inline vec set1(double v) noexcept
    {
        return _mm_set1_pd(v);
    }
    static inline vec setzero() noexcept
    {
        return _mm_setzero_pd();
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return _mm_add_pd(a, b);
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return _mm_sub_pd(a, b);
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return _mm_mul_pd(a, b);
    }
    static inline vec div(vec a, vec b) noexcept
    {
        return _mm_div_pd(a, b);
    }
    static inline vec sqrt(vec a) noexcept
    {
        return _mm_sqrt_pd(a);
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
#if defined(__FMA__)
        return _mm_fmadd_pd(a, b, c);
#else
        return _mm_add_pd(_mm_mul_pd(a, b), c);
#endif
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return _mm_min_pd(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return _mm_max_pd(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return _mm_and_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    }
    static inline vec neg(vec a) noexcept
    {
        return _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x8000000000000000LL)));
    }
    static inline vec nearbyint(vec a) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
        // 2^52 magic: round-to-nearest-even under the default FP mode.
        const vec c = _mm_set1_pd(4503599627370496.0);
        vec t = _mm_sub_pd(_mm_add_pd(abs(a), c), c);
        t = blend(a, t, cmplt(abs(a), c)); // |a| >= 2^52 already integral
        return _mm_or_pd(t, _mm_and_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x8000000000000000LL))));
#endif
    }
    static inline vec floor(vec a) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_floor_pd(a);
#else
        const vec r = nearbyint(a);
        return blend(r, sub(r, set1(1.0)), cmplt(a, r));
#endif
    }
    static inline vec ceil(vec a) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_ceil_pd(a);
#else
        const vec r = nearbyint(a);
        return blend(r, add(r, set1(1.0)), cmplt(r, a));
#endif
    }
    static inline vec trunc(vec a) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
        const vec t = floor(abs(a));
        return _mm_or_pd(t, _mm_and_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x8000000000000000LL))));
#endif
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return _mm_cmpeq_pd(a, b);
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return _mm_cmplt_pd(a, b);
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return _mm_cmple_pd(a, b);
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return _mm_and_pd(a, b);
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return _mm_or_pd(a, b);
    }
    static inline mask not_mask(mask a) noexcept
    {
        return _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi32(-1)));
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_blendv_pd(a, b, m);
#else
        return _mm_or_pd(_mm_andnot_pd(m, a), _mm_and_pd(m, b));
#endif
    }
    static inline bool any(mask m) noexcept
    {
        return _mm_movemask_pd(m) != 0;
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        const int bits = _mm_movemask_pd(m);
        lanes[0] = (bits & 1) ? 1u : 0u;
        lanes[1] = (bits & 2) ? 1u : 0u;
    }
    static inline vec scale2k(vec k) noexcept
    {
        // |k| <= 2048 in every caller: exact i32 convert, sign-extend to i64,
        // add the double bias, shift into the exponent field.
        __m128i i = _mm_cvtpd_epi32(k);
        const __m128i s = _mm_srai_epi32(i, 31);
        __m128i w = _mm_unpacklo_epi32(i, s);
        w = _mm_add_epi64(w, _mm_set1_epi64x(1023));
        w = _mm_slli_epi64(w, 52);
        return _mm_castsi128_pd(w);
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        const __m128i ix = _mm_castpd_si128(x);
        __m128i e64 = _mm_srli_epi64(ix, 52);
        e64 = _mm_and_si128(e64, _mm_set1_epi64x(0x7ff));
        const __m128i e32 = _mm_shuffle_epi32(e64, _MM_SHUFFLE(2, 0, 2, 0));
        e = _mm_sub_epi32(e32, _mm_set1_epi32(1023));
        const __m128i m =
            _mm_or_si128(_mm_and_si128(ix, _mm_set1_epi64x(0xFFFFFFFFFFFFFLL)), _mm_set1_epi64x(0x3FF0000000000000LL));
        return _mm_castsi128_pd(m);
    }
    static inline vec from_int(ivec e) noexcept
    {
        return _mm_cvtepi32_pd(e);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        return _mm_or_pd(_mm_and_pd(mag, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL))),
                         _mm_and_pd(sgn, _mm_castsi128_pd(_mm_set1_epi64x(0x8000000000000000LL))));
    }
};

struct SseF32
{
    using scalar = float;
    using vec = __m128;
    using ivec = __m128i;
    using mask = __m128;
    static constexpr std::size_t W = 4;
    static inline vec loadu(const float *p) noexcept
    {
        return _mm_loadu_ps(p);
    }
    static inline void storeu(vec v, float *p) noexcept
    {
        _mm_storeu_ps(p, v);
    }
    static inline vec set1(float v) noexcept
    {
        return _mm_set1_ps(v);
    }
    static inline vec setzero() noexcept
    {
        return _mm_setzero_ps();
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return _mm_add_ps(a, b);
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return _mm_sub_ps(a, b);
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return _mm_mul_ps(a, b);
    }
    static inline vec div(vec a, vec b) noexcept
    {
        return _mm_div_ps(a, b);
    }
    static inline vec sqrt(vec a) noexcept
    {
        return _mm_sqrt_ps(a);
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
#if defined(__FMA__)
        return _mm_fmadd_ps(a, b, c);
#else
        return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return _mm_min_ps(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return _mm_max_ps(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
    }
    static inline vec neg(vec a) noexcept
    {
        return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000))));
    }
    static inline vec nearbyint(vec a) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
        const vec c = _mm_set1_ps(8388608.0f); // 2^23
        vec t = _mm_sub_ps(_mm_add_ps(abs(a), c), c);
        t = blend(a, t, cmplt(abs(a), c));
        return _mm_or_ps(t, _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000)))));
#endif
    }
    static inline vec floor(vec a) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_floor_ps(a);
#else
        const vec r = nearbyint(a);
        return blend(r, sub(r, set1(1.0f)), cmplt(a, r));
#endif
    }
    static inline vec ceil(vec a) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_ceil_ps(a);
#else
        const vec r = nearbyint(a);
        return blend(r, add(r, set1(1.0f)), cmplt(r, a));
#endif
    }
    static inline vec trunc(vec a) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
        const vec t = floor(abs(a));
        return _mm_or_ps(t, _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000)))));
#endif
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return _mm_cmpeq_ps(a, b);
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return _mm_cmplt_ps(a, b);
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return _mm_cmple_ps(a, b);
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return _mm_and_ps(a, b);
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return _mm_or_ps(a, b);
    }
    static inline mask not_mask(mask a) noexcept
    {
        return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1)));
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
#if defined(NP_VECMATH_HAS_SSE41)
        return _mm_blendv_ps(a, b, m);
#else
        return _mm_or_ps(_mm_andnot_ps(m, a), _mm_and_ps(m, b));
#endif
    }
    static inline bool any(mask m) noexcept
    {
        return _mm_movemask_ps(m) != 0;
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        const int bits = _mm_movemask_ps(m);
        for (int j = 0; j < 4; ++j)
        {
            lanes[static_cast<std::size_t>(j)] = (bits & (1 << j)) ? 1u : 0u;
        }
    }
    static inline vec scale2k(vec k) noexcept
    {
        __m128i i = _mm_cvtps_epi32(k);
        i = _mm_add_epi32(i, _mm_set1_epi32(127));
        i = _mm_slli_epi32(i, 23);
        return _mm_castsi128_ps(i);
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        const __m128i ix = _mm_castps_si128(x);
        __m128i e32 = _mm_srli_epi32(ix, 23);
        e32 = _mm_and_si128(e32, _mm_set1_epi32(0xff));
        e = _mm_sub_epi32(e32, _mm_set1_epi32(127));
        const __m128i m = _mm_or_si128(_mm_and_si128(ix, _mm_set1_epi32(0x7FFFFF)), _mm_set1_epi32(0x3F800000));
        return _mm_castsi128_ps(m);
    }
    static inline vec from_int(ivec e) noexcept
    {
        return _mm_cvtepi32_ps(e);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        return _mm_or_ps(_mm_and_ps(mag, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))),
                         _mm_and_ps(sgn, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000)))));
    }
};
#endif // NP_VECMATH_HAS_SSE2

#if defined(NP_VECMATH_HAS_AVX2)
// 256-bit tier. f32 uses native 256-bit integer ops; f64 bit tricks reuse the
// 128-bit helpers per half (AVX2 lacks i64->f64 conversion).
struct AvxF64
{
    using scalar = double;
    using vec = __m256d;
    using ivec = __m128i; // 4x i32, converted with _mm256_cvtepi32_pd
    using mask = __m256d;
    static constexpr std::size_t W = 4;
    static inline vec loadu(const double *p) noexcept
    {
        return _mm256_loadu_pd(p);
    }
    static inline void storeu(vec v, double *p) noexcept
    {
        _mm256_storeu_pd(p, v);
    }
    static inline vec set1(double v) noexcept
    {
        return _mm256_set1_pd(v);
    }
    static inline vec setzero() noexcept
    {
        return _mm256_setzero_pd();
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return _mm256_add_pd(a, b);
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return _mm256_sub_pd(a, b);
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return _mm256_mul_pd(a, b);
    }
    static inline vec div(vec a, vec b) noexcept
    {
        return _mm256_div_pd(a, b);
    }
    static inline vec sqrt(vec a) noexcept
    {
        return _mm256_sqrt_pd(a);
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
#if defined(__FMA__)
        return _mm256_fmadd_pd(a, b, c);
#else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return _mm256_min_pd(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return _mm256_max_pd(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return _mm256_and_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    }
    static inline vec neg(vec a) noexcept
    {
        return _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000LL)));
    }
    static inline vec nearbyint(vec a) noexcept
    {
        return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    static inline vec floor(vec a) noexcept
    {
        return _mm256_floor_pd(a);
    }
    static inline vec ceil(vec a) noexcept
    {
        return _mm256_ceil_pd(a);
    }
    static inline vec trunc(vec a) noexcept
    {
        return _mm256_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return _mm256_and_pd(a, b);
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return _mm256_or_pd(a, b);
    }
    static inline mask not_mask(mask a) noexcept
    {
        return _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
        return _mm256_blendv_pd(a, b, m);
    }
    static inline bool any(mask m) noexcept
    {
        return _mm256_movemask_pd(m) != 0;
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        const int bits = _mm256_movemask_pd(m);
        for (int j = 0; j < 4; ++j)
        {
            lanes[static_cast<std::size_t>(j)] = (bits & (1 << j)) ? 1u : 0u;
        }
    }
    static inline vec scale2k(vec k) noexcept
    {
        const __m128d lo = _mm256_castpd256_pd128(k);
        const __m128d hi = _mm256_extractf128_pd(k, 1);
        const __m128d rlo = SseF64::scale2k(lo);
        const __m128d rhi = SseF64::scale2k(hi);
        __m256d r = _mm256_castpd128_pd256(rlo);
        r = _mm256_insertf128_pd(r, rhi, 1);
        return r;
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        const __m128d lo = _mm256_castpd256_pd128(x);
        const __m128d hi = _mm256_extractf128_pd(x, 1);
        __m128i elo = _mm_setzero_si128();
        __m128i ehi = _mm_setzero_si128();
        const __m128d mlo = SseF64::norm_mant(lo, elo);
        const __m128d mhi = SseF64::norm_mant(hi, ehi);
        // elo holds [e0,e1], ehi [e2,e3] in the low 64 bits each.
        e = _mm_unpacklo_epi64(elo, ehi);
        __m256d m = _mm256_castpd128_pd256(mlo);
        m = _mm256_insertf128_pd(m, mhi, 1);
        return m;
    }
    static inline vec from_int(ivec e) noexcept
    {
        return _mm256_cvtepi32_pd(e);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        return _mm256_or_pd(_mm256_and_pd(mag, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL))),
                            _mm256_and_pd(sgn, _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000LL))));
    }
};

struct AvxF32
{
    using scalar = float;
    using vec = __m256;
    using ivec = __m256i;
    using mask = __m256;
    static constexpr std::size_t W = 8;
    static inline vec loadu(const float *p) noexcept
    {
        return _mm256_loadu_ps(p);
    }
    static inline void storeu(vec v, float *p) noexcept
    {
        _mm256_storeu_ps(p, v);
    }
    static inline vec set1(float v) noexcept
    {
        return _mm256_set1_ps(v);
    }
    static inline vec setzero() noexcept
    {
        return _mm256_setzero_ps();
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return _mm256_add_ps(a, b);
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return _mm256_sub_ps(a, b);
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return _mm256_mul_ps(a, b);
    }
    static inline vec div(vec a, vec b) noexcept
    {
        return _mm256_div_ps(a, b);
    }
    static inline vec sqrt(vec a) noexcept
    {
        return _mm256_sqrt_ps(a);
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
#if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
#else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return _mm256_min_ps(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return _mm256_max_ps(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    }
    static inline vec neg(vec a) noexcept
    {
        return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000))));
    }
    static inline vec nearbyint(vec a) noexcept
    {
        return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    static inline vec floor(vec a) noexcept
    {
        return _mm256_floor_ps(a);
    }
    static inline vec ceil(vec a) noexcept
    {
        return _mm256_ceil_ps(a);
    }
    static inline vec trunc(vec a) noexcept
    {
        return _mm256_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return _mm256_and_ps(a, b);
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return _mm256_or_ps(a, b);
    }
    static inline mask not_mask(mask a) noexcept
    {
        return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
        return _mm256_blendv_ps(a, b, m);
    }
    static inline bool any(mask m) noexcept
    {
        return _mm256_movemask_ps(m) != 0;
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        const int bits = _mm256_movemask_ps(m);
        for (int j = 0; j < 8; ++j)
        {
            lanes[static_cast<std::size_t>(j)] = (bits & (1 << j)) ? 1u : 0u;
        }
    }
    static inline vec scale2k(vec k) noexcept
    {
        __m256i i = _mm256_cvtps_epi32(k);
        i = _mm256_add_epi32(i, _mm256_set1_epi32(127));
        i = _mm256_slli_epi32(i, 23);
        return _mm256_castsi256_ps(i);
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        const __m256i ix = _mm256_castps_si256(x);
        __m256i e32 = _mm256_srli_epi32(ix, 23);
        e32 = _mm256_and_si256(e32, _mm256_set1_epi32(0xff));
        e = _mm256_sub_epi32(e32, _mm256_set1_epi32(127));
        const __m256i m =
            _mm256_or_si256(_mm256_and_si256(ix, _mm256_set1_epi32(0x7FFFFF)), _mm256_set1_epi32(0x3F800000));
        return _mm256_castsi256_ps(m);
    }
    static inline vec from_int(ivec e) noexcept
    {
        return _mm256_cvtepi32_ps(e);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        return _mm256_or_ps(_mm256_and_ps(mag, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))),
                            _mm256_and_ps(sgn, _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000)))));
    }
};
#endif // NP_VECMATH_HAS_AVX2

#if defined(NP_VECMATH_HAS_AVX512)
// 512-bit tier. Polynomials run at full width; the integer bit tricks use a
// scalar assist through stack buffers (DQ-independent, obviously correct; the
// assist cost is negligible next to polynomial evaluation).
struct Avx512F64
{
    using scalar = double;
    using vec = __m512d;
    using ivec = __m512i;
    using mask = __mmask8;
    static constexpr std::size_t W = 8;
    static inline vec loadu(const double *p) noexcept
    {
        return _mm512_loadu_pd(p);
    }
    static inline void storeu(vec v, double *p) noexcept
    {
        _mm512_storeu_pd(p, v);
    }
    static inline vec set1(double v) noexcept
    {
        return _mm512_set1_pd(v);
    }
    static inline vec setzero() noexcept
    {
        return _mm512_setzero_pd();
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return _mm512_add_pd(a, b);
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return _mm512_sub_pd(a, b);
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return _mm512_mul_pd(a, b);
    }
    static inline vec div(vec a, vec b) noexcept
    {
        return _mm512_div_pd(a, b);
    }
    static inline vec sqrt(vec a) noexcept
    {
        return _mm512_sqrt_pd(a);
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
        return _mm512_fmadd_pd(a, b, c);
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return _mm512_min_pd(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return _mm512_max_pd(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a), _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL)));
    }
    static inline vec neg(vec a) noexcept
    {
        return _mm512_castsi512_pd(
            _mm512_xor_si512(_mm512_castpd_si512(a), _mm512_set1_epi64(static_cast<long long>(0x8000000000000000LL))));
    }
    static inline vec nearbyint(vec a) noexcept
    {
        return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    static inline vec floor(vec a) noexcept
    {
        return _mm512_roundscale_pd(a, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
    }
    static inline vec ceil(vec a) noexcept
    {
        return _mm512_roundscale_pd(a, _MM_FROUND_CEIL | _MM_FROUND_NO_EXC);
    }
    static inline vec trunc(vec a) noexcept
    {
        return _mm512_roundscale_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return static_cast<mask>(a & b);
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return static_cast<mask>(a | b);
    }
    static inline mask not_mask(mask a) noexcept
    {
        return static_cast<mask>(~a);
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
        return _mm512_mask_blend_pd(m, a, b);
    }
    static inline bool any(mask m) noexcept
    {
        return m != 0;
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        for (std::size_t j = 0; j < W; ++j)
        {
            lanes[j] = (m & (static_cast<mask>(1u) << j)) ? 1u : 0u;
        }
    }
    static inline vec scale2k(vec k) noexcept
    {
        alignas(64) double kt[W];
        alignas(64) double st[W];
        _mm512_store_pd(kt, k);
        for (std::size_t j = 0; j < W; ++j)
        {
            const std::uint64_t bits = static_cast<std::uint64_t>(static_cast<std::int64_t>(kt[j]) + 1023) << 52;
            std::memcpy(&st[j], &bits, sizeof(bits));
        }
        return _mm512_load_pd(st);
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        alignas(64) double xt[W];
        alignas(64) double mt[W];
        alignas(64) std::int64_t et[W];
        _mm512_store_pd(xt, x);
        for (std::size_t j = 0; j < W; ++j)
        {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &xt[j], sizeof(bits));
            et[j] = static_cast<std::int64_t>((bits >> 52) & 0x7ffu) - 1023;
            bits = (bits & 0xFFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
            std::memcpy(&mt[j], &bits, sizeof(bits));
        }
        e = _mm512_load_si512(et);
        return _mm512_load_pd(mt);
    }
    static inline vec from_int(ivec e) noexcept
    {
        alignas(64) std::int64_t et[W];
        alignas(64) double ft[W];
        _mm512_store_si512(et, e);
        for (std::size_t j = 0; j < W; ++j)
        {
            ft[j] = static_cast<double>(et[j]);
        }
        return _mm512_load_pd(ft);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        __m512i mi = _mm512_and_si512(_mm512_castpd_si512(mag), _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL));
        __m512i si =
            _mm512_and_si512(_mm512_castpd_si512(sgn), _mm512_set1_epi64(static_cast<long long>(0x8000000000000000LL)));
        return _mm512_castsi512_pd(_mm512_or_si512(mi, si));
    }
};

struct Avx512F32
{
    using scalar = float;
    using vec = __m512;
    using ivec = __m512i;
    using mask = __mmask16;
    static constexpr std::size_t W = 16;
    static inline vec loadu(const float *p) noexcept
    {
        return _mm512_loadu_ps(p);
    }
    static inline void storeu(vec v, float *p) noexcept
    {
        _mm512_storeu_ps(p, v);
    }
    static inline vec set1(float v) noexcept
    {
        return _mm512_set1_ps(v);
    }
    static inline vec setzero() noexcept
    {
        return _mm512_setzero_ps();
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return _mm512_add_ps(a, b);
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return _mm512_sub_ps(a, b);
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return _mm512_mul_ps(a, b);
    }
    static inline vec div(vec a, vec b) noexcept
    {
        return _mm512_div_ps(a, b);
    }
    static inline vec sqrt(vec a) noexcept
    {
        return _mm512_sqrt_ps(a);
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
        return _mm512_fmadd_ps(a, b, c);
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return _mm512_min_ps(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return _mm512_max_ps(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_set1_epi32(0x7FFFFFFF)));
    }
    static inline vec neg(vec a) noexcept
    {
        return _mm512_castsi512_ps(
            _mm512_xor_si512(_mm512_castps_si512(a), _mm512_set1_epi32(static_cast<int>(0x80000000))));
    }
    static inline vec nearbyint(vec a) noexcept
    {
        return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    static inline vec floor(vec a) noexcept
    {
        return _mm512_roundscale_ps(a, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
    }
    static inline vec ceil(vec a) noexcept
    {
        return _mm512_roundscale_ps(a, _MM_FROUND_CEIL | _MM_FROUND_NO_EXC);
    }
    static inline vec trunc(vec a) noexcept
    {
        return _mm512_roundscale_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return static_cast<mask>(a & b);
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return static_cast<mask>(a | b);
    }
    static inline mask not_mask(mask a) noexcept
    {
        return static_cast<mask>(~a);
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
        return _mm512_mask_blend_ps(m, a, b);
    }
    static inline bool any(mask m) noexcept
    {
        return m != 0;
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        for (std::size_t j = 0; j < W; ++j)
        {
            lanes[j] = (m & (static_cast<mask>(1u) << j)) ? 1u : 0u;
        }
    }
    static inline vec scale2k(vec k) noexcept
    {
        __m512i i = _mm512_cvtps_epi32(k);
        i = _mm512_add_epi32(i, _mm512_set1_epi32(127));
        i = _mm512_slli_epi32(i, 23);
        return _mm512_castsi512_ps(i);
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        const __m512i ix = _mm512_castps_si512(x);
        __m512i e32 = _mm512_srli_epi32(ix, 23);
        e32 = _mm512_and_si512(e32, _mm512_set1_epi32(0xff));
        e = _mm512_sub_epi32(e32, _mm512_set1_epi32(127));
        const __m512i m =
            _mm512_or_si512(_mm512_and_si512(ix, _mm512_set1_epi32(0x7FFFFF)), _mm512_set1_epi32(0x3F800000));
        return _mm512_castsi512_ps(m);
    }
    static inline vec from_int(ivec e) noexcept
    {
        return _mm512_cvtepi32_ps(e);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        const __m512i mi = _mm512_and_si512(_mm512_castps_si512(mag), _mm512_set1_epi32(0x7FFFFFFF));
        const __m512i si = _mm512_and_si512(_mm512_castps_si512(sgn), _mm512_set1_epi32(static_cast<int>(0x80000000)));
        return _mm512_castsi512_ps(_mm512_or_si512(mi, si));
    }
};
#endif // NP_VECMATH_HAS_AVX512

#if defined(NP_VECMATH_HAS_NEON)
// ARM NEON tier (128-bit). f64 requires AArch64; 32-bit ARM uses scalar f64.
struct NeonF32
{
    using scalar = float;
    using vec = float32x4_t;
    using ivec = int32x4_t;
    using mask = uint32x4_t;
    static constexpr std::size_t W = 4;
    static inline vec loadu(const float *p) noexcept
    {
        return vld1q_f32(p);
    }
    static inline void storeu(vec v, float *p) noexcept
    {
        vst1q_f32(p, v);
    }
    static inline vec set1(float v) noexcept
    {
        return vdupq_n_f32(v);
    }
    static inline vec setzero() noexcept
    {
        return vdupq_n_f32(0.0f);
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return vaddq_f32(a, b);
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return vsubq_f32(a, b);
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return vmulq_f32(a, b);
    }
    static inline vec div(vec a, vec b) noexcept
    {
#if defined(__aarch64__)
        return vdivq_f32(a, b);
#else
        float ta[4], tb[4], tr[4];
        vst1q_f32(ta, a);
        vst1q_f32(tb, b);
        for (int j = 0; j < 4; ++j)
        {
            tr[j] = ta[j] / tb[j];
        }
        return vld1q_f32(tr);
#endif
    }
    static inline vec sqrt(vec a) noexcept
    {
#if defined(__aarch64__)
        return vsqrtq_f32(a);
#else
        float ta[4], tr[4];
        vst1q_f32(ta, a);
        for (int j = 0; j < 4; ++j)
        {
            tr[j] = std::sqrt(ta[j]);
        }
        return vld1q_f32(tr);
#endif
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
#if defined(__aarch64__)
        return vfmaq_f32(c, a, b);
#else
        return vaddq_f32(vmulq_f32(a, b), c);
#endif
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return vminq_f32(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return vmaxq_f32(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return vabsq_f32(a);
    }
    static inline vec neg(vec a) noexcept
    {
        return vnegq_f32(a);
    }
    static inline vec nearbyint(vec a) noexcept
    {
#if defined(__aarch64__)
        return vrndnq_f32(a);
#else
        float t[4];
        vst1q_f32(t, a);
        for (int j = 0; j < 4; ++j)
        {
            t[j] = std::nearbyint(t[j]);
        }
        return vld1q_f32(t);
#endif
    }
    static inline vec floor(vec a) noexcept
    {
#if defined(__aarch64__)
        return vrndmq_f32(a);
#else
        const vec r = nearbyint(a);
        return blend(r, sub(r, set1(1.0f)), cmplt(a, r));
#endif
    }
    static inline vec ceil(vec a) noexcept
    {
#if defined(__aarch64__)
        return vrndpq_f32(a);
#else
        const vec r = nearbyint(a);
        return blend(r, add(r, set1(1.0f)), cmplt(r, a));
#endif
    }
    static inline vec trunc(vec a) noexcept
    {
#if defined(__aarch64__)
        return vrndq_f32(a);
#else
        const vec t = floor(abs(a));
        const uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(a), vdupq_n_u32(0x80000000u));
        return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(t), sign));
#endif
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return vceqq_f32(a, b);
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return vcltq_f32(a, b);
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return vcleq_f32(a, b);
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return vandq_u32(a, b);
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return vorrq_u32(a, b);
    }
    static inline mask not_mask(mask a) noexcept
    {
        return vmvnq_u32(a);
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
        return vbslq_f32(m, b, a);
    }
    static inline bool any(mask m) noexcept
    {
#if defined(__aarch64__)
        return vmaxvq_u32(m) != 0u;
#else
        const uint32x2_t o = vorr_u32(vget_low_u32(m), vget_high_u32(m));
        return (vget_lane_u32(o, 0) | vget_lane_u32(o, 1)) != 0u;
#endif
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        alignas(16) std::uint32_t t[4];
        vst1q_u32(t, m);
        for (int j = 0; j < 4; ++j)
        {
            lanes[static_cast<std::size_t>(j)] = t[j] ? 1u : 0u;
        }
    }
    static inline vec scale2k(vec k) noexcept
    {
        int32x4_t i = vcvtq_s32_f32(k);
        i = vaddq_s32(i, vdupq_n_s32(127));
        i = vshlq_n_s32(i, 23);
        return vreinterpretq_f32_s32(i);
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        const int32x4_t ix = vreinterpretq_s32_f32(x);
        int32x4_t e32 = vshrq_n_s32(ix, 23);
        e32 = vandq_s32(e32, vdupq_n_s32(0xff));
        e = vsubq_s32(e32, vdupq_n_s32(127));
        const int32x4_t m = vorrq_s32(vandq_s32(ix, vdupq_n_s32(0x7FFFFF)), vdupq_n_s32(0x3F800000));
        return vreinterpretq_f32_s32(m);
    }
    static inline vec from_int(ivec e) noexcept
    {
        return vcvtq_f32_s32(e);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        const uint32x4_t m = vandq_u32(vreinterpretq_u32_f32(mag), vdupq_n_u32(0x7FFFFFFFu));
        const uint32x4_t s = vandq_u32(vreinterpretq_u32_f32(sgn), vdupq_n_u32(0x80000000u));
        return vreinterpretq_f32_u32(vorrq_u32(m, s));
    }
};

#if defined(NP_VECMATH_HAS_NEON64)
struct NeonF64
{
    using scalar = double;
    using vec = float64x2_t;
    using ivec = int64x2_t;
    using mask = uint64x2_t;
    static constexpr std::size_t W = 2;
    static inline vec loadu(const double *p) noexcept
    {
        return vld1q_f64(p);
    }
    static inline void storeu(vec v, double *p) noexcept
    {
        vst1q_f64(p, v);
    }
    static inline vec set1(double v) noexcept
    {
        return vdupq_n_f64(v);
    }
    static inline vec setzero() noexcept
    {
        return vdupq_n_f64(0.0);
    }
    static inline vec add(vec a, vec b) noexcept
    {
        return vaddq_f64(a, b);
    }
    static inline vec sub(vec a, vec b) noexcept
    {
        return vsubq_f64(a, b);
    }
    static inline vec mul(vec a, vec b) noexcept
    {
        return vmulq_f64(a, b);
    }
    static inline vec div(vec a, vec b) noexcept
    {
        return vdivq_f64(a, b);
    }
    static inline vec sqrt(vec a) noexcept
    {
        return vsqrtq_f64(a);
    }
    static inline vec fma(vec a, vec b, vec c) noexcept
    {
        return vfmaq_f64(c, a, b);
    }
    static inline vec min(vec a, vec b) noexcept
    {
        return vminq_f64(a, b);
    }
    static inline vec max(vec a, vec b) noexcept
    {
        return vmaxq_f64(a, b);
    }
    static inline vec abs(vec a) noexcept
    {
        return vabsq_f64(a);
    }
    static inline vec neg(vec a) noexcept
    {
        return vnegq_f64(a);
    }
    static inline vec nearbyint(vec a) noexcept
    {
        return vrndnq_f64(a);
    }
    static inline vec floor(vec a) noexcept
    {
        return vrndmq_f64(a);
    }
    static inline vec ceil(vec a) noexcept
    {
        return vrndpq_f64(a);
    }
    static inline vec trunc(vec a) noexcept
    {
        return vrndq_f64(a);
    }
    static inline mask cmpeq(vec a, vec b) noexcept
    {
        return vceqq_f64(a, b);
    }
    static inline mask cmplt(vec a, vec b) noexcept
    {
        return vcltq_f64(a, b);
    }
    static inline mask cmple(vec a, vec b) noexcept
    {
        return vcleq_f64(a, b);
    }
    static inline mask and_mask(mask a, mask b) noexcept
    {
        return vandq_u64(a, b);
    }
    static inline mask or_mask(mask a, mask b) noexcept
    {
        return vorrq_u64(a, b);
    }
    static inline mask not_mask(mask a) noexcept
    {
        // vmvn has no 64-bit form; round-trip through u32.
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(a)));
    }
    static inline vec blend(vec a, vec b, mask m) noexcept
    {
        return vbslq_f64(m, b, a);
    }
    static inline bool any(mask m) noexcept
    {
        return vmaxvq_u64(m) != 0u;
    }
    static inline void mask_lanes(mask m, std::uint64_t *lanes) noexcept
    {
        alignas(16) std::uint64_t t[2];
        vst1q_u64(t, m);
        lanes[0] = t[0] ? 1u : 0u;
        lanes[1] = t[1] ? 1u : 0u;
    }
    static inline vec scale2k(vec k) noexcept
    {
        int64x2_t i = vcvtq_s64_f64(k);
        i = vaddq_s64(i, vdupq_n_s64(1023));
        i = vshlq_n_s64(i, 52);
        return vreinterpretq_f64_s64(i);
    }
    static inline vec norm_mant(vec x, ivec &e) noexcept
    {
        const uint64x2_t ix = vreinterpretq_u64_f64(x);
        uint64x2_t e64 = vshrq_n_u64(ix, 52);
        e64 = vandq_u64(e64, vdupq_n_u64(0x7ffu));
        e = vsubq_s64(vreinterpretq_s64_u64(e64), vdupq_n_s64(1023));
        const uint64x2_t m =
            vorrq_u64(vandq_u64(ix, vdupq_n_u64(0xFFFFFFFFFFFFFULL)), vdupq_n_u64(0x3FF0000000000000ULL));
        return vreinterpretq_f64_u64(m);
    }
    static inline vec from_int(ivec e) noexcept
    {
        return vcvtq_f64_s64(e);
    }
    static inline vec copysign_to(vec mag, vec sgn) noexcept
    {
        const uint64x2_t m = vandq_u64(vreinterpretq_u64_f64(mag), vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL));
        const uint64x2_t s = vandq_u64(vreinterpretq_u64_f64(sgn), vdupq_n_u64(0x8000000000000000ULL));
        return vreinterpretq_f64_u64(vorrq_u64(m, s));
    }
};
#endif // NP_VECMATH_HAS_NEON64
#endif // NP_VECMATH_HAS_NEON

// Traits selection: widest usable tier. Plain AVX without AVX2 deliberately
// falls back to the 128-bit tier (no 256-bit integer arithmetic there).
template <typename T> struct Traits;
#if defined(NP_VECMATH_HAS_AVX512)
template <> struct Traits<float>
{
    using type = Avx512F32;
};
template <> struct Traits<double>
{
    using type = Avx512F64;
};
#elif defined(NP_VECMATH_HAS_AVX2)
template <> struct Traits<float>
{
    using type = AvxF32;
};
template <> struct Traits<double>
{
    using type = AvxF64;
};
#elif defined(NP_VECMATH_HAS_SSE2)
template <> struct Traits<float>
{
    using type = SseF32;
};
template <> struct Traits<double>
{
    using type = SseF64;
};
#elif defined(NP_VECMATH_HAS_NEON)
template <> struct Traits<float>
{
    using type = NeonF32;
};
#if defined(NP_VECMATH_HAS_NEON64)
template <> struct Traits<double>
{
    using type = NeonF64;
};
#else
template <> struct Traits<double>
{
    using type = ScalarTraits<double>;
};
#endif
#else
template <> struct Traits<float>
{
    using type = ScalarTraits<float>;
};
template <> struct Traits<double>
{
    using type = ScalarTraits<double>;
};
#endif

// Shared mask predicates.
template <typename Tr> inline typename Tr::mask is_nan(typename Tr::vec x) noexcept
{
    return Tr::not_mask(Tr::cmpeq(x, x));
}
template <typename Tr> inline typename Tr::mask is_inf(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    return Tr::cmpeq(Tr::abs(x), Tr::set1(std::numeric_limits<S>::infinity()));
}
template <typename Tr> inline typename Tr::mask not_finite(typename Tr::vec x) noexcept
{
    return Tr::or_mask(is_nan<Tr>(x), is_inf<Tr>(x));
}
template <typename Tr> inline typename Tr::mask is_finite(typename Tr::vec x) noexcept
{
    return Tr::not_mask(not_finite<Tr>(x));
}

// exp Taylor of degree 12 around 0 (tail < 3e-16 for |t| <= ln2/2).
template <typename Tr> inline typename Tr::vec exp_taylor(typename Tr::vec t) noexcept
{
    using S = typename Tr::scalar;
    typename Tr::vec p = Tr::set1(S(2.08767569878681e-9));
    p = Tr::fma(p, t, Tr::set1(S(2.505210838544172e-8)));
    p = Tr::fma(p, t, Tr::set1(S(2.755731922398589e-7)));
    p = Tr::fma(p, t, Tr::set1(S(2.7557319223985893e-6)));
    p = Tr::fma(p, t, Tr::set1(S(2.48015873015873e-5)));
    p = Tr::fma(p, t, Tr::set1(S(1.984126984126984e-4)));
    p = Tr::fma(p, t, Tr::set1(S(1.388888888888889e-3)));
    p = Tr::fma(p, t, Tr::set1(S(8.333333333333333e-3)));
    p = Tr::fma(p, t, Tr::set1(S(4.1666666666666664e-2)));
    p = Tr::fma(p, t, Tr::set1(S(1.6666666666666666e-1)));
    p = Tr::fma(p, t, Tr::set1(S(5.0e-1)));
    p = Tr::fma(p, t, Tr::set1(S(1.0)));
    p = Tr::fma(p, t, Tr::set1(S(1.0)));
    return p;
}

// exp kernel: k = round(x/ln2), r in [-ln2/2, ln2/2], degree-12 Taylor of
// exp(r) (tail < 3e-16 there), scaled by the exact 2^k.
template <typename Tr> inline typename Tr::vec exp_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec k = Tr::nearbyint(Tr::mul(x, Tr::set1(C::inv_ln2())));
    const typename Tr::vec r =
        Tr::sub(Tr::sub(x, Tr::mul(k, Tr::set1(C::ln2_hi()))), Tr::mul(k, Tr::set1(C::ln2_lo())));
    const typename Tr::vec p = exp_taylor<Tr>(r);
    // Integer conversion of out-of-range k is undefined; clamp into the exact
    // range (exotic lanes are overwritten by the blends below). The scale is
    // split in halves so k = +/-1024 stays exact instead of overflowing.
    typename Tr::vec ks = Tr::blend(k, Tr::setzero(), not_finite<Tr>(x));
    ks = Tr::max(Tr::min(ks, Tr::set1(S{2048})), Tr::set1(S{-2048}));
    const typename Tr::vec kh = Tr::floor(Tr::mul(ks, Tr::set1(S{0.5})));
    const typename Tr::vec kl = Tr::sub(ks, kh);
    typename Tr::vec y = Tr::mul(Tr::mul(p, Tr::scale2k(kh)), Tr::scale2k(kl));
    const typename Tr::vec inf = Tr::set1(std::numeric_limits<S>::infinity());
    y = Tr::blend(y, inf, Tr::or_mask(Tr::cmpeq(x, inf), Tr::cmplt(Tr::set1(C::exp_hi()), x)));
    y = Tr::blend(y, Tr::setzero(), Tr::or_mask(Tr::cmpeq(x, Tr::neg(inf)), Tr::cmplt(x, Tr::set1(C::exp_lo()))));
    return y; // NaN flows through the polynomial untouched.
}

// log kernel: mantissa in [1,2) via bit trick, halve above sqrt(2), atanh
// series in s = (m-1)/(m+1) (|s| <= 0.1716, degree-17 tail < 3e-16).
template <typename Tr> inline typename Tr::vec log_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    typename Tr::ivec e;
    std::memset(&e, 0, sizeof(e)); // written by norm_mant below; zeroed for MSVC vector-type init
    const typename Tr::vec inf = Tr::set1(std::numeric_limits<S>::infinity());
    // Only normal positive finite inputs take the fast path.
    const typename Tr::mask ok = Tr::and_mask(Tr::and_mask(Tr::cmplt(Tr::setzero(), x), Tr::cmplt(x, inf)),
                                              Tr::cmple(Tr::set1(C::min_normal()), x));
    const typename Tr::vec xs = Tr::blend(x, Tr::set1(S{1}), Tr::not_mask(ok));
    typename Tr::vec m = Tr::norm_mant(xs, e);
    const typename Tr::mask gt = Tr::cmplt(Tr::set1(C::sqrt2()), m);
    m = Tr::blend(m, Tr::mul(m, Tr::set1(S{0.5})), gt);
    typename Tr::vec ev = Tr::from_int(e);
    ev = Tr::blend(ev, Tr::add(ev, Tr::set1(S{1})), gt);
    const typename Tr::vec s = Tr::div(Tr::sub(m, Tr::set1(S{1})), Tr::add(m, Tr::set1(S{1})));
    const typename Tr::vec s2 = Tr::mul(s, s);
    typename Tr::vec p = Tr::set1(S(2.0 / 21.0));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 19.0)));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 17.0)));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 15.0)));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 13.0)));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 11.0)));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 9.0)));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 7.0)));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 5.0)));
    p = Tr::fma(p, s2, Tr::set1(S(2.0 / 3.0)));
    p = Tr::fma(p, s2, Tr::set1(S{2}));
    p = Tr::mul(s, p);
    typename Tr::vec y = Tr::fma(ev, Tr::set1(C::ln2()), p);
    const typename Tr::vec nan = Tr::set1(std::numeric_limits<S>::quiet_NaN());
    y = Tr::blend(y, Tr::neg(inf), Tr::cmpeq(x, Tr::setzero()));
    y = Tr::blend(y, nan, Tr::and_mask(Tr::cmplt(x, Tr::setzero()), Tr::not_mask(is_nan<Tr>(x))));
    y = Tr::blend(y, inf, Tr::cmpeq(x, inf));
    y = Tr::blend(y, nan, is_nan<Tr>(x));
    return y;
}

// sincos kernel: Cody-Waite reduction (exact for |x| <= trig_big via the
// PIO2_1 split), degree-15 sine / degree-16 cosine Taylor on [-pi/4, pi/4],
// quadrant swap. need_trig_fixup() flags out-of-range lanes for libm.
template <typename Tr> inline typename Tr::mask need_trig_fixup(typename Tr::vec x) noexcept
{
    using C = Consts<typename Tr::scalar>;
    return Tr::and_mask(is_finite<Tr>(x), Tr::cmplt(Tr::set1(C::trig_big()), Tr::abs(x)));
}
template <typename Tr> inline void sincos_vec(typename Tr::vec x, typename Tr::vec &s, typename Tr::vec &c) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    // Clamp the reduction input so k stays integral and k*PIO2_1 stays exact;
    // flagged lanes are recomputed scalarly by the driver.
    const typename Tr::vec xc = Tr::blend(x, Tr::setzero(), Tr::or_mask(not_finite<Tr>(x), need_trig_fixup<Tr>(x)));
    const typename Tr::vec k = Tr::nearbyint(Tr::mul(xc, Tr::set1(C::inv_pio2())));
    const typename Tr::vec r =
        Tr::sub(Tr::sub(xc, Tr::mul(k, Tr::set1(C::pio2_1()))), Tr::mul(k, Tr::set1(C::pio2_1t())));
    const typename Tr::vec r2 = Tr::mul(r, r);
    // sin(r) = r * P(r^2), degree-15 Taylor (tail < 5e-17 on [-pi/4, pi/4]).
    // P(u) = 1 - u/6 + u^2/120 - ... - u^7/15! evaluated innermost-first.
    typename Tr::vec sn = Tr::set1(S(-7.647163731819816e-13));  // -1/15!
    sn = Tr::fma(sn, r2, Tr::set1(S(1.6059043836821613e-10)));  // 1/13!
    sn = Tr::fma(sn, r2, Tr::set1(S(-2.505210838544172e-08)));  // -1/11!
    sn = Tr::fma(sn, r2, Tr::set1(S(2.7557319223985893e-06)));  // 1/9!
    sn = Tr::fma(sn, r2, Tr::set1(S(-1.984126984126984e-04)));  // -1/7!
    sn = Tr::fma(sn, r2, Tr::set1(S(8.333333333333333e-03)));   // 1/5!
    sn = Tr::fma(sn, r2, Tr::set1(S(-1.6666666666666666e-01))); // -1/3!
    sn = Tr::fma(sn, r2, Tr::set1(S(1.0)));
    const typename Tr::vec sinr = Tr::mul(sn, r);
    typename Tr::vec cs = Tr::set1(S(4.779477332387385e-14));
    cs = Tr::fma(cs, r2, Tr::set1(S(-1.1470745597729725e-11)));
    cs = Tr::fma(cs, r2, Tr::set1(S(2.08767569878681e-09)));
    cs = Tr::fma(cs, r2, Tr::set1(S(-2.7557319223985893e-07)));
    cs = Tr::fma(cs, r2, Tr::set1(S(2.48015873015873e-05)));
    cs = Tr::fma(cs, r2, Tr::set1(S(-1.388888888888889e-03)));
    cs = Tr::fma(cs, r2, Tr::set1(S(4.1666666666666664e-02)));
    cs = Tr::fma(cs, r2, Tr::set1(S(-5.0e-1)));
    cs = Tr::fma(cs, r2, Tr::set1(S{1}));
    const typename Tr::vec cosr = cs;
    const typename Tr::vec q = Tr::sub(k, Tr::mul(Tr::floor(Tr::mul(k, Tr::set1(S{0.25}))), Tr::set1(S{4})));
    const typename Tr::mask q1 = Tr::cmpeq(q, Tr::set1(S{1}));
    const typename Tr::mask q2 = Tr::cmpeq(q, Tr::set1(S{2}));
    const typename Tr::mask q3 = Tr::cmpeq(q, Tr::set1(S{3}));
    s = Tr::blend(sinr, cosr, q1);
    s = Tr::blend(s, Tr::neg(sinr), q2);
    s = Tr::blend(s, Tr::neg(cosr), q3);
    c = Tr::blend(cosr, Tr::neg(sinr), q1);
    c = Tr::blend(c, Tr::neg(cosr), q2);
    c = Tr::blend(c, sinr, q3);
    // sin/cos of inf or NaN is NaN.
    const typename Tr::vec nan = Tr::set1(std::numeric_limits<S>::quiet_NaN());
    s = Tr::blend(s, nan, not_finite<Tr>(x));
    c = Tr::blend(c, nan, not_finite<Tr>(x));
}

// atan kernel: fold |x| into [0,1] by inversion, two half-angle steps
// (a2 <= tan(pi/16) ~= 0.199), degree-19 odd Taylor (tail < 2e-16).
template <typename Tr> inline typename Tr::vec atan_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec a = Tr::abs(x);
    const typename Tr::mask inv = Tr::cmplt(Tr::set1(S{1}), a);
    typename Tr::vec as = Tr::blend(a, Tr::div(Tr::set1(S{1}), a), inv);
    as = Tr::blend(as, Tr::setzero(), is_nan<Tr>(x));
    const typename Tr::vec a1 = Tr::div(as, Tr::add(Tr::set1(S{1}), Tr::sqrt(Tr::fma(as, as, Tr::set1(S{1})))));
    const typename Tr::vec a2 = Tr::div(a1, Tr::add(Tr::set1(S{1}), Tr::sqrt(Tr::fma(a1, a1, Tr::set1(S{1})))));
    const typename Tr::vec w = Tr::mul(a2, a2);
    typename Tr::vec p = Tr::set1(S(-1.0 / 19.0));
    p = Tr::fma(p, w, Tr::set1(S(1.0 / 17.0)));
    p = Tr::fma(p, w, Tr::set1(S(-1.0 / 15.0)));
    p = Tr::fma(p, w, Tr::set1(S(1.0 / 13.0)));
    p = Tr::fma(p, w, Tr::set1(S(-1.0 / 11.0)));
    p = Tr::fma(p, w, Tr::set1(S(1.0 / 9.0)));
    p = Tr::fma(p, w, Tr::set1(S(-1.0 / 7.0)));
    p = Tr::fma(p, w, Tr::set1(S(1.0 / 5.0)));
    p = Tr::fma(p, w, Tr::set1(S(-1.0 / 3.0)));
    p = Tr::fma(p, w, Tr::set1(S{1}));
    typename Tr::vec r = Tr::mul(Tr::mul(a2, p), Tr::set1(S{4}));
    r = Tr::blend(r, Tr::sub(Tr::set1(C::pi_2()), r), inv);
    r = Tr::copysign_to(r, x);
    // Below 2^-26 (2^-12 float) atan(x) rounds to x; this also keeps
    // subnormals exact, which the half-angle steps would flush to zero.
    r = Tr::blend(r, x, Tr::cmple(Tr::abs(x), Tr::set1(C::tiny())));
    r = Tr::blend(r, Tr::set1(std::numeric_limits<S>::quiet_NaN()), is_nan<Tr>(x));
    return r;
}

// asin kernel via atan: |x| <= 0.5 uses atan(x/sqrt(1-x^2)), larger arguments
// use pi/2 - 2*asin(sqrt((1-x)/2)) so the inner argument stays <= 0.5.
template <typename Tr> inline typename Tr::vec asin_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec ax = Tr::abs(x);
    const typename Tr::vec direct = atan_vec<Tr>(Tr::div(x, Tr::sqrt(Tr::sub(Tr::set1(S{1}), Tr::mul(x, x)))));
    const typename Tr::vec z = Tr::sqrt(Tr::mul(Tr::sub(Tr::set1(S{1}), ax), Tr::set1(S{0.5})));
    const typename Tr::vec inner = atan_vec<Tr>(Tr::div(z, Tr::sqrt(Tr::sub(Tr::set1(S{1}), Tr::mul(z, z)))));
    typename Tr::vec rec = Tr::sub(Tr::set1(C::pi_2()), Tr::mul(inner, Tr::set1(S{2})));
    rec = Tr::copysign_to(rec, x);
    typename Tr::vec r = Tr::blend(direct, rec, Tr::cmplt(Tr::set1(S{0.5}), ax));
    r = Tr::blend(r, Tr::copysign_to(Tr::set1(C::pi_2()), x), Tr::cmpeq(ax, Tr::set1(S{1})));
    r = Tr::blend(r, Tr::set1(std::numeric_limits<S>::quiet_NaN()),
                  Tr::or_mask(Tr::cmplt(Tr::set1(S{1}), ax), is_nan<Tr>(x)));
    return r;
}

// expm1 kernel: (u-1)*x/log(u) with u = exp(x) is accurate down to x = 0
// (avoids the catastrophic cancellation of exp(x)-1); u == 1 returns x.
template <typename Tr> inline typename Tr::vec expm1_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec u = exp_vec<Tr>(x);
    const typename Tr::vec formula = Tr::div(Tr::mul(Tr::sub(u, Tr::set1(S{1})), x), log_vec<Tr>(u));
    typename Tr::vec r = Tr::blend(formula, x, Tr::cmpeq(u, Tr::set1(S{1})));
    r = Tr::blend(r, Tr::set1(std::numeric_limits<S>::infinity()), Tr::cmplt(Tr::set1(C::exp_hi()), x));
    r = Tr::blend(r, Tr::set1(S(-1)), Tr::cmplt(x, Tr::set1(C::exp_lo())));
    return r;
}

// log1p kernel: Kahan's trick log1p(x) = log(t) - ((t-1)-x)/t with t = 1+x
// keeps full accuracy for x near 0, where log(1+x) would lose everything.
template <typename Tr> inline typename Tr::vec log1p_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    const typename Tr::vec inf = Tr::set1(std::numeric_limits<S>::infinity());
    const typename Tr::vec t = Tr::add(Tr::set1(S{1}), x);
    const typename Tr::vec z = log_vec<Tr>(t);
    const typename Tr::vec corr = Tr::div(Tr::sub(Tr::sub(t, Tr::set1(S{1})), x), t);
    // For t <= 1/2 the input t = 1+x is exact (Sterbenz), so log(t) alone is
    // accurate; the correction would only divide (t-1)'s rounding noise by a
    // tiny t. x = -1 and x < -1 keep their -inf/NaN from log(t) this way too.
    typename Tr::vec r = Tr::blend(Tr::sub(z, corr), z, Tr::cmple(t, Tr::set1(S{0.5})));
    r = Tr::blend(r, inf, Tr::cmpeq(x, inf));
    r = Tr::blend(r, Tr::neg(inf), Tr::cmpeq(x, Tr::set1(S{-1}))); // t = 0 cancels above
    r = Tr::copysign_to(r, x);                                     // restores the sign of zero for x = -0
    return r;
}
template <typename Tr> inline typename Tr::mask need_log1p_fixup(typename Tr::vec x) noexcept
{
    using C = Consts<typename Tr::scalar>;
    const typename Tr::vec t = Tr::add(Tr::set1(typename Tr::scalar{1}), x);
    // t subnormal positive (x just above -1): delegate to scalar log1p.
    return Tr::and_mask(Tr::and_mask(Tr::cmplt(Tr::setzero(), t), Tr::cmplt(t, Tr::set1(C::min_normal()))),
                        is_finite<Tr>(x));
}

// Hyperbolic kernels from exp/expm1 (no cancellation: the small-argument
// sinh uses t*(t+2)/(2u) with t = expm1, exact at 0).
template <typename Tr> inline typename Tr::vec sinh_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    const typename Tr::vec ax = Tr::abs(x);
    const typename Tr::vec ex = exp_vec<Tr>(ax);
    const typename Tr::vec t = expm1_vec<Tr>(ax);
    const typename Tr::vec u = Tr::add(t, Tr::set1(S{1}));
    const typename Tr::vec small = Tr::div(Tr::mul(t, Tr::add(t, Tr::set1(S{2}))), Tr::mul(u, Tr::set1(S{2})));
    const typename Tr::vec big = Tr::div(Tr::sub(ex, Tr::div(Tr::set1(S{1}), ex)), Tr::set1(S{2}));
    typename Tr::vec r = Tr::blend(big, small, Tr::cmplt(ax, Tr::set1(S{1})));
    return Tr::copysign_to(r, x);
}
template <typename Tr> inline typename Tr::vec cosh_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    const typename Tr::vec ex = exp_vec<Tr>(Tr::abs(x));
    return Tr::div(Tr::add(ex, Tr::div(Tr::set1(S{1}), ex)), Tr::set1(S{2}));
}
template <typename Tr> inline typename Tr::vec tanh_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    const typename Tr::vec ax = Tr::abs(x);
    const typename Tr::vec t = expm1_vec<Tr>(Tr::mul(ax, Tr::set1(S{2})));
    typename Tr::vec r = Tr::div(t, Tr::add(t, Tr::set1(S{2})));
    r = Tr::blend(r, Tr::set1(S{1}), Tr::cmplt(Tr::set1(S{20}), ax));
    return Tr::copysign_to(r, x);
}

// Inverse hyperbolic kernels from log/sqrt. asinh/acosh switch to
// ln2+log(x) above asinh_big so x*x cannot overflow to inf.
template <typename Tr> inline typename Tr::vec asinh_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec a = Tr::abs(x);
    typename Tr::vec r = log_vec<Tr>(Tr::add(a, Tr::sqrt(Tr::fma(a, a, Tr::set1(S{1})))));
    const typename Tr::vec big = Tr::add(Tr::set1(C::ln2()), log_vec<Tr>(a));
    r = Tr::blend(r, big, Tr::cmplt(Tr::set1(C::asinh_big()), a));
    r = Tr::blend(r, x, Tr::cmple(a, Tr::set1(C::tiny()))); // asinh(x) rounds to x here
    return Tr::copysign_to(r, x);
}
template <typename Tr> inline typename Tr::vec acosh_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec general =
        log_vec<Tr>(Tr::add(x, Tr::mul(Tr::sqrt(Tr::sub(x, Tr::set1(S{1}))), Tr::sqrt(Tr::add(x, Tr::set1(S{1}))))));
    const typename Tr::vec big = Tr::add(Tr::set1(C::ln2()), log_vec<Tr>(x));
    typename Tr::vec r = Tr::blend(general, big, Tr::cmplt(Tr::set1(C::asinh_big()), x));
    r = Tr::blend(r, Tr::set1(std::numeric_limits<S>::quiet_NaN()), Tr::cmplt(x, Tr::set1(S{1})));
    return r;
}
template <typename Tr> inline typename Tr::vec atanh_vec(typename Tr::vec x) noexcept
{
    // Two formulations: log1p(2x/(1-x)) is tight near 0 but its intermediate
    // quotient rounding is amplified by 1/(1-|x|) near |x| = 1, where instead
    // (log1p(x)-log1p(-x))/2 is exact-safe (1+/-x are exact by Sterbenz).
    using S = typename Tr::scalar;
    const typename Tr::vec ax = Tr::abs(x);
    const typename Tr::vec small =
        Tr::mul(log1p_vec<Tr>(Tr::div(Tr::mul(Tr::set1(S{2}), x), Tr::sub(Tr::set1(S{1}), x))), Tr::set1(S{0.5}));
    const typename Tr::vec big = Tr::mul(Tr::sub(log1p_vec<Tr>(x), log1p_vec<Tr>(Tr::neg(x))), Tr::set1(S{0.5}));
    typename Tr::vec r = Tr::blend(big, small, Tr::cmple(ax, Tr::set1(S{0.5})));
    r = Tr::blend(r, Tr::copysign_to(Tr::set1(std::numeric_limits<S>::infinity()), x), Tr::cmpeq(ax, Tr::set1(S{1})));
    r = Tr::blend(r, Tr::set1(std::numeric_limits<S>::quiet_NaN()), Tr::cmplt(Tr::set1(S{1}), ax));
    return Tr::copysign_to(r, x);
}

// cbrt kernel: exp(log(x)/3) followed by one Newton step, which squares the
// ~2 ULP seed error to well below 1 ULP. Zero/inf/NaN never enter Newton.
template <typename Tr> inline typename Tr::vec cbrt_vec(typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    const typename Tr::vec ax = Tr::abs(x);
    const typename Tr::vec seed = exp_vec<Tr>(Tr::div(log_vec<Tr>(ax), Tr::set1(S{3})));
    const typename Tr::vec step =
        Tr::div(Tr::add(Tr::mul(seed, Tr::set1(S{2})), Tr::div(ax, Tr::mul(seed, seed))), Tr::set1(S{3}));
    const typename Tr::mask ok = Tr::and_mask(Tr::cmplt(Tr::setzero(), ax), is_finite<Tr>(ax));
    const typename Tr::vec c = Tr::blend(seed, step, ok);
    return Tr::copysign_to(c, x);
}
template <typename Tr> inline typename Tr::mask need_cbrt_fixup(typename Tr::vec x) noexcept
{
    using C = Consts<typename Tr::scalar>;
    const typename Tr::vec ax = Tr::abs(x);
    return Tr::and_mask(Tr::and_mask(Tr::cmplt(Tr::setzero(), ax), Tr::cmplt(ax, Tr::set1(C::min_normal()))),
                        is_finite<Tr>(x));
}

// atan2 kernel: general path atan(y/x) plus quadrant shift, with explicit
// IEEE masks for signed zeros and infinities (matches std::atan2 exactly).
template <typename Tr> inline typename Tr::vec atan2_vec(typename Tr::vec y, typename Tr::vec x) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec nan = Tr::set1(std::numeric_limits<S>::quiet_NaN());
    const typename Tr::vec inf = Tr::set1(std::numeric_limits<S>::infinity());
    const typename Tr::vec pi = Tr::set1(C::pi());
    const typename Tr::vec pi_2 = Tr::set1(C::pi_2());
    typename Tr::vec r = atan_vec<Tr>(Tr::div(y, x));
    r = Tr::blend(r, Tr::add(r, Tr::copysign_to(pi, y)), Tr::cmplt(x, Tr::setzero()));
    // x == +/-0: y == 0 keeps the zero semantics (+/-0 vs +/-pi), else +/-pi/2.
    const typename Tr::mask xzero = Tr::cmpeq(x, Tr::setzero());
    const typename Tr::mask yzero = Tr::cmpeq(y, Tr::setzero());
    const typename Tr::vec neg_zero_case = Tr::blend(Tr::copysign_to(pi_2, y), Tr::copysign_to(pi, y), yzero);
    const typename Tr::vec pos_zero_case =
        Tr::blend(Tr::copysign_to(pi_2, y), Tr::copysign_to(Tr::setzero(), y), yzero);
    // Signed-zero test via 1/x: 1/+0 = +inf, 1/-0 = -inf.
    const typename Tr::mask xnegzero = Tr::and_mask(xzero, Tr::cmpeq(Tr::div(Tr::set1(S{1}), x), Tr::neg(inf)));
    r = Tr::blend(r, Tr::blend(pos_zero_case, neg_zero_case, xnegzero), xzero);
    // x == +/-inf.
    const typename Tr::mask xposinf = Tr::cmpeq(x, inf);
    const typename Tr::mask xneginf = Tr::cmpeq(x, Tr::neg(inf));
    const typename Tr::vec pinf_case =
        Tr::blend(Tr::copysign_to(Tr::setzero(), y),
                  Tr::blend(Tr::set1(C::pi_2() / 2), Tr::neg(Tr::set1(C::pi_2() / 2)), Tr::cmplt(y, Tr::setzero())),
                  Tr::or_mask(Tr::cmpeq(y, inf), Tr::cmpeq(y, Tr::neg(inf))));
    const typename Tr::vec ninf_case =
        Tr::blend(Tr::copysign_to(pi, y),
                  Tr::blend(Tr::mul(Tr::set1(S{3}), Tr::set1(C::pi_2() / 2)),
                            Tr::neg(Tr::mul(Tr::set1(S{3}), Tr::set1(C::pi_2() / 2))), Tr::cmplt(y, Tr::setzero())),
                  Tr::or_mask(Tr::cmpeq(y, inf), Tr::cmpeq(y, Tr::neg(inf))));
    // |y| == inf with finite x is +/-pi/2; x == +/-inf cases above already
    // hold the exact quadrant values, so nothing more is needed for them.
    r = Tr::blend(r, pinf_case, xposinf);
    r = Tr::blend(r, ninf_case, xneginf);
    r = Tr::blend(r, Tr::copysign_to(pi_2, y),
                  Tr::and_mask(Tr::or_mask(Tr::cmpeq(y, inf), Tr::cmpeq(y, Tr::neg(inf))), is_finite<Tr>(x)));
    r = Tr::blend(r, nan, Tr::or_mask(is_nan<Tr>(x), is_nan<Tr>(y)));
    return r;
}

// hypot kernel: max*sqrt(1+(min/max)^2) cannot overflow; max == 0 gives 0,
// either inf gives +inf (even with a NaN other side, per IEEE).
template <typename Tr> inline typename Tr::vec hypot_vec(typename Tr::vec a, typename Tr::vec b) noexcept
{
    using S = typename Tr::scalar;
    const typename Tr::vec ax = Tr::abs(a);
    const typename Tr::vec ay = Tr::abs(b);
    const typename Tr::vec m = Tr::max(ax, ay);
    const typename Tr::vec n = Tr::min(ax, ay);
    const typename Tr::vec ratio = Tr::div(n, m);
    typename Tr::vec r = Tr::mul(m, Tr::sqrt(Tr::fma(ratio, ratio, Tr::set1(S{1}))));
    r = Tr::blend(r, Tr::setzero(), Tr::cmpeq(m, Tr::setzero()));
    r = Tr::blend(r, Tr::set1(std::numeric_limits<S>::infinity()), Tr::or_mask(is_inf<Tr>(a), is_inf<Tr>(b)));
    const typename Tr::mask nan_no_inf = Tr::and_mask(Tr::or_mask(is_nan<Tr>(a), is_nan<Tr>(b)),
                                                      Tr::not_mask(Tr::or_mask(is_inf<Tr>(a), is_inf<Tr>(b))));
    r = Tr::blend(r, Tr::set1(std::numeric_limits<S>::quiet_NaN()), nan_no_inf);
    return r;
}

// pow kernel fast path: exp2(b*log2(a)) for positive finite bases; every
// other lane (non-positive base, 0^0, non-finite inputs, ...) is flagged for
// the scalar std::pow fallback so edges match libm bit for bit.
// The product b*log2(a) is computed in double-double (exact via FMA, Dekker
// otherwise): its plain rounding would be amplified by exp into ~|s| ULP.
template <typename Tr> inline typename Tr::vec pow_vec(typename Tr::vec a, typename Tr::vec b) noexcept
{
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec L = Tr::mul(log_vec<Tr>(a), Tr::set1(C::inv_ln2()));
    typename Tr::vec ph = Tr::setzero();
    typename Tr::vec pl = Tr::setzero();
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
    ph = Tr::mul(b, L);
    pl = Tr::fma(b, L, Tr::neg(ph)); // exact error term (true FMA only)
#else
    // Veltkamp split (Dekker): halves with disjoint mantissas.
    const typename Tr::vec vc = Tr::set1(C::veltkamp());
    const typename Tr::vec tb = Tr::mul(b, vc);
    const typename Tr::vec bh = Tr::sub(tb, Tr::sub(tb, b));
    const typename Tr::vec bl = Tr::sub(b, bh);
    const typename Tr::vec tl = Tr::mul(L, vc);
    const typename Tr::vec Lh = Tr::sub(tl, Tr::sub(tl, L));
    const typename Tr::vec Ll = Tr::sub(L, Lh);
    ph = Tr::mul(b, L);
    pl = Tr::add(Tr::add(Tr::sub(Tr::mul(bh, Lh), ph), Tr::mul(bh, Ll)), Tr::add(Tr::mul(bl, Lh), Tr::mul(bl, Ll)));
#endif
    const typename Tr::vec s = Tr::add(ph, pl); // exponent sum for the blends
    const typename Tr::vec k = Tr::nearbyint(s);
    const typename Tr::vec fr = Tr::add(Tr::sub(ph, k), pl); // exact head, then tail
    const typename Tr::vec ef = exp_taylor<Tr>(Tr::mul(fr, Tr::set1(C::ln2())));
    typename Tr::vec ks = Tr::blend(k, Tr::setzero(), not_finite<Tr>(s));
    ks = Tr::max(Tr::min(ks, Tr::set1(S{2048})), Tr::set1(S{-2048}));
    const typename Tr::vec kh = Tr::floor(Tr::mul(ks, Tr::set1(S{0.5})));
    const typename Tr::vec kl = Tr::sub(ks, kh);
    typename Tr::vec y = Tr::mul(Tr::mul(ef, Tr::scale2k(kh)), Tr::scale2k(kl));
    y = Tr::blend(y, Tr::set1(std::numeric_limits<S>::infinity()), Tr::cmplt(Tr::set1(C::exp_hi()), s));
    y = Tr::blend(y, Tr::setzero(), Tr::cmplt(s, Tr::set1(C::exp_lo())));
    // Exact small-exponent paths: these single-rounding forms are
    // bit-identical to correctly-rounded libm pow (and much faster).
    const typename Tr::mask m0 = Tr::cmpeq(b, Tr::setzero());
    const typename Tr::mask m1 = Tr::cmpeq(b, Tr::set1(S{1}));
    const typename Tr::mask m2 = Tr::cmpeq(b, Tr::set1(S{2}));
    const typename Tr::mask mn1 = Tr::cmpeq(b, Tr::set1(S{-1}));
    const typename Tr::mask mh = Tr::cmpeq(b, Tr::set1(S{0.5}));
    const typename Tr::mask mnh = Tr::cmpeq(b, Tr::set1(S{-0.5}));
    const typename Tr::mask m_small =
        Tr::or_mask(Tr::or_mask(m0, m1), Tr::or_mask(Tr::or_mask(m2, mn1), Tr::or_mask(mh, mnh)));
    if (Tr::any(m_small))
    {
        const typename Tr::vec a2 = Tr::mul(a, a);
        const typename Tr::vec sq = Tr::sqrt(a);
        typename Tr::vec ys = y;
        ys = Tr::blend(ys, Tr::set1(S{1}), m0);
        ys = Tr::blend(ys, a, m1);
        ys = Tr::blend(ys, a2, m2);
        ys = Tr::blend(ys, Tr::div(Tr::set1(S{1}), a), mn1);
        ys = Tr::blend(ys, sq, mh);
        ys = Tr::blend(ys, Tr::div(Tr::set1(S{1}), sq), mnh);
        y = Tr::blend(y, ys, m_small);
    }
    return y;
}
template <typename Tr> inline typename Tr::mask need_pow_fixup(typename Tr::vec a, typename Tr::vec b) noexcept
{
    return Tr::not_mask(Tr::and_mask(Tr::and_mask(Tr::cmplt(Tr::setzero(), a), is_finite<Tr>(a)), is_finite<Tr>(b)));
}

// Operation descriptors: vector kernel + fallback mask + scalar reference.
// The driver evaluates the whole block with SIMD, then recomputes only the
// flagged lanes scalarly (edge cases stay bit-compatible with libm).
template <typename Tr> inline typename Tr::mask no_fixup(typename Tr::vec) noexcept
{
    using S = typename Tr::scalar;
    return Tr::cmplt(Tr::set1(S{1}), Tr::set1(S{0})); // all-false
}

// Small composed kernels (defined before the op table: unqualified calls in
// template definitions resolve at definition time under two-phase lookup).
template <typename Tr> inline typename Tr::vec sin_vec(typename Tr::vec x) noexcept
{
    typename Tr::vec s = Tr::setzero();
    typename Tr::vec c = Tr::setzero();
    sincos_vec<Tr>(x, s, c);
    return s;
}
template <typename Tr> inline typename Tr::vec cos_vec(typename Tr::vec x) noexcept
{
    typename Tr::vec s = Tr::setzero();
    typename Tr::vec c = Tr::setzero();
    sincos_vec<Tr>(x, s, c);
    return c;
}
template <typename Tr> inline typename Tr::vec tan_vec(typename Tr::vec x) noexcept
{
    typename Tr::vec s = Tr::setzero();
    typename Tr::vec c = Tr::setzero();
    sincos_vec<Tr>(x, s, c);
    return Tr::div(s, c);
}
template <typename Tr> inline typename Tr::vec acos_vec(typename Tr::vec x) noexcept
{
    // Direct form: acos(x) = atan(sqrt(1-x^2)/|x|) folded by sign. Unlike
    // pi/2 - asin(x) this never subtracts nearly-equal numbers near |x| = 1.
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec ax = Tr::abs(x);
    const typename Tr::vec s = Tr::sqrt(Tr::mul(Tr::sub(Tr::set1(S{1}), ax), Tr::add(Tr::set1(S{1}), ax)));
    const typename Tr::vec a = atan_vec<Tr>(Tr::div(s, ax));
    return Tr::blend(a, Tr::sub(Tr::set1(C::pi()), a), Tr::cmplt(x, Tr::setzero()));
}
template <typename Tr> inline typename Tr::vec exp2_vec(typename Tr::vec x) noexcept
{
    // Dedicated reduction: k = round(x) is exact, so r = x - k is exact
    // (Sterbenz) and no absolute error gets amplified by the exponential.
    // (Routing through exp(x*ln2) would cost ~x/2 ULP.) Thresholds bracket
    // the exact overflow/underflow points: 2^1024 = inf, 2^-1075 = 0.
    using S = typename Tr::scalar;
    using C = Consts<S>;
    const typename Tr::vec k = Tr::nearbyint(x);
    const typename Tr::vec r = Tr::sub(x, k);
    const typename Tr::vec t = Tr::mul(r, Tr::set1(C::ln2()));
    const typename Tr::vec p = exp_taylor<Tr>(t);
    typename Tr::vec ks = Tr::blend(k, Tr::setzero(), not_finite<Tr>(x));
    ks = Tr::max(Tr::min(ks, Tr::set1(S{2048})), Tr::set1(S{-2048}));
    const typename Tr::vec kh = Tr::floor(Tr::mul(ks, Tr::set1(S{0.5})));
    const typename Tr::vec kl = Tr::sub(ks, kh);
    typename Tr::vec y = Tr::mul(Tr::mul(p, Tr::scale2k(kh)), Tr::scale2k(kl));
    y = Tr::blend(y, Tr::set1(std::numeric_limits<S>::infinity()), Tr::cmplt(Tr::set1(C::exp2_hi()), x));
    y = Tr::blend(y, Tr::setzero(), Tr::cmplt(x, Tr::set1(C::exp2_lo())));
    return y;
}
template <typename Tr> inline typename Tr::vec log10_vec(typename Tr::vec x) noexcept
{
    using C = Consts<typename Tr::scalar>;
    return Tr::mul(log_vec<Tr>(x), Tr::set1(C::inv_ln10()));
}
template <typename Tr> inline typename Tr::vec log2_vec(typename Tr::vec x) noexcept
{
    using C = Consts<typename Tr::scalar>;
    return Tr::mul(log_vec<Tr>(x), Tr::set1(C::inv_ln2()));
}
template <typename Tr> inline typename Tr::vec sqrt_vec(typename Tr::vec x) noexcept
{
    return Tr::sqrt(x); // native instruction: correctly rounded, IEEE exact
}
template <typename Tr> inline typename Tr::vec floor_vec(typename Tr::vec x) noexcept
{
    return Tr::floor(x);
}
template <typename Tr> inline typename Tr::vec ceil_vec(typename Tr::vec x) noexcept
{
    return Tr::ceil(x);
}
template <typename Tr> inline typename Tr::vec trunc_vec(typename Tr::vec x) noexcept
{
    return Tr::trunc(x);
}
template <typename Tr> inline typename Tr::vec rint_vec(typename Tr::vec x) noexcept
{
    return Tr::nearbyint(x);
}
template <typename Tr> inline typename Tr::vec fabs_vec(typename Tr::vec x) noexcept
{
    return Tr::abs(x);
}
template <typename Tr> inline typename Tr::mask need_log_fixup(typename Tr::vec x) noexcept
{
    using C = Consts<typename Tr::scalar>;
    return Tr::and_mask(Tr::and_mask(Tr::cmplt(Tr::setzero(), x),
                                     Tr::cmplt(x, Tr::set1(std::numeric_limits<typename Tr::scalar>::infinity()))),
                        Tr::cmplt(x, Tr::set1(C::min_normal())));
}

#define NP_VECMATH_UNARY_OP(Name, Kernel, Fixup, ScalarFn)                                                             \
    struct Op##Name                                                                                                    \
    {                                                                                                                  \
        template <typename Tr> static inline typename Tr::vec vec(typename Tr::vec x) noexcept                         \
        {                                                                                                              \
            return Kernel<Tr>(x);                                                                                      \
        }                                                                                                              \
        template <typename Tr> static inline typename Tr::mask need_fixup(typename Tr::vec x) noexcept                 \
        {                                                                                                              \
            return Fixup<Tr>(x);                                                                                       \
        }                                                                                                              \
        template <typename S> static inline S scalar(S x) noexcept                                                     \
        {                                                                                                              \
            return ScalarFn(x);                                                                                        \
        }                                                                                                              \
    };

NP_VECMATH_UNARY_OP(Exp, exp_vec, no_fixup, std::exp)
NP_VECMATH_UNARY_OP(Expm1, expm1_vec, no_fixup, std::expm1)
NP_VECMATH_UNARY_OP(Log, log_vec, need_log_fixup, std::log)
NP_VECMATH_UNARY_OP(Log1p, log1p_vec, need_log1p_fixup, std::log1p)
NP_VECMATH_UNARY_OP(Sin, sin_vec, need_trig_fixup, std::sin)
NP_VECMATH_UNARY_OP(Cos, cos_vec, need_trig_fixup, std::cos)
NP_VECMATH_UNARY_OP(Tan, tan_vec, need_trig_fixup, std::tan)
NP_VECMATH_UNARY_OP(Asin, asin_vec, no_fixup, std::asin)
NP_VECMATH_UNARY_OP(Acos, acos_vec, no_fixup, std::acos)
NP_VECMATH_UNARY_OP(Atan, atan_vec, no_fixup, std::atan)
NP_VECMATH_UNARY_OP(Sinh, sinh_vec, no_fixup, std::sinh)
NP_VECMATH_UNARY_OP(Cosh, cosh_vec, no_fixup, std::cosh)
NP_VECMATH_UNARY_OP(Tanh, tanh_vec, no_fixup, std::tanh)
NP_VECMATH_UNARY_OP(Asinh, asinh_vec, no_fixup, std::asinh)
NP_VECMATH_UNARY_OP(Acosh, acosh_vec, no_fixup, std::acosh)
NP_VECMATH_UNARY_OP(Atanh, atanh_vec, no_fixup, std::atanh)
NP_VECMATH_UNARY_OP(Sqrt, sqrt_vec, no_fixup, std::sqrt)
NP_VECMATH_UNARY_OP(Cbrt, cbrt_vec, need_cbrt_fixup, std::cbrt)
NP_VECMATH_UNARY_OP(Floor, floor_vec, no_fixup, std::floor)
NP_VECMATH_UNARY_OP(Ceil, ceil_vec, no_fixup, std::ceil)
NP_VECMATH_UNARY_OP(Trunc, trunc_vec, no_fixup, std::trunc)
NP_VECMATH_UNARY_OP(Rint, rint_vec, no_fixup, std::rint)
NP_VECMATH_UNARY_OP(Fabs, fabs_vec, no_fixup, std::fabs)
NP_VECMATH_UNARY_OP(Exp2, exp2_vec, no_fixup, std::exp2)
NP_VECMATH_UNARY_OP(Log10, log10_vec, need_log_fixup, std::log10)
NP_VECMATH_UNARY_OP(Log2, log2_vec, need_log_fixup, std::log2)

#undef NP_VECMATH_UNARY_OP

// Generic drivers: SIMD block loop with lane-granular scalar fallback, plus a
// scalar tail. Only Nutzungs of raw pointers: the caller guarantees storage.
template <typename Tr, typename Op>
inline void run_unary(const typename Tr::scalar *in, typename Tr::scalar *out, std::size_t n) noexcept
{
    constexpr std::size_t W = Tr::W;
    std::size_t i = 0;
    const std::size_t vend = n - (n % W);
    for (; i < vend; i += W)
    {
        const typename Tr::vec v = Tr::loadu(in + i);
        Tr::storeu(Op::template vec<Tr>(v), out + i);
        const typename Tr::mask f = Op::template need_fixup<Tr>(v);
        if (Tr::any(f))
        {
            std::uint64_t lanes[W];
            Tr::mask_lanes(f, lanes);
            for (std::size_t j = 0; j < W; ++j)
            {
                if (lanes[j] != 0u)
                {
                    out[i + j] = Op::scalar(in[i + j]);
                }
            }
        }
    }
    for (; i < n; ++i)
    {
        out[i] = Op::scalar(in[i]);
    }
}

template <typename Tr, typename Op>
inline void run_binary(const typename Tr::scalar *a, const typename Tr::scalar *b, typename Tr::scalar *out,
                       std::size_t n) noexcept
{
    constexpr std::size_t W = Tr::W;
    std::size_t i = 0;
    const std::size_t vend = n - (n % W);
    for (; i < vend; i += W)
    {
        const typename Tr::vec va = Tr::loadu(a + i);
        const typename Tr::vec vb = Tr::loadu(b + i);
        Tr::storeu(Op::template vec<Tr>(va, vb), out + i);
        const typename Tr::mask f = Op::template need_fixup<Tr>(va, vb);
        if (Tr::any(f))
        {
            std::uint64_t lanes[W];
            Tr::mask_lanes(f, lanes);
            for (std::size_t j = 0; j < W; ++j)
            {
                if (lanes[j] != 0u)
                {
                    out[i + j] = Op::scalar(a[i + j], b[i + j]);
                }
            }
        }
    }
    for (; i < n; ++i)
    {
        out[i] = Op::scalar(a[i], b[i]);
    }
}

struct OpAtan2
{
    template <typename Tr> static inline typename Tr::vec vec(typename Tr::vec y, typename Tr::vec x) noexcept
    {
        return atan2_vec<Tr>(y, x);
    }
    template <typename Tr> static inline typename Tr::mask need_fixup(typename Tr::vec, typename Tr::vec) noexcept
    {
        return no_fixup<Tr>(Tr::setzero());
    }
    template <typename S> static inline S scalar(S y, S x) noexcept
    {
        return std::atan2(y, x);
    }
};

struct OpPow
{
    template <typename Tr> static inline typename Tr::vec vec(typename Tr::vec a, typename Tr::vec b) noexcept
    {
        return pow_vec<Tr>(a, b);
    }
    template <typename Tr> static inline typename Tr::mask need_fixup(typename Tr::vec a, typename Tr::vec b) noexcept
    {
        return need_pow_fixup<Tr>(a, b);
    }
    template <typename S> static inline S scalar(S a, S b) noexcept
    {
        return std::pow(a, b);
    }
};

struct OpHypot
{
    template <typename Tr> static inline typename Tr::vec vec(typename Tr::vec a, typename Tr::vec b) noexcept
    {
        return hypot_vec<Tr>(a, b);
    }
    template <typename Tr> static inline typename Tr::mask need_fixup(typename Tr::vec, typename Tr::vec) noexcept
    {
        return no_fixup<Tr>(Tr::setzero());
    }
    template <typename S> static inline S scalar(S a, S b) noexcept
    {
        return std::hypot(a, b);
    }
};

template <typename Tr>
inline void run_sincos(const typename Tr::scalar *in, typename Tr::scalar *out_s, typename Tr::scalar *out_c,
                       std::size_t n) noexcept
{
    constexpr std::size_t W = Tr::W;
    std::size_t i = 0;
    const std::size_t vend = n - (n % W);
    for (; i < vend; i += W)
    {
        const typename Tr::vec v = Tr::loadu(in + i);
        typename Tr::vec s = Tr::setzero();
        typename Tr::vec c = Tr::setzero();
        sincos_vec<Tr>(v, s, c);
        Tr::storeu(s, out_s + i);
        Tr::storeu(c, out_c + i);
        const typename Tr::mask f = need_trig_fixup<Tr>(v);
        if (Tr::any(f))
        {
            std::uint64_t lanes[W];
            Tr::mask_lanes(f, lanes);
            for (std::size_t j = 0; j < W; ++j)
            {
                if (lanes[j] != 0u)
                {
                    out_s[i + j] = std::sin(in[i + j]);
                    out_c[i + j] = std::cos(in[i + j]);
                }
            }
        }
    }
    for (; i < n; ++i)
    {
        out_s[i] = std::sin(in[i]);
        out_c[i] = std::cos(in[i]);
    }
}

} // namespace detail

// Public batch API. Preconditions: `in`/`out` point to at least `n`
// elements (binary: `a`, `b`, `out` each); n == 0 or null is a no-op.
// Span overloads process min(in.size(), out.size()) elements.
#define NP_VECMATH_BATCH_UNARY(Name, Op)                                                                               \
    template <typename T> inline void Name(const T *in, T *out, std::size_t n) noexcept                                \
    {                                                                                                                  \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "vecmath: float/double only");            \
        if (in == nullptr || out == nullptr || n == 0)                                                                 \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        detail::run_unary<typename detail::Traits<T>::type, detail::Op>(in, out, n);                                   \
    }                                                                                                                  \
    template <typename T> inline void Name(std::span<const T> in, std::span<T> out) noexcept                           \
    {                                                                                                                  \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "vecmath: float/double only");            \
        const std::size_t n = in.size() < out.size() ? in.size() : out.size();                                         \
        if (n != 0)                                                                                                    \
        {                                                                                                              \
            Name(in.data(), out.data(), n);                                                                            \
        }                                                                                                              \
    }

NP_VECMATH_BATCH_UNARY(exp, OpExp)
NP_VECMATH_BATCH_UNARY(expm1, OpExpm1)
NP_VECMATH_BATCH_UNARY(exp2, OpExp2)
NP_VECMATH_BATCH_UNARY(log, OpLog)
NP_VECMATH_BATCH_UNARY(log10, OpLog10)
NP_VECMATH_BATCH_UNARY(log2, OpLog2)
NP_VECMATH_BATCH_UNARY(log1p, OpLog1p)
NP_VECMATH_BATCH_UNARY(sin, OpSin)
NP_VECMATH_BATCH_UNARY(cos, OpCos)
NP_VECMATH_BATCH_UNARY(tan, OpTan)
NP_VECMATH_BATCH_UNARY(asin, OpAsin)
NP_VECMATH_BATCH_UNARY(acos, OpAcos)
NP_VECMATH_BATCH_UNARY(atan, OpAtan)
NP_VECMATH_BATCH_UNARY(sinh, OpSinh)
NP_VECMATH_BATCH_UNARY(cosh, OpCosh)
NP_VECMATH_BATCH_UNARY(tanh, OpTanh)
NP_VECMATH_BATCH_UNARY(asinh, OpAsinh)
NP_VECMATH_BATCH_UNARY(acosh, OpAcosh)
NP_VECMATH_BATCH_UNARY(atanh, OpAtanh)
NP_VECMATH_BATCH_UNARY(sqrt, OpSqrt)
NP_VECMATH_BATCH_UNARY(cbrt, OpCbrt)
NP_VECMATH_BATCH_UNARY(floor, OpFloor)
NP_VECMATH_BATCH_UNARY(ceil, OpCeil)
NP_VECMATH_BATCH_UNARY(trunc, OpTrunc)
NP_VECMATH_BATCH_UNARY(rint, OpRint)
NP_VECMATH_BATCH_UNARY(fabs, OpFabs)

#undef NP_VECMATH_BATCH_UNARY

#define NP_VECMATH_BATCH_BINARY(Name, Op)                                                                              \
    template <typename T> inline void Name(const T *a, const T *b, T *out, std::size_t n) noexcept                     \
    {                                                                                                                  \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "vecmath: float/double only");            \
        if (a == nullptr || b == nullptr || out == nullptr || n == 0)                                                  \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        detail::run_binary<typename detail::Traits<T>::type, detail::Op>(a, b, out, n);                                \
    }

NP_VECMATH_BATCH_BINARY(atan2, OpAtan2)
NP_VECMATH_BATCH_BINARY(pow, OpPow)
NP_VECMATH_BATCH_BINARY(hypot, OpHypot)

#undef NP_VECMATH_BATCH_BINARY

/** @brief Fused sin+cos in a single pass (one range reduction). */
template <typename T> inline void sincos(const T *in, T *out_s, T *out_c, std::size_t n) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "vecmath: float/double only");
    if (in == nullptr || out_s == nullptr || out_c == nullptr || n == 0)
    {
        return;
    }
    detail::run_sincos<typename detail::Traits<T>::type>(in, out_s, out_c, n);
}

} // namespace vecmath
} // namespace np

#endif // NP_VECMATH_HPP
