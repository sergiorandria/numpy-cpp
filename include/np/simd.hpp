/**
 * @file simd.hpp
 * @brief SIMD optimizations for array operations.
 *
 * Provides CPU-specific SIMD optimizations for common array operations.
 * Supports multiple instruction sets: SSE, AVX, AVX2, AVX-512, NEON.
 *
 * Reference: numpy-reference/reference/simd/
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_SIMD_HPP
#define NP_SIMD_HPP

#include "api_macros.hpp"
#include "powerful.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "pqc.hpp"

#ifdef NP_HAS_SLEEF
#include <sleef.h>
#endif

// Detect SIMD capabilities at compile time
// AVX512 implies AVX2 and AVX, AVX2 implies AVX – define all lower levels
// so kernels gated on NP_SIMD_AVX are visible in AVX2/AVX-512 builds.
#if defined(__AVX512F__)
#define NP_SIMD_AVX512
#define NP_SIMD_AVX2
#define NP_SIMD_AVX
#include <immintrin.h>
#elif defined(__AVX2__)
#define NP_SIMD_AVX2
#define NP_SIMD_AVX
#include <immintrin.h>
#elif defined(__AVX__)
#define NP_SIMD_AVX
#include <immintrin.h>
#elif defined(__SSE4_2__)
#define NP_SIMD_SSE42
#include <nmmintrin.h>
#elif defined(__SSE4_1__)
#define NP_SIMD_SSE41
#include <smmintrin.h>
#elif defined(__SSSE3__)
#define NP_SIMD_SSSE3
#include <tmmintrin.h>
#elif defined(__SSE3__)
#define NP_SIMD_SSE3
#include <pmmintrin.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define NP_SIMD_SSE2
#include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NP_SIMD_NEON
#include <arm_neon.h>
#endif

namespace np
{
namespace simd
{

// SIMD Trait Detection
/**
 * @brief Compile-time detection of available SIMD features.
 */
struct Features
{
    static constexpr bool has_sse2 =
#ifdef NP_SIMD_SSE2
        true;
#else
        false;
#endif

    static constexpr bool has_sse3 =
#ifdef NP_SIMD_SSE3
        true;
#else
        false;
#endif

    static constexpr bool has_ssse3 =
#ifdef NP_SIMD_SSSE3
        true;
#else
        false;
#endif

    static constexpr bool has_sse41 =
#ifdef NP_SIMD_SSE41
        true;
#else
        false;
#endif

    static constexpr bool has_sse42 =
#ifdef NP_SIMD_SSE42
        true;
#else
        false;
#endif

    static constexpr bool has_avx =
#ifdef NP_SIMD_AVX
        true;
#else
        false;
#endif

    static constexpr bool has_avx2 =
#ifdef NP_SIMD_AVX2
        true;
#else
        false;
#endif

    static constexpr bool has_avx512 =
#ifdef NP_SIMD_AVX512
        true;
#else
        false;
#endif

    static constexpr bool has_neon =
#ifdef NP_SIMD_NEON
        true;
#else
        false;
#endif

    static constexpr bool has_sleef =
#ifdef NP_HAS_SLEEF
        true;
#else
        false;
#endif
};

// Vectorized Operations
/**
 * @brief Vector width for different types and instruction sets.
 */
template <typename T> struct VectorWidth
{
#if defined(NP_SIMD_AVX512)
    static constexpr std::size_t value = 64 / sizeof(T); // 512 bits
#elif defined(NP_SIMD_AVX) || defined(NP_SIMD_AVX2)
    static constexpr std::size_t value = 32 / sizeof(T); // 256 bits
#elif defined(NP_SIMD_SSE2)
    static constexpr std::size_t value = 16 / sizeof(T); // 128 bits
#elif defined(NP_SIMD_NEON)
    static constexpr std::size_t value = 16 / sizeof(T); // 128 bits
#else
    static constexpr std::size_t value = 1; // Scalar fallback
#endif
};

// SSE2/SSE4.1 Optimizations (x86-64)
#ifdef NP_SIMD_SSE2

/**
 * @brief Vectorized addition for double arrays (SSE2).
 */
inline void add_f64_sse2(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 2);

    for (; i < vec_end; i += 2)
    {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vout = _mm_add_pd(va, vb);
        _mm_storeu_pd(out + i, vout);
    }

    // Handle remainder
    for (; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Vectorized addition for float arrays (SSE).
 */
inline void add_f32_sse(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vout = _mm_add_ps(va, vb);
        _mm_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Vectorized multiplication for double arrays (SSE2).
 */
inline void mul_f64_sse2(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 2);

    for (; i < vec_end; i += 2)
    {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vout = _mm_mul_pd(va, vb);
        _mm_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Vectorized multiplication for float arrays (SSE).
 */
inline void mul_f32_sse(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vout = _mm_mul_ps(va, vb);
        _mm_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Vectorized sum reduction for double arrays (SSE2).
 */
inline double sum_f64_sse2(const double *data, std::size_t n)
{
    __m128d vsum = _mm_setzero_pd();
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 2);

    for (; i < vec_end; i += 2)
    {
        __m128d v = _mm_loadu_pd(data + i);
        vsum = _mm_add_pd(vsum, v);
    }

    // Horizontal sum
    double temp[2];
    _mm_storeu_pd(temp, vsum);
    double sum = temp[0] + temp[1];

    // Handle remainder
    for (; i < n; ++i)
    {
        sum += data[i];
    }

    return sum;
}

/**
 * @brief Vectorized sum reduction for float arrays (SSE).
 */
inline float sum_f32_sse(const float *data, std::size_t n)
{
    __m128 vsum = _mm_setzero_ps();
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m128 v = _mm_loadu_ps(data + i);
        vsum = _mm_add_ps(vsum, v);
    }

    // Horizontal sum
    float temp[4];
    _mm_storeu_ps(temp, vsum);
    float sum = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < n; ++i)
    {
        sum += data[i];
    }

    return sum;
}

/**
 * @brief Vectorized subtraction for double arrays (SSE2).
 */
inline void sub_f64_sse2(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 2);

    for (; i < vec_end; i += 2)
    {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vout = _mm_sub_pd(va, vb);
        _mm_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] - b[i];
    }
}

/**
 * @brief Vectorized subtraction for float arrays (SSE).
 */
inline void sub_f32_sse(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vout = _mm_sub_ps(va, vb);
        _mm_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] - b[i];
    }
}

/**
 * @brief Vectorized division for double arrays (SSE2).
 */
inline void div_f64_sse2(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 2);

    for (; i < vec_end; i += 2)
    {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vout = _mm_div_pd(va, vb);
        _mm_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] / b[i];
    }
}

/**
 * @brief Vectorized division for float arrays (SSE).
 */
inline void div_f32_sse(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vout = _mm_div_ps(va, vb);
        _mm_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] / b[i];
    }
}

#endif // NP_SIMD_SSE2

// AVX/AVX2 Optimizations (x86-64)
#ifdef NP_SIMD_AVX

/**
 * @brief Vectorized addition for double arrays (AVX).
 */
inline void add_f64_avx(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vout = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Vectorized addition for float arrays (AVX).
 */
inline void add_f32_avx(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Vectorized multiplication for double arrays (AVX).
 */
inline void mul_f64_avx(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vout = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Vectorized multiplication for float arrays (AVX).
 */
inline void mul_f32_avx(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Vectorized sum reduction for double arrays (AVX).
 */
inline double sum_f64_avx(const double *data, std::size_t n)
{
    __m256d vsum = _mm256_setzero_pd();
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m256d v = _mm256_loadu_pd(data + i);
        vsum = _mm256_add_pd(vsum, v);
    }

    // Horizontal sum
    double temp[4];
    _mm256_storeu_pd(temp, vsum);
    double sum = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < n; ++i)
    {
        sum += data[i];
    }

    return sum;
}

/**
 * @brief Vectorized sum reduction for float arrays (AVX).
 */
inline float sum_f32_avx(const float *data, std::size_t n)
{
    __m256 vsum = _mm256_setzero_ps();
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        vsum = _mm256_add_ps(vsum, v);
    }

    // Horizontal sum
    float temp[8];
    _mm256_storeu_ps(temp, vsum);
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j)
    {
        sum += temp[j];
    }

    for (; i < n; ++i)
    {
        sum += data[i];
    }

    return sum;
}

/**
 * @brief Vectorized subtraction for double arrays (AVX).
 */
inline void sub_f64_avx(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vout = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] - b[i];
    }
}

/**
 * @brief Vectorized subtraction for float arrays (AVX).
 */
inline void sub_f32_avx(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] - b[i];
    }
}

/**
 * @brief Vectorized division for double arrays (AVX).
 */
inline void div_f64_avx(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vout = _mm256_div_pd(va, vb);
        _mm256_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] / b[i];
    }
}

/**
 * @brief Vectorized division for float arrays (AVX).
 */
inline void div_f32_avx(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] / b[i];
    }
}

#endif // NP_SIMD_AVX

// AVX-512 Optimizations (x86-64)
#ifdef NP_SIMD_AVX512

/**
 * @brief Vectorized addition for double arrays (AVX-512).
 */
inline void add_f64_avx512(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vout = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Vectorized addition for float arrays (AVX-512).
 */
inline void add_f32_avx512(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 16);

    for (; i < vec_end; i += 16)
    {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Vectorized sum reduction for double arrays (AVX-512).
 */
inline double sum_f64_avx512(const double *data, std::size_t n)
{
    __m512d vsum = _mm512_setzero_pd();
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m512d v = _mm512_loadu_pd(data + i);
        vsum = _mm512_add_pd(vsum, v);
    }

    double sum = _mm512_reduce_add_pd(vsum);

    for (; i < n; ++i)
    {
        sum += data[i];
    }

    return sum;
}

/**
 * @brief Vectorized sum reduction for float arrays (AVX-512).
 */
inline float sum_f32_avx512(const float *data, std::size_t n)
{
    __m512 vsum = _mm512_setzero_ps();
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 16);

    for (; i < vec_end; i += 16)
    {
        __m512 v = _mm512_loadu_ps(data + i);
        vsum = _mm512_add_ps(vsum, v);
    }

    float sum = _mm512_reduce_add_ps(vsum);

    for (; i < n; ++i)
    {
        sum += data[i];
    }

    return sum;
}

/**
 * @brief Vectorized subtraction for double arrays (AVX-512).
 */
inline void sub_f64_avx512(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vout = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] - b[i];
    }
}

/**
 * @brief Vectorized subtraction for float arrays (AVX-512).
 */
inline void sub_f32_avx512(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 16);

    for (; i < vec_end; i += 16)
    {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] - b[i];
    }
}

/**
 * @brief Vectorized multiplication for double arrays (AVX-512).
 */
inline void mul_f64_avx512(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vout = _mm512_mul_pd(va, vb);
        _mm512_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Vectorized multiplication for float arrays (AVX-512).
 */
inline void mul_f32_avx512(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 16);

    for (; i < vec_end; i += 16)
    {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Vectorized division for double arrays (AVX-512).
 */
inline void div_f64_avx512(const double *a, const double *b, double *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8)
    {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vout = _mm512_div_pd(va, vb);
        _mm512_storeu_pd(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] / b[i];
    }
}

/**
 * @brief Vectorized division for float arrays (AVX-512).
 */
inline void div_f32_avx512(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 16);

    for (; i < vec_end; i += 16)
    {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_div_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] / b[i];
    }
}

#endif // NP_SIMD_AVX512

// ARM NEON Optimizations
#ifdef NP_SIMD_NEON

/**
 * @brief Vectorized addition for float arrays (NEON).
 */
inline void add_f32_neon(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vout = vaddq_f32(va, vb);
        vst1q_f32(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Vectorized multiplication for float arrays (NEON).
 */
inline void mul_f32_neon(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vout = vmulq_f32(va, vb);
        vst1q_f32(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Vectorized sum reduction for float arrays (NEON).
 */
inline float sum_f32_neon(const float *data, std::size_t n)
{
    float32x4_t vsum = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vsum = vaddq_f32(vsum, v);
    }

    // Horizontal sum
    float32x2_t vsum_low = vget_low_f32(vsum);
    float32x2_t vsum_high = vget_high_f32(vsum);
    float32x2_t vsum_pair = vadd_f32(vsum_low, vsum_high);
    float sum = vget_lane_f32(vsum_pair, 0) + vget_lane_f32(vsum_pair, 1);

    for (; i < n; ++i)
    {
        sum += data[i];
    }

    return sum;
}

/**
 * @brief Vectorized subtraction for float arrays (NEON).
 */
inline void sub_f32_neon(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vout = vsubq_f32(va, vb);
        vst1q_f32(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] - b[i];
    }
}

/**
 * @brief Vectorized division for float arrays (NEON).
 */
inline void div_f32_neon(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vout = vdivq_f32(va, vb);
        vst1q_f32(out + i, vout);
    }

    for (; i < n; ++i)
    {
        out[i] = a[i] / b[i];
    }
}

#endif // NP_SIMD_NEON

// WASM SIMD128 (wasm_simd128.h) – 128-bit baseline
#ifdef NP_SIMD_WASM
inline void add_f32_wasm(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);
    for (; i < vec_end; i += 4)
    {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        v128_t vout = wasm_f32x4_add(va, vb);
        wasm_v128_store(out + i, vout);
    }
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
}
inline void mul_f32_wasm(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);
    for (; i < vec_end; i += 4)
    {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        v128_t vout = wasm_f32x4_mul(va, vb);
        wasm_v128_store(out + i, vout);
    }
    for (; i < n; ++i)
        out[i] = a[i] * b[i];
}
#endif

// RISC-V Vector (RVV 1.0) – VLEN agnostic, fallback to scalar if not available
#ifdef NP_SIMD_RVV
inline void add_f32_rvv(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t vl;
    std::size_t i = 0;
    for (; i < n; i += vl)
    {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vc = __riscv_vfadd_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(out + i, vc, vl);
    }
}
inline void mul_f32_rvv(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t vl;
    std::size_t i = 0;
    for (; i < n; i += vl)
    {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vc = __riscv_vfmul_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(out + i, vc, vl);
    }
}
#endif

// ARM SVE – variable vector length
#ifdef NP_SIMD_SVE
inline void add_f32_sve(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svptrue_b32();
    std::size_t vl = svcntw();
    for (; i + vl <= n; i += vl)
    {
        svfloat32_t va = svld1(pg, a + i);
        svfloat32_t vb = svld1(pg, b + i);
        svfloat32_t vc = svadd_f32_z(pg, va, vb);
        svst1(pg, out + i, vc);
    }
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
}
#endif

// POWER VSX / Altivec
#ifdef NP_SIMD_VSX
inline void add_f32_vsx(const float *a, const float *b, float *out, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4);
    for (; i < vec_end; i += 4)
    {
        vector float va = vec_xl(0, a + i);
        vector float vb = vec_xl(0, b + i);
        vector float vc = vec_add(va, vb);
        vec_xst(vc, 0, out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
}
#endif

// Generic Dispatch Functions (Runtime Selection)
/**
 * @brief Vectorized addition with automatic dispatch.
 */
template <typename T> inline void add_vectorized(const T *a, const T *b, T *out, std::size_t n)
{
    if (!tune::should_use_simd(n))
    {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
        return;
    }
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_SIMD_AVX512)
        add_f64_avx512(a, b, out, n);
#elif defined(NP_SIMD_AVX)
        add_f64_avx(a, b, out, n);
#elif defined(NP_SIMD_SSE2)
        add_f64_sse2(a, b, out, n);
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] + b[i];
        }
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_SIMD_AVX512)
        add_f32_avx512(a, b, out, n);
#elif defined(NP_SIMD_AVX)
        add_f32_avx(a, b, out, n);
#elif defined(NP_SIMD_SSE2)
        add_f32_sse(a, b, out, n);
#elif defined(NP_SIMD_NEON)
        add_f32_neon(a, b, out, n);
#elif defined(NP_SIMD_SVE)
        add_f32_sve(a, b, out, n);
#elif defined(NP_SIMD_RVV)
        add_f32_rvv(a, b, out, n);
#elif defined(NP_SIMD_WASM)
        add_f32_wasm(a, b, out, n);
#elif defined(NP_SIMD_VSX)
        add_f32_vsx(a, b, out, n);
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] + b[i];
        }
#endif
    }
    else
    {
        // Scalar fallback
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }
}

/**
 * @brief Vectorized multiplication with automatic dispatch.
 */
template <typename T> inline void mul_vectorized(const T *a, const T *b, T *out, std::size_t n)
{
    if (!tune::should_use_simd(n)) { for (std::size_t i=0;i<n;++i) out[i]=a[i]*b[i]; return; }
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_SIMD_AVX512)
        mul_f64_avx512(a, b, out, n);
#elif defined(NP_SIMD_AVX)
        mul_f64_avx(a, b, out, n);
#elif defined(NP_SIMD_SSE2)
        mul_f64_sse2(a, b, out, n);
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] * b[i];
        }
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_SIMD_AVX512)
        mul_f32_avx512(a, b, out, n);
#elif defined(NP_SIMD_AVX)
        mul_f32_avx(a, b, out, n);
#elif defined(NP_SIMD_SSE2)
        mul_f32_sse(a, b, out, n);
#elif defined(NP_SIMD_NEON)
        mul_f32_neon(a, b, out, n);
#elif defined(NP_SIMD_SVE)
        // SVE/RVV/WASM/VSX fallback to scalar for mul (or add kernel if available)
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] * b[i];
#elif defined(NP_SIMD_RVV)
        mul_f32_rvv(a, b, out, n);
#elif defined(NP_SIMD_WASM)
        mul_f32_wasm(a, b, out, n);
#elif defined(NP_SIMD_VSX)
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] * b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] * b[i];
        }
#endif
    }
    else
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] * b[i];
        }
    }
}

/**
 * @brief Vectorized sum reduction with automatic dispatch.
 */
template <typename T> inline T sum_vectorized(const T *data, std::size_t n)
{
    if (!tune::should_use_simd(n)) { T s{}; for (std::size_t i=0;i<n;++i) s+=data[i]; return s; }
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_SIMD_AVX512)
        return sum_f64_avx512(data, n);
#elif defined(NP_SIMD_AVX)
        return sum_f64_avx(data, n);
#elif defined(NP_SIMD_SSE2)
        return sum_f64_sse2(data, n);
#else
        T sum = T{0};
        for (std::size_t i = 0; i < n; ++i)
        {
            sum += data[i];
        }
        return sum;
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_SIMD_AVX512)
        return sum_f32_avx512(data, n);
#elif defined(NP_SIMD_AVX)
        return sum_f32_avx(data, n);
#elif defined(NP_SIMD_SSE2)
        return sum_f32_sse(data, n);
#elif defined(NP_SIMD_NEON)
        return sum_f32_neon(data, n);
#elif defined(NP_SIMD_SVE) || defined(NP_SIMD_RVV) || defined(NP_SIMD_WASM) || defined(NP_SIMD_VSX)
        {
            T sum = T{0};
            for (std::size_t i = 0; i < n; ++i)
                sum += data[i];
            return sum;
        }
#else
        T sum = T{0};
        for (std::size_t i = 0; i < n; ++i)
        {
            sum += data[i];
        }
        return sum;
#endif
    }
    else
    {
        T sum = T{0};
        for (std::size_t i = 0; i < n; ++i)
        {
            sum += data[i];
        }
        return sum;
    }
}

/**
 * @brief Vectorized subtraction with automatic dispatch.
 */
template <typename T> inline void sub_vectorized(const T *a, const T *b, T *out, std::size_t n)
{
    if (!tune::should_use_simd(n)) { for (std::size_t i=0;i<n;++i) out[i]=a[i]-b[i]; return; }
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_SIMD_AVX512)
        sub_f64_avx512(a, b, out, n);
#elif defined(NP_SIMD_AVX)
        sub_f64_avx(a, b, out, n);
#elif defined(NP_SIMD_SSE2)
        sub_f64_sse2(a, b, out, n);
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] - b[i];
        }
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_SIMD_AVX512)
        sub_f32_avx512(a, b, out, n);
#elif defined(NP_SIMD_AVX)
        sub_f32_avx(a, b, out, n);
#elif defined(NP_SIMD_SSE2)
        sub_f32_sse(a, b, out, n);
#elif defined(NP_SIMD_NEON)
        sub_f32_neon(a, b, out, n);
#elif defined(NP_SIMD_WASM) || defined(NP_SIMD_SVE) || defined(NP_SIMD_RVV) || defined(NP_SIMD_VSX)
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] - b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] - b[i];
        }
#endif
    }
    else
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] - b[i];
        }
    }
}

/**
 * @brief Vectorized division with automatic dispatch.
 */
template <typename T> inline void div_vectorized(const T *a, const T *b, T *out, std::size_t n)
{
    if (!tune::should_use_simd(n)) { for (std::size_t i=0;i<n;++i) out[i]=a[i]/b[i]; return; }
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_SIMD_AVX512)
        div_f64_avx512(a, b, out, n);
#elif defined(NP_SIMD_AVX)
        div_f64_avx(a, b, out, n);
#elif defined(NP_SIMD_SSE2)
        div_f64_sse2(a, b, out, n);
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] / b[i];
        }
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_SIMD_AVX512)
        div_f32_avx512(a, b, out, n);
#elif defined(NP_SIMD_AVX)
        div_f32_avx(a, b, out, n);
#elif defined(NP_SIMD_SSE2)
        div_f32_sse(a, b, out, n);
#elif defined(NP_SIMD_NEON)
        div_f32_neon(a, b, out, n);
#elif defined(NP_SIMD_WASM) || defined(NP_SIMD_SVE) || defined(NP_SIMD_RVV) || defined(NP_SIMD_VSX)
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] / b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] / b[i];
        }
#endif
    }
    else
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] / b[i];
        }
    }
}

/**
 * @brief Constant-time barrier wrappers for SIMD kernels (PQC).
 *
 * Ensures vector loads/stores are not reordered across a PQC boundary.
 * Wraps `pqc::ct_barrier` around an existing vectorized call.
 * Reference: pqc.hpp:ct_barrier
 */
template <typename T> inline void add_vectorized_ct(const T *a, const T *b, T *out, std::size_t n)
{
    pqc::ct_barrier();
    add_vectorized(a, b, out, n);
    pqc::ct_barrier();
}

template <typename T> inline void mul_vectorized_ct(const T *a, const T *b, T *out, std::size_t n)
{
    pqc::ct_barrier();
    mul_vectorized(a, b, out, n);
    pqc::ct_barrier();
}

template <typename T> inline void sub_vectorized_ct(const T *a, const T *b, T *out, std::size_t n)
{
    pqc::ct_barrier();
    sub_vectorized(a, b, out, n);
    pqc::ct_barrier();
}

// FMA: out[i] += a * b[i] with broadcast scalar a (for matmul inner loop)
template <typename T> inline void fma_vectorized(const T *b, T a, T *out, std::size_t n)
{
    if (!tune::should_use_simd(n)) { for (std::size_t i=0;i<n;++i) out[i]+=a*b[i]; return; }
    if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_SIMD_AVX512)
        std::size_t i = 0;
        __m512 va = _mm512_set1_ps(a);
        for (; i + 16 <= n; i += 16)
        {
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 vo = _mm512_loadu_ps(out + i);
#if defined(__FMA__)
            __m512 vr = _mm512_fmadd_ps(va, vb, vo);
#else
            __m512 vr = _mm512_add_ps(vo, _mm512_mul_ps(va, vb));
#endif
            _mm512_storeu_ps(out + i, vr);
        }
        for (; i < n; ++i)
            out[i] += a * b[i];
#elif defined(NP_SIMD_AVX)
        std::size_t i = 0;
        __m256 va = _mm256_set1_ps(a);
        for (; i + 8 <= n; i += 8)
        {
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vo = _mm256_loadu_ps(out + i);
#if defined(__FMA__)
            __m256 vr = _mm256_fmadd_ps(va, vb, vo);
#else
            __m256 vr = _mm256_add_ps(vo, _mm256_mul_ps(va, vb));
#endif
            _mm256_storeu_ps(out + i, vr);
        }
        for (; i < n; ++i)
            out[i] += a * b[i];
#elif defined(NP_SIMD_SSE2)
        std::size_t i = 0;
        __m128 va = _mm_set1_ps(a);
        for (; i + 4 <= n; i += 4)
        {
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vo = _mm_loadu_ps(out + i);
            __m128 vr = _mm_add_ps(vo, _mm_mul_ps(va, vb));
            _mm_storeu_ps(out + i, vr);
        }
        for (; i < n; ++i)
            out[i] += a * b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] += a * b[i];
#endif
    }
    else if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_SIMD_AVX512)
        std::size_t i = 0;
        __m512d va = _mm512_set1_pd(a);
        for (; i + 8 <= n; i += 8)
        {
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d vo = _mm512_loadu_pd(out + i);
#if defined(__FMA__)
            __m512d vr = _mm512_fmadd_pd(va, vb, vo);
#else
            __m512d vr = _mm512_add_pd(vo, _mm512_mul_pd(va, vb));
#endif
            _mm512_storeu_pd(out + i, vr);
        }
        for (; i < n; ++i)
            out[i] += a * b[i];
#elif defined(NP_SIMD_AVX)
        std::size_t i = 0;
        __m256d va = _mm256_set1_pd(a);
        for (; i + 4 <= n; i += 4)
        {
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d vo = _mm256_loadu_pd(out + i);
#if defined(__FMA__)
            __m256d vr = _mm256_fmadd_pd(va, vb, vo);
#else
            __m256d vr = _mm256_add_pd(vo, _mm256_mul_pd(va, vb));
#endif
            _mm256_storeu_pd(out + i, vr);
        }
        for (; i < n; ++i)
            out[i] += a * b[i];
#elif defined(NP_SIMD_SSE2)
        std::size_t i = 0;
        __m128d va = _mm_set1_pd(a);
        for (; i + 2 <= n; i += 2)
        {
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d vo = _mm_loadu_pd(out + i);
            __m128d vr = _mm_add_pd(vo, _mm_mul_pd(va, vb));
            _mm_storeu_pd(out + i, vr);
        }
        for (; i < n; ++i)
            out[i] += a * b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] += a * b[i];
#endif
    }
    else
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] += a * b[i];
    }
}

// ── Transcendental (SLEEF-accelerated, scalar fallback) ───────────────
template <typename T> inline void sin_vectorized(const T *in, T *out, std::size_t n)
    if (!tune::should_use_simd(n)) { for (std::size_t i=0;i<n;++i) out[i]=std::sin(in[i]); return; }
{
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_HAS_SLEEF) && defined(NP_SIMD_AVX)
        std::size_t i = 0;
#if defined(NP_SIMD_AVX512)
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(in + i);
            __m512d r = Sleef_sind8_u10(v);
            _mm512_storeu_pd(out + i, r);
        }
#endif
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(in + i);
            __m256d r = Sleef_sind4_u10(v);
            _mm256_storeu_pd(out + i, r);
        }
        for (; i < n; ++i)
            out[i] = std::sin(in[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::sin(in[i]);
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_HAS_SLEEF) && defined(NP_SIMD_AVX)
        std::size_t i = 0;
#if defined(NP_SIMD_AVX512)
        for (; i + 16 <= n; i += 16)
        {
            __m512 v = _mm512_loadu_ps(in + i);
            __m512 r = Sleef_sinf16_u10(v);
            _mm512_storeu_ps(out + i, r);
        }
#endif
        for (; i + 8 <= n; i += 8)
        {
            __m256 v = _mm256_loadu_ps(in + i);
            __m256 r = Sleef_sinf8_u10(v);
            _mm256_storeu_ps(out + i, r);
        }
        for (; i < n; ++i)
            out[i] = std::sin(in[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::sin(in[i]);
#endif
    }
    else
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::sin(in[i]);
    }
}
template <typename T> inline void cos_vectorized(const T *in, T *out, std::size_t n)
    if (!tune::should_use_simd(n)) { for (std::size_t i=0;i<n;++i) out[i]=std::cos(in[i]); return; }
{
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_HAS_SLEEF) && defined(NP_SIMD_AVX)
        std::size_t i = 0;
#if defined(NP_SIMD_AVX512)
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(in + i);
            __m512d r = Sleef_cosd8_u10(v);
            _mm512_storeu_pd(out + i, r);
        }
#endif
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(in + i);
            __m256d r = Sleef_cosd4_u10(v);
            _mm256_storeu_pd(out + i, r);
        }
        for (; i < n; ++i)
            out[i] = std::cos(in[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::cos(in[i]);
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_HAS_SLEEF) && defined(NP_SIMD_AVX)
        std::size_t i = 0;
#if defined(NP_SIMD_AVX512)
        for (; i + 16 <= n; i += 16)
        {
            __m512 v = _mm512_loadu_ps(in + i);
            __m512 r = Sleef_cosf16_u10(v);
            _mm512_storeu_ps(out + i, r);
        }
#endif
        for (; i + 8 <= n; i += 8)
        {
            __m256 v = _mm256_loadu_ps(in + i);
            __m256 r = Sleef_cosf8_u10(v);
            _mm256_storeu_ps(out + i, r);
        }
        for (; i < n; ++i)
            out[i] = std::cos(in[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::cos(in[i]);
#endif
    }
    else
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::cos(in[i]);
    }
}
    if (!tune::should_use_simd(n)) { for (std::size_t i=0;i<n;++i) out[i]=std::exp(in[i]); return; }
template <typename T> inline void exp_vectorized(const T *in, T *out, std::size_t n)
{
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_HAS_SLEEF) && defined(NP_SIMD_AVX)
        std::size_t i = 0;
#if defined(NP_SIMD_AVX512)
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(in + i);
            __m512d r = Sleef_expd8_u10(v);
            _mm512_storeu_pd(out + i, r);
        }
#endif
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(in + i);
            __m256d r = Sleef_expd4_u10(v);
            _mm256_storeu_pd(out + i, r);
        }
        for (; i < n; ++i)
            out[i] = std::exp(in[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::exp(in[i]);
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_HAS_SLEEF) && defined(NP_SIMD_AVX)
        std::size_t i = 0;
#if defined(NP_SIMD_AVX512)
        for (; i + 16 <= n; i += 16)
        {
            __m512 v = _mm512_loadu_ps(in + i);
            __m512 r = Sleef_expf16_u10(v);
            _mm512_storeu_ps(out + i, r);
        }
#endif
        for (; i + 8 <= n; i += 8)
        {
            __m256 v = _mm256_loadu_ps(in + i);
            __m256 r = Sleef_expf8_u10(v);
            _mm256_storeu_ps(out + i, r);
        }
        for (; i < n; ++i)
            out[i] = std::exp(in[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::exp(in[i]);
#endif
    }
    else
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::exp(in[i]);
    }
    if (!tune::should_use_simd(n)) { for (std::size_t i=0;i<n;++i) out[i]=std::log(in[i]); return; }
}
template <typename T> inline void log_vectorized(const T *in, T *out, std::size_t n)
{
    if constexpr (std::is_same_v<T, double>)
    {
#if defined(NP_HAS_SLEEF) && defined(NP_SIMD_AVX)
        std::size_t i = 0;
#if defined(NP_SIMD_AVX512)
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(in + i);
            __m512d r = Sleef_logd8_u10(v);
            _mm512_storeu_pd(out + i, r);
        }
#endif
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(in + i);
            __m256d r = Sleef_logd4_u10(v);
            _mm256_storeu_pd(out + i, r);
        }
        for (; i < n; ++i)
            out[i] = std::log(in[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::log(in[i]);
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
#if defined(NP_HAS_SLEEF) && defined(NP_SIMD_AVX)
        std::size_t i = 0;
#if defined(NP_SIMD_AVX512)
        for (; i + 16 <= n; i += 16)
        {
            __m512 v = _mm512_loadu_ps(in + i);
            __m512 r = Sleef_logf16_u10(v);
            _mm512_storeu_ps(out + i, r);
        }
#endif
        for (; i + 8 <= n; i += 8)
        {
            __m256 v = _mm256_loadu_ps(in + i);
            __m256 r = Sleef_logf8_u10(v);
            _mm256_storeu_ps(out + i, r);
        }
        for (; i < n; ++i)
            out[i] = std::log(in[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::log(in[i]);
#endif
    }
    else
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = std::log(in[i]);
    }
}

} // namespace simd
} // namespace np

#endif // NP_SIMD_HPP
