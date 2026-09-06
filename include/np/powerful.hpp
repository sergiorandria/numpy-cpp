/**
 * @file powerful.hpp
 * @brief Tuning for very powerful workstation + GPU — cache, threads, GPU thresholds.
 *
 * Central tuning hub for all subsystems (linalg, simd, gpu, memory, tensor,
 * fft, random, statistics, window, polynomial, differential, spectral, etc.).
 * Header-only, C++20, uses runtime cache detection, SIMD width, NUMA, and
 * GPU caps. All subsystems should query tune:: rather than hard-coding.
 *
 * @author Sergio Randriamihoatra
 */
#ifndef NP_POWERFUL_HPP
#define NP_POWERFUL_HPP

#include "api_macros.hpp"
#include <algorithm>
#include <cstddef>
#include <string>
#include <thread>

#if defined(__linux__)
#include <unistd.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

// Tuning constants
#define NP_TUNE_KB 1024
#define NP_TUNE_MB (1024 * 1024)

#define NP_TUNE_L1_DEFAULT_BYTES (32 * NP_TUNE_KB)
#define NP_TUNE_L2_DEFAULT_BYTES (256 * NP_TUNE_KB)
#define NP_TUNE_L3_DEFAULT_BYTES (12 * NP_TUNE_MB)
#define NP_TUNE_L3_FALLBACK_MULT 8
#define NP_TUNE_HW_THREADS_DEFAULT 8
#define NP_TUNE_HW_CORES_DIV 2

#define NP_TUNE_SIMD_AVX512_F32 16
#define NP_TUNE_SIMD_AVX512_F64 8
#define NP_TUNE_SIMD_AVX2_F32 8
#define NP_TUNE_SIMD_AVX2_F64 4
#define NP_TUNE_SIMD_NEON_F32 4
#define NP_TUNE_SIMD_NEON_F64 2

#define NP_TUNE_BLOCK_MIN 32
#define NP_TUNE_BLOCK_ALIGN 8
#define NP_TUNE_BLOCK_BASE 32
#define NP_TUNE_BLOCK_32M 256
#define NP_TUNE_BLOCK_16M 192
#define NP_TUNE_BLOCK_8M 128
#define NP_TUNE_BLOCK_4M 96
#define NP_TUNE_BLOCK_2M 64
#define NP_TUNE_L3_THRESH_32M (32 * NP_TUNE_MB)
#define NP_TUNE_L3_THRESH_16M (16 * NP_TUNE_MB)
#define NP_TUNE_L3_THRESH_8M (8 * NP_TUNE_MB)
#define NP_TUNE_L3_THRESH_4M (4 * NP_TUNE_MB)
#define NP_TUNE_L3_THRESH_2M (2 * NP_TUNE_MB)
#define NP_TUNE_BLOCK_F64_NUM 3
#define NP_TUNE_BLOCK_F64_DEN 4
#define NP_TUNE_L2_THRESH_1M (1 * NP_TUNE_MB)
#define NP_TUNE_L2_THRESH_512K (512 * NP_TUNE_KB)
#define NP_TUNE_FFT_BLOCK_8K 8192
#define NP_TUNE_FFT_BLOCK_4K 4096
#define NP_TUNE_FFT_BLOCK_2K 2048
#define NP_TUNE_GPU_THREADS_64 64
#define NP_TUNE_GPU_THREADS_32 32
#define NP_TUNE_GPU_THREADS_16 16
#define NP_TUNE_GPU_FLOPS_8M 8000000
#define NP_TUNE_GPU_FLOPS_4M 4000000
#define NP_TUNE_GPU_FLOPS_2M 2000000
#define NP_TUNE_GPU_FLOPS_1M 1000000
#define NP_TUNE_THREADING_FACTOR 1024
#define NP_TUNE_SIMD_THRESHOLD 64
#define NP_TUNE_FFT_THRESHOLD 8192
#define NP_TUNE_RANDOM_THRESHOLD 10000
#define NP_TUNE_WINDOW_THRESHOLD 2048
#define NP_TUNE_POLY_THRESHOLD 1000
#define NP_TUNE_THREAD_CHUNK_DIV 4
#define NP_TUNE_THREAD_CHUNK_MIN 1

namespace np::tune
{

// CPU topology
NP_NODISCARD inline std::size_t l1_cache_bytes() noexcept
{
#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
    long v = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (v > 0)
        return static_cast<std::size_t>(v);
#endif
    return NP_TUNE_L1_DEFAULT_BYTES;
}
NP_NODISCARD inline std::size_t l2_cache_bytes() noexcept
{
#if defined(__linux__) && defined(_SC_LEVEL2_CACHE_SIZE)
    long v = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (v > 0)
        return static_cast<std::size_t>(v);
#endif
    return NP_TUNE_L2_DEFAULT_BYTES;
}
NP_NODISCARD inline std::size_t l3_cache_bytes() noexcept
{
#if defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)
    long v = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (v > 0)
        return static_cast<std::size_t>(v);
#endif
#if defined(_SC_LEVEL2_CACHE_SIZE)
    long v2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (v2 > 0)
        return static_cast<std::size_t>(v2) * NP_TUNE_L3_FALLBACK_MULT;
#endif
    return NP_TUNE_L3_DEFAULT_BYTES;
}
NP_NODISCARD inline std::size_t hardware_threads() noexcept
{
    std::size_t n = std::thread::hardware_concurrency();
    return n ? n : NP_TUNE_HW_THREADS_DEFAULT;
}
NP_NODISCARD inline std::size_t hardware_cores() noexcept
{
    // Physical cores ≈ threads / 2 on x86 with HT
    std::size_t t = hardware_threads();
#if defined(__linux__)
    long c = sysconf(_SC_NPROCESSORS_ONLN);
    if (c > 0)
        return static_cast<std::size_t>(c);
#endif
    return (t + 1) / NP_TUNE_HW_CORES_DIV;
}
NP_NODISCARD inline std::size_t numa_nodes() noexcept
{
#if defined(__linux__) && defined(_SC_NPROCESSORS_CONF)
    // Heuristic: threads / cores
    std::size_t t = hardware_threads(), c = hardware_cores();
    if (c == 0)
        return 1;
    std::size_t n = t / c;
    return n ? n : 1;
#else
    return 1;
#endif
}

// SIMD width
struct SimdInfo
{
    int width_f32 = 1, width_f64 = 1;
    bool has_avx512 = false, has_avx2 = false, has_neon = false;
};
NP_NODISCARD inline SimdInfo simd_info() noexcept
{
    SimdInfo s;
#if defined(__AVX512F__)
    s.has_avx512 = true;
    s.width_f32 = NP_TUNE_SIMD_AVX512_F32;
    s.width_f64 = NP_TUNE_SIMD_AVX512_F64;
#elif defined(__AVX2__) || defined(__AVX__)
    s.has_avx2 = true;
    s.width_f32 = NP_TUNE_SIMD_AVX2_F32;
    s.width_f64 = NP_TUNE_SIMD_AVX2_F64;
#elif defined(__ARM_NEON)
    s.has_neon = true;
    s.width_f32 = NP_TUNE_SIMD_NEON_F32;
    s.width_f64 = NP_TUNE_SIMD_NEON_F64;
#endif
    return s;
}

// GPU caps
struct GpuInfo
{
    int count = 0;
    std::size_t mem = 0;
    bool has_fp8 = false, has_fp4 = false, is_blackwell = false;
};
NP_NODISCARD inline GpuInfo gpu_info() noexcept;

// Blocking
NP_NODISCARD inline std::size_t optimal_block_f32() noexcept
{
    std::size_t l3 = l3_cache_bytes();
    std::size_t b = NP_TUNE_BLOCK_BASE;
    if (l3 >= NP_TUNE_L3_THRESH_32M)
        b = NP_TUNE_BLOCK_32M;
    else if (l3 >= NP_TUNE_L3_THRESH_16M)
        b = NP_TUNE_BLOCK_16M;
    else if (l3 >= NP_TUNE_L3_THRESH_8M)
        b = NP_TUNE_BLOCK_8M;
    else if (l3 >= NP_TUNE_L3_THRESH_4M)
        b = NP_TUNE_BLOCK_4M;
    else if (l3 >= NP_TUNE_L3_THRESH_2M)
        b = NP_TUNE_BLOCK_2M;
    b = (b / NP_TUNE_BLOCK_ALIGN) * NP_TUNE_BLOCK_ALIGN;
    return std::max<std::size_t>(NP_TUNE_BLOCK_MIN, b);
}
NP_NODISCARD inline std::size_t optimal_block_f64() noexcept
{
    return (optimal_block_f32() * NP_TUNE_BLOCK_F64_NUM) / NP_TUNE_BLOCK_F64_DEN;
}
NP_NODISCARD inline std::size_t optimal_block_int() noexcept
{
    return optimal_block_f32();
}
NP_NODISCARD inline std::size_t optimal_fft_block() noexcept
{
    // FFT radix-2 benefits from L2-sized blocks
    std::size_t l2 = l2_cache_bytes();
    if (l2 >= NP_TUNE_L2_THRESH_1M)
        return NP_TUNE_FFT_BLOCK_8K;
    if (l2 >= NP_TUNE_L2_THRESH_512K)
        return NP_TUNE_FFT_BLOCK_4K;
    return NP_TUNE_FFT_BLOCK_2K;
}
NP_NODISCARD inline std::size_t optimal_einsum_block() noexcept
{
    return optimal_block_f32();
}

// Thresholds
NP_NODISCARD inline std::size_t gpu_threshold_flops() noexcept
{
    std::size_t threads = hardware_threads();
    if (threads >= NP_TUNE_GPU_THREADS_64)
        return NP_TUNE_GPU_FLOPS_8M;
    if (threads >= NP_TUNE_GPU_THREADS_32)
        return NP_TUNE_GPU_FLOPS_4M;
    if (threads >= NP_TUNE_GPU_THREADS_16)
        return NP_TUNE_GPU_FLOPS_2M;
    return NP_TUNE_GPU_FLOPS_1M;
}
NP_NODISCARD inline std::size_t gpu_threshold_bytes() noexcept
{
    return gpu_threshold_flops() * sizeof(float);
}
NP_NODISCARD inline std::size_t threading_threshold() noexcept
{
    std::size_t t = hardware_threads();
    return t * NP_TUNE_THREADING_FACTOR;
}
NP_NODISCARD inline std::size_t simd_threshold() noexcept
{
    return NP_TUNE_SIMD_THRESHOLD;
}
NP_NODISCARD inline std::size_t fft_threshold() noexcept
{
    // Use GPU FFT for N >= 8192 when GPU available, else SIMD FFT
    return NP_TUNE_FFT_THRESHOLD;
}
NP_NODISCARD inline std::size_t random_threshold() noexcept
{
    return NP_TUNE_RANDOM_THRESHOLD;
}
NP_NODISCARD inline std::size_t window_threshold() noexcept
{
    return NP_TUNE_WINDOW_THRESHOLD;
}
NP_NODISCARD inline std::size_t poly_threshold() noexcept
{
    return NP_TUNE_POLY_THRESHOLD;
}

// Helpers
NP_NODISCARD inline std::size_t thread_chunk(std::size_t n) noexcept
{
    std::size_t t = hardware_threads();
    return std::max<std::size_t>(NP_TUNE_THREAD_CHUNK_MIN, n / (t * NP_TUNE_THREAD_CHUNK_DIV));
}
NP_NODISCARD inline bool should_use_gpu(std::size_t flops) noexcept
{
    // Defined in gpu.hpp, but provide lightweight check here
    return flops > gpu_threshold_flops();
}
NP_NODISCARD inline bool should_use_threading(std::size_t n) noexcept
{
    return n > threading_threshold();
}
NP_NODISCARD inline bool should_use_simd(std::size_t n) noexcept
{
    return n > simd_threshold();
}
NP_NODISCARD inline std::string tune_summary() noexcept
{
    SimdInfo s = simd_info();
    return "L3=" + std::to_string(l3_cache_bytes() / NP_TUNE_MB) + "MB threads=" + std::to_string(hardware_threads()) +
           " simd_f32=" + std::to_string(s.width_f32);
}

} // namespace np::tune

#endif // NP_POWERFUL_HPP
