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
#pragma once

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

namespace np::tune::detail
{
// Tuning constants (constexpr; values identical to the former NP_TUNE_* macros).
inline constexpr std::size_t kKb = 1024;
inline constexpr std::size_t kMb = 1024 * 1024;

inline constexpr std::size_t kL1DefaultBytes = 32 * kKb;
inline constexpr std::size_t kL2DefaultBytes = 256 * kKb;
inline constexpr std::size_t kL3DefaultBytes = 12 * kMb;
inline constexpr std::size_t kL3FallbackMult = 8;
inline constexpr std::size_t kHwThreadsDefault = 8;
inline constexpr std::size_t kHwCoresDiv = 2;

inline constexpr int kSimdAvx512F32 = 16;
inline constexpr int kSimdAvx512F64 = 8;
inline constexpr int kSimdAvx2F32 = 8;
inline constexpr int kSimdAvx2F64 = 4;
inline constexpr int kSimdNeonF32 = 4;
inline constexpr int kSimdNeonF64 = 2;

inline constexpr std::size_t kBlockMin = 32;
inline constexpr std::size_t kBlockAlign = 8;
inline constexpr std::size_t kBlockBase = 32;
inline constexpr std::size_t kBlock32M = 256;
inline constexpr std::size_t kBlock16M = 192;
inline constexpr std::size_t kBlock8M = 128;
inline constexpr std::size_t kBlock4M = 96;
inline constexpr std::size_t kBlock2M = 64;
inline constexpr std::size_t kL3Thresh32M = 32 * kMb;
inline constexpr std::size_t kL3Thresh16M = 16 * kMb;
inline constexpr std::size_t kL3Thresh8M = 8 * kMb;
inline constexpr std::size_t kL3Thresh4M = 4 * kMb;
inline constexpr std::size_t kL3Thresh2M = 2 * kMb;
inline constexpr std::size_t kBlockF64Num = 3;
inline constexpr std::size_t kBlockF64Den = 4;
inline constexpr std::size_t kL2Thresh1M = 1 * kMb;
inline constexpr std::size_t kL2Thresh512K = 512 * kKb;
inline constexpr std::size_t kFftBlock8K = 8192;
inline constexpr std::size_t kFftBlock4K = 4096;
inline constexpr std::size_t kFftBlock2K = 2048;
inline constexpr std::size_t kGpuThreads64 = 64;
inline constexpr std::size_t kGpuThreads32 = 32;
inline constexpr std::size_t kGpuThreads16 = 16;
inline constexpr std::size_t kGpuFlops8M = 8000000;
inline constexpr std::size_t kGpuFlops4M = 4000000;
inline constexpr std::size_t kGpuFlops2M = 2000000;
inline constexpr std::size_t kGpuFlops1M = 1000000;
inline constexpr std::size_t kThreadingFactor = 1024;
inline constexpr std::size_t kSimdThreshold = 64;
inline constexpr std::size_t kFftThreshold = 8192;
inline constexpr std::size_t kRandomThreshold = 10000;
inline constexpr std::size_t kWindowThreshold = 2048;
inline constexpr std::size_t kPolyThreshold = 1000;
inline constexpr std::size_t kThreadChunkDiv = 4;
inline constexpr std::size_t kThreadChunkMin = 1;
} // namespace np::tune::detail

namespace np::tune
{

// CPU topology (static values cached on first call; sysconf runs once)
NP_NODISCARD inline std::size_t l1_cache_bytes() noexcept
{
    static const std::size_t cached = [] {
#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
        long v = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        if (v > 0)
            return static_cast<std::size_t>(v);
#endif
        return detail::kL1DefaultBytes;
    }();
    return cached;
}
NP_NODISCARD inline std::size_t l2_cache_bytes() noexcept
{
    static const std::size_t cached = [] {
#if defined(__linux__) && defined(_SC_LEVEL2_CACHE_SIZE)
        long v = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (v > 0)
            return static_cast<std::size_t>(v);
#endif
        return detail::kL2DefaultBytes;
    }();
    return cached;
}
NP_NODISCARD inline std::size_t l3_cache_bytes() noexcept
{
    static const std::size_t cached = [] {
#if defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)
        long v = sysconf(_SC_LEVEL3_CACHE_SIZE);
        if (v > 0)
            return static_cast<std::size_t>(v);
#endif
#if defined(_SC_LEVEL2_CACHE_SIZE)
        long v2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (v2 > 0)
            return static_cast<std::size_t>(v2) * detail::kL3FallbackMult;
#endif
        return detail::kL3DefaultBytes;
    }();
    return cached;
}
NP_NODISCARD inline std::size_t hardware_threads() noexcept
{
    static const std::size_t cached = [] {
        std::size_t n = std::thread::hardware_concurrency();
        return n ? n : detail::kHwThreadsDefault;
    }();
    return cached;
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
    return (t + 1) / detail::kHwCoresDiv;
}
NP_NODISCARD inline std::size_t numa_nodes() noexcept
{
    static const std::size_t cached = [] {
#if defined(__linux__) && defined(_SC_NPROCESSORS_CONF)
        // Heuristic: threads / cores
        std::size_t t = hardware_threads(), c = hardware_cores();
        if (c == 0)
            return std::size_t{1};
        std::size_t n = t / c;
        return n ? n : std::size_t{1};
#else
        return std::size_t{1};
#endif
    }();
    return cached;
}

// SIMD width
struct SimdInfo
{
    int width_f32 = 1, width_f64 = 1;
    bool has_avx512 = false, has_avx2 = false, has_neon = false;
};
NP_NODISCARD inline SimdInfo simd_info() noexcept
{
    static const SimdInfo cached = [] {
        SimdInfo s;
#if defined(__AVX512F__)
        s.has_avx512 = true;
        s.width_f32 = detail::kSimdAvx512F32;
        s.width_f64 = detail::kSimdAvx512F64;
#elif defined(__AVX2__) || defined(__AVX__)
        s.has_avx2 = true;
        s.width_f32 = detail::kSimdAvx2F32;
        s.width_f64 = detail::kSimdAvx2F64;
#elif defined(__ARM_NEON)
        s.has_neon = true;
        s.width_f32 = detail::kSimdNeonF32;
        s.width_f64 = detail::kSimdNeonF64;
#endif
        return s;
    }();
    return cached;
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
    std::size_t b = detail::kBlockBase;
    if (l3 >= detail::kL3Thresh32M)
        b = detail::kBlock32M;
    else if (l3 >= detail::kL3Thresh16M)
        b = detail::kBlock16M;
    else if (l3 >= detail::kL3Thresh8M)
        b = detail::kBlock8M;
    else if (l3 >= detail::kL3Thresh4M)
        b = detail::kBlock4M;
    else if (l3 >= detail::kL3Thresh2M)
        b = detail::kBlock2M;
    b = (b / detail::kBlockAlign) * detail::kBlockAlign;
    return std::max<std::size_t>(detail::kBlockMin, b);
}
NP_NODISCARD inline std::size_t optimal_block_f64() noexcept
{
    return (optimal_block_f32() * detail::kBlockF64Num) / detail::kBlockF64Den;
}
NP_NODISCARD inline std::size_t optimal_block_int() noexcept
{
    return optimal_block_f32();
}
NP_NODISCARD inline std::size_t optimal_fft_block() noexcept
{
    // FFT radix-2 benefits from L2-sized blocks
    std::size_t l2 = l2_cache_bytes();
    if (l2 >= detail::kL2Thresh1M)
        return detail::kFftBlock8K;
    if (l2 >= detail::kL2Thresh512K)
        return detail::kFftBlock4K;
    return detail::kFftBlock2K;
}
NP_NODISCARD inline std::size_t optimal_einsum_block() noexcept
{
    return optimal_block_f32();
}

// Thresholds
NP_NODISCARD inline std::size_t gpu_threshold_flops() noexcept
{
    std::size_t threads = hardware_threads();
    if (threads >= detail::kGpuThreads64)
        return detail::kGpuFlops8M;
    if (threads >= detail::kGpuThreads32)
        return detail::kGpuFlops4M;
    if (threads >= detail::kGpuThreads16)
        return detail::kGpuFlops2M;
    return detail::kGpuFlops1M;
}
NP_NODISCARD inline std::size_t gpu_threshold_bytes() noexcept
{
    return gpu_threshold_flops() * sizeof(float);
}
NP_NODISCARD inline std::size_t threading_threshold() noexcept
{
    std::size_t t = hardware_threads();
    return t * detail::kThreadingFactor;
}
NP_NODISCARD inline std::size_t simd_threshold() noexcept
{
    return detail::kSimdThreshold;
}
NP_NODISCARD inline std::size_t fft_threshold() noexcept
{
    // Use GPU FFT for N >= 8192 when GPU available, else SIMD FFT
    return detail::kFftThreshold;
}
NP_NODISCARD inline std::size_t random_threshold() noexcept
{
    return detail::kRandomThreshold;
}
NP_NODISCARD inline std::size_t window_threshold() noexcept
{
    return detail::kWindowThreshold;
}
NP_NODISCARD inline std::size_t poly_threshold() noexcept
{
    return detail::kPolyThreshold;
}

// Helpers
NP_NODISCARD inline std::size_t thread_chunk(std::size_t n) noexcept
{
    std::size_t t = hardware_threads();
    return std::max<std::size_t>(detail::kThreadChunkMin, n / (t * detail::kThreadChunkDiv));
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
    return "L3=" + std::to_string(l3_cache_bytes() / detail::kMb) + "MB threads=" + std::to_string(hardware_threads()) +
           " simd_f32=" + std::to_string(s.width_f32);
}

} // namespace np::tune

#endif // NP_POWERFUL_HPP
