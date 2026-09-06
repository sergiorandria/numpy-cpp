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
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace np::tune
{

// ── CPU topology ───────────────────────────────────────────────────────────
NP_NODISCARD inline std::size_t l1_cache_bytes() noexcept
{
#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
    long v = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (v > 0) return static_cast<std::size_t>(v);
#endif
    return 32 * 1024;
}
NP_NODISCARD inline std::size_t l2_cache_bytes() noexcept
{
#if defined(__linux__) && defined(_SC_LEVEL2_CACHE_SIZE)
    long v = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (v > 0) return static_cast<std::size_t>(v);
#endif
    return 256 * 1024;
}
NP_NODISCARD inline std::size_t l3_cache_bytes() noexcept
{
#if defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)
    long v = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (v > 0) return static_cast<std::size_t>(v);
#endif
#if defined(_SC_LEVEL2_CACHE_SIZE)
    long v2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (v2 > 0) return static_cast<std::size_t>(v2) * 8;
#endif
    return 12 * 1024 * 1024;
}
NP_NODISCARD inline std::size_t hardware_threads() noexcept
{
    std::size_t n = std::thread::hardware_concurrency();
    return n ? n : 8;
}
NP_NODISCARD inline std::size_t hardware_cores() noexcept
{
    // Physical cores ≈ threads / 2 on x86 with HT
    std::size_t t = hardware_threads();
#if defined(__linux__)
    long c = sysconf(_SC_NPROCESSORS_ONLN);
    if (c > 0) return static_cast<std::size_t>(c);
#endif
    return (t + 1) / 2;
}
NP_NODISCARD inline std::size_t numa_nodes() noexcept
{
#if defined(__linux__) && defined(_SC_NPROCESSORS_CONF)
    // Heuristic: threads / cores
    std::size_t t = hardware_threads(), c = hardware_cores();
    if (c == 0) return 1;
    std::size_t n = t / c;
    return n ? n : 1;
#else
    return 1;
#endif
}

// ── SIMD width ─────────────────────────────────────────────────────────────
struct SimdInfo
{
    int width_f32 = 1, width_f64 = 1;
    bool has_avx512 = false, has_avx2 = false, has_neon = false;
};
NP_NODISCARD inline SimdInfo simd_info() noexcept
{
    SimdInfo s;
#if defined(__AVX512F__)
    s.has_avx512 = true; s.width_f32 = 16; s.width_f64 = 8;
#elif defined(__AVX2__) || defined(__AVX__)
    s.has_avx2 = true; s.width_f32 = 8; s.width_f64 = 4;
#elif defined(__ARM_NEON)
    s.has_neon = true; s.width_f32 = 4; s.width_f64 = 2;
#endif
    return s;
}

// ── GPU caps ───────────────────────────────────────────────────────────────
struct GpuInfo
{
    int count = 0;
    std::size_t mem = 0;
    bool has_fp8 = false, has_fp4 = false, is_blackwell = false;
};
NP_NODISCARD inline GpuInfo gpu_info() noexcept;

// ── Blocking ───────────────────────────────────────────────────────────────
NP_NODISCARD inline std::size_t optimal_block_f32() noexcept
{
    std::size_t l3 = l3_cache_bytes();
    std::size_t b = 32;
    if (l3 >= 32 * 1024 * 1024) b = 256;
    else if (l3 >= 16 * 1024 * 1024) b = 192;
    else if (l3 >= 8 * 1024 * 1024) b = 128;
    else if (l3 >= 4 * 1024 * 1024) b = 96;
    else if (l3 >= 2 * 1024 * 1024) b = 64;
    b = (b / 8) * 8;
    return std::max<std::size_t>(32, b);
}
NP_NODISCARD inline std::size_t optimal_block_f64() noexcept
{
    return (optimal_block_f32() * 3) / 4;
}
NP_NODISCARD inline std::size_t optimal_block_int() noexcept
{
    return optimal_block_f32();
}
NP_NODISCARD inline std::size_t optimal_fft_block() noexcept
{
    // FFT radix-2 benefits from L2-sized blocks
    std::size_t l2 = l2_cache_bytes();
    if (l2 >= 1024 * 1024) return 8192;
    if (l2 >= 512 * 1024) return 4096;
    return 2048;
}
NP_NODISCARD inline std::size_t optimal_einsum_block() noexcept
{
    return optimal_block_f32();
}

// ── Thresholds ─────────────────────────────────────────────────────────────
NP_NODISCARD inline std::size_t gpu_threshold_flops() noexcept
{
    std::size_t threads = hardware_threads();
    if (threads >= 64) return 8'000'000;
    if (threads >= 32) return 4'000'000;
    if (threads >= 16) return 2'000'000;
    return 1'000'000;
}
NP_NODISCARD inline std::size_t gpu_threshold_bytes() noexcept
{
    return gpu_threshold_flops() * sizeof(float);
}
NP_NODISCARD inline std::size_t threading_threshold() noexcept
{
    std::size_t t = hardware_threads();
    return t * 1024;
}
NP_NODISCARD inline std::size_t simd_threshold() noexcept { return 64; }
NP_NODISCARD inline std::size_t fft_threshold() noexcept
{
    // Use GPU FFT for N >= 8192 when GPU available, else SIMD FFT
    return 8192;
}
NP_NODISCARD inline std::size_t random_threshold() noexcept { return 10000; }
NP_NODISCARD inline std::size_t window_threshold() noexcept { return 2048; }
NP_NODISCARD inline std::size_t poly_threshold() noexcept { return 1000; }

// ── Helpers ────────────────────────────────────────────────────────────────
NP_NODISCARD inline std::size_t thread_chunk(std::size_t n) noexcept
{
    std::size_t t = hardware_threads();
    return std::max<std::size_t>(1, n / (t * 4));
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
    return "L3=" + std::to_string(l3_cache_bytes() / (1024 * 1024)) + "MB threads=" +
           std::to_string(hardware_threads()) + " simd_f32=" + std::to_string(s.width_f32);
}

} // namespace np::tune

#endif // NP_POWERFUL_HPP
