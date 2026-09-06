/**
 * @file powerful.hpp
 * @brief Tuning for very powerful workstation + GPU — cache, threads, GPU thresholds.
 *
 * Provides np::tune with runtime cache detection, optimal blocking, GPU thresholds,
 * and NUMA-aware helpers. Header-only, used by linalg, simd, gpu, memory.
 *
 * @author Sergio Randriamihoatra
 */
#ifndef NP_POWERFUL_HPP
#define NP_POWERFUL_HPP

#include "api_macros.hpp"
#include <algorithm>
#include <cstddef>
#include <thread>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace np::tune
{

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
        return static_cast<std::size_t>(v2) * 8;
#endif
    return 12 * 1024 * 1024;
}

NP_NODISCARD inline std::size_t hardware_threads() noexcept
{
    std::size_t n = std::thread::hardware_concurrency();
    return n ? n : 8;
}

NP_NODISCARD inline std::size_t optimal_block_f32() noexcept
{
    std::size_t l3 = l3_cache_bytes();
    std::size_t b = 32;
    if (l3 >= 8 * 1024 * 1024)
        b = 128;
    else if (l3 >= 4 * 1024 * 1024)
        b = 96;
    else if (l3 >= 2 * 1024 * 1024)
        b = 64;
    b = (b / 8) * 8;
    return std::max<std::size_t>(32, b);
}

NP_NODISCARD inline std::size_t optimal_block_f64() noexcept
{
    return optimal_block_f32() * 3 / 4;
}

NP_NODISCARD inline std::size_t gpu_threshold_flops() noexcept
{
    std::size_t threads = hardware_threads();
    if (threads >= 32)
        return 4'000'000;
    if (threads >= 16)
        return 2'000'000;
    return 1'000'000;
}

NP_NODISCARD inline std::size_t thread_chunk(std::size_t n) noexcept
{
    std::size_t t = hardware_threads();
    return std::max<std::size_t>(1, n / (t * 4));
}

} // namespace np::tune

#endif // NP_POWERFUL_HPP
