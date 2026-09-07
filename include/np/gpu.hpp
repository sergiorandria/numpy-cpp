/**
 * @file gpu.hpp
 * @brief Unified GPU abstraction for powerful computers — CUDA/HIP/OpenMP target.
 *
 * Header-only, no hard CUDA/HIP dependency. At runtime:
 *  - Tries CUDA driver via dlopen("libcuda.so.1" / "nvcuda.dll" / "libcuda.dylib")
 *    and cuInit/cuDeviceGetCount without needing <cuda_runtime.h> at build time.
 *  - Tries OpenMP target offload via omp_get_num_devices() when _OPENMP is available.
 *  - Falls back to CPU ThreadPool + SIMD when no GPU is present.
 *
 * Provides np::gpu::is_available(), device_count(), try_matmul<float/double>,
 * pinned memory helpers, and async stream abstraction.
 *
 * Integration: linalg::dot dispatches to gpu::try_matmul for large contiguous
 * float GEMMs (rows*cols*k > 1M) when NP_ENABLE_GPU is on; accelerator::GPUAccelerator
 * and tensor::HopperBackend delegate here; memory::GpuArray uses managed memory
 * when available.
 *
 * Powerful-machine tuning: cache-aware blocking (128), NUMA-friendly OpenMP,
 * AVX2 FMA micro-kernel, huge-page hint, and LTO/native CMake preset.
 *
 * @author Sergio Randriamihoatra
 */
#ifndef NP_GPU_HPP
#define NP_GPU_HPP

#include "api_macros.hpp"
#include "cuda.hpp"
#include "powerful.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(__has_include)
#if __has_include(<dlfcn.h>) && !defined(_WIN32)
#include <dlfcn.h>
#endif
#if __has_include(<omp.h>)
#include <omp.h>
#endif
#endif

#if defined(NP_ENABLE_CUDA) && __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define NP_GPU_HAS_CUDA_RUNTIME 1
#endif
#if defined(NP_ENABLE_HIP) && __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define NP_GPU_HAS_HIP_RUNTIME 1
#endif

namespace np::gpu
{

enum class Backend : std::uint8_t
{
    None = 0,
    CudaDriver = 1,
    OpenMPTarget = 2,
    CudaRuntime = 3,
    HipRuntime = 4
};

struct DeviceInfo
{
    Backend backend = Backend::None;
    int id = -1;
    std::string name = "none";
    std::size_t total_mem = 0;
    bool available = false;
};

namespace detail
{

inline bool probe_cuda_driver(int *out_count = nullptr) noexcept
{
#if defined(_WIN32)
    return false;
#else
#if defined(__has_include) && __has_include(<dlfcn.h>)
    void *h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!h)
        h = dlopen("libcuda.so", RTLD_LAZY);
    if (!h)
        return false;
    using cuInit_t = int (*)(unsigned int);
    using cuDeviceGetCount_t = int (*)(int *);
    auto cuInit = reinterpret_cast<cuInit_t>(dlsym(h, "cuInit"));
    auto cuDeviceGetCount = reinterpret_cast<cuDeviceGetCount_t>(dlsym(h, "cuDeviceGetCount"));
    bool ok = false;
    if (cuInit && cuDeviceGetCount)
    {
        if (cuInit(0) == 0)
        {
            int cnt = 0;
            if (cuDeviceGetCount(&cnt) == 0)
            {
                ok = cnt > 0;
                if (out_count)
                    *out_count = cnt;
            }
        }
    }
    dlclose(h);
    return ok;
#else
    (void)out_count;
    return false;
#endif
#endif
}

inline bool probe_openmp_target(int *out_count = nullptr) noexcept
{
#if defined(_OPENMP) && defined(__has_include) && __has_include(<omp.h>)
#if defined(NP_ENABLE_GPU) || defined(NP_ENABLE_OPENMP)
    int cnt = 0;
#if defined(_OPENMP)
    cnt = omp_get_num_devices();
#endif
    if (out_count)
        *out_count = cnt;
    return cnt > 0;
#else
    (void)out_count;
    return false;
#endif
#else
    (void)out_count;
    return false;
#endif
}

// Cached probe to avoid repeated dlopen/cuInit on hot path (thread-safe static init in C++11+)
inline bool probe_cuda_driver_cached(int *out_count = nullptr) noexcept
{
    static const auto cached = [] {
        int cnt = 0;
        bool ok = probe_cuda_driver(&cnt);
        return std::pair<bool, int>{ok, cnt};
    }();
    if (out_count)
        *out_count = cached.second;
    return cached.first;
}

inline void cpu_gemm_blocked_f32(const float *a, const float *b, float *c, std::size_t M, std::size_t N, std::size_t K)
{
    constexpr std::size_t BLOCK = 128;
    for (std::size_t i = 0; i < M * N; ++i)
        c[i] = 0.0f;

#if defined(_OPENMP) && defined(NP_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t ii = 0; ii < M; ii += BLOCK)
    {
        for (std::size_t jj = 0; jj < N; jj += BLOCK)
        {
            for (std::size_t pp = 0; pp < K; pp += BLOCK)
            {
                std::size_t i_max = std::min(ii + BLOCK, M);
                std::size_t j_max = std::min(jj + BLOCK, N);
                std::size_t p_max = std::min(pp + BLOCK, K);
                for (std::size_t i = ii; i < i_max; ++i)
                {
                    for (std::size_t p = pp; p < p_max; ++p)
                    {
                        float av = a[i * K + p];
                        std::size_t j = jj;
#if defined(__AVX512F__) && defined(__FMA__)
                        for (; j + 15 < j_max; j += 16)
                        {
                            __m512 bv = _mm512_loadu_ps(b + p * N + j);
                            __m512 cv = _mm512_loadu_ps(c + i * N + j);
                            __m512 avb = _mm512_set1_ps(av);
                            cv = _mm512_fmadd_ps(avb, bv, cv);
                            _mm512_storeu_ps(c + i * N + j, cv);
                        }
#elif defined(__AVX2__) && defined(__FMA__)
                        for (; j + 7 < j_max; j += 8)
                        {
                            __m256 bv = _mm256_loadu_ps(b + p * N + j);
                            __m256 cv = _mm256_loadu_ps(c + i * N + j);
                            __m256 avb = _mm256_set1_ps(av);
                            cv = _mm256_fmadd_ps(avb, bv, cv);
                            _mm256_storeu_ps(c + i * N + j, cv);
                        }
#endif
                        for (; j < j_max; ++j)
                            c[i * N + j] += av * b[p * N + j];
                    }
                }
            }
        }
    }
#else
    for (std::size_t ii = 0; ii < M; ii += BLOCK)
    {
        for (std::size_t jj = 0; jj < N; jj += BLOCK)
        {
            for (std::size_t pp = 0; pp < K; pp += BLOCK)
            {
                std::size_t i_max = std::min(ii + BLOCK, M);
                std::size_t j_max = std::min(jj + BLOCK, N);
                std::size_t p_max = std::min(pp + BLOCK, K);
                for (std::size_t i = ii; i < i_max; ++i)
                {
                    for (std::size_t p = pp; p < p_max; ++p)
                    {
                        float av = a[i * K + p];
                        std::size_t j = jj;
#if defined(__AVX512F__) && defined(__FMA__)
                        for (; j + 15 < j_max; j += 16)
                        {
                            __m512 bv = _mm512_loadu_ps(b + p * N + j);
                            __m512 cv = _mm512_loadu_ps(c + i * N + j);
                            __m512 avb = _mm512_set1_ps(av);
                            cv = _mm512_fmadd_ps(avb, bv, cv);
                            _mm512_storeu_ps(c + i * N + j, cv);
                        }
#elif defined(__AVX2__) && defined(__FMA__)
                        for (; j + 7 < j_max; j += 8)
                        {
                            __m256 bv = _mm256_loadu_ps(b + p * N + j);
                            __m256 cv = _mm256_loadu_ps(c + i * N + j);
                            __m256 avb = _mm256_set1_ps(av);
                            cv = _mm256_fmadd_ps(avb, bv, cv);
                            _mm256_storeu_ps(c + i * N + j, cv);
                        }
#endif
                        for (; j < j_max; ++j)
                            c[i * N + j] += av * b[p * N + j];
                    }
                }
            }
        }
    }
#endif
}

inline void cpu_gemm_blocked_f64(const double *a, const double *b, double *c, std::size_t M, std::size_t N,
                                 std::size_t K)
{
    constexpr std::size_t BLOCK = 96;
    for (std::size_t i = 0; i < M * N; ++i)
        c[i] = 0.0;

#if defined(_OPENMP) && defined(NP_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t ii = 0; ii < M; ii += BLOCK)
    {
        for (std::size_t jj = 0; jj < N; jj += BLOCK)
        {
            for (std::size_t pp = 0; pp < K; pp += BLOCK)
            {
                std::size_t i_max = std::min(ii + BLOCK, M);
                std::size_t j_max = std::min(jj + BLOCK, N);
                std::size_t p_max = std::min(pp + BLOCK, K);
                for (std::size_t i = ii; i < i_max; ++i)
                    for (std::size_t p = pp; p < p_max; ++p)
                    {
                        double av = a[i * K + p];
                        std::size_t j = jj;
#if defined(__AVX512F__) && defined(__FMA__)
                        for (; j + 7 < j_max; j += 8)
                        {
                            __m512d bv = _mm512_loadu_pd(b + p * N + j);
                            __m512d cv = _mm512_loadu_pd(c + i * N + j);
                            __m512d avb = _mm512_set1_pd(av);
                            cv = _mm512_fmadd_pd(avb, bv, cv);
                            _mm512_storeu_pd(c + i * N + j, cv);
                        }
#endif
                        for (; j < j_max; ++j)
                            c[i * N + j] += av * b[p * N + j];
                    }
            }
        }
    }
#else
    for (std::size_t ii = 0; ii < M; ii += BLOCK)
        for (std::size_t jj = 0; jj < N; jj += BLOCK)
            for (std::size_t pp = 0; pp < K; pp += BLOCK)
            {
                std::size_t i_max = std::min(ii + BLOCK, M);
                std::size_t j_max = std::min(jj + BLOCK, N);
                std::size_t p_max = std::min(pp + BLOCK, K);
                for (std::size_t i = ii; i < i_max; ++i)
                    for (std::size_t p = pp; p < p_max; ++p)
                    {
                        double av = a[i * K + p];
                        std::size_t j = jj;
#if defined(__AVX512F__) && defined(__FMA__)
                        for (; j + 7 < j_max; j += 8)
                        {
                            __m512d bv = _mm512_loadu_pd(b + p * N + j);
                            __m512d cv = _mm512_loadu_pd(c + i * N + j);
                            __m512d avb = _mm512_set1_pd(av);
                            cv = _mm512_fmadd_pd(avb, bv, cv);
                            _mm512_storeu_pd(c + i * N + j, cv);
                        }
#endif
                        for (; j < j_max; ++j)
                            c[i * N + j] += av * b[p * N + j];
                    }
            }
#endif
}

} // namespace detail

NP_NODISCARD inline std::vector<DeviceInfo> enumerate_devices() noexcept
{
    std::vector<DeviceInfo> out;
    int cnt = 0;
    if (detail::probe_cuda_driver_cached(&cnt) && cnt > 0)
    {
        for (int i = 0; i < cnt; ++i)
            out.push_back(DeviceInfo{Backend::CudaDriver, i, "CUDA device " + std::to_string(i), 0, true});
    }
    if (detail::probe_openmp_target(&cnt) && cnt > 0)
    {
        for (int i = 0; i < cnt; ++i)
            out.push_back(DeviceInfo{Backend::OpenMPTarget, i, "OpenMP target " + std::to_string(i), 0, true});
    }
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    {
        int c = 0;
        if (cudaGetDeviceCount(&c) == cudaSuccess && c > 0)
            for (int i = 0; i < c; ++i)
                out.push_back(DeviceInfo{Backend::CudaRuntime, i, "CUDA runtime " + std::to_string(i), 0, true});
    }
#endif
#if defined(NP_GPU_HAS_HIP_RUNTIME)
    {
        int c = 0;
        if (hipGetDeviceCount(&c) == hipSuccess && c > 0)
            for (int i = 0; i < c; ++i)
                out.push_back(DeviceInfo{Backend::HipRuntime, i, "HIP runtime " + std::to_string(i), 0, true});
    }
#endif
    if (out.empty())
        out.push_back(DeviceInfo{Backend::None, -1, "CPU fallback", 0, false});
    return out;
}

NP_NODISCARD inline bool is_available() noexcept
{
    int c = 0;
    if (detail::probe_cuda_driver_cached(&c) && c > 0)
        return true;
    if (detail::probe_openmp_target(&c) && c > 0)
        return true;
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    {
        int cc = 0;
        if (cudaGetDeviceCount(&cc) == cudaSuccess && cc > 0)
            return true;
    }
#endif
#if defined(NP_GPU_HAS_HIP_RUNTIME)
    {
        int cc = 0;
        if (hipGetDeviceCount(&cc) == hipSuccess && cc > 0)
            return true;
    }
#endif
    return false;
}

NP_NODISCARD inline int device_count() noexcept
{
    int driver_cnt = 0, omp_cnt = 0, runtime_cnt = 0;
    bool has_driver = detail::probe_cuda_driver_cached(&driver_cnt);
    bool has_omp = detail::probe_openmp_target(&omp_cnt);
    (void)has_driver;
    (void)has_omp;
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    if (cudaGetDeviceCount(&runtime_cnt) != cudaSuccess)
        runtime_cnt = 0;
#endif
    // Avoid double-counting same physical GPUs when both driver and runtime are present
    int cuda_total = 0;
    if (driver_cnt > 0 && runtime_cnt > 0)
        cuda_total = std::max(driver_cnt, runtime_cnt);
    else
        cuda_total = driver_cnt + runtime_cnt;
    return cuda_total + omp_cnt;
}

NP_NODISCARD inline Backend preferred_backend() noexcept
{
    int c = 0;
    if (detail::probe_cuda_driver_cached(&c) && c > 0)
        return Backend::CudaDriver;
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    if (cudaGetDeviceCount(&c) == cudaSuccess && c > 0)
        return Backend::CudaRuntime;
#endif
    if (detail::probe_openmp_target(&c) && c > 0)
        return Backend::OpenMPTarget;
    return Backend::None;
}

template <typename T>
NP_NODISCARD inline bool try_matmul(const T *a, const T *b, T *c, std::size_t M, std::size_t N, std::size_t K) noexcept
{
    if (M == 0 || N == 0 || K == 0 || !a || !b || !c)
        return false;
    // Use 128-bit to avoid size_t overflow on large dims
    __int128 prod = static_cast<__int128>(M) * static_cast<__int128>(N) * static_cast<__int128>(K);
    __int128 mn = static_cast<__int128>(M) * static_cast<__int128>(N);
    if (prod < 1000000 && mn < 65536)
        return false;
    if (!is_available())
        return false;

#if defined(_OPENMP) && (defined(NP_ENABLE_GPU) || defined(NP_ENABLE_OPENMP))
    if (detail::probe_openmp_target())
    {
#if defined(NP_ENABLE_GPU)
        try
        {
#pragma omp target data map(to : a[0 : M * K], b[0 : K * N]) map(from : c[0 : M * N])
            {
#pragma omp target teams distribute parallel for collapse(2) if (M * N > 4096)
                for (std::size_t i = 0; i < M; ++i)
                {
                    for (std::size_t j = 0; j < N; ++j)
                    {
                        T sum = T{0};
                        for (std::size_t p = 0; p < K; ++p)
                            sum += a[i * K + p] * b[p * N + j];
                        c[i * N + j] = sum;
                    }
                }
            }
            return true;
        }
        catch (...)
        {
            return false;
        }
#else
        (void)a;
        (void)b;
        (void)c;
        return false;
#endif
    }
#endif
    return false;
}

template <typename T>
inline void cpu_matmul(const T *a, const T *b, T *c, std::size_t M, std::size_t N, std::size_t K) noexcept
{
    if constexpr (std::is_same_v<T, float>)
        detail::cpu_gemm_blocked_f32(a, b, c, M, N, K);
    else if constexpr (std::is_same_v<T, double>)
        detail::cpu_gemm_blocked_f64(a, b, c, M, N, K);
    else
    {
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t j = 0; j < N; ++j)
            {
                T sum = T{0};
                for (std::size_t p = 0; p < K; ++p)
                    sum += a[i * K + p] * b[p * N + j];
                c[i * N + j] = sum;
            }
    }
}

template <typename T>
inline void matmul(const T *a, const T *b, T *c, std::size_t M, std::size_t N, std::size_t K) noexcept
{
    if (!try_matmul(a, b, c, M, N, K))
        cpu_matmul(a, b, c, M, N, K);
}

// ── FFT GPU offload (cuFFT dlopen + OpenMP) ──────────────────────────
namespace fft_detail
{
inline bool probe_cufft() noexcept
{
#if defined(_WIN32)
    return false;
#else
#if defined(__has_include) && __has_include(<dlfcn.h>)
    void *h = dlopen("libcufft.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcufft.so.11", RTLD_LAZY);
    if (!h)
        return false;
    dlclose(h);
    return is_available();
#else
    return false;
#endif
#endif
}
} // namespace fft_detail

template <typename Cplx>
NP_NODISCARD inline bool try_fft(const Cplx *in, Cplx *out, std::size_t N, bool inverse) noexcept
{
    if (N < tune::fft_threshold())
        return false; // CPU radix2 already very fast for small N (tune::fft_threshold)
    if (!is_available())
        return false;
#if defined(NP_GPU_HAS_CUDA_RUNTIME) && defined(NP_ENABLE_CUDA)
    if (fft_detail::probe_cufft())
    {
        // cuFFT path would be via dlopen cufftPlan1d/cufftExecZ2Z
        // For header-only, we fall through to OpenMP target as portable
        // (real cuFFT would require linking -lcufft, which we avoid here)
    }
#endif
#if defined(_OPENMP) && defined(NP_ENABLE_GPU)
    if (detail::probe_openmp_target())
    {
        try
        {
            // Naive DFT offload for demonstration – radix2 would be better
            // Use OpenMP target to compute DFT in parallel (O(N^2) but parallel)
            // For benchmark, we offload the existing radix2 butterflies via target
            // Here we just do a simple parallel DFT for large N when GPU is present
            // Fallback to CPU if N is not power of two
            if ((N & (N - 1)) != 0)
                return false;
#pragma omp target data map(to : in[0 : N]) map(from : out[0 : N])
            {
#pragma omp target teams distribute parallel for
                for (std::size_t k = 0; k < N; ++k)
                {
                    Cplx sum{0, 0};
                    for (std::size_t n = 0; n < N; ++n)
                    {
                        double angle = (inverse ? 1 : -1) * 2 * 3.141592653589793 * double(k * n) / double(N);
                        Cplx w{std::cos(angle), std::sin(angle)};
                        sum += in[n] * w;
                    }
                    out[k] = sum;
                }
            }
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
#endif
    (void)in;
    (void)out;
    (void)inverse;
    return false;
}

inline void *pinned_alloc(std::size_t bytes) noexcept
{
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    void *p = nullptr;
    if (cudaMallocHost(&p, bytes) == cudaSuccess)
        return p;
#endif
#if defined(__linux__)
    void *p = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
    if (p)
    {
#ifdef MADV_HUGEPAGE
        madvise(p, bytes, MADV_HUGEPAGE);
#endif
    }
    return p;
#else
    return std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
#endif
}

inline void pinned_free(void *p, std::size_t bytes) noexcept
{
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    if (p && cudaFreeHost(p) == cudaSuccess)
        return;
    (void)bytes;
#endif
#if defined(__linux__)
    (void)bytes;
#endif
    std::free(p);
}

// Unified managed memory via dlopen cudaMallocManaged (no link-time dep)
inline void *managed_alloc(std::size_t bytes) noexcept
{
#if defined(_WIN32)
    return pinned_alloc(bytes);
#else
#if defined(__has_include) && __has_include(<dlfcn.h>)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (h)
    {
        using cudaMallocManaged_t = int (*)(void **, std::size_t, unsigned int);
        auto sym = reinterpret_cast<cudaMallocManaged_t>(dlsym(h, "cudaMallocManaged"));
        if (!sym)
            sym = reinterpret_cast<cudaMallocManaged_t>(dlsym(h, "cuMemAllocManaged"));
        if (sym)
        {
            void *ptr = nullptr;
            if (sym(&ptr, bytes, 0x01) == 0 && ptr) // 0x01 = cudaMemAttachGlobal
            {
                dlclose(h);
                return ptr;
            }
        }
        dlclose(h);
    }
#endif
    return pinned_alloc(bytes);
#endif
}

inline void managed_free(void *p) noexcept
{
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    // Try cudaFree (runtime API)
    if (p && cudaFree(p) == cudaSuccess)
        return;
#endif
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    // Try cudaFree via dlopen (runtime)
    {
        void *h = dlopen("libcudart.so", RTLD_LAZY);
        if (!h)
            h = dlopen("libcudart.so.12", RTLD_LAZY);
        if (h)
        {
            using cudaFree_t = int (*)(void *);
            auto sym = reinterpret_cast<cudaFree_t>(dlsym(h, "cudaFree"));
            if (sym && sym(p) == 0)
            {
                dlclose(h);
                return;
            }
            dlclose(h);
        }
    }
    // Try driver API cuMemFree (mirrors managed_alloc fallback to cuMemAllocManaged)
    {
        void *h = dlopen("libcuda.so.1", RTLD_LAZY);
        if (!h)
            h = dlopen("libcuda.so", RTLD_LAZY);
        if (h)
        {
            using cuMemFree_t = int (*)(void *);
            auto sym = reinterpret_cast<cuMemFree_t>(dlsym(h, "cuMemFree"));
            if (!sym)
                sym = reinterpret_cast<cuMemFree_t>(dlsym(h, "cuMemFree_v2"));
            if (sym && sym(p) == 0)
            {
                dlclose(h);
                return;
            }
            dlclose(h);
        }
    }
#endif
    pinned_free(p, 0);
}

// Forward declare for batch_matmul (defined later)
template <typename T>
NP_NODISCARD inline bool try_graph_batch_matmul(const std::vector<const T *> &As, const std::vector<const T *> &Bs,
                                                std::vector<T *> &Cs, std::size_t M, std::size_t N,
                                                std::size_t K) noexcept;

// ── Async streams & batch for powerful multi-GPU ────────────────────────
struct Stream
{
    int device = 0;
    int id = 0;
    // For CPU fallback, use ThreadPool; for GPU, OpenMP target nowait
    template <typename Fn> auto enqueue(Fn &&fn) -> std::future<std::invoke_result_t<Fn>>
    {
        using R = std::invoke_result_t<Fn>;
        // Use async with launch::async to overlap with caller; on powerful
        // machines this maps to ThreadPool or GPU stream
#if defined(NP_ENABLE_GPU) && defined(_OPENMP)
        if (is_available())
        {
            // GPU path: use OpenMP target task with heap-allocated packaged_task
            // to avoid dangling reference when task is deferred.
            auto pt = std::make_shared<std::packaged_task<R()>>(std::forward<Fn>(fn));
            auto fut = pt->get_future();
#pragma omp task firstprivate(pt)
            {
                (*pt)();
            }
            return fut;
        }
#endif
        return std::async(std::launch::async, std::forward<Fn>(fn));
    }
};

NP_NODISCARD inline std::vector<Stream> make_streams(int n = 4) noexcept
{
    int devs = device_count();
    if (devs == 0)
        devs = 1;
    std::vector<Stream> s;
    s.reserve(n);
    for (int i = 0; i < n; ++i)
        s.push_back(Stream{i % devs, i});
    return s;
}

// Batch GEMM: vector of (A,B,C) where each is MxK, KxN, MxN
template <typename T>
inline void batch_matmul(const std::vector<const T *> &As, const std::vector<const T *> &Bs, std::vector<T *> &Cs,
                         std::size_t M, std::size_t N, std::size_t K) noexcept
{
    std::size_t batch = As.size();
    if (batch == 0)
        return;
    // CUDA 12 graphs: try to capture batch as graph for fast replay (e.g., transformer)
    if (try_graph_batch_matmul(As, Bs, Cs, M, N, K))
        return;
    int devs = device_count();
    if (devs == 0)
        devs = 1;
    // Shard batch across devices/streams
    auto streams = make_streams(std::min<std::size_t>(batch, devs * 2));
#if defined(NP_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < batch; ++b)
    {
        int s = b % streams.size();
        (void)s;
        matmul(As[b], Bs[b], Cs[b], M, N, K);
    }
#else
    for (std::size_t b = 0; b < batch; ++b)
        matmul(As[b], Bs[b], Cs[b], M, N, K);
#endif
}

// Overlap CPU and GPU: if GPU available, run half batch on GPU, half on CPU
template <typename T>
inline void hybrid_batch_matmul(const std::vector<const T *> &As, const std::vector<const T *> &Bs,
                                std::vector<T *> &Cs, std::size_t M, std::size_t N, std::size_t K) noexcept
{
    std::size_t batch = As.size();
    if (batch == 0)
        return;
    if (!is_available() || batch < 4)
    {
        batch_matmul(As, Bs, Cs, M, N, K);
        return;
    }
    std::size_t gpu_batch = batch / 2;
    std::vector<const T *> As_gpu(As.begin(), As.begin() + gpu_batch);
    std::vector<const T *> Bs_gpu(Bs.begin(), Bs.begin() + gpu_batch);
    std::vector<T *> Cs_gpu(Cs.begin(), Cs.begin() + gpu_batch);
    std::vector<const T *> As_cpu(As.begin() + gpu_batch, As.end());
    std::vector<const T *> Bs_cpu(Bs.begin() + gpu_batch, Bs.end());
    std::vector<T *> Cs_cpu(Cs.begin() + gpu_batch, Cs.end());
    auto fut = std::async(std::launch::async, [&] { batch_matmul(As_gpu, Bs_gpu, Cs_gpu, M, N, K); });
    batch_matmul(As_cpu, Bs_cpu, Cs_cpu, M, N, K);
    fut.wait();
}

// Multi-GPU sharding for very large single GEMM (e.g., 4096) — split M across devices
template <typename T>
inline void sharded_matmul(const T *a, const T *b, T *c, std::size_t M, std::size_t N, std::size_t K) noexcept
{
    int devs = device_count();
    if (devs <= 1 || M < 1024 || M * N * K < 64ULL * 1024 * 1024)
    {
        matmul(a, b, c, M, N, K);
        return;
    }
    std::size_t rows_per_dev = (M + devs - 1) / devs;
#if defined(NP_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
    for (int d = 0; d < devs; ++d)
    {
        std::size_t start = d * rows_per_dev;
        std::size_t end = std::min(start + rows_per_dev, M);
        if (start >= end)
            continue;
        // Each shard is (end-start) x N
        matmul(a + start * K, b, c + start * N, end - start, N, K);
    }
#else
    for (int d = 0; d < devs; ++d)
    {
        std::size_t start = d * rows_per_dev;
        std::size_t end = std::min(start + rows_per_dev, M);
        if (start >= end)
            continue;
        matmul(a + start * K, b, c + start * N, end - start, N, K);
    }
#endif
}

// ── CUDA 12/13 new features (header-only, dlopen) ────────────────────────
NP_NODISCARD inline bool is_blackwell() noexcept
{
    return cuda::is_blackwell(10);
}
NP_NODISCARD inline bool has_fp8_tensor() noexcept
{
    return cuda::has_fp8_tensor();
}
NP_NODISCARD inline bool has_fp4_tensor() noexcept
{
    return cuda::has_fp4_tensor();
}
NP_NODISCARD inline int cuda_driver_version() noexcept
{
    return cuda::driver_version();
}
NP_NODISCARD inline int cuda_runtime_version() noexcept
{
    return cuda::runtime_version();
}

// Stream-ordered async alloc (CUDA 11.2+): try cudaMallocAsync, fallback to pinned
NP_NODISCARD inline void *async_alloc(std::size_t bytes, void *stream = nullptr) noexcept
{
    if (void *p = cuda::malloc_async(bytes, stream))
        return p;
    return pinned_alloc(bytes);
}
inline void async_free(void *p, void *stream = nullptr) noexcept
{
    if (cuda::free_async(p, stream) == 0)
        return;
    pinned_free(p, 0);
}

// Graph-captured batch GEMM (CUDA 10+): try to capture batch as graph for replay
template <typename T>
NP_NODISCARD inline bool try_graph_batch_matmul(const std::vector<const T *> &As, const std::vector<const T *> &Bs,
                                                std::vector<T *> &Cs, std::size_t M, std::size_t N,
                                                std::size_t K) noexcept
{
    if (As.empty() || !is_available())
        return false;
    // Use cuda::try_cuda_graph_batch_matmul as probe (dlopen); fallback to streams
    if (cuda::try_cuda_graph_batch_matmul(static_cast<const void *>(As[0]), static_cast<const void *>(Bs[0]),
                                          static_cast<void *>(Cs[0]), M, N, K, As.size()))
        return true;
    return false;
}

} // namespace np::gpu

#endif // NP_GPU_HPP
