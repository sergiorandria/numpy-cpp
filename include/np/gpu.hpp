/**
 * @file gpu.hpp
 * @brief Unified GPU abstraction — CUDA compute via dlopen, OpenMP-target
 *        fallback, CPU blocked-GEMM fallback. Header-only, no hard link dep.
 *
 * What actually runs where (verified against the implementation):
 *  - Detection: CUDA driver via dlopen("libcuda.so.1") + cuInit/cuDeviceGetCount
 *    (cached); OpenMP-target via omp_get_num_devices(); HIP runtime only when
 *    built with NP_ENABLE_HIP and its headers.
 *  - Compute: float/double GEMM runs on CUDA via dlopen'd cuBLAS
 *    (try_cuda_matmul) when libcublas + libcudart are present; otherwise
 *    OpenMP-target naive GEMM (compiler offload, same physical GPUs); else
 *    CPU cache-blocked GEMM. Large FFTs run on CUDA via dlopen'd cuFFT
 *    (try_cuda_fft) when libcufft is present.
 *  - Not compute: HIP has detection only, no kernels. Batch GEMM dispatches
 *    per entry over streams — no graph-capture replay path exists, by
 *    decision (capture would need per-key exec caches with node updates for
 *    marginal gain over already-async GEMMs). Cooperative groups / wgmma /
 *    TMA have no wrappers (need compiled device code, out of reach for a
 *    dlopen design).
 *  - Failures: public try_* keep bool signatures; the reason is observable
 *    via last_error() (CudaStatus), set on every CUDA-path outcome.
 *
 * Integration: linalg::dot dispatches to gpu::try_matmul for large contiguous
 * float GEMMs; accelerator::GPUAccelerator and tensor::GpuFp32Backend delegate
 * here; memory tags are host-resident (use pinned_alloc/managed_alloc here
 * directly for real pinned/managed buffers); fft_core calls
 * try_fft with an explicit caller-side scale (try_fft returns UNSCALED
 * directional DFT, matching both the CPU radix-2 path and cuFFT).
 *
 * @author Sergio Randriamihoatra
 */
#ifndef NP_GPU_HPP
#define NP_GPU_HPP

#include "api_macros.hpp"
#include "cuda.hpp"
#include "powerful.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
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

// NP_ENABLE_CUDA is a hard requirement on the CUDA toolkit headers.
// Driver-only probing (no headers) remains available via NP_ENABLE_GPU.
#if defined(NP_ENABLE_CUDA) && !defined(NP_GPU_HAS_CUDA_RUNTIME)
#error "NP_ENABLE_CUDA requires <cuda_runtime.h> (CUDA toolkit); else use -DNP_ENABLE_GPU."
#endif

namespace np::gpu
{

/// @brief Machine-readable reason for the last CUDA-path outcome.
///
/// Public try_* functions keep their bool signatures for backward
/// compatibility; this status (thread-local, so concurrent callers do not
/// clobber each other) tells *why* a call returned false. Set on every
/// CUDA-path exit: Ok on success, a specific reason on failure. Untouched
/// when no CUDA path is attempted (e.g. below-size-threshold early-outs).
enum class CudaStatus : std::uint8_t
{
    Ok = 0,          ///< Last CUDA operation succeeded (or none attempted yet).
    NoDevice,        ///< A CUDA device call failed (no device / bad ordinal).
    NoLibrary,       ///< libcudart/libcublas/libcufft (or a symbol) is absent.
    OutOfMemory,     ///< Device allocation failed (cuda/CL status 2/3).
    InvalidValue,    ///< Null pointer, zero dimension, or oversized dim.
    UnsupportedType, ///< Type or architecture the backend cannot execute.
    LaunchFailed,    ///< Kernel/BLAS/FFT execution reported failure.
    NotImplemented,  ///< Recognized request with no implementation yet.
    DriverError      ///< Any other CUDA error (see cuda::rt_error_string).
};

NP_NODISCARD inline CudaStatus &last_error_slot() noexcept
{
    thread_local CudaStatus s = CudaStatus::Ok;
    return s;
}

/// @brief Reason for the last CUDA-path outcome on this thread.
NP_NODISCARD inline CudaStatus last_error() noexcept
{
    return last_error_slot();
}

NP_NODISCARD inline const char *last_error_string() noexcept
{
    switch (last_error_slot())
    {
    case CudaStatus::Ok:
        return "ok";
    case CudaStatus::NoDevice:
        return "no CUDA device";
    case CudaStatus::NoLibrary:
        return "CUDA library/symbol absent (no toolkit/driver userspace?)";
    case CudaStatus::OutOfMemory:
        return "CUDA out of memory";
    case CudaStatus::InvalidValue:
        return "invalid argument to CUDA path";
    case CudaStatus::UnsupportedType:
        return "type/architecture unsupported by CUDA path";
    case CudaStatus::LaunchFailed:
        return "CUDA execution failed";
    case CudaStatus::NotImplemented:
        return "CUDA path not implemented";
    case CudaStatus::DriverError:
        return "CUDA driver/runtime error";
    }
    return "unknown CUDA status";
}

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

// True when a CUDA backend (runtime or driver) reports devices.
// OpenMP-target offload is only a fallback when this returns false.
inline bool has_cuda_backend() noexcept
{
    int c = 0;
    if (probe_cuda_driver_cached(&c) && c > 0)
        return true;
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    if (cudaGetDeviceCount(&c) == cudaSuccess && c > 0)
        return true;
#endif
    return false;
}

// ── CUDA-path internals ─────────────────────────────────────────────────
namespace cuda_detail
{
inline void set_status(CudaStatus s) noexcept
{
    last_error_slot() = s;
}

NP_NODISCARD inline CudaStatus blas_status_to(int code) noexcept
{
    switch (code)
    {
    case cuda::kCublasSuccess:
        return CudaStatus::Ok;
    case cuda::kCublasAllocFailed:
        return CudaStatus::OutOfMemory;
    case cuda::kCublasExecutionFailed:
        return CudaStatus::LaunchFailed;
    case cuda::kCublasArchMismatch:
        return CudaStatus::UnsupportedType;
    case cuda::kCublasInvalidValue:
        return CudaStatus::InvalidValue;
    case cuda::kCublasNotSupported:
        return CudaStatus::NotImplemented;
    default:
        return CudaStatus::DriverError;
    }
}

NP_NODISCARD inline CudaStatus cufft_status_to(int code) noexcept
{
    switch (code)
    {
    case cuda::kCufftSuccess:
        return CudaStatus::Ok;
    case cuda::kCufftAllocFailed:
        return CudaStatus::OutOfMemory;
    case cuda::kCufftExecFailed:
        return CudaStatus::LaunchFailed;
    case cuda::kCufftInvalidType:
    case cuda::kCufftInvalidSize:
        return CudaStatus::UnsupportedType;
    case cuda::kCufftInvalidValue:
        return CudaStatus::InvalidValue;
    default:
        return CudaStatus::DriverError;
    }
}

NP_NODISCARD inline CudaStatus rt_status_to(int code) noexcept
{
    if (code == cuda::kCudaSuccess)
        return CudaStatus::Ok;
    if (code == cuda::kCudaErrorMemoryAllocation)
        return CudaStatus::OutOfMemory;
    if (code == cuda::kCudaErrorNoDevice)
        return CudaStatus::NoDevice;
    return CudaStatus::DriverError;
}

// RAII device allocation (no raw new/delete; freed even on early return).
struct device_buffer
{
    void *p = nullptr;
    int code = -1; // raw cudaMalloc status for error mapping
    explicit device_buffer(std::size_t bytes) noexcept
    {
        code = cuda::rt_malloc(&p, bytes);
        if (code != cuda::kCudaSuccess)
            p = nullptr;
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    ~device_buffer() noexcept
    {
        if (p != nullptr)
            cuda::rt_free(p);
    }
    NP_NODISCARD explicit operator bool() const noexcept
    {
        return p != nullptr;
    }
};

// Per-thread, per-device cuBLAS handles. Handles are expensive to create and
// not safe to share across threads without locking, so each thread keeps its
// own array (bounded by cuda::kMaxCachedDevices; devices beyond that report
// InvalidValue). Destroyed at thread exit; the dlopen'd libraries themselves
// outlive all threads (never dlclosed).
struct blas_thread_cache
{
    std::array<void *, cuda::kMaxCachedDevices> handles{};
    blas_thread_cache() = default;
    blas_thread_cache(const blas_thread_cache &) = delete;
    blas_thread_cache &operator=(const blas_thread_cache &) = delete;
    ~blas_thread_cache() noexcept
    {
        for (void *h : handles)
        {
            if (h != nullptr)
                cuda::blas_destroy(h);
        }
    }
};

NP_NODISCARD inline void *blas_handle_for(int device) noexcept
{
    if (device < 0 || device >= cuda::kMaxCachedDevices)
        return nullptr;
    thread_local blas_thread_cache cache;
    void *&slot = cache.handles[static_cast<std::size_t>(device)];
    if (slot == nullptr)
    {
        void *h = nullptr;
        if (cuda::blas_create(&h) != cuda::kCublasSuccess || h == nullptr)
            return nullptr;
        slot = h;
    }
    return slot;
}

// Host callback thunk for Stream::enqueue's CUDA path (cudaLaunchHostFunc
// takes a C function pointer + void*; the heap-allocated thunk carries the
// packaged_task and deletes itself after running).
template <typename R> struct host_thunk
{
    std::shared_ptr<std::packaged_task<R()>> task;
};

template <typename R> void host_trampoline(void *arg) noexcept
{
    auto *th = static_cast<host_thunk<R> *>(arg);
    if (th == nullptr)
        return;
    try
    {
        (*th->task)();
    }
    catch (...)
    {
        // packaged_task already captures user exceptions into the future;
        // this guards against broken-promise misuse. Never let exceptions
        // escape into the CUDA callback thread.
    }
    delete th;
}

// Bounded per-thread cuFFT plan cache keyed by (n, type, direction).
// Plans are expensive; eviction destroys the oldest slot (FIFO by victim
// index). Array-based so every path stays noexcept (no allocation).
struct cufft_plan_cache
{
    struct entry
    {
        int n = 0;
        int type = 0;
        int dir = 0;
        void *plan = nullptr;
        bool used = false;
    };
    std::array<entry, 8> slots{};
    std::size_t victim = 0;
    cufft_plan_cache() = default;
    cufft_plan_cache(const cufft_plan_cache &) = delete;
    cufft_plan_cache &operator=(const cufft_plan_cache &) = delete;
    ~cufft_plan_cache() noexcept
    {
        for (auto &e : slots)
        {
            if (e.used && e.plan != nullptr)
                cuda::fft_destroy(e.plan);
        }
    }
};

NP_NODISCARD inline void *cufft_plan_for(int n, int type, int dir, CudaStatus &out_status) noexcept
{
    thread_local cufft_plan_cache cache;
    for (auto &e : cache.slots)
    {
        if (e.used && e.n == n && e.type == type && e.dir == dir)
        {
            out_status = CudaStatus::Ok;
            return e.plan;
        }
    }
    void *plan = nullptr;
    const int rc = cuda::fft_plan1d(&plan, n, type, 1);
    if (rc != cuda::kCufftSuccess || plan == nullptr)
    {
        out_status = rc == cuda::kCufftSuccess ? CudaStatus::DriverError : cufft_status_to(rc);
        return nullptr;
    }
    cufft_plan_cache::entry &slot = cache.slots[cache.victim];
    cache.victim = (cache.victim + 1) % cache.slots.size();
    if (slot.used && slot.plan != nullptr)
        cuda::fft_destroy(slot.plan);
    slot = cufft_plan_cache::entry{n, type, dir, plan, true};
    out_status = CudaStatus::Ok;
    return plan;
}
} // namespace cuda_detail

// CUDA GEMM via dlopen'd cuBLAS (cublasSgemm/Dgemm) on `device`.
//
// Row-major transpose trick: the library stores C(M×N) = A(M×K)·B(K×N)
// row-major, while cuBLAS is column-major. Since C_row == C_col^T and
// (A·B)^T == B^T·A^T, we call gemm(N, M, K, dB, dA, dC) with OP_N/OP_N:
// dB (K×N row-major) reads as an N×K column-major matrix with lda=N, and
// likewise dA as K×M with ldb=K, producing dC as N×M column-major == M×N
// row-major with ldc=N. No explicit transpose, no extra memory.
//
// Synchronous contract (matches cpu_matmul): device-synchronized before
// returning true, so `c` is populated on return. Every failure sets
// last_error() and returns false; callers fall back to CPU.
//
// Design note (dlopen vs link): resolved via dlopen to preserve the
// header-only, no-hard-link guarantee — a build using NP_ENABLE_CUDA with
// linked cuBLAS still satisfies these dlopens from the same .so, while
// driver-only and no-GPU builds degrade gracefully with NoLibrary.
template <typename T>
NP_NODISCARD inline bool try_cuda_matmul(const T *a, const T *b, T *c, std::size_t M, std::size_t N, std::size_t K,
                                         int device = 0) noexcept
{
    if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>)
    {
        cuda_detail::set_status(CudaStatus::UnsupportedType);
        return false;
    }
    else
    {
        if (a == nullptr || b == nullptr || c == nullptr || M == 0 || N == 0 || K == 0)
        {
            cuda_detail::set_status(CudaStatus::InvalidValue);
            return false;
        }
        // cuBLAS takes int dims; reject oversized before any narrowing.
        constexpr double kIntMax = static_cast<double>((std::numeric_limits<int>::max)());
        if (static_cast<double>(M) > kIntMax || static_cast<double>(N) > kIntMax || static_cast<double>(K) > kIntMax)
        {
            cuda_detail::set_status(CudaStatus::InvalidValue);
            return false;
        }
        // Guard size_t overflow on byte counts (double is exact to 2^53,
        // far above any allocatable buffer; beyond that it is unallocatable).
        const double bytes_a = static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(sizeof(T));
        const double bytes_b = static_cast<double>(K) * static_cast<double>(N) * static_cast<double>(sizeof(T));
        const double bytes_c = static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(sizeof(T));
        constexpr double kSizeMax = static_cast<double>((std::numeric_limits<std::size_t>::max)());
        if (bytes_a > kSizeMax || bytes_b > kSizeMax || bytes_c > kSizeMax)
        {
            cuda_detail::set_status(CudaStatus::InvalidValue);
            return false;
        }
        if (device < 0 || device >= cuda::kMaxCachedDevices)
        {
            cuda_detail::set_status(CudaStatus::InvalidValue);
            return false;
        }
        if (cuda::detail::cublas_lib() == nullptr || cuda::detail::cudart_lib() == nullptr)
        {
            cuda_detail::set_status(CudaStatus::NoLibrary);
            return false;
        }
        if (cuda::rt_set_device(device) != cuda::kCudaSuccess)
        {
            cuda_detail::set_status(CudaStatus::NoDevice);
            return false;
        }
        void *handle = cuda_detail::blas_handle_for(device);
        if (handle == nullptr)
        {
            cuda_detail::set_status(CudaStatus::DriverError);
            return false;
        }
        cuda_detail::device_buffer da(static_cast<std::size_t>(bytes_a));
        cuda_detail::device_buffer db(static_cast<std::size_t>(bytes_b));
        cuda_detail::device_buffer dc(static_cast<std::size_t>(bytes_c));
        if (!da || !db || !dc)
        {
            const int code = !da ? da.code : (!db ? db.code : dc.code);
            cuda_detail::set_status(code == cuda::kCudaErrorMemoryAllocation ? CudaStatus::OutOfMemory
                                                                             : CudaStatus::DriverError);
            return false;
        }
        if (cuda::rt_memcpy(da.p, a, static_cast<std::size_t>(bytes_a), cuda::kMemcpyHostToDevice) !=
                cuda::kCudaSuccess ||
            cuda::rt_memcpy(db.p, b, static_cast<std::size_t>(bytes_b), cuda::kMemcpyHostToDevice) !=
                cuda::kCudaSuccess)
        {
            cuda_detail::set_status(CudaStatus::DriverError);
            return false;
        }
        // C(M×N) = A·B row-major  <=>  C^T(N×M) = B^T·A^T column-major.
        const int m = static_cast<int>(N);
        const int n = static_cast<int>(M);
        const int k = static_cast<int>(K);
        int rc = -1;
        if constexpr (std::is_same_v<T, float>)
        {
            const float alpha = 1.0f, beta = 0.0f;
            rc = cuda::blas_sgemm(handle, cuda::kCublasOpN, cuda::kCublasOpN, m, n, k, &alpha,
                                  static_cast<const float *>(db.p), m, static_cast<const float *>(da.p), k, &beta,
                                  static_cast<float *>(dc.p), m);
        }
        else
        {
            const double alpha = 1.0, beta = 0.0;
            rc = cuda::blas_dgemm(handle, cuda::kCublasOpN, cuda::kCublasOpN, m, n, k, &alpha,
                                  static_cast<const double *>(db.p), m, static_cast<const double *>(da.p), k, &beta,
                                  static_cast<double *>(dc.p), m);
        }
        if (rc != cuda::kCublasSuccess)
        {
            cuda_detail::set_status(cuda_detail::blas_status_to(rc));
            return false;
        }
        if (cuda::rt_device_synchronize() != cuda::kCudaSuccess)
        {
            cuda_detail::set_status(CudaStatus::DriverError);
            return false;
        }
        if (cuda::rt_memcpy(c, dc.p, static_cast<std::size_t>(bytes_c), cuda::kMemcpyDeviceToHost) !=
            cuda::kCudaSuccess)
        {
            cuda_detail::set_status(CudaStatus::DriverError);
            return false;
        }
        cuda_detail::set_status(CudaStatus::Ok);
        return true;
    }
}

namespace cuda_detail
{
// Large-N FFT on CUDA via dlopen'd cuFFT. Contract matches the CPU path and
// fft_core's use: returns the UNSCALED directional DFT (forward exp(-2πi),
// inverse exp(+2πi)); the caller applies its own scale factor afterwards.
// Only complex<float> (cufftExecC2C) and complex<double> (cufftExecZ2Z) are
// supported; anything else fails at compile time for direct instantiation
// and reports UnsupportedType at runtime. Plans are cached per thread keyed
// by (n, type, direction); device buffers are transient per call.
template <typename Cplx>
NP_NODISCARD inline bool try_cuda_fft(const Cplx *in, Cplx *out, std::size_t n, bool inverse, int device = 0) noexcept
{
    static_assert(std::is_same_v<Cplx, std::complex<float>> || std::is_same_v<Cplx, std::complex<double>>,
                  "try_cuda_fft supports std::complex<float> and std::complex<double> only");
    constexpr bool is_f = std::is_same_v<Cplx, std::complex<float>>;
    static_assert(sizeof(Cplx) == 2 * sizeof(typename Cplx::value_type),
                  "std::complex must be an interleaved pair for cuFFT");
    if (!std::is_same_v<Cplx, std::complex<float>> && !std::is_same_v<Cplx, std::complex<double>>)
    {
        set_status(CudaStatus::UnsupportedType);
        return false;
    }
    if (in == nullptr || out == nullptr || n == 0 || n > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    {
        set_status(CudaStatus::InvalidValue);
        return false;
    }
    const double bytes_d = static_cast<double>(n) * static_cast<double>(sizeof(Cplx));
    if (bytes_d > static_cast<double>((std::numeric_limits<std::size_t>::max)()))
    {
        set_status(CudaStatus::InvalidValue);
        return false;
    }
    if (device < 0 || device >= cuda::kMaxCachedDevices)
    {
        set_status(CudaStatus::InvalidValue);
        return false;
    }
    if (cuda::detail::cufft_lib() == nullptr || cuda::detail::cudart_lib() == nullptr)
    {
        set_status(CudaStatus::NoLibrary);
        return false;
    }
    if (cuda::rt_set_device(device) != cuda::kCudaSuccess)
    {
        set_status(CudaStatus::NoDevice);
        return false;
    }
    const int type = is_f ? cuda::kCufftC2C : cuda::kCufftZ2Z;
    const int dir = inverse ? cuda::kCufftInverse : cuda::kCufftForward;
    CudaStatus plan_status = CudaStatus::DriverError;
    void *plan = cufft_plan_for(static_cast<int>(n), type, dir, plan_status);
    if (plan == nullptr)
    {
        set_status(plan_status);
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(bytes_d);
    device_buffer din(bytes);
    device_buffer dout(bytes);
    if (!din || !dout)
    {
        const int code = !din ? din.code : dout.code;
        set_status(code == cuda::kCudaErrorMemoryAllocation ? CudaStatus::OutOfMemory : CudaStatus::DriverError);
        return false;
    }
    if (cuda::rt_memcpy(din.p, in, bytes, cuda::kMemcpyHostToDevice) != cuda::kCudaSuccess)
    {
        set_status(CudaStatus::DriverError);
        return false;
    }
    const int rc = is_f ? cuda::fft_exec_c2c(plan, din.p, dout.p, dir) : cuda::fft_exec_z2z(plan, din.p, dout.p, dir);
    if (rc != cuda::kCufftSuccess)
    {
        set_status(cufft_status_to(rc));
        return false;
    }
    if (cuda::rt_device_synchronize() != cuda::kCudaSuccess)
    {
        set_status(CudaStatus::DriverError);
        return false;
    }
    if (cuda::rt_memcpy(out, dout.p, bytes, cuda::kMemcpyDeviceToHost) != cuda::kCudaSuccess)
    {
        set_status(CudaStatus::DriverError);
        return false;
    }
    set_status(CudaStatus::Ok);
    return true;
}
} // namespace cuda_detail

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
    // Threshold check in double to avoid size_t overflow without
    // non-standard 128-bit integers (which fail -Wpedantic -Werror).
    // Thresholds (1e6 / 65536) are far below 2^53 so the comparison
    // is exact for small sizes and safely over-threshold for huge ones.
    const double prod = static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double mn = static_cast<double>(M) * static_cast<double>(N);
    if (prod < 1000000.0 && mn < 65536.0)
        return false;
    if (!is_available())
        return false;

    // CUDA first: real cuBLAS compute when the libraries are present. On
    // failure last_error() says why (NoLibrary/NoDevice/OOM/...) and the
    // caller (matmul() below, linalg::dot, ...) falls back to CPU.
    if (detail::has_cuda_backend())
        return detail::try_cuda_matmul(a, b, c, M, N, K);

    // OpenMP-target fallback: only when no CUDA backend was detected.
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

// FFT GPU offload (cuFFT dlopen + OpenMP)
namespace fft_detail
{
// True when libcufft can be opened AND a CUDA device is present. Uses the
// cached library handle (no per-call dlopen); the versioned names cover
// CUDA 11/12/13 plus the Arch (/opt/cuda) toolkit location.
inline bool probe_cufft() noexcept
{
#if defined(_WIN32)
    return false;
#else
    if (cuda::detail::cufft_lib() == nullptr)
        return false;
    return is_available();
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
    // CUDA first: real cuFFT compute when libcufft is present. true means
    // `out` holds the unscaled directional DFT (caller applies its scale);
    // false with last_error() set means "fell through", and fft_core runs
    // the CPU radix-2/Bluestein path. OpenMP offload is skipped while a CUDA
    // backend owns the devices (it would target the same physical GPUs).
    if (detail::has_cuda_backend())
        return detail::cuda_detail::try_cuda_fft(in, out, N, inverse);
    // OpenMP-target fallback: only when no CUDA backend was detected.
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
    {
        // Braced scope: the fallback below declares its own `p`. Both
        // declarations coexisted (latent redeclaration) until a real toolkit
        // build exposed it.
        void *p = nullptr;
        if (cudaMallocHost(&p, bytes) == cudaSuccess)
            return p;
    }
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

// Unified managed memory via dlopen cudaMallocManaged (no link-time dep).
// Uses the cached runtime/driver handles (covers .so.11/12/13 plus the Arch
// toolkit path); falls back to pinned host memory when CUDA is absent.
inline void *managed_alloc(std::size_t bytes) noexcept
{
#if defined(_WIN32)
    return pinned_alloc(bytes);
#else
#if defined(__has_include) && __has_include(<dlfcn.h>)
    {
        void *h = cuda::detail::cudart_lib();
        using cudaMallocManaged_t = int (*)(void **, std::size_t, unsigned int);
        auto sym = reinterpret_cast<cudaMallocManaged_t>(cuda::detail::lookup(h, "cudaMallocManaged"));
        if (sym != nullptr)
        {
            void *ptr = nullptr;
            if (sym(&ptr, bytes, 0x01) == cuda::kCudaSuccess && ptr != nullptr) // 0x01 = cudaMemAttachGlobal
                return ptr;
        }
    }
    {
        void *h = cuda::detail::driver_lib();
        using cuMemAllocManaged_t = int (*)(void **, std::size_t, unsigned int);
        // cuMemAllocManaged takes (ptr, bytes, flags) like the runtime call.
        auto sym = reinterpret_cast<cuMemAllocManaged_t>(cuda::detail::lookup(h, "cuMemAllocManaged"));
        if (sym != nullptr)
        {
            void *ptr = nullptr;
            if (sym(&ptr, bytes, 0x01) == cuda::kCudaSuccess && ptr != nullptr)
                return ptr;
        }
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
    // Try cudaFree via the cached runtime handle.
    if (cuda::rt_free(p) == cuda::kCudaSuccess)
        return;
    // Try driver API cuMemFree (mirrors managed_alloc fallback to cuMemAllocManaged).
    {
        void *h = cuda::detail::driver_lib();
        using cuMemFree_t = int (*)(void *);
        auto sym = reinterpret_cast<cuMemFree_t>(cuda::detail::lookup(h, "cuMemFree", "cuMemFree_v2"));
        if (sym != nullptr && sym(p) == cuda::kCudaSuccess)
            return;
    }
#endif
    pinned_free(p, 0);
}

// CUDA device count (driver + runtime agree via max; OpenMP targets excluded).
// Used for real device targeting; device_count() above mixes in OpenMP
// devices and must not be used to pick a cudaSetDevice ordinal.
NP_NODISCARD inline int cuda_device_count() noexcept
{
    int driver = 0;
    detail::probe_cuda_driver_cached(&driver);
    int runtime = 0;
    if (cuda::rt_device_get_count(&runtime) != cuda::kCudaSuccess)
        runtime = 0;
    return driver > runtime ? driver : runtime;
}

// ── Async streams & batch for powerful multi-GPU ────────────────────────
namespace stream_detail
{
// Shared stream state: one cudaStream_t per Stream object graph, created
// lazily against `device` (cudaSetDevice first) and destroyed with the last
// reference. Copies of a Stream share the same native stream — documented,
// not accidental. Null stream == CPU mode (no CUDA or creation failed).
struct state
{
    int device = 0;
    void *stream = nullptr;
    std::mutex m;
    explicit state(int d) noexcept : device(d)
    {
    }
    state(const state &) = delete;
    state &operator=(const state &) = delete;
    ~state() noexcept
    {
        if (stream != nullptr)
            cuda::rt_stream_destroy(stream);
    }
    NP_NODISCARD void *handle() noexcept
    {
        if (stream != nullptr)
            return stream;
        try
        {
            std::lock_guard<std::mutex> lk(m);
            if (stream == nullptr)
            {
                if (cuda::rt_set_device(device) != cuda::kCudaSuccess)
                    return nullptr;
                void *s = nullptr;
                if (cuda::rt_stream_create(&s) != cuda::kCudaSuccess || s == nullptr)
                    return nullptr;
                stream = s;
            }
            return stream;
        }
        catch (...)
        {
            return nullptr;
        }
    }
};
} // namespace stream_detail

struct Stream
{
    int device = 0;
    int id = 0;
    std::shared_ptr<stream_detail::state> state;

    Stream() : state(std::make_shared<stream_detail::state>(0))
    {
    }
    Stream(int device_, int id_) : device(device_), id(id_), state(std::make_shared<stream_detail::state>(device_))
    {
    }
    // Copies share the native stream (see stream_detail::state).

    /// @brief Native cudaStream_t (void*), creating it on first use.
    /// @return The stream, or nullptr in CPU mode.
    NP_NODISCARD void *native_handle() const noexcept
    {
        return state ? state->handle() : nullptr;
    }

    // Host work ordered against this stream's device work. A host callable
    // cannot execute *on* a CUDA stream, so the CUDA path submits it as a
    // host callback (cudaLaunchHostFunc): it runs after previously queued
    // stream work completes, and the future becomes ready when it finishes.
    // Without CUDA this keeps the previous behavior (OpenMP task, else
    // std::async).
    template <typename Fn> auto enqueue(Fn &&fn) -> std::future<std::invoke_result_t<Fn>>
    {
        using R = std::invoke_result_t<Fn>;
        auto pt = std::make_shared<std::packaged_task<R()>>(std::forward<Fn>(fn));
        auto fut = pt->get_future();
        if (void *s = native_handle())
        {
            auto *th = new (std::nothrow) detail::cuda_detail::host_thunk<R>{pt};
            if (th != nullptr)
            {
                if (cuda::rt_launch_host_func(s, &detail::cuda_detail::host_trampoline<R>, th) == cuda::kCudaSuccess)
                    return fut;
                delete th;
            }
        }
#if defined(NP_ENABLE_GPU) && defined(_OPENMP)
        if (is_available())
        {
            // GPU path: use OpenMP target task with heap-allocated packaged_task
            // to avoid dangling reference when task is deferred.
#pragma omp task firstprivate(pt)
            {
                (*pt)();
            }
            return fut;
        }
#endif
        // CPU path: packaged_task is already shared; run it on a worker and
        // hand back the future (std::async would re-wrap and copy the task).
        std::thread([pt]() {
            try
            {
                (*pt)();
            }
            catch (...)
            {
            }
        }).detach();
        return fut;
    }
};

NP_NODISCARD inline std::vector<Stream> make_streams(int n = 4)
{
    int devs = cuda_device_count();
    if (devs <= 0)
        devs = 1;
    std::vector<Stream> s;
    s.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        s.emplace_back(i % devs, i);
    return s;
}

// Batch GEMM: vector of (A,B,C) where each is MxK, KxN, MxN.
// Dispatched per entry over streams (each entry runs the same
// CUDA-first matmul() path); there is deliberately no graph-capture replay
// path — see the design note at the end of this file.
template <typename T>
inline void batch_matmul(const std::vector<const T *> &As, const std::vector<const T *> &Bs, std::vector<T *> &Cs,
                         std::size_t M, std::size_t N, std::size_t K) noexcept
{
    std::size_t batch = As.size();
    if (batch == 0)
        return;
    int devs = device_count();
    if (devs == 0)
        devs = 1;
    // Shard batch across devices/streams. Stream objects are affinity hints
    // here (matmul() picks the compute path per call); a failed make_streams
    // degrades to unsharded dispatch rather than throwing (noexcept).
    std::vector<Stream> streams;
    try
    {
        streams = make_streams(std::min<std::size_t>(batch, devs * 2));
    }
    catch (...)
    {
    }
    const std::size_t n_streams = streams.empty() ? 1 : streams.size();
#if defined(NP_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < batch; ++b)
    {
        int s = b % n_streams;
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

// Multi-GPU sharding for very large single GEMM (e.g., 4096) — split M across
// devices. Each shard binds its CUDA device (cudaSetDevice) and runs cuBLAS
// on it; a shard whose device call fails falls back to plain matmul (CPU).
// Shards are row-disjoint, so no cross-device synchronization is needed.
// NOTE: every shard re-reads the full B matrix on its own device (memory for
// bandwidth); acceptable for the large-GEMM regime this targets.
// Multi-GPU verification: run a long sharded GEMM and watch per-GPU
// utilization with `nvidia-smi dmon` — each device should show activity.
template <typename T>
inline void sharded_matmul(const T *a, const T *b, T *c, std::size_t M, std::size_t N, std::size_t K) noexcept
{
    const int devs = cuda_device_count();
    if (devs <= 1 || M < 1024 || M * N * K < 64ULL * 1024 * 1024)
    {
        matmul(a, b, c, M, N, K);
        return;
    }
    const std::size_t rows_per_dev = (M + static_cast<std::size_t>(devs) - 1) / static_cast<std::size_t>(devs);
    const auto run_shard = [&](int d) {
        const std::size_t start = static_cast<std::size_t>(d) * rows_per_dev;
        const std::size_t end = std::min(start + rows_per_dev, M);
        if (start >= end)
            return;
        const std::size_t rows = end - start;
        // Each shard is (rows × N); bind device d before dispatching.
        bool done = false;
        if (cuda::rt_set_device(d) == cuda::kCudaSuccess)
            done = detail::try_cuda_matmul(a + start * K, b, c + start * N, rows, N, K, d);
        if (!done)
            matmul(a + start * K, b, c + start * N, rows, N, K);
    };
#if defined(NP_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
    for (int d = 0; d < devs; ++d)
        run_shard(d);
#else
    for (int d = 0; d < devs; ++d)
        run_shard(d);
#endif
}

// Architecture queries (device-indexed; mixed-architecture multi-GPU machines
// can differ per device). Query the real silicon via cuda::*, never the
// installed driver version.
NP_NODISCARD inline bool is_blackwell(int device = 0) noexcept
{
    return cuda::is_blackwell_device(device);
}
NP_NODISCARD inline bool has_fp8_tensor(int device = 0) noexcept
{
    return cuda::has_fp8_tensor(device);
}
NP_NODISCARD inline bool has_fp4_tensor(int device = 0) noexcept
{
    return cuda::has_fp4_tensor(device);
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

// Design note (no graph-capture path, by decision): batch GEMM is dispatched
// per entry over streams (see batch_matmul above). Replayable graph capture
// would need stream-captured device kernels plus a (batch, M, N, K)-keyed
// cudaGraphExec_t cache with per-call node updates for the changing device
// pointers — machinery whose maintenance cost exceeds the marginal replay
// gain over already-asynchronous per-entry GEMMs. There is no graph batch
// path, none is planned, and no stub remains pretending otherwise.

} // namespace np::gpu

#endif // NP_GPU_HPP
