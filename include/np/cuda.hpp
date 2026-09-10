/**
 * @file cuda.hpp
 * @brief CUDA dlopen wrappers for gpu.hpp — header-only, no hard link dep.
 *
 * Everything in `np::cuda` is resolved at runtime with dlopen/dlsym so the
 * library builds and runs on machines without any CUDA toolkit or driver.
 * When a library or symbol is absent, functions return 0/false/nullptr and
 * callers fall back to CPU paths. Nothing here links `-lcudart`, `-lcublas`
 * or `-lcufft`, even when NP_ENABLE_CUDA is defined (that macro only gates
 * the *linked* runtime-API fast paths in gpu.hpp).
 *
 * Currently wrapped (all dlopen'd, all optional at runtime):
 *  - Version + device queries: cuDriverGetVersion, cuInit, cuDeviceGet,
 *    cuDeviceGetAttribute (compute capability, cached per device).
 *  - Runtime: cudaMalloc/Free/Memcpy, cudaSetDevice/GetDevice,
 *    cudaStreamCreate/Destroy/Synchronize, cudaDeviceSynchronize.
 *  - Stream-ordered alloc: cudaMallocAsync / cudaFreeAsync, default mempool.
 *  - Graphs: cudaGraphCreate/Destroy, stream begin/end capture.
 *  - cuBLAS: cublasCreate/Destroy/SetStream/Sgemm/Dgemm (used by
 *    gpu::try_cuda_matmul).
 *  - cuFFT: cufftPlan1d/ExecC2C/ExecZ2Z/Destroy (used by gpu::try_fft).
 *
 * Explicitly NOT implemented: cooperative-group launch, wgmma/TMA intrinsics
 * (these need compiled device code, which a header-only dlopen design cannot
 * provide). Batch GEMM has no graph-capture path by decision (per-entry
 * stream dispatch instead); cooperative-group launch and wgmma/TMA need
 * compiled device code and are out of reach for a dlopen design.
 */
#ifndef NP_CUDA_HPP
#define NP_CUDA_HPP

#include "api_macros.hpp"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

// When built with NP_ENABLE_CUDA and the toolkit headers are available, use
// the real CUDA runtime types throughout (this must come first: defining the
// void* stand-ins below and *then* including <cuda_runtime.h> redeclares
// cudaStream_t/cudaError_t/cudaSuccess and fails to compile — caught live
// with cuda 13.3 headers).
#if defined(NP_ENABLE_CUDA) && defined(__has_include) && __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif

#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
#include <dlfcn.h>
#endif

// CUDA version thresholds (macros, no magic numbers in logic)
#define NP_CUDA_DRIVER_COOP_MIN 9000
#define NP_CUDA_DRIVER_HOPPER_MIN 11080
#define NP_CUDA_DRIVER_BLACKWELL_MIN 12080

// Opaque handle stand-ins for builds without <cuda_runtime.h>.
// Skipped whenever the real header was included above (it defines
// CUDART_VERSION; very old toolkits used __CUDART_VERSION__): the stand-ins
// would redeclare cudaStream_t, cudaError_t, cudaSuccess, etc. Nothing below
// names these aliases in a signature — all wrappers use void*/int — so both
// spellings interoperate.
#if !defined(CUDART_VERSION) && !defined(__CUDART_VERSION__)
#ifndef NP_CUDA_TYPES_DEFINED
#define NP_CUDA_TYPES_DEFINED
using cudaStream_t = void *;
using cudaGraph_t = void *;
using cudaGraphExec_t = void *;
using cudaGraphNode_t = void *;
using cudaMemPool_t = void *;
using cudaEvent_t = void *;
using cudaFunction_t = void *;
using cudaError_t = int;
static constexpr cudaError_t cudaSuccess = 0;
#endif
#endif

namespace np::cuda
{

// ── Stable CUDA ABI constants ─────────────────────────────────────────────
// Numeric values from cuda_runtime_api.h / cublas_api.h / cufft.h / cuda.h.
// These enums have been ABI-stable for a decade+; they are spelled out here
// so this header never needs the toolkit at build time. Only the values
// actually consumed below are defined (success,OOM/no-device, op codes,
// memcpy kinds, fft types/directions).
inline constexpr int kCudaSuccess = 0;
inline constexpr int kCudaErrorMemoryAllocation = 2; // cudaErrorMemoryAllocation / CUDA_ERROR_OUT_OF_MEMORY
inline constexpr int kCudaErrorNoDevice = 100;       // "no CUDA-capable device is detected"
// cudaMemcpyKind
inline constexpr int kMemcpyHostToHost = 0;
inline constexpr int kMemcpyHostToDevice = 1;
inline constexpr int kMemcpyDeviceToHost = 2;
inline constexpr int kMemcpyDeviceToDevice = 3;
// cublasOperation_t
inline constexpr int kCublasOpN = 0;
inline constexpr int kCublasOpT = 1;
// cublasStatus_t (subset)
inline constexpr int kCublasSuccess = 0;
inline constexpr int kCublasNotInitialized = 1;
inline constexpr int kCublasAllocFailed = 3;
inline constexpr int kCublasInvalidValue = 7;
inline constexpr int kCublasArchMismatch = 8;
inline constexpr int kCublasMappingError = 11;
inline constexpr int kCublasExecutionFailed = 13;
inline constexpr int kCublasNotSupported = 15;
// cufftType (subset)
inline constexpr int kCufftC2C = 0x29;
inline constexpr int kCufftZ2Z = 0x69;
// cufft direction
inline constexpr int kCufftForward = -1;
inline constexpr int kCufftInverse = 1;
// cufftResult (subset)
inline constexpr int kCufftSuccess = 0;
inline constexpr int kCufftAllocFailed = 2;
inline constexpr int kCufftInvalidType = 3;
inline constexpr int kCufftInvalidValue = 4;
inline constexpr int kCufftInternalError = 5;
inline constexpr int kCufftExecFailed = 6;
inline constexpr int kCufftSetupFailed = 7;
inline constexpr int kCufftInvalidSize = 8;
// CUdevice_attribute used for compute-capability queries
inline constexpr int kAttrComputeMajor = 75;
inline constexpr int kAttrComputeMinor = 76;
// Upper bound for thread-local per-device handle caches (cuBLAS).
inline constexpr int kMaxCachedDevices = 16;

namespace detail
{
// Try each candidate soname/path in order; return the first handle that
// opens, or nullptr. The caller owns the handle (dlclose when done).
inline void *open_first(const char *const *names, std::size_t n) noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    for (std::size_t i = 0; i < n; ++i)
    {
        if (names[i] == nullptr)
            continue;
        if (void *h = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL))
            return h;
    }
#else
    (void)names;
    (void)n;
#endif
    return nullptr;
}

// Library handles below are opened once per process and intentionally never
// closed: closing would invalidate cached symbols and re-dlopen on the hot
// path. Each list covers unversioned + versioned sonames plus the Arch
// (/opt/cuda) and default (/usr/local/cuda) toolkit locations, since those
// directories are often absent from ld.so.conf.
inline void *driver_lib() noexcept
{
    static const char *const names[] = {"libcuda.so.1", "libcuda.so"};
    static void *h = open_first(names, 2);
    return h;
}

inline void *cudart_lib() noexcept
{
    static const char *const names[] = {"libcudart.so",
                                        "libcudart.so.13",
                                        "libcudart.so.12",
                                        "libcudart.so.11",
                                        "/opt/cuda/lib64/libcudart.so",
                                        "/usr/local/cuda/lib64/libcudart.so"};
    static void *h = open_first(names, 6);
    return h;
}

inline void *cublas_lib() noexcept
{
    static const char *const names[] = {"libcublas.so",
                                        "libcublas.so.13",
                                        "libcublas.so.12",
                                        "libcublas.so.11",
                                        "/opt/cuda/lib64/libcublas.so",
                                        "/usr/local/cuda/lib64/libcublas.so"};
    static void *h = open_first(names, 6);
    return h;
}

inline void *cufft_lib() noexcept
{
    static const char *const names[] = {"libcufft.so",
                                        "libcufft.so.12",
                                        "libcufft.so.11",
                                        "libcufft.so.10",
                                        "/opt/cuda/lib64/libcufft.so",
                                        "/usr/local/cuda/lib64/libcufft.so"};
    static void *h = open_first(names, 6);
    return h;
}

// Look up `primary`, falling back to `alternate` when non-null. Returns
// nullptr when the library handle is null or neither symbol exists.
inline void *lookup(void *lib, const char *primary, const char *alternate = nullptr) noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    if (lib == nullptr || primary == nullptr)
        return nullptr;
    if (void *s = dlsym(lib, primary))
        return s;
    if (alternate != nullptr)
        return dlsym(lib, alternate);
#else
    (void)lib;
    (void)primary;
    (void)alternate;
#endif
    return nullptr;
}
} // namespace detail

// ── CUDA version helpers ──────────────────────────────────────────────────
// Uncached probes (single dlopen cycle each). Prefer the cached
// driver_version()/runtime_version() below on hot paths.
NP_NODISCARD inline int driver_version_uncached() noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!h)
        h = dlopen("libcuda.so", RTLD_LAZY);
    if (!h)
        return 0;
    using cuDriverGetVersion_t = int (*)(int *);
    auto sym = reinterpret_cast<cuDriverGetVersion_t>(dlsym(h, "cuDriverGetVersion"));
    int v = 0;
    if (sym)
        sym(&v);
    dlclose(h);
    return v; // e.g. 12080 for CUDA 12.8
#else
    return 0;
#endif
}

NP_NODISCARD inline int runtime_version_uncached() noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.13", RTLD_LAZY);
    if (!h)
        return 0;
    using cudaRuntimeGetVersion_t = int (*)(int *);
    auto sym = reinterpret_cast<cudaRuntimeGetVersion_t>(dlsym(h, "cudaRuntimeGetVersion"));
    int v = 0;
    if (sym)
        sym(&v);
    dlclose(h);
    return v;
#else
    return 0;
#endif
}

// Cached versions: one dlopen per process (thread-safe static init).
NP_NODISCARD inline int driver_version() noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    static const int cached = driver_version_uncached();
    return cached;
#else
    return 0;
#endif
}

NP_NODISCARD inline int runtime_version() noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    static const int cached = runtime_version_uncached();
    return cached;
#else
    return 0;
#endif
}

// ── Device compute-capability query ───────────────────────────────────────
// Driver version only tells which CUDA API the installed driver supports,
// not what silicon is present. Capability checks must query the device.

// Pure architecture predicates on (major, minor). constexpr and
// dependency-free so they are unit-testable without a GPU.
NP_NODISCARD constexpr bool arch_is_blackwell(int major, int minor) noexcept
{
    // Blackwell is SM 100+ (100/103/120/121, ...); Hopper is SM 90.
    (void)minor;
    return major >= 10;
}
NP_NODISCARD constexpr bool arch_has_fp8(int major, int minor) noexcept
{
    // FP8 tensor cores: Hopper (SM90+) and newer.
    (void)minor;
    return major >= 9;
}
NP_NODISCARD constexpr bool arch_has_fp4(int major, int minor) noexcept
{
    // FP4 tensor cores: Blackwell (SM100+) and newer.
    (void)minor;
    return major >= 10;
}

// Uncached primitive: one dlopen cycle per call. Prefer
// cached_compute_capability() below on any repeated path.
NP_NODISCARD inline bool device_compute_capability(int device, int &major, int &minor) noexcept
{
    major = 0;
    minor = 0;
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = detail::driver_lib();
    if (h == nullptr)
        return false;
    using cuInit_t = int (*)(unsigned int);
    // CUdevice is an int in the CUDA driver API; avoid pulling <cuda.h>.
    using cuDeviceGet_t = int (*)(int *, int);
    using cuDeviceGetAttribute_t = int (*)(int *, int, int);
    auto cuInit = reinterpret_cast<cuInit_t>(detail::lookup(h, "cuInit"));
    auto cuDeviceGet = reinterpret_cast<cuDeviceGet_t>(detail::lookup(h, "cuDeviceGet"));
    auto cuDeviceGetAttribute = reinterpret_cast<cuDeviceGetAttribute_t>(detail::lookup(h, "cuDeviceGetAttribute"));
    if (cuInit == nullptr || cuDeviceGet == nullptr || cuDeviceGetAttribute == nullptr)
        return false;
    int cu_dev = 0;
    if (cuInit(0) != 0 || cuDeviceGet(&cu_dev, device) != 0)
        return false;
    int maj = 0, min = 0;
    if (cuDeviceGetAttribute(&maj, kAttrComputeMajor, cu_dev) != 0 ||
        cuDeviceGetAttribute(&min, kAttrComputeMinor, cu_dev) != 0)
        return false;
    major = maj;
    minor = min;
    return true;
#else
    (void)device;
    return false;
#endif
}

namespace detail
{
// Per-device capability cache entry. Both success and failure are cached:
// on machines without a driver every predicate would otherwise re-dlopen.
struct capability_entry
{
    int major = 0;
    int minor = 0;
    bool probed = false;
    bool present = false;
};
inline std::mutex &capability_mutex() noexcept
{
    static std::mutex m;
    return m;
}
inline std::vector<capability_entry> &capability_cache()
{
    static std::vector<capability_entry> v;
    return v;
}
} // namespace detail

// Cached capability query keyed by device index (thread-safe). Falls back
// to an uncached probe if the cache itself cannot be maintained (e.g. mutex
// or vector allocation failure — must not throw: this function is noexcept,
// so every exception path degrades to a direct probe returning false).
NP_NODISCARD inline bool cached_compute_capability(int device, int &major, int &minor) noexcept
{
    major = 0;
    minor = 0;
    if (device < 0)
        return false;
    try
    {
        std::lock_guard<std::mutex> lk(detail::capability_mutex());
        auto &cache = detail::capability_cache();
        const auto idx = static_cast<std::size_t>(device);
        if (idx < cache.size() && cache[idx].probed)
        {
            major = cache[idx].major;
            minor = cache[idx].minor;
            return cache[idx].present;
        }
        // Probe once per device while holding the lock; afterwards every
        // caller is served from the cache with no dlopen.
        int maj = 0, min = 0;
        const bool ok = device_compute_capability(device, maj, min);
        if (idx >= cache.size())
            cache.resize(idx + 1);
        cache[idx].major = maj;
        cache[idx].minor = min;
        cache[idx].probed = true;
        cache[idx].present = ok;
        major = maj;
        minor = min;
        return ok;
    }
    catch (const std::exception &)
    {
        // Cache maintenance failed (mutex/vector); fall through to a direct
        // noexcept probe so capability queries degrade instead of throwing.
        return device_compute_capability(device, major, minor);
    }
    catch (...)
    {
        // Non-std exception (e.g. corrupt cache state); same degradation.
        return device_compute_capability(device, major, minor);
    }
}

// ── Runtime core: device memory, transfers, devices, streams ──────────────
// Thin dlopen'd wrappers over the CUDA runtime + driver libraries. All use
// the cached library handles from detail:: (opened once, never closed) and
// return raw status codes (kCudaSuccess == 0); gpu.hpp maps these onto
// CudaStatus. All degrade to failure (nonzero / nullptr / false) when the
// libraries are absent — no-ops on machines without CUDA.
NP_NODISCARD inline void *rt_lib_cudart() noexcept
{
    return detail::cudart_lib();
}

NP_NODISCARD inline int rt_malloc(void **out, std::size_t bytes) noexcept
{
    if (out == nullptr)
        return -1;
    *out = nullptr;
    void *h = detail::cudart_lib();
    using fn_t = int (*)(void **, std::size_t);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaMalloc"));
    if (sym == nullptr)
        return -1;
    const int rc = sym(out, bytes);
    if (rc != kCudaSuccess)
        *out = nullptr; // guarantee null-on-failure regardless of driver behavior
    return rc;
}

inline int rt_free(void *p) noexcept
{
    void *h = detail::cudart_lib();
    using fn_t = int (*)(void *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaFree"));
    if (sym == nullptr)
        return -1;
    return sym(p);
}

NP_NODISCARD inline int rt_memcpy(void *dst, const void *src, std::size_t bytes, int kind) noexcept
{
    void *h = detail::cudart_lib();
    using fn_t = int (*)(void *, const void *, std::size_t, int);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaMemcpy"));
    if (sym == nullptr)
        return -1;
    return sym(dst, src, bytes, kind);
}

inline int rt_set_device(int device) noexcept
{
    // Runtime API only: the driver equivalent (cuCtxCreate per device) needs
    // context-lifetime management that is out of scope here; without libcudart
    // there is no device targeting and callers must take the CPU fallback.
    void *h = detail::cudart_lib();
    using fn_t = int (*)(int);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaSetDevice"));
    if (sym == nullptr)
        return -1;
    return sym(device);
}

NP_NODISCARD inline int rt_get_device(int *device) noexcept
{
    if (device == nullptr)
        return -1;
    void *h = detail::cudart_lib();
    using fn_t = int (*)(int *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaGetDevice"));
    if (sym == nullptr)
        return -1;
    return sym(device);
}

NP_NODISCARD inline int rt_device_get_count(int *count) noexcept
{
    if (count == nullptr)
        return -1;
    *count = 0;
    void *h = detail::cudart_lib();
    using fn_t = int (*)(int *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaGetDeviceCount"));
    if (sym == nullptr)
        return -1;
    return sym(count);
}

NP_NODISCARD inline int rt_stream_create(void **out) noexcept
{
    if (out == nullptr)
        return -1;
    *out = nullptr;
    void *h = detail::cudart_lib();
    using fn_t = int (*)(void **);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaStreamCreate"));
    if (sym == nullptr)
        return -1;
    return sym(out);
}

inline int rt_stream_destroy(void *stream) noexcept
{
    void *h = detail::cudart_lib();
    using fn_t = int (*)(void *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaStreamDestroy"));
    if (sym == nullptr)
        return -1;
    return sym(stream);
}

inline int rt_stream_synchronize(void *stream) noexcept
{
    void *h = detail::cudart_lib();
    using fn_t = int (*)(void *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaStreamSynchronize"));
    if (sym == nullptr)
        return -1;
    return sym(stream);
}

inline int rt_device_synchronize() noexcept
{
    void *h = detail::cudart_lib();
    using fn_t = int (*)();
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaDeviceSynchronize"));
    if (sym == nullptr)
        return -1;
    return sym();
}

// Enqueue a host callback after all previously queued work on `stream`
// completes (cudaLaunchHostFunc). The callback runs on a driver thread and
// must not throw; used by gpu::Stream::enqueue's CUDA path.
using host_callback_t = void (*)(void *);
inline int rt_launch_host_func(void *stream, host_callback_t fn, void *arg) noexcept
{
    if (fn == nullptr)
        return -1;
    void *h = detail::cudart_lib();
    using fn_t = int (*)(void *, void (*)(void *), void *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaLaunchHostFunc"));
    if (sym == nullptr)
        return -1;
    return sym(stream, fn, arg);
}

NP_NODISCARD inline const char *rt_error_string(int code) noexcept
{
    void *h = detail::cudart_lib();
    using fn_t = const char *(*)(int);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cudaGetErrorString"));
    if (sym == nullptr)
        return "cudaGetErrorString unavailable (no CUDA runtime)";
    return sym(code);
}

// ── cuBLAS (dlopen'd; signatures per cublas_api.h, _v2 entry points) ───────
NP_NODISCARD inline int blas_create(void **handle) noexcept
{
    if (handle == nullptr)
        return -1;
    *handle = nullptr;
    void *h = detail::cublas_lib();
    using fn_t = int (*)(void **);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cublasCreate_v2", "cublasCreate"));
    if (sym == nullptr)
        return -1;
    return sym(handle);
}

inline int blas_destroy(void *handle) noexcept
{
    void *h = detail::cublas_lib();
    using fn_t = int (*)(void *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cublasDestroy_v2", "cublasDestroy"));
    if (sym == nullptr)
        return -1;
    return sym(handle);
}

inline int blas_set_stream(void *handle, void *stream) noexcept
{
    void *h = detail::cublas_lib();
    using fn_t = int (*)(void *, void *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cublasSetStream_v2", "cublasSetStream"));
    if (sym == nullptr)
        return -1;
    return sym(handle, stream);
}

// Column-major GEMM: C(m×n) = op(A)(m×k) * op(B)(k×n). See gpu::try_cuda_matmul
// for the row-major transpose trick used with these.
NP_NODISCARD inline int blas_sgemm(void *handle, int transa, int transb, int m, int n, int k, const float *alpha,
                                   const float *a, int lda, const float *b, int ldb, const float *beta, float *c,
                                   int ldc) noexcept
{
    void *h = detail::cublas_lib();
    using fn_t = int (*)(void *, int, int, int, int, int, const float *, const float *, int, const float *, int,
                         const float *, float *, int);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cublasSgemm_v2", "cublasSgemm"));
    if (sym == nullptr)
        return -1;
    return sym(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

NP_NODISCARD inline int blas_dgemm(void *handle, int transa, int transb, int m, int n, int k, const double *alpha,
                                   const double *a, int lda, const double *b, int ldb, const double *beta, double *c,
                                   int ldc) noexcept
{
    void *h = detail::cublas_lib();
    using fn_t = int (*)(void *, int, int, int, int, int, const double *, const double *, int, const double *, int,
                         const double *, double *, int);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cublasDgemm_v2", "cublasDgemm"));
    if (sym == nullptr)
        return -1;
    return sym(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ── cuFFT (dlopen'd; signatures per cufft.h) ───────────────────────────────
NP_NODISCARD inline int fft_plan1d(void **plan, int nx, int type, int batch) noexcept
{
    if (plan == nullptr)
        return -1;
    *plan = nullptr;
    void *h = detail::cufft_lib();
    using fn_t = int (*)(void **, int, int, int);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cufftPlan1d"));
    if (sym == nullptr)
        return -1;
    return sym(plan, nx, type, batch);
}

NP_NODISCARD inline int fft_exec_c2c(void *plan, void *idata, void *odata, int direction) noexcept
{
    void *h = detail::cufft_lib();
    using fn_t = int (*)(void *, void *, void *, int);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cufftExecC2C"));
    if (sym == nullptr)
        return -1;
    return sym(plan, idata, odata, direction);
}

NP_NODISCARD inline int fft_exec_z2z(void *plan, void *idata, void *odata, int direction) noexcept
{
    void *h = detail::cufft_lib();
    using fn_t = int (*)(void *, void *, void *, int);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cufftExecZ2Z"));
    if (sym == nullptr)
        return -1;
    return sym(plan, idata, odata, direction);
}

inline int fft_destroy(void *plan) noexcept
{
    void *h = detail::cufft_lib();
    using fn_t = int (*)(void *);
    auto sym = reinterpret_cast<fn_t>(detail::lookup(h, "cufftDestroy"));
    if (sym == nullptr)
        return -1;
    return sym(plan);
}

// ── Stream-ordered / async alloc (CUDA 11.2+ / 12) ───────────────────────
NP_NODISCARD inline void *malloc_async(std::size_t bytes, void *stream = nullptr) noexcept
{
    void *h = detail::cudart_lib();
    using cudaMallocAsync_t = int (*)(void **, std::size_t, void *);
    auto sym = reinterpret_cast<cudaMallocAsync_t>(detail::lookup(h, "cudaMallocAsync"));
    void *p = nullptr;
    if (sym != nullptr && sym(&p, bytes, stream) == kCudaSuccess && p != nullptr)
        return p;
    (void)bytes;
    (void)stream;
    return nullptr;
}

inline int free_async(void *p, void *stream = nullptr) noexcept
{
    void *h = detail::cudart_lib();
    using cudaFreeAsync_t = int (*)(void *, void *);
    auto sym = reinterpret_cast<cudaFreeAsync_t>(detail::lookup(h, "cudaFreeAsync"));
    if (sym == nullptr)
    {
        (void)p;
        (void)stream;
        return -1;
    }
    return sym(p, stream);
}

// MemPool (CUDA 11.2+) — get default pool and set thresholds
NP_NODISCARD inline void *mempool_default(int device = 0) noexcept
{
    void *h = detail::cudart_lib();
    using cudaDeviceGetDefaultMemPool_t = int (*)(void **, int);
    auto sym = reinterpret_cast<cudaDeviceGetDefaultMemPool_t>(detail::lookup(h, "cudaDeviceGetDefaultMemPool"));
    void *pool = nullptr;
    if (sym != nullptr)
        sym(&pool, device);
    else
        (void)device;
    return pool;
}

// ── Graphs (CUDA 10+ / 12) ────────────────────────────────────────────────
NP_NODISCARD inline int graph_create(void **out) noexcept
{
    if (out == nullptr)
        return -1;
    void *h = detail::cudart_lib();
    using cudaGraphCreate_t = int (*)(void **, unsigned int);
    auto sym = reinterpret_cast<cudaGraphCreate_t>(detail::lookup(h, "cudaGraphCreate"));
    if (sym == nullptr)
        return -1;
    return sym(out, 0);
}

inline int graph_destroy(void *g) noexcept
{
    void *h = detail::cudart_lib();
    using cudaGraphDestroy_t = int (*)(void *);
    auto sym = reinterpret_cast<cudaGraphDestroy_t>(detail::lookup(h, "cudaGraphDestroy"));
    if (sym == nullptr)
    {
        (void)g;
        return -1;
    }
    return sym(g);
}

// Stream capture for graphs
NP_NODISCARD inline int stream_begin_capture(void *stream) noexcept
{
    void *h = detail::cudart_lib();
    using cudaStreamBeginCapture_t = int (*)(void *, int);
    auto sym = reinterpret_cast<cudaStreamBeginCapture_t>(detail::lookup(h, "cudaStreamBeginCapture"));
    if (sym == nullptr)
    {
        (void)stream;
        return -1;
    }
    return sym(stream, 0); // cudaStreamCaptureModeGlobal
}

NP_NODISCARD inline int stream_end_capture(void *stream, void **out_graph) noexcept
{
    void *h = detail::cudart_lib();
    using cudaStreamEndCapture_t = int (*)(void *, void **);
    auto sym = reinterpret_cast<cudaStreamEndCapture_t>(detail::lookup(h, "cudaStreamEndCapture"));
    if (sym == nullptr)
    {
        (void)stream;
        (void)out_graph;
        return -1;
    }
    return sym(stream, out_graph);
}

// ── Cooperative launch (CUDA 9+ / 12) ─────────────────────────────────────
NP_NODISCARD inline bool has_cooperative() noexcept
{
    int v = driver_version();
    return v >= NP_CUDA_DRIVER_COOP_MIN;
}

// ── Blackwell / Hopper arch helpers ─────────────────────────────────────────
// NOTE: driver version only indicates which CUDA API the driver supports.
// Architecture predicates below query the actual device compute capability
// (device 0 by default) and fall back to the driver-version heuristic only
// when no device can be queried (e.g. no GPU present).
NP_NODISCARD inline bool is_blackwell_device(int device = 0) noexcept
{
    int major = 0, minor = 0;
    if (cached_compute_capability(device, major, minor))
        return arch_is_blackwell(major, minor);
    return driver_version() >= NP_CUDA_DRIVER_BLACKWELL_MIN;
}

NP_NODISCARD inline bool is_blackwell(int major = 10) noexcept
{
    // Legacy heuristic overload kept for compatibility: `major` is the
    // expected compute-major to test for. Prefer is_blackwell_device().
    int actual_major = 0, actual_minor = 0;
    if (cached_compute_capability(0, actual_major, actual_minor))
    {
        if (major >= 10)
            return arch_is_blackwell(actual_major, actual_minor);
        if (major == 9)
            return actual_major == 9;
        return false;
    }
    const int v = driver_version();
    if (major >= 10)
        return v >= NP_CUDA_DRIVER_BLACKWELL_MIN;
    if (major == 9)
        return v >= NP_CUDA_DRIVER_HOPPER_MIN && v < NP_CUDA_DRIVER_BLACKWELL_MIN;
    return false;
}

NP_NODISCARD inline bool has_fp8_tensor(int device = 0) noexcept
{
    // FP8 tensor cores: Hopper (SM90+) and newer.
    int major = 0, minor = 0;
    if (cached_compute_capability(device, major, minor))
        return arch_has_fp8(major, minor);
    return driver_version() >= NP_CUDA_DRIVER_HOPPER_MIN;
}

NP_NODISCARD inline bool has_fp4_tensor(int device = 0) noexcept
{
    // FP4: Blackwell (SM100+) + CUDA 12.8+.
    int major = 0, minor = 0;
    if (cached_compute_capability(device, major, minor))
        return arch_has_fp4(major, minor) && driver_version() >= NP_CUDA_DRIVER_BLACKWELL_MIN;
    return driver_version() >= NP_CUDA_DRIVER_BLACKWELL_MIN;
}

// NOTE: there is intentionally no graph-captured batch GEMM helper here.
// Batch GEMM is dispatched per entry over streams (gpu::batch_matmul); a
// replayable graph path would need captured kernels plus a (batch, M, N, K)-
// keyed exec cache with per-call node updates, which is out of scope by
// decision, not deferred. The graph_create/destroy + stream capture wrappers
// above remain as general infrastructure.

} // namespace np::cuda

#endif // NP_CUDA_HPP
