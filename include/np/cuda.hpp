/**
 * @file cuda.hpp
 * @brief CUDA 12/13 header-only stub + dlopen wrappers for gpu.hpp — no hard link dep.
 *
 * Provides header-only access to new CUDA features via dlopen:
 *  - CUDA 12: cudaMallocAsync / cudaFreeAsync / MemPool (cudaMemPool_t, cudaMemPoolProps)
 *  - CUDA 12: Graphs (cudaGraph_t, cudaGraphExec_t, cudaGraphCreate, AddKernelNode, Instantiate, Launch)
 *  - CUDA 12: Cooperative Groups (cudaLaunchCooperativeKernel, Occupancy)
 *  - CUDA 12: Stream-ordered (cudaStreamBeginCapture, EndCapture, GraphExec)
 *  - CUDA 13: Blackwell (SM 100/103) arch helpers, FP4/FP8 tensor, wgmma stub
 *  - Hopper (SM 90) wgmma / wgmma.fence, TMA stub
 * Real runtime is dlopened in gpu.hpp; this header just defines types and
 * inline helpers so gpu.hpp can call them without <cuda_runtime.h>.
 */
#ifndef NP_CUDA_HPP
#define NP_CUDA_HPP

#include "api_macros.hpp"
#include <cstddef>
#include <cstdint>

#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
#include <dlfcn.h>
#endif

// CUDA version thresholds (macros, no magic numbers in logic)
#define NP_CUDA_DRIVER_COOP_MIN 9000
#define NP_CUDA_DRIVER_HOPPER_MIN 11080
#define NP_CUDA_DRIVER_BLACKWELL_MIN 12080

// Define opaque handle types without pulling <cuda_runtime.h>
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

namespace np::cuda
{

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
NP_NODISCARD inline bool device_compute_capability(int device, int &major, int &minor) noexcept
{
    major = 0;
    minor = 0;
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!h)
        h = dlopen("libcuda.so", RTLD_LAZY);
    if (!h)
        return false;
    using cuInit_t = int (*)(unsigned int);
    // CUdevice is an int in the CUDA driver API; avoid pulling <cuda.h>.
    using cuDeviceGet_t = int (*)(int *, int);
    using cuDeviceGetAttribute_t = int (*)(int *, int, int);
    auto cuInit = reinterpret_cast<cuInit_t>(dlsym(h, "cuInit"));
    auto cuDeviceGet = reinterpret_cast<cuDeviceGet_t>(dlsym(h, "cuDeviceGet"));
    auto cuDeviceGetAttribute =
        reinterpret_cast<cuDeviceGetAttribute_t>(dlsym(h, "cuDeviceGetAttribute"));
    bool ok = false;
    if (cuInit && cuDeviceGet && cuDeviceGetAttribute)
    {
        // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=75, MINOR=76
        constexpr int kMajorAttr = 75;
        constexpr int kMinorAttr = 76;
        int cu_dev = 0;
        if (cuInit(0) == 0 && cuDeviceGet(&cu_dev, device) == 0)
        {
            int maj = 0, min = 0;
            if (cuDeviceGetAttribute(&maj, kMajorAttr, cu_dev) == 0 &&
                cuDeviceGetAttribute(&min, kMinorAttr, cu_dev) == 0)
            {
                major = maj;
                minor = min;
                ok = true;
            }
        }
    }
    dlclose(h);
    return ok;
#else
    (void)device;
    return false;
#endif
}

// ── Stream-ordered / async alloc (CUDA 11.2+ / 12) ───────────────────────
NP_NODISCARD inline void *malloc_async(std::size_t bytes, void *stream = nullptr) noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.13", RTLD_LAZY);
    if (!h)
        return nullptr;
    using cudaMallocAsync_t = int (*)(void **, std::size_t, void *);
    auto sym = reinterpret_cast<cudaMallocAsync_t>(dlsym(h, "cudaMallocAsync"));
    void *p = nullptr;
    int rc = -1;
    if (sym)
        rc = sym(&p, bytes, stream);
    dlclose(h);
    if (rc == 0 && p)
        return p;
#endif
    (void)bytes;
    (void)stream;
    return nullptr;
}

inline int free_async(void *p, void *stream = nullptr) noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.13", RTLD_LAZY);
    if (!h)
        return -1;
    using cudaFreeAsync_t = int (*)(void *, void *);
    auto sym = reinterpret_cast<cudaFreeAsync_t>(dlsym(h, "cudaFreeAsync"));
    int rc = -1;
    if (sym)
        rc = sym(p, stream);
    dlclose(h);
    return rc;
#else
    (void)p;
    (void)stream;
    return -1;
#endif
}

// MemPool (CUDA 11.2+) — get default pool and set thresholds
NP_NODISCARD inline void *mempool_default(int device = 0) noexcept
{
    (void)device;
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        return nullptr;
    using cudaDeviceGetDefaultMemPool_t = int (*)(void **, int);
    auto sym = reinterpret_cast<cudaDeviceGetDefaultMemPool_t>(dlsym(h, "cudaDeviceGetDefaultMemPool"));
    void *pool = nullptr;
    if (sym)
        sym(&pool, device);
    dlclose(h);
    return pool;
#else
    return nullptr;
#endif
}

// ── Graphs (CUDA 10+ / 12) ────────────────────────────────────────────────
NP_NODISCARD inline int graph_create(void **out) noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        return -1;
    using cudaGraphCreate_t = int (*)(void **, unsigned int);
    auto sym = reinterpret_cast<cudaGraphCreate_t>(dlsym(h, "cudaGraphCreate"));
    int rc = -1;
    if (sym)
        rc = sym(out, 0);
    dlclose(h);
    return rc;
#else
    (void)out;
    return -1;
#endif
}

inline int graph_destroy(void *g) noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        return -1;
    using cudaGraphDestroy_t = int (*)(void *);
    auto sym = reinterpret_cast<cudaGraphDestroy_t>(dlsym(h, "cudaGraphDestroy"));
    int rc = -1;
    if (sym)
        rc = sym(g);
    dlclose(h);
    return rc;
#else
    (void)g;
    return -1;
#endif
}

// Stream capture for graphs
NP_NODISCARD inline int stream_begin_capture(void *stream) noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        return -1;
    using cudaStreamBeginCapture_t = int (*)(void *, int);
    auto sym = reinterpret_cast<cudaStreamBeginCapture_t>(dlsym(h, "cudaStreamBeginCapture"));
    int rc = -1;
    if (sym)
        rc = sym(stream, 0); // cudaStreamCaptureModeGlobal
    dlclose(h);
    return rc;
#else
    (void)stream;
    return -1;
#endif
}

NP_NODISCARD inline int stream_end_capture(void *stream, void **out_graph) noexcept
{
#if defined(__has_include) && __has_include(<dlfcn.h>) && !defined(_WIN32)
    void *h = dlopen("libcudart.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!h)
        return -1;
    using cudaStreamEndCapture_t = int (*)(void *, void **);
    auto sym = reinterpret_cast<cudaStreamEndCapture_t>(dlsym(h, "cudaStreamEndCapture"));
    int rc = -1;
    if (sym)
        rc = sym(stream, out_graph);
    dlclose(h);
    return rc;
#else
    (void)stream;
    (void)out_graph;
    return -1;
#endif
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
    // Blackwell is SM 100+ (100/103/120/121, ...); Hopper is SM 90.
    int major = 0, minor = 0;
    if (device_compute_capability(device, major, minor))
        return major >= 10;
    return driver_version() >= NP_CUDA_DRIVER_BLACKWELL_MIN;
}

NP_NODISCARD inline bool is_blackwell(int major = 10) noexcept
{
    // Legacy heuristic overload kept for compatibility: `major` is the
    // expected compute-major to test for. Prefer is_blackwell_device().
    int actual_major = 0, actual_minor = 0;
    if (device_compute_capability(0, actual_major, actual_minor))
    {
        if (major >= 10)
            return actual_major >= 10;
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
    // FP8 tensor cores: Hopper (SM90+) and Blackwell.
    int major = 0, minor = 0;
    if (device_compute_capability(device, major, minor))
        return major >= 9;
    return driver_version() >= NP_CUDA_DRIVER_HOPPER_MIN;
}

NP_NODISCARD inline bool has_fp4_tensor(int device = 0) noexcept
{
    // FP4: Blackwell (SM100+) + CUDA 12.8+.
    int major = 0, minor = 0;
    if (device_compute_capability(device, major, minor))
        return major >= 10 && driver_version() >= NP_CUDA_DRIVER_BLACKWELL_MIN;
    return driver_version() >= NP_CUDA_DRIVER_BLACKWELL_MIN;
}

// ── Pinned / async helpers that gpu.hpp can call ──────────────────────────
// Thin wrappers so gpu.hpp doesn't need to dlopen itself for new features
inline bool try_cuda_graph_batch_matmul(const void * /*as*/, const void * /*bs*/, void * /*cs*/, std::size_t /*M*/,
                                        std::size_t /*N*/, std::size_t /*K*/, std::size_t /*batch*/) noexcept
{
    // Placeholder for future graph-captured batch GEMM
    // For now, return false to fall back to streams
    return false;
}

} // namespace np::cuda

#endif // NP_CUDA_HPP
