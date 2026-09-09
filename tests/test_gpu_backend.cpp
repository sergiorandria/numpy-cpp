/**
 * @file test_gpu_backend.cpp
 * @brief Tests for the real CUDA backend (np::gpu / np::cuda).
 *
 * Strategy: pure-logic checks (arch predicates, status plumbing, argument
 * validation) run everywhere and are fully deterministic. Live-GPU checks
 * (cuBLAS GEMM vs CPU within tolerance, cuFFT roundtrip, OOM mapping) run
 * only when the corresponding libraries + devices are present; otherwise
 * they verify the graceful fallback (false + informative last_error()).
 * Nothing here requires a GPU to pass.
 */
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "np/cuda.hpp"
#include "np/gpu.hpp"
#include "test_util.hpp"

namespace
{

template <typename T> void fill_seq(std::vector<T> &v, T scale = T{1})
{
    for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<T>((i % 13) + 1) * scale / T{7};
}

template <typename T>
void cpu_ref(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &c, std::size_t M, std::size_t N,
             std::size_t K)
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

template <typename T> bool close(const std::vector<T> &x, const std::vector<T> &y, double eps)
{
    if (x.size() != y.size())
        return false;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double xa = static_cast<double>(x[i]);
        const double ya = static_cast<double>(y[i]);
        const double scale = 1.0 + std::fabs(xa) + std::fabs(ya);
        if (std::fabs(xa - ya) > eps * scale)
            return false;
    }
    return true;
}

} // namespace

int main()
{
    using np::gpu::CudaStatus;

    // 1. Pure architecture predicates (no GPU needed; static truth table).
    test::check(np::cuda::arch_is_blackwell(10, 0), "blackwell sm100");
    test::check(np::cuda::arch_is_blackwell(10, 3), "blackwell sm103");
    test::check(!np::cuda::arch_is_blackwell(9, 0), "hopper is not blackwell");
    test::check(!np::cuda::arch_is_blackwell(8, 0), "ampere is not blackwell");
    test::check(!np::cuda::arch_is_blackwell(7, 5), "turing is not blackwell");
    test::check(np::cuda::arch_has_fp8(9, 0), "hopper fp8");
    test::check(np::cuda::arch_has_fp8(10, 0), "blackwell fp8");
    test::check(!np::cuda::arch_has_fp8(8, 0), "ampere no fp8");
    test::check(!np::cuda::arch_has_fp8(7, 5), "turing no fp8");
    test::check(np::cuda::arch_has_fp4(10, 0), "blackwell fp4");
    test::check(!np::cuda::arch_has_fp4(9, 0), "hopper no fp4");
    test::check(!np::cuda::arch_has_fp4(8, 0), "ampere no fp4");

    // 2. Status plumbing defaults.
    test::check(np::gpu::last_error() == CudaStatus::Ok, "last_error starts Ok");
    test::check(np::gpu::last_error_string() != nullptr, "last_error_string non-null");

    // 3. Argument validation is deterministic with or without CUDA.
    test::check(!np::gpu::detail::try_cuda_matmul<int>(nullptr, nullptr, nullptr, 0, 0, 0), "int matmul rejected");
    test::check(np::gpu::last_error() == CudaStatus::UnsupportedType, "int matmul status");
    {
        float a = 1.0f, b = 2.0f, c = 0.0f;
        test::check(!np::gpu::detail::try_cuda_matmul(&a, &b, &c, 0, 1, 1), "zero-M rejected");
        test::check(np::gpu::last_error() == CudaStatus::InvalidValue, "zero-M status");
        test::check(!np::gpu::detail::try_cuda_matmul<float>(nullptr, &b, &c, 1, 1, 1), "null rejected");
        test::check(np::gpu::last_error() == CudaStatus::InvalidValue, "null status");
    }

    // 4. Live device query agrees with the pure predicates (when a device
    //    exists); otherwise the driver-version fallback is self-consistent.
    {
        int major = 0, minor = 0;
        const bool present = np::cuda::cached_compute_capability(0, major, minor);
        if (present)
        {
            test::check(major >= 0 && minor >= 0, "capability sane");
            test::check(np::gpu::has_fp8_tensor(0) == np::cuda::arch_has_fp8(major, minor), "fp8 live");
            test::check(np::gpu::has_fp4_tensor(0) == (np::cuda::arch_has_fp4(major, minor) &&
                                                       np::cuda::driver_version() >= NP_CUDA_DRIVER_BLACKWELL_MIN),
                        "fp4 live");
            test::check(np::gpu::is_blackwell(0) == np::cuda::arch_is_blackwell(major, minor), "blackwell live");
        }
        else
        {
            // No queryable device: predicates must equal the documented
            // driver-version fallback, never crash.
            const int v = np::cuda::driver_version();
            test::check(np::gpu::has_fp8_tensor(0) == (v >= NP_CUDA_DRIVER_HOPPER_MIN), "fp8 fallback");
            test::check(np::gpu::has_fp4_tensor(0) == (v >= NP_CUDA_DRIVER_BLACKWELL_MIN), "fp4 fallback");
        }
        // Negative device index never queries.
        int mj = -1, mn = -1;
        test::check(!np::cuda::cached_compute_capability(-1, mj, mn), "negative device");
    }

    // 5. cuBLAS GEMM: runs on real hardware when present (tolerance per the
    //    acceptance checklist: 1e-5 float, 1e-12 double); otherwise asserts
    //    the failure is reported, not silent.
    {
        constexpr std::size_t M = 96, N = 96, K = 96;
        std::vector<float> af(M * K), bf(K * N), cf(M * N, 0.0f), ref(M * N);
        fill_seq(af);
        fill_seq(bf, 0.5f);
        cpu_ref(af, bf, ref, M, N, K);
        const bool ok = np::gpu::detail::try_cuda_matmul(af.data(), bf.data(), cf.data(), M, N, K);
        if (ok)
        {
            std::printf("note: LIVE cublasSgemm path executed (96x96x96)\n");
            test::check(np::gpu::last_error() == CudaStatus::Ok, "cublas status Ok");
            test::check(close(cf, ref, 1e-5), "cublas sgemm matches CPU");
        }
        else
        {
            std::printf("note: sgemm fell through, last_error=%s\n", np::gpu::last_error_string());
            test::check(np::gpu::last_error() != CudaStatus::Ok, "cublas failure reported");
        }
    }
    {
        constexpr std::size_t M = 64, N = 48, K = 80; // non-square exercises lda/ldb/ldc
        std::vector<double> ad(M * K), bd(K * N), cd(M * N, 0.0), ref(M * N);
        fill_seq(ad);
        fill_seq(bd, 0.25);
        cpu_ref(ad, bd, ref, M, N, K);
        const bool ok = np::gpu::detail::try_cuda_matmul(ad.data(), bd.data(), cd.data(), M, N, K);
        if (ok)
        {
            std::printf("note: LIVE cublasDgemm path executed (64x48x80)\n");
            test::check(close(cd, ref, 1e-12), "cublas dgemm matches CPU");
        }
        else
        {
            std::printf("note: dgemm fell through, last_error=%s\n", np::gpu::last_error_string());
            test::check(np::gpu::last_error() != CudaStatus::Ok, "dgemm failure reported");
        }
    }

    // 6. Public dispatch still honors the CPU contract everywhere: matmul()
    //    is correct whether the GPU path ran or not.
    {
        constexpr std::size_t M = 130, N = 130, K = 130; // above small-size fast paths
        std::vector<double> ad(M * K), bd(K * N), cd(M * N, 0.0), ref(M * N);
        fill_seq(ad);
        fill_seq(bd, 0.5);
        cpu_ref(ad, bd, ref, M, N, K);
        np::gpu::matmul(ad.data(), bd.data(), cd.data(), M, N, K);
        test::check(close(cd, ref, 1e-12), "gpu::matmul correct via any path");
    }
    {
        // Sharded path (single-device here degrades to matmul; still correct).
        constexpr std::size_t M = 40, N = 32, K = 24;
        std::vector<float> af(M * K), bf(K * N), cf(M * N, 0.0f), ref(M * N);
        fill_seq(af);
        fill_seq(bf, 0.5f);
        cpu_ref(af, bf, ref, M, N, K);
        np::gpu::sharded_matmul(af.data(), bf.data(), cf.data(), M, N, K);
        test::check(close(cf, ref, 1e-5), "sharded_matmul correct");
    }

    // 7. Induced allocation failure is observable. Requesting an
    //    unallocatable buffer must fail (never returns a dangling pointer),
    //    and the status mapping reports it instead of a bare false.
    {
        void *p = reinterpret_cast<void *>(0x1);
        const std::size_t huge = (std::numeric_limits<std::size_t>::max)() / 2;
        const int code = np::cuda::rt_malloc(&p, huge);
        test::check(code != np::cuda::kCudaSuccess && p == nullptr, "huge alloc fails cleanly");
    }

    // 8. FFT: below-threshold short-circuits; at/above threshold the CUDA
    //    path roundtrips (forward+inverse, manual 1/N) when hardware exists,
    //    else reports failure for CPU fallback.
    {
        using C = std::complex<double>;
        test::check(!np::gpu::try_fft<C>(nullptr, nullptr, 16, false), "small fft short-circuit");
        const std::size_t N = np::tune::fft_threshold();
        std::vector<C> in(N), fwd(N), back(N);
        for (std::size_t i = 0; i < N; ++i)
            in[i] = C{std::sin(0.01 * static_cast<double>(i)), std::cos(0.013 * static_cast<double>(i))};
        const bool fwd_ok = np::gpu::detail::cuda_detail::try_cuda_fft(in.data(), fwd.data(), N, false);
        if (fwd_ok)
        {
            std::printf("note: LIVE cuFFT forward path executed (N=%zu)\n", N);
            const bool inv_ok = np::gpu::detail::cuda_detail::try_cuda_fft(fwd.data(), back.data(), N, true);
            test::check(inv_ok, "cufft inverse runs");
            if (inv_ok)
            {
                const double inv_n = 1.0 / static_cast<double>(N);
                bool ok = true;
                for (std::size_t i = 0; i < N && ok; ++i)
                {
                    const C got = back[i] * inv_n;
                    const double scale = 1.0 + std::abs(in[i]);
                    if (std::abs(got - in[i]) > 1e-9 * scale)
                        ok = false;
                }
                test::check(ok, "cufft roundtrip matches input");
            }
        }
        else
        {
            test::check(np::gpu::last_error() != CudaStatus::Ok, "fft failure reported");
        }
    }

    // 9. Streams construct, copy (shared native handle), and enqueue host
    //    work correctly. native_handle() is null without a CUDA runtime and
    //    non-null when stream creation succeeds — either is valid, but it
    //    must agree with library presence.
    {
        np::gpu::Stream s(0, 0);
        const bool have_rt = np::cuda::detail::cudart_lib() != nullptr;
        void *nh = s.native_handle();
        if (nh != nullptr)
            std::printf("note: LIVE cudaStream_t created (%p)\n", nh);
        // Null is always legal (CPU mode: no runtime, no device, or device
        // down — e.g. under ASan, whose allocator the NVIDIA driver rejects).
        // Non-null additionally requires the runtime library to be present.
        test::check(nh == nullptr || have_rt, "native handle matches runtime");
        auto fut = s.enqueue([] { return 42; });
        test::check(fut.get() == 42, "stream enqueue runs host work");
        np::gpu::Stream copy = s; // shares the native stream by design
        test::check(copy.native_handle() == s.native_handle(), "stream copy shares handle");
        auto streams = np::gpu::make_streams(4);
        test::check(streams.size() == 4, "make_streams count");
        int total = 0;
        for (auto &st : streams)
            total += st.enqueue([] { return 1; }).get();
        test::check(total == 4, "make_streams enqueue");
        // Device count is environment-dependent; assert self-consistency:
        // zero iff no CUDA backend is detected.
        test::check((np::gpu::cuda_device_count() == 0) == !np::gpu::detail::has_cuda_backend(),
                    "device count consistent");
    }

    // 10. No-driver degradation: detection is false, large try_matmul is
    //     false, and last_error explains the CUDA miss. (On GPU machines
    //     these branches flip to live checks above; the matmul-correctness
    //     asserts in (6) hold on both.)
    if (!np::gpu::is_available())
    {
        std::vector<float> a(1100 * 1100, 1.0f), b(1100 * 1100, 1.0f), c(1100 * 1100, 0.0f);
        test::check(!np::gpu::try_matmul(a.data(), b.data(), c.data(), 1100, 1100, 1100),
                    "try_matmul false without GPU");
        std::printf("note: no CUDA backend; live-GPU asserts skipped\n");
    }

    return test::failures() ? 1 : 0;
}
