/**
 * @file bench_hardware.cpp
 * @brief Micro-benchmark for hardware backends — HBM, tensor, neuromorphic, padic,
 * lattice + powerful GPU/CPU GEMM.
 *
 * Measures throughput for:
 *   1. HBM migrate (mem::migrate_to_hbm)
 *   2. Tensor FP8 matmul (tensor::matmul_fp8)
 *   3. ReRAM crossbar dot (analog::Crossbar)
 *   4. Photonics mesh apply
 *   5. Neuromorphic spike encode + LIF
 *   6. Padic Hensel lift
 *   7. Lattice LLL
 *   8. Powerful GEMM 512/1024 CPU vs GPU vs Auto (gpu::, accelerator::)
 *
 * Build powerful: cmake --preset powerful && cmake --build --preset powerful -j && ./build/tests/bench_hardware
 * Not part of ctest; run: ./build/tests/bench_hardware
 */
#include <chrono>
#include <cstdio>
#include <np/np.hpp>

using Clock = std::chrono::steady_clock;
template <typename Fn> double ms(Fn &&fn, int iters = 3)
{
    double best = 1e18;
    for (int i = 0; i < iters; ++i)
    {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        double d = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (d < best)
            best = d;
    }
    return best;
}

template <typename T> double bench_gemm(int N, const char *label)
{
    auto a = np::eye<T>(N);
    auto b = np::eye<T>(N);
    double t_linalg = ms([&] {
        auto c = np::linalg::matmul(a, b);
        (void)c;
    });
    double t_gpu = 0, t_auto = 0, t_tensor = 0;
    if constexpr (std::is_same_v<T, float>)
    {
        auto gpu_acc = np::accelerator::AcceleratorFactory::gpu();
        t_gpu = ms([&] {
            auto c = gpu_acc->matmul(a, b);
            (void)c;
        });
        auto auto_acc = np::accelerator::AcceleratorFactory::auto_select();
        t_auto = ms([&] {
            auto c = auto_acc->matmul(a, b);
            (void)c;
        });
        t_tensor = ms([&] {
            auto c = np::tensor::matmul_fp8(a, b);
            (void)c;
        });
    }
    else
    {
        t_gpu = t_auto = t_tensor = 0;
    }
    printf("%s %dx%d: linalg %.2f ms | GPU %.2f ms | Auto %.2f ms | tensor_fp8 %.2f ms (gpu %s)\n", label, N, N,
           t_linalg, t_gpu, t_auto, t_tensor, np::gpu::is_available() ? "yes" : "no");
    return t_linalg;
}

int main()
{
    auto a = np::eye<float>(64);
    auto b = np::eye<float>(64);
    printf("=== hardware backends (64) ===\n");
    printf("HBM migrate: %.2f ms\n", ms([&] {
               auto h = np::mem::tag_hbm_hint(a);
               (void)h;
           }));
    printf("tensor matmul_fp8: %.2f ms\n", ms([&] {
               auto c = np::tensor::matmul_fp8(a, b);
               (void)c;
           }));
    printf("ReRAM dot: %.2f ms\n", ms([&] {
               np::analog::Crossbar cb(a);
               auto x = np::ndarray<float>(std::vector<int>{64});
               for (int i = 0; i < 64; ++i)
                   x[i] = 1.0f;
               auto y = cb.dot(x);
               (void)y;
           }));
    printf("photonics: %.2f ms\n", ms([&] {
               auto mesh = np::photonics::PhotonicsFactory::identity(4);
               auto x = np::ndarray<std::complex<double>>(std::vector<int>{4});
               for (int i = 0; i < 4; ++i)
                   x[i] = {1, 0};
               auto y = mesh.apply(x);
               (void)y;
           }));
    printf("neuromorphic encode: %.2f ms\n", ms([&] {
               auto arr = np::ndarray<float>(std::vector<int>{64});
               for (int i = 0; i < 64; ++i)
                   arr[i] = 0.5f;
               auto ea = np::spike::encode_rate(arr, 100, 100);
               (void)ea;
           }));
    printf("padic Hensel: %.2f ms\n", ms([&] {
               np::padic::Padic<int64_t> x0(7, 3, 10);
               auto f = [](const np::padic::Padic<int64_t> &x) {
                   return np::padic::Padic<int64_t>(x.p, x.value * x.value - 2, x.prec);
               };
               auto df = [](const np::padic::Padic<int64_t> &x) {
                   return np::padic::Padic<int64_t>(x.p, 2 * x.value, x.prec);
               };
               auto r = np::padic::HenselStrategy<int64_t>(5).lift(x0, f, df);
               (void)r;
           }));
    printf("lattice LLL: %.2f ms\n", ms([&] {
               auto lat = np::lattice::LatticeFactory::cubic<double>(4);
               auto r = lat.lll_reduce();
               (void)r;
           }));

    printf("\n=== powerful GEMM (CPU SIMD+OpenMP vs GPU) ===\n");
    printf("GPU available: %s (%d devices, backend %d) | OpenMP %s | AVX2 %s\n", np::gpu::is_available() ? "yes" : "no",
           np::gpu::device_count(), (int)np::gpu::preferred_backend(),
#if defined(_OPENMP)
           "yes",
#else
           "no",
#endif
#if defined(__AVX2__)
           "yes"
#else
           "no"
#endif
    );
    bench_gemm<float>(64, "GEMM float");
    bench_gemm<float>(256, "GEMM float");
    bench_gemm<float>(512, "GEMM float");
    bench_gemm<float>(1024, "GEMM float");
    bench_gemm<double>(512, "GEMM double");

    printf("\n=== memory (HBM/GPU pinned) ===\n");
    {
        auto arr = np::eye<float>(512);
        printf("migrate_to_hbm 512: %.2f ms\n", ms([&] {
                   auto h = np::mem::tag_hbm_hint(arr);
                   (void)h;
               }));
        printf("migrate_to_device 512: %.2f ms\n", ms([&] {
                   auto g = np::mem::tag_device_hint(arr);
                   (void)g;
               }));
        printf("migrate_to_pinned 512: %.2f ms\n", ms([&] {
                   auto p = np::mem::tag_pinned_hint(arr);
                   (void)p;
               }));
    }
    return 0;
}
