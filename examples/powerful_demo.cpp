/**
 * @file powerful_demo.cpp
 * @brief Powerful workstation + GPU demo — AVX2, OpenMP, GPU dispatch.
 *
 * Build: cmake --preset powerful && cmake --build --preset powerful -j && ./build/examples/powerful_demo
 * Or: g++ -O3 -mavx2 -mfma -fopenmp -DNP_ENABLE_GPU -DNP_ENABLE_OPENMP -I include examples/powerful_demo.cpp -o
 * /tmp/demo -ldl -lomp && /tmp/demo
 */
#include <chrono>
#include <cstdio>
#include <np/np.hpp>

int main()
{
    printf("tune: L3 %zu KB, threads %zu, block_f32 %zu, gpu_thresh %zu\n", np::tune::l3_cache_bytes() / 1024,
           np::tune::hardware_threads(), np::tune::optimal_block_f32(), np::tune::gpu_threshold_flops());
    printf("gpu: available %s, devices %d, backend %d\n", np::gpu::is_available() ? "yes" : "no",
           np::gpu::device_count(), (int)np::gpu::preferred_backend());

    for (int N : {256, 512, 1024})
    {
        auto a = np::eye<float>(N);
        auto b = np::eye<float>(N);
        auto t0 = std::chrono::steady_clock::now();
        auto c = np::linalg::matmul(a, b);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("linalg matmul %4dx%4d: %.2f ms (first 0,0=%.1f)\n", N, N, ms, c(0, 0));

        auto gpu = np::accelerator::AcceleratorFactory::gpu();
        t0 = std::chrono::steady_clock::now();
        auto cg = gpu->matmul(a, b);
        t1 = std::chrono::steady_clock::now();
        printf("  GPUAccelerator: %.2f ms (avail %s)\n", std::chrono::duration<double, std::milli>(t1 - t0).count(),
               gpu->is_available() ? "yes" : "no");
        (void)cg;

        auto ten = np::tensor::TensorFactory::hopper();
        t0 = std::chrono::steady_clock::now();
        auto ct = ten->matmul(a, b);
        t1 = std::chrono::steady_clock::now();
        printf("  Hopper FP8: %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
        (void)ct;
    }

    {
        auto arr = np::eye<float>(512);
        auto h = np::mem::migrate_to_hbm(arr);
        auto g = np::mem::migrate_to_device(arr);
        auto p = np::mem::migrate_to_pinned(arr);
        printf("mem: hbm %zu, device %zu (on_device %d), pinned %zu\n", h.size(), g.size(), g.on_device, p.size());
    }
    return 0;
}
