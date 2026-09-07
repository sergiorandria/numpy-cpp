/**
 * @file accelerator.hpp
 * @brief Heterogeneous accelerator dispatcher — CPU/GPU/Loihi/ReRAM/Photonics.
 *
 * GPU path now dispatches via np::gpu (OpenMP target / CUDA driver dlopen) for
 * powerful workstations. CPU path uses blocked+SIMD+ThreadPool. AutoAccelerator
 * benchmarks CPU vs GPU on first call and caches the winner.
 */
#ifndef NP_ACCELERATOR_HPP
#define NP_ACCELERATOR_HPP

#include "api_macros.hpp"
#include "gpu.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"
#include <chrono>
#include <memory>
#include <string>

// GPU offload tuning (macros, no magic numbers in logic)
#define NP_ACCEL_GPU_SIZE_THRESH 1000000
#define NP_ACCEL_BENCH_DIM 128

namespace np::accelerator
{

struct IAccelerator
{
    virtual ~IAccelerator() = default;
    virtual ndarray<float> matmul(const ndarray<float> &a, const ndarray<float> &b) = 0;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
    NP_NODISCARD virtual bool is_available() const noexcept
    {
        return true;
    }
};

struct CPUAccelerator : IAccelerator
{
    ndarray<float> matmul(const ndarray<float> &a, const ndarray<float> &b) override
    {
        return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "CPU";
    }
};

struct GPUAccelerator : IAccelerator
{
    ndarray<float> matmul(const ndarray<float> &a, const ndarray<float> &b) override
    {
        if (a.ndim() != 2 || b.ndim() != 2)
            return linalg::matmul(a, b);
        if (!a.is_contiguous() || !b.is_contiguous())
            return linalg::matmul(a, b);
        const std::size_t M = static_cast<std::size_t>(a.shape[0]);
        const std::size_t K = static_cast<std::size_t>(a.shape[1]);
        const std::size_t N = static_cast<std::size_t>(b.shape[1]);
        if (K != static_cast<std::size_t>(b.shape[0]))
            return linalg::matmul(a, b);
        if (a.size() > 0 && b.size() > 0)
        {
            ndarray<float> out(std::vector<int>{static_cast<int>(M), static_cast<int>(N)});
            const float *ad = a.data().data();
            const float *bd = b.data().data();
            float *cd = out.data().data();
            if (gpu::try_matmul(ad, bd, cd, M, N, K))
                return out;
            gpu::matmul(ad, bd, cd, M, N, K);
            return out;
        }
        return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "GPU";
    }
    NP_NODISCARD bool is_available() const noexcept override
    {
        return gpu::is_available();
    }
};

struct LoihiAccelerator : IAccelerator
{
    ndarray<float> matmul(const ndarray<float> &a, const ndarray<float> &b) override
    {
        return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "Loihi2";
    }
};

struct ReRAMAccelerator : IAccelerator
{
    ndarray<float> matmul(const ndarray<float> &a, const ndarray<float> &b) override
    {
        return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "ReRAM";
    }
};

struct AutoAccelerator : IAccelerator
{
    mutable std::shared_ptr<IAccelerator> cached_;
    mutable std::once_flag once_;
    ndarray<float> matmul(const ndarray<float> &a, const ndarray<float> &b) override
    {
        std::call_once(once_, [&] {
            if (gpu::is_available() && a.size() * b.size() > NP_ACCEL_GPU_SIZE_THRESH)
            {
                auto bench = [](IAccelerator &acc) -> double {
                    auto aa = np::eye<float>(NP_ACCEL_BENCH_DIM);
                    auto bb = np::eye<float>(NP_ACCEL_BENCH_DIM);
                    auto t0 = std::chrono::steady_clock::now();
                    auto cc = acc.matmul(aa, bb);
                    auto t1 = std::chrono::steady_clock::now();
                    (void)cc;
                    return std::chrono::duration<double, std::milli>(t1 - t0).count();
                };
                CPUAccelerator cpu;
                GPUAccelerator gpu;
                double t_cpu = bench(cpu);
                double t_gpu = bench(gpu);
                cached_ = (t_gpu < t_cpu) ? std::static_pointer_cast<IAccelerator>(std::make_shared<GPUAccelerator>())
                                          : std::static_pointer_cast<IAccelerator>(std::make_shared<CPUAccelerator>());
            }
            else
            {
                cached_ = std::make_shared<CPUAccelerator>();
            }
        });
        return cached_->matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        if (cached_)
            return "Auto(" + cached_->name() + ")";
        return gpu::is_available() ? "Auto(GPU|CPU)" : "Auto(CPU)";
    }
    NP_NODISCARD bool is_available() const noexcept override
    {
        return true;
    }
};

struct AcceleratorFactory
{
    NP_NODISCARD static std::shared_ptr<IAccelerator> cpu()
    {
        return std::make_shared<CPUAccelerator>();
    }
    NP_NODISCARD static std::shared_ptr<IAccelerator> gpu()
    {
        return std::make_shared<GPUAccelerator>();
    }
    NP_NODISCARD static std::shared_ptr<IAccelerator> loihi()
    {
        return std::make_shared<LoihiAccelerator>();
    }
    NP_NODISCARD static std::shared_ptr<IAccelerator> reram()
    {
        return std::make_shared<ReRAMAccelerator>();
    }
    NP_NODISCARD static std::shared_ptr<IAccelerator> auto_select()
    {
        return std::make_shared<AutoAccelerator>();
    }
    NP_NODISCARD static std::shared_ptr<IAccelerator> powerful()
    {
        if (gpu::is_available())
            return gpu();
        return auto_select();
    }
};

} // namespace np::accelerator

#endif // NP_ACCELERATOR_HPP
