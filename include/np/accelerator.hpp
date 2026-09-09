/**
 * @file accelerator.hpp
 * @brief Heterogeneous accelerator dispatcher — CPU/GPU/ReRAM-sim.
 *
 * GPU path dispatches via np::gpu (OpenMP target / CUDA driver dlopen).
 * CPU path uses blocked+SIMD+ThreadPool. ReRAM path runs the analog
 * crossbar VMM simulation from memristor.hpp (DAC/ADC quantization +
 * device noise), not real ReRAM hardware. AutoAccelerator benchmarks CPU
 * vs GPU per workload size-class and caches the winner per class.
 *
 * NOTE (honesty audit): an earlier revision had a LoihiAccelerator named
 * "Loihi2" whose matmul() was linalg::matmul — no Loihi hardware or
 * simulation of any kind. It is deleted; there is no neuromorphic matmul
 * path in this file.
 */
#ifndef NP_ACCELERATOR_HPP
#define NP_ACCELERATOR_HPP

#include "api_macros.hpp"
#include "gpu.hpp"
#include "linalg.hpp"
#include "memristor.hpp"
#include "ndarray.hpp"
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
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

// Analog in-memory-compute matmul via the memristor crossbar simulation:
// each output row is a hardware-aware VMM (DAC quantization, analog dot,
// ADC quantization, deterministic seed) over the B matrix held as crossbar
// weights. This is a functional device model, not a performance path and
// not real ReRAM hardware — hence "ReRAM-sim", with is_available() true
// because software simulation needs no device.
//
// Precision/range caveat (measured, not assumed): the default 8-bit DAC/ADC
// model is coarse — small well-conditioned inputs stay within ~1% of ideal,
// but magnitudes beyond the DAC full-scale (MemristorConfig::max_input_
// voltage, default 1.0) saturate and large dynamic ranges can show tens of
// percent relative error, exactly like naive-mapped analog hardware. Pass a
// custom MemristorConfig (e.g. ideal 0-bit quantization) for bit-close
// results, or normalized inputs for representative analog behavior.
struct ReRAMAccelerator : IAccelerator
{
    analog::MemristorConfig config;

    ReRAMAccelerator()
    {
        config.dac_bits = 8;
        config.adc_bits = 8;
        config.seed = 0xC0FFEEu;
    }
    explicit ReRAMAccelerator(analog::MemristorConfig cfg) : config(std::move(cfg))
    {
    }
    ndarray<float> matmul(const ndarray<float> &a, const ndarray<float> &b) override
    {
        if (a.ndim() != 2 || b.ndim() != 2 || a.shape[1] != b.shape[0] || a.size() == 0 || b.size() == 0)
        {
            return linalg::matmul(a, b);
        }
        analog::Crossbar xb(b, config);
        const int m = a.shape[0];
        const int kdim = a.shape[1];
        const int n = b.shape[1];
        ndarray<float> out(std::vector<int>{m, n});
        ndarray<float> row(std::vector<int>{kdim});
        for (int i = 0; i < m; ++i)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            for (int k = 0; k < kdim; ++k)
            {
                // Logical access: a may be a strided view.
                row.data()[static_cast<std::size_t>(k)] = a(ii, static_cast<std::size_t>(k));
            }
            const ndarray<float> y = xb.apply(row);
            for (int j = 0; j < n; ++j)
            {
                const std::size_t jj = static_cast<std::size_t>(j);
                out(ii, jj) = y.data()[jj];
            }
        }
        return out;
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "ReRAM-sim";
    }
};

struct AutoAccelerator : IAccelerator
{
    // NOTE (honesty audit): an earlier revision benchmarked once on fixed
    // 128x128 identity inputs and cached the winner forever, so every later
    // shape/dtype got the stale choice. The cache is now keyed by workload
    // size class (floor(log2(flops))), and each class is benchmarked on the
    // actual caller inputs the first time it appears.
    mutable std::mutex mtx_;
    mutable std::map<std::uint64_t, std::shared_ptr<IAccelerator>> winners_;
    static std::uint64_t size_class(std::size_t flops) noexcept
    {
        std::uint64_t bucket = 0;
        while ((flops >>= 1) != 0)
        {
            ++bucket;
        }
        return bucket;
    }
    ndarray<float> matmul(const ndarray<float> &a, const ndarray<float> &b) override
    {
        const std::uint64_t key = (a.ndim() == 2 && b.ndim() == 2)
                                      ? size_class(a.size() * b.size())
                                      : std::numeric_limits<std::uint64_t>::max();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            auto it = winners_.find(key);
            if (it != winners_.end())
            {
                return it->second->matmul(a, b);
            }
        }
        std::shared_ptr<IAccelerator> winner = std::make_shared<CPUAccelerator>();
        if (gpu::is_available() && a.size() * b.size() > NP_ACCEL_GPU_SIZE_THRESH)
        {
            auto bench = [&](IAccelerator &acc) -> double {
                const auto t0 = std::chrono::steady_clock::now();
                auto cc = acc.matmul(a, b);
                const auto t1 = std::chrono::steady_clock::now();
                (void)cc;
                return std::chrono::duration<double, std::milli>(t1 - t0).count();
            };
            CPUAccelerator cpu;
            GPUAccelerator gpu;
            if (bench(gpu) < bench(cpu))
            {
                winner = std::make_shared<GPUAccelerator>();
            }
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            winners_[key] = winner;
        }
        return winner->matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "Auto";
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
