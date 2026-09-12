/**
 * @file test_tensor_core.cpp
 * @brief Tests for np::tensor backends, honest multiply counts, and
 * capability degradation.
 */
#include "test_util.hpp"

#include "np/linalg.hpp"
#include "np/tensor_core.hpp"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

// Counting scalar: routes through alpha_evolve::matmul_4x4_49 to count the
// ACTUAL scalar multiplies executed (not the comment). All ops noexcept so
// the noexcept kernel accepts this type; production instantiates float.
struct CountMul
{
    float v = 0.0f;
    inline static std::size_t mults = 0;
    CountMul() = default;
    CountMul(float x) noexcept : v(x)
    {
    }
    CountMul(const CountMul &) = default;
    CountMul &operator=(const CountMul &) = default;
    friend CountMul operator+(const CountMul &a, const CountMul &b) noexcept
    {
        return CountMul(a.v + b.v);
    }
    friend CountMul operator-(const CountMul &a, const CountMul &b) noexcept
    {
        return CountMul(a.v - b.v);
    }
    friend CountMul operator*(const CountMul &a, const CountMul &b) noexcept
    {
        ++mults;
        return CountMul(a.v * b.v);
    }
};

namespace
{

np::ndarray<float> eye4()
{
    np::ndarray<float> e(std::vector<int>{4, 4});
    std::fill(e.data().begin(), e.data().end(), 0.0f);
    for (int i = 0; i < 4; ++i)
        e(i, i) = 1.0f;
    return e;
}

np::ndarray<float> rand_mat(std::mt19937 &rng, int m, int n)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    np::ndarray<float> a(std::vector<int>{m, n});
    for (auto &v : a.data())
        v = dist(rng);
    return a;
}

bool close_rel(const np::ndarray<float> &x, const np::ndarray<float> &y, double eps)
{
    if (x.shape != y.shape)
        return false;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double xa = x.data()[i], ya = y.data()[i];
        if (std::fabs(xa - ya) > eps * (1.0 + std::fabs(xa) + std::fabs(ya)))
            return false;
    }
    return true;
}

} // namespace

int main()
{
    using namespace np::tensor;

    // Honest backend names (regression: these used to claim Hopper/AMX).
    {
        auto cpu = TensorFactory::cpu();
        test::check(cpu->name() == "CPU", "CPU backend");
        auto gpu = TensorFactory::gpu_fp32();
        test::check(gpu->name() == "GPU-FP32", "GPU-FP32 name");
        test::check(gpu->is_available() == np::gpu::is_available(), "GPU-FP32 availability honest");
        auto blk = TensorFactory::cpu_blocked();
        test::check(blk->name() == "CPU-blocked", "CPU-blocked name");
        auto s44 = TensorFactory::strassen_4x4();
        test::check(s44->name() == "Strassen-49-4x4", "tiled backend name");
        test::check(s44->rank() == 49, "tiled backend rank");
        static_assert(np::tensor::alpha_evolve::rank_4x4 == 49, "rank_4x4 must be 49, not 48");
        test::check(np::tensor::alpha_evolve::optimizer::best_rank(4, 4, 4) == 49, "optimizer 4x4 rank");
    }

    // Multiply count, measured on the executed path: 7 products + 7 each.
    {
        float X[4] = {1, 2, 3, 4}, Y[4] = {5, 6, 7, 8};
        CountMul CX[4], CY[4], CO[7]; // helper writes exactly 7 products
        for (int i = 0; i < 4; ++i)
        {
            CX[i] = X[i];
            CY[i] = Y[i];
        }
        CountMul::mults = 0;
        np::tensor::alpha_evolve::strassen_2x2_products<CountMul>(CX, CY, CO);
        test::check(CountMul::mults == 7, "2x2 helper is 7 mults");

        CountMul A[16], B[16], C[16];
        for (int i = 0; i < 16; ++i)
        {
            A[i] = static_cast<float>(i + 1);
            B[i] = static_cast<float>(16 - i);
        }
        CountMul::mults = 0;
        np::tensor::alpha_evolve::matmul_4x4_49<CountMul>(A, B, C);
        test::check(CountMul::mults == 49, "4x4 kernel is 49 mults, not 48");
        // The counted path must compute the same result as float.
        float Af[16], Bf[16], Cf[16];
        for (int i = 0; i < 16; ++i)
        {
            Af[i] = static_cast<float>(i + 1);
            Bf[i] = static_cast<float>(16 - i);
        }
        np::tensor::alpha_evolve::matmul_4x4_49<float>(Af, Bf, Cf);
        bool same = true;
        for (int i = 0; i < 16 && same; ++i)
            same = (C[i].v == Cf[i]);
        test::check(same, "counted kernel matches float kernel");
    }

    // Randomized correctness vs linalg::matmul.
    {
        std::mt19937 rng(12345);
        for (int trial = 0; trial < 20; ++trial) // 4x4 fast path
        {
            auto a = rand_mat(rng, 4, 4), b = rand_mat(rng, 4, 4);
            auto got = alpha_evolve::matmul(a, b);
            auto ref = np::linalg::matmul(a, b);
            if (!close_rel(got, ref, 1e-6))
            {
                test::check(false, "4x4 randomized");
                break;
            }
        }
        test::check(true, "4x4 randomized done");
        for (int trial = 0; trial < 5; ++trial) // tiled 8x8 / 16x16
        {
            auto a = rand_mat(rng, 8, 8), b = rand_mat(rng, 8, 8);
            test::check(close_rel(alpha_evolve::matmul(a, b), np::linalg::matmul(a, b), 1e-5), "8x8 tiled");
            auto c = rand_mat(rng, 16, 16), d = rand_mat(rng, 16, 16);
            test::check(close_rel(alpha_evolve::matmul(c, d), np::linalg::matmul(c, d), 1e-5), "16x16 tiled");
        }
        // Non-multiples of 4 fall through to Strassen/naive, still correct.
        {
            auto a = rand_mat(rng, 6, 6), b = rand_mat(rng, 6, 6);
            test::check(close_rel(alpha_evolve::matmul(a, b), np::linalg::matmul(a, b), 1e-5), "6x6 fallback");
            auto e = rand_mat(rng, 4, 6), f = rand_mat(rng, 6, 8);
            test::check(close_rel(alpha_evolve::matmul(e, f), np::linalg::matmul(e, f), 1e-5), "4x6x8 fallback");
        }
        // Backend-level entry points agree too.
        {
            auto a = rand_mat(rng, 4, 4), b = rand_mat(rng, 4, 4);
            test::check(close_rel(Strassen4x4Backend{}.matmul(a, b), np::linalg::matmul(a, b), 1e-6), "backend 4x4");
            auto c = rand_mat(rng, 8, 8), d = rand_mat(rng, 8, 8);
            test::check(close_rel(Strassen4x4Backend{}.matmul(c, d), np::linalg::matmul(c, d), 1e-5), "backend 8x8");
        }
    }

    // Capability honesty on non-Hopper hardware: a new-enough driver must
    // not imply tensor-core support (the old driver-version heuristic did).
    {
        int mj = 0, mn = 0;
        if (np::cuda::cached_compute_capability(0, mj, mn))
        {
            test::check(np::gpu::has_fp8_tensor(0) == np::cuda::arch_has_fp8(mj, mn), "fp8 live");
            test::check(np::gpu::has_fp4_tensor(0) == (np::cuda::arch_has_fp4(mj, mn) &&
                                                       np::cuda::driver_version() >= NP_CUDA_DRIVER_BLACKWELL_MIN),
                        "fp4 live");
            test::check(np::gpu::is_blackwell(0) == np::cuda::arch_is_blackwell(mj, mn), "blackwell live");
            if (mj < 9) // pre-Hopper silicon: all three must be false regardless of driver
            {
                test::check(!np::gpu::has_fp8_tensor(0), "pre-hopper no fp8");
                test::check(!np::gpu::has_fp4_tensor(0), "pre-hopper no fp4");
                test::check(!np::gpu::is_blackwell(0), "pre-hopper no blackwell");
            }
        }
    }

    // Pre-existing coverage, kept working.
    {
        auto e = eye4();
        auto c = matmul_fp8(e, e, 1.0f, 1.0f);
        test::check(std::abs(c(0, 0) - 1) < 1e-3, "tensor matmul_fp8");
        QuantizedTensor<float> qt{e, 0.5f, TensorDtype::FP8};
        auto dq = qt.dequantize();
        test::check(dq.size() == 16, "quantized dequant");
        auto q = quantize(e, 0.5f);
        test::check(q.size() == 16, "quantize");
    }

    return test::failures() ? 1 : 0;
}
