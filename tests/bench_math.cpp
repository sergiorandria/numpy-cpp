/**
 * @file bench_math.cpp
 * @brief Micro-benchmark for math.hpp binary/unary ufunc paths.
 *
 * Compares:
 *   1. Old-style ufunc_binary (per-element index mapping with
 *      get/set) vs. the new detail::elementwise implementation
 *      (precomputed adjusted strides + Odometer).
 *   2. np::square SIMD fast path vs. the scalar ufunc_unary loop.
 *   3. np::maximum with broadcasting (contiguous lhs, size-2 rhs).
 *
 * Built with -O2 -mavx so the np::simd kernels (NP_SIMD_AVX) are
 * actually compiled and dispatched. Not part of `ctest`; run it
 * directly: ./build/tests/bench_math
 */
#include <np/np.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

using Clock = std::chrono::steady_clock;

template <typename Fn> double ms(Fn &&fn, int iters = 5)
{
    double best = 1e18;
    for (int i = 0; i < iters; ++i)
    {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        const double d = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best = std::min(best, d);
    }
    return best;
}

// Old-style ufunc_binary: recomputes index vectors per element and
// maps through get()/set(). This mirrors the implementation that was
// replaced by np::detail::elementwise.
auto old_binary_max(const np::ndarray<double> &lhs, const np::ndarray<double> &rhs) -> np::ndarray<double>
{
    using R = double;
    const auto out_shape = np::detail::broadcast_shapes(lhs.shape, rhs.shape);
    np::ndarray<double> result(out_shape, np::dtype_of<R>);

    const auto ndim_out = out_shape.size();
    std::vector<std::size_t> idx(ndim_out, 0);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        std::vector<std::size_t> idx_lhs(lhs.ndim(), 0);
        std::vector<std::size_t> idx_rhs(rhs.ndim(), 0);

        for (std::size_t d = 0; d < ndim_out; ++d)
        {
            if (d >= ndim_out - lhs.ndim())
            {
                const auto d_lhs = d - (ndim_out - lhs.ndim());
                idx_lhs[d_lhs] = (lhs.shape[d_lhs] == 1) ? 0 : idx[d];
            }
            if (d >= ndim_out - rhs.ndim())
            {
                const auto d_rhs = d - (ndim_out - rhs.ndim());
                idx_rhs[d_rhs] = (rhs.shape[d_rhs] == 1) ? 0 : idx[d];
            }
        }

        const double val_lhs = lhs.get(idx_lhs);
        const double val_rhs = rhs.get(idx_rhs);
        result.set(idx, std::max(val_lhs, val_rhs));

        for (std::size_t d = ndim_out; d-- > 0;)
        {
            if (++idx[d] < static_cast<std::size_t>(out_shape[d]))
            {
                break;
            }
            idx[d] = 0;
        }
    }
    return result;
}

int main()
{
    using namespace np;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const std::size_t N = 1U << 20;

    auto a = linspace(1.0, 2.0, static_cast<int>(N));
    auto b = linspace(0.5, 1.5, static_cast<int>(N));

    std::printf("N = %zu doubles (%.1f MiB each)\n", N, N * 8.0 / (1024.0 * 1024.0));

    // --- binary contiguous: maximum ---
    const double t_old = ms([&] { (void)old_binary_max(a, b); }, 1);
    const double t_new = ms([&] { (void)maximum(a, b); });
    std::printf("maximum (1D, contiguous):  old ufunc_binary %7.3f ms | "
                "elementwise %7.3f ms | %.2fx\n",
                t_old, t_new, t_old / t_new);

    // --- binary broadcast: maximum(a2d, row) ---
    auto a2d = linspace(1.0, 2.0, static_cast<int>(N)); // {N/2, 2}
    a2d = a2d.reshape({static_cast<int>(N) / 2, 2});
    auto row = asarray(std::vector<double>{1.0, 2.0});
    const double t_bc = ms([&] { (void)maximum(a2d, row); });
    std::printf("maximum (broadcast {N/2,2} vs {2}): elementwise %7.3f ms\n", t_bc);

    // --- binary divide (SIMD fast path when contiguous same-typed) ---
    const double t_div = ms([&] { (void)divide(a, b); });
    std::printf("divide (contiguous SIMD):  %7.3f ms\n", t_div);

    // --- unary square: SIMD vs scalar ---
    const double t_simd = ms([&] { (void)square(a); });
    const double t_scalar = ms([&] { (void)detail::ufunc_unary(a, [](const double &v) { return v * v; }); });
    std::printf("square:  SIMD fast path %7.3f ms | scalar ufunc_unary %7.3f ms | "
                "%.2fx\n",
                t_simd, t_scalar, t_scalar / t_simd);

    // --- fma (single fused pass) ---
    auto c = linspace(0.1, 0.9, static_cast<int>(N));
    const double t_fma = ms([&] { (void)fma(a, b, c); });
    std::printf("fma (ternary, fused):      %7.3f ms\n", t_fma);

    return 0;
}
