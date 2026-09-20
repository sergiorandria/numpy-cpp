/**
 * @file test_vecmath.cpp
 * @brief Tests for the portable SIMD vector math library (vecmath.hpp).
 *
 * Covers three layers:
 *   1. np::vecmath batch kernels: worst-case ULP error vs. correctly rounded
 *      libm over wide sweeps (bounds documented in vecmath.hpp), for float
 *      and double, at aligned and misaligned sizes.
 *   2. Special values: bitwise-exact match with std:: for zeros, infinities,
 *      NaN, subnormals, overflow/underflow and the atan2/pow/hypot edge
 *      tables.
 *   3. Integration: np:: math ufuncs and np::simd wrappers route through the
 *      vector kernels for contiguous float/double arrays (including the
 *      `out` overloads and broadcast fallbacks).
 */

#include "test_util.hpp"
#include <np/np.hpp>
#include <np/vecmath.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

double ulp_of(double ref)
{
    if (!std::isfinite(ref) || ref == 0.0)
    {
        return std::numeric_limits<double>::min();
    }
    const double nxt = std::nextafter(ref, std::numeric_limits<double>::infinity());
    const double d = std::abs(nxt - ref);
    return d == 0.0 ? std::numeric_limits<double>::min() : d;
}

float ulp_of(float ref)
{
    if (!std::isfinite(ref) || ref == 0.0f)
    {
        return std::numeric_limits<float>::min();
    }
    const float nxt = std::nextafterf(ref, std::numeric_limits<float>::infinity());
    const float d = std::abs(nxt - ref);
    return d == 0.0f ? std::numeric_limits<float>::min() : d;
}

// Worst-case ULP error of a unary kernel over [lo, hi] (linear sampling).
template <typename T, typename FVec, typename FStd>
double sweep_unary(FVec fv, FStd fs, T lo, T hi, std::size_t n, bool log_spaced = false)
{
    std::vector<T> in(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const double t = n == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n - 1);
        if (log_spaced)
        {
            in[i] = static_cast<T>(static_cast<double>(lo) *
                                   std::pow(static_cast<double>(hi) / static_cast<double>(lo), t));
        }
        else
        {
            in[i] = static_cast<T>(static_cast<double>(lo) + (static_cast<double>(hi) - static_cast<double>(lo)) * t);
        }
    }
    fv(in.data(), out.data(), n);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const T ref = fs(in[i]);
        const T got = out[i];
        double e = 0.0;
        if (std::isnan(ref))
        {
            e = std::isnan(got) ? 0.0 : 1e18;
        }
        else if (std::isinf(ref))
        {
            e = (got == ref) ? 0.0 : 1e18;
        }
        else if (ref == T(0))
        {
            e = (got == T(0) && std::signbit(got) == std::signbit(ref)) ? 0.0 : 1e18;
        }
        else
        {
            const double u = std::is_same_v<T, double> ? ulp_of(static_cast<double>(ref))
                                                       : static_cast<double>(ulp_of(static_cast<float>(ref)));
            e = std::abs(static_cast<double>(got) - static_cast<double>(ref)) / u;
            if (!std::isfinite(e))
            {
                e = 1e18;
            }
        }
        worst = std::max(worst, e);
    }
    return worst;
}

template <typename T, typename FVec, typename FStd>
void check_unary(const char *name, FVec fv, FStd fs, T lo, T hi, double bound, bool log_spaced = false)
{
    constexpr std::size_t n = 4096;
    const double w = sweep_unary<T>(fv, fs, lo, hi, n, log_spaced);
    if (!(w <= bound))
    {
        char buf[192];
        std::snprintf(buf, sizeof(buf), "%s worst %.3g ulp exceeds %.3g", name, w, bound);
        test::check(false, "vecmath accuracy", buf);
    }
}

// Absolute-error check for trig: near zeros the relative ULP metric is
// meaningless (both libm and this library have reduction error there), so
// the error is measured in ULPs of 1.0.
template <typename T, typename FVec, typename FStd>
void check_trig(const char *name, FVec fv, FStd fs, T lo, T hi, double bound_ulp1)
{
    constexpr std::size_t n = 8192;
    std::vector<T> in(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        in[i] = static_cast<T>(static_cast<double>(lo) + (static_cast<double>(hi) - static_cast<double>(lo)) *
                                                             static_cast<double>(i) / static_cast<double>(n - 1));
    }
    fv(in.data(), out.data(), n);
    const double u1 = std::is_same_v<T, double> ? 2.220446049250313e-16 : 1.1920928955078125e-07;
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double e = std::abs(static_cast<double>(out[i]) - static_cast<double>(fs(in[i])));
        if (std::isfinite(e))
        {
            worst = std::max(worst, e / u1);
        }
    }
    if (!(worst <= bound_ulp1))
    {
        char buf[192];
        std::snprintf(buf, sizeof(buf), "%s worst %.3g ulp(1) exceeds %.3g", name, worst, bound_ulp1);
        test::check(false, "vecmath trig accuracy", buf);
    }
}

// Bitwise-exact match (NaN == NaN) of a unary kernel at hand-picked specials.
template <typename T, typename FVec, typename FStd>
void check_specials(const char *name, FVec fv, FStd fs, const std::vector<T> &vals)
{
    std::vector<T> in(vals.size() + 2), out(vals.size() + 2);
    for (std::size_t i = 0; i < vals.size(); ++i)
    {
        in[i] = vals[i];
    }
    in[vals.size()] = vals.empty() ? T(0) : vals[0];
    in[vals.size() + 1] = vals.empty() ? T(0) : vals[0];
    fv(in.data(), out.data(), in.size());
    for (std::size_t i = 0; i < vals.size(); ++i)
    {
        const T ref = fs(vals[i]);
        const T got = out[i];
        const bool ok = (got == ref) || (std::isnan(got) && std::isnan(ref));
        if (!ok)
        {
            char buf[192];
            std::snprintf(buf, sizeof(buf), "%s(%g): got %g want %g", name, static_cast<double>(vals[i]),
                          static_cast<double>(got), static_cast<double>(ref));
            test::check(false, "vecmath specials", buf);
        }
    }
}

#define VEC1D(F) [](const double *a, double *b, std::size_t n) { np::vecmath::F(a, b, n); }
#define VEC1F(F) [](const float *a, float *b, std::size_t n) { np::vecmath::F(a, b, n); }

void test_accuracy_f64()
{
    check_trig<double>("sin", VEC1D(sin), static_cast<double (*)(double)>(std::sin), -100.0, 100.0, 2.0);
    check_trig<double>("cos", VEC1D(cos), static_cast<double (*)(double)>(std::cos), -100.0, 100.0, 2.0);
    check_trig<double>("tan", VEC1D(tan), static_cast<double (*)(double)>(std::tan), -1.4, 1.4, 16.0);
    check_unary<double>("asin", VEC1D(asin), static_cast<double (*)(double)>(std::asin), -1.0, 1.0, 6.0);
    check_unary<double>("acos", VEC1D(acos), static_cast<double (*)(double)>(std::acos), -1.0, 1.0, 6.0);
    check_unary<double>("atan", VEC1D(atan), static_cast<double (*)(double)>(std::atan), -1e6, 1e6, 4.0);
    check_unary<double>("sinh", VEC1D(sinh), static_cast<double (*)(double)>(std::sinh), -10.0, 10.0, 4.0);
    check_unary<double>("cosh", VEC1D(cosh), static_cast<double (*)(double)>(std::cosh), -10.0, 10.0, 4.0);
    check_unary<double>("tanh", VEC1D(tanh), static_cast<double (*)(double)>(std::tanh), -10.0, 10.0, 4.0);
    check_unary<double>("asinh", VEC1D(asinh), static_cast<double (*)(double)>(std::asinh), -1e10, 1e10, 4.0);
    check_unary<double>("acosh", VEC1D(acosh), static_cast<double (*)(double)>(std::acosh), 1.0, 1e10, 4.0);
    check_unary<double>("atanh", VEC1D(atanh), static_cast<double (*)(double)>(std::atanh), -0.999999, 0.999999, 4.0);
    check_unary<double>("exp", VEC1D(exp), static_cast<double (*)(double)>(std::exp), -700.0, 700.0, 2.0);
    check_unary<double>("expm1", VEC1D(expm1), static_cast<double (*)(double)>(std::expm1), -10.0, 10.0, 3.0);
    check_unary<double>("expm1-tiny", VEC1D(expm1), static_cast<double (*)(double)>(std::expm1), -1e-8, 1e-8, 2.0);
    check_unary<double>("exp2", VEC1D(exp2), static_cast<double (*)(double)>(std::exp2), -1000.0, 1000.0, 2.0);
    check_unary<double>("log", VEC1D(log), static_cast<double (*)(double)>(std::log), 1e-300, 1e300, 2.0, true);
    check_unary<double>("log-mid", VEC1D(log), static_cast<double (*)(double)>(std::log), 0.5, 2.0, 2.0);
    check_unary<double>("log10", VEC1D(log10), static_cast<double (*)(double)>(std::log10), 1e-300, 1e300, 2.0, true);
    check_unary<double>("log2", VEC1D(log2), static_cast<double (*)(double)>(std::log2), 1e-300, 1e300, 2.0, true);
    check_unary<double>("log1p", VEC1D(log1p), static_cast<double (*)(double)>(std::log1p), -0.999999, 10.0, 2.0);
    check_unary<double>("log1p-tiny", VEC1D(log1p), static_cast<double (*)(double)>(std::log1p), -1e-8, 1e-8, 2.0);
    check_unary<double>("cbrt", VEC1D(cbrt), static_cast<double (*)(double)>(std::cbrt), -1e9, 1e9, 3.0);
    // Correctly-rounded kernels: bitwise exact.
    check_unary<double>("sqrt", VEC1D(sqrt), static_cast<double (*)(double)>(std::sqrt), 0.0, 1e300, 0.0, true);
    check_unary<double>("floor", VEC1D(floor), static_cast<double (*)(double)>(std::floor), -100.0, 100.0, 0.0);
    check_unary<double>("ceil", VEC1D(ceil), static_cast<double (*)(double)>(std::ceil), -100.0, 100.0, 0.0);
    check_unary<double>("trunc", VEC1D(trunc), static_cast<double (*)(double)>(std::trunc), -100.0, 100.0, 0.0);
    check_unary<double>("rint", VEC1D(rint), static_cast<double (*)(double)>(std::rint), -100.0, 100.0, 0.0);
    check_unary<double>("fabs", VEC1D(fabs), static_cast<double (*)(double)>(std::fabs), -100.0, 100.0, 0.0);
}

void test_accuracy_f32()
{
    check_trig<float>("sinf", VEC1F(sin), static_cast<float (*)(float)>(std::sin), -100.0f, 100.0f, 2.0);
    check_trig<float>("cosf", VEC1F(cos), static_cast<float (*)(float)>(std::cos), -100.0f, 100.0f, 2.0);
    check_trig<float>("tanf", VEC1F(tan), static_cast<float (*)(float)>(std::tan), -1.4f, 1.4f, 16.0);
    check_unary<float>("asinf", VEC1F(asin), static_cast<float (*)(float)>(std::asin), -1.0f, 1.0f, 6.0);
    check_unary<float>("acosf", VEC1F(acos), static_cast<float (*)(float)>(std::acos), -1.0f, 1.0f, 6.0);
    check_unary<float>("atanf", VEC1F(atan), static_cast<float (*)(float)>(std::atan), -1e4f, 1e4f, 4.0);
    check_unary<float>("sinhf", VEC1F(sinh), static_cast<float (*)(float)>(std::sinh), -10.0f, 10.0f, 4.0);
    check_unary<float>("coshf", VEC1F(cosh), static_cast<float (*)(float)>(std::cosh), -10.0f, 10.0f, 4.0);
    check_unary<float>("tanhf", VEC1F(tanh), static_cast<float (*)(float)>(std::tanh), -10.0f, 10.0f, 4.0);
    check_unary<float>("expf", VEC1F(exp), static_cast<float (*)(float)>(std::exp), -80.0f, 80.0f, 2.0);
    check_unary<float>("expm1f", VEC1F(expm1), static_cast<float (*)(float)>(std::expm1), -5.0f, 5.0f, 3.0);
    check_unary<float>("exp2f", VEC1F(exp2), static_cast<float (*)(float)>(std::exp2), -120.0f, 120.0f, 2.0);
    check_unary<float>("logf", VEC1F(log), static_cast<float (*)(float)>(std::log), 1e-30f, 1e30f, 2.0, true);
    check_unary<float>("log10f", VEC1F(log10), static_cast<float (*)(float)>(std::log10), 1e-30f, 1e30f, 3.0, true);
    check_unary<float>("log2f", VEC1F(log2), static_cast<float (*)(float)>(std::log2), 1e-30f, 1e30f, 3.0, true);
    check_unary<float>("log1pf", VEC1F(log1p), static_cast<float (*)(float)>(std::log1p), -0.999f, 10.0f, 2.0);
    check_unary<float>("cbrtf", VEC1F(cbrt), static_cast<float (*)(float)>(std::cbrt), -1e6f, 1e6f, 3.0);
    check_unary<float>("sqrtf", VEC1F(sqrt), static_cast<float (*)(float)>(std::sqrt), 0.0f, 1e30f, 0.0, true);
}

void test_accuracy_binary()
{
    constexpr std::size_t n = 4096;
    // pow with |b*log2(a)| <= 12: tight bound (error grows with |s|, see docs).
    {
        std::vector<double> a(n), b(n), out(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a[i] = 0.125 * std::pow(64.0, static_cast<double>(i) / static_cast<double>(n - 1));
            b[i] = -4.0 + 8.0 * static_cast<double>((i * 37) % n) / static_cast<double>(n - 1);
        }
        np::vecmath::pow(a.data(), b.data(), out.data(), n);
        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double ref = std::pow(a[i], b[i]);
            worst = std::max(worst, std::abs(out[i] - ref) / ulp_of(ref));
        }
        test::check(worst <= 16.0, "pow accuracy");
    }
    // pow exact small exponents: bitwise identical to libm.
    {
        const double avals[8] = {0.5, 1.5, 2.0, 3.7, 100.0, 0.1, 7.0, 9.0};
        const double bvals[8] = {0.0, 1.0, 2.0, -1.0, 0.5, -0.5, 0.0, 2.0};
        double a[10], b[10], out[10];
        for (int i = 0; i < 8; ++i)
        {
            a[i] = avals[i];
            b[i] = bvals[i];
        }
        a[8] = 2.0;
        b[8] = 2.0;
        a[9] = 0.25;
        b[9] = -0.5;
        np::vecmath::pow(a, b, out, 10);
        for (int i = 0; i < 10; ++i)
        {
            test::check(out[i] == std::pow(a[i], b[i]), "pow exact exponents");
        }
    }
    // hypot over extreme magnitudes.
    {
        std::vector<double> a(n), b(n), out(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a[i] = -1e300 + 2e300 * static_cast<double>(i) / static_cast<double>(n - 1);
            b[i] = -1e300 + 2e300 * static_cast<double>((i * 37) % n) / static_cast<double>(n - 1);
        }
        np::vecmath::hypot(a.data(), b.data(), out.data(), n);
        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double ref = std::hypot(a[i], b[i]);
            worst = std::max(worst, std::abs(out[i] - ref) / ulp_of(ref));
        }
        test::check(worst <= 3.0, "hypot accuracy");
    }
    // atan2 over quadrants.
    {
        std::vector<double> y(n), x(n), out(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            y[i] = -10.0 + 20.0 * static_cast<double>(i) / static_cast<double>(n - 1);
            x[i] = -10.0 + 20.0 * static_cast<double>((i * 37) % n) / static_cast<double>(n - 1);
        }
        np::vecmath::atan2(y.data(), x.data(), out.data(), n);
        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double ref = std::atan2(y[i], x[i]);
            worst = std::max(worst, std::abs(out[i] - ref) / ulp_of(ref));
        }
        test::check(worst <= 4.0, "atan2 accuracy");
    }
    // sincos fused matches separate sin/cos bit-for-bit in the fast path.
    {
        std::vector<double> in(n), s1(n), c1(n), s2(n), c2(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            in[i] = -10.0 + 20.0 * static_cast<double>(i) / static_cast<double>(n - 1);
        }
        np::vecmath::sincos(in.data(), s1.data(), c1.data(), n);
        np::vecmath::sin(in.data(), s2.data(), n);
        np::vecmath::cos(in.data(), c2.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            test::check(s1[i] == s2[i], "sincos/sin consistency");
            test::check(c1[i] == c2[i], "sincos/cos consistency");
        }
    }
}

void test_specials()
{
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double den = std::numeric_limits<double>::denorm_min();
    check_specials<double>("sin", VEC1D(sin), static_cast<double (*)(double)>(std::sin),
                           {0.0, -0.0, inf, -inf, nan, 1e10, -1e10, 1e300});
    check_specials<double>("cos", VEC1D(cos), static_cast<double (*)(double)>(std::cos),
                           {0.0, -0.0, inf, -inf, nan, 1e10, -1e10, 1e300});
    check_specials<double>("tan", VEC1D(tan), static_cast<double (*)(double)>(std::tan),
                           {0.0, -0.0, inf, -inf, nan, 1e10, -1e10});
    check_specials<double>("asin", VEC1D(asin), static_cast<double (*)(double)>(std::asin),
                           {0.0, -0.0, 1.0, -1.0, 2.0, -2.0, inf, nan, den});
    check_specials<double>("acos", VEC1D(acos), static_cast<double (*)(double)>(std::acos),
                           {0.0, -0.0, 1.0, -1.0, 2.0, -2.0, inf, nan});
    check_specials<double>("atan", VEC1D(atan), static_cast<double (*)(double)>(std::atan),
                           {0.0, -0.0, inf, -inf, nan, den, 1e300});
    check_specials<double>("sinh", VEC1D(sinh), static_cast<double (*)(double)>(std::sinh),
                           {0.0, -0.0, inf, -inf, nan, den});
    check_specials<double>("cosh", VEC1D(cosh), static_cast<double (*)(double)>(std::cosh), {0.0, inf, -inf, nan});
    check_specials<double>("tanh", VEC1D(tanh), static_cast<double (*)(double)>(std::tanh),
                           {0.0, -0.0, inf, -inf, nan});
    check_specials<double>("asinh", VEC1D(asinh), static_cast<double (*)(double)>(std::asinh),
                           {0.0, -0.0, inf, -inf, nan, den, 1e-10});
    check_specials<double>("acosh", VEC1D(acosh), static_cast<double (*)(double)>(std::acosh),
                           {1.0, inf, 0.0, -1.0, -inf, nan});
    check_specials<double>("atanh", VEC1D(atanh), static_cast<double (*)(double)>(std::atanh),
                           {0.0, -0.0, 1.0, -1.0, 2.0, -2.0, inf, nan});
    check_specials<double>("exp", VEC1D(exp), static_cast<double (*)(double)>(std::exp),
                           {0.0, -0.0, inf, -inf, nan, 1000.0, -1000.0, den});
    check_specials<double>("expm1", VEC1D(expm1), static_cast<double (*)(double)>(std::expm1),
                           {0.0, -0.0, inf, -inf, nan, den});
    check_specials<double>("exp2", VEC1D(exp2), static_cast<double (*)(double)>(std::exp2),
                           {0.0, -0.0, inf, -inf, nan, 2000.0, -2000.0});
    check_specials<double>("log", VEC1D(log), static_cast<double (*)(double)>(std::log),
                           {0.0, 1.0, inf, -1.0, -inf, nan, den});
    check_specials<double>("log10", VEC1D(log10), static_cast<double (*)(double)>(std::log10),
                           {0.0, 1.0, inf, -1.0, nan});
    check_specials<double>("log2", VEC1D(log2), static_cast<double (*)(double)>(std::log2), {0.0, 1.0, inf, -1.0, nan});
    check_specials<double>("log1p", VEC1D(log1p), static_cast<double (*)(double)>(std::log1p),
                           {0.0, -0.0, -1.0, inf, -2.0, nan});
    check_specials<double>("sqrt", VEC1D(sqrt), static_cast<double (*)(double)>(std::sqrt),
                           {0.0, -0.0, inf, -1.0, nan});
    check_specials<double>("cbrt", VEC1D(cbrt), static_cast<double (*)(double)>(std::cbrt),
                           {0.0, -0.0, inf, -inf, nan});
    check_specials<double>("floor", VEC1D(floor), static_cast<double (*)(double)>(std::floor),
                           {0.5, -0.5, 2.0, -2.0, inf, -inf, nan});
    check_specials<double>("ceil", VEC1D(ceil), static_cast<double (*)(double)>(std::ceil),
                           {0.5, -0.5, 2.0, -2.0, inf, -inf, nan});
    check_specials<double>("rint", VEC1D(rint), static_cast<double (*)(double)>(std::rint),
                           {0.5, 1.5, 2.5, -0.5, inf, -inf, nan});

    // atan2/pow/hypot edge tables (IEEE exact cases).
    {
        const double ys[7] = {0.0, -0.0, 1.0, -1.0, inf, -inf, nan};
        const double xs[7] = {0.0, -0.0, 1.0, -1.0, inf, -inf, nan};
        double ia[4], ib[4], oo[4];
        for (double y : ys)
        {
            for (double x : xs)
            {
                // Skip finite non-edge pairs (covered by the ULP sweep).
                if (std::isfinite(y) && std::isfinite(x) && y != 0.0 && x != 0.0)
                {
                    continue;
                }
                ia[0] = ia[1] = ia[2] = ia[3] = y;
                ib[0] = ib[1] = ib[2] = ib[3] = x;
                np::vecmath::atan2(ia, ib, oo, 4);
                const double ref = std::atan2(y, x);
                test::check(oo[0] == ref || (std::isnan(oo[0]) && std::isnan(ref)), "atan2 edges");
                np::vecmath::hypot(ia, ib, oo, 4);
                const double href = std::hypot(y, x);
                test::check(oo[0] == href || (std::isnan(oo[0]) && std::isnan(href)), "hypot edges");
            }
        }
        const double pb[10][2] = {{0.0, 0.0}, {2.0, 0.0}, {nan, 0.0}, {inf, 0.0}, {0.0, 1.0},
                                  {5.0, 1.0}, {4.0, 2.0}, {9.0, 0.5}, {0.0, 2.0}, {-2.0, 3.0}};
        for (const auto &pr : pb)
        {
            ia[0] = ia[1] = ia[2] = ia[3] = pr[0];
            ib[0] = ib[1] = ib[2] = ib[3] = pr[1];
            np::vecmath::pow(ia, ib, oo, 4);
            const double ref = std::pow(pr[0], pr[1]);
            test::check(oo[0] == ref || (std::isnan(oo[0]) && std::isnan(ref)), "pow edges");
        }
    }

    // Huge trig arguments take the scalar libm lane fallback (bitwise exact).
    {
        const double hx[4] = {1e7, -1e7, 1e15, -1e15};
        double ho[4] = {};
        np::vecmath::sin(hx, ho, 4);
        np::vecmath::cos(hx, ho, 4);
        np::vecmath::tan(hx, ho, 4);
        for (int i = 0; i < 4; ++i)
        {
            test::check(ho[i] == std::tan(hx[i]), "tan huge fallback");
        }
        np::vecmath::sin(hx, ho, 4);
        for (int i = 0; i < 4; ++i)
        {
            test::check(ho[i] == std::sin(hx[i]), "sin huge fallback");
        }
    }
}

void test_misaligned()
{
    // Sizes that defeat every vector width (incl. tails and single lanes).
    for (const std::size_t n : {1u, 2u, 3u, 5u, 7u, 9u, 15u, 17u, 31u, 63u, 100u, 1000u})
    {
        std::vector<double> in(n), out(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            in[i] = -2.0 + 4.0 * static_cast<double>(i) / static_cast<double>(n == 1 ? 1 : n - 1);
        }
        np::vecmath::exp(in.data(), out.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double ref = std::exp(in[i]);
            test::check(std::abs(out[i] - ref) <= 2.0 * ulp_of(ref), "misaligned exp");
        }
        np::vecmath::atan2(in.data(), in.data(), out.data(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double ref = std::atan2(in[i], in[i]);
            test::check(std::abs(out[i] - ref) <= 4.0 * ulp_of(ref), "misaligned atan2");
        }
    }
}

void test_integration()
{
    using namespace np;
    // Contiguous double arrays route through the vector kernels (incl. out=).
    {
        auto x = linspace(-3.0, 3.0, 256);
        auto s = sin(x);
        auto c = cos(x);
        auto e = exp(x);
        auto t = tan(x);
        auto as = arctan(x);
        auto sh = sinh(x);
        auto th = tanh(x);
        auto l1 = log1p(abs(x) * 0.1);
        auto sq = sqrt(abs(x));
        auto cb = cbrt(x);
        auto pw = power(abs(x) + 0.5, full_like(x, 1.5));
        auto hy = hypot(x, x);
        auto a2 = arctan2(x, full_like(x, 1.0));
        test::check(s.shape == x.shape, "np::sin shape");
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            const double xv = x.at(i);
            test::check(test::approx(s.at(i), std::sin(xv), 1e-12), "np::sin values");
            test::check(test::approx(c.at(i), std::cos(xv), 1e-12), "np::cos values");
            test::check(test::approx(e.at(i), std::exp(xv), 1e-12), "np::exp values");
            test::check(test::approx(t.at(i), std::tan(xv), 1e-12), "np::tan values");
            test::check(test::approx(as.at(i), std::atan(xv), 1e-12), "np::arctan values");
            test::check(test::approx(sh.at(i), std::sinh(xv), 1e-12), "np::sinh values");
            test::check(test::approx(th.at(i), std::tanh(xv), 1e-12), "np::tanh values");
            test::check(test::approx(sq.at(i), std::sqrt(std::abs(xv)), 1e-12), "np::sqrt values");
            test::check(test::approx(cb.at(i), std::cbrt(xv), 1e-12), "np::cbrt values");
            test::check(test::approx(pw.at(i), std::pow(std::abs(xv) + 0.5, 1.5), 1e-9), "np::power values");
            test::check(test::approx(hy.at(i), std::hypot(xv, xv), 1e-12), "np::hypot values");
            test::check(test::approx(a2.at(i), std::atan2(xv, 1.0), 1e-12), "np::arctan2 values");
            test::check(test::approx(l1.at(i), std::log1p(std::abs(xv) * 0.1), 1e-12), "np::log1p values");
        }
        ndarray<double> out(x.shape);
        sin(x, out);
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            test::check(test::approx(out.at(i), std::sin(x.at(i)), 1e-12), "np::sin out= values");
        }
        // Broadcast fallback still works alongside the fast path.
        auto bc = arctan2(x, ones<double>({1}));
        test::check(bc.size() == x.size(), "np::arctan2 broadcast");
    }
    // Float arrays take the float kernels.
    {
        auto x = linspace(0.1f, 3.0f, 128);
        auto l = log(x);
        auto e2 = exp2(x);
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            test::check(test::approx(static_cast<double>(l.at(i)), static_cast<double>(std::log(x.at(i))), 1e-6),
                        "np::logf values");
            test::check(test::approx(static_cast<double>(e2.at(i)), static_cast<double>(std::exp2(x.at(i))), 1e-6),
                        "np::exp2f values");
        }
    }
    // Feature flags.
    test::check(np::simd::Features::has_vecmath, "simd vecmath flag");
    test::check(np::vecmath::Features::width_f32 >= 1 && np::vecmath::Features::width_f64 >= 1, "vecmath widths");
}

} // namespace

int main()
{
    test_accuracy_f64();
    test_accuracy_f32();
    test_accuracy_binary();
    test_specials();
    test_misaligned();
    test_integration();
    return test::failures() ? 1 : 0;
}
