/**
 * @file modular.hpp
 * @brief Modular forms, q-expansions, Eisenstein series, Dedekind eta, j-invariant, Hecke
 * operators.
 *
 * Header-only, exact `np::bigint` coefficients where possible (Eisenstein),
 * `std::complex<double>` for analytic values (eta, j). Formulas follow
 * Diamond–Shurman, *A First Course in Modular Forms* and
 * Apostol, *Modular Functions and Dirichlet Series*.
 *
 *   - `sigma(k, n)` divisor power sum
 *   - `bernoulli(k)` / `bernoulli_opt(k)` (even k <= 30; odd k>1 is 0)
 *   - `eisenstein_series(k, N)` q-expansion `1 - (2k/Bk) Σ σ_{k-1}(n) q^n`
 *   - `eisenstein_series_double(k, N)` floating-point variant (works for every even k)
 *   - `dedekind_eta(tau, terms)` via `q^{1/24} ∏(1-q^n)`
 *   - `j_invariant(tau, terms)` analytic via `1728 E4^3/(E4^3-E6^2)`
 *   - `j_invariant_series(N)` Fourier coefficients from `q^0` onward
 *     (`[744, 196884, 21493760, ...]`, i.e. `j(q) = q^{-1} + 744 + ...`)
 *   - `hecke_operator(a, k, p)` on q-expansion `a`
 *   - `modular_discriminant`, `ramanujan_tau`
 *
 * Reference: https://en.wikipedia.org/wiki/Eisenstein_series
 *            https://en.wikipedia.org/wiki/Dedekind_eta_function
 *            https://en.wikipedia.org/wiki/J-invariant
 *            https://en.wikipedia.org/wiki/Hecke_operator
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MODULAR_HPP
#define NP_MODULAR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "ndarray.hpp"

namespace np::modular
{

namespace detail
{
// Small deterministic primality test (trial division, fine for Hecke indices).
NP_NODISCARD inline bool is_prime_small(int n) noexcept
{
    if (n < 2)
        return false;
    for (int p = 2; p <= n / p; ++p)
    {
        if (n % p == 0)
            return false;
    }
    return true;
}

// Binary exponentiation for exact bigint powers (avoids O(k) repeated multiply).
NP_NODISCARD inline bigint pow_bigint(bigint base, int exp)
{
    if (exp < 0)
        throw std::invalid_argument("pow_bigint: exp>=0");
    bigint out = 1;
    while (exp > 0)
    {
        if (exp & 1)
            out *= base;
        base *= base;
        exp >>= 1;
    }
    return out;
}

NP_NODISCARD inline double bigint_to_double(const bigint &b)
{
#if NP_HAS_CPP_INT
    return b.convert_to<double>();
#else
    try
    {
        return std::stod(b.value);
    }
    catch (...)
    {
        return 0.0;
    }
#endif
}

NP_NODISCARD inline std::string bigint_to_string(const bigint &b)
{
#if NP_HAS_CPP_INT
    return b.convert_to<std::string>();
#else
    return b.value;
#endif
}

// Truncated product ∏_{n=1}^{terms} (1 - q^n), shared by eta / discriminant.
NP_NODISCARD inline std::complex<double> eta_product(std::complex<double> q, int terms)
{
    std::complex<double> prod = 1.0;
    std::complex<double> qpow = q;
    for (int n = 1; n <= terms; ++n)
    {
        prod *= (1.0 - qpow);
        qpow *= q;
    }
    return prod;
}

template <typename T> void require_1d_nonempty(const ndarray<T> &a, const char *what)
{
    if (a.ndim() != 1)
        throw std::invalid_argument(std::string(what) + ": need 1-D q-expansion");
    if (a.size() == 0)
        throw std::invalid_argument(std::string(what) + ": empty q-expansion");
}

// Truncated power-series multiply: c[n] = Σ_{i=0}^{n} x[i] y[n-i].
NP_NODISCARD inline std::vector<double> series_mul(const std::vector<double> &x, const std::vector<double> &y)
{
    const std::size_t n = x.size();
    std::vector<double> c(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (x[i] == 0.0)
            continue;
        for (std::size_t j = 0; j + i < n; ++j)
            c[i + j] += x[i] * y[j];
    }
    return c;
}
} // namespace detail

NP_NODISCARD inline bool is_prime(int n) noexcept
{
    return detail::is_prime_small(n);
}

NP_NODISCARD inline bigint sigma(int k, int n)
{
    if (k < 0)
        throw std::invalid_argument("sigma: k>=0");
    if (n <= 0)
        return bigint(0);
    bigint s = 0;
    for (int d = 1; d <= n / d; ++d)
    {
        if (n % d == 0)
        {
            s += detail::pow_bigint(bigint(d), k);
            const int other = n / d;
            if (other != d)
                s += detail::pow_bigint(bigint(other), k);
        }
    }
    return s;
}

// Exact Bernoulli numbers B_k = num/den (reduced), B_1 = -1/2 convention.
// Odd k > 1 vanish; even k supported up to 30 (von Staudt–Clausen denominators).
NP_NODISCARD inline std::optional<std::pair<bigint, bigint>> bernoulli_opt(int k)
{
    if (k < 0)
        return std::nullopt;
    if (k == 1)
        return std::make_pair(bigint(-1), bigint(2));
    if (k % 2 == 1)
        return std::make_pair(bigint(0), bigint(1));
    switch (k)
    {
    case 0:
        return std::make_pair(bigint(1), bigint(1));
    case 2:
        return std::make_pair(bigint(1), bigint(6));
    case 4:
        return std::make_pair(bigint(-1), bigint(30));
    case 6:
        return std::make_pair(bigint(1), bigint(42));
    case 8:
        return std::make_pair(bigint(-1), bigint(30));
    case 10:
        return std::make_pair(bigint(5), bigint(66));
    case 12:
        return std::make_pair(bigint(-691), bigint(2730));
    case 14:
        return std::make_pair(bigint(7), bigint(6));
    case 16:
        return std::make_pair(bigint(-3617), bigint(510));
    case 18:
        return std::make_pair(bigint(43867), bigint(798));
    case 20:
        return std::make_pair(bigint(-174611), bigint(330));
    case 22:
        return std::make_pair(bigint(854513), bigint(138));
    case 24:
        return std::make_pair(bigint(-236364091), bigint(2730));
    case 26:
        return std::make_pair(bigint(8553103), bigint(6));
    case 28:
        return std::make_pair(bigint(-23749461029LL), bigint(870));
    case 30:
        return std::make_pair(bigint("8615841276005"), bigint(14322));
    default:
        return std::nullopt;
    }
}
NP_NODISCARD inline std::pair<bigint, bigint> bernoulli(int k)
{
    if (auto o = bernoulli_opt(k))
        return *o;
    throw std::invalid_argument("bernoulli: only k=0, k=1, odd k (trivially 0), and even k<=30 supported");
}

/**
 * @brief Eisenstein series `E_k` q-expansion `a_0 + Σ a_n q^n`, `n=0..N-1`.
 * `a_0=1`, `a_n = - (2k / B_k) * σ_{k-1}(n)` for even `k≥4`.
 * Returns `ndarray<bigint>` exact coefficients.
 *
 * @throws std::logic_error if the `a_0=1` normalization is non-integral
 * (e.g. k=12, where `65520/691` is fractional); use
 * `eisenstein_series_double` for the floating-point expansion instead.
 */
NP_NODISCARD inline ndarray<bigint> eisenstein_series(int k, int N)
{
    if (k < 4 || k % 2 != 0)
        throw std::invalid_argument("eisenstein_series: need even k>=4");
    if (N <= 0)
        throw std::invalid_argument("eisenstein_series: N>0");
    const auto [num, den] = bernoulli(k);
    if (num == 0)
        throw std::logic_error("eisenstein_series: Bernoulli number vanished");
    // factor = -2k / B_k = (-2k*den) / num; applied per-coefficient so that
    // intermediate exactness is checked on `factor_num * sigma` rather than on
    // the factor alone.
    const bigint factor_num = bigint(-2 * k) * den;
    const bigint factor_den = num;
    ndarray<bigint> a(std::vector<int>{N});
    a.at(0) = bigint(1);
    for (int n = 1; n < N; ++n)
    {
        const bigint s = sigma(k - 1, n);
        const bigint scaled = factor_num * s;
        if (scaled % factor_den != 0)
            throw std::logic_error("eisenstein_series: non-integral coefficient for this k "
                                   "(try eisenstein_series_double)");
        a.at(static_cast<std::size_t>(n)) = scaled / factor_den;
    }
    return a;
}

/**
 * @brief Floating-point Eisenstein q-expansion (valid for every even `k≥4`,
 * including non-integral normalizations such as k=12).
 */
NP_NODISCARD inline ndarray<double> eisenstein_series_double(int k, int N)
{
    if (k < 4 || k % 2 != 0)
        throw std::invalid_argument("eisenstein_series_double: need even k>=4");
    if (N <= 0)
        throw std::invalid_argument("eisenstein_series_double: N>0");
    const auto [num, den] = bernoulli(k);
    const double bk = detail::bigint_to_double(num) / detail::bigint_to_double(den);
    if (bk == 0.0)
        throw std::logic_error("eisenstein_series_double: Bernoulli number vanished");
    const double factor = -2.0 * static_cast<double>(k) / bk;
    ndarray<double> a(std::vector<int>{N});
    a.at(0) = 1.0;
    for (int n = 1; n < N; ++n)
        a.at(static_cast<std::size_t>(n)) = factor * detail::bigint_to_double(sigma(k - 1, n));
    return a;
}

NP_NODISCARD inline ndarray<bigint> eisenstein_E4(int N)
{
    return eisenstein_series(4, N);
}
NP_NODISCARD inline ndarray<bigint> eisenstein_E6(int N)
{
    return eisenstein_series(6, N);
}
NP_NODISCARD inline ndarray<bigint> eisenstein_E8(int N)
{
    return eisenstein_series(8, N);
}
NP_NODISCARD inline ndarray<bigint> eisenstein_E10(int N)
{
    return eisenstein_series(10, N);
}

/**
 * @brief Dedekind eta `η(τ) = e^{π i τ/12} ∏_{n≥1}(1 - q^n)`, `q = e^{2π i τ}`.
 * Truncated product to `terms`. Requires `Im(τ) > 0` (`|q| < 1`).
 */
NP_NODISCARD inline std::complex<double> dedekind_eta(std::complex<double> tau, int terms = 50)
{
    if (terms <= 0)
        throw std::invalid_argument("dedekind_eta: terms>0");
    if (tau.imag() <= 0.0)
        throw std::domain_error("dedekind_eta: need Im(tau)>0 for convergence");
    constexpr double pi = std::numbers::pi_v<double>;
    const std::complex<double> q = std::exp(std::complex<double>(0, 2 * pi) * tau);
    const std::complex<double> prod = detail::eta_product(q, terms);
    // q^{1/24} = exp(2*pi*i*tau/24) directly, not pow(q,1/24) which is multi-valued
    const std::complex<double> q24 = std::exp(std::complex<double>(0, 2 * pi) * tau / 24.0);
    return q24 * prod;
}

/**
 * @brief Dedekind eta from `q` directly: `q^{1/24} ∏(1-q^n)`.
 * The 24th root is branch-cut dependent (`std::pow` principal branch);
 * prefer the `tau` overload whenever the phase matters.
 */
NP_NODISCARD inline std::complex<double> dedekind_eta_q(std::complex<double> q, int terms = 50)
{
    if (terms <= 0)
        throw std::invalid_argument("dedekind_eta_q: terms>0");
    if (q == std::complex<double>(0.0, 0.0) || std::abs(q) >= 1.0)
        throw std::domain_error("dedekind_eta_q: need 0<|q|<1 for convergence");
    return std::pow(q, 1.0 / 24.0) * detail::eta_product(q, terms);
}

/**
 * @brief Modular discriminant `Δ(τ) = (2π)^{12} η(τ)^{24}` via q-expansion `q
 * ∏(1-q^n)^{24}`. Returns `q * ∏_{n=1}^{terms} (1 - q^n)^{24}` (without `(2π)^{12}`
 * factor for algebraic normalization).
 */
NP_NODISCARD inline std::complex<double> modular_discriminant_q(std::complex<double> q, int terms = 50)
{
    if (terms <= 0)
        throw std::invalid_argument("modular_discriminant_q: terms>0");
    if (std::abs(q) >= 1.0)
        throw std::domain_error("modular_discriminant_q: need |q|<1 for convergence");
    const std::complex<double> prod = detail::eta_product(q, terms);
    // ∏(1-q^n)^{24} = (∏(1-q^n))^{24}: raise once instead of per-factor pow loop.
    return q * std::pow(prod, 24);
}

/**
 * @brief Analytic discriminant `Δ(τ) = (2π)^12 η(τ)^24` with the transcendental factor.
 */
NP_NODISCARD inline std::complex<double> modular_discriminant(std::complex<double> tau, int terms = 50)
{
    const std::complex<double> eta = dedekind_eta(tau, terms);
    constexpr double pi = std::numbers::pi_v<double>;
    const std::complex<double> base = std::complex<double>(2.0 * pi, 0.0) * eta;
    // base^24 by binary exponentiation (avoids std::pow branch-cut concerns).
    std::complex<double> out = 1.0;
    std::complex<double> bpow = base;
    int exp = 24;
    while (exp > 0)
    {
        if (exp & 1)
            out *= bpow;
        bpow *= bpow;
        exp >>= 1;
    }
    return out;
}

/**
 * @brief Ramanujan tau `τ(n)` = coefficient of `q^n` in `Δ(q) = Σ τ(n) q^n`.
 * Computed via `Δ` product truncated to `N`. Exact `bigint` for small N.
 */
NP_NODISCARD inline ndarray<bigint> ramanujan_tau(int N)
{
    if (N <= 0)
        throw std::invalid_argument("ramanujan_tau: N>0");
    // binom(24,k) fits in 64 bits; hoist out of the O(N^2) multiply loop.
    // C(24,7) = C(24,17) = 346104 (note: not 346504).
    constexpr std::array<long long, 25> kBinom24 = {
        1,       24,      276,     2024,   10626,  42504,  134596, 346104, 735471, 1307504, 1961256, 2496144, 2704156,
        2496144, 1961256, 1307504, 735471, 346104, 134596, 42504,  10626,  2024,   276,     24,      1};
    // Use product (1 - q^n) expansion.
    // For small N we multiply polynomials in bigint.
    // Represent series as vector<bigint> length N: prod_{n=1}^{N-1} (1 - q^n)^{24}
    std::vector<bigint> prod(static_cast<std::size_t>(N), bigint(0));
    prod[0] = bigint(1);
    std::vector<bigint> next(static_cast<std::size_t>(N));
    for (int n = 1; n < N; ++n)
    {
        // multiply by (1 - q^n)^{24} = Σ_{k=0}^{24} binom(24,k) (-1)^k q^{n k}
        std::ranges::fill(next, bigint(0));
        for (int i = 0; i < N; ++i)
        {
            if (prod[static_cast<std::size_t>(i)] == 0)
                continue;
            for (int k = 0; k <= 24; ++k)
            {
                const int j = i + n * k;
                if (j >= N)
                    break;
                bigint coeff = bigint(kBinom24[static_cast<std::size_t>(k)]);
                if (k % 2 == 1)
                    coeff = -coeff;
                next[static_cast<std::size_t>(j)] += prod[static_cast<std::size_t>(i)] * coeff;
            }
        }
        prod.swap(next);
    }
    // Δ = q * prod, so shift by 1: τ(n) = prod[n-1]
    ndarray<bigint> tau(std::vector<int>{N});
    tau.at(0) = bigint(0);
    for (int n = 1; n < N; ++n)
        tau.at(static_cast<std::size_t>(n)) = prod[static_cast<std::size_t>(n - 1)];
    return tau;
}

/**
 * @brief `j`-invariant via `j = 1728 E4^3 / (E4^3 - E6^2)` as `q`-series ratio.
 * Returns `ndarray<double>` of length `N` with `j(q) = q^{-1} + 744 + 196884 q + ...`
 * Coefficients are stored from `q^0` onward (`out[0]=744`), since the `q^{-1}`
 * pole has no finite index.
 */
NP_NODISCARD inline ndarray<double> j_invariant_series(int N)
{
    if (N <= 0)
        throw std::invalid_argument("j_invariant_series: N>0");
    // Need one extra coefficient internally: out[N-1] = s[N] below.
    const int M = N + 1;
    const ndarray<bigint> E4b = eisenstein_series(4, M);
    const ndarray<bigint> E6b = eisenstein_series(6, M);
    std::vector<double> e4(static_cast<std::size_t>(M)), e6(static_cast<std::size_t>(M));
    for (int i = 0; i < M; ++i)
    {
        e4[static_cast<std::size_t>(i)] = detail::bigint_to_double(E4b.at(static_cast<std::size_t>(i)));
        e6[static_cast<std::size_t>(i)] = detail::bigint_to_double(E6b.at(static_cast<std::size_t>(i)));
    }
    const std::vector<double> e4sq = detail::series_mul(e4, e4);
    const std::vector<double> f = detail::series_mul(e4sq, e4); // E4^3
    const std::vector<double> g = detail::series_mul(e6, e6);   // E6^2
    // d = E4^3 - E6^2 = 1728 q + ..., so delta_red[n] = d[n+1]/1728 starts at 1.
    std::vector<double> delta_red(static_cast<std::size_t>(M));
    for (int i = 0; i < M; ++i)
    {
        const int j = i + 1;
        const double d = (j < M) ? (f[static_cast<std::size_t>(j)] - g[static_cast<std::size_t>(j)]) : 0.0;
        delta_red[static_cast<std::size_t>(i)] = d / 1728.0;
    }
    if (delta_red[0] == 0.0)
        throw std::logic_error("j_invariant_series: degenerate denominator");
    // s = f / delta_red, then j = q^{-1} s.
    std::vector<double> s(static_cast<std::size_t>(M), 0.0);
    for (int n = 0; n < M; ++n)
    {
        double acc = f[static_cast<std::size_t>(n)];
        for (int i = 0; i < n; ++i)
            acc -= s[static_cast<std::size_t>(i)] * delta_red[static_cast<std::size_t>(n - i)];
        s[static_cast<std::size_t>(n)] = acc / delta_red[0];
    }
    ndarray<double> out(std::vector<int>{N});
    for (int n = 0; n < N; ++n)
        out.at(static_cast<std::size_t>(n)) = s[static_cast<std::size_t>(n) + 1];
    return out;
}

/**
 * @brief Analytic `j`-invariant `j(τ) = 1728 E4(τ)^3 / (E4(τ)^3 - E6(τ)^2)`.
 * `E4, E6` are evaluated from `terms` Fourier coefficients at `q=e^{2πiτ}`.
 */
NP_NODISCARD inline std::complex<double> j_invariant(std::complex<double> tau, int terms = 50)
{
    if (terms <= 0)
        throw std::invalid_argument("j_invariant: terms>0");
    if (tau.imag() <= 0.0)
        throw std::domain_error("j_invariant: need Im(tau)>0 for convergence");
    constexpr double pi = std::numbers::pi_v<double>;
    const std::complex<double> q = std::exp(std::complex<double>(0, 2 * pi) * tau);
    const ndarray<bigint> E4b = eisenstein_series(4, terms);
    const ndarray<bigint> E6b = eisenstein_series(6, terms);
    std::complex<double> e4 = 0.0, e6 = 0.0;
    std::complex<double> qpow = 1.0;
    for (int n = 0; n < terms; ++n)
    {
        e4 += detail::bigint_to_double(E4b.at(static_cast<std::size_t>(n))) * qpow;
        e6 += detail::bigint_to_double(E6b.at(static_cast<std::size_t>(n))) * qpow;
        qpow *= q;
    }
    const std::complex<double> f = e4 * e4 * e4;
    const std::complex<double> den = f - e6 * e6;
    if (std::abs(den) == 0.0)
        throw std::domain_error("j_invariant: singular (E4^3 == E6^2 at this tau)");
    return 1728.0 * f / den;
}

/**
 * @brief Hecke operator `T_p` on q-expansion `a` (weight `k`).
 * `(T_p a)_n = a_{pn} + p^{k-1} a_{n/p}` (with `a_{n/p}=0` if `p∤n`).
 * `a` length `N`, result length `N`.
 *
 * The input is truncated to length `N`, so `(T_p a)_n` is exact only for
 * `p*n < N`; higher indices silently drop the `a_{pn}` contribution.
 */
NP_NODISCARD inline ndarray<bigint> hecke_operator(const ndarray<bigint> &a, int k, int p)
{
    if (p <= 1)
        throw std::invalid_argument("hecke_operator: p must be prime >1");
    if (k < 2)
        throw std::invalid_argument("hecke_operator: k>=2");
    detail::require_1d_nonempty(a, "hecke_operator");
    const int N = static_cast<int>(a.size());
    ndarray<bigint> b(std::vector<int>{N});
    const bigint p_pow = detail::pow_bigint(bigint(p), k - 1);
    for (int n = 0; n < N; ++n)
    {
        bigint term1 = 0;
        if (p * n < N)
            term1 = a.at(static_cast<std::size_t>(p * n));
        bigint term2 = 0;
        if (n % p == 0)
            term2 = p_pow * a.at(static_cast<std::size_t>(n / p));
        b.at(static_cast<std::size_t>(n)) = term1 + term2;
    }
    return b;
}

NP_NODISCARD inline ndarray<double> hecke_operator(const ndarray<double> &a, int k, int p)
{
    if (p <= 1)
        throw std::invalid_argument("hecke_operator: p must be prime >1");
    if (k < 2)
        throw std::invalid_argument("hecke_operator: k>=2");
    detail::require_1d_nonempty(a, "hecke_operator");
    const int N = static_cast<int>(a.size());
    ndarray<double> b(std::vector<int>{N});
    const double p_pow = std::pow(static_cast<double>(p), k - 1);
    for (int n = 0; n < N; ++n)
    {
        const double t1 = (p * n < N) ? a.at(static_cast<std::size_t>(p * n)) : 0.0;
        const double t2 = (n % p == 0) ? p_pow * a.at(static_cast<std::size_t>(n / p)) : 0.0;
        b.at(static_cast<std::size_t>(n)) = t1 + t2;
    }
    return b;
}

/**
 * @brief Check modular form q-expansion is a Hecke eigenform (up to `primes`).
 *
 * Eigenvalue `λ_p` is derived from the first nonzero coefficient
 * (`λ = (T_p a)_r / a_r`), which reduces to `λ = a_p` for cusp forms
 * normalized with `a_1 = 1` and to `λ = σ_{k-1}(p)` for Eisenstein series.
 * Primes with `p >= size(a)` are skipped (truncated expansion carries no data),
 * and indices with `p*n >= size(a)` are skipped (`a_{pn}` unknown there).
 */
NP_NODISCARD inline bool is_hecke_eigenform(const ndarray<bigint> &a, int k, const std::vector<int> &primes = {2, 3, 5})
{
    if (k < 2)
        throw std::invalid_argument("is_hecke_eigenform: k>=2");
    detail::require_1d_nonempty(a, "is_hecke_eigenform");
    const int N = static_cast<int>(a.size());
    for (int p : primes)
    {
        if (p <= 1 || !detail::is_prime_small(p))
            throw std::invalid_argument("is_hecke_eigenform: primes must be prime >1");
        if (p >= N)
            continue;
        const ndarray<bigint> Tp = hecke_operator(a, k, p);
        // First nonzero reference coefficient determines the eigenvalue.
        int ref = -1;
        for (int n = 0; n < N; ++n)
        {
            if (a.at(static_cast<std::size_t>(n)) != 0)
            {
                ref = n;
                break;
            }
        }
        if (ref < 0)
            return false; // zero form carries no eigenvalue
        const bigint &ar = a.at(static_cast<std::size_t>(ref));
        const bigint &tr = Tp.at(static_cast<std::size_t>(ref));
        if (tr % ar != 0)
            return false;
        const bigint lambda = tr / ar;
        // Truncated q-expansion: (T_p a)_n needs a_{pn}, so only n with p*n<N is verifiable.
        for (int n = 0; n < N; ++n)
        {
            if (p * n >= N)
                continue;
            if (Tp.at(static_cast<std::size_t>(n)) != lambda * a.at(static_cast<std::size_t>(n)))
                return false;
        }
    }
    return true;
}

NP_NODISCARD inline bool is_hecke_eigenform(const ndarray<double> &a, int k, const std::vector<int> &primes = {2, 3, 5},
                                            double tol = 1e-9)
{
    if (k < 2)
        throw std::invalid_argument("is_hecke_eigenform: k>=2");
    if (!(tol > 0.0))
        throw std::invalid_argument("is_hecke_eigenform: tol>0");
    detail::require_1d_nonempty(a, "is_hecke_eigenform");
    const int N = static_cast<int>(a.size());
    for (int p : primes)
    {
        if (p <= 1 || !detail::is_prime_small(p))
            throw std::invalid_argument("is_hecke_eigenform: primes must be prime >1");
        if (p >= N)
            continue;
        const ndarray<double> Tp = hecke_operator(a, k, p);
        int ref = -1;
        for (int n = 0; n < N; ++n)
        {
            if (std::abs(a.at(static_cast<std::size_t>(n))) > tol)
            {
                ref = n;
                break;
            }
        }
        if (ref < 0)
            return false;
        const double lambda = Tp.at(static_cast<std::size_t>(ref)) / a.at(static_cast<std::size_t>(ref));
        // Truncated q-expansion: (T_p a)_n needs a_{pn}, so only n with p*n<N is verifiable.
        for (int n = 0; n < N; ++n)
        {
            if (p * n >= N)
                continue;
            const double expected = lambda * a.at(static_cast<std::size_t>(n));
            if (std::abs(Tp.at(static_cast<std::size_t>(n)) - expected) > tol * (1.0 + std::abs(expected)))
                return false;
        }
    }
    return true;
}

// Ergonomic ModularForm wrapper

struct ModularForm
{
    int weight = 0, level = 1;
    ndarray<bigint> qexp;
    ModularForm() = default;
    ModularForm(int k, int lvl, ndarray<bigint> q) : weight(k), level(lvl), qexp(std::move(q))
    {
        if (weight < 0)
            throw std::invalid_argument("ModularForm: weight>=0");
        if (level < 1)
            throw std::invalid_argument("ModularForm: level>=1");
        detail::require_1d_nonempty(qexp, "ModularForm");
    }
    ModularForm(int k, ndarray<bigint> q) : weight(k), qexp(std::move(q))
    {
        if (weight < 0)
            throw std::invalid_argument("ModularForm: weight>=0");
        detail::require_1d_nonempty(qexp, "ModularForm");
    }

    NP_NODISCARD std::size_t size() const noexcept
    {
        return qexp.size();
    }
    NP_NODISCARD ndarray<bigint> hecke(int p) const
    {
        return hecke_operator(qexp, weight, p);
    }
    NP_NODISCARD bool is_eigenform(const std::vector<int> &primes = {2, 3, 5}) const
    {
        return is_hecke_eigenform(qexp, weight, primes);
    }
    NP_NODISCARD bigint coeff(int n) const
    {
        if (n < 0 || static_cast<std::size_t>(n) >= qexp.size())
            return bigint(0);
        return qexp.at(static_cast<std::size_t>(n));
    }
    NP_NODISCARD std::optional<bigint> coeff_opt(int n) const
    {
        if (n < 0 || static_cast<std::size_t>(n) >= qexp.size())
            return std::nullopt;
        return qexp.at(static_cast<std::size_t>(n));
    }
    NP_NODISCARD bool operator==(const ModularForm &o) const
    {
        if (weight != o.weight || level != o.level || qexp.size() != o.qexp.size())
            return false;
        for (std::size_t i = 0; i < qexp.size(); ++i)
        {
            if (qexp.at(i) != o.qexp.at(i))
                return false;
        }
        return true;
    }
    NP_NODISCARD std::string to_string(int max_terms = 5) const
    {
        if (max_terms <= 0)
            throw std::invalid_argument("ModularForm::to_string: max_terms>0");
        std::string s = "ModularForm k=" + std::to_string(weight) + " qexp: ";
        const int show = std::min(max_terms, static_cast<int>(qexp.size()));
        for (int i = 0; i < show; ++i)
        {
            if (i)
                s += " + ";
            s += detail::bigint_to_string(qexp.at(static_cast<std::size_t>(i))) + "*q^" + std::to_string(i);
        }
        return s;
    }
};

NP_NODISCARD inline ModularForm make_eisenstein(int k, int N, int level = 1)
{
    return ModularForm(k, level, eisenstein_series(k, N));
}
NP_NODISCARD inline ModularForm make_delta(int N)
{
    return ModularForm(12, 1, ramanujan_tau(N));
}

} // namespace np::modular

#endif // NP_MODULAR_HPP
